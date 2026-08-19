"""Calibration-only OneRel proposal plus frozen pair-classifier reranking.

This diagnostic tests the missing bridge between the proven high-recall
OneRel proposal set and the existing fit-only relation classifier.  It never
reads outer or official-dev data and selects only one global candidate
threshold plus schema conditioning on/off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.bench.npu.om_manifest import sha256_tree
from dki.evidence import sha256_file
from dki.nlp.cmeie_candidate_reranker import (
    PairRerankCandidate,
    expand_pair_rerank_inputs,
    micro_spo_metric,
    predicate_signature_map,
    select_global_rerank_policy,
)
from dki.nlp.cmeie_pair import load_relation_contract, mark_pair


MAX_LENGTH = 320
PAIR_RANK = 192
LORA_RANK = 8
LORA_ALPHA = 16.0
LINK_THRESHOLD = -0.5
CANDIDATE_CAP = 4096
COORDINATE_CANDIDATE_CAP = 32768
THRESHOLDS = tuple(round(value / 20.0, 2) for value in range(2, 20))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _gold(row: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (
            str(spo.get("subject", "")),
            str(spo.get("predicate", "")),
            str(spo.get("object", "")),
        )
        for spo in row.get("spo_list", [])
        if spo.get("subject") and spo.get("predicate") and spo.get("object")
    }


def _rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _base_model_tree_sha256(path: Path) -> str:
    """Match the content hash used by ``train_cmeie_onerel.py``.

    Hugging Face snapshot members are normally symlinks into the blob cache.
    The training receipt hashes the resolved top-level regular-file contents,
    so evaluation must use that same binding rather than the stricter OM-tree
    policy that rejects all symlinks.
    """

    entries: list[tuple[str, int, str]] = []
    for item in sorted(candidate for candidate in path.iterdir() if candidate.is_file()):
        entries.append((item.name, item.stat().st_size, sha256_file(item)))
    if not entries:
        raise ValueError("base model snapshot has no files")
    digest = hashlib.sha256()
    for name, size, file_sha256 in entries:
        digest.update(f"{name}\0{size}\0{file_sha256}\n".encode("utf-8"))
    return digest.hexdigest()


def _create_once_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _density(values: Sequence[int]) -> dict[str, float | int]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"mean": 0.0, "p95": 0, "p99": 0, "max": 0}

    def percentile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

    return {
        "mean": sum(ordered) / len(ordered),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(ordered),
    }


def _tokenize(rows: Sequence[dict[str, Any]], tokenizer: Any, device: Any):
    encoded = tokenizer(
        [str(row.get("text", "")) for row in rows],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = [
        [(int(left), int(right)) for left, right in row]
        for row in encoded.pop("offset_mapping").tolist()
    ]
    return {name: value.to(device) for name, value in encoded.items()}, offsets


def _surface_candidate(
    *,
    row_index: int,
    text: str,
    offsets: Sequence[tuple[int, int]],
    candidate: Any,
    predicates: Sequence[str],
) -> PairRerankCandidate | None:
    indices = (
        candidate.subject_start,
        candidate.subject_end,
        candidate.object_start,
        candidate.object_end,
    )
    if any(not 0 <= index < len(offsets) for index in indices):
        return None
    ss, se, os_, oe = indices
    if any(offsets[index][1] <= offsets[index][0] for index in indices):
        return None
    subject_span = (offsets[ss][0], offsets[se][1])
    object_span = (offsets[os_][0], offsets[oe][1])
    subject = text[subject_span[0] : subject_span[1]]
    obj = text[object_span[0] : object_span[1]]
    if not subject or not obj or subject_span == object_span:
        return None
    marked = mark_pair(
        text,
        subject,
        obj,
        subject_span=subject_span,
        object_span=object_span,
    )
    if marked is None:
        return None
    predicate_id = int(candidate.predicate_id)
    if not 0 <= predicate_id < len(predicates):
        raise ValueError("OneRel candidate predicate is out of range")
    return PairRerankCandidate(
        row_index=row_index,
        predicate_id=predicate_id,
        subject=subject,
        predicate=str(predicates[predicate_id]),
        object=obj,
        marked_text=marked,
        relation_score=float(candidate.relation_score),
        link_score=float(candidate.link_score),
        boundary_score=float(candidate.boundary_score),
    )


def _load_onerel(
    *,
    checkpoint_path: Path,
    base_model: Path,
    device: Any,
    torch: Any,
):
    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_onerel import CMeIEOneRel, inject_bert_attention_lora

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_family") != "onerel":
        raise ValueError("checkpoint is not a OneRel model")
    predicates = [str(value) for value in checkpoint.get("predicates", [])]
    if len(predicates) != 44 or len(set(predicates)) != 44:
        raise ValueError("OneRel checkpoint must bind exactly 44 predicates")
    expected_base_sha = checkpoint.get("base_model_tree_sha256")
    actual_base_sha = _base_model_tree_sha256(base_model.resolve())
    if expected_base_sha and expected_base_sha != actual_base_sha:
        raise ValueError("OneRel checkpoint base-model tree hash differs")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=True, local_files_only=True
    )
    if not tokenizer.is_fast:
        raise ValueError("OneRel proposal extraction requires a fast tokenizer")
    encoder = AutoModel.from_pretrained(base_model, local_files_only=True)
    inject_bert_attention_lora(encoder, rank=LORA_RANK, alpha=LORA_ALPHA)
    model = CMeIEOneRel(
        encoder,
        int(encoder.config.hidden_size),
        len(predicates),
        pair_rank=PAIR_RANK,
    )
    missing, unexpected = model.load_state_dict(
        checkpoint.get("model_delta", {}), strict=False
    )
    critical_prefixes = (
        "context_adapter.",
        "left_projection.",
        "right_projection.",
        "pair_norm.",
        "pair_hidden.",
        "link_weight",
        "link_bias",
        "relation_presence.",
        "boundary.",
    )
    critical_missing = [
        name for name in missing if name.startswith(critical_prefixes)
    ]
    missing_lora = [
        name
        for name in missing
        if name.startswith("encoder.")
        and (name.endswith(".a.weight") or name.endswith(".b.weight"))
    ]
    if critical_missing or missing_lora or unexpected:
        raise ValueError(
            "OneRel proposal checkpoint mismatch: "
            f"critical_missing={critical_missing}, missing_lora={missing_lora}, "
            f"unexpected={unexpected}"
        )
    return model.to(device).eval(), tokenizer, predicates, checkpoint, actual_base_sha


def _mask_padded_link_logits(link_logits: Any, attention_mask: Any) -> Any:
    """Exclude both padded link endpoints before CPU sparsification."""

    if link_logits.ndim != 5 or attention_mask.ndim != 2:
        raise ValueError("link logits and attention mask have invalid ranks")
    if (
        link_logits.shape[0] != attention_mask.shape[0]
        or link_logits.shape[-2] != attention_mask.shape[1]
        or link_logits.shape[-1] != attention_mask.shape[1]
    ):
        raise ValueError("link logits and attention mask shapes differ")
    valid = attention_mask.bool()
    pair_mask = valid[:, None, None, :, None] & valid[:, None, None, None, :]
    return link_logits.masked_fill(~pair_mask, float("-inf"))


def _generate_candidates(
    *,
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, Any]],
    predicates: Sequence[str],
    device: Any,
    torch: Any,
    batch_size: int,
    fp16: bool,
    relation_threshold: float = -math.inf,
    link_threshold: float = LINK_THRESHOLD,
    boundary_threshold: float = -math.inf,
    enforce_surface_cap: bool = True,
) -> tuple[list[PairRerankCandidate], dict[str, Any]]:
    from dki.nlp.cmeie_onerel import (
        OneRelDensityError,
        enumerate_onerel_span_candidates,
        sparsify_onerel_logits,
    )

    candidates: list[PairRerankCandidate] = []
    row_counts: list[int] = []
    link_counts: list[int] = []
    dropped_invalid_spans = 0
    duplicate_coordinates = 0
    model.eval()
    model.encoder.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inputs, offsets = _tokenize(batch, tokenizer, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=fp16,
        ):
            hidden = model.encode(**inputs)
            relation_logits, boundary_logits = model.auxiliary_logits(
                hidden, inputs["attention_mask"]
            )
            link_logits = _mask_padded_link_logits(
                model.score_dense(hidden), inputs["attention_mask"]
            )
        sparse = sparsify_onerel_logits(link_logits, link_threshold)
        relation_rows = relation_logits.float().cpu().tolist()
        boundary_rows = boundary_logits.float().cpu().tolist()
        for local_index, row in enumerate(batch):
            row_index = start + local_index
            link_counts.append(len(sparse[local_index]))
            try:
                proposed = enumerate_onerel_span_candidates(
                    relation_rows[local_index],
                    boundary_rows[local_index],
                    sparse[local_index],
                    len(predicates),
                    (relation_threshold, link_threshold, boundary_threshold),
                    maximum_span_candidates=COORDINATE_CANDIDATE_CAP,
                )
            except OneRelDensityError as error:
                raise RuntimeError(
                    f"row {row_index} violates the "
                    f"{COORDINATE_CANDIDATE_CAP} raw-coordinate safety cap"
                ) from error
            row_surface: dict[
                tuple[int, str, str, str], PairRerankCandidate
            ] = {}
            for proposed_item in proposed:
                surface = _surface_candidate(
                    row_index=row_index,
                    text=str(row.get("text", "")),
                    offsets=offsets[local_index],
                    candidate=proposed_item,
                    predicates=predicates,
                )
                if surface is None:
                    dropped_invalid_spans += 1
                    continue
                key = (
                    surface.predicate_id,
                    surface.subject,
                    surface.predicate,
                    surface.object,
                )
                if key in row_surface:
                    duplicate_coordinates += 1
                previous = row_surface.get(key)
                if previous is None or (
                    surface.link_score,
                    surface.relation_score,
                    surface.boundary_score,
                ) > (
                    previous.link_score,
                    previous.relation_score,
                    previous.boundary_score,
                ):
                    row_surface[key] = surface
            row_values = [row_surface[key] for key in sorted(row_surface)]
            if enforce_surface_cap and len(row_values) > CANDIDATE_CAP:
                raise RuntimeError(
                    f"row {row_index} surface candidates exceed {CANDIDATE_CAP}"
                )
            row_counts.append(len(row_values))
            candidates.extend(row_values)
        print(
            json.dumps(
                {
                    "stage": "onerel_proposals",
                    "rows_completed": min(start + len(batch), len(rows)),
                    "rows_total": len(rows),
                    "candidates": len(candidates),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return candidates, {
        "link_density": _density(link_counts),
        "surface_candidate_density": _density(row_counts),
        "dropped_invalid_spans": dropped_invalid_spans,
        "duplicate_surface_coordinates": duplicate_coordinates,
        "raw_coordinate_safety_cap_per_row": COORDINATE_CANDIDATE_CAP,
        "candidate_cap_per_row": CANDIDATE_CAP,
        "proposal_thresholds": {
            "relation": (
                "-inf"
                if math.isinf(relation_threshold) and relation_threshold < 0
                else relation_threshold
            ),
            "link": (
                "-inf"
                if math.isinf(link_threshold) and link_threshold < 0
                else link_threshold
            ),
            "boundary": (
                "-inf"
                if math.isinf(boundary_threshold) and boundary_threshold < 0
                else boundary_threshold
            ),
        },
        "surface_cap_enforced": bool(enforce_surface_cap),
        "inference_truncation": False,
    }


def _score_candidates(
    *,
    candidates: Sequence[PairRerankCandidate],
    signatures_by_predicate: dict[int, tuple[str, ...]],
    schema_on: bool,
    pair_model: Any,
    pair_tokenizer: Any,
    pair_max_length: int,
    batch_size: int,
    device: Any,
    torch: Any,
    fp16: bool,
    n_predicates: int,
) -> tuple[float, ...]:
    expanded = expand_pair_rerank_inputs(
        candidates, signatures_by_predicate, schema_on=schema_on
    )
    scores = [-1.0] * len(candidates)
    for start in range(0, len(expanded), batch_size):
        batch = expanded[start : start + batch_size]
        encoded = pair_tokenizer(
            [item.marked_text for item in batch],
            [item.signature for item in batch],
            padding=True,
            truncation=True,
            max_length=pair_max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=fp16,
        ):
            probabilities = pair_model(**encoded).logits.float().sigmoid().cpu()
        if probabilities.shape[1] != n_predicates:
            raise ValueError("pair classifier output width differs from predicates")
        for item, row in zip(batch, probabilities.tolist()):
            predicate_id = candidates[item.candidate_index].predicate_id
            scores[item.candidate_index] = max(
                scores[item.candidate_index], float(row[predicate_id])
            )
        if start == 0 or start + len(batch) == len(expanded):
            print(
                json.dumps(
                    {
                        "stage": "pair_rerank",
                        "schema_on": schema_on,
                        "inputs_completed": min(start + len(batch), len(expanded)),
                        "inputs_total": len(expanded),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return tuple(scores)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--onerel-checkpoint", type=Path, required=True)
    parser.add_argument("--pair-model", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposal-batch-size", type=int, default=2)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--fp16", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_tree_sha256):
        raise SystemExit("source-tree-sha256 must be lowercase SHA-256")
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    for path in (
        args.calibration_data,
        args.split_manifest,
        args.base_model,
        args.onerel_checkpoint,
        args.pair_model,
    ):
        if not path.exists():
            raise SystemExit(f"required input does not exist: {path}")
    if args.proposal_batch_size < 1 or args.pair_batch_size < 1:
        raise SystemExit("batch sizes must be positive")

    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("CMeIE candidate reranking requires an Ascend NPU")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    started = time.time()
    rows = _jsonl(args.calibration_data)
    gold_rows = [_gold(row) for row in rows]
    onerel, onerel_tokenizer, predicates, checkpoint, base_model_sha = _load_onerel(
        checkpoint_path=args.onerel_checkpoint,
        base_model=args.base_model,
        device=device,
        torch=torch,
    )
    candidates, proposal_audit = _generate_candidates(
        model=onerel,
        tokenizer=onerel_tokenizer,
        rows=rows,
        predicates=predicates,
        device=device,
        torch=torch,
        batch_size=args.proposal_batch_size,
        fp16=args.fp16,
    )
    candidate_rows = [set() for _ in rows]
    for candidate in candidates:
        candidate_rows[candidate.row_index].add(candidate.triple)
    candidate_oracle = micro_spo_metric(gold_rows, candidate_rows)
    recall = float(candidate_oracle["recall"])
    oracle_f1_at_precision_one = 2.0 * recall / max(1e-12, 1.0 + recall)

    del onerel
    torch.npu.empty_cache()
    relation_contract = load_relation_contract(args.pair_model)
    pair_predicates = json.loads(
        (args.pair_model / "predicates.json").read_text(encoding="utf-8")
    )
    if list(pair_predicates) != list(predicates):
        raise ValueError("OneRel and pair-classifier predicate tables differ")
    signatures = predicate_signature_map(relation_contract, predicates)
    if any(not values for values in signatures.values()):
        missing = [predicates[index] for index, values in signatures.items() if not values]
        raise ValueError(f"pair schema has predicates without signatures: {missing}")
    pair_tokenizer = AutoTokenizer.from_pretrained(
        args.pair_model, local_files_only=True
    )
    pair_model = AutoModelForSequenceClassification.from_pretrained(
        args.pair_model, local_files_only=True
    ).to(device).eval()
    pair_max_length = int(relation_contract.get("max_len", 256))
    score_sets = {
        schema_on: _score_candidates(
            candidates=candidates,
            signatures_by_predicate=signatures,
            schema_on=schema_on,
            pair_model=pair_model,
            pair_tokenizer=pair_tokenizer,
            pair_max_length=pair_max_length,
            batch_size=args.pair_batch_size,
            device=device,
            torch=torch,
            fp16=args.fp16,
            n_predicates=len(predicates),
        )
        for schema_on in (False, True)
    }
    selection = select_global_rerank_policy(
        gold_rows=gold_rows,
        candidates=candidates,
        score_sets=score_sets,
        thresholds=THRESHOLDS,
    )
    payload = {
        "schema_version": 1,
        "model_family": "onerel_pair_reranker_diagnostic",
        "dataset_role": "calibration_selection_used",
        "outer_accessed": False,
        "official_dev_accessed": False,
        "status": "passed_calibration_gate"
        if selection["metric"]["f1"] >= 0.45
        else "rejected_calibration_ablation",
        "calibration_gate": 0.45,
        "calibration_gate_passed": selection["metric"]["f1"] >= 0.45,
        "source_tree_sha256": args.source_tree_sha256,
        "calibration_data": str(args.calibration_data.resolve()),
        "calibration_data_sha256": sha256_file(args.calibration_data),
        "calibration_rows_sha256": _rows_sha256(rows),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "base_model_tree_sha256": base_model_sha,
        "onerel_checkpoint": str(args.onerel_checkpoint.resolve()),
        "onerel_checkpoint_sha256": sha256_file(args.onerel_checkpoint),
        "onerel_checkpoint_architecture_version": checkpoint.get(
            "architecture_version"
        ),
        "pair_model": str(args.pair_model.resolve()),
        "pair_model_tree_sha256": sha256_tree(args.pair_model.resolve()),
        "script_sha256": sha256_file(Path(__file__)),
        "n_rows": len(rows),
        "n_gold_spo": sum(len(row) for row in gold_rows),
        "n_surface_candidates": len(candidates),
        "proposal": {
            "relation_threshold": "off",
            "link_threshold": LINK_THRESHOLD,
            "boundary_threshold": "off",
            **proposal_audit,
            "candidate_oracle_recall": recall,
            "candidate_oracle_f1_at_precision_one": oracle_f1_at_precision_one,
            "candidate_surface_metric_without_filtering": candidate_oracle,
        },
        "selection": selection,
        "threshold_grid": list(THRESHOLDS),
        "selection_dimensions": ["global_candidate_threshold", "schema_on"],
        "per_predicate_thresholds": False,
        "device": "npu:0",
        "device_name": torch.npu.get_device_name(0),
        "fp16": bool(args.fp16),
        "elapsed_seconds": time.time() - started,
    }
    _create_once_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "candidate_oracle_recall": recall,
                "selected": selection,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if payload["calibration_gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
