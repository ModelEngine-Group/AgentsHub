"""Evaluate fit-only surface schema on one frozen candidate reranker.

This development-only evaluator never reads outer or official-dev data.  It
regenerates (or validates and reuses) the frozen calibration proposals, scores
them once with an already selected checkpoint, and compares only two declared
policies: schema off and conservative fit-only schema on.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
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

from dki.evidence import sha256_file
from dki.nlp.cmeie_candidate_reranker import (
    PairRerankCandidate,
    apply_fit_surface_schema,
    build_fit_surface_schema,
)
from dki.nlp.cmeie_pair import MARKERS
from scripts.eval_cmeie_onerel_pair_reranker import (
    _base_model_tree_sha256,
    _generate_candidates,
    _gold,
    _jsonl,
    _load_onerel,
)
from scripts.train_cmeie_candidate_cross_encoder import (
    _apply_selected_proposal_policy,
    _build_span_pair_model,
    _encode_candidates,
    _select_threshold,
    _write_json_once,
)


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _coverage_cap(counts: Sequence[int], *, target: float = 0.995) -> int:
    if not counts or not 0.0 < target <= 1.0 or any(value < 1 for value in counts):
        raise ValueError("cardinality cap requires positive counts and coverage")
    total = sum(counts)
    for cap in range(1, max(counts) + 1):
        if sum(min(value, cap) for value in counts) / total >= target:
            return cap
    raise AssertionError("cardinality coverage cap is unreachable")


def _fit_cardinality_contract(
    rows: Sequence[dict[str, Any]], *, target: float = 0.995
) -> dict[str, Any]:
    pair_counts: list[int] = []
    subject_predicate_counts: list[int] = []
    for row in rows:
        by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
        by_subject_predicate: dict[tuple[str, str], set[str]] = defaultdict(set)
        for spo in row["spo_list"]:
            subject = str(spo["subject"])
            predicate = str(spo["predicate"])
            obj = str(spo["object"])
            by_pair[(subject, obj)].add(predicate)
            by_subject_predicate[(subject, predicate)].add(obj)
        pair_counts.extend(len(values) for values in by_pair.values())
        subject_predicate_counts.extend(
            len(values) for values in by_subject_predicate.values()
        )

    def summary(counts: Sequence[int]) -> dict[str, Any]:
        cap = _coverage_cap(counts, target=target)
        return {
            "groups": len(counts),
            "instances": sum(counts),
            "histogram": {
                str(key): value for key, value in sorted(Counter(counts).items())
            },
            "cap": cap,
            "target_coverage": target,
            "retained_instance_coverage": sum(
                min(value, cap) for value in counts
            )
            / sum(counts),
        }

    pair = summary(pair_counts)
    subject_predicate = summary(subject_predicate_counts)
    return {
        "selection_role": "fit_only_fixed_before_calibration_scoring",
        "pair_predicate": pair,
        "subject_predicate_object": subject_predicate,
    }


def _apply_group_cardinality(
    *,
    candidates: Sequence[PairRerankCandidate],
    scores: Sequence[float],
    eligible_indices: Sequence[int],
    pair_predicate_cap: int,
    subject_predicate_object_cap: int,
) -> tuple[tuple[int, ...], dict[str, int]]:
    """Apply deterministic score-ranked caps learned only from fit counts."""

    if len(candidates) != len(scores):
        raise ValueError("candidate and score counts differ")
    if pair_predicate_cap < 1 or subject_predicate_object_cap < 1:
        raise ValueError("group cardinality caps must be positive")
    eligible = tuple(int(index) for index in eligible_indices)
    if len(set(eligible)) != len(eligible) or any(
        not 0 <= index < len(candidates) for index in eligible
    ):
        raise ValueError("eligible candidate indices are invalid")

    def rank(index: int) -> tuple[float, float, float, float, int]:
        candidate = candidates[index]
        return (
            -float(scores[index]),
            -float(candidate.relation_score),
            -float(candidate.link_score),
            -float(candidate.boundary_score),
            index,
        )

    by_pair: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for index in eligible:
        candidate = candidates[index]
        by_pair[(candidate.row_index, candidate.subject, candidate.object)].append(
            index
        )
    after_pair: list[int] = []
    for key in sorted(by_pair):
        after_pair.extend(
            sorted(by_pair[key], key=rank)[:pair_predicate_cap]
        )

    by_subject_predicate: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for index in after_pair:
        candidate = candidates[index]
        by_subject_predicate[
            (candidate.row_index, candidate.subject, candidate.predicate)
        ].append(index)
    retained: list[int] = []
    for key in sorted(by_subject_predicate):
        retained.extend(
            sorted(by_subject_predicate[key], key=rank)[
                :subject_predicate_object_cap
            ]
        )
    retained.sort()
    return tuple(retained), {
        "eligible_candidates": len(eligible),
        "after_pair_predicate_cap": len(after_pair),
        "retained_candidates": len(retained),
        "rejected_by_pair_predicate_cap": len(eligible) - len(after_pair),
        "rejected_by_subject_predicate_object_cap": len(after_pair)
        - len(retained),
    }


def _factorized_probabilities(
    *, outputs: Any, predicate_ids: Any, model: Any, torch: Any
) -> Any:
    """Return the fixed probability product implied by the three heads."""

    fused_logits = outputs.logits.reshape(-1).float()
    pair_logits = outputs.pair_logits.reshape(-1).float()
    relation_logits = outputs.relation_logits.float().gather(
        1, predicate_ids.reshape(-1, 1)
    ).reshape(-1)
    pair_scale = torch.sigmoid(model.pair_scale_logit).float()
    relation_scale = torch.sigmoid(model.relation_scale_logit).float()
    verifier_logits = (
        fused_logits
        - pair_scale * pair_logits
        - relation_scale * relation_logits
    )
    return (
        torch.sigmoid(verifier_logits)
        * torch.sigmoid(pair_logits)
        * torch.sigmoid(relation_logits)
    )


def _score_candidate_views(
    *,
    model: Any,
    tokenizer: Any,
    candidates: Sequence[PairRerankCandidate],
    device: Any,
    torch: Any,
    fp16: bool,
    batch_size: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    fused_scores: list[float] = []
    factorized_scores: list[float] = []
    model.eval()
    for start in range(0, len(candidates), batch_size):
        indices = list(range(start, min(start + batch_size, len(candidates))))
        encoded = _encode_candidates(
            tokenizer, candidates, indices, device, torch
        )
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=fp16,
        ):
            outputs = model(**encoded)
            probabilities = _factorized_probabilities(
                outputs=outputs,
                predicate_ids=encoded["predicate_ids"],
                model=model,
                torch=torch,
            )
            fused_probabilities = outputs.logits.reshape(-1).float().sigmoid()
        fused_scores.extend(fused_probabilities.cpu().tolist())
        factorized_scores.extend(probabilities.cpu().tolist())
    return tuple(fused_scores), tuple(factorized_scores)


def _candidate_payload(candidate: PairRerankCandidate) -> dict[str, Any]:
    return {
        "row_index": candidate.row_index,
        "predicate_id": candidate.predicate_id,
        "subject": candidate.subject,
        "predicate": candidate.predicate,
        "object": candidate.object,
        "marked_text": candidate.marked_text,
        "relation_score": candidate.relation_score,
        "link_score": candidate.link_score,
        "boundary_score": candidate.boundary_score,
    }


def _candidate_from_payload(payload: dict[str, Any]) -> PairRerankCandidate:
    required = {
        "row_index",
        "predicate_id",
        "subject",
        "predicate",
        "object",
        "marked_text",
        "relation_score",
        "link_score",
        "boundary_score",
    }
    if set(payload) != required:
        raise ValueError("candidate cache row fields differ from schema")
    candidate = PairRerankCandidate(
        row_index=int(payload["row_index"]),
        predicate_id=int(payload["predicate_id"]),
        subject=str(payload["subject"]),
        predicate=str(payload["predicate"]),
        object=str(payload["object"]),
        marked_text=str(payload["marked_text"]),
        relation_score=float(payload["relation_score"]),
        link_score=float(payload["link_score"]),
        boundary_score=float(payload["boundary_score"]),
    )
    if not all(
        math.isfinite(value)
        for value in (
            candidate.relation_score,
            candidate.link_score,
            candidate.boundary_score,
        )
    ):
        raise ValueError("candidate cache contains a non-finite score")
    return candidate


def _private_cache_path(path: Path) -> Path:
    absolute = path.resolve(strict=False)
    normalized = absolute.as_posix()
    if "/outputs/private/" not in normalized:
        raise ValueError("candidate cache must live under outputs/private")
    return absolute


def _write_candidate_cache_once(
    path: Path,
    *,
    binding: dict[str, Any],
    candidates: Sequence[PairRerankCandidate],
) -> str:
    path = _private_cache_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(
                    {
                        "kind": "header",
                        "schema_version": 1,
                        "binding": binding,
                        "n_candidates": len(candidates),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            for candidate in candidates:
                stream.write(
                    json.dumps(
                        _candidate_payload(candidate),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _load_candidate_cache(
    path: Path,
    *,
    expected_binding: dict[str, Any],
) -> tuple[list[PairRerankCandidate], str]:
    path = _private_cache_path(path).resolve(strict=True)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        first = stream.readline()
        if not first:
            raise ValueError("candidate cache is empty")
        header = json.loads(first)
        if header != {
            "kind": "header",
            "schema_version": 1,
            "binding": expected_binding,
            "n_candidates": header.get("n_candidates"),
        }:
            raise ValueError("candidate cache binding differs from request")
        candidates = [
            _candidate_from_payload(json.loads(line))
            for line in stream
            if line.strip()
        ]
    if len(candidates) != header.get("n_candidates"):
        raise ValueError("candidate cache count differs from header")
    return candidates, sha256_file(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onerel-base-model", type=Path, required=True)
    parser.add_argument("--onerel-checkpoint", type=Path, required=True)
    parser.add_argument("--cross-encoder-base-model", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposal-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_tree_sha256):
        raise SystemExit("source tree hash must be lowercase SHA-256")
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    started = time.time()

    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModel, AutoTokenizer

    if not torch.npu.is_available():
        raise SystemExit("candidate schema evaluation requires an Ascend NPU")
    torch.npu.set_device(0)
    device = torch.device("npu:0")

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if (
        selection.get("model_family") != "candidate_cross_encoder"
        or selection.get("dataset_role") != "calibration_selection_used"
        or selection.get("outer_accessed") is not False
        or selection.get("official_dev_accessed") is not False
    ):
        raise SystemExit("selection is not a calibration-only candidate model")
    if sha256_file(args.checkpoint) != selection.get("checkpoint_sha256"):
        raise SystemExit("candidate checkpoint hash differs from selection")
    if sha256_file(args.fit_data) != selection.get("fit_data_sha256"):
        raise SystemExit("fit data hash differs from selection")
    if sha256_file(args.calibration_data) != selection.get(
        "calibration_data_sha256"
    ):
        raise SystemExit("calibration data hash differs from selection")
    if sha256_file(args.onerel_checkpoint) != selection.get(
        "onerel_checkpoint_sha256"
    ):
        raise SystemExit("OneRel checkpoint hash differs from selection")
    cross_base_sha = _base_model_tree_sha256(args.cross_encoder_base_model)
    if cross_base_sha != selection.get("cross_encoder_base_model_tree_sha256"):
        raise SystemExit("cross-encoder base tree differs from selection")

    fit_rows = _jsonl(args.fit_data)
    calibration_rows = _jsonl(args.calibration_data)
    calibration_gold = [_gold(row) for row in calibration_rows]
    schema_contract = build_fit_surface_schema(fit_rows)
    schema_sha = _canonical_sha(schema_contract)
    cardinality_contract = _fit_cardinality_contract(fit_rows)
    cardinality_sha = _canonical_sha(cardinality_contract)

    policy = selection.get("calibration_proposal", {}).get(
        "global_proposal_policy"
    )
    if not isinstance(policy, dict):
        raise SystemExit("selection lacks the frozen calibration proposal policy")
    cache_binding = {
        "calibration_data_sha256": sha256_file(args.calibration_data),
        "onerel_checkpoint_sha256": sha256_file(args.onerel_checkpoint),
        "onerel_base_model_tree_sha256": _base_model_tree_sha256(
            args.onerel_base_model
        ),
        "proposal_policy_sha256": _canonical_sha(policy),
        "candidate_generator_sha256": sha256_file(
            ROOT / "scripts/eval_cmeie_onerel_pair_reranker.py"
        ),
        "candidate_contract_sha256": sha256_file(
            ROOT / "dki/nlp/cmeie_candidate_reranker.py"
        ),
    }
    if args.candidate_cache.exists():
        calibration_candidates, cache_sha = _load_candidate_cache(
            args.candidate_cache,
            expected_binding=cache_binding,
        )
        cache_status = "reused"
    else:
        onerel, onerel_tokenizer, predicates, _, _ = _load_onerel(
            checkpoint_path=args.onerel_checkpoint,
            base_model=args.onerel_base_model,
            device=device,
            torch=torch,
        )
        raw_candidates, _ = _generate_candidates(
            model=onerel,
            tokenizer=onerel_tokenizer,
            rows=calibration_rows,
            predicates=predicates,
            device=device,
            torch=torch,
            batch_size=args.proposal_batch_size,
            fp16=args.fp16,
            enforce_surface_cap=False,
        )
        calibration_candidates, _ = _apply_selected_proposal_policy(
            raw_candidates,
            n_rows=len(calibration_rows),
            policy=policy,
        )
        cache_sha = _write_candidate_cache_once(
            args.candidate_cache,
            binding=cache_binding,
            candidates=calibration_candidates,
        )
        cache_status = "created"
        del onerel, raw_candidates
        torch.npu.empty_cache()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture_version = int(checkpoint.get("architecture_version", 1))
    selection_architecture = int(
        selection.get("architecture_version", architecture_version)
    )
    if selection_architecture != architecture_version:
        raise SystemExit("candidate architecture version differs from selection")
    predicates = tuple(str(value) for value in checkpoint.get("predicates", ()))
    if not predicates:
        raise SystemExit("candidate checkpoint lacks predicates")
    tokenizer = AutoTokenizer.from_pretrained(
        args.cross_encoder_base_model,
        local_files_only=True,
        use_fast=True,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": list(MARKERS)})
    marker_token_ids = tuple(
        int(tokenizer.convert_tokens_to_ids(marker)) for marker in MARKERS
    )
    if marker_token_ids != tuple(checkpoint.get("marker_token_ids", ())):
        raise SystemExit("candidate marker token IDs differ from checkpoint")
    model = _build_span_pair_model(
        base_model=args.cross_encoder_base_model,
        marker_token_ids=marker_token_ids,
        n_predicates=len(predicates),
        torch=torch,
        auto_model=AutoModel,
        factor_head_version=architecture_version,
    )
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device)
    fused_scores, scores = _score_candidate_views(
        model=model,
        tokenizer=tokenizer,
        candidates=calibration_candidates,
        device=device,
        torch=torch,
        fp16=args.fp16,
        batch_size=args.eval_batch_size,
    )
    policy_scores = fused_scores if architecture_version >= 2 else scores
    unstructured_schema_off = _select_threshold(
        gold_rows=calibration_gold,
        candidates=calibration_candidates,
        scores=fused_scores,
    )
    unstructured_factorized_reference = _select_threshold(
        gold_rows=calibration_gold,
        candidates=calibration_candidates,
        scores=scores,
    )
    schema_retained, schema_audit = apply_fit_surface_schema(
        calibration_candidates, schema_contract
    )
    policies: list[dict[str, Any]] = []
    cardinality_audits: dict[str, dict[str, int]] = {}
    for schema_on, eligible in (
        (False, tuple(range(len(calibration_candidates)))),
        (True, schema_retained),
    ):
        retained, cardinality_audit = _apply_group_cardinality(
            candidates=calibration_candidates,
            scores=policy_scores,
            eligible_indices=eligible,
            pair_predicate_cap=int(
                cardinality_contract["pair_predicate"]["cap"]
            ),
            subject_predicate_object_cap=int(
                cardinality_contract["subject_predicate_object"]["cap"]
            ),
        )
        structured_candidates = [
            calibration_candidates[index] for index in retained
        ]
        structured_scores = [policy_scores[index] for index in retained]
        policies.append(
            {
                "schema_on": schema_on,
                **_select_threshold(
                    gold_rows=calibration_gold,
                    candidates=structured_candidates,
                    scores=structured_scores,
                ),
            }
        )
        cardinality_audits[str(schema_on).lower()] = cardinality_audit
    candidates = policies
    selected = max(
        candidates,
        key=lambda item: (
            item["metric"]["f1"],
            item["metric"]["precision"],
            item["threshold"],
            int(item["schema_on"]),
        ),
    )
    payload = {
        "schema_version": 1,
        "status": "passed" if selected["metric"]["f1"] >= 0.45 else "failed",
        "dataset_role": "calibration_schema_selection_used",
        "model_family": "candidate_cross_encoder",
        "architecture_version": architecture_version,
        "authoritative_score": (
            "sigmoid_additive_factor_fusion"
            if architecture_version >= 2
            else "legacy_three_head_probability_product"
        ),
        "source_tree_sha256": args.source_tree_sha256,
        "evaluated_checkpoint_source_tree_sha256": selection.get(
            "source_tree_sha256"
        ),
        "selection_sha256": sha256_file(args.selection),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fit_data_sha256": sha256_file(args.fit_data),
        "calibration_data_sha256": sha256_file(args.calibration_data),
        "fit_surface_schema_sha256": schema_sha,
        "fit_cardinality_contract_sha256": cardinality_sha,
        "fit_cardinality_contract": cardinality_contract,
        "candidate_cache_sha256": cache_sha,
        "candidate_cache_status": cache_status,
        "candidate_cache_binding": cache_binding,
        "n_candidates": len(calibration_candidates),
        "schema_audit": schema_audit,
        "cardinality_audits": cardinality_audits,
        "unstructured_schema_off_reference": unstructured_schema_off,
        "unstructured_factorized_reference": unstructured_factorized_reference,
        "candidate_score": (
            "sigmoid(exact_verifier_logit)*sigmoid(pair_existence_logit)*"
            "sigmoid(selected_relation_logit)"
        ),
        "policies": candidates,
        "selected": selected,
        "metric": selected["metric"],
        "selected_threshold": selected["threshold"],
        "schema_on": selected["schema_on"],
        "outer_accessed": False,
        "official_dev_accessed": False,
        "device": "npu:0",
        "device_name": torch.npu.get_device_name(0),
        "fp16": bool(args.fp16),
        "script_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    _write_json_once(args.output / "schema_selection.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "metric": payload["metric"],
                "schema_on": payload["schema_on"],
                "threshold": payload["selected_threshold"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
