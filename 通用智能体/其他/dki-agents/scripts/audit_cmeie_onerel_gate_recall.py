"""Audit the gold-coordinate recall ceiling of a trained CMeIE OneRel model.

The audit is intentionally cheaper and more diagnostic than a full decode.  It
scores only aligned gold coordinates, then reports how much recall is retained
by the OneRel link score and by the optional relation/boundary proposal gates.
No outer or official-dev input is accepted by this entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


THRESHOLDS = (-0.5, 0.0, 0.5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _predicates(rows: Iterable[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(spo.get("predicate", ""))
            for row in rows
            for spo in row.get("spo_list", [])
            if str(spo.get("predicate", ""))
        }
    )


def gate_key(
    link_threshold: float,
    relation_threshold: float | None,
    boundary_threshold: float | None,
) -> str:
    relation = "off" if relation_threshold is None else f"{relation_threshold:.1f}"
    boundary = "off" if boundary_threshold is None else f"{boundary_threshold:.1f}"
    return f"link={link_threshold:.1f}|relation={relation}|boundary={boundary}"


def gate_passes(
    scores: tuple[float, float, float],
    *,
    link_threshold: float,
    relation_threshold: float | None,
    boundary_threshold: float | None,
) -> bool:
    """Return whether one exact gold mention survives the proposal gates."""

    link_score, relation_score, boundary_score = scores
    return (
        link_score > float(link_threshold)
        and (
            relation_threshold is None
            or relation_score > float(relation_threshold)
        )
        and (
            boundary_threshold is None
            or boundary_score > float(boundary_threshold)
        )
    )


def surface_gold_coordinates(
    row: dict[str, Any],
    offsets: Sequence[tuple[int, int]],
    predicate2id: dict[str, int],
) -> dict[tuple[str, str, str], tuple[tuple[int, int, int, int, int], ...]]:
    """Map every surface SPO to all deterministic aligned mention coordinates."""

    from dki.nlp.cmeie_spo import (
        char_span_to_tokens,
        select_mention_pairs,
    )

    text = str(row.get("text", ""))
    result: dict[
        tuple[str, str, str], set[tuple[int, int, int, int, int]]
    ] = {}
    offset_list = list(offsets)
    for spo in row.get("spo_list", []):
        subject = str(spo.get("subject", ""))
        predicate = str(spo.get("predicate", ""))
        obj = str(spo.get("object", ""))
        surface = (subject, predicate, obj)
        coordinates = result.setdefault(surface, set())
        relation = predicate2id.get(predicate)
        if relation is None:
            continue
        for subject_chars, object_chars in select_mention_pairs(
            text,
            subject,
            obj,
            allow_cross_sentence_fallback=True,
        ):
            subject_tokens = char_span_to_tokens(offset_list, subject_chars)
            object_tokens = char_span_to_tokens(offset_list, object_chars)
            if subject_tokens is None or object_tokens is None:
                continue
            ss, se = subject_tokens
            os_, oe = object_tokens
            coordinates.add((relation, ss, se, os_, oe))
    return {
        surface: tuple(sorted(coordinates))
        for surface, coordinates in sorted(result.items())
    }


def _grid() -> tuple[tuple[float, float | None, float | None], ...]:
    optional = (None, *THRESHOLDS)
    return tuple(
        (link, relation, boundary)
        for link in THRESHOLDS
        for relation in optional
        for boundary in optional
    )


def summarize_gate_recall(
    rows: Sequence[
        tuple[
            str,
            dict[
                tuple[str, str, str],
                Sequence[tuple[float, float, float]],
            ],
        ]
    ],
) -> dict[str, Any]:
    """Aggregate surface-level candidate recall for the declared gate grid."""

    combinations = _grid()
    covered = {gate_key(*combination): 0 for combination in combinations}
    totals = {key: 0 for key in covered}
    unaligned = 0
    per_predicate: dict[str, dict[str, list[int]]] = {
        f"{threshold:.1f}": {} for threshold in THRESHOLDS
    }
    for _row_id, surfaces in rows:
        for surface, observations in surfaces.items():
            predicate = surface[1]
            if not observations:
                unaligned += 1
            for combination in combinations:
                key = gate_key(*combination)
                totals[key] += 1
                if any(
                    gate_passes(
                        observation,
                        link_threshold=combination[0],
                        relation_threshold=combination[1],
                        boundary_threshold=combination[2],
                    )
                    for observation in observations
                ):
                    covered[key] += 1
            for threshold in THRESHOLDS:
                bucket = per_predicate[f"{threshold:.1f}"].setdefault(
                    predicate, [0, 0]
                )
                bucket[1] += 1
                if any(observation[0] > threshold for observation in observations):
                    bucket[0] += 1
    grid: dict[str, dict[str, float | int]] = {}
    for key in sorted(covered):
        total = totals[key]
        recall = covered[key] / max(1, total)
        grid[key] = {
            "covered": covered[key],
            "total": total,
            "recall": recall,
            "oracle_f1_at_precision_1": 2.0 * recall / max(1e-12, 1.0 + recall),
        }
    predicate_output = {
        threshold: {
            predicate: {
                "covered": values[0],
                "total": values[1],
                "recall": values[0] / max(1, values[1]),
            }
            for predicate, values in sorted(predicates.items())
        }
        for threshold, predicates in sorted(per_predicate.items())
    }
    best = max(
        grid.items(),
        key=lambda item: (float(item[1]["recall"]), item[0]),
    )
    return {
        "surface_gold": next(iter(totals.values()), 0),
        "unaligned_surface_gold": unaligned,
        "grid_size": len(grid),
        "grid": grid,
        "best_gate_recall": {"configuration": best[0], **best[1]},
        "link_only_per_predicate": predicate_output,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--pair-rank", type=int, default=192)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if len(args.source_tree_sha256) != 64:
        raise SystemExit("source tree SHA-256 must contain 64 hexadecimal characters")
    if args.batch_size < 1 or args.max_length != 320 or args.pair_rank != 192:
        raise SystemExit("formal audit dimensions are batch>=1, max_length=320, pair_rank=192")

    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_onerel import (
        CMeIEOneRel,
        HB_TB,
        HB_TE,
        HE_TE,
        inject_bert_attention_lora,
    )

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("gate recall audit requires an Ascend NPU")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    fit = _jsonl(args.fit_data)
    calibration = _jsonl(args.calibration_data)
    predicates = _predicates(fit)
    if len(predicates) != 44:
        raise SystemExit(f"expected 44 fit predicates, received {len(predicates)}")
    predicate2id = {predicate: index for index, predicate in enumerate(predicates)}

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True
    )
    encoder = AutoModel.from_pretrained(args.base_model, local_files_only=True)
    inject_bert_attention_lora(
        encoder, rank=args.lora_rank, alpha=args.lora_alpha
    )
    model = CMeIEOneRel(
        encoder,
        int(encoder.config.hidden_size),
        len(predicates),
        pair_rank=args.pair_rank,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model_family") != "onerel":
        raise SystemExit("checkpoint is not a OneRel model")
    if list(checkpoint.get("predicates", [])) != predicates:
        raise SystemExit("checkpoint predicate table differs from fit")
    missing, unexpected = model.load_state_dict(
        checkpoint.get("model_delta", {}), strict=False
    )
    migration_missing = (
        {"proposal_calibrator.weight", "proposal_calibrator.bias"}
        if int(checkpoint.get("architecture_version", -1)) == 2
        else set()
    )
    illegal_missing = [
        name
        for name in missing
        if (
            not name.startswith("encoder.")
            or name.endswith(".a.weight")
            or name.endswith(".b.weight")
        )
        and name not in migration_missing
    ]
    if illegal_missing or unexpected:
        raise SystemExit(
            f"checkpoint delta mismatch: missing={illegal_missing}, unexpected={unexpected}"
        )
    model.to(device).eval()
    model.encoder.eval()

    observed_rows = []
    for start in range(0, len(calibration), args.batch_size):
        batch = calibration[start : start + args.batch_size]
        encoded = tokenizer(
            [str(row.get("text", "")) for row in batch],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = [
            [(int(left), int(right)) for left, right in row]
            for row in encoded.pop("offset_mapping").tolist()
        ]
        inputs = {name: value.to(device) for name, value in encoded.items()}
        surface_rows = [
            surface_gold_coordinates(row, row_offsets, predicate2id)
            for row, row_offsets in zip(batch, offsets)
        ]
        coordinate_rows = [
            sorted({coordinate for values in row.values() for coordinate in values})
            for row in surface_rows
        ]
        parents: list[int] = []
        relations: list[int] = []
        tags: list[int] = []
        lefts: list[int] = []
        rights: list[int] = []
        flat_coordinates: list[tuple[int, tuple[int, int, int, int, int]]] = []
        for parent, coordinates in enumerate(coordinate_rows):
            for relation, ss, se, os_, oe in coordinates:
                flat_coordinates.append((parent, (relation, ss, se, os_, oe)))
                for tag, left, right in (
                    (HB_TB, ss, os_),
                    (HB_TE, ss, oe),
                    (HE_TE, se, oe),
                ):
                    parents.append(parent)
                    relations.append(relation)
                    tags.append(tag)
                    lefts.append(left)
                    rights.append(right)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=True
        ):
            hidden = model.encode(**inputs)
            relation_logits, boundary_logits = model.auxiliary_logits(
                hidden, inputs["attention_mask"]
            )
            if parents:
                link_logits = model.score_sparse(
                    hidden,
                    torch.tensor(parents, dtype=torch.long, device=device),
                    torch.tensor(relations, dtype=torch.long, device=device),
                    torch.tensor(tags, dtype=torch.long, device=device),
                    torch.tensor(lefts, dtype=torch.long, device=device),
                    torch.tensor(rights, dtype=torch.long, device=device),
                ).reshape(-1, 3)
            else:
                link_logits = torch.empty((0, 3), device=device)
        relation_cpu = relation_logits.float().cpu().tolist()
        boundary_cpu = boundary_logits.float().cpu().tolist()
        link_cpu = link_logits.float().cpu().tolist()
        coordinate_scores: list[dict[tuple[int, int, int, int, int], tuple[float, float, float]]] = [
            {} for _ in batch
        ]
        for (parent, coordinate), link_values in zip(flat_coordinates, link_cpu):
            relation, ss, se, os_, oe = coordinate
            coordinate_scores[parent][coordinate] = (
                min(float(value) for value in link_values),
                float(relation_cpu[parent][relation]),
                min(
                    float(boundary_cpu[parent][ss][0]),
                    float(boundary_cpu[parent][se][1]),
                    float(boundary_cpu[parent][os_][2]),
                    float(boundary_cpu[parent][oe][3]),
                ),
            )
        for offset, (row, surfaces, scores) in enumerate(
            zip(batch, surface_rows, coordinate_scores)
        ):
            observed_rows.append(
                (
                    str(row.get("id", start + offset)),
                    {
                        surface: [scores[coordinate] for coordinate in coordinates]
                        for surface, coordinates in surfaces.items()
                    },
                )
            )

    summary = summarize_gate_recall(observed_rows)
    payload = {
        "schema_version": 1,
        "kind": "cmeie_onerel_gold_coordinate_gate_recall_audit",
        "dataset_role": "calibration_diagnostic_selection_used",
        "fit_sha256": _sha256(args.fit_data),
        "calibration_sha256": _sha256(args.calibration_data),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "base_model_tree_sha256": checkpoint.get("base_model_tree_sha256"),
        "source_tree_sha256": args.source_tree_sha256,
        "script_sha256": _sha256(Path(__file__)),
        "device": "npu:0",
        "device_name": torch.npu.get_device_name(0),
        "rows": len(calibration),
        "thresholds": list(THRESHOLDS),
        "outer_accessed": False,
        "official_dev_accessed": False,
        **summary,
    }
    _write_json_once(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
