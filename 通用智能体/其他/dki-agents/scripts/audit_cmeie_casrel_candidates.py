"""Calibration-only predicate-aware candidate recall audit for CMeIE CasRel.

This audit deliberately uses gold subject spans.  It measures the maximum
recall available to a later span verifier before that verifier is trained; it
does not read outer or official-dev data and it never silently truncates dense
candidate sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"candidate audit output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    args.selection = args.selection.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.calibration_data = args.calibration_data.resolve()
    args.base_model = args.base_model.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise SystemExit(f"candidate audit output exists: {args.output}")
    if args.batch_size < 1:
        raise SystemExit("batch size must be positive")
    if not all(path.is_file() for path in (args.selection, args.checkpoint, args.calibration_data)):
        raise SystemExit("selection, checkpoint, and calibration data must be regular files")
    if not args.base_model.is_dir():
        raise SystemExit("base model directory is missing")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("kind") != "cmeie_casrel_calibration_selection":
        raise SystemExit("candidate audit requires a CasRel calibration selection")
    declared_checkpoint_sha = selection.get("checkpoint_sha256")
    if declared_checkpoint_sha != _sha256(args.checkpoint):
        raise SystemExit("checkpoint hash differs from calibration selection")
    predicates = selection.get("predicates")
    if not isinstance(predicates, list) or len(predicates) != 44 or len(set(predicates)) != 44:
        raise SystemExit("calibration selection must bind exactly 44 predicates")

    import torch
    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_casrel import (
        CandidateDensityError,
        CMeIECasRel,
        build_casrel_labels,
        predicate_object_candidates,
    )

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("candidate audit requires Ascend NPU")
    device = torch.device("npu:0")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    encoder = AutoModel.from_pretrained(args.base_model)
    model = CMeIECasRel(encoder, int(encoder.config.hidden_size), len(predicates)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise SystemExit("checkpoint does not contain a CasRel model state")
    missing, unexpected = model.load_state_dict(state, strict=False)
    disallowed_missing = [key for key in missing if not key.startswith("span_verifier.")]
    if disallowed_missing or unexpected:
        raise SystemExit(
            "checkpoint is incompatible with current CasRel architecture: "
            f"missing={disallowed_missing}, unexpected={list(unexpected)}"
        )
    model.eval()
    rows = _rows(args.calibration_data)
    predicate2id = {predicate: index for index, predicate in enumerate(predicates)}
    total_gold = 0
    exact_covered = 0
    per_predicate_gold: Counter[int] = Counter()
    per_predicate_covered: Counter[int] = Counter()
    boundary_errors = 0
    wrong_predicate = 0
    crossed_subject = 0
    self_candidates = 0
    density_rejections: list[dict[str, Any]] = []
    per_row_counts: list[int] = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer(
                [str(row["text"]) for row in batch],
                return_offsets_mapping=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=320,
            )
            offsets = [
                [tuple(pair) for pair in mapping.tolist()]
                for mapping in encoded.pop("offset_mapping")
            ]
            labels = [
                build_casrel_labels(row, offsets[index], predicate2id)
                for index, row in enumerate(batch)
            ]
            inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            flat_subjects: list[tuple[int, int]] = []
            parents: list[int] = []
            row_subjects: list[list[tuple[int, int]]] = []
            for index, label in enumerate(labels):
                subjects = sorted(label.object_spans)
                row_subjects.append(subjects)
                flat_subjects.extend(subjects)
                parents.extend([index] * len(subjects))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16):
                hidden = model.encode(**inputs)
                object_logits = None
                if flat_subjects:
                    object_logits = model.object_logits(
                        hidden,
                        torch.tensor(flat_subjects, dtype=torch.long, device=device),
                        torch.tensor(parents, dtype=torch.long, device=device),
                    ).float().cpu()
            cursor = 0
            for row_index, label in enumerate(labels):
                subjects = row_subjects[row_index]
                gold = {
                    (subject, predicate_id, object_start, object_end)
                    for subject, triples in label.object_spans.items()
                    for predicate_id, object_start, object_end in triples
                }
                total_gold += len(gold)
                for _, predicate_id, _, _ in gold:
                    per_predicate_gold[predicate_id] += 1
                if not subjects or object_logits is None:
                    per_row_counts.append(0)
                    cursor += len(subjects)
                    continue
                local_logits = object_logits[cursor : cursor + len(subjects)]
                cursor += len(subjects)
                try:
                    candidates = predicate_object_candidates(
                        local_logits,
                        offsets[row_index],
                        -0.5,
                        maximum_per_subject_predicate=8,
                        maximum_per_row=4096,
                    )
                except CandidateDensityError as exc:
                    density_rejections.append(
                        {"row_index": start + row_index, "reason": str(exc)}
                    )
                    per_row_counts.append(4097)
                    continue
                per_row_counts.append(len(candidates))
                candidate_set = {
                    (subjects[subject_index], predicate_id, object_start, object_end)
                    for subject_index, predicate_id, object_start, object_end, _ in candidates
                }
                exact = gold & candidate_set
                exact_covered += len(exact)
                for _, predicate_id, _, _ in exact:
                    per_predicate_covered[predicate_id] += 1
                for subject, predicate_id, object_start, object_end in gold - candidate_set:
                    same_subject_predicate = [
                        item
                        for item in candidate_set
                        if item[0] == subject and item[1] == predicate_id
                    ]
                    if same_subject_predicate:
                        boundary_errors += 1
                    if any(
                        item[0] == subject
                        and item[2] == object_start
                        and item[3] == object_end
                        and item[1] != predicate_id
                        for item in candidate_set
                    ):
                        wrong_predicate += 1
                    if any(
                        item[0] != subject
                        and item[1] == predicate_id
                        and item[2] == object_start
                        and item[3] == object_end
                        for item in candidate_set
                    ):
                        crossed_subject += 1
                for subject_index, _predicate_id, object_start, object_end, _ in candidates:
                    subject = subjects[subject_index]
                    if (object_start, object_end) == subject:
                        self_candidates += 1
    candidate_recall = exact_covered / max(1, total_gold)
    per_predicate = {
        predicates[predicate_id]: {
            "gold": per_predicate_gold[predicate_id],
            "covered": per_predicate_covered[predicate_id],
            "recall": per_predicate_covered[predicate_id]
            / max(1, per_predicate_gold[predicate_id]),
        }
        for predicate_id in range(len(predicates))
        if per_predicate_gold[predicate_id]
    }
    density = {
        "mean": sum(per_row_counts) / max(1, len(per_row_counts)),
        "p95": _percentile(per_row_counts, 0.95),
        "p99": _percentile(per_row_counts, 0.99),
        "max": max(per_row_counts, default=0),
        "rejections": len(density_rejections),
        "maximum_per_subject_predicate": 8,
        "maximum_per_row": 4096,
    }
    payload = {
        "schema_version": 1,
        "kind": "cmeie_casrel_predicate_aware_candidate_audit",
        "dataset_role": "calibration_diagnostic_only",
        "selection_used": True,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "object_threshold": -0.5,
        "gold_subjects": True,
        "candidate_recall": candidate_recall,
        "candidate_recall_counts": {"covered": exact_covered, "gold": total_gold},
        "density": density,
        "diagnostics": {
            "boundary_errors": boundary_errors,
            "wrong_predicate": wrong_predicate,
            "crossed_subject": crossed_subject,
            "self_candidates": self_candidates,
            "density_rejections": density_rejections,
            "per_predicate": per_predicate,
        },
        "gate": {
            "candidate_recall_gte_0_60": candidate_recall >= 0.60,
            "density_rejections_zero": not density_rejections,
            "passed": candidate_recall >= 0.60 and not density_rejections,
        },
        "bindings": {
            "selection_sha256": _sha256(args.selection),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "calibration_sha256": _sha256(args.calibration_data),
            "base_model": str(args.base_model),
            "source_tree_sha256": args.source_tree_sha256,
            "script_sha256": _sha256(Path(__file__)),
            "missing_new_verifier_keys": list(missing),
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    _write_json_once(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if payload["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
