"""Aggregate-only diagnostic for a failed CMeIE PRGC overfit preflight.

It is deliberately limited to the fixed first eight fit rows and writes no
text, offsets, gold triples, or per-row predictions.  The output distinguishes
an optimisation failure from an alignment, masking, label-index or decoder
failure before anyone considers another overfit attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"PRGC audit output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(values)
    def percentile(fraction: float) -> float:
        index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
        return float(ordered[index])
    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "p05": percentile(0.05),
        "median": percentile(0.50),
        "p95": percentile(0.95),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
    }


def _gold(row: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (str(spo.get("subject", "")), str(spo.get("predicate", "")), str(spo.get("object", "")))
        for spo in row.get("spo_list", [])
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.fit_data = args.fit_data.resolve()
    args.base_model = args.base_model.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise SystemExit(f"PRGC audit output exists: {args.output}")
    if not args.checkpoint.is_file() or not args.fit_data.is_file() or not args.base_model.is_dir():
        raise SystemExit("PRGC checkpoint/fit/base-model input is missing")

    import torch
    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_prgc import (
        CMeIEPRGC,
        PRGC_ARCHITECTURE_VERSION,
        PRGCDensityError,
        build_prgc_labels,
        decode_prgc,
    )

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("PRGC audit requires Ascend NPU")
    rows = [row for row in _jsonl(args.fit_data) if row.get("spo_list")][:8]
    if len(rows) != 8 or sum(len(row.get("spo_list", [])) for row in rows) != 44:
        raise SystemExit("PRGC audit requires the fixed 8 fit rows / 44 SPOs")
    predicates = sorted(
        {
            str(spo.get("predicate", ""))
            for row in rows
            for spo in row.get("spo_list", [])
            if spo.get("predicate")
        }
    )
    # The model always contains all fit predicates; construct the table from
    # the full fit corpus below so a subset cannot accidentally renumber IDs.
    all_rows = [row for row in _jsonl(args.fit_data) if row.get("spo_list")]
    all_predicates = sorted(
        {
            str(spo.get("predicate", ""))
            for row in all_rows
            for spo in row.get("spo_list", [])
            if spo.get("predicate")
        }
    )
    if len(all_predicates) != 44:
        raise SystemExit("PRGC audit expected 44 predicates")
    predicate2id = {value: index for index, value in enumerate(all_predicates)}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("model_family") != "prgc"
        or checkpoint.get("architecture_version") != PRGC_ARCHITECTURE_VERSION
        or checkpoint.get("predicates") != all_predicates
    ):
        raise SystemExit("PRGC checkpoint model family or predicate table differs")
    device = torch.device("npu:0")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)
    encoder = AutoModel.from_pretrained(args.base_model, local_files_only=True)
    model = CMeIEPRGC(encoder, int(encoder.config.hidden_size), len(all_predicates)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    relation_positive: list[float] = []
    span_positive = {"subject_start": [], "subject_end": [], "object_start": [], "object_end": []}
    correspondence_positive: list[float] = []
    active = {
        str(threshold): {"relation": 0, "span": 0, "correspondence": 0}
        for threshold in (-0.5, 0.0, 0.5)
    }
    decoded_counts = {
        (relation, span, correspondence, schema): {"tp": 0, "fp": 0, "fn": 0, "rejected_rows": 0}
        for relation, span, correspondence in itertools.product((-0.5, 0.0, 0.5), repeat=3)
        for schema in (False, True)
    }
    dropped = 0
    with torch.no_grad():
        for start in range(0, len(rows), 4):
            batch = rows[start : start + 4]
            encoded = tokenizer(
                [str(row["text"]) for row in batch],
                return_offsets_mapping=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=320,
            )
            offsets = [[tuple(pair) for pair in item.tolist()] for item in encoded.pop("offset_mapping")]
            labels = [build_prgc_labels(row, offsets[index], predicate2id) for index, row in enumerate(batch)]
            dropped += sum(item.dropped_spo for item in labels)
            inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            with torch.autocast(device_type="npu", dtype=torch.float16, enabled=args.fp16):
                output = model(**inputs)
            relation = output["relation"].float().cpu()
            span = output["span"].float().cpu().reshape(len(batch), len(all_predicates), -1, 4)
            correspondence = output["correspondence"].float().cpu().reshape(len(batch), len(all_predicates), -1, span.shape[2])
            for threshold in (-0.5, 0.0, 0.5):
                active[str(threshold)]["relation"] += int((relation > threshold).sum())
                active[str(threshold)]["span"] += int((span > threshold).sum())
                active[str(threshold)]["correspondence"] += int((correspondence > threshold).sum())
            for index, row in enumerate(batch):
                for predicate_id, triples in labels[index].triples.items():
                    relation_positive.append(float(relation[index, predicate_id]))
                    for subject_start, subject_end, object_start, object_end in triples:
                        span_positive["subject_start"].append(float(span[index, predicate_id, subject_start, 0]))
                        span_positive["subject_end"].append(float(span[index, predicate_id, subject_end, 1]))
                        span_positive["object_start"].append(float(span[index, predicate_id, object_start, 2]))
                        span_positive["object_end"].append(float(span[index, predicate_id, object_end, 3]))
                        correspondence_positive.append(float(correspondence[index, predicate_id, subject_start, object_start]))
                expected = _gold(row)
                for key, count in decoded_counts.items():
                    relation_threshold, span_threshold, correspondence_threshold, schema = key
                    try:
                        prediction = decode_prgc(
                            str(row["text"]), offsets[index], relation[index], span[index], correspondence[index],
                            all_predicates, (relation_threshold, span_threshold, correspondence_threshold),
                            schema_constraint=schema,
                        )
                    except PRGCDensityError:
                        count["rejected_rows"] += 1
                        continue
                    count["tp"] += len(prediction & expected)
                    count["fp"] += len(prediction - expected)
                    count["fn"] += len(expected - prediction)
    scored = []
    for key, count in decoded_counts.items():
        if count["rejected_rows"]:
            continue
        tp, fp, fn = count["tp"], count["fp"], count["fn"]
        scored.append((
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            },
            key,
        ))
    best_metric, best_key = max(scored, key=lambda item: (item[0]["f1"], item[0]["precision"], sum(item[1][:3]), item[1][3]))
    payload = {
        "schema_version": 1,
        "kind": "cmeie_prgc_overfit_aggregate_diagnostic",
        "architecture_version": PRGC_ARCHITECTURE_VERSION,
        "dataset_role": "fit_overfit_diagnostic_only",
        "selection_used": False,
        "calibration_accessed": False,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "rows": len(rows),
        "spo_count": sum(len(row.get("spo_list", [])) for row in rows),
        "dropped_spo": dropped,
        "best_declared_grid": {
            "relation_threshold": best_key[0],
            "span_threshold": best_key[1],
            "correspondence_threshold": best_key[2],
            "schema_constraint": "on" if best_key[3] else "off",
            "metric": best_metric,
        },
        "gold_logit_summary": {
            "relation": _summary(relation_positive),
            "subject_start": _summary(span_positive["subject_start"]),
            "subject_end": _summary(span_positive["subject_end"]),
            "object_start": _summary(span_positive["object_start"]),
            "object_end": _summary(span_positive["object_end"]),
            "correspondence": _summary(correspondence_positive),
        },
        "active_logit_counts": active,
        "density_rejected_grid_combinations": sum(1 for count in decoded_counts.values() if count["rejected_rows"]),
        "bindings": {
            "checkpoint_sha256": _sha(args.checkpoint),
            "fit_sha256": _sha(args.fit_data),
            "base_model_config_sha256": _sha(args.base_model / "config.json"),
            "source_tree_sha256": args.source_tree_sha256,
            "script_sha256": _sha(Path(__file__)),
        },
    }
    _write_once(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
