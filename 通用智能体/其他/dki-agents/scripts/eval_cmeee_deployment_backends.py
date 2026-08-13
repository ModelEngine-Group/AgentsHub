"""Evaluate the bound CMeEE OM and source ONNX on official CMeEE dev."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.bench.npu.om_manifest import sha256_tree
from dki.evidence import sha256_file


def _metric(counts):
    true_positive, false_positive, false_negative = counts
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "micro_f1": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def _gold(row):
    return {
        (
            entity["type"],
            int(entity["start"]),
            int(entity["end"]),
        )
        for entity in row.get("entities", [])
    }


def _predicted(entities):
    return {
        (
            entity.get("cmeee_type") or entity["type"],
            int(entity["span"][0]),
            int(entity["span"][1]),
        )
        for entity in entities
    }


def _evaluate(backend, rows, batch_size, types):
    counts = [0, 0, 0]
    per_type_counts = {entity_type: [0, 0, 0] for entity_type in types}
    started = time.perf_counter()
    documents = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        predictions = backend.execute_batch([row["text"] for row in batch])
        documents += len(predictions)
        for row, entities in zip(batch, predictions):
            gold = _gold(row)
            predicted = _predicted(entities)
            counts[0] += len(gold & predicted)
            counts[1] += len(predicted - gold)
            counts[2] += len(gold - predicted)
            for entity_type in types:
                gold_type = {item for item in gold if item[0] == entity_type}
                predicted_type = {
                    item for item in predicted if item[0] == entity_type
                }
                target = per_type_counts[entity_type]
                target[0] += len(gold_type & predicted_type)
                target[1] += len(predicted_type - gold_type)
                target[2] += len(gold_type - predicted_type)
    seconds = time.perf_counter() - started
    by_type = {
        entity_type: _metric(type_counts)
        for entity_type, type_counts in per_type_counts.items()
    }
    macro_f1 = sum(item["micro_f1"] for item in by_type.values()) / len(by_type)
    return {
        **_metric(counts),
        "macro_f1": round(macro_f1, 6),
        "by_type": by_type,
        "documents": documents,
        "seconds": round(seconds, 3),
        "docs_per_second": round(documents / seconds, 3),
    }


def _write_create_once(path: Path, payload) -> None:
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
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--om", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--atc-log", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-dependencies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--onnx-threads", type=int, default=24)
    parser.add_argument("--om-threshold", type=float, default=-1.005)
    parser.add_argument("--onnx-threshold", type=float, default=-1.0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.onnx_threads <= 0:
        parser.error("batch-size and onnx-threads must be positive")
    if args.output.exists():
        parser.error("output is a create-once artifact")
    if not args.data.is_file() or not args.onnx_dependencies.is_dir():
        parser.error("data and onnx-dependencies must exist")

    rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sys.path.insert(0, str(args.onnx_dependencies.resolve()))
    from dki.bench.npu.cmeee_ner_infer import (
        CMEEE_TYPES,
        CmeeeGlobalPointerOmBackend,
        CmeeeGlobalPointerOnnxBackend,
    )

    om_backend = CmeeeGlobalPointerOmBackend(
        model_dir=str(args.model),
        om_path=str(args.om),
        manifest_path=str(args.manifest),
        export_receipt_path=str(args.export_receipt),
        atc_log_path=str(args.atc_log),
        map_to_project=False,
        threshold=args.om_threshold,
    )
    try:
        om_metrics = _evaluate(
            om_backend, rows, args.batch_size, CMEEE_TYPES,
        )
    finally:
        om_backend.close()

    onnx_backend = CmeeeGlobalPointerOnnxBackend(
        model_dir=str(args.model),
        onnx_path=str(args.onnx),
        map_to_project=False,
        threshold=args.onnx_threshold,
        intra_op_threads=args.onnx_threads,
    )
    try:
        onnx_metrics = _evaluate(
            onnx_backend, rows, args.batch_size, CMEEE_TYPES,
        )
    finally:
        onnx_backend.close()

    delta = om_metrics["micro_f1"] - onnx_metrics["micro_f1"]
    payload = {
        "schema_version": 1,
        "benchmark": "CBLUE CMeEE-V2 deployment backend parity",
        "dataset_role": "official_dev_evaluation_only",
        "selection_used": False,
        "evaluation_scope": "exact_span_type",
        "n_dev": len(rows),
        "data_sha256": sha256_file(args.data),
        "model_sha256": sha256_tree(args.model.resolve()),
        "om_sha256": sha256_file(args.om),
        "manifest_sha256": sha256_file(args.manifest),
        "onnx_sha256": sha256_file(args.onnx),
        "om_threshold": args.om_threshold,
        "onnx_threshold": args.onnx_threshold,
        "om": om_metrics,
        "onnx": onnx_metrics,
        "om_minus_onnx_micro_f1": round(delta, 6),
        "absolute_micro_f1_delta": round(abs(delta), 6),
        "quality_gate": "absolute official-dev Micro-F1 delta <= 0.001",
        "passed": abs(delta) <= 0.001,
    }
    _write_create_once(args.output, payload)
    print(json.dumps({
        "n_dev": len(rows),
        "om": {
            key: om_metrics[key]
            for key in ("micro_f1", "precision", "recall", "docs_per_second")
        },
        "onnx": {
            key: onnx_metrics[key]
            for key in ("micro_f1", "precision", "recall", "docs_per_second")
        },
        "absolute_micro_f1_delta": payload["absolute_micro_f1_delta"],
        "passed": payload["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
