"""Calibrate the FP16 OM threshold against FP32 ONNX on a frozen corpus slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.bench.npu.om_manifest import sha256_tree
from dki.evidence import sha256_file
from scripts.compare_cmeee_om_onnx import (
    _key,
    _metric,
    _write_create_once,
)


def _compare(reference_rows, candidate_rows):
    true_positive = false_positive = false_negative = 0
    exact_documents = 0
    for reference, candidate in zip(reference_rows, candidate_rows):
        reference_keys = {_key(entity) for entity in reference}
        candidate_keys = {_key(entity) for entity in candidate}
        true_positive += len(reference_keys & candidate_keys)
        false_positive += len(candidate_keys - reference_keys)
        false_negative += len(reference_keys - candidate_keys)
        exact_documents += reference_keys == candidate_keys
    return {
        **_metric(true_positive, false_positive, false_negative),
        "exact_documents": exact_documents,
        "document_match_rate": round(
            exact_documents / max(1, len(reference_rows)), 6,
        ),
    }


def _texts_sha256(texts):
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--om", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--atc-log", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-dependencies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--onnx-threads", type=int, default=24)
    parser.add_argument("--reference-threshold", type=float, default=-1.0)
    parser.add_argument(
        "--threshold-grid",
        default="-1.03,-1.02,-1.01,-1.005,-1.0,-0.995,-0.99,-0.98,-0.97",
    )
    args = parser.parse_args()
    try:
        threshold_grid = sorted({
            float(value.strip())
            for value in args.threshold_grid.split(",")
            if value.strip()
        })
    except ValueError as error:
        parser.error(f"invalid threshold-grid: {error}")
    if (
        args.documents <= 0
        or args.start_index < 0
        or args.onnx_threads <= 0
        or not threshold_grid
    ):
        parser.error("documents, threads, and threshold-grid must be valid")
    if args.output.exists():
        parser.error("output is a create-once artifact")

    sys.path.insert(0, str(args.onnx_dependencies.resolve()))
    from dki.bench.npu.cmeee_ner_infer import (
        CmeeeGlobalPointerOmBackend,
        CmeeeGlobalPointerOnnxBackend,
    )

    texts = [
        path.read_text(encoding="utf-8")
        for path in sorted((args.root / "data" / "raw").glob("emr_*.txt"))[
            args.start_index:args.start_index + args.documents
        ]
    ]
    if len(texts) != args.documents:
        parser.error("insufficient EMR documents")

    reference = CmeeeGlobalPointerOnnxBackend(
        model_dir=str(args.model),
        onnx_path=str(args.onnx),
        intra_op_threads=args.onnx_threads,
        threshold=args.reference_threshold,
    )
    try:
        reference_rows = reference.execute_batch(texts)
    finally:
        reference.close()

    backend = CmeeeGlobalPointerOmBackend(
        model_dir=str(args.model),
        om_path=str(args.om),
        manifest_path=str(args.manifest),
        export_receipt_path=str(args.export_receipt),
        atc_log_path=str(args.atc_log),
        threshold=min(threshold_grid),
    )
    started = time.perf_counter()
    try:
        minimum_threshold_rows = backend.execute_batch(texts)
    finally:
        backend.close()
    om_seconds = time.perf_counter() - started

    candidates = {}
    for threshold in threshold_grid:
        probability_cutoff = 1.0 / (1.0 + math.exp(-threshold))
        rows = [
            [
                entity
                for entity in entities
                if float(entity["confidence"]) > probability_cutoff
            ]
            for entities in minimum_threshold_rows
        ]
        candidates[str(threshold)] = {
            **_compare(reference_rows, rows),
            "docs_per_second": round(len(texts) / om_seconds, 3),
        }

    selected_threshold = max(
        threshold_grid,
        key=lambda threshold: (
            candidates[str(threshold)]["micro_f1"],
            candidates[str(threshold)]["precision"],
            -abs(threshold - args.reference_threshold),
        ),
    )
    payload = {
        "schema_version": 1,
        "kind": "cmeee_fp16_om_threshold_calibration",
        "selection_scope": "frozen_project_emr_calibration_slice",
        "official_cmeee_dev_used_for_selection": False,
        "start_index": args.start_index,
        "documents": len(texts),
        "texts_sha256": _texts_sha256(texts),
        "reference_backend": "fp32_onnx_cpu",
        "reference_threshold": args.reference_threshold,
        "threshold_grid": threshold_grid,
        "candidates": candidates,
        "selected_threshold": selected_threshold,
        "selected_metrics": candidates[str(selected_threshold)],
        "model_sha256": sha256_tree(args.model.resolve()),
        "om_sha256": sha256_file(args.om),
        "manifest_sha256": sha256_file(args.manifest),
        "onnx_sha256": sha256_file(args.onnx),
    }
    _write_create_once(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
