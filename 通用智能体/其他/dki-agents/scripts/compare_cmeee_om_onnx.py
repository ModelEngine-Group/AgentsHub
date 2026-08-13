"""Compare CMeEE FP16 OM output with its FP32 ONNX source artifact."""

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


def _key(entity):
    return (
        entity.get("mention") or entity.get("name", ""),
        entity["type"],
        tuple(entity["span"]),
    )


def _metric(true_positive, false_positive, false_negative):
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
    parser.add_argument("--documents", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--onnx-threads", type=int, default=24)
    parser.add_argument("--om-threshold", type=float)
    parser.add_argument("--onnx-threshold", type=float)
    args = parser.parse_args()
    if args.documents <= 0 or args.start_index < 0 or args.onnx_threads <= 0:
        parser.error("documents and onnx-threads must be positive")
    if args.output.exists():
        parser.error("output is a create-once artifact")
    required_files = (
        args.om,
        args.manifest,
        args.export_receipt,
        args.atc_log,
        args.onnx,
    )
    if not all(path.is_file() for path in required_files):
        parser.error("all model evidence files must exist")
    if not args.model.is_dir() or not args.onnx_dependencies.is_dir():
        parser.error("model and onnx-dependencies must be directories")

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

    om_backend = CmeeeGlobalPointerOmBackend(
        model_dir=str(args.model),
        om_path=str(args.om),
        manifest_path=str(args.manifest),
        export_receipt_path=str(args.export_receipt),
        atc_log_path=str(args.atc_log),
        threshold=args.om_threshold,
    )
    onnx_backend = CmeeeGlobalPointerOnnxBackend(
        model_dir=str(args.model),
        onnx_path=str(args.onnx),
        intra_op_threads=args.onnx_threads,
        threshold=args.onnx_threshold,
    )
    try:
        started = time.perf_counter()
        om_rows = om_backend.execute_batch(texts)
        om_seconds = time.perf_counter() - started
        started = time.perf_counter()
        onnx_rows = onnx_backend.execute_batch(texts)
        onnx_seconds = time.perf_counter() - started
    finally:
        om_backend.close()
        onnx_backend.close()

    true_positive = false_positive = false_negative = 0
    exact_documents = 0
    differences = []
    for index, (om_entities, onnx_entities) in enumerate(
        zip(om_rows, onnx_rows)
    ):
        om_by_key = {_key(entity): entity for entity in om_entities}
        onnx_by_key = {_key(entity): entity for entity in onnx_entities}
        om_keys = set(om_by_key)
        onnx_keys = set(onnx_by_key)
        true_positive += len(om_keys & onnx_keys)
        false_positive += len(om_keys - onnx_keys)
        false_negative += len(onnx_keys - om_keys)
        if om_keys == onnx_keys:
            exact_documents += 1
            continue
        if len(differences) < 20:
            differences.append({
                "document_index": index,
                "om_only": [
                    {
                        "mention": key[0],
                        "type": key[1],
                        "span": list(key[2]),
                        "confidence": om_by_key[key].get("confidence"),
                    }
                    for key in sorted(om_keys - onnx_keys)
                ],
                "onnx_only": [
                    {
                        "mention": key[0],
                        "type": key[1],
                        "span": list(key[2]),
                        "confidence": onnx_by_key[key].get("confidence"),
                    }
                    for key in sorted(onnx_keys - om_keys)
                ],
            })

    payload = {
        "schema_version": 1,
        "kind": "cmeee_fp16_om_vs_fp32_onnx_quality",
        "documents": len(texts),
        "start_index": args.start_index,
        "om_threshold": om_backend.threshold,
        "onnx_threshold": onnx_backend.threshold,
        "exact_documents": exact_documents,
        "document_match_rate": round(exact_documents / len(texts), 6),
        "entity_parity": _metric(
            true_positive, false_positive, false_negative,
        ),
        "om_entities": sum(len(row) for row in om_rows),
        "onnx_entities": sum(len(row) for row in onnx_rows),
        "om_docs_per_second": round(len(texts) / om_seconds, 3),
        "onnx_docs_per_second": round(len(texts) / onnx_seconds, 3),
        "model_sha256": sha256_tree(args.model.resolve()),
        "om_sha256": sha256_file(args.om),
        "manifest_sha256": sha256_file(args.manifest),
        "onnx_sha256": sha256_file(args.onnx),
        "first_differences": differences,
        "passed": bool(
            len(om_rows) == len(onnx_rows) == len(texts)
            and _metric(
                true_positive, false_positive, false_negative,
            )["micro_f1"] >= 0.999
        ),
        "acceptance_rule": "entity-level OM-vs-ONNX micro-F1 >= 0.999",
    }
    _write_create_once(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
