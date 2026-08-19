"""Select a CPU reference batch size for the frozen CMeEE checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.bench.npu.om_manifest import sha256_tree
from dki.evidence import sha256_file


def _canonical(rows):
    return [
        sorted(
            (
                entity.get("mention") or entity.get("name", ""),
                entity["type"],
                tuple(entity["span"]),
            )
            for entity in entities
        )
        for entities in rows
    ]


def _percentile(values, fraction):
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, round(fraction * (len(ordered) - 1))),
    )
    return ordered[index]


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
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=32)
    parser.add_argument("--batch-sizes", default="8,16,32")
    parser.add_argument("--warmup-documents", type=int, default=4)
    parser.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch")
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--intra-op-threads", type=int, default=0)
    args = parser.parse_args()
    try:
        batch_sizes = sorted({
            int(value.strip())
            for value in args.batch_sizes.split(",")
            if value.strip()
        })
    except ValueError as error:
        parser.error(f"invalid batch-sizes: {error}")
    if (
        args.documents <= 0
        or args.warmup_documents < 0
        or not batch_sizes
        or any(batch <= 0 for batch in batch_sizes)
        or args.intra_op_threads < 0
    ):
        parser.error("documents and batch sizes must be positive")
    if args.output.exists():
        parser.error("output is a create-once artifact")
    if args.backend == "onnx" and (
        args.onnx is None or not args.onnx.is_file()
    ):
        parser.error("backend=onnx requires an existing --onnx")

    texts = [
        path.read_text(encoding="utf-8")
        for path in sorted((args.root / "data" / "raw").glob("emr_*.txt"))[
            :args.documents
        ]
    ]
    if len(texts) != args.documents:
        parser.error("insufficient EMR documents")

    if args.backend == "onnx":
        from dki.bench.npu.cmeee_ner_infer import (
            CmeeeGlobalPointerOnnxBackend,
        )

        backend = CmeeeGlobalPointerOnnxBackend(
            model_dir=str(args.model),
            onnx_path=str(args.onnx),
            intra_op_threads=args.intra_op_threads,
        )
    else:
        from dki.bench.npu.cmeee_ner_infer import CmeeeGlobalPointerBackend

        backend = CmeeeGlobalPointerBackend(
            model_dir=str(args.model),
            device="cpu",
        )
    if args.warmup_documents:
        backend.extract(texts[:args.warmup_documents])

    runs = {}
    baseline = None
    for batch_size in batch_sizes:
        outputs = []
        latencies = []
        started = time.perf_counter()
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset:offset + batch_size]
            call_started = time.perf_counter()
            outputs.extend(backend.extract(batch))
            latencies.append((time.perf_counter() - call_started) * 1000)
        wall = time.perf_counter() - started
        canonical = _canonical(outputs)
        if baseline is None:
            baseline = canonical
        mismatches = [
            index
            for index, (left, right) in enumerate(zip(baseline, canonical))
            if left != right
        ]
        runs[str(batch_size)] = {
            "docs_per_second": round(len(outputs) / wall, 4),
            "wall_seconds": round(wall, 3),
            "batch_latency_ms": {
                "p50": round(_percentile(latencies, 0.5), 3),
                "p95": round(_percentile(latencies, 0.95), 3),
            },
            "documents": len(outputs),
            "mismatches_vs_first_batch": len(mismatches),
            "first_mismatch_indices": mismatches[:10],
        }

    eligible = [
        (int(batch), values["docs_per_second"])
        for batch, values in runs.items()
        if values["mismatches_vs_first_batch"] == 0
    ]
    selected_batch, selected_rate = max(eligible, key=lambda item: item[1])
    payload = {
        "schema_version": 1,
        "kind": "cmeee_cpu_batch_selection",
        "model": str(args.model.resolve()),
        "model_sha256": sha256_tree(args.model.resolve()),
        "backend": args.backend,
        "onnx": str(args.onnx.resolve()) if args.onnx else None,
        "onnx_sha256": sha256_file(args.onnx) if args.onnx else None,
        "documents": len(texts),
        "runs": runs,
        "selected_batch": selected_batch,
        "selected_docs_per_second": selected_rate,
        "selection_rule": "fastest exact-output-equivalent candidate",
        "passed": bool(eligible),
    }
    _write_create_once(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
