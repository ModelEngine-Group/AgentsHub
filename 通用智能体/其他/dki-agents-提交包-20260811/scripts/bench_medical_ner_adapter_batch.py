#!/usr/bin/env python3
"""Replay the same EMR corpus through serial and cross-document OM batching.

This measures the production text->entities surface, not static tensor samples/s.
The candidate passes only when every document's entities exactly match serial.
"""
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


def _percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _canonical(entities):
    return sorted(
        (
            item.get("name") or item.get("mention", ""),
            item["type"],
            item.get("mention") or item.get("name", ""),
            tuple(item["span"]),
        )
        for item in entities
    )


def _mismatch_indices(left, right, expected):
    mismatches = {
        index
        for index, (left_item, right_item) in enumerate(zip(left, right))
        if _canonical(left_item) != _canonical(right_item)
    }
    if len(left) != expected or len(right) != expected:
        mismatches.update(
            range(min(len(left), len(right)), max(expected, len(left), len(right)))
        )
    return sorted(mismatches)


def _entity_parity(reference, candidate):
    true_positive = false_positive = false_negative = 0
    for reference_entities, candidate_entities in zip(reference, candidate):
        reference_set = set(_canonical(reference_entities))
        candidate_set = set(_canonical(candidate_entities))
        true_positive += len(reference_set & candidate_set)
        false_positive += len(candidate_set - reference_set)
        false_negative += len(reference_set - candidate_set)
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


def _write_json_create_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence artifact: {path}")
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


def _run_serial(adapter, texts):
    latencies, outputs = [], []
    started = time.perf_counter()
    for text in texts:
        item_started = time.perf_counter()
        outputs.append(adapter.execute(text))
        latencies.append((time.perf_counter() - item_started) * 1000)
    wall = time.perf_counter() - started
    return outputs, wall, latencies


def _run_batched(adapter, texts, document_batch):
    latencies, outputs = [], []
    started = time.perf_counter()
    for offset in range(0, len(texts), document_batch):
        batch = texts[offset:offset + document_batch]
        item_started = time.perf_counter()
        outputs.extend(adapter.execute_batch(batch))
        latencies.append((time.perf_counter() - item_started) * 1000)
    wall = time.perf_counter() - started
    return outputs, wall, latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/dki-agents"))
    parser.add_argument("--om", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--export-receipt", type=Path)
    parser.add_argument("--atc-log", type=Path)
    parser.add_argument("--backend", choices=["auto", "bert", "cmeee"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=200)
    parser.add_argument("--document-batch", type=int, default=64)
    parser.add_argument("--warmup-documents", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--cpu-batch",
        type=int,
        default=8,
        help="CPU reference batch size; independent from the static OM batch",
    )
    parser.add_argument(
        "--power-marker",
        type=Path,
        help="create-once marker around one extra batched run for synchronized power sampling",
    )
    parser.add_argument(
        "--power-min-seconds",
        type=float,
        default=10.0,
        help="repeat the power corpus until this synchronized measurement duration",
    )
    parser.add_argument(
        "--cpu-reference",
        action="store_true",
        help="run the same CMeEE checkpoint on CPU three times and compare exact entities",
    )
    parser.add_argument(
        "--cpu-backend",
        choices=["pytorch", "onnx"],
        default="pytorch",
    )
    parser.add_argument("--cpu-onnx", type=Path)
    parser.add_argument("--cpu-intra-op-threads", type=int, default=0)
    parser.add_argument("--cpu-dependency-path", type=Path)
    parser.add_argument("--cpu-quality-evidence", type=Path)
    parser.add_argument("--om-threshold", type=float)
    parser.add_argument("--soc-version", default="Ascend910B2C")
    parser.add_argument("--cann-version", default="8.5.1")
    args = parser.parse_args()
    if (
        args.document_batch <= 0
        or args.cpu_batch <= 0
        or args.documents <= 0
        or args.warmup_documents < 0
        or args.repeats < 3
        or args.power_min_seconds <= 0
        or args.cpu_intra_op_threads < 0
    ):
        parser.error(
            "documents/document-batch/cpu-batch must be positive, warmup must be "
            "non-negative, and repeats must be >= 3"
        )
    if args.output.exists():
        parser.error("output is a create-once evidence artifact")
    if args.power_marker and args.power_marker.exists():
        parser.error("power-marker is a create-once evidence artifact")
    if (
        args.power_marker
        and args.output.resolve() == args.power_marker.resolve()
    ):
        parser.error("output and power-marker must be different files")

    root = args.root
    om = args.om or root / "outputs/bench/npu_int8/ner_ft_bs64_seq256_int8.om"
    model = args.model or root / "models/medical_ner_bert"
    manifest = args.manifest or root / "outputs/bench/npu_int8/ner_ft_bs64_seq256_int8.manifest.json"
    if not om.is_file() or not manifest.is_file() or not model.is_dir():
        parser.error("OM, manifest, and model/checkpoint directory must exist")
    try:
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"invalid OM manifest: {exc}")
    backend = args.backend
    if backend == "auto":
        backend = (
            "cmeee"
            if manifest_payload.get("model_family") == "cmeee_globalpointer"
            else "bert"
        )
    texts = [
        path.read_text(encoding="utf-8")
        for path in sorted((root / "data/raw").glob("emr_*.txt"))[:args.documents]
    ]
    if len(texts) != args.documents:
        parser.error(
            f"requested {args.documents} documents but found only {len(texts)}"
        )
    if backend == "cmeee":
        checkpoint = model.resolve()
        evidence_paths = [args.output.resolve()]
        if args.power_marker:
            evidence_paths.append(args.power_marker.resolve())
        if any(checkpoint == path or checkpoint in path.parents for path in evidence_paths):
            parser.error("benchmark evidence must be outside the bound checkpoint tree")
    if args.cpu_reference and args.cpu_backend == "onnx":
        if args.cpu_onnx is None or not args.cpu_onnx.is_file():
            parser.error("cpu-backend=onnx requires an existing --cpu-onnx")
        if (
            args.cpu_dependency_path is not None
            and not args.cpu_dependency_path.is_dir()
        ):
            parser.error("cpu-dependency-path must be an existing directory")
    quality_evidence = None
    quality_evidence_valid = False
    if args.cpu_quality_evidence:
        if not args.cpu_quality_evidence.is_file():
            parser.error("cpu-quality-evidence must be an existing file")
        if args.cpu_onnx is None:
            parser.error("cpu-quality-evidence requires cpu-onnx")
        from dki.bench.npu.om_manifest import sha256_tree
        from dki.evidence import sha256_file

        quality_evidence = json.loads(
            args.cpu_quality_evidence.read_text(encoding="utf-8")
        )
        quality_evidence_valid = bool(
            quality_evidence.get("passed")
            and quality_evidence.get("om_sha256") == sha256_file(om)
            and quality_evidence.get("onnx_sha256") == sha256_file(args.cpu_onnx)
            and quality_evidence.get("model_sha256")
            == sha256_tree(model.resolve())
            and quality_evidence.get("absolute_micro_f1_delta", 1.0) <= 0.001
            and args.om_threshold is not None
            and quality_evidence.get("om_threshold") == args.om_threshold
        )
        if not quality_evidence_valid:
            parser.error(
                "cpu-quality-evidence failed artifact, threshold, or quality binding"
            )
    result = {
        "schema_version": 1,
        "probe": "medical_ner_adapter_same_corpus_serial_vs_microbatch",
        "n_docs": len(texts), "document_batch": args.document_batch,
        "repeats": args.repeats, "backend": backend,
        "om": str(om), "model": str(model),
        "formal_microbatch": backend == "cmeee",
        "cpu_batch": args.cpu_batch if args.cpu_reference else None,
        "cpu_backend": args.cpu_backend if args.cpu_reference else None,
        "cpu_onnx": str(args.cpu_onnx) if args.cpu_onnx else None,
        "om_threshold": args.om_threshold,
        "cpu_quality_evidence": (
            str(args.cpu_quality_evidence)
            if args.cpu_quality_evidence
            else None
        ),
        "cpu_quality_evidence_valid": quality_evidence_valid,
    }
    if backend == "cmeee":
        if not args.export_receipt or not args.atc_log:
            parser.error("cmeee backend requires --export-receipt and --atc-log")
        if not args.export_receipt.is_file() or not args.atc_log.is_file():
            parser.error("CMeEE export receipt and ATC log must exist")
        from dki.bench.npu.cmeee_ner_infer import CmeeeGlobalPointerOmBackend

        adapter = CmeeeGlobalPointerOmBackend(
            model_dir=str(model),
            om_path=str(om),
            manifest_path=str(manifest),
            export_receipt_path=str(args.export_receipt),
            atc_log_path=str(args.atc_log),
            threshold=args.om_threshold,
        )
    else:
        from dki.bench.npu.medical_ner_om_adapter import MedicalNerOmAdapter

        adapter = MedicalNerOmAdapter(
            omPath=str(om), modelName=str(model), seqLen=256, device=0, overlap=32,
            manifestPath=str(manifest) if manifest.is_file() else None,
            tokenizerPath=str(model / "tokenizer.json"), labelsPath=str(model / "config.json"),
            socVersion=args.soc_version, cannVersion=args.cann_version,
        )
        adapter.open()
    power_run = None
    try:
        for text in texts[:args.warmup_documents]:
            adapter.execute(text)
        runs = []
        for repeat in range(args.repeats):
            serial, serial_wall, serial_latencies = _run_serial(adapter, texts)
            batched, batch_wall, batch_latencies = _run_batched(
                adapter, texts, args.document_batch
            )
            mismatches = _mismatch_indices(serial, batched, len(texts))
            cardinality_ok = len(serial) == len(batched) == len(texts)
            runs.append({
                "repeat": repeat + 1,
                "serial": serial,
                "batched": batched,
                "serial_wall": serial_wall,
                "batch_wall": batch_wall,
                "serial_latencies": serial_latencies,
                "batch_latencies": batch_latencies,
                "cardinality_ok": cardinality_ok,
                "mismatches": mismatches,
            })
        if args.power_marker:
            from dki.evidence import sha256_file

            start_mono = time.monotonic()
            power_passes = 0
            power_documents = 0
            power_mismatch_union = set()
            power_cardinality_ok = True
            while True:
                power_outputs, _, _ = _run_batched(
                    adapter, texts, args.document_batch
                )
                power_passes += 1
                power_documents += len(power_outputs)
                power_cardinality_ok = (
                    power_cardinality_ok and len(power_outputs) == len(texts)
                )
                power_mismatch_union.update(
                    _mismatch_indices(
                        runs[0]["serial"], power_outputs, len(texts)
                    )
                )
                if time.monotonic() - start_mono >= args.power_min_seconds:
                    break
            end_mono = time.monotonic()
            power_mismatches = sorted(power_mismatch_union)
            power_run = {
                "start_mono": start_mono,
                "end_mono": end_mono,
                "duration_s": end_mono - start_mono,
                "n_docs": power_documents,
                "corpus_documents": len(texts),
                "passes": power_passes,
                "docs_per_second": power_documents / (end_mono - start_mono),
                "document_batch": args.document_batch,
                "backend": backend,
                "om_sha256": sha256_file(om),
                "cardinality_ok": power_cardinality_ok,
                "mismatches": len(power_mismatches),
                "first_mismatch_indices": power_mismatches[:10],
            }
            _write_json_create_once(args.power_marker, power_run)
    finally:
        adapter.close()

    serial_rates = [len(item["serial"]) / item["serial_wall"] for item in runs]
    batch_rates = [len(item["batched"]) / item["batch_wall"] for item in runs]
    serial_rate = statistics.median(serial_rates)
    batch_rate = statistics.median(batch_rates)
    mismatches = sorted({index for item in runs for index in item["mismatches"]})
    cardinality_ok = all(item["cardinality_ok"] for item in runs)
    serial_latencies = [value for item in runs for value in item["serial_latencies"]]
    batch_latencies = [value for item in runs for value in item["batch_latencies"]]
    serial = runs[0]["serial"]
    batched = runs[0]["batched"]
    cpu_reference = None
    if args.cpu_reference:
        if backend != "cmeee":
            parser.error("cpu-reference currently requires the CMeEE backend")
        if args.cpu_backend == "onnx":
            if args.cpu_dependency_path:
                sys.path.insert(0, str(args.cpu_dependency_path.resolve()))
            from dki.bench.npu.cmeee_ner_infer import (
                CmeeeGlobalPointerOnnxBackend,
            )

            cpu = CmeeeGlobalPointerOnnxBackend(
                model_dir=str(model),
                onnx_path=str(args.cpu_onnx),
                intra_op_threads=args.cpu_intra_op_threads,
            )
        else:
            from dki.bench.npu.cmeee_ner_infer import CmeeeGlobalPointerBackend

            cpu = CmeeeGlobalPointerBackend(model_dir=str(model), device="cpu")
        cpu.extract(texts[:args.warmup_documents])
        cpu_runs = []
        cpu_mismatch_union = set()
        for repeat in range(args.repeats):
            started = time.perf_counter()
            current = []
            latencies = []
            for offset in range(0, len(texts), args.cpu_batch):
                batch = texts[offset:offset + args.cpu_batch]
                call_started = time.perf_counter()
                current.extend(cpu.extract(batch))
                latencies.append((time.perf_counter() - call_started) * 1000)
            wall = time.perf_counter() - started
            run_mismatches = _mismatch_indices(current, batched, len(texts))
            cpu_mismatch_union.update(run_mismatches)
            cpu_runs.append({
                "repeat": repeat + 1,
                "docs_per_second": len(current) / wall,
                "batch_latencies": latencies,
                "cardinality_ok": len(current) == len(texts),
                "mismatches": len(run_mismatches),
            })
        cpu_mismatches = sorted(cpu_mismatch_union)
        cpu_entity_parity = _entity_parity(batched, current)
        cpu_cardinality_ok = all(item["cardinality_ok"] for item in cpu_runs)
        cpu_rates = [item["docs_per_second"] for item in cpu_runs]
        cpu_latencies = [
            value for item in cpu_runs for value in item["batch_latencies"]
        ]
        cpu_reference = {
            "backend": args.cpu_backend,
            "onnx": str(args.cpu_onnx) if args.cpu_onnx else None,
            "batch": args.cpu_batch,
            "cardinality_ok": cpu_cardinality_ok,
            "docs_per_second": round(statistics.median(cpu_rates), 3),
            "per_run_docs_per_second": [round(value, 3) for value in cpu_rates],
            "batch_latency_ms": {
                "p50": round(_percentile(cpu_latencies, 0.50), 3),
                "p95": round(_percentile(cpu_latencies, 0.95), 3),
            },
            "exact_document_matches": len(texts) - len(
                [index for index in cpu_mismatches if index < len(texts)]
            ),
            "mismatches": len(cpu_mismatches),
            "entity_parity": cpu_entity_parity,
            "official_quality_evidence_passed": quality_evidence_valid,
            "official_quality_evidence_sha256": (
                sha256_file(args.cpu_quality_evidence)
                if args.cpu_quality_evidence
                else None
            ),
            "first_mismatch_indices": cpu_mismatches[:10],
            "run_parity": [
                {
                    "repeat": item["repeat"],
                    "cardinality_ok": item["cardinality_ok"],
                    "mismatches": item["mismatches"],
                }
                for item in cpu_runs
            ],
        }
    result.update({
        "parity": {
            "cardinality_ok": cardinality_ok,
            "serial_documents": len(serial), "batch_documents": len(batched),
            "exact_document_matches": min(len(serial), len(batched)) - len([
                index for index in mismatches if index < min(len(serial), len(batched))
            ]),
            "mismatches": len(mismatches), "first_mismatch_indices": mismatches[:10],
            "total_serial_entities": sum(map(len, serial)),
            "total_batch_entities": sum(map(len, batched)),
        },
        "serial": {
            "docs_per_second": round(serial_rate, 3),
            "per_run_docs_per_second": [round(value, 3) for value in serial_rates],
            "document_latency_ms": {
                "mean": round(statistics.fmean(serial_latencies), 3),
                "p50": round(_percentile(serial_latencies, 0.50), 3),
                "p95": round(_percentile(serial_latencies, 0.95), 3),
            },
        },
        "microbatch": {
            "docs_per_second": round(batch_rate, 3),
            "per_run_docs_per_second": [round(value, 3) for value in batch_rates],
            "n_calls": len(batch_latencies),
            "batch_latency_ms": {
                "mean": round(statistics.fmean(batch_latencies), 3),
                "p50": round(_percentile(batch_latencies, 0.50), 3),
                "p95": round(_percentile(batch_latencies, 0.95), 3),
            },
        },
        "speedup": round(batch_rate / serial_rate, 4),
        "run_parity": [
            {
                "repeat": item["repeat"],
                "cardinality_ok": item["cardinality_ok"],
                "mismatches": len(item["mismatches"]),
            }
            for item in runs
        ],
        "power_run": power_run,
        "cpu_reference": cpu_reference,
        "npu_vs_cpu_speedup": (
            None
            if cpu_reference is None or cpu_reference["docs_per_second"] <= 0
            else round(batch_rate / cpu_reference["docs_per_second"], 4)
        ),
    })
    result["passed"] = (
        len(texts) == args.documents and cardinality_ok
        and not mismatches and batch_rate > serial_rate
        and (
            power_run is None
            or (power_run["cardinality_ok"] and power_run["mismatches"] == 0)
        )
        and (
            cpu_reference is None
            or (
                cpu_reference["cardinality_ok"]
                and (
                    cpu_reference["mismatches"] == 0
                    or quality_evidence_valid
                )
            )
        )
    )
    _write_json_create_once(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
