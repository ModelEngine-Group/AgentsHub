"""Run same-checkpoint CPU/NPU parity and synchronized NPU energy capture."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.evidence import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate three-run CMeEE OM acceptance with power sampling"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--om", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--atc-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=200)
    parser.add_argument("--document-batch", type=int, default=64)
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--power-min-seconds", type=float, default=10.0)
    parser.add_argument("--cpu-batch", type=int, default=16)
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
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--idle-watts", type=float, required=True)
    parser.add_argument("--soc-version", default="Ascend910B2C")
    parser.add_argument("--cann-version", default="8.5.1")
    args = parser.parse_args()
    if (
        args.documents <= 0
        or args.document_batch <= 0
        or args.duration <= 0
        or args.power_min_seconds <= 0
        or args.cpu_batch <= 0
        or args.cpu_intra_op_threads < 0
    ):
        parser.error(
            "documents, batches, duration, and power-min-seconds must be positive"
        )
    if args.interval <= 0 or args.idle_watts < 0:
        parser.error("interval must be positive and idle-watts non-negative")
    if args.cpu_backend == "onnx" and (
        args.cpu_onnx is None or not args.cpu_onnx.is_file()
    ):
        parser.error("cpu-backend=onnx requires an existing --cpu-onnx")
    if (
        args.cpu_dependency_path is not None
        and not args.cpu_dependency_path.is_dir()
    ):
        parser.error("cpu-dependency-path must be an existing directory")
    if (
        args.cpu_quality_evidence is not None
        and not args.cpu_quality_evidence.is_file()
    ):
        parser.error("cpu-quality-evidence must be an existing file")

    output_dir = args.output_dir.resolve()
    benchmark = output_dir / "cmeee_cpu_npu_acceptance.json"
    marker = output_dir / "cmeee_npu_power_marker.json"
    power_csv = output_dir / "cmeee_npu_power.csv"
    power_summary = output_dir / "cmeee_npu_power.summary.json"
    receipt = output_dir / "cmeee_npu_release_receipt.json"
    artifacts = (benchmark, marker, power_csv, power_summary, receipt)
    existing = [str(path) for path in artifacts if path.exists()]
    if existing:
        parser.error(f"acceptance outputs are create-once: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler_command = [
        sys.executable,
        str(ROOT / "scripts" / "sample_npu_power.py"),
        "--out",
        str(power_csv),
        "--summary",
        str(power_summary),
        "--marker",
        str(marker),
        "--duration",
        str(args.duration),
        "--interval",
        str(args.interval),
        "--device-index",
        str(args.device_index),
        "--idle-watts",
        str(args.idle_watts),
        "--measurement-state",
        "benchmark",
        "--stop-after-marker",
    ]
    benchmark_command = [
        sys.executable,
        str(ROOT / "scripts" / "bench_medical_ner_adapter_batch.py"),
        "--root",
        str(args.root.resolve()),
        "--om",
        str(args.om.resolve()),
        "--model",
        str(args.checkpoint.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--export-receipt",
        str(args.export_receipt.resolve()),
        "--atc-log",
        str(args.atc_log.resolve()),
        "--backend",
        "cmeee",
        "--output",
        str(benchmark),
        "--documents",
        str(args.documents),
        "--document-batch",
        str(args.document_batch),
        "--repeats",
        "3",
        "--power-marker",
        str(marker),
        "--power-min-seconds",
        str(args.power_min_seconds),
        "--cpu-batch",
        str(args.cpu_batch),
        "--cpu-backend",
        args.cpu_backend,
        "--cpu-intra-op-threads",
        str(args.cpu_intra_op_threads),
        "--cpu-reference",
        "--soc-version",
        args.soc_version,
        "--cann-version",
        args.cann_version,
    ]
    if args.om_threshold is not None:
        benchmark_command.extend(["--om-threshold", str(args.om_threshold)])
    if args.cpu_onnx:
        benchmark_command.extend(["--cpu-onnx", str(args.cpu_onnx.resolve())])
    if args.cpu_dependency_path:
        benchmark_command.extend([
            "--cpu-dependency-path",
            str(args.cpu_dependency_path.resolve()),
        ])
    if args.cpu_quality_evidence:
        benchmark_command.extend([
            "--cpu-quality-evidence",
            str(args.cpu_quality_evidence.resolve()),
        ])

    sampler = subprocess.Popen(sampler_command, cwd=str(ROOT))
    time.sleep(max(1.0, args.interval * 3))
    benchmark_run = subprocess.run(benchmark_command, cwd=str(ROOT), check=False)
    if benchmark_run.returncode != 0 and not marker.exists():
        sampler.terminate()
    try:
        sampler_returncode = sampler.wait(timeout=args.duration + 30)
    except subprocess.TimeoutExpired:
        sampler.terminate()
        sampler_returncode = sampler.wait(timeout=10)

    benchmark_payload = (
        json.loads(benchmark.read_text(encoding="utf-8"))
        if benchmark.is_file()
        else {"passed": False}
    )
    power_payload = (
        json.loads(power_summary.read_text(encoding="utf-8"))
        if power_summary.is_file()
        else {"passed": False}
    )
    payload = {
        "schema_version": 1,
        "kind": "cmeee_same_checkpoint_cpu_npu_energy_acceptance",
        "manifest_sha256": sha256_file(args.manifest),
        "om_sha256": sha256_file(args.om),
        "export_receipt_sha256": sha256_file(args.export_receipt),
        "benchmark_sha256": sha256_file(benchmark) if benchmark.is_file() else None,
        "power_csv_sha256": sha256_file(power_csv) if power_csv.is_file() else None,
        "power_summary_sha256": (
            sha256_file(power_summary) if power_summary.is_file() else None
        ),
        "sampler_returncode": sampler_returncode,
        "benchmark_returncode": benchmark_run.returncode,
        "npu_docs_per_second": benchmark_payload.get("microbatch", {}).get(
            "docs_per_second"
        ),
        "cpu_docs_per_second": (
            benchmark_payload.get("cpu_reference") or {}
        ).get("docs_per_second"),
        "npu_vs_cpu_speedup": benchmark_payload.get("npu_vs_cpu_speedup"),
        "exact_cpu_npu_document_matches": (
            benchmark_payload.get("cpu_reference") or {}
        ).get("exact_document_matches"),
        "cpu_npu_entity_parity": (
            benchmark_payload.get("cpu_reference") or {}
        ).get("entity_parity"),
        "official_quality_evidence_sha256": (
            sha256_file(args.cpu_quality_evidence)
            if args.cpu_quality_evidence
            else None
        ),
        "official_quality_evidence_passed": (
            benchmark_payload.get("cpu_reference") or {}
        ).get("official_quality_evidence_passed"),
        "joules_per_document": power_payload.get("joules_per_document"),
        "idle_subtracted_joules_per_document": power_payload.get(
            "idle_subtracted_joules_per_document"
        ),
        "power_coverage_ratio": power_payload.get("coverage_ratio"),
        "passed": bool(
            benchmark_run.returncode == 0
            and sampler_returncode == 0
            and benchmark_payload.get("passed")
            and power_payload.get("passed")
        ),
    }
    receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
