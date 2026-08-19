"""Run task-2 KG operator CPU/NPU benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.npu_ops import benchmark_task2_kg_ops

DEFAULT_INPUT = ROOT / "data" / "samples" / "task2_medical_notes.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark task-2 KG operators.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Medical text input path.")
    parser.add_argument("--iterations", type=int, default=20, help="Benchmark iterations.")
    parser.add_argument(
        "--skip-npu-probe",
        action="store_true",
        help="Detect the NPU runtime without running the tensor execution probe.",
    )
    parser.add_argument(
        "--npu-probe-iterations",
        type=int,
        default=5,
        help="Iterations for the optional torch_npu runtime execution probe.",
    )
    parser.add_argument(
        "--npu-probe-size",
        type=int,
        default=64,
        help="Square matrix size for the optional torch_npu runtime execution probe.",
    )
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    text = source.read_text(encoding="utf-8")
    report = benchmark_task2_kg_ops(
        text,
        iterations=args.iterations,
        npu_probe=not args.skip_npu_probe,
        npu_probe_iterations=args.npu_probe_iterations,
        npu_probe_size=args.npu_probe_size,
    )
    report["input"]["path"] = str(source)

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
