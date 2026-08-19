"""Run task-3 graph analysis CPU/NPU benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.analysis_ops import load_graph
from src.operators.npu_ops import benchmark_task3_analysis_ops
from src.pipelines.task2_kg_pipeline import run_task2_pipeline

DEFAULT_GRAPH = ROOT / "outputs" / "task2" / "medical_kg.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark task-3 analysis operators.")
    parser.add_argument("--graph-file", default=str(DEFAULT_GRAPH), help="Task-2 graph JSON path.")
    parser.add_argument("--question", default="哪些疾病关联最多症状？")
    parser.add_argument("--iterations", type=int, default=20)
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
    graph_path = Path(args.graph_file)
    if not graph_path.exists():
        bootstrap = run_task2_pipeline(output_dir=graph_path.parent)
        if bootstrap.status != "completed":
            print(json.dumps({"status": "failed", "message": bootstrap.message}, ensure_ascii=False))
            return 1
        graph_path = Path(bootstrap.artifacts["graph"]["output_path"])

    graph = load_graph(graph_path)
    report = benchmark_task3_analysis_ops(
        graph=graph,
        question=args.question,
        iterations=args.iterations,
        npu_probe=not args.skip_npu_probe,
        npu_probe_iterations=args.npu_probe_iterations,
        npu_probe_size=args.npu_probe_size,
    )
    report["input"]["path"] = str(graph_path)

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
