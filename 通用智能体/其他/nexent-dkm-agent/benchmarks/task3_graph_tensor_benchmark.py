"""Run Task-3 graph tensor CPU/NPU degree-centrality benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.npu_ops.graph_tensor_ops import (
    GRAPH_DEGREE_BENCHMARK_MODES,
    benchmark_graph_degree_centrality,
)


def parse_amortized_runs(value: str) -> list[int]:
    try:
        runs = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a comma-separated integer list") from exc
    if not runs:
        raise argparse.ArgumentTypeError("must not be empty")
    if any(run < 1 for run in runs):
        raise argparse.ArgumentTypeError("all run counts must be positive")
    return runs


def parse_benchmark_modes(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    if not modes:
        raise argparse.ArgumentTypeError("must not be empty")
    valid_modes = set(GRAPH_DEGREE_BENCHMARK_MODES) | {"all"}
    invalid = [mode for mode in modes if mode not in valid_modes]
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported benchmark mode(s): {', '.join(invalid)}")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Task-3 tensor degree centrality.")
    parser.add_argument("--nodes", type=int, default=1000, help="Synthetic graph node count.")
    parser.add_argument("--edges", type=int, default=10000, help="Synthetic graph edge count.")
    parser.add_argument("--iterations", type=int, default=20, help="Benchmark iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic graph random seed.")
    parser.add_argument("--prefer-device", default="auto", choices=["auto", "npu", "cuda", "cpu"])
    parser.add_argument(
        "--amortized-runs",
        type=parse_amortized_runs,
        default=[1, 2, 5, 10, 20],
        help="Comma-separated run counts for prepare-once NPU amortization.",
    )
    parser.add_argument(
        "--profile-breakdown",
        action="store_true",
        help="Include one step-by-step NPU graph degree breakdown profile.",
    )
    parser.add_argument(
        "--benchmark-modes",
        type=parse_benchmark_modes,
        default=None,
        help=(
            "Comma-separated optimization benchmark modes. Use 'all' for "
            f"{', '.join(GRAPH_DEGREE_BENCHMARK_MODES)}."
        ),
    )
    parser.add_argument(
        "--monitor-npu",
        action="store_true",
        help="Sample npu-smi AICore%%/utilization/power during the benchmark.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.5,
        help="npu-smi sampling interval in seconds (default 0.5).",
    )
    parser.add_argument(
        "--npu-id",
        type=int,
        default=None,
        help="npu-smi NPU id to sample (default: auto-detect).",
    )
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sampler = None
    if args.monitor_npu:
        from benchmarks.npu_monitor import NpuUtilizationSampler

        sampler = NpuUtilizationSampler(npu_id=args.npu_id, interval_s=args.monitor_interval)
        sampler.start()

    try:
        report = benchmark_graph_degree_centrality(
            node_count=args.nodes,
            edge_count=args.edges,
            iterations=args.iterations,
            seed=args.seed,
            prefer_device=args.prefer_device,
            amortized_runs=args.amortized_runs,
            profile_breakdown=args.profile_breakdown,
            benchmark_modes=args.benchmark_modes,
        )
    finally:
        if sampler is not None:
            sampler.stop()

    if sampler is not None:
        report["npu_utilization"] = sampler.result()

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
