"""Benchmark Task-3 centrality integration with optional NPU top-hubs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.analysis_ops.graph_analytics import compute_centrality
from src.operators.npu_ops.graph_tensor_ops import (
    generate_synthetic_graph,
    generate_synthetic_graph_multi_type,
    prepare_graph_degree_tensor_cache,
)

CENTRALITY_BENCHMARK_MODES = (
    "cpu_compute_centrality",
    "optimized_uncached_compute_centrality",
    "cached_npu_top_hubs_cpu_type_centrality",
)


def benchmark_task3_centrality_integration(
    node_count: int = 1000,
    edge_count: int = 10000,
    iterations: int = 20,
    seed: int = 42,
    prefer_device: str = "auto",
    amortized_runs: list[int] | tuple[int, ...] = (1, 2, 5, 10, 20),
    multi_type: bool = False,
) -> dict[str, Any]:
    """Compare CPU centrality with the optimized NPU-top-hubs centrality path."""

    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    if multi_type:
        graph = generate_synthetic_graph_multi_type(
            node_count=node_count, edge_count=edge_count, seed=seed,
        )
    else:
        graph = generate_synthetic_graph(node_count=node_count, edge_count=edge_count, seed=seed)

    cpu_warmup = compute_centrality(graph, prefer_device="cpu")
    cpu_durations = []
    cpu_last = cpu_warmup
    for _ in range(iterations):
        started = time.perf_counter()
        cpu_last = compute_centrality(graph, prefer_device="cpu")
        cpu_durations.append(time.perf_counter() - started)

    optimized_warmup = compute_centrality(graph, prefer_device=prefer_device)
    optimized_durations = []
    optimized_last = optimized_warmup
    for _ in range(iterations):
        started = time.perf_counter()
        optimized_last = compute_centrality(graph, prefer_device=prefer_device)
        optimized_durations.append(time.perf_counter() - started)

    cache_started = time.perf_counter()
    degree_tensor_cache = prepare_graph_degree_tensor_cache(graph, prefer_device=prefer_device)
    cache_prepare_ms = round((time.perf_counter() - cache_started) * 1000, 4)
    cached_warmup = compute_centrality(
        graph,
        prefer_device=prefer_device,
        degree_tensor_cache=degree_tensor_cache,
    )
    cached_durations = []
    cached_last = cached_warmup
    for _ in range(iterations):
        started = time.perf_counter()
        cached_last = compute_centrality(
            graph,
            prefer_device=prefer_device,
            degree_tensor_cache=degree_tensor_cache,
        )
        cached_durations.append(time.perf_counter() - started)

    cpu_metrics = {
        **_timing_metrics(cpu_durations, iterations, edge_count),
        "status": cpu_last.get("status"),
        "backend": "python",
        "top_hubs_backend": cpu_last.get("top_hubs_backend"),
        "top_degree": _top_degree(cpu_last),
    }
    optimized_metrics = {
        **_timing_metrics(optimized_durations, iterations, edge_count),
        "status": optimized_last.get("status"),
        "backend": optimized_last.get("top_hubs_backend"),
        "top_hubs_backend": optimized_last.get("top_hubs_backend"),
        "top_hubs_npu_reason": optimized_last.get("top_hubs_npu_reason"),
        "top_degree": _top_degree(optimized_last),
    }
    cached_metrics = {
        **_timing_metrics(cached_durations, iterations, edge_count),
        "status": cached_last.get("status"),
        "backend": cached_last.get("top_hubs_backend"),
        "top_hubs_backend": cached_last.get("top_hubs_backend"),
        "top_hubs_npu_reason": cached_last.get("top_hubs_npu_reason"),
        "top_degree": _top_degree(cached_last),
        "cache_status": degree_tensor_cache.get("status", "unavailable"),
        "cache_backend": degree_tensor_cache.get("backend"),
        "cache_device": degree_tensor_cache.get("device"),
        "cache_prepare_ms": cache_prepare_ms,
    }
    speedup = None
    if optimized_metrics["latency_ms_avg"] and cpu_metrics["latency_ms_avg"] is not None:
        speedup = round(cpu_metrics["latency_ms_avg"] / optimized_metrics["latency_ms_avg"], 4)
    cached_speedup = None
    if cached_metrics["latency_ms_avg"] and cpu_metrics["latency_ms_avg"] is not None:
        cached_speedup = round(cpu_metrics["latency_ms_avg"] / cached_metrics["latency_ms_avg"], 4)

    # Collect type_centrality info from results
    cpu_type_count = len(cpu_last.get("type_centrality", {}))
    cached_type_count = len(cached_last.get("type_centrality", {}))

    return {
        "task": "task3_centrality_integration",
        "graph": {
            "node_count": node_count,
            "edge_count": edge_count,
            "seed": seed,
            "multi_type": multi_type,
            "type_count": cpu_type_count,
            "cached_type_count": cached_type_count,
        },
        "cpu_compute_centrality": cpu_metrics,
        "optimized_compute_centrality": optimized_metrics,
        "cached_npu_top_hubs_cpu_type_centrality": cached_metrics,
        "correctness": _compare_centrality_results(cpu_last, optimized_last),
        "cached_correctness": _compare_centrality_results(cpu_last, cached_last),
        "speedup": speedup,
        "cached_speedup": cached_speedup,
        "amortized_cached_centrality": _amortized_metrics(
            cpu_latency_ms=cpu_metrics["latency_ms_avg"],
            cached_latency_ms=cached_metrics["latency_ms_avg"],
            cache_prepare_ms=cache_prepare_ms,
            runs=amortized_runs,
        ),
        "notes": [
            "CPU baseline forces prefer_device=cpu.",
            "Optimized path uses compute_centrality(prefer_device) without an external tensor cache.",
            "Cached path prepares the NPU degree tensor cache once, then reuses it across benchmark iterations.",
            "When NPU is unavailable, optimized and cached paths fall back to CPU and speedup should be interpreted as fallback overhead.",
        ],
    }


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
    valid_modes = set(CENTRALITY_BENCHMARK_MODES) | {"all"}
    invalid = [mode for mode in modes if mode not in valid_modes]
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported benchmark mode(s): {', '.join(invalid)}")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Task-3 centrality integration.")
    parser.add_argument("--nodes", type=int, default=1000, help="Synthetic graph node count.")
    parser.add_argument("--edges", type=int, default=10000, help="Synthetic graph edge count.")
    parser.add_argument("--iterations", type=int, default=20, help="Benchmark iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic graph random seed.")
    parser.add_argument("--prefer-device", default="auto", choices=["auto", "npu", "cuda", "cpu"])
    parser.add_argument(
        "--amortized-runs",
        type=parse_amortized_runs,
        default=[1, 2, 5, 10, 20],
        help="Comma-separated run counts for prepare-once cached centrality amortization.",
    )
    parser.add_argument(
        "--benchmark-modes",
        type=parse_benchmark_modes,
        default=["all"],
        help=(
            "Accepted for parity with graph tensor benchmarks. Current report "
            "always includes cpu, uncached optimized, and cached modes."
        ),
    )
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    parser.add_argument(
        "--multi-type",
        action="store_true",
        help="Use multi-type medical KG nodes instead of single Synthetic type.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = benchmark_task3_centrality_integration(
        node_count=args.nodes,
        edge_count=args.edges,
        iterations=args.iterations,
        seed=args.seed,
        prefer_device=args.prefer_device,
        amortized_runs=args.amortized_runs,
        multi_type=args.multi_type,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    return 0


def _timing_metrics(durations: list[float], iterations: int, edge_count: int) -> dict[str, Any]:
    if not durations:
        return {
            "iterations": iterations,
            "latency_ms_avg": None,
            "latency_ms_min": None,
            "latency_ms_max": None,
            "throughput_edges_per_sec": None,
        }
    total = sum(durations)
    return {
        "iterations": iterations,
        "latency_ms_avg": round(mean(durations) * 1000, 4),
        "latency_ms_min": round(min(durations) * 1000, 4),
        "latency_ms_max": round(max(durations) * 1000, 4),
        "throughput_edges_per_sec": _throughput(edge_count * iterations, total),
    }


def _throughput(iterations: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return round(iterations / seconds, 4)


def _top_degree(result: dict[str, Any]) -> int:
    top_hubs = result.get("top_hubs") or []
    return int(top_hubs[0].get("degree", 0)) if top_hubs else 0


def _compare_centrality_results(cpu: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    cpu_top_hubs = cpu.get("top_hubs", [])
    optimized_top_hubs = optimized.get("top_hubs", [])
    top_hubs_exact_match = _top_hub_signature(cpu) == _top_hub_signature(optimized)
    top_degree_equal = _top_degree(cpu) == _top_degree(optimized)
    top_hub_count_equal = len(cpu_top_hubs) == len(optimized_top_hubs)
    cpu_boundary_degree = int(cpu_top_hubs[-1].get("degree", 0)) if cpu_top_hubs else 0
    optimized_above_cpu_boundary = all(
        int(item.get("degree", 0)) >= cpu_boundary_degree
        for item in optimized_top_hubs
    )
    type_centrality_equal = cpu.get("type_centrality") == optimized.get("type_centrality")
    passed = (
        top_degree_equal
        and top_hub_count_equal
        and optimized_above_cpu_boundary
        and type_centrality_equal
    )
    return {
        "status": "passed" if passed else "failed",
        "top_hubs_exact_match": top_hubs_exact_match,
        "top_degree_equal": top_degree_equal,
        "top_hub_count_equal": top_hub_count_equal,
        "optimized_above_cpu_boundary": optimized_above_cpu_boundary,
        "cpu_boundary_degree": cpu_boundary_degree,
        "type_centrality_equal": type_centrality_equal,
        "cpu_top_hubs_backend": cpu.get("top_hubs_backend"),
        "optimized_top_hubs_backend": optimized.get("top_hubs_backend"),
    }


def _top_hub_signature(result: dict[str, Any]) -> list[tuple[str, int]]:
    return [
        (str(item.get("id", "")), int(item.get("degree", 0)))
        for item in result.get("top_hubs", [])
    ]


def _amortized_metrics(
    cpu_latency_ms: float | None,
    cached_latency_ms: float | None,
    cache_prepare_ms: float | None,
    runs: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    run_counts = _normalize_run_counts(runs)
    cpu_totals = [
        round(cpu_latency_ms * run_count, 4) if cpu_latency_ms is not None else None
        for run_count in run_counts
    ]
    if cached_latency_ms is None or cache_prepare_ms is None:
        return {
            "runs": run_counts,
            "cpu_total_ms": cpu_totals,
            "cached_total_ms": [None for _ in run_counts],
            "speedup": [None for _ in run_counts],
            "breakeven_runs": None,
        }

    cached_totals = [
        round(cache_prepare_ms + cached_latency_ms * run_count, 4)
        for run_count in run_counts
    ]
    speedups = [
        round(cpu_total / cached_total, 4) if cpu_total is not None and cached_total > 0 else None
        for cpu_total, cached_total in zip(cpu_totals, cached_totals)
    ]
    breakeven_runs = next(
        (run_count for run_count, speedup in zip(run_counts, speedups) if speedup is not None and speedup >= 1),
        None,
    )
    return {
        "runs": run_counts,
        "cpu_total_ms": cpu_totals,
        "cached_total_ms": cached_totals,
        "speedup": speedups,
        "breakeven_runs": breakeven_runs,
    }


def _normalize_run_counts(runs: list[int] | tuple[int, ...]) -> list[int]:
    run_counts = [int(value) for value in runs]
    if not run_counts:
        raise ValueError("amortized_runs must not be empty")
    if any(value < 1 for value in run_counts):
        raise ValueError("amortized_runs must contain positive integers")
    return run_counts


if __name__ == "__main__":
    raise SystemExit(main())
