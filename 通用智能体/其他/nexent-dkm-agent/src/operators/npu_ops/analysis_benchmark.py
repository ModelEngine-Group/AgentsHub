"""Task 3 graph analysis benchmark helpers with optional NPU detection."""

from __future__ import annotations

import time
from statistics import mean
from typing import Any

from src.operators.analysis_ops import (
    build_analysis_visualizations,
    build_graph_sqlite,
    execute_sql,
    generate_association_analysis,
    generate_statistical_summary,
    generate_trend_analysis,
    translate_question_to_sql,
)
from src.operators.npu_ops.kg_benchmark import detect_npu_runtime


def benchmark_task3_analysis_ops(
    graph: dict[str, Any],
    question: str | None = None,
    iterations: int = 5,
    npu_probe: bool = True,
    npu_probe_iterations: int = 5,
    npu_probe_size: int = 64,
) -> dict[str, Any]:
    """Benchmark the deterministic task-3 graph analysis chain on CPU."""

    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    warmup = _run_analysis_ops(graph, question)
    durations = []
    last_run = warmup
    for _ in range(iterations):
        started = time.perf_counter()
        last_run = _run_analysis_ops(graph, question)
        durations.append(time.perf_counter() - started)

    total_duration = sum(durations)
    node_count = len(graph.get("nodes", []))
    edge_count = len(graph.get("edges", []))
    return {
        "task": "task3_analysis_agent",
        "input": {
            "node_count": node_count,
            "edge_count": edge_count,
            "iterations": iterations,
            "npu_probe": {
                "enabled": npu_probe,
                "iterations": npu_probe_iterations,
                "matrix_size": npu_probe_size,
            },
        },
        "cpu": {
            "status": "completed",
            "backend": "python",
            "iterations": iterations,
            "latency_ms_avg": round(mean(durations) * 1000, 4),
            "latency_ms_min": round(min(durations) * 1000, 4),
            "latency_ms_max": round(max(durations) * 1000, 4),
            "throughput_edges_per_sec": _throughput(edge_count * iterations, total_duration),
            "chart_count": len(last_run["visualizations"].get("charts", {})),
            "sql_row_count": len(last_run["nl2sql"].get("rows", [])),
            "disease_profile_count": len(last_run["associations"].get("disease_profiles", [])),
        },
        "npu": detect_npu_runtime(
            probe=npu_probe,
            probe_iterations=npu_probe_iterations,
            probe_size=npu_probe_size,
        ),
        "notes": [
            "CPU numbers are measured locally with deterministic graph-analysis operators.",
            "NPU runtime probes are reported separately from task-operator CPU timings.",
            "Task-level NPU speedup and energy metrics require operator-specific NPU kernels and profiler support.",
        ],
    }


def _run_analysis_ops(graph: dict[str, Any], question: str | None) -> dict[str, Any]:
    statistics = generate_statistical_summary(graph)
    associations = generate_association_analysis(graph)
    trends = generate_trend_analysis(graph)
    sql_plan = translate_question_to_sql(question)
    sql_rows = execute_sql(build_graph_sqlite(graph), sql_plan["sql"])
    visualizations = build_analysis_visualizations(statistics, associations, trends)
    return {
        "statistics": statistics,
        "associations": associations,
        "trends": trends,
        "nl2sql": {**sql_plan, "rows": sql_rows},
        "visualizations": visualizations,
    }


def _throughput(items: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return round(items / seconds, 4)
