"""One-command reviewer smoke check for task 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.analysis_agent.nexent_adapter import build_nexent_agent_spec
from src.operators.analysis_ops import load_graph
from src.operators.npu_ops import benchmark_task3_analysis_ops
from src.pipelines.task3_insight_pipeline import run_task3_pipeline


def run_task3_smoke(
    graph_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    question: str | None = None,
    iterations: int = 3,
) -> dict[str, Any]:
    """Run a compact task-3 smoke suite for reviewers."""

    effective_question = question or "哪些疾病关联最多症状？"
    result = run_task3_pipeline(
        graph_file=graph_file,
        output_dir=output_dir,
        question=effective_question,
    )
    artifacts = result.artifacts
    insight = artifacts.get("insight_report", {})
    graph_path = artifacts.get("input", {}).get("path")
    centrality = artifacts.get("centrality", {})
    top_hubs_backend = centrality.get("top_hubs_backend")

    benchmark = None
    if result.status == "completed" and graph_path:
        benchmark = benchmark_task3_analysis_ops(
            graph=load_graph(graph_path),
            question=effective_question,
            iterations=iterations,
        )

    agent_spec = build_nexent_agent_spec(model_name="main_model")
    checks = {
        "pipeline_completed": result.status == "completed",
        "quality_passed": artifacts.get("quality_report", {}).get("status") == "passed",
        "html_report_exists": _path_exists(insight.get("html_path")),
        "markdown_report_exists": _path_exists(insight.get("markdown_path")),
        "dashboard_exists": _path_exists(insight.get("dashboard_path")),
        "benchmark_completed": bool(benchmark and benchmark.get("cpu", {}).get("status") == "completed"),
        "nexent_spec_ready": agent_spec.get("tools", [{}])[0].get("name") == "task3_graph_analysis",
        "top_hubs_backend_recorded": top_hubs_backend in {"python", "torch_npu"},
    }
    status = "completed" if all(checks.values()) else "warning"
    return {
        "task": "task3_analysis_agent",
        "status": status,
        "checks": checks,
        "artifacts": {
            "graph_file": graph_path,
            "json_report": artifacts.get("export", {}).get("output_path"),
            "html_report": insight.get("html_path"),
            "markdown_report": insight.get("markdown_path"),
            "dashboard": insight.get("dashboard_path"),
            "chart_count": artifacts.get("quality_report", {}).get("metrics", {}).get("chart_count", 0),
            "nl2sql_row_count": artifacts.get("quality_report", {}).get("metrics", {}).get("nl2sql_row_count", 0),
            "top_hubs_backend": top_hubs_backend,
            "top_hubs_npu_reason": centrality.get("top_hubs_npu_reason"),
        },
        "benchmark": {
            "latency_ms_avg": (benchmark or {}).get("cpu", {}).get("latency_ms_avg"),
            "throughput_edges_per_sec": (benchmark or {}).get("cpu", {}).get("throughput_edges_per_sec"),
            "npu_status": (benchmark or {}).get("npu", {}).get("status"),
        },
        "nexent": {
            "agent_name": agent_spec.get("name"),
            "tool_name": agent_spec.get("tools", [{}])[0].get("name"),
        },
    }


def _path_exists(path: str | None) -> bool:
    return bool(path and Path(path).exists())
