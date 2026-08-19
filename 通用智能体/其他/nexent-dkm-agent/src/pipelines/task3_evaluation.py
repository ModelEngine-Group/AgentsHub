"""Reproducible evaluation helper for task 3."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.agents.analysis_agent.agent import DEFAULT_OUTPUT_DIR
from src.pipelines.task3_insight_pipeline import run_task3_pipeline

DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "task3_quality_report.json"


def run_task3_evaluation(
    graph_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    question: str | None = None,
    task_request: str | None = None,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Run task 3 and write a compact JSON evidence report."""

    result = run_task3_pipeline(
        graph_file=graph_file,
        output_dir=output_dir,
        question=question,
        task_request=task_request,
    )
    payload = _evaluation_payload(asdict(result))
    if report_path:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _evaluation_payload(result: dict[str, Any]) -> dict[str, Any]:
    artifacts = result.get("artifacts", {})
    nl2sql = artifacts.get("nl2sql", {})
    visualizations = artifacts.get("visualizations", {})
    return {
        "task": result.get("task"),
        "status": result.get("status"),
        "message": result.get("message"),
        "input": artifacts.get("input", {}),
        "graph": artifacts.get("graph", {}),
        "statistics": artifacts.get("statistics", {}),
        "associations": {
            "status": artifacts.get("associations", {}).get("status"),
            "profile_count": len(artifacts.get("associations", {}).get("disease_profiles", [])),
            "top_associations": artifacts.get("associations", {}).get("top_associations", []),
        },
        "trends": {
            "status": artifacts.get("trends", {}).get("status"),
            "record_count": len(artifacts.get("trends", {}).get("record_trends", [])),
            "peak_record": artifacts.get("trends", {}).get("peak_record"),
        },
        "nl2sql": {
            "status": nl2sql.get("status"),
            "intent": nl2sql.get("intent"),
            "sql": nl2sql.get("sql"),
            "row_count": len(nl2sql.get("rows", [])),
            "rows": nl2sql.get("rows", []),
        },
        "visualizations": {
            "status": visualizations.get("status"),
            "chart_names": sorted(visualizations.get("charts", {}).keys()),
        },
        "insight_report": artifacts.get("insight_report", {}),
        "quality_report": artifacts.get("quality_report", {}),
        "run_state": artifacts.get("run_state", {}),
    }
