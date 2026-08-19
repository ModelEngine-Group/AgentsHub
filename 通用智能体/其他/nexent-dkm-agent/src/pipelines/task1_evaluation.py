"""Reproducible evaluation helper for task 1."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.agents.data_processing_agent.agent import DEFAULT_OUTPUT_DIR
from src.pipelines.task1_data_pipeline import run_task1_pipeline

DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "task1_quality_report.json"


def run_task1_evaluation(
    task_request: str | None = None,
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    datamate_base_url: str | None = "http://localhost:18000",
    datamate_mode: str = "dry_run",
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Run task 1 and write a compact JSON evidence report."""

    result = run_task1_pipeline(
        task_request=task_request,
        input_path=input_path,
        output_dir=output_dir,
        datamate_base_url=datamate_base_url,
        datamate_mode=datamate_mode,
    )
    payload = _evaluation_payload(asdict(result))
    if report_path:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return payload


def _evaluation_payload(result: dict[str, Any]) -> dict[str, Any]:
    artifacts = result.get("artifacts", {})
    return {
        "task": result.get("task"),
        "status": result.get("status"),
        "message": result.get("message"),
        "input": artifacts.get("input", {}),
        "understanding": artifacts.get("understanding", {}),
        "plan": {
            "operators": artifacts.get("plan", {}).get("operators", []),
            "confidence": artifacts.get("plan", {}).get("confidence"),
            "quality_checks": artifacts.get("plan", {}).get("quality_checks", []),
        },
        "profile": artifacts.get("profile", {}),
        "cleaning": artifacts.get("cleaning", {}),
        "validation": artifacts.get("validation", {}),
        "datamate": artifacts.get("quality_report", {}).get("datamate", {}),
        "quality_report": artifacts.get("quality_report", {}),
        "run_state": artifacts.get("run_state", {}),
    }
