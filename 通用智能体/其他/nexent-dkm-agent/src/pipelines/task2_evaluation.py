"""Reproducible evaluation helper for task 2."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.agents.kg_agent.agent import DEFAULT_OUTPUT_DIR
from src.pipelines.task2_kg_pipeline import run_task2_pipeline

DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "task2_quality_report.json"


def run_task2_evaluation(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    question: str | None = None,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Run task 2 and write a compact JSON evidence report."""

    result = run_task2_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        question=question,
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
    extraction = artifacts.get("extraction", {})
    validation = artifacts.get("validation", {})
    graph = artifacts.get("graph", {})
    qa = artifacts.get("qa", {})
    return {
        "task": result.get("task"),
        "status": result.get("status"),
        "message": result.get("message"),
        "input": artifacts.get("input", {}),
        "entity_counts": extraction.get("entity_counts", {}),
        "normalization": extraction.get("normalization", {}),
        "validation": {
            "status": validation.get("status"),
            "valid_count": validation.get("valid_count"),
            "invalid_count": validation.get("invalid_count"),
            "duplicate_count": validation.get("duplicate_count"),
        },
        "graph": graph,
        "qa": {
            "status": qa.get("status"),
            "question": qa.get("question"),
            "answer": qa.get("answer"),
            "evidence_count": len(qa.get("evidence", [])),
        },
        "quality_report": artifacts.get("quality_report", {}),
        "run_state": artifacts.get("run_state", {}),
    }
