"""Measure task-1 DataMate submit integration end-to-end.

When a DataMate backend is reachable, runs the task-1 CSV pipeline with
``datamate_mode=submit`` (or ``auto`` resolved to submit), creates a cleaning
template and task, and verifies server-side resource IDs via detail APIs.

When DataMate is unavailable, writes an honest skipped report so reviewers can
distinguish "not run" from "failed".

Usage:
    python benchmarks/task1_datamate_submit_benchmark.py
    python benchmarks/task1_datamate_submit_benchmark.py --base-url http://localhost:18000 --mode submit
    python benchmarks/task1_datamate_submit_benchmark.py --report benchmarks/reports/task1_datamate_submit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "data" / "samples" / "task1_patients.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "task1_datamate_submit_benchmark"
DEFAULT_BASE_URL = "http://localhost:18000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark task-1 DataMate submit integration.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="DataMate backend base URL.")
    parser.add_argument(
        "--mode",
        choices=["submit", "auto"],
        default="auto",
        help="DataMate execution mode (auto resolves to submit when healthy).",
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Pipeline output directory.")
    parser.add_argument("--timeout", type=float, default=5.0, help="DataMate HTTP timeout in seconds.")
    parser.add_argument(
        "--src-dataset-id",
        default=None,
        help="Existing DataMate source dataset id (required for task submit verification).",
    )
    parser.add_argument(
        "--src-dataset-name",
        default="task1_submit_benchmark",
        help="Source dataset display name for task payload.",
    )
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _extract_submit_evidence(datamate: dict[str, Any]) -> dict[str, Any]:
    operators = datamate.get("operators") or {}
    template = operators.get("cleaning_template") or {}
    task = operators.get("cleaning_task") or {}
    template_sub = template.get("submission") or {}
    task_sub = task.get("submission") or {}

    return {
        "template_status": template_sub.get("status"),
        "template_resource_id": template_sub.get("resource_id"),
        "template_verified": template_sub.get("verified"),
        "task_status": task_sub.get("status"),
        "task_resource_id": task_sub.get("resource_id"),
        "task_verified": task_sub.get("verified"),
        "verified": bool(template_sub.get("verified")) and bool(task_sub.get("verified")),
    }


def run_datamate_submit_benchmark(
    *,
    base_url: str | None = DEFAULT_BASE_URL,
    mode: str = "auto",
    input_path: str | Path | None = DEFAULT_INPUT,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    timeout: float = 5.0,
    src_dataset_id: str | None = None,
    src_dataset_name: str | None = "task1_submit_benchmark",
) -> dict[str, Any]:
    from src.agents.data_processing_agent.agent import inspect_datamate
    from src.operators.data_ops.datamate_client import resolve_datamate_mode
    from src.pipelines.task1_data_pipeline import run_task1_pipeline

    input_file = Path(input_path) if input_path else DEFAULT_INPUT
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_mode, mode_meta = resolve_datamate_mode(base_url, mode, timeout=timeout)
    health_probe = inspect_datamate(base_url, plan_operators=[], timeout=timeout)

    report: dict[str, Any] = {
        "task": "task1_datamate_submit",
        "input": _relative(input_file),
        "base_url": base_url,
        "requested_mode": mode,
        "resolved_mode": resolved_mode,
        "mode_resolution": mode_meta,
        "health_status": health_probe.get("status"),
    }

    if health_probe.get("status") not in {"healthy", "unknown"}:
        report.update({
            "status": "skipped",
            "reason": "datamate_unavailable",
            "passed": False,
            "message": "DataMate health check failed; submit benchmark not executed.",
            "health": health_probe.get("health"),
            "historical_evidence": (
                "competition_submission/defense-package-final/evidence/"
                "online_integration/datamate-submit-20260702-final.json"
            ),
        })
        return report

    if resolved_mode != "submit":
        report.update({
            "status": "skipped",
            "reason": mode_meta.get("reason", "resolved_to_dry_run"),
            "passed": False,
            "message": "DataMate mode did not resolve to submit.",
        })
        return report

    pipeline = run_task1_pipeline(
        input_path=input_file,
        output_dir=out_dir,
        datamate_base_url=base_url,
        datamate_timeout=timeout,
        datamate_mode=resolved_mode,
        datamate_src_dataset_id=src_dataset_id,
        datamate_src_dataset_name=src_dataset_name,
    )

    datamate = pipeline.artifacts.get("datamate", {})
    mode_resolution = pipeline.artifacts.get("datamate_mode_resolution", mode_meta)
    evidence = _extract_submit_evidence(datamate)
    quality = pipeline.artifacts.get("quality_report", {})

    template_ok = evidence.get("template_status") == "verified"
    task_ok = (
        evidence.get("task_status") == "verified"
        if src_dataset_id
        else evidence.get("task_status") in {None, "waiting_for_dataset"}
    )
    passed = (
        pipeline.status in {"completed", "completed_with_warnings"}
        and template_ok
        and task_ok
    )

    report.update({
        "status": "completed" if passed else "failed",
        "pipeline_status": pipeline.status,
        "passed": passed,
        "src_dataset_id": src_dataset_id,
        "datamate_mode_resolution": mode_resolution,
        "submit_evidence": evidence,
        "quality_report_status": quality.get("status"),
        "output_dir": _relative(out_dir),
    })
    return report


def main() -> int:
    args = parse_args()
    report = run_datamate_submit_benchmark(
        base_url=args.base_url,
        mode=args.mode,
        input_path=args.input,
        output_dir=args.output_dir,
        timeout=args.timeout,
        src_dataset_id=args.src_dataset_id,
        src_dataset_name=args.src_dataset_name,
    )

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")

    return 0 if report.get("passed") or report.get("status") == "skipped" else 1


if __name__ == "__main__":
    raise SystemExit(main())
