"""Run the task-1 data quality benchmark.

The benchmark measures deterministic CSV cleaning quality rather than NPU
speedup. Task 1 is dominated by data profiling, cleaning, validation, and
DataMate payload preparation, so its reviewer-facing metric is whether data
quality improves reproducibly on a known dirty sample.

Usage:
    python benchmarks/task1_data_quality_benchmark.py
    python benchmarks/task1_data_quality_benchmark.py --report benchmarks/reports/task1_data_quality.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.task1_evaluation import run_task1_evaluation

DEFAULT_INPUT = ROOT / "data" / "samples" / "task1_patients.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "task1_benchmark"
DEFAULT_TASK = "请清洗患者CSV，删除重复记录，填补空值，并统一字段类型后导出"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark task-1 data quality.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task-1 cleaning request.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV path.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of repeated pipeline runs used for latency averaging.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated benchmark run outputs.",
    )
    parser.add_argument("--report", default=None, help="Optional JSON report output path.")
    parser.add_argument(
        "--max-duplicate-rows-after",
        type=int,
        default=0,
        help="Maximum allowed duplicate rows after cleaning.",
    )
    parser.add_argument(
        "--max-missing-values-after",
        type=int,
        default=0,
        help="Maximum allowed missing values after cleaning.",
    )
    parser.add_argument(
        "--min-quality-score-after",
        type=float,
        default=1.0,
        help="Minimum required post-cleaning quality score.",
    )
    return parser.parse_args()


def run_task1_quality_benchmark(
    *,
    task_request: str | None = DEFAULT_TASK,
    input_path: str | Path | None = DEFAULT_INPUT,
    iterations: int = 3,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    report_path: str | Path | None = None,
    max_duplicate_rows_after: int = 0,
    max_missing_values_after: int = 0,
    min_quality_score_after: float = 1.0,
) -> dict[str, Any]:
    """Run task 1 repeatedly and return a reproducible quality report."""

    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    base_output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    latencies_ms: list[float] = []
    payload: dict[str, Any] | None = None

    for index in range(iterations):
        run_output_dir = base_output_dir / f"run_{index + 1}"
        start = time.perf_counter()
        payload = run_task1_evaluation(
            task_request=task_request,
            input_path=input_path,
            output_dir=run_output_dir,
            datamate_base_url=None,
            report_path=None,
        )
        latencies_ms.append((time.perf_counter() - start) * 1000)

    assert payload is not None
    metrics = _quality_metrics(payload)
    thresholds = {
        "max_duplicate_rows_after": max_duplicate_rows_after,
        "max_missing_values_after": max_missing_values_after,
        "min_quality_score_after": min_quality_score_after,
    }
    checks = {
        "pipeline_completed": payload.get("status") in {"completed", "completed_with_warnings"},
        "validation_passed": payload.get("validation", {}).get("status") == "passed",
        "duplicates_removed": metrics["duplicate_rows_after"] <= max_duplicate_rows_after,
        "missing_values_filled": metrics["missing_values_after"] <= max_missing_values_after,
        "quality_score_met": metrics["quality_score_after"] >= min_quality_score_after,
        "quality_improved": metrics["quality_score_after"] >= metrics["quality_score_before"],
    }

    report = {
        "task": payload.get("task"),
        "benchmark_type": "data_quality",
        "benchmark": _relative(Path(input_path) if input_path else DEFAULT_INPUT),
        "iterations": iterations,
        "status": payload.get("status"),
        "passed": all(checks.values()),
        "thresholds": thresholds,
        "checks": checks,
        "quality_metrics": metrics,
        "timing": {
            "latency_ms_avg": statistics.fmean(latencies_ms),
            "latency_ms_min": min(latencies_ms),
            "latency_ms_max": max(latencies_ms),
        },
        "plan": payload.get("plan", {}),
        "run_state_status": payload.get("run_state", {}).get("status"),
        "notes": [
            "Task 1 benchmark measures deterministic data-quality improvement, not NPU acceleration.",
            "DataMate is disabled for benchmark determinism; integration payloads are covered by task-1 demo/tests.",
        ],
    }

    if report_path:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return report


def _quality_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    profile = payload.get("profile", {})
    cleaning = payload.get("cleaning", {})
    validation = payload.get("validation", {})
    before = validation.get("before", {})
    after = validation.get("after", {})

    input_rows = profile.get("row_count", before.get("row_count", 0))
    output_rows = cleaning.get("output_rows", after.get("row_count", 0))
    column_count = profile.get("column_count", before.get("column_count", 0))
    duplicate_rows_before = profile.get("duplicate_rows", before.get("duplicate_rows", 0))
    duplicate_rows_after = after.get("duplicate_rows", 0)
    missing_values_before = _missing_total(profile.get("missing_cells", before.get("missing_cells", {})))
    missing_values_after = _missing_total(after.get("missing_cells", {}))

    return {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "column_count": column_count,
        "duplicate_rows_before": duplicate_rows_before,
        "duplicate_rows_after": duplicate_rows_after,
        "duplicate_rows_removed": cleaning.get("duplicate_rows_removed", 0),
        "missing_values_before": missing_values_before,
        "missing_values_after": missing_values_after,
        "missing_values_filled": cleaning.get("missing_values_filled", 0),
        "row_count_delta": output_rows - input_rows,
        "quality_score_before": _quality_score(
            row_count=input_rows,
            column_count=column_count,
            duplicate_rows=duplicate_rows_before,
            missing_values=missing_values_before,
        ),
        "quality_score_after": _quality_score(
            row_count=output_rows,
            column_count=column_count,
            duplicate_rows=duplicate_rows_after,
            missing_values=missing_values_after,
        ),
        "validation_status": validation.get("status"),
        "quality_report_status": payload.get("quality_report", {}).get("status"),
    }


def _quality_score(
    *,
    row_count: int,
    column_count: int,
    duplicate_rows: int,
    missing_values: int,
) -> float:
    total_cells = max(row_count * max(column_count, 1), 1)
    issue_count = duplicate_rows + missing_values
    return round(max(0.0, 1.0 - issue_count / total_cells), 6)


def _missing_total(missing_cells: dict[str, int]) -> int:
    return sum(missing_cells.values())


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def main() -> int:
    args = parse_args()
    report = run_task1_quality_benchmark(
        task_request=args.task,
        input_path=args.input,
        iterations=args.iterations,
        output_dir=args.output_dir,
        report_path=args.report,
        max_duplicate_rows_after=args.max_duplicate_rows_after,
        max_missing_values_after=args.max_missing_values_after,
        min_quality_score_after=args.min_quality_score_after,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
