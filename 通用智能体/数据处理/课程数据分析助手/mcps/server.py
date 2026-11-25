"""MCP tools for course data analysis.

This module implements deterministic standard-library tools and exposes them
through FastMCP when it is installed. It intentionally reads only the local
course CSV and never writes or deletes data.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "course_records.csv"
REQUIRED_COLUMNS = [
    "student_id",
    "gender",
    "grade",
    "attendance_rate",
    "homework_score",
    "midterm_score",
    "final_score",
    "learning_hours",
    "experiment_score",
]
NUMERIC_COLUMNS = [
    "attendance_rate",
    "homework_score",
    "midterm_score",
    "final_score",
    "learning_hours",
    "experiment_score",
]


def _load_rows() -> list[dict[str, Any]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")

    with open(DATA_PATH, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
    if missing:
        raise ValueError(f"数据缺少必需字段: {', '.join(missing)}")
    return rows


def _as_float(value: str) -> float:
    return float(value)


def _error(tool: str, message: str) -> dict[str, Any]:
    return {"tool": tool, "status": "error", "message": message}


def inspect_dataset_tool() -> dict[str, Any]:
    """Inspect row count, columns, missing values, and numeric ranges."""
    try:
        rows = _load_rows()
    except Exception as exc:
        return _error("inspect_dataset", str(exc))

    missing_count = {
        column: sum(row.get(column) in (None, "") for row in rows)
        for column in REQUIRED_COLUMNS
    }
    ranges: dict[str, dict[str, float | int]] = {}
    for column in NUMERIC_COLUMNS:
        values = [_as_float(row[column]) for row in rows if row.get(column) not in (None, "")]
        if values:
            ranges[column] = {
                "min": min(values),
                "max": max(values),
            }

    return {
        "tool": "inspect_dataset",
        "status": "success",
        "row_count": len(rows),
        "column_count": len(REQUIRED_COLUMNS),
        "columns": REQUIRED_COLUMNS,
        "missing_count": missing_count,
        "numeric_ranges": ranges,
        "data_file": str(DATA_PATH),
    }


def _values(rows: list[dict[str, Any]], column: str) -> list[float]:
    return [_as_float(row[column]) for row in rows]


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("相关系数计算需要两个长度相同且至少两行的数值序列")
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    denominator_left = math.sqrt(sum((x - mean_left) ** 2 for x in left))
    denominator_right = math.sqrt(sum((y - mean_right) ** 2 for y in right))
    if denominator_left == 0 or denominator_right == 0:
        raise ValueError("某列方差为 0，无法计算相关系数")
    return numerator / (denominator_left * denominator_right)


def analyze_course_data_tool(at_risk_threshold: int = 60) -> dict[str, Any]:
    """Compute course score statistics, correlations, and at-risk counts."""
    try:
        rows = _load_rows()
        if not rows:
            raise ValueError("数据文件没有记录")
        threshold = int(at_risk_threshold)
        if not 0 <= threshold <= 100:
            raise ValueError("at_risk_threshold 必须在 0 到 100 之间")

        final_scores = _values(rows, "final_score")
        correlations = {
            column: _correlation(_values(rows, column), final_scores)
            for column in NUMERIC_COLUMNS
            if column != "final_score"
        }
        at_risk = [row["student_id"] for row in rows if _as_float(row["final_score"]) < threshold]

        return {
            "tool": "analyze_course_data",
            "status": "success",
            "sample_count": len(rows),
            "final_score": {
                "mean": round(statistics.fmean(final_scores), 2),
                "stdev": round(statistics.stdev(final_scores), 2),
                "min": min(final_scores),
                "max": max(final_scores),
                "quantiles": {
                    "q25": round(statistics.quantiles(final_scores, n=4, method="inclusive")[0], 2),
                    "median": round(statistics.median(final_scores), 2),
                    "q75": round(statistics.quantiles(final_scores, n=4, method="inclusive")[2], 2),
                },
            },
            "correlations_with_final_score": {
                column: round(value, 3)
                for column, value in correlations.items()
            },
            "at_risk_count": len(at_risk),
            "at_risk_rate": round(len(at_risk) / len(rows), 4),
            "at_risk_threshold": threshold,
        }
    except Exception as exc:
        return _error("analyze_course_data", str(exc))


def query_course_data_tool(
    min_final_score: int | None = None,
    max_final_score: int | None = None,
    min_attendance_rate: float | None = None,
    max_attendance_rate: float | None = None,
    gender: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Query course records with safe numeric and categorical filters."""
    try:
        rows = _load_rows()
        if not 1 <= limit <= 500:
            raise ValueError("limit 必须在 1 到 500 之间")
        if gender is not None and gender not in {"男", "女"}:
            raise ValueError("gender 仅支持 男 或 女")

        filtered = rows
        if min_final_score is not None:
            filtered = [row for row in filtered if _as_float(row["final_score"]) >= min_final_score]
        if max_final_score is not None:
            filtered = [row for row in filtered if _as_float(row["final_score"]) <= max_final_score]
        if min_attendance_rate is not None:
            filtered = [
                row for row in filtered
                if _as_float(row["attendance_rate"]) >= min_attendance_rate
            ]
        if max_attendance_rate is not None:
            filtered = [
                row for row in filtered
                if _as_float(row["attendance_rate"]) <= max_attendance_rate
            ]
        if gender is not None:
            filtered = [row for row in filtered if row["gender"] == gender]

        return {
            "tool": "query_course_data",
            "status": "success",
            "matched_count": len(filtered),
            "returned_count": min(len(filtered), limit),
            "records": filtered[:limit],
        }
    except Exception as exc:
        return _error("query_course_data", str(exc))


try:
    from fastmcp import FastMCP
except ImportError:
    FastMCP = None


if FastMCP is not None:
    mcp = FastMCP("course-data-analysis")
    mcp.tool(name="inspect_dataset")(inspect_dataset_tool)
    mcp.tool(name="analyze_course_data")(analyze_course_data_tool)
    mcp.tool(name="query_course_data")(query_course_data_tool)


if __name__ == "__main__":
    if FastMCP is None:
        raise SystemExit("请先安装 fastmcp，或在导入模式下调用本文件中的工具函数。")
    mcp.run(transport="http", host="0.0.0.0", port=8001, path="/mcp")
