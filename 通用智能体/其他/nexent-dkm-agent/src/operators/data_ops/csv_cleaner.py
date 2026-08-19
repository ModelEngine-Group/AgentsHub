"""CSV cleaning utilities for task 1."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.operators.data_ops.csv_profile import MISSING_MARKERS, profile_csv

DEDUP_OPERATOR = "drop_duplicate_rows"
FILL_OPERATOR = "fill_missing_values"
NORMALIZE_OPERATOR = "normalize_column_types"


def clean_csv(
    input_path: str | Path,
    profile: dict[str, Any],
    output_dir: str | Path,
    operators: list[str] | None = None,
) -> dict[str, Any]:
    """Apply the planned task-1 cleaning operators and export a cleaned CSV.

    ``operators`` is the operator list produced by the planner. Only the
    sub-steps named in the plan run, so the agent's plan genuinely drives
    execution. ``operators=None`` preserves the full dedup+fill+normalize
    flow (backward-compatible default).
    """

    csv_path = Path(input_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{csv_path.stem}_cleaned.csv"

    if operators is None:
        do_dedup = do_fill = do_normalize = True
    else:
        op_set = set(operators)
        do_dedup = DEDUP_OPERATOR in op_set
        do_fill = FILL_OPERATOR in op_set
        do_normalize = NORMALIZE_OPERATOR in op_set

    columns = [column["name"] for column in profile["columns"]]
    type_by_column = {
        column["name"]: column["inferred_type"] for column in profile["columns"]
    }
    rows = _read_rows(csv_path, columns)
    if do_dedup:
        working_rows, duplicate_rows_removed = _deduplicate_rows(rows, columns)
    else:
        working_rows, duplicate_rows_removed = list(rows), 0
    cleaned_rows, missing_values_filled = _fill_and_normalize_rows(
        working_rows,
        columns,
        type_by_column,
        do_fill=do_fill,
        do_normalize=do_normalize,
    )
    _write_rows(output_path, columns, cleaned_rows)

    operators_applied = []
    if do_dedup:
        operators_applied.append(DEDUP_OPERATOR)
    if do_fill:
        operators_applied.append(FILL_OPERATOR)
    if do_normalize:
        operators_applied.append(NORMALIZE_OPERATOR)

    return {
        "status": "completed",
        "output_path": str(output_path),
        "input_rows": len(rows),
        "output_rows": len(cleaned_rows),
        "duplicate_rows_removed": duplicate_rows_removed,
        "missing_values_filled": missing_values_filled,
        "operators_applied": operators_applied,
    }


def validate_cleaning_result(
    before_profile: dict[str, Any],
    cleaning_result: dict[str, Any],
) -> dict[str, Any]:
    """Profile the exported CSV and verify core cleaning invariants."""

    after_profile = profile_csv(cleaning_result["output_path"])
    applied = cleaning_result.get("operators_applied")
    dedup_applied = applied is None or DEDUP_OPERATOR in applied
    fill_applied = applied is None or FILL_OPERATOR in applied

    checks: dict[str, bool] = {}
    if dedup_applied:
        checks["duplicate_rows_removed"] = after_profile["duplicate_rows"] == 0
    if fill_applied:
        checks["missing_values_filled"] = _missing_total(after_profile) == 0
    checks["row_count_matches_export"] = (
        after_profile["row_count"] == cleaning_result["output_rows"]
    )
    checks["row_count_not_increased"] = (
        after_profile["row_count"] <= before_profile["row_count"]
    )
    status = "passed" if all(checks.values()) else "failed"
    return {
        "status": status,
        "checks": checks,
        "before": {
            "row_count": before_profile["row_count"],
            "duplicate_rows": before_profile["duplicate_rows"],
            "missing_cells": before_profile["missing_cells"],
        },
        "after": {
            "row_count": after_profile["row_count"],
            "duplicate_rows": after_profile["duplicate_rows"],
            "missing_cells": after_profile["missing_cells"],
        },
    }


def _read_rows(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {column: (row.get(column) or "").strip() for column in columns}
            for row in reader
        ]


def _deduplicate_rows(
    rows: list[dict[str, str]],
    columns: list[str],
) -> tuple[list[dict[str, str]], int]:
    seen: set[tuple[str, ...]] = set()
    unique_rows: list[dict[str, str]] = []
    duplicate_count = 0
    for row in rows:
        key = tuple(row[column] for column in columns)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows, duplicate_count


def _fill_and_normalize_rows(
    rows: list[dict[str, str]],
    columns: list[str],
    type_by_column: dict[str, str],
    do_fill: bool = True,
    do_normalize: bool = True,
) -> tuple[list[dict[str, str]], int]:
    cleaned_rows: list[dict[str, str]] = []
    missing_values_filled = 0
    for row in rows:
        cleaned_row = {}
        for column in columns:
            value = row[column]
            is_missing = _is_missing(value)
            if is_missing and do_fill:
                missing_values_filled += 1
                value = _default_value(type_by_column[column])
                is_missing = False
            if do_normalize and not is_missing:
                value = _normalize_value(value, type_by_column[column])
            cleaned_row[column] = value
        cleaned_rows.append(cleaned_row)
    return cleaned_rows, missing_values_filled


def _write_rows(
    output_path: Path,
    columns: list[str],
    rows: list[dict[str, str]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_MARKERS


def _default_value(inferred_type: str) -> str:
    if inferred_type == "integer":
        return "0"
    if inferred_type == "float":
        return "0.0"
    if inferred_type == "boolean":
        return "false"
    return "unknown"


def _normalize_value(value: str, inferred_type: str) -> str:
    if inferred_type == "integer":
        return str(int(value))
    if inferred_type == "float":
        return str(float(value))
    if inferred_type == "boolean":
        return value.strip().lower()
    return value.strip()


def _missing_total(profile: dict[str, Any]) -> int:
    return sum(profile.get("missing_cells", {}).values())
