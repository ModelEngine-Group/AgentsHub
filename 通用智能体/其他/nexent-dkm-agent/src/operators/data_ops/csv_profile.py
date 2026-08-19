"""CSV profiling utilities for task 1."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MISSING_MARKERS = {"", "na", "n/a", "null", "none"}


def profile_csv(path: str | Path) -> dict[str, Any]:
    """Read a CSV file and return a small, deterministic data quality profile."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {csv_path}")

        columns = [column.strip() for column in reader.fieldnames]
        rows = [
            {column: (row.get(column) or "").strip() for column in columns}
            for row in reader
        ]

    missing_cells = {
        column: sum(1 for row in rows if _is_missing(row[column])) for column in columns
    }
    duplicate_rows = _count_duplicate_rows(rows, columns)

    return {
        "file_name": csv_path.name,
        "row_count": len(rows),
        "column_count": len(columns),
        "columns": [
            {
                "name": column,
                "inferred_type": _infer_type(row[column] for row in rows),
                "missing_count": missing_cells[column],
                "non_empty_count": len(rows) - missing_cells[column],
            }
            for column in columns
        ],
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
    }


def _is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_MARKERS


def _count_duplicate_rows(rows: list[dict[str, str]], columns: list[str]) -> int:
    seen: set[tuple[str, ...]] = set()
    duplicates = 0
    for row in rows:
        key = tuple(row[column] for column in columns)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def _infer_type(values: Iterable[str]) -> str:
    present_values = [
        str(value).strip() for value in values if not _is_missing(str(value))
    ]
    if not present_values:
        return "empty"
    if all(_is_integer(value) for value in present_values):
        return "integer"
    if all(_is_float(value) for value in present_values):
        return "float"
    if all(value.lower() in {"true", "false"} for value in present_values):
        return "boolean"
    return "text"


def _is_integer(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True
