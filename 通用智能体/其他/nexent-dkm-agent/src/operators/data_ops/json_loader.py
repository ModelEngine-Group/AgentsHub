"""JSON record loading for task 1.

Converts a JSON dataset of records into a flat CSV so the existing CSV
profiling/cleaning/validation operators can be reused unchanged. Supported
JSON shapes:

- a top-level list of objects: ``[{...}, {...}]``
- a wrapper object with a records key: ``{"records": [...]}``,
  ``{"data": [...]}``, or ``{"rows": [...]}``

Nested object/array values are JSON-serialized into their CSV cell so the
tabular schema stays flat and deterministic.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

_RECORD_KEYS = ("records", "data", "rows", "items")


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON dataset and return a list of record dicts."""

    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {json_path}")

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Input JSON is not valid: {exc}") from exc

    if isinstance(payload, dict):
        for key in _RECORD_KEYS:
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            # A single object is treated as a one-row dataset.
            payload = [payload]

    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list of records or a wrapper object containing one.")
    if not payload:
        raise ValueError("Input JSON contains no records.")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"JSON record at index {index} is not an object.")
        records.append(item)
    return records


def json_records_to_csv(
    input_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Convert a JSON record dataset into a flat CSV for the task-1 CSV pipeline."""

    json_path = Path(input_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    records = load_json_records(json_path)

    # Preserve first-seen column order across all records.
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)

    csv_path = target_dir / f"{json_path.stem}_from_json.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({col: _cell(record.get(col)) for col in columns})

    return {
        "status": "completed",
        "csv_path": str(csv_path),
        "record_count": len(records),
        "columns": columns,
    }


def _cell(value: Any) -> str:
    """Render a JSON value as a flat CSV cell."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
