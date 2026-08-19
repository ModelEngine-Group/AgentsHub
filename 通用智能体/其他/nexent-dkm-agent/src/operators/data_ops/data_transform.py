"""Data transform operators for task 1."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def transform_csv(
    input_path: str | Path,
    output_dir: str | Path,
    transforms: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply column-level transforms to a CSV: select, rename, filter, convert types."""

    csv_path = Path(input_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{csv_path.stem}_transformed.csv"

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        original_columns = list(reader.fieldnames or [])
        rows = list(reader)

    transforms = transforms or []
    applied = []
    columns = list(original_columns)
    result_rows = list(rows)

    for t in transforms:
        kind = t.get("kind")
        if kind == "select":
            cols = t.get("columns", [])
            columns = [c for c in columns if c in cols]
            result_rows = [
                {c: row.get(c, "") for c in columns}
                for row in result_rows
            ]
            applied.append(f"selected {len(cols)} columns")
        elif kind == "rename":
            old_name = t.get("from", "")
            new_name = t.get("to", "")
            if old_name in columns:
                columns = [new_name if c == old_name else c for c in columns]
                result_rows = [
                    {new_name if k == old_name else k: v for k, v in row.items()}
                    for row in result_rows
                ]
                applied.append(f"renamed {old_name} -> {new_name}")
        elif kind == "filter":
            col = t.get("column", "")
            op = t.get("op", "not_empty")
            if op == "not_empty":
                result_rows = [
                    row for row in result_rows
                    if row.get(col, "").strip() != ""
                ]
                applied.append(f"filtered {col} not_empty")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result_rows)

    return {
        "status": "completed",
        "output_path": str(output_path),
        "input_rows": len(rows),
        "output_rows": len(result_rows),
        "original_columns": original_columns,
        "output_columns": columns,
        "transforms_applied": applied,
    }


def extract_fields_from_text(
    text_path: str | Path,
    output_dir: str | Path,
    field_spec: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Extract structured fields from a cleaned text file and export as CSV."""

    from src.operators.data_ops.text_processor import extract_medical_entities

    path = Path(text_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{path.stem}_entities.csv"

    raw_text = path.read_text(encoding="utf-8-sig")
    records = raw_text.split("---")

    field_spec = field_spec or {}
    entity_records = []
    for i, record in enumerate(records):
        record = record.strip()
        if not record:
            continue
        entities = extract_medical_entities(record)
        entity_records.append({
            "record_id": i + 1,
            "diseases": "; ".join(entities["diseases"]) or "none",
            "drugs": "; ".join(entities["drugs"]) or "none",
            "examinations": "; ".join(entities["examinations"]) or "none",
        })

    if entity_records:
        columns = list(entity_records[0].keys())
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(entity_records)

    return {
        "status": "completed",
        "output_path": str(output_path),
        "records_processed": len(entity_records),
        "fields_extracted": list(entity_records[0].keys()) if entity_records else [],
    }
