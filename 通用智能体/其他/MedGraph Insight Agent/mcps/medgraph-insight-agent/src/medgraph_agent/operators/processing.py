from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from medgraph_agent.core.models import DataRecord, stable_id, utc_now
from medgraph_agent.operators.base import Operator, fail_result, ok_result


def load_records(source: str | Path) -> list[DataRecord]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"source does not exist: {path}")
    if path.is_dir():
        records: list[DataRecord] = []
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in {".jsonl", ".csv", ".txt", ".md"}:
                records.extend(load_records(child))
        return records
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    if path.suffix.lower() in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
        return [DataRecord(id=stable_id("rec", str(path), text), source=str(path), text=text)]
    raise ValueError(f"unsupported source format: {path.suffix}")


def _load_jsonl(path: Path) -> list[DataRecord]:
    records: list[DataRecord] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        text = str(payload.get("text") or payload.get("content") or "")
        if not text.strip():
            continue
        metadata = {k: v for k, v in payload.items() if k not in {"text", "content"}}
        record_id = str(payload.get("id") or stable_id("rec", str(path), str(line_no), text))
        records.append(DataRecord(id=record_id, source=str(path), text=text, metadata=metadata))
    return records


def _load_csv(path: Path) -> list[DataRecord]:
    records: list[DataRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row_no, row in enumerate(reader, start=1):
            text_fields = [
                str(value).strip()
                for key, value in row.items()
                if value and key.lower() not in {"id", "source"}
            ]
            text = "。".join(text_fields)
            if not text:
                continue
            record_id = str(row.get("id") or stable_id("rec", str(path), str(row_no), text))
            records.append(DataRecord(id=record_id, source=str(path), text=text, metadata=dict(row)))
    return records


def normalize_medical_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[；;]+", "；", text)
    text = re.sub(r"[。]{2,}", "。", text)
    text = text.replace(" ,", ",").replace(" 。", "。").strip()
    return text


class DataIngestionOperator(Operator):
    name = "data_ingestion"

    def execute(self, context: dict[str, Any]):
        started_at = utc_now()
        try:
            records = load_records(context["source"])
            context["records"] = records
            return ok_result(
                self.name,
                started_at,
                records_processed=len(records),
                output={"records": [record.__dict__ for record in records]},
            )
        except Exception as exc:
            return fail_result(self.name, started_at, exc)


class TextCleaningOperator(Operator):
    name = "text_cleaning"

    def execute(self, context: dict[str, Any]):
        started_at = utc_now()
        try:
            cleaned: list[DataRecord] = []
            for record in context.get("records", []):
                text = normalize_medical_text(record.text)
                cleaned.append(
                    DataRecord(
                        id=record.id,
                        source=record.source,
                        text=text,
                        metadata={**record.metadata, "cleaned_at": utc_now()},
                    )
                )
            context["records"] = cleaned
            return ok_result(
                self.name,
                started_at,
                records_processed=len(cleaned),
                output={"records": [record.__dict__ for record in cleaned]},
            )
        except Exception as exc:
            return fail_result(self.name, started_at, exc)
