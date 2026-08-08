"""Deterministic dataset quality profiling and duplicate detection."""
from __future__ import annotations

import hashlib
import statistics

from medigraph.operators.base import BaseOperator, OperatorMeta


class DataQualityOperator(BaseOperator):
    def __init__(self):
        self.meta = OperatorMeta(
            name="data_quality",
            version="1.0.0",
            description=(
                "检查文档/记录的数据质量：空值、重复、长度异常和字段缺失；默认去重。"
                "输入 {documents?或text?,required_fields?}, 输出 {documents,text,quality_report}。"
            ),
            input_schema={"type": "object", "properties": {"documents": {"type": "array"}}},
            output_schema={
                "type": "object",
                "properties": {
                    "documents": {"type": "array"},
                    "text": {"type": "string"},
                    "quality_report": {"type": "object"},
                },
                "required": ["quality_report"],
            },
        )

    def run(self, inputs: dict, **kwargs) -> dict:
        documents = inputs.get("documents")
        if not isinstance(documents, list):
            documents = [{"fileName": "inline", "text": str(inputs.get("text", "") or "")}]
        required = inputs.get("required_fields", [])
        required = required if isinstance(required, list) else []
        deduplicate = bool(inputs.get("deduplicate", True))
        clean, duplicates, empty, missing_fields = [], [], [], []
        seen: dict[str, int] = {}
        lengths = []
        for index, document in enumerate(documents):
            if not isinstance(document, dict):
                empty.append(index)
                continue
            text = str(document.get("text", "") or "").strip()
            if not text:
                empty.append(index)
                continue
            missing = [field for field in required if document.get(field) in (None, "", [])]
            if missing:
                missing_fields.append({"index": index, "fields": missing})
            digest = hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()
            if digest in seen:
                duplicates.append({"index": index, "duplicate_of": seen[digest], "sha256": digest})
                if deduplicate:
                    continue
            else:
                seen[digest] = index
            value = dict(document)
            value["content_sha256"] = digest
            clean.append(value)
            lengths.append(len(text))
        report = {
            "input_records": len(documents),
            "output_records": len(clean),
            "empty_records": len(empty),
            "duplicate_records": len(duplicates),
            "missing_field_records": len(missing_fields),
            "completeness": round((len(documents) - len(empty)) / len(documents), 4) if documents else 1.0,
            "uniqueness": round((len(documents) - len(duplicates)) / len(documents), 4) if documents else 1.0,
            "length": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "mean": round(statistics.fmean(lengths), 2) if lengths else 0.0,
                "median": round(statistics.median(lengths), 2) if lengths else 0.0,
            },
            "duplicate_details": duplicates[:100],
            "missing_field_details": missing_fields[:100],
        }
        return {
            "documents": clean,
            "text": "\n\n".join(str(document.get("text", "")) for document in clean),
            "quality_report": report,
        }
