from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from medinsight.common.operators.base import OperatorSpec
from medinsight.common.schemas import Document
from medinsight.common.utils import clean_text


LIST_FIELDS = {"symptom", "drug", "checklist", "treatment", "department", "complication"}
FIELD_ALIASES = {
    "name": "disease",
    "disease": "disease",
    "疾病": "disease",
    "symptom": "symptom",
    "symptoms": "symptom",
    "症状": "symptom",
    "drug": "drug",
    "drugs": "drug",
    "药品": "drug",
    "checklist": "checklist",
    "checks": "checklist",
    "检查": "checklist",
    "treatment": "treatment",
    "治疗": "treatment",
    "department": "department",
    "科室": "department",
    "complication": "complication",
    "并发症": "complication",
    "source_dataset": "source_dataset",
}


def split_cell(value: Any) -> list[str]:
    text = clean_text(str(value or ""))
    if not text:
        return []
    for sep in ["；", ";", "，", ",", "|", "、"]:
        text = text.replace(sep, " ")
    return [item.strip() for item in text.split() if item.strip()]


class MedicalTableProfiler:
    spec = OperatorSpec(
        name="medical_table_profiler",
        description="Profile structured medical CSV/table records, including schema, missing values and field types.",
        input_schema={"rows": "list[dict]"},
        output_schema={"profile": "dict"},
    )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rows = payload["rows"]
        total = len(rows)
        columns = sorted({field for row in rows for field in row})
        missing = {
            column: sum(1 for row in rows if not str(row.get(column) or "").strip())
            for column in columns
        }
        type_hints = {column: self._type_hint([row.get(column) for row in rows]) for column in columns}
        profile = {
            "row_count": total,
            "column_count": len(columns),
            "columns": columns,
            "missing_count": missing,
            "missing_ratio": {
                column: round(count / total, 4) if total else 0.0
                for column, count in missing.items()
            },
            "type_hints": type_hints,
            "canonical_columns": sorted({FIELD_ALIASES.get(column, column) for column in columns}),
            "detected_mode": "structured_table",
        }
        return {"profile": profile}

    def _type_hint(self, values: list[Any]) -> str:
        non_empty = [str(value).strip() for value in values if str(value or "").strip()]
        if not non_empty:
            return "empty"
        numeric = 0
        for value in non_empty:
            try:
                float(value)
                numeric += 1
            except ValueError:
                pass
        if numeric == len(non_empty):
            return "numeric"
        if len(set(non_empty)) <= max(3, len(non_empty) // 2):
            return "categorical"
        return "text"


class MedicalTableNormalizer:
    spec = OperatorSpec(
        name="medical_table_normalizer",
        description="Normalize structured medical table rows into a canonical disease-centric schema.",
        input_schema={"rows": "list[dict]"},
        output_schema={"rows": "list[dict]", "anomalies": "list[dict]"},
    )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized: List[dict[str, Any]] = []
        anomalies: List[dict[str, Any]] = []
        for index, row in enumerate(payload["rows"], start=1):
            item: dict[str, Any] = {"row_id": str(row.get("row_id") or f"table-row-{index}")}
            for key, value in row.items():
                canonical = FIELD_ALIASES.get(str(key).strip(), str(key).strip())
                item[canonical] = clean_text(str(value or ""))
            disease = item.get("disease", "")
            if not disease:
                disease = f"unknown-disease-{index}"
                item["disease"] = disease
                anomalies.append(
                    {
                        "row_id": item["row_id"],
                        "level": "warning",
                        "action": "filled_default_disease",
                        "message": "missing disease/name field; generated replacement value",
                    }
                )
            for field in LIST_FIELDS:
                values = split_cell(item.get(field))
                item[f"{field}_list"] = values
                if not values:
                    anomalies.append(
                        {
                            "row_id": item["row_id"],
                            "level": "warning",
                            "action": "empty_structured_field",
                            "field": field,
                            "message": f"empty structured field: {field}",
                        }
                    )
            normalized.append(item)
        return {"rows": normalized, "anomalies": anomalies}


class MedicalTableDocumentBuilder:
    spec = OperatorSpec(
        name="medical_table_document_builder",
        description="Convert normalized structured medical table rows into reusable Document objects.",
        input_schema={"rows": "list[dict]"},
        output_schema={"documents": "list[Document]"},
    )

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        documents: list[Document] = []
        for index, row in enumerate(payload["rows"], start=1):
            disease = str(row.get("disease") or f"document-{index}")
            text = (
                f"{disease}的症状包括{self._join(row.get('symptom_list'))}。"
                f"{disease}的诊断检查可参考{self._join(row.get('checklist_list'))}。"
                f"{disease}的治疗方式包括{self._join(row.get('treatment_list'))}。"
                f"{disease}相关药品包括{self._join(row.get('drug_list'))}。"
                f"{disease}可能并发{self._join(row.get('complication_list'))}。"
                f"{disease}建议就诊科室包括{self._join(row.get('department_list'))}。"
            )
            documents.append(
                Document(
                    doc_id=str(row.get("row_id") or f"table-row-{index}"),
                    title=disease,
                    text=clean_text(text),
                    metadata={k: v for k, v in row.items() if k not in {"row_id", "disease"}},
                )
            )
        return {"documents": documents}

    def _join(self, value: Any) -> str:
        if isinstance(value, list) and value:
            return "、".join(str(item) for item in value)
        return "未提供"
