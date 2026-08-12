from __future__ import annotations

from typing import Any

from medinsight.common.knowledge_graph import stable_id
from medinsight.common.operators.base import OperatorSpec
from medinsight.task2.operators.entity_extraction import extract_object_value, normalize_medical_text_records
from medinsight.task2.operators.text_kg_extraction import extract_text_relations


class MedicalRelationExtractor:
    spec = OperatorSpec(
        name="medical_relation_extractor",
        description="Extract medical relations from generic medical text, with optional CMeIE SPO compatibility.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "medical_text_records": {"type": "array"},
                "cmeie_records": {"type": "array"},
            },
        },
        output_schema={"type": "object", "properties": {"relations": {"type": "array"}}},
    )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        records = payload.get("cmeie_records") or []
        split = str(payload.get("cmeie_split") or "train")
        relations: list[dict[str, Any]] = []
        relations.extend(self._from_medical_text(normalize_medical_text_records(payload)))
        for doc_index, record in enumerate(records):
            doc_id = str(record.get("doc_id") or f"cmeie-{split}-{doc_index:06d}")
            text = str(record.get("text") or "")
            for spo_index, spo in enumerate(record.get("spo_list") or []):
                subject = str(spo.get("subject") or "").strip()
                predicate = str(spo.get("predicate") or "").strip()
                object_value = extract_object_value(spo.get("object"))
                subject_type = str(spo.get("subject_type") or "未知").strip()
                object_type = extract_object_value(spo.get("object_type")) or "未知"
                if not subject or not predicate or not object_value:
                    continue
                relations.append(
                    {
                        "relation_id": stable_id(
                            "rel",
                            "cmeie",
                            doc_id,
                            spo_index,
                            subject,
                            predicate,
                            object_value,
                        ),
                        "doc_id": doc_id,
                        "subject": subject,
                        "subject_type": subject_type,
                        "predicate": predicate,
                        "object": object_value,
                        "object_type": object_type,
                        "evidence": text,
                        "source_dataset": "CMeIE-V2",
                        "source_split": split,
                        "confidence": 1.0,
                        "properties": {
                            "combined": bool(spo.get("Combined", False)),
                        },
                    }
                )
        return {
            "relations": relations,
            "summary": {
                "total_relations": len(relations),
                "from_medical_text": sum(1 for item in relations if item["source_dataset"] == "generic_medical_text"),
                "from_cmeie": sum(1 for item in relations if item["source_dataset"] == "CMeIE-V2"),
                "source_split": split,
            },
        }

    def _from_medical_text(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        relations: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            doc_id = str(record.get("doc_id") or f"medical-text-{index:06d}")
            text = str(record.get("text") or "")
            for relation in extract_text_relations(text, doc_id):
                relation["source_dataset"] = "generic_medical_text"
                relation["properties"]["source_dataset_label"] = "generic_medical_text"
                for meta_key in (
                    "source_refs",
                    "dataset",
                    "split",
                    "source_file",
                    "source_index",
                    "disease",
                    "disease_topic",
                    "title",
                    "metadata",
                ):
                    if meta_key in record:
                        relation["properties"][meta_key] = record[meta_key]
                relations.append(relation)
        return relations
