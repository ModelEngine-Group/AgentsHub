from __future__ import annotations

from typing import Any

from medinsight.common.knowledge_graph import stable_id
from medinsight.common.operators.base import OperatorSpec
from medinsight.task2.operators.text_kg_extraction import SORTED_ENTITY_LEXICONS, detect_disease, normalize_text, split_sentences


CMEEE_ENTITY_TYPES = {
    "dis": "疾病",
    "sym": "症状",
    "dru": "药物",
    "pro": "医疗程序",
    "bod": "身体部位",
    "ite": "医学检查项目",
    "dep": "科室",
    "equ": "医疗设备",
    "mic": "微生物",
}


def normalize_entity_type(type_code: str | None) -> str:
    if not type_code:
        return "未知"
    return CMEEE_ENTITY_TYPES.get(type_code, type_code)


class MedicalEntityExtractor:
    spec = OperatorSpec(
        name="medical_entity_extractor",
        description="Extract medical entities from generic medical text, with optional CBLUE annotation compatibility.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "medical_text_records": {"type": "array"},
                "cmeee_records": {"type": "array"},
                "cmeie_records": {"type": "array"},
            },
        },
        output_schema={"type": "object", "properties": {"entities": {"type": "array"}}},
    )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        entities: list[dict[str, Any]] = []
        text_records = normalize_medical_text_records(payload)
        entities.extend(self._from_medical_text(text_records))
        entities.extend(self._from_cmeee(payload.get("cmeee_records") or [], payload.get("cmeee_split", "train")))
        entities.extend(self._from_cmeie(payload.get("cmeie_records") or [], payload.get("cmeie_split", "train")))
        return {
            "entities": entities,
            "summary": {
                "total_entities": len(entities),
                "from_medical_text": sum(1 for item in entities if item["source_dataset"] == "generic_medical_text"),
                "from_cmeee": sum(1 for item in entities if item["source_dataset"] == "CMeEE-V2"),
                "from_cmeie": sum(1 for item in entities if item["source_dataset"] == "CMeIE-V2"),
            },
        }

    def _from_medical_text(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for doc_index, record in enumerate(records):
            doc_id = str(record.get("doc_id") or f"medical-text-{doc_index:06d}")
            text = normalize_text(str(record.get("text") or ""))
            if not text:
                continue
            for sentence in split_sentences(text):
                disease = detect_disease(sentence)
                if disease:
                    self._append_text_entity(entities, seen, doc_id, text, disease, "疾病", "disease_trigger")
            for entity_type, terms in SORTED_ENTITY_LEXICONS.items():
                for term in terms:
                    if term in text:
                        self._append_text_entity(entities, seen, doc_id, text, term, entity_type, "lexicon_match")
        return entities

    def _append_text_entity(
        self,
        entities: list[dict[str, Any]],
        seen: set[tuple[str, str, str]],
        doc_id: str,
        text: str,
        name: str,
        entity_type: str,
        method: str,
    ) -> None:
        key = (doc_id, name, entity_type)
        if key in seen:
            return
        seen.add(key)
        start_idx = text.find(name)
        entities.append(
            {
                "mention_id": stable_id("mention", "medical-text", doc_id, name, entity_type),
                "doc_id": doc_id,
                "name": name,
                "type": entity_type,
                "type_code": entity_type,
                "start_idx": start_idx if start_idx >= 0 else None,
                "end_idx": start_idx + len(name) - 1 if start_idx >= 0 else None,
                "source_dataset": "generic_medical_text",
                "source_split": "user_input",
                "extraction_method": method,
                "confidence": 0.82 if method == "lexicon_match" else 0.74,
            }
        )

    def _from_cmeee(self, records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        for doc_index, record in enumerate(records):
            doc_id = str(record.get("doc_id") or f"cmeee-{split}-{doc_index:06d}")
            text = str(record.get("text") or "")
            for entity_index, item in enumerate(record.get("entities") or []):
                name = str(item.get("entity") or text[item.get("start_idx", 0) : item.get("end_idx", 0) + 1])
                type_code = str(item.get("type") or "")
                entity_type = normalize_entity_type(type_code)
                entities.append(
                    {
                        "mention_id": stable_id("mention", "cmeee", doc_id, entity_index, name, type_code),
                        "doc_id": doc_id,
                        "name": name,
                        "type": entity_type,
                        "type_code": type_code,
                        "start_idx": item.get("start_idx"),
                        "end_idx": item.get("end_idx"),
                        "source_dataset": "CMeEE-V2",
                        "source_split": split,
                        "extraction_method": "gold_annotation",
                        "confidence": 1.0,
                    }
                )
        return entities

    def _from_cmeie(self, records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for doc_index, record in enumerate(records):
            doc_id = str(record.get("doc_id") or f"cmeie-{split}-{doc_index:06d}")
            for spo_index, spo in enumerate(record.get("spo_list") or []):
                subject = str(spo.get("subject") or "").strip()
                subject_type = str(spo.get("subject_type") or "未知").strip()
                object_value = extract_object_value(spo.get("object"))
                object_type = extract_object_value(spo.get("object_type"))
                for role, name, entity_type in (
                    ("subject", subject, subject_type),
                    ("object", object_value, object_type),
                ):
                    if not name:
                        continue
                    key = (doc_id, name, entity_type)
                    if key in seen:
                        continue
                    seen.add(key)
                    entities.append(
                        {
                            "mention_id": stable_id("mention", "cmeie", doc_id, role, spo_index, name, entity_type),
                            "doc_id": doc_id,
                            "name": name,
                            "type": entity_type or "未知",
                            "type_code": entity_type or "未知",
                            "start_idx": None,
                            "end_idx": None,
                            "source_dataset": "CMeIE-V2",
                            "source_split": split,
                            "extraction_method": "relation_anchor",
                            "confidence": 1.0,
                        }
                    )
        return entities


def extract_object_value(value: Any) -> str:
    if isinstance(value, dict):
        if "@value" in value:
            return str(value.get("@value") or "").strip()
        for item in value.values():
            if isinstance(item, str) and item.strip():
                return item.strip()
        return ""
    return str(value or "").strip()


def normalize_medical_text_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    text = str(payload.get("text") or "").strip()
    if text:
        records.append({"doc_id": payload.get("doc_id") or "medical-text-000001", "text": text})
    for key in ("medical_text_records", "medical_records", "documents"):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        for index, item in enumerate(value):
            if isinstance(item, dict):
                item_text = str(item.get("text") or item.get("content") or "").strip()
                if item_text:
                    record = {
                        "doc_id": item.get("doc_id") or item.get("id") or f"medical-text-{index:06d}",
                        "text": item_text,
                    }
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
                        if meta_key in item:
                            record[meta_key] = item[meta_key]
                    records.append(record)
            elif isinstance(item, str) and item.strip():
                records.append({"doc_id": f"medical-text-{index:06d}", "text": item.strip()})
    return records
