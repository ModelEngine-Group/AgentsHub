from __future__ import annotations

from typing import Any

from medinsight.common.knowledge_graph import stable_id
from medinsight.common.operators.base import OperatorSpec


ENHANCEMENT_RELATION_SCHEMAS = [
    {
        "subject_type": "临床术语",
        "predicate": "标准化为",
        "object_type": "标准术语",
        "source_dataset": "CHIP-CDN",
    },
    {
        "subject_type": "标准术语",
        "predicate": "ICD编码",
        "object_type": "ICD-10编码",
        "source_dataset": "CHIP-CDN-ICD10",
    },
    {
        "subject_type": "临床术语",
        "predicate": "标准化为",
        "object_type": "标准手术名称",
        "source_dataset": "CHIP-CDN-SurgeryLexicon",
    },
    {
        "subject_type": "临床术语",
        "predicate": "手术编码",
        "object_type": "手术编码",
        "source_dataset": "CHIP-CDN-SurgeryLexicon",
    },
    {
        "subject_type": "标准手术名称",
        "predicate": "手术编码",
        "object_type": "手术编码",
        "source_dataset": "CHIP-CDN-SurgeryLexicon",
    },
    {
        "subject_type": "医学事件",
        "predicate": "因果相关",
        "object_type": "医学事件",
        "source_dataset": "CMedCausal",
    },
    {
        "subject_type": "医学事件",
        "predicate": "条件因果",
        "object_type": "医学事件",
        "source_dataset": "CMedCausal",
    },
    {
        "subject_type": "医学事件",
        "predicate": "概念包含",
        "object_type": "医学事件",
        "source_dataset": "CMedCausal",
    },
]


CAUSAL_RELATION_LABELS = {
    1: "因果相关",
    2: "条件因果",
    3: "概念包含",
}


class MedicalGraphEnricher:
    spec = OperatorSpec(
        name="medical_graph_enricher",
        description="Enhance the medical graph with CHIP-CDN normalization aliases and CMedCausal causal relations.",
        input_schema={
            "type": "object",
            "properties": {
                "chip_cdn_records": {"type": "array"},
                "cmedcausal_records": {"type": "array"},
                "icd10_terms": {"type": "array"},
                "surgery_terms": {"type": "array"},
            },
        },
        output_schema={"type": "object", "properties": {"relations": {"type": "array"}}},
    )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        chip_split = str(payload.get("chip_cdn_split") or "train")
        causal_split = str(payload.get("cmedcausal_split") or "train")
        chip_relations = self._from_chip_cdn(payload.get("chip_cdn_records") or [], chip_split)
        causal_relations = self._from_cmedcausal(payload.get("cmedcausal_records") or [], causal_split)
        icd10_relations = self._from_icd10(payload.get("icd10_terms") or [])
        surgery_relations = self._from_surgery_lexicon(payload.get("surgery_terms") or [])
        relations = chip_relations + icd10_relations + surgery_relations + causal_relations
        return {
            "relations": relations,
            "schemas": ENHANCEMENT_RELATION_SCHEMAS,
            "summary": {
                "total_enhanced_relations": len(relations),
                "from_chip_cdn": len(chip_relations),
                "from_icd10": len(icd10_relations),
                "from_surgery_lexicon": len(surgery_relations),
                "from_cmedcausal": len(causal_relations),
                "schemas": ENHANCEMENT_RELATION_SCHEMAS,
            },
        }

    def _from_chip_cdn(self, records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        relations: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for doc_index, record in enumerate(records):
            raw_text = str(record.get("text") or "").strip()
            if not raw_text:
                continue
            doc_id = str(record.get("doc_id") or f"chip-cdn-{split}-{doc_index:06d}")
            for normalized in split_normalized_terms(record.get("normalized_result")):
                if not normalized or normalized == raw_text:
                    continue
                key = (raw_text, normalized)
                if key in seen:
                    continue
                seen.add(key)
                relations.append(
                    {
                        "relation_id": stable_id("rel", "chip-cdn", doc_id, raw_text, "标准化为", normalized),
                        "doc_id": doc_id,
                        "subject": raw_text,
                        "subject_type": "临床术语",
                        "predicate": "标准化为",
                        "object": normalized,
                        "object_type": "标准术语",
                        "evidence": raw_text,
                        "source_dataset": "CHIP-CDN",
                        "source_split": split,
                        "confidence": 1.0,
                        "properties": {
                            "normalization_source": "gold_normalized_result",
                        },
                    }
                )
        return relations

    def _from_icd10(self, terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
        relations: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for index, term in enumerate(terms):
            code = str(term.get("code") or "").strip()
            name = str(term.get("name") or "").strip()
            if not code or not name:
                continue
            key = (name, code)
            if key in seen:
                continue
            seen.add(key)
            doc_id = f"chip-cdn-icd10-{index:06d}"
            relations.append(
                {
                    "relation_id": stable_id("rel", "chip-cdn-icd10", name, "ICD编码", code),
                    "doc_id": doc_id,
                    "subject": name,
                    "subject_type": "标准术语",
                    "predicate": "ICD编码",
                    "object": code,
                    "object_type": "ICD-10编码",
                    "evidence": f"{code}\t{name}",
                    "source_dataset": "CHIP-CDN-ICD10",
                    "source_split": "lexicon",
                    "confidence": 1.0,
                    "properties": {
                        "code_system": "ICD-10 北京临床版 v601",
                    },
                }
            )
        return relations

    def _from_surgery_lexicon(self, terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
        relations: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for index, term in enumerate(terms):
            raw_name = str(term.get("raw_name") or "").strip()
            code = str(term.get("code") or "").strip()
            standard_name = str(term.get("standard_name") or "").strip()
            if not raw_name or not code or not standard_name:
                continue
            doc_id = f"chip-cdn-surgery-{index:06d}"
            relations.extend(
                self._surgery_relation(
                    seen,
                    doc_id,
                    raw_name,
                    "临床术语",
                    "标准化为",
                    standard_name,
                    "标准手术名称",
                    f"{raw_name}\t{code}\t{standard_name}",
                    code,
                )
            )
            relations.extend(
                self._surgery_relation(
                    seen,
                    doc_id,
                    raw_name,
                    "临床术语",
                    "手术编码",
                    code,
                    "手术编码",
                    f"{raw_name}\t{code}\t{standard_name}",
                    code,
                )
            )
            relations.extend(
                self._surgery_relation(
                    seen,
                    doc_id,
                    standard_name,
                    "标准手术名称",
                    "手术编码",
                    code,
                    "手术编码",
                    f"{raw_name}\t{code}\t{standard_name}",
                    code,
                )
            )
        return relations

    def _surgery_relation(
        self,
        seen: set[tuple[str, str, str]],
        doc_id: str,
        subject: str,
        subject_type: str,
        predicate: str,
        object_value: str,
        object_type: str,
        evidence: str,
        code: str,
    ) -> list[dict[str, Any]]:
        key = (subject, predicate, object_value)
        if key in seen or subject == object_value:
            return []
        seen.add(key)
        return [
            {
                "relation_id": stable_id("rel", "chip-cdn-surgery", subject, predicate, object_value),
                "doc_id": doc_id,
                "subject": subject,
                "subject_type": subject_type,
                "predicate": predicate,
                "object": object_value,
                "object_type": object_type,
                "evidence": evidence,
                "source_dataset": "CHIP-CDN-SurgeryLexicon",
                "source_split": "lexicon",
                "confidence": 1.0,
                "properties": {
                    "code_system": "Chinese surgery terminology code",
                    "surgery_code": code,
                },
            }
        ]

    def _from_cmedcausal(self, records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        relations: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for doc_index, record in enumerate(records):
            doc_id = str(record.get("doc_id") or f"cmedcausal-{split}-{doc_index:06d}")
            text = str(record.get("text") or "")
            for relation_index, relation in enumerate(record.get("relation_of_mention") or []):
                for extracted in flatten_causal_relation(relation):
                    subject = extracted.get("head")
                    object_value = extracted.get("tail")
                    relation_code = extracted.get("relation")
                    predicate = CAUSAL_RELATION_LABELS.get(relation_code, f"CMedCausal-R{relation_code}")
                    if not subject or not object_value:
                        continue
                    key = (subject, predicate, object_value)
                    if key in seen:
                        continue
                    seen.add(key)
                    relations.append(
                        {
                            "relation_id": stable_id(
                                "rel",
                                "cmedcausal",
                                doc_id,
                                relation_index,
                                subject,
                                predicate,
                                object_value,
                            ),
                            "doc_id": doc_id,
                            "subject": subject,
                            "subject_type": "医学事件",
                            "predicate": predicate,
                            "object": object_value,
                            "object_type": "医学事件",
                            "evidence": text,
                            "source_dataset": "CMedCausal",
                            "source_split": split,
                            "confidence": 0.92,
                            "properties": {
                                "relation_code": relation_code,
                                "mapping_note": "CMedCausal labels are preserved as relation_code; predicate is a conservative display label.",
                            },
                        }
                    )
        return relations


def split_normalized_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_terms = [str(item) for item in value]
    else:
        raw_terms = str(value).split("##")
    terms: list[str] = []
    for item in raw_terms:
        cleaned = item.strip().strip('"').strip("'").strip()
        if cleaned and cleaned not in terms:
            terms.append(cleaned)
    return terms


def mention_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("mention") or "").strip()
    return ""


def flatten_causal_relation(relation: dict[str, Any]) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    head = mention_text(relation.get("head"))
    relation_code = relation.get("relation")
    tail = relation.get("tail") or {}
    if isinstance(tail, dict) and tail.get("type") == "relation":
        nested = {
            "head": tail.get("head"),
            "relation": tail.get("relation"),
            "tail": tail.get("tail"),
        }
        extracted.extend(flatten_causal_relation(nested))
        nested_tail = mention_text(tail.get("tail"))
        if head and nested_tail:
            extracted.append({"head": head, "relation": relation_code, "tail": nested_tail})
        return extracted

    tail_text = mention_text(tail)
    if head and tail_text:
        extracted.append({"head": head, "relation": relation_code, "tail": tail_text})
    return extracted
