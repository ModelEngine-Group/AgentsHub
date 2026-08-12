from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from medinsight.common.knowledge_graph import stable_id
from medinsight.common.operators.base import OperatorSpec


class MedicalTripleBuilder:
    spec = OperatorSpec(
        name="medical_triple_builder",
        description="Build canonical medical triples from extracted relations.",
        input_schema={"type": "object", "properties": {"relations": {"type": "array"}}},
        output_schema={"type": "object", "properties": {"triples": {"type": "array"}}},
    )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        relations = payload.get("relations") or []
        triples_by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        for relation in relations:
            key = (
                str(relation.get("subject") or ""),
                str(relation.get("subject_type") or ""),
                str(relation.get("predicate") or ""),
                str(relation.get("object") or ""),
                str(relation.get("object_type") or ""),
            )
            if key not in triples_by_key:
                triples_by_key[key] = {
                    "triple_id": stable_id("triple", *key),
                    "subject": key[0],
                    "subject_type": key[1],
                    "predicate": key[2],
                    "object": key[3],
                    "object_type": key[4],
                    "confidence": float(relation.get("confidence") or 1.0),
                    "source_dataset": relation.get("source_dataset"),
                    "evidence": [],
                    "properties": {
                        "source_relation_ids": [],
                    },
                }
            triple = triples_by_key[key]
            triple["properties"]["source_relation_ids"].append(relation.get("relation_id"))
            if len(triple["evidence"]) < 3:
                evidence_item = {
                    "doc_id": relation.get("doc_id"),
                    "text": relation.get("evidence"),
                    "source_split": relation.get("source_split"),
                }
                relation_properties = relation.get("properties") or {}
                for meta_key in (
                    "source_refs",
                    "dataset",
                    "split",
                    "source_file",
                    "source_index",
                    "disease",
                    "title",
                ):
                    if meta_key in relation_properties:
                        evidence_item[meta_key] = relation_properties[meta_key]
                triple["evidence"].append(evidence_item)
        triples = list(triples_by_key.values())
        return {
            "triples": triples,
            "summary": {
                "raw_relations": len(relations),
                "unique_triples": len(triples),
                "deduplicated": len(relations) - len(triples),
            },
        }


class MedicalTripleValidator:
    spec = OperatorSpec(
        name="medical_triple_validator",
        description="Validate triples against CMeIE schema and basic graph quality rules.",
        input_schema={
            "type": "object",
            "properties": {
                "triples": {"type": "array"},
                "schema": {"type": "array"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "valid_triples": {"type": "array"},
                "invalid_triples": {"type": "array"},
            },
        },
    )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        triples = payload.get("triples") or []
        schema = (payload.get("schema") or []) + (payload.get("extra_schema") or [])
        exact_schema = {
            (
                str(item.get("subject_type") or ""),
                str(item.get("predicate") or ""),
                str(item.get("object_type") or ""),
            )
            for item in schema
        }
        predicates = {str(item.get("predicate") or "") for item in schema}
        valid: list[dict[str, Any]] = []
        invalid: list[dict[str, Any]] = []
        invalid_reasons = Counter()
        for triple in triples:
            reasons: list[str] = []
            if not triple.get("subject") or not triple.get("predicate") or not triple.get("object"):
                reasons.append("empty_field")
            schema_key = (
                str(triple.get("subject_type") or ""),
                str(triple.get("predicate") or ""),
                str(triple.get("object_type") or ""),
            )
            if schema_key not in exact_schema:
                if triple.get("predicate") in predicates:
                    reasons.append("type_mismatch_with_schema")
                else:
                    reasons.append("predicate_not_in_schema")
            if reasons:
                invalid_item = dict(triple)
                invalid_item["validation_errors"] = reasons
                invalid.append(invalid_item)
                invalid_reasons.update(reasons)
            else:
                valid_item = dict(triple)
                valid_item["validation_status"] = "valid"
                valid.append(valid_item)
        return {
            "valid_triples": valid,
            "invalid_triples": invalid,
            "summary": {
                "input_triples": len(triples),
                "valid_triples": len(valid),
                "invalid_triples": len(invalid),
                "valid_rate": round(len(valid) / len(triples), 4) if triples else 0.0,
                "invalid_reasons": dict(invalid_reasons),
            },
        }


class MedicalTripleRepairer:
    spec = OperatorSpec(
        name="medical_triple_repairer",
        description="Repair schema type mismatches when the target medical relation schema is unambiguous.",
        input_schema={
            "type": "object",
            "properties": {
                "invalid_triples": {"type": "array"},
                "schema": {"type": "array"},
                "extra_schema": {"type": "array"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "repaired_triples": {"type": "array"},
                "unrepaired_triples": {"type": "array"},
                "repair_report": {"type": "object"},
            },
        },
    )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        invalid_triples = payload.get("invalid_triples") or []
        schema = (payload.get("schema") or []) + (payload.get("extra_schema") or [])
        schema_by_predicate: dict[str, list[dict[str, Any]]] = {}
        for item in schema:
            predicate = str(item.get("predicate") or "")
            if predicate:
                schema_by_predicate.setdefault(predicate, []).append(item)

        repaired: list[dict[str, Any]] = []
        unrepaired: list[dict[str, Any]] = []
        repair_reasons = Counter()
        for triple in invalid_triples:
            candidate = self._select_repair_schema(triple, schema_by_predicate)
            if candidate:
                repaired_item = deepcopy(triple)
                original_types = {
                    "subject_type": repaired_item.get("subject_type"),
                    "object_type": repaired_item.get("object_type"),
                }
                repaired_item["subject_type"] = candidate["subject_type"]
                repaired_item["object_type"] = candidate["object_type"]
                repaired_item["triple_id"] = stable_id(
                    "triple",
                    str(repaired_item.get("subject") or ""),
                    str(repaired_item.get("subject_type") or ""),
                    str(repaired_item.get("predicate") or ""),
                    str(repaired_item.get("object") or ""),
                    str(repaired_item.get("object_type") or ""),
                )
                repaired_item["validation_status"] = "schema_repaired"
                repaired_item["original_validation_errors"] = repaired_item.pop("validation_errors", [])
                repaired_item.setdefault("properties", {})
                repaired_item["properties"]["repair"] = {
                    "operator": self.spec.name,
                    "strategy": "schema_type_alignment",
                    "original_types": original_types,
                    "repaired_types": {
                        "subject_type": candidate["subject_type"],
                        "object_type": candidate["object_type"],
                    },
                    "reason": "The predicate has an unambiguous or type-compatible schema candidate.",
                }
                repaired.append(repaired_item)
                repair_reasons.update(["schema_type_alignment"])
            else:
                unrepaired_item = deepcopy(triple)
                unrepaired_item["repair_status"] = "unrepaired"
                unrepaired_item["repair_reason"] = self._unrepaired_reason(triple, schema_by_predicate)
                unrepaired.append(unrepaired_item)
                repair_reasons.update([unrepaired_item["repair_reason"]])

        return {
            "repaired_triples": repaired,
            "unrepaired_triples": unrepaired,
            "repair_report": {
                "input_invalid_triples": len(invalid_triples),
                "repaired_triples": len(repaired),
                "unrepaired_triples": len(unrepaired),
                "repair_rate": round(len(repaired) / len(invalid_triples), 4) if invalid_triples else 0.0,
                "repair_reasons": dict(repair_reasons),
            },
        }

    def _select_repair_schema(
        self,
        triple: dict[str, Any],
        schema_by_predicate: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        errors = set(triple.get("validation_errors") or [])
        if "empty_field" in errors or "predicate_not_in_schema" in errors:
            return None
        if "type_mismatch_with_schema" not in errors:
            return None
        predicate = str(triple.get("predicate") or "")
        candidates = schema_by_predicate.get(predicate) or []
        if len(candidates) == 1:
            return candidates[0]

        subject_type = str(triple.get("subject_type") or "")
        object_type = str(triple.get("object_type") or "")
        compatible = [
            item
            for item in candidates
            if subject_type in {"", str(item.get("subject_type") or "")}
            or object_type in {"", str(item.get("object_type") or "")}
        ]
        return compatible[0] if len(compatible) == 1 else None

    def _unrepaired_reason(
        self,
        triple: dict[str, Any],
        schema_by_predicate: dict[str, list[dict[str, Any]]],
    ) -> str:
        errors = set(triple.get("validation_errors") or [])
        if "empty_field" in errors:
            return "empty_field_not_repairable"
        if "predicate_not_in_schema" in errors:
            return "unknown_predicate_not_repairable"
        predicate = str(triple.get("predicate") or "")
        if not schema_by_predicate.get(predicate):
            return "no_schema_candidate"
        return "ambiguous_schema_candidate"


class MedicalConflictDetector:
    spec = OperatorSpec(
        name="medical_conflict_detector",
        description="Detect duplicate and schema-type conflicts in generated medical triples.",
        input_schema={"type": "object", "properties": {"triples": {"type": "array"}}},
        output_schema={
            "type": "object",
            "properties": {
                "conflicts": {"type": "array"},
                "summary": {"type": "object"},
            },
        },
    )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        triples = payload.get("triples") or []
        exact_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        spo_type_groups: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
        for triple in triples:
            exact_key = (
                str(triple.get("subject") or ""),
                str(triple.get("subject_type") or ""),
                str(triple.get("predicate") or ""),
                str(triple.get("object") or ""),
                str(triple.get("object_type") or ""),
            )
            exact_groups.setdefault(exact_key, []).append(triple)
            spo_key = (
                str(triple.get("subject") or ""),
                str(triple.get("predicate") or ""),
                str(triple.get("object") or ""),
            )
            spo_type_groups.setdefault(spo_key, set()).add((exact_key[1], exact_key[4]))

        conflicts: list[dict[str, Any]] = []
        for key, group in exact_groups.items():
            if len(group) > 1:
                conflicts.append(
                    {
                        "type": "duplicate_triple",
                        "subject": key[0],
                        "predicate": key[2],
                        "object": key[3],
                        "count": len(group),
                        "triple_ids": [item.get("triple_id") for item in group],
                    }
                )
        for key, type_pairs in spo_type_groups.items():
            if len(type_pairs) > 1:
                conflicts.append(
                    {
                        "type": "entity_type_conflict",
                        "subject": key[0],
                        "predicate": key[1],
                        "object": key[2],
                        "type_pairs": [
                            {"subject_type": subject_type, "object_type": object_type}
                            for subject_type, object_type in sorted(type_pairs)
                        ],
                    }
                )

        conflict_types = Counter(item["type"] for item in conflicts)
        return {
            "conflicts": conflicts,
            "summary": {
                "input_triples": len(triples),
                "conflict_count": len(conflicts),
                "conflict_types": dict(conflict_types),
                "passed": len(conflicts) == 0,
            },
        }
