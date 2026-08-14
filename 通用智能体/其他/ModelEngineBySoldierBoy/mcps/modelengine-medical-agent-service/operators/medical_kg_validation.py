"""Validation and aggregation helpers for the medical knowledge graph."""

from __future__ import annotations

from collections import Counter
from typing import Any


VALID_ENTITY_TYPES = {
    "Disease",
    "Symptom",
    "Drug",
    "Test",
    "Treatment",
    "Department",
    "RiskFactor",
    "Anatomy",
    "ClinicalStage",
    "PatientGroup",
    "History",
    "TransmissionRoute",
    "Epidemiology",
}
VALID_RELATIONS = {
    "has_symptom": {"Disease": {"Symptom"}},
    "treated_by": {"Disease": {"Drug", "Treatment"}},
    "diagnosed_by": {"Disease": {"Test"}},
    "has_complication": {"Disease": {"Disease"}},
    "belongs_to_department": {"Disease": {"Department"}},
    "contraindicated_for": {"Drug": {"Disease", "Symptom"}},
    "has_risk_factor": {"Disease": {"Disease", "Symptom", "Treatment", "RiskFactor"}},
    "prevented_by": {"Disease": {"Treatment"}},
    "causes": {
        "Disease": {"Disease", "Symptom"},
        "RiskFactor": {"Disease", "Symptom"},
        "Treatment": {"Disease", "Symptom"},
    },
    "located_in": {"Disease": {"Anatomy"}},
    "has_stage": {"Disease": {"ClinicalStage"}},
    "has_patient_group": {"Disease": {"PatientGroup"}},
    "has_history": {"Disease": {"History"}},
    "spreads_via": {"Disease": {"TransmissionRoute"}},
    "has_epidemiology": {"Disease": {"Epidemiology"}},
}


def _source_doc_ids_from_triple(triple: dict[str, Any]) -> list[str]:
    source_doc_ids = triple.get("source_doc_ids")
    if isinstance(source_doc_ids, list):
        values = [str(item) for item in source_doc_ids if str(item).strip()]
        if values:
            return values
    source_doc_id = str(triple.get("source_doc_id", "")).strip()
    return [source_doc_id] if source_doc_id else []


def _evidence_quality_score(triple: dict[str, Any]) -> float:
    evidence_quality = triple.get("evidence_quality")
    if not isinstance(evidence_quality, dict):
        return 0.0
    try:
        return float(evidence_quality.get("score", 0))
    except (TypeError, ValueError):
        return 0.0


def _support_score(confidence: float, evidence: str, evidence_quality_score: float, source_doc_count: int) -> float:
    evidence_bonus = 0.03 if evidence else 0.0
    source_bonus = min(0.05, max(0, source_doc_count - 1) * 0.015)
    quality_bonus = min(0.05, evidence_quality_score * 0.05)
    return round(min(1.0, confidence + evidence_bonus + source_bonus + quality_bonus), 4)


def _is_valid_triple(triple: dict[str, Any]) -> bool:
    subject = str(triple.get("subject", "")).strip()
    subject_type = str(triple.get("subject_type", "")).strip()
    predicate = str(triple.get("predicate", "")).strip()
    object_type = str(triple.get("object_type", "")).strip()
    object_name = str(triple.get("object", "")).strip()
    confidence = triple.get("confidence", 0)

    if not subject or not object_name or not predicate:
        return False
    if subject_type not in VALID_ENTITY_TYPES or object_type not in VALID_ENTITY_TYPES:
        return False
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        return False

    relation_spec = VALID_RELATIONS.get(predicate)
    if relation_spec is None:
        return False
    allowed_objects = relation_spec.get(subject_type)
    if not allowed_objects or object_type not in allowed_objects:
        return False
    return True


def aggregate_and_validate_triples(triples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate triples and merge identical facts across source documents."""
    aggregated: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    duplicate_count = 0
    rejected_count = 0
    conflict_filtered_count = 0
    conflict_reasons: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    arbitration_events: list[dict[str, Any]] = []

    for triple in triples:
        if not _is_valid_triple(triple):
            rejected_count += 1
            subject_type = str(triple.get("subject_type", "")).strip()
            predicate = str(triple.get("predicate", "")).strip()
            object_type = str(triple.get("object_type", "")).strip()
            if predicate not in VALID_RELATIONS:
                rejection_reasons["invalid_relation"] += 1
            elif subject_type not in VALID_ENTITY_TYPES or object_type not in VALID_ENTITY_TYPES:
                rejection_reasons["invalid_entity_type"] += 1
            else:
                rejection_reasons["relation_schema_mismatch"] += 1
            continue

        opposite_key = (
            str(triple["object"]),
            str(triple["object_type"]),
            str(triple["predicate"]),
            str(triple["subject"]),
            str(triple["subject_type"]),
        )
        if str(triple["predicate"]) in {"has_complication", "causes"} and opposite_key in aggregated:
            existing = aggregated[opposite_key]
            confidence = float(triple.get("confidence", 0))
            if confidence <= float(existing.get("confidence", 0)):
                conflict_filtered_count += 1
                conflict_reasons[f"reverse_{triple['predicate']}"] += 1
                arbitration_events.append(
                    {
                        "type": f"reverse_{triple['predicate']}",
                        "winner": {
                            "subject": existing.get("subject"),
                            "predicate": existing.get("predicate"),
                            "object": existing.get("object"),
                            "confidence": existing.get("confidence"),
                        },
                        "discarded": {
                            "subject": triple.get("subject"),
                            "predicate": triple.get("predicate"),
                            "object": triple.get("object"),
                            "confidence": confidence,
                        },
                        "rule": "keep_higher_confidence_direction",
                    }
                )
                continue
            del aggregated[opposite_key]
            conflict_filtered_count += 1
            conflict_reasons[f"reverse_{triple['predicate']}"] += 1
            arbitration_events.append(
                {
                    "type": f"reverse_{triple['predicate']}",
                    "winner": {
                        "subject": triple.get("subject"),
                        "predicate": triple.get("predicate"),
                        "object": triple.get("object"),
                        "confidence": confidence,
                    },
                    "discarded": {
                        "subject": existing.get("subject"),
                        "predicate": existing.get("predicate"),
                        "object": existing.get("object"),
                        "confidence": existing.get("confidence"),
                    },
                    "rule": "replace_lower_confidence_direction",
                }
            )

        key = (
            str(triple["subject"]),
            str(triple["subject_type"]),
            str(triple["predicate"]),
            str(triple["object"]),
            str(triple["object_type"]),
        )
        bucket = aggregated.get(key)
        source_doc_ids = _source_doc_ids_from_triple(triple)
        evidence = str(triple.get("evidence", "")).strip()
        confidence = float(triple.get("confidence", 0))
        evidence_quality = triple.get("evidence_quality") if isinstance(triple.get("evidence_quality"), dict) else {}
        evidence_quality_score = _evidence_quality_score(triple)

        if bucket is None:
            support_score = _support_score(confidence, evidence, evidence_quality_score, len(source_doc_ids))
            aggregated[key] = {
                "subject": key[0],
                "subject_type": key[1],
                "predicate": key[2],
                "object": key[3],
                "object_type": key[4],
                "source_doc_id": source_doc_ids[0] if source_doc_ids else "",
                "source_doc_ids": source_doc_ids[:],
                "evidence": evidence,
                "evidence_samples": [evidence] if evidence else [],
                "confidence": round(confidence, 2),
                "evidence_quality": evidence_quality,
                "evidence_quality_score": round(evidence_quality_score, 4),
                "evidence_quality_samples": [evidence_quality] if evidence_quality else [],
                "support_score": support_score,
                "support_doc_count": len(set(source_doc_ids)),
                "aggregation": {
                    "merged_count": 1,
                    "confidence_values": [round(confidence, 4)],
                    "evidence_quality_values": [round(evidence_quality_score, 4)] if evidence_quality else [],
                },
            }
            continue

        duplicate_count += 1
        for source_doc_id in source_doc_ids:
            if source_doc_id and source_doc_id not in bucket["source_doc_ids"]:
                bucket["source_doc_ids"].append(source_doc_id)
        if evidence and evidence not in bucket["evidence_samples"]:
            bucket["evidence_samples"].append(evidence)
        if evidence_quality and evidence_quality not in bucket["evidence_quality_samples"]:
            bucket["evidence_quality_samples"].append(evidence_quality)
        if evidence_quality_score > float(bucket.get("evidence_quality_score", 0)):
            bucket["evidence_quality"] = evidence_quality
            bucket["evidence_quality_score"] = round(evidence_quality_score, 4)
        if confidence > bucket["confidence"]:
            bucket["confidence"] = round(confidence, 2)
        bucket["aggregation"]["merged_count"] += 1
        bucket["aggregation"]["confidence_values"].append(round(confidence, 4))
        if evidence_quality:
            bucket["aggregation"]["evidence_quality_values"].append(round(evidence_quality_score, 4))
        bucket["support_doc_count"] = len(set(bucket["source_doc_ids"]))
        bucket["support_score"] = _support_score(
            float(bucket.get("confidence", 0)),
            str(bucket.get("evidence", "")),
            float(bucket.get("evidence_quality_score", 0)),
            int(bucket.get("support_doc_count", 0)),
        )

    rows = []
    for index, item in enumerate(
        sorted(aggregated.values(), key=lambda row: (row["subject_type"], row["subject"], row["predicate"], row["object"])),
        start=1,
    ):
        rows.append({**item, "triple_id": f"T{index:04d}", "source_doc_ids": sorted(item["source_doc_ids"])})

    relation_counts = Counter(item["predicate"] for item in rows)
    entity_pair_counts = Counter((item["subject_type"], item["object_type"]) for item in rows)
    source_doc_coverage = sorted({doc_id for item in rows for doc_id in item["source_doc_ids"] if doc_id})
    confidence_values = [item["confidence"] for item in rows]
    evidence_quality_values = [float(item.get("evidence_quality_score", 0)) for item in rows if item.get("evidence_quality")]
    support_scores = [float(item.get("support_score", 0)) for item in rows]
    multi_source_facts = [item for item in rows if int(item.get("support_doc_count", 0)) >= 2]
    evidence_sample_counts = [len(item.get("evidence_samples", [])) for item in rows]
    relation_confidence: dict[str, list[float]] = {}
    relation_support: dict[str, list[float]] = {}
    for item in rows:
        relation = str(item.get("predicate", "unknown"))
        relation_confidence.setdefault(relation, []).append(float(item.get("confidence", 0)))
        relation_support.setdefault(relation, []).append(float(item.get("support_score", 0)))
    report = {
        "candidate_triples": len(triples),
        "accepted_triples": len(rows),
        "rejected_triples": rejected_count,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "duplicate_triples": duplicate_count,
        "conflict_filtered_triples": conflict_filtered_count,
        "conflict_reasons": dict(sorted(conflict_reasons.items())),
        "arbitration": {
            "enabled": True,
            "duplicate_merge_count": duplicate_count,
            "multi_source_fact_count": len(multi_source_facts),
            "conflict_filtered_count": conflict_filtered_count,
            "events": arbitration_events[:50],
            "rules": [
                "validate_entity_relation_schema",
                "merge_duplicate_facts",
                "keep_higher_confidence_reverse_causal_or_complication_edge",
                "raise_support_score_with_evidence_quality_and_multi_source_support",
            ],
        },
        "relation_type_counts": dict(sorted(relation_counts.items())),
        "entity_pair_type_counts": {
            f"{subject_type}->{object_type}": count
            for (subject_type, object_type), count in sorted(entity_pair_counts.items())
        },
        "source_doc_count": len(source_doc_coverage),
        "evidence_fusion": {
            "facts_with_evidence": sum(1 for item in rows if str(item.get("evidence", "")).strip()),
            "avg_evidence_samples_per_fact": round(sum(evidence_sample_counts) / len(evidence_sample_counts), 4)
            if evidence_sample_counts else 0,
            "multi_source_fact_count": len(multi_source_facts),
            "support_score": {
                "min": round(min(support_scores), 4) if support_scores else 0,
                "max": round(max(support_scores), 4) if support_scores else 0,
                "avg": round(sum(support_scores) / len(support_scores), 4) if support_scores else 0,
            },
            "by_relation": {
                relation: {
                    "count": len(values),
                    "avg_confidence": round(sum(values) / len(values), 4) if values else 0,
                    "avg_support_score": round(sum(relation_support.get(relation, [])) / len(relation_support.get(relation, [])), 4)
                    if relation_support.get(relation) else 0,
                }
                for relation, values in sorted(relation_confidence.items())
            },
        },
        "confidence": {
            "min": round(min(confidence_values), 2) if confidence_values else 0,
            "max": round(max(confidence_values), 2) if confidence_values else 0,
            "avg": round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else 0,
        },
        "evidence_quality": {
            "scored_triples": len(evidence_quality_values),
            "avg_score": round(sum(evidence_quality_values) / len(evidence_quality_values), 4)
            if evidence_quality_values else 0,
        },
    }
    return rows, report
