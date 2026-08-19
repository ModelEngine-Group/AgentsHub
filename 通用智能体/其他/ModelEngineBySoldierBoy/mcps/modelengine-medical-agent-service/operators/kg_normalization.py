"""Normalization and confidence calibration for medical KG candidates."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


ENTITY_ALIAS_MAP: dict[tuple[str, str], tuple[str, ...]] = {
    ("Disease", "原发性高血压"): ("高血压", "血压升高"),
    ("Disease", "2型糖尿病"): ("糖尿病", "血糖升高"),
    ("Disease", "慢性阻塞性肺疾病"): ("慢阻肺", "COPD"),
    ("Disease", "胃食管反流病"): ("胃食道反流", "胃食道反流病", "胃食管反流"),
    ("Disease", "消化性溃疡"): ("胃溃疡", "十二指肠溃疡"),
    ("Disease", "抑郁障碍"): ("抑郁症",),
    ("Disease", "泌尿系感染"): ("尿路感染",),
    ("Disease", "支气管哮喘"): ("哮喘", "咳嗽变异性哮喘"),
    ("Disease", "甲状腺功能亢进"): ("甲亢",),
    ("Disease", "甲状腺功能减退"): ("甲减",),
    ("Drug", "质子泵抑制剂"): ("PPI",),
    ("Drug", "β受体阻滞剂"): ("β受体阻断剂", "贝塔受体阻滞剂"),
    ("Test", "头颅MRI"): ("颅脑MRI", "头部MRI"),
    ("Test", "头颅CT"): ("颅脑CT", "头部CT"),
    ("Treatment", "生活方式干预"): ("生活方式调整", "生活方式管理"),
}

CANONICAL_BY_TYPE_AND_ALIAS: dict[tuple[str, str], str] = {}
for (entity_type, canonical), aliases in ENTITY_ALIAS_MAP.items():
    CANONICAL_BY_TYPE_AND_ALIAS[(entity_type, canonical)] = canonical
    for alias in aliases:
        CANONICAL_BY_TYPE_AND_ALIAS[(entity_type, alias)] = canonical


RELATION_BASE_CONFIDENCE = {
    "has_symptom": 0.9,
    "treated_by": 0.88,
    "diagnosed_by": 0.88,
    "has_complication": 0.84,
    "belongs_to_department": 0.92,
    "contraindicated_for": 0.84,
    "has_risk_factor": 0.82,
    "prevented_by": 0.8,
    "causes": 0.86,
    "located_in": 0.84,
    "has_stage": 0.8,
    "has_patient_group": 0.78,
    "has_history": 0.78,
    "spreads_via": 0.78,
    "has_epidemiology": 0.76,
}


def _clean_name(name: str) -> str:
    text = str(name or "").strip()
    text = re.sub(r"\s+", "", text)
    text = text.strip("，。；;、:：,.()（）[]【】")
    return text


def canonicalize_entity(name: str, entity_type: str) -> tuple[str, list[str]]:
    cleaned = _clean_name(name)
    canonical = CANONICAL_BY_TYPE_AND_ALIAS.get((entity_type, cleaned), cleaned)
    aliases = [cleaned]
    if canonical != cleaned:
        aliases.append(canonical)
    configured_aliases = ENTITY_ALIAS_MAP.get((entity_type, canonical), ())
    aliases.extend(alias for alias in configured_aliases if alias in {cleaned, canonical})
    return canonical, sorted({alias for alias in aliases if alias})


def _is_overlong_causal_phrase(text: str) -> bool:
    return len(text) > 24 or text.count("、") >= 3 or text.count(",") >= 2 or text.count("，") >= 2


def _trim_causal_phrase(text: str) -> str:
    cleaned = _clean_name(text)
    if not cleaned:
        return cleaned
    connectors = ("导致", "造成", "引起", "诱发", "从而", "进而", "使")
    for connector in connectors:
        if connector in cleaned:
            left, right = cleaned.rsplit(connector, 1)
            if right:
                return right[:24]
            if left:
                return left[:24]
    pieces = [piece for piece in re.split(r"[，,；;、]", cleaned) if piece]
    if pieces:
        return min(pieces, key=len) if min(len(piece) for piece in pieces) <= 24 else pieces[-1][:24]
    return cleaned[:24] if len(cleaned) > 24 else cleaned


def normalize_triple_candidate(triple: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return a normalized triple and a small audit record."""
    subject_type = str(triple.get("subject_type", "")).strip()
    object_type = str(triple.get("object_type", "")).strip()
    predicate = str(triple.get("predicate", "")).strip()
    subject_raw = _clean_name(str(triple.get("subject", "")))
    object_raw = _clean_name(str(triple.get("object", "")))
    audit: dict[str, Any] = {
        "predicate": predicate,
        "changed": False,
        "dropped": False,
        "reasons": [],
    }
    if not subject_raw or not object_raw:
        audit["dropped"] = True
        audit["reasons"].append("empty_subject_or_object")
        return None, audit

    if predicate == "causes":
        if _is_overlong_causal_phrase(subject_raw):
            trimmed = _trim_causal_phrase(subject_raw)
            if trimmed and trimmed != subject_raw:
                audit["changed"] = True
                audit["reasons"].append("trimmed_long_causal_subject")
                subject_raw = trimmed
        if _is_overlong_causal_phrase(object_raw):
            trimmed = _trim_causal_phrase(object_raw)
            if trimmed and trimmed != object_raw:
                audit["changed"] = True
                audit["reasons"].append("trimmed_long_causal_object")
                object_raw = trimmed
        if subject_raw == object_raw:
            audit["dropped"] = True
            audit["reasons"].append("self_loop_after_normalization")
            return None, audit

    subject, subject_aliases = canonicalize_entity(subject_raw, subject_type)
    object_name, object_aliases = canonicalize_entity(object_raw, object_type)
    if subject != subject_raw or object_name != object_raw:
        audit["changed"] = True
        audit["reasons"].append("canonicalized_entity_alias")

    normalized = dict(triple)
    normalized.update(
        {
            "subject": subject,
            "object": object_name,
            "subject_aliases": sorted(set(triple.get("subject_aliases", []) or []) | set(subject_aliases)),
            "object_aliases": sorted(set(triple.get("object_aliases", []) or []) | set(object_aliases)),
        }
    )
    evidence = str(normalized.get("evidence", ""))
    try:
        input_confidence = float(normalized.get("confidence", 0))
    except (TypeError, ValueError):
        input_confidence = 0.0
    base = RELATION_BASE_CONFIDENCE.get(predicate, 0.78)
    confidence = max(input_confidence, base)
    if subject_raw in evidence and object_raw in evidence:
        confidence += 0.03
    if normalized.get("source_dataset") in {"CBLUE-CMeIE", "CBLUE-CMedCausal", "CBLUE-CMeEE"}:
        confidence += 0.02
    if audit["changed"]:
        confidence -= 0.03
    evidence_quality = normalized.get("evidence_quality")
    if isinstance(evidence_quality, dict):
        try:
            quality_score = float(evidence_quality.get("score", 0))
        except (TypeError, ValueError):
            quality_score = 0.0
        if quality_score > 0:
            confidence += min(0.03, quality_score * 0.03)
    normalized["confidence"] = round(min(max(confidence, 0.0), 0.97), 2)
    normalized["normalization"] = {
        "changed": audit["changed"],
        "reasons": audit["reasons"],
    }
    return normalized, audit


def normalize_triple_candidates(triples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    changed_count = 0
    dropped_count = 0
    reason_counts: Counter[str] = Counter()
    predicate_counts: Counter[str] = Counter()
    evidence_quality_scores: list[float] = []
    negation_checked_count = 0
    for triple in triples:
        normalized, audit = normalize_triple_candidate(triple)
        if audit["changed"]:
            changed_count += 1
        if audit["dropped"]:
            dropped_count += 1
        for reason in audit["reasons"]:
            reason_counts[reason] += 1
        if normalized is not None:
            predicate_counts[str(normalized.get("predicate", "unknown"))] += 1
            evidence_quality = normalized.get("evidence_quality")
            if isinstance(evidence_quality, dict):
                try:
                    evidence_quality_scores.append(float(evidence_quality.get("score", 0)))
                except (TypeError, ValueError):
                    pass
                if evidence_quality.get("negation_checked"):
                    negation_checked_count += 1
            normalized_rows.append(normalized)
    report = {
        "input_triples": len(triples),
        "output_triples": len(normalized_rows),
        "changed_triples": changed_count,
        "dropped_triples": dropped_count,
        "reason_counts": dict(sorted(reason_counts.items())),
        "predicate_counts": dict(sorted(predicate_counts.items())),
        "alias_rule_count": len(CANONICAL_BY_TYPE_AND_ALIAS),
        "evidence_quality": {
            "scored_triples": len(evidence_quality_scores),
            "negation_checked_triples": negation_checked_count,
            "avg_score": round(sum(evidence_quality_scores) / len(evidence_quality_scores), 4)
            if evidence_quality_scores else 0,
        },
    }
    return normalized_rows, report
