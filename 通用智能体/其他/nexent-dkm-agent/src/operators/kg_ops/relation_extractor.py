"""Relation extraction and triple generation for task 2.

Confidence scores are computed based on textual proximity between the
subject and object entities within each record.  Entities that appear
closer together in the text receive higher confidence, reflecting the
assumption that nearby mentions are more likely to be truly related.
"""

from __future__ import annotations

import re
from typing import Any

RELATION_SCHEMA = {
    "has_symptom": ("Disease", "Symptom"),
    "treated_by": ("Disease", "Drug"),
    "diagnosed_by": ("Disease", "Examination"),
    "recommended_treatment": ("Disease", "Treatment"),
    "complication_of": ("Disease", "Disease"),
}

# Base confidence for each relation type, before distance adjustment
_RELATION_BASE_CONFIDENCE = {
    "has_symptom": 0.80,
    "treated_by": 0.75,
    "diagnosed_by": 0.73,
    "recommended_treatment": 0.70,
    "complication_of": 0.60,
}

# Maximum text distance (chars) for full proximity bonus
_PROXIMITY_WINDOW = 50


def _compute_confidence(
    predicate: str,
    subject: str,
    object_: str,
    record_text: str,
) -> float:
    """Compute a data-driven confidence score based on text proximity.

    The score starts from a relation-type-specific base and is boosted
    (up to +0.15) when the two entities appear close together in the
    source text, and penalised (down to -0.10) when they are far apart
    or one is absent from the text.
    """
    base = _RELATION_BASE_CONFIDENCE.get(predicate, 0.65)

    subj_pos = record_text.find(subject)
    obj_pos = record_text.find(object_)

    if subj_pos < 0 or obj_pos < 0:
        # Entity not found in text — reduce confidence
        return round(max(base - 0.10, 0.30), 2)

    distance = abs(subj_pos - obj_pos)
    if distance <= _PROXIMITY_WINDOW:
        # Close proximity: boost
        bonus = 0.15 * (1.0 - distance / _PROXIMITY_WINDOW)
        return round(min(base + bonus, 0.99), 2)

    # Distant: mild penalty scaled by distance
    penalty = min(0.10, 0.10 * (distance - _PROXIMITY_WINDOW) / 200)
    return round(max(base - penalty, 0.30), 2)


def extract_relations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate medical KG triples from per-record entities."""

    triples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for record in records:
        record_id = record["record_id"]
        record_text = record.get("text", "")
        entities = record.get("entities", {})
        diseases = entities.get("Disease", [])

        # Detect complication structure so drugs can be attributed to the
        # primary disease rather than its complication (avoids the classic
        # over-pairing that credits 心力衰竭 with 冠心病's medications).
        complication_subjects = {
            subject
            for subject, _object, _conf in _oriented_complication_pairs(
                record_text, diseases
            )
        }

        # Drugs that appear next to an ancillary purpose marker are used for
        # symptomatic relief (退热/退烧/止痛/镇痛), not for treating the disease.
        symptomatic_drugs = _detect_symptomatic_drugs(
            record_text, entities.get("Drug", [])
        )
        explicit_treated_by_pairs = _explicit_treated_by_pairs(
            record_text,
            diseases,
            entities.get("Drug", []),
        )

        for disease in diseases:
            for symptom in entities.get("Symptom", []):
                confidence = _compute_confidence("has_symptom", disease, symptom, record_text)
                _append_triple(
                    triples, seen, disease, "has_symptom", symptom,
                    record_id, confidence, "Disease", "Symptom", record_text,
                )
            for drug in entities.get("Drug", []):
                if not _allow_treated_by_pair(
                    disease,
                    drug,
                    complication_subjects=complication_subjects,
                    symptomatic_drugs=symptomatic_drugs,
                    explicit_pairs=explicit_treated_by_pairs,
                ):
                    continue
                confidence = _compute_confidence("treated_by", disease, drug, record_text)
                _append_triple(
                    triples, seen, disease, "treated_by", drug,
                    record_id, confidence, "Disease", "Drug", record_text,
                )
            for examination in entities.get("Examination", []):
                confidence = _compute_confidence("diagnosed_by", disease, examination, record_text)
                _append_triple(
                    triples, seen, disease, "diagnosed_by", examination,
                    record_id, confidence, "Disease", "Examination", record_text,
                )
            for treatment in entities.get("Treatment", []):
                confidence = _compute_confidence("recommended_treatment", disease, treatment, record_text)
                _append_triple(
                    triples, seen, disease, "recommended_treatment", treatment,
                    record_id, confidence, "Disease", "Treatment", record_text,
                )

        for left, right, confidence in _disease_pairs(record_text, diseases):
            _append_triple(
                triples, seen, left, "complication_of", right,
                record_id, confidence, "Disease", "Disease", record_text,
            )

    return triples


def extract_relations_tensorized(
    records: list[dict[str, Any]],
    backend: str = "cpu",
    prefer_device: str = "auto",
) -> dict[str, Any]:
    """Generate KG triples using tensorized relation scoring (CPU/NPU).

    Real ``(Disease, Object)`` candidate pairs are encoded into a feature
    matrix and scored with a deterministic relation-projection matmul on the
    requested ``backend`` (``"cpu"`` or ``"npu"``). The matmul's ``argmax``
    recovers each candidate's predicate, which is then materialised into a
    triple with a proximity-based confidence (identical to the rule path).
    ``complication_of`` edges still come from the rule-based oriented cues.

    Returns a dict with the produced ``triples`` plus scoring metadata
    (``scoring_backend``, ``scoring_device``, ``candidate_count``,
    ``status``). On any failure the function falls back to the pure rule path
    and reports ``status == "fallback_rule"``.
    """

    from src.operators.kg_ops.relation_features import (
        build_relation_candidates,
        build_scoring_inputs,
    )

    normalized = (backend or "cpu").lower()
    if normalized not in {"cpu", "npu"}:
        raise ValueError(f"unsupported tensorized backend: {backend!r}")

    fallback = {
        "triples": extract_relations(records),
        "scoring_backend": "rule",
        "scoring_device": "cpu",
        "candidate_count": 0,
        "status": "fallback_rule",
    }

    candidates = build_relation_candidates(records)
    if not candidates:
        # No object-typed candidates; only complication edges (rule) may exist.
        return {
            "triples": extract_relations(records),
            "scoring_backend": normalized,
            "scoring_device": "cpu",
            "candidate_count": 0,
            "status": "no_candidates",
        }

    try:
        from src.operators.npu_ops.kg_tensor_ops import (
            prepare_relation_tensor_cache,
            score_cached_argmax_labels,
            score_relation_candidates_cpu,
            score_relation_candidates_npu,
        )

        scoring_inputs = build_scoring_inputs(candidates)
        if normalized == "npu":
            # Default NPU path: prepare a reusable tensor cache and score with
            # cached_argmax_labels, which avoids copying full logits to CPU.
            # Falls back to the legacy path when the cache cannot be prepared.
            cache = prepare_relation_tensor_cache(scoring_inputs, prefer_device=prefer_device)
            if cache.get("status") == "completed":
                cached_result = score_cached_argmax_labels(cache)
                if cached_result.get("status") == "completed":
                    effective_backend = "npu"
                    result = cached_result
                else:
                    result = score_relation_candidates_npu(scoring_inputs, prefer_device=prefer_device)
                    if result.get("status") != "completed":
                        result = score_relation_candidates_cpu(scoring_inputs)
                        effective_backend = "cpu"
                    else:
                        effective_backend = "npu"
            else:
                result = score_relation_candidates_npu(scoring_inputs, prefer_device=prefer_device)
                if result.get("status") != "completed":
                    result = score_relation_candidates_cpu(scoring_inputs)
                    effective_backend = "cpu"
                else:
                    effective_backend = "npu"
        else:
            result = score_relation_candidates_cpu(scoring_inputs)
            effective_backend = "cpu"

        if result.get("status") != "completed":
            return fallback

        predicted = result.get("predicted_relations", [])
        if len(predicted) != len(candidates):
            return fallback
    except Exception:
        return fallback

    triples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    # Pre-compute per-record gating sets so the tensor path mirrors the rule
    # path's medical-correctness filters without modifying any NPU op.
    record_gating: dict[str, dict[str, Any]] = {}
    for record in records:
        rid = record.get("record_id", "")
        rtext = record.get("text", "")
        rdiseases = record.get("entities", {}).get("Disease", [])
        rdrugs = record.get("entities", {}).get("Drug", [])
        record_gating[rid] = {
            "complication_subjects": {
                subject
                for subject, _o, _c in _oriented_complication_pairs(rtext, rdiseases)
            },
            "symptomatic_drugs": _detect_symptomatic_drugs(rtext, rdrugs),
            "explicit_treated_by_pairs": _explicit_treated_by_pairs(
                rtext,
                rdiseases,
                rdrugs,
            ),
        }

    for candidate, predicate in zip(candidates, predicted):
        # The projection guarantees predicate == candidate["predicate_hint"];
        # honour the scored prediction so the NPU output drives the triple.
        subject = candidate["subject"]
        object_ = candidate["object"]
        record_text = candidate["record_text"]
        gating = record_gating.get(candidate.get("record_id", ""), {})
        # Apply the same medical-correctness gating used by the rule path.
        if predicate == "treated_by":
            if not _allow_treated_by_pair(
                subject,
                object_,
                complication_subjects=gating.get("complication_subjects", set()),
                symptomatic_drugs=gating.get("symptomatic_drugs", set()),
                explicit_pairs=gating.get("explicit_treated_by_pairs", set()),
            ):
                continue
        confidence = _compute_confidence(predicate, subject, object_, record_text)
        object_type = RELATION_SCHEMA.get(predicate, ("Disease", candidate["object_type"]))[1]
        _append_triple(
            triples, seen, subject, predicate, object_,
            candidate["record_id"], confidence, "Disease", object_type, record_text,
        )

    # complication_of edges rely on oriented linguistic cues, kept rule-based.
    for record in records:
        record_id = record.get("record_id", "")
        record_text = record.get("text", "")
        diseases = record.get("entities", {}).get("Disease", [])
        for left, right, confidence in _disease_pairs(record_text, diseases):
            _append_triple(
                triples, seen, left, "complication_of", right,
                record_id, confidence, "Disease", "Disease", record_text,
            )

    return {
        "triples": triples,
        "scoring_backend": effective_backend,
        "scoring_device": result.get("device", "cpu"),
        "scoring_mode": result.get("scoring_mode", "legacy"),
        "candidate_count": len(candidates),
        "status": "completed",
    }


def _append_triple(
    triples: list[dict[str, Any]],
    seen: set[tuple[str, str, str, str]],
    subject: str,
    predicate: str,
    object_: str,
    record_id: str,
    confidence: float,
    subject_type: str,
    object_type: str,
    record_text: str,
) -> None:
    key = (subject, predicate, object_, record_id)
    if key in seen:
        return
    seen.add(key)
    triples.append(
        {
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "record_id": record_id,
            "confidence": confidence,
            "subject_type": subject_type,
            "object_type": object_type,
            "evidence": _evidence_snippet(record_text, subject, object_),
        }
    )


def _disease_pairs(text: str, diseases: list[str]) -> list[tuple[str, str, float]]:
    """Return complication pairs only when an explicit linguistic cue is present.

    Earlier revisions also emitted a triple for *every* unordered pair of
    co-occurring diseases, which produced large numbers of spurious and
    unexplainable ``complication_of`` edges (e.g. two unrelated chronic
    conditions mentioned in the same note).  Complications are now derived
    solely from oriented cues such as ``并发`` / ``合并`` / ``继发于``, keeping
    every edge backed by a concrete textual signal.
    """
    return _oriented_complication_pairs(text, diseases)


def _detect_symptomatic_drugs(text: str, drugs: list[str]) -> set[str]:
    """Identify drugs that are followed by an ancillary purpose marker.

    When a drug name is immediately (within a short window) followed by
    markers such as 退热 / 退烧 / 止痛 / 镇痛, the drug is used for
    symptomatic relief rather than as a disease-specific treatment, so it
    should not be paired with the disease under ``treated_by``.
    """

    symptomatic: set[str] = set()
    purpose_markers = ("退热", "退烧", "止痛", "镇痛", "解热", "消炎止痛")
    for drug in drugs:
        search_pos = 0
        while True:
            pos = text.find(drug, search_pos)
            if pos < 0:
                break
            tail = text[pos + len(drug) : pos + len(drug) + 6]
            if any(marker in tail for marker in purpose_markers):
                symptomatic.add(drug)
                break
            search_pos = pos + len(drug)
    return symptomatic


def _explicit_treated_by_pairs(
    text: str,
    diseases: list[str],
    drugs: list[str],
) -> set[tuple[str, str]]:
    """Return drug-disease pairs backed by an explicit treatment phrase."""

    pairs: set[tuple[str, str]] = set()
    clause_chars = r"[^。；;，,\n]"
    for disease in diseases:
        for drug in drugs:
            patterns = (
                (
                    rf"{re.escape(drug)}{clause_chars}{{0,8}}"
                    rf"(?:用于|用来|治疗|控制|缓解){clause_chars}{{0,8}}"
                    rf"{re.escape(disease)}"
                ),
                (
                    rf"{re.escape(disease)}{clause_chars}{{0,12}}"
                    rf"(?:服用|使用|给予|口服|应用|采用|加用|继续服用)"
                    rf"{clause_chars}{{0,8}}{re.escape(drug)}"
                ),
            )
            if any(re.search(pattern, text) for pattern in patterns):
                pairs.add((disease, drug))
    return pairs


def _allow_treated_by_pair(
    disease: str,
    drug: str,
    *,
    complication_subjects: set[str],
    symptomatic_drugs: set[str],
    explicit_pairs: set[tuple[str, str]],
) -> bool:
    """Apply conservative record gating without hiding explicit attribution."""

    explicit_targets = {
        target_disease
        for target_disease, target_drug in explicit_pairs
        if target_drug == drug
    }
    if explicit_targets:
        return disease in explicit_targets
    if disease in complication_subjects:
        return False
    if drug in symptomatic_drugs:
        return False
    return True


def _oriented_complication_pairs(text: str, diseases: list[str]) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    for primary in diseases:
        for complication in diseases:
            if primary == complication:
                continue
            patterns = [
                rf"{re.escape(primary)}[^。；;，,]{{0,8}}(?:并发|合并|伴发){re.escape(complication)}",
                rf"{re.escape(complication)}[^。；;，,]{{0,8}}(?:继发于|由){re.escape(primary)}",
                rf"{re.escape(complication)}[^。；;，,]{{0,16}}{re.escape(primary)}的并发症",
            ]
            if any(re.search(pattern, text) for pattern in patterns):
                # Oriented complication has higher confidence due to explicit linguistic signal
                pairs.append((complication, primary, 0.78))
    return _dedupe_pairs(pairs)


def _dedupe_pairs(pairs: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    deduped: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for subject, object_, confidence in pairs:
        key = (subject, object_)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((subject, object_, confidence))
    return deduped


def _evidence_snippet(text: str, subject: str, object_: str, window: int = 28) -> str:
    positions = [pos for pos in (text.find(subject), text.find(object_)) if pos >= 0]
    if not positions:
        return text[: window * 2].strip()
    start = max(min(positions) - window, 0)
    end = min(max(pos + len(term) for pos, term in ((text.find(subject), subject), (text.find(object_), object_)) if pos >= 0) + window, len(text))
    return text[start:end].strip()
