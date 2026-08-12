# -*- coding: utf-8 -*-
"""任务二的离线优先级联抽取。

流程固定为：

1. 离线词典和规则扫描全文；
2. 只把证据充分的低可靠关系、少量中可靠关系和离线未覆盖的医学句子交给 LLM；
3. LLM 复核通过的新增事实才进入最终结果，复核失败或不确定不新增；
4. LLM 调用失败时完整保留离线结果，并由上层返回显式错误。

这使 hybrid 成为一个安全的级联入口，而不是两个抽取器的无条件并集。
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import Any

from .llm_client import LLMClient
from .medical_extraction_validation import (
    relation_evidence_supports_pair,
    relations_to_triples,
)
from .medical_offline_extraction import (
    extract_entities_offline,
    extract_relations_offline,
)
from .schemas import Entity, Relation, Triple
from .task2_cascade_schemas import CascadeOutput, CascadeSegment, ReviewCandidate
from .task2_verifier import (
    extract_gap_facts,
    normalize_verified_entity,
    review_candidates_parallel,
)


_SENTENCE_RE = re.compile(r"[^。！？!?；;@\n]+(?:[。！？!?；;@\n]|$)")
_GAP_CUES = (
    "诊断",
    "症状",
    "表现",
    "病因",
    "并发",
    "治疗",
    "用药",
    "服用",
    "口服",
    "给予",
    "首选",
    "出现",
    "伴有",
    "可见",
    "检查",
    "监测",
    "扫描",
    "分型",
    "分为",
    "感染",
    "预防",
    "导致",
    "高危",
    "风险",
    "包括",
    "患者",
    "疾病",
    "药物",
    "手术",
    "检验",
)
_SECTION_CONTEXT_GAP_CUES = (
    "有",
    "出现",
    "可见",
    "表现",
    "症状",
    "口服",
    "用药",
    "治疗",
    "给予",
    "首选",
    "检查",
    "检测",
    "扫描",
    "诊断",
    "并发",
    "合并",
    "分为",
    "分型",
    "类型",
    "包括",
    "相关",
    "危险因素",
    "高危",
    "转移",
    "侵犯",
)


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in _SENTENCE_RE.finditer(text or ""):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        start = match.start() + leading
        end = match.start() + trailing
        if start < end:
            spans.append((start, end, text[start:end]))
    return spans


def _entity_in_segment(entity: Entity, start: int, end: int) -> bool:
    if entity.start_idx is None:
        return False
    entity_end = (entity.end_idx if entity.end_idx is not None else entity.start_idx) + 1
    return entity.start_idx < end and entity_end > start


def _preceding_section_disease(
    text: str,
    segment_start: int,
    entities: list[Entity],
) -> Entity | None:
    """Find the disease heading that owns the current ``@`` section.

    CMeIE contains source fragments such as ``疾病@症状描述``.  The marker is
    a formatting boundary, not a semantic boundary: the disease heading is
    still the subject of facts in later sentences of that section.  The old
    eight-character marker window worked only for the first sentence after
    ``@`` and left later relation segments without a subject.  Keep the lookup
    bounded to the recent section and never cross a newer ``@`` marker.
    """

    if segment_start <= 0:
        return None
    marker = text.rfind("@", max(0, segment_start - 240), segment_start)
    if marker < 0:
        return None
    candidates = [
        entity
        for entity in entities
        if entity.type == "dis"
        and entity.start_idx is not None
        and entity.end_idx is not None
        and entity.end_idx < marker
        and marker - entity.end_idx <= 240
    ]
    return max(candidates, key=lambda item: item.end_idx or -1, default=None)


def _segment_relations(segment_text: str, relations: list[Relation]) -> list[Relation]:
    return [
        relation
        for relation in relations
        if relation.subject in segment_text and relation.object in segment_text
    ]


_CASCADE_CAUSAL_CUES = (
    "\u6f5c\u5728\u7684\u75c5\u56e0",
    "\u75c5\u56e0",
    "\u539f\u56e0",
    "\u7531\u4e8e",
    "\u5f15\u8d77",
    "\u5bfc\u81f4",
    "\u8bf1\u53d1",
)
_CASCADE_BODY_SITE_CUES = (
    "\u53d1\u75c5\u90e8\u4f4d",
    "\u597d\u53d1\u4e8e",
    "\u53d1\u751f\u4e8e",
    "\u4f4d\u4e8e",
    "\u7d2f\u53ca",
    "\u4fb5\u72af",
    "\u8f6c\u79fb\u81f3",
)
_CASCADE_LIST_CUES = ("\u5982", "\u5305\u62ec", "\u4f8b\u5982")
_CASCADE_CAUSAL_RELATIONS = {"\u75c5\u56e0", "\u76f8\u5173\uff08\u5bfc\u81f4\uff09"}
_CASCADE_BODY_SITE_RELATIONS = {"\u53d1\u75c5\u90e8\u4f4d", "\u5916\u4fb5\u90e8\u4f4d", "\u8f6c\u79fb\u90e8\u4f4d"}

_TARGETED_LOW_REVIEW_CUES = {
    "\u4e34\u5e8a\u8868\u73b0": ("\u8868\u73b0", "\u4f34\u6709", "\u51fa\u73b0", "\u4e3b\u8bc9"),
    "\u836f\u7269\u6cbb\u7597": ("\u6cbb\u7597", "\u7528\u836f", "\u670d\u7528", "\u53e3\u670d", "\u6ce8\u5c04"),
    "\u8f85\u52a9\u6cbb\u7597": ("\u6cbb\u7597", "\u624b\u672f", "\u5904\u7406", "\u5e72\u9884"),
    "\u8f85\u52a9\u68c0\u67e5": ("\u68c0\u67e5", "\u68c0\u6d4b", "\u76d1\u6d4b", "\u590d\u67e5"),
    "\u5e76\u53d1\u75c7": ("\u5e76\u53d1", "\u5408\u5e76"),
    "\u75c5\u56e0": _CASCADE_CAUSAL_CUES,
    "\u53d1\u75c5\u90e8\u4f4d": _CASCADE_BODY_SITE_CUES,
    "\u5916\u4fb5\u90e8\u4f4d": ("\u4fb5\u72af",),
    "\u8f6c\u79fb\u90e8\u4f4d": ("\u8f6c\u79fb\u81f3",),
}


def _segment_requires_gap_review(
    sentence: str,
    sentence_entities: list[Entity],
    sentence_relations: list[Relation],
) -> bool:
    """Detect partial coverage, rather than treating one relation as complete."""

    has_causal_cue = any(cue in sentence for cue in _CASCADE_CAUSAL_CUES)
    has_body_site_cue = any(cue in sentence for cue in _CASCADE_BODY_SITE_CUES)
    has_location_list = (
        "\u5728" in sentence
        and any(entity.type == "bod" for entity in sentence_entities)
        and any(cue in sentence for cue in _CASCADE_LIST_CUES)
    )
    has_gap_cue = (
        any(cue in sentence for cue in _GAP_CUES)
        or has_causal_cue
        or has_body_site_cue
        or has_location_list
    )
    if not has_gap_cue:
        return False
    if not sentence_entities or not sentence_relations:
        return True
    if any(cue in sentence for cue in _CASCADE_LIST_CUES) and (
        has_causal_cue or has_body_site_cue or has_location_list
    ):
        return True

    relation_endpoints = {
        endpoint
        for relation in sentence_relations
        for endpoint in (relation.subject, relation.object)
    }
    if any(
        entity.type != "dis" and entity.text not in relation_endpoints
        for entity in sentence_entities
    ):
        return True
    if has_causal_cue and not any(
        relation.predicate in _CASCADE_CAUSAL_RELATIONS
        for relation in sentence_relations
    ):
        return True
    if (has_body_site_cue or has_location_list) and not any(
        relation.predicate in _CASCADE_BODY_SITE_RELATIONS
        for relation in sentence_relations
    ):
        return True
    return False


def _max_gap_segments() -> int:
    raw = os.getenv("CCF_TASK2_CASCADE_MAX_GAP_SEGMENTS", "12")
    try:
        return max(1, min(32, int(raw)))
    except ValueError:
        return 12


def _max_review_candidates() -> int:
    """Keep one-record hybrid requests bounded when the model is slow."""

    raw = os.getenv("CCF_TASK2_CASCADE_MAX_REVIEW_CANDIDATES", "256")
    try:
        return max(1, min(512, int(raw)))
    except ValueError:
        return 256


def _find_gap_segments(
    text: str,
    entities: list[Entity],
    relations: list[Relation],
) -> list[CascadeSegment]:
    gaps: list[CascadeSegment] = []
    for index, (start, end, sentence) in enumerate(_sentence_spans(text)):
        context_disease = _preceding_section_disease(text, start, entities)
        segment_start = (
            context_disease.start_idx
            if context_disease is not None and context_disease.start_idx is not None
            else start
        )
        segment_text = text[segment_start:end]
        sentence_entities = [
            entity for entity in entities if _entity_in_segment(entity, start, end)
        ]
        if context_disease is not None and context_disease not in sentence_entities:
            sentence_entities.insert(0, context_disease)
        sentence_relations = _segment_relations(segment_text, relations)
        reasons: list[str] = []
        has_section_context_cue = context_disease is not None and any(
            cue in sentence for cue in _SECTION_CONTEXT_GAP_CUES
        )
        if has_section_context_cue:
            reasons.append("section_context_gap")
        if _segment_requires_gap_review(sentence, sentence_entities, sentence_relations):
            reason = (
                "partial_relation_coverage"
                if sentence_relations
                else "entity_without_relation"
            )
            if reason not in reasons:
                reasons.append(reason)
        if reasons:
            gaps.append(
                CascadeSegment(
                    segment_id=f"s{index}",
                    start_idx=segment_start,
                    end_idx=end,
                    text=segment_text,
                    reasons=tuple(reasons),
                )
            )
    return gaps[: _max_gap_segments()]


def _scoped_candidate_id(candidate_scope: str, candidate_id: str) -> str:
    return f"{candidate_scope}:{candidate_id}" if candidate_scope else candidate_id


def _offline_relation_candidates(
    relations: list[Relation], *, candidate_scope: str = ""
) -> list[ReviewCandidate]:
    candidates: list[ReviewCandidate] = []
    for index, relation in enumerate(relations):
        if relation.reliability_level == "high":
            continue
        candidates.append(
            ReviewCandidate(
                candidate_id=_scoped_candidate_id(candidate_scope, f"offline_relation:{index}"),
                kind="relation",
                source="offline",
                evidence=relation.evidence,
                reliability_level=relation.reliability_level,
                confidence=relation.confidence,
                subject=relation.subject,
                subject_type=relation.subject_type,
                predicate=relation.predicate,
                object=relation.object,
                object_type=relation.object_type,
                extraction_method=relation.extraction_method,
            )
        )
    return candidates


def _offline_entity_candidates(
    entities: list[Entity], *, candidate_scope: str = ""
) -> list[ReviewCandidate]:
    candidates: list[ReviewCandidate] = []
    for index, entity in enumerate(entities):
        if entity.reliability_level == "high":
            continue
        candidates.append(
            ReviewCandidate(
                candidate_id=_scoped_candidate_id(candidate_scope, f"offline_entity:{index}"),
                kind="entity",
                source="offline",
                evidence=entity.evidence,
                reliability_level=entity.reliability_level,
                confidence=entity.confidence,
                entity_text=entity.text,
                entity_type=entity.type,
            )
        )
    return candidates


def _targeted_low_relation_candidate(candidate: ReviewCandidate) -> bool:
    """Route only calibrated, evidence-bearing low relations to LLM review.

    ``low`` also contains small-support known pairs whose measured precision is
    useful.  It must not reopen context/sentence groups measured near random.
    The confidence field is the group's validation precision for offline facts.
    """

    if candidate.kind != "relation" or candidate.reliability_level != "low":
        return False
    if candidate.extraction_method in {
        "clinical_context_rule",
        "explicit_medication_frame",
    }:
        return relation_evidence_supports_pair(
            candidate.evidence,
            candidate.predicate,
            candidate.subject,
            candidate.object,
            subject_type=candidate.subject_type,
            object_type=candidate.object_type,
            require_disease_subject=True,
        )
    if candidate.confidence < 0.55:
        return False
    cues = _TARGETED_LOW_REVIEW_CUES.get(candidate.predicate, ())
    return bool(cues) and any(cue in candidate.evidence for cue in cues)


def _gap_candidates(
    entities: list[Entity],
    relations: list[Relation],
    segments: list[CascadeSegment],
    *,
    candidate_scope: str = "",
) -> list[ReviewCandidate]:
    candidates: list[ReviewCandidate] = []
    for index, entity in enumerate(entities):
        segment_id = ""
        for segment in segments:
            if _entity_in_segment(entity, segment.start_idx, segment.end_idx):
                segment_id = segment.segment_id
                break
        candidates.append(
            ReviewCandidate(
                candidate_id=_scoped_candidate_id(
                    candidate_scope, f"gap_entity:{segment_id}:{index}"
                ),
                kind="entity",
                source="llm_gap",
                evidence=entity.evidence,
                reliability_level=entity.reliability_level,
                confidence=entity.confidence,
                entity_text=entity.text,
                entity_type=entity.type,
                segment_id=segment_id,
            )
        )
    for index, relation in enumerate(relations):
        segment_id = ""
        for segment in segments:
            if relation.subject in segment.text and relation.object in segment.text:
                segment_id = segment.segment_id
                break
        candidates.append(
            ReviewCandidate(
                candidate_id=_scoped_candidate_id(
                    candidate_scope, f"gap_relation:{segment_id}:{index}"
                ),
                kind="relation",
                source="llm_gap",
                evidence=relation.evidence,
                reliability_level=relation.reliability_level,
                confidence=relation.confidence,
                subject=relation.subject,
                subject_type=relation.subject_type,
                predicate=relation.predicate,
                object=relation.object,
                object_type=relation.object_type,
                extraction_method=relation.extraction_method,
                segment_id=segment_id,
            )
        )
    return candidates


def _candidate_semantic_key(candidate: ReviewCandidate) -> tuple:
    scope = candidate.candidate_id.split(":", 1)[0] if candidate.candidate_id.startswith("r") else ""
    if candidate.kind == "entity":
        return (
            scope,
            "entity",
            candidate.entity_text,
            candidate.entity_type,
        )
    return (
        scope,
        "relation",
        candidate.subject,
        candidate.predicate,
        candidate.object,
    )


def dedupe_review_candidates(
    candidates: list[ReviewCandidate],
) -> list[ReviewCandidate]:
    """Deduplicate semantic duplicates before spending an LLM review call."""

    result: list[ReviewCandidate] = []
    seen: set[tuple] = set()
    for candidate in candidates:
        key = _candidate_semantic_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result

def prepare_cascade_targets(
    text: str,
    entities: list[Entity],
    relations: list[Relation],
    *,
    candidate_scope: str = "",
) -> tuple[list[CascadeSegment], list[ReviewCandidate]]:
    """Prepare the offline review queue and uncovered segments for batch mode.

    Medium/high facts are the quality-controlled offline baseline and bypass
    model review.  Only low-reliability entities and evidence-bearing low
    relations enter the queue.  This makes every review call capable of
    changing the final result instead of spending tokens on facts that the
    merge policy preserves regardless of the decision.
    """

    gap_segments = _find_gap_segments(text, entities, relations)
    entity_candidates = [
        candidate
        for candidate in _offline_entity_candidates(
            entities, candidate_scope=candidate_scope
        )
        if candidate.reliability_level == "low"
    ][:32]
    relation_candidates = [
        candidate
        for candidate in _offline_relation_candidates(
            relations, candidate_scope=candidate_scope
        )
        if _targeted_low_relation_candidate(candidate)
    ][:64]
    return gap_segments, dedupe_review_candidates(
        entity_candidates + relation_candidates
    )


def count_skipped_offline_candidates(
    entities: list[Entity],
    relations: list[Relation],
    review_candidates: list[ReviewCandidate],
    *,
    candidate_scope: str = "",
) -> int:
    """Count offline candidates intentionally kept out of LLM review."""

    all_candidates = (
        _offline_entity_candidates(entities, candidate_scope=candidate_scope)
        + _offline_relation_candidates(relations, candidate_scope=candidate_scope)
    )
    reviewed_ids = {candidate.candidate_id for candidate in review_candidates}
    return sum(
        1
        for candidate in all_candidates
        if candidate.candidate_id not in reviewed_ids
    )


def build_gap_review_candidates(
    entities: list[Entity],
    relations: list[Relation],
    segments: list[CascadeSegment],
    *,
    candidate_scope: str = "",
) -> list[ReviewCandidate]:
    """Build review candidates for facts extracted from uncovered segments."""

    return _gap_candidates(
        entities,
        relations,
        segments,
        candidate_scope=candidate_scope,
    )


_AUTO_ACCEPT_GAP_ENTITY_TYPES = {"bod", "dru", "pro", "mic", "dep"}
_AUTO_ACCEPT_GAP_RELATIONS = {
    "临床表现",
    "发病部位",
}


def _gap_auto_accept_confidence() -> float:
    raw = os.getenv("CCF_TASK2_LLM_GAP_AUTO_ACCEPT_CONFIDENCE", "0.90")
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.90


def select_auto_accepted_gap_candidate_ids(
    candidates: list[ReviewCandidate],
) -> set[str]:
    """Select high-confidence gap facts that already passed deterministic gates.

    The first model call extracts facts, while the parser verifies exact spans,
    endpoint types, relation wording and negation.  Repeating a second model
    call for the strongest surviving facts wastes the bounded review budget.
    Only entity groups with good observed incremental precision and relations
    with explicit textual frames use this fast path.  All other gap candidates
    still require normal LLM review.
    """

    threshold = _gap_auto_accept_confidence()
    accepted: set[str] = set()
    accepted_relation_endpoints: set[str] = set()
    for candidate in candidates:
        if candidate.source != "llm_gap" or candidate.confidence < threshold:
            continue
        if candidate.kind == "entity":
            if candidate.entity_type in _AUTO_ACCEPT_GAP_ENTITY_TYPES:
                accepted.add(candidate.candidate_id)
            continue
        if candidate.predicate not in _AUTO_ACCEPT_GAP_RELATIONS:
            continue
        if not relation_evidence_supports_pair(
            candidate.evidence,
            candidate.predicate,
            candidate.subject,
            candidate.object,
            subject_type=candidate.subject_type,
            object_type=candidate.object_type,
            require_disease_subject=True,
        ):
            continue
        accepted.add(candidate.candidate_id)
        accepted_relation_endpoints.update((candidate.subject, candidate.object))

    # A directly accepted relation must keep both of its validated anchors even
    # when an endpoint type is not in the entity-only fast-path allowlist.
    accepted.update(
        candidate.candidate_id
        for candidate in candidates
        if candidate.source == "llm_gap"
        and candidate.kind == "entity"
        and candidate.confidence >= threshold
        and candidate.entity_text in accepted_relation_endpoints
    )
    return accepted


def _merge_entities(primary: list[Entity], secondary: list[Entity]) -> list[Entity]:
    merged: list[Entity] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for entity in primary + secondary:
        key = (entity.text, entity.start_idx, entity.end_idx)
        if key in seen:
            continue
        seen.add(key)
        merged.append(entity)
    return sorted(merged, key=lambda item: (item.start_idx or 0, -len(item.text)))


def _dedupe_relations(relations: list[Relation]) -> list[Relation]:
    result: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in relations:
        key = (relation.subject, relation.predicate, relation.object)
        if key in seen:
            continue
        seen.add(key)
        result.append(relation)
    return result


def _rebind_relation_to_entities(
    relation: Relation,
    entities: list[Entity],
) -> Relation | None:
    """Bind relation endpoint types to the surviving entity set by text."""

    type_by_text: dict[str, str] = {}
    for entity in entities:
        type_by_text.setdefault(entity.text, entity.type)
    subject_type = type_by_text.get(relation.subject)
    object_type = type_by_text.get(relation.object)
    if not subject_type or not object_type:
        return None
    return replace(
        relation,
        subject_type=subject_type,
        object_type=object_type,
    )


def _ensure_relation_anchors(
    entities: list[Entity],
    relations: list[Relation],
    source_entities: list[Entity],
) -> list[Entity]:
    """Keep endpoint entities when entity review rejected only their mention."""

    result = list(entities)
    for relation in relations:
        for endpoint_text, endpoint_type in (
            (relation.subject, relation.subject_type),
            (relation.object, relation.object_type),
        ):
            if not endpoint_text or any(item.text == endpoint_text for item in result):
                continue
            template = next(
                (item for item in source_entities if item.text == endpoint_text),
                None,
            )
            if template is not None:
                result.append(
                    replace(
                        template,
                        type=endpoint_type or template.type,
                        extraction_method="relation_anchor",
                        reliability_level=relation.reliability_level
                        or template.reliability_level,
                    )
                )
            elif endpoint_type:
                result.append(
                    Entity(
                        text=endpoint_text,
                        type=endpoint_type,
                        confidence=relation.confidence,
                        evidence=relation.evidence,
                        extraction_method="relation_anchor",
                        reliability_level=relation.reliability_level,
                    )
                )
    return _merge_entities(result, [])


def filter_low_reliability_results(
    entities: list[Entity],
    relations: list[Relation],
) -> tuple[list[Entity], list[Relation], int]:
    """Remove low-reliability offline candidates from the final result.

    The extractor still produces low-reliability candidates for recall and
    audit purposes.  They must not silently become final graph facts.  A
    retained medium/high relation may reference a low-reliability entity; in
    that case an explicit relation anchor is kept so endpoint validation does
    not erase the relation after entity gating.
    """

    kept_entities = [
        entity
        for entity in entities
        if str(entity.reliability_level or "").lower() != "low"
    ]
    kept_relations = [
        relation
        for relation in relations
        if str(relation.reliability_level or "").lower() != "low"
    ]
    filtered_count = (
        len(entities)
        - len(kept_entities)
        + len(relations)
        - len(kept_relations)
    )
    kept_entities = _ensure_relation_anchors(
        kept_entities,
        kept_relations,
        entities,
    )
    kept_relations = [
        bound_relation
        for relation in kept_relations
        if (bound_relation := _rebind_relation_to_entities(relation, kept_entities))
        is not None
    ]
    return kept_entities, kept_relations, filtered_count


def _decision_name(decisions: dict[str, Any], candidate_id: str) -> str:
    decision = decisions.get(candidate_id)
    return str(getattr(decision, "decision", "") or "").lower()


def _reviewed_offline_relation_is_supported(relation: Relation) -> bool:
    """Keep a low relation only when its exact wording supports the label."""

    evidence = str(getattr(relation, "evidence", "") or "")
    if not relation_evidence_supports_pair(
        evidence,
        relation.predicate,
        relation.subject,
        relation.object,
        subject_type=relation.subject_type,
        object_type=relation.object_type,
        require_disease_subject=True,
    ):
        return False
    if relation.predicate == "\u8f85\u52a9\u6cbb\u7597" and relation.object in {
        "\u6cbb\u7597",
        "\u6cbb\u7597\u65b9\u6cd5",
        "\u6cbb\u7597\u65b9\u6848",
        "\u6cbb\u7597\u63aa\u65bd",
        "\u8054\u5408\u6cbb\u7597",
        "\u7efc\u5408\u6cbb\u7597",
    }:
        # A generic treatment heading is not an actionable graph object;
        # keep the specific procedure labels for the gap path to add.
        return False
    if relation.predicate == "\u75c5\u56e0":
        if (
            relation.object_type == "dis"
            and "\u5bfc\u81f4" in evidence
            and not any(cue in evidence for cue in ("\u75c5\u56e0", "\u539f\u56e0", "\u7531\u4e8e"))
        ):
            return False
        if "\u7f3a\u4e4f" in evidence and "\u7f3a\u4e4f" not in relation.object:
            return False
    if relation.predicate == "\u836f\u7269\u6cbb\u7597":
        local_text = evidence
        if any(cue in local_text for cue in ("\u65e0\u53cd\u5e94", "\u4e0d\u9700\u8981", "\u65e0\u9700", "\u672a\u4f7f\u7528")):
            return False
    return True


def _retain_offline_candidate(
    item: Any,
    candidate_id: str,
    decisions: dict[str, Any],
    reviewed_ids: set[str],
    *,
    text: str = "",
    preserve_medium: bool = False,
) -> tuple[bool, Any]:
    """Apply the cascade policy to one offline candidate.

    High-reliability facts bypass review.  Medium facts are the safe offline
    baseline: only an explicit reject removes them.  A low fact is never kept
    by default; it must be explicitly accepted by the targeted LLM review and
    is promoted to a verified high-reliability fact after acceptance.
    """

    level = str(getattr(item, "reliability_level", "") or "").lower()
    decision_record = decisions.get(candidate_id)
    decision = _decision_name(decisions, candidate_id)
    if level == "low":
        if candidate_id in reviewed_ids and decision == "accept":
            if isinstance(item, Relation) and not _reviewed_offline_relation_is_supported(
                item
            ):
                return False, item
            if isinstance(item, Entity):
                normalized_entity = normalize_verified_entity(text, item)
                if normalized_entity is None:
                    return False, item
                item = normalized_entity
            review_confidence = float(
                getattr(decision_record, "confidence", 0.0) or 0.0
            )
            replacement = {
                "extraction_method": "llm_review_verified",
                "reliability_level": "high",
            }
            if review_confidence > 0:
                replacement["confidence"] = min(1.0, max(0.0, review_confidence))
            return True, replace(item, **replacement)
        return False, item
    if preserve_medium and level == "medium":
        return True, item
    if candidate_id in reviewed_ids and decision == "reject":
        return False, item
    return True, item


def _retain_offline_items(
    items: list[Any],
    kind: str,
    candidate_scope: str,
    decisions: dict[str, Any],
    reviewed_ids: set[str],
    *,
    text: str = "",
    preserve_medium: bool = False,
) -> tuple[list[Any], int]:
    """Retain offline items according to the shared entity/relation policy."""

    kept: list[Any] = []
    filtered_low_count = 0
    for index, item in enumerate(items):
        candidate_id = _scoped_candidate_id(
            candidate_scope, f"offline_{kind}:{index}"
        )
        should_keep, normalized = _retain_offline_candidate(
            item,
            candidate_id,
            decisions,
            reviewed_ids,
            text=text,
            preserve_medium=preserve_medium,
        )
        if should_keep:
            kept.append(normalized)
        elif str(getattr(item, "reliability_level", "") or "").lower() == "low":
            filtered_low_count += 1
    return kept, filtered_low_count


def _min_gap_relation_confidence() -> float:
    raw = os.getenv("CCF_TASK2_LLM_GAP_RELATION_MIN_CONFIDENCE", "0.80")
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.80


def _min_gap_entity_confidence() -> float:
    raw = os.getenv("CCF_TASK2_LLM_GAP_ENTITY_MIN_CONFIDENCE", "0.80")
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.80


def _accept_gap_entity(decision: Any) -> bool:
    if getattr(decision, "decision", "") != "accept":
        return False
    try:
        confidence = float(getattr(decision, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return confidence >= _min_gap_entity_confidence()


def _accept_gap_relation(
    text: str,
    relation: Relation,
    decision: Any,
) -> bool:
    if getattr(decision, "decision", "") != "accept":
        return False
    try:
        confidence = float(getattr(decision, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < _min_gap_relation_confidence():
        return False
    return relation_evidence_supports_pair(
        text,
        relation.predicate,
        relation.subject,
        relation.object,
        subject_type=relation.subject_type,
        object_type=relation.object_type,
        require_disease_subject=True,
    )


def extract_medical_knowledge_cascade(
    text: str,
    *,
    kg_db_path: str = "",
    llm: LLMClient,
) -> CascadeOutput:
    """执行离线优先、LLM 查缺补漏和候选复核。"""

    offline_entities = extract_entities_offline(text, kg_db_path)
    offline_relations = extract_relations_offline(
        text,
        entities=offline_entities,
        db_path=kg_db_path,
    )
    gap_segments, offline_review_candidates = prepare_cascade_targets(
        text,
        offline_entities,
        offline_relations,
    )
    offline_review_skipped = count_skipped_offline_candidates(
        offline_entities,
        offline_relations,
        offline_review_candidates,
    )
    gap_entities, gap_relations = extract_gap_facts(
        text,
        gap_segments,
        llm,
        anchor_entities=offline_entities,
    )
    gap_candidates = _gap_candidates(gap_entities, gap_relations, gap_segments)
    auto_accepted_ids = select_auto_accepted_gap_candidate_ids(gap_candidates)
    review_queue = dedupe_review_candidates(
        [
            candidate
            for candidate in gap_candidates + offline_review_candidates
            if candidate.candidate_id not in auto_accepted_ids
        ]
    )[: _max_review_candidates()]
    decisions = review_candidates_parallel(llm, review_queue)
    return apply_cascade_merge(
        text=text,
        entities=offline_entities,
        relations=offline_relations,
        gap_segments=gap_segments,
        review_candidates=offline_review_candidates,
        decisions=decisions,
        gap_entities=gap_entities,
        gap_relations=gap_relations,
        reviewed_candidate_ids={candidate.candidate_id for candidate in review_queue},
        auto_accepted_candidate_ids=auto_accepted_ids,
        offline_review_skipped_count=offline_review_skipped,
    )


def apply_cascade_merge(
    text: str,
    entities: list[Entity],
    relations: list[Relation],
    *,
    gap_segments: list[CascadeSegment],
    review_candidates: list[ReviewCandidate],
    decisions: dict[str, Any],
    gap_entities: list[Entity],
    gap_relations: list[Relation],
    candidate_scope: str = "",
    reviewed_candidate_ids: set[str] | None = None,
    auto_accepted_candidate_ids: set[str] | None = None,
    offline_review_skipped_count: int = 0,
) -> CascadeOutput:
    """Apply a batch review result to one record without cross-record leakage.

    ``reviewed_candidate_ids`` is the authoritative queue sent to the LLM. It
    keeps reported review counts honest when a batch-level safety cap is used.
    A missing decision preserves medium/high offline facts, but does not allow
    a low-reliability fact or a gap fact into the final result; the gated
    offline result is therefore the safe fallback.
    """
    gap_candidates = _gap_candidates(
        gap_entities,
        gap_relations,
        gap_segments,
        candidate_scope=candidate_scope,
    )
    all_candidates = dedupe_review_candidates(
        gap_candidates + review_candidates
    )
    all_candidate_ids = {candidate.candidate_id for candidate in all_candidates}
    if reviewed_candidate_ids is None:
        reviewed_ids = set(all_candidate_ids)
    else:
        reviewed_ids = set(reviewed_candidate_ids) & all_candidate_ids
    auto_accepted_ids = set(auto_accepted_candidate_ids or set()) & all_candidate_ids
    auto_accepted_ids -= reviewed_ids

    retained_offline_entities, filtered_entity_count = _retain_offline_items(
        entities,
        "entity",
        candidate_scope,
        decisions,
        reviewed_ids,
        text=text,
        preserve_medium=True,
    )
    retained_offline_relations, filtered_relation_count = _retain_offline_items(
        relations,
        "relation",
        candidate_scope,
        decisions,
        reviewed_ids,
        text=text,
        preserve_medium=True,
    )

    accepted_gap_entity_ids = {
        candidate.candidate_id
        for candidate in gap_candidates
        if candidate.kind == "entity"
        and (
            candidate.candidate_id in auto_accepted_ids
            or (
                candidate.candidate_id in reviewed_ids
                and _accept_gap_entity(decisions.get(candidate.candidate_id))
            )
        )
    }
    accepted_gap_entities: list[Entity] = []
    for index, entity in enumerate(gap_entities):
        candidate_id = next(
            (
                candidate.candidate_id
                for candidate in gap_candidates
                if candidate.kind == "entity"
                and candidate.entity_text == entity.text
                and candidate.entity_type == entity.type
                and candidate.candidate_id.endswith(f":{index}")
            ),
            "",
        )
        if candidate_id not in accepted_gap_entity_ids:
            continue
        normalized_entity = normalize_verified_entity(text, entity)
        if normalized_entity is None:
            continue
        accepted_gap_entities.append(
            replace(
                normalized_entity,
                extraction_method="llm_gap_verified",
                reliability_level="high",
            )
        )

    accepted_gap_relations: list[Relation] = []
    for index, relation in enumerate(gap_relations):
        candidate_id = next(
            (
                candidate.candidate_id
                for candidate in gap_candidates
                if candidate.kind == "relation"
                and candidate.subject == relation.subject
                and candidate.predicate == relation.predicate
                and candidate.object == relation.object
                and candidate.candidate_id.endswith(f":{index}")
            ),
            "",
        )
        if not candidate_id or candidate_id not in reviewed_ids | auto_accepted_ids:
            continue
        if candidate_id in auto_accepted_ids:
            candidate = next(
                item for item in gap_candidates if item.candidate_id == candidate_id
            )
            accepted = (
                candidate.confidence >= _gap_auto_accept_confidence()
                and relation_evidence_supports_pair(
                    text,
                    relation.predicate,
                    relation.subject,
                    relation.object,
                    subject_type=relation.subject_type,
                    object_type=relation.object_type,
                    require_disease_subject=True,
                )
            )
        else:
            accepted = _accept_gap_relation(
                text,
                relation,
                decisions.get(candidate_id),
            )
        if not accepted:
            continue
        bound_relation = _rebind_relation_to_entities(
            relation,
            retained_offline_entities + accepted_gap_entities,
        )
        if bound_relation is None:
            continue
        accepted_gap_relations.append(
            replace(
                bound_relation,
                extraction_method="llm_gap_verified",
                reliability_level="high",
            )
        )

    merged_entities = _merge_entities(retained_offline_entities, accepted_gap_entities)
    merged_relations = _dedupe_relations(
        retained_offline_relations + accepted_gap_relations
    )
    merged_entities = _ensure_relation_anchors(
        merged_entities,
        merged_relations,
        entities + gap_entities,
    )
    merged_relations = [
        bound_relation
        for relation in merged_relations
        if (bound_relation := _rebind_relation_to_entities(relation, merged_entities))
        is not None
    ]
    offline_relation_keys = {
        (relation.subject, relation.predicate, relation.object)
        for relation in retained_offline_relations
    }
    llm_added_relations = [
        relation
        for relation in accepted_gap_relations
        if (relation.subject, relation.predicate, relation.object) not in offline_relation_keys
    ]
    offline_entity_keys = {
        (entity.text, entity.start_idx, entity.end_idx)
        for entity in retained_offline_entities
    }
    llm_added_entities = [
        entity
        for entity in accepted_gap_entities
        if (entity.text, entity.start_idx, entity.end_idx) not in offline_entity_keys
    ]
    triples: list[Triple] = relations_to_triples(merged_relations, min_confidence=0.0)
    rejected_count = sum(
        1
        for candidate in all_candidates
        if candidate.candidate_id in reviewed_ids
        and getattr(decisions.get(candidate.candidate_id), "decision", "") == "reject"
    )
    offline_filtered_count = filtered_entity_count + filtered_relation_count
    return CascadeOutput(
        entities=merged_entities,
        relations=merged_relations,
        triples=triples,
        gap_segment_count=len(gap_segments),
        gap_candidate_count=len(gap_candidates),
        reviewed_candidate_count=len(reviewed_ids),
        auto_accepted_candidate_count=len(auto_accepted_ids),
        review_skipped_candidate_count=(
            offline_review_skipped_count
            + len(all_candidate_ids - reviewed_ids - auto_accepted_ids)
        ),
        offline_filtered_candidate_count=offline_filtered_count,
        rejected_candidate_count=rejected_count,
        llm_added_count=len(llm_added_relations),
        llm_added_entity_count=len(llm_added_entities),
        llm_added_relation_count=len(llm_added_relations),
    )
