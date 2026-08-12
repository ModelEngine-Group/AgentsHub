# -*- coding: utf-8 -*-
"""任务二级联抽取的 LLM 适配层。

本模块只做两件事：批量复核已经存在的低可靠候选，以及从被离线链路漏掉
的句子中提出候选。最终是否进入结果由 ``task2_cascade`` 的严格门控决定。
LLM 不会在这里直接写数据库。
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any, Iterable

from .llm_client import LLMClient
from .medical_extraction_validation import (
    normalize_entity_type,
    normalize_relation_type,
    relation_evidence_supports_pair,
    validate_entities,
    validate_relations,
)
from .medical_lexicon import load_known_relation_pairs
from .medical_ner import ENTITY_TYPES
from .medical_re import RELATION_TYPES
from .schemas import Entity, Relation
from .task2_cascade_schemas import CascadeSegment, ReviewCandidate, ReviewDecision


_DECISION_ALIASES = {
    "accept": "accept",
    "accepted": "accept",
    "yes": "accept",
    "支持": "accept",
    "保留": "accept",
    "reject": "reject",
    "rejected": "reject",
    "no": "reject",
    "不支持": "reject",
    "删除": "reject",
    "uncertain": "uncertain",
    "unknown": "uncertain",
    "不确定": "uncertain",
    "无法判断": "uncertain",
}


def _as_list(payload: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _normalize_decision(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return _DECISION_ALIASES.get(raw, "uncertain")


def _normalize_confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


_FORWARD_GAP_RELATIONS = {
    "\u4e34\u5e8a\u8868\u73b0",
    "\u53d1\u75c5\u90e8\u4f4d",
    "\u5916\u4fb5\u90e8\u4f4d",
    "\u8f6c\u79fb\u90e8\u4f4d",
    "\u75c5\u56e0",
    "\u5e76\u53d1\u75c7",
    "\u75c5\u7406\u5206\u578b",
}
_DISEASE_SUBJECT_SUFFIXES = (
    "\u75c5",
    "\u708e",
    "\u764c",
    "\u75c7",
    "\u7624",
    "\u611f\u67d3",
    "\u8870\u7aed",
    "\u7efc\u5408\u5f81",
    "\u635f\u4f24",
)
_NON_DISEASE_ABBREVIATIONS = {"ICU", "ERCP", "CT", "MRI", "E2", "FSH", "LH", "CMV"}
_ABBREVIATION_RE = re.compile(r"\b[A-Z][A-Za-z0-9-]{1,5}\b")
_GENERIC_GAP_ENTITY_TERMS = {
    "\u75c7\u72b6",
    "\u4e34\u5e8a\u75c7\u72b6",
    "\u4e34\u5e8a\u8868\u73b0",
    "\u4e34\u5e8a\u7efc\u5408\u5f81",
    "\u4f53\u5f81",
    "\u60a3\u8005",
    "\u60a3\u513f",
    "\u5c0f\u513f",
    "\u5c0f\u5a74\u513f",
    "\u513f\u7ae5",
    "\u6210\u4eba",
    "\u8001\u4eba",
    "\u75c5\u4eba",
    "\u4eba\u7fa4",
    "\u75c5\u7a0b\u65e9\u671f",
    "\u75c5\u7a0b\u4e2d\u671f",
    "\u75c5\u7a0b\u665a\u671f",
    "\u9274\u522b\u8bca\u65ad",
    "\u8bca\u65ad\u6807\u51c6",
    "\u6d41\u884c\u75c5\u5b66\u53f2",
    "\u4f20\u67d3\u6027",
    "\u65e0\u660e\u663e\u6548\u679c",
    "\u591a\u79cd\u591a\u6837\u7684\u4e34\u5e8a\u8868\u73b0",
    "\u5a74\u5e7c\u513f",
    "\u8425\u517b\u4e0d\u826f\u513f",
    "\u80ce\u513f",
    "\u5916\u5468",
    "\u75c5\u6b7b\u7387",
    "\u53d1\u75c5\u7387",
    "\u7ec4\u7ec7\u5b66\u5206\u578b",
    "\u75c5\u7406\u5206\u578b",
    "\u80bf\u7624\u8d1f\u8377",
    "\u6bcf\u5c0f\u65f6\u5c3f\u91cf",
    "\u5438\u5165\u6c27\u6d53\u5ea6",
    "Poiseuille\u516c\u5f0f",
    "\u5206\u5ea6",
    "\u963b\u529b",
    "\u4f53\u5faa\u73af\u963b\u529b",
    "\u8212\u5f20\u671f",
    "\u6536\u7f29\u671f",
    "\u53d1\u70ed\u671f",
    "\u51fa\u75b9\u671f",
    "\u524d\u9a71\u671f",
    "\u4eba\u6216\u7334\u80be\u7ec6\u80de",
    "\u513f\u79d1\u7528\u836f",
    "\u539f\u836f\u7269",
    "\u5b9e\u9a8c\u836f\u7269",
    "\u836f\u7269",
    "\u5168\u8840",
}
_GENERIC_GAP_ENTITY_FRAGMENTS = (
    "\u60a3\u513f",
    "\u5c0f\u513f",
    "\u5a74\u513f",
    "\u513f\u7ae5",
    "\u4f53\u5f81\u51fa\u73b0",
    "\u4f53\u5f81\u51fa\u73b0\u65e9",
    "\u5e74\u957f\u513f",
    "\u6210\u4eba",
    "\u6708\u9f84",
    "\u5e7c\u5a74",
    "\u5c0f\u513f",
    "\u6bcd\u4eb2",
    "\u75c5\u521d\u671f",
    "\u75c5\u7a0b",
    "\u591a\u79cd\u591a\u6837",
    "\u65e0\u660e\u663e\u6548\u679c",
    "\u4f20\u67d3\u6027",
    "\u65e0\u6548",
)
_NON_ENTITY_PUNCTUATION = set("\u3001\uff0c\uff1b\uff1a\uff08\uff09()")
_GAP_MEASUREMENT_ENTITY_MARKERS = (
    "\u538b\u529b",
    "\u963b\u529b",
    "\u987a\u5e94\u6027",
    "\u8840\u6d41\u91cf",
    "\u8f93\u51fa\u91cf",
    "\u8840\u538b",
    "\u589e\u5927",
    "\u51cf\u5c11",
    "\u589e\u591a",
)
_GAP_GENERIC_EQUIPMENT_TERMS = {
    "12\u5bfc\u5fc3\u7535\u56fe",
    "V1",
    "V3R",
    "V4R",
    "\u53f3\u80f8\u524d\u5bfc\u8054QRS-T\u6ce2\u7fa4",
    "HFO",
    "\u632f\u8361\u5668",
    "\u6fc0\u5149",
}
_GAP_GENERIC_PROCEDURE_TERMS = {
    "\u6cbb\u7597",
    "\u836f\u7406\u4f5c\u7528",
    "\u9ad8\u9891\u5207\u6362\u6c14\u6d41",
    "\u80ba\u6ce1\u547c\u6c14\u672b\u6b63\u538b",
    "\u5e38\u89c4\u65b9\u6848",
    "\u5e38\u89c4\u6cbb\u7597\u65b9\u6848",
    "\u2161\u671f\u4e34\u5e8a\u65b9\u6848",
    "\u8d25\u8840\u75c7\u7684\u9632\u6cbb",
    "\u7279\u6b8a\u90e8\u4f4d\u70e7\u70eb\u4f24\u7684\u5904\u7406",
}
_GAP_GENERIC_BODY_TERMS = {
    "\u8db3\u6708\u80ce\u513f",
    "\u80ba\u4f53\u5faa\u73af",
    "\u4f53\u5faa\u73af",
    "\u5fc3\u5ba4",
    "\u810f\u5668",
    "\u521b\u9762",
    "\u95ed\u9501\u74e3\u819c",
}
_COMPOSITE_SYM_FRAGMENTS = (
    "\u9634\u5f71",
    "\u6d78\u6da6",
    "\u4f53\u5f81",
    "\u542c\u8bca",
    "\u75c7\u72b6",
    "\u75c5\u60c5",
    "\u75c5\u7076",
    "\u8303\u56f4",
    "\u6539\u53d8",
    "\u5206\u6ccc\u7269",
    "\u8d77\u75c5",
    "\u597d\u8f6c",
    "\u75ca\u6108",
    "\u8f83\u8f7b",
    "\u7f13\u6162",
)
_TEST_PROCEDURE_CUES = (
    "\u8bd5\u9a8c",
    "\u68c0\u67e5",
    "\u6cbb\u7597",
    "\u57f9\u517b",
    "\u76d1\u62a4",
    "X\u7ebf\u7247",
    "PCR",
    "\u76ae\u8bd5",
    "\u68c0\u6d4b",
    "\u6d4b\u5b9a",
    "\u6d82\u7247",
    "\u955c\u68c0",
    "\u5206\u79bb",
    "\u5fc3\u7535\u56fe",
    "\u8111\u7535\u56fe",
    "\u8d85\u58f0",
)
_GAP_DIRECTION_ONLY_BODY_TERMS = {
    "\u5355\u4fa7",
    "\u53cc\u4fa7",
    "\u4e0b\u53f6",
    "\u5916\u5e26",
    "\u5fc3\u5f71\u540e",
}
_GAP_TEST_INDICATOR_SUFFIXES = (
    "\u529f\u80fd",
    "\u8ba1\u6570",
    "\u6c34\u5e73",
    "\u6bd4\u503c",
    "\u6307\u6807",
)
_GAP_TEST_ANALYTE_SUFFIXES = (
    "\u9176",
    "\u86cb\u767d",
    "\u80c6\u7ea2\u7d20",
    "\u8461\u8404\u7cd6",
    "\u5c3f\u7d20\u6c2e",
    "\u5c3f\u9178",
    "\u808c\u9150",
)
_GAP_SUBJECT_CONNECTOR_MARKERS = (
    "\u60a3\u8005",
    "\u75c5\u53f2",
    "\u6216",
    "\u4ee5\u53ca",
    "\u53ca",
    "\uff08",
    "\uff09",
    "(",
    ")",
)
_GAP_CLINICAL_NEGATIVE_CUES = (
    "\u8be2\u95ee",
    "\u7b5b\u67e5",
    "\u95ee\u5377",
    "\u91cf\u8868",
    "\u8bc4\u4f30",
    "\u98ce\u9669",
    "\u8003\u8651",
)
_GAP_DRUG_RELATION_CUES = (
    "\u836f\u7269\u6cbb\u7597",
    "\u6cbb\u7597",
    "\u7528\u836f",
    "\u670d\u7528",
    "\u7ed9\u4e88",
    "\u4f7f\u7528",
    "\u5e94\u7528",
    "\u6ce8\u5c04",
)
_GAP_MICROBE_SUFFIXES = (
    "\u75c5\u6bd2",
    "\u7ec6\u83cc",
    "\u652f\u539f\u4f53",
    "\u8863\u539f\u4f53",
    "\u771f\u83cc",
    "\u87ba\u65cb\u4f53",
    "\u7403\u83cc",
    "\u6746\u83cc",
)
_GAP_PROCEDURE_HEADS = (
    "\u68c0\u67e5",
    "\u68c0\u6d4b",
    "\u6d4b\u5b9a",
    "\u57f9\u517b",
    "\u6d82\u7247",
    "\u955c\u68c0",
    "\u6297\u539f",
    "\u6297\u4f53",
)
_GAP_GENERIC_RELATION_OBJECTS = {
    "\u75c7\u72b6",
    "\u4e34\u5e8a\u75c7\u72b6",
    "\u4e34\u5e8a\u8868\u73b0",
    "\u4e34\u5e8a\u7efc\u5408\u5f81",
    "\u8868\u73b0",
    "\u68c0\u67e5",
    "\u6cbb\u7597",
    "\u4fdd\u80ce\u836f",
    "\u5b89\u7720\u836f",
    "\u6297\u7126\u8651\u5242",
}
_GAP_CAUSAL_CUES = (
    "\u75c5\u56e0",
    "\u539f\u56e0",
    "\u7531\u4e8e",
    "\u5f15\u8d77",
    "\u5bfc\u81f4",
    "\u75c5\u539f\u4f53",
    "\u611f\u67d3",
)
_GAP_OUTCOME_SUFFIXES = (
    "\u5206\u6ccc\u660e\u663e\u51cf\u5c11",
    "\u751f\u7269\u5b66\u6d3b\u6027\u964d\u4f4e",
    "\u529f\u80fd\u964d\u4f4e",
    "\u6c34\u5e73\u4e0b\u964d",
    "\u542b\u91cf\u964d\u4f4e",
    "\u6d3b\u6027\u4e0b\u964d",
)


def _looks_like_disease_subject(value: str) -> bool:
    normalized = str(value or "").strip()
    return bool(normalized) and any(
        normalized.endswith(suffix) for suffix in _DISEASE_SUBJECT_SUFFIXES
    )


def _text_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    if not value:
        return []
    result: list[tuple[int, int]] = []
    start = text.find(value)
    while start >= 0:
        result.append((start, start + len(value)))
        start = text.find(value, start + 1)
    return result


def _is_structured_relation_text(text: str) -> bool:
    """Detect the CMeIE-style disease/section serialization.

    CMeIE records use ``@`` and ``###`` as explicit context markers.  CMeEE
    records are ordinary NER sentences, where coordinated noun phrases are
    often annotated as one maximal span.  The distinction is used only to
    specialize the gap prompt and post-validation; it does not change the
    public extraction contract.
    """

    return "@" in text or "###" in text


def _local_entity_type(
    entities: list[Entity], value: str, fallback: str
) -> str:
    for entity in entities:
        if entity.text == value and entity.type:
            return entity.type
    return fallback


def _relation_between_windows(
    text: str,
    subject_spans: list[tuple[int, int]],
    object_spans: list[tuple[int, int]],
    *,
    forward_only: bool = False,
) -> list[str]:
    """Return the text between every non-overlapping endpoint occurrence."""

    windows: list[str] = []
    for subject_start, subject_end in subject_spans:
        for object_start, object_end in object_spans:
            if object_start > subject_end:
                windows.append(text[subject_end:object_start])
            elif not forward_only and subject_start > object_end:
                windows.append(text[object_end:subject_start])
    return windows


def _gap_relation_context_is_supported(
    text: str,
    relation: Relation,
    local_entities: list[Entity],
    anchor_entities: list[Entity] | None = None,
) -> bool:
    """Apply endpoint-type and wording gates before accepting a gap relation."""

    endpoint_entities = [*(anchor_entities or []), *local_entities]
    subject_type = _local_entity_type(
        endpoint_entities, relation.subject, relation.subject_type
    )
    object_type = _local_entity_type(
        endpoint_entities, relation.object, relation.object_type
    )
    subject_spans = _text_occurrences(text, relation.subject)
    object_spans = _text_occurrences(text, relation.object)
    if not subject_spans or not object_spans:
        return False
    start = min(subject_spans[0][0], object_spans[0][0])
    end = max(subject_spans[0][1], object_spans[0][1])
    local_text = text[max(0, start - 24) : min(len(text), end + 24)]
    between_windows = _relation_between_windows(
        text,
        subject_spans,
        object_spans,
        forward_only=relation.predicate in _FORWARD_GAP_RELATIONS,
    )
    known_predicate = load_known_relation_pairs().get(
        (relation.subject, relation.object)
    )
    if known_predicate and known_predicate != relation.predicate:
        return False
    if relation.predicate == "\u4e34\u5e8a\u8868\u73b0":
        if object_type not in {"sym", "bod"}:
            return False
        if relation.object in _GAP_GENERIC_RELATION_OBJECTS:
            return False
        if relation.object.endswith("\u75c7\u72b6"):
            return False
        if any(suffix in relation.object for suffix in _GAP_OUTCOME_SUFFIXES):
            return False
        if object_type == "sym" and any(
            cue in local_text for cue in ("\u8111\u7535\u56fe", "\u8111\u7535\u751f\u7406", "\u8111\u7535\u6d3b\u52a8")
        ):
            return False
        if not any(
            cue in window
            for window in between_windows
            for cue in ("\u4e34\u5e8a\u8868\u73b0", "\u4f53\u5f81", "\u75c7\u72b6", "\u8868\u73b0\u4e3a", "\u4f34\u6709")
        ):
            return False
        if not any(
            cue in window
            for window in between_windows
            for cue in ("\u4f53\u5f81/\u75c7\u72b6", "\u4e34\u5e8a\u8868\u73b0")
        ) and any(len(window) > 64 for window in between_windows):
            return False
        if not any(
            cue in window
            for window in between_windows
            for cue in ("\u4f53\u5f81/\u75c7\u72b6", "\u4e34\u5e8a\u8868\u73b0")
        ) and not any(
            cue in text[max(0, object_start - 24) : object_start]
            for object_start, _ in object_spans
            for cue in ("\u8868\u73b0\u4e3a", "\u4f34\u6709", "\u51fa\u73b0")
        ):
            return False
    if relation.predicate == "\u75c5\u56e0":
        if not between_windows:
            return False
        if any(suffix in relation.object for suffix in _GAP_OUTCOME_SUFFIXES):
            return False
        if not any(
            cue in window
            for window in between_windows
            for cue in ("\u75c5\u56e0", "\u539f\u56e0", "\u7531\u4e8e", "\u7531", "\u5f15\u8d77", "\u5bfc\u81f4", "\u8bf1\u53d1", "\u75c5\u539f\u4f53")
        ):
            return False
        if (
            any("\u76f8\u5173" in window for window in between_windows)
            and not any(
                cue in window
                for window in between_windows
                for cue in _GAP_CAUSAL_CUES
            )
        ):
            return False
        if any(
            cue in window for window in between_windows for cue in ("\u5bfc\u81f4", "\u5f15\u8d77", "\u8bf1\u53d1")
        ) and not any(
            cue in window
            for window in between_windows
            for cue in ("\u7531", "\u75c5\u56e0", "\u539f\u56e0", "\u75c5\u539f\u4f53", "\u611f\u67d3")
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
        return False
    if relation.predicate == "\u836f\u7269\u6cbb\u7597":
        if object_type != "dru":
            return False
        if relation.object in _GAP_GENERIC_RELATION_OBJECTS or relation.object.endswith(
            ("\u6cbb\u7597\u836f\u7269", "\u836f\u7269")
        ):
            return False
        if len(relation.object) > 8 and relation.object.endswith(
            ("\u6291\u5236\u5242", "\u6fc0\u52a8\u5242", "\u62ee\u6297\u5242")
        ):
            return False
        for subject_start, subject_end in subject_spans:
            for object_start, object_end in object_spans:
                if object_start > subject_end:
                    between = text[subject_end:object_start]
                elif subject_start > object_end:
                    between = text[object_end:subject_start]
                else:
                    continue
                if any(cue in between[:48] for cue in _GAP_DRUG_RELATION_CUES):
                    break
            else:
                continue
            break
        else:
            return False
        if any(
            ("\u9884\u9632\u6027" in window or "\u9884\u9632\u7528" in window)
            and "\u6cbb\u7597" not in window
            for window in between_windows
        ):
            return False
        for subject_start, subject_end in subject_spans:
            subject_prefix = text[max(0, subject_start - 12) : subject_start]
            subject_suffix = text[subject_end : subject_end + 4]
            if any(cue in subject_prefix for cue in ("\u4ea7\u751f", "\u5bfc\u81f4", "\u5f15\u8d77", "\u5e76\u53d1")) and subject_suffix.startswith("\u7684"):
                return False
        if _is_structured_relation_text(text):
            for subject_start, _ in subject_spans:
                marker = max(text.rfind("@", 0, subject_start), text.rfind("###", 0, subject_start))
                if marker < 0:
                    continue
                inline_prefix = text[marker + 1 : subject_start]
                if any(cue in inline_prefix for cue in ("\u75c5\u60c5\u7a33\u5b9a", "\u8054\u5408", "\u82e5\u6709", "\u5982\u679c\u6709")):
                    return False
    if relation.predicate == "\u8f85\u52a9\u6cbb\u7597":
        if relation.object in {"\u5fc3\u7406\u793e\u4f1a\u5b66", "\u75c7\u72b6\u76f8\u5173\u5e72\u9884"} or relation.object.endswith("\u5e72\u9884"):
            return False
    if relation.predicate in {"\u5b9e\u9a8c\u5ba4\u68c0\u67e5", "\u8f85\u52a9\u68c0\u67e5", "\u5f71\u50cf\u5b66\u68c0\u67e5"}:
        subject_end = max(end for _, end in subject_spans)
        object_start = min(start for start, _ in object_spans)
        between = text[subject_end:object_start]
        action_cues = ("\u68c0\u67e5", "\u68c0\u6d4b", "\u6d4b\u5b9a", "\u5316\u9a8c", "\u8fdb\u884c", "\u53ef\u4f5c", "\u590d\u67e5")
        object_is_test_phrase = any(cue in relation.object for cue in _GAP_PROCEDURE_HEADS)
        if not any(cue in between for cue in action_cues) and not object_is_test_phrase:
            return False
        heading_cues = ("\u5b9e\u9a8c\u5ba4\u68c0\u67e5", "\u8f85\u52a9\u68c0\u67e5", "\u5f71\u50cf\u5b66\u68c0\u67e5")
        heading_matches = [
            (between.rfind(cue), cue) for cue in heading_cues if cue in between
        ]
        if heading_matches:
            heading_position, heading = max(heading_matches)
            trailing = between[heading_position + len(heading) :]
            if not any(cue in trailing for cue in action_cues) and not object_is_test_phrase:
                return False
    if relation.predicate == "\u4e34\u5e8a\u8868\u73b0":
        if "\u4e3b\u8bc9" in local_text:
            return False
        if any(cue in local_text for cue in _GAP_CLINICAL_NEGATIVE_CUES):
            return False
    return True


def _gap_subject_context_is_supported(
    text: str,
    relation: Relation,
    local_entities: list[Entity],
    anchor_entities: list[Entity] | None,
) -> bool:
    """Reject a gap relation whose subject is a wrong section or age term."""

    known_diseases = {
        entity.text
        for entity in [*(anchor_entities or []), *local_entities]
        if entity.type == "dis" and entity.text
    }
    anchor_diseases = {
        entity.text
        for entity in anchor_entities or []
        if entity.type == "dis" and entity.text
    }
    if anchor_diseases:
        anchor_supported = any(
            anchor == relation.subject
            or (len(anchor) >= 2 and anchor in relation.subject)
            for anchor in anchor_diseases
        )
        if not anchor_supported:
            return False
        if relation.subject != next(
            (anchor for anchor in anchor_diseases if anchor == relation.subject),
            "",
        ) and any(
            marker in relation.subject for marker in _GAP_SUBJECT_CONNECTOR_MARKERS
        ):
            return False
    if relation.subject not in known_diseases and not _looks_like_disease_subject(
        relation.subject
    ):
        return False

    subject_spans = _text_occurrences(text, relation.subject)
    object_spans = _text_occurrences(text, relation.object)
    if not subject_spans or not object_spans:
        return False
    for subject_start, subject_end in subject_spans:
        section_prefix = text[max(0, subject_start - 96) : subject_start]
        competing_section_disease = any(
            disease != relation.subject and disease in section_prefix
            for disease in known_diseases
        )
        if (
            competing_section_disease
            and ("@" in section_prefix or "###" in section_prefix)
            and "###" not in section_prefix[-16:]
            and relation.predicate
            in {
                "\u4e34\u5e8a\u8868\u73b0",
                "\u836f\u7269\u6cbb\u7597",
                "\u8f85\u52a9\u68c0\u67e5",
                "\u5b9e\u9a8c\u5ba4\u68c0\u67e5",
                "\u5f71\u50cf\u5b66\u68c0\u67e5",
            }
        ):
            continue
        for object_start, object_end in object_spans:
            if relation.predicate in _FORWARD_GAP_RELATIONS and subject_start > object_start:
                continue
            left, right = sorted((subject_end, object_start))
            if left > right:
                continue
            competing_disease = any(
                disease != relation.subject
                and disease in text[left:right]
                for disease in known_diseases
            )
            competing_alias = any(
                token not in _NON_DISEASE_ABBREVIATIONS
                and token != relation.subject
                for token in _ABBREVIATION_RE.findall(text[left:right])
            )
            if not competing_disease and not competing_alias:
                return True
    return False


def _gap_section_subject_is_supported(
    text: str,
    relation: Relation,
    known_diseases: set[str],
) -> bool:
    """Prefer the disease heading that owns a ``###`` section.

    CMeIE records use ``###`` as a section boundary.  A later repeated
    ``disease@`` marker can otherwise make a relation inherit the preceding
    disease even when the section heading names another disease.  This gate
    applies only to section-bound facts; cross-disease diagnostic/transform
    relations remain eligible for the normal relation-specific checks.
    """

    section_scoped_predicates = {
        "临床表现",
        "并发症",
        "病因",
        "发病部位",
        "药物治疗",
        "辅助检查",
        "实验室检查",
        "影像学检查",
    }
    if relation.predicate not in section_scoped_predicates:
        return True

    subject_spans = _text_occurrences(text, relation.subject)
    object_spans = _text_occurrences(text, relation.object)
    if not subject_spans or not object_spans:
        return False
    for subject_start, _ in subject_spans:
        for object_start, _ in object_spans:
            if (
                relation.predicate in _FORWARD_GAP_RELATIONS
                and subject_start > object_start
            ):
                continue
            left, right = sorted(
                (
                    subject_start + len(relation.subject),
                    object_start,
                )
            )
            between = text[left:right]
            if "###" in between:
                continue
            marker = text.rfind("###", 0, subject_start)
            if marker < 0:
                return True
            section_text = text[marker + 3 : subject_start]
            section_diseases = [
                disease
                for disease in known_diseases
                if disease and disease in section_text
            ]
            if not section_diseases:
                return True
            heading = max(
                section_diseases,
                key=lambda disease: section_text.rfind(disease) + len(disease),
            )
            if heading == relation.subject:
                return True
    return False


def _gap_entity_context_is_supported(
    entity: Entity,
    segment: CascadeSegment,
    anchor_entities: list[Entity] | None,
) -> bool:
    """Keep specific gap mentions and reject headings, demographics, and fragments."""

    value = str(entity.text or "").strip()
    structured_relation_text = _is_structured_relation_text(segment.text)
    if not value or value in _GENERIC_GAP_ENTITY_TERMS or value in {
        "\u68c0\u67e5",
        "\u5b9e\u9a8c\u5ba4\u68c0\u67e5",
        "\u4e34\u5e8a\u68c0\u67e5",
    }:
        return False
    if any(fragment in value for fragment in _GENERIC_GAP_ENTITY_FRAGMENTS):
        return False
    if any(char in value for char in _NON_ENTITY_PUNCTUATION):
        return False
    if entity.type == "sym" and any(
        fragment in value for fragment in _COMPOSITE_SYM_FRAGMENTS
    ):
        return False
    if entity.type == "sym" and (
        value.startswith("\u5404\u79cd")
        or "\u75c5\u53f2" in value
        or "\u672a\u6709\u6548\u63a7\u5236" in value
    ):
        return False
    if entity.type == "ite" and any(
        fragment in value
        for fragment in ("\u4e0b\u964d", "\u5347\u9ad8", "\u660e\u663e", "\u9633\u6027", "\u9634\u6027")
    ):
        return False
    if entity.type == "bod" and any(
        token in value for token in ("\u5355\u4fa7", "\u53cc\u4fa7", "\u5916\u5e26", "\u5fc3\u5f71\u540e")
    ):
        return False
    if entity.type == "mic" and value.endswith("\u6027"):
        return False
    if not structured_relation_text and entity.type == "mic":
        if any(
            marker in segment.text
            for marker in ("\u75c5\u6bd2\u6027", "\u7ec6\u83cc\u6027", "\u652f\u539f\u4f53", "\u8863\u539f\u4f53")
        ) and "\u3001" in segment.text:
            return False

    # CMeEE annotates a number of coordinated examination/treatment phrases as
    # one maximal ``pro`` span.  Splitting them into isolated analytes,
    # pathogens, or body materials creates plausible-looking but wrong NER
    # facts.  Keep the structured CMeIE path unchanged: its relation objects
    # are intentionally enumerated one by one.
    if not structured_relation_text:
        if entity.type == "dis" and len(value) <= 1:
            return False
        if entity.type == "dis" and (
            re.search(r"\d", value)
            or "\u5e7c\u5a74" in value
            or "\u6708\u9f84" in value
            or value.endswith("\u6bcd\u4eb2")
        ):
            return False
        if entity.type == "sym" and any(
            cue in value
            for cue in (
                "\u6548\u679c",
                "\u4f20\u67d3\u6027",
                "\u5355\u4fa7",
                "\u53cc\u4fa7",
                "\u65e0\u660e\u663e",
                "\u591a\u79cd\u591a\u6837",
            )
        ):
            return False
        if entity.type == "pro" and value == "\u6d41\u884c\u75c5\u5b66\u53f2":
            return False
        if entity.type == "equ":
            # Keep explicit equipment mentions while rejecting lead names and
            # bare device abbreviations returned by the gap prompt.
            if value in _GAP_GENERIC_EQUIPMENT_TERMS or not any(
                cue in value
                for cue in (
                    "\u8bbe\u5907",
                    "\u4eea\u5668",
                    "\u76d1\u62a4",
                    "\u8d77\u640f\u5668",
                    "\u6cf5",
                    "\u673a",
                )
            ):
                return False
        if entity.type == "ite" and any(
            marker in value for marker in _GAP_MEASUREMENT_ENTITY_MARKERS
        ):
            return False
        if entity.type == "bod" and (
            value in _GAP_GENERIC_BODY_TERMS
            or "\u6216" in value
            or any(marker in value for marker in _GAP_MEASUREMENT_ENTITY_MARKERS[:6])
        ):
            return False
        if entity.type == "pro" and value in _GAP_GENERIC_PROCEDURE_TERMS:
            return False
        if entity.type == "mic" and "\u591a\u6838\u5de8\u7ec6\u80de" in value and "\u75c5\u6bd2" not in value:
            return False
        if entity.type in {"bod", "ite"} and value in {"\u75f0", "\u54bd\u5206\u6ccc\u7269"} and any(
            cue in segment.text for cue in _GAP_PROCEDURE_HEADS
        ):
            return False
        if entity.type == "pro" and value in {"\u6d82\u7247", "\u955c\u68c0"} and any(
            cue in segment.text for cue in ("\u6d82\u7247\u76f4\u63a5\u955c\u68c0", "\u65b9\u6cd5\u68c0\u6d4b")
        ):
            return False
        if entity.type == "sym" and value in {"\u767d\u75f0", "\u8113\u75f0"} and "\u54b3\u55fd\u591a\u4f34\u6709" in segment.text:
            return False
        if "\u6297\u539f\u6d4b\u5b9a" in segment.text and "\u3001" in segment.text:
            if entity.type != "pro":
                return False
        if "\u548c\uff08\u6216\uff09" in segment.text and value != segment.text.strip():
            if "\u68c0\u6d4b" in segment.text or "\u6297\u4f53" in segment.text:
                return False
        local_entity_start = entity.start_idx or 0
        local_prefix = segment.text[max(0, local_entity_start - 18) : local_entity_start]
        if entity.type == "sym" and re.search(
            r"(?:\u65e0|\u6ca1\u6709|\u672a\u89c1|\u672a\u53d1\u73b0|\u5426\u8ba4|\u4e0d\u8003\u8651|\u6392\u9664)$",
            local_prefix,
        ):
            return False

    global_start = segment.start_idx + (entity.start_idx or 0)
    global_end = segment.start_idx + (
        entity.end_idx if entity.end_idx is not None else (entity.start_idx or 0)
    )
    for anchor in anchor_entities or []:
        if anchor.start_idx is None or anchor.end_idx is None:
            continue
        overlaps = global_start <= anchor.end_idx and global_end >= anchor.start_idx
        if not overlaps:
            continue
        same_span = global_start == anchor.start_idx and global_end == anchor.end_idx
        contained = (
            anchor.start_idx <= global_start and global_end <= anchor.end_idx
        )
        if same_span or contained:
            return False
    return True


def _normalize_gap_entity(entity: Entity, segment: CascadeSegment) -> Entity | None:
    """Correct high-signal type labels before the final gap-entity gate."""

    value = str(entity.text or "").strip()
    entity_type = entity.type
    structured_relation_text = _is_structured_relation_text(segment.text)
    local_start = entity.start_idx or 0
    local_end = entity.end_idx if entity.end_idx is not None else local_start + len(value) - 1
    prefix = segment.text[:local_start]
    suffix = segment.text[local_end + 1 :]

    if not structured_relation_text:
        # Reviewed low-reliability mentions and gap mentions must obey the
        # same CMeEE boundary/type contract.
        if entity_type == "bod" and value in _GAP_DIRECTION_ONLY_BODY_TERMS:
            if value == "\u4e0b\u53f6" and prefix.endswith("\u80ba"):
                value = "\u80ba" + value
                local_start -= 1
            else:
                return None
        if entity_type in {"bod", "dru"}:
            indicator = next(
                (item for item in _GAP_TEST_INDICATOR_SUFFIXES if suffix.startswith(item)),
                "",
            )
            if indicator:
                value += indicator
                local_end += len(indicator)
                entity_type = "ite"
                suffix = segment.text[local_end + 1 :]
            elif (
                any(
                    cue in segment.text
                    for cue in ("\u68c0\u67e5", "\u68c0\u6d4b", "\u6307\u6807", "\u590d\u67e5")
                )
                and (
                    value.endswith(_GAP_TEST_ANALYTE_SUFFIXES)
                    or re.fullmatch(r"[A-Z][A-Z0-9-]{1,8}", value)
                )
            ):
                entity_type = "ite"
        if entity_type == "sym" and value == "\u4f11\u514b":
            entity_type = "dis"
        if entity_type == "sym" and any(
            cue in value
            for cue in ("\u6ca1\u6709", "\u672a\u89c1", "\u65e0\u660e\u663e", "\u5426\u8ba4")
        ):
            return None
        if re.fullmatch(r"[A-Za-z0-9-]+", value):
            left = segment.text[local_start - 1 : local_start] if local_start else ""
            right = segment.text[local_end + 1 : local_end + 2]
            if (left and left.isalnum()) or (right and right.isalnum()):
                return None
    if entity_type in {"ite", "equ"} and any(
        cue in value for cue in _TEST_PROCEDURE_CUES
    ):
        entity_type = "pro"
    if entity_type == "sym" and any(
        cue in value for cue in ("\u75c5\u53d8", "\u810f\u5668\u635f\u5bb3")
    ):
        entity_type = "dis"
    if entity_type == "mic" and value.endswith("\u6027"):
        return None
    if structured_relation_text and entity_type != "mic":
        if value.endswith(_GAP_MICROBE_SUFFIXES) and not value.endswith(
            ("\u611f\u67d3", "\u80ba\u708e", "\u75c5")
        ):
            entity_type = "mic"
    if entity_type == "sym" and value.endswith("\u5f81") and value[:-1].endswith(
        ("\u80bf", "\u708e", "\u764c", "\u75c5")
    ):
        value = value[:-1]
        local_end -= 1
        entity_type = "dis"

    if not structured_relation_text and local_start >= 2:
        if segment.text[local_start - 2 : local_start] == "\u5e8a\u8fb9":
            value = "\u5e8a\u8fb9" + value
            local_start -= 2
    if not structured_relation_text and entity_type == "dis" and local_start >= 2:
        if segment.text[local_start - 2 : local_start] == "\u5176\u4ed6":
            value = "\u5176\u4ed6" + value
            local_start -= 2

    if entity_type == "dis" and value.startswith("\u4e00\u822c") and len(value) > 2:
        value = value[2:]
        local_start += 2
    return replace(
        entity,
        text=value,
        type=entity_type,
        start_idx=local_start,
        end_idx=local_end,
    )


def normalize_verified_entity(text: str, entity: Entity) -> Entity | None:
    """Apply one deterministic entity gate to every LLM-approved route."""

    segment = CascadeSegment(
        segment_id="verified-entity",
        start_idx=0,
        end_idx=len(text),
        text=text,
        reasons=("verified_entity",),
    )
    normalized = _normalize_gap_entity(entity, segment)
    if normalized is None:
        return None
    start = normalized.start_idx
    end = normalized.end_idx
    if start is None or end is None:
        located = text.find(normalized.text)
        if located < 0 or text.find(normalized.text, located + 1) >= 0:
            return None
        start, end = located, located + len(normalized.text) - 1
        normalized = replace(normalized, start_idx=start, end_idx=end)
    if not (
        0 <= start <= end < len(text)
        and text[start : end + 1] == normalized.text
    ):
        # LLM outputs such as “胸痛（术后）” must not pass when only “胸痛”
        # occurs in the source.  A verified entity is always an exact span.
        return None
    if not _gap_entity_context_is_supported(normalized, segment, None):
        return None
    return normalized


def _review_prompt(candidates: Iterable[ReviewCandidate]) -> str:
    candidate_payload = [candidate.to_prompt_dict() for candidate in candidates]
    return (
        "You are a conservative medical fact verifier. Review only the supplied candidates; "
        "do not create new facts and do not rewrite any text.\n"
        "Review each offline mention independently. An offline candidate is the baseline: "
        "reject it only when the evidence explicitly contradicts, negates, or clearly "
        "invalidates the candidate. Do not reject a candidate merely because the term is "
        "generic, appears inside a longer mention, or has an ambiguous type; use uncertain "
        "and keep the offline baseline in those cases. A relation decision is independent "
        "from the review decision for either endpoint entity.\n"
        "For a gap candidate, accept means every endpoint and the relation are directly "
        "supported by its evidence; reject or uncertain means do not add it. Be conservative.\n"
        "For a low-reliability offline candidate, accept only when the evidence directly "
        "supports the exact entity or relation. Reject or uncertain means remove it from "
        "the final result; do not preserve a low-reliability candidate merely because it "
        "is plausible.\n"
        "Return JSON only in this form: "
        '{"decisions":[{"candidate_id":"...","decision":"accept|reject|uncertain",'
        '"reason":"short reason","confidence":0.0}]}\n'
        "Use each candidate_id at most once.\n"
        "Candidates:\n"
        + json.dumps(candidate_payload, ensure_ascii=False)
    )


def review_candidates(
    llm: LLMClient,
    candidates: list[ReviewCandidate],
    *,
    batch_size: int = 32,
) -> dict[str, ReviewDecision]:
    """批量复核候选；缺失或无法解析的结果一律视为 uncertain。"""

    decisions: dict[str, ReviewDecision] = {}
    size = max(1, int(batch_size))
    for offset in range(0, len(candidates), size):
        batch = candidates[offset : offset + size]
        payload = llm.chat_json(_review_prompt(batch))
        for item in _as_list(payload, "decisions", "results"):
            candidate_id = str(item.get("candidate_id", "")).strip()
            if not candidate_id or candidate_id not in {c.candidate_id for c in batch}:
                continue
            decisions[candidate_id] = ReviewDecision(
                candidate_id=candidate_id,
                decision=_normalize_decision(item.get("decision")),
                reason=str(item.get("reason", "") or "").strip()[:240],
                confidence=_normalize_confidence(item.get("confidence")),
            )
    return decisions


def review_candidates_parallel(
    llm: LLMClient,
    candidates: list[ReviewCandidate],
    *,
    batch_size: int = 64,
    max_workers: int = 4,
) -> dict[str, ReviewDecision]:
    """Review independent candidate batches concurrently.

    Candidate IDs are unique within the batch cascade, so the batches do not
    share state.  If any request fails, fail closed for the whole review pass;
    the caller can then keep the offline baseline and expose the error instead
    of presenting a partial model result as complete.
    """

    if not candidates:
        return {}
    size = max(1, int(batch_size))
    batches = [
        candidates[offset : offset + size]
        for offset in range(0, len(candidates), size)
    ]
    if len(batches) == 1:
        return review_candidates(llm, batches[0], batch_size=len(batches[0]))

    worker_count = max(1, min(int(max_workers), len(batches)))
    decisions: dict[str, ReviewDecision] = {}
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                review_candidates,
                llm,
                batch,
                batch_size=len(batch),
            ): batch
            for batch in batches
        }
        for future in as_completed(futures):
            try:
                decisions.update(future.result())
            except BaseException as exc:  # re-raised below with fail-closed semantics
                errors.append(exc)
    if errors:
        raise RuntimeError(
            f"candidate review batch failed: {type(errors[0]).__name__}: "
            f"{str(errors[0])[:200]}"
        ) from errors[0]
    return decisions


def _gap_prompt(segments: list[CascadeSegment]) -> str:
    segment_payload = [
        {
            "segment_id": segment.segment_id,
            "text": segment.text[:600],
            "reasons": segment.reasons,
            "annotation_style": (
                "structured_relation"
                if _is_structured_relation_text(segment.text)
                else "span_ner"
            ),
        }
        for segment in segments
    ]
    entity_types = ", ".join(f"{key}={value}" for key, value in ENTITY_TYPES.items())
    relation_types = ", ".join(RELATION_TYPES)
    return (
        "You extract only directly stated medical facts from the supplied Chinese text segments. "
        "Do not use outside medical knowledge. Preserve exact source strings.\n"
        f"Entity types: {entity_types}\n"
        f"Relation types: {relation_types}\n"
        "Each segment declares an annotation_style. For structured_relation segments "
        "(CMeIE-style text with disease/section markers), enumerate each item introduced "
        "by \u5982, \u5305\u62ec, \u4f8b\u5982, or joined by \u548c/\u3001, and propagate the nearest "
        "explicit relation frame across that list. For example, \u5728\u5173\u8282(\u5982 A \u548c B) "
        "yields separate \u53d1\u75c5\u90e8\u4f4d relations to \u5173\u8282, A, and B; \u6f5c\u5728\u7684\u75c5\u56e0"
        "(\u5982 C\u3001D) yields separate \u75c5\u56e0 relations to C and D. For span_ner "
        "segments (CMeEE-style ordinary sentences), preserve the maximal exact entity span "
        "supported by the wording. Do not split a coordinated examination, treatment, or "
        "symptom phrase into isolated ingredients when the phrase has one shared head such "
        "as \u68c0\u67e5, \u68c0\u6d4b, \u6d4b\u5b9a, \u6d82\u7247, or \u955c\u68c0. Do not output section "
        "headings, age ranges, demographic roles, negated outcomes, adjectives, or single "
        "character fragments as entities. Do not invent a cause entity type; use one of the "
        "listed CMeIE entity types, and keep the relation predicate as \u75c5\u56e0.\n"
        "For every relation, include both endpoints in that segment's entities. "
        "Use an empty list when no fact is directly stated.\n"
        "Return JSON only in this form: "
        '{"segments":[{"segment_id":"s0","entities":[{"text":"...","type":"dis",'
        '"start_idx":0,"end_idx":1,"confidence":0.9}],"relations":[{"subject":"...",'
        '"subject_type":"dis","predicate":"...","object":"...",'
        '"object_type":"sym","confidence":0.9}]}]}\n'
        "Segments:\n"
        + json.dumps(segment_payload, ensure_ascii=False)
    )


def _add_missing_relation_endpoints(raw_entities: Any, raw_relations: Any) -> list[dict[str, Any]]:
    """Make relation endpoints visible to validation when the model omitted them."""

    entities = [item for item in raw_entities if isinstance(item, dict)] if isinstance(raw_entities, list) else []
    relations = raw_relations if isinstance(raw_relations, list) else []
    known_texts = {
        str(item.get("text", item.get("entity", "")) or "").strip()
        for item in entities
    }
    for item in relations:
        if not isinstance(item, dict):
            continue
        predicate = normalize_relation_type(item.get("predicate"))
        object_value = item.get("object", "")
        if isinstance(object_value, dict):
            object_value = object_value.get("@value", object_value.get("value", ""))
        endpoint_specs = (
            ("subject", "subject_type", "dis"),
            ("object", "object_type", "sym"),
        )
        for value_key, type_key, fallback_type in endpoint_specs:
            raw_value = object_value if value_key == "object" else item.get(value_key, "")
            value = str(raw_value or "").strip()
            if not value or value in known_texts:
                continue
            entity_type = normalize_entity_type(item.get(type_key))
            if not entity_type and predicate == "\u53d1\u75c5\u90e8\u4f4d" and value_key == "object":
                entity_type = "bod"
            elif not entity_type and predicate == "\u75c5\u56e0" and value_key == "object":
                entity_type = "sym"
            elif not entity_type:
                entity_type = fallback_type
            if entity_type:
                entities.append({"text": value, "type": entity_type})
                known_texts.add(value)
    return entities


def _parse_gap_facts_payload(
    text: str,
    segments: list[CascadeSegment],
    payload: Any,
    *,
    anchor_entities: list[Entity] | None = None,
) -> tuple[list[Entity], list[Relation]]:
    """Validate one record's portion of a gap response."""

    segment_by_id = {segment.segment_id: segment for segment in segments}
    entities: list[Entity] = []
    relations: list[Relation] = []

    for segment_item in _as_list(payload, "segments", "results"):
        segment_id = str(segment_item.get("segment_id", "")).strip()
        segment = segment_by_id.get(segment_id)
        if segment is None:
            continue
        raw_entities = _add_missing_relation_endpoints(
            segment_item.get("entities", []),
            segment_item.get("relations", []),
        )
        local_entities = validate_entities(segment.text, raw_entities)
        local_relations = validate_relations(
            segment.text,
            segment_item.get("relations", []),
            entities=local_entities,
        )
        local_relations = [
            relation
            for relation in local_relations
            if relation_evidence_supports_pair(
                segment.text,
                relation.predicate,
                relation.subject,
                relation.object,
                subject_type=_local_entity_type(
                    local_entities, relation.subject, relation.subject_type
                ),
                object_type=_local_entity_type(
                    local_entities, relation.object, relation.object_type
                ),
                require_disease_subject=True,
            )
            and _gap_subject_context_is_supported(
                segment.text,
                relation,
                local_entities,
                anchor_entities,
            )
            and _gap_section_subject_is_supported(
                text,
                relation,
                {
                    entity.text
                    for entity in [*(anchor_entities or []), *local_entities]
                    if entity.type == "dis" and entity.text
                },
            )
            and _gap_relation_context_is_supported(
                segment.text, relation, local_entities, anchor_entities
            )
        ]
        accepted_local_entities = [
            normalized_entity
            for entity in local_entities
            if (normalized_entity := _normalize_gap_entity(entity, segment)) is not None
            and _gap_entity_context_is_supported(
                normalized_entity, segment, anchor_entities
            )
        ]
        for entity in accepted_local_entities:
            local_start = entity.start_idx or 0
            local_end = entity.end_idx if entity.end_idx is not None else local_start + len(entity.text) - 1
            left = max(0, segment.start_idx + local_start - 20)
            right = min(len(text), segment.start_idx + local_end + 21)
            entities.append(
                replace(
                    entity,
                    start_idx=segment.start_idx + local_start,
                    end_idx=segment.start_idx + local_end,
                    evidence=text[left:right],
                    extraction_method="llm_gap_candidate",
                    reliability_level="medium",
                )
            )
        for relation in local_relations:
            relations.append(
                replace(
                    relation,
                    evidence=segment.text[:500],
                    extraction_method="llm_gap_candidate",
                    reliability_level="medium",
                )
            )
    return entities, relations


def extract_gap_facts(
    text: str,
    segments: list[CascadeSegment],
    llm: LLMClient,
    *,
    anchor_entities: list[Entity] | None = None,
) -> tuple[list[Entity], list[Relation]]:
    """用一次受限批量请求处理一个记录的缺口句子。"""

    if not segments:
        return [], []
    payload = llm.chat_json(_gap_prompt(segments))
    return _parse_gap_facts_payload(
        text,
        segments,
        payload,
        anchor_entities=anchor_entities,
    )


def extract_gap_facts_batch(
    records: list[tuple[int, str, list[CascadeSegment], list[Entity]]],
    llm: LLMClient,
) -> dict[int, tuple[list[Entity], list[Relation]]]:
    """Extract gap facts for several records in one model request.

    Segment IDs are scoped by record before the prompt is sent.  The response
    is split back into record-local payloads and passed through the same
    validation gates as the single-record path, so batching changes request
    scheduling but not acceptance rules or source offsets.
    """

    result = {record_index: ([], []) for record_index, _, _, _ in records}
    active_records = [item for item in records if item[2]]
    if not active_records:
        return result

    scoped_segments: list[CascadeSegment] = []
    scoped_to_record: dict[str, tuple[int, str]] = {}
    for record_index, _, segments, _ in active_records:
        for segment in segments:
            scoped_id = f"r{record_index}:{segment.segment_id}"
            scoped_segments.append(replace(segment, segment_id=scoped_id))
            scoped_to_record[scoped_id] = (record_index, segment.segment_id)

    payload = llm.chat_json(_gap_prompt(scoped_segments))
    grouped: dict[int, list[dict[str, Any]]] = {
        record_index: [] for record_index, _, _, _ in active_records
    }
    for segment_item in _as_list(payload, "segments", "results"):
        scoped_id = str(segment_item.get("segment_id", "")).strip()
        record_info = scoped_to_record.get(scoped_id)
        if record_info is None:
            continue
        record_index, original_segment_id = record_info
        normalized_item = dict(segment_item)
        normalized_item["segment_id"] = original_segment_id
        grouped[record_index].append(normalized_item)

    for record_index, text, segments, anchor_entities in active_records:
        record_payload = {"segments": grouped.get(record_index, [])}
        result[record_index] = _parse_gap_facts_payload(
            text,
            segments,
            record_payload,
            anchor_entities=anchor_entities,
        )
    return result
