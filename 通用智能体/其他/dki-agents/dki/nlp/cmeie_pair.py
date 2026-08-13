"""Shared CMeIE entity-pair encoding and frozen-checkpoint contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


MARKERS = ("[SUBJ]", "[/SUBJ]", "[OBJ]", "[/OBJ]")
SENTENCE_ENDS = frozenset("。！？；;\n")

PROJECT_TO_CMEIE_TYPE = {
    "Disease": "疾病",
    "Symptom": "症状",
    "Drug": "药物",
    "Exam": "检查",
    "Treatment": "其他治疗",
    "Department": "社会学",
}

CMEEE_TO_CMEIE_TYPE = {
    "dis": "疾病",
    "sym": "症状",
    "dru": "药物",
    "equ": "检查",
    "pro": "其他治疗",
    "bod": "部位",
    "ite": "检查",
    "mic": "其他",
    "dep": "社会学",
}


def normalize_entity_type(entity_type: str) -> str:
    value = str(entity_type or "").strip()
    return PROJECT_TO_CMEIE_TYPE.get(
        value,
        CMEEE_TO_CMEIE_TYPE.get(value, value or "未知"),
    )


def signature_text(subject_type: str, object_type: str) -> str:
    return (
        f"主体类型={normalize_entity_type(subject_type)}，"
        f"客体类型={normalize_entity_type(object_type)}"
    )


def _occurrences(text: str, mention: str) -> list[Tuple[int, int]]:
    if not mention:
        return []
    spans: list[Tuple[int, int]] = []
    start = 0
    while True:
        index = text.find(mention, start)
        if index < 0:
            return spans
        spans.append((index, index + len(mention)))
        start = index + 1


def choose_pair_spans(
    text: str,
    subject: str,
    obj: str,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Choose the closest deterministic non-overlapping mention pair."""
    candidates = []
    for subject_span in _occurrences(text, subject):
        for object_span in _occurrences(text, obj):
            if subject_span[0] < object_span[1] and object_span[0] < subject_span[1]:
                continue
            subject_sentence = sum(
                character in SENTENCE_ENDS for character in text[:subject_span[0]]
            )
            object_sentence = sum(
                character in SENTENCE_ENDS for character in text[:object_span[0]]
            )
            if subject_sentence != object_sentence:
                continue
            distance = abs(
                (subject_span[0] + subject_span[1])
                - (object_span[0] + object_span[1])
            )
            candidates.append(
                (distance, subject_span[0], object_span[0], subject_span, object_span)
            )
    if not candidates:
        return None
    return min(candidates)[-2:]


def _valid_span(text: str, span: Optional[Sequence[int]], mention: str) -> bool:
    if span is None or len(span) != 2:
        return False
    start, end = int(span[0]), int(span[1])
    return 0 <= start < end <= len(text) and text[start:end] == mention


def mark_pair(
    text: str,
    subject: str,
    obj: str,
    *,
    subject_span: Optional[Sequence[int]] = None,
    object_span: Optional[Sequence[int]] = None,
) -> Optional[str]:
    """Insert directional markers at the supplied or closest real mentions."""
    if _valid_span(text, subject_span, subject) and _valid_span(text, object_span, obj):
        chosen = (
            (int(subject_span[0]), int(subject_span[1])),
            (int(object_span[0]), int(object_span[1])),
        )
        if chosen[0][0] < chosen[1][1] and chosen[1][0] < chosen[0][1]:
            return None
    else:
        chosen = choose_pair_spans(text, subject, obj)
    if chosen is None:
        return None

    subject_pos, object_pos = chosen
    insertions = [
        (subject_pos[0], 1, MARKERS[0]),
        (subject_pos[1], 0, MARKERS[1]),
        (object_pos[0], 1, MARKERS[2]),
        (object_pos[1], 0, MARKERS[3]),
    ]
    marked = text
    for position, _, marker in sorted(insertions, reverse=True):
        marked = marked[:position] + marker + marked[position:]
    return marked


def load_relation_contract(model_path: Path) -> Dict[str, Any]:
    path = Path(model_path) / "relation_config.json"
    if not path.is_file():
        legacy = Path(model_path) / "re_config.json"
        path = legacy if legacy.is_file() else path
    if not path.is_file():
        raise FileNotFoundError(f"CMeIE checkpoint 缺少 relation_config.json: {model_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("multi_label", True):
        raise ValueError("CMeIE checkpoint 不是多标签关系分类模型")
    return payload


def allowed_label_ids(
    contract: Dict[str, Any],
    subject_type: str,
    object_type: str,
    n_labels: int,
) -> set[int]:
    constraints = contract.get("schema_constraints") or {}
    if not constraints:
        return set(range(n_labels))
    key = (
        f"{normalize_entity_type(subject_type)}\t"
        f"{normalize_entity_type(object_type)}"
    )
    if key not in constraints:
        return set()
    labels: Iterable[Any] = constraints[key]
    allowed = {
        int(label)
        for label in labels
        if isinstance(label, int) or str(label).isdigit()
    }
    return {label for label in allowed if 0 <= label < n_labels}
