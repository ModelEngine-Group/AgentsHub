"""Deterministic entity-level evaluation utilities.

New evaluations use exact character span plus entity type as their primary metric.
Relaxed overlap is retained only as an explicitly named diagnostic.  Character
coordinates are Python ``[start, end)`` code-point offsets.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

Span = Tuple[int, int, str]
ENTITY_TYPES = ("Disease", "Symptom", "Drug", "Exam", "Treatment", "Department")


class SpanValidationError(ValueError):
    """Raised when an evaluation record does not contain an unambiguous span."""


@dataclass(frozen=True, order=True)
class EntitySpan:
    start: int
    end: int
    entity_type: str

    def as_tuple(self) -> Span:
        return self.start, self.end, self.entity_type


def validate_span(span: Sequence[object], text: str | None = None) -> Tuple[int, int]:
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise SpanValidationError(f"span must be [start, end), got {span!r}")
    if isinstance(span[0], bool) or isinstance(span[1], bool):
        raise SpanValidationError(f"span offsets must be integers, got {span!r}")
    try:
        start, end = int(span[0]), int(span[1])
    except (TypeError, ValueError) as exc:
        raise SpanValidationError(f"span offsets must be integers, got {span!r}") from exc
    if start < 0 or end <= start:
        raise SpanValidationError(f"invalid half-open span [{start}, {end})")
    if text is not None and end > len(text):
        raise SpanValidationError(
            f"span [{start}, {end}) exceeds text length {len(text)}"
        )
    return start, end


def explicit_spans(text: str, entities: Iterable[Mapping[str, object]]) -> List[Span]:
    """Read strict gold/prediction spans without guessing from a mention.

    Mention-only records are intentionally rejected: locating the first textual
    occurrence silently changes the denominator when a mention repeats.
    """
    out: List[Span] = []
    for index, entity in enumerate(entities):
        if "span" in entity and entity.get("span") is not None:
            raw_span = entity["span"]
        elif "start" in entity and "end" in entity:
            raw_span = [entity["start"], entity["end"]]
        else:
            raise SpanValidationError(f"entity {index} has no explicit span")
        start, end = validate_span(raw_span, text)
        entity_type = entity.get("type")
        if not isinstance(entity_type, str) or not entity_type:
            raise SpanValidationError(f"entity {index} has no valid type")
        mention = entity.get("mention")
        if mention is not None and text[start:end] != mention:
            raise SpanValidationError(
                f"entity {index} mention {mention!r} != text[{start}:{end}] {text[start:end]!r}"
            )
        out.append((start, end, entity_type))
    return out


def _overlap(left: Span, right: Span) -> bool:
    return left[2] == right[2] and not (left[1] <= right[0] or left[0] >= right[1])


def _exact(left: Span, right: Span) -> bool:
    return left == right


def _maximum_matching(
    predictions: Sequence[Span],
    gold: Sequence[Span],
    compatible: Callable[[Span, Span], bool],
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Return deterministic maximum-cardinality bipartite matching.

    Sorting canonicalizes input order.  A standard augmenting-path matcher then
    guarantees maximum cardinality rather than the order-dependent greedy count
    used by the historical evaluator.
    """
    pred_order = sorted(range(len(predictions)), key=lambda i: (predictions[i], i))
    gold_order = sorted(range(len(gold)), key=lambda i: (gold[i], i))
    adjacency = {
        pi: [gi for gi in gold_order if compatible(predictions[pi], gold[gi])]
        for pi in pred_order
    }
    gold_to_pred: Dict[int, int] = {}

    def augment(pred_index: int, seen: set[int]) -> bool:
        for gold_index in adjacency[pred_index]:
            if gold_index in seen:
                continue
            seen.add(gold_index)
            incumbent = gold_to_pred.get(gold_index)
            if incumbent is None or augment(incumbent, seen):
                gold_to_pred[gold_index] = pred_index
                return True
        return False

    for pred_index in pred_order:
        augment(pred_index, set())
    pairs = sorted((pred, gold_index) for gold_index, pred in gold_to_pred.items())
    matched_pred = {pred for pred, _ in pairs}
    matched_gold = set(gold_to_pred)
    return (
        pairs,
        [i for i in range(len(predictions)) if i not in matched_pred],
        [i for i in range(len(gold)) if i not in matched_gold],
    )


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
    }


def evaluate_spans(
    pred_by_doc: Mapping[str, Sequence[Span]],
    gold_by_doc: Mapping[str, Sequence[Span]],
    *,
    match: str = "exact",
) -> Dict[str, object]:
    """Evaluate exact or relaxed spans with micro and type-safety metrics."""
    if match not in {"exact", "overlap"}:
        raise ValueError("match must be 'exact' or 'overlap'")
    compatible = _exact if match == "exact" else _overlap
    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0])
    all_doc_ids = sorted(set(pred_by_doc) | set(gold_by_doc))
    for doc_id in all_doc_ids:
        pred = [tuple(item) for item in pred_by_doc.get(doc_id, ())]
        gold = [tuple(item) for item in gold_by_doc.get(doc_id, ())]
        pairs, unmatched_pred, unmatched_gold = _maximum_matching(pred, gold, compatible)
        for pred_index, _ in pairs:
            counts[pred[pred_index][2]][0] += 1
        for pred_index in unmatched_pred:
            counts[pred[pred_index][2]][1] += 1
        for gold_index in unmatched_gold:
            counts[gold[gold_index][2]][2] += 1

    observed_types = sorted(set(ENTITY_TYPES) | set(counts))
    by_type = {
        entity_type: _prf(*counts[entity_type]) for entity_type in observed_types
    }
    tp = sum(value[0] for value in counts.values())
    fp = sum(value[1] for value in counts.values())
    fn = sum(value[2] for value in counts.values())
    micro = _prf(tp, fp, fn)
    supported = [metric for metric in by_type.values() if metric["support"] > 0]
    macro_f1 = sum(float(metric["f1"]) for metric in supported) / len(supported) if supported else 0.0
    worst = min(supported, key=lambda metric: float(metric["f1"])) if supported else None
    worst_types = [
        entity_type for entity_type, metric in by_type.items()
        if worst is not None and metric["support"] > 0 and metric["f1"] == worst["f1"]
    ]
    return {
        **micro,
        "match": "exact_span+type" if match == "exact" else "relaxed_overlap+type",
        "macro_f1": round(macro_f1, 4),
        "worst_type_f1": None if worst is None else worst["f1"],
        "worst_types": worst_types,
        "by_type": by_type,
    }


def evaluate_both(
    pred_by_doc: Mapping[str, Sequence[Span]],
    gold_by_doc: Mapping[str, Sequence[Span]],
) -> Dict[str, object]:
    return {
        "primary_exact": evaluate_spans(pred_by_doc, gold_by_doc, match="exact"),
        "relaxed_diagnostic": evaluate_spans(pred_by_doc, gold_by_doc, match="overlap"),
    }
