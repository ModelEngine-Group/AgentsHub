"""CMeIE direct-SPO label alignment, decoding and exact scoring helpers."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import torch


SENTENCE_BREAKS = frozenset("。！？!?\n")


def _validate_entity_label_mode(entity_label_mode: str) -> None:
    if entity_label_mode not in {"typed", "role_only"}:
        raise ValueError(
            "entity_label_mode must be either 'typed' or 'role_only'"
        )


def labels_from_fit(
    rows: list[dict[str, Any]],
    *,
    entity_label_mode: str = "typed",
) -> tuple[list[str], list[str]]:
    _validate_entity_label_mode(entity_label_mode)
    entity = set()
    predicates = set()
    for row in rows:
        for spo in row.get("spo_list", []):
            if spo.get("subject"):
                if entity_label_mode == "role_only":
                    entity.add("subject:*")
                elif spo.get("subject_type"):
                    entity.add(f"subject:{spo['subject_type']}")
            if spo.get("object"):
                if entity_label_mode == "role_only":
                    entity.add("object:*")
                elif spo.get("object_type"):
                    entity.add(f"object:{spo['object_type']}")
            if spo.get("predicate"):
                predicates.add(str(spo["predicate"]))
    return sorted(entity), sorted(predicates)


def schema_constraints_from_fit(
    rows: list[dict[str, Any]],
    *,
    entity_label_mode: str = "typed",
) -> dict[str, list[tuple[str, str]]]:
    """Relation type signatures learned exclusively from the fit split."""
    _validate_entity_label_mode(entity_label_mode)
    constraints: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        for spo in row.get("spo_list", []):
            predicate = str(spo.get("predicate", ""))
            if entity_label_mode == "role_only":
                subject = "subject:*" if spo.get("subject") else "subject:"
                obj = "object:*" if spo.get("object") else "object:"
            else:
                subject = f"subject:{spo.get('subject_type', '')}"
                obj = f"object:{spo.get('object_type', '')}"
            if predicate and subject != "subject:" and obj != "object:":
                constraints.setdefault(predicate, set()).add((subject, obj))
    return {predicate: sorted(values) for predicate, values in sorted(constraints.items())}


def all_occurrences(text: str, term: str) -> list[tuple[int, int]]:
    if not term:
        return []
    result, start = [], 0
    while True:
        index = text.find(term, start)
        if index < 0:
            return result
        result.append((index, index + len(term)))
        start = index + 1


def same_sentence(text: str, left: tuple[int, int], right: tuple[int, int]) -> bool:
    return not any(
        char in SENTENCE_BREAKS
        for char in text[min(left[1], right[1]) : max(left[0], right[0])]
    )


def _span_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] <= right[0]:
        return right[0] - left[1]
    if right[1] <= left[0]:
        return left[0] - right[1]
    return 0


def _mention_pair_sort_key(
    pair: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, int, int, int, int, int]:
    return (
        _span_distance(pair[0], pair[1]),
        abs(pair[0][0] - pair[1][0]),
        pair[0][0],
        pair[1][0],
        pair[0][1],
        pair[1][1],
    )


def _minimum_cost_monotonic_edge_cover(
    subject_spans: list[tuple[int, int]],
    object_spans: list[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Cover ordered mentions with the least-cost non-crossing links.

    A path from the first subject/object pair to the last may advance the
    subject, the object, or both.  Consequently every occurrence is covered,
    unequal mention counts remain representable, and an aligned repetition
    such as ``甲…乙，甲…乙`` cannot gain the spurious middle cross-link produced
    by taking the union of independent nearest neighbours.
    """

    if not subject_spans or not object_spans:
        return []
    subjects = sorted(subject_spans)
    objects = sorted(object_spans)
    states: dict[
        tuple[int, int],
        tuple[
            tuple[int, int, int, tuple[tuple[tuple[int, int], tuple[int, int]], ...]],
            tuple[tuple[tuple[int, int], tuple[int, int]], ...],
        ],
    ] = {}
    for subject_index, subject_span in enumerate(subjects):
        for object_index, object_span in enumerate(objects):
            pair = (subject_span, object_span)
            predecessors = (
                (subject_index - 1, object_index - 1),
                (subject_index - 1, object_index),
                (subject_index, object_index - 1),
            )
            candidate_paths = [
                states[index][1] + (pair,)
                for index in predecessors
                if index in states
            ]
            if subject_index == 0 and object_index == 0:
                candidate_paths.append((pair,))
            ranked = []
            for path in candidate_paths:
                ranked.append(
                    (
                        (
                            sum(_span_distance(left, right) for left, right in path),
                            sum(abs(left[0] - right[0]) for left, right in path),
                            len(path),
                            path,
                        ),
                        path,
                    )
                )
            states[(subject_index, object_index)] = min(
                ranked,
                key=lambda item: item[0],
            )
    return list(states[(len(subjects) - 1, len(objects) - 1)][1])


def select_mention_pairs(
    text: str,
    subject: str,
    obj: str,
    *,
    allow_cross_sentence_fallback: bool = False,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return deterministic nearest links covering all same-sentence mentions.

    CMeIE supplies surface strings but no offsets.  Treating every Cartesian
    product of repeated mentions as positive invents relations, while selecting
    only one pair makes the other valid occurrences false negatives.  For each
    subject occurrence and each object occurrence, keep its nearest
    same-sentence partner and take the deterministic union.  This covers every
    legal repeated mention without adding distant cross-links.

    A corpus record with no same-sentence pair may opt into one closest
    cross-sentence fallback.  The fallback remains singular so it cannot create
    a Cartesian set of unsupported long-distance relations.
    """

    subject_spans = all_occurrences(text, subject)
    object_spans = all_occurrences(text, obj)
    if not subject_spans or not object_spans:
        return []
    all_pairs = [
        (subject_span, object_span)
        for subject_span in subject_spans
        for object_span in object_spans
    ]
    same_sentence_pairs = [
        pair for pair in all_pairs if same_sentence(text, pair[0], pair[1])
    ]
    if not same_sentence_pairs:
        if not allow_cross_sentence_fallback:
            return []
        return [min(all_pairs, key=_mention_pair_sort_key)]

    def sentence_index(span: tuple[int, int]) -> int:
        return sum(char in SENTENCE_BREAKS for char in text[: span[0]])

    subjects_by_sentence: dict[int, list[tuple[int, int]]] = defaultdict(list)
    objects_by_sentence: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for span in subject_spans:
        subjects_by_sentence[sentence_index(span)].append(span)
    for span in object_spans:
        objects_by_sentence[sentence_index(span)].append(span)

    selected = set()
    for index in sorted(subjects_by_sentence.keys() & objects_by_sentence.keys()):
        selected.update(
            _minimum_cost_monotonic_edge_cover(
                subjects_by_sentence[index],
                objects_by_sentence[index],
            )
        )
    return sorted(selected, key=_mention_pair_sort_key)


def select_mention_pair(
    text: str,
    subject: str,
    obj: str,
    *,
    allow_cross_sentence_fallback: bool = False,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Choose one deterministic mention pair for an SPO surface form.

    CMeIE supplies surface strings rather than annotated character offsets.
    Same-sentence pairs are preferred and the closest pair is selected instead
    of labelling every Cartesian-product combination.  The direct joint model
    may enable the cross-sentence fallback because the official corpus contains
    many valid long-distance gold relations; candidate-pair product paths keep
    the stricter default.
    """

    pairs = select_mention_pairs(
        text,
        subject,
        obj,
        allow_cross_sentence_fallback=allow_cross_sentence_fallback,
    )
    return pairs[0] if pairs else None


def char_span_to_tokens(offsets: list[tuple[int, int]], span: tuple[int, int]) -> tuple[int, int] | None:
    """Map a source character span to the first/last overlapping tokens.

    Fast BERT tokenizers intentionally omit whitespace and some control
    characters from ``offset_mapping``.  Requiring the exact first and last
    character to be owned by a token therefore drops otherwise representable
    mentions.  Overlap mapping preserves the source slice while still refusing
    spans with no represented token at all.
    """

    start, end = span
    if start < 0 or end <= start:
        return None
    token_indices = []
    for index, (left, right) in enumerate(offsets):
        if right <= left:
            continue
        if right > start and left < end:
            token_indices.append(index)
    if not token_indices:
        return None
    return token_indices[0], token_indices[-1]


def build_labels(
    row: dict[str, Any],
    offsets: list[tuple[int, int]],
    entity2id: dict[str, int],
    predicate2id: dict[str, int],
    *,
    allow_cross_sentence_fallback: bool = False,
) -> tuple[dict[str, torch.Tensor], int]:
    """Build dense GPLinker labels; returns labels and non-alignable SPO count."""
    seq = len(offsets)
    labels = {
        "entity": torch.zeros((len(entity2id), seq, seq), dtype=torch.float32),
        "relation_head": torch.zeros((len(predicate2id), seq, seq), dtype=torch.float32),
        "relation_tail": torch.zeros((len(predicate2id), seq, seq), dtype=torch.float32),
    }
    dropped = 0
    text = str(row.get("text", ""))
    for spo in row.get("spo_list", []):
        subject, obj = str(spo.get("subject", "")), str(spo.get("object", ""))
        typed_subject_label = f"subject:{spo.get('subject_type', '')}"
        typed_object_label = f"object:{spo.get('object_type', '')}"
        subject_label = (
            typed_subject_label
            if typed_subject_label in entity2id
            else "subject:*"
            if "subject:*" in entity2id
            else None
        )
        object_label = (
            typed_object_label
            if typed_object_label in entity2id
            else "object:*"
            if "object:*" in entity2id
            else None
        )
        predicate = str(spo.get("predicate", ""))
        if (
            subject_label is None
            or object_label is None
            or predicate not in predicate2id
        ):
            dropped += 1
            continue
        pairs = select_mention_pairs(
            text,
            subject,
            obj,
            allow_cross_sentence_fallback=allow_cross_sentence_fallback,
        )
        if not pairs:
            dropped += 1
            continue
        aligned = 0
        for pair in pairs:
            subject_tokens = char_span_to_tokens(offsets, pair[0])
            object_tokens = char_span_to_tokens(offsets, pair[1])
            if subject_tokens is None or object_tokens is None:
                continue
            ss, se = subject_tokens
            os, oe = object_tokens
            labels["entity"][entity2id[subject_label], ss, se] = 1
            labels["entity"][entity2id[object_label], os, oe] = 1
            labels["relation_head"][predicate2id[predicate], ss, os] = 1
            labels["relation_tail"][predicate2id[predicate], se, oe] = 1
            aligned += 1
        if not aligned:
            dropped += 1
    return labels, dropped


def gold_spo_set(row: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {(str(spo.get("subject", "")), str(spo.get("predicate", "")), str(spo.get("object", ""))) for spo in row.get("spo_list", [])}


def _entity_maps(
    text: str,
    offsets: list[tuple[int, int]],
    entity_logits: torch.Tensor,
    entity_labels: list[str],
    threshold: float,
) -> tuple[
    dict[int, list[tuple[str, int, str]]],
    dict[int, list[tuple[str, int, str]]],
]:
    subject_by_start: dict[int, list[tuple[str, int, str]]] = defaultdict(list)
    object_by_start: dict[int, list[tuple[str, int, str]]] = defaultdict(list)
    for label_id, start, end in (
        entity_logits > threshold
    ).nonzero(as_tuple=False).tolist():
        if (
            start > end
            or offsets[start][1] <= offsets[start][0]
            or offsets[end][1] <= offsets[end][0]
        ):
            continue
        entity_label = entity_labels[label_id]
        role, _ = entity_label.split(":", 1)
        value = text[offsets[start][0] : offsets[end][1]]
        if not value:
            continue
        if role == "subject":
            subject_by_start[start].append((entity_label, end, value))
        elif role == "object":
            object_by_start[start].append((entity_label, end, value))
    return subject_by_start, object_by_start


def _join_pointer_predictions(
    subject_by_start: dict[int, list[tuple[str, int, str]]],
    object_by_start: dict[int, list[tuple[str, int, str]]],
    heads: set[tuple[int, int, int]],
    tails: set[tuple[int, int, int]],
    predicates: list[str],
    allowed: dict[str, set[tuple[str, str]]] | None,
) -> set[tuple[str, str, str]]:
    predicted = set()
    for predicate_id, ss, os in heads:
        subjects = subject_by_start.get(ss)
        objects = object_by_start.get(os)
        if not subjects or not objects:
            continue
        predicate = predicates[predicate_id]
        for subject_label, se, subject in subjects:
            for object_label, oe, obj in objects:
                if (predicate_id, se, oe) not in tails:
                    continue
                if allowed is not None and (
                    subject_label,
                    object_label,
                ) not in allowed.get(predicate, set()):
                    continue
                predicted.add((subject, predicate, obj))
    return predicted


def decode_spo(
    text: str,
    offsets: list[tuple[int, int]],
    entity_logits: torch.Tensor,
    head_logits: torch.Tensor,
    tail_logits: torch.Tensor,
    entity_labels: list[str],
    predicates: list[str],
    thresholds: tuple[float, float, float],
    schema_constraints: dict[str, list[tuple[str, str]]] | None = None,
) -> set[tuple[str, str, str]]:
    entity_threshold, head_threshold, tail_threshold = thresholds
    subject_by_start, object_by_start = _entity_maps(
        text, offsets, entity_logits, entity_labels, entity_threshold
    )
    heads = {(p, s, o) for p, s, o in (head_logits > head_threshold).nonzero(as_tuple=False).tolist()}
    tails = {(p, s, o) for p, s, o in (tail_logits > tail_threshold).nonzero(as_tuple=False).tolist()}
    allowed = (
        {
            predicate: set(signatures)
            for predicate, signatures in schema_constraints.items()
        }
        if schema_constraints is not None
        else None
    )
    return _join_pointer_predictions(
        subject_by_start,
        object_by_start,
        heads,
        tails,
        predicates,
        allowed,
    )


def decode_spo_grid(
    text: str,
    offsets: list[tuple[int, int]],
    entity_logits: torch.Tensor,
    head_logits: torch.Tensor,
    tail_logits: torch.Tensor,
    entity_labels: list[str],
    predicates: list[str],
    threshold_values: tuple[float, ...],
    schema_constraints: dict[str, list[tuple[str, str]]],
) -> dict[tuple[tuple[float, float, float], bool], set[tuple[str, str, str]]]:
    """Decode the declared threshold grid from dense logits.

    Dense tensors are converted to exact candidates above the lowest declared
    threshold, then delegated to :func:`decode_spo_grid_sparse`.  Training and
    remote evaluation should sparsify on the accelerator before transferring to
    CPU; this compatibility wrapper keeps the public decoder/test API stable.
    """
    threshold_values = tuple(sorted(threshold_values))
    lowest = threshold_values[0]
    return decode_spo_grid_sparse(
        text,
        offsets,
        sparsify_pointer_logits(entity_logits, lowest),
        sparsify_pointer_logits(head_logits, lowest),
        sparsify_pointer_logits(tail_logits, lowest),
        entity_labels,
        predicates,
        threshold_values,
        schema_constraints,
    )


def sparsify_pointer_logits(
    logits: torch.Tensor,
    minimum_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact indices/scores above ``minimum_threshold``.

    The comparison and gather execute on the input device.  Only the sparse
    candidate tensors cross to CPU, avoiding transfer of three dense
    ``labels × sequence × sequence`` heads for every calibration row.
    """
    indices = (logits > minimum_threshold).nonzero(as_tuple=False)
    if indices.numel():
        scores = logits[tuple(indices.transpose(0, 1))]
    else:
        scores = logits.new_empty((0,))
    return indices.cpu(), scores.float().cpu()


def sparse_join_match_lower_bound(
    offsets: list[tuple[int, int]],
    entity_candidates: tuple[torch.Tensor, torch.Tensor],
    head_candidates: tuple[torch.Tensor, torch.Tensor],
    tail_candidates: tuple[torch.Tensor, torch.Tensor],
    entity_labels: list[str],
    thresholds: tuple[float, float, float],
    cap: int,
) -> dict[str, int | bool]:
    """Count four-pointer span joins up to a deterministic safety cap.

    This is a lower bound on work, not a metric and not a prediction.  It lets
    early, poorly calibrated checkpoints defer exact grid enumeration without
    silently dropping any candidate or reading another data split.
    """
    if cap < 1:
        raise ValueError("join safety cap must be positive")
    entity_threshold, head_threshold, tail_threshold = thresholds
    subject_by_start: dict[int, list[int]] = defaultdict(list)
    object_by_span: dict[tuple[int, int], int] = defaultdict(int)
    entity_indices, entity_scores = entity_candidates
    for (label_id, start, end), score in zip(
        entity_indices.tolist(),
        entity_scores.tolist(),
    ):
        if (
            score <= entity_threshold
            or start > end
            or offsets[start][1] <= offsets[start][0]
            or offsets[end][1] <= offsets[end][0]
        ):
            continue
        role, _ = entity_labels[label_id].split(":", 1)
        if role == "subject":
            subject_by_start[start].append(end)
        elif role == "object":
            object_by_span[(start, end)] += 1

    tail_by_subject_end: dict[tuple[int, int], list[int]] = defaultdict(list)
    tail_indices, tail_scores = tail_candidates
    for (predicate_id, subject_end, object_end), score in zip(
        tail_indices.tolist(),
        tail_scores.tolist(),
    ):
        if score > tail_threshold:
            tail_by_subject_end[(predicate_id, subject_end)].append(object_end)

    matches = 0
    head_indices, head_scores = head_candidates
    for (predicate_id, subject_start, object_start), head_score in zip(
        head_indices.tolist(),
        head_scores.tolist(),
    ):
        if head_score <= head_threshold:
            continue
        for subject_end in subject_by_start.get(subject_start, ()):
            for object_end in tail_by_subject_end.get(
                (predicate_id, subject_end),
                (),
            ):
                matches += object_by_span.get((object_start, object_end), 0)
                if matches >= cap:
                    return {
                        "match_lower_bound": matches,
                        "cap": cap,
                        "capped": True,
                    }
    return {
        "match_lower_bound": matches,
        "cap": cap,
        "capped": False,
    }


def _sparse_entity_maps_grid(
    text: str,
    offsets: list[tuple[int, int]],
    candidates: tuple[torch.Tensor, torch.Tensor],
    entity_labels: list[str],
    threshold_values: tuple[float, ...],
) -> dict[
    float,
    tuple[
        dict[int, list[tuple[str, int, str]]],
        dict[tuple[int, int], list[tuple[str, str]]],
    ],
]:
    entities = {
        threshold: (defaultdict(list), defaultdict(list))
        for threshold in threshold_values
    }
    indices, scores = candidates
    for (label_id, start, end), score in zip(
        indices.tolist(),
        scores.tolist(),
    ):
        if (
            start > end
            or offsets[start][1] <= offsets[start][0]
            or offsets[end][1] <= offsets[end][0]
        ):
            continue
        entity_label = entity_labels[label_id]
        role, _ = entity_label.split(":", 1)
        value = text[offsets[start][0] : offsets[end][1]]
        if not value:
            continue
        for threshold in threshold_values:
            if score <= threshold:
                continue
            subject_by_start, object_by_span = entities[threshold]
            if role == "subject":
                subject_by_start[start].append(
                    (entity_label, end, value)
                )
            elif role == "object":
                object_by_span[(start, end)].append(
                    (entity_label, value)
                )
    return entities


def decode_spo_grid_sparse(
    text: str,
    offsets: list[tuple[int, int]],
    entity_candidates: tuple[torch.Tensor, torch.Tensor],
    head_candidates: tuple[torch.Tensor, torch.Tensor],
    tail_candidates: tuple[torch.Tensor, torch.Tensor],
    entity_labels: list[str],
    predicates: list[str],
    threshold_values: tuple[float, ...],
    schema_constraints: dict[str, list[tuple[str, str]]],
) -> dict[tuple[tuple[float, float, float], bool], set[tuple[str, str, str]]]:
    """Decode the declared symmetric grid from exact sparse candidates."""
    return decode_spo_grid_sparse_axes(
        text,
        offsets,
        entity_candidates,
        head_candidates,
        tail_candidates,
        entity_labels,
        predicates,
        threshold_values,
        threshold_values,
        threshold_values,
        schema_constraints,
    )


def decode_spo_grid_sparse_axes(
    text: str,
    offsets: list[tuple[int, int]],
    entity_candidates: tuple[torch.Tensor, torch.Tensor],
    head_candidates: tuple[torch.Tensor, torch.Tensor],
    tail_candidates: tuple[torch.Tensor, torch.Tensor],
    entity_labels: list[str],
    predicates: list[str],
    entity_threshold_values: tuple[float, ...],
    head_threshold_values: tuple[float, ...],
    tail_threshold_values: tuple[float, ...],
    schema_constraints: dict[str, list[tuple[str, str]]],
) -> dict[tuple[tuple[float, float, float], bool], set[tuple[str, str, str]]]:
    """Decode an exact Cartesian grid with independent threshold axes."""
    entity_threshold_values = tuple(sorted(entity_threshold_values))
    head_threshold_values = tuple(sorted(head_threshold_values))
    tail_threshold_values = tuple(sorted(tail_threshold_values))
    entities = _sparse_entity_maps_grid(
        text,
        offsets,
        entity_candidates,
        entity_labels,
        entity_threshold_values,
    )
    allowed = {
        predicate: set(signatures)
        for predicate, signatures in schema_constraints.items()
    }
    head_indices, head_scores = head_candidates
    tail_indices, tail_scores = tail_candidates
    tail_by_subject_end: dict[
        tuple[int, int],
        list[tuple[int, float]],
    ] = defaultdict(list)
    for (predicate_id, subject_end, object_end), score in zip(
        tail_indices.tolist(),
        tail_scores.tolist(),
    ):
        tail_by_subject_end[(predicate_id, subject_end)].append(
            (object_end, float(score))
        )
    decoded = {
        (
            (entity_threshold, head_threshold, tail_threshold),
            schema_on,
        ): set()
        for entity_threshold in entity_threshold_values
        for head_threshold in head_threshold_values
        for tail_threshold in tail_threshold_values
        for schema_on in (False, True)
    }
    for entity_threshold in entity_threshold_values:
        subject_by_start, object_by_span = entities[entity_threshold]
        if not subject_by_start or not object_by_span:
            continue
        for (predicate_id, ss, os), head_score in zip(
            head_indices.tolist(),
            head_scores.tolist(),
        ):
            subjects = subject_by_start.get(ss)
            if not subjects:
                continue
            passed_head_thresholds = [
                value for value in head_threshold_values if head_score > value
            ]
            if not passed_head_thresholds:
                continue
            predicate = predicates[predicate_id]
            for subject_label, se, subject in subjects:
                for oe, tail_score in tail_by_subject_end.get(
                    (predicate_id, se),
                    (),
                ):
                    passed_tail_thresholds = [
                        value
                        for value in tail_threshold_values
                        if tail_score > value
                    ]
                    if not passed_tail_thresholds:
                        continue
                    for object_label, obj in object_by_span.get(
                        (os, oe),
                        (),
                    ):
                        triple = (subject, predicate, obj)
                        schema_allowed = (
                            subject_label,
                            object_label,
                        ) in allowed.get(predicate, set())
                        for head_threshold in passed_head_thresholds:
                            for tail_threshold in passed_tail_thresholds:
                                thresholds = (
                                    entity_threshold,
                                    head_threshold,
                                    tail_threshold,
                                )
                                decoded[(thresholds, False)].add(triple)
                                if schema_allowed:
                                    decoded[(thresholds, True)].add(triple)
    return decoded


def exact_micro_f1(predicted: list[set[tuple[str, str, str]]], gold: list[set[tuple[str, str, str]]]) -> dict[str, float | int]:
    tp = sum(len(p & g) for p, g in zip(predicted, gold))
    fp = sum(len(p - g) for p, g in zip(predicted, gold))
    fn = sum(len(g - p) for p, g in zip(predicted, gold))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall)}
