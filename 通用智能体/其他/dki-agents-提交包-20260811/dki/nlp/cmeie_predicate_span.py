"""Predicate-conditioned subject/object spans plus one directional link head.

Unlike the earlier GPLinker candidate, entity boundaries are learned inside
each predicate channel.  The link head only has to disambiguate repeated
subject/object mentions; it is not also responsible for recovering both ends
of each entity span.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dki.nlp.cmeie_gplinker import DirectionalGlobalPointer
from dki.nlp.cmeie_spo import char_span_to_tokens, same_sentence, select_mention_pairs
from dki.nlp.globalpointer import GlobalPointer


LOSS_MODE_HARD_NEGATIVE_BCE = "hard_negative_bce"
LOSS_MODE_BALANCED_PER_HEAD_SPARSE_LOGSUMEXP = (
    "balanced_per_head_sparse_logsumexp"
)
PREDICATE_SPAN_LOSS_MODES = (
    LOSS_MODE_HARD_NEGATIVE_BCE,
    LOSS_MODE_BALANCED_PER_HEAD_SPARSE_LOGSUMEXP,
)


@dataclass(frozen=True)
class ThresholdDensityRejection:
    """One unsafe head/threshold pair for a single decoded example."""

    head: str
    threshold: float
    active_candidates: int


@dataclass(frozen=True)
class ThresholdCombinationDensityRejection:
    """One unsafe threshold combination before its Cartesian join runs."""

    thresholds: tuple[float, float, float]
    join_work: int
    maximum_join_work: int


@dataclass(frozen=True)
class PredicateSpanThresholdGridDecode:
    """Safe decodes plus explicit density exclusions for one example."""

    decoded: dict[
        tuple[float, float, float], set[tuple[str, str, str]]
    ]
    rejected_combinations: tuple[tuple[float, float, float], ...]
    density_rejections: tuple[ThresholdDensityRejection, ...]
    active_candidates_by_combination: dict[
        tuple[float, float, float], tuple[int, int, int]
    ] = field(default_factory=dict)
    join_work_by_combination: dict[
        tuple[float, float, float], int
    ] = field(default_factory=dict)
    combination_density_rejections: tuple[
        ThresholdCombinationDensityRejection, ...
    ] = ()


class CMeIEPredicateSpanLinker(nn.Module):
    """Encoder with predicate-specific subject, object and start-link heads."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        predicates: int,
        inner_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.subject = GlobalPointer(hidden_size, predicates, inner_dim)
        self.object = GlobalPointer(hidden_size, predicates, inner_dim)
        self.link = DirectionalGlobalPointer(hidden_size, predicates, inner_dim)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        encoded = self.encoder(input_ids, **kwargs)
        hidden = (
            encoded.last_hidden_state
            if hasattr(encoded, "last_hidden_state")
            else encoded[0]
        )
        return {
            "subject": self.subject(hidden, attention_mask),
            "object": self.object(hidden, attention_mask),
            "link": self.link(hidden, attention_mask),
        }


def build_predicate_span_labels(
    row: dict[str, Any],
    offsets: list[tuple[int, int]],
    predicate2id: dict[str, int],
    *,
    allow_cross_sentence_fallback: bool = True,
) -> tuple[dict[str, torch.Tensor], int]:
    """Create labels, including one deterministic cross-sentence fallback.

    CMeIE contains gold surface pairs for which no two occurrences share a
    sentence.  In that case, and only when explicitly enabled, the existing
    mention selector contributes its single nearest deterministic pair.  This
    keeps training labels representable without introducing a Cartesian set
    of unsupported cross-sentence links.
    """

    sequence = len(offsets)
    shape = (len(predicate2id), sequence, sequence)
    labels = {
        # The loss consumes these tensors as booleans.  Keeping the dense
        # predicate/sequence grids boolean reduces host and device label
        # memory by 4x for the formal max_length=320 training run.
        "subject": torch.zeros(shape, dtype=torch.bool),
        "object": torch.zeros(shape, dtype=torch.bool),
        "link": torch.zeros(shape, dtype=torch.bool),
    }
    dropped = 0
    text = str(row.get("text", ""))
    for spo in row.get("spo_list", []):
        predicate = str(spo.get("predicate", ""))
        predicate_id = predicate2id.get(predicate)
        if predicate_id is None:
            dropped += 1
            continue
        pairs = select_mention_pairs(
            text,
            str(spo.get("subject", "")),
            str(spo.get("object", "")),
            allow_cross_sentence_fallback=allow_cross_sentence_fallback,
        )
        aligned = 0
        for subject_chars, object_chars in pairs:
            subject_tokens = char_span_to_tokens(offsets, subject_chars)
            object_tokens = char_span_to_tokens(offsets, object_chars)
            if subject_tokens is None or object_tokens is None:
                continue
            subject_start, subject_end = subject_tokens
            object_start, object_end = object_tokens
            labels["subject"][predicate_id, subject_start, subject_end] = 1
            labels["object"][predicate_id, object_start, object_end] = 1
            labels["link"][predicate_id, subject_start, object_start] = 1
            aligned += 1
        if not aligned:
            dropped += 1
    return labels, dropped


def hard_negative_pointer_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    negatives_per_head: int = 32,
    inactive_weight: float = 0.25,
) -> torch.Tensor:
    """Balanced BCE using the highest-scoring valid negatives in each head."""

    if negatives_per_head < 1:
        raise ValueError("negatives_per_head must be positive")
    if not 0.0 <= inactive_weight <= 1.0:
        raise ValueError("inactive_weight must be between zero and one")
    batch, heads = logits.shape[:2]
    flat_logits = logits.float().reshape(batch, heads, -1)
    flat_labels = labels.bool().reshape(batch, heads, -1)
    valid = flat_logits > -1e11
    positive_count = flat_labels.sum(dim=-1).clamp_min(1)
    positive_loss = torch.where(
        flat_labels,
        F.softplus(-flat_logits),
        torch.zeros_like(flat_logits),
    ).sum(dim=-1) / positive_count

    negative_logits = flat_logits.masked_fill(~(valid & ~flat_labels), -1e12)
    k = min(int(negatives_per_head), negative_logits.shape[-1])
    hardest = torch.topk(negative_logits, k=k, dim=-1).values
    hardest_valid = hardest > -1e11
    negative_loss = torch.where(
        hardest_valid,
        F.softplus(hardest),
        torch.zeros_like(hardest),
    ).sum(dim=-1) / hardest_valid.sum(dim=-1).clamp_min(1)

    active = flat_labels.any(dim=-1)
    per_head_loss = torch.where(
        active,
        positive_loss + negative_loss,
        negative_loss,
    )
    per_head_weight = torch.where(
        active,
        torch.ones_like(per_head_loss),
        torch.full_like(per_head_loss, inactive_weight),
    )
    per_sample = (per_head_loss * per_head_weight).sum(dim=1)
    per_sample = per_sample / per_head_weight.sum(dim=1).clamp_min(1e-6)
    return per_sample.mean()


def balanced_per_head_sparse_logsumexp_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    inactive_weight: float = 0.1,
) -> torch.Tensor:
    """Exact multilabel logsumexp with explicit per-head balancing.

    The positive and negative terms use the stable GlobalPointer multilabel
    identity.  Reduction is deliberately separate: every active predicate
    head has weight ``1`` and every inactive predicate head has weight
    ``inactive_weight`` before normalising inside each sample.  Consequently,
    adding more inactive predicate heads cannot silently change the requested
    active-to-inactive *per-head* weight ratio.

    This loss is an explicit experiment.  The formal/default path remains
    :func:`hard_negative_pointer_loss`.
    """

    if not 0.0 <= inactive_weight <= 1.0:
        raise ValueError("inactive_weight must be between zero and one")
    batch, heads = logits.shape[:2]
    y_pred = logits.float().reshape(batch, heads, -1)
    y_true = labels.bool().reshape(batch, heads, -1)
    valid = y_pred > -1e11
    positive = y_true & valid
    negative = valid & ~y_true
    negative_logits = y_pred.masked_fill(~negative, -1e12)
    positive_logits = (-y_pred).masked_fill(~positive, -1e12)
    zeros = torch.zeros_like(y_pred[..., :1])
    negative_loss = torch.logsumexp(
        torch.cat([negative_logits, zeros], dim=-1),
        dim=-1,
    )
    positive_loss = torch.logsumexp(
        torch.cat([positive_logits, zeros], dim=-1),
        dim=-1,
    )
    per_head_loss = negative_loss + positive_loss
    active = positive.any(dim=-1)
    per_head_weight = torch.where(
        active,
        torch.ones_like(per_head_loss),
        torch.full_like(per_head_loss, inactive_weight),
    )
    per_sample = (per_head_loss * per_head_weight).sum(dim=1)
    per_sample = per_sample / per_head_weight.sum(dim=1).clamp_min(1e-6)
    return per_sample.mean()


def _allowed_mention_pair(
    text: str,
    subject_chars: tuple[int, int],
    object_chars: tuple[int, int],
    subject: str,
    obj: str,
    fallback_cache: dict[
        tuple[str, str],
        frozenset[tuple[tuple[int, int], tuple[int, int]]],
    ],
) -> bool:
    """Accept same-sentence pairs or the surface pair's unique fallback."""

    if same_sentence(text, subject_chars, object_chars):
        return True
    key = (subject, obj)
    allowed = fallback_cache.get(key)
    if allowed is None:
        allowed = frozenset(
            select_mention_pairs(
                text,
                subject,
                obj,
                allow_cross_sentence_fallback=True,
            )
        )
        fallback_cache[key] = allowed
    return (subject_chars, object_chars) in allowed


def predicate_span_loss(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    *,
    negatives_per_head: int = 32,
    inactive_weight: float = 0.25,
    loss_mode: str = LOSS_MODE_HARD_NEGATIVE_BCE,
) -> torch.Tensor:
    mode = str(loss_mode).casefold()
    if mode == LOSS_MODE_HARD_NEGATIVE_BCE:
        losses = [
            hard_negative_pointer_loss(
                outputs[name],
                labels[name],
                negatives_per_head=negatives_per_head,
                inactive_weight=inactive_weight,
            )
            for name in ("subject", "object", "link")
        ]
    elif mode == LOSS_MODE_BALANCED_PER_HEAD_SPARSE_LOGSUMEXP:
        losses = [
            balanced_per_head_sparse_logsumexp_loss(
                outputs[name],
                labels[name],
                inactive_weight=inactive_weight,
            )
            for name in ("subject", "object", "link")
        ]
    else:
        raise ValueError(f"unsupported predicate-span loss mode: {loss_mode}")
    return sum(losses) / len(losses)


def decode_predicate_spans(
    text: str,
    offsets: list[tuple[int, int]],
    outputs: dict[str, torch.Tensor],
    predicates: list[str],
    thresholds: tuple[float, float, float],
) -> set[tuple[str, str, str]]:
    """Decode exact surface SPOs, requiring a same-predicate start link."""

    subject_by_start: dict[tuple[int, int], list[tuple[int, tuple[int, int], str]]] = defaultdict(list)
    object_by_start: dict[tuple[int, int], list[tuple[int, tuple[int, int], str]]] = defaultdict(list)
    for name, target, threshold in (
        ("subject", subject_by_start, thresholds[0]),
        ("object", object_by_start, thresholds[1]),
    ):
        for predicate_id, start, end in (
            outputs[name].float() > threshold
        ).nonzero(as_tuple=False).tolist():
            if (
                start > end
                or offsets[start][1] <= offsets[start][0]
                or offsets[end][1] <= offsets[end][0]
            ):
                continue
            chars = (offsets[start][0], offsets[end][1])
            value = text[chars[0] : chars[1]]
            if value:
                target[(predicate_id, start)].append((end, chars, value))

    predicted: set[tuple[str, str, str]] = set()
    fallback_cache: dict[
        tuple[str, str],
        frozenset[tuple[tuple[int, int], tuple[int, int]]],
    ] = {}
    for predicate_id, subject_start, object_start in (
        outputs["link"].float() > thresholds[2]
    ).nonzero(as_tuple=False).tolist():
        for _, subject_chars, subject in subject_by_start.get(
            (predicate_id, subject_start), ()
        ):
            for _, object_chars, obj in object_by_start.get(
                (predicate_id, object_start), ()
            ):
                if _allowed_mention_pair(
                    text,
                    subject_chars,
                    object_chars,
                    subject,
                    obj,
                    fallback_cache,
                ):
                    predicted.add((subject, predicates[predicate_id], obj))
    return predicted


def decode_predicate_span_threshold_grid(
    text: str,
    offsets: list[tuple[int, int]],
    outputs: dict[str, torch.Tensor],
    predicates: list[str],
    threshold_values: tuple[float, ...],
    *,
    maximum_active_candidates: int | None = None,
    maximum_join_work: int | None = None,
    threshold_combinations: tuple[
        tuple[float, float, float], ...
    ] | None = None,
) -> PredicateSpanThresholdGridDecode:
    """Decode a declared global threshold grid after one sparse extraction.

    Applying a dense ``[44, sequence, sequence]`` comparison 27 times per
    example dominates calibration evaluation.  Density is checked separately
    for every head and threshold.  Combinations containing an unsafe threshold
    are reported and omitted, while candidates are extracted once at the
    lowest still-safe threshold for each head.  Before any subject/object
    Cartesian join is materialized, its exact work is counted per combination;
    only combinations over ``maximum_join_work`` are rejected.  Every returned
    decode is exactly equivalent to a :func:`decode_predicate_spans` call.
    """

    if not threshold_values:
        raise ValueError("threshold_values must not be empty")
    if maximum_join_work is not None and maximum_join_work < 1:
        raise ValueError("maximum_join_work must be positive")
    values = tuple(float(value) for value in threshold_values)
    if len(set(values)) != len(values):
        raise ValueError("threshold_values must be unique")
    if threshold_combinations is None:
        combinations = tuple(product(values, repeat=3))
    else:
        combinations = tuple(
            tuple(float(value) for value in thresholds)
            for thresholds in threshold_combinations
        )
        if not combinations:
            raise ValueError("threshold_combinations must not be empty")
        if len(set(combinations)) != len(combinations):
            raise ValueError("threshold_combinations must be unique")
        if any(
            len(thresholds) != 3
            or any(value not in values for value in thresholds)
            for thresholds in combinations
        ):
            raise ValueError(
                "threshold_combinations must use the declared threshold values"
            )

    density_rejections: list[ThresholdDensityRejection] = []
    active_by_head_threshold: dict[tuple[str, float], int] = {}
    unsafe: set[tuple[str, float]] = set()
    if maximum_active_candidates is not None and maximum_active_candidates < 1:
        raise ValueError("maximum_active_candidates must be positive")
    for threshold_index, name in enumerate(("subject", "object", "link")):
        tensor = outputs[name].float()
        for threshold in sorted(
            {item[threshold_index] for item in combinations}
        ):
            active = int((tensor > threshold).sum().item())
            active_by_head_threshold[(name, threshold)] = active
            if (
                maximum_active_candidates is not None
                and active > maximum_active_candidates
            ):
                unsafe.add((name, threshold))
                density_rejections.append(
                    ThresholdDensityRejection(
                        head=name,
                        threshold=threshold,
                        active_candidates=active,
                    )
                )

    head_names = ("subject", "object", "link")
    rejected_combinations = tuple(
        thresholds
        for thresholds in combinations
        if any(
            (head_names[index], thresholds[index]) in unsafe
            for index in range(3)
        )
    )
    rejected_set = set(rejected_combinations)
    safe_combinations = tuple(
        thresholds
        for thresholds in combinations
        if thresholds not in rejected_set
    )
    if not safe_combinations:
        details = ", ".join(
            f"{item.head}@{item.threshold}={item.active_candidates}"
            for item in density_rejections
        )
        raise OverflowError(
            "candidate density rejects every remaining threshold combination "
            f"at maximum {maximum_active_candidates}: {details}"
        )

    extraction_thresholds = {
        name: min(
            thresholds[index] for thresholds in safe_combinations
        )
        for index, name in enumerate(head_names)
    }
    subject_by_start: dict[
        tuple[int, int], list[tuple[tuple[int, int], str, float]]
    ] = defaultdict(list)
    object_by_start: dict[
        tuple[int, int], list[tuple[tuple[int, int], str, float]]
    ] = defaultdict(list)
    for name, target in (
        ("subject", subject_by_start),
        ("object", object_by_start),
    ):
        tensor = outputs[name].float()
        extraction_threshold = extraction_thresholds[name]
        for predicate_id, start, end in (
            tensor > extraction_threshold
        ).nonzero(as_tuple=False).tolist():
            if (
                start > end
                or offsets[start][1] <= offsets[start][0]
                or offsets[end][1] <= offsets[end][0]
            ):
                continue
            chars = (offsets[start][0], offsets[end][1])
            value = text[chars[0] : chars[1]]
            if value:
                target[(predicate_id, start)].append(
                    (chars, value, float(tensor[predicate_id, start, end]))
                )

    link_tensor = outputs["link"].float()
    link_extraction_threshold = extraction_thresholds["link"]
    links = [
        (
            predicate_id,
            subject_start,
            object_start,
            float(link_tensor[predicate_id, subject_start, object_start]),
        )
        for predicate_id, subject_start, object_start in (
            link_tensor > link_extraction_threshold
        ).nonzero(as_tuple=False).tolist()
    ]
    join_work_by_combination: dict[
        tuple[float, float, float], int
    ] = {}
    combination_density_rejections: list[
        ThresholdCombinationDensityRejection
    ] = []
    join_safe_combinations: list[tuple[float, float, float]] = []
    for thresholds in safe_combinations:
        subject_threshold, object_threshold, link_threshold = thresholds
        join_work = 0
        for predicate_id, subject_start, object_start, link_score in links:
            if link_score <= link_threshold:
                continue
            subject_count = sum(
                subject_score > subject_threshold
                for _, _, subject_score in subject_by_start.get(
                    (predicate_id, subject_start), ()
                )
            )
            if not subject_count:
                continue
            object_count = sum(
                object_score > object_threshold
                for _, _, object_score in object_by_start.get(
                    (predicate_id, object_start), ()
                )
            )
            join_work += subject_count * object_count
        if maximum_join_work is not None and join_work > maximum_join_work:
            combination_density_rejections.append(
                ThresholdCombinationDensityRejection(
                    thresholds=thresholds,
                    join_work=join_work,
                    maximum_join_work=maximum_join_work,
                )
            )
            continue
        join_work_by_combination[thresholds] = join_work
        join_safe_combinations.append(thresholds)

    if not join_safe_combinations:
        details = ", ".join(
            f"{item.thresholds}={item.join_work}"
            for item in combination_density_rejections
        )
        raise OverflowError(
            "join-work density rejects every remaining threshold combination "
            f"at maximum {maximum_join_work}: {details}"
        )

    all_rejected_combinations = tuple(
        dict.fromkeys(
            rejected_combinations
            + tuple(
                item.thresholds
                for item in combination_density_rejections
            )
        )
    )
    active_candidates_by_combination = {
        thresholds: tuple(
            active_by_head_threshold[(head_names[index], thresholds[index])]
            for index in range(3)
        )
        for thresholds in join_safe_combinations
    }
    decoded: dict[
        tuple[float, float, float], set[tuple[str, str, str]]
    ] = {}
    fallback_cache: dict[
        tuple[str, str],
        frozenset[tuple[tuple[int, int], tuple[int, int]]],
    ] = {}
    for thresholds in join_safe_combinations:
        subject_threshold, object_threshold, link_threshold = thresholds
        predicted: set[tuple[str, str, str]] = set()
        for predicate_id, subject_start, object_start, link_score in links:
            if link_score <= link_threshold:
                continue
            for subject_chars, subject, subject_score in subject_by_start.get(
                (predicate_id, subject_start), ()
            ):
                if subject_score <= subject_threshold:
                    continue
                for object_chars, obj, object_score in object_by_start.get(
                    (predicate_id, object_start), ()
                ):
                    if object_score > object_threshold and _allowed_mention_pair(
                        text,
                        subject_chars,
                        object_chars,
                        subject,
                        obj,
                        fallback_cache,
                    ):
                        predicted.add((subject, predicates[predicate_id], obj))
        decoded[thresholds] = predicted
    return PredicateSpanThresholdGridDecode(
        decoded=decoded,
        rejected_combinations=all_rejected_combinations,
        density_rejections=tuple(density_rejections),
        active_candidates_by_combination=active_candidates_by_combination,
        join_work_by_combination=join_work_by_combination,
        combination_density_rejections=tuple(
            combination_density_rejections
        ),
    )
