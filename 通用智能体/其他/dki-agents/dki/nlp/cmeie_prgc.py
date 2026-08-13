"""PRGC-style direct CMeIE text-to-SPO extraction primitives.

This is the fail-closed fallback after a conditional-object candidate audit
shows that no bounded verifier can recover enough gold objects.  The model
keeps relation presence, predicate-conditioned argument boundaries, and
subject/object correspondence as separately observable heads.  It deliberately
does *not* share a dense 44-predicate correspondence tensor during training:
only every gold predicate and a deterministic set of at most four current hard
non-gold predicates are materialised for each source row.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dki.nlp.cmeie_spo import (
    char_span_to_tokens,
    same_sentence,
    select_mention_pairs,
)


PRGC_ARCHITECTURE_VERSION = "prgc_predicate_query_v2"


@dataclass(frozen=True)
class PRGCLabels:
    """Gold relation presence plus exact aligned spans for one input row."""

    relation: torch.Tensor
    triples: dict[int, tuple[tuple[int, int, int, int], ...]]
    dropped_spo: int


class PRGCDensityError(RuntimeError):
    """Raised when a decode would need an undeclared candidate truncation."""


def build_prgc_labels(
    row: dict[str, Any],
    offsets: Sequence[tuple[int, int]],
    predicate2id: dict[str, int],
    *,
    allow_cross_sentence_fallback: bool = True,
) -> PRGCLabels:
    """Align gold CMeIE SPOs without inventing repeated-mention Cartesian pairs."""

    relation = torch.zeros((len(predicate2id),), dtype=torch.float32)
    triples: dict[int, set[tuple[int, int, int, int]]] = {}
    dropped = 0
    text = str(row.get("text", ""))
    for spo in row.get("spo_list", []):
        predicate = str(spo.get("predicate", ""))
        predicate_id = predicate2id.get(predicate)
        if predicate_id is None:
            dropped += 1
            continue
        # Predicate presence includes every recognisable gold predicate even
        # when one span cannot be aligned.  Span/correspondence loss records
        # the latter separately in dropped_spo rather than silently erasing it.
        relation[predicate_id] = 1.0
        pairs = select_mention_pairs(
            text,
            str(spo.get("subject", "")),
            str(spo.get("object", "")),
            allow_cross_sentence_fallback=allow_cross_sentence_fallback,
        )
        aligned = 0
        for subject_chars, object_chars in pairs:
            subject_tokens = char_span_to_tokens(list(offsets), subject_chars)
            object_tokens = char_span_to_tokens(list(offsets), object_chars)
            if subject_tokens is None or object_tokens is None:
                continue
            subject_start, subject_end = subject_tokens
            object_start, object_end = object_tokens
            if not (
                0 <= subject_start <= subject_end < len(offsets)
                and 0 <= object_start <= object_end < len(offsets)
            ):
                continue
            triples.setdefault(predicate_id, set()).add(
                (subject_start, subject_end, object_start, object_end)
            )
            aligned += 1
        if not aligned:
            dropped += 1
    return PRGCLabels(
        relation=relation,
        triples={key: tuple(sorted(value)) for key, value in sorted(triples.items())},
        dropped_spo=dropped,
    )


def select_training_predicates(
    relation_logits: torch.Tensor,
    labels: Sequence[PRGCLabels],
    *,
    maximum_hard_non_gold: int = 4,
) -> tuple[list[int], list[int]]:
    """Return all gold predicates plus deterministic hard non-gold predicates.

    Selection is intentionally detached from autograd.  It determines which
    relation-specific argument/correspondence tensors are evaluated, while the
    relation-presence loss continues to receive all 44 predicate logits.
    """

    if relation_logits.ndim != 2:
        raise ValueError("relation_logits must have shape (batch,predicates)")
    if len(labels) != int(relation_logits.shape[0]):
        raise ValueError("labels and relation logits batch size differ")
    if maximum_hard_non_gold < 0:
        raise ValueError("maximum_hard_non_gold must be non-negative")
    scores = relation_logits.detach().float().cpu()
    parents: list[int] = []
    predicate_ids: list[int] = []
    for row_index, label in enumerate(labels):
        if int(label.relation.numel()) != int(relation_logits.shape[1]):
            raise ValueError("relation label predicate count differs from logits")
        positives = [
            predicate_id
            for predicate_id, enabled in enumerate(label.relation.tolist())
            if float(enabled) > 0.0
        ]
        positive_set = set(positives)
        negatives = [
            predicate_id
            for predicate_id in range(int(relation_logits.shape[1]))
            if predicate_id not in positive_set
        ]
        # Sorting by predicate id after the negative score makes equal logits
        # deterministic across accelerator kernels.
        negatives.sort(key=lambda value: (-float(scores[row_index, value]), value))
        chosen = sorted(positives) + negatives[:maximum_hard_non_gold]
        parents.extend([row_index] * len(chosen))
        predicate_ids.extend(chosen)
    if not parents:
        raise ValueError("training batch selected no predicate channels")
    return parents, predicate_ids


def build_selected_targets(
    labels: Sequence[PRGCLabels],
    parents: Sequence[int],
    predicate_ids: Sequence[int],
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct exact start/end and start-to-start targets for selected heads."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if len(parents) != len(predicate_ids):
        raise ValueError("parents and predicate_ids must have equal length")
    span = torch.zeros((len(parents), sequence_length, 4), dtype=torch.float32)
    correspondence = torch.zeros(
        (len(parents), sequence_length, sequence_length), dtype=torch.float32
    )
    for index, (parent, predicate_id) in enumerate(zip(parents, predicate_ids)):
        if parent < 0 or parent >= len(labels):
            raise ValueError("selected parent index is out of range")
        for subject_start, subject_end, object_start, object_end in labels[parent].triples.get(
            int(predicate_id), ()
        ):
            if not (
                0 <= subject_start <= subject_end < sequence_length
                and 0 <= object_start <= object_end < sequence_length
            ):
                raise ValueError("selected target span is out of range")
            span[index, subject_start, 0] = 1.0
            span[index, subject_end, 1] = 1.0
            span[index, object_start, 2] = 1.0
            span[index, object_end, 3] = 1.0
            correspondence[index, subject_start, object_start] = 1.0
    return span, correspondence


class CMeIEPRGC(nn.Module):
    """Predicate-conditioned span and low-rank correspondence model.

    ``predicate_ids`` and ``parent_indices`` describe a ragged collection of
    relation channels.  Evaluation may pass all 44 predicates for every input;
    formal training only passes all gold predicates and the bounded hard
    negatives selected by :func:`select_training_predicates`.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        predicates: int,
        *,
        predicate_rank: int = 128,
        correspondence_rank: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if predicates < 1:
            raise ValueError("predicates must be positive")
        self.encoder = encoder
        self.predicates = int(predicates)
        self.predicate_rank = int(predicate_rank)
        self.correspondence_rank = int(correspondence_rank)
        self.dropout = nn.Dropout(float(dropout))
        self.relation_presence = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.predicates),
        )
        self.predicate_embedding = nn.Embedding(self.predicates, self.predicate_rank)
        self.predicate_condition = nn.Linear(self.predicate_rank, hidden_size)
        self.conditioned_norm = nn.LayerNorm(hidden_size)
        # A predicate-dependent additive shift followed by one shared linear
        # head can mostly change a channel bias, but cannot reliably change the
        # ranking of tokens.  PRGC needs a real token/predicate interaction.
        # Each predicate therefore generates separate bilinear queries for the
        # four boundary roles.  Subject and object generators remain separate
        # so their gradients can still be clipped and audited independently.
        self.boundary_token_projection = nn.Linear(hidden_size, self.predicate_rank)
        self.boundary_token_norm = nn.LayerNorm(self.predicate_rank)
        self.subject_boundary = nn.Linear(
            self.predicate_rank, 2 * self.predicate_rank
        )
        self.object_boundary = nn.Linear(
            self.predicate_rank, 2 * self.predicate_rank
        )
        self.subject_boundary_bias = nn.Linear(self.predicate_rank, 2)
        self.object_boundary_bias = nn.Linear(self.predicate_rank, 2)
        self.span_logit_prior = nn.Parameter(torch.full((4,), -1.0))
        self.correspondence_subject = nn.Linear(hidden_size, self.correspondence_rank)
        self.correspondence_object = nn.Linear(hidden_size, self.correspondence_rank)
        self.correspondence_predicate = nn.Embedding(
            self.predicates, self.correspondence_rank
        )
        self.correspondence_logit_prior = nn.Parameter(torch.tensor(-1.0))

    def encode(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        kwargs = {"attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        encoded = self.encoder(input_ids, **kwargs)
        return (
            encoded.last_hidden_state
            if hasattr(encoded, "last_hidden_state")
            else encoded[0]
        )

    def _pooled(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def all_predicate_indices(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        predicate_ids = torch.arange(self.predicates, device=device).repeat(batch_size)
        parents = torch.arange(batch_size, device=device).repeat_interleave(self.predicates)
        return parents, predicate_ids

    def relation_logits(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the 44-way predicate-presence logits without dense channels."""

        return self.relation_presence(
            self.dropout(self._pooled(hidden, attention_mask))
        ).float()

    def score_channels(
        self,
        hidden: torch.Tensor,
        *,
        parent_indices: torch.Tensor,
        predicate_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score a ragged set of predicate channels against encoded rows."""

        if parent_indices.ndim != 1 or predicate_ids.ndim != 1:
            raise ValueError("parent_indices and predicate_ids must be one-dimensional")
        if int(parent_indices.numel()) != int(predicate_ids.numel()):
            raise ValueError("parent_indices and predicate_ids must have equal length")
        if not int(predicate_ids.numel()):
            raise ValueError("at least one predicate channel is required")
        parent_indices = parent_indices.long()
        predicate_ids = predicate_ids.long()
        if (
            (parent_indices < 0).any()
            or (parent_indices >= int(hidden.shape[0])).any()
            or (predicate_ids < 0).any()
            or (predicate_ids >= self.predicates).any()
        ):
            raise ValueError("predicate channel index is out of range")
        selected = hidden.index_select(0, parent_indices)
        predicate_embedding = self.predicate_embedding(predicate_ids)
        condition = self.predicate_condition(predicate_embedding)
        conditioned = self.conditioned_norm(selected + condition[:, None, :])
        boundary_tokens = self.boundary_token_norm(
            self.boundary_token_projection(self.dropout(selected))
        )
        subject_queries = self.subject_boundary(predicate_embedding).reshape(
            -1, 2, self.predicate_rank
        )
        object_queries = self.object_boundary(predicate_embedding).reshape(
            -1, 2, self.predicate_rank
        )
        subject_span = torch.einsum(
            "clr,cbr->clb", boundary_tokens, subject_queries
        ) / math.sqrt(float(self.predicate_rank))
        object_span = torch.einsum(
            "clr,cbr->clb", boundary_tokens, object_queries
        ) / math.sqrt(float(self.predicate_rank))
        subject_span = subject_span + self.subject_boundary_bias(
            predicate_embedding
        )[:, None, :]
        object_span = object_span + self.object_boundary_bias(
            predicate_embedding
        )[:, None, :]
        span = torch.cat([subject_span, object_span], dim=-1)
        span = (span + self.span_logit_prior).float()
        subject = self.correspondence_subject(self.dropout(conditioned))
        obj = self.correspondence_object(self.dropout(conditioned))
        predicate = self.correspondence_predicate(predicate_ids).unsqueeze(1)
        correspondence = torch.matmul(
            subject * predicate,
            obj.transpose(1, 2),
        ) / math.sqrt(float(self.correspondence_rank))
        correspondence = correspondence + self.correspondence_logit_prior
        return {
            "span": span.float(),
            "correspondence": correspondence.float(),
            "parent_indices": parent_indices,
            "predicate_ids": predicate_ids,
        }

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        *,
        parent_indices: torch.Tensor | None = None,
        predicate_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(input_ids, attention_mask, token_type_ids)
        relation = self.relation_logits(hidden, attention_mask)
        if parent_indices is None or predicate_ids is None:
            parent_indices, predicate_ids = self.all_predicate_indices(
                int(hidden.shape[0]), hidden.device
            )
        channels = self.score_channels(
            hidden,
            parent_indices=parent_indices,
            predicate_ids=predicate_ids,
        )
        return {"hidden": hidden, "relation": relation, **channels}


def balanced_hard_negative_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    hard_negative_ratio: int = 8,
    inactive_channel_weight: float = 0.25,
) -> torch.Tensor:
    """Balanced BCE with at most ``HNR`` current highest-scoring negatives.

    A channel with no positive label still contributes a bounded hard-negative
    term.  This is essential for the selected non-gold predicate channels and
    avoids pretending an all-negative batch has zero loss.
    """

    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have identical shapes")
    if hard_negative_ratio < 1:
        raise ValueError("hard_negative_ratio must be positive")
    if not 0.0 < inactive_channel_weight <= 1.0:
        raise ValueError("inactive_channel_weight must be in (0, 1]")
    values = logits.float()
    target = labels.bool()
    if valid_mask is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        valid = valid_mask.bool()
        while valid.ndim < target.ndim:
            valid = valid.unsqueeze(-1)
        valid = valid.expand_as(target)
    flat_logits = values.reshape(values.shape[0], -1)
    flat_target = (target & valid).reshape(values.shape[0], -1)
    flat_valid = valid.reshape(values.shape[0], -1)
    positive_count = flat_target.sum(dim=-1)
    negative_mask = flat_valid & ~flat_target
    negative_count = negative_mask.sum(dim=-1)
    desired_negative = torch.where(
        positive_count > 0,
        positive_count * hard_negative_ratio,
        torch.full_like(positive_count, hard_negative_ratio),
    )
    desired_negative = torch.minimum(desired_negative, negative_count)
    max_negative = int(desired_negative.max().detach().cpu()) if desired_negative.numel() else 0
    positive_loss = torch.where(
        flat_target,
        F.softplus(-flat_logits),
        torch.zeros_like(flat_logits),
    ).sum(dim=-1) / positive_count.clamp_min(1)
    if max_negative:
        ranked = torch.topk(
            flat_logits.masked_fill(~negative_mask, float("-inf")),
            k=max_negative,
            dim=-1,
            largest=True,
            sorted=True,
        ).values
        positions = torch.arange(max_negative, device=logits.device).unsqueeze(0)
        chosen = positions < desired_negative.unsqueeze(1)
        negative_loss = torch.where(
            chosen,
            F.softplus(ranked),
            torch.zeros_like(ranked),
        ).sum(dim=-1) / desired_negative.clamp_min(1)
    else:
        negative_loss = torch.zeros_like(positive_loss)
    positive_present = positive_count.gt(0).to(dtype=values.dtype)
    # A selected hard non-gold predicate must still learn to reject its
    # boundaries, but it must not outweigh every positive predicate channel.
    # Without this channel-level balance the fixed 8-row overfit diagnostic
    # showed correctly learned relation/correspondence logits while all gold
    # boundary logits remained below the declared -0.5 threshold.
    per_channel = (positive_loss * positive_present + negative_loss) / (
        positive_present + 1.0
    )
    channel_weight = torch.where(
        positive_count.gt(0),
        torch.ones_like(per_channel),
        torch.full_like(per_channel, float(inactive_channel_weight)),
    )
    return (per_channel * channel_weight).sum() / channel_weight.sum().clamp_min(1e-6)


def prgc_loss(
    outputs: dict[str, torch.Tensor],
    relation_labels: torch.Tensor,
    span_labels: torch.Tensor,
    correspondence_labels: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    hard_negative_ratio: int = 8,
    inactive_channel_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equal-weight relation, boundary and correspondence losses."""

    parents = outputs["parent_indices"]
    selected_mask = attention_mask.index_select(0, parents).bool()
    pair_mask = selected_mask.unsqueeze(2) & selected_mask.unsqueeze(1)
    relation_loss = balanced_hard_negative_bce(
        outputs["relation"],
        relation_labels,
        hard_negative_ratio=hard_negative_ratio,
        inactive_channel_weight=inactive_channel_weight,
    )
    span_loss = balanced_hard_negative_bce(
        outputs["span"],
        span_labels,
        valid_mask=selected_mask,
        hard_negative_ratio=hard_negative_ratio,
        inactive_channel_weight=inactive_channel_weight,
    )
    correspondence_loss = balanced_hard_negative_bce(
        outputs["correspondence"],
        correspondence_labels,
        valid_mask=pair_mask,
        hard_negative_ratio=hard_negative_ratio,
        inactive_channel_weight=inactive_channel_weight,
    )
    total = relation_loss + span_loss + correspondence_loss
    return total, {
        "relation": relation_loss.detach(),
        "span": span_loss.detach(),
        "correspondence": correspondence_loss.detach(),
    }


def _span_candidates(
    logits: torch.Tensor,
    offsets: Sequence[tuple[int, int]],
    *,
    start_column: int,
    end_column: int,
    threshold: float,
    maximum_candidates: int,
) -> list[tuple[int, int, tuple[int, int], str]]:
    starts = [
        index
        for index, value in enumerate(logits[:, start_column].float().tolist())
        if value > threshold and offsets[index][1] > offsets[index][0]
    ]
    ends = [
        index
        for index, value in enumerate(logits[:, end_column].float().tolist())
        if value > threshold and offsets[index][1] > offsets[index][0]
    ]
    # Start and end heads are marginal boundary predictors.  Their Cartesian
    # product invents spans that were never represented by either the PRGC
    # labels or the start-to-start correspondence head.  Pair every start with
    # its nearest legal end, deterministically, as in the stable CasRel
    # decoder.  This remains fail-closed: the cap is checked on the complete
    # nearest-pair set and no candidate is silently truncated.
    ordered_ends = sorted(ends)
    pairs: list[tuple[int, int]] = []
    for start in sorted(starts):
        end = next((value for value in ordered_ends if value >= start), None)
        if end is not None:
            pairs.append((start, end))
    if len(pairs) > maximum_candidates:
        raise PRGCDensityError(
            f"span candidates exceed cap ({len(pairs)} > {maximum_candidates})"
        )
    return [
        (start, end, (offsets[start][0], offsets[end][1]), "")
        for start, end in pairs
    ]


def _allowed_pair(
    text: str,
    subject_chars: tuple[int, int],
    object_chars: tuple[int, int],
    subject: str,
    obj: str,
    cache: dict[
        tuple[str, str],
        frozenset[tuple[tuple[int, int], tuple[int, int]]],
    ],
) -> bool:
    if same_sentence(text, subject_chars, object_chars):
        return True
    key = (subject, obj)
    allowed = cache.get(key)
    if allowed is None:
        allowed = frozenset(
            select_mention_pairs(
                text,
                subject,
                obj,
                allow_cross_sentence_fallback=True,
            )
        )
        cache[key] = allowed
    return (subject_chars, object_chars) in allowed


def decode_prgc(
    text: str,
    offsets: Sequence[tuple[int, int]],
    relation_logits: torch.Tensor,
    span_logits: torch.Tensor,
    correspondence_logits: torch.Tensor,
    predicates: Sequence[str],
    thresholds: tuple[float, float, float],
    *,
    schema_constraint: bool,
    maximum_span_candidates: int = 64,
    maximum_correspondence_candidates: int = 512,
    maximum_triples: int = 4096,
) -> set[tuple[str, str, str]]:
    """Decode PRGC outputs exactly or fail closed before any truncation.

    The CMeIE metric does not include entity types, so this architecture has no
    type prediction available for a stronger schema filter.  ``schema_constraint``
    therefore applies the only type-free rule supported by the model: reject a
    self-surface triple.  The setting remains explicit in the 54-combination
    calibration receipt rather than pretending a non-existent type head exists.
    """

    if len(predicates) != int(relation_logits.numel()):
        raise ValueError("relation logits and predicate table differ")
    if span_logits.shape[:2] != (len(predicates), len(offsets)):
        raise ValueError("span logits do not match predicate/token dimensions")
    if correspondence_logits.shape[:3] != (
        len(predicates),
        len(offsets),
        len(offsets),
    ):
        raise ValueError("correspondence logits do not match predicate/token dimensions")
    relation_threshold, span_threshold, correspondence_threshold = thresholds
    predicted: set[tuple[str, str, str]] = set()
    fallback_cache: dict[
        tuple[str, str],
        frozenset[tuple[tuple[int, int], tuple[int, int]]],
    ] = {}
    for predicate_id, predicate in enumerate(predicates):
        if float(relation_logits[predicate_id]) <= relation_threshold:
            continue
        subjects = _span_candidates(
            span_logits[predicate_id],
            offsets,
            start_column=0,
            end_column=1,
            threshold=span_threshold,
            maximum_candidates=maximum_span_candidates,
        )
        objects = _span_candidates(
            span_logits[predicate_id],
            offsets,
            start_column=2,
            end_column=3,
            threshold=span_threshold,
            maximum_candidates=maximum_span_candidates,
        )
        subject_by_start: dict[int, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
        object_by_start: dict[int, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
        for start, end, chars, _ in subjects:
            subject_by_start[start].append((end, chars))
        for start, end, chars, _ in objects:
            object_by_start[start].append((end, chars))
        correspondence = [
            (int(subject_start), int(object_start))
            for subject_start, object_start in (
                correspondence_logits[predicate_id].float() > correspondence_threshold
            ).nonzero(as_tuple=False).tolist()
            if subject_start in subject_by_start and object_start in object_by_start
        ]
        if len(correspondence) > maximum_correspondence_candidates:
            raise PRGCDensityError(
                "correspondence candidates exceed cap "
                f"({len(correspondence)} > {maximum_correspondence_candidates})"
            )
        for subject_start, object_start in correspondence:
            for _subject_end, subject_chars in subject_by_start[subject_start]:
                for _object_end, object_chars in object_by_start[object_start]:
                    subject = text[subject_chars[0] : subject_chars[1]]
                    obj = text[object_chars[0] : object_chars[1]]
                    if not subject or not obj:
                        continue
                    if schema_constraint and subject == obj:
                        continue
                    if not _allowed_pair(
                        text,
                        subject_chars,
                        object_chars,
                        subject,
                        obj,
                        fallback_cache,
                    ):
                        continue
                    predicted.add((subject, str(predicate), obj))
                    if len(predicted) > maximum_triples:
                        raise PRGCDensityError(
                            f"decoded triples exceed cap ({len(predicted)} > {maximum_triples})"
                        )
    return predicted


def exact_spo_metric(
    predicted: Iterable[set[tuple[str, str, str]]],
    gold: Iterable[set[tuple[str, str, str]]],
) -> dict[str, float | int]:
    tp = fp = fn = 0
    for candidate, expected in zip(predicted, gold):
        tp += len(candidate & expected)
        fp += len(candidate - expected)
        fn += len(expected - candidate)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }
