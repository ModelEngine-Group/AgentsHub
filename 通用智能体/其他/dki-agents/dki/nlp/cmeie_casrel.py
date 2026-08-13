"""CasRel-style conditional extraction primitives for CMeIE.

The first head finds subject spans.  The second head conditions every token on
one selected subject span and predicts object start/end positions for all
predicates.  This changes relation learning from a sparse ``predicate x L x L``
matrix into a considerably denser ``subject x predicate x L`` problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dki.nlp.cmeie_spo import char_span_to_tokens, select_mention_pairs


@dataclass(frozen=True)
class CasRelLabels:
    """Token labels for one source row."""

    subject: torch.Tensor
    objects: dict[tuple[int, int], torch.Tensor]
    # Exact relation-aware object spans are retained independently from the
    # boundary tensor.  A boundary tensor alone cannot distinguish two object
    # spans that share a start or end under different predicates.
    object_spans: dict[tuple[int, int], tuple[tuple[int, int, int], ...]]
    dropped_spo: int


def build_casrel_labels(
    row: dict[str, Any],
    offsets: Sequence[tuple[int, int]],
    predicate2id: dict[str, int],
    *,
    allow_cross_sentence_fallback: bool = True,
) -> CasRelLabels:
    """Align all representable SPOs without using entity-type labels."""

    sequence = len(offsets)
    subject = torch.zeros((sequence, 2), dtype=torch.float32)
    objects: dict[tuple[int, int], torch.Tensor] = {}
    object_spans: dict[tuple[int, int], set[tuple[int, int, int]]] = {}
    dropped = 0
    text = str(row.get("text", ""))
    for spo in row.get("spo_list", []):
        predicate = str(spo.get("predicate", ""))
        if predicate not in predicate2id:
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
            subject_tokens = char_span_to_tokens(list(offsets), subject_chars)
            object_tokens = char_span_to_tokens(list(offsets), object_chars)
            if subject_tokens is None or object_tokens is None:
                continue
            ss, se = subject_tokens
            os_, oe = object_tokens
            if not (0 <= ss <= se < sequence and 0 <= os_ <= oe < sequence):
                continue
            subject[ss, 0] = 1
            subject[se, 1] = 1
            target = objects.setdefault(
                (ss, se),
                torch.zeros(
                    (sequence, len(predicate2id), 2),
                    dtype=torch.float32,
                ),
            )
            target[os_, predicate2id[predicate], 0] = 1
            target[oe, predicate2id[predicate], 1] = 1
            object_spans.setdefault((ss, se), set()).add(
                (predicate2id[predicate], os_, oe)
            )
            aligned += 1
        if not aligned:
            dropped += 1
    return CasRelLabels(
        subject=subject,
        objects=objects,
        object_spans={
            span: tuple(sorted(values)) for span, values in object_spans.items()
        },
        dropped_spo=dropped,
    )


class SubjectObjectInteraction(nn.Module):
    """Bounded predicate-specific compatibility for one subject and every token.

    The original additive CasRel condition shifts every token by the same
    subject vector before a linear predicate head.  That route can still give
    two semantically similar subjects almost identical object logits.  This
    low-rank interaction supplies the missing multiplicative evidence while
    keeping logits bounded, which is important before any encoder unfreeze.
    """

    def __init__(
        self,
        hidden_size: int,
        predicates: int,
        rank: int | None = None,
    ) -> None:
        super().__init__()
        self.predicates = int(predicates)
        self.rank = min(64, int(hidden_size)) if rank is None else int(rank)
        if self.rank < 1:
            raise ValueError("interaction rank must be positive")
        self.subject_query = nn.Linear(
            hidden_size * 2,
            self.predicates * 2 * self.rank,
        )
        self.object_key = nn.Linear(hidden_size, 2 * self.rank)
        # softplus(raw_scale) + 1 starts at eight.  Cosine compatibility is
        # therefore expressive enough to veto a residual false positive but
        # cannot create the unbounded products seen in the failed joint runs.
        self.raw_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(7.0)), dtype=torch.float32)
        )

    def forward(
        self,
        selected_hidden: torch.Tensor,
        subject_start: torch.Tensor,
        subject_end: torch.Tensor,
    ) -> torch.Tensor:
        count, sequence, _hidden = selected_hidden.shape
        query = self.subject_query(
            torch.cat([subject_start, subject_end], dim=-1)
        ).view(count, self.predicates, 2, self.rank)
        key = self.object_key(selected_hidden).view(count, sequence, 2, self.rank)
        query = F.normalize(query.float(), dim=-1, eps=1e-6)
        key = F.normalize(key.float(), dim=-1, eps=1e-6)
        compatibility = torch.einsum("npbr,nlbr->nlpb", query, key)
        scale = (F.softplus(self.raw_scale.float()) + 1.0).clamp(max=16.0)
        return compatibility * scale


class ResidualContextAdapter(nn.Module):
    """Small trainable domain adapter over a frozen contextual encoder.

    Full encoder unfreezing produced extreme gradients in the conditional
    object objective.  This bottleneck keeps the pretrained representation
    fixed while allowing every contextual token to move in a bounded residual
    subspace.  The zero-initialized up projection preserves the exact frozen
    baseline at step zero.
    """

    def __init__(self, hidden_size: int, bottleneck: int | None = None) -> None:
        super().__init__()
        width = (
            min(192, max(4, hidden_size // 4))
            if bottleneck is None
            else int(bottleneck)
        )
        if width < 1:
            raise ValueError("adapter bottleneck must be positive")
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, width)
        self.up = nn.Linear(width, hidden_size)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(self.norm(hidden))))
        scale = self.scale.float().clamp(min=0.0, max=1.0)
        return hidden + residual * scale


class PredicateAwareSpanVerifier(nn.Module):
    """Score one exact ``(subject, predicate, object span)`` candidate.

    Boundary heads remain the recall-oriented first stage.  This verifier is a
    separate, compact precision stage so adding it cannot change the historical
    boundary logits at step zero.
    """

    def __init__(self, hidden_size: int, predicates: int, rank: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.rank = int(rank)
        if self.rank < 1:
            raise ValueError("verifier rank must be positive")
        self.subject_projection = nn.Linear(hidden_size * 2, self.rank)
        self.object_projection = nn.Linear(hidden_size * 2, self.rank)
        self.predicate_embedding = nn.Embedding(int(predicates), self.rank)
        width = self.rank * 5
        self.norm = nn.LayerNorm(width)
        self.hidden = nn.Linear(width, self.rank)
        self.dropout = nn.Dropout(float(dropout))
        self.output = nn.Linear(self.rank, 1)
        # A zero final projection makes every verifier logit zero initially and
        # leaves the pre-existing object boundary path byte-for-byte unchanged.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        subject_start: torch.Tensor,
        subject_end: torch.Tensor,
        object_start: torch.Tensor,
        object_end: torch.Tensor,
        predicate_ids: torch.Tensor,
    ) -> torch.Tensor:
        subject = self.subject_projection(torch.cat([subject_start, subject_end], dim=-1))
        obj = self.object_projection(torch.cat([object_start, object_end], dim=-1))
        predicate = self.predicate_embedding(predicate_ids.long())
        features = torch.cat(
            [subject, obj, subject * obj, torch.abs(subject - obj), predicate], dim=-1
        )
        return self.output(self.dropout(F.gelu(self.hidden(self.norm(features))))).squeeze(-1).float()


class CMeIECasRel(nn.Module):
    """Encoder with subject and subject-conditioned predicate-object heads."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        predicates: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predicates = int(predicates)
        self.dropout = nn.Dropout(float(dropout))
        self.context_adapter = ResidualContextAdapter(hidden_size)
        self.subject = nn.Linear(hidden_size, 2)
        self.subject_condition = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
        )
        self.conditioned_norm = nn.LayerNorm(hidden_size)
        self.object = nn.Linear(hidden_size, self.predicates * 2)
        self.subject_object_interaction = SubjectObjectInteraction(
            hidden_size,
            self.predicates,
        )
        self.span_verifier = PredicateAwareSpanVerifier(
            hidden_size,
            self.predicates,
            dropout=dropout,
        )

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        encoded = self.encoder(input_ids, **kwargs)
        hidden = (
            encoded.last_hidden_state
            if hasattr(encoded, "last_hidden_state")
            else encoded[0]
        )
        return self.dropout(self.context_adapter(hidden))

    def subject_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.subject(hidden).float()

    def object_logits(
        self,
        hidden: torch.Tensor,
        subject_spans: torch.Tensor,
        parent_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict objects for ``(start,end)`` subject token pairs.

        ``parent_indices`` maps each subject to a row in ``hidden``.  Omitting
        it is valid when there is exactly one subject per encoded row.
        """

        if subject_spans.ndim != 2 or subject_spans.shape[-1] != 2:
            raise ValueError("subject_spans must have shape (n,2)")
        count = int(subject_spans.shape[0])
        if parent_indices is None:
            if count != int(hidden.shape[0]):
                raise ValueError(
                    "parent_indices are required when subjects and rows differ"
                )
            parent_indices = torch.arange(count, device=hidden.device)
        if parent_indices.ndim != 1 or int(parent_indices.shape[0]) != count:
            raise ValueError("parent_indices must have shape (n,)")
        parent_indices = parent_indices.long()
        subject_spans = subject_spans.long()
        selected = hidden.index_select(0, parent_indices)
        rows = torch.arange(count, device=hidden.device)
        start = selected[rows, subject_spans[:, 0]]
        end = selected[rows, subject_spans[:, 1]]
        condition = self.subject_condition(torch.cat([start, end], dim=-1))
        conditioned = self.conditioned_norm(selected + condition[:, None, :])
        residual = self.object(self.dropout(conditioned)).view(
            count,
            hidden.shape[1],
            self.predicates,
            2,
        ).float()
        interaction = self.subject_object_interaction(selected, start, end)
        return residual + interaction

    def score_object_spans(
        self,
        hidden: torch.Tensor,
        subject_spans: torch.Tensor,
        object_spans: torch.Tensor,
        predicate_ids: torch.Tensor,
        parent_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return raw verifier logits for aligned exact span candidates.

        ``subject_spans``, ``object_spans`` and ``predicate_ids`` are all
        candidate-aligned.  ``parent_indices`` maps each candidate to an input
        row and is mandatory whenever candidates and encoded rows differ.
        """

        if subject_spans.ndim != 2 or subject_spans.shape[-1] != 2:
            raise ValueError("subject_spans must have shape (n,2)")
        if object_spans.ndim != 2 or object_spans.shape[-1] != 2:
            raise ValueError("object_spans must have shape (n,2)")
        count = int(subject_spans.shape[0])
        if int(object_spans.shape[0]) != count or int(predicate_ids.numel()) != count:
            raise ValueError("verifier candidate tensors must have the same length")
        if parent_indices is None:
            if count != int(hidden.shape[0]):
                raise ValueError(
                    "parent_indices are required when candidates and rows differ"
                )
            parent_indices = torch.arange(count, device=hidden.device)
        if parent_indices.ndim != 1 or int(parent_indices.shape[0]) != count:
            raise ValueError("parent_indices must have shape (n,)")
        sequence = int(hidden.shape[1])
        subject_spans = subject_spans.long()
        object_spans = object_spans.long()
        predicate_ids = predicate_ids.long()
        if (
            (subject_spans < 0).any()
            or (object_spans < 0).any()
            or (subject_spans >= sequence).any()
            or (object_spans >= sequence).any()
            or (subject_spans[:, 0] > subject_spans[:, 1]).any()
            or (object_spans[:, 0] > object_spans[:, 1]).any()
            or (predicate_ids < 0).any()
            or (predicate_ids >= self.predicates).any()
        ):
            raise ValueError("verifier candidate index is out of range")
        selected = hidden.index_select(0, parent_indices.long())
        rows = torch.arange(count, device=hidden.device)
        return self.span_verifier(
            selected[rows, subject_spans[:, 0]],
            selected[rows, subject_spans[:, 1]],
            selected[rows, object_spans[:, 0]],
            selected[rows, object_spans[:, 1]],
            predicate_ids,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        *,
        subject_spans: torch.Tensor | None = None,
        parent_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(input_ids, attention_mask, token_type_ids)
        result = {"subject": self.subject_logits(hidden), "hidden": hidden}
        if subject_spans is not None:
            result["object"] = self.object_logits(
                hidden,
                subject_spans,
                parent_indices,
            )
        return result


def balanced_token_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    hard_negative_ratio: int | None = None,
) -> torch.Tensor:
    """Give positive and negative token mass equal weight per example."""

    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have identical shapes")
    while mask.ndim < logits.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=torch.bool).expand_as(logits)
    positive = labels.bool() & mask
    negative = ~labels.bool() & mask
    if hard_negative_ratio is not None:
        if hard_negative_ratio < 1:
            raise ValueError("hard_negative_ratio must be positive")
        flat_logits = logits.float().reshape(logits.shape[0], -1)
        flat_positive = positive.reshape(logits.shape[0], -1)
        flat_negative = negative.reshape(logits.shape[0], -1)
        positive_count = flat_positive.sum(-1)
        positive_loss = torch.where(
            flat_positive,
            F.softplus(-flat_logits),
            torch.zeros_like(flat_logits),
        ).sum(-1) / positive_count.clamp_min(1)
        desired_negative_count = (positive_count * hard_negative_ratio).clamp_min(1)
        negative_count = flat_negative.sum(-1)
        desired_negative_count = torch.minimum(
            desired_negative_count,
            negative_count.clamp_min(1),
        )
        maximum_hard = min(
            int(flat_logits.shape[1]),
            max(1, int(desired_negative_count.max().detach().cpu())),
        )
        ranked_negatives = torch.topk(
            torch.where(
                flat_negative,
                flat_logits,
                torch.full_like(flat_logits, float("-inf")),
            ),
            k=maximum_hard,
            dim=-1,
            largest=True,
            sorted=True,
        ).values
        ranks = torch.arange(
            maximum_hard,
            device=logits.device,
        ).unsqueeze(0)
        selected = ranks < desired_negative_count.unsqueeze(1)
        negative_loss = torch.where(
            selected,
            F.softplus(ranked_negatives),
            torch.zeros_like(ranked_negatives),
        ).sum(-1) / desired_negative_count
        has_positive = positive_count.gt(0).float()
        return (
            (positive_loss * has_positive + negative_loss)
            / (has_positive + 1.0)
        ).mean()
    positive_count = positive.reshape(logits.shape[0], -1).sum(-1).clamp_min(1)
    negative_count = negative.reshape(logits.shape[0], -1).sum(-1).clamp_min(1)
    positive_loss = torch.where(
        positive,
        F.softplus(-logits.float()),
        torch.zeros_like(logits, dtype=torch.float32),
    ).reshape(logits.shape[0], -1).sum(-1) / positive_count
    negative_loss = torch.where(
        negative,
        F.softplus(logits.float()),
        torch.zeros_like(logits, dtype=torch.float32),
    ).reshape(logits.shape[0], -1).sum(-1) / negative_count
    has_positive = positive.reshape(logits.shape[0], -1).any(-1).float()
    return ((positive_loss * has_positive + negative_loss) / (has_positive + 1)).mean()


def casrel_loss(
    subject_logits: torch.Tensor,
    subject_labels: torch.Tensor,
    attention_mask: torch.Tensor,
    object_logits: torch.Tensor,
    object_labels: torch.Tensor,
    object_attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    subject_loss = balanced_token_bce(
        subject_logits,
        subject_labels,
        attention_mask,
        hard_negative_ratio=8,
    )
    object_loss = balanced_token_bce(
        object_logits,
        object_labels,
        object_attention_mask,
        hard_negative_ratio=8,
    )
    return subject_loss + object_loss, {
        "subject_loss": subject_loss.detach(),
        "object_loss": object_loss.detach(),
    }


def verifier_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Balanced binary verifier loss with explicit positive/negative mass."""

    if logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError("verifier logits and labels must be aligned one-dimensional tensors")
    labels = labels.float()
    positive = labels > 0.5
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("verifier batch must contain positive and negative candidates")
    positive_loss = F.softplus(-logits[positive]).mean()
    negative_loss = F.softplus(logits[negative]).mean()
    return positive_loss + negative_loss


def pair_boundary_logits(
    starts: torch.Tensor,
    ends: torch.Tensor,
    threshold: float,
    *,
    max_span_tokens: int = 64,
    maximum: int | None = 64,
) -> list[tuple[int, int, float]]:
    """Pair each positive start with its closest admissible positive end."""

    start_indices = (starts > threshold).nonzero(as_tuple=False).flatten().tolist()
    end_indices = (ends > threshold).nonzero(as_tuple=False).flatten().tolist()
    candidates: dict[tuple[int, int], float] = {}
    for start in start_indices:
        matching = [
            end
            for end in end_indices
            if start <= end and end - start + 1 <= max_span_tokens
        ]
        if not matching:
            continue
        end = min(matching)
        candidates[(start, end)] = float(starts[start] + ends[end])
    ranked = sorted(
        ((start, end, score) for (start, end), score in candidates.items()),
        key=lambda item: (-item[2], item[0], item[1]),
    )
    return ranked if maximum is None else ranked[:maximum]


class CandidateDensityError(RuntimeError):
    """Raised instead of silently truncating a verifier candidate population."""


def predicate_object_candidates(
    object_logits: torch.Tensor,
    offsets: Sequence[tuple[int, int]],
    object_threshold: float,
    *,
    maximum_per_subject_predicate: int = 8,
    maximum_per_row: int = 4096,
) -> list[tuple[int, int, int, int, float]]:
    """Enumerate every admissible boundary candidate or fail closed on density.

    The tuple is ``(subject_index, predicate_id, object_start, object_end,
    boundary_score)``.  The fixed caps are validation gates, not ranking caps:
    an overflow raises so a run cannot hide recall/density tradeoffs by dropping
    candidates after seeing logits.
    """

    if object_logits.ndim != 4 or object_logits.shape[-1] != 2:
        raise ValueError("object_logits must have shape (subjects,sequence,predicates,2)")
    if maximum_per_subject_predicate < 1 or maximum_per_row < 1:
        raise ValueError("candidate density caps must be positive")
    if int(object_logits.shape[1]) != len(offsets):
        raise ValueError("object logits and offsets have different sequence lengths")
    candidates: list[tuple[int, int, int, int, float]] = []
    for subject_index in range(int(object_logits.shape[0])):
        for predicate_id in range(int(object_logits.shape[2])):
            spans = pair_boundary_logits(
                object_logits[subject_index, :, predicate_id, 0],
                object_logits[subject_index, :, predicate_id, 1],
                object_threshold,
                maximum=None,
            )
            spans = [
                (start, end, score)
                for start, end, score in spans
                if offsets[start][1] > offsets[start][0]
                and offsets[end][1] > offsets[end][0]
            ]
            if len(spans) > maximum_per_subject_predicate:
                raise CandidateDensityError(
                    "predicate-aware object candidates exceed cap "
                    f"({len(spans)} > {maximum_per_subject_predicate}) for "
                    f"subject={subject_index}, predicate={predicate_id}"
                )
            candidates.extend(
                (subject_index, predicate_id, start, end, score)
                for start, end, score in spans
            )
            if len(candidates) > maximum_per_row:
                raise CandidateDensityError(
                    "row verifier candidates exceed cap "
                    f"({len(candidates)} > {maximum_per_row})"
                )
    return candidates


def decode_verified_casrel(
    text: str,
    offsets: Sequence[tuple[int, int]],
    subject_spans: Sequence[tuple[int, int, float]],
    object_logits: torch.Tensor,
    predicates: Sequence[str],
    object_threshold: float,
    verifier_scores: torch.Tensor | Sequence[float],
    verifier_threshold: float,
    *,
    maximum_per_subject_predicate: int = 8,
    maximum_per_row: int = 4096,
) -> set[tuple[str, str, str]]:
    """Decode boundary candidates accepted by one global verifier threshold."""

    candidates = predicate_object_candidates(
        object_logits,
        offsets,
        object_threshold,
        maximum_per_subject_predicate=maximum_per_subject_predicate,
        maximum_per_row=maximum_per_row,
    )
    scores = torch.as_tensor(verifier_scores, dtype=torch.float32).flatten()
    if int(scores.numel()) != len(candidates):
        raise ValueError("verifier score count differs from boundary candidates")
    triples: set[tuple[str, str, str]] = set()
    for candidate, score in zip(candidates, scores.tolist()):
        if score <= verifier_threshold:
            continue
        subject_index, predicate_id, object_start, object_end, _ = candidate
        subject_start, subject_end, _ = subject_spans[subject_index]
        if (
            offsets[subject_start][1] <= offsets[subject_start][0]
            or offsets[subject_end][1] <= offsets[subject_end][0]
        ):
            continue
        subject = text[offsets[subject_start][0] : offsets[subject_end][1]]
        obj = text[offsets[object_start][0] : offsets[object_end][1]]
        if subject and obj and subject != obj:
            triples.add((subject, str(predicates[predicate_id]), obj))
    return triples


def decode_casrel(
    text: str,
    offsets: Sequence[tuple[int, int]],
    subject_spans: Sequence[tuple[int, int, float]],
    object_logits: torch.Tensor,
    predicates: Sequence[str],
    object_threshold: float,
) -> set[tuple[str, str, str]]:
    """Decode subject-conditioned object logits into exact surface triples."""

    triples: set[tuple[str, str, str]] = set()
    for subject_index, (ss, se, _) in enumerate(subject_spans):
        if (
            offsets[ss][1] <= offsets[ss][0]
            or offsets[se][1] <= offsets[se][0]
        ):
            continue
        subject = text[offsets[ss][0] : offsets[se][1]]
        if not subject:
            continue
        # A self-referential object span (surface text == subject) is not a
        # valid CMeIE triple; it is the dominant systematic decode-layer FP.
        # Compare surface text, not token span: the same entity may appear at
        # multiple token spans, and each fires as a self-loop candidate.
        subject_text = subject
        for predicate_id, predicate in enumerate(predicates):
            objects = pair_boundary_logits(
                object_logits[subject_index, :, predicate_id, 0],
                object_logits[subject_index, :, predicate_id, 1],
                object_threshold,
            )
            for os_, oe, _ in objects:
                if (
                    offsets[os_][1] <= offsets[os_][0]
                    or offsets[oe][1] <= offsets[oe][0]
                ):
                    continue
                obj = text[offsets[os_][0] : offsets[oe][1]]
                if not obj:
                    continue
                # Skip a self-referential object span (same surface text as
                # the subject). A CMeIE triple cannot relate an entity to
                # itself; this removes the systematic self-loop FP class.
                if obj == subject_text:
                    continue
                triples.add((subject, str(predicate), obj))
    return triples


def exact_spo_metric(
    predictions: Iterable[set[tuple[str, str, str]]],
    gold: Iterable[set[tuple[str, str, str]]],
) -> dict[str, float | int]:
    counts = {"tp": 0, "fp": 0, "fn": 0}
    for predicted, expected in zip(predictions, gold):
        counts["tp"] += len(predicted & expected)
        counts["fp"] += len(predicted - expected)
        counts["fn"] += len(expected - predicted)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        **counts,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }
