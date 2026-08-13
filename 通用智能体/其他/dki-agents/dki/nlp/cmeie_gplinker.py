"""Direct CMeIE text-to-SPO model primitives.

Entity spans use a triangular GlobalPointer.  Relation head/tail links are
directional token pairs, so they deliberately use a non-triangular pointer;
reusing the entity mask here would silently discard links whose object precedes
its subject in text.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from dki.nlp.globalpointer import EfficientGlobalPointer, GlobalPointer


def _scale_positive_loss(
    positive_loss: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    """Scale positive loss without Ascend's pathological host-scalar path.

    The CMeIE fit-only audit selected an integer weight of four from measured
    positive/negative gradient mass. Repeated addition operates only on the
    tiny ``(batch, heads)`` loss tensor and avoids the severe runtime/RSS
    regression observed with ``npu_tensor * Python_float``.
    """
    integer_weight = int(positive_weight)
    if positive_weight == integer_weight and 1 <= integer_weight <= 16:
        scaled = positive_loss
        for _ in range(1, integer_weight):
            scaled = scaled + positive_loss
        return scaled
    return positive_loss * positive_weight


class RelationPairPointer(nn.Module):
    """Efficient directional pointer over subject-token/object-token pairs."""

    def __init__(self, hidden_size: int, heads: int, inner_dim: int = 64):
        super().__init__()
        self.heads = int(heads)
        self.inner_dim = int(inner_dim)
        self.q = nn.Linear(hidden_size, inner_dim)
        self.k = nn.Linear(hidden_size, inner_dim)
        self.bias = nn.Linear(hidden_size, heads * 2)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        q, k = self.q(hidden), self.k(hidden)
        logits = torch.einsum("bmd,bnd->bmn", q, k) / (self.inner_dim ** 0.5)
        bias = self.bias(hidden).view(hidden.size(0), hidden.size(1), self.heads, 2)
        logits = logits[:, None] + bias[..., 0].permute(0, 2, 1)[:, :, :, None] / 2
        logits = logits + bias[..., 1].permute(0, 2, 1)[:, :, None, :] / 2
        positions = torch.arange(hidden.size(1), device=hidden.device)
        lengths = attention_mask.long().sum(dim=1).clamp_min(1)
        valid = attention_mask.bool()
        valid &= positions[None, :] != 0
        valid &= positions[None, :] != (lengths - 1)[:, None]
        return logits.float().masked_fill(~(valid[:, None, :, None] & valid[:, None, None, :]), -1e12)


class DirectionalGlobalPointer(nn.Module):
    """Classic per-head Q/K pointer without a lower-triangular restriction.

    EfficientGlobalPointer shares its Q/K projection across all labels.  That
    sharing is attractive for two entity-role heads, but can make the gradient
    from dozens of empty typed/predicate heads dominate the sparse positives.
    This diagnostic/product candidate gives each head an independent Q/K
    projection while preserving the directional subject/object token matrix.
    """

    def __init__(self, hidden_size: int, heads: int, inner_dim: int = 64):
        super().__init__()
        self.heads = int(heads)
        self.inner_dim = int(inner_dim)
        self.dense = nn.Linear(
            hidden_size,
            self.heads * self.inner_dim * 2,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        projected = self.dense(hidden).view(
            batch,
            sequence,
            self.heads,
            self.inner_dim * 2,
        )
        query = projected[..., : self.inner_dim]
        key = projected[..., self.inner_dim :]
        logits = torch.einsum(
            "bmhd,bnhd->bhmn",
            query,
            key,
        ) / (self.inner_dim ** 0.5)
        positions = torch.arange(sequence, device=hidden.device)
        lengths = attention_mask.long().sum(dim=1).clamp_min(1)
        valid = attention_mask.bool()
        valid &= positions[None, :] != 0
        valid &= positions[None, :] != (lengths - 1)[:, None]
        valid_pairs = (
            valid[:, None, :, None]
            & valid[:, None, None, :]
        )
        return logits.float().masked_fill(~valid_pairs, -1e12)


class CMeIEGPLinker(nn.Module):
    """Encoder + entity / relation-head / relation-tail pointer heads."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        entity_labels: int,
        predicates: int,
        inner_dim: int = 64,
        pointer_variant: str = "efficient",
    ):
        super().__init__()
        self.encoder = encoder
        self.pointer_variant = str(pointer_variant).casefold()
        if self.pointer_variant == "efficient":
            self.entity = EfficientGlobalPointer(
                hidden_size,
                entity_labels,
                inner_dim,
            )
            relation_factory = RelationPairPointer
        elif self.pointer_variant == "classic":
            self.entity = GlobalPointer(
                hidden_size,
                entity_labels,
                inner_dim,
            )
            relation_factory = DirectionalGlobalPointer
        else:
            raise ValueError(
                f"unsupported GPLinker pointer variant: {pointer_variant}"
            )
        self.relation_head = relation_factory(
            hidden_size,
            predicates,
            inner_dim,
        )
        self.relation_tail = relation_factory(
            hidden_size,
            predicates,
            inner_dim,
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        encoded = self.encoder(input_ids, **kwargs)
        hidden = encoded.last_hidden_state if hasattr(encoded, "last_hidden_state") else encoded[0]
        return {
            "entity": self.entity(hidden, attention_mask),
            "relation_head": self.relation_head(hidden, attention_mask),
            "relation_tail": self.relation_tail(hidden, attention_mask),
        }


def pointer_group_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    head_reduction: str = "sum",
    positive_weight: float = 1.0,
    positive_margin: float = 0.0,
) -> torch.Tensor:
    """Sparse multilabel loss with official GPLinker head reduction.

    Losses are summed across pointer heads inside each sample and then averaged
    across the batch.  Averaging over ``batch * heads`` makes the rare positive
    relation head roughly 44 times weaker when CMeIE has 44 predicates, and
    permits the degenerate all-negative solution observed in the seed-42
    calibration audit.
    """
    y_true = labels.float()
    y_pred = logits.float()
    positive_weight = float(positive_weight)
    if not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("positive_weight must be finite and positive")
    positive_margin = float(positive_margin)
    if not math.isfinite(positive_margin) or positive_margin < 0:
        raise ValueError("positive_margin must be finite and non-negative")
    batch, heads = y_pred.shape[:2]
    reduction = str(head_reduction).casefold()
    if reduction not in {
        "sum",
        "balanced",
        "active_margin",
        "active_bce",
        "balanced_bce",
        "global",
    }:
        raise ValueError(
            f"unsupported pointer head reduction: {head_reduction}"
        )
    if reduction == "global" and positive_margin != 0.0:
        raise ValueError(
            "positive_margin is unsupported with global reduction"
        )
    if reduction in {"active_bce", "balanced_bce"} and (
        positive_weight != 1.0 or positive_margin != 0.0
    ):
        raise ValueError(
            "BCE reductions require default positive weight and margin"
        )
    y_true = y_true.reshape(batch, heads, -1)
    y_pred = y_pred.reshape(batch, heads, -1)
    if reduction in {"active_bce", "balanced_bce"}:
        positive = y_true.bool()
        valid = y_pred > -1e11
        negative = valid & ~positive
        active = positive.any(dim=-1)
        positive_count = positive.sum(dim=-1).clamp_min(1)
        negative_count = negative.sum(dim=-1).clamp_min(1)
        positive_loss = torch.where(
            positive,
            F.softplus(-y_pred),
            torch.zeros_like(y_pred),
        ).sum(dim=-1) / positive_count
        negative_loss = torch.where(
            negative,
            F.softplus(y_pred),
            torch.zeros_like(y_pred),
        ).sum(dim=-1) / negative_count
        head_loss = positive_loss + negative_loss
        active_count = active.sum(dim=1).clamp_min(1)
        active_loss = (
            (head_loss * active).sum(dim=1)
            / active_count
        )
        if reduction == "active_bce":
            return active_loss.mean()

        inactive = ~active
        inactive_count = inactive.sum(dim=1).clamp_min(1)
        inactive_loss = (
            (negative_loss * inactive).sum(dim=1)
            / inactive_count
        )
        has_active = active.any(dim=1).to(head_loss.dtype)
        has_inactive = inactive.any(dim=1).to(head_loss.dtype)
        numerator = (
            active_loss * has_active
            + inactive_loss * has_inactive
        )
        denominator = (
            has_active + has_inactive
        ).clamp_min(1)
        return (numerator / denominator).mean()

    if reduction in {"sum", "balanced", "active_margin"}:
        signed = (1 - 2 * y_true) * y_pred
        y_pred_neg = signed - y_true * 1e12
        y_pred_pos = signed - (1 - y_true) * 1e12
        zeros = torch.zeros_like(signed[..., :1])
        positive_floor = (
            zeros
            if positive_margin == 0.0
            else torch.full_like(zeros, -positive_margin)
        )
        negative_loss = torch.logsumexp(
            torch.cat([y_pred_neg, zeros], dim=-1),
            dim=-1,
        )
        positive_loss = torch.logsumexp(
            torch.cat([y_pred_pos, positive_floor], dim=-1),
            dim=-1,
        )
        head_loss = negative_loss + _scale_positive_loss(
            positive_loss,
            positive_weight,
        )
        if reduction == "sum":
            return head_loss.sum(dim=1).mean()

        active = y_true.bool().any(dim=-1)
        if reduction == "active_margin":
            active_count = active.sum(dim=1).clamp_min(1)
            return (
                (head_loss * active).sum(dim=1)
                / active_count
            ).mean()

        inactive = ~active
        active_count = active.sum(dim=1).clamp_min(1)
        inactive_count = inactive.sum(dim=1).clamp_min(1)
        active_loss = (
            (head_loss * active).sum(dim=1)
            / active_count
        )
        inactive_loss = (
            (head_loss * inactive).sum(dim=1)
            / inactive_count
        )
        has_active = active.any(dim=1)
        has_inactive = inactive.any(dim=1)
        denominator = (
            has_active.to(head_loss.dtype)
            + has_inactive.to(head_loss.dtype)
        ).clamp_min(1)
        return (
            (
                active_loss * has_active
                + inactive_loss * has_inactive
            )
            / denominator
        ).mean()

    # Exact sparse multilabel categorical cross-entropy identity.  One
    # hierarchical logsumexp covers all logits in a sample; positive logits
    # are gathered, and their probability mass is removed from the total to
    # obtain the negative term.  This is algebraically identical to masking
    # two dense copies of (head, start, end), with substantially lower memory
    # traffic on Ascend NPU.
    # Ascend's reduction kernel for the exact seq² trailing dimension is
    # pathologically slow.  Appending an effectively negative-infinite
    # sentinel preserves logsumexp while selecting the same fast seq²+1
    # kernel shape already exercised by the per-head implementation.
    head_sentinel = y_pred.new_full(
        (*y_pred.shape[:-1], 1),
        -1e12,
    )
    all_by_head = torch.logsumexp(
        torch.cat([y_pred, head_sentinel], dim=-1),
        dim=-1,
    )
    zero_by_sample = torch.zeros_like(all_by_head[:, :1])
    all_loss = torch.logsumexp(
        torch.cat([all_by_head, zero_by_sample], dim=-1),
        dim=-1,
    )
    sample_losses = []
    epsilon = torch.finfo(y_pred.dtype).eps
    positive_mask = y_true.bool()
    for index in range(batch):
        positive_values = y_pred[index][positive_mask[index]]
        if positive_values.numel():
            zero = positive_values.new_zeros((1,))
            positive_loss = torch.logsumexp(
                torch.cat([-positive_values, zero], dim=0),
                dim=0,
            )
            log_positive_mass = (
                torch.logsumexp(positive_values, dim=0)
                - all_loss[index]
            )
            positive_mass = torch.exp(log_positive_mass).clamp(
                max=1.0 - epsilon
            )
            negative_loss = (
                all_loss[index]
                + torch.log1p(-positive_mass)
            )
        else:
            positive_loss = all_loss[index].new_zeros(())
            negative_loss = all_loss[index]
        sample_losses.append(
            negative_loss
            + _scale_positive_loss(positive_loss, positive_weight)
        )
    return torch.stack(sample_losses).mean()


def gplinker_loss(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    head_reduction: str = "sum",
    positive_weight: float = 1.0,
    positive_margin: float = 0.0,
) -> torch.Tensor:
    """Equal-weight entity/head/tail groups with summed type heads."""
    return (
        pointer_group_loss(
            outputs["entity"],
            labels["entity"],
            head_reduction,
            positive_weight,
            positive_margin,
        )
        + pointer_group_loss(
            outputs["relation_head"],
            labels["relation_head"],
            head_reduction,
            positive_weight,
            positive_margin,
        )
        + pointer_group_loss(
            outputs["relation_tail"],
            labels["relation_tail"],
            head_reduction,
            positive_weight,
            positive_margin,
        )
    ) / 3.0


def decode_spo_token_links(
    entity_spans: Iterable[tuple[int, int, int]],
    head_links: Iterable[tuple[int, int, int]],
    tail_links: Iterable[tuple[int, int, int]],
) -> set[tuple[int, int, int, int, int, int, int]]:
    """Return typed (subj label/span, predicate, obj label/span) candidates."""
    heads = {(predicate, s, o) for predicate, s, o in head_links}
    tails = {(predicate, s, o) for predicate, s, o in tail_links}
    entities = list(entity_spans)
    decoded = set()
    for subject_type, subject_start, subject_end in entities:
        for object_type, object_start, object_end in entities:
            for predicate, s_head, o_head in heads:
                if (s_head, o_head) == (subject_start, object_start) and (predicate, subject_end, object_end) in tails:
                    decoded.add((subject_type, subject_start, subject_end, predicate, object_type, object_start, object_end))
    return decoded
