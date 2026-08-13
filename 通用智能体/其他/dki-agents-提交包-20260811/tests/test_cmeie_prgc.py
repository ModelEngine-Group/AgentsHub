from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from dki.nlp.cmeie_prgc import (
    CMeIEPRGC,
    PRGCDensityError,
    _span_candidates,
    balanced_hard_negative_bce,
    build_prgc_labels,
    build_selected_targets,
    decode_prgc,
    prgc_loss,
    select_training_predicates,
)


class _TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, hidden_size)
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_prgc_labels_keep_repeated_mentions_and_relation_presence():
    row = {
        "text": "甲见乙。甲见丙。",
        "spo_list": [
            {"subject": "甲", "predicate": "关联", "object": "乙"},
            {"subject": "甲", "predicate": "关联", "object": "丙"},
        ],
    }
    labels = build_prgc_labels(
        row,
        [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 0)],
        {"关联": 0},
    )
    assert labels.relation.tolist() == [1.0]
    assert len(labels.triples[0]) == 2
    assert labels.dropped_spo == 0


def test_prgc_selection_includes_all_gold_and_at_most_four_hard_negatives():
    labels = [
        build_prgc_labels(
            {
                "text": "甲乙",
                "spo_list": [
                    {"subject": "甲", "predicate": "p0", "object": "乙"},
                    {"subject": "甲", "predicate": "p1", "object": "乙"},
                ],
            },
            [(0, 0), (0, 1), (1, 2), (0, 0)],
            {f"p{i}": i for i in range(7)},
        )
    ]
    parents, predicates = select_training_predicates(
        torch.tensor([[0.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0]]), labels
    )
    assert parents == [0] * len(predicates)
    assert predicates[:2] == [0, 1]
    assert predicates[2:] == [2, 3, 4, 5]


def test_prgc_model_and_three_losses_accept_ragged_predicate_channels():
    torch.manual_seed(3)
    model = CMeIEPRGC(_TinyEncoder(), 12, 3, predicate_rank=8, correspondence_rank=6)
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    labels = [
        build_prgc_labels(
            {"text": "甲乙", "spo_list": [{"subject": "甲", "predicate": "p0", "object": "乙"}]},
            [(0, 0), (0, 1), (1, 2), (0, 0)],
            {"p0": 0, "p1": 1, "p2": 2},
        ),
        build_prgc_labels(
            {"text": "丙丁", "spo_list": [{"subject": "丙", "predicate": "p1", "object": "丁"}]},
            [(0, 0), (0, 1), (1, 2), (0, 0)],
            {"p0": 0, "p1": 1, "p2": 2},
        ),
    ]
    hidden = model.encode(input_ids, mask)
    relation = model.relation_logits(hidden, mask)
    parents, predicate_ids = select_training_predicates(relation, labels, maximum_hard_non_gold=1)
    channels = model.score_channels(
        hidden,
        parent_indices=torch.tensor(parents),
        predicate_ids=torch.tensor(predicate_ids),
    )
    channels["relation"] = relation
    span, correspondence = build_selected_targets(labels, parents, predicate_ids, 4)
    loss, parts = prgc_loss(
        channels,
        torch.stack([item.relation for item in labels]),
        span,
        correspondence,
        mask,
    )
    assert channels["span"].shape == (4, 4, 4)
    assert channels["correspondence"].shape == (4, 4, 4)
    assert torch.isfinite(loss)
    assert set(parts) == {"relation", "span", "correspondence"}
    loss.backward()
    assert model.subject_boundary.weight.grad is not None
    assert model.object_boundary.weight.grad is not None
    assert model.boundary_token_projection.weight.grad is not None


def test_prgc_boundary_queries_change_token_ranking_by_predicate():
    """Predicate conditioning must alter token ranking, not only a bias."""

    model = CMeIEPRGC(
        _TinyEncoder(hidden_size=8),
        8,
        2,
        predicate_rank=4,
        correspondence_rank=3,
        dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        model.boundary_token_projection.weight.zero_()
        model.boundary_token_projection.bias.zero_()
        model.boundary_token_projection.weight[0, 0] = 1.0
        model.boundary_token_projection.weight[1, 1] = 1.0
        model.predicate_embedding.weight.zero_()
        model.predicate_embedding.weight[0, 0] = 1.0
        model.predicate_embedding.weight[1, 1] = 1.0
        model.subject_boundary.weight.zero_()
        model.subject_boundary.bias.zero_()
        model.subject_boundary.weight[0, 0] = 1.0
        model.subject_boundary.weight[1, 1] = 1.0
        model.subject_boundary_bias.weight.zero_()
        model.subject_boundary_bias.bias.zero_()
    hidden = torch.tensor(
        [
            [
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    scored = model.score_channels(
        hidden,
        parent_indices=torch.tensor([0, 0]),
        predicate_ids=torch.tensor([0, 1]),
    )["span"]
    assert int(scored[0, :, 0].argmax()) == 0
    assert int(scored[1, :, 0].argmax()) == 1


def test_prgc_sparse_priors_start_below_declared_minimum_threshold():
    model = CMeIEPRGC(
        _TinyEncoder(hidden_size=8),
        8,
        2,
        predicate_rank=4,
        correspondence_rank=3,
    )
    assert float(model.span_logit_prior.detach().max()) < -0.5
    assert float(model.correspondence_logit_prior.detach()) < -0.5


def test_prgc_decoder_requires_relation_boundaries_and_correspondence():
    offsets = [(0, 0), (0, 1), (1, 2), (0, 0)]
    relation = torch.tensor([2.0])
    span = torch.full((1, 4, 4), -5.0)
    span[0, 1, 0] = 2.0
    span[0, 1, 1] = 2.0
    span[0, 2, 2] = 2.0
    span[0, 2, 3] = 2.0
    correspondence = torch.full((1, 4, 4), -5.0)
    correspondence[0, 1, 2] = 2.0
    assert decode_prgc(
        "甲乙",
        offsets,
        relation,
        span,
        correspondence,
        ["p"],
        (0.0, 0.0, 0.0),
        schema_constraint=False,
    ) == {("甲", "p", "乙")}


def test_prgc_boundary_decoder_pairs_each_start_with_nearest_end():
    offsets = [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (0, 0)]
    logits = torch.full((6, 2), -5.0)
    logits[1, 0] = 3.0
    logits[2, 0] = 3.0
    logits[2, 1] = 3.0
    logits[4, 1] = 3.0
    candidates = _span_candidates(
        logits,
        offsets,
        start_column=0,
        end_column=1,
        threshold=0.0,
        maximum_candidates=8,
    )
    assert [(start, end) for start, end, _chars, _surface in candidates] == [
        (1, 2),
        (2, 2),
    ]


def test_prgc_density_is_fail_closed_and_loss_keeps_negative_channels_active():
    offsets = [(0, 0), (0, 1), (1, 2), (2, 3), (0, 0)]
    with pytest.raises(PRGCDensityError):
        decode_prgc(
            "甲乙丙",
            offsets,
            torch.tensor([1.0]),
            torch.ones((1, 5, 4)),
            torch.ones((1, 5, 5)),
            ["p"],
            (0.0, 0.0, 0.0),
            schema_constraint=False,
            maximum_span_candidates=1,
        )
    value = balanced_hard_negative_bce(
        torch.tensor([[3.0, 2.0, 1.0]]), torch.zeros((1, 3))
    )
    assert float(value) > 0.0
    logits = torch.tensor([[2.0], [2.0]])
    labels = torch.tensor([[1.0], [0.0]])
    weighted = balanced_hard_negative_bce(
        logits, labels, inactive_channel_weight=0.25
    )
    unweighted = balanced_hard_negative_bce(
        logits, labels, inactive_channel_weight=1.0
    )
    assert float(weighted) < float(unweighted)
