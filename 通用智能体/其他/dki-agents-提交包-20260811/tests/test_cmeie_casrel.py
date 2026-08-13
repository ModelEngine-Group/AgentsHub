from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dki.nlp.cmeie_casrel import (
    CandidateDensityError,
    CMeIECasRel,
    balanced_token_bce,
    build_casrel_labels,
    decode_casrel,
    decode_verified_casrel,
    pair_boundary_logits,
    predicate_object_candidates,
    verifier_bce_loss,
)
from scripts.train_cmeie_casrel import (
    _clip_casrel_gradients,
    _configure_encoder_trainability,
    _cpu_state_dict,
    _joint_stability_passed,
)


class _Encoder(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _LayeredEncoder(nn.Module):
    def __init__(self, hidden_size=8, layers=4):
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        )

    def forward(self, input_ids, **_kwargs):
        hidden = self.embedding(input_ids)
        for layer in self.encoder.layer:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def test_casrel_model_shapes_with_multiple_subjects_per_row():
    model = CMeIECasRel(_Encoder(), 8, predicates=3, dropout=0)
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    attention_mask = torch.ones_like(input_ids)
    hidden = model.encode(input_ids, attention_mask)
    subjects = torch.tensor([[1, 2], [0, 1], [2, 3]])
    parents = torch.tensor([0, 1, 1])

    assert model.subject_logits(hidden).shape == (2, 4, 2)
    assert model.object_logits(hidden, subjects, parents).shape == (3, 4, 3, 2)


def test_context_adapter_is_an_exact_zero_initialised_residual():
    model = CMeIECasRel(_Encoder(), 8, predicates=3, dropout=0)
    hidden = torch.randn(2, 5, 8)

    adapted = model.context_adapter(hidden)

    assert torch.equal(adapted, hidden)


def test_casrel_object_logits_include_explicit_subject_token_interaction():
    model = CMeIECasRel(_Encoder(hidden_size=2), 2, predicates=1, dropout=0)
    with torch.no_grad():
        model.object.weight.zero_()
        model.object.bias.zero_()
        interaction = model.subject_object_interaction
        interaction.subject_query.weight.zero_()
        interaction.subject_query.bias.zero_()
        interaction.object_key.weight.zero_()
        interaction.object_key.bias.zero_()
        # Start-boundary predicate query and start-boundary token key both use
        # hidden dimension zero. Subjects with opposite signs must therefore
        # produce opposite compatibility for the same candidate token.
        interaction.subject_query.weight[0, 0] = 1.0
        interaction.object_key.weight[0, 0] = 1.0

    hidden = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]])
    subjects = torch.tensor([[0, 0], [1, 1]])
    parents = torch.tensor([0, 0])
    logits = model.object_logits(hidden, subjects, parents)

    assert logits[0, 0, 0, 0] > 0
    assert logits[1, 0, 0, 0] < 0


def test_casrel_alignment_keeps_repeated_mentions_and_multiple_predicates():
    row = {
        "text": "甲治乙。甲伴乙。",
        "spo_list": [
            {"subject": "甲", "predicate": "治疗", "object": "乙"},
            {"subject": "甲", "predicate": "并发", "object": "乙"},
        ],
    }
    offsets = [(0, 0)] + [(index, index + 1) for index in range(len(row["text"]))] + [(0, 0)]
    labels = build_casrel_labels(row, offsets, {"治疗": 0, "并发": 1})

    assert labels.dropped_spo == 0
    assert int(labels.subject[:, 0].sum()) == 2
    assert int(labels.subject[:, 1].sum()) == 2
    assert sum(int(value.sum()) for value in labels.objects.values()) == 8
    assert labels.object_spans
    assert all(
        len(values) == 2
        for values in labels.object_spans.values()
    )


def test_balanced_token_bce_rewards_correct_separation():
    labels = torch.tensor([[[1.0], [0.0], [0.0]]])
    mask = torch.ones((1, 3), dtype=torch.long)
    good = torch.tensor([[[6.0], [-6.0], [-6.0]]])
    bad = -good

    assert balanced_token_bce(good, labels, mask) < balanced_token_bce(bad, labels, mask)


def test_balanced_token_bce_can_focus_on_the_hardest_negatives():
    labels = torch.tensor([[[1.0], [0.0], [0.0], [0.0], [0.0]]])
    mask = torch.ones((1, 5), dtype=torch.long)
    logits = torch.tensor([[[4.0], [3.0], [-8.0], [-8.0], [-8.0]]])

    averaged = balanced_token_bce(logits, labels, mask)
    hard_mined = balanced_token_bce(
        logits,
        labels,
        mask,
        hard_negative_ratio=1,
    )

    assert hard_mined > averaged


def test_pair_boundary_logits_uses_nearest_valid_end_and_cap():
    starts = torch.tensor([-2.0, 2.0, -2.0, 3.0, -2.0])
    ends = torch.tensor([-2.0, -2.0, 2.5, -2.0, 4.0])

    assert pair_boundary_logits(starts, ends, 0.0, maximum=1) == [(3, 4, 7.0)]
    assert pair_boundary_logits(starts, ends, 0.0, maximum=2) == [
        (3, 4, 7.0),
        (1, 2, 4.5),
    ]


def test_decode_casrel_emits_exact_surface_triple():
    text = "甲治疗乙"
    offsets = [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (0, 0)]
    logits = torch.full((1, len(offsets), 1, 2), -5.0)
    logits[0, 4, 0, 0] = 4.0
    logits[0, 4, 0, 1] = 4.0

    assert decode_casrel(
        text,
        offsets,
        [(1, 1, 8.0)],
        logits,
        ["治疗"],
        0.0,
    ) == {("甲", "治疗", "乙")}


def test_casrel_gradient_groups_are_clipped_independently():
    model = CMeIECasRel(_Encoder(), 8, predicates=3, dropout=0)
    for parameter in model.encoder.parameters():
        parameter.grad = torch.full_like(parameter, 1000.0)
    for parameter in model.subject.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)
    for module in (
        model.context_adapter,
        model.subject_condition,
        model.conditioned_norm,
        model.object,
        model.subject_object_interaction,
    ):
        for parameter in module.parameters():
            parameter.grad = torch.full_like(parameter, 500.0)

    norms = _clip_casrel_gradients(model, torch, encoder_frozen=False)

    assert norms["encoder"] > 1.0
    assert norms["subject"] > 1.0
    assert norms["object"] > 1.0
    subject_after = torch.sqrt(
        sum(
            (parameter.grad.float() ** 2).sum()
            for parameter in model.subject.parameters()
        )
    )
    assert subject_after <= 1.0001


def test_top_layer_scope_freezes_embeddings_and_lower_encoder_layers():
    model = CMeIECasRel(_LayeredEncoder(), 8, predicates=3, dropout=0)

    selected, scope = _configure_encoder_trainability(model, top_layers=2)

    assert scope == "top_2_transformer_layers"
    assert selected
    assert not model.encoder.embedding.weight.requires_grad
    assert all(
        not parameter.requires_grad
        for layer in model.encoder.encoder.layer[:2]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in model.encoder.encoder.layer[-2:]
        for parameter in layer.parameters()
    )


def test_joint_stability_gate_rejects_quality_collapse_and_extreme_gradients():
    training = {
        "maximum_pre_clip_gradient_norms": {"encoder": 8.0},
        "nonfinite_gradient_steps": 0,
    }
    good = {"metric": {"f1": 0.96}}

    assert _joint_stability_passed(training, good)
    assert not _joint_stability_passed(training, {"metric": {"f1": 0.94}})
    training["maximum_pre_clip_gradient_norms"]["encoder"] = 20_000.0
    assert not _joint_stability_passed(training, good)


def test_cpu_state_dict_is_portable_contiguous_and_detached():
    model = CMeIECasRel(_Encoder(), 8, predicates=3, dropout=0)
    state = _cpu_state_dict(model)

    assert state
    assert all(value.device.type == "cpu" for value in state.values())
    assert all(value.is_contiguous() for value in state.values())
    original = model.subject.weight.detach().clone()
    state["subject.weight"].zero_()
    assert torch.equal(model.subject.weight.detach(), original)


def test_span_verifier_is_candidate_aligned_and_does_not_change_boundaries():
    model = CMeIECasRel(_Encoder(), 8, predicates=3, dropout=0)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    hidden = model.encode(input_ids, torch.ones_like(input_ids))
    subjects = torch.tensor([[0, 0], [1, 1]])
    parents = torch.tensor([0, 0])
    before = model.object_logits(hidden, subjects, parents).detach().clone()
    scores = model.score_object_spans(
        hidden,
        subjects,
        torch.tensor([[2, 2], [3, 3]]),
        torch.tensor([0, 1]),
        parents,
    )
    after = model.object_logits(hidden, subjects, parents)

    assert scores.shape == (2,)
    assert torch.equal(before, after)
    assert torch.equal(scores, torch.zeros_like(scores))


def test_verifier_loss_requires_both_candidate_classes():
    logits = torch.tensor([2.0, -2.0])
    labels = torch.tensor([1.0, 0.0])

    assert verifier_bce_loss(logits, labels) < verifier_bce_loss(-logits, labels)


def test_predicate_object_candidates_fail_closed_instead_of_truncating():
    offsets = [(0, 0), (0, 1), (1, 2), (2, 3), (0, 0)]
    logits = torch.full((1, len(offsets), 1, 2), -3.0)
    logits[0, 1:4, 0, 0] = 2.0
    logits[0, 1:4, 0, 1] = 2.0

    with pytest.raises(CandidateDensityError, match="exceed cap"):
        predicate_object_candidates(
            logits,
            offsets,
            0.0,
            maximum_per_subject_predicate=1,
        )


def test_verified_decoder_requires_matching_verifier_score():
    text = "甲治乙"
    offsets = [(0, 0), (0, 1), (1, 2), (2, 3), (0, 0)]
    logits = torch.full((1, len(offsets), 1, 2), -5.0)
    logits[0, 3, 0, 0] = 3.0
    logits[0, 3, 0, 1] = 3.0

    assert decode_verified_casrel(
        text,
        offsets,
        [(1, 1, 1.0)],
        logits,
        ["治疗"],
        0.0,
        torch.tensor([1.0]),
        0.0,
    ) == {("甲", "治疗", "乙")}
    assert not decode_verified_casrel(
        text,
        offsets,
        [(1, 1, 1.0)],
        logits,
        ["治疗"],
        0.0,
        torch.tensor([-1.0]),
        0.0,
    )
