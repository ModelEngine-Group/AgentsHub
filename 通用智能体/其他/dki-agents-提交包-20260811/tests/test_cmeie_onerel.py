from __future__ import annotations

import torch
import torch.nn as nn

from dki.nlp.cmeie_onerel import (
    CMeIEOneRel,
    HB_TB,
    HB_TE,
    HE_TE,
    LINK_NAMES,
    ONEREL_ARCHITECTURE_VERSION,
    OneRelDensityError,
    OneRelSpanCandidate,
    build_onerel_labels,
    decode_onerel_candidates,
    decode_verified_onerel_candidates,
    dense_onerel_targets,
    enumerate_onerel_span_candidates,
    grouped_balanced_hard_negative_loss,
    grouped_margin_separation_loss,
    grouped_tail_anchor_loss,
    select_fixed_proposal_candidates,
    structured_negative_links,
    structured_triple_candidates,
)


class _Encoder(nn.Module):
    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, hidden)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        del attention_mask, token_type_ids
        return type("Encoded", (), {"last_hidden_state": self.embedding(input_ids)})()


def _offsets(text: str):
    return [(0, 0)] + [(index, index + 1) for index in range(len(text))] + [(0, 0)]


def test_labels_keep_three_relation_specific_corners_and_collisions():
    row = {
        "text": "甲病乙症丙症",
        "spo_list": [
            {"subject": "甲病", "predicate": "临床表现", "object": "乙症"},
            {"subject": "甲病", "predicate": "临床表现", "object": "丙症"},
        ],
    }
    labels = build_onerel_labels(row, _offsets(row["text"]), {"临床表现": 0})
    assert labels.dropped_spo == 0
    assert labels.triples == ((0, 1, 2, 3, 4), (0, 1, 2, 5, 6))
    assert (0, HB_TB, 1, 3) in labels.links
    assert (0, HB_TE, 1, 4) in labels.links
    assert (0, HE_TE, 2, 4) in labels.links
    target, relation, boundary = dense_onerel_targets(labels, 8, 1)
    assert target.shape == (1, len(LINK_NAMES), 8, 8)
    assert int(target.sum()) == 6
    assert relation.tolist() == [1.0]
    assert boundary[1, 0] == 1 and boundary[2, 1] == 1


def test_structured_negatives_are_gold_free_and_deterministic():
    row = {
        "text": "甲病乙症",
        "spo_list": [{"subject": "甲病", "predicate": "临床表现", "object": "乙症"}],
    }
    labels = build_onerel_labels(row, _offsets(row["text"]), {"临床表现": 0, "病因": 1})
    first = structured_negative_links(labels, range(1, 5), 2, seed=42)
    second = structured_negative_links(labels, range(1, 5), 2, seed=42)
    assert first == second
    assert set(first).isdisjoint(labels.links)
    assert (1, HB_TB, 1, 3) in first


def test_verifier_candidates_include_link_ghosts_and_are_deterministic():
    row = {
        "text": "甲病乙症丙症",
        "spo_list": [
            {"subject": "甲病", "predicate": "临床表现", "object": "乙症"},
            {"subject": "甲病", "predicate": "临床表现", "object": "丙症"},
        ],
    }
    labels = build_onerel_labels(
        row,
        _offsets(row["text"]),
        {"临床表现": 0, "病因": 1},
    )
    first = structured_triple_candidates(labels, range(1, 7), 2, seed=42)
    second = structured_triple_candidates(labels, range(1, 7), 2, seed=42)
    assert first == second
    positives, negatives = first
    assert set(positives) == set(labels.triples)
    assert set(negatives).isdisjoint(positives)
    assert (0, 1, 2, 3, 6) in negatives


def test_sparse_and_dense_scores_match():
    torch.manual_seed(7)
    model = CMeIEOneRel(_Encoder(), 8, 2, pair_rank=8, dropout=0.0)
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids)
    hidden = model.encode(ids, mask)
    dense = model.score_dense(hidden)
    sparse = model.score_sparse(
        hidden,
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
        torch.tensor([HB_TB, HE_TE]),
        torch.tensor([1, 2]),
        torch.tensor([3, 0]),
    )
    assert torch.allclose(sparse[0], dense[0, 0, HB_TB, 1, 3], atol=1e-6)
    assert torch.allclose(sparse[1], dense[0, 1, HE_TE, 2, 0], atol=1e-6)


def test_exact_triple_verifier_changes_with_span_and_predicate():
    torch.manual_seed(11)
    model = CMeIEOneRel(_Encoder(), 8, 2, pair_rank=8, dropout=0.0)
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    hidden = model.encode(ids, torch.ones_like(ids))
    scores = model.score_triples(
        hidden,
        torch.tensor([0, 0, 0]),
        torch.tensor([0, 1, 0]),
        torch.tensor([0, 0, 1]),
        torch.tensor([1, 1, 1]),
        torch.tensor([2, 2, 2]),
        torch.tensor([3, 3, 3]),
    )
    assert scores.shape == (3,)
    assert not torch.allclose(scores[0], scores[1])
    assert not torch.allclose(scores[0], scores[2])


def test_zero_initialized_proposal_calibrator_preserves_v3_then_uses_scores():
    assert ONEREL_ARCHITECTURE_VERSION == 4
    torch.manual_seed(13)
    model = CMeIEOneRel(_Encoder(), 8, 2, pair_rank=8, dropout=0.0)
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    hidden = model.encode(ids, torch.ones_like(ids))
    indices = (
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        torch.tensor([1, 1]),
        torch.tensor([2, 2]),
        torch.tensor([3, 3]),
    )
    baseline = model.score_triples(hidden, *indices)
    proposal = torch.tensor([[3.0, 3.0, 3.0], [-3.0, -3.0, -3.0]])
    zero_initialized = model.score_triples(
        hidden, *indices, proposal_scores=proposal
    )
    assert torch.allclose(baseline, zero_initialized, atol=1e-7)
    with torch.no_grad():
        model.proposal_calibrator.weight.fill_(1.0)
    calibrated = model.score_triples(hidden, *indices, proposal_scores=proposal)
    assert calibrated[0] > calibrated[1]


def test_zero_initialized_context_branch_preserves_v3_then_uses_window_context():
    torch.manual_seed(17)
    model = CMeIEOneRel(_Encoder(), 8, 2, pair_rank=8, dropout=0.0)
    model.eval()
    hidden = torch.zeros((2, 5, 8))
    hidden[0, 2] = 4.0
    hidden[1, 2] = -4.0
    indices = (
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
        torch.tensor([1, 1]),
        torch.tensor([1, 1]),
        torch.tensor([3, 3]),
        torch.tensor([3, 3]),
    )
    zero_started = model.score_triples(hidden, *indices)
    assert torch.allclose(zero_started[0], zero_started[1], atol=1e-7)
    with torch.no_grad():
        model.verifier_context_output.weight.fill_(0.25)
    context_enabled = model.score_triples(hidden, *indices)
    assert not torch.allclose(context_enabled[0], context_enabled[1])


def test_verifier_hard_negative_mining_is_balanced_per_source_row():
    logits = torch.tensor([2.0, 1.0, -1.0, 3.0, -2.0, -3.0])
    targets = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    combined = grouped_balanced_hard_negative_loss(
        logits,
        targets,
        groups,
        negative_ratio=1,
        minimum_negatives=1,
    )
    row_losses = [
        grouped_balanced_hard_negative_loss(
            logits[groups == row],
            targets[groups == row],
            torch.zeros(3, dtype=torch.long),
            negative_ratio=1,
            minimum_negatives=1,
        )
        for row in (0, 1)
    ]
    assert torch.allclose(combined, torch.stack(row_losses).mean())


def test_verifier_margin_penalizes_overlapping_positive_and_negative_tails():
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    groups = torch.zeros(4, dtype=torch.long)
    overlapping = grouped_margin_separation_loss(
        torch.tensor([1.0, -1.0, 2.0, -2.0]), targets, groups, margin=1.0
    )
    separated = grouped_margin_separation_loss(
        torch.tensor([4.0, 3.0, 0.0, -1.0]), targets, groups, margin=1.0
    )
    assert separated < overlapping


def test_verifier_tail_anchor_prefers_scores_around_declared_global_grid():
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    groups = torch.zeros(4, dtype=torch.long)
    unanchored = grouped_tail_anchor_loss(
        torch.tensor([0.2, -0.2, 0.3, -0.3]), targets, groups
    )
    anchored = grouped_tail_anchor_loss(
        torch.tensor([2.0, 1.5, -1.5, -2.0]), targets, groups
    )
    assert anchored < unanchored


def test_fixed_proposal_sampling_keeps_gold_and_hnr8_hardest_deterministically():
    gold = {(0, 1, 1, 2, 2)}
    proposed = tuple(
        OneRelSpanCandidate(
            predicate_id=0,
            subject_start=1,
            subject_end=1,
            object_start=index,
            object_end=index,
            relation_score=1.0,
            link_score=float(index) / 10.0,
            boundary_score=1.0,
        )
        for index in range(2, 14)
    )
    first, statistics = select_fixed_proposal_candidates(
        proposed, gold, negative_ratio=8
    )
    second, _ = select_fixed_proposal_candidates(
        tuple(reversed(proposed)), gold, negative_ratio=8
    )
    assert first == second
    assert gold <= set(first)
    assert len(first) == 9
    assert statistics["raw_proposal_candidates"] == 12
    assert statistics["selected_negative_candidates"] == 8
    assert statistics["negative_ratio"] == 8
    assert (0, 1, 1, 13, 13) in first
    assert (0, 1, 1, 5, 5) not in first


def test_candidate_conditioned_sampling_excludes_unreachable_gold():
    reachable = (0, 1, 1, 2, 2)
    unreachable = (0, 1, 1, 9, 9)
    proposed = (
        OneRelSpanCandidate(0, 1, 1, 2, 2, 1.0, 1.0, 1.0),
        OneRelSpanCandidate(0, 1, 1, 3, 3, 0.8, 0.8, 0.8),
    )
    selected, statistics = select_fixed_proposal_candidates(
        proposed,
        {reachable, unreachable},
        negative_ratio=8,
        retain_unproposed_gold=False,
    )
    assert reachable in selected
    assert unreachable not in selected
    assert statistics["reachable_gold_candidates"] == 1
    assert statistics["excluded_unproposed_gold_candidates"] == 1
    assert statistics["retain_unproposed_gold"] == 0


def test_decoder_requires_all_three_links_and_global_auxiliary_gates():
    text = "甲病乙症"
    offsets = _offsets(text)
    candidates = [
        (0, HB_TB, 1, 3, 2.0),
        (0, HB_TE, 1, 4, 2.0),
        (0, HE_TE, 2, 4, 2.0),
    ]
    relation = [2.0]
    boundary = [[-9.0] * 4 for _ in offsets]
    boundary[1][0] = boundary[2][1] = boundary[3][2] = boundary[4][3] = 2.0
    assert decode_onerel_candidates(
        text, offsets, relation, boundary, candidates, ["临床表现"], (0.0, 0.0, 0.0)
    ) == {("甲病", "临床表现", "乙症")}
    assert not decode_onerel_candidates(
        text, offsets, relation, boundary, candidates[:-1], ["临床表现"], (0.0, 0.0, 0.0)
    )


def test_verifier_removes_perfect_link_cross_product_ghosts():
    text = "甲病乙症丙症"
    offsets = _offsets(text)
    candidates = [
        (0, HB_TB, 1, 3, 2.0),
        (0, HB_TE, 1, 4, 2.0),
        (0, HE_TE, 2, 4, 2.0),
        (0, HB_TB, 1, 5, 2.0),
        (0, HB_TE, 1, 6, 2.0),
        (0, HE_TE, 2, 6, 2.0),
    ]
    boundary = [[-9.0] * 4 for _ in offsets]
    boundary[1][0] = boundary[2][1] = 2.0
    for start, end in ((3, 4), (5, 6)):
        boundary[start][2] = boundary[end][3] = 2.0
    proposed = enumerate_onerel_span_candidates(
        [2.0], boundary, candidates, 1, (0.0, 0.0, 0.0)
    )
    # The reverse boundary pair (object_start=5, object_end=4) is invalid,
    # leaving two gold spans and the forward cross-product ghost (3, 6).
    assert len(proposed) == 3
    scores = [
        2.0
        if (candidate.object_start, candidate.object_end) in {(3, 4), (5, 6)}
        else -2.0
        for candidate in proposed
    ]
    assert decode_verified_onerel_candidates(
        text,
        offsets,
        proposed,
        scores,
        ["临床表现"],
        (0.0, 0.0, 0.0, 0.0),
    ) == {
        ("甲病", "临床表现", "乙症"),
        ("甲病", "临床表现", "丙症"),
    }


def test_decoder_density_is_fail_closed():
    with torch.no_grad():
        try:
            decode_onerel_candidates(
                "甲",
                [(0, 1)],
                [1.0],
                [[1.0] * 4],
                [(0, HB_TB, 0, 0, 1.0), (0, HB_TE, 0, 0, 1.0)],
                ["同义词"],
                (0.0, 0.0, 0.0),
                maximum_link_candidates=1,
            )
        except OneRelDensityError:
            pass
        else:
            raise AssertionError("density overflow was silently accepted")
