"""Predicate-conditioned CMeIE span/link unit contracts."""

import pytest
import torch

from dki.nlp.cmeie_predicate_span import (
    LOSS_MODE_BALANCED_PER_HEAD_SPARSE_LOGSUMEXP,
    PredicateSpanThresholdGridDecode,
    ThresholdCombinationDensityRejection,
    ThresholdDensityRejection,
    balanced_per_head_sparse_logsumexp_loss,
    build_predicate_span_labels,
    decode_predicate_span_threshold_grid,
    decode_predicate_spans,
    hard_negative_pointer_loss,
    predicate_span_loss,
)
from scripts.train_cmeie_predicate_span import (
    _CalibrationGrid,
    _set_training_mode,
)


def test_predicate_span_labels_keep_repeated_same_sentence_pairs():
    row = {
        "text": "甲用乙，甲也用乙。",
        "spo_list": [
            {"subject": "甲", "predicate": "治疗", "object": "乙"},
        ],
    }
    offsets = [(0, 0)] + [(index, index + 1) for index in range(len(row["text"]))] + [(0, 0)]
    labels, dropped = build_predicate_span_labels(row, offsets, {"治疗": 0})

    assert dropped == 0
    assert int(labels["subject"].sum()) == 2
    assert int(labels["object"].sum()) == 2
    assert int(labels["link"].sum()) == 2
    assert labels["subject"].dtype == torch.bool


def test_predicate_span_labels_use_only_nearest_cross_sentence_fallback():
    row = {
        "text": "A。A。B。B",
        "spo_list": [
            {"subject": "A", "predicate": "rel", "object": "B"},
        ],
    }
    offsets = [(0, 0)] + [
        (index, index + 1) for index in range(len(row["text"]))
    ] + [(0, 0)]

    labels, dropped = build_predicate_span_labels(
        row, offsets, {"rel": 0}
    )
    strict, strict_dropped = build_predicate_span_labels(
        row,
        offsets,
        {"rel": 0},
        allow_cross_sentence_fallback=False,
    )

    assert dropped == 0
    assert int(labels["subject"].sum()) == 1
    assert int(labels["object"].sum()) == 1
    assert int(labels["link"].sum()) == 1
    assert labels["link"][0, 3, 5]
    assert strict_dropped == 1
    assert int(strict["link"].sum()) == 0


def test_hard_negative_loss_is_finite_and_rewards_positive_margin():
    labels = torch.zeros((1, 2, 4, 4))
    labels[0, 0, 1, 1] = 1
    weak = torch.zeros_like(labels)
    strong = weak.clone()
    strong[0, 0, 1, 1] = 4
    strong[0, :, 2:, :] = -4

    weak_loss = hard_negative_pointer_loss(weak, labels, negatives_per_head=4)
    strong_loss = hard_negative_pointer_loss(strong, labels, negatives_per_head=4)
    assert torch.isfinite(weak_loss)
    assert torch.isfinite(strong_loss)
    assert strong_loss < weak_loss


def test_hard_negative_loss_weights_every_inactive_head_before_mean():
    logits = torch.zeros((1, 4, 1, 2))
    labels = torch.zeros_like(logits, dtype=torch.bool)
    labels[0, 0, 0, 0] = True

    loss = hard_negative_pointer_loss(
        logits,
        labels,
        negatives_per_head=2,
        inactive_weight=0.1,
    )
    log_two = torch.log(torch.tensor(2.0))
    expected = (2 * log_two + 3 * 0.1 * log_two) / (1 + 3 * 0.1)

    assert torch.isclose(loss, expected, atol=1e-6)


def test_balanced_sparse_logsumexp_is_finite_with_correct_gradient_directions():
    logits = torch.zeros((1, 3, 1, 3), requires_grad=True)
    labels = torch.zeros_like(logits, dtype=torch.bool)
    labels[0, 0, 0, 0] = True

    loss = balanced_per_head_sparse_logsumexp_loss(
        logits,
        labels,
        inactive_weight=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 0, 0] < 0
    assert logits.grad[0, 0, 0, 1] > 0
    assert logits.grad[0, 1, 0, 1] > 0


def test_balanced_sparse_logsumexp_preserves_per_head_weight_ratio():
    def gradient_ratio(heads: int) -> float:
        logits = torch.zeros((1, heads, 1, 3), requires_grad=True)
        with torch.no_grad():
            # Give active and inactive heads the same valid-negative set so
            # their gradient ratio isolates only the requested head weights.
            logits[0, 1:, 0, 0] = -1e12
        labels = torch.zeros_like(logits, dtype=torch.bool)
        labels[0, 0, 0, 0] = True
        loss = balanced_per_head_sparse_logsumexp_loss(
            logits,
            labels,
            inactive_weight=0.1,
        )
        loss.backward()
        active_negative = float(logits.grad[0, 0, 0, 1])
        inactive_negative = float(logits.grad[0, 1, 0, 1])
        return active_negative / inactive_negative

    two_heads = gradient_ratio(2)
    six_heads = gradient_ratio(6)
    assert two_heads == pytest.approx(10.0, rel=1e-5)
    assert six_heads == pytest.approx(10.0, rel=1e-5)
    assert six_heads == pytest.approx(two_heads, rel=1e-5)


def test_predicate_span_rejects_unknown_loss_mode():
    outputs = {
        name: torch.zeros((1, 1, 1, 1))
        for name in ("subject", "object", "link")
    }
    labels = {
        name: torch.zeros_like(value, dtype=torch.bool)
        for name, value in outputs.items()
    }
    with pytest.raises(ValueError, match="unsupported predicate-span loss mode"):
        predicate_span_loss(
            outputs,
            labels,
            loss_mode=LOSS_MODE_BALANCED_PER_HEAD_SPARSE_LOGSUMEXP + "_bad",
        )


def test_sparse_threshold_grid_matches_repeated_dense_decode():
    text = "AxxB"
    offsets = [(0, 0)] + [
        (index, index + 1) for index in range(len(text))
    ] + [(0, 0)]
    shape = (1, len(offsets), len(offsets))
    outputs = {
        name: torch.full(shape, -5.0)
        for name in ("subject", "object", "link")
    }
    outputs["subject"][0, 1, 1] = 0.25
    outputs["object"][0, 4, 4] = 0.75
    outputs["link"][0, 1, 4] = 1.25
    values = (-0.5, 0.5, 1.0)

    result = decode_predicate_span_threshold_grid(
        text, offsets, outputs, ["rel"], values
    )
    actual = result.decoded
    expected = {
        thresholds: decode_predicate_spans(
            text, offsets, outputs, ["rel"], thresholds
        )
        for thresholds in actual
    }
    assert actual == expected
    assert result.rejected_combinations == ()
    assert result.density_rejections == ()

    with pytest.raises(OverflowError, match="candidate density"):
        decode_predicate_span_threshold_grid(
            text,
            offsets,
            {name: torch.zeros_like(value) for name, value in outputs.items()},
            ["rel"],
            (-0.5,),
            maximum_active_candidates=1,
        )


def test_density_rejects_only_unsafe_low_threshold_combinations():
    text = "AxxB"
    offsets = [(0, 0)] + [
        (index, index + 1) for index in range(len(text))
    ] + [(0, 0)]
    shape = (1, len(offsets), len(offsets))
    outputs = {
        name: torch.zeros(shape)
        for name in ("subject", "object", "link")
    }
    outputs["subject"][0, 1, 1] = 0.75
    outputs["object"][0, 4, 4] = 0.75
    outputs["link"][0, 1, 4] = 0.75
    values = (-0.5, 0.5)

    result = decode_predicate_span_threshold_grid(
        text,
        offsets,
        outputs,
        ["rel"],
        values,
        maximum_active_candidates=1,
    )

    assert set(result.decoded) == {(0.5, 0.5, 0.5)}
    assert result.decoded[(0.5, 0.5, 0.5)] == decode_predicate_spans(
        text, offsets, outputs, ["rel"], (0.5, 0.5, 0.5)
    )
    assert len(result.rejected_combinations) == 7
    assert {
        (item.head, item.threshold) for item in result.density_rejections
    } == {
        ("subject", -0.5),
        ("object", -0.5),
        ("link", -0.5),
    }


def test_join_work_rejects_only_unsafe_threshold_combinations():
    text = "ABCDE"
    offsets = [(0, 0)] + [
        (index, index + 1) for index in range(len(text))
    ] + [(0, 0)]
    shape = (1, len(offsets), len(offsets))
    outputs = {
        name: torch.full(shape, -5.0)
        for name in ("subject", "object", "link")
    }
    outputs["subject"][0, 1, 1] = 0.75
    outputs["subject"][0, 1, 2] = 0.25
    outputs["object"][0, 4, 4] = 0.75
    outputs["object"][0, 4, 5] = 0.25
    outputs["link"][0, 1, 4] = 0.75

    result = decode_predicate_span_threshold_grid(
        text,
        offsets,
        outputs,
        ["rel"],
        (0.0, 0.5),
        maximum_active_candidates=10,
        maximum_join_work=2,
    )

    rejected = {(0.0, 0.0, 0.0), (0.0, 0.0, 0.5)}
    assert set(result.rejected_combinations) == rejected
    assert set(result.decoded) == {
        (subject, obj, link)
        for subject in (0.0, 0.5)
        for obj in (0.0, 0.5)
        for link in (0.0, 0.5)
        if (subject, obj, link) not in rejected
    }
    assert {
        item.thresholds for item in result.combination_density_rejections
    } == rejected
    assert all(
        item.join_work == 4
        for item in result.combination_density_rejections
    )
    assert result.join_work_by_combination[(0.0, 0.5, 0.0)] == 2
    assert result.active_candidates_by_combination[(0.5, 0.5, 0.5)] == (
        1,
        1,
        1,
    )


def test_calibration_grid_removes_prior_counts_for_later_density_rejection():
    low = (-0.5, -0.5, -0.5)
    high = (0.5, 0.5, 0.5)
    grid = _CalibrationGrid([low, high])
    grid.observe(
        PredicateSpanThresholdGridDecode(
            decoded={
                low: {
                    ("bad-1", "rel", "pair"),
                    ("bad-2", "rel", "pair"),
                },
                high: set(),
            },
            rejected_combinations=(),
            density_rejections=(),
            active_candidates_by_combination={
                low: (9, 9, 9),
                high: (1, 1, 1),
            },
            join_work_by_combination={low: 99, high: 1},
        ),
        set(),
        calibration_row=1,
    )
    assert grid.counts[low]["fp"] == 2

    grid.observe(
        PredicateSpanThresholdGridDecode(
            decoded={high: {("gold", "rel", "pair")}},
            rejected_combinations=(low,),
            density_rejections=(
                ThresholdDensityRejection(
                    head="link",
                    threshold=-0.5,
                    active_candidates=2,
                ),
            ),
            active_candidates_by_combination={high: (2, 2, 2)},
            join_work_by_combination={high: 2},
        ),
        {("gold", "rel", "pair")},
        calibration_row=2,
    )

    assert low not in grid.counts
    assert grid.counts[high]["tp"] == 1
    report = grid.density_report(selected_thresholds=high)
    assert report["declared_threshold_combination_count"] == 2
    assert report["density_rejected_combinations"] == [list(low)]
    assert report["remaining_threshold_combination_count"] == 1
    assert report["density_rejected_thresholds"] == [
        {
            "head": "link",
            "threshold": -0.5,
            "maximum_active_candidates_observed": 2,
            "first_rejected_calibration_row": 2,
        }
    ]
    assert report["grid_maximum_predictions_per_example"] == 1
    assert report["selected_maximum_predictions_per_example"] == 1
    assert report["grid_maximum_join_work_per_example"] == 2
    assert report["selected_maximum_join_work_per_example"] == 2
    assert report["grid_maximum_active_candidates_per_head"] == {
        "subject": 2,
        "object": 2,
        "link": 2,
    }
    assert [item["thresholds"] for item in report["per_combination_density"]] == [
        list(high)
    ]


def test_calibration_grid_records_join_rejection_and_clears_history():
    low = (0.0, 0.0, 0.0)
    high = (0.5, 0.5, 0.5)
    grid = _CalibrationGrid([low, high])
    grid.observe(
        PredicateSpanThresholdGridDecode(
            decoded={low: {("bad", "rel", "pair")}, high: set()},
            rejected_combinations=(),
            density_rejections=(),
            active_candidates_by_combination={
                low: (4, 4, 1),
                high: (1, 1, 1),
            },
            join_work_by_combination={low: 16, high: 1},
        ),
        set(),
        calibration_row=1,
    )
    grid.observe(
        PredicateSpanThresholdGridDecode(
            decoded={high: set()},
            rejected_combinations=(low,),
            density_rejections=(),
            active_candidates_by_combination={high: (1, 1, 1)},
            join_work_by_combination={high: 1},
            combination_density_rejections=(
                ThresholdCombinationDensityRejection(
                    thresholds=low,
                    join_work=25,
                    maximum_join_work=20,
                ),
            ),
        ),
        set(),
        calibration_row=2,
    )

    report = grid.density_report(selected_thresholds=high)
    assert low not in grid.counts
    assert report["grid_maximum_join_work_per_example"] == 1
    assert report["join_work_rejected_combinations"] == [
        {
            "thresholds": list(low),
            "maximum_join_work_observed": 25,
            "maximum_join_work_allowed": 20,
            "first_rejected_calibration_row": 2,
        }
    ]


def test_calibration_grid_fails_closed_when_all_combinations_disappear():
    only = (0.0, 0.0, 0.0)
    grid = _CalibrationGrid([only])
    with pytest.raises(OverflowError, match="every declared threshold"):
        grid.observe(
            PredicateSpanThresholdGridDecode(
                decoded={},
                rejected_combinations=(only,),
                density_rejections=(
                    ThresholdDensityRejection(
                        head="link",
                        threshold=0.0,
                        active_candidates=2,
                    ),
                ),
            ),
            set(),
            calibration_row=1,
        )


def test_decode_allows_only_unique_cross_sentence_surface_fallback():
    text = "A。A。B。B"
    offsets = [(0, 0)] + [
        (index, index + 1) for index in range(len(text))
    ] + [(0, 0)]
    shape = (1, len(offsets), len(offsets))
    subject = torch.full(shape, -5.0)
    obj = torch.full(shape, -5.0)
    link = torch.full(shape, -5.0)
    for start in (1, 3):
        subject[0, start, start] = 5
    for start in (5, 7):
        obj[0, start, start] = 5

    link[0, 1, 7] = 5
    outputs = {"subject": subject, "object": obj, "link": link}
    assert decode_predicate_spans(
        text, offsets, outputs, ["rel"], (0.0, 0.0, 0.0)
    ) == set()

    link[0, 1, 7] = -5
    link[0, 3, 5] = 5
    expected = {("A", "rel", "B")}
    assert decode_predicate_spans(
        text, offsets, outputs, ["rel"], (0.0, 0.0, 0.0)
    ) == expected
    grid = decode_predicate_span_threshold_grid(
        text,
        offsets,
        outputs,
        ["rel"],
        (0.0,),
        maximum_join_work=1,
    )
    assert grid.decoded[(0.0, 0.0, 0.0)] == expected
    assert grid.join_work_by_combination[(0.0, 0.0, 0.0)] == 1


def test_frozen_head_only_mode_keeps_encoder_in_eval():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(torch.nn.Dropout(0.5))
            self.head = torch.nn.Linear(1, 1)

    model = Model()
    _set_training_mode(model, encoder_trainable=False)
    assert model.training
    assert not model.encoder.training
    assert model.head.training

    _set_training_mode(model, encoder_trainable=True)
    assert model.training
    assert model.encoder.training
    assert model.head.training


def test_decode_requires_matching_predicate_link_and_same_sentence():
    text = "甲治乙。丙治丁。"
    offsets = [(0, 0)] + [(index, index + 1) for index in range(len(text))] + [(0, 0)]
    shape = (1, len(offsets), len(offsets))
    subject = torch.full(shape, -5.0)
    obj = torch.full(shape, -5.0)
    link = torch.full(shape, -5.0)
    subject[0, 1, 1] = 5
    subject[0, 5, 5] = 5
    obj[0, 3, 3] = 5
    obj[0, 7, 7] = 5
    link[0, 1, 3] = 5
    link[0, 5, 7] = 5

    decoded = decode_predicate_spans(
        text,
        offsets,
        {"subject": subject, "object": obj, "link": link},
        ["治疗"],
        (0.0, 0.0, 0.0),
    )
    assert decoded == {("甲", "治疗", "乙"), ("丙", "治疗", "丁")}
