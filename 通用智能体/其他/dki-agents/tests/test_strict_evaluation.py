from __future__ import annotations

import pytest

from dki.nlp.evaluation import SpanValidationError, evaluate_both, evaluate_spans, explicit_spans


def test_exact_metrics_include_type_safety():
    gold = {"d": [(0, 2, "Disease"), (4, 6, "Treatment")]}
    pred = {"d": [(0, 2, "Disease"), (4, 5, "Treatment"), (8, 9, "Drug")]}
    result = evaluate_spans(pred, gold)
    assert result["tp"] == 1
    assert result["fp"] == 2
    assert result["fn"] == 1
    assert result["by_type"]["Treatment"]["recall"] == 0.0
    assert result["worst_type_f1"] == 0.0
    assert set(result["worst_types"]) == {"Treatment"}


def test_relaxed_matching_is_maximum_and_order_independent():
    # p0 can match both gold spans, p1 only g0. Greedy p0->g0 would score 1;
    # maximum matching finds p1->g0 and p0->g1, scoring 2.
    gold = {"d": [(0, 3, "Disease"), (3, 6, "Disease")]}
    pred = {"d": [(2, 4, "Disease"), (0, 1, "Disease")]}
    first = evaluate_spans(pred, gold, match="overlap")
    second = evaluate_spans({"d": list(reversed(pred["d"]))}, {"d": list(reversed(gold["d"]))}, match="overlap")
    assert first["tp"] == 2
    assert first == second


def test_exact_and_relaxed_are_separately_named():
    result = evaluate_both({"d": [(0, 2, "Exam")]}, {"d": [(1, 3, "Exam")]})
    assert result["primary_exact"]["tp"] == 0
    assert result["relaxed_diagnostic"]["tp"] == 1
    assert result["primary_exact"]["match"] == "exact_span+type"


def test_explicit_spans_reject_ambiguous_mention_only_gold():
    with pytest.raises(SpanValidationError, match="no explicit span"):
        explicit_spans("痛痛", [{"mention": "痛", "type": "Symptom"}])


def test_explicit_spans_validate_mention_and_unicode_offsets():
    text = "甲🙂乙\r\n"
    spans = explicit_spans(text, [{"span": [1, 2], "mention": "🙂", "type": "Symptom"}])
    assert spans == [(1, 2, "Symptom")]
    with pytest.raises(SpanValidationError, match="mention"):
        explicit_spans(text, [{"span": [0, 1], "mention": "乙", "type": "Symptom"}])


def test_predictions_for_unknown_document_are_false_positives():
    result = evaluate_spans({"extra": [(0, 1, "Drug")]}, {})
    assert result["fp"] == 1
    assert result["tp"] == result["fn"] == 0
