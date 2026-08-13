"""Deterministic bootstrap CI + summary statistics for the performance contract."""

import pytest

from dki.evidence.stats import bootstrap_ci, summarize


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    values = [3085.06, 3036.71, 3054.45, 2995.73, 3069.43]
    a = bootstrap_ci(values, iterations=2000, seed=7)
    b = bootstrap_ci(values, iterations=2000, seed=7)
    assert a == b  # deterministic
    assert a["lower"] <= a["point"] <= a["upper"]
    assert a["confidence"] == 0.95


def test_bootstrap_ci_single_value_is_degenerate():
    r = bootstrap_ci([42.0])
    assert r["lower"] == r["point"] == r["upper"] == 42.0
    assert r["iterations"] == 0


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_summarize_fields_and_ci():
    values = [10.0, 12.0, 11.0, 13.0, 9.0]
    s = summarize(values)
    assert s["n"] == 5
    assert s["min"] == 9.0 and s["max"] == 13.0
    assert "ci95" in s and s["ci95"]["lower"] <= s["mean"] <= s["ci95"]["upper"]


def test_narrower_ci_with_less_spread():
    tight = summarize([100.0, 100.5, 99.5, 100.2, 99.8])["ci95"]
    wide = summarize([50.0, 150.0, 100.0, 20.0, 180.0])["ci95"]
    assert (tight["upper"] - tight["lower"]) < (wide["upper"] - wide["lower"])
