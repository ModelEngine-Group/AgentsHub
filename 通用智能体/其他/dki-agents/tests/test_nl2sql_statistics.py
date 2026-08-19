"""Focused uncertainty tests for Wave 1C NL2SQL evaluation."""

import pytest

from dki.analysis.nl2sql_statistics import (
    FAMILY_CLUSTER_BOOTSTRAP_POLICY_HASH,
    WILSON_POLICY_HASH,
    family_cluster_bootstrap,
    wilson_interval,
)


def test_wilson_known_values():
    interval = wilson_interval(5, 10)

    assert interval["correct"] == 5
    assert interval["total"] == 10
    assert interval["estimate"] == 0.5
    assert interval["lower"] == pytest.approx(0.2365930905)
    assert interval["upper"] == pytest.approx(0.7634069095)
    assert interval["policy_hash"] == WILSON_POLICY_HASH


def test_wilson_known_all_successes_value():
    interval = wilson_interval(10, 10)

    assert interval["lower"] == pytest.approx(0.7224672001)
    assert interval["upper"] == pytest.approx(1.0)


def test_wilson_rejects_pathological_counts():
    with pytest.raises(ValueError, match="greater than zero"):
        wilson_interval(0, 0)
    with pytest.raises(ValueError, match="between zero and total"):
        wilson_interval(11, 10)
    with pytest.raises(TypeError, match="integer"):
        wilson_interval(True, 10)


def test_family_bootstrap_is_deterministic_and_item_weighted():
    # Macro family accuracy would be (1.0 + 0.0) / 2 == 0.5; the required
    # item-weighted point estimate is 3 / 4 == 0.75.
    rows = [
        {"partition_family_id": "large", "correct": True},
        {"partition_family_id": "large", "correct": True},
        {"partition_family_id": "large", "correct": True},
        {"partition_family_id": "small", "correct": False},
    ]

    first = family_cluster_bootstrap(rows, iterations=2_000, seed=17)
    second = family_cluster_bootstrap(rows, iterations=2_000, seed=17)

    assert first == second
    assert first["estimate"] == 0.75
    assert first["correct"] == 3
    assert first["total"] == 4
    assert first["n_families"] == 2
    assert first["min_family_size"] == 1
    assert first["max_family_size"] == 3
    assert first["policy_hash"] == FAMILY_CLUSTER_BOOTSTRAP_POLICY_HASH
    assert first["lower"] <= first["estimate"] <= first["upper"]


def test_family_bootstrap_missing_or_invalid_family_fails_closed():
    with pytest.raises(ValueError, match="missing required family key"):
        family_cluster_bootstrap([{"correct": True}], iterations=10, seed=1)
    with pytest.raises(ValueError, match="invalid family ID"):
        family_cluster_bootstrap(
            [{"partition_family_id": "  ", "correct": True}],
            iterations=10,
            seed=1,
        )
    with pytest.raises(ValueError, match="boolean"):
        family_cluster_bootstrap(
            [{"partition_family_id": "f1", "correct": 1}],
            iterations=10,
            seed=1,
        )


def test_family_bootstrap_requires_explicit_integer_seed():
    with pytest.raises(TypeError, match="seed must be an integer"):
        family_cluster_bootstrap(
            [{"partition_family_id": "f1", "correct": True}],
            iterations=10,
            seed=True,
        )


def test_family_bootstrap_rejects_one_family_and_one_iteration():
    with pytest.raises(ValueError, match="at least two distinct families"):
        family_cluster_bootstrap(
            [
                {"partition_family_id": "f1", "correct": True},
                {"partition_family_id": "f1", "correct": False},
            ],
            iterations=10,
            seed=1,
        )

    with pytest.raises(ValueError, match="iterations must be at least two"):
        family_cluster_bootstrap(
            [
                {"partition_family_id": "f1", "correct": True},
                {"partition_family_id": "f2", "correct": False},
            ],
            iterations=1,
            seed=1,
        )
