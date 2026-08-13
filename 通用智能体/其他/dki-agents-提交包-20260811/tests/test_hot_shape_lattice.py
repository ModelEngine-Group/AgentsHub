"""Hot-shape lattice planning + deterministic arrival replay (CPU sim)."""

import random

import pytest

from dki.nlp.hot_shape import (
    choose_node,
    compare_fixed_vs_lattice,
    plan_lattice,
    simulate_arrivals,
)


def _synthetic(n=300, seed=1):
    rng = random.Random(seed)
    return [rng.choice([30, 45, 60, 90, 120, 200, 300]) for _ in range(n)]


def test_choose_smallest_containing_node():
    assert choose_node(60, [64, 128, 256]) == 64
    assert choose_node(65, [64, 128, 256]) == 128
    assert choose_node(300, [64, 128, 256]) is None


def test_plan_lattice_bounded_nodes_and_overflow():
    lengths = [10, 60, 61, 100, 200, 300]
    plan = plan_lattice(lengths, sequence_lengths=(64, 128, 256), max_nodes=2)
    assert len(plan.nodes) <= 2
    assert plan.total == 6
    assert plan.overflow >= 1  # 300 tokens + 2 specials > 256
    assert plan.total_padding_waste() >= 0


def test_plan_lattice_padding_waste_accounting():
    # all docs need 62 tokens -> node 64 wastes 2 each
    plan = plan_lattice([60] * 10, sequence_lengths=(64, 128, 256), max_nodes=3, special_tokens=2)
    assert plan.node_lengths() == [64]
    assert plan.nodes[0].padding_waste == 10 * (64 - 62)


def test_lattice_reduces_padding_vs_fixed_seq256():
    lengths = _synthetic()
    cmp = compare_fixed_vs_lattice(
        lengths, sequence_lengths=(64, 128, 256), fixed_seq=256,
        batch_size=8, arrival_interval=4.0,
    )
    for mode in ("open_loop", "closed_loop"):
        fixed = cmp["runs"][mode]["fixed"]
        lattice = cmp["runs"][mode]["lattice"]
        # same trace -> same completed+dropped accounting
        assert fixed["completed"] + fixed["dropped"] == lattice["completed"] + lattice["dropped"]
        # lattice packs shorter shapes -> strictly less padding waste here
        assert lattice["padding_waste_tokens"] < fixed["padding_waste_tokens"]
        assert cmp["runs"][mode]["padding_waste_reduction"] > 0.0


def test_simulation_is_deterministic():
    lengths = _synthetic()
    a = simulate_arrivals(lengths, schedule="lattice", mode="open_loop", seed=42)
    b = simulate_arrivals(lengths, schedule="lattice", mode="open_loop", seed=42)
    assert a.to_dict() == b.to_dict()


def test_closed_loop_backpressure_drops_when_saturated():
    lengths = _synthetic(n=200)
    # tiny inflight cap + fast arrivals + slow device -> some drops
    result = simulate_arrivals(
        lengths, schedule="fixed", mode="closed_loop", fixed_seq=256,
        batch_size=1, arrival_interval=0.01, per_token_cost=1.0, max_inflight=2,
    )
    assert result.dropped > 0


def test_invalid_args_rejected():
    with pytest.raises(ValueError):
        simulate_arrivals([10], schedule="bogus")
    with pytest.raises(ValueError):
        plan_lattice([10], max_nodes=0)
