from dataclasses import FrozenInstanceError

import pytest

from dki.analysis.nl2sql_ablation import (
    KNOWN_RULE_CLUSTER_IDS,
    RULE_CLUSTER_CLOSED_STRUCTURAL_OWNERS,
    RULE_CLUSTER_EXACT_LEXICAL_CURATED,
    RULE_CLUSTER_KG,
    RULE_CLUSTER_OUTPUT_FORMATTING,
    RULE_CLUSTER_TEMPORAL_COHORT_WINDOW_TOPK,
    RULE_CLUSTER_VERSIONED_COMPILERS,
    AblationAcceptanceCriteria,
    DEFAULT_ABLATION_ACCEPTANCE_CRITERIA,
    DEFAULT_RULE_REGISTRY,
    NL2SQLAblationConfig,
    RouteProvenance,
    RuleActivationReason,
    RuleDecision,
    RuleDecisionKind,
    RuleRegistration,
    RuleRegistrySnapshot,
    assess_ablation_acceptance,
    compare_paired_correctness,
    compare_stability_populations,
    validate_rule_cluster_id,
    validate_rule_registry,
)


EXPECTED_CLUSTERS = (
    "closed_structural_owners",
    "versioned_compilers",
    "temporal_cohort_window_topk",
    "kg",
    "exact_lexical_curated",
    "output_formatting",
)
SOURCE_HASH = "1" * 64


def _comparison(baseline, variant):
    return compare_paired_correctness(
        {f"item-{index}": value for index, value in enumerate(baseline)},
        {f"item-{index}": value for index, value in enumerate(variant)},
    )


def _stability(*, worsening=False):
    baseline = {
        "family_a": {
            "run_1": {"a": True, "b": False, "c": True, "d": False},
            "run_2": {"a": True, "b": False, "c": True, "d": False},
        },
        "family_b": {
            "run_1": {"e": True, "f": False},
            "run_2": {"e": True, "f": False},
        },
    }
    variant = {
        family: {run: dict(values) for run, values in runs.items()}
        for family, runs in baseline.items()
    }
    if worsening:
        variant["family_a"]["run_1"] = {
            "a": True,
            "b": True,
            "c": True,
            "d": False,
        }
    return compare_stability_populations(baseline, variant)


def test_known_cluster_ids_are_stable_complete_and_validated():
    assert KNOWN_RULE_CLUSTER_IDS == EXPECTED_CLUSTERS
    assert {
        RULE_CLUSTER_CLOSED_STRUCTURAL_OWNERS,
        RULE_CLUSTER_VERSIONED_COMPILERS,
        RULE_CLUSTER_TEMPORAL_COHORT_WINDOW_TOPK,
        RULE_CLUSTER_KG,
        RULE_CLUSTER_EXACT_LEXICAL_CURATED,
        RULE_CLUSTER_OUTPUT_FORMATTING,
    } == set(EXPECTED_CLUSTERS)
    for cluster_id in EXPECTED_CLUSTERS:
        assert validate_rule_cluster_id(cluster_id) == cluster_id
    with pytest.raises(ValueError, match="unknown NL2SQL rule cluster"):
        validate_rule_cluster_id("compiler_v24")


def test_registry_snapshot_validates_unknown_and_cluster_mismatched_rules():
    entry = RuleRegistration(
        "compiler.v23",
        RULE_CLUSTER_VERSIONED_COMPILERS,
        "Independent v23 compiler",
    )
    registry = RuleRegistrySnapshot((entry,))

    assert registry.cluster_for("compiler.v23") == RULE_CLUSTER_VERSIONED_COMPILERS
    registry.validate_membership("compiler.v23", RULE_CLUSTER_VERSIONED_COMPILERS)
    with pytest.raises(ValueError, match="unknown rule_id"):
        registry.cluster_for("compiler.v24")
    with pytest.raises(ValueError, match="rule/cluster mismatch"):
        registry.validate_membership("compiler.v23", RULE_CLUSTER_KG)
    with pytest.raises(ValueError, match="duplicate rule_id"):
        validate_rule_registry((entry, entry))


def test_config_is_immutable_registry_bound_and_resolves_cluster_then_rule():
    config = NL2SQLAblationConfig.without(
        RULE_CLUSTER_KG,
        profile_name="baseline_minus_kg",
        disabled_rule_ids=("compiler.v23",),
    )

    assert config.disabled_clusters == (RULE_CLUSTER_KG,)
    assert config.resolve(RULE_CLUSTER_KG, "kg.rule").reason is (
        RuleActivationReason.CLUSTER_DISABLED
    )
    assert config.resolve(
        RULE_CLUSTER_VERSIONED_COMPILERS,
        "compiler.v23",
    ).reason is RuleActivationReason.RULE_DISABLED
    assert config.is_enabled(
        RULE_CLUSTER_VERSIONED_COMPILERS,
        "compiler.v21",
    )
    with pytest.raises(ValueError, match="unknown rule_id"):
        NL2SQLAblationConfig(disabled_rule_ids=("compiler.v24",))
    with pytest.raises(ValueError, match="rule/cluster mismatch"):
        config.resolve(RULE_CLUSTER_KG, "compiler.v23")
    with pytest.raises(FrozenInstanceError):
        config.profile_name = "mutated"


def test_policy_hash_excludes_profile_identity_and_inert_disabled_rules():
    first = NL2SQLAblationConfig.without(
        RULE_CLUSTER_KG,
        profile_name="profile_one",
        disabled_rule_ids=("kg.rule",),
    )
    second = NL2SQLAblationConfig.without(
        RULE_CLUSTER_KG,
        profile_name="profile_two",
    )
    changed = NL2SQLAblationConfig.without(
        RULE_CLUSTER_OUTPUT_FORMATTING,
        profile_name="profile_one",
    )

    assert first.disabled_rule_ids == ()
    assert first.policy_hash == second.policy_hash
    assert first.behavior_hash == second.behavior_hash
    assert first.canonical_hash == first.policy_hash
    assert first.profile_hash != second.profile_hash
    assert first.policy_hash != changed.policy_hash


def test_config_manifest_round_trip_recomputes_policy_and_profile_hashes():
    config = NL2SQLAblationConfig.without(
        RULE_CLUSTER_KG,
        profile_name="baseline_minus_kg",
        disabled_rule_ids=("compiler.v23",),
    )
    loaded = NL2SQLAblationConfig.from_manifest_payload(
        config.manifest_payload(),
        registry=DEFAULT_RULE_REGISTRY,
    )

    assert loaded == config
    assert loaded.policy_hash == config.policy_hash
    assert loaded.profile_hash == config.profile_hash


def test_rule_decision_rejects_unknown_or_wrong_cluster_membership():
    with pytest.raises(ValueError, match="unknown rule_id"):
        RuleDecision(
            "compiler.v24",
            RULE_CLUSTER_VERSIONED_COMPILERS,
            RuleDecisionKind.DISABLED,
        )
    with pytest.raises(ValueError, match="rule/cluster mismatch"):
        RuleDecision(
            "compiler.v23",
            RULE_CLUSTER_KG,
            RuleDecisionKind.SQL,
        )


def test_route_provenance_has_canonical_round_trip_and_alias_compatibility():
    config = NL2SQLAblationConfig.baseline()
    decision = RuleDecision(
        rule_id="compiler.v23",
        cluster_id=RULE_CLUSTER_VERSIONED_COMPILERS,
        decision=RuleDecisionKind.SQL,
        reason_codes=("ast.read_only", "contract.valid"),
    )
    provenance = RouteProvenance(
        route="bounded_rule_before_model",
        policy_hash=config.policy_hash,
        profile_hash=config.profile_hash,
        source_hash=SOURCE_HASH,
        decisions=(decision,),
        rule_id="compiler.v23",
        cluster_id=RULE_CLUSTER_VERSIONED_COMPILERS,
    )

    payload = provenance.as_dict()
    assert RouteProvenance.from_dict(payload) == provenance
    alias_payload = dict(payload)
    alias_payload["ablation_config_hash"] = alias_payload.pop("policy_hash")
    alias_payload["selected_rule_id"] = alias_payload.pop("rule_id")
    alias_payload["selected_rule_cluster"] = alias_payload.pop("cluster_id")
    alias_payload["rule_decisions"] = alias_payload.pop("decisions")
    assert RouteProvenance.from_dict(alias_payload) == provenance
    assert "sql" not in payload
    assert "question" not in payload


def test_route_provenance_rejects_alias_conflicts_and_registry_mismatch():
    config = NL2SQLAblationConfig.baseline()
    payload = {
        "route": "rule",
        "policy_hash": config.policy_hash,
        "ablation_config_hash": "2" * 64,
        "profile_hash": config.profile_hash,
        "source_hash": SOURCE_HASH,
        "rule_id": "compiler.v23",
        "cluster_id": RULE_CLUSTER_VERSIONED_COMPILERS,
    }
    with pytest.raises(ValueError, match="conflicting aliases for policy_hash"):
        RouteProvenance.from_dict(payload)

    wrong_registry = RuleRegistrySnapshot(
        (RuleRegistration("only.rule", RULE_CLUSTER_KG),)
    )
    good = RouteProvenance(
        route="rule",
        policy_hash=config.policy_hash,
        profile_hash=config.profile_hash,
        source_hash=SOURCE_HASH,
        decisions=(
            RuleDecision(
                "compiler.v23",
                RULE_CLUSTER_VERSIONED_COMPILERS,
                RuleDecisionKind.SQL,
            ),
        ),
        rule_id="compiler.v23",
        cluster_id=RULE_CLUSTER_VERSIONED_COMPILERS,
    ).as_dict()
    with pytest.raises(ValueError, match="registry_hash"):
        RouteProvenance.from_dict(good, registry=wrong_registry)


def test_strict_comparison_keeps_every_item_and_exact_transition_counts():
    comparison = compare_paired_correctness(
        {"a": True, "b": False, "c": False, "d": True},
        {"a": False, "b": True, "c": True, "d": True},
    )

    assert comparison.baseline.as_dict() == {
        "n": 4,
        "n_correct": 2,
        "n_incorrect": 2,
        "accuracy": 0.5,
    }
    assert comparison.variant.n_correct == 3
    assert comparison.delta_n_correct == 1
    assert comparison.delta == 0.25
    assert comparison.n_improved == 2
    assert comparison.n_regressed == 1
    with pytest.raises(ValueError, match="unpaired result IDs"):
        compare_paired_correctness({"a": True}, {"b": True})


def test_stability_compares_range_and_stddev_on_identical_family_run_populations():
    stable = _stability(worsening=False)
    worsening = _stability(worsening=True)

    assert stable.range_not_worse is True
    assert stable.stddev_not_worse is True
    assert stable.variant.max_accuracy_range == 0.0
    assert stable.variant.max_accuracy_stddev == 0.0
    assert worsening.range_not_worse is False
    assert worsening.stddev_not_worse is False
    assert worsening.variant.max_accuracy_range == 0.25
    assert worsening.variant.max_accuracy_stddev == 0.125


def test_stability_rejects_mixed_family_run_or_item_populations():
    baseline = {"family_a": {"run_1": {"a": True}, "run_2": {"a": True}}}
    with pytest.raises(ValueError, match="family populations differ"):
        compare_stability_populations(
            baseline,
            {"family_b": {"run_1": {"a": True}, "run_2": {"a": True}}},
        )
    with pytest.raises(ValueError, match="run populations differ"):
        compare_stability_populations(
            baseline,
            {"family_a": {"run_1": {"a": True}}},
        )
    with pytest.raises(ValueError, match="item population differs"):
        compare_stability_populations(
            baseline,
            {
                "family_a": {
                    "run_1": {"a": True},
                    "run_2": {"b": True},
                }
            },
        )


def test_acceptance_passes_all_four_hardened_gates():
    decision = assess_ablation_acceptance(
        pooled=_comparison((True, False, False, True), (True, True, False, True)),
        shadow={
            "slice_a_family_a": _comparison((True, False), (True, False)),
            "slice_a_family_b": _comparison((True, True), (True, True)),
            "slice_b_family_a": _comparison((False, True), (True, True)),
        },
        alternate_db={
            "nulls": _comparison((True, False), (True, True)),
            "ties": _comparison((False, True), (False, True)),
            "duplicates": _comparison((True, True), (True, True)),
        },
        stability=_stability(worsening=False),
    )

    assert decision.accepted is True
    payload = decision.as_dict()
    assert set(payload["gates"]) == {
        "pooled_positive_delta",
        "shadow_worst_family_loss",
        "alternate_db_no_loss",
        "stability_no_worsening",
    }
    assert payload["gates"]["shadow_worst_family_loss"]["measured"] == 0.0


def test_acceptance_uses_worst_shadow_family_and_rejects_worsening_stability():
    decision = assess_ablation_acceptance(
        pooled=_comparison((False, False, True, True), (True, False, True, True)),
        shadow={
            # Aggregate across these two could be flat, but the first family loses 50pp.
            "same_slice_family_a": _comparison((True, True), (True, False)),
            "same_slice_family_b": _comparison((True, False), (True, True)),
            "other_slice_family_a": _comparison((True, False), (True, False)),
        },
        alternate_db={
            "a": _comparison((True,), (True,)),
            "b": _comparison((False,), (False,)),
            "c": _comparison((True,), (True,)),
        },
        stability=_stability(worsening=True),
    )
    gates = {gate.name: gate for gate in decision.gates}

    assert decision.accepted is False
    assert gates["shadow_worst_family_loss"].measured == 0.5
    assert gates["shadow_worst_family_loss"].passed is False
    assert gates["stability_no_worsening"].passed is False
    assert gates["stability_no_worsening"].measured["range_not_worse"] is False
    assert gates["stability_no_worsening"].measured["stddev_not_worse"] is False


def test_protocol_defaults_are_one_immutable_shared_policy():
    assert DEFAULT_ABLATION_ACCEPTANCE_CRITERIA == AblationAcceptanceCriteria()
    assert DEFAULT_ABLATION_ACCEPTANCE_CRITERIA.min_shadow_groups == 3
    assert DEFAULT_ABLATION_ACCEPTANCE_CRITERIA.min_alternate_dbs == 3
    assert DEFAULT_ABLATION_ACCEPTANCE_CRITERIA.min_stability_runs == 2
    assert DEFAULT_ABLATION_ACCEPTANCE_CRITERIA.max_stability_range == 0.02
    assert DEFAULT_ABLATION_ACCEPTANCE_CRITERIA.max_stability_stddev == 0.02
