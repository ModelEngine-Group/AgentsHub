import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from dki.analysis.nl2sql_ablation import (
    RULE_CLUSTER_KG,
    RULE_CLUSTER_OUTPUT_FORMATTING,
    RULE_CLUSTER_VERSIONED_COMPILERS,
    AblationAcceptanceCriteria,
    DEFAULT_ABLATION_ACCEPTANCE_CRITERIA,
    DEFAULT_RULE_REGISTRY,
    NL2SQLAblationConfig,
    RouteProvenance,
    RuleDecision,
    RuleDecisionKind,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "eval_nl2sql_ablations",
    ROOT / "scripts" / "eval_nl2sql_ablations.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

SOURCE_HASH = "1" * 64
DATASET_HASH = "2" * 64
FAMILY_HASH = "3" * 64
RUN_HASH = "4" * 64
DATABASE_HASH = "5" * 64
DATABASE_IDS = ("alt_a", "alt_b", "alt_c", "base")
FAMILY_IDS = ("family_a", "family_b", "family_c")
RUN_IDS = ("primary", "repeat_2")
SHADOW_GROUPS = (
    ("shadow_a", "family_a"),
    ("shadow_a", "family_b"),
    ("shadow_b", "family_c"),
)


def _provenance(config, *, route="bounded_rule_before_model"):
    decision = RuleDecision(
        "compiler.v23",
        RULE_CLUSTER_VERSIONED_COMPILERS,
        RuleDecisionKind.SQL,
    )
    return RouteProvenance(
        route=route,
        policy_hash=config.policy_hash,
        profile_hash=config.profile_hash,
        source_hash=SOURCE_HASH,
        decisions=(decision,),
        rule_id="compiler.v23",
        cluster_id=RULE_CLUSTER_VERSIONED_COMPILERS,
    ).as_dict()


def _row(
    config,
    item_id,
    correct,
    *,
    role="fresh_dev_b",
    slice_id="dev",
    db_id="base",
    run_id="primary",
    family_id="family_a",
    category="category_a",
    provenance_level="nested",
):
    provenance = _provenance(config)
    payload = {
        "id": item_id,
        "correct": correct,
        "dataset_role": role,
        "slice_id": slice_id,
        "db_id": db_id,
        "run_id": run_id,
        "difficulty": "hard",
        "category": category,
        "family_id": family_id,
        "capabilities": ["date", "output_contract"],
        "engine": "deterministic_rule",
        "error_category": None if correct else "semantic_reasoning",
    }
    if provenance_level == "nested":
        payload["provenance"] = provenance
    elif provenance_level == "plan":
        payload["plan"] = provenance
    elif provenance_level == "top":
        payload.update(provenance)
    else:
        raise AssertionError(provenance_level)
    return payload


def _arm_rows(config, *, variant=False):
    rows = []
    selection = (
        ("sel_a", "family_a", "category_a", False, variant),
        ("sel_b", "family_b", "category_b", True, True),
        ("sel_c", "family_c", "category_c", True, True),
    )
    for run_id in RUN_IDS:
        for item_id, family_id, category, before, after in selection:
            rows.append(
                _row(
                    config,
                    item_id,
                    after if variant else before,
                    run_id=run_id,
                    family_id=family_id,
                    category=category,
                )
            )
    for db_id in DATABASE_IDS:
        if db_id == "base":
            continue
        for item_id, family_id, category, before, after in selection:
            rows.append(
                _row(
                    config,
                    item_id,
                    after if variant else before,
                    db_id=db_id,
                    family_id=family_id,
                    category=category,
                )
            )
    for slice_id, family_id in SHADOW_GROUPS:
        for suffix, correct in (("one", True), ("two", False)):
            rows.append(
                _row(
                    config,
                    f"{slice_id}_{family_id}_{suffix}",
                    correct,
                    role="shadow",
                    slice_id=slice_id,
                    family_id=family_id,
                    category="shadow_category",
                )
            )
    return rows


def _parse_rows(rows, registry=DEFAULT_RULE_REGISTRY):
    return tuple(
        sorted(
            (
                MODULE.parse_precomputed_result(
                    row,
                    registry=registry,
                    source=f"row-{index}",
                )
                for index, row in enumerate(rows)
            ),
            key=lambda row: row.identity,
        )
    )


def _manifest(
    config,
    parsed_rows,
    *,
    arm_id,
    arm_role,
    target_clusters=(),
    target_rules=(),
):
    return MODULE.AblationArmManifest(
        arm_id=arm_id,
        arm_role=arm_role,
        registry=config.registry,
        config=config,
        target_disabled_clusters=tuple(target_clusters),
        target_disabled_rule_ids=tuple(target_rules),
        source_hash=SOURCE_HASH,
        dataset_manifest_hash=DATASET_HASH,
        family_manifest_hash=FAMILY_HASH,
        run_manifest_hash=RUN_HASH,
        database_manifest_hash=DATABASE_HASH,
        selection_role="fresh_dev_b",
        shadow_role="shadow",
        base_db_id="base",
        primary_run_id="primary",
        database_ids=DATABASE_IDS,
        family_ids=FAMILY_IDS,
        run_ids=RUN_IDS,
        shadow_groups=SHADOW_GROUPS,
        result_rows_hash=MODULE.canonical_rows_sha256(parsed_rows),
    )


def _bound_arm(config, rows, *, arm_id, arm_role, target_clusters=(), target_rules=()):
    parsed = _parse_rows(rows, config.registry)
    manifest = _manifest(
        config,
        parsed,
        arm_id=arm_id,
        arm_role=arm_role,
        target_clusters=target_clusters,
        target_rules=target_rules,
    )
    return MODULE.bind_arm(
        manifest,
        MODULE.LoadedResultSet("f" * 64, parsed),
    )


def _passing_arms():
    baseline_config = NL2SQLAblationConfig.baseline(profile_name="baseline")
    variant_config = NL2SQLAblationConfig.without(
        RULE_CLUSTER_KG,
        profile_name="baseline_minus_kg",
    )
    baseline = _bound_arm(
        baseline_config,
        _arm_rows(baseline_config),
        arm_id="baseline",
        arm_role="baseline",
    )
    variant = _bound_arm(
        variant_config,
        _arm_rows(variant_config, variant=True),
        arm_id="baseline_minus_kg",
        arm_role="variant",
        target_clusters=(RULE_CLUSTER_KG,),
    )
    return baseline, variant


def _write_jsonl(path, rows):
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_bound_inputs(
    tmp_path,
    *,
    name,
    config,
    rows,
    arm_role,
    target_clusters=(),
    target_rules=(),
):
    result_path = tmp_path / f"{name}.jsonl"
    _write_jsonl(result_path, rows)
    loaded = MODULE.load_precomputed_results(result_path, registry=config.registry)
    manifest = _manifest(
        config,
        loaded.rows,
        arm_id=name,
        arm_role=arm_role,
        target_clusters=target_clusters,
        target_rules=target_rules,
    )
    manifest_path = tmp_path / f"{name}.manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest.canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return result_path, manifest_path, manifest


def test_manifest_bound_report_passes_and_hashes_all_data_dimensions():
    baseline, variant = _passing_arms()

    report = MODULE.build_ablation_report(
        baseline,
        {"baseline_minus_kg": variant},
    )
    public_variant_id = MODULE.public_label_hash("arm", "baseline_minus_kg")
    result = report["variants"][public_variant_id]
    rendered = MODULE.encode_report(report).decode("utf-8")

    assert report["all_variants_accepted"] is True
    assert result["acceptance"]["accepted"] is True
    assert result["acceptance"]["status"] == "accepted"
    assert result["acceptance"]["measurements"]["pooled"]["n"] == 3
    assert result["acceptance"]["measurements"]["pooled"][
        "delta_n_correct"
    ] == 1
    assert len(result["acceptance"]["measurements"]["shadow_by_slice_family"]) == 3
    assert len(result["acceptance"]["measurements"]["alternate_db"]) == 3
    assert result["acceptance"]["measurements"]["stability"][
        "range_not_worse"
    ] is True
    assert all(key.startswith("sha256:") for key in result["by_family"])
    assert all(key.startswith("sha256:") for key in result["by_category"])
    assert all(key.startswith("sha256:") for key in result["by_database"])
    assert "family_a" not in rendered
    assert "category_a" not in rendered
    assert "bounded_rule_before_model" not in rendered
    assert "deterministic_rule" not in rendered
    assert "question" not in rendered
    assert "gold_sql" not in rendered
    assert "pred_sql" not in rendered
    assert "path" not in report["baseline"]["results"]


def test_report_is_deterministic_for_variant_mapping_and_row_order():
    baseline, variant = _passing_arms()
    reversed_baseline = MODULE.BoundArm(
        baseline.manifest,
        MODULE.LoadedResultSet(
            baseline.results.file_sha256,
            tuple(reversed(baseline.results.rows)),
        ),
    )
    reversed_variant = MODULE.BoundArm(
        variant.manifest,
        MODULE.LoadedResultSet(
            variant.results.file_sha256,
            tuple(reversed(variant.results.rows)),
        ),
    )

    first = MODULE.build_ablation_report(
        reversed_baseline,
        {"baseline_minus_kg": reversed_variant},
    )
    second = MODULE.build_ablation_report(
        baseline,
        {"baseline_minus_kg": variant},
    )

    assert MODULE.encode_report(first) == MODULE.encode_report(second)


def test_route_provenance_round_trips_from_top_plan_and_provenance_levels():
    config = NL2SQLAblationConfig.baseline()
    for level in ("top", "plan", "nested"):
        row = _row(config, f"item_{level}", True, provenance_level=level)
        parsed = MODULE.parse_precomputed_result(
            row,
            registry=config.registry,
        )
        assert parsed.provenance == RouteProvenance.from_dict(_provenance(config))


def test_route_provenance_rejects_top_level_nested_alias_conflicts():
    config = NL2SQLAblationConfig.baseline()
    row = _row(config, "item", True)
    row["policy_hash"] = "9" * 64
    with pytest.raises(ValueError, match="conflicting policy_hash values across levels"):
        MODULE.parse_precomputed_result(row, registry=config.registry)

    row = _row(config, "item", True, provenance_level="top")
    row["selected_rule_id"] = "compiler.v21"
    with pytest.raises(ValueError, match="conflicting rule_id aliases"):
        MODULE.parse_precomputed_result(row, registry=config.registry)


@pytest.mark.parametrize(
    ("field", "alias", "bad_value", "message"),
    (
        ("family_id", "family", "family_b", "conflicting aliases for family_id"),
        ("slice_id", "shadow_id", "other_slice", "conflicting aliases for slice_id"),
        ("db_id", "database_id", "alt_a", "conflicting aliases for db_id"),
        (
            "capabilities",
            "capability",
            "window",
            "conflicting aliases for capabilities",
        ),
    ),
)
def test_all_row_alias_conflicts_fail_closed(field, alias, bad_value, message):
    config = NL2SQLAblationConfig.baseline()
    row = _row(config, "item", True)
    assert field in row
    row[alias] = bad_value
    with pytest.raises(ValueError, match=message):
        MODULE.parse_precomputed_result(row, registry=config.registry)


def test_jsonl_rejects_duplicate_keys_nonfinite_values_and_unknown_fields(tmp_path):
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text('{"id":"a","id":"b","correct":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        MODULE.load_precomputed_results(duplicate, registry=DEFAULT_RULE_REGISTRY)

    nonfinite = tmp_path / "nonfinite.jsonl"
    nonfinite.write_text('{"id":"a","correct":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON value"):
        MODULE.load_precomputed_results(nonfinite, registry=DEFAULT_RULE_REGISTRY)

    overflow = tmp_path / "overflow.jsonl"
    overflow.write_text(
        '{"id":"a","correct":true,"score":1e999}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite JSON value"):
        MODULE.load_precomputed_results(overflow, registry=DEFAULT_RULE_REGISTRY)

    config = NL2SQLAblationConfig.baseline()
    unknown = _row(config, "item", True)
    unknown["question"] = "sensitive"
    with pytest.raises(ValueError, match="unknown fields"):
        MODULE.parse_precomputed_result(unknown, registry=config.registry)


def test_rows_require_complete_coherent_provenance_and_stable_public_labels():
    config = NL2SQLAblationConfig.baseline()
    missing = _row(config, "item", True)
    del missing["provenance"]["source_hash"]
    with pytest.raises(ValueError, match="missing source_hash"):
        MODULE.parse_precomputed_result(missing, registry=config.registry)

    leaking = _row(config, "item", True)
    leaking["category"] = "SELECT * FROM patients"
    with pytest.raises(ValueError, match="lowercase stable identifier"):
        MODULE.parse_precomputed_result(leaking, registry=config.registry)

    wrong_cluster = _row(config, "item", True)
    wrong_cluster["provenance"]["cluster_id"] = RULE_CLUSTER_KG
    with pytest.raises(ValueError, match="rule/cluster mismatch"):
        MODULE.parse_precomputed_result(wrong_cluster, registry=config.registry)


def test_arm_binding_rejects_mixed_policy_profile_source_and_disabled_selected_rule():
    config = NL2SQLAblationConfig.baseline()
    rows = _arm_rows(config)
    parsed = list(_parse_rows(rows))
    manifest = _manifest(
        config,
        tuple(parsed),
        arm_id="baseline",
        arm_role="baseline",
    )

    changed = replace(
        parsed[0],
        provenance=replace(parsed[0].provenance, profile_hash="9" * 64),
    )
    mixed_rows = tuple([changed, *parsed[1:]])
    mixed_manifest = replace(
        manifest,
        result_rows_hash=MODULE.canonical_rows_sha256(mixed_rows),
    )
    with pytest.raises(ValueError, match="profile provenance"):
        MODULE.bind_arm(
            mixed_manifest,
            MODULE.LoadedResultSet("f" * 64, mixed_rows),
        )

    variant_config = NL2SQLAblationConfig.without(
        RULE_CLUSTER_KG,
        profile_name="baseline_minus_kg",
    )
    variant_rows = list(_parse_rows(_arm_rows(variant_config, variant=True)))
    disabled_decision = RuleDecision(
        "kg.rule",
        RULE_CLUSTER_KG,
        RuleDecisionKind.SQL,
    )
    disabled_provenance = replace(
        variant_rows[0].provenance,
        decisions=(disabled_decision,),
        rule_id="kg.rule",
        cluster_id=RULE_CLUSTER_KG,
    )
    variant_rows[0] = replace(variant_rows[0], provenance=disabled_provenance)
    variant_rows_tuple = tuple(variant_rows)
    variant_manifest = _manifest(
        variant_config,
        variant_rows_tuple,
        arm_id="baseline_minus_kg",
        arm_role="variant",
        target_clusters=(RULE_CLUSTER_KG,),
    )
    with pytest.raises(ValueError, match="selected rule is disabled"):
        MODULE.bind_arm(
            variant_manifest,
            MODULE.LoadedResultSet("e" * 64, variant_rows_tuple),
        )


def test_manifest_rejects_registry_policy_profile_and_target_mismatches():
    config = NL2SQLAblationConfig.baseline()
    parsed = _parse_rows(_arm_rows(config))
    manifest = _manifest(
        config,
        parsed,
        arm_id="baseline",
        arm_role="baseline",
    )
    payload = manifest.canonical_payload()

    wrong = dict(payload)
    wrong["registry_hash"] = "9" * 64
    with pytest.raises(ValueError, match="registry_hash does not match"):
        MODULE.AblationArmManifest.from_payload(wrong)

    wrong = dict(payload)
    wrong["policy_hash"] = "9" * 64
    with pytest.raises(ValueError, match="policy_hash does not match"):
        MODULE.AblationArmManifest.from_payload(wrong)

    wrong = dict(payload)
    wrong["profile_hash"] = "9" * 64
    with pytest.raises(ValueError, match="profile_hash does not match"):
        MODULE.AblationArmManifest.from_payload(wrong)

    variant_config = NL2SQLAblationConfig.without(
        RULE_CLUSTER_KG,
        profile_name="baseline_minus_kg",
    )
    variant_rows = _parse_rows(_arm_rows(variant_config, variant=True))
    with pytest.raises(ValueError, match="not disabled by the config"):
        MODULE.AblationArmManifest(
            **{
                **_manifest(
                    variant_config,
                    variant_rows,
                    arm_id="baseline_minus_kg",
                    arm_role="variant",
                    target_clusters=(RULE_CLUSTER_KG,),
                ).__dict__,
                "target_disabled_clusters": (RULE_CLUSTER_VERSIONED_COMPILERS,),
            }
        )


def test_arm_binding_rejects_each_mixed_required_provenance_hash():
    config = NL2SQLAblationConfig.baseline()
    parsed = list(_parse_rows(_arm_rows(config)))
    manifest = _manifest(
        config,
        tuple(parsed),
        arm_id="baseline",
        arm_role="baseline",
    )
    cases = (
        ("policy_hash", "policy provenance"),
        ("profile_hash", "profile provenance"),
        ("source_hash", "source provenance"),
    )
    for field_name, message in cases:
        changed = replace(
            parsed[0],
            provenance=replace(
                parsed[0].provenance,
                **{field_name: "9" * 64},
            ),
        )
        changed_rows = tuple([changed, *parsed[1:]])
        changed_manifest = replace(
            manifest,
            result_rows_hash=MODULE.canonical_rows_sha256(changed_rows),
        )
        with pytest.raises(ValueError, match=message):
            MODULE.bind_arm(
                changed_manifest,
                MODULE.LoadedResultSet("f" * 64, changed_rows),
            )


def test_cross_arm_manifest_population_hashes_must_match():
    baseline, variant = _passing_arms()
    cases = (
        ("dataset_manifest_hash", "dataset_manifest_hash"),
        ("family_manifest_hash", "family_manifest_hash"),
        ("run_manifest_hash", "run_manifest_hash"),
        ("database_manifest_hash", "database_manifest_hash"),
    )
    for field_name, message in cases:
        changed_manifest = replace(
            variant.manifest,
            **{field_name: "9" * 64},
        )
        changed_variant = MODULE.BoundArm(changed_manifest, variant.results)
        with pytest.raises(ValueError, match=message):
            MODULE.pair_bound_arms(baseline, changed_variant)


def test_manifest_target_must_equal_effective_policy_delta():
    baseline_config = NL2SQLAblationConfig.baseline(profile_name="baseline")
    variant_config = NL2SQLAblationConfig.without(
        RULE_CLUSTER_KG,
        RULE_CLUSTER_OUTPUT_FORMATTING,
        profile_name="baseline_minus_two_clusters",
    )
    baseline = _bound_arm(
        baseline_config,
        _arm_rows(baseline_config),
        arm_id="baseline",
        arm_role="baseline",
    )
    parsed = _parse_rows(_arm_rows(variant_config, variant=True))
    wrong_manifest = _manifest(
        variant_config,
        parsed,
        arm_id="baseline_minus_two_clusters",
        arm_role="variant",
        target_clusters=(RULE_CLUSTER_KG,),
    )
    wrong_target = MODULE.BoundArm(
        wrong_manifest,
        MODULE.LoadedResultSet("d" * 64, parsed),
    )
    with pytest.raises(ValueError, match="target clusters"):
        MODULE.pair_bound_arms(baseline, wrong_target)


def test_alternate_db_and_run_populations_must_match_fixed_manifest():
    config = NL2SQLAblationConfig.baseline()
    rows = _arm_rows(config)
    rows = [
        row
        for row in rows
        if not (
            row["db_id"] == "alt_b"
            and row["id"] == "sel_c"
            and row["run_id"] == "primary"
        )
    ]
    parsed = _parse_rows(rows)
    manifest = _manifest(
        config,
        parsed,
        arm_id="baseline",
        arm_role="baseline",
    )
    with pytest.raises(ValueError, match="identical selection population"):
        MODULE.bind_arm(
            manifest,
            MODULE.LoadedResultSet("f" * 64, parsed),
        )

    rows = [
        row
        for row in _arm_rows(config)
        if not (
            row["db_id"] == "base"
            and row["run_id"] == "repeat_2"
            and row["id"] == "sel_c"
        )
    ]
    parsed = _parse_rows(rows)
    manifest = _manifest(
        config,
        parsed,
        arm_id="baseline",
        arm_role="baseline",
    )
    with pytest.raises(ValueError, match="identical family/item population"):
        MODULE.bind_arm(
            manifest,
            MODULE.LoadedResultSet("f" * 64, parsed),
        )


def test_shadow_gate_uses_worst_slice_family_instead_of_slice_pooling():
    baseline, variant = _passing_arms()
    changed = list(variant.results.rows)
    for index, row in enumerate(changed):
        if row.dataset_role != "shadow":
            continue
        if row.slice_id == "shadow_a" and row.family_id == "family_a":
            # One regression in this fixed family.
            if row.item_id.endswith("one"):
                changed[index] = replace(row, correct=False, error_category="semantic_reasoning")
        elif row.slice_id == "shadow_a" and row.family_id == "family_b":
            # Offset at the pooled slice level, which must not hide the regression.
            if row.item_id.endswith("two"):
                changed[index] = replace(row, correct=True, error_category=None)
    changed_rows = tuple(changed)
    changed_manifest = replace(
        variant.manifest,
        result_rows_hash=MODULE.canonical_rows_sha256(changed_rows),
    )
    changed_variant = MODULE.bind_arm(
        changed_manifest,
        MODULE.LoadedResultSet("d" * 64, changed_rows),
    )

    report = MODULE.build_ablation_report(
        baseline,
        {"baseline_minus_kg": changed_variant},
    )
    gate = report["variants"][MODULE.public_label_hash("arm", "baseline_minus_kg")]["acceptance"]["gates"][
        "shadow_worst_family_loss"
    ]
    assert gate["passed"] is False
    assert gate["measured"] == 0.5


def test_stability_gate_rejects_range_and_stddev_worsening_from_baseline():
    baseline, variant = _passing_arms()
    changed = list(variant.results.rows)
    for index, row in enumerate(changed):
        if (
            row.dataset_role == "fresh_dev_b"
            and row.db_id == "base"
            and row.run_id == "repeat_2"
            and row.family_id == "family_a"
        ):
            changed[index] = replace(
                row,
                correct=False,
                error_category="semantic_reasoning",
            )
    changed_rows = tuple(changed)
    changed_manifest = replace(
        variant.manifest,
        result_rows_hash=MODULE.canonical_rows_sha256(changed_rows),
    )
    changed_variant = MODULE.bind_arm(
        changed_manifest,
        MODULE.LoadedResultSet("d" * 64, changed_rows),
    )

    report = MODULE.build_ablation_report(
        baseline,
        {"baseline_minus_kg": changed_variant},
    )
    gate = report["variants"][MODULE.public_label_hash("arm", "baseline_minus_kg")]["acceptance"]["gates"][
        "stability_no_worsening"
    ]
    assert gate["passed"] is False
    assert gate["measured"]["range_not_worse"] is False
    assert gate["measured"]["stddev_not_worse"] is False


def test_missing_evidence_keeps_all_four_explicit_failed_gates():
    criteria = DEFAULT_ABLATION_ACCEPTANCE_CRITERIA
    payload = MODULE._missing_acceptance(criteria, "missing evidence")

    assert payload["accepted"] is False
    assert payload["status"] == "not_evaluable"
    assert set(payload["gates"]) == {
        "pooled_positive_delta",
        "shadow_worst_family_loss",
        "alternate_db_no_loss",
        "stability_no_worsening",
    }
    assert all(not gate["passed"] for gate in payload["gates"].values())
    assert all(gate["evidence_count"] == 0 for gate in payload["gates"].values())


def test_no_rule_model_and_failure_rows_remain_in_strict_denominator():
    config = NL2SQLAblationConfig.baseline()
    no_rule = RouteProvenance(
        route="schema_linking_query_plan",
        policy_hash=config.policy_hash,
        profile_hash=config.profile_hash,
        source_hash=SOURCE_HASH,
        reason_codes=("model.abstained",),
    ).as_dict()
    row = _row(config, "model_failure", False)
    row["engine"] = "model"
    row["error_category"] = "abstention"
    row["provenance"] = no_rule

    parsed = MODULE.parse_precomputed_result(row, registry=config.registry)

    assert parsed.provenance.rule_id is None
    assert parsed.provenance.cluster_id is None
    comparison = MODULE.comparison_for_pairs(((parsed, parsed),))
    assert comparison.baseline.n == 1
    assert comparison.baseline.n_correct == 0


def test_manifest_bound_no_rule_failure_row_is_retained():
    baseline, variant = _passing_arms()

    def replace_first(arm):
        rows = list(arm.results.rows)
        target = rows[0]
        no_rule = RouteProvenance(
            route="schema_linking_query_plan",
            policy_hash=arm.manifest.config.policy_hash,
            profile_hash=arm.manifest.config.profile_hash,
            source_hash=SOURCE_HASH,
            reason_codes=("model.abstained",),
        )
        rows[0] = replace(
            target,
            correct=False,
            engine="model",
            provenance=no_rule,
            error_category="abstention",
        )
        rows_tuple = tuple(rows)
        manifest = replace(
            arm.manifest,
            result_rows_hash=MODULE.canonical_rows_sha256(rows_tuple),
        )
        return MODULE.bind_arm(
            manifest,
            MODULE.LoadedResultSet(arm.results.file_sha256, rows_tuple),
        )

    baseline_with_failure = replace_first(baseline)
    variant_with_failure = replace_first(variant)
    pairs = MODULE.pair_bound_arms(baseline_with_failure, variant_with_failure)
    comparison = MODULE.comparison_for_pairs(pairs)

    assert comparison.baseline.n == len(pairs)
    assert any(before.provenance.rule_id is None for before, _ in pairs)


def test_policy_delta_allows_cluster_disable_to_supersede_baseline_rule_disable():
    baseline_config = NL2SQLAblationConfig(
        profile_name="baseline_rule_disabled",
        disabled_rule_ids=("kg.rule",),
    )
    variant_config = NL2SQLAblationConfig.without(
        RULE_CLUSTER_KG,
        profile_name="variant_cluster_disabled",
    )
    baseline_rows = _parse_rows(_arm_rows(baseline_config))
    variant_rows = _parse_rows(_arm_rows(variant_config, variant=True))
    baseline = MODULE.BoundArm(
        _manifest(
            baseline_config,
            baseline_rows,
            arm_id="baseline_rule_disabled",
            arm_role="baseline",
        ),
        MODULE.LoadedResultSet("a" * 64, baseline_rows),
    )
    variant = MODULE.BoundArm(
        _manifest(
            variant_config,
            variant_rows,
            arm_id="variant_cluster_disabled",
            arm_role="variant",
            target_clusters=(RULE_CLUSTER_KG,),
        ),
        MODULE.LoadedResultSet("b" * 64, variant_rows),
    )

    assert MODULE.pair_bound_arms(baseline, variant)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", 0, "unsupported provenance schema_version"),
        ("schema_version", False, "unsupported provenance schema_version"),
        ("reason_codes", False, "reason_codes must be an iterable"),
        ("decisions", 0, "decisions must be a sequence"),
    ),
)
def test_falsy_provenance_values_fail_closed(field, value, message):
    config = NL2SQLAblationConfig.baseline()
    row = _row(config, "item", True)
    row["provenance"][field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        MODULE.parse_precomputed_result(row, registry=config.registry)


def test_decision_objects_reject_unknown_fields():
    config = NL2SQLAblationConfig.baseline()
    row = _row(config, "item", True)
    row["provenance"]["decisions"] = [
        {
            "rule_id": "compiler.v23",
            "cluster_id": RULE_CLUSTER_VERSIONED_COMPILERS,
            "decision": "sql",
            "reason_codes": [],
            "unexpected": "value",
        }
    ]
    with pytest.raises(ValueError, match="rule decision has unknown fields"):
        MODULE.parse_precomputed_result(row, registry=config.registry)


def test_manifest_direct_constructor_canonicalizes_immutable_collections():
    config = NL2SQLAblationConfig.without(
        RULE_CLUSTER_KG,
        profile_name="baseline_minus_kg",
    )
    parsed = _parse_rows(_arm_rows(config, variant=True))
    manifest = MODULE.AblationArmManifest(
        arm_id="baseline_minus_kg",
        arm_role="variant",
        registry=config.registry,
        config=config,
        target_disabled_clusters=[RULE_CLUSTER_KG],
        target_disabled_rule_ids=[],
        source_hash=SOURCE_HASH,
        dataset_manifest_hash=DATASET_HASH,
        family_manifest_hash=FAMILY_HASH,
        run_manifest_hash=RUN_HASH,
        database_manifest_hash=DATABASE_HASH,
        selection_role="fresh_dev_b",
        shadow_role="shadow",
        base_db_id="base",
        primary_run_id="primary",
        database_ids=["base", "alt_c", "alt_a", "alt_b"],
        family_ids=["family_c", "family_a", "family_b"],
        run_ids=["repeat_2", "primary"],
        shadow_groups=[
            ["shadow_b", "family_c"],
            ["shadow_a", "family_b"],
            ["shadow_a", "family_a"],
        ],
        result_rows_hash=MODULE.canonical_rows_sha256(parsed),
    )

    assert manifest.database_ids == DATABASE_IDS
    assert manifest.family_ids == FAMILY_IDS
    assert manifest.run_ids == RUN_IDS
    assert manifest.shadow_groups == SHADOW_GROUPS
    assert isinstance(manifest.database_ids, tuple)
    with pytest.raises(ValueError, match="duplicate database_ids"):
        replace(manifest, database_ids=["base", "base"])


def test_decision_trace_must_be_policy_consistent():
    config = NL2SQLAblationConfig.without(
        RULE_CLUSTER_KG,
        profile_name="baseline_minus_kg",
    )
    rows = list(_parse_rows(_arm_rows(config, variant=True)))
    contradictory = RuleDecision(
        "kg.rule",
        RULE_CLUSTER_KG,
        RuleDecisionKind.SQL,
    )
    rows[0] = replace(
        rows[0],
        provenance=replace(
            rows[0].provenance,
            decisions=(
                RuleDecision(
                    "compiler.v23",
                    RULE_CLUSTER_VERSIONED_COMPILERS,
                    RuleDecisionKind.SQL,
                ),
                contradictory,
            ),
        ),
    )
    rows_tuple = tuple(rows)
    manifest = _manifest(
        config,
        rows_tuple,
        arm_id="baseline_minus_kg",
        arm_role="variant",
        target_clusters=(RULE_CLUSTER_KG,),
    )
    with pytest.raises(ValueError, match="disabled rule cannot have an SQL decision"):
        MODULE.bind_arm(
            manifest,
            MODULE.LoadedResultSet("f" * 64, rows_tuple),
        )


def test_loaded_result_set_rejects_duplicate_direct_rows_and_mutable_input():
    config = NL2SQLAblationConfig.baseline()
    row = _parse_rows([_row(config, "item", True)])[0]
    mutable = [row]
    loaded = MODULE.LoadedResultSet("f" * 64, mutable)
    mutable.append(row)
    assert loaded.rows == (row,)
    with pytest.raises(ValueError, match="duplicate result identity"):
        MODULE.LoadedResultSet("f" * 64, (row, row))
    with pytest.raises(ValueError, match="file_sha256"):
        MODULE.LoadedResultSet("not-a-hash", (row,))


    args = MODULE._parser().parse_args(
        [
            "--baseline",
            "baseline.jsonl",
            "--baseline-manifest",
            "baseline.json",
            "--variant",
            "variant_a=variant.jsonl",
            "--variant-manifest",
            "variant_a=variant.manifest.json",
            "--output",
            "report.json",
        ]
    )
    defaults = AblationAcceptanceCriteria()

    assert args.min_pooled_delta == defaults.min_pooled_delta
    assert args.max_shadow_loss == defaults.max_shadow_loss
    assert args.min_shadow_groups == defaults.min_shadow_groups
    assert args.min_alternate_dbs == defaults.min_alternate_dbs
    assert args.max_stability_range == defaults.max_stability_range
    assert args.max_stability_stddev == defaults.max_stability_stddev
    assert args.min_stability_runs == defaults.min_stability_runs


def test_cli_writes_create_once_manifest_bound_path_independent_report(tmp_path):
    baseline_config = NL2SQLAblationConfig.baseline(profile_name="baseline")
    variant_config = NL2SQLAblationConfig.without(
        RULE_CLUSTER_KG,
        profile_name="baseline_minus_kg",
    )
    baseline_path, baseline_manifest_path, _ = _write_bound_inputs(
        tmp_path,
        name="baseline",
        config=baseline_config,
        rows=_arm_rows(baseline_config),
        arm_role="baseline",
    )
    variant_path, variant_manifest_path, _ = _write_bound_inputs(
        tmp_path,
        name="baseline_minus_kg",
        config=variant_config,
        rows=_arm_rows(variant_config, variant=True),
        arm_role="variant",
        target_clusters=(RULE_CLUSTER_KG,),
    )
    output = tmp_path / "report.json"

    exit_code = MODULE.main(
        [
            "--baseline",
            str(baseline_path),
            "--baseline-manifest",
            str(baseline_manifest_path),
            "--variant",
            f"baseline_minus_kg={variant_path}",
            "--variant-manifest",
            f"baseline_minus_kg={variant_manifest_path}",
            "--output",
            str(output),
            "--require-acceptance",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["kind"] == "nl2sql_manifest_bound_ablation_report"
    assert payload["all_variants_accepted"] is True
    assert output.read_bytes() == MODULE.encode_report(payload)
    assert str(tmp_path) not in output.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        MODULE.main(
            [
                "--baseline",
                str(baseline_path),
                "--baseline-manifest",
                str(baseline_manifest_path),
                "--variant",
                f"baseline_minus_kg={variant_path}",
                "--variant-manifest",
                f"baseline_minus_kg={variant_manifest_path}",
                "--output",
                str(output),
            ]
        )
