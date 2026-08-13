"""Focused Wave 1A tests for complete results and runtime profiles."""
from __future__ import annotations

import sqlite3
from dataclasses import FrozenInstanceError

import pytest

from dki.analysis.nl2sql_protocol import CompleteResultLimitError
from dki.analysis.nl2sql_protocol import CompleteResultSerializationError
from dki.analysis.nl2sql_protocol import execute_complete_readonly
from dki.analysis.nl2sql_protocol import execute_complete_readonly_fingerprint
from dki.analysis.nl2sql_protocol import fingerprint_complete_result
from dki.analysis.nl2sql_runtime_profile import NL2SQLRuntimeProfile
from dki.analysis.nl2sql_runtime_profile import construct_nl2sql


_COMPONENT_HASH = "a" * 64


def _profile(**overrides) -> NL2SQLRuntimeProfile:
    values = {
        "runtime_mode": "query_plan_v3",
        "rule_policy": "plan_first",
        "llm_samples": 3,
        "rounds": 2,
        "reasoning_mode": "adaptive",
        "token_budget": 4096,
        "fail_closed": True,
        "selector_version": "selector-v1",
        "selector_hash": _COMPONENT_HASH,
        "scorer_version": "scorer-v1",
        "scorer_hash": "b" * 64,
        "counterfactual_version": "counterfactual-v1",
        "counterfactual_hash": "c" * 64,
        "executor_version": "executor-v1",
        "executor_hash": "d" * 64,
    }
    values.update(overrides)
    return NL2SQLRuntimeProfile(**values)


def _write_patients_db(path, *, tail_value: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE patients (id INTEGER, payload TEXT)")
        connection.executemany(
            "INSERT INTO patients VALUES (?, ?)",
            [
                (index, tail_value if index == 249 else f"value-{index}")
                for index in range(250)
            ],
        )
        connection.commit()
    finally:
        connection.close()


def test_complete_execution_fingerprints_tail_after_row_200(tmp_path):
    first_db = tmp_path / "first.db"
    second_db = tmp_path / "second.db"
    _write_patients_db(first_db, tail_value="first-tail")
    _write_patients_db(second_db, tail_value="second-tail")
    sql = "SELECT id, payload FROM patients ORDER BY id"

    legacy_columns, legacy_rows = execute_complete_readonly(sql, first_db)
    first = execute_complete_readonly_fingerprint(sql, first_db)
    second = execute_complete_readonly_fingerprint(sql, second_db)

    assert legacy_columns == ["id", "payload"]
    assert len(legacy_rows) == 250
    assert first.row_count == 250
    assert first.rows[249][1] == "first-tail"
    assert second.rows[249][1] == "second-tail"
    assert first.result_sha256 != second.result_sha256


def test_ordered_and_unordered_row_semantics_preserve_multiplicity():
    columns = ["value"]
    rows = [(1,), (2,), (1,)]
    reordered = [(1,), (1,), (2,)]

    assert fingerprint_complete_result(columns, rows, row_semantics="ordered") != (
        fingerprint_complete_result(columns, reordered, row_semantics="ordered")
    )
    assert fingerprint_complete_result(columns, rows, row_semantics="unordered") == (
        fingerprint_complete_result(columns, reordered, row_semantics="unordered")
    )
    assert fingerprint_complete_result(
        columns,
        rows,
        row_semantics="unordered",
    ) != fingerprint_complete_result(
        columns,
        [(1,), (2,)],
        row_semantics="unordered",
    )


def test_result_identity_includes_column_arity_and_positional_order():
    one_column = fingerprint_complete_result(["left"], [(1,)])
    two_columns = fingerprint_complete_result(["left", "right"], [(1, None)])
    left_right = fingerprint_complete_result(["left", "right"], [(1, "x")])
    right_left = fingerprint_complete_result(["right", "left"], [("x", 1)])

    assert one_column != two_columns
    assert left_right != right_left


def test_result_identity_distinguishes_null_and_heterogeneous_scalar_types():
    values = [(None,), (1,), (1.0,), ("1",), (b"1",)]
    fingerprint = fingerprint_complete_result(["value"], values)

    assert fingerprint == fingerprint_complete_result(["value"], values)
    for replacement in ("null", 0, "1.0", "01", b"01"):
        changed = list(values)
        changed[0] = (replacement,)
        assert fingerprint != fingerprint_complete_result(["value"], changed)


def test_complete_result_fails_closed_on_bounds_and_unsupported_scalars():
    with pytest.raises(CompleteResultLimitError, match="max_rows"):
        fingerprint_complete_result(["value"], [(1,), (2,)], max_rows=1)
    with pytest.raises(CompleteResultLimitError, match="max_canonical_bytes"):
        fingerprint_complete_result(
            ["value"],
            [("large-result-value",)],
            max_canonical_bytes=8,
        )
    with pytest.raises(CompleteResultSerializationError, match="unsupported"):
        fingerprint_complete_result(["value"], [(object(),)])
    with pytest.raises(CompleteResultSerializationError, match="arity"):
        fingerprint_complete_result(["first", "second"], [(1,)])


@pytest.mark.parametrize(
    ("runtime_mode", "rule_policy"),
    [
        ("legacy_sql", "rule_first"),
        ("legacy_sql", "model_first"),
        ("query_plan", "rule_first"),
        ("query_plan", "plan_first"),
        ("query_plan_v2", "rule_first"),
        ("query_plan_v2", "plan_first"),
        ("query_plan_v3", "rule_first"),
        ("query_plan_v3", "plan_first"),
    ],
)
def test_runtime_profile_accepts_all_compatible_mode_policy_pairs(
    runtime_mode,
    rule_policy,
):
    profile = _profile(runtime_mode=runtime_mode, rule_policy=rule_policy)
    assert profile.runtime_mode == runtime_mode
    assert profile.rule_policy == rule_policy


def test_runtime_profile_hash_is_deterministic_sensitive_and_immutable():
    first = _profile()
    second = _profile()

    assert first.canonical_payload() == second.canonical_payload()
    assert first.canonical_sha256 == second.canonical_sha256
    assert first.canonical_hash == first.canonical_sha256
    assert _profile(rounds=3).canonical_sha256 != first.canonical_sha256
    with pytest.raises(FrozenInstanceError):
        first.rounds = 4
    with pytest.raises(TypeError):
        first.canonical_payload()["rounds"] = 4


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"runtime_mode": "future_mode"}, "runtime_mode"),
        ({"rule_policy": "future_policy"}, "rule_policy"),
        (
            {"runtime_mode": "legacy_sql", "rule_policy": "plan_first"},
            "incompatible",
        ),
        ({"llm_samples": 0}, "llm_samples"),
        ({"llm_samples": True}, "llm_samples"),
        ({"rounds": 0}, "rounds"),
        ({"reasoning_mode": "always"}, "reasoning_mode"),
        ({"token_budget": 0}, "token_budget"),
        ({"token_budget": True}, "token_budget"),
        ({"token_budget": 131_073}, "token_budget"),
        ({"fail_closed": 1}, "fail_closed"),
        ({"selector_version": ""}, "selector_version"),
        ({"selector_hash": "ABC"}, "selector_hash"),
    ],
)
def test_runtime_profile_rejects_invalid_fields(overrides, message):
    with pytest.raises(ValueError, match=message):
        _profile(**overrides)


def test_runtime_profile_lazy_constructor_mapping(monkeypatch, tmp_path):
    import dki.analysis.nl2sql as nl2sql_module

    captured = {}

    class FakeNL2SQL:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(nl2sql_module, "NL2SQL", FakeNL2SQL)
    db_path = tmp_path / "runtime.db"
    profile = _profile()

    instance = construct_nl2sql(profile, db_path, llm_temperature=0.25)

    assert isinstance(instance, FakeNL2SQL)
    assert captured == {
        "db_path": db_path,
        "llm_samples": 3,
        "llm_rounds": 2,
        "llm_strategy": "query_plan_v3",
        "agent_rule_policy": "plan_first",
        "llm_reasoning_mode": "adaptive",
        "llm_completion_tokens": 4096,
        "llm_fail_closed": True,
        "llm_temperature": 0.25,
    }
    with pytest.raises(ValueError, match="profile-controlled"):
        construct_nl2sql(profile, db_path, llm_samples=99)
