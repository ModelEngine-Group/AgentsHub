"""Core Agent-only regressions for the sealed NL2SQL metric path."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from dki.analysis.nl2sql_eval import _score_item
from dki.analysis.nl2sql_seal import _agent_runtime_profile


class _SingleCallAgent:
    def __init__(self, response: dict):
        self.response = response
        self.calls: list[str] = []

    def query(self, question: str) -> dict:
        self.calls.append(question)
        return dict(self.response)


def _wide_database(path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE patients(patient_id TEXT)")
    connection.executemany(
        "INSERT INTO patients VALUES (?)",
        [(f"p{index:03d}",) for index in range(205)],
    )
    connection.commit()
    connection.close()


def test_agent_evaluator_calls_product_query_once_and_does_not_truncate(tmp_path):
    database = tmp_path / "blind.sqlite"
    _wide_database(database)
    sql = "SELECT patient_id FROM patients"
    agent = _SingleCallAgent({"sql": sql, "engine": "llm"})

    detail = _score_item(
        agent,
        {
            "id": "agent-001",
            "question": "return every patient id",
            "gold_sql": sql,
            "difficulty": "hard",
            "category": "filtering",
        },
        "agent",
        database,
        strict=True,
    )

    assert agent.calls == ["return every patient id"]
    assert detail["correct"] is True
    assert detail["engine"] == "llm"


def test_agent_evaluator_preserves_terminal_blocked_abstention(tmp_path):
    database = tmp_path / "blind.sqlite"
    _wide_database(database)
    agent = _SingleCallAgent(
        {
            "sql": "",
            "engine": "none",
            "terminal": True,
            "disposition": "blocked",
        }
    )

    detail = _score_item(
        agent,
        {
            "id": "agent-002",
            "question": "unsafe partial relation request",
            "gold_sql": "SELECT patient_id FROM patients",
            "difficulty": "hard",
            "category": "knowledge_graph",
        },
        "agent",
        database,
        strict=True,
    )

    assert agent.calls == ["unsafe partial relation request"]
    assert detail["correct"] is False
    assert detail["terminal"] is True
    assert detail["reason"] == "blocked"


def test_agent_profile_binds_provider_without_leaking_api_key(monkeypatch):
    import dki.paths as paths

    monkeypatch.setattr(paths, "LLM_API_KEY", "secret-never-published")
    monkeypatch.setattr(paths, "LLM_API_URL", "https://example.invalid/v1")
    monkeypatch.setattr(paths, "LLM_MODEL_NAME", "model-a")
    first = _agent_runtime_profile()

    assert first["contains_api_key"] is False
    assert "secret-never-published" not in repr(first)
    assert "https://example.invalid/v1" not in repr(first)
    assert first["llm_samples"] == 1
    assert first["llm_rounds"] == 2
    assert first["llm_temperature"] == 0.0
    assert first["runtime_error_policy"] == "fail_closed"
    assert first["llm_strategy"] == "legacy_sql"
    assert first["agent_rule_policy"] == "rule_first"
    assert first["llm_reasoning_mode"] == "disabled"
    assert first["llm_completion_tokens"] == 1024
    assert "query_plan_source_sha256" not in first
    assert "query_plan_schema_version" not in first

    monkeypatch.setattr(paths, "LLM_API_URL", "https://other.invalid/v1")
    second = _agent_runtime_profile()
    assert second["api_url_sha256"] != first["api_url_sha256"]


def test_query_plan_agent_profile_binds_strategy_source_without_secret(monkeypatch):
    import dki.paths as paths

    monkeypatch.setattr(paths, "LLM_API_KEY", "query-plan-secret")
    monkeypatch.setattr(paths, "LLM_API_URL", "https://example.invalid/v1")
    monkeypatch.setattr(paths, "LLM_MODEL_NAME", "model-plan")

    profile = _agent_runtime_profile(
        llm_strategy="query_plan",
        agent_rule_policy="plan_first",
        llm_reasoning_mode="adaptive",
        llm_completion_tokens=3072,
    )
    query_plan_source = (
        Path(__file__).resolve().parents[1]
        / "dki"
        / "analysis"
        / "nl2sql_query_plan.py"
    )

    assert profile["schema_version"] == 1
    assert profile["llm_strategy"] == "query_plan"
    assert profile["agent_rule_policy"] == "plan_first"
    assert profile["llm_reasoning_mode"] == "adaptive"
    assert profile["llm_completion_tokens"] == 3072
    assert profile["query_plan_schema_version"] == 1
    assert profile["query_plan_source_sha256"] == hashlib.sha256(
        query_plan_source.read_bytes()
    ).hexdigest()
    assert profile["contains_api_key"] is False
    assert "query-plan-secret" not in repr(profile)
    assert "https://example.invalid/v1" not in repr(profile)


def test_query_plan_v2_profile_and_development_evaluator_are_wired(monkeypatch, tmp_path):
    import dki.paths as paths
    import scripts.eval_nl2sql_frozen_development as evaluator

    monkeypatch.setattr(paths, "LLM_API_KEY", "v2-secret")
    monkeypatch.setattr(paths, "LLM_API_URL", "https://api.xiaomimimo.com/v1")
    monkeypatch.setattr(paths, "LLM_MODEL_NAME", "mimo-v2.5")
    profile = _agent_runtime_profile(
        llm_strategy="query_plan_v2",
        agent_rule_policy="plan_first",
        llm_reasoning_mode="adaptive",
        llm_completion_tokens=8192,
    )
    assert profile["query_plan_schema_version"] == 2
    assert profile["provider_profile"] == "mimo_adaptive_reasoning"

    captured = {}

    class _CapturedNL2SQL:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(evaluator, "NL2SQL", _CapturedNL2SQL)
    built = evaluator._nl2sql_for_evaluation(
        tmp_path / "development.sqlite",
        engine="agent",
        llm_strategy="query_plan_v2",
        agent_rule_policy="plan_first",
        llm_reasoning_mode="adaptive",
        llm_completion_tokens=8192,
    )
    assert isinstance(built, _CapturedNL2SQL)
    assert captured["llm_strategy"] == "query_plan_v2"
    assert captured["agent_rule_policy"] == "plan_first"


def test_minimax_m2_profile_does_not_claim_reasoning_control(monkeypatch):
    import dki.paths as paths

    monkeypatch.setattr(paths, "LLM_API_KEY", "m2-secret")
    monkeypatch.setattr(paths, "LLM_API_URL", "https://api.minimax.example/v1")
    monkeypatch.setattr(paths, "LLM_MODEL_NAME", "MiniMax-M2.7-highspeed")
    profile = _agent_runtime_profile(
        llm_strategy="query_plan_v2",
        agent_rule_policy="plan_first",
        llm_reasoning_mode="adaptive",
        llm_completion_tokens=8192,
    )
    assert profile["provider_profile"] == "minimax_m2_provider_default_reasoning"
    assert profile["llm_reasoning_mode"] == "adaptive"
    assert profile["llm_reasoning_mode_effective"] == "provider_default_reasoning"
    assert profile["llm_reasoning_control_supported"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_reasoning_mode": "always"},
        {"llm_completion_tokens": 0},
        {"llm_completion_tokens": 131073},
        {"llm_completion_tokens": True},
    ],
)
def test_agent_profile_rejects_invalid_inference_controls(overrides):
    with pytest.raises(ValueError):
        _agent_runtime_profile(**overrides)


@pytest.mark.parametrize(
    ("llm_strategy", "agent_rule_policy"),
    [
        ("legacy_sql", "plan_first"),
        ("query_plan", "model_first"),
    ],
)
def test_agent_profile_rejects_cross_strategy_rule_policies(
    monkeypatch,
    llm_strategy,
    agent_rule_policy,
):
    import dki.paths as paths

    monkeypatch.setattr(paths, "LLM_API_KEY", "secret-never-published")
    monkeypatch.setattr(paths, "LLM_API_URL", "https://example.invalid/v1")
    monkeypatch.setattr(paths, "LLM_MODEL_NAME", "model-a")

    with pytest.raises(ValueError, match="strategy/rule-policy combination"):
        _agent_runtime_profile(
            llm_strategy=llm_strategy,
            agent_rule_policy=agent_rule_policy,
        )


def test_frozen_development_engine_binds_strategy_and_rejects_rule_override(
    monkeypatch,
    tmp_path,
):
    import scripts.eval_nl2sql_frozen_development as evaluator

    captured = {}

    class _CapturedNL2SQL:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(evaluator, "NL2SQL", _CapturedNL2SQL)
    database = tmp_path / "development.sqlite"

    built = evaluator._nl2sql_for_evaluation(
        database,
        engine="agent",
        llm_strategy="legacy_sql",
        agent_rule_policy="model_first",
        llm_reasoning_mode="adaptive",
        llm_completion_tokens=4096,
    )

    assert isinstance(built, _CapturedNL2SQL)
    assert captured == {
        "db_path": database,
        "llm_strategy": "legacy_sql",
        "agent_rule_policy": "model_first",
        "llm_reasoning_mode": "adaptive",
        "llm_completion_tokens": 4096,
        "llm_rounds": 2,
        "llm_fail_closed": True,
    }
    with pytest.raises(ValueError, match="require engine=agent"):
        evaluator._nl2sql_for_evaluation(
            database,
            engine="rule",
            llm_strategy="query_plan",
            agent_rule_policy="rule_first",
        )


@pytest.mark.parametrize(
    ("llm_reasoning_mode", "llm_completion_tokens"),
    [
        ("adaptive", 1024),
        ("disabled", 2048),
    ],
)
def test_frozen_development_rule_engine_rejects_nondefault_llm_controls(
    tmp_path,
    llm_reasoning_mode,
    llm_completion_tokens,
):
    import scripts.eval_nl2sql_frozen_development as evaluator

    with pytest.raises(ValueError, match="require engine=agent"):
        evaluator._nl2sql_for_evaluation(
            tmp_path / "development.sqlite",
            engine="rule",
            llm_strategy="legacy_sql",
            agent_rule_policy="rule_first",
            llm_reasoning_mode=llm_reasoning_mode,
            llm_completion_tokens=llm_completion_tokens,
        )


@pytest.mark.parametrize(
    ("llm_strategy", "agent_rule_policy"),
    [
        ("legacy_sql", "plan_first"),
        ("query_plan", "model_first"),
    ],
)
def test_frozen_development_rejects_cross_strategy_rule_policies(
    tmp_path,
    llm_strategy,
    agent_rule_policy,
):
    import scripts.eval_nl2sql_frozen_development as evaluator

    with pytest.raises(ValueError, match="strategy/rule-policy combination"):
        evaluator._nl2sql_for_evaluation(
            tmp_path / "development.sqlite",
            engine="agent",
            llm_strategy=llm_strategy,
            agent_rule_policy=agent_rule_policy,
        )


def test_blind_evaluator_binds_selected_agent_profile(monkeypatch, tmp_path):
    import scripts.eval_nl2sql_blind as evaluator

    captured = {}

    class _CapturedNL2SQL:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(evaluator, "NL2SQL", _CapturedNL2SQL)
    database = tmp_path / "blind.sqlite"
    built = evaluator._nl2sql_for_evaluation(
        database,
        engine="agent",
        llm_strategy="legacy_sql",
        agent_rule_policy="model_first",
        llm_reasoning_mode="adaptive",
        llm_completion_tokens=8192,
    )

    assert isinstance(built, _CapturedNL2SQL)
    assert captured == {
        "db_path": database,
        "llm_strategy": "legacy_sql",
        "agent_rule_policy": "model_first",
        "llm_reasoning_mode": "adaptive",
        "llm_completion_tokens": 8192,
        "llm_fail_closed": True,
    }


def test_blind_agent_runtime_preflight_fails_before_holdout_on_provider_error(
    monkeypatch,
    tmp_path,
):
    import dki.llm as llm
    import scripts.eval_nl2sql_blind as evaluator

    class _Cleaner:
        @staticmethod
        def _clean_sql(value):
            return value

    monkeypatch.setattr(evaluator.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(llm, "llm_available", lambda: True)

    def _fail(*args, **kwargs):
        raise TimeoutError("synthetic provider timeout")

    monkeypatch.setattr(llm, "chat", _fail)

    with pytest.raises(TimeoutError, match="synthetic provider timeout"):
        evaluator._preflight_agent_runtime(
            _Cleaner(),
            engine="agent",
            database=tmp_path / "never-opened.sqlite",
            llm_reasoning_mode="adaptive",
            llm_completion_tokens=8192,
        )


def test_blind_rule_runtime_preflight_does_not_call_provider(tmp_path):
    import scripts.eval_nl2sql_blind as evaluator

    result = evaluator._preflight_agent_runtime(
        object(),
        engine="rule",
        database=tmp_path / "unused.sqlite",
        llm_reasoning_mode="disabled",
        llm_completion_tokens=1024,
    )

    assert result == {"status": "not_applicable", "engine": "rule"}


def test_blind_rule_engine_rejects_dormant_agent_override(tmp_path):
    import scripts.eval_nl2sql_blind as evaluator

    with pytest.raises(ValueError, match="require engine=agent"):
        evaluator._nl2sql_for_evaluation(
            tmp_path / "blind.sqlite",
            engine="rule",
            llm_strategy="legacy_sql",
            agent_rule_policy="model_first",
        )
