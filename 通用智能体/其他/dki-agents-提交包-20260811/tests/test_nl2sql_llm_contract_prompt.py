"""Core regression for the semantic-contract LLM compiler prompt."""

from __future__ import annotations

import json
import sqlite3

import pytest

from dki.analysis.nl2sql import NL2SQL
from dki.analysis.nl2sql_compile_result import CompileDisposition, CompileResult


def _empty_analysis_db(path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE patients (
                patient_id TEXT, name TEXT, gender TEXT, age INTEGER, region TEXT
            );
            CREATE TABLE visits (
                visit_id TEXT, patient_id TEXT, visit_date TEXT,
                department TEXT, disease TEXT, drug_prescribed TEXT,
                exam TEXT, cost REAL, los_days INTEGER
            );
            CREATE TABLE kg_entities (
                name TEXT, type TEXT, mention_count INTEGER
            );
            CREATE TABLE kg_triples (
                head TEXT, head_type TEXT, relation TEXT, relation_zh TEXT,
                tail TEXT, tail_type TEXT, confidence REAL, count INTEGER
            );
            INSERT INTO kg_entities VALUES ('心律失常', 'Disease', 2);
            INSERT INTO kg_entities VALUES ('冠心病', 'Disease', 2);
            INSERT INTO kg_entities VALUES ('心肌梗死', 'Disease', 3);
            INSERT INTO kg_entities VALUES ('心力衰竭', 'Disease', 4);
            INSERT INTO kg_triples VALUES
                ('心律失常', 'Disease', 'DIAGNOSED_BY', '诊断检查',
                 '心电图', 'Exam', 0.95, 3),
                ('冠心病', 'Disease', 'DIAGNOSED_BY', '诊断检查',
                 '心电图', 'Exam', 0.92, 2),
                ('心肌梗死', 'Disease', 'HAS_COMPLICATION', '并发症',
                 '心力衰竭', 'Disease', 0.91, 2);
            """
        )


def test_llm_prompt_binds_contract_rounding_and_total_order(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    captured = {}

    def fake_chat(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return (
            "SELECT department, ROUND(AVG(cost), 2) AS avg_cost "
            "FROM visits GROUP BY department "
            "ORDER BY avg_cost DESC, department"
        )

    monkeypatch.setattr("dki.llm.chat", fake_chat)
    nl = NL2SQL(
        db_path,
        llm_samples=1,
        llm_rounds=1,
        llm_reasoning_mode="adaptive",
        llm_completion_tokens=4096,
    )

    candidates = nl._llm_candidates("按科室计算平均费用，按平均费用降序。")

    assert len(candidates) == 1
    assert '"grouping": ["department"]' in captured["prompt"]
    assert "AVG(confidence) 使用 ROUND(..., 4)" in captured["prompt"]
    assert "时间分组必须以时间键升序为第一优先" in captured["prompt"]
    assert "DIAGNOSED_BY Disease->Exam" in captured["prompt"]
    assert "结构示例（只学习结构与输出合同，不复制常量）" in captured["prompt"]
    assert "WITH s AS" in captured["prompt"]
    assert "BELONGS_TO_DEPT" in captured["prompt"]
    assert "Drug 实体按不同头疾病数 TopK" in captured["prompt"]
    assert "组合结构示例（示例常量不得复制" in captured["prompt"]
    assert "DENSE_RANK() OVER (PARTITION BY department" in captured["prompt"]
    assert "relation_zh IN ('临床表现', '诊断检查')" in captured["prompt"]
    assert "SELECT MAX(x.cost) FROM visits x" in captured["prompt"]
    assert "'患者地区' AS source_name" in captured["prompt"]
    assert "strftime('%m', visit_date) = '09'" in captured["prompt"]
    assert "COUNT(*) OVER (PARTITION BY department)" in captured["prompt"]
    assert "禁止把 patient_id 换成 name" in captured["prompt"]
    assert captured["kwargs"]["max_tokens"] == 4096
    assert captured["kwargs"]["reasoning_mode"] == "adaptive"


def test_contract_guided_llm_defaults_use_one_candidate_and_one_retry(tmp_path) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)

    nl = NL2SQL(db_path)

    assert nl.llm_samples == 1
    assert nl.llm_rounds == 2
    assert nl.llm_temperature == 0.0
    assert nl.llm_strategy == "legacy_sql"
    assert nl.agent_rule_policy == "rule_first"
    assert nl.llm_reasoning_mode == "disabled"
    assert nl.llm_completion_tokens == 1024

    route_meta = nl._route_meta("rule", "supported")
    assert route_meta["llm_reasoning_mode"] == "disabled"
    assert route_meta["llm_completion_tokens"] == 1024
    assert all("key" not in key.casefold() for key in route_meta)


def test_result_fingerprint_handles_null_text_and_numeric_equivalence(tmp_path) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    nl = NL2SQL(db_path)

    forward = nl._result_key("SELECT NULL AS value UNION ALL SELECT 'text'")
    reverse = nl._result_key("SELECT 'text' AS value UNION ALL SELECT NULL")

    assert forward is not None
    assert forward == reverse
    assert nl._result_key("SELECT 1") == nl._result_key("SELECT 1.0")
    assert nl._result_key("SELECT NULL") != nl._result_key("SELECT 'None'")


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("llm_strategy", "free_sql"),
        ("agent_rule_policy", "rules_maybe"),
        ("llm_reasoning_mode", "always"),
        ("llm_completion_tokens", 0),
        ("llm_completion_tokens", 131073),
        ("llm_completion_tokens", True),
        ("llm_completion_tokens", "1024"),
    ),
)
def test_unknown_agent_routing_values_are_rejected(
    tmp_path,
    keyword,
    value,
) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)

    with pytest.raises(ValueError):
        NL2SQL(db_path, **{keyword: value})


@pytest.mark.parametrize(
    ("llm_strategy", "agent_rule_policy"),
    (
        ("legacy_sql", "plan_first"),
        ("query_plan", "model_first"),
    ),
)
def test_cross_strategy_first_policies_are_rejected(
    tmp_path,
    llm_strategy,
    agent_rule_policy,
) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)

    with pytest.raises(ValueError):
        NL2SQL(
            db_path,
            llm_strategy=llm_strategy,
            agent_rule_policy=agent_rule_policy,
        )


def _stub_query_plan_compiler(monkeypatch, compiled_sql: str) -> None:
    monkeypatch.setattr("dki.llm.llm_available", lambda: True)
    monkeypatch.setattr(
        "dki.analysis.nl2sql_query_plan.query_plan_schema",
        lambda: {"type": "object", "additionalProperties": False},
    )
    monkeypatch.setattr(
        "dki.analysis.nl2sql_query_plan.parse_query_plan",
        lambda raw: {"plan_version": 1, "raw_seen": raw},
    )
    monkeypatch.setattr(
        "dki.analysis.nl2sql_query_plan.validate_plan_grounding",
        lambda plan, question, *, allowed_domain_values=(): None,
    )
    monkeypatch.setattr(
        "dki.analysis.nl2sql_query_plan.compile_query_plan",
        lambda plan: compiled_sql,
    )
    monkeypatch.setattr(
        "dki.analysis.nl2sql_query_plan.canonical_plan_sha256",
        lambda plan: "a" * 64,
    )


def test_query_plan_compiles_json_and_records_plan_identity(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    prompts = []
    chat_kwargs = []

    def fake_chat(prompt, **kwargs):
        prompts.append(prompt)
        chat_kwargs.append(kwargs)
        return json.dumps(
            {
                "plan_version": 1,
                "from": {"table": "visits", "alias": "v"},
                "select": [
                    {
                        "expr": {
                            "type": "column",
                            "table_alias": "v",
                            "column": "visit_id",
                        }
                    }
                ],
            }
        )

    monkeypatch.setattr("dki.llm.llm_available", lambda: True)
    monkeypatch.setattr("dki.llm.chat", fake_chat)
    # Hermetic provider neutrality: the developer .env may select a provider
    # with a completion-token floor (for example MiniMax M2's 2048); this
    # test asserts the configured budget itself, so pin a neutral provider.
    monkeypatch.setattr("dki.paths.LLM_API_URL", "https://neutral.example/v1")
    monkeypatch.setattr("dki.paths.LLM_MODEL_NAME", "neutral-model")
    nl = NL2SQL(db_path, llm_strategy="query_plan", llm_rounds=1)

    def unavailable_advisory(question, sql):
        raise RuntimeError("heuristic validator unavailable")

    monkeypatch.setattr(
        nl,
        "_query_plan_semantic_advisory",
        unavailable_advisory,
    )
    monkeypatch.setattr(
        nl,
        "_rule_sql_normalized_result",
        lambda question: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )

    result = nl.to_sql("列出 visits 表的 visit_id。")

    assert result["engine"] == "llm"
    normalized_sql = result["sql"].replace('"', "")
    assert "SELECT v.visit_id" in normalized_sql
    assert "FROM visits AS v" in normalized_sql
    assert result["llm_meta"]["strategy"] == "query_plan"
    assert result["llm_meta"]["plan_version"] == 1
    assert len(result["llm_meta"]["plan_sha256"]) == 64
    int(result["llm_meta"]["plan_sha256"], 16)
    assert result["llm_meta"]["execution_validated"] is True
    assert result["llm_meta"]["semantic_validated"] is False
    assert result["llm_meta"]["semantic_warnings"] == [
        "semantic_advisory_unavailable"
    ]
    assert result["plan"]["agent_rule_policy"] == "rule_first"
    assert "只输出一个严格符合上述 JSON Schema 的 JSON 对象" in prompts[0]
    assert "禁止 sql、raw_sql" in prompts[0]
    assert "只输出 SQL，不要解释" not in prompts[0]
    assert "示例问题：列出心血管内科的就诊编号" in prompts[0]
    assert '"having":{' in prompts[0]
    assert '"path":"patients_visits"' in prompts[0]
    assert chat_kwargs == [
        {
            "temperature": 0.0,
            # Provider-neutral tests must retain the configured budget.  The
            # MiniMax M2 floor is covered separately with an explicit provider
            # profile; relying on a developer .env made this test non-hermetic.
            "max_tokens": 1024,
            "reasoning_mode": "disabled",
        }
    ]
    assert result["plan"]["llm_reasoning_mode"] == "disabled"
    assert result["plan"]["llm_completion_tokens"] == 1024
    assert result["llm_meta"]["llm_reasoning_mode"] == "disabled"
    assert result["llm_meta"]["llm_completion_tokens"] == 1024


def test_invalid_query_plan_retries_without_raw_feedback_then_abstains(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    raw_responses = [
        "SELECT * FROM visits",
        '{"plan_version": 1, "sql": "SELECT * FROM visits"}',
    ]
    prompts = []

    monkeypatch.setattr("dki.llm.llm_available", lambda: True)

    def fake_chat(prompt, **kwargs):
        prompts.append(prompt)
        return raw_responses[len(prompts) - 1]

    def reject_raw_sql(raw):
        raise ValueError("invalid plan contains private literal: do-not-echo")

    monkeypatch.setattr("dki.llm.chat", fake_chat)
    monkeypatch.setattr(
        "dki.analysis.nl2sql_query_plan.query_plan_schema",
        lambda: {"type": "object"},
    )
    monkeypatch.setattr(
        "dki.analysis.nl2sql_query_plan.parse_query_plan",
        reject_raw_sql,
    )
    nl = NL2SQL(db_path, llm_strategy="query_plan", llm_rounds=2)
    monkeypatch.setattr(
        nl,
        "_rule_sql_normalized_result",
        lambda question: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )
    monkeypatch.setattr(
        nl,
        "_llm_candidates",
        lambda *args, **kwargs: pytest.fail("QueryPlan must not use free-SQL fallback"),
    )

    result = nl.to_sql("无法由规则处理的查询")

    assert result["engine"] == "none"
    assert len(prompts) == 2
    assert "阶段 json_schema 失败" in prompts[1]
    assert "do-not-echo" not in prompts[1]
    assert raw_responses[0] not in prompts[1]
    assert result["llm_meta"]["strategy"] == "query_plan"
    assert result["llm_meta"]["abstained"] is True
    assert nl._last_llm_meta["abstained"] is True


def test_invalid_query_plan_can_recover_with_valid_json_on_next_round(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    valid_plan = json.dumps(
        {
            "plan_version": 1,
            "from": {"table": "visits", "alias": "v"},
            "select": [
                {
                    "expr": {
                        "type": "column",
                        "table_alias": "v",
                        "column": "visit_id",
                    }
                }
            ],
        }
    )
    responses = ["not-json-and-not-sql", valid_plan]
    prompts = []

    def fake_chat(prompt, **kwargs):
        prompts.append(prompt)
        return responses[len(prompts) - 1]

    monkeypatch.setattr("dki.llm.llm_available", lambda: True)
    monkeypatch.setattr("dki.llm.chat", fake_chat)
    nl = NL2SQL(db_path, llm_strategy="query_plan", llm_rounds=2)
    monkeypatch.setattr(
        nl,
        "_rule_sql_normalized_result",
        lambda question: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )

    result = nl.to_sql("列出 visits 表的 visit_id。")

    assert result["engine"] == "llm"
    assert result["llm_meta"]["rounds"] == 2
    assert result["llm_meta"]["attempts"] == 2
    assert "阶段 json_schema 失败" in prompts[1]
    assert responses[0] not in prompts[1]


def test_query_plan_grounding_receives_original_question(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    captured = {}
    _stub_query_plan_compiler(
        monkeypatch,
        "SELECT COUNT(*) AS n FROM visits",
    )
    monkeypatch.setattr(
        "dki.analysis.nl2sql_query_plan.validate_plan_grounding",
        lambda plan, question, *, allowed_domain_values=(): captured.update(
            question=question,
            allowed_domain_values=allowed_domain_values,
        ),
    )
    monkeypatch.setattr("dki.llm.chat", lambda *args, **kwargs: '{"plan_version":1}')
    nl = NL2SQL(db_path, llm_strategy="query_plan", llm_rounds=1)
    monkeypatch.setattr(nl, "_normalize", lambda question: "数据库规范化后的别名")
    monkeypatch.setattr(
        nl,
        "_rule_sql_normalized_result",
        lambda question: CompileResult(CompileDisposition.NOT_APPLICABLE),
    )

    result = nl.to_sql("用户原始别名")

    assert result["engine"] == "llm"
    assert captured["question"] == "用户原始别名"
    assert captured["question"] != "数据库规范化后的别名"


def test_legacy_execution_retry_feedback_contains_no_query_sql_or_exception(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    nl = NL2SQL(
        db_path,
        agent_rule_policy="model_first",
        llm_rounds=2,
    )
    question = "QUESTION_EXECUTION_SENTINEL"
    rejected_sql = "SELECT EXECUTION_SQL_SENTINEL FROM missing_table"
    accepted_sql = "SELECT 1 AS repaired"
    feedbacks = []

    monkeypatch.setattr("dki.llm.llm_available", lambda: True)

    def candidates(seen_question, feedback=""):
        assert seen_question == question
        feedbacks.append(feedback)
        return [[rejected_sql], [accepted_sql]][len(feedbacks) - 1]

    monkeypatch.setattr(nl, "_llm_candidates", candidates)
    monkeypatch.setattr(
        nl,
        "_result_key",
        lambda sql: None if sql == rejected_sql else "executable",
    )
    result = nl._llm_pipeline(question)

    assert result is not None
    assert result["sql"] == accepted_sql
    assert feedbacks[0] == ""
    assert "execution_validation" in feedbacks[1]
    assert question not in feedbacks[1]
    assert rejected_sql not in feedbacks[1]
    assert "EXECUTION_SQL_SENTINEL" not in feedbacks[1]
    assert "missing_table" not in feedbacks[1]


def test_model_first_overrides_rule_sql_and_query_executes_model_route(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    nl = NL2SQL(db_path, agent_rule_policy="model_first")
    rule_result = CompileResult(
        CompileDisposition.SQL,
        "SELECT COUNT(*) AS rule_count FROM patients",
    )
    monkeypatch.setattr(
        nl,
        "_rule_sql_normalized_result",
        lambda question: rule_result,
    )
    calls = []

    def model_result(question):
        calls.append(question)
        return {
            "sql": "SELECT COUNT(*) AS model_count FROM visits",
            "confidence": 0.8,
            "n_candidates": 1,
            "n_agree": 1,
            "rounds": 1,
        }

    monkeypatch.setattr(nl, "_llm_pipeline", model_result)

    compiled = nl.to_sql("count records")
    queried = nl.query("count records")

    assert compiled["engine"] == "llm"
    assert "model_count" in compiled["sql"]
    assert queried["engine"] == "llm"
    assert queried["columns"] == ["model_count"]
    assert queried["provenance"]["agent_rule_policy"] == "model_first"
    assert queried["provenance"]["route"] == "schema_linking_llm"
    assert calls == ["count records", "count records"]


def test_model_first_failure_falls_back_only_to_existing_rule(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    nl = NL2SQL(db_path, agent_rule_policy="model_first")
    rule_result = CompileResult(
        CompileDisposition.SQL,
        "SELECT COUNT(*) AS rule_count FROM patients",
    )
    monkeypatch.setattr(
        nl,
        "_rule_sql_normalized_result",
        lambda question: rule_result,
    )
    calls = []

    def model_abstains(question):
        calls.append(question)
        nl._last_llm_meta = nl._llm_result_meta({"abstained": True})
        return None

    monkeypatch.setattr(nl, "_llm_pipeline", model_abstains)

    compiled = nl.to_sql("count records")
    queried = nl.query("count records")

    assert compiled["engine"] == "rule"
    assert compiled["sql"] == rule_result.sql
    assert compiled["plan"]["route"] == "rule_after_model_abstain"
    assert compiled["llm_meta"]["abstained"] is True
    assert queried["engine"] == "rule"
    assert queried["columns"] == ["rule_count"]
    assert queried["provenance"]["route"] == "rule_after_model_abstain"
    assert calls == ["count records", "count records"]


def test_plan_first_overrides_rule_sql_and_query_executes_same_route(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    calls = []

    def fake_chat(prompt, **kwargs):
        calls.append(prompt)
        return '{"plan_version": 1}'

    _stub_query_plan_compiler(
        monkeypatch,
        "SELECT COUNT(*) AS planned_count FROM visits",
    )
    monkeypatch.setattr("dki.llm.chat", fake_chat)
    nl = NL2SQL(
        db_path,
        llm_strategy="query_plan",
        agent_rule_policy="plan_first",
        llm_rounds=1,
    )
    rule_result = CompileResult(
        CompileDisposition.SQL,
        "SELECT COUNT(*) AS rule_count FROM patients",
    )
    monkeypatch.setattr(
        nl,
        "_rule_sql_normalized_result",
        lambda question: rule_result,
    )

    compiled = nl.to_sql("统计记录数")
    queried = nl.query("统计记录数")

    assert compiled["engine"] == "llm"
    assert "planned_count" in compiled["sql"]
    assert queried["engine"] == "llm"
    assert queried["columns"] == ["planned_count"]
    assert queried["provenance"]["agent_rule_policy"] == "plan_first"
    assert len(calls) == 2


def test_plan_first_falls_back_to_existing_rule_after_plan_abstention(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    monkeypatch.setattr("dki.llm.llm_available", lambda: True)
    monkeypatch.setattr("dki.llm.chat", lambda *args, **kwargs: "not json")
    nl = NL2SQL(
        db_path,
        llm_strategy="query_plan",
        agent_rule_policy="plan_first",
        llm_rounds=1,
    )
    rule_result = CompileResult(
        CompileDisposition.SQL,
        "SELECT COUNT(*) AS n FROM patients",
        contract_self_validated=True,
    )
    monkeypatch.setattr(
        nl,
        "_rule_sql_normalized_result",
        lambda question: rule_result,
    )
    monkeypatch.setattr(
        nl,
        "_authoritative_rule_before_model",
        lambda question, result: True,
    )

    result = nl.to_sql("统计患者数")
    queried = nl.query("统计患者数")

    assert result["engine"] == "rule"
    assert result["sql"] == rule_result.sql
    assert result["plan"]["route"] == "rule_after_query_plan_abstain"
    assert queried["engine"] == "rule"
    assert queried["columns"] == ["n"]
    assert queried["provenance"]["route"] == "rule_after_query_plan_abstain"


def test_default_rule_first_keeps_rule_precedence_with_query_plan_enabled(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    nl = NL2SQL(db_path, llm_strategy="query_plan")
    rule_result = CompileResult(
        CompileDisposition.SQL,
        "SELECT COUNT(*) AS n FROM patients",
        contract_self_validated=True,
    )
    monkeypatch.setattr(
        nl,
        "_rule_sql_normalized_result",
        lambda question: rule_result,
    )
    monkeypatch.setattr(
        nl,
        "_authoritative_rule_before_model",
        lambda question, result: True,
    )
    monkeypatch.setattr(
        nl,
        "_llm_pipeline",
        lambda question: pytest.fail("rule_first must keep a rule SQL result"),
    )

    result = nl.to_sql("统计患者数")

    assert result["engine"] == "rule"
    assert result["sql"] == rule_result.sql
    assert result["plan"]["agent_rule_policy"] == "rule_first"


def test_structured_plan_never_uses_unverified_legacy_rule_as_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    nl = NL2SQL(
        db_path,
        llm_strategy="query_plan",
        agent_rule_policy="plan_first",
    )
    monkeypatch.setattr(
        nl,
        "_rule_sql_normalized_result",
        lambda question: CompileResult(
            CompileDisposition.SQL,
            "SELECT COUNT(*) AS unsafe_partial_count FROM patients",
        ),
    )
    monkeypatch.setattr(nl, "_llm_pipeline", lambda question: None)
    monkeypatch.setattr(nl, "_llm_query", lambda question: None)

    compiled = nl.to_sql("一个没有完整结构合同的问题")
    queried = nl.query("一个没有完整结构合同的问题")

    assert compiled["engine"] == "none"
    assert compiled["sql"] == ""
    assert queried["engine"] == "none"
    assert queried["sql"] == ""


@pytest.mark.parametrize(
    ("llm_strategy", "agent_rule_policy", "structured_attempts"),
    (
        ("legacy_sql", "model_first", False),
        ("query_plan", "plan_first", True),
    ),
)
def test_blocked_rule_allows_only_fresh_structured_parse(
    tmp_path,
    monkeypatch,
    llm_strategy,
    agent_rule_policy,
    structured_attempts,
) -> None:
    """BLOCKED admits exactly one fresh structured parse per API, never a
    relaxed legacy pipeline.  Unified parity contract: ``to_sql`` and
    ``query`` make the same structured attempt (via ``_llm_pipeline`` and
    ``_llm_query`` respectively); the legacy strategy must not call the
    model at all on a BLOCKED rule."""

    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    nl = NL2SQL(
        db_path,
        llm_strategy=llm_strategy,
        agent_rule_policy=agent_rule_policy,
    )
    monkeypatch.setattr(
        nl,
        "_rule_sql_normalized_result",
        lambda question: CompileResult(CompileDisposition.BLOCKED),
    )
    pipeline_calls = []
    query_calls = []

    def pipeline_abstains(question):
        pipeline_calls.append(question)
        return None

    def structured_abstains(question):
        query_calls.append(question)
        return None

    monkeypatch.setattr(
        nl,
        "_llm_pipeline",
        pipeline_abstains
        if structured_attempts
        else lambda question: pytest.fail("legacy BLOCKED must not call the model"),
    )
    monkeypatch.setattr(
        nl,
        "_llm_query",
        structured_abstains
        if structured_attempts
        else lambda question: pytest.fail("legacy BLOCKED must not call the model"),
    )

    compiled = nl.to_sql("包含未支持限定条件的问题")
    queried = nl.query("包含未支持限定条件的问题")

    assert compiled["disposition"] == "blocked"
    assert queried["disposition"] == "blocked"
    assert compiled["plan"]["agent_rule_policy"] == agent_rule_policy
    assert queried["provenance"]["agent_rule_policy"] == agent_rule_policy
    assert pipeline_calls == (
        ["包含未支持限定条件的问题"] if structured_attempts else []
    )
    assert query_calls == (
        ["包含未支持限定条件的问题"] if structured_attempts else []
    )


def test_shared_tail_peer_is_compiled_without_invented_metric_filter(tmp_path) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    nl = NL2SQL(db_path)

    sql = nl.rule_sql(
        "请找出和心律失常拥有相同诊断检查的其余疾病及共同检查，"
        "关系使用 DIAGNOSED_BY。"
    )

    assert sql is not None
    assert "other.tail = base.tail" in sql
    assert "other.head <> base.head" in sql
    assert "base.head = '心律失常'" in sql
    assert "confidence" not in sql
    assert "count" not in sql


def test_kg_endpoint_join_is_compiled_without_extra_edge_columns(tmp_path) -> None:
    db_path = tmp_path / "analysis.db"
    _empty_analysis_db(db_path)
    nl = NL2SQL(db_path)

    sql = nl.rule_sql(
        "将图谱关系 HAS_COMPLICATION 的头实体和尾实体分别连接实体表，"
        "列出两侧名称及各自提及次数。"
    )

    assert sql is not None
    assert "JOIN kg_entities h ON h.name = t.head" in sql
    assert "JOIN kg_entities e ON e.name = t.tail" in sql
    assert "t.relation = 'HAS_COMPLICATION'" in sql
    select_clause = sql.split(" FROM ", 1)[0]
    assert "confidence" not in select_clause
    assert "t.relation" not in select_clause
