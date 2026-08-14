from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis import nl2sql_templates
from analysis.open_nl2sql import intent_router, schema_linker, sql_candidate_builder
from analysis.open_nl2sql.sql_explainer import (
    build_metric_definition,
    build_sql_response,
    summarize_rows,
)
from analysis.query_schema import QueryPlan, TimeWindow
from orchestration.execution_trace import write_trace
from runtime_common.common import read_json


@pytest.mark.parametrize(
    ("intent", "expected_route"),
    [
        ("graph_sql_joint_analysis", "graph_driven"),
        ("kg_subgraph", "graph_driven"),
        ("future_followup_chart", "analysis"),
        ("risk_distribution", "analysis"),
        ("cohort_stats", "analysis"),
        ("nl2sql", "analysis"),
        ("unknown", "standard"),
    ],
)
def test_legacy_intent_router_maps_planner_intents(
    monkeypatch: pytest.MonkeyPatch,
    intent: str,
    expected_route: str,
) -> None:
    monkeypatch.setattr(
        intent_router,
        "plan_query",
        lambda question: SimpleNamespace(intent=intent),
    )
    result = intent_router.route_intent("普通问题")
    assert result["route"] == expected_route


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        (
            "未来 30 天需要随访的高风险患者的疾病类型分布是什么？",
            "future_30d_high_risk_followup_disease_distribution",
        ),
        ("高盐饮食患者的血压异常比例是多少？", "high_salt_bp_abnormal_rate"),
        (
            "高血压合并糖尿病患者的 HbA1c、血压和 LDL-C 异常情况如何？",
            "hypertension_diabetes_multi_indicator",
        ),
        ("根据未来随访人数，绘制折线图，饼状图", "future_followup_chart_bundle"),
    ],
)
def test_legacy_intent_router_keeps_explicit_compatibility_routes(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    intent: str,
) -> None:
    monkeypatch.setattr(
        intent_router,
        "plan_query",
        lambda value: SimpleNamespace(intent="unknown"),
    )
    result = intent_router.route_intent(question)
    assert result == {"route": "graph_driven", "intent": intent}


def test_legacy_schema_linker_deduplicates_tables_and_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        schema_linker,
        "get_schema_registry",
        lambda: [
            {
                "table": "patient_profile",
                "field": "disease_tags",
                "chinese_alias": ["疾病", "病种"],
            },
            {
                "table": "patient_profile",
                "field": "patient_id",
                "chinese_alias": ["患者"],
            },
            {
                "table": "lab_result",
                "field": "item_name",
                "chinese_alias": ["指标"],
            },
        ],
    )
    result = schema_linker.build_schema_links("患者疾病指标")
    assert result["tables"] == ["patient_profile", "lab_result"]
    assert result["columns"] == ["disease_tags", "patient_id", "item_name"]


def _plan(
    intent: str,
    diseases: list[str] | None = None,
    days: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        intent=intent,
        disease_filters=diseases or [],
        time_window=SimpleNamespace(value=days) if days else None,
    )


@pytest.mark.parametrize(
    ("route_intent", "fragment"),
    [
        (
            "future_30d_high_risk_followup_disease_distribution",
            "followup_plan",
        ),
        ("high_salt_bp_abnormal_rate", "salt_intake_level"),
        ("hypertension_diabetes_multi_indicator", "disease_tags"),
        ("future_followup_chart_bundle", "GROUP BY followup_date"),
    ],
)
def test_legacy_candidate_builder_explicit_routes(
    monkeypatch: pytest.MonkeyPatch,
    route_intent: str,
    fragment: str,
) -> None:
    monkeypatch.setattr(
        sql_candidate_builder,
        "plan_query",
        lambda question: _plan("unknown"),
    )
    result = sql_candidate_builder.build_sql_candidate(
        "问题",
        {"intent": route_intent},
        {},
    )
    assert result["executable"] is True
    assert fragment in result["sql"]


@pytest.mark.parametrize(
    ("plan", "question", "fragment"),
    [
        (_plan("risk_distribution"), "风险分布", "risk_level"),
        (
            _plan("cohort_stats", ["hypertension", "diabetes"]),
            "高血压糖尿病",
            "AND lower(p.disease_tags)",
        ),
        (
            _plan("cohort_stats", ["hypertension"]),
            "高血压",
            "hypertension",
        ),
        (_plan("cohort_stats", ["diabetes"]), "糖尿病", "diabetes"),
        (
            _plan("future_followup_chart", days=9),
            "未来9天随访",
            "'+9 day'",
        ),
    ],
)
def test_legacy_candidate_builder_planner_routes(
    monkeypatch: pytest.MonkeyPatch,
    plan: SimpleNamespace,
    question: str,
    fragment: str,
) -> None:
    monkeypatch.setattr(sql_candidate_builder, "plan_query", lambda value: plan)
    result = sql_candidate_builder.build_sql_candidate(
        question,
        {"intent": "planner"},
        {},
    )
    assert result["executable"] is True
    assert fragment in result["sql"]


def test_legacy_candidate_builder_returns_fallback_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sql_candidate_builder,
        "plan_query",
        lambda value: _plan("unknown"),
    )
    links = {"tables": ["patient_profile"]}
    result = sql_candidate_builder.build_sql_candidate(
        "无法模板化",
        {"intent": "planner"},
        links,
    )
    assert result["executable"] is False
    assert result["sql"] is None
    assert result["schema_links"] == links


def test_sql_explainer_definitions_summaries_and_response() -> None:
    filters = {
        "time_window": {"label": "未来7天"},
        "disease_filters": ["hypertension", "diabetes"],
        "risk_filters": ["high"],
    }
    definition = build_metric_definition("随访人数", filters)
    assert "未来7天" in definition
    assert "hypertension, diabetes" in definition
    assert "high" in definition
    assert summarize_rows([]) == "查询已执行，但没有返回结果。"
    assert "count=2" in summarize_rows([{"count": 2}])
    assert "2 行结果" in summarize_rows([{"id": 1}, {"id": 2}])
    response = build_sql_response(
        question="随访人数",
        sql="SELECT 1",
        result=[{"count": 2}],
        filters=filters,
        warnings=["demo"],
    )
    assert response["is_safe"] is True
    assert response["warnings"] == ["demo"]
    assert response["summary"].startswith("查询返回 1 行")


def test_build_sql_candidates_handles_success_parse_failure_and_missing_template() -> None:
    items, errors = nl2sql_templates.build_sql_candidates(
        [
            {
                "id": "q1",
                "question": "人数",
                "intent": "count",
                "status": "success",
            },
            {
                "id": "q2",
                "question": "失败",
                "intent": "count",
                "status": "failed",
            },
            {
                "id": "q3",
                "question": "缺模板",
                "intent": "unknown",
                "status": "success",
            },
            {
                "id": "q4",
                "question": "自带模板",
                "intent": "custom",
                "status": "success",
                "sql_template": " SELECT 4 ",
                "expected_chart_type": "table",
            },
        ],
        {"count": "SELECT COUNT(*)"},
    )
    assert items[0]["sql"] == "SELECT COUNT(*)"
    assert items[1]["error"] == "question_parse_failed"
    assert items[2]["error"] == "template_not_found"
    assert items[3]["sql"] == "SELECT 4"
    assert items[3]["expected_chart_type"] == "table"
    assert len(errors) == 2


def test_query_schema_serializes_nested_time_window() -> None:
    window = TimeWindow(value=7, label="未来7天")
    plan = QueryPlan(
        intent="future_followup_chart",
        time_window=window,
        disease_filters=["hypertension"],
        confidence=0.95,
    )
    assert window.to_dict()["direction"] == "future"
    payload = plan.to_dict()
    assert payload["time_window"]["value"] == 7
    assert payload["disease_filters"] == ["hypertension"]


def test_execution_trace_writes_complete_audit_record(tmp_path: Path) -> None:
    path = tmp_path / "trace" / "run.json"
    write_trace(
        path=path,
        run_id="run-1",
        user_goal="统计人数",
        plan=[{"tool": "analysis"}],
        steps=[{"status": "success"}, {"status": "success"}],
        final_answer="共2000人",
        safety_note="仅用于辅助分析",
        agents_used=["ChronicCare"],
        artifacts_used=["report.json"],
    )
    trace = read_json(path)
    assert trace["run_id"] == "run-1"
    assert trace["tool_call_count"] == 2
    assert trace["created_at"]
    assert trace["agents_used"] == ["ChronicCare"]
