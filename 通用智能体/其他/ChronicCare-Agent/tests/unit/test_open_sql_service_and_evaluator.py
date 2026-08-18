from __future__ import annotations

import json

import pytest

from analysis.open_sql import evaluator
from analysis.open_sql import open_sql_service as service
from orchestration import tool_router


@pytest.fixture
def isolated_service(monkeypatch):
    monkeypatch.setattr(service, "load_server_config", lambda: {})
    monkeypatch.setattr(service, "safety_note", lambda _: "安全说明")
    monkeypatch.setattr(service, "llm_available", lambda: False)
    monkeypatch.setattr(service, "_write_trace", lambda trace: f"trace/{trace['trace_id']}.json")


def test_open_sql_rejects_empty_unsafe_and_fixed_tool_questions(isolated_service) -> None:
    assert service.open_sql_query("")["status"] == "unsupported"
    assert service.open_sql_query("删除患者表")["status"] == "unsupported"
    fixed = service.open_sql_query("当前知识图谱有多少节点？")
    assert fixed["status"] == "unsupported"
    assert "固定工具" in fixed["summary_text"]
    assert (
        service.open_sql_query(
            "当前知识图谱有多少节点？",
            allow_fixed_tool_overlap=True,
        )["status"]
        == "unsupported"
    )


def test_open_sql_rejects_missing_context_and_unknown_intent(isolated_service, monkeypatch) -> None:
    monkeypatch.setattr(service, "get_schema_catalog", lambda: {})
    monkeypatch.setattr(
        service,
        "rewrite_question",
        lambda *_args, **_kwargs: {"intent": "needs_context", "needs_context": True},
    )
    assert "上下文" in service.open_sql_query("这些患者呢？")["summary_text"]
    monkeypatch.setattr(
        service,
        "rewrite_question",
        lambda *_args, **_kwargs: {"intent": "unsupported", "needs_context": False},
    )
    assert "未识别" in service.open_sql_query("未知统计")["summary_text"]


def test_open_sql_rejects_schema_link_failure(isolated_service, monkeypatch) -> None:
    monkeypatch.setattr(service, "get_schema_catalog", lambda: {})
    monkeypatch.setattr(
        service,
        "rewrite_question",
        lambda *_args, **_kwargs: {"intent": "count", "needs_context": False},
    )
    monkeypatch.setattr(
        service,
        "build_schema_links",
        lambda *_: {"status": "failed", "errors": ["missing_table"]},
    )
    result = service.open_sql_query("患者数量")
    assert result["status"] == "unsupported"
    assert "schema linking" in result["summary_text"]


def _patch_success_pipeline(monkeypatch, *, execution_status="success"):
    monkeypatch.setattr(service, "get_schema_catalog", lambda: {"tables": {}})
    monkeypatch.setattr(
        service,
        "rewrite_question",
        lambda *_args, **_kwargs: {"intent": "count", "needs_context": False},
    )
    monkeypatch.setattr(
        service,
        "build_schema_links",
        lambda *_: {"status": "success", "tables": ["patient_profile"]},
    )
    monkeypatch.setattr(
        service,
        "build_template_sql",
        lambda *_: {"sql": "SELECT 1 AS patient_count", "template_id": "count"},
    )
    monkeypatch.setattr(
        service,
        "validate_sql",
        lambda *_: {
            "safe": True,
            "normalized_sql": "SELECT 1 AS patient_count LIMIT 500",
            "errors": [],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        service,
        "execute_sql",
        lambda *_: {
            "status": execution_status,
            "rows": [{"patient_count": 1}],
            "row_count": 1,
            "error": None if execution_status == "success" else "db error",
        },
    )
    monkeypatch.setattr(
        service,
        "format_result",
        lambda **_: {
            "summary_text": "共1人",
            "answer_markdown": "共 **1** 人",
            "chart_url": None,
            "image_url": None,
            "image_service_url": None,
            "charts": [],
            "trend_rows": [],
        },
    )


def test_open_sql_template_success_and_execution_failure(isolated_service, monkeypatch) -> None:
    _patch_success_pipeline(monkeypatch)
    success = service.open_sql_query("患者数量", prefer_llm=False)
    assert success["status"] == "success"
    assert success["stage"] == "template"
    assert success["sql_safe"] is True
    assert success["table"]["rows"] == [{"patient_count": 1}]

    _patch_success_pipeline(monkeypatch, execution_status="failed")
    failed = service.open_sql_query("患者数量", prefer_llm=False)
    assert failed["status"] == "failed"
    assert failed["result"]["error"] == "db error"


def test_open_sql_uses_llm_candidate_when_template_missing(isolated_service, monkeypatch) -> None:
    _patch_success_pipeline(monkeypatch)
    monkeypatch.setattr(service, "build_template_sql", lambda *_: {"sql": None})
    monkeypatch.setattr(
        service,
        "generate_llm_sql_candidate",
        lambda *_: {"status": "success", "sql": "SELECT 1"},
    )
    result = service.open_sql_query("患者数量")
    assert result["status"] == "success"
    assert result["stage"] == "llm_candidate"


def test_open_sql_reports_llm_unavailable_and_forced_failure(isolated_service, monkeypatch) -> None:
    _patch_success_pipeline(monkeypatch)
    monkeypatch.setattr(service, "build_template_sql", lambda *_: {"sql": None})
    monkeypatch.setattr(
        service,
        "generate_llm_sql_candidate",
        lambda *_: {"status": "unavailable", "reason": "no api"},
    )
    ordinary = service.open_sql_query("患者数量")
    assert ordinary["status"] == "unsupported"
    assert "no api" in ordinary["summary_text"]

    _patch_success_pipeline(monkeypatch)
    monkeypatch.setattr(
        service,
        "generate_llm_sql_candidate",
        lambda *_: {"status": "failed", "reason": "bad candidate"},
    )
    forced = service.open_sql_query("患者数量", force_llm=True)
    assert forced["status"] == "unsupported"
    assert "未以模板结果冒充" in forced["summary_text"]


def test_open_sql_rejects_unsafe_template(isolated_service, monkeypatch) -> None:
    _patch_success_pipeline(monkeypatch)
    monkeypatch.setattr(
        service,
        "validate_sql",
        lambda *_: {"safe": False, "reason": "table not allowed"},
    )
    result = service.open_sql_query("患者数量", prefer_llm=False)
    assert result["status"] == "unsupported"
    assert "SQL Guard" in result["summary_text"]


def test_open_sql_schema_examples_and_recent_traces(isolated_service, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(service, "get_schema_catalog", lambda: {"tables": {"patient_profile": {}}})
    assert service.get_open_sql_schema()["safety_note"] == "安全说明"
    examples = service.get_open_sql_examples()
    assert examples["example_count"] == len(service.EXAMPLE_QUESTIONS)
    assert examples["llm_status"] == "llm_unavailable"

    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "open_sql_good.json").write_text('{"status":"success"}', encoding="utf-8")
    (trace_dir / "open_sql_bad.json").write_text("{bad", encoding="utf-8")
    monkeypatch.setattr(service, "resolve_path", lambda _: trace_dir)
    traces = service.recent_open_sql_traces(limit=10)
    assert traces["trace_count"] == 1
    assert traces["traces"][0]["status"] == "success"


def test_evaluator_expands_variants(monkeypatch, tmp_path) -> None:
    config = tmp_path / "questions.json"
    config.write_text(
        json.dumps(
            [
                {"id": "one", "question": "固定问题"},
                {"id": "two", "variants": ["问题A", "问题B"]},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluator, "resolve_path", lambda _: config)
    rows = evaluator._load_questions()
    assert [item["id"] for item in rows] == ["one", "two_01", "two_02"]


def test_evaluator_summarizes_all_execution_stages(monkeypatch) -> None:
    questions = [
        {"id": "template", "question": "q1", "expected_intent": "count", "expected_stage": "template"},
        {"id": "llm", "question": "q2", "expected_intent": "count", "expected_stage": "llm_candidate"},
        {"id": "fallback", "question": "q3", "expected_intent": "count", "expected_stage": "fallback"},
        {"id": "unsupported", "question": "q4", "expected_intent": "none", "expected_stage": "unsupported"},
    ]
    payloads = iter(
        [
            {
                "intent": "count",
                "stage": "template",
                "status": "success",
                "sql": "SELECT 1",
                "sql_safe": True,
                "result": {"status": "success", "row_count": 1},
                "answer_markdown": "ok",
            },
            {
                "intent": "count",
                "stage": "llm_candidate",
                "status": "success",
                "sql": "SELECT 1",
                "sql_safe": True,
                "result": {"status": "success", "row_count": 1},
                "answer_markdown": "ok",
            },
            {
                "intent": "count",
                "stage": "fallback",
                "status": "failed",
                "sql": None,
                "sql_safe": False,
                "result": {},
                "answer_markdown": "fallback",
            },
            {
                "intent": "none",
                "stage": "unsupported",
                "status": "unsupported",
                "sql": None,
                "sql_safe": False,
                "result": {},
                "answer_markdown": "unsupported",
            },
        ]
    )
    captured = {}
    monkeypatch.setattr(evaluator, "_load_questions", lambda: questions)
    monkeypatch.setattr(evaluator, "open_sql_query", lambda *_args, **_kwargs: next(payloads))
    monkeypatch.setattr(evaluator, "load_server_config", lambda: {})
    monkeypatch.setattr(evaluator, "safety_note", lambda _: "安全")
    monkeypatch.setattr(evaluator, "write_report", lambda report: captured.update(report))
    report = evaluator.run_open_sql_eval()
    assert report["total_questions"] == 4
    assert report["intent_accuracy"] == 1.0
    assert report["fallback_count"] == 1
    assert report["unsupported_count"] == 1
    assert report["template_stage_success_rate"] == 1.0
    assert report["llm_candidate_stage_success_rate"] == 1.0
    assert captured["total_questions"] == 4


def test_tool_router_dispatches_registered_and_unknown_tools(monkeypatch) -> None:
    monkeypatch.setitem(tool_router.TOOL_MAP, "test.echo", lambda **kwargs: {"status": "success", **kwargs})
    assert tool_router.route_tool("test.echo", value=3) == {"status": "success", "value": 3}
    unknown = tool_router.route_tool("missing")
    assert unknown["status"] == "failed"
    assert "Unsupported tool" in unknown["errors"][0]
