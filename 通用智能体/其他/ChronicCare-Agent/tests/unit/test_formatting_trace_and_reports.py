from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_adapter import trace_logger
from orchestration.answer_formatter import format_answer
from visualization.report_writer import (
    build_index_html,
    build_markdown_report,
    markdown_to_html,
)


def test_answer_formatter_preserves_failure_and_existing_answer() -> None:
    failure = {"status": "error", "message": "failed"}
    existing = {"status": "success", "answer": "already"}
    assert format_answer({}, failure) is failure
    assert format_answer({}, existing) == existing


@pytest.mark.parametrize(
    ("intent", "payload", "fragment"),
    [
        (
            "kg_summary",
            {
                "patient_count": 2000,
                "visit_count": 8231,
                "lab_result_count": 131323,
                "medication_record_count": 18248,
                "node_count": 197404,
                "edge_count": 396928,
            },
            "患者 2000 人",
        ),
        ("report_summary", {}, "HTTP URL"),
        ("capability_examples", {}, "不以固定题目总数"),
        ("kg_relation_query", {"insight": "风险因素关系"}, "风险因素关系"),
        ("kg_entity_query", {"text": "实体结果"}, "实体结果"),
        ("system_status", {}, "正常运行"),
        ("datamate_pipeline_status", {"summary": "最近成功"}, "最近成功"),
        ("datamate_pipeline_run", {}, "执行摘要"),
    ],
)
def test_answer_formatter_builds_expected_summary(
    intent: str,
    payload: dict,
    fragment: str,
) -> None:
    result = format_answer({"intent": intent}, {"status": "success", **payload})
    assert fragment in result["answer"]
    assert result["summary_text"] == result["answer"]


def test_answer_formatter_sets_graph_url_for_subgraph() -> None:
    result = format_answer(
        {"intent": "kg_subgraph_render"},
        {"status": "success", "html_url": "http://localhost/graph.html"},
    )
    assert result["graph_url"] == result["html_url"]
    assert result["html_url"] in result["answer"]


def test_answer_formatter_falls_back_to_summary_text() -> None:
    result = format_answer(
        {"intent": "unknown"},
        {"status": "success", "summary_text": "通用摘要"},
    )
    assert result["answer"] == "通用摘要"


def _trace_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    trace_file = tmp_path / "traces" / "calls.jsonl"
    summary_file = tmp_path / "traces" / "summary.json"
    monkeypatch.setenv("CHRONICCARE_TRACE_FILE", str(trace_file))
    monkeypatch.setenv("CHRONICCARE_TRACE_SUMMARY_FILE", str(summary_file))
    monkeypatch.setenv("CHRONICCARE_TRACE_DIR", str(trace_file.parent))
    return trace_file, summary_file


def test_trace_logger_append_deduplicate_and_summarize(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_file, summary_file = _trace_env(monkeypatch, tmp_path)
    trace_logger.append_trace(
        {
            "trace_id": "same",
            "tool_name": "tool_a",
            "status": "success",
            "latency_ms": 10,
        }
    )
    trace_logger.append_trace(
        {
            "trace_id": "same",
            "tool_name": "tool_a",
            "status": "success",
            "latency_ms": 20,
        }
    )
    trace_logger.append_trace(
        {
            "trace_id": "error",
            "tool_name": "tool_b",
            "status": "error",
            "latency_ms": 30,
        }
    )
    recent = trace_logger.load_recent_traces(limit=50)
    summary = trace_logger.summarize_traces()
    persisted = json.loads(summary_file.read_text(encoding="utf-8"))
    assert trace_file.exists()
    assert len(recent) == 2
    assert recent[0]["latency_ms"] == 20
    assert summary["total_calls"] == 2
    assert summary["success_calls"] == 1
    assert summary["error_calls"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["avg_latency_ms"] == 25.0
    assert persisted["tool_counts"] == {"tool_a": 1, "tool_b": 1}


def test_trace_logger_ignores_invalid_rows_and_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trace_file, _ = _trace_env(monkeypatch, tmp_path)
    trace_file.parent.mkdir(parents=True)
    trace_file.write_text(
        '\n{"trace_id":"a","tool_name":"x","status":"success"}\ninvalid\n[]\n'
        '{"trace_id":"b","tool_name":"y","status":"success"}\n',
        encoding="utf-8",
    )
    assert trace_logger.load_recent_traces(0) == []
    assert [item["trace_id"] for item in trace_logger.load_recent_traces(1)] == ["b"]


def test_trace_summary_is_empty_when_file_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _trace_env(monkeypatch, tmp_path)
    summary = trace_logger.summarize_traces()
    assert summary["total_calls"] == 0
    assert summary["success_rate"] == 0.0
    assert summary["recent_trace"] is None


def test_build_markdown_report_contains_metrics_and_safety_note() -> None:
    markdown = build_markdown_report(
        insights={
            "analysis_items": [
                {
                    "id": "q1",
                    "title": "疾病分布",
                    "question": "常见病有哪些？",
                    "chart_type": "bar",
                    "chart_path": "chart.html",
                    "insight": "高血压人数最多",
                }
            ],
            "safety_note": "仅用于辅助分析。",
        },
        indicator_doc={"items": [{"id": "q1"}]},
        nl2sql_eval={
            "sql_generation_success_rate": 0.95,
            "sql_executable_rate": 0.9,
            "result_success_rate": 0.88,
        },
        graph_summary={"node_count": 10, "edge_count": 20},
        kg_quality={},
        chart_index_path="index.html",
        sqlite_report={
            "tables": {
                "patient_profile": 2000,
                "visit_record": 8231,
                "lab_result": 131323,
                "medication_record": 18248,
            }
        },
    )
    assert "# ChronicCare-Agent分析报告" in markdown
    assert "图谱节点数: 10" in markdown
    assert "高血压人数最多" in markdown
    assert "仅用于辅助分析。" in markdown


def test_markdown_to_html_escapes_untrusted_text() -> None:
    html = markdown_to_html(
        "# 标题\n## 二级\n### 三级\n- <script>alert(1)</script>\n普通文本",
        "报告 <标题>",
    )
    assert "<h1>标题</h1>" in html
    assert "&lt;script&gt;" in html
    assert "<script>" not in html
    assert "<title>报告 &lt;标题&gt;</title>" in html


def test_build_index_html_renders_rows_and_escapes_safety_note() -> None:
    html = build_index_html(
        [
            {
                "title": "趋势图",
                "question": "最近趋势？",
                "chart_type": "line",
                "path": "trend.html",
            }
        ],
        "不得用于<诊断>",
    )
    assert "ChronicCare-Agent图表索引" in html
    assert "trend.html" in html
    assert "不得用于&lt;诊断&gt;" in html
