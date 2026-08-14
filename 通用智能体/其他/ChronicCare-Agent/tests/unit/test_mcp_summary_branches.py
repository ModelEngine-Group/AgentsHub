from __future__ import annotations

import pytest

from mcp_adapter import server


def test_core_mcp_summaries_render_structured_payloads() -> None:
    assert "发布阶段" in server.summarize_health(
        {"status": "ok", "project": "ChronicCare", "stage": "release"},
        "http://service",
    )
    kg = server.summarize_kg(
        {
            "patient_count": 2000,
            "visit_count": 8231,
            "lab_result_count": 131323,
            "medication_record_count": 18248,
            "node_count": 197404,
            "edge_count": 396928,
            "entity_type_total_count": 14,
            "relation_type_total_count": 15,
            "top_entity_types": [["Patient", 2000]],
            "top_relation_types": [["has_lab", 131323]],
            "graph_url": "http://graph",
        }
    )
    assert "197404" in kg and "http://graph" in kg
    data = server.summarize_data_summary(
        {
            "data_version": "synthetic_chroniccare",
            "patient_count": 2000,
            "visit_count": 8231,
            "lab_result_count": 131323,
            "medication_record_count": 18248,
            "node_count": 197404,
            "edge_count": 396928,
            "table": {"rows": [{"指标": "患者", "数值": "2,000"}]},
        }
    )
    assert "synthetic_chroniccare" in data and "2,000" in data


def test_datamate_and_npu_mcp_summaries_cover_all_modes() -> None:
    datamate = server.summarize_datamate(
        {
            "status": "success",
            "steps": [
                {
                    "operator": "clean",
                    "status": "success",
                    "execution_seconds": 1.25,
                },
                {
                    "operator": "extract",
                    "status": "success",
                    "execution_seconds": "bad",
                    "execution_seconds_is_reference": True,
                },
            ],
            "metrics": {"node_count": 10, "edge_count": 20, "question_count": 60},
            "timing": {
                "pure_execution_seconds": 2,
                "pipeline_execution_seconds": 3,
                "outer_flow_seconds": 4,
            },
            "metric_definition": "固定口径",
            "report_path": "report.json",
        }
    )
    assert "固定口径" in datamate and "1.2500" in datamate
    overview = server.summarize_datamate_overview(
        {
            "pipelines": [{"pipeline_name": "etl", "operators": ["clean", "normalize"]}],
            "operator_count": 2,
            "latest_run": {"status": "success"},
            "invocation_mode": "DAG",
        }
    )
    assert "etl" in overview and "2 个主线算子" in overview

    row = {
        "operator": "entity",
        "status": "success",
        "backend": "npu",
        "npu_record_count": 10000,
        "cpu_benchmark_records": 2048,
        "cpu_rule_seconds": 1,
        "cpu_bge_sample_seconds": 2,
        "npu_bge_sample_seconds": 0.5,
        "npu_bge_full_seconds": 3,
        "estimated_cpu_bge_full_seconds": 10,
        "sample_speedup": 4,
        "cpu_resource_utilization_percent": 6400,
    }
    benchmark = server.summarize_npu(
        {
            "status": "success",
            "runtime": {"backend": "npu", "npu_available": True},
            "operator_results": [{"status": "success"}],
            "npu_comparison_rows": [row],
            "fallback_used": False,
            "report_path": "npu.json",
        }
    )
    assert "CPU吞吐量" in benchmark and "2048" in benchmark
    pipeline = server.summarize_npu(
        {
            "status": "success",
            "base_pipeline": {"status": "success"},
            "npu_benchmark": {
                "fallback_used": False,
                "npu_comparison_rows": [row],
            },
        }
    )
    assert "四列表格" in pipeline
    supported = server.summarize_npu({"supported_operators": [{"operator": "entity_npu"}]})
    assert "entity_npu" in supported
    readiness = server.summarize_npu(
        {
            "status": "success",
            "runtime": {
                "backend": "cpu",
                "npu_available": False,
                "fallback_required": True,
                "message": "unavailable",
            },
            "report_path": "ready.json",
        }
    )
    assert "unavailable" in readiness


def test_kg_detail_summary_covers_generated_and_structured_results() -> None:
    generated = server.summarize_kg_detail(
        {
            "status": "success",
            "subgraph_id": "hypertension",
            "query": "高血压子图",
            "html_url": "http://127.0.0.1:28088/graph.html",
            "preview_png_url": "http://127.0.0.1:28088/preview.png",
            "html_route_path": "/artifacts/graph.html",
            "preview_route_path": "/artifacts/preview.png",
            "graph_scope_explanation": "局部图",
            "cohort_patient_count": 433,
            "safety_note": "安全",
        }
    )
    assert "备用入口" in generated
    assert "![子图预览]" in generated
    assert "433" in generated

    structured = server.summarize_kg_detail(
        {
            "status": "success",
            "query": "关系查询",
            "text": "真实结构化结果",
            "answer_guardrail": "只复述表格",
            "table": {
                "rows": [{"指标": "HbA1c", "数量": 10}],
                "allowed_names": ["HbA1c"],
            },
            "node_count": 3,
            "edge_count": 2,
            "seed_labels": ["高血压"],
            "cohort_patient_count": 433,
            "display_patient_node_count": 5,
            "semantic_node_count": 8,
            "top_indicators": [{"indicator": "HbA1c"}],
            "associated_indicators": [{"target_label": "LDL-C"}],
            "top_risk_events": [{"event_type": "血压偏高"}],
            "associated_risk_events": [{"target_label": "肾脏风险"}],
            "top_drugs": [{"drug_name": "缬沙坦"}],
            "associated_drugs": [{"target_label": "氨氯地平"}],
            "graph_url": "http://graph",
            "preview_url": "http://preview.svg",
            "safety_note": "安全",
        }
    )
    assert "HbA1c" in structured
    assert "缬沙坦" in structured
    assert "![图谱子图预览]" in structured

    locked = server.summarize_kg_detail(
        {
            "status": "success",
            "query": "严格表格",
            "table": {
                "rows": [{"疾病": "高血压", "人数": 433}],
                "strict_rows_only": True,
            },
            "final_answer_lock": "不得改写",
            "cohort_patient_count": 433,
        }
    )
    assert "必须原样复述" in locked


def test_analysis_summary_covers_followup_inventory_and_combinations() -> None:
    followup = server.summarize_analysis(
        {
            "question": "未来7天人数",
            "intent": "future_n_days_high_risk_followup",
            "window_days": 7,
            "metric": {"value": 17, "unit": "人"},
            "window": {"start_date": "2026-07-28", "end_date": "2026-08-03"},
            "table": {
                "rows": [{"指标": "人数", "数值": 17}],
                "trend_rows": [
                    {"date": "2026-07-28", "patient_count": 7},
                    {"date": "2026-07-29", "patient_count": 10},
                ],
            },
            "chart_url": "http://127.0.0.1:28088/chart.png",
            "charts": [
                {"name": "趋势", "png_url": "http://host/trend.png"},
            ],
            "safety_note": "安全",
        }
    )
    assert "逐日合计 17" in followup
    assert "![趋势]" in followup

    inventory = server.summarize_analysis(
        {
            "question": "常见病",
            "matched_id": "kg_disease_inventory",
            "intent": "disease_distribution",
            "patient_count": 2000,
            "disease_labels": ["高血压", "糖尿病"],
            "metric": {"name": "matched_disease_patient_count"},
            "table": {"detail_rows": [{"疾病名称": "高血压", "患者人数": 433, "占比": "21.65%"}]},
            "final_answer_lock": "433",
        }
    )
    assert "433" in inventory and "完整疾病类型列表" in inventory

    combos = server.summarize_analysis(
        {
            "question": "共病组合",
            "matched_id": "disease_combination_distribution",
            "intent": "disease_combination_distribution",
            "metric": {"value": 782, "unit": "人"},
            "table": {"detail_rows": [{"疾病组合": "高血压 + 糖尿病", "患者人数": 25}]},
        }
    )
    assert "782" in combos and "精确多病组合" in combos


def test_open_sql_agent_report_and_trace_summaries() -> None:
    schema = server.summarize_open_sql_schema(
        {
            "tables": {"patient_profile": {"columns": [{"name": "patient_id"}, "disease_tags"]}},
            "joins": [{"left": "patient_id", "right": "patient_id"}],
            "safety_note": "安全",
        }
    )
    assert "patient_profile" in schema and "白名单 Join 数量：1" in schema
    evaluation = server.summarize_open_sql_eval(
        {
            "status": "success",
            "total_questions": 240,
            "intent_accuracy": 1.0,
            "result_success_rate": 0.9,
            "report_path": "eval.json",
        }
    )
    assert "240" in evaluation and "eval.json" in evaluation
    examples = server.summarize_open_sql_examples(
        {
            "examples": ["患者人数", "随访趋势"],
            "supported_intents": ["count", "trend"],
            "llm_status": "available",
        }
    )
    assert "2. 随访趋势" in examples
    agent = server.summarize_agent(
        {
            "user_goal": "分析",
            "plan": {"step": 1},
            "tool_results": [{"status": "ok"}],
            "final_answer": "完成",
            "artifacts_used": ["report"],
        }
    )
    assert "最终回答：完成" in agent
    report = server.summarize_report(
        {
            "report_url": "http://report",
            "chart_index_url": "http://charts",
            "graph_url": "http://graph",
            "summary_text": "摘要",
            "latest_graph_driven_analysis": {
                "report_url": "http://latest",
                "graph_url": "http://subgraph",
            },
        },
        {
            "charts": [
                {"title": "图一", "url": "http://host/chart.png"},
            ]
        },
    )
    assert "![图一]" in report and "http://latest" in report

    assert server.jsonish(None) == "null"
    assert server.jsonish("text") == "text"
    assert "{" in server.jsonish({"a": 1})
    assert server.truncate_text("short", 10) == "short"
    assert server.truncate_text("abcdefghijk", 8).endswith("...")
    assert server.build_trace_id().startswith("trace_")
    assert server.summarize_trace_input({"long": "x" * 300})["long"].endswith("...")


@pytest.mark.parametrize(
    ("tool_name", "payload", "key"),
    [
        ("chroniccare_health_check", {"status": "ok", "project": "p"}, "project"),
        ("chroniccare_kg_summary", {"node_count": 1, "edge_count": 2}, "edge_count"),
        (
            "chroniccare_datamate_pipeline_run",
            {"status": "success", "steps": [{}], "metrics": {"node_count": 1}},
            "step_count",
        ),
        (
            "chroniccare_npu_readiness",
            {"runtime": {"backend": "npu", "npu_available": True}},
            "backend",
        ),
        (
            "chroniccare_datamate_pipelines",
            {"pipelines": [{}, {}], "operator_count": 11},
            "pipeline_count",
        ),
        (
            "chroniccare_datamate_pipeline_latest",
            {"run_id": "r1", "pipeline_name": "etl"},
            "run_id",
        ),
        (
            "chroniccare_datamate_pipeline_report",
            {"report_path": "r.json"},
            "report_path",
        ),
        (
            "chroniccare_kg_subgraph_render",
            {"node_count": 3, "subgraph_id": "s"},
            "subgraph_id",
        ),
        (
            "chroniccare_data_summary",
            {"patient_count": 2000},
            "patient_count",
        ),
        (
            "chroniccare_analysis_query",
            {
                "question": "平均值",
                "metric": {"name": "avg"},
                "table": {"rows": [{"avg_value": 7.1}]},
            },
            "value",
        ),
        (
            "chroniccare_disease_distribution",
            {"question": "疾病", "summary_text": "结果"},
            "summary_text",
        ),
        (
            "chroniccare_agent_run",
            {"run_id": "a", "artifacts_used": ["x"]},
            "run_id",
        ),
        (
            "chroniccare_report_summary",
            {"analysis_report_html": "r"},
            "analysis_report_html",
        ),
        (
            "chroniccare_open_sql_query",
            {"status": "success", "trace_id": "t"},
            "trace_id",
        ),
        (
            "chroniccare_trace_summary",
            {"total_calls": 2, "success_rate": 1.0},
            "total_calls",
        ),
        ("unknown", {"status": "ok"}, "status"),
    ],
)
def test_trace_output_summary_branches(tool_name: str, payload: dict, key: str) -> None:
    assert key in server.summarize_trace_output(tool_name, payload)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:28088/a", "127.0.0.1:18089"),
        ("http://localhost:28088/a", "127.0.0.1:18089"),
        ("http://127.0.0.1:18089/a", "127.0.0.1:28088"),
        ("http://localhost:18089/a", "127.0.0.1:28088"),
        ("http://example/a", None),
    ],
)
def test_alternate_local_artifact_urls(url: str, expected: str | None) -> None:
    result = server._alternate_local_artifact_url(url)
    if expected is None:
        assert result is None
    else:
        assert expected in result
