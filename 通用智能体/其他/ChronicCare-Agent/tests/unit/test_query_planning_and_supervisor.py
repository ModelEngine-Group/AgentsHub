from __future__ import annotations

import pytest

from analysis import query_planner
from orchestration import planner, query_executor, supervisor


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("7", 7),
        ("十", 10),
        ("十二", 12),
        ("二十", 20),
        ("二十三", 23),
        ("百", None),
        ("", None),
    ],
)
def test_chinese_number_parser(token: str, expected: int | None) -> None:
    assert query_planner._parse_chinese_number(token) == expected


@pytest.mark.parametrize(
    ("question", "value", "direction"),
    [
        ("未来十二天随访", 12, "future"),
        ("接下来 8 日随访", 8, "future"),
        ("近30天记录", 30, "past"),
        ("本月随访", 30, "current"),
    ],
)
def test_detect_time_window(question: str, value: int, direction: str) -> None:
    result = query_planner.detect_time_window(question)
    assert result is not None
    assert result.value == value
    assert result.direction == direction


def test_entity_and_output_preference_detection() -> None:
    assert query_planner.detect_diseases("高血压、糖尿病与慢阻肺") == [
        "hypertension",
        "diabetes",
        "copd",
    ]
    assert query_planner.detect_risks("高风险和低风险") == ["high", "low"]
    assert query_planner.detect_chart_types("请给趋势图和表格") == ["line", "table"]
    assert query_planner.detect_time_window("没有时间") is None


@pytest.mark.parametrize(
    ("question", "intent", "route"),
    [
        ("为什么图谱中的高血压患者变多？", "graph_sql_joint_analysis", "graph_driven"),
        ("当前有哪些疾病？", "nl2sql", "analysis"),
        ("给出分析报告入口", "report_summary", "report"),
        ("图谱中高血压患者有多少？", "graph_sql_joint_analysis", "graph_driven"),
        ("当前图谱节点质量如何？", "kg_summary", "kg"),
        ("生成一个知识图谱关系图", "kg_subgraph", "kg"),
        ("未来7天随访人数趋势图", "future_followup_chart", "graph_driven"),
        ("不同风险等级占比", "risk_distribution", "analysis"),
        ("高血压患者有多少？", "cohort_stats", "analysis"),
        ("糖尿病患者情况", "nl2sql", "analysis"),
        ("执行一条SQL查询", "nl2sql", "analysis"),
        ("天气怎么样", "unknown", "open"),
    ],
)
def test_query_planner_routes_all_major_intents(question: str, intent: str, route: str) -> None:
    result = query_planner.plan_query(
        question,
        context={"canonical_question": "标准问题"},
    )
    assert result.intent == intent
    assert result.route == route
    assert result.tool_plan
    assert result.canonical_question == "标准问题"


@pytest.fixture
def isolated_planner(monkeypatch):
    monkeypatch.setattr(planner, "load_server_config", lambda: {})
    monkeypatch.setattr(planner, "safety_note", lambda _: "安全说明")


@pytest.mark.parametrize(
    ("goal", "tools", "primary_group"),
    [
        ("当前常见病有哪些？", ["analysis.open_query"], "data_analysis_tools"),
        ("生成糖尿病患者知识图谱子图", ["kg.subgraph_render"], None),
        (
            "请重新运行 DataMate 全流程",
            ["datamate.pipeline_run", "datamate.pipeline_status", "datamate.pipeline_report"],
            "data_processing_tools",
        ),
        (
            "查看最新 DataMate pipeline 状态",
            ["datamate.pipelines", "datamate.pipeline_status", "datamate.pipeline_report"],
            "data_processing_tools",
        ),
        ("未来7天随访人数趋势图", ["analysis.open_query"], "data_analysis_tools"),
        ("高血压相关知识图谱", ["kg.subgraph_render"], None),
        ("生成统计报告", ["analysis.query", "reports.summary"], "data_analysis_tools"),
        ("展示系统入口", ["artifacts.status", "reports.summary"], "shared_tools"),
        ("普通未知目标", ["artifacts.status", "reports.summary"], "shared_tools"),
    ],
)
def test_agent_planner_builds_expected_steps(
    isolated_planner,
    goal: str,
    tools: list[str],
    primary_group: str | None,
) -> None:
    result = planner.build_plan(goal)
    assert result["status"] == "success"
    assert [item["tool"] for item in result["plan"]] == tools
    if primary_group is not None:
        assert result["primary_tool_group"] == primary_group


@pytest.mark.parametrize(
    ("goal", "expected"),
    [
        ("糖尿病分析", "高血压合并糖尿病患者的平均 HbA1c 是多少？"),
        ("LDL血脂分析", "不同疾病组合的 LDL-C 异常比例是多少？"),
        ("空腹血糖趋势", "不同月份的空腹血糖异常人数趋势如何？"),
        ("BMI情况", "BMI 偏高患者中血压异常比例是多少？"),
        ("其他分析", "高血压合并糖尿病患者的平均 HbA1c 是多少？"),
    ],
)
def test_infer_analysis_question(goal: str, expected: str) -> None:
    assert planner._infer_analysis_question(goal) == expected


@pytest.mark.parametrize(
    ("goal", "entity_id"),
    [
        ("糖尿病", "Disease::diabetes"),
        ("高血压", "Disease::hypertension"),
        ("空腹血糖", "Indicator::fasting_glucose"),
        ("药物", "Drug::metformin"),
        ("未知", "Disease::hypertension"),
    ],
)
def test_infer_kg_entity(goal: str, entity_id: str) -> None:
    assert planner._infer_kg_entity(goal)["entity_id"] == entity_id


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("datamate.pipeline_run", {"task_id": "supervisor_datamate_run_001", "force": True, "safe_run": True}),
        ("datamate.pipeline_status", {"run_id": "latest"}),
        ("analysis.open_query", {"question": "目标"}),
        ("analysis.graph_driven", {"question": "目标"}),
        ("kg.subgraph_render", {"query": "目标", "max_nodes": 96}),
        ("kg.query", {"query_type": "disease_profile", "entity_id": "Disease::hypertension"}),
        ("unknown", {}),
    ],
)
def test_supervisor_infers_tool_inputs(tool: str, expected: dict) -> None:
    assert supervisor._infer_tool_input(tool, "目标", {}) == expected
    assert supervisor._infer_tool_input(tool, "目标", {"input_hint": {"fixed": True}}) == {"fixed": True}


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"status": "failed", "errors": ["错误"]}, "错误"),
        ({"status": "success", "summary_text": "摘要"}, "摘要"),
        ({"status": "success", "answer": "答案"}, "答案"),
        ({"status": "success", "insight": "洞察"}, "洞察"),
        (
            {"status": "success", "artifacts": {"db": {"exists": True}, "graph": {"exists": False}}},
            "已确认可用产物：db",
        ),
        (
            {"status": "success", "pipeline_name": "chroniccare_datamate_full_pipeline", "summary": "完成"},
            "完成",
        ),
        ({"status": "success", "analysis_report_html": "/report"}, "已整理可访问的分析报告入口"),
        ({"status": "success", "html_url": "/graph"}, "已生成可访问的问题驱动子图入口"),
        ({"status": "success", "graph_url": "/graph"}, "已生成可访问的图谱子图入口"),
        ({"status": "success", "questions": [1, 2]}, "已读取分析问题集，共 2 个问题。"),
        ({"status": "success"}, "工具执行成功"),
    ],
)
def test_supervisor_summarizes_tool_outputs(output: dict, expected: str) -> None:
    assert supervisor._summarize_tool_output(output) == expected


def test_supervisor_collects_links_charts_and_pipeline_details() -> None:
    results = [
        {
            "raw_output": {
                "html_url": "/artifacts/subgraphs/one.html",
                "report_url": "http://server/report",
                "ignored": "/tmp/private",
                "charts": [
                    {"name": "趋势", "url": "http://server/chart.png"},
                    {"name": "重复", "url": "http://server/chart.png"},
                ],
                "artifacts": {
                    "graph": {"path": "/artifacts/graph.html"},
                    "private": {"path": "/tmp/private"},
                },
            }
        }
    ]
    paths = supervisor._collect_evidence_paths(results)
    assert paths == [
        "/artifacts/subgraphs/one.html",
        "http://server/report",
        "/artifacts/graph.html",
    ]
    assert supervisor._collect_chart_markdown(results) == ["![趋势](http://server/chart.png)"]
    assert supervisor._find_first_url(results, ["report_url"]) == "http://server/report"
    assert supervisor._find_first_url(results, ["missing"]) is None

    expanded = supervisor._expand_pipeline_run_steps(
        {"step": 1},
        {
            "steps": [
                {
                    "operator": "chronic_nl2sql_analyze",
                    "status": "success",
                    "summary": {"question_count": 60, "success_count": 60},
                },
                {
                    "operator": "chronic_triple_validate",
                    "status": "success",
                    "summary": {"triples_clean": 10, "triples_rejected": 2},
                },
                {"operator": "other", "status": "success", "summary": {}},
            ]
        },
    )
    assert [item["step"] for item in expanded] == ["1.1", "1.2", "1.3"]
    assert "question_count=60" in expanded[0]["output_summary"]
    assert "剔除异常三元组=2" in expanded[1]["output_summary"]


def test_supervisor_composes_subgraph_and_general_answers() -> None:
    subgraph = [
        {
            "agent": "KG",
            "tool": "kg.subgraph_render",
            "output_summary": "已生成",
            "status": "success",
            "raw_output": {"html_url": "http://server/subgraph"},
        }
    ]
    answer = supervisor._compose_final_answer("生成糖尿病患者知识图谱子图", subgraph, "安全")
    assert "http://server/subgraph" in answer
    assert "医疗安全说明：安全" in answer

    general = [
        {
            "agent": "Analysis",
            "tool": "analysis.query",
            "output_summary": "统计完成",
            "status": "success",
            "time_cost_sec": 0.1,
            "raw_output": {
                "chart_url": "http://server/charts",
                "report_url": "http://server/report",
                "graph_url": "http://server/graph",
            },
        }
    ]
    answer = supervisor._compose_final_answer("生成统计报告", general, "安全")
    assert "统计完成" in answer
    assert "查看图表总览" in answer
    assert "查看分析报告页面" in answer


def test_run_supervisor_executes_plan_and_writes_public_result(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(supervisor, "load_server_config", lambda: {"paths": {"agent_runs_dir": "runs"}})
    monkeypatch.setattr(supervisor, "safety_note", lambda _: "安全")
    monkeypatch.setattr(
        supervisor,
        "build_plan",
        lambda _: {
            "plan": [
                {
                    "step": 1,
                    "agent": "Analysis",
                    "tool": "analysis.query",
                    "description": "分析",
                    "input_hint": {"question": "测试"},
                    "expected_output": "摘要",
                }
            ]
        },
    )
    monkeypatch.setattr(
        supervisor,
        "route_tool",
        lambda *_args, **_kwargs: {"status": "success", "summary_text": "完成"},
    )
    monkeypatch.setattr(supervisor, "resolve_path", lambda _: tmp_path / "runs")
    monkeypatch.setattr(supervisor, "relative_to_project", lambda path: str(path))
    captured = {}
    monkeypatch.setattr(
        supervisor,
        "write_trace",
        lambda path, *args: captured.update({"path": path, "args": args}),
    )
    result = supervisor.run_supervisor("执行测试")
    assert result["status"] == "success"
    assert result["tool_call_count"] == 1
    assert result["agents_used"] == ["Analysis"]
    assert result["tool_results"][0]["output_summary"] == "完成"
    assert "raw_output" not in result["tool_results"][0]
    assert captured["path"].parent == tmp_path / "runs"


def test_query_executor_handles_local_safety_routes(monkeypatch) -> None:
    assert query_executor.execute_query_plan({"executor": "legacy_open_analysis"}) is None
    monkeypatch.setattr(
        query_executor, "load_server_config", lambda: {"server": {"browser_base_url": "http://server/"}}
    )
    monkeypatch.setattr(query_executor, "safety_note", lambda _: "安全")
    unsupported = query_executor.execute_query_plan(
        {
            "executor": "direct_tool",
            "intent": "unsupported_negation_query",
            "query": "排除糖尿病",
        }
    )
    assert unsupported["status"] == "success"
    assert unsupported["supported_followups"]
    status = query_executor.execute_query_plan({"executor": "direct_tool", "intent": "system_status", "query": "状态"})
    assert status["base_url"] == "http://server"
    missing_patient = query_executor.execute_query_plan(
        {
            "executor": "direct_tool",
            "intent": "kg_patient_path_query",
            "query": "查询患者路径",
        }
    )
    assert missing_patient["status"] == "failed"
    assert query_executor.execute_query_plan({"executor": "direct_tool", "intent": "unknown", "query": "未知"}) is None
