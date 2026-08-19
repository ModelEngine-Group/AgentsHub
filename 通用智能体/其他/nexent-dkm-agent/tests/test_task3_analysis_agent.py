import json
import re
import sqlite3
from pathlib import Path

import pytest

from src.agents.analysis_agent import GraphAnalysisAgent
from src.agents.analysis_agent.nexent_adapter import (
    GraphAnalysisAgentTool,
    build_nexent_agent_spec,
    build_nexent_tool_spec,
)
from src.operators.analysis_ops import (
    INTENT_SQL,
    build_analysis_report,
    build_analysis_visualizations,
    classify_question_intent,
    evaluate_nl2sql_accuracy,
    export_analysis_dashboard,
    export_echarts_dashboard,
    export_insight_report,
    build_graph_sqlite,
    execute_sql,
    generate_association_analysis,
    generate_statistical_summary,
    generate_trend_analysis,
    load_graph,
    plan_analysis_task,
    translate_question_to_sql,
    validate_read_only_sql,
)
from src.operators.analysis_ops.echarts_dashboard import _svg_pie
from src.pipelines.task2_kg_pipeline import run_task2_pipeline
from src.pipelines.task3_evaluation import run_task3_evaluation
from src.pipelines.task3_insight_pipeline import run_task3_pipeline
from src.pipelines.task3_smoke import run_task3_smoke


ROOT = Path(__file__).resolve().parents[1]


def _graph_path(tmp_path: Path) -> Path:
    result = run_task2_pipeline(output_dir=tmp_path / "task2")
    assert result.status == "completed"
    return Path(result.artifacts["graph"]["output_path"])


def test_load_graph_validates_required_fields(tmp_path):
    path = _graph_path(tmp_path)
    graph = load_graph(path)

    assert graph["statistics"]["node_count"] > 0
    assert graph["nodes"]
    assert graph["edges"]


def test_statistical_summary_counts_entities_and_relations(tmp_path):
    graph = load_graph(_graph_path(tmp_path))
    summary = generate_statistical_summary(graph)

    assert summary["status"] == "completed"
    assert summary["entity_type_counts"]["Disease"] >= 3
    assert summary["relation_type_counts"]["has_symptom"] > 0
    assert summary["top_degree_nodes"]
    assert summary["confidence"]["average"] > 0


def test_association_analysis_builds_disease_profiles(tmp_path):
    graph = load_graph(_graph_path(tmp_path))
    associations = generate_association_analysis(graph)

    assert associations["status"] == "completed"
    assert associations["disease_profiles"]
    assert any(
        profile["symptoms"] or profile["drugs"] or profile["examinations"]
        for profile in associations["disease_profiles"]
    )


def test_trend_analysis_uses_record_sequence(tmp_path):
    graph = load_graph(_graph_path(tmp_path))
    trends = generate_trend_analysis(graph)

    assert trends["status"] == "completed"
    assert trends["record_trends"]
    assert trends["record_trends"][0]["record_id"].startswith("record_")
    assert "edge_count" in trends["record_trends"][0]


def test_nl2sql_translates_and_executes_question(tmp_path):
    graph = load_graph(_graph_path(tmp_path))
    conn = build_graph_sqlite(graph)

    sql_plan = translate_question_to_sql("哪些疾病关联最多症状？")
    rows = execute_sql(conn, sql_plan["sql"])

    assert sql_plan["status"] == "completed"
    assert "has_symptom" in sql_plan["sql"]
    assert rows
    assert rows[0]["symptom_count"] >= 1


def _entity_aware_graph() -> dict:
    """Small deterministic graph: 高血压 and 糖尿病 with distinct drugs/symptoms."""
    return {
        "nodes": [
            {"id": "高血压", "name": "高血压", "type": "Disease", "mention_count": 5},
            {"id": "糖尿病", "name": "糖尿病", "type": "Disease", "mention_count": 4},
            {"id": "氨氯地平", "name": "氨氯地平", "type": "Drug", "mention_count": 2},
            {"id": "阿司匹林", "name": "阿司匹林", "type": "Drug", "mention_count": 2},
            {"id": "二甲双胍", "name": "二甲双胍", "type": "Drug", "mention_count": 1},
            {"id": "头晕", "name": "头晕", "type": "Symptom", "mention_count": 1},
            {"id": "多尿", "name": "多尿", "type": "Symptom", "mention_count": 1},
        ],
        "edges": [
            {"source": "高血压", "target": "氨氯地平", "predicate": "treated_by", "confidence": 0.9},
            {"source": "高血压", "target": "阿司匹林", "predicate": "treated_by", "confidence": 0.8},
            {"source": "糖尿病", "target": "二甲双胍", "predicate": "treated_by", "confidence": 0.9},
            {"source": "高血压", "target": "头晕", "predicate": "has_symptom", "confidence": 0.9},
            {"source": "糖尿病", "target": "多尿", "predicate": "has_symptom", "confidence": 0.9},
        ],
        "statistics": {"node_count": 7, "edge_count": 5},
    }


def test_entity_aware_nl2sql_filters_to_specific_disease():
    from src.operators.analysis_ops.nl2sql import (
        build_graph_sqlite,
        disease_names_from_graph,
        execute_sql,
        translate_question_to_sql,
    )

    graph = _entity_aware_graph()
    conn = build_graph_sqlite(graph)
    names = disease_names_from_graph(graph)

    result = translate_question_to_sql("高血压用什么药", disease_names=names)
    assert result["intent"] == "disease_specific_drugs"
    assert result["entity"] == "高血压"

    rows = execute_sql(conn, result["sql"])
    drugs = {row["drug"] for row in rows}
    # Must return 高血压's drugs only -- NOT 糖尿病's 二甲双胍.
    assert drugs == {"氨氯地平", "阿司匹林"}


def test_nl2sql_without_entity_context_uses_aggregate_template():
    """Backward compat: no disease_names -> existing aggregate intent behavior."""
    result = translate_question_to_sql("高血压用什么药")
    assert result["intent"] == "top_disease_drugs"


def test_evaluate_nl2sql_execution_accuracy_matches_gold_rows():
    from src.operators.analysis_ops.nl2sql import evaluate_nl2sql_execution_accuracy

    graph = _entity_aware_graph()
    cases = [
        {
            "question": "高血压用什么药",
            "gold_sql": (
                "SELECT t.name AS drug FROM edges e "
                "JOIN nodes s ON e.source = s.id JOIN nodes t ON e.target = t.id "
                "WHERE e.predicate = 'treated_by' AND s.name = '高血压' "
                "ORDER BY drug ASC LIMIT 20"
            ),
        },
        {
            "question": "糖尿病有哪些症状",
            "gold_sql": (
                "SELECT t.name AS symptom FROM edges e "
                "JOIN nodes s ON e.source = s.id JOIN nodes t ON e.target = t.id "
                "WHERE e.predicate = 'has_symptom' AND s.name = '糖尿病' "
                "ORDER BY symptom ASC LIMIT 20"
            ),
        },
        {
            "question": "图谱中有哪些关系类型",
            "gold_sql": (
                "SELECT predicate, COUNT(*) AS edge_count FROM edges "
                "GROUP BY predicate ORDER BY edge_count DESC, predicate ASC"
            ),
        },
    ]

    report = evaluate_nl2sql_execution_accuracy(cases, graph)
    assert report["total"] == 3
    assert report["accuracy"] == 1.0


_NL2SQL_BENCHMARK = ROOT / "benchmarks" / "data" / "nl2sql_benchmark.json"


def _load_nl2sql_benchmark():
    data = json.loads(_NL2SQL_BENCHMARK.read_text(encoding="utf-8"))
    return data["cases"]


def test_classify_relation_distribution_over_generic_entity_keywords():
    """Relation-specific phrases must beat generic distribution/statistics keywords."""
    assert classify_question_intent("给我统计一下关系类型分布") == "relation_distribution"
    assert classify_question_intent("统计关系类型分布") == "relation_distribution"
    assert classify_question_intent("症状最多的疾病有哪些") == "top_disease_symptoms"


def test_nl2sql_intents_all_have_sql():
    """Every classifiable intent must map to an executable SQL template."""
    for case in _load_nl2sql_benchmark():
        intent = classify_question_intent(case["question"])
        assert intent in INTENT_SQL


def test_nl2sql_each_intent_sql_executes(tmp_path):
    """All canonical intent SQL templates run safely against the graph schema."""
    graph = load_graph(_graph_path(tmp_path))
    conn = build_graph_sqlite(graph)
    for intent, sql in INTENT_SQL.items():
        rows = execute_sql(conn, sql)
        assert isinstance(rows, list), intent


def test_nl2sql_benchmark_accuracy_above_threshold():
    """Template NL2SQL intent accuracy must meet the task-3 >=85% requirement."""
    report = evaluate_nl2sql_accuracy(_load_nl2sql_benchmark())
    assert report["total"] >= 30
    assert report["accuracy"] >= 0.85, report["mistakes"]


_NL2SQL_EXEC_BENCHMARK = ROOT / "benchmarks" / "data" / "nl2sql_execution_benchmark.json"
_NL2SQL_HOLDOUT_BENCHMARK = ROOT / "benchmarks" / "data" / "nl2sql_holdout_benchmark.json"


def test_nl2sql_execution_accuracy_above_threshold():
    """Execution-level NL2SQL accuracy (rows vs gold SQL) must meet >=85%."""
    from src.operators.analysis_ops import evaluate_nl2sql_execution_accuracy

    data = json.loads(_NL2SQL_EXEC_BENCHMARK.read_text(encoding="utf-8"))
    report = evaluate_nl2sql_execution_accuracy(data["cases"], data["graph"])
    assert report["total"] >= 10
    assert report["accuracy"] >= 0.85, report["mistakes"]


def test_nl2sql_resolves_medical_alias_to_known_disease():
    data = json.loads(_NL2SQL_HOLDOUT_BENCHMARK.read_text(encoding="utf-8"))
    disease_names = [
        node["name"]
        for node in data["graph"]["nodes"]
        if node["type"] == "Disease"
    ]

    result = translate_question_to_sql(
        "血糖偏高的患者有什么饮食上的建议",
        disease_names=disease_names,
    )

    assert result["intent"] == "disease_specific_treatments"
    assert result["entity"] == "糖尿病"
    assert "s.name = '糖尿病'" in result["sql"]


def test_nl2sql_limits_mention_comparison_to_named_diseases():
    data = json.loads(_NL2SQL_HOLDOUT_BENCHMARK.read_text(encoding="utf-8"))
    disease_names = [
        node["name"]
        for node in data["graph"]["nodes"]
        if node["type"] == "Disease"
    ]

    result = translate_question_to_sql(
        "高血压和糖尿病哪个被提及更多次",
        disease_names=disease_names,
    )

    assert result["intent"] == "disease_mention_comparison"
    assert result["entities"] == ["高血压", "糖尿病"]
    assert "name IN ('高血压', '糖尿病')" in result["sql"]


def test_nl2sql_holdout_generalization_above_threshold():
    """The expanded paraphrase regression set must stay above the rubric threshold."""
    from src.operators.analysis_ops import evaluate_nl2sql_execution_accuracy

    data = json.loads(_NL2SQL_HOLDOUT_BENCHMARK.read_text(encoding="utf-8"))
    report = evaluate_nl2sql_execution_accuracy(data["cases"], data["graph"])
    assert report["total"] >= 20
    assert report["accuracy"] >= 0.95, report["mistakes"]


def test_nl2sql_rejects_unsafe_sql():
    with pytest.raises(ValueError):
        validate_read_only_sql("SELECT * FROM nodes; DROP TABLE nodes")
    with pytest.raises(ValueError):
        validate_read_only_sql("DELETE FROM nodes")
    with pytest.raises(ValueError):
        validate_read_only_sql("SELECT * FROM sqlite_master")
    with pytest.raises(ValueError):
        validate_read_only_sql(
            'SELECT id, name, type, mention_count FROM nodes '
            'UNION ALL SELECT name, sql, type, rootpage FROM "sqlite_master"'
        )


def test_nl2sql_execution_blocks_system_table_bypass():
    from src.operators.analysis_ops import execute_read_only_sql

    conn = build_graph_sqlite(
        {
            "nodes": [
                {
                    "id": "Disease:test",
                    "name": "test",
                    "type": "Disease",
                    "mention_count": 1,
                }
            ],
            "edges": [],
        }
    )

    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        execute_read_only_sql(
            conn,
            "SELECT id, name, type, mention_count FROM nodes "
            "UNION ALL SELECT name, sql, type, rootpage FROM sqlite_master",
        )


def test_nl2sql_rejects_comma_limit_row_bound_bypass():
    with pytest.raises(ValueError):
        validate_read_only_sql("SELECT * FROM nodes LIMIT 1, 100000")


def test_nl2sql_bounds_safe_sql_results():
    assert validate_read_only_sql("SELECT * FROM nodes").endswith("LIMIT 20")
    assert validate_read_only_sql("SELECT * FROM nodes LIMIT 100").endswith("LIMIT 20")
    assert validate_read_only_sql("SELECT * FROM edges LIMIT 5").endswith("LIMIT 5")


def test_nl2sql_execution_bounds_outer_rows_when_subquery_has_limit():
    from src.operators.analysis_ops import execute_read_only_sql

    conn = build_graph_sqlite(
        {
            "nodes": [
                {
                    "id": f"Disease:{index}",
                    "name": f"disease-{index}",
                    "type": "Disease",
                    "mention_count": index,
                }
                for index in range(30)
            ],
            "edges": [],
        }
    )

    result = execute_read_only_sql(
        conn,
        "SELECT n.* FROM nodes n "
        "JOIN (SELECT id FROM nodes LIMIT 1) sample ON 1 = 1",
        max_limit=20,
    )

    assert len(result["rows"]) == 20


def test_visualization_specs_are_serializable(tmp_path):
    graph = load_graph(_graph_path(tmp_path))
    stats = generate_statistical_summary(graph)
    associations = generate_association_analysis(graph)
    trends = generate_trend_analysis(graph)
    visuals = build_analysis_visualizations(stats, associations, trends)

    assert visuals["status"] == "completed"
    assert {"entity_distribution", "relation_distribution", "record_trend", "disease_network"}.issubset(
        visuals["charts"]
    )
    json.dumps(visuals, ensure_ascii=False)


def test_export_insight_report_writes_markdown_and_html(tmp_path):
    graph = load_graph(_graph_path(tmp_path))
    stats = generate_statistical_summary(graph)
    associations = generate_association_analysis(graph)
    trends = generate_trend_analysis(graph)
    visuals = build_analysis_visualizations(stats, associations, trends)
    nl2sql = {
        **translate_question_to_sql("哪些疾病关联最多症状？"),
        "rows": [{"disease": "高血压", "symptom_count": 4}],
    }

    report = export_insight_report(
        target_dir=tmp_path / "reports",
        graph=graph,
        statistics=stats,
        associations=associations,
        trends=trends,
        nl2sql=nl2sql,
        visualizations=visuals,
    )

    markdown_path = Path(report["markdown_path"])
    html_path = Path(report["html_path"])
    assert report["status"] == "completed"
    assert markdown_path.exists()
    assert html_path.exists()
    assert "Task 3 Graph Analysis Insight Report" in markdown_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "insight-chart-grid" in html
    assert "Entity type distribution" in html
    assert "<svg" in html


def test_insight_report_includes_graph_analytics_sections(tmp_path):
    graph = load_graph(_graph_path(tmp_path))
    stats = generate_statistical_summary(graph)
    associations = generate_association_analysis(graph)
    trends = generate_trend_analysis(graph)
    visuals = build_analysis_visualizations(stats, associations, trends)
    nl2sql = {**translate_question_to_sql("哪些疾病关联最多症状？"), "rows": []}
    centrality = {
        "status": "completed",
        "top_hubs_backend": "python",
        "top_hubs": [
            {"id": "Disease:高血压", "name": "高血压", "type": "Disease",
             "degree": 4, "degree_centrality": 0.4},
        ],
        "type_centrality": {
            "Disease": {"count": 2, "avg_degree": 3.0, "max_degree": 4, "top_node": "高血压"},
        },
    }
    graph_analytics = {
        "status": "completed",
        "start_hub": "Disease:高血压",
        "communities": {
            "community_count": 1,
            "communities": [
                {"community_id": "c1", "size": 5,
                 "type_distribution": {"Disease": 2, "Symptom": 3}, "members": []},
            ],
        },
        "shortest_paths": {"status": "reachable", "reachable_count": 6, "max_depth": 4, "nodes": []},
    }

    report = export_insight_report(
        target_dir=tmp_path / "reports",
        graph=graph,
        statistics=stats,
        associations=associations,
        trends=trends,
        nl2sql=nl2sql,
        visualizations=visuals,
        centrality=centrality,
        graph_analytics=graph_analytics,
    )

    markdown = Path(report["markdown_path"]).read_text(encoding="utf-8")
    assert "## Graph Centrality" in markdown
    assert "高血压" in markdown
    assert "## Community Structure" in markdown
    assert "## Reachability From Top Hub" in markdown


def test_export_analysis_dashboard_renders_static_charts(tmp_path):
    graph = load_graph(_graph_path(tmp_path))
    stats = generate_statistical_summary(graph)
    associations = generate_association_analysis(graph)
    trends = generate_trend_analysis(graph)
    visuals = build_analysis_visualizations(stats, associations, trends)
    nl2sql = {
        **translate_question_to_sql("disease symptom count"),
        "rows": [{"disease": "hypertension", "symptom_count": 4}],
    }

    dashboard = export_analysis_dashboard(
        target_dir=tmp_path / "dashboard",
        statistics=stats,
        associations=associations,
        trends=trends,
        nl2sql=nl2sql,
        visualizations=visuals,
    )

    dashboard_path = Path(dashboard["dashboard_path"])
    html = dashboard_path.read_text(encoding="utf-8")
    assert dashboard["status"] == "completed"
    assert dashboard_path.exists()
    assert "analysis-dashboard" in html
    assert "Entity type distribution" in html
    assert "hypertension" in html


def test_export_analysis_dashboard_includes_entity_trend_and_top_hubs(tmp_path):
    graph = load_graph(_graph_path(tmp_path))
    stats = generate_statistical_summary(graph)
    associations = generate_association_analysis(graph)
    trends = generate_trend_analysis(graph)
    visuals = build_analysis_visualizations(stats, associations, trends)
    nl2sql = {
        **translate_question_to_sql("disease symptom count"),
        "rows": [{"disease": "hypertension", "symptom_count": 4}],
    }

    dashboard = export_analysis_dashboard(
        target_dir=tmp_path / "dashboard",
        statistics=stats,
        associations=associations,
        trends=trends,
        nl2sql=nl2sql,
        visualizations=visuals,
    )
    content = Path(dashboard["dashboard_path"]).read_text(encoding="utf-8")

    expected_entity_counts = [row["entity_count"] for row in trends["record_trends"]]
    expected_edge_counts = [row["edge_count"] for row in trends["record_trends"]]

    edge_polyline = re.search(
        r'<polyline[^>]*points="([^"]+)"',
        content,
    )
    assert edge_polyline, "trend edge polyline should be rendered"
    assert "实体数" in content, "static dashboard should label the entity trend series"
    entity_polyline = re.search(
        r'aria-label="[^"]*entity[^"]*".*?<polyline[^>]*points="([^"]+)"',
        content,
        re.S,
    )
    assert entity_polyline, "entity trend polyline should be rendered"
    assert "Top Hub" in content, "static dashboard should include a Top Hub section"
    assert "暂无数据" not in content
    assert all(count > 0 for count in expected_entity_counts)
    assert all(count > 0 for count in expected_edge_counts)



def test_task3_pipeline_runs_full_analysis(tmp_path):
    graph_file = _graph_path(tmp_path)
    result = run_task3_pipeline(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
        question="哪些疾病关联最多症状？",
        task_request="分析图谱并生成可视化",
    )

    assert result.task == "task3_analysis_agent"
    assert result.status == "completed"
    assert result.artifacts["plan"]["operators"][0] == "load_graph"
    assert result.artifacts["nl2sql"]["rows"]
    assert result.artifacts["visualizations"]["charts"]
    assert Path(result.artifacts["export"]["output_path"]).exists()
    assert Path(result.artifacts["insight_report"]["html_path"]).exists()
    assert Path(result.artifacts["insight_report"]["markdown_path"]).exists()
    assert Path(result.artifacts["insight_report"]["dashboard_path"]).exists()
    assert result.artifacts["quality_report"]["status"] == "passed"
    assert "prepare_degree_tensor_cache" in {
        step["name"] for step in result.artifacts["run_state"]["steps"]
    }


def test_task3_pipeline_bootstraps_task2_graph_when_missing(tmp_path):
    result = run_task3_pipeline(
        graph_file=tmp_path / "missing.json",
        output_dir=tmp_path / "task3",
    )

    assert result.status == "completed"
    assert result.artifacts["input"]["source"] == "bootstrapped_task2"
    assert result.artifacts["graph"]["node_count"] > 0


def test_analysis_agent_tracks_failed_input(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")

    result = GraphAnalysisAgent().run(graph_file=bad, output_dir=tmp_path / "out")

    assert result.status == "failed"
    assert result.artifacts["error"]["type"] in {"ValueError", "KeyError"}
    assert result.artifacts["run_state"]["status"] == "failed"


def test_nexent_adapter_wraps_task3_pipeline(tmp_path):
    graph_file = _graph_path(tmp_path)
    tool = GraphAnalysisAgentTool(output_dir=str(tmp_path / "task3"))
    payload = json.loads(
        tool.forward(
            graph_file=str(graph_file),
            question="哪些疾病关联最多症状？",
        )
    )

    assert payload["status"] == "completed"
    assert payload["artifacts"]["visualizations"]["charts"]

    tool_spec = build_nexent_tool_spec(output_dir=str(tmp_path / "task3"))
    agent_spec = build_nexent_agent_spec(model_name="main_model")
    assert tool_spec["name"] == "task3_graph_analysis"
    assert agent_spec["name"] == "task3_graph_analysis_agent"
    assert agent_spec["tools"][0]["name"] == "task3_graph_analysis"


def test_analysis_report_scores_readiness(tmp_path):
    graph = load_graph(_graph_path(tmp_path))
    stats = generate_statistical_summary(graph)
    associations = generate_association_analysis(graph)
    trends = generate_trend_analysis(graph)
    visuals = build_analysis_visualizations(stats, associations, trends)
    report = build_analysis_report(
        graph=graph,
        statistics=stats,
        associations=associations,
        trends=trends,
        nl2sql={"status": "completed", "rows": [{"x": 1}]},
        visualizations=visuals,
        insight_report={
            "status": "completed",
            "html_path": "x.html",
            "markdown_path": "x.md",
            "dashboard_path": "x_dashboard.html",
        },
    )

    assert report["status"] == "passed"
    assert report["readiness"]["graph_loaded"]
    assert report["readiness"]["nl2sql_answered"]
    assert report["readiness"]["insight_report_exported"]
    assert report["readiness"]["dashboard_exported"]
    assert report["metrics"]["chart_count"] >= 4


def test_plan_analysis_task_selects_expected_operators():
    plan = plan_analysis_task("做统计分析、趋势分析和可视化", question="哪些疾病最多？")

    assert plan["task_type"] == "full_analysis"
    assert "generate_statistical_summary" in plan["operators"]
    assert "translate_question_to_sql" in plan["operators"]


def test_plan_detects_graph_analytics_intent():
    plan = plan_analysis_task("分析关键枢纽节点、社区结构和最短路径")
    assert "graph_analytics" in plan["intent_keywords"]

    default_plan = plan_analysis_task(None, question="哪些疾病关联最多症状？")
    assert "graph_analytics" not in default_plan["intent_keywords"]


def test_agent_runs_extended_analytics_when_planned(tmp_path):
    graph_file = _graph_path(tmp_path)
    result = GraphAnalysisAgent().run(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
        task_request="请分析图谱的核心枢纽节点、社区结构与最短路径",
    )

    assert result.status == "completed"
    assert result.artifacts["plan_execution"]["extended_analytics"] is True
    analytics = result.artifacts["graph_analytics"]
    assert analytics["status"] == "completed"
    assert analytics["communities"]["status"] == "completed"
    assert analytics["shortest_paths"]["status"] in {"reachable", "path_found", "no_path"}
    assert "compute_shortest_paths" in result.artifacts["plan_execution"]["executed_optional_operators"]


def test_agent_skips_extended_analytics_by_default(tmp_path):
    graph_file = _graph_path(tmp_path)
    result = GraphAnalysisAgent().run(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
        question="哪些疾病关联最多症状？",
    )

    assert result.status == "completed"
    assert result.artifacts["plan_execution"]["extended_analytics"] is False
    assert result.artifacts["graph_analytics"]["status"] == "skipped"
    assert "compute_shortest_paths" not in result.artifacts["plan_execution"]["executed_optional_operators"]


def test_agent_records_full_operator_execution_mapping(tmp_path):
    graph_file = _graph_path(tmp_path)
    result = GraphAnalysisAgent().run(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
        question="哪些疾病关联最多症状？",
    )

    plan_execution = result.artifacts["plan_execution"]
    assert "generate_statistical_summary" in plan_execution["planned_operators"]
    assert "generate_statistical_summary" in plan_execution["executed_operators"]
    assert "compute_centrality" in plan_execution["executed_operators"]
    # Extended graph operators are opt-in; default run must not list them.
    assert "compute_shortest_paths" not in plan_execution["executed_operators"]
    assert "detect_communities" not in plan_execution["executed_operators"]


def test_agent_execution_mapping_includes_extended_when_planned(tmp_path):
    graph_file = _graph_path(tmp_path)
    result = GraphAnalysisAgent().run(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
        task_request="请分析图谱的核心枢纽节点、社区结构与最短路径",
    )

    executed = result.artifacts["plan_execution"]["executed_operators"]
    assert "compute_shortest_paths" in executed
    assert "detect_communities" in executed


def test_should_run_extended_analytics_honors_llm_operator_selection():
    from src.agents.analysis_agent.agent import _should_run_extended_analytics

    # Rule planner: triggered only by the graph_analytics intent keyword.
    assert _should_run_extended_analytics(
        {"planner_mode": "rule", "intent_keywords": ["graph_analytics"],
         "operators": ["load_graph"]}
    ) is True
    # Rule planner always lists every operator; without the intent it must skip,
    # otherwise extended analytics would never be opt-in for rule-based plans.
    assert _should_run_extended_analytics(
        {"planner_mode": "rule", "intent_keywords": ["statistics"],
         "operators": ["compute_shortest_paths", "detect_communities"]}
    ) is False
    # LLM planner: free-form keywords (no 'graph_analytics') but it explicitly
    # selected the extended operators, so its selection drives execution.
    assert _should_run_extended_analytics(
        {"planner_mode": "llm", "intent_keywords": ["图谱", "社区结构"],
         "operators": ["load_graph", "detect_communities"]}
    ) is True
    # LLM planner that did not select extended operators -> skipped.
    assert _should_run_extended_analytics(
        {"planner_mode": "llm", "intent_keywords": ["统计"],
         "operators": ["load_graph", "generate_statistical_summary"]}
    ) is False


def test_operator_selected_rule_vs_llm_contract():
    from src.agents.analysis_agent.agent import _operator_selected

    # Rule planner lists every operator, so optional steps default to running.
    assert _operator_selected(
        {"planner_mode": "rule", "operators": ["load_graph"]},
        "translate_question_to_sql",
    ) is True
    # LLM planner with the operator selected -> run.
    assert _operator_selected(
        {"planner_mode": "llm", "operators": ["translate_question_to_sql", "execute_sql"]},
        "translate_question_to_sql",
    ) is True
    # LLM planner that omitted the operator -> skip.
    assert _operator_selected(
        {"planner_mode": "llm", "operators": ["generate_statistical_summary"]},
        "translate_question_to_sql",
    ) is False


def test_agent_skips_nl2sql_when_llm_plan_omits_sql(tmp_path, monkeypatch):
    """An LLM plan that omits the SQL operators must skip NL2SQL (Task 2 parity)."""
    from src.agents.analysis_agent.agent import AnalysisHybridPlanner

    graph_file = _graph_path(tmp_path)

    def _fake_plan(self, task_request, question=None, graph_summary=None):
        return {
            "planner_mode": "llm",
            "intent_keywords": ["统计"],
            "operators": [
                "load_graph",
                "generate_statistical_summary",
                "generate_association_analysis",
                "generate_trend_analysis",
                "compute_centrality",
                "build_analysis_visualizations",
                "build_analysis_report",
            ],
        }

    monkeypatch.setattr(AnalysisHybridPlanner, "plan", _fake_plan)
    result = GraphAnalysisAgent().run(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
        question="哪些疾病关联最多症状？",
    )

    assert result.status == "completed"
    assert result.artifacts["nl2sql"]["status"] == "skipped"
    plan_execution = result.artifacts["plan_execution"]
    assert plan_execution["nl2sql_executed"] is False
    assert "translate_question_to_sql" not in plan_execution["executed_operators"]
    step_names = [s["name"] for s in result.artifacts["run_state"]["steps"]]
    assert "translate_and_execute_sql" not in step_names
    assert "build_sqlite" not in step_names


def test_agent_runs_nl2sql_for_rule_plan(tmp_path):
    """The rule planner lists all operators, so NL2SQL must still run by default."""
    graph_file = _graph_path(tmp_path)
    result = GraphAnalysisAgent().run(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
        question="哪些疾病关联最多症状？",
    )
    assert result.artifacts["plan_execution"]["nl2sql_executed"] is True
    step_names = [s["name"] for s in result.artifacts["run_state"]["steps"]]
    assert "translate_and_execute_sql" in step_names


def test_task3_evaluation_writes_compact_report(tmp_path):
    graph_file = _graph_path(tmp_path)
    report_path = tmp_path / "reports" / "task3_quality_report.json"

    payload = run_task3_evaluation(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
        report_path=report_path,
        question="哪些疾病关联最多症状？",
    )

    assert payload["task"] == "task3_analysis_agent"
    assert payload["status"] == "completed"
    assert payload["quality_report"]["status"] == "passed"
    assert payload["graph"]["node_count"] > 0
    assert payload["nl2sql"]["row_count"] > 0
    assert report_path.exists()


def test_task3_api_process_status_report_and_sql(tmp_path):
    from fastapi.testclient import TestClient
    from src.pipelines.task3_api_server import app, _tasks

    graph_file = _graph_path(tmp_path)
    _tasks.clear()

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["service"] == "task3_graph_analysis_agent"

    operators = client.get("/api/task3/operators")
    assert operators.status_code == 200
    assert "generate_statistical_summary" in operators.json()["operators"]

    submitted = client.post(
        "/api/task3/process",
        json={
            "graph_file": str(graph_file),
            "output_dir": str(tmp_path / "task3"),
            "question": "哪些疾病关联最多症状？",
        },
    )
    assert submitted.status_code == 200
    task_id = submitted.json()["task_id"]

    status = client.get(f"/api/task3/status/{task_id}")
    assert status.json()["status"] == "completed"

    report = client.get(f"/api/task3/report/{task_id}")
    assert report.status_code == 200
    assert report.json()["artifacts"]["quality_report"]["status"] == "passed"

    sql = client.post(
        "/api/task3/sql",
        json={"task_id": task_id, "question": "哪些疾病关联最多症状？"},
    )
    assert sql.status_code == 200
    assert sql.json()["rows"]


def test_task3_api_nl2sql_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    from src.pipelines.task3_api_server import app

    graph_file = _graph_path(tmp_path)
    client = TestClient(app)
    response = client.post(
        "/api/nl2sql",
        json={
            "question": "哪些疾病关联最多症状？",
            "graph_file": str(graph_file),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["sql"]
    assert payload["rows"]


def test_task3_api_404_for_unknown_task():
    from fastapi.testclient import TestClient
    from src.pipelines.task3_api_server import app, _tasks

    _tasks.clear()
    client = TestClient(app)
    response = client.get("/api/task3/status/missing")

    assert response.status_code == 404


def test_task3_benchmark_reports_cpu_metrics(tmp_path):
    from src.operators.npu_ops import benchmark_task3_analysis_ops

    graph = load_graph(_graph_path(tmp_path))
    report = benchmark_task3_analysis_ops(graph, question="哪些疾病关联最多症状？", iterations=3)

    assert report["task"] == "task3_analysis_agent"
    assert report["cpu"]["status"] == "completed"
    assert report["cpu"]["iterations"] == 3
    assert report["cpu"]["latency_ms_avg"] >= 0
    assert report["cpu"]["chart_count"] >= 4
    assert report["npu"]["status"] in {"available", "unavailable"}


def test_task3_benchmark_forwards_npu_probe_options(monkeypatch, tmp_path):
    import src.operators.npu_ops.analysis_benchmark as analysis_benchmark

    captured = {}

    def fake_detect_npu_runtime(**kwargs):
        captured.update(kwargs)
        return {"status": "available", "backend": "fake"}

    monkeypatch.setattr(analysis_benchmark, "detect_npu_runtime", fake_detect_npu_runtime)

    graph = load_graph(_graph_path(tmp_path))
    report = analysis_benchmark.benchmark_task3_analysis_ops(
        graph,
        question="哪些疾病关联最多症状？",
        iterations=1,
        npu_probe=False,
        npu_probe_iterations=9,
        npu_probe_size=16,
    )

    assert captured == {"probe": False, "probe_iterations": 9, "probe_size": 16}
    assert report["input"]["npu_probe"] == {
        "enabled": False,
        "iterations": 9,
        "matrix_size": 16,
    }
    assert report["npu"]["backend"] == "fake"


def test_task3_smoke_summarizes_reviewer_path(tmp_path):
    graph_file = _graph_path(tmp_path)

    payload = run_task3_smoke(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
        iterations=2,
    )

    assert payload["status"] == "completed"
    assert payload["checks"]["pipeline_completed"]
    assert payload["checks"]["quality_passed"]
    assert payload["checks"]["html_report_exists"]
    assert payload["checks"]["markdown_report_exists"]
    assert payload["checks"]["dashboard_exists"]
    assert payload["checks"]["benchmark_completed"]
    assert payload["checks"]["nexent_spec_ready"]
    assert payload["checks"]["top_hubs_backend_recorded"]
    assert payload["artifacts"]["chart_count"] >= 4
    assert payload["artifacts"]["top_hubs_backend"] in {"python", "torch_npu"}


def test_task3_smoke_cli_outputs_json(tmp_path):
    import subprocess
    import sys

    graph_file = _graph_path(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "demos/task3_smoke.py",
            "--graph-file", str(graph_file),
            "--output-dir", str(tmp_path / "task3"),
            "--iterations", "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["status"] == "completed"
    assert payload["checks"]["pipeline_completed"]
    assert payload["checks"]["top_hubs_backend_recorded"]
    assert payload["artifacts"]["top_hubs_backend"] in {"python", "torch_npu"}


def test_task3_demo_cli_prints_hubs_backend(tmp_path):
    import subprocess
    import sys

    graph_file = _graph_path(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "demos/task3_demo.py",
            "--graph-file", str(graph_file),
            "--output-dir", str(tmp_path / "task3_demo"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert proc.returncode == 0
    assert "[Hubs Backend]" in proc.stdout


# ---- Phase 1 Enhancement Tests ----


def test_compute_centrality_finds_hubs(tmp_path):
    """Centrality analysis should identify high-degree hub nodes."""
    graph_file = _graph_path(tmp_path)
    from src.operators.analysis_ops.graph_analytics import compute_centrality
    graph = load_graph(graph_file)
    result = compute_centrality(graph)
    assert result["status"] == "completed"
    assert len(result["top_hubs"]) > 0
    assert result["top_hubs"][0]["degree"] > 0
    assert "type_centrality" in result
    # Disease nodes should have highest degree in medical KG
    top = result["top_hubs"][0]
    assert top["type"] in {"Disease", "Symptom", "Drug", "Examination", "Treatment"}


def test_compute_centrality_uses_npu_top_hubs_when_available(monkeypatch):
    import src.operators.analysis_ops.graph_analytics as graph_analytics

    graph = {
        "nodes": [
            {"id": "d1", "name": "disease", "type": "Disease"},
            {"id": "s1", "name": "symptom", "type": "Symptom"},
            {"id": "s2", "name": "symptom2", "type": "Symptom"},
        ],
        "edges": [
            {"source": "d1", "target": "s1"},
            {"source": "d1", "target": "s2"},
        ],
    }
    monkeypatch.setattr(
        graph_analytics,
        "compute_degree_topk_npu_cached",
        lambda _graph, prefer_device="auto", top_k=10, kernel="bincount": {
            "status": "completed",
            "backend": "torch_npu",
            "device": "npu:0",
            "kernel": kernel,
            "top_hubs": [
                {
                    "id": "d1",
                    "name": "disease",
                    "type": "Disease",
                    "degree": 2,
                    "degree_centrality": 1.0,
                }
            ],
        },
        raising=False,
    )

    result = graph_analytics.compute_centrality(graph)

    assert result["top_hubs_backend"] == "torch_npu"
    assert result["top_hubs"] == [
        {
            "id": "d1",
            "name": "disease",
            "type": "Disease",
            "degree": 2,
            "degree_centrality": 1.0,
        }
    ]
    assert result["type_centrality"]["Disease"]["top_node"] == "disease"


def test_compute_centrality_reuses_degree_tensor_cache(monkeypatch):
    import src.operators.analysis_ops.graph_analytics as graph_analytics

    graph = {
        "nodes": [
            {"id": "d1", "name": "disease", "type": "Disease"},
            {"id": "s1", "name": "symptom", "type": "Symptom"},
        ],
        "edges": [{"source": "d1", "target": "s1"}],
    }
    cache = {"operator": "graph_degree_tensor_cache", "status": "completed"}
    captured = {}

    def fake_topk(graph_or_cache, prefer_device="auto", top_k=10, kernel="bincount"):
        captured["graph_or_cache"] = graph_or_cache
        captured["prefer_device"] = prefer_device
        return {
            "status": "completed",
            "backend": "torch_npu",
            "top_hubs": [
                {
                    "id": "d1",
                    "name": "disease",
                    "type": "Disease",
                    "degree": 1,
                    "degree_centrality": 1.0,
                }
            ],
        }

    monkeypatch.setattr(graph_analytics, "compute_degree_topk_npu_cached", fake_topk)

    result = graph_analytics.compute_centrality(
        graph,
        prefer_device="auto",
        degree_tensor_cache=cache,
    )

    assert captured["graph_or_cache"] is cache
    assert captured["prefer_device"] == "auto"
    assert result["top_hubs_backend"] == "torch_npu"
    assert result["top_hubs"][0]["id"] == "d1"


def test_compute_centrality_falls_back_to_cpu_top_hubs_when_npu_unavailable(monkeypatch):
    import src.operators.analysis_ops.graph_analytics as graph_analytics

    graph = {
        "nodes": [
            {"id": "d1", "name": "disease", "type": "Disease"},
            {"id": "s1", "name": "symptom", "type": "Symptom"},
            {"id": "s2", "name": "symptom2", "type": "Symptom"},
        ],
        "edges": [
            {"source": "d1", "target": "s1"},
            {"source": "d1", "target": "s2"},
        ],
    }
    monkeypatch.setattr(
        graph_analytics,
        "compute_degree_topk_npu_cached",
        lambda _graph, prefer_device="auto", top_k=10, kernel="bincount": {
            "status": "unavailable",
            "reason": "no npu",
            "top_hubs": [],
        },
        raising=False,
    )

    result = graph_analytics.compute_centrality(graph)

    assert result["top_hubs_backend"] == "python"
    assert result["top_hubs"][0]["id"] == "d1"
    assert result["top_hubs_npu_reason"] == "no npu"
    assert "type_centrality" in result


def test_compute_shortest_paths_finds_route(tmp_path):
    """Shortest path between two connected entities should be found."""
    graph_file = _graph_path(tmp_path)
    from src.operators.analysis_ops.graph_analytics import compute_shortest_paths
    graph = load_graph(graph_file)
    result = compute_shortest_paths(graph, "高血压", "氨氯地平", max_depth=4)
    assert result["status"] == "path_found"
    assert result["hop_count"] >= 1
    assert len(result["paths"]) == 1
    steps = result["paths"][0]["steps"]
    assert any("高血压" in s["source"] or "高血压" in s["target"] for s in steps)


def test_detect_communities_groups_nodes(tmp_path):
    """Community detection should group nodes into clusters."""
    graph_file = _graph_path(tmp_path)
    from src.operators.analysis_ops.graph_analytics import detect_communities
    graph = load_graph(graph_file)
    result = detect_communities(graph)
    assert result["status"] == "completed"
    assert result["community_count"] >= 1
    assert len(result["communities"]) >= 1
    # Largest community should have multiple nodes
    assert result["communities"][0]["size"] >= 2


def test_llm_nl2sql_falls_back_to_template(tmp_path):
    """When LLM is unavailable, NL2SQL should fall back to template."""
    graph_file = _graph_path(tmp_path)
    from src.operators.analysis_ops.llm_nl2sql import translate_question_to_sql_with_llm
    graph = load_graph(graph_file)
    conn = build_graph_sqlite(graph)

    # Use invalid LLM config to trigger fallback
    bad_config = {"base_url": "http://localhost:1", "api_key": "fake", "timeout": 1.0}
    result = translate_question_to_sql_with_llm(
        "高血压有哪些症状？", conn, llm_config=bad_config,
    )
    assert result["status"] == "completed"
    assert result["translator"] == "template"
    assert len(result["rows"]) > 0


def test_llm_nl2sql_template_mode_without_llm(tmp_path):
    """Without LLM config, NL2SQL should use template translation."""
    graph_file = _graph_path(tmp_path)
    from src.operators.analysis_ops.llm_nl2sql import translate_question_to_sql_with_llm
    graph = load_graph(graph_file)
    conn = build_graph_sqlite(graph)

    result = translate_question_to_sql_with_llm(
        "高血压有哪些症状？", conn, llm_config=None,
    )
    assert result["status"] == "completed"
    assert result["translator"] == "template"


def test_hybrid_planner_rule_fallback():
    """Hybrid planner should fall back to rule-based without LLM."""
    from src.operators.analysis_ops.hybrid_planner import AnalysisHybridPlanner

    planner = AnalysisHybridPlanner(llm_config=None)
    plan = planner.plan("统计分析", question="哪些疾病关联最多症状？")
    assert plan["planner_mode"] == "rule"
    assert len(plan["operators"]) >= 2


def test_task3_pipeline_with_llm_config(tmp_path):
    """Pipeline should accept llm_config and still complete (LLM may fail)."""
    graph_file = _graph_path(tmp_path)
    result = run_task3_pipeline(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
        question="高血压有哪些症状？",
        llm_config=None,  # No LLM available in test
    )
    assert result.status == "completed"
    assert "centrality" in result.artifacts
    assert result.artifacts["plan"]["planner_mode"] == "rule"


def test_echarts_dashboard_generates_html(tmp_path):
    """ECharts dashboard should generate a valid HTML file with chart options."""
    graph_file = _graph_path(tmp_path)
    result = run_task3_pipeline(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
    )
    assert result.status == "completed"
    echarts = result.artifacts.get("echarts_dashboard", {})
    assert echarts.get("status") == "completed"
    assert Path(echarts["dashboard_path"]).exists()
    content = Path(echarts["dashboard_path"]).read_text(encoding="utf-8")
    assert "echarts" in content.lower()
    assert "Medical KG Analysis Dashboard" in content
    assert "entity-chart" in content  # chart container IDs


def test_echarts_dashboard_has_svg_fallback(tmp_path):
    """ECharts dashboard should contain inline SVG fallback for offline use."""
    graph_file = _graph_path(tmp_path)
    result = run_task3_pipeline(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
    )
    echarts = result.artifacts["echarts_dashboard"]
    content = Path(echarts["dashboard_path"]).read_text(encoding="utf-8")
    # Must have fallback divs
    assert "entity-fallback" in content
    assert "relation-fallback" in content
    assert "trend-fallback" in content
    assert "hub-fallback" in content
    # Must have inline SVG in fallback sections
    assert "<svg" in content
    assert "fallback" in content


def test_echarts_dashboard_trend_and_hub_series_use_real_counts(tmp_path):
    graph_file = _graph_path(tmp_path)
    result = run_task3_pipeline(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
    )
    content = Path(result.artifacts["echarts_dashboard"]["dashboard_path"]).read_text(encoding="utf-8")
    record_trends = result.artifacts["trends"]["record_trends"]
    expected_entity_counts = [row["entity_count"] for row in record_trends]
    expected_edge_counts = [row["edge_count"] for row in record_trends]

    entity_match = _extract_echarts_series_data(content, "实体数")
    edge_match = _extract_echarts_series_data(content, "边数")
    assert entity_match == expected_entity_counts
    assert edge_match == expected_edge_counts
    assert all(count > 0 for count in expected_entity_counts)
    assert "Top Hub 节点" in content
    assert "暂无数据" not in content
    assert '"type": "category"' in content
    assert '"type": "value"' in content


def _extract_echarts_series_data(html_content: str, series_name: str) -> list[int]:
    match = re.search(
        rf'"name": "{re.escape(series_name)}".*?"data": (\[[^\]]+\])',
        html_content,
        re.S,
    )
    assert match, f"ECharts series {series_name!r} not found in dashboard HTML"
    return json.loads(match.group(1))


def test_echarts_dashboard_shows_fallback_before_async_cdn_load(tmp_path):
    graph_file = _graph_path(tmp_path)
    result = run_task3_pipeline(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
    )
    content = Path(
        result.artifacts["echarts_dashboard"]["dashboard_path"]
    ).read_text(encoding="utf-8")

    assert ".chart { width: 100%; height: 320px; display: none; }" in content
    assert ".fallback { display: block;" in content
    assert '<script src="https://cdn.jsdelivr.net/' not in content
    assert "document.createElement('script')" in content
    assert "echarts@5.6.0/dist/echarts.min.js" in content


def test_echarts_dashboard_defers_init_until_layout_ready(tmp_path):
    graph_file = _graph_path(tmp_path)
    result = run_task3_pipeline(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
    )
    content = Path(result.artifacts["echarts_dashboard"]["dashboard_path"]).read_text(encoding="utf-8")

    assert "requestAnimationFrame" in content
    assert "hideFallbackPanels" in content
    assert "chart.resize()" in content
    assert '"animation": false' in content
    assert '"grid": {"left": 48' in content


def test_echarts_svg_pie_uses_valid_svg_legend_elements():
    svg = _svg_pie(
        [
            {"category": "Disease", "value": 3},
            {"category": "Drug", "value": 2},
        ],
        "实体类型分布",
    )

    assert "<div" not in svg
    assert "<span" not in svg
    assert "<rect" in svg
    assert "Disease (60%)" in svg


def test_echarts_dashboard_contains_kg_force_graph(tmp_path):
    """ECharts dashboard must render KG structure as force-directed graph."""
    graph_file = _graph_path(tmp_path)
    result = run_task3_pipeline(
        graph_file=graph_file,
        output_dir=tmp_path / "task3",
    )
    echarts = result.artifacts["echarts_dashboard"]
    content = Path(echarts["dashboard_path"]).read_text(encoding="utf-8")
    # Must have KG graph container
    assert "kg-chart" in content
    assert "kg-fallback" in content
    # Must contain ECharts graph series with force layout
    assert "type\":\"graph" in content or '"type": "graph"' in content
    assert "force" in content
    assert "Disease" in content
    assert "Symptom" in content
    # Must have tooltip for edges
    assert "kg-chart" in content


def test_echarts_dashboard_escapes_script_context_data(tmp_path):
    """Dashboard JSON payloads should not break out of inline script tags."""
    visualizations = {
        "charts": {
            "entity_distribution": {"data": [{"category": "</script><script>alert(1)</script>", "value": 1}]},
            "relation_distribution": {"data": []},
            "record_trend": {"data": []},
        }
    }
    dashboard = export_echarts_dashboard(
        target_dir=tmp_path / "dashboard",
        statistics={"confidence": {}, "entity_type_counts": {"Disease": 1}, "relation_type_counts": {}},
        associations={"top_associations": []},
        trends={},
        nl2sql={"rows": [], "intent": "test", "translator": "test", "sql": "SELECT * FROM nodes LIMIT 1"},
        visualizations=visualizations,
        centrality={"top_hubs": []},
    )

    content = Path(dashboard["dashboard_path"]).read_text(encoding="utf-8")
    assert "</script><script>alert(1)</script>" not in content
    assert "\\u003c/script\\u003e" in content


def test_echarts_kg_tooltip_uses_safe_text_mode(tmp_path):
    dashboard = export_echarts_dashboard(
        target_dir=tmp_path / "dashboard",
        statistics={
            "confidence": {},
            "entity_type_counts": {"Disease": 1},
            "relation_type_counts": {"has_symptom": 1},
        },
        associations={"top_associations": []},
        trends={},
        nl2sql={"rows": [], "sql": "SELECT 1"},
        visualizations={
            "charts": {
                "entity_distribution": {"data": []},
                "relation_distribution": {"data": []},
                "record_trend": {"data": []},
                "disease_network": {
                    "nodes": [
                        {
                            "id": "Disease:test",
                            "label": "<img src=x onerror=alert(1)>",
                            "type": "Disease",
                        }
                    ],
                    "edges": [
                        {
                            "source": "Disease:test",
                            "target": "Symptom:test",
                            "relation": "has_symptom",
                        }
                    ],
                },
            }
        },
        centrality={"top_hubs": []},
    )

    content = Path(dashboard["dashboard_path"]).read_text(encoding="utf-8")

    assert '"renderMode": "richText"' in content
    assert '"formatter": "{b}"' in content
    assert '"name": "has_symptom"' in content
    assert "function(params)" not in content
    assert "\\u003cbr/\\u003e" not in content


# ---- Issue 2: API endpoint tests for graph analytics ----


def test_task3_api_centrality_endpoint(tmp_path):
    """POST /api/task3/centrality should return centrality analysis."""
    from fastapi.testclient import TestClient
    from src.pipelines.task3_api_server import app, _tasks

    graph_file = _graph_path(tmp_path)
    _tasks.clear()
    client = TestClient(app)

    resp = client.post(
        "/api/task3/centrality",
        json={"graph_file": str(graph_file)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert len(body["top_hubs"]) > 0
    assert body["top_hubs"][0]["degree"] > 0


def test_task3_api_paths_endpoint(tmp_path):
    """POST /api/task3/paths should find shortest paths between entities."""
    from fastapi.testclient import TestClient
    from src.pipelines.task3_api_server import app, _tasks

    graph_file = _graph_path(tmp_path)
    _tasks.clear()
    client = TestClient(app)

    # First submit a task to get a task_id
    submitted = client.post(
        "/api/task3/process",
        json={"graph_file": str(graph_file), "output_dir": str(tmp_path / "task3")},
    )
    task_id = submitted.json()["task_id"]

    resp = client.post(
        "/api/task3/paths",
        json={"task_id": task_id, "start_entity": "高血压", "end_entity": "氨氯地平"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "path_found"
    assert body["hop_count"] >= 1


def test_task3_api_paths_rejects_excessive_max_depth():
    from fastapi.testclient import TestClient
    from src.pipelines.task3_api_server import app

    client = TestClient(app)
    resp = client.post(
        "/api/task3/paths",
        json={"task_id": "unused", "start_entity": "高血压", "max_depth": 101},
    )

    assert resp.status_code == 422


def test_task3_api_communities_endpoint(tmp_path):
    """POST /api/task3/communities should detect communities."""
    from fastapi.testclient import TestClient
    from src.pipelines.task3_api_server import app, _tasks

    graph_file = _graph_path(tmp_path)
    _tasks.clear()
    client = TestClient(app)

    resp = client.post(
        "/api/task3/communities",
        json={"graph_file": str(graph_file)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["community_count"] >= 1


def test_task3_api_centrality_404_for_missing_graph(tmp_path):
    """POST /api/task3/centrality should 404 for non-existent graph file."""
    from fastapi.testclient import TestClient
    from src.pipelines.task3_api_server import app, _tasks

    _tasks.clear()
    client = TestClient(app)

    resp = client.post(
        "/api/task3/centrality",
        json={"graph_file": str(tmp_path / "nonexistent.json")},
    )
    assert resp.status_code == 404


def test_hybrid_planner_llm_uses_openai():
    """Hybrid planner LLM path should use openai library (mock test)."""
    import sys
    from unittest.mock import MagicMock
    from src.operators.analysis_ops.hybrid_planner import AnalysisHybridPlanner

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "task_type": "full_analysis",
        "operators": ["load_graph", "generate_statistical_summary", "translate_question_to_sql"],
        "intent_keywords": ["statistics", "nl2sql"],
        "confidence": 0.92,
    })

    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.OpenAI.return_value = mock_client

    fake_config = {
        "base_url": "https://fake.api/v1",
        "api_key": "test-key",
        "model_name": "test-model",
        "timeout": 10.0,
    }

    saved = sys.modules.get("openai")
    try:
        sys.modules["openai"] = mock_openai
        planner = AnalysisHybridPlanner(llm_config=fake_config)
        plan = planner.plan("统计分析", question="高血压有哪些症状？")
    finally:
        if saved is not None:
            sys.modules["openai"] = saved
        else:
            sys.modules.pop("openai", None)

    assert plan["planner_mode"] == "llm"
    assert plan["confidence"] == 0.92
    assert "generate_statistical_summary" in plan["operators"]
    assert "translate_question_to_sql" in plan["operators"]
    mock_openai.OpenAI.assert_called_once_with(
        base_url="https://fake.api/v1",
        api_key="test-key",
    )
    mock_client.chat.completions.create.assert_called_once()


def test_hybrid_planner_llm_fallback_on_openai_error():
    """Hybrid planner should fallback to rules when openai call fails."""
    import sys
    from unittest.mock import MagicMock
    from src.operators.analysis_ops.hybrid_planner import AnalysisHybridPlanner

    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API error")
    mock_openai.OpenAI.return_value = mock_client

    fake_config = {
        "base_url": "https://fake.api/v1",
        "api_key": "test-key",
    }

    saved = sys.modules.get("openai")
    try:
        sys.modules["openai"] = mock_openai
        planner = AnalysisHybridPlanner(llm_config=fake_config)
        plan = planner.plan("统计分析")
    finally:
        if saved is not None:
            sys.modules["openai"] = saved
        else:
            sys.modules.pop("openai", None)

    assert plan["planner_mode"] == "rule"


def test_hybrid_planner_llm_fallback_on_import_error():
    """Hybrid planner should fallback to rules when openai is not installed."""
    from unittest.mock import patch
    from src.operators.analysis_ops.hybrid_planner import AnalysisHybridPlanner

    fake_config = {
        "base_url": "https://fake.api/v1",
        "api_key": "test-key",
    }

    # Simulate openai not being installed
    with patch.dict("sys.modules", {"openai": None}):
        planner = AnalysisHybridPlanner(llm_config=fake_config)
        plan = planner.plan("统计分析")

    assert plan["planner_mode"] == "rule"


def test_hybrid_planner_llm_json_with_code_fences():
    """Planner should parse LLM response wrapped in markdown code fences."""
    from src.operators.analysis_ops.hybrid_planner import _parse_plan_json

    raw = '```json\n{"task_type": "full_analysis", "operators": ["load_graph", "generate_statistical_summary"], "confidence": 0.85}\n```'
    result = _parse_plan_json(raw)
    assert result["task_type"] == "full_analysis"
    assert len(result["operators"]) == 2
    assert result["confidence"] == 0.85


def test_llm_nl2sql_success_with_mock(tmp_path):
    """LLM NL2SQL should return llm translator when openai succeeds."""
    import sys
    from unittest.mock import MagicMock
    from src.operators.analysis_ops.llm_nl2sql import translate_question_to_sql_with_llm

    graph_file = _graph_path(tmp_path)
    graph = load_graph(graph_file)
    conn = build_graph_sqlite(graph)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = (
        "SELECT n.name AS entity FROM edges e "
        "JOIN nodes n ON e.target = n.id "
        "WHERE e.predicate = 'has_symptom' "
        "ORDER BY entity ASC LIMIT 10"
    )

    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.OpenAI.return_value = mock_client

    fake_config = {
        "base_url": "https://fake.api/v1",
        "api_key": "test-key",
        "model_name": "test-model",
        "timeout": 10.0,
    }

    saved = sys.modules.get("openai")
    try:
        sys.modules["openai"] = mock_openai
        result = translate_question_to_sql_with_llm(
            "List all symptom entities", conn, llm_config=fake_config,
        )
    finally:
        if saved is not None:
            sys.modules["openai"] = saved
        else:
            sys.modules.pop("openai", None)

    assert result["status"] == "completed"
    assert result["translator"] == "llm"
    assert result["intent"] == "llm_generated"
    assert "SELECT" in result["sql"]
    assert isinstance(result["rows"], list)
    mock_client.chat.completions.create.assert_called_once()


def test_pipeline_llm_success_path(tmp_path):
    """Full pipeline should produce planner=llm and translator=llm when mocked."""
    import sys
    from unittest.mock import MagicMock

    graph_file = _graph_path(tmp_path)

    # Mock planner response
    planner_response = MagicMock()
    planner_response.choices = [MagicMock()]
    planner_response.choices[0].message.content = json.dumps({
        "task_type": "full_analysis",
        "operators": [
            "load_graph", "generate_statistical_summary",
            "generate_association_analysis", "generate_trend_analysis",
            "translate_question_to_sql", "execute_sql",
            "build_analysis_visualizations", "build_analysis_report",
        ],
        "intent_keywords": ["statistics", "nl2sql"],
        "confidence": 0.95,
    })

    # Mock NL2SQL response
    nl2sql_response = MagicMock()
    nl2sql_response.choices = [MagicMock()]
    nl2sql_response.choices[0].message.content = (
        "SELECT type, COUNT(*) AS cnt FROM nodes GROUP BY type ORDER BY cnt DESC LIMIT 10"
    )

    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [planner_response, nl2sql_response]
    mock_openai.OpenAI.return_value = mock_client

    fake_config = {
        "base_url": "https://fake.api/v1",
        "api_key": "test-key",
        "model_name": "test-model",
        "timeout": 10.0,
    }

    saved = sys.modules.get("openai")
    try:
        sys.modules["openai"] = mock_openai
        result = run_task3_pipeline(
            graph_file=graph_file,
            output_dir=tmp_path / "task3",
            question="Show entity distribution",
            llm_config=fake_config,
        )
    finally:
        if saved is not None:
            sys.modules["openai"] = saved
        else:
            sys.modules.pop("openai", None)

    assert result.status == "completed"
    assert result.artifacts["plan"]["planner_mode"] == "llm"
    assert result.artifacts["plan"]["confidence"] == 0.95
    assert result.artifacts["nl2sql"]["translator"] == "llm"
    assert result.artifacts["nl2sql"]["intent"] == "llm_generated"
    assert "SELECT" in result.artifacts["nl2sql"]["sql"]
    assert isinstance(result.artifacts["nl2sql"]["rows"], list)


def test_local_model_planning_used_when_predict_plan_succeeds(tmp_path):
    """When the local model returns a valid plan, the planner adopts it."""
    from unittest.mock import patch

    from src.operators.analysis_ops.hybrid_planner import AnalysisHybridPlanner

    model_dir = tmp_path / "analysis_model"
    model_dir.mkdir()

    fake_plan = {
        "task_type": "full_analysis",
        "operators": ["load_graph", "compute_centrality", "detect_communities"],
        "intent_keywords": ["graph_analytics"],
        "confidence": 0.91,
    }

    with patch(
        "src.operators.analysis_ops.local_model_planning.predict_plan",
        return_value=fake_plan,
    ) as mock_predict:
        planner = AnalysisHybridPlanner(local_model_path=str(model_dir))
        plan = planner.plan("分析核心枢纽和社区", question="高血压有哪些症状？")

    assert plan["planner_mode"] == "local_model"
    assert "compute_centrality" in plan["operators"]
    assert plan["confidence"] == 0.91
    mock_predict.assert_called_once()


def test_local_model_planning_falls_back_when_none(tmp_path):
    """A None local-model result must fall back to the rule planner."""
    from unittest.mock import patch

    from src.operators.analysis_ops.hybrid_planner import AnalysisHybridPlanner

    model_dir = tmp_path / "analysis_model"
    model_dir.mkdir()

    with patch(
        "src.operators.analysis_ops.local_model_planning.predict_plan",
        return_value=None,
    ):
        planner = AnalysisHybridPlanner(local_model_path=str(model_dir))
        plan = planner.plan("做统计分析")

    assert plan["planner_mode"] == "rule"


def test_local_model_nl2sql_used_when_predict_sql_succeeds(tmp_path):
    """The hybrid translator prefers the local-model SQL when it executes."""
    from unittest.mock import patch

    from src.operators.analysis_ops.llm_nl2sql import translate_question_with_fallbacks

    graph_file = _graph_path(tmp_path)
    conn = build_graph_sqlite(load_graph(graph_file))

    sql = INTENT_SQL["entity_distribution"]
    with patch(
        "src.operators.analysis_ops.local_model_nl2sql.predict_sql",
        return_value=sql,
    ) as mock_predict:
        result = translate_question_with_fallbacks(
            "各类实体的数量分布", conn, local_model_path=str(tmp_path),
        )

    assert result["translator"] == "local_model"
    assert "SELECT" in result["sql"]
    assert isinstance(result["rows"], list)
    mock_predict.assert_called_once()


def test_local_model_nl2sql_falls_back_to_template(tmp_path):
    """When the local model yields no SQL and no LLM is set, use the template."""
    from unittest.mock import patch

    from src.operators.analysis_ops.llm_nl2sql import translate_question_with_fallbacks

    graph_file = _graph_path(tmp_path)
    conn = build_graph_sqlite(load_graph(graph_file))

    with patch(
        "src.operators.analysis_ops.local_model_nl2sql.predict_sql",
        return_value=None,
    ):
        result = translate_question_with_fallbacks(
            "哪些疾病关联最多症状？", conn, local_model_path=str(tmp_path),
        )

    assert result["translator"] == "template"
    assert result["intent"] == "top_disease_symptoms"


def test_training_data_instructions_match_canonical_prompts():
    """Generated instructions must equal the shared prompts used at inference.

    A drift here would train an adapter on one prompt and query it with another,
    silently degrading the fine-tuned model to base behaviour.
    """
    import importlib.util

    from src.operators.analysis_ops.analysis_prompts import (
        NL2SQL_INSTRUCTION,
        PLANNING_INSTRUCTION,
    )

    gen_path = ROOT / "data" / "training" / "generate_analysis_training_data.py"
    spec = importlib.util.spec_from_file_location("_gen_prompts", gen_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.generate_planning_samples(1)[0]["instruction"] == PLANNING_INSTRUCTION
    assert module.generate_nl2sql_samples(1)[0]["instruction"] == NL2SQL_INSTRUCTION


def test_generate_analysis_training_data_produces_valid_samples():
    """The analysis training-data generator emits planning and NL2SQL samples."""
    import importlib.util

    gen_path = ROOT / "data" / "training" / "generate_analysis_training_data.py"
    spec = importlib.util.spec_from_file_location("_gen_analysis", gen_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    planning = module.generate_planning_samples(20)
    nl2sql = module.generate_nl2sql_samples(20)

    assert len(planning) == 20
    assert len(nl2sql) == 20
    for sample in planning:
        assert {"instruction", "input", "output"} <= set(sample)
        parsed = json.loads(sample["output"])
        assert isinstance(parsed["operators"], list) and parsed["operators"]
    for sample in nl2sql:
        assert {"instruction", "input", "output"} <= set(sample)
        assert sample["output"].strip().upper().startswith("SELECT")


def test_generate_graph_insights_produces_grounded_narrative(tmp_path):
    """Graph-driven insights must be derived from the graph structure itself."""
    from src.operators.analysis_ops import compute_centrality, generate_graph_insights

    graph = load_graph(_graph_path(tmp_path))
    associations = generate_association_analysis(graph)
    centrality = compute_centrality(graph, prefer_device="cpu")

    insight = generate_graph_insights(graph, {}, associations, centrality, None)

    assert insight["status"] == "completed"
    assert insight["insights"]
    # Scale sentence must reflect the real node/edge counts.
    assert str(insight["metrics"]["node_count"]) in insight["insights"][0]
    # Predicate distribution must be summarised.
    assert any("has_symptom" in s for s in insight["insights"])
    assert insight["metrics"]["predicate_counts"]


def test_generate_graph_insights_skips_empty_graph():
    from src.operators.analysis_ops import generate_graph_insights

    insight = generate_graph_insights({"nodes": [], "edges": []})
    assert insight["status"] == "skipped"
    assert insight["insights"] == []


def test_insight_report_includes_graph_driven_section(tmp_path):
    """The exported markdown must contain the graph-driven NL insight section."""
    from src.operators.analysis_ops import compute_centrality

    graph = load_graph(_graph_path(tmp_path))
    stats = generate_statistical_summary(graph)
    associations = generate_association_analysis(graph)
    trends = generate_trend_analysis(graph)
    visuals = build_analysis_visualizations(stats, associations, trends)
    centrality = compute_centrality(graph, prefer_device="cpu")
    nl2sql = {**translate_question_to_sql("哪些疾病关联最多症状？"), "rows": []}

    report = export_insight_report(
        target_dir=tmp_path / "reports",
        graph=graph,
        statistics=stats,
        associations=associations,
        trends=trends,
        nl2sql=nl2sql,
        visualizations=visuals,
        centrality=centrality,
    )

    markdown = Path(report["markdown_path"]).read_text(encoding="utf-8")
    assert "图谱驱动的自然语言洞察" in markdown
    assert "知识图谱由" in markdown


def test_nl2sql_execution_accuracy_reports_per_translator_breakdown():
    """The execution evaluator must attribute accuracy to the producing path."""
    from src.operators.analysis_ops.nl2sql import evaluate_nl2sql_execution_accuracy

    graph = _entity_aware_graph()
    cases = [
        {
            "question": "图谱中有哪些关系类型",
            "gold_sql": (
                "SELECT predicate, COUNT(*) AS edge_count FROM edges "
                "GROUP BY predicate ORDER BY edge_count DESC, predicate ASC"
            ),
        },
    ]

    # Default (template) path.
    default_report = evaluate_nl2sql_execution_accuracy(cases, graph)
    assert "template" in default_report["per_translator"]
    assert default_report["per_translator"]["template"]["total"] == 1

    # A custom translator path must be attributed to its own label.
    from src.operators.analysis_ops.nl2sql import translate_question_to_sql

    def fake_local_model(question, _conn, disease_names):
        result = translate_question_to_sql(question, disease_names=disease_names)
        return {**result, "translator": "local_model"}

    custom_report = evaluate_nl2sql_execution_accuracy(
        cases, graph, translator=fake_local_model
    )
    assert "local_model" in custom_report["per_translator"]
    assert custom_report["per_translator"]["local_model"]["total"] == 1
    assert custom_report["accuracy"] == 1.0


def test_nl2sql_benchmark_measures_local_model_path(tmp_path):
    """With a mocked local model, the fallback-chain translator path is measured."""
    from unittest.mock import patch

    from src.operators.analysis_ops.llm_nl2sql import translate_question_with_fallbacks
    from src.operators.analysis_ops.nl2sql import evaluate_nl2sql_execution_accuracy

    graph = _entity_aware_graph()
    cases = [
        {
            "question": "图谱中有哪些关系类型",
            "gold_sql": (
                "SELECT predicate, COUNT(*) AS edge_count FROM edges "
                "GROUP BY predicate ORDER BY edge_count DESC, predicate ASC"
            ),
        },
    ]
    gold_sql = cases[0]["gold_sql"]

    def translator(question, conn, _disease_names):
        return translate_question_with_fallbacks(
            question, conn, local_model_path=str(tmp_path)
        )

    with patch(
        "src.operators.analysis_ops.local_model_nl2sql.predict_sql",
        return_value=gold_sql,
    ):
        report = evaluate_nl2sql_execution_accuracy(cases, graph, translator=translator)

    assert "local_model" in report["per_translator"]
    assert report["per_translator"]["local_model"]["accuracy"] == 1.0


def test_nl2sql_llm_only_skips_without_config(tmp_path):
    from src.operators.analysis_ops.llm_nl2sql import translate_question_to_sql_llm_only

    conn = build_graph_sqlite(_entity_aware_graph())
    result = translate_question_to_sql_llm_only("列出所有疾病", conn, llm_config=None)

    assert result["translator"] == "llm"
    assert result["status"] == "skipped"


def test_nl2sql_local_only_skips_without_model(tmp_path):
    from src.operators.analysis_ops.llm_nl2sql import translate_question_to_sql_local_only

    conn = build_graph_sqlite(_entity_aware_graph())
    result = translate_question_to_sql_local_only("列出所有疾病", conn, local_model_path=None)

    assert result["translator"] == "local_model"
    assert result["status"] == "skipped"


def test_nl2sql_benchmark_reports_independent_paths(tmp_path):
    from unittest.mock import patch

    from benchmarks.task3_nl2sql_benchmark import main as benchmark_main

    report_path = tmp_path / "nl2sql_paths.json"
    with patch(
        "sys.argv",
        [
            "task3_nl2sql_benchmark.py",
            "--report",
            str(report_path),
        ],
    ):
        assert benchmark_main() == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "independent_paths" in report
    assert report["independent_paths"]["template"]["status"] == "evaluated"
    assert report["independent_paths"]["template"]["execution"]["accuracy"] == 1.0
    assert "enhancement_config" in report
    for label in ("llm", "local_model"):
        path = report["independent_paths"][label]
        assert path["status"] in {"evaluated", "not_configured"}


def test_nl2sql_independent_llm_path_with_mock(tmp_path):
    from unittest.mock import patch

    from src.operators.analysis_ops.llm_nl2sql import translate_question_to_sql_llm_only
    from src.operators.analysis_ops.nl2sql import evaluate_nl2sql_execution_accuracy

    graph = _entity_aware_graph()
    cases = [
        {
            "question": "图谱中有哪些关系类型",
            "gold_sql": (
                "SELECT predicate, COUNT(*) AS edge_count FROM edges "
                "GROUP BY predicate ORDER BY edge_count DESC, predicate ASC"
            ),
        },
    ]
    gold_sql = cases[0]["gold_sql"]

    def translator(question, conn, _disease_names):
        return translate_question_to_sql_llm_only(
            question, conn, llm_config={"base_url": "http://test", "api_key": "k"}
        )

    with patch(
        "src.operators.analysis_ops.llm_nl2sql._llm_translate",
        return_value=gold_sql,
    ):
        report = evaluate_nl2sql_execution_accuracy(cases, graph, translator=translator)

    assert report["per_translator"]["llm"]["accuracy"] == 1.0
