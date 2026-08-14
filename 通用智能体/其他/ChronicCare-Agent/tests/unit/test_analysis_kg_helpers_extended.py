from __future__ import annotations

import base64
from pathlib import Path

import pytest

from tool_server import analysis_tools, kg_tools


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (" 高血压 合并 糖尿病患者 HbA1c 均值 ", "高血压合并糖尿病患者的平均 HbA1c 是多少？"),
        ("糖尿病患者空腹血糖平均是多少", "糖尿病患者的空腹血糖平均值是多少？"),
        ("高脂血症患者 LDL-C 异常比例", "高脂血症患者的 LDL-C 异常比例是多少？"),
        ("BMI 超标人数有多少", "BMI 超标患者有多少人？"),
        ("普通问题", "普通问题"),
    ],
)
def test_controlled_question_normalization(question: str, expected: str) -> None:
    assert analysis_tools._normalize_controlled_question(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("最近 6 个月 HbA1c 异常人数趋势", 6),
        ("最近99个月HbA1c异常人数趋势", 24),
        ("最近半年 HbA1c 异常人数趋势", 6),
        ("最近 3 个月 HbA1c 平均值", None),
    ],
)
def test_extract_hba1c_trend_months(question: str, expected: int | None) -> None:
    assert analysis_tools._extract_hba1c_abnormal_trend_months(question) == expected


def test_analysis_string_and_translation_helpers() -> None:
    assert len(analysis_tools._question_slug("同一问题")) == 12
    assert analysis_tools._extract_body_fragment("<html><body><b>正文</b></body></html>") == "<b>正文</b>"
    assert analysis_tools._extract_body_fragment("无 body") == "无 body"
    assert analysis_tools._display_disease_label("hypertension") == "高血压"
    assert analysis_tools._display_disease_label("custom_disease") == "custom disease"
    assert analysis_tools._display_disease_label("nan") == ""
    assert analysis_tools._normalize_disease_label("diabetes") == "糖尿病"
    assert analysis_tools._normalize_disease_label("糖尿病") == "糖尿病"
    assert analysis_tools._translate_scalar("female") == "女"
    assert analysis_tools._translate_scalar(3) == 3
    assert analysis_tools._translate_disease_tags("hypertension;diabetes;nan") == "高血压、糖尿病"
    assert analysis_tools._extract_patient_id("查询 p0042 的路径") == "P0042"
    assert analysis_tools._extract_patient_id("无患者编号") is None
    assert analysis_tools._public_question_id("NLQ007") == "AQ007"
    assert analysis_tools._public_question_id("") == "AQ000"


def test_humanize_metric_and_distribution_helpers() -> None:
    row = analysis_tools._humanize_row({"gender": "male", "disease_tags": "hypertension;diabetes", "custom": 2})
    assert "男" in row.values()
    assert "高血压、糖尿病" in row.values()
    assert analysis_tools._first_metric({"name": "x", "count": 3}) == ("count", 3)
    assert analysis_tools._first_metric({"name": "x"}) == (None, None)
    metric = analysis_tools._metric_from_question("未知问题", {"patient_count": 9})
    assert metric["name"] == "patient_count"
    assert metric["value"] == 9
    assert analysis_tools._format_disease_distribution_text([]).startswith("当前未识别")
    text = analysis_tools._format_disease_distribution_text([{"疾病名称": "高血压", "患者人数": 10, "占比": 0.25}])
    assert text == "高血压（10人，25.0%）"


def test_analysis_html_and_svg_helpers_escape_and_render() -> None:
    bars = analysis_tools._bar_chart_html(
        "分布", [{"name": "A", "value": 3}, {"name": "B", "value": 1}], "name", "value"
    )
    cards = analysis_tools._metric_cards_html("指标", [{"label": "患者", "value": 20}])
    table = analysis_tools._html_table(["列"], [["值"]])
    month_table = analysis_tools._month_rows_to_table([{"month": "2026-07", "count": 2}], "count")
    line = analysis_tools._line_chart_svg(
        "趋势", "副标题", [{"month": "2026-06", "count": 1}, {"month": "2026-07", "count": 4}], "month", "count"
    )
    pie = analysis_tools._pie_chart_svg(
        "风险", "副标题", [{"risk": "high", "count": 2}, {"risk": "low", "count": 1}], "risk", "count"
    )
    single_pie = analysis_tools._pie_chart_svg("风险", "副标题", [{"risk": "high", "count": 2}], "risk", "count")
    horizontal = analysis_tools._bar_chart_svg("疾病", "副标题", [{"name": "<高血压>", "count": 3}], "name", "count")
    bundle = analysis_tools._chart_bundle_html(
        "综合", [{"label": "人数", "value": 3}], [{"title": "图表", "content": "<svg/>"}]
    )
    assert "A (3.0)" in bars
    assert "患者" in cards
    assert "<table" in table and "2026-07" in month_table
    assert all("<svg" in item for item in (line, pie, single_pie, horizontal))
    assert "&lt;高血压&gt;" in horizontal
    assert "综合" in bundle and "<svg/>" in bundle


def test_analysis_entry_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert analysis_tools._entry_button("打开", "") == ""
    assert analysis_tools._entry_meta("地址", "") == ""
    assert "linear-gradient" in analysis_tools._entry_button("打开", "/x", primary=True)
    monkeypatch.setattr(analysis_tools, "artifact_exists_for_route", lambda path: path == "/ok")
    assert analysis_tools._entry_row({"label": "缺失", "route_path": "/missing"}) == ""
    row = analysis_tools._entry_row(
        {"label": "报告", "title": "完整报告", "route_path": "/ok", "service_url": "/service"}
    )
    assert "完整报告" in row and "/service" in row


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("high", "高风险"), ("medium", "中风险"), ("urgent", "紧急"), ("unknown", "unknown"), ("", "未标注")],
)
def test_risk_level_labels(raw: str, expected: str) -> None:
    assert analysis_tools._risk_level_label(raw) == expected


def _nodes() -> dict[str, dict[str, str]]:
    return {
        "Patient::P0001": {"id": "Patient::P0001", "type": "Patient", "display_name": "P0001"},
        "Patient::P0002": {"id": "Patient::P0002", "type": "Patient", "display_name": "P0002"},
        "Disease::hypertension": {
            "id": "Disease::hypertension",
            "type": "Disease",
            "display_name": "hypertension",
        },
        "Indicator::hba1c": {"id": "Indicator::hba1c", "type": "Indicator", "display_name": "hba1c"},
        "Drug::metformin": {"id": "Drug::metformin", "type": "Drug", "display_name": "metformin"},
        "RiskEvent::stroke": {"id": "RiskEvent::stroke", "type": "RiskEvent", "display_name": "stroke"},
    }


def test_kg_topic_alias_and_entity_normalization() -> None:
    nodes = _nodes()
    assert kg_tools._topic_from_query("生成高血压的知识图谱子图") == "高血压"
    assert kg_tools._topic_from_query("") == "当前问题相关主题"
    payload = kg_tools._synthetic_query_subgraph_payload("画出未知疾病关系图")
    assert payload["status"] == "success" and payload["node_count"] == 1
    assert kg_tools._normalize_lookup_text(" HbA1c-Test ") == "hba1ctest"
    aliases = kg_tools._build_entity_alias_map(nodes)
    assert aliases[kg_tools._normalize_lookup_text("高血压")] == "Disease::hypertension"
    found = kg_tools._normalize_entity_ids("高血压患者 P0001 的 HbA1c 和二甲双胍", nodes)
    assert {
        "Disease::hypertension",
        "Patient::P0001",
        "Indicator::hba1c",
        "Drug::metformin",
    }.issubset(found)


def test_kg_labels_tables_and_intents() -> None:
    nodes = _nodes()
    assert kg_tools._label_for("Patient::P0001", nodes) == "患者 P0001"
    assert "高血压" in kg_tools._label_for("Disease::hypertension", nodes)
    assert "指标" in kg_tools._label_for("Indicator::hba1c", nodes)
    indicator_rows = [{"indicator": "hba1c", "display_name": "糖化血红蛋白", "patient_count": 2, "record_count": 4}]
    drug_rows = [{"drug_name": "metformin", "display_name": "二甲双胍", "value": 2, "record_count": 3}]
    risk_rows = [{"event_type": "stroke", "display_name": "脑卒中", "patient_count": 1, "record_count": 1}]
    assert kg_tools._indicator_table(indicator_rows)["rows"][0]["覆盖患者数"] == 2
    assert kg_tools._drug_table(drug_rows)["allowed_names"] == ["二甲双胍"]
    assert kg_tools._risk_event_table(risk_rows)["kind"] == "risk_event"
    combined = kg_tools._kg_detail_table_for_intent("检查指标、药物和风险事件", indicator_rows, drug_rows, risk_rows)
    assert combined["kind"] == "combined" and combined["row_count"] == 3
    assert kg_tools._kg_detail_table_for_intent("用药", indicator_rows, drug_rows, risk_rows)["kind"] == "drug"


def test_kg_graph_traversal_and_signatures() -> None:
    nodes = _nodes()
    edges = [
        {"source": "Disease::hypertension", "relation": "has_indicator", "target": "Indicator::hba1c"},
        {"source": "Patient::P0001", "relation": "has_disease", "target": "Disease::hypertension"},
        {"source": "Patient::P0002", "relation": "has_disease", "target": "Disease::hypertension"},
    ]
    outgoing = {
        "Disease::hypertension": [edges[0]],
        "Patient::P0001": [edges[1]],
        "Patient::P0002": [edges[2]],
    }
    incoming = {"Indicator::hba1c": [edges[0]], "Disease::hypertension": edges[1:]}
    selected, selected_edges = kg_tools._expand_frontier(
        ["Disease::hypertension"], outgoing, incoming, nodes, max_nodes=4
    )
    assert "Indicator::hba1c" in selected and selected_edges
    assert kg_tools._sample_patient_ids(["P3", "P1", "P2"], 2) == ["P1", "P2"]
    assert kg_tools._risk_level_entity_id("HIGH") == "RiskScore::high"
    assert kg_tools._cohort_query_signature("同时生成子图", ["Disease::hypertension", "Disease::diabetes"]).startswith(
        "cohort_subgraph"
    )
    assert kg_tools._node_type("Unknown::x", nodes) == "Unknown"
    assert kg_tools._edge_priority(edges[0], "Disease::hypertension", nodes)[0] == 1


def test_kg_preview_data_uri_and_html(tmp_path: Path) -> None:
    payload = {
        "seed_labels": ["<高血压>"],
        "cohort_patient_count": 10,
        "display_patient_node_count": 2,
        "semantic_node_count": 3,
        "node_count": 5,
        "edge_count": 4,
        "graph_scope_explanation": "<范围>",
    }
    svg = kg_tools._subgraph_preview_svg("高血压", payload)
    uri = kg_tools._subgraph_preview_data_uri("高血压", payload)
    decoded = base64.b64decode(uri.split(",", 1)[1]).decode("utf-8")
    assert "&lt;高血压&gt;" in svg and decoded == svg
    assert kg_tools._subgraph_id_from_query("高血压").startswith("subgraph_graph_query_")
    assert kg_tools._subgraph_id_from_query("x", signature="risk high") == "subgraph_risk_high"
    output = tmp_path / "subgraph.html"
    kg_tools._render_subgraph_html(
        output,
        "<查询>",
        list(_nodes().values()),
        [{"source": "Patient::P0001", "target": "Disease::hypertension", "relation": "has_disease"}],
        {"graph_scope_explanation": "<说明>"},
    )
    content = output.read_text(encoding="utf-8")
    assert "<svg" in content and "has_disease" in content and "高血压" in content
