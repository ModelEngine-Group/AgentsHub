from __future__ import annotations

import json
from pathlib import Path

import pytest

from tool_server import analysis_tools, kg_tools


def test_analysis_file_and_question_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(analysis_tools, "resolve_path", lambda value: tmp_path / value)
    assert analysis_tools._safe_replace(tmp_path / "missing") == tmp_path / "missing"
    existing = tmp_path / "old"
    existing.write_text("old", encoding="utf-8")
    analysis_tools._safe_replace(existing)
    assert not existing.exists()
    base, path = analysis_tools._first_writable_output_dir(["one", "two"], ["a.json"])
    assert base == "one" and path.is_dir()
    assert analysis_tools._question_slug("问题") == analysis_tools._question_slug("问题")
    assert analysis_tools._extract_body_fragment("<html><body><b>x</b></body></html>") == "<b>x</b>"
    assert analysis_tools._extract_body_fragment("plain") == "plain"
    assert analysis_tools._extract_hba1c_abnormal_trend_months("最近 30 个月 HbA1c 异常人数趋势") == 24
    assert analysis_tools._extract_hba1c_abnormal_trend_months("最近半年 HbA1c 异常人数趋势") == 6
    assert analysis_tools._extract_hba1c_abnormal_trend_months("普通问题") is None


def test_analysis_json_and_planner_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    second.write_text('{"value":2}', encoding="utf-8")
    mapping = {"first": first, "second": second}
    monkeypatch.setattr(analysis_tools, "resolve_path", lambda value: mapping.get(value, tmp_path / value))
    assert analysis_tools._read_json_first(["first", "second"]) == {"value": 2}
    assert analysis_tools._read_json_first(["first"]) == {}

    monkeypatch.setattr(analysis_tools, "load_server_config", lambda: {"paths": {"indicator_results": "missing"}})
    assert analysis_tools._load_indicator_items() == []

    planner_dir = tmp_path / "planner"
    monkeypatch.setattr(analysis_tools, "PLANNER_LOG_DIR", "planner")
    monkeypatch.setattr(analysis_tools, "resolve_path", lambda value: tmp_path / value)
    result = analysis_tools._write_planner_log("测试问题", {"intent": "test"}, {"entry": "unit"})
    assert result.endswith(".json")
    payload = json.loads(next(planner_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["entry"] == "unit"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("高血压和糖尿病平均 hba1c", "高血压合并糖尿病患者的平均 HbA1c 是多少？"),
        ("糖尿病空腹血糖均值", "糖尿病患者的空腹血糖平均值是多少？"),
        ("高脂血症 LDL-C 异常比例", "高脂血症患者的 LDL-C 异常比例是多少？"),
        ("BMI 超标人数是多少", "BMI 超标患者有多少人？"),
        ("最近6个月血压异常人数趋势", "最近 6 个月血压异常人数趋势如何？"),
        (" 原始   问题 ", "原始 问题"),
    ],
)
def test_normalize_controlled_questions(question: str, expected: str) -> None:
    assert analysis_tools._normalize_controlled_question(question) == expected


def test_disease_inventory_and_combinations(monkeypatch: pytest.MonkeyPatch) -> None:
    source = [
        {"patient_id": "P1", "disease_tags": "hypertension;diabetes;hypertension"},
        {"patient_id": "P2", "disease_tags": "hypertension;obesity"},
        {"patient_id": "P3", "disease_tags": "nan;custom"},
    ]
    monkeypatch.setattr(analysis_tools, "fetch_rows", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(analysis_tools, "fetch_one", lambda *_args, **_kwargs: {"patient_count": 4})
    inventory = analysis_tools._collect_disease_inventory_rows()
    hypertension = next(row for row in inventory if row["疾病名称"] == "高血压")
    assert hypertension["患者人数"] == 2
    assert hypertension["占比"] == 0.5
    assert hypertension["英文标准Code"] == "hypertension"

    grouped = [
        {"disease_tags": "hypertension;diabetes;hypertension", "patient_count": 3},
        {"disease_tags": "obesity", "patient_count": 5},
        {"disease_tags": "nan;custom", "patient_count": 2},
    ]
    monkeypatch.setattr(analysis_tools, "fetch_rows", lambda *_args, **_kwargs: grouped)
    combos = analysis_tools._collect_disease_combination_rows(limit=1)
    assert combos == [
        {
            "疾病组合": "高血压 + 糖尿病",
            "患者人数": 3,
            "疾病标签数": 2,
            "统计口径": "精确多病组合",
        }
    ]
    lengths = analysis_tools._collect_exact_combo_length_rows()
    assert {row["疾病标签数"]: row["患者人数"] for row in lengths} == {1: 7, 2: 3}

    monkeypatch.setattr(analysis_tools, "fetch_rows", lambda *_args, **_kwargs: source)
    pairs = analysis_tools._collect_disease_pairwise_combination_rows(limit=2)
    assert len(pairs) == 2
    assert all(row["统计口径"] == "两两共现" for row in pairs)


def test_analysis_intent_detection_and_translation() -> None:
    assert analysis_tools._looks_like_disease_inventory_question("当前常见疾病有哪些")
    assert not analysis_tools._looks_like_disease_inventory_question("不同疾病组合有哪些")
    assert not analysis_tools._looks_like_disease_inventory_question("风险等级分布")
    assert analysis_tools._looks_like_disease_combination_question("多病共病组合")
    assert analysis_tools._looks_like_cohort_disease_question("这些患者有什么疾病")
    assert analysis_tools._looks_like_cohort_risk_question("这些患者风险等级分布")
    assert analysis_tools._contains_explicit_disease_name("高血压患者")
    assert not analysis_tools._contains_explicit_disease_name("这些患者")
    assert analysis_tools._display_disease_label("hypertension") == "高血压"
    assert analysis_tools._display_disease_label("new_disease") == "new disease"
    assert analysis_tools._display_disease_label("nan") == ""
    assert analysis_tools._translate_disease_tags("hypertension;diabetes;nan") == "高血压、糖尿病"
    assert analysis_tools._humanize_row({"gender": "male", "disease_tags": "hypertension"})["性别"] == "男"
    assert analysis_tools._first_metric({"name": "x", "count": 3}) == ("count", 3)
    assert analysis_tools._first_metric({"name": "x"}) == (None, None)


def test_kg_query_rows_and_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    assert kg_tools._query_patient_ids_by_diseases([]) == []
    assert kg_tools._top_indicator_rows([]) == []
    assert kg_tools._top_risk_event_rows([]) == []
    assert kg_tools._top_drug_rows([]) == []
    assert kg_tools._cohort_disease_distribution([]) == []

    captured: list[tuple[str, list[str]]] = []

    def rows(sql: str, params: list[str]) -> list[dict]:
        captured.append((sql, params))
        if "lab_result" in sql:
            return [{"indicator": "hba1c", "patient_count": 2, "record_count": 4}]
        if "risk_event" in sql:
            return [{"event_type": "blood_pressure_high", "patient_count": 2, "record_count": 3}]
        if "medication_record" in sql:
            return [{"drug_name": "metformin", "patient_count": 1, "record_count": 2}]
        if "SELECT patient_id" in sql:
            return [{"patient_id": "P1"}, {"patient_id": "P2"}]
        if "disease_tags" in sql:
            return [{"disease_tags": "hypertension;diabetes"}, {"disease_tags": "hypertension"}]
        return [{"patient_id": "P1"}, {"patient_id": "P2"}]

    monkeypatch.setattr(kg_tools, "fetch_rows", rows)
    assert kg_tools._query_patient_ids_by_diseases(["hypertension"]) == ["P1", "P2"]
    assert kg_tools._query_patient_ids_by_risk_level("high") == ["P1", "P2"]
    indicators = kg_tools._top_indicator_rows(["P1", "P2"], limit=None)
    risks = kg_tools._top_risk_event_rows(["P1"], limit=5)
    drugs = kg_tools._top_drug_rows(["P1"])
    assert indicators[0]["display_name"] == "糖化血红蛋白"
    assert risks[0]["display_name"] == "血压偏高"
    assert drugs[0]["display_name"] == "二甲双胍"
    distribution = kg_tools._cohort_disease_distribution(["P1", "P2"])
    assert distribution[0] == {"disease": "hypertension", "patient_count": 2}
    assert any("LIMIT 5" in sql for sql, _params in captured)


def test_kg_detail_tables_and_aliases() -> None:
    indicators = [{"indicator": "hba1c", "display_name": "糖化血红蛋白", "patient_count": 2, "record_count": 4}]
    drugs = [{"drug_name": "metformin", "display_name": "二甲双胍", "patient_count": 1, "record_count": 2}]
    risks = [{"event_type": "blood_pressure_high", "display_name": "血压偏高", "patient_count": 2, "record_count": 3}]
    assert kg_tools._kg_detail_table_for_intent("检查指标", indicators, drugs, risks)["kind"] == "indicator"
    assert kg_tools._kg_detail_table_for_intent("相关药物", indicators, drugs, risks)["kind"] == "drug"
    assert kg_tools._kg_detail_table_for_intent("风险事件", indicators, drugs, risks)["kind"] == "risk_event"
    combined = kg_tools._kg_detail_table_for_intent("指标、药物和风险事件", indicators, drugs, risks)
    assert combined["kind"] == "combined" and combined["row_count"] == 3
    assert kg_tools._sample_patient_ids(["P3", "P1", "P2"], 2) == ["P1", "P2"]

    nodes = {
        "Disease::hypertension": {"type": "Disease", "display_name": "hypertension"},
        "Indicator::hba1c": {"type": "Indicator", "display_name": "hba1c"},
        "Patient::P0001": {"type": "Patient", "display_name": "P0001"},
    }
    found = kg_tools._normalize_entity_ids("患者 P0001 高血压和 HbA1c", nodes)
    assert set(found) == {"Disease::hypertension", "Indicator::hba1c", "Patient::P0001"}
    assert kg_tools._label_for("Disease::hypertension", nodes) == "高血压"
    assert kg_tools._label_for("Indicator::hba1c", nodes).startswith("指标")
    synthetic = kg_tools._synthetic_query_subgraph_payload("请生成未知疾病的知识图谱子图")
    assert synthetic["status"] == "success" and synthetic["edge_count"] == 0
