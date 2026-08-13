"""分析层测试：建库 / SQL 安全 / NL2SQL / 指标 / BI 报告。"""

import csv
import json

import pytest

from dki.analysis.bi import ReportBuilder, _md_to_html, build_default_report
from dki.analysis.database import build_database, run_sql
from dki.analysis.metrics import overview_stats, standard_metrics
from dki.analysis.nl2sql import NL2SQL
from dki.kg.store import KnowledgeGraph


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("analysis")
    patients = tmp / "patients.csv"
    visits = tmp / "visits.csv"
    with open(patients, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "name", "gender", "age", "region"])
        w.writerows([
            ["P1", "张三", "男", 65, "北京"], ["P2", "李四", "女", 30, "上海"],
            ["P3", "王五", "男", 50, "北京"],
        ])
    with open(visits, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["visit_id", "patient_id", "visit_date", "department", "disease",
                    "drug_prescribed", "exam", "cost", "los_days"])
        w.writerows([
            ["v1", "P1", "2026-01-10", "心血管内科", "高血压", "硝苯地平", "血压测量", 1200.5, 0],
            ["v2", "P1", "2026-02-11", "心血管内科", "冠心病", "阿司匹林", "心电图", 8000.0, 5],
            ["v3", "P2", "2025-12-01", "内分泌科", "2型糖尿病", "二甲双胍", "空腹血糖检测", 900.0, 0],
            ["v4", "P3", "2026-03-05", "内分泌科", "2型糖尿病", "胰岛素", "糖化血红蛋白检测", 1500.0, 3],
            ["v5", "P2", "2026-04-02", "呼吸内科", "肺炎", "左氧氟沙星", "胸部CT", 2200.0, 7],
        ])
    kg = KnowledgeGraph()
    kg.add_triple("高血压", "Disease", "HAS_SYMPTOM", "头痛", "Symptom", 0.9, "v1")
    kg.add_triple("高血压", "Disease", "TREATED_BY_DRUG", "硝苯地平", "Drug", 0.9, "v1")
    db_path = tmp / "insight.db"
    build_database(db_path=db_path, visits_csv=visits, patients_csv=patients, kg=kg)
    return db_path


def test_sql_whitelist(db):
    with pytest.raises(PermissionError):
        run_sql("UPDATE visits SET cost=0", db)
    with pytest.raises(PermissionError):
        run_sql("SELECT 1; SELECT 2", db)
    with pytest.raises(PermissionError):
        run_sql("PRAGMA table_info(visits)", db)
    cols, rows = run_sql("SELECT COUNT(*) FROM visits", db)
    assert rows[0][0] == 5


def test_nl2sql_basic(db):
    nl = NL2SQL(db_path=db)
    r = nl.query("按科室统计就诊量")
    assert r["engine"] == "rule"
    assert dict((a, b) for a, b in r["rows"]) == {"心血管内科": 2, "内分泌科": 2, "呼吸内科": 1}
    r = nl.query("2型糖尿病的就诊次数")
    assert r["rows"][0][0] == 2
    r = nl.query("男性患者的就诊次数")
    assert r["rows"][0][0] == 3
    r = nl.query("图谱里高血压有哪些症状")
    assert r["rows"][0][0] == "头痛"
    assert r["columns"] == ["tail"]


def test_nl2sql_unparseable_no_llm(db, monkeypatch):
    import dki.paths as paths

    monkeypatch.setattr(paths, "LLM_API_KEY", "")
    nl = NL2SQL(db_path=db)
    r = nl.query("今天天气怎么样")
    assert r.get("error") and not r["rows"]


def test_standard_metrics_and_overview(db):
    ms = standard_metrics(db)
    keys = {m["key"] for m in ms}
    assert {"dept_visits", "top_diseases", "monthly_trend"} <= keys
    ov = overview_stats(db)
    assert ov["就诊记录"] == 5 and ov["患者数"] == 3


def test_report_builder(db, tmp_path):
    b = ReportBuilder()
    b.set_overview({"就诊记录": 5})
    b.add_chart("bar", "测试图", {"categories": ["a", "b"], "series": [{"name": "n", "values": [1, 2]}]},
                "a为1，b为2。")
    b.add_markdown("## 结论\n- **测试**通过")
    out = b.render("测试报告", out_path=tmp_path / "r.html")
    html = out.read_text(encoding="utf-8")
    assert "echarts" in html and "测试图" in html and "<strong>测试</strong>" in html


def test_build_default_report(db, tmp_path):
    kg = KnowledgeGraph()
    kg.add_triple("高血压", "Disease", "HAS_SYMPTOM", "头痛", "Symptom", 0.9, "v1")
    out = build_default_report(db, kg=kg, out_path=tmp_path / "full.html")
    html = out.read_text(encoding="utf-8")
    assert "医疗数据洞察报告" in html and '"type": "graph"' in html


def test_markdown_renderer_escapes_input_without_shadowing_html_module():
    rendered = _md_to_html("## 标题\n- **安全** <script>alert(1)</script>")

    assert "<h2>标题</h2>" in rendered
    assert "<strong>安全</strong>" in rendered
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
