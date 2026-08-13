"""API parity: ``to_sql()`` and ``query()`` must make the same SQL decision."""

import csv

import pytest

from dki.analysis.database import build_database
from dki.analysis.nl2sql import CompileDisposition, CompileResult, NL2SQL
from dki.kg.store import KnowledgeGraph


@pytest.fixture(scope="module")
def parity_db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api_parity")
    patients = tmp / "patients.csv"
    visits = tmp / "visits.csv"
    with open(patients, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "name", "gender", "age", "region"])
        w.writerows([
            ["P1", "张三", "男", 65, "北京"],
            ["P2", "李四", "女", 30, "上海"],
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
        ])
    kg = KnowledgeGraph()
    kg.add_triple("高血压", "Disease", "HAS_SYMPTOM", "头痛", "Symptom", 0.9, "v1")
    db_path = tmp / "insight.db"
    build_database(db_path=db_path, visits_csv=visits, patients_csv=patients, kg=kg)
    return db_path


def _no_llm(monkeypatch):
    import dki.llm

    monkeypatch.setattr(dki.llm, "llm_available", lambda: False)


PARITY_QUESTIONS = (
    "按科室统计就诊量",
    "高血压的就诊次数",
    "统计每个地区的患者数",
    "费用最高的一次就诊",
)


@pytest.mark.parametrize("question", PARITY_QUESTIONS)
def test_rule_engine_parity_without_llm(parity_db, monkeypatch, question):
    _no_llm(monkeypatch)
    engine = NL2SQL(db_path=parity_db)

    sql_view = engine.to_sql(question)
    exec_view = engine.query(question)

    assert sql_view.get("sql", "") == exec_view.get("sql", "")
    assert sql_view.get("engine") == exec_view.get("engine")


@pytest.mark.parametrize(
    "strategy", ["query_plan", "query_plan_v2", "query_plan_v3"]
)
def test_structured_strategy_parity_without_llm(parity_db, monkeypatch, strategy):
    _no_llm(monkeypatch)
    engine = NL2SQL(
        db_path=parity_db,
        llm_strategy=strategy,
        agent_rule_policy="plan_first",
    )
    for question in PARITY_QUESTIONS:
        sql_view = engine.to_sql(question)
        exec_view = engine.query(question)
        assert sql_view.get("sql", "") == exec_view.get("sql", ""), question
        assert sql_view.get("engine") == exec_view.get("engine"), question


def test_blocked_rule_prefers_same_structured_candidate(parity_db, monkeypatch):
    """Both APIs must try the structured pipeline first on a BLOCKED rule."""

    engine = NL2SQL(
        db_path=parity_db,
        llm_strategy="query_plan_v2",
        agent_rule_policy="plan_first",
    )
    candidate = {
        "sql": "SELECT COUNT(*) FROM visits",
        "confidence": 0.9,
        "strategy": "query_plan_v2",
        "plan_version": 2,
        "execution_validated": True,
    }
    monkeypatch.setattr(
        engine,
        "_rule_sql_normalized_result",
        lambda q: CompileResult(CompileDisposition.BLOCKED),
    )
    monkeypatch.setattr(engine, "_llm_pipeline", lambda question: dict(candidate))

    sql_view = engine.to_sql("构造性 BLOCKED 问题")
    exec_view = engine.query("构造性 BLOCKED 问题")

    assert sql_view["sql"] == candidate["sql"]
    assert exec_view["sql"] == candidate["sql"]
    assert sql_view["engine"] == exec_view["engine"] == "llm"
    assert exec_view["validated"] is True
    assert exec_view["rows"] == [[3]]
