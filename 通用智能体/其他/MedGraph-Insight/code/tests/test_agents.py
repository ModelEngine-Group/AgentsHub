"""智能体与端到端流程测试：建图、问答、分析、NL2SQL 准确率。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mgi.agents.data_agent import DataProcessingAgent
from mgi.agents.kg_agent import KGBuildAgent, KGQAAgent
from mgi.agents.analysis_agent import AnalysisAgent
from mgi.analysis.nl2sql import NL2SQL, eval_accuracy
from mgi.kg import MedicalKG

ROOT = Path(__file__).resolve().parents[1]
CORPUS = str(ROOT / "data" / "corpus" / "medical_notes.jsonl")
EVAL = str(ROOT / "data" / "eval" / "nl2sql_eval.jsonl")


@pytest.fixture(scope="module")
def built_kg(tmp_path_factory) -> MedicalKG:
    agent = KGBuildAgent()
    agent.build_from_corpus(CORPUS)
    return agent.kg


def test_data_agent_pipeline(tmp_path) -> None:
    out = tmp_path / "cleaned.jsonl"
    agent = DataProcessingAgent()
    run = agent.process_file("清洗医疗文本：去HTML、繁简转换、脱敏、去重", CORPUS, str(out))
    assert out.exists()
    assert all(r.status == "ok" for r in run.records)
    assert len(run.plan) >= 1


def test_kg_build_produces_graph(built_kg) -> None:
    stats = built_kg.stats()
    assert stats["num_entities"] > 50
    assert stats["num_relations"] > 50
    assert "HAS_SYMPTOM" in stats["relation_types"]


def test_kg_qa_symptom(built_kg) -> None:
    qa = KGQAAgent(built_kg)
    r = qa.answer("2型糖尿病有哪些症状？")
    assert "多尿" in r["answer"] or "多饮" in r["answer"]


def test_kg_qa_multiple_intents(built_kg) -> None:
    qa = KGQAAgent(built_kg)
    cases = [
        ("高血压吃什么药？", "药"),
        ("冠心病需要做哪些检查？", "检查"),
        ("脑梗死怎么治疗？", "治疗"),
        ("痛风好发于哪些人群？", "人群"),
    ]
    for q, _ in cases:
        r = qa.answer(q)
        assert r["answer"] and "暂无" not in r["answer"], f"未答出: {q} -> {r['answer']}"


def test_nl2sql_accuracy_above_threshold(built_kg, tmp_path) -> None:
    db = tmp_path / "kg.sqlite"
    built_kg.to_sqlite(db)
    report = eval_accuracy(NL2SQL(db), EVAL)
    assert report["accuracy"] >= 0.95, f"NL2SQL 准确率不足: {report['accuracy']}"


def test_nl2sql_alias_linking(built_kg, tmp_path) -> None:
    db = tmp_path / "kg.sqlite"
    built_kg.to_sqlite(db)
    sql = NL2SQL(db).to_sql("T2DM有哪些症状")
    assert "2型糖尿病" in sql and "HAS_SYMPTOM" in sql


def test_nl2sql_multi_condition_intersect(built_kg, tmp_path) -> None:
    db = tmp_path / "kg.sqlite"
    built_kg.to_sqlite(db)
    r = NL2SQL(db).query("既有胸痛又有心悸的疾病有哪些")
    assert "INTERSECT" in r["sql"]
    assert ("冠心病",) in r["rows"]


def test_nl2sql_aggregation_having(built_kg, tmp_path) -> None:
    db = tmp_path / "kg.sqlite"
    built_kg.to_sqlite(db)
    sql = NL2SQL(db).to_sql("症状超过4个的疾病")
    assert "HAVING" in sql and "cnt > 4" in sql


def test_nl2sql_confidence_filter(built_kg, tmp_path) -> None:
    db = tmp_path / "kg.sqlite"
    built_kg.to_sqlite(db)
    sql = NL2SQL(db).to_sql("置信度大于0.85的用药关系")
    assert "confidence >= 0.85" in sql and "TREATED_BY_DRUG" in sql


def test_nl2sql_join_type(built_kg, tmp_path) -> None:
    db = tmp_path / "kg.sqlite"
    built_kg.to_sqlite(db)
    sql = NL2SQL(db).to_sql("2型糖尿病的药物属于什么类型")
    assert "JOIN entities" in sql and "etype_zh" in sql


def test_nl2sql_reverse_lookup(built_kg, tmp_path) -> None:
    db = tmp_path / "kg.sqlite"
    built_kg.to_sqlite(db)
    r = NL2SQL(db).query("哪些疾病会并发心力衰竭")
    assert "tail = '心力衰竭'" in r["sql"]
    assert ("急性心肌梗死",) in r["rows"]


def test_analysis_agent_report(built_kg, tmp_path) -> None:
    graph = tmp_path / "kg.json"
    db = tmp_path / "kg.sqlite"
    report = tmp_path / "report.html"
    built_kg.save(graph)
    agent = AnalysisAgent(str(graph), str(db))
    run = agent.analyze("分析2型糖尿病相关知识并生成报告", out_path=str(report))
    assert report.exists()
    html = report.read_text(encoding="utf-8")
    assert html.strip().lower().startswith("<!doctype html")
    assert all(r.status == "ok" for r in run.records)
    # 报告包含新增的覆盖热力图与并发症关系网络两类可视化
    tools = {r.step.tool for r in run.records}
    assert "coverage_heatmap" in tools
    assert "complication_network" in tools
    assert "覆盖热力图" in "".join(r.summary for r in run.records)
