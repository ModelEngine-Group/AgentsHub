"""基础层测试：知识图谱 Schema 校验 / 存储查询 / 问答 / 评测。"""

import pytest

from dki.kg import KnowledgeGraph, validate_triple
from dki.kg.eval import evaluate_triples
from dki.kg.qa import answer_question


@pytest.fixture()
def kg():
    g = KnowledgeGraph()
    g.add_triple("高血压", "Disease", "HAS_SYMPTOM", "头痛", "Symptom", 0.9, "d1", "主诉头痛")
    g.add_triple("高血压", "Disease", "HAS_SYMPTOM", "头晕", "Symptom", 0.8, "d1")
    g.add_triple("高血压", "Disease", "TREATED_BY_DRUG", "硝苯地平", "Drug", 0.9, "d1", "给予硝苯地平口服")
    g.add_triple("高血压", "Disease", "HAS_COMPLICATION", "脑卒中", "Disease", 0.7, "d2")
    g.add_triple("高血压", "Disease", "BELONGS_TO_DEPT", "心血管内科", "Department", 0.9, "d1")
    g.add_triple("硝苯地平", "Drug", "CONTRAINDICATED_FOR", "低血压", "Disease", 0.8, "d3")
    g.entities["高血压"]["aliases"].append("高血压病")
    return g


def test_schema_validation():
    ok, _ = validate_triple("糖尿病", "Disease", "HAS_SYMPTOM", "多饮", "Symptom")
    assert ok
    ok, reason = validate_triple("糖尿病", "Disease", "HAS_SYMPTOM", "二甲双胍", "Drug")
    assert not ok and "Symptom" in reason
    ok, reason = validate_triple("糖尿病", "Disease", "HAS_COMPLICATION", "糖尿病", "Disease")
    assert not ok and "自环" in reason


def test_confidence_merge_noisy_or(kg):
    kg.add_triple("高血压", "Disease", "HAS_SYMPTOM", "头痛", "Symptom", 0.5, "d9")
    t = [t for t in kg.triples if t["tail"] == "头痛"][0]
    assert t["count"] == 2 and t["confidence"] > 0.9
    assert "d9" in t["sources"]


def test_alias_resolution_and_neighbors(kg):
    ns = kg.neighbors("高血压病", relation="HAS_SYMPTOM")
    assert {n["entity"] for n in ns} == {"头痛", "头晕"}


def test_reverse_direction(kg):
    ns = kg.neighbors("头痛", direction="in")
    assert ns and ns[0]["entity"] == "高血压"


def test_subgraph_and_path(kg):
    sub = kg.subgraph(["高血压"], hops=1)
    assert "硝苯地平" in sub["entities"] and len(sub["triples"]) >= 4
    path = kg.find_path("头痛", "硝苯地平", max_hops=3)
    assert path, "应能通过 高血压 中转找到路径"


def test_save_load_roundtrip(kg, tmp_path):
    p = kg.save(tmp_path / "kg.json")
    kg2 = KnowledgeGraph.load(p)
    assert kg2.stats() == kg.stats()
    assert kg2.neighbors("高血压病")  # 别名仍可解析


def test_echarts_export(kg):
    g = kg.to_echarts_graph()
    assert g["nodes"] and g["links"] and g["categories"]
    names = {n["name"] for n in g["nodes"]}
    assert "高血压" in names


def test_qa_templates(kg):
    r = answer_question(kg, "高血压有哪些症状？")
    assert r["answered"] and "头痛" in r["answer"]
    r = answer_question(kg, "高血压病用什么药？")  # 别名问法
    assert r["answered"] and "硝苯地平" in r["answer"]
    r = answer_question(kg, "高血压有什么并发症？")
    assert r["answered"] and "脑卒中" in r["answer"]
    r = answer_question(kg, "低血压不能用什么药")
    assert r["answered"] and "硝苯地平" in r["answer"]
    r = answer_question(kg, "头痛和硝苯地平有什么关系？")
    assert r["answered"] and "高血压" in r["answer"]
    r = answer_question(kg, "不知所云的问题")
    assert not r["answered"]


def test_triple_evaluation(tmp_path):
    gold_dir = tmp_path / "triples"
    gold_dir.mkdir()
    import json

    (gold_dir / "emr_0001.json").write_text(
        json.dumps(
            [
                {"head": "高血压", "head_type": "Disease", "relation": "HAS_SYMPTOM",
                 "tail": "头痛", "tail_type": "Symptom"},
                {"head": "高血压", "head_type": "Disease", "relation": "TREATED_BY_DRUG",
                 "tail": "硝苯地平", "tail_type": "Drug"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    preds = {
        "emr_0001": [
            {"head": "高血压", "relation": "HAS_SYMPTOM", "tail": "头痛"},   # TP
            {"head": "高血压", "relation": "HAS_SYMPTOM", "tail": "咳嗽"},   # FP
        ]
    }
    res = evaluate_triples(preds, gold_dir=gold_dir)
    assert res["tp"] == 1 and res["fp"] == 1 and res["fn"] == 1
    assert res["precision"] == 0.5 and res["recall"] == 0.5
