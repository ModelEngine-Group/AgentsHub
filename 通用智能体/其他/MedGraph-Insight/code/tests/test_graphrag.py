"""GraphRAG 语义检索问答 单测。"""

from __future__ import annotations

import pytest

from mgi.agents.kg_agent import KGBuildAgent
from mgi.kg.graphrag import GraphRAG, TfidfIndex


@pytest.fixture(scope="module")
def rag() -> GraphRAG:
    kg = KGBuildAgent().build_from_corpus("data/corpus/medical_notes.jsonl").result
    return GraphRAG(kg)


def test_tfidf_ranks_relevant_doc_first():
    idx = TfidfIndex(["糖尿病 多饮 多尿 二甲双胍", "冠心病 胸痛 心电图", "痛风 关节疼痛"])
    hits = idx.query("血糖高 多尿", k=1)
    assert hits and hits[0][0] == 0


def test_alias_english_links_to_canonical(rag: GraphRAG):
    # 英文缩写 T2DM 应经别名归一化链接到规范名
    r = rag.answer("T2DM有哪些症状？")
    assert r["entity"] == "2型糖尿病"
    assert "多尿" in r["answer"]


def test_semantic_link_without_exact_mention(rag: GraphRAG):
    # 问题未含精确实体串，靠语义检索/别名命中
    ent, score, method = rag.link_entity("血糖控制不佳应该吃什么降糖药")
    assert ent == "2型糖尿病"
    assert method in {"alias", "semantic"}


def test_treatment_intent_unions_drug_and_therapy(rag: GraphRAG):
    r = rag.answer("得了痛风怎么治疗？")
    assert r["entity"] == "痛风"
    # 药物（别嘌醇等）应包含在治疗答案中
    assert "别嘌醇" in r["answer"]


def test_evidence_is_grounded(rag: GraphRAG):
    r = rag.answer("冠心病需要做哪些检查？")
    assert r["relation"] == "DIAGNOSED_BY"
    assert r["evidence"] and any(e.get("evidence") for e in r["evidence"])


def test_multi_hop_complication_chain(rag: GraphRAG):
    r = rag.answer("2型糖尿病的并发症有哪些症状？")
    assert r["hops"] == 2
