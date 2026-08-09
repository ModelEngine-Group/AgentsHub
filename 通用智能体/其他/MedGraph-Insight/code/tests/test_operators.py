"""算子单元测试：NER / 关系抽取 / 三元组构建，以及 NPU 后端一致性。"""

from __future__ import annotations

from mgi.operators.entity_extract import EntityExtractOperator
from mgi.operators.relation_extract import RelationExtractOperator
from mgi.operators.triple_builder import TripleBuildOperator
from mgi.operators.npu.ner_npu import EntityExtractNPUOperator


TEXT = "2型糖尿病表现为多饮、多尿，服用二甲双胍治疗。"


def test_entity_extract_ac_finds_terms() -> None:
    op = EntityExtractOperator()
    ents = op.process([TEXT])[0]
    names = {(e.text, e.etype) for e in ents}
    assert ("2型糖尿病", "Disease") in names
    assert ("多饮", "Symptom") in names
    assert ("二甲双胍", "Drug") in names


def test_entity_extract_ac_matches_naive() -> None:
    op = EntityExtractOperator()
    docs = [TEXT, "高血压患者服用氨氯地平，可行心电图检查。"]
    ac = [sorted((e.start, e.end, e.etype) for e in d) for d in op.process(docs, mode="ac")]
    naive = [sorted((e.start, e.end, e.etype) for e in d) for d in op.process(docs, mode="naive")]
    assert ac == naive


def test_entity_extract_longest_match() -> None:
    # "2型糖尿病" 应整体匹配，而非子串 "糖尿病"
    op = EntityExtractOperator()
    ents = op.process(["确诊为2型糖尿病"])[0]
    assert any(e.text == "2型糖尿病" for e in ents)


def test_npu_backend_matches_cpu() -> None:
    cpu = EntityExtractOperator()
    npu = EntityExtractNPUOperator()
    docs = [TEXT, "冠心病诊断需要冠状动脉造影和心电图。"]
    a = [sorted((e.start, e.end, e.text) for e in d) for d in cpu.process(docs, mode="ac")]
    b = [sorted((e.start, e.end, e.text) for e in d) for d in npu.process(docs)]
    assert a == b


def test_relation_extract_basic() -> None:
    ner = EntityExtractOperator()
    re_op = RelationExtractOperator()
    ents = ner.process([TEXT])[0]
    triples = re_op.process([(TEXT, ents)])[0]
    rels = {(t.head, t.relation, t.tail) for t in triples}
    assert ("2型糖尿病", "HAS_SYMPTOM", "多饮") in rels
    assert ("2型糖尿病", "TREATED_BY_DRUG", "二甲双胍") in rels


def test_relation_cross_sentence_topic() -> None:
    text = "冠心病典型症状为胸痛。诊断需要心电图和冠状动脉造影。"
    ner = EntityExtractOperator()
    re_op = RelationExtractOperator()
    ents = ner.process([text])[0]
    triples = re_op.process([(text, ents)])[0]
    rels = {(t.head, t.relation, t.tail) for t in triples}
    # 第二句无「冠心病」，需通过话题回退补全 DIAGNOSED_BY
    assert ("冠心病", "DIAGNOSED_BY", "心电图") in rels


def test_triple_build_dedup_and_schema() -> None:
    ner = EntityExtractOperator()
    re_op = RelationExtractOperator()
    tb = TripleBuildOperator()
    ents = ner.process([TEXT])[0]
    cand = re_op.process([(TEXT, ents)])[0]
    built = tb.process(cand + cand)  # 传入重复，验证去重
    keys = [(t.head, t.relation, t.tail) for t in built]
    assert len(keys) == len(set(keys))


def test_operator_timing_instrumented() -> None:
    op = EntityExtractOperator()
    result = op.run([TEXT])
    assert result.elapsed_ms >= 0
    assert result.throughput >= 0
    assert result.num_in == 1
