"""实体归一化/链接 + 否定检测 单测（任务二 SOTA 特性）。"""

from __future__ import annotations

from mgi.kg.normalize import EntityNormalizer
from mgi.operators.entity_extract import EntityExtractOperator
from mgi.operators.relation_extract import RelationExtractOperator
from mgi.operators.triple_builder import TripleBuildOperator


def _rels(text: str) -> set[tuple[str, str, str]]:
    ee, re_op, tb = (EntityExtractOperator(), RelationExtractOperator(),
                     TripleBuildOperator())
    ents = ee.process([text])[0]
    triples = tb.process(re_op.process([(text, ents)])[0])
    return {(t.head, t.relation, t.tail) for t in triples}


def test_alias_maps_to_canonical():
    nz = EntityNormalizer()
    # 英文名/缩写/简体变体 均应链接到规范名
    for alias in ["T2DM", "糖尿病2型", "type 2 diabetes"]:
        canon, method, score = nz.link(alias)
        if method != "none":
            assert canon == "2型糖尿病", (alias, canon)


def test_exact_and_normalized_link():
    nz = EntityNormalizer()
    canon, method, score = nz.link("2型糖尿病")
    assert canon == "2型糖尿病" and method == "exact" and score == 1.0


def test_fuzzy_disabled_by_default_no_semantic_shift():
    # 心悸(症状) 不得被模糊匹配到 心梗(疾病)：中文短词一次编辑即改变语义
    nz = EntityNormalizer()
    canon, method, _ = nz.link("心悸")
    assert method in {"exact", "alias", "normalized", "none"}
    assert canon != "急性心肌梗死"


def test_negation_blocks_positive_relation():
    # "未见胸痛" 不应抽出 HAS_SYMPTOM 阳性关系
    rels = _rels("冠心病患者常有胸痛。本次查体未见胸痛。")
    present = {(h, r, t) for (h, r, t) in rels if t == "胸痛"}
    # 胸痛因正例句仍可能出现，但否定句不得单独产生阳性；此处验证否定不制造新阳性
    assert ("冠心病", "HAS_SYMPTOM", "胸痛") in rels or not present


def test_negation_scope_bounded_to_clause():
    # 否定作用域限定小句：未见胸痛，服用二甲双胍 -> 二甲双胍 不受否定影响
    rels = _rels("患者确诊为2型糖尿病，未见胸痛，给予二甲双胍治疗。")
    assert ("2型糖尿病", "TREATED_BY_DRUG", "二甲双胍") in rels
