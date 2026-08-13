"""弱标注器测试：BIO 合法性、数据集构建（纯 python，不依赖 torch）。"""

import pytest

from dki.nlp.labels import KNOWN_TYPES, LABELS, NUM_LABELS
from dki.nlp.weak_label import label_one, spans_to_bio


def test_label_space():
    assert NUM_LABELS == 13
    assert LABELS[0] == "O"
    # 6 类实体各有 B-/I-
    for etype in KNOWN_TYPES:
        assert f"B-{etype}" in LABELS
        assert f"I-{etype}" in LABELS


def test_spans_to_bio_basic():
    text = "冠心病患者"
    # 冠心病 [0,3) 是 Disease
    tags = spans_to_bio(text, [[0, 3, "Disease"]])
    assert tags == ["B-Disease", "I-Disease", "I-Disease", "O", "O"]


def test_spans_to_bio_no_orphan_i():
    text = "患者有高血压和糖尿病"
    spans = [[3, 6, "Disease"], [7, 10, "Disease"]]
    tags = spans_to_bio(text, spans)
    prev = "O"
    for t in tags:
        if t.startswith("I-"):
            assert prev == t or prev == "B-" + t[2:], f"孤立 I- 标签: {prev}->{t}"
        prev = t


def test_spans_to_bio_rejects_bad_type():
    with pytest.raises(ValueError):
        spans_to_bio("测试", [[0, 2, "NotAType"]])


def test_label_one_on_real_dict():
    """用词典对含实体的文本做弱标注，应产出合法 span（类型∈6类、start<end）。"""
    try:
        from dki.operators.medical_entity_extractor.process import load_vocab
    except Exception:
        pytest.skip("算子依赖不可用（DataMate/loguru 缺失）")
    vocab = load_vocab()
    text = "患者主诉胸痛，诊断冠心病，给予阿司匹林口服。"
    rec = label_one(text, vocab)
    assert "spans" in rec
    for s, e, etype in rec["spans"]:
        assert 0 <= s < e <= len(text)
        assert etype in KNOWN_TYPES
    # 至少应命中冠心病(Disease) 与 阿司匹林(Drug)
    types = {t for _, _, t in rec["spans"]}
    assert "Disease" in types
