"""质量增强 NER 数据、跨度评测和解码回归测试。"""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


@pytest.fixture(autouse=True)
def _scripts_on_path(monkeypatch):
    import sys

    monkeypatch.syspath_prepend(str(SCRIPTS))
    sys.modules.pop("build_quality_data", None)
    sys.modules.pop("train_ner_quality", None)


def test_validate_quality_rows_rejects_test_overlap():
    from build_quality_data import _text_hash, _validate_rows

    rows = [{
        "doc_id": "quality_0000",
        "text": "患者最近没睡好。",
        "entities": [{"mention": "没睡好", "name": "失眠", "type": "Symptom"}],
    }]
    with pytest.raises(ValueError, match="最终 OOD 测试文本重合"):
        _validate_rows(rows, {_text_hash(rows[0]["text"])})


def test_validate_quality_rows_rejects_missing_mention():
    from build_quality_data import _validate_rows

    rows = [{
        "doc_id": "quality_0000",
        "text": "患者最近头晕。",
        "entities": [{"mention": "没睡好", "name": "失眠", "type": "Symptom"}],
    }]
    with pytest.raises(ValueError, match="非法实体标注"):
        _validate_rows(rows, set())


def test_mentions_to_spans_prefers_longest_non_overlapping():
    from train_ner_quality import _mentions_to_spans

    row = {
        "doc_id": "q",
        "text": "最近胸口发闷。",
        "entities": [
            {"mention": "胸口发闷", "type": "Symptom"},
            {"mention": "发闷", "type": "Symptom"},
        ],
    }
    assert _mentions_to_spans(row)["spans"] == [[2, 6, "Symptom"]]


def test_span_prf_requires_overlap_and_type():
    from train_ner_quality import _span_prf

    gold = [{"spans": [[2, 6, "Symptom"], [8, 11, "Disease"]]}]
    pred = [[
        {"span": [3, 6], "type": "Symptom"},
        {"span": [8, 11], "type": "Drug"},
    ]]
    result = _span_prf(pred, gold)
    assert {key: result[key] for key in ("precision", "recall", "f1", "tp", "fp", "fn")} == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "tp": 1,
        "fp": 1,
        "fn": 1,
    }
    assert result["by_type"]["Symptom"]["f1"] == 1.0
    assert result["by_type"]["Drug"]["fp"] == 1
    assert result["by_type"]["Disease"]["fn"] == 1


def test_quality_selection_score_rewards_macro_types():
    from train_ner_quality import _selection_score

    weak_macro = {"f1": 0.8, "recall": 0.8, "by_type": {"Symptom": {"f1": 0.9}, "Exam": {"f1": 0.3}}}
    strong_macro = {"f1": 0.8, "recall": 0.8, "by_type": {"Symptom": {"f1": 0.8}, "Exam": {"f1": 0.8}}}
    assert _selection_score(strong_macro, 0.1, 0.15) > _selection_score(weak_macro, 0.1, 0.15)


def test_decode_bio_returns_character_span():
    from train_ner import _decode_bio

    text = "患者胸闷明显"
    offsets = [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 0)]
    pred = [0, 0, 0, 1, 2, 0, 0, 0]
    labels = {0: "O", 1: "B-Symptom", 2: "I-Symptom"}
    entities = _decode_bio(text, offsets, pred, labels, {})
    assert entities == [{
        "name": "胸闷",
        "type": "Symptom",
        "mention": "胸闷",
        "span": [2, 4],
    }]
