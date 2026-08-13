"""实体级评测测试：P/R/F1 数学正确性（纯 python）。"""

from dki.nlp.eval_ner import prf_from_pairs


def test_perfect_match():
    gold = [("高血压", "Disease"), ("头痛", "Symptom")]
    r = prf_from_pairs(gold, gold)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0
    assert r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 0


def test_partial_with_fp_and_fn():
    pred = [("高血压", "Disease"), ("头痛", "Symptom"), ("误报", "Drug")]
    gold = [("高血压", "Disease"), ("头晕", "Symptom")]
    r = prf_from_pairs(pred, gold)
    assert r["tp"] == 1  # 只有高血压对
    assert r["fp"] == 2  # 头痛、误报
    assert r["fn"] == 1  # 头晕漏
    # precision/recall 结果四舍五入到 4 位小数
    assert abs(r["precision"] - 1 / 3) < 1e-4
    assert abs(r["recall"] - 1 / 2) < 1e-4


def test_type_sensitivity():
    """同名不同类型算不同实体。"""
    pred = [("胃炎", "Symptom")]
    gold = [("胃炎", "Disease")]
    r = prf_from_pairs(pred, gold)
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 1


def test_empty_prediction():
    r = prf_from_pairs([], [("高血压", "Disease")])
    assert r["tp"] == 0 and r["recall"] == 0.0 and r["f1"] == 0.0
