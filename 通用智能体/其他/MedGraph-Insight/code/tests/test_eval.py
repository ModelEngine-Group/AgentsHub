"""评测框架单测：PRF1 计算 + 语料整体 NER/RE 指标门槛。"""

from __future__ import annotations

from mgi.eval.metrics import PRF1, prf1
from mgi.eval.runner import run_eval


def test_prf1_basic():
    r = prf1({("a", "X"), ("b", "Y")}, {("a", "X"), ("c", "Z")})
    assert (r.tp, r.fp, r.fn) == (1, 1, 1)
    assert abs(r.precision - 0.5) < 1e-9
    assert abs(r.recall - 0.5) < 1e-9
    assert abs(r.f1 - 0.5) < 1e-9


def test_prf1_empty():
    r = PRF1()
    assert r.precision == 0.0 and r.recall == 0.0 and r.f1 == 0.0


def test_corpus_metrics_meet_threshold():
    rep = run_eval(
        "data/corpus/medical_notes.jsonl",
        "data/eval/kg_gold.jsonl",
    )
    assert rep["docs"] >= 15
    # NER 实体级、RE 三元组级 F1 需达到高精度门槛（防回归）
    assert rep["ner"]["f1"] >= 0.95, rep["ner"]
    assert rep["re"]["precision"] >= 0.95, rep["re"]
    assert rep["re"]["f1"] >= 0.90, rep["re"]
