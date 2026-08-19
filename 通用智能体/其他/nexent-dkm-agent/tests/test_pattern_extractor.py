"""Tests for OOV suffix-pattern entity discovery."""

from __future__ import annotations

from src.operators.kg_ops.entity_extractor import reload_entity_dictionary
from src.operators.kg_ops.extraction_eval import evaluate_extraction_vocabulary_split
from src.operators.kg_ops.pattern_extractor import find_pattern_entities


def test_find_pattern_entities_fabry_disease():
    text = "记录: 患儿确诊法布里病，反复肢端灼痛，建议酶替代治疗随访。"
    found = {etype: term for etype, term in find_pattern_entities(text)}
    assert found.get("Disease") == "法布里病"
    assert found.get("Symptom") == "肢端灼痛"
    assert found.get("Treatment") == "酶替代治疗"


def test_pattern_skips_lesion_false_positive():
    text = "记录: 脑卒中后遗留头晕，MRI评估病灶。"
    diseases = [t for et, t in find_pattern_entities(text) if et == "Disease"]
    assert "评估病" not in diseases


def test_oov_benchmark_recall_above_threshold():
    import json
    from pathlib import Path

    reload_entity_dictionary()
    gold_path = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "kg_extraction_oov_gold.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    report = evaluate_extraction_vocabulary_split(gold["records"])
    assert report["vocabulary_split"]["out_of_vocabulary"]["recall"] >= 0.95


def test_holdout_precision_unchanged():
    import json
    from pathlib import Path

    from src.operators.kg_ops.extraction_eval import evaluate_extraction_quality

    reload_entity_dictionary()
    gold_path = Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "kg_extraction_gold.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    report = evaluate_extraction_quality(gold["records"])
    assert report["overall"]["precision"] == 1.0
    assert all(not rec["false_positives"] for rec in report["records"])
