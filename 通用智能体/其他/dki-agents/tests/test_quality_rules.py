"""可选 quality NER 的规则校准测试。"""

from dki.nlp.quality_rules import calibrate_quality_entities, merge_quality_recall


def _ent(text, mention, entity_type):
    start = text.index(mention)
    return {"mention": mention, "name": mention, "type": entity_type,
            "span": [start, start + len(mention)], "source": "bert"}


def test_calibrate_splits_blood_pressure_and_glucose_findings():
    text = "患者血压高血糖也高，刚查了血糖。"
    raw = [_ent(text, "血压高血糖也高", "Symptom"), _ent(text, "查", "Exam")]
    out = calibrate_quality_entities(text, raw)

    assert ("血压高", "Disease") in {(e["mention"], e["type"]) for e in out}
    assert ("血糖也高", "Disease") in {(e["mention"], e["type"]) for e in out}
    assert ("查", "Exam") not in {(e["mention"], e["type"]) for e in out}


def test_calibrate_adds_explicit_exam_patterns():
    text = "查体BP 150/95，FBG 8.2，后来又做了CT。"
    out = calibrate_quality_entities(text, [])
    pairs = {(e["mention"], e["type"]) for e in out}

    assert ("BP 150/95", "Exam") in pairs
    assert ("FBG 8.2", "Exam") in pairs
    assert ("CT", "Exam") in pairs


def test_calibrate_leaves_normal_entities_unchanged():
    text = "患者头晕，服用二甲双胍。"
    raw = [_ent(text, "头晕", "Symptom"), _ent(text, "二甲双胍", "Drug")]

    assert calibrate_quality_entities(text, raw) == raw


def test_quality_recall_rescues_complementary_entities_and_abbreviations():
    text = "患者打鼾，PSG提示异常，予CBT-I，短期服用NSAIDs。"
    quality = [_ent(text, "打鼾", "Symptom")]
    baseline = [_ent(text, "PSG", "Exam"), _ent(text, "异常", "Symptom")]
    out = merge_quality_recall(text, quality, baseline)
    pairs = {(e["mention"], e["type"]) for e in out}

    assert ("打鼾", "Symptom") in pairs
    assert ("PSG", "Exam") in pairs
    assert ("CBT-I", "Treatment") in pairs
    assert ("NSAIDs", "Drug") in pairs


def test_quality_recall_filters_single_cjk_fragments():
    text = "患者血压高，已查心电图。"
    quality = [_ent(text, "血", "Symptom"), _ent(text, "心电图", "Exam")]
    out = merge_quality_recall(text, quality, [])

    assert ("血", "Symptom") not in {(e["mention"], e["type"]) for e in out}
    assert ("心电图", "Exam") in {(e["mention"], e["type"]) for e in out}
