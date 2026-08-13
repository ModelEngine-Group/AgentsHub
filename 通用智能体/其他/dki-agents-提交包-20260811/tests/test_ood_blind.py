"""冻结后 OOD 盲测数据生成器测试。"""

from scripts.gen_ood_blind import _hash, _json_array, _validated_records


def test_blind_records_require_literal_mentions_and_three_types():
    texts = ["患者昨晚起剧烈咳嗽伴反复发热，急诊明确诊断为社区获得性肺炎，遂口服阿莫西林克拉维酸，并安排胸部CT检查复诊。"]
    labels = [{
        "index": 0,
        "entities": [
            {"mention": "咳嗽", "name": "咳嗽", "type": "Symptom"},
            {"mention": "发热", "name": "发热", "type": "Symptom"},
            {"mention": "社区获得性肺炎", "name": "社区获得性肺炎", "type": "Disease"},
            {"mention": "阿莫西林克拉维酸", "name": "阿莫西林克拉维酸", "type": "Drug"},
            {"mention": "胸部CT", "name": "胸部CT", "type": "Exam"},
            {"mention": "不存在", "name": "幻觉", "type": "Disease"},
        ],
    }]
    rows = _validated_records(texts, labels)

    assert len(rows) == 1
    assert len(rows[0]["entities"]) == 5
    assert all(e["mention"] in texts[0] for e in rows[0]["entities"])


def test_blind_records_reject_underlabelled_case():
    texts = ["患者昨晚起剧烈咳嗽伴反复发热，急诊明确诊断为社区获得性肺炎，遂口服阿莫西林克拉维酸，并安排胸部CT检查复诊。"]
    labels = [{"index": 0, "entities": [
        {"mention": "咳嗽", "name": "咳嗽", "type": "Symptom"},
        {"mention": "发热", "name": "发热", "type": "Symptom"},
        {"mention": "肺炎", "name": "肺炎", "type": "Disease"},
    ]}]

    assert _validated_records(texts, labels) == []


def test_json_array_ignores_reasoning_brackets():
    raw = '<think>先检查类型 [Disease, Symptom]。</think>\n[{"index":0,"entities":[]}]'
    assert _json_array(raw) == [{"index": 0, "entities": []}]


def test_blind_hash_is_stable_and_strips_outer_space():
    assert _hash(" 病历文本 ") == _hash("病历文本")
