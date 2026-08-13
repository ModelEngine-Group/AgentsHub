"""广域质量语料生成、审计和分层切分回归测试。"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


@pytest.fixture(autouse=True)
def _scripts_on_path(monkeypatch):
    import sys

    monkeypatch.syspath_prepend(str(SCRIPTS))
    for name in ("gen_ood_quality_v2", "build_quality_data_v2"):
        sys.modules.pop(name, None)


def test_quality_v2_clean_entities_drops_conflicts_and_fragments():
    from gen_ood_quality_v2 import _clean_entities

    text = "查血常规后诊断肺炎，给予阿莫西林。"
    entities = _clean_entities(text, [
        {"mention": "查", "name": "检查", "type": "Exam"},
        {"mention": "血常规", "name": "血常规", "type": "Exam"},
        {"mention": "肺炎", "name": "肺炎", "type": "Disease"},
        {"mention": "肺炎", "name": "肺炎", "type": "Symptom"},
        {"mention": "阿莫西林", "name": "阿莫西林", "type": "Drug"},
    ])
    assert [(entity["mention"], entity["type"]) for entity in entities] == [
        ("血常规", "Exam"),
        ("阿莫西林", "Drug"),
    ]


def test_quality_v2_validation_requires_anchor_types():
    from gen_ood_quality_v2 import _validated_records

    text = "患者因持续咳嗽并发热到呼吸科就诊，检查胸部CT后诊断肺炎，给予阿莫西林胶囊口服并建议呼吸训练和两周后复查。"
    labels = [{
        "index": 0,
        "entities": [
            {"mention": "咳嗽", "name": "咳嗽", "type": "Symptom"},
            {"mention": "胸部CT", "name": "胸部CT", "type": "Exam"},
            {"mention": "肺炎", "name": "肺炎", "type": "Disease"},
            {"mention": "阿莫西林胶囊", "name": "阿莫西林", "type": "Drug"},
            {"mention": "呼吸训练", "name": "呼吸训练", "type": "Treatment"},
            {"mention": "持续咳嗽", "name": "咳嗽", "type": "Symptom"},
        ],
    }]
    assert len(_validated_records([text], labels)) == 1
    without_drug = json.loads(json.dumps(labels))
    without_drug[0]["entities"] = [
        entity for entity in without_drug[0]["entities"] if entity["type"] != "Drug"
    ]
    accepted = _validated_records([text], without_drug, min_entities=4)
    assert len(accepted) == 1
    assert {entity["type"] for entity in accepted[0]["entities"]} == {
        "Disease", "Symptom", "Exam", "Treatment",
    }


def test_quality_v2_audit_rejects_blind_overlap():
    from build_quality_data_v2 import _hash, _validate_rows

    row = {
        "doc_id": "v2_1",
        "text": "患者咳嗽，诊断肺炎，给予阿莫西林并做胸部CT检查。",
        "entities": [
            {"mention": "咳嗽", "name": "咳嗽", "type": "Symptom"},
            {"mention": "肺炎", "name": "肺炎", "type": "Disease"},
            {"mention": "阿莫西林", "name": "阿莫西林", "type": "Drug"},
            {"mention": "胸部CT", "name": "胸部CT", "type": "Exam"},
            {"mention": "检查", "name": "检查", "type": "Treatment"},
        ],
    }
    with pytest.raises(ValueError, match="历史 OOD/盲测文本重合"):
        _validate_rows([row], {_hash(row["text"])})


def test_quality_v2_stratified_split_is_deterministic():
    from build_quality_data_v2 import _stratified_split

    rows = [
        {"doc_id": f"resp_{i}", "text": f"患者咳嗽伴肺部症状{i}", "entities": []}
        for i in range(10)
    ] + [
        {"doc_id": f"skin_{i}", "text": f"患者皮肤红疹瘙痒{i}", "entities": []}
        for i in range(10)
    ]
    one = _stratified_split(rows, 20260719, 0.7, 0.15)
    two = _stratified_split(list(reversed(rows)), 20260719, 0.7, 0.15)
    assert {name: [row["doc_id"] for row in data] for name, data in one.items()} == {
        name: [row["doc_id"] for row in data] for name, data in two.items()
    }
    assert {name: len(data) for name, data in one.items()} == {
        "train": 14,
        "dev": 4,
        "holdout": 2,
    }
