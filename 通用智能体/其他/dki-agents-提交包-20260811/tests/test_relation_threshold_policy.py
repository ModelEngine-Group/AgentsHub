import json

import pytest

from dki.operators.medical_relation_extractor.process import (
    MedicalRelationExtractor,
    _load_threshold_map,
    _sha256_tree,
)


def test_threshold_map_is_checkpoint_and_predicate_bound(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"weights")
    predicates = ["关系一", "关系二"]
    policy = tmp_path / "thresholds.json"
    policy.write_text(
        json.dumps({
            "re_model_sha256": _sha256_tree(model),
            "selected_thresholds": {"关系一": 0.7, "关系二": None},
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    assert _load_threshold_map(policy, model, predicates) == {
        "关系一": 0.7,
        "关系二": None,
    }

    (model / "weights.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="哈希"):
        _load_threshold_map(policy, model, predicates)


def test_candidate_pairs_exclude_same_name_mentions():
    entities = [
        {
            "name": "同名",
            "mention": "同名",
            "type": "Disease",
            "section": "全文",
            "span": [0, 2],
        },
        {
            "name": "同名",
            "mention": "同名",
            "type": "Symptom",
            "section": "全文",
            "span": [3, 5],
        },
    ]
    extractor = object.__new__(MedicalRelationExtractor)
    assert extractor._candidate_pairs(entities, {"全文": "同名 同名。"}) == []
