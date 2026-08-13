import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_cblue.py"
SPEC = importlib.util.spec_from_file_location("download_cblue", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cmeie_normalizer_preserves_schema_types_and_combined_fields():
    row = {
        "text": "甲乙",
        "spo_list": [{
            "subject": "甲", "subject_type": "疾病", "predicate": "用药",
            "object": {"@value": "乙", "剂量": "每日一次"},
            "object_type": {"@value": "药物", "剂量": "频率"},
        }],
    }
    normalized = MODULE._norm_cmeie_row(row)
    spo = normalized["spo_list"][0]
    assert spo["subject_type"] == "疾病"
    assert spo["object_type"] == "药物"
    assert spo["object_fields"] == {"剂量": "每日一次"}


def test_cmeie_normalizer_remains_compatible_with_flat_rows():
    normalized = MODULE._norm_cmeie_row({
        "text": "甲乙", "spo_list": [
            {"subject": "甲", "predicate": "相关", "object": "乙"},
        ],
    })
    assert normalized == {"text": "甲乙", "spo_list": [
        {"subject": "甲", "predicate": "相关", "object": "乙"},
    ]}
