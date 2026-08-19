import importlib.util
from pathlib import Path

import pytest

from dki.nlp.cmeie_pair import allowed_label_ids, normalize_entity_type


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_re_cmeie.py"
SPEC = importlib.util.spec_from_file_location("train_re_cmeie", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mark_pair_marks_directional_mentions_in_context():
    marked = MODULE._mark_pair("高血压导致头痛", "高血压", "头痛")
    assert marked == "[SUBJ]高血压[/SUBJ]导致[OBJ]头痛[/OBJ]"


def test_mark_pair_repeated_mentions_is_deterministic_and_closest():
    text = "甲导致乙，后来甲再次导致乙"
    first = MODULE._mark_pair(text, "甲", "乙")
    second = MODULE._mark_pair(text, "甲", "乙")
    assert first == second
    assert first.count("[SUBJ]") == first.count("[OBJ]") == 1
    assert first.startswith("[SUBJ]甲[/SUBJ]导致[OBJ]乙[/OBJ]")


def test_mark_pair_rejects_cross_sentence_mentions():
    assert MODULE._mark_pair("甲出现。随后乙出现。", "甲", "乙") is None


def test_pair_examples_preserve_multiple_predicates_for_same_pair():
    row = {
        "text": "高血压会导致头痛",
        "spo_list": [
            {"subject": "高血压", "predicate": "相关", "object": "头痛"},
            {"subject": "高血压", "predicate": "并发症", "object": "头痛"},
        ],
    }
    examples = MODULE._pair_examples(row, {"相关": 0, "并发症": 1}, neg_per_pos=0)
    assert len(examples) == 1
    assert examples[0][3] == (0, 1)


def test_pair_examples_preserve_all_type_signatures_for_same_pair():
    row = {
        "text": "甲与乙有关",
        "spo_list": [
            {
                "subject": "甲", "subject_type": "疾病",
                "predicate": "临床表现", "object": "乙", "object_type": "症状",
            },
            {
                "subject": "甲", "subject_type": "疾病",
                "predicate": "病因", "object": "乙", "object_type": "社会学",
            },
        ],
    }
    example = MODULE._pair_examples(
        row, {"临床表现": 0, "病因": 1}, neg_per_pos=0,
    )[0]
    assert example[3] == (0, 1)
    assert example[4] == (("疾病", "症状"), ("疾病", "社会学"))
    assert MODULE._allowed_labels(
        example[4], {("疾病", "症状"): {0}, ("疾病", "社会学"): {1}}, 3,
    ) == {0, 1}


def test_signature_constraints_are_built_from_fit_rows_only():
    pred2id = {"fit-rel": 0, "internal-only-rel": 1}
    fit_rows = [{
        "text": "甲关联乙",
        "spo_list": [{
            "subject": "甲", "subject_type": "疾病",
            "predicate": "fit-rel", "object": "乙", "object_type": "症状",
        }],
    }]
    internal_rows = [{
        "text": "丙关联丁",
        "spo_list": [{
            "subject": "丙", "subject_type": "疾病",
            "predicate": "internal-only-rel", "object": "丁", "object_type": "症状",
        }],
    }]
    constraints = MODULE._build_signature_labels(fit_rows, pred2id)
    assert constraints == {("疾病", "症状"): {0}}
    assert 1 not in constraints[("疾病", "症状")]
    assert internal_rows  # validation fixture is intentionally excluded


def test_pair_examples_negative_has_empty_multilabel_target():
    row = {
        "text": "甲影响乙和丙",
        "spo_list": [
            {"subject": "甲", "predicate": "相关", "object": "乙"},
            {"subject": "甲", "predicate": "相关", "object": "丙"},
        ],
    }
    examples = MODULE._pair_examples(row, {"相关": 0}, neg_per_pos=1)
    assert any(not example[3] for example in examples)
    assert all(isinstance(example[3], tuple) for example in examples)


def test_negative_pair_uses_real_entity_type_signature():
    row = {
        "text": "甲影响乙和丙",
        "spo_list": [
            {
                "subject": "甲",
                "subject_type": "疾病",
                "predicate": "临床表现",
                "object": "乙",
                "object_type": "症状",
            },
            {
                "subject": "甲",
                "subject_type": "疾病",
                "predicate": "药物治疗",
                "object": "丙",
                "object_type": "药物",
            },
        ],
    }
    examples = MODULE._pair_examples(
        row,
        {"临床表现": 0, "药物治疗": 1},
        all_negatives=True,
    )
    negative_signatures = {
        example[4] for example in examples if not example[3]
    }
    assert (("症状", "药物"),) in negative_signatures
    assert (("未知", "未知"),) not in negative_signatures


def test_metric_counts_multilabel_relations_not_pairs_only():
    metric = MODULE._metric([{0, 1}, {2}], [{0, 2}, set()], 0.5)
    assert metric["tp"] == 1
    assert metric["fp"] == 1
    assert metric["fn"] == 2
    assert metric["pred_empty"] == 1


def test_gold_pair_decode_keeps_top_one_when_all_below_threshold():
    decoded = MODULE._decode_probabilities([[0.2, 0.4, 0.1]], 0.5, force_nonempty=True)
    assert decoded == [{1}]
    assert MODULE._decode_probabilities([[0.2, 0.4, 0.1]], 0.5) == [set()]


def test_decode_applies_schema_allowed_predicates_before_top_one():
    decoded = MODULE._decode_probabilities(
        [[0.9, 0.4, 0.8]], 0.5,
        force_nonempty=True, allowed_labels=[{1}],
    )
    assert decoded == [{1}]
    with pytest.raises(ValueError, match="constraints"):
        MODULE._decode_probabilities([[0.1]], 0.5, allowed_labels=[])
    assert MODULE._allowed_labels(
        (("症状", "疾病"),),
        {("疾病", "症状"): {0}},
        2,
    ) == set()


def test_runtime_schema_constraints_fail_closed_for_unseen_signature():
    contract = {"schema_constraints": {"疾病\t症状": [0, 2]}}
    assert allowed_label_ids(contract, "Disease", "Symptom", 4) == {0, 2}
    assert allowed_label_ids(contract, "Symptom", "Disease", 4) == set()
    assert allowed_label_ids({}, "Symptom", "Disease", 4) == {0, 1, 2, 3}
    assert normalize_entity_type("Treatment") == "其他治疗"
    assert normalize_entity_type("Department") == "社会学"
    assert normalize_entity_type("dis") == "疾病"
    assert normalize_entity_type("ite") == "检查"
    assert normalize_entity_type("bod") == "部位"


def test_text_group_split_does_not_leak_duplicates():
    rows = [{"text": f"文本{i}", "spo_list": []} for i in range(10)]
    rows += [{"text": "重复", "spo_list": []}, {"text": "重复", "spo_list": []}]
    fit, internal = MODULE._split_rows(rows, 0.2, 42)
    assert {row["text"] for row in fit}.isdisjoint({row["text"] for row in internal})
    assert len(fit) + len(internal) == len(rows)


def test_atomic_json_write_replaces_complete_document(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text('{"stale": true}', encoding="utf-8")
    MODULE._write_json_atomic(path, {"fresh": True})
    assert path.read_text(encoding="utf-8") == '{\n  "fresh": true\n}'


def test_model_promotion_keeps_old_model_when_staging_fails(tmp_path):
    output = tmp_path / "model"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")

    def fail_write(staging):
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        MODULE._promote_model_directory(output, fail_write)
    assert (output / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (output / "partial.txt").exists()
