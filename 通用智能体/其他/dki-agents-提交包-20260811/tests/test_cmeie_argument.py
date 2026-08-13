import json
import hashlib
import math

import pytest

torch = pytest.importorskip("torch")

from dki.nlp.cmeie_argument import (
    ARGUMENT_BILSTM_WEIGHTS,
    ARGUMENT_HEAD_WEIGHTS,
    ARGUMENT_MODEL_CONFIG,
    ARGUMENT_SOFTLEXICON,
    ARGUMENT_SOFTLEXICON_WEIGHTS,
    argument_model_contract,
    build_labels,
    decode_mentions,
    gold_mentions,
    labels_from_rows,
    load_argument_model_contract,
    mention_metric_by_label,
    validate_argument_model_contract,
)


def _row():
    return {
        "text": "疾病甲伴症状乙。疾病甲需检查丙。",
        "spo_list": [
            {"subject": "疾病甲", "subject_type": "疾病", "predicate": "临床表现", "object": "症状乙", "object_type": "症状"},
            {"subject": "疾病甲", "subject_type": "疾病", "predicate": "辅助检查", "object": "检查丙", "object_type": "检查"},
        ],
    }


def test_argument_labels_keep_native_types_and_repeated_mentions():
    row = _row()
    labels = labels_from_rows([row])
    assert labels == ["检查", "疾病", "症状"]
    offsets = [(0, 0)] + [(i, i + 1) for i in range(len(row["text"]))] + [(0, 0)]
    dense, dropped = build_labels(row, offsets, {name: i for i, name in enumerate(labels)})
    assert dropped == 0
    assert dense[labels.index("疾病")].sum().item() == 2
    assert len(gold_mentions(row)) == 4


def test_argument_decode_uses_exact_char_offsets():
    row = _row()
    labels = ["疾病"]
    offsets = [(0, 0)] + [(i, i + 1) for i in range(len(row["text"]))] + [(0, 0)]
    logits = torch.full((1, len(offsets), len(offsets)), -10.0)
    logits[0, 1, 3] = 1.0
    assert decode_mentions(row["text"], offsets, logits, labels, 0.0) == {("疾病", 0, 3, "疾病甲")}


def test_argument_model_contract_is_exact_and_loadable(tmp_path):
    labels = [f"类型{index:02d}" for index in range(11)]
    contract = argument_model_contract(
        labels=labels,
        max_length=320,
        selected_threshold=-0.5,
    )
    assert validate_argument_model_contract(contract) == contract
    model_dir = tmp_path / "argument-model"
    model_dir.mkdir()
    (model_dir / ARGUMENT_MODEL_CONFIG).write_text(
        json.dumps(contract, ensure_ascii=False),
        encoding="utf-8",
    )
    (model_dir / ARGUMENT_HEAD_WEIGHTS).write_bytes(b"weights")
    assert load_argument_model_contract(model_dir) == contract
    tampered = dict(contract)
    tampered["pointer_variant"] = "unsupported"
    with pytest.raises(ValueError, match="pointer_variant"):
        validate_argument_model_contract(tampered)
    with pytest.raises(ValueError, match="finite"):
        argument_model_contract(
            labels=labels,
            max_length=320,
            selected_threshold=math.inf,
        )


def test_argument_model_contract_loads_legacy_nonnegative_schema_v1():
    labels = [f"legacy-{index:02d}" for index in range(11)]
    legacy = {
        "schema_version": 1,
        "kind": "cmeie_native_argument_globalpointer",
        "pointer_variant": "efficient",
        "inner_dim": 64,
        "labels": labels,
        "n_labels": 11,
        "max_length": 320,
        "selected_threshold": 0.25,
        "head_weights": ARGUMENT_HEAD_WEIGHTS,
    }
    assert validate_argument_model_contract(legacy) == legacy
    legacy["selected_threshold"] = -0.5
    with pytest.raises(ValueError, match="schema-v1"):
        validate_argument_model_contract(legacy)


def test_argument_model_contract_binds_classic_bilstm_weights(tmp_path):
    labels = [f"类型{index:02d}" for index in range(11)]
    contract = argument_model_contract(
        labels=labels,
        max_length=320,
        selected_threshold=-1.0,
        pointer_variant="classic",
        use_bilstm=True,
    )
    assert contract["pointer_variant"] == "classic"
    assert contract["bilstm_weights"] == ARGUMENT_BILSTM_WEIGHTS
    model_dir = tmp_path / "classic-bilstm"
    model_dir.mkdir()
    (model_dir / ARGUMENT_MODEL_CONFIG).write_text(
        json.dumps(contract, ensure_ascii=False),
        encoding="utf-8",
    )
    (model_dir / ARGUMENT_HEAD_WEIGHTS).write_bytes(b"head")
    with pytest.raises(FileNotFoundError, match="BiLSTM"):
        load_argument_model_contract(model_dir)
    (model_dir / ARGUMENT_BILSTM_WEIGHTS).write_bytes(b"bilstm")
    assert load_argument_model_contract(model_dir) == contract


def test_argument_model_contract_binds_fit_softlexicon(tmp_path):
    labels = [f"type-{index:02d}" for index in range(11)]
    surfaces = b'{"kind":"test"}\n'
    digest = hashlib.sha256(surfaces).hexdigest()
    contract = argument_model_contract(
        labels=labels,
        max_length=320,
        selected_threshold=0.0,
        pointer_variant="classic",
        use_bilstm=True,
        softlexicon_sha256=digest,
        softlexicon_min_chars=2,
    )
    assert contract["schema_version"] == 4
    assert contract["softlexicon_mode"] == "add"
    assert contract["softlexicon_dim"] == 44
    model_dir = tmp_path / "softlex-argument-model"
    model_dir.mkdir()
    (model_dir / ARGUMENT_MODEL_CONFIG).write_text(
        json.dumps(contract, ensure_ascii=False),
        encoding="utf-8",
    )
    (model_dir / ARGUMENT_HEAD_WEIGHTS).write_bytes(b"head")
    (model_dir / ARGUMENT_BILSTM_WEIGHTS).write_bytes(b"bilstm")
    (model_dir / ARGUMENT_SOFTLEXICON_WEIGHTS).write_bytes(b"projection")
    (model_dir / ARGUMENT_SOFTLEXICON).write_bytes(surfaces)
    assert load_argument_model_contract(model_dir) == contract
    (model_dir / ARGUMENT_SOFTLEXICON).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SoftLexicon SHA-256"):
        load_argument_model_contract(model_dir)


def test_argument_metrics_report_every_native_type():
    labels = ["疾病", "症状"]
    gold = [
        {
            ("疾病", 0, 3, "疾病甲"),
            ("症状", 4, 7, "症状乙"),
        }
    ]
    predicted = [{("疾病", 0, 3, "疾病甲")}]
    by_type = mention_metric_by_label(predicted, gold, labels)
    assert by_type["疾病"]["f1"] == 1.0
    assert by_type["症状"]["recall"] == 0.0
