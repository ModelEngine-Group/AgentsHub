import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_argument_v2",
    ROOT / "scripts" / "train_argument_ner_cmeie_v2.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_partial_accumulation_uses_actual_tail_group_size():
    sizes = [
        MODULE.accumulation_group_size(index, 10, 4)
        for index in range(1, 11)
    ]
    assert sizes == [4, 4, 4, 4, 4, 4, 4, 4, 2, 2]
    assert MODULE.optimizer_steps_per_epoch(10, 4) == 3


def test_sparse_early_stopping_does_not_spend_zero_f1_patience():
    best, stale, improved = MODULE.advance_sparse_early_stopping(
        None,
        0,
        (0.0, 0.0, 1.0),
    )
    assert improved is True
    for _ in range(6):
        best, stale, improved = MODULE.advance_sparse_early_stopping(
            best,
            stale,
            (0.0, 0.0, 0.75),
        )
        assert improved is False
        assert stale == 0
    best, stale, improved = MODULE.advance_sparse_early_stopping(
        best,
        stale,
        (0.4, 0.5, 0.25),
    )
    assert improved is True
    best, stale, improved = MODULE.advance_sparse_early_stopping(
        best,
        stale,
        (0.3, 0.6, 0.5),
    )
    assert improved is False
    assert stale == 1


def test_threshold_grid_is_deterministic_and_accepts_negative_values():
    assert MODULE.parse_threshold_grid("1,-0.5,0,-0.5") == (
        -0.5,
        0.0,
        1.0,
    )
    with pytest.raises(ValueError, match="finite"):
        MODULE.parse_threshold_grid("0,nan")


def test_argument_precision_profile_binds_autocast_and_scaler_policy():
    assert MODULE.argument_precision_profile(
        "fp16",
        bf16_supported=False,
    ) == {
        "precision": "fp16",
        "uses_grad_scaler": True,
        "bf16_supported": False,
    }
    assert MODULE.argument_precision_profile(
        "BF16",
        bf16_supported=True,
    ) == {
        "precision": "bf16",
        "uses_grad_scaler": False,
        "bf16_supported": True,
    }
    with pytest.raises(ValueError, match="not supported"):
        MODULE.argument_precision_profile(
            "bf16",
            bf16_supported=False,
        )
    with pytest.raises(ValueError, match="fp16 or bf16"):
        MODULE.argument_precision_profile(
            "fp32",
            bf16_supported=True,
        )


def test_phase_learning_rate_has_independent_warmup_and_zero_tail():
    assert MODULE.linear_phase_learning_rate(
        step=1,
        total_steps=10,
        warmup_steps=2,
        peak=1e-4,
    ) == pytest.approx(5e-5)
    assert MODULE.linear_phase_learning_rate(
        step=2,
        total_steps=10,
        warmup_steps=2,
        peak=1e-4,
    ) == pytest.approx(1e-4)
    assert MODULE.linear_phase_learning_rate(
        step=10,
        total_steps=10,
        warmup_steps=2,
        peak=1e-4,
    ) == 0.0


def test_argument_training_strategy_allows_only_predeclared_schedules():
    assert MODULE.argument_training_strategy(
        head_warmup_epochs=2,
        epochs=8,
        encoder_scope="all",
    ) == "phased_head_then_joint"
    assert MODULE.argument_training_strategy(
        head_warmup_epochs=8,
        epochs=8,
        encoder_scope="none",
    ) == "frozen_encoder_ablation"
    assert MODULE.argument_training_strategy(
        head_warmup_epochs=4,
        epochs=8,
        encoder_scope="last2",
    ) == "top2_encoder_low_lr_ablation"
    with pytest.raises(ValueError, match="encoder schedule"):
        MODULE.argument_training_strategy(
            head_warmup_epochs=3,
            epochs=8,
            encoder_scope="last2",
        )


def test_softlex_ablation_contract_allows_only_the_declared_pair():
    common = {
        "seed": 43,
        "epochs": 4,
        "head_warmup_epochs": 4,
        "encoder_scope": "none",
    }
    assert MODULE.softlex_ablation_contract(
        role="baseline",
        softlexicon=False,
        **common,
    ) == "baseline"
    assert MODULE.softlex_ablation_contract(
        role="softlex",
        softlexicon=True,
        **common,
    ) == "softlex"
    with pytest.raises(ValueError, match="enablement"):
        MODULE.softlex_ablation_contract(
            role="baseline",
            softlexicon=True,
            **common,
        )
    with pytest.raises(ValueError, match="seed43"):
        MODULE.softlex_ablation_contract(
            role="softlex",
            softlexicon=True,
            **{**common, "seed": 42},
        )


def test_last2_encoder_scope_selects_only_highest_two_numbered_layers():
    names = [
        "embeddings.word_embeddings.weight",
        "encoder.layer.0.attention.weight",
        "encoder.layer.9.attention.weight",
        "encoder.layer.10.attention.weight",
        "encoder.layer.11.attention.weight",
        "pooler.dense.weight",
    ]
    assert MODULE.selected_encoder_parameter_names(
        names,
        encoder_scope="last2",
    ) == frozenset(
        {
            "encoder.layer.10.attention.weight",
            "encoder.layer.11.attention.weight",
        }
    )


def test_calibration_history_preserves_complete_sweep_and_selected_separately():
    sweep = [
        {
            "threshold": 0.0,
            "aggregate": {"f1": 0.2, "precision": 0.1},
            "by_type": {"disease": {"f1": 0.2}},
            "predicted_mentions": 12,
            "gold_mentions": 5,
        },
        {
            "threshold": 0.5,
            "aggregate": {"f1": 0.4, "precision": 0.6},
            "by_type": {"disease": {"f1": 0.4}},
            "predicted_mentions": 4,
            "gold_mentions": 5,
        },
    ]

    record = MODULE.calibration_history_fields(sweep)

    assert record["calibration_sweep"] == sweep
    assert record["selected_calibration"] == sweep[1]
    assert record["calibration_sweep"] is not sweep
    assert record["selected_calibration"] is not sweep[1]
    sweep[0]["aggregate"]["f1"] = 1.0
    assert record["calibration_sweep"][0]["aggregate"]["f1"] == 0.2


@pytest.mark.parametrize(
    ("f1", "predicted_mentions"),
    [
        (0.0, 7),
        (-0.1, 7),
        (0.5, 0),
    ],
)
def test_zero_value_operating_points_decide_not_to_publish(
    f1,
    predicted_mentions,
):
    publish, reason = MODULE.candidate_publication_decision(
        {
            "aggregate": {"f1": f1},
            "predicted_mentions": predicted_mentions,
        }
    )

    assert publish is False
    assert reason == MODULE.REJECTION_REASON


def test_positive_operating_point_decides_to_publish():
    assert MODULE.candidate_publication_decision(
        {
            "aggregate": {"f1": 0.01},
            "predicted_mentions": 1,
        }
    ) == (True, None)


def test_started_receipt_is_create_once(tmp_path):
    receipt = tmp_path / "started.json"
    MODULE.publish_json_create_once(receipt, {"schema_version": 1})
    assert receipt.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError):
        MODULE.publish_json_create_once(receipt, {"schema_version": 2})


def test_candidate_directory_publication_is_no_replace(tmp_path):
    output = tmp_path / "candidate"
    MODULE._publish_model_directory(
        output,
        lambda staging: (staging / "payload.txt").write_text(
            "first",
            encoding="utf-8",
        ),
    )
    assert (output / "payload.txt").read_text(encoding="utf-8") == "first"
    with pytest.raises(FileExistsError):
        MODULE._publish_model_directory(
            output,
            lambda staging: (staging / "payload.txt").write_text(
                "second",
                encoding="utf-8",
            ),
        )
    assert (output / "payload.txt").read_text(encoding="utf-8") == "first"


def test_candidate_payload_tree_can_exclude_receipt(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"model")
    before = MODULE.sha256_tree(tmp_path)
    (tmp_path / "training_receipt.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    excluding = MODULE.sha256_tree(
        tmp_path,
        excluded_relative_paths=frozenset({"training_receipt.json"}),
    )
    assert excluding == before
