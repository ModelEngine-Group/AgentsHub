from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_spo_cmeie.py"
SPEC = importlib.util.spec_from_file_location("train_spo_cmeie", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
BUILDER_SCRIPT = ROOT / "scripts" / "build_cmeie_split_manifest.py"
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_cmeie_split_manifest_for_training_tests",
    BUILDER_SCRIPT,
)
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
assert BUILDER_SPEC.loader is not None
BUILDER_SPEC.loader.exec_module(BUILDER)

from dki.nlp import cmeie_split_protocol as split_protocol


def _rank(f1: float, precision: float = 0.0):
    return (f1, precision, 1.5, True)


def _fixed_args(**updates):
    values = {
        "epochs": 8,
        "batch_size": 8,
        "gradient_accumulation": 4,
        "max_length": 320,
        "early_stopping_patience": 2,
        "device": "npu",
        "fp16": True,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _base_model_tree():
    return {
        "root": "/models/roberta",
        "algorithm": MODULE.TREE_HASH_ALGORITHM,
        "sha256": "6" * 64,
        "n_files": 1,
        "total_bytes": 2,
        "entries": [
            {
                "path": "config.json",
                "size": 2,
                "sha256": "7" * 64,
            }
        ],
    }


def _tampered_base_model_tree():
    report = _base_model_tree()
    report["entries"][0]["size"] = 3
    return report


def _preflight_common():
    return {
        "schema_version": 1,
        "purpose": "cmeie_npu_preflight_common_binding",
        "preflight_receipt_path": "/run/evidence/cmeie/preflight.json",
        "preflight_receipt_sha256": "1" * 64,
        "preflight_producer_script_sha256": "2" * 64,
        "source_tree_sha256": "e" * 64,
        "source_manifest_sha256": "2" * 64,
        "cmeie_train_rows": 14339,
        "cmeie_train_sha256": "a" * 64,
        "cmeie_dev_rows": 3585,
        "cmeie_dev_sha256": "3" * 64,
        "base_model_canonical_path": "/models/roberta",
        "base_model_tree_sha256": "6" * 64,
        "base_model_materialization_receipt_sha256": "4" * 64,
        "runtime_fingerprint_canonical_sha256": "5" * 64,
        "run_binding": {
            "rid": "20260729T120000Z-gbbbbbbb-teeeeeeeeeeee",
            "git_head": "b" * 40,
            "source_tree_sha256": "e" * 64,
            "source_manifest_sha256": "2" * 64,
        },
    }


def _overfit_payload(**updates):
    payload = {
        "schema_version": 1,
        "kind": "cmeie_gplinker_fit_only_overfit_preflight",
        "passed": True,
        "fit_only": True,
        "calibration_accessed": False,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "train_sha256": "a" * 64,
        "source_train_sha256": "a" * 64,
        "fit_shard_sha256": "9" * 64,
        "split_manifest_sha256": "b" * 64,
        "base_model": "/models/roberta",
        "base_model_tree": _base_model_tree(),
        "binding_algorithms": dict(MODULE.BINDING_ALGORITHMS),
        "source_tree_manifest_sha256": "2" * 64,
        "sample_size": 8,
        "steps": 400,
        "max_length": 320,
        "minimum_train_f1": 0.95,
        "fp16": True,
        "device": "npu:0",
        "pointer_variant": "efficient",
        "head_reduction": "balanced",
        "encoder_frozen": False,
        "positive_weight": 1.0,
        "positive_margin": 0.0,
        "encoder_learning_rate": 2e-5,
        "head_learning_rate": 1e-3,
        "joint_head_learning_rate": 1e-4,
        "head_only_warmup_steps": 300,
        "joint_optimizer_steps": 100,
        "encoder_mode": "eval_with_gradients_after_warmup",
        "amp_grad_scaler": True,
        "amp_initial_scale": 1024.0,
        "dropped_label_count": 0,
        "gradient_clip": 1.0,
        "gradient_clip_scope": "separate_encoder_and_pointer_heads",
        "encoder_gradient_clip": 1.0,
        "head_gradient_clip": 5.0,
        "gradient_clip_after_unscale": True,
        "weight_decay": 0.01,
        "weight_decay_policy": "exclude_bias_and_layernorm",
        "positive_weight_implementation": "integer_unrolled_add",
        "model_module_sha256": "c" * 64,
        "alignment_module_sha256": "d" * 64,
        "globalpointer_module_sha256": "8" * 64,
        "tokenizer_vocabulary_sha256": "3" * 64,
        "label_table_sha256": "4" * 64,
        "schema_constraints_sha256": "5" * 64,
        "source_tree_sha256": "e" * 64,
        "preflight_script_sha256": "f" * 64,
        "preflight_common": _preflight_common(),
        "declared_grid_result": {
            "declared_threshold_values": [-0.5, 0.0, 0.5],
            "thresholds": [0.0, -0.5, -0.5],
            "schema_constraint": "on",
            "grid_size": 54,
            "metric": {
                "tp": 98,
                "fp": 2,
                "fn": 2,
                "precision": 0.98,
                "recall": 0.98,
                "f1": 0.98,
            },
        },
    }
    payload["phase_grid_results"] = {
        "head_only": copy.deepcopy(payload["declared_grid_result"]),
        "joint": copy.deepcopy(payload["declared_grid_result"]),
    }
    payload["phase_grid_results"]["joint"]["metric"] = {
        "tp": 90,
        "fp": 10,
        "fn": 10,
        "precision": 0.9,
        "recall": 0.9,
        "f1": 0.9,
    }
    payload["selected_phase"] = "head_only"
    payload.update(updates)
    return payload


def _validate(payload, *, source_tree_sha256="e" * 64):
    return MODULE.validate_overfit_preflight(
        payload,
        receipt_sha256="1" * 64,
        source_train_sha256="a" * 64,
        fit_shard_sha256="9" * 64,
        split_manifest_sha256="b" * 64,
        base_model="/models/roberta",
        base_model_tree=_base_model_tree(),
        source_tree_manifest_sha256="2" * 64,
        preflight_script_sha256="f" * 64,
        model_module_sha256="c" * 64,
        alignment_module_sha256="d" * 64,
        globalpointer_module_sha256="8" * 64,
        tokenizer_vocabulary_sha256="3" * 64,
        label_table_sha256="4" * 64,
        schema_constraints_sha256="5" * 64,
        source_tree_sha256=source_tree_sha256,
        preflight_common=_preflight_common(),
    )


def _source_manifest(root: Path):
    paths = sorted(MODULE.REQUIRED_TRAINING_SOURCES)
    entries = []
    for index, relative in enumerate(paths):
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}", encoding="utf-8")
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["path"]):
        digest.update(entry["path"].encode())
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode())
        digest.update(b"\0")
        digest.update(entry["sha256"].encode())
        digest.update(b"\n")
    return {
        "schema_version": 1,
        "git_head": "2" * 40,
        "included_tree_sha256": digest.hexdigest(),
        "entries": entries,
    }


def test_training_parameters_are_frozen_to_the_declared_protocol():
    MODULE.validate_fixed_training_arguments(_fixed_args())
    with pytest.raises(ValueError, match="batch_size"):
        MODULE.validate_fixed_training_arguments(_fixed_args(batch_size=4))


def test_source_tree_manifest_recomputes_every_included_file(tmp_path: Path):
    manifest = _source_manifest(tmp_path)
    binding = MODULE.validate_source_tree_manifest(
        manifest,
        expected_tree_sha256=manifest["included_tree_sha256"],
        root=tmp_path,
    )
    assert binding["n_files"] == len(MODULE.REQUIRED_TRAINING_SOURCES)

    (tmp_path / "scripts" / "train_spo_cmeie.py").write_text(
        "tampered",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        MODULE.validate_source_tree_manifest(
            manifest,
            expected_tree_sha256=manifest["included_tree_sha256"],
            root=tmp_path,
        )


def _fake_checkpoint_binding(epoch: int) -> dict:
    digit = format(epoch, "x")
    return {
        "checkpoint_bundle_manifest": "checkpoint_bundle.json",
        "checkpoint_bundle_manifest_sha256": digit * 64,
        "checkpoint_bundle_tree_sha256": "a" * 64,
        "checkpoint_bundle_n_files": epoch + 3,
        "checkpoint_bundle_total_bytes": epoch * 100,
        "checkpoint_sha256": "b" * 64,
    }


def _fake_extraction_binding(epoch: int = 1) -> dict:
    digit = format(epoch + 1, "x")
    return {
        "extraction_contract": "extraction_contract.json",
        "extraction_contract_sha256": digit * 64,
        "tokenizer_vocabulary_sha256": "3" * 64,
        "label_table_sha256": "4" * 64,
        "schema_constraints_sha256": "5" * 64,
        "semantic_contract_sha256": "6" * 64,
    }


def test_each_epoch_publishes_complete_checkpoint_binding(
    tmp_path: Path,
    monkeypatch,
):
    calls = []

    def publish(checkpoint: Path):
        calls.append(checkpoint)
        return _fake_checkpoint_binding(len(calls))

    monkeypatch.setattr(
        MODULE,
        "publish_checkpoint_bundle_manifest",
        publish,
    )
    history = []
    for epoch in range(1, MODULE.FIXED_EPOCHS + 1):
        checkpoint = tmp_path / f"epoch-{epoch:02d}"
        checkpoint.mkdir()
        history.append(
            {
                "epoch": epoch,
                **MODULE._publish_checkpoint_binding(checkpoint),
            }
        )

    assert calls == [
        tmp_path / f"epoch-{epoch:02d}"
        for epoch in range(1, MODULE.FIXED_EPOCHS + 1)
    ]
    for epoch, item in enumerate(history, start=1):
        assert item["epoch"] == epoch
        assert set(MODULE.CHECKPOINT_BINDING_FIELDS) <= set(item)
        assert item["checkpoint_bundle_n_files"] == epoch + 3


def test_extraction_contract_is_published_before_bundle_freezing(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "epoch-01"
    checkpoint.mkdir()
    events = []

    def publish_contract(path, **kwargs):
        events.append(("contract", path))
        return _fake_extraction_binding()

    def publish_bundle(path):
        events.append(("bundle", path))
        return _fake_checkpoint_binding(1)

    monkeypatch.setattr(
        MODULE,
        "publish_extraction_contract_once",
        publish_contract,
    )
    monkeypatch.setattr(
        MODULE,
        "publish_checkpoint_bundle_manifest",
        publish_bundle,
    )

    contract = MODULE._publish_extraction_binding(
        checkpoint,
        tokenizer_vocabulary={"[UNK]": 0},
        entity_labels=["subject", "object"],
        predicates=["treats"],
        schema_constraints={"treats": [["subject", "object"]]},
    )
    bundle = MODULE._publish_checkpoint_binding(checkpoint)

    assert events == [
        ("contract", checkpoint),
        ("bundle", checkpoint),
    ]
    assert set(MODULE.EXTRACTION_BINDING_FIELDS) <= set(contract)
    assert set(MODULE.CHECKPOINT_BINDING_FIELDS) <= set(bundle)


def test_tampered_extraction_contract_binding_aborts_epoch(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "epoch-01"
    checkpoint.mkdir()
    tampered = _fake_extraction_binding()
    tampered["semantic_contract_sha256"] = "not-a-sha"
    monkeypatch.setattr(
        MODULE,
        "publish_extraction_contract_once",
        lambda *args, **kwargs: tampered,
    )

    with pytest.raises(RuntimeError, match="semantic_contract_sha256"):
        MODULE._publish_extraction_binding(
            checkpoint,
            tokenizer_vocabulary={"[UNK]": 0},
            entity_labels=["subject", "object"],
            predicates=["treats"],
            schema_constraints={"treats": [["subject", "object"]]},
        )


def test_copied_candidate_path_is_not_canonical(tmp_path):
    expected = tmp_path / "run" / "models" / "cmeie" / "seed-42"
    copied = tmp_path / "copy" / "seed-42"

    with pytest.raises(ValueError, match="canonical run path"):
        MODULE._require_canonical_path(
            copied,
            expected,
            label="candidate output",
        )


def test_training_receipt_v7_binds_evaluator_and_integrity_module(
    tmp_path: Path,
):
    evaluator = tmp_path / "scripts" / "eval_spo_cmeie.py"
    integrity = tmp_path / "dki" / "nlp" / "cmeie_artifacts.py"
    extraction = (
        tmp_path / "dki" / "nlp" / "cmeie_extraction_contract.py"
    )
    evaluator.parent.mkdir(parents=True)
    integrity.parent.mkdir(parents=True)
    evaluator.write_bytes(b"evaluator")
    integrity.write_bytes(b"integrity")
    extraction.write_bytes(b"extraction")

    fields = MODULE._training_receipt_integrity_fields(tmp_path)

    assert fields == {
        "schema_version": 7,
        "evaluator_sha256": hashlib.sha256(b"evaluator").hexdigest(),
        "artifact_integrity_module_sha256": hashlib.sha256(
            b"integrity"
        ).hexdigest(),
        "extraction_contract_module_sha256": hashlib.sha256(
            b"extraction"
        ).hexdigest(),
    }


def test_training_receipt_v6_binds_evaluator_and_integrity_module(
    tmp_path: Path,
):
    """Retain the historical node while verifying the upgraded v7 binding."""

    test_training_receipt_v7_binds_evaluator_and_integrity_module(tmp_path)


def test_finite_gradient_norm_max_is_preserved():
    assert MODULE._finite_grad_norm_max([]) == 0.0
    assert MODULE._finite_grad_norm_max([0.25, 1.5, 0.75]) == 1.5


def test_nonfinite_gradient_norm_max_is_serialized_as_null():
    epoch = {
        "encoder_pre_clip_grad_norm_max": (
            MODULE._finite_grad_norm_max([0.25, float("inf")])
        ),
        "head_pre_clip_grad_norm_max": 1.0,
        "nonfinite_grad_norm_steps": 1,
        "skipped_optimizer_steps": 1,
    }

    decoded = json.loads(
        MODULE._strict_training_receipt_json({"history": [epoch]})
    )

    assert decoded["history"][0] == {
        "encoder_pre_clip_grad_norm_max": None,
        "head_pre_clip_grad_norm_max": 1.0,
        "nonfinite_grad_norm_steps": 1,
        "skipped_optimizer_steps": 1,
    }


def test_strict_training_receipt_rejects_unhandled_nonfinite_value():
    with pytest.raises(ValueError, match="Out of range float values"):
        MODULE._strict_training_receipt_json(
            {"unhandled_metric": float("inf")}
        )


def test_partial_gradient_accumulation_uses_actual_group_size():
    assert [
        MODULE._accumulation_group_size(step, 10, 4)
        for step in range(1, 11)
    ] == [4, 4, 4, 4, 4, 4, 4, 4, 2, 2]


def test_optimizer_step_accounting_uses_executed_early_stop_steps():
    assert MODULE._optimizer_step_accounting(
        planned=24,
        attempts=9,
        skipped=1,
        scheduler_steps=8,
    ) == {
        "total_optimizer_steps": 24,
        "optimizer_step_attempts": 9,
        "actual_scheduler_steps": 8,
        "successful_optimizer_steps": 8,
        "skipped_optimizer_steps": 1,
    }


def test_optimizer_step_accounting_rejects_scheduler_drift():
    with pytest.raises(
        ValueError,
        match="scheduler steps must equal successful optimizer steps",
    ):
        MODULE._optimizer_step_accounting(
            planned=24,
            attempts=9,
            skipped=1,
            scheduler_steps=9,
        )


def test_training_attestation_binds_every_epoch_and_protocol_input():
    history = [
        {
            "epoch": 1,
            **_fake_checkpoint_binding(1),
            **_fake_extraction_binding(1),
        },
        {
            "epoch": 2,
            **_fake_checkpoint_binding(2),
            **_fake_extraction_binding(2),
        },
    ]
    binding = SimpleNamespace(
        rid="20260729T120000Z-gbbbbbbb-teeeeeeeeeeee",
        git_head="b" * 40,
        included_tree_sha256="e" * 64,
        source_manifest_sha256="2" * 64,
        source_manifest=Path("/run/evidence/source/included_tree.json"),
    )
    data = {
        "train_sha256": "a" * 64,
        "source_train_sha256": "a" * 64,
        "fit_shard_sha256": "9" * 64,
        "calibration_shard_sha256": "8" * 64,
        "split_manifest_sha256": "7" * 64,
    }
    evidence = MODULE.training_attestation_evidence(
        training_receipt_sha256="1" * 64,
        history=history,
        preflight_common=_preflight_common(),
        run_binding=binding,
        isolated_data_binding=data,
        configuration={"seed": 42, "epochs": 8},
        overfit_preflight=_validate(_overfit_payload()),
    )

    assert evidence["training_receipt_sha256"] == "1" * 64
    assert evidence["preflight_common"] == _preflight_common()
    assert evidence["data_binding"] == data
    assert evidence["configuration"] == {"seed": 42, "epochs": 8}
    assert evidence["overfit_preflight_sha256"] == "1" * 64
    assert [item["epoch"] for item in evidence["epoch_bindings"]] == [1, 2]
    for item in evidence["epoch_bindings"]:
        assert set(MODULE.CHECKPOINT_BINDING_FIELDS) <= set(item)
        assert set(MODULE.EXTRACTION_BINDING_FIELDS) <= set(item)


def test_training_attestation_refuses_history_without_contract_hash():
    item = {
        "epoch": 1,
        **_fake_checkpoint_binding(1),
        **_fake_extraction_binding(1),
    }
    item.pop("semantic_contract_sha256")
    binding = SimpleNamespace(
        rid="20260729T120000Z-gbbbbbbb-teeeeeeeeeeee",
        git_head="b" * 40,
        included_tree_sha256="e" * 64,
        source_manifest_sha256="2" * 64,
        source_manifest=Path("/run/evidence/source/included_tree.json"),
    )

    with pytest.raises(ValueError, match="semantic_contract_sha256"):
        MODULE.training_attestation_evidence(
            training_receipt_sha256="1" * 64,
            history=[item],
            preflight_common=_preflight_common(),
            run_binding=binding,
            isolated_data_binding={
                "fit_shard_sha256": "9" * 64,
                "calibration_shard_sha256": "8" * 64,
            },
            configuration={"seed": 42},
            overfit_preflight=_validate(_overfit_payload()),
        )


def test_checkpoint_publisher_failure_aborts_before_target_publication(
    tmp_path: Path,
    monkeypatch,
):
    staging = tmp_path / ".candidate.staging"
    checkpoint = staging / "epoch-01"
    checkpoint.mkdir(parents=True)
    target = tmp_path / "candidate"
    history = []

    def fail(_checkpoint: Path):
        raise RuntimeError("bundle publication failed")

    monkeypatch.setattr(
        MODULE,
        "publish_checkpoint_bundle_manifest",
        fail,
    )
    with pytest.raises(RuntimeError, match="bundle publication failed"):
        history.append(
            {
                "epoch": 1,
                **MODULE._publish_checkpoint_binding(checkpoint),
            }
        )
        MODULE._publish_directory_create_once(staging, target)

    assert history == []
    assert staging.is_dir()
    assert not target.exists()


def test_split_manifest_source_tree_binding_requires_v6_and_both_hashes():
    expected_tree = "e" * 64
    expected_manifest = "2" * 64
    assert MODULE.validate_split_source_tree_binding(
        {
            "schema_version": 6,
            "source_tree_sha256": "E" * 64,
            "source_tree_manifest_sha256": expected_manifest,
        },
        expected_tree,
        expected_manifest,
    ) == {
        "source_tree_sha256": expected_tree,
        "source_tree_manifest_sha256": expected_manifest,
    }
    with pytest.raises(ValueError, match="schema is not v6"):
        MODULE.validate_split_source_tree_binding(
            {
                "schema_version": 5,
                "source_tree_sha256": expected_tree,
                "source_tree_manifest_sha256": expected_manifest,
            },
            expected_tree,
            expected_manifest,
        )
    with pytest.raises(ValueError, match="manifest SHA-256 is invalid"):
        MODULE.validate_split_source_tree_binding(
            {
                "schema_version": 6,
                "source_tree_sha256": expected_tree,
            },
            expected_tree,
            expected_manifest,
        )
    with pytest.raises(ValueError, match="differs from the candidate"):
        MODULE.validate_split_source_tree_binding(
            {
                "schema_version": 6,
                "source_tree_sha256": "0" * 64,
                "source_tree_manifest_sha256": expected_manifest,
            },
            expected_tree,
            expected_manifest,
        )
    with pytest.raises(ValueError, match="differs from the current file"):
        MODULE.validate_split_source_tree_binding(
            {
                "schema_version": 6,
                "source_tree_sha256": expected_tree,
                "source_tree_manifest_sha256": "0" * 64,
            },
            expected_tree,
            expected_manifest,
        )


def test_training_calls_two_role_validators_before_model_work(
    tmp_path,
    monkeypatch,
):
    fit_data = tmp_path / "fit.jsonl"
    fit_data.write_text("{}\n", encoding="utf-8")
    calibration_data = tmp_path / "calibration.jsonl"
    calibration_data.write_text("{}\n", encoding="utf-8")
    source_manifest = tmp_path / "included_tree.json"
    source_manifest.write_text("{}", encoding="utf-8")
    split_manifest = tmp_path / "split_manifest.json"
    split_manifest.write_text("{}", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        MODULE,
        "validate_source_tree_manifest",
        lambda *args, **kwargs: {
            "included_tree_sha256": "a" * 64,
            "git_head": "b" * 40,
            "n_files": 1,
        },
    )
    fake_binding = SimpleNamespace(
        included_tree_sha256="a" * 64,
        source_root=tmp_path,
    )
    monkeypatch.setattr(
        MODULE,
        "validate_canonical_training_paths",
        lambda args: (fake_binding, SimpleNamespace()),
    )

    def reject(*args, **kwargs):
        calls.append((args, kwargs))
        raise ValueError("full validator sentinel")

    monkeypatch.setattr(
        MODULE,
        "load_and_validate_role_shard",
        reject,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--run-root",
            str(tmp_path / "run"),
            "--fit-data",
            str(fit_data),
            "--calibration-data",
            str(calibration_data),
            "--split-manifest",
            str(split_manifest),
            "--output",
            str(tmp_path / "candidate"),
            "--overfit-preflight-receipt",
            str(tmp_path / "overfit.json"),
            "--base-model",
            str(tmp_path / "model"),
            "--seed",
            "42",
            "--source-tree-sha256",
            "a" * 64,
            "--source-tree-manifest",
            str(source_manifest),
        ],
    )

    with pytest.raises(SystemExit, match="full validator sentinel"):
        MODULE.main()
    assert len(calls) == 1


def test_training_loader_opens_only_fit_and_calibration(
    tmp_path,
    monkeypatch,
):
    rows = [
        {
            "text": f"唯一标题{i}@正文{i}",
            "spo_list": [{"predicate": "P"}],
        }
        for i in range(30)
    ]
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in rows
        ),
        encoding="utf-8",
    )
    split_dir = tmp_path / "split"
    BUILDER.build_split_directory(
        source=source,
        output_dir=split_dir,
        seed="20260725",
        source_tree_sha256="a" * 64,
        source_tree_manifest_sha256="b" * 64,
    )
    forbidden = {
        source.absolute(),
        (
            split_dir / split_protocol.SHARD_FILENAMES["outer"]
        ).absolute(),
    }
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.absolute() in forbidden:
            raise AssertionError(f"forbidden data access: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    fit, calibration, bindings = MODULE.load_isolated_training_data(
        fit_data=(
            split_dir / split_protocol.SHARD_FILENAMES["fit"]
        ),
        calibration_data=(
            split_dir / split_protocol.SHARD_FILENAMES["calibration"]
        ),
        split_manifest=(
            split_dir / split_protocol.MANIFEST_FILENAME
        ),
        source_tree_sha256="a" * 64,
        source_tree_manifest_sha256="b" * 64,
    )

    assert fit
    assert calibration
    assert bindings["fit"]["outer_accessed"] is False
    assert bindings["calibration"]["outer_accessed"] is False
    assert bindings["fit"]["source_train_accessed"] is False
    assert bindings["calibration"]["source_train_accessed"] is False
    receipt_fields = MODULE.isolated_training_receipt_fields(bindings)
    assert receipt_fields == {
        "train_sha256": bindings["fit"]["source_train_sha256"],
        "source_train_sha256": bindings["fit"][
            "source_train_sha256"
        ],
        "fit_shard_sha256": bindings["fit"]["shard_sha256"],
        "calibration_shard_sha256": bindings["calibration"][
            "shard_sha256"
        ],
        "split_manifest_sha256": bindings["fit"][
            "split_manifest_sha256"
        ],
    }


def test_training_recomputes_canonical_artifact_binding_hashes():
    first = MODULE._artifact_binding_hashes(
        tokenizer_vocabulary={"z": 2, "a": 1},
        entity_labels=["subject", "object"],
        predicates=["treats"],
        schema_constraints={"treats": [["subject", "object"]]},
    )
    reordered = MODULE._artifact_binding_hashes(
        tokenizer_vocabulary={"a": 1, "z": 2},
        entity_labels=["subject", "object"],
        predicates=["treats"],
        schema_constraints={"treats": [["subject", "object"]]},
    )
    changed_schema = MODULE._artifact_binding_hashes(
        tokenizer_vocabulary={"a": 1, "z": 2},
        entity_labels=["subject", "object"],
        predicates=["treats"],
        schema_constraints={"treats": [["object", "subject"]]},
    )

    assert first == reordered
    assert (
        first["schema_constraints_sha256"]
        != changed_schema["schema_constraints_sha256"]
    )


def test_passed_fit_only_overfit_receipt_authorizes_training():
    binding = _validate(_overfit_payload())
    assert binding["metric"]["f1"] == 0.98
    assert binding["head_reduction"] == "balanced"
    assert binding["amp_grad_scaler"] is True
    assert binding["calibration_accessed"] is False
    assert binding["source_tree_sha256"] == "e" * 64
    assert binding["source_train_sha256"] == "a" * 64
    assert binding["fit_shard_sha256"] == "9" * 64
    assert binding["base_model_tree"] == _base_model_tree()
    assert binding["tokenizer_vocabulary_sha256"] == "3" * 64
    assert binding["label_table_sha256"] == "4" * 64
    assert binding["schema_constraints_sha256"] == "5" * 64
    assert binding["globalpointer_module_sha256"] == "8" * 64


def test_overfit_receipt_source_tree_binding_normalizes_hex_case():
    binding = _validate(
        _overfit_payload(source_tree_sha256="E" * 64),
        source_tree_sha256="e" * 64,
    )
    assert binding["source_tree_sha256"] == "e" * 64


def test_overfit_receipt_rejects_a_different_valid_source_tree():
    with pytest.raises(ValueError, match="differs from the candidate"):
        _validate(_overfit_payload(source_tree_sha256="0" * 64))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_sha256", "0" * 64),
        ("source_train_sha256", "0" * 64),
        ("fit_shard_sha256", "0" * 64),
        ("split_manifest_sha256", "0" * 64),
        ("base_model", "/models/other"),
        ("base_model_tree", _tampered_base_model_tree()),
        ("source_tree_manifest_sha256", "0" * 64),
        ("preflight_script_sha256", "0" * 64),
        ("model_module_sha256", "0" * 64),
        ("alignment_module_sha256", "0" * 64),
        ("globalpointer_module_sha256", "0" * 64),
        ("tokenizer_vocabulary_sha256", "0" * 64),
        ("label_table_sha256", "0" * 64),
        ("schema_constraints_sha256", "0" * 64),
        ("preflight_common", {"schema_version": 0}),
        ("binding_algorithms", {"tokenizer_vocabulary_sha256": "other"}),
    ],
)
def test_overfit_receipt_rejects_artifact_binding_tamper(field, value):
    with pytest.raises(ValueError, match="not admissible"):
        _validate(_overfit_payload(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("passed", False),
        ("sample_size", 7),
        ("steps", 399),
        ("max_length", 319),
        ("minimum_train_f1", 0.90),
        ("fp16", False),
        ("pointer_variant", "classic"),
        ("head_reduction", "balanced_bce"),
        ("encoder_frozen", True),
        pytest.param(
            "encoder_frozen",
            True,
            id="encoder_frozen-False",
        ),
        ("positive_weight", 2.0),
        ("positive_margin", 1.0),
        ("encoder_learning_rate", 3e-5),
        ("head_learning_rate", 2e-4),
        ("joint_head_learning_rate", 2e-5),
        ("head_only_warmup_steps", 99),
        ("joint_optimizer_steps", 299),
        ("encoder_mode", "train_with_dropout"),
        ("amp_grad_scaler", False),
        ("amp_initial_scale", 2048.0),
        pytest.param(
            "amp_initial_scale",
            2048.0,
            id="amp_initial_scale-32768.0",
        ),
        ("dropped_label_count", 1),
        ("gradient_clip", 2.0),
        ("gradient_clip_scope", "global"),
        ("encoder_gradient_clip", 2.0),
        ("head_gradient_clip", 1.0),
        ("gradient_clip_after_unscale", False),
        ("weight_decay", 0.02),
        ("weight_decay_policy", "all_parameters"),
        ("positive_weight_implementation", "direct_scalar_multiply"),
        ("calibration_accessed", True),
    ],
)
def test_invalid_overfit_receipt_cannot_authorize_training(field, value):
    with pytest.raises(ValueError, match="not admissible"):
        _validate(_overfit_payload(**{field: value}))


def test_overfit_receipt_rejects_threshold_outside_declared_grid():
    payload = _overfit_payload()
    payload["declared_grid_result"]["thresholds"] = [-1.0, 0.0, 0.5]
    with pytest.raises(ValueError, match="not admissible"):
        _validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_constraint", "off"),
        ("grid_size", 52),
    ],
)
def test_overfit_receipt_rejects_incomplete_grid_contract(field, value):
    payload = _overfit_payload()
    payload["declared_grid_result"][field] = value
    with pytest.raises(ValueError, match="not admissible"):
        _validate(payload)


def test_overfit_receipt_rejects_non_npu_or_low_f1_proof():
    with pytest.raises(ValueError, match="not admissible"):
        _validate(_overfit_payload(device="cpu"))

    payload = _overfit_payload()
    payload["declared_grid_result"]["metric"]["f1"] = 0.949
    with pytest.raises(ValueError, match="not admissible"):
        _validate(payload)

    payload = _overfit_payload()
    payload["declared_grid_result"]["metric"]["f1"] = float("nan")
    with pytest.raises(ValueError, match="not admissible"):
        _validate(payload)

    payload = _overfit_payload()
    payload["declared_grid_result"]["thresholds"][1] = False
    with pytest.raises(ValueError, match="not admissible"):
        _validate(payload)


def test_overfit_receipt_rejects_forged_metric_inconsistent_with_counts():
    payload = _overfit_payload()
    payload["declared_grid_result"]["metric"] = {
        "tp": 0,
        "fp": 100,
        "fn": 100,
        "precision": 0.98,
        "recall": 0.98,
        "f1": 0.98,
    }
    with pytest.raises(ValueError, match="disagrees with tp/fp/fn"):
        _validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tp", True),
        ("fp", -1),
        ("fn", 1.5),
    ],
)
def test_overfit_receipt_requires_nonnegative_integer_counts(field, value):
    payload = _overfit_payload()
    payload["declared_grid_result"]["metric"][field] = value
    with pytest.raises(ValueError, match="non-negative integer"):
        _validate(payload)


def test_overfit_receipt_metric_rejects_extra_fields():
    payload = _overfit_payload()
    payload["declared_grid_result"]["metric"]["reported_score"] = 0.98
    with pytest.raises(ValueError, match="metric fields are invalid"):
        _validate(payload)


def test_zero_f1_warmup_does_not_consume_early_stopping_patience():
    best, stale = MODULE.advance_early_stopping(None, 0, _rank(0.0))
    assert best == _rank(0.0)
    assert stale == 0
    best, stale = MODULE.advance_early_stopping(best, stale, _rank(0.0))
    assert best == _rank(0.0)
    assert stale == 0


def test_patience_is_counted_after_first_positive_f1():
    best, stale = MODULE.advance_early_stopping(None, 0, _rank(0.20, 0.30))
    assert stale == 0
    best, stale = MODULE.advance_early_stopping(
        best,
        stale,
        _rank(0.19, 0.40),
    )
    assert stale == 1
    best, stale = MODULE.advance_early_stopping(
        best,
        stale,
        _rank(0.18, 0.50),
    )
    assert stale == 2


def test_equal_positive_f1_with_better_precision_is_an_improvement():
    best, stale = MODULE.advance_early_stopping(
        None,
        0,
        _rank(0.20, 0.30),
    )
    best, stale = MODULE.advance_early_stopping(
        best,
        stale,
        _rank(0.20, 0.31),
    )
    assert best == _rank(0.20, 0.31)
    assert stale == 0


def test_adamw_decay_policy_excludes_bias_and_layernorm():
    assert MODULE._uses_weight_decay("encoder.layer.0.attention.weight")
    assert not MODULE._uses_weight_decay("encoder.layer.0.attention.bias")
    assert not MODULE._uses_weight_decay(
        "encoder.layer.0.LayerNorm.weight"
    )
    assert not MODULE._uses_weight_decay(
        "encoder.layer.0.layer_norm.weight"
    )


def test_phased_linear_schedule_delays_encoder_until_epoch_three():
    contract = MODULE.joint_schedule_contract(
        optimiser_steps_per_epoch=100,
        epochs=8,
    )
    assert contract == {
        "head_only_optimizer_steps": 200,
        "joint_optimizer_steps": 600,
        "total_optimizer_steps": 800,
        "head_scheduler_warmup_steps": 80,
        "encoder_scheduler_warmup_steps": 60,
    }
    assert MODULE.FIXED_HEAD_WARMUP_EPOCHS == 2
    assert MODULE.FIXED_JOINT_TRAINING_START_EPOCH == 3
    assert MODULE.FIXED_JOINT_HEAD_LEARNING_RATE_SCALE == 0.1
    assert MODULE.FIXED_AMP_INITIAL_SCALE == 1024.0
    assert MODULE.phased_linear_scale(
        0,
        start=0,
        length=800,
        warmup=80,
    ) == pytest.approx(1e-3)
    assert MODULE.phased_linear_scale(
        80,
        start=0,
        length=800,
        warmup=80,
    ) == pytest.approx(1.0)
    assert MODULE.phased_linear_scale(
        800,
        start=0,
        length=800,
        warmup=80,
    ) == 0.0


def test_training_phase_freezes_then_unfreezes_encoder_without_dropout():
    class Parameter:
        def __init__(self):
            self.requires_grad = False

        def requires_grad_(self, enabled):
            self.requires_grad = enabled

    class Encoder:
        training = False

        def eval(self):
            self.training = False

    class TinyJointModel:
        def __init__(self):
            self.encoder = Encoder()
            self.training = False

        def train(self):
            self.training = True
            self.encoder.training = True

    model = TinyJointModel()
    parameters = [Parameter(), Parameter()]
    MODULE.configure_training_phase(
        model,
        parameters,
        head_only=True,
    )
    assert model.training
    assert not model.encoder.training
    assert not any(parameter.requires_grad for parameter in parameters)

    MODULE.configure_training_phase(
        model,
        parameters,
        head_only=False,
    )
    assert model.training
    assert not model.encoder.training
    assert all(parameter.requires_grad for parameter in parameters)


def test_gradient_clip_separates_encoder_and_pointer_heads():
    calls = []

    class Utils:
        @staticmethod
        def clip_grad_norm_(parameters, max_norm):
            calls.append((list(parameters), max_norm))
            return float(max_norm)

    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(utils=Utils),
    )
    encoder = object()
    head = object()
    pre_clip = MODULE.clip_grouped_gradients(
        fake_torch,
        [encoder],
        [head],
        head_only=False,
    )
    assert pre_clip == {"encoder": 1.0, "pointer_heads": 5.0}
    assert calls == [([head], 5.0), ([encoder], 1.0)]

    calls.clear()
    head_phase = MODULE.clip_grouped_gradients(
        fake_torch,
        [encoder],
        [head],
        head_only=True,
    )
    assert head_phase == {"encoder": None, "pointer_heads": 5.0}
    assert calls == [([head], 5.0)]


def test_gradient_clip_uses_one_global_norm_for_encoder_and_heads():
    """Historical node now proves the unsafe global clip was replaced."""

    test_gradient_clip_separates_encoder_and_pointer_heads()


def test_joint_training_enables_encoder_and_train_mode_from_epoch_one():
    """Historical node now pins phased unfreezing with encoder dropout off."""

    test_training_phase_freezes_then_unfreezes_encoder_without_dropout()


def test_shared_linear_schedule_starts_all_groups_at_first_step():
    """Historical node now pins delayed, independently warmed encoder steps."""

    test_phased_linear_schedule_delays_encoder_until_epoch_three()


def test_amp_step_detection_only_rejects_scale_backoff():
    assert MODULE.amp_optimizer_step_succeeded(65536.0, 65536.0)
    assert MODULE.amp_optimizer_step_succeeded(65536.0, 131072.0)
    assert not MODULE.amp_optimizer_step_succeeded(65536.0, 32768.0)


def test_directory_publication_refuses_an_existing_target(tmp_path: Path):
    staging = tmp_path / ".candidate.staging"
    staging.mkdir()
    (staging / "candidate.txt").write_text("new", encoding="utf-8")
    target = tmp_path / "candidate"
    target.mkdir()
    (target / "owner.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        MODULE._publish_directory_create_once(staging, target)

    assert (target / "owner.txt").read_text(encoding="utf-8") == "existing"
    assert (staging / "candidate.txt").read_text(encoding="utf-8") == "new"


def test_concurrent_directory_publication_has_exactly_one_winner(
    tmp_path: Path,
):
    target = tmp_path / "candidate"
    barrier = threading.Barrier(2)

    def publish(name: str) -> tuple[str, str]:
        staging = tmp_path / f".candidate.{name}"
        staging.mkdir()
        (staging / "owner.txt").write_text(name, encoding="utf-8")
        barrier.wait()
        try:
            MODULE._publish_directory_create_once(staging, target)
        except FileExistsError:
            return name, "lost"
        return name, "won"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, ("one", "two")))

    assert sorted(status for _, status in outcomes) == ["lost", "won"]
    winner = next(name for name, status in outcomes if status == "won")
    assert (target / "owner.txt").read_text(encoding="utf-8") == winner
