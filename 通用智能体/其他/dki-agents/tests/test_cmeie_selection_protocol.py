from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dki.nlp import cmeie_run_protocol as run_protocol
from dki.nlp import cmeie_selection_protocol as protocol
from dki.nlp.cmeie_artifacts import (
    publish_checkpoint_bundle_manifest,
)
from dki.nlp.cmeie_extraction_contract import (
    BINDING_ALGORITHMS,
    EXTRACTION_CONTRACT_FILENAME,
    publish_extraction_contract_once,
)
from dki.nlp.cmeie_preflight_protocol import (
    validate_cmeie_npu_preflight_receipt,
)
from dki.nlp.cmeie_overfit_protocol import (
    canonical_overfit_json_bytes,
    validate_overfit_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
VOCABULARY = {"[PAD]": 0, "[UNK]": 1, "药": 2, "病": 3}
ENTITY_LABELS = ["subject:疾病", "object:药物"]
PREDICATES = ["治疗", "预防"]
SCHEMA_CONSTRAINTS = {
    "治疗": [("subject:疾病", "object:药物")],
    "预防": [("subject:疾病", "object:药物")],
}
SOURCE_FIELDS = {
    "training_script_sha256": "scripts/train_spo_cmeie.py",
    "evaluator_sha256": "scripts/eval_spo_cmeie.py",
    "model_module_sha256": "dki/nlp/cmeie_gplinker.py",
    "alignment_module_sha256": "dki/nlp/cmeie_spo.py",
    "globalpointer_module_sha256": "dki/nlp/globalpointer.py",
    "artifact_integrity_module_sha256": "dki/nlp/cmeie_artifacts.py",
    "extraction_contract_module_sha256": (
        "dki/nlp/cmeie_extraction_contract.py"
    ),
    "selector_sha256": "scripts/select_spo_cmeie_checkpoint.py",
    "selection_protocol_sha256": "dki/nlp/cmeie_selection_protocol.py",
    "run_protocol_sha256": "dki/nlp/cmeie_run_protocol.py",
    "preflight_protocol_sha256": "dki/nlp/cmeie_preflight_protocol.py",
    "split_protocol_sha256": "dki/nlp/cmeie_split_protocol.py",
    "overfit_protocol_sha256": "dki/nlp/cmeie_overfit_protocol.py",
    "smoke_script_sha256": "scripts/smoke_cmeie_gplinker_overfit.py",
}
TREE_HASH_ALGORITHM = (
    "sha256 of sorted UTF-8 path NUL decimal-size NUL "
    "lowercase-sha256(file bytes) LF"
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _tree_sha(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["path"]):
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _metric(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _base_model_tree(base_model: Path) -> dict[str, Any]:
    files = {
        "config.json": b"{}\n",
        "materialization_receipt.json": _canonical(
            {"schema_version": 1, "materialized": True}
        ),
    }
    entries = []
    for relative, raw in sorted(files.items()):
        (base_model / relative).write_bytes(raw)
        entries.append(
            {
                "path": relative,
                "size": len(raw),
                "sha256": _sha(raw),
            }
        )
    return {
        "root": str(base_model),
        "algorithm": TREE_HASH_ALGORITHM,
        "sha256": _tree_sha(entries),
        "n_files": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "entries": entries,
    }


def _strict_preflight_payload(
    binding: run_protocol.RunBinding,
    paths: run_protocol.CmeieRunPaths,
    *,
    train_sha256: str,
    dev_sha256: str,
    model_tree: dict[str, Any],
) -> dict[str, Any]:
    def cblue(rows: int, sha256: str) -> dict[str, Any]:
        return {
            "exists": True,
            "rows": rows,
            "expected_rows": rows,
            "sha256": sha256,
        }

    preflight_tree_algorithm = (
        "sha256 of sorted path NUL size NUL file_sha256 LF"
    )
    return {
        "schema_version": 2,
        "purpose": "preflight_only_no_acl_model_load",
        "root": str(paths.run_root / "cblue"),
        "cblue": {
            "CMeEE-V2/train.jsonl": cblue(15000, "9" * 64),
            "CMeEE-V2/dev.jsonl": cblue(5000, "a" * 64),
            "CMeIE-V2/train.jsonl": cblue(14339, train_sha256),
            "CMeIE-V2/dev.jsonl": cblue(3585, dev_sha256),
        },
        "required_artifacts": [],
        "cmeee_checkpoint": None,
        "om_manifest": None,
        "cmeie_binding": {
            "complete_arguments": True,
            "valid": True,
            "source_root": str(binding.source_root),
            "source_tree": {
                "valid": True,
                "requested_source_tree_sha256": (
                    binding.included_tree_sha256
                ),
                "manifest_path": str(binding.source_manifest),
                "root": str(binding.source_root),
                "preflight_script": {
                    "path": str(
                        binding.source_root
                        / "scripts"
                        / "remote_npu_preflight.py"
                    ),
                    "size": 32768,
                    "sha256": "5" * 64,
                },
                "manifest_size": binding.source_manifest_size,
                "manifest_sha256": binding.source_manifest_sha256,
                "source_tree_sha256": binding.included_tree_sha256,
                "manifest_schema_version": 1,
                "manifest_rid": binding.rid,
                "manifest_git_head": binding.git_head,
                "manifest_algorithm": preflight_tree_algorithm,
                "file_count": len(
                    json.loads(
                        binding.source_manifest.read_text(encoding="utf-8")
                    )["entries"]
                ),
            },
            "cmeie_base_model": {
                "valid": True,
                "model_dir": str(paths.base_model),
                "algorithm": preflight_tree_algorithm,
                "file_count": model_tree["n_files"],
                "tree_sha256": model_tree["sha256"],
                "files": copy.deepcopy(model_tree["entries"]),
            },
        },
        "runtime": {
            "python": "3.11.6",
            "platform": "Linux-aarch64",
            "torch": {"available": True, "version": "2.7.1"},
            "torch_npu": {"available": True, "version": "2.7.1.post6"},
            "transformers": {"available": True, "version": "4.48.0"},
            "torch_npu_available": True,
            "torch_npu_probe": {
                "npu_available": True,
                "probe": {
                    "available": True,
                    "command": ["/usr/bin/python", "-c", "import torch_npu"],
                    "returncode": 0,
                    "stdout": '{"npu_available": true}\n',
                    "stderr": "",
                },
            },
            "npu_smi": {
                "available": True,
                "command": ["npu-smi", "info"],
                "returncode": 0,
                "stdout": "NPU 0 OK\n",
                "stderr": "",
            },
            "atc": {
                "available": False,
                "command": ["atc", "--version"],
            },
            "ascend_env_present": {
                "ASCEND_HOME_PATH": True,
                "ASCEND_OPP_PATH": True,
                "LD_LIBRARY_PATH": True,
            },
        },
        "strict": True,
        "ready_for_explicit_npu_runs": True,
        "npu_work_not_performed": [
            "ACL initialization",
            "OM model load",
            "OM inference",
            "training",
            "benchmarking",
        ],
    }


def _make_epoch(
    candidate: Path,
    epoch: int,
    metric: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    directory = candidate / f"epoch-{epoch:02d}"
    directory.mkdir(parents=True)
    (directory / "gplinker.pt").write_bytes(
        f"checkpoint-{epoch}".encode("ascii")
    )
    (directory / "config.json").write_text(
        json.dumps({"hidden_size": epoch}),
        encoding="utf-8",
    )
    (directory / "model.safetensors").write_bytes(
        f"encoder-{epoch}".encode("ascii")
    )
    (directory / "tokenizer.json").write_bytes(
        _canonical(
            {
                "version": "1.0",
                "added_tokens": [],
                "model": {
                    "type": "WordPiece",
                    "vocab": VOCABULARY,
                },
            }
        )
    )
    contract = publish_extraction_contract_once(
        directory,
        tokenizer_vocabulary=VOCABULARY,
        entity_labels=ENTITY_LABELS,
        predicates=PREDICATES,
        schema_constraints=SCHEMA_CONSTRAINTS,
    )
    bundle = publish_checkpoint_bundle_manifest(directory)
    return (
        {
            "epoch": epoch,
            **bundle,
            **contract,
            "calibration_declared_grid_selection": {
                "status": "completed",
                "thresholds": [0.0, 0.0, 0.0],
                "schema_constraint": "off",
                "grid_size": 54,
                "metric": copy.deepcopy(metric),
            },
        },
        contract,
    )


@dataclass
class CandidateFixture:
    root: Path
    binding: run_protocol.RunBinding
    candidate: Path
    receipt_path: Path
    attestation_path: Path
    selection_path: Path

    def receipt(self) -> dict[str, Any]:
        return json.loads(self.receipt_path.read_text(encoding="utf-8"))

    def write_receipt(
        self,
        value: dict[str, Any],
        *,
        refresh_attestation: bool = True,
    ) -> None:
        self.receipt_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if refresh_attestation:
            if self.attestation_path.exists():
                self.attestation_path.unlink()
            _publish_training_attestation(
                self.binding,
                self.candidate,
                value,
            )


def _publish_training_attestation(
    binding: run_protocol.RunBinding,
    candidate: Path,
    receipt: dict[str, Any],
) -> Path:
    epoch_fields = (
        "epoch",
        "checkpoint_bundle_manifest",
        "checkpoint_bundle_manifest_sha256",
        "checkpoint_bundle_tree_sha256",
        "checkpoint_bundle_n_files",
        "checkpoint_bundle_total_bytes",
        "checkpoint_sha256",
        "extraction_contract",
        "extraction_contract_sha256",
        "tokenizer_vocabulary_sha256",
        "label_table_sha256",
        "schema_constraints_sha256",
        "semantic_contract_sha256",
    )
    configuration = {
        "seed": receipt["seed"],
        "epochs": receipt["epochs"],
        "batch_size": receipt["batch_size"],
        "gradient_accumulation": receipt["gradient_accumulation"],
        "calibration_batch_size": receipt["calibration_batch_size"],
        "max_length": receipt["max_length"],
        "fp16": receipt["fp16"],
        "device": "npu",
        "pointer_variant": receipt["training_protocol"][
            "pointer_variant"
        ],
        "head_reduction": receipt["training_protocol"]["head_reduction"],
        "encoder_learning_rate": receipt["training_protocol"][
            "joint_training"
        ]["encoder_learning_rate"],
        "head_learning_rate": receipt["training_protocol"][
            "joint_training"
        ]["head_learning_rate"],
        "head_warmup_epochs": receipt["training_protocol"][
            "head_only_warmup"
        ]["epochs"],
        "head_warmup_learning_rate": receipt["training_protocol"][
            "head_only_warmup"
        ]["head_learning_rate"],
        "joint_head_learning_rate_scale": receipt["training_protocol"][
            "joint_training"
        ]["head_learning_rate_scale"],
        "encoder_gradient_clip": receipt["training_protocol"][
            "joint_training"
        ]["encoder_gradient_clip"],
        "head_gradient_clip": receipt["training_protocol"][
            "joint_training"
        ]["head_gradient_clip"],
        "encoder_mode": receipt["training_protocol"][
            "joint_training"
        ]["encoder_mode"],
    }
    evidence = {
        "training_receipt_sha256": _sha(
            (candidate / "training_receipt.json").read_bytes()
        ),
        "epoch_bindings": [
            {
                field: copy.deepcopy(item[field])
                for field in epoch_fields
                if field in item
            }
            for item in receipt["history"]
        ],
        "preflight_common": copy.deepcopy(receipt["preflight_common"]),
        "source_binding": {
            "rid": binding.rid,
            "git_head": binding.git_head,
            "source_tree_sha256": binding.included_tree_sha256,
            "source_manifest_sha256": binding.source_manifest_sha256,
            "source_manifest_path": str(binding.source_manifest),
        },
        "data_binding": {
            field: receipt[field]
            for field in (
                "train_sha256",
                "source_train_sha256",
                "fit_shard_sha256",
                "calibration_shard_sha256",
                "split_manifest_sha256",
            )
        },
        "configuration": configuration,
        "overfit_preflight_sha256": receipt["overfit_preflight"][
            "path_sha256"
        ],
    }
    return run_protocol.publish_training_attestation_once(
        binding,
        receipt["seed"],
        evidence,
    )


def make_run_candidate(
    tmp_path: Path,
    *,
    seed: int = 42,
) -> CandidateFixture:
    source_entries = []
    source_bytes: dict[str, bytes] = {}
    for relative in sorted(set(SOURCE_FIELDS.values())):
        raw = ROOT.joinpath(*relative.split("/")).read_bytes()
        source_bytes[relative] = raw
        source_entries.append(
            {
                "path": relative,
                "size": len(raw),
                "sha256": _sha(raw),
            }
        )
    tree = run_protocol.included_tree_sha256(source_entries)
    head = "30fce24" + ("a" * 33)
    rid = f"20260729T120000Z-g{head[:7]}-t{tree[:12]}"
    root = tmp_path / f"dki-agents-{rid}"
    source_root = root / "source" / "dki-agents"
    for relative, raw in source_bytes.items():
        path = source_root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    manifest_path = root / "evidence" / "source" / "included_tree.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(
        _canonical(
            {
                "schema_version": 1,
                "rid": rid,
                "git_head": head,
                "algorithm": run_protocol.SOURCE_MANIFEST_ALGORITHM,
                "included_tree_sha256": tree,
                "entries": source_entries,
            }
        )
    )
    binding = run_protocol.load_run_binding(root)
    paths = run_protocol.canonical_run_paths(binding)
    candidate = paths.seed_model(seed)

    history = []
    contracts = []
    for epoch, metric in (
        (1, _metric(0, 1, 1)),
        (2, _metric(1, 1, 1)),
        (3, _metric(2, 0, 1)),
    ):
        item, contract = _make_epoch(candidate, epoch, metric)
        history.append(item)
        contracts.append(contract)
    assert len({item["semantic_contract_sha256"] for item in contracts}) == 1

    base_model = paths.base_model
    base_model.mkdir(parents=True)
    model_tree = _base_model_tree(base_model)
    split_manifest_raw = _canonical(
        {"schema_version": 6, "kind": "test_split_manifest"}
    )
    fit_raw = b'{"text":"fit"}\n'
    calibration_raw = b'{"text":"calibration"}\n'
    paths.split_dir.mkdir(parents=True)
    (paths.split_dir / "split_manifest.json").write_bytes(
        split_manifest_raw
    )
    (paths.split_dir / "fit.jsonl").write_bytes(fit_raw)
    (paths.split_dir / "calibration.jsonl").write_bytes(calibration_raw)
    data = {
        "train_sha256": "1" * 64,
        "source_train_sha256": "1" * 64,
        "fit_shard_sha256": _sha(fit_raw),
        "calibration_shard_sha256": _sha(calibration_raw),
        "split_manifest_sha256": _sha(split_manifest_raw),
    }
    model = {
        "base_model": str(base_model),
        "base_model_tree": model_tree,
        "tokenizer_vocabulary_sha256": contracts[0][
            "tokenizer_vocabulary_sha256"
        ],
        "label_table_sha256": contracts[0]["label_table_sha256"],
        "schema_constraints_sha256": contracts[0][
            "schema_constraints_sha256"
        ],
    }
    paths.preflight_receipt.parent.mkdir(parents=True, exist_ok=True)
    dev_sha256 = "6" * 64
    paths.preflight_receipt.write_text(
        json.dumps(
            _strict_preflight_payload(
                binding,
                paths,
                train_sha256=data["source_train_sha256"],
                dev_sha256=dev_sha256,
                model_tree=model_tree,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    preflight_common = validate_cmeie_npu_preflight_receipt(
        str(paths.preflight_receipt),
        expected_source_tree_sha256=binding.included_tree_sha256,
        expected_source_manifest_sha256=binding.source_manifest_sha256,
        expected_cmeie_train_sha256=data["source_train_sha256"],
        expected_base_model_canonical_path=str(base_model),
        expected_base_model_tree_sha256=model_tree["sha256"],
        expected_run_binding={
            "rid": binding.rid,
            "git_head": binding.git_head,
            "source_tree_sha256": binding.included_tree_sha256,
            "source_manifest_sha256": binding.source_manifest_sha256,
        },
    )
    overfit_payload = {
        "schema_version": 1,
        "kind": "cmeie_gplinker_fit_only_overfit_preflight",
        "passed": True,
        "fit_only": True,
        "calibration_accessed": False,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "minimum_train_f1": 0.95,
        "sample_size": 8,
        "steps": 400,
        "max_length": 320,
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
        "source_tree_sha256": binding.included_tree_sha256,
        "source_tree_manifest_sha256": binding.source_manifest_sha256,
        "train_sha256": data["train_sha256"],
        "source_train_sha256": data["source_train_sha256"],
        "fit_shard_sha256": data["fit_shard_sha256"],
        "split_manifest_sha256": data["split_manifest_sha256"],
        "base_model": model["base_model"],
        "base_model_tree": model["base_model_tree"],
        "binding_algorithms": dict(BINDING_ALGORITHMS),
        "preflight_script_sha256": _sha(
            source_bytes["scripts/smoke_cmeie_gplinker_overfit.py"]
        ),
        "model_module_sha256": _sha(
            source_bytes["dki/nlp/cmeie_gplinker.py"]
        ),
        "alignment_module_sha256": _sha(
            source_bytes["dki/nlp/cmeie_spo.py"]
        ),
        "globalpointer_module_sha256": _sha(
            source_bytes["dki/nlp/globalpointer.py"]
        ),
        "tokenizer_vocabulary_sha256": model[
            "tokenizer_vocabulary_sha256"
        ],
        "label_table_sha256": model["label_table_sha256"],
        "schema_constraints_sha256": model[
            "schema_constraints_sha256"
        ],
        "preflight_common": preflight_common,
        "declared_grid_result": {
            "declared_threshold_values": [-0.5, 0.0, 0.5],
            "thresholds": [0.0, 0.0, 0.0],
            "schema_constraint": "on",
            "grid_size": 54,
            "metric": _metric(98, 2, 2),
        },
    }
    overfit_payload["phase_grid_results"] = {
        "head_only": copy.deepcopy(
            overfit_payload["declared_grid_result"]
        ),
        "joint": copy.deepcopy(overfit_payload["declared_grid_result"]),
    }
    overfit_payload["phase_grid_results"]["joint"]["metric"] = {
        "tp": 90,
        "fp": 10,
        "fn": 10,
        "precision": 0.9,
        "recall": 0.9,
        "f1": 0.9,
    }
    overfit_payload["selected_phase"] = "head_only"
    overfit_raw = canonical_overfit_json_bytes(overfit_payload)
    paths.overfit.write_bytes(overfit_raw)
    preflight = validate_overfit_preflight(
        overfit_payload,
        receipt_sha256=_sha(overfit_raw),
        source_train_sha256=data["source_train_sha256"],
        fit_shard_sha256=data["fit_shard_sha256"],
        split_manifest_sha256=data["split_manifest_sha256"],
        base_model=model["base_model"],
        base_model_tree=model["base_model_tree"],
        source_tree_manifest_sha256=binding.source_manifest_sha256,
        preflight_script_sha256=overfit_payload[
            "preflight_script_sha256"
        ],
        model_module_sha256=overfit_payload["model_module_sha256"],
        alignment_module_sha256=overfit_payload[
            "alignment_module_sha256"
        ],
        globalpointer_module_sha256=overfit_payload[
            "globalpointer_module_sha256"
        ],
        tokenizer_vocabulary_sha256=model[
            "tokenizer_vocabulary_sha256"
        ],
        label_table_sha256=model["label_table_sha256"],
        schema_constraints_sha256=model[
            "schema_constraints_sha256"
        ],
        source_tree_sha256=binding.included_tree_sha256,
        preflight_common=preflight_common,
    )
    receipt = {
        "schema_version": 7,
        "seed": seed,
        "source_tree_sha256": binding.included_tree_sha256,
        "source_tree_manifest_sha256": binding.source_manifest_sha256,
        **data,
        **model,
        "max_length": 320,
        "parameter_count": 102_000_000,
        "binding_algorithms": dict(BINDING_ALGORITHMS),
        "overfit_preflight": preflight,
        "preflight_common": preflight_common,
        "epochs": 8,
        "batch_size": 8,
        "gradient_accumulation": 4,
        "calibration_batch_size": 8,
        "fp16": True,
        "device": "npu:0",
        "training_protocol": {
            "pointer_variant": "efficient",
            "head_reduction": "balanced",
            "head_only_warmup": {
                "epochs": 2,
                "head_learning_rate": 1e-4,
            },
            "joint_training": {
                "encoder_learning_rate": 2e-5,
                "head_learning_rate": 1e-5,
                "head_learning_rate_scale": 0.1,
                "encoder_gradient_clip": 1.0,
                "head_gradient_clip": 5.0,
                "encoder_mode": "eval_with_gradients",
            },
        },
        "history": history,
        **{
            field: _sha(source_bytes[relative])
            for field, relative in SOURCE_FIELDS.items()
            if field
            not in {
                "selector_sha256",
                "selection_protocol_sha256",
                "run_protocol_sha256",
            }
        },
    }
    receipt_path = candidate / "training_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    attestation_path = _publish_training_attestation(
        binding,
        candidate,
        receipt,
    )
    return CandidateFixture(
        root=root,
        binding=binding,
        candidate=candidate,
        receipt_path=receipt_path,
        attestation_path=attestation_path,
        selection_path=protocol.canonical_selection_path(binding, seed),
    )


def _publish(fixture: CandidateFixture) -> dict[str, Any]:
    payload = protocol.compute_canonical_selection(
        fixture.candidate,
        run_binding=fixture.binding,
    )
    protocol.publish_canonical_selection_once(
        fixture.selection_path,
        payload,
        run_binding=fixture.binding,
    )
    return payload


def _rewrite_attestation(
    fixture: CandidateFixture,
    payload: dict[str, Any],
    *,
    recompute_evidence_digest: bool = False,
) -> None:
    if recompute_evidence_digest:
        evidence = {
            field: payload[field]
            for field in (
                "training_receipt_sha256",
                "epoch_bindings",
                "preflight_common",
                "source_binding",
                "data_binding",
                "configuration",
                "overfit_preflight_sha256",
            )
        }
        payload["training_evidence_sha256"] = _sha(_canonical(evidence))
    fixture.attestation_path.write_bytes(_canonical(payload))


def _forge_overfit_with_embedded_sha(
    fixture: CandidateFixture,
    payload: dict[str, Any],
    *,
    receipt_binding_updates: dict[str, Any] | None = None,
) -> None:
    raw = canonical_overfit_json_bytes(payload)
    paths = run_protocol.canonical_run_paths(fixture.binding)
    paths.overfit.write_bytes(raw)
    receipt = fixture.receipt()
    receipt["overfit_preflight"]["path_sha256"] = _sha(raw)
    if receipt_binding_updates:
        receipt["overfit_preflight"].update(receipt_binding_updates)
    fixture.write_receipt(receipt)


def test_compute_publish_and_reload_recompute_complete_selection(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    payload = _publish(fixture)

    assert payload["schema_version"] == 3
    assert payload["selected_epoch"] == 3
    assert payload["rid"] == fixture.binding.rid
    assert payload["head"] == fixture.binding.git_head
    assert payload["tree"] == fixture.binding.included_tree_sha256
    assert payload["bundle_binding"]["checkpoint_sha256"] == (
        payload["checkpoint_sha256"]
    )
    assert payload["contract_binding"]["semantic_contract_sha256"] == (
        payload["semantic_contract_sha256"]
    )
    assert payload["training_attestation_path"] == str(
        fixture.attestation_path
    )
    assert payload["training_attestation_sha256"] == _sha(
        fixture.attestation_path.read_bytes()
    )
    assert payload["training_attestation_binding"][
        "split_artifact_binding"
    ] == payload["split_artifact_binding"]
    assert payload["canonical_overfit_preflight_path"] == str(
        run_protocol.canonical_run_paths(fixture.binding).overfit
    )
    assert payload["canonical_overfit_preflight_sha256"] == (
        payload["canonical_overfit_validated_binding"]["path_sha256"]
    )
    assert fixture.selection_path.read_bytes() == _canonical(payload)
    assert (
        protocol.load_and_validate_selection_receipt(
            fixture.selection_path,
            run_binding=fixture.binding,
        )
        == payload
    )
    with pytest.raises(FileExistsError, match="output exists"):
        protocol.publish_canonical_selection_once(
            fixture.selection_path,
            payload,
            run_binding=fixture.binding,
        )


def test_each_seed_has_a_distinct_canonical_create_once_selection(
    tmp_path: Path,
):
    fixture = make_run_candidate(tmp_path)
    candidates = {42: fixture.candidate}
    for seed in (43, 44):
        candidate = run_protocol.canonical_run_paths(
            fixture.binding
        ).seed_model(seed)
        shutil.copytree(fixture.candidate, candidate)
        receipt_path = candidate / "training_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["seed"] = seed
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _publish_training_attestation(
            fixture.binding,
            candidate,
            receipt,
        )
        candidates[seed] = candidate

    published = []
    for seed in (42, 43, 44):
        candidate = candidates[seed]
        payload = protocol.compute_canonical_selection(
            candidate,
            run_binding=fixture.binding,
        )
        output = protocol.canonical_selection_path(
            fixture.binding,
            seed,
        )
        protocol.publish_canonical_selection_once(
            output,
            payload,
            run_binding=fixture.binding,
        )
        published.append(output)

    assert len(set(published)) == 3
    assert [path.parent.name for path in published] == [
        "seed-42",
        "seed-43",
        "seed-44",
    ]
    assert all(path.name == "selection.json" for path in published)
    assert all(path.is_file() for path in published)


def test_full_tie_break_is_f1_precision_then_earliest_epoch(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    receipt = fixture.receipt()
    receipt["history"][0]["calibration_declared_grid_selection"][
        "metric"
    ] = _metric(1, 1, 1)
    receipt["history"][1]["calibration_declared_grid_selection"][
        "metric"
    ] = _metric(1, 0, 2)
    receipt["history"][2]["calibration_declared_grid_selection"][
        "metric"
    ] = _metric(1, 0, 2)
    fixture.write_receipt(receipt)

    payload = protocol.compute_canonical_selection(
        fixture.candidate,
        run_binding=fixture.binding,
    )
    assert payload["selected_epoch"] == 2
    assert payload["calibration_metric"] == _metric(1, 0, 2)
    assert payload["tie_break"] == [
        "f1_desc",
        "precision_desc",
        "epoch_asc",
    ]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("thresholds", [0.5, 0.0, 0.0]),
        ("schema_constraint", "on"),
        ("calibration_metric", _metric(1, 1, 1)),
        ("selected_epoch", 2),
        ("training_receipt_sha256", "0" * 64),
    ],
)
def test_loader_rejects_manual_selection_field_tampering(
    tmp_path: Path,
    field: str,
    replacement: object,
):
    fixture = make_run_candidate(tmp_path)
    payload = _publish(fixture)
    payload[field] = replacement
    fixture.selection_path.write_bytes(_canonical(payload))

    with pytest.raises(ValueError, match="canonical recomputation"):
        protocol.load_and_validate_selection_receipt(
            fixture.selection_path,
            run_binding=fixture.binding,
        )


def test_candidate_copy_replay_is_rejected_before_receipt_trust(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    replay = tmp_path / "copied-candidate"
    shutil.copytree(fixture.candidate, replay)

    with pytest.raises(ValueError, match=r"seed_model\(seed\)"):
        protocol.compute_canonical_selection(
            replay,
            run_binding=fixture.binding,
        )


def test_all_completed_epoch_bundles_are_verified_not_only_winner(
    tmp_path: Path,
):
    fixture = make_run_candidate(tmp_path)
    (fixture.candidate / "epoch-01" / "model.safetensors").write_bytes(
        b"tampered"
    )

    with pytest.raises(ValueError, match="epoch 1.*bundle verification"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_contract_semantic_binding_is_recomputed_after_valid_bundle_rewrite(
    tmp_path: Path,
):
    fixture = make_run_candidate(tmp_path)
    epoch = fixture.candidate / "epoch-01"
    old_contract = fixture.receipt()["history"][0][
        "semantic_contract_sha256"
    ]
    (epoch / "checkpoint_bundle.json").unlink()
    (epoch / EXTRACTION_CONTRACT_FILENAME).unlink()
    publish_extraction_contract_once(
        epoch,
        tokenizer_vocabulary=VOCABULARY,
        entity_labels=["subject:疾病", "object:治疗方法"],
        predicates=PREDICATES,
        schema_constraints={
            predicate: [
                ("subject:疾病", "object:治疗方法")
            ]
            for predicate in PREDICATES
        },
    )
    bundle = publish_checkpoint_bundle_manifest(epoch)
    receipt = fixture.receipt()
    receipt["history"][0].update(bundle)
    assert receipt["history"][0]["semantic_contract_sha256"] == old_contract
    fixture.write_receipt(receipt)

    with pytest.raises(
        ValueError,
        match="extraction contract verification failed",
    ):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_metric_must_be_recomputable_from_counts(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    receipt = fixture.receipt()
    receipt["history"][2]["calibration_declared_grid_selection"]["metric"][
        "f1"
    ] = 0.99
    fixture.write_receipt(receipt)

    with pytest.raises(ValueError, match="disagrees with tp/fp/fn"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_missing_new_contract_field_names_the_training_contract(
    tmp_path: Path,
):
    fixture = make_run_candidate(tmp_path)
    receipt = fixture.receipt()
    del receipt["history"][1]["semantic_contract_sha256"]
    fixture.write_receipt(receipt)

    with pytest.raises(
        ValueError,
        match="history lacks attested epoch fields: "
        "semantic_contract_sha256",
    ):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_training_receipt_duplicate_key_is_rejected(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    text = fixture.receipt_path.read_text(encoding="utf-8")
    fixture.receipt_path.write_text(
        text.replace(
            '"schema_version": 7,',
            '"schema_version": 7, "schema_version": 7,',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_selection_receipt_duplicate_key_is_rejected(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    _publish(fixture)
    text = fixture.selection_path.read_text(encoding="utf-8")
    fixture.selection_path.write_text(
        text.replace(
            '"schema_version":3,',
            '"schema_version":3,"schema_version":3,',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        protocol.load_and_validate_selection_receipt(
            fixture.selection_path,
            run_binding=fixture.binding,
        )


def test_training_receipt_toctou_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = make_run_candidate(tmp_path)
    original = protocol.verify_checkpoint_bundle
    changed = False

    def racing_verifier(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            changed = True
            receipt = fixture.receipt()
            receipt["raced_after_bound_read"] = True
            fixture.write_receipt(receipt)
        return result

    monkeypatch.setattr(protocol, "verify_checkpoint_bundle", racing_verifier)
    with pytest.raises(ValueError, match="changed during canonical selection"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_selection_receipt_toctou_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = make_run_candidate(tmp_path)
    _publish(fixture)
    original = protocol.compute_canonical_selection
    changed = False

    def racing_recomputation(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            changed = True
            payload = json.loads(
                fixture.selection_path.read_text(encoding="utf-8")
            )
            payload["selection_used"] = False
            fixture.selection_path.write_bytes(_canonical(payload))
        return result

    monkeypatch.setattr(
        protocol,
        "compute_canonical_selection",
        racing_recomputation,
    )
    with pytest.raises(ValueError, match="changed during canonical selection"):
        protocol.load_and_validate_selection_receipt(
            fixture.selection_path,
            run_binding=fixture.binding,
        )


def test_loader_rejects_noncanonical_path_even_with_valid_payload(
    tmp_path: Path,
):
    fixture = make_run_candidate(tmp_path)
    payload = protocol.compute_canonical_selection(
        fixture.candidate,
        run_binding=fixture.binding,
    )
    wrong = tmp_path / "selection-copy.json"
    wrong.write_bytes(_canonical(payload))

    with pytest.raises(ValueError, match="canonical seed selection path"):
        protocol.load_and_validate_selection_receipt(
            wrong,
            run_binding=fixture.binding,
        )


def test_selection_requires_canonical_external_training_attestation(
    tmp_path: Path,
):
    fixture = make_run_candidate(tmp_path)
    fixture.attestation_path.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="training attestation.*does not exist",
    ):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_training_attestation_hardlink_alias_is_rejected(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    alias = tmp_path / "training-attestation-alias.json"
    os.link(fixture.attestation_path, alias)

    with pytest.raises(ValueError, match="multiple hard links"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_training_attestation_core_path_tampering_is_rejected(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    payload = json.loads(
        fixture.attestation_path.read_text(encoding="utf-8")
    )
    payload["candidate_directory"] = str(tmp_path / "copied-candidate")
    _rewrite_attestation(fixture, payload)

    with pytest.raises(ValueError, match="run binding mismatch.*candidate"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_training_attestation_evidence_digest_tampering_is_rejected(
    tmp_path: Path,
):
    fixture = make_run_candidate(tmp_path)
    payload = json.loads(
        fixture.attestation_path.read_text(encoding="utf-8")
    )
    payload["data_binding"]["fit_shard_sha256"] = "0" * 64
    _rewrite_attestation(fixture, payload)

    with pytest.raises(ValueError, match="evidence digest is inconsistent"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_attested_training_receipt_sha_is_independently_checked(
    tmp_path: Path,
):
    fixture = make_run_candidate(tmp_path)
    payload = json.loads(
        fixture.attestation_path.read_text(encoding="utf-8")
    )
    payload["training_receipt_sha256"] = "0" * 64
    _rewrite_attestation(
        fixture,
        payload,
        recompute_evidence_digest=True,
    )

    with pytest.raises(ValueError, match="training receipt SHA-256 mismatch"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_strict_preflight_is_fully_revalidated_not_only_hash_compared(
    tmp_path: Path,
):
    fixture = make_run_candidate(tmp_path)
    path = run_protocol.canonical_run_paths(
        fixture.binding
    ).preflight_receipt
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strict"] = False
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"receipt\.strict"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_canonical_split_shard_tampering_is_rejected(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    fit = (
        run_protocol.canonical_run_paths(fixture.binding).split_dir
        / "fit.jsonl"
    )
    fit.write_bytes(b'{"text":"tampered"}\n')

    with pytest.raises(ValueError, match="fit_shard binding differs"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_candidate_extra_path_is_rejected(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    (fixture.candidate / "unrecorded.bin").write_bytes(b"extra")

    with pytest.raises(ValueError, match="extra path"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_candidate_checkpoint_hardlink_alias_is_rejected(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    checkpoint = fixture.candidate / "epoch-01" / "gplinker.pt"
    os.link(checkpoint, tmp_path / "checkpoint-alias.pt")

    with pytest.raises(ValueError, match="checkpoint path has multiple"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_selection_reads_canonical_overfit_before_candidate(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    paths = run_protocol.canonical_run_paths(fixture.binding)
    paths.overfit.unlink()
    (fixture.candidate / "unrecorded.bin").write_bytes(b"extra")

    with pytest.raises(
        FileNotFoundError,
        match="overfit preflight.*does not exist",
    ):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_selection_rejects_canonical_overfit_hardlink(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    path = run_protocol.canonical_run_paths(fixture.binding).overfit
    os.link(path, tmp_path / "overfit-alias.json")

    with pytest.raises(ValueError, match="multiple hard links"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_selection_rejects_noncanonical_overfit_json(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    path = run_protocol.canonical_run_paths(fixture.binding).overfit
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not canonical JSON"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_selection_rejects_module_hash_tamper_even_with_embedded_sha_forged(
    tmp_path: Path,
):
    fixture = make_run_candidate(tmp_path)
    path = run_protocol.canonical_run_paths(fixture.binding).overfit
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model_module_sha256"] = "0" * 64
    _forge_overfit_with_embedded_sha(
        fixture,
        payload,
        receipt_binding_updates={"model_module_sha256": "0" * 64},
    )

    with pytest.raises(ValueError, match="model_module_sha256"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_selection_rejects_schema_off_overfit_even_with_sha_forged(
    tmp_path: Path,
):
    fixture = make_run_candidate(tmp_path)
    path = run_protocol.canonical_run_paths(fixture.binding).overfit
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["declared_grid_result"]["schema_constraint"] = "off"
    _forge_overfit_with_embedded_sha(fixture, payload)

    with pytest.raises(ValueError, match="schema constraint on"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_selection_rejects_low_count_derived_overfit_f1(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    path = run_protocol.canonical_run_paths(fixture.binding).overfit
    payload = json.loads(path.read_text(encoding="utf-8"))
    low = _metric(94, 6, 6)
    payload["declared_grid_result"]["metric"] = low
    _forge_overfit_with_embedded_sha(
        fixture,
        payload,
        receipt_binding_updates={"metric": low},
    )

    with pytest.raises(ValueError, match="count-derived overfit F1"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )


def test_selection_rejects_forged_f1_despite_embedded_sha(tmp_path: Path):
    fixture = make_run_candidate(tmp_path)
    path = run_protocol.canonical_run_paths(fixture.binding).overfit
    payload = json.loads(path.read_text(encoding="utf-8"))
    forged = {
        "tp": 0,
        "fp": 100,
        "fn": 100,
        "precision": 0.98,
        "recall": 0.98,
        "f1": 0.98,
    }
    payload["declared_grid_result"]["metric"] = forged
    _forge_overfit_with_embedded_sha(
        fixture,
        payload,
        receipt_binding_updates={"metric": forged},
    )

    with pytest.raises(ValueError, match="disagrees with tp/fp/fn"):
        protocol.compute_canonical_selection(
            fixture.candidate,
            run_binding=fixture.binding,
        )
