from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import dki.nlp.cmeie_preflight_protocol as protocol
from dki.nlp.cmeie_preflight_protocol import (
    validate_cmeie_npu_preflight_receipt,
)


SOURCE_TREE_SHA = "a" * 64
SOURCE_MANIFEST_SHA = "c" * 64
TRAIN_SHA = "d" * 64
DEV_SHA = "e" * 64
SCRIPT_SHA = "f" * 64
GIT_HEAD = "b" * 40
RID = f"20260729T104110Z-g{GIT_HEAD[:7]}-t{SOURCE_TREE_SHA[:12]}"
MODEL_PATH = "/srv/models/chinese-roberta-wwm-ext"
TREE_ALGORITHM = "sha256 of sorted path NUL size NUL file_sha256 LF"


def _tree_sha256(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


MODEL_FILES = [
    {
        "path": "config.json",
        "size": 41,
        "sha256": "1" * 64,
    },
    {
        "path": "materialization_receipt.json",
        "size": 97,
        "sha256": "2" * 64,
    },
    {
        "path": "tokenizer.json",
        "size": 131,
        "sha256": "3" * 64,
    },
]
MODEL_TREE_SHA = _tree_sha256(MODEL_FILES)


def _cblue_item(rows: int, sha256: str) -> dict[str, Any]:
    return {
        "exists": True,
        "rows": rows,
        "expected_rows": rows,
        "sha256": sha256,
    }


def _valid_receipt() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "purpose": "preflight_only_no_acl_model_load",
        "root": "/srv/cblue-checkout",
        "cblue": {
            "CMeEE-V2/train.jsonl": _cblue_item(15000, "4" * 64),
            "CMeEE-V2/dev.jsonl": _cblue_item(5000, "5" * 64),
            "CMeIE-V2/train.jsonl": _cblue_item(14339, TRAIN_SHA),
            "CMeIE-V2/dev.jsonl": _cblue_item(3585, DEV_SHA),
        },
        "required_artifacts": [],
        "cmeee_checkpoint": None,
        "om_manifest": None,
        "cmeie_binding": {
            "complete_arguments": True,
            "valid": True,
            "source_root": "/srv/run/source/dki-agents",
            "source_tree": {
                "valid": True,
                "requested_source_tree_sha256": SOURCE_TREE_SHA,
                "manifest_path": "/srv/run/evidence/included_tree.json",
                "root": "/srv/run/source/dki-agents",
                "preflight_script": {
                    "path": (
                        "/srv/run/source/dki-agents/"
                        "scripts/remote_npu_preflight.py"
                    ),
                    "size": 32768,
                    "sha256": SCRIPT_SHA,
                },
                "manifest_size": 8192,
                "manifest_sha256": SOURCE_MANIFEST_SHA,
                "source_tree_sha256": SOURCE_TREE_SHA,
                "manifest_schema_version": 1,
                "manifest_rid": RID,
                "manifest_git_head": GIT_HEAD,
                "manifest_algorithm": TREE_ALGORITHM,
                "file_count": 347,
            },
            "cmeie_base_model": {
                "valid": True,
                "model_dir": MODEL_PATH,
                "algorithm": TREE_ALGORITHM,
                "file_count": len(MODEL_FILES),
                "tree_sha256": MODEL_TREE_SHA,
                "files": copy.deepcopy(MODEL_FILES),
            },
        },
        "runtime": {
            "python": "3.11.10 (main, Jul 1 2026)",
            "platform": "Linux-5.10-aarch64-with-glibc2.28",
            "torch": {"available": True, "version": "2.7.1+cpu"},
            "torch_npu": {"available": True, "version": "2.7.1.post6"},
            "transformers": {"available": True, "version": "4.48.0"},
            "torch_npu_available": True,
            "torch_npu_probe": {
                "npu_available": True,
                "probe": {
                    "available": True,
                    "command": ["/usr/bin/python", "-c", "import torch_npu"],
                    "returncode": 0,
                    "stdout": (
                        "runtime diagnostic\n"
                        '{"npu_available": true}\n'
                    ),
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


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _validate(path: Path, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "expected_source_tree_sha256": SOURCE_TREE_SHA,
        "expected_source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "expected_cmeie_train_sha256": TRAIN_SHA,
        "expected_base_model_canonical_path": MODEL_PATH,
        "expected_base_model_tree_sha256": MODEL_TREE_SHA,
    }
    arguments.update(overrides)
    return validate_cmeie_npu_preflight_receipt(str(path), **arguments)


def test_valid_receipt_returns_descriptor_bound_common_binding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    receipt = _valid_receipt()
    _write_receipt(path, receipt)

    binding = _validate(
        path,
        expected_run_binding={
            "rid": RID,
            "git_head": GIT_HEAD,
            "source_tree_sha256": SOURCE_TREE_SHA,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        },
    )

    raw = path.read_bytes()
    assert binding == {
        "schema_version": 1,
        "purpose": "cmeie_npu_preflight_common_binding",
        "preflight_receipt_path": str(path.resolve()),
        "preflight_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "preflight_producer_script_sha256": SCRIPT_SHA,
        "source_tree_sha256": SOURCE_TREE_SHA,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "cmeie_train_rows": 14339,
        "cmeie_train_sha256": TRAIN_SHA,
        "cmeie_dev_rows": 3585,
        "cmeie_dev_sha256": DEV_SHA,
        "base_model_canonical_path": MODEL_PATH,
        "base_model_tree_sha256": MODEL_TREE_SHA,
        "base_model_materialization_receipt_sha256": "2" * 64,
        "runtime_fingerprint_canonical_sha256": hashlib.sha256(
            json.dumps(
                receipt["runtime"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "run_binding": {
            "rid": RID,
            "git_head": GIT_HEAD,
            "source_tree_sha256": SOURCE_TREE_SHA,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        },
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value.__setitem__("schema_version", True),
            "schema_version",
        ),
        (
            lambda value: value.__setitem__("schema_version", 1),
            "schema_version",
        ),
        (
            lambda value: value.__setitem__("purpose", "training"),
            "purpose",
        ),
        (
            lambda value: value.__setitem__("strict", False),
            "receipt.strict",
        ),
        (
            lambda value: value.__setitem__(
                "ready_for_explicit_npu_runs", False
            ),
            "ready_for_explicit_npu_runs",
        ),
        (
            lambda value: value["cblue"]["CMeIE-V2/train.jsonl"].__setitem__(
                "rows", 14338
            ),
            "14339 rows",
        ),
        (
            lambda value: value["cblue"]["CMeIE-V2/train.jsonl"].__setitem__(
                "rows", True
            ),
            "booleans are not integers",
        ),
        (
            lambda value: value["cblue"]["CMeIE-V2/train.jsonl"].__setitem__(
                "sha256", "0" * 64
            ),
            "expected source training data",
        ),
        (
            lambda value: value["cblue"]["CMeIE-V2/dev.jsonl"].__setitem__(
                "rows", 3584
            ),
            "3585 rows",
        ),
        (
            lambda value: value["cblue"]["CMeIE-V2/dev.jsonl"].__setitem__(
                "sha256", "not-a-hash"
            ),
            "lowercase SHA-256",
        ),
        (
            lambda value: value["cmeie_binding"].__setitem__(
                "complete_arguments", False
            ),
            "complete_arguments",
        ),
        (
            lambda value: value["cmeie_binding"].__setitem__("valid", False),
            "cmeie_binding.valid",
        ),
        (
            lambda value: value["cmeie_binding"]["source_tree"].__setitem__(
                "manifest_sha256", "0" * 64
            ),
            "expected source manifest",
        ),
        (
            lambda value: value["cmeie_binding"]["source_tree"].__setitem__(
                "requested_source_tree_sha256", "0" * 64
            ),
            "source-tree hashes must agree",
        ),
        (
            lambda value: value["cmeie_binding"]["source_tree"][
                "preflight_script"
            ].__setitem__("sha256", "NaN"),
            "lowercase SHA-256",
        ),
        (
            lambda value: value["cmeie_binding"][
                "cmeie_base_model"
            ].__setitem__("valid", False),
            "cmeie_base_model.valid",
        ),
        (
            lambda value: value["cmeie_binding"]["cmeie_base_model"][
                "files"
            ].pop(1),
            "file_count",
        ),
        (
            lambda value: value["cmeie_binding"]["cmeie_base_model"][
                "files"
            ][0].__setitem__("size", True),
            "booleans are not integers",
        ),
        (
            lambda value: value["runtime"]["torch"].__setitem__(
                "available", False
            ),
            "torch.available",
        ),
        (
            lambda value: value["runtime"].__setitem__(
                "torch_npu_available", False
            ),
            "torch_npu_available",
        ),
        (
            lambda value: value["runtime"]["torch_npu_probe"][
                "probe"
            ].__setitem__("stdout", '{"npu_available": false}\n'),
            "contradicts",
        ),
        (
            lambda value: value["runtime"]["npu_smi"].__setitem__(
                "returncode", False
            ),
            "booleans are not integers",
        ),
        (
            lambda value: value["runtime"]["npu_smi"].__setitem__(
                "returncode", 1
            ),
            "return code 0",
        ),
        (
            lambda value: value["runtime"].__setitem__(
                "diagnostic_value", float("nan")
            ),
            "non-finite",
        ),
        (
            lambda value: value["runtime"].pop("transformers"),
            "missing required field",
        ),
    ],
)
def test_rejects_tampered_or_self_inconsistent_receipts(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    receipt = _valid_receipt()
    mutation(receipt)
    path = tmp_path / "receipt.json"
    _write_receipt(path, receipt)

    with pytest.raises(ValueError, match=match):
        _validate(path)


def test_rejects_wrong_expected_model_path(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    _write_receipt(path, _valid_receipt())

    with pytest.raises(ValueError, match="expected materialized model path"):
        _validate(
            path,
            expected_base_model_canonical_path="/srv/models/other-model",
        )


def test_rejects_wrong_expected_model_tree(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    _write_receipt(path, _valid_receipt())

    with pytest.raises(ValueError, match="model-tree hashes must agree"):
        _validate(path, expected_base_model_tree_sha256="0" * 64)


def test_rejects_recomputed_model_tree_mismatch(tmp_path: Path) -> None:
    receipt = _valid_receipt()
    receipt["cmeie_binding"]["cmeie_base_model"]["files"][0][
        "sha256"
    ] = "9" * 64
    path = tmp_path / "receipt.json"
    _write_receipt(path, receipt)

    with pytest.raises(ValueError, match="model-tree hashes must agree"):
        _validate(path)


def test_rejects_model_without_materialization_receipt(tmp_path: Path) -> None:
    receipt = _valid_receipt()
    model = receipt["cmeie_binding"]["cmeie_base_model"]
    model["files"][1]["path"] = "special_tokens_map.json"
    model["files"].sort(key=lambda item: item["path"])
    model["tree_sha256"] = _tree_sha256(model["files"])
    path = tmp_path / "receipt.json"
    _write_receipt(path, receipt)

    with pytest.raises(ValueError, match="materialization_receipt.json"):
        _validate(
            path,
            expected_base_model_tree_sha256=model["tree_sha256"],
        )


def test_rejects_run_binding_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    _write_receipt(path, _valid_receipt())

    with pytest.raises(ValueError, match="expected_run_binding.rid"):
        _validate(
            path,
            expected_run_binding={
                "rid": RID.replace("104110", "104111"),
                "git_head": GIT_HEAD,
                "source_tree_sha256": SOURCE_TREE_SHA,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA,
            },
        )


def test_duplicate_json_key_is_rejected_by_bound_loader(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    text = json.dumps(_valid_receipt(), ensure_ascii=False, indent=2)
    text = text.replace(
        '  "schema_version": 2,',
        '  "schema_version": 2,\n  "schema_version": 2,',
        1,
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key"):
        _validate(path)


def test_descriptor_bound_loader_is_called_once_and_errors_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "receipt.json"
    _write_receipt(path, _valid_receipt())
    calls = 0

    def changed_while_reading(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise ValueError("evidence file changed while reading")

    monkeypatch.setattr(protocol, "load_bound_json", changed_while_reading)
    with pytest.raises(ValueError, match="changed while reading"):
        _validate(path)
    assert calls == 1
