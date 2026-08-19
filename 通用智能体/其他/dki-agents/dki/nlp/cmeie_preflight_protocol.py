"""Strict consumer for CMeIE Ascend NPU preflight receipts.

The producer receipt is an authorization boundary: training code must not
silently reinterpret a partial or self-inconsistent preflight report as
permission to use an NPU.  This module therefore validates the complete CMeIE
binding needed by downstream runs and returns a small, hash-bound common
binding that can be copied into their own receipts.

The receipt pathname is opened exactly once through
``dki.evidence_io.load_bound_json``.  In particular, its SHA-256 is taken from
the same descriptor-bound read that supplied the parsed JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from dki.evidence_io import load_bound_json


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_HEAD_RE = re.compile(r"^[0-9a-f]{40}$")
_RID_RE = re.compile(
    r"^(?P<timestamp>[0-9]{8}T[0-9]{6}Z)-"
    r"g(?P<head>[0-9a-f]{7})-t(?P<tree>[0-9a-f]{12})$"
)
_TREE_ALGORITHM = "sha256 of sorted path NUL size NUL file_sha256 LF"
_PURPOSE = "preflight_only_no_acl_model_load"
_COMMON_PURPOSE = "cmeie_npu_preflight_common_binding"
_MATERIALIZATION_RECEIPT = "materialization_receipt.json"
_CBLUE_ROWS = {
    "CMeEE-V2/train.jsonl": 15000,
    "CMeEE-V2/dev.jsonl": 5000,
    "CMeIE-V2/train.jsonl": 14339,
    "CMeIE-V2/dev.jsonl": 3585,
}
_RUN_BINDING_KEYS = frozenset(
    {
        "rid",
        "git_head",
        "source_tree_sha256",
        "source_manifest_sha256",
    }
)


def _error(label: str, message: str) -> ValueError:
    return ValueError(f"{label}: {message}")


def _object(value: object, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise _error(label, "must be a JSON object")
    return value


def _array(value: object, *, label: str) -> list[Any]:
    if type(value) is not list:
        raise _error(label, "must be a JSON array")
    return value


def _required(mapping: Mapping[str, Any], key: str, *, label: str) -> Any:
    if key not in mapping:
        raise _error(label, f"missing required field {key!r}")
    return mapping[key]


def _true(value: object, *, label: str) -> None:
    if value is not True:
        raise _error(label, "must be the JSON boolean true")


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise _error(label, "must be a JSON boolean")
    return value


def _integer(
    value: object,
    *,
    label: str,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise _error(label, "must be an integer (booleans are not integers)")
    if minimum is not None and value < minimum:
        raise _error(label, f"must be at least {minimum}")
    return value


def _text(value: object, *, label: str, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise _error(label, "must be text")
    if nonempty and not value:
        raise _error(label, "must not be empty")
    if "\0" in value:
        raise _error(label, "must not contain a NUL byte")
    return value


def _sha256(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if _SHA256_RE.fullmatch(text) is None:
        raise _error(label, "must be a lowercase SHA-256")
    return text


def _git_head(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if _GIT_HEAD_RE.fullmatch(text) is None:
        raise _error(label, "must be a lowercase 40-character Git object ID")
    return text


def _absolute_canonical_path(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if "/" in text and "\\" in text:
        raise _error(label, "must not mix path separator styles")

    if "\\" in text or PureWindowsPath(text).drive:
        path = PureWindowsPath(text)
        if (
            not path.is_absolute()
            or any(part in {".", ".."} for part in path.parts)
            or str(path) != text
        ):
            raise _error(label, "must be an absolute canonical Windows path")
        return text

    path = PurePosixPath(text)
    if (
        not path.is_absolute()
        or text.startswith("//")
        or any(part in {".", ".."} for part in path.parts)
        or str(path) != text
    ):
        raise _error(label, "must be an absolute canonical POSIX path")
    return text


def _relative_posix_path(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    path = PurePosixPath(text)
    if (
        "\\" in text
        or path.is_absolute()
        or PureWindowsPath(text).drive
        or any(part in {"", ".", ".."} for part in text.split("/"))
        or str(path) != text
    ):
        raise _error(label, "must be a canonical POSIX relative path")
    return text


def _reject_nonfinite(value: object, *, label: str) -> None:
    """Reject JSON extensions such as NaN at every depth."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise _error(label, "contains a non-finite number")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise _error(label, "contains a non-text object key")
            _reject_nonfinite(child, label=f"{label}.{key}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _reject_nonfinite(child, label=f"{label}[{index}]")
        return
    if value is None or type(value) in {bool, int, str}:
        return
    raise _error(label, f"contains an unsupported value type {type(value).__name__}")


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _validate_command_report(
    value: object,
    *,
    label: str,
    require_available: bool,
    require_success: bool,
) -> dict[str, Any]:
    report = _object(value, label=label)
    available = _boolean(
        _required(report, "available", label=label),
        label=f"{label}.available",
    )
    command = _array(
        _required(report, "command", label=label),
        label=f"{label}.command",
    )
    if not command:
        raise _error(f"{label}.command", "must not be empty")
    for index, item in enumerate(command):
        _text(item, label=f"{label}.command[{index}]")

    if require_available and not available:
        raise _error(label, "reports that the executable is unavailable")
    if not available:
        if "returncode" in report:
            raise _error(label, "unavailable command must not report a return code")
        return report

    returncode = _integer(
        _required(report, "returncode", label=label),
        label=f"{label}.returncode",
    )
    _text(
        _required(report, "stdout", label=label),
        label=f"{label}.stdout",
        nonempty=False,
    )
    _text(
        _required(report, "stderr", label=label),
        label=f"{label}.stderr",
        nonempty=False,
    )
    if "error" in report:
        raise _error(label, "cannot report both command completion and an error")
    if require_success and returncode != 0:
        raise _error(label, "must have return code 0")
    return report


def _strict_embedded_json(line: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"invalid JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        line,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _validate_probe_stdout(stdout: str, *, label: str) -> None:
    for line in reversed(stdout.splitlines()):
        try:
            payload = _strict_embedded_json(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if type(payload) is dict and payload.get("npu_available") is True:
            return
        if type(payload) is dict and "npu_available" in payload:
            raise _error(label, "embedded NPU availability contradicts the receipt")
    raise _error(label, "does not contain a successful NPU availability result")


def _validate_package(value: object, *, label: str) -> str:
    package = _object(value, label=label)
    _true(
        _required(package, "available", label=label),
        label=f"{label}.available",
    )
    version = _text(
        _required(package, "version", label=label),
        label=f"{label}.version",
    )
    if version.strip().lower() == "unknown":
        raise _error(f"{label}.version", "must identify the installed version")
    if "error" in package:
        raise _error(label, "available package must not report an error")
    return version


def _validate_runtime(runtime_value: object) -> str:
    runtime = _object(runtime_value, label="receipt.runtime")
    _text(
        _required(runtime, "python", label="receipt.runtime"),
        label="receipt.runtime.python",
    )
    _text(
        _required(runtime, "platform", label="receipt.runtime"),
        label="receipt.runtime.platform",
    )
    for package_name in ("torch", "torch_npu", "transformers"):
        _validate_package(
            _required(runtime, package_name, label="receipt.runtime"),
            label=f"receipt.runtime.{package_name}",
        )

    _true(
        _required(runtime, "torch_npu_available", label="receipt.runtime"),
        label="receipt.runtime.torch_npu_available",
    )
    probe = _object(
        _required(runtime, "torch_npu_probe", label="receipt.runtime"),
        label="receipt.runtime.torch_npu_probe",
    )
    _true(
        _required(probe, "npu_available", label="receipt.runtime.torch_npu_probe"),
        label="receipt.runtime.torch_npu_probe.npu_available",
    )
    probe_command = _validate_command_report(
        _required(probe, "probe", label="receipt.runtime.torch_npu_probe"),
        label="receipt.runtime.torch_npu_probe.probe",
        require_available=True,
        require_success=True,
    )
    _validate_probe_stdout(
        probe_command["stdout"],
        label="receipt.runtime.torch_npu_probe.probe.stdout",
    )

    npu_smi = _validate_command_report(
        _required(runtime, "npu_smi", label="receipt.runtime"),
        label="receipt.runtime.npu_smi",
        require_available=True,
        require_success=True,
    )
    if npu_smi["command"][0] != "npu-smi":
        raise _error("receipt.runtime.npu_smi.command", "must invoke npu-smi")

    _validate_command_report(
        _required(runtime, "atc", label="receipt.runtime"),
        label="receipt.runtime.atc",
        require_available=False,
        require_success=False,
    )
    ascend_environment = _object(
        _required(runtime, "ascend_env_present", label="receipt.runtime"),
        label="receipt.runtime.ascend_env_present",
    )
    expected_environment_keys = {
        "ASCEND_HOME_PATH",
        "ASCEND_OPP_PATH",
        "LD_LIBRARY_PATH",
    }
    if set(ascend_environment) != expected_environment_keys:
        raise _error(
            "receipt.runtime.ascend_env_present",
            "must contain exactly the three producer environment flags",
        )
    for key in sorted(expected_environment_keys):
        _boolean(
            ascend_environment[key],
            label=f"receipt.runtime.ascend_env_present.{key}",
        )
    return _canonical_json_sha256(runtime)


def _validate_cblue(
    value: object,
    *,
    expected_train_sha256: str,
) -> tuple[str, str]:
    cblue = _object(value, label="receipt.cblue")
    if set(cblue) != set(_CBLUE_ROWS):
        raise _error(
            "receipt.cblue",
            "must contain exactly the four expected CBLUE data files",
        )
    hashes: dict[str, str] = {}
    for relative, expected_rows in _CBLUE_ROWS.items():
        label = f"receipt.cblue.{relative}"
        item = _object(cblue[relative], label=label)
        _true(_required(item, "exists", label=label), label=f"{label}.exists")
        rows = _integer(
            _required(item, "rows", label=label),
            label=f"{label}.rows",
            minimum=0,
        )
        claimed_expected_rows = _integer(
            _required(item, "expected_rows", label=label),
            label=f"{label}.expected_rows",
            minimum=0,
        )
        if rows != expected_rows or claimed_expected_rows != expected_rows:
            raise _error(label, f"must report exactly {expected_rows} rows")
        hashes[relative] = _sha256(
            _required(item, "sha256", label=label),
            label=f"{label}.sha256",
        )

    train_hash = hashes["CMeIE-V2/train.jsonl"]
    if train_hash != expected_train_sha256:
        raise _error(
            "receipt.cblue.CMeIE-V2/train.jsonl.sha256",
            "does not match the expected source training data",
        )
    return train_hash, hashes["CMeIE-V2/dev.jsonl"]


def _validate_auxiliary_ready_facts(receipt: Mapping[str, Any]) -> None:
    artifacts = _array(
        _required(receipt, "required_artifacts", label="receipt"),
        label="receipt.required_artifacts",
    )
    for index, value in enumerate(artifacts):
        label = f"receipt.required_artifacts[{index}]"
        item = _object(value, label=label)
        _text(_required(item, "path", label=label), label=f"{label}.path")
        _true(_required(item, "exists", label=label), label=f"{label}.exists")
        if "size" in item:
            _integer(item["size"], label=f"{label}.size", minimum=0)
        if "sha256" in item:
            _sha256(item["sha256"], label=f"{label}.sha256")

    checkpoint = _required(receipt, "cmeee_checkpoint", label="receipt")
    if checkpoint is not None:
        item = _object(checkpoint, label="receipt.cmeee_checkpoint")
        _true(
            _required(item, "complete", label="receipt.cmeee_checkpoint"),
            label="receipt.cmeee_checkpoint.complete",
        )
        errors = _array(
            _required(
                item,
                "validation_errors",
                label="receipt.cmeee_checkpoint",
            ),
            label="receipt.cmeee_checkpoint.validation_errors",
        )
        if errors:
            raise _error(
                "receipt.cmeee_checkpoint.validation_errors",
                "must be empty when the receipt is ready",
            )

    om_manifest = _required(receipt, "om_manifest", label="receipt")
    if om_manifest is not None:
        item = _object(om_manifest, label="receipt.om_manifest")
        _true(
            _required(item, "complete_arguments", label="receipt.om_manifest"),
            label="receipt.om_manifest.complete_arguments",
        )
        _true(
            _required(item, "valid", label="receipt.om_manifest"),
            label="receipt.om_manifest.valid",
        )


def _validate_source_tree(
    value: object,
    *,
    expected_tree_sha256: str,
    expected_manifest_sha256: str,
) -> tuple[dict[str, str], str]:
    source = _object(value, label="receipt.cmeie_binding.source_tree")
    _true(
        _required(source, "valid", label="receipt.cmeie_binding.source_tree"),
        label="receipt.cmeie_binding.source_tree.valid",
    )
    requested_tree = _sha256(
        _required(
            source,
            "requested_source_tree_sha256",
            label="receipt.cmeie_binding.source_tree",
        ),
        label="receipt.cmeie_binding.source_tree.requested_source_tree_sha256",
    )
    actual_tree = _sha256(
        _required(
            source,
            "source_tree_sha256",
            label="receipt.cmeie_binding.source_tree",
        ),
        label="receipt.cmeie_binding.source_tree.source_tree_sha256",
    )
    manifest_hash = _sha256(
        _required(
            source,
            "manifest_sha256",
            label="receipt.cmeie_binding.source_tree",
        ),
        label="receipt.cmeie_binding.source_tree.manifest_sha256",
    )
    if requested_tree != actual_tree or actual_tree != expected_tree_sha256:
        raise _error(
            "receipt.cmeie_binding.source_tree",
            "requested, verified, and expected source-tree hashes must agree",
        )
    if manifest_hash != expected_manifest_sha256:
        raise _error(
            "receipt.cmeie_binding.source_tree.manifest_sha256",
            "does not match the expected source manifest",
        )

    _absolute_canonical_path(
        _required(
            source,
            "manifest_path",
            label="receipt.cmeie_binding.source_tree",
        ),
        label="receipt.cmeie_binding.source_tree.manifest_path",
    )
    _absolute_canonical_path(
        _required(source, "root", label="receipt.cmeie_binding.source_tree"),
        label="receipt.cmeie_binding.source_tree.root",
    )
    _integer(
        _required(
            source,
            "manifest_size",
            label="receipt.cmeie_binding.source_tree",
        ),
        label="receipt.cmeie_binding.source_tree.manifest_size",
        minimum=1,
    )
    if (
        _integer(
            _required(
                source,
                "manifest_schema_version",
                label="receipt.cmeie_binding.source_tree",
            ),
            label="receipt.cmeie_binding.source_tree.manifest_schema_version",
        )
        != 1
    ):
        raise _error(
            "receipt.cmeie_binding.source_tree.manifest_schema_version",
            "must be 1",
        )
    if (
        _text(
            _required(
                source,
                "manifest_algorithm",
                label="receipt.cmeie_binding.source_tree",
            ),
            label="receipt.cmeie_binding.source_tree.manifest_algorithm",
        )
        != _TREE_ALGORITHM
    ):
        raise _error(
            "receipt.cmeie_binding.source_tree.manifest_algorithm",
            "is unsupported",
        )
    _integer(
        _required(
            source,
            "file_count",
            label="receipt.cmeie_binding.source_tree",
        ),
        label="receipt.cmeie_binding.source_tree.file_count",
        minimum=1,
    )

    rid = _text(
        _required(
            source,
            "manifest_rid",
            label="receipt.cmeie_binding.source_tree",
        ),
        label="receipt.cmeie_binding.source_tree.manifest_rid",
    )
    rid_match = _RID_RE.fullmatch(rid)
    if rid_match is None:
        raise _error(
            "receipt.cmeie_binding.source_tree.manifest_rid",
            "has an invalid immutable run ID",
        )
    git_head = _git_head(
        _required(
            source,
            "manifest_git_head",
            label="receipt.cmeie_binding.source_tree",
        ),
        label="receipt.cmeie_binding.source_tree.manifest_git_head",
    )
    if (
        rid_match.group("head") != git_head[:7]
        or rid_match.group("tree") != actual_tree[:12]
    ):
        raise _error(
            "receipt.cmeie_binding.source_tree.manifest_rid",
            "does not bind the reported Git HEAD and source tree",
        )

    script = _object(
        _required(
            source,
            "preflight_script",
            label="receipt.cmeie_binding.source_tree",
        ),
        label="receipt.cmeie_binding.source_tree.preflight_script",
    )
    _absolute_canonical_path(
        _required(
            script,
            "path",
            label="receipt.cmeie_binding.source_tree.preflight_script",
        ),
        label="receipt.cmeie_binding.source_tree.preflight_script.path",
    )
    _integer(
        _required(
            script,
            "size",
            label="receipt.cmeie_binding.source_tree.preflight_script",
        ),
        label="receipt.cmeie_binding.source_tree.preflight_script.size",
        minimum=1,
    )
    script_hash = _sha256(
        _required(
            script,
            "sha256",
            label="receipt.cmeie_binding.source_tree.preflight_script",
        ),
        label="receipt.cmeie_binding.source_tree.preflight_script.sha256",
    )
    return {
        "rid": rid,
        "git_head": git_head,
        "source_tree_sha256": actual_tree,
        "source_manifest_sha256": manifest_hash,
    }, script_hash


def _validate_base_model(
    value: object,
    *,
    expected_canonical_path: str,
    expected_tree_sha256: str,
) -> str:
    model = _object(value, label="receipt.cmeie_binding.cmeie_base_model")
    _true(
        _required(model, "valid", label="receipt.cmeie_binding.cmeie_base_model"),
        label="receipt.cmeie_binding.cmeie_base_model.valid",
    )
    model_dir = _absolute_canonical_path(
        _required(
            model,
            "model_dir",
            label="receipt.cmeie_binding.cmeie_base_model",
        ),
        label="receipt.cmeie_binding.cmeie_base_model.model_dir",
    )
    if model_dir != expected_canonical_path:
        raise _error(
            "receipt.cmeie_binding.cmeie_base_model.model_dir",
            "does not match the expected materialized model path",
        )
    if (
        _text(
            _required(
                model,
                "algorithm",
                label="receipt.cmeie_binding.cmeie_base_model",
            ),
            label="receipt.cmeie_binding.cmeie_base_model.algorithm",
        )
        != _TREE_ALGORITHM
    ):
        raise _error(
            "receipt.cmeie_binding.cmeie_base_model.algorithm",
            "is unsupported",
        )

    files_value = _array(
        _required(
            model,
            "files",
            label="receipt.cmeie_binding.cmeie_base_model",
        ),
        label="receipt.cmeie_binding.cmeie_base_model.files",
    )
    if not files_value:
        raise _error(
            "receipt.cmeie_binding.cmeie_base_model.files",
            "must not be empty",
        )
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(files_value):
        label = f"receipt.cmeie_binding.cmeie_base_model.files[{index}]"
        item = _object(value, label=label)
        path = _relative_posix_path(
            _required(item, "path", label=label),
            label=f"{label}.path",
        )
        if path in seen:
            raise _error(f"{label}.path", "duplicates an earlier model file")
        seen.add(path)
        entries.append(
            {
                "path": path,
                "size": _integer(
                    _required(item, "size", label=label),
                    label=f"{label}.size",
                    minimum=0,
                ),
                "sha256": _sha256(
                    _required(item, "sha256", label=label),
                    label=f"{label}.sha256",
                ),
            }
        )
    if entries != sorted(entries, key=lambda entry: entry["path"]):
        raise _error(
            "receipt.cmeie_binding.cmeie_base_model.files",
            "must be sorted by canonical relative path",
        )
    reported_count = _integer(
        _required(
            model,
            "file_count",
            label="receipt.cmeie_binding.cmeie_base_model",
        ),
        label="receipt.cmeie_binding.cmeie_base_model.file_count",
        minimum=1,
    )
    if reported_count != len(entries):
        raise _error(
            "receipt.cmeie_binding.cmeie_base_model.file_count",
            "does not match the model file inventory",
        )

    reported_tree = _sha256(
        _required(
            model,
            "tree_sha256",
            label="receipt.cmeie_binding.cmeie_base_model",
        ),
        label="receipt.cmeie_binding.cmeie_base_model.tree_sha256",
    )
    computed_tree = _tree_sha256(entries)
    if (
        reported_tree != computed_tree
        or reported_tree != expected_tree_sha256
    ):
        raise _error(
            "receipt.cmeie_binding.cmeie_base_model.tree_sha256",
            "reported, recomputed, and expected model-tree hashes must agree",
        )
    materialization = next(
        (
            entry
            for entry in entries
            if entry["path"] == _MATERIALIZATION_RECEIPT
        ),
        None,
    )
    if materialization is None:
        raise _error(
            "receipt.cmeie_binding.cmeie_base_model.files",
            f"must include {_MATERIALIZATION_RECEIPT}",
        )
    return materialization["sha256"]


def _validate_expected_run_binding(
    expected: Mapping[str, str] | None,
    *,
    actual: dict[str, str],
) -> None:
    if expected is None:
        return
    if not isinstance(expected, Mapping):
        raise TypeError("expected_run_binding must be a mapping or None")
    if set(expected) != _RUN_BINDING_KEYS:
        raise ValueError(
            "expected_run_binding must contain exactly rid, git_head, "
            "source_tree_sha256, and source_manifest_sha256"
        )
    for key in sorted(_RUN_BINDING_KEYS):
        value = _text(expected[key], label=f"expected_run_binding.{key}")
        if value != actual[key]:
            raise ValueError(
                f"expected_run_binding.{key} does not match the receipt"
            )


def validate_cmeie_npu_preflight_receipt(
    receipt_path: str,
    *,
    expected_source_tree_sha256: str,
    expected_source_manifest_sha256: str,
    expected_cmeie_train_sha256: str,
    expected_base_model_canonical_path: str,
    expected_base_model_tree_sha256: str,
    expected_run_binding: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate one strict preflight receipt and return its common binding.

    All expected digests must be lowercase SHA-256 strings.  The base-model
    path is compared as an exact canonical path rather than resolved on the
    consumer host, allowing a local orchestrator to validate a receipt emitted
    by a remote Linux NPU host.
    """

    source_tree_hash = _sha256(
        expected_source_tree_sha256,
        label="expected_source_tree_sha256",
    )
    source_manifest_hash = _sha256(
        expected_source_manifest_sha256,
        label="expected_source_manifest_sha256",
    )
    train_hash = _sha256(
        expected_cmeie_train_sha256,
        label="expected_cmeie_train_sha256",
    )
    model_path = _absolute_canonical_path(
        expected_base_model_canonical_path,
        label="expected_base_model_canonical_path",
    )
    model_tree_hash = _sha256(
        expected_base_model_tree_sha256,
        label="expected_base_model_tree_sha256",
    )

    receipt_value, receipt_binding = load_bound_json(
        receipt_path,
        "CMeIE strict NPU preflight receipt",
        reject_multiple_links=True,
        require_object=True,
    )
    receipt = _object(receipt_value, label="receipt")
    _reject_nonfinite(receipt, label="receipt")

    schema_version = _integer(
        _required(receipt, "schema_version", label="receipt"),
        label="receipt.schema_version",
    )
    if schema_version != 2:
        raise _error("receipt.schema_version", "must be 2")
    if (
        _text(
            _required(receipt, "purpose", label="receipt"),
            label="receipt.purpose",
        )
        != _PURPOSE
    ):
        raise _error("receipt.purpose", f"must be {_PURPOSE!r}")
    _true(
        _required(receipt, "strict", label="receipt"),
        label="receipt.strict",
    )
    _true(
        _required(receipt, "ready_for_explicit_npu_runs", label="receipt"),
        label="receipt.ready_for_explicit_npu_runs",
    )
    _absolute_canonical_path(
        _required(receipt, "root", label="receipt"),
        label="receipt.root",
    )

    _validate_auxiliary_ready_facts(receipt)
    verified_train_hash, dev_hash = _validate_cblue(
        _required(receipt, "cblue", label="receipt"),
        expected_train_sha256=train_hash,
    )

    cmeie_binding = _object(
        _required(receipt, "cmeie_binding", label="receipt"),
        label="receipt.cmeie_binding",
    )
    _true(
        _required(
            cmeie_binding,
            "complete_arguments",
            label="receipt.cmeie_binding",
        ),
        label="receipt.cmeie_binding.complete_arguments",
    )
    _true(
        _required(cmeie_binding, "valid", label="receipt.cmeie_binding"),
        label="receipt.cmeie_binding.valid",
    )
    source_run_binding, producer_script_hash = _validate_source_tree(
        _required(
            cmeie_binding,
            "source_tree",
            label="receipt.cmeie_binding",
        ),
        expected_tree_sha256=source_tree_hash,
        expected_manifest_sha256=source_manifest_hash,
    )
    source_root = _absolute_canonical_path(
        _required(
            cmeie_binding,
            "source_root",
            label="receipt.cmeie_binding",
        ),
        label="receipt.cmeie_binding.source_root",
    )
    source_tree_root = _absolute_canonical_path(
        cmeie_binding["source_tree"]["root"],
        label="receipt.cmeie_binding.source_tree.root",
    )
    if source_root != source_tree_root:
        raise _error(
            "receipt.cmeie_binding.source_root",
            "does not match the verified source-tree root",
        )
    materialization_receipt_hash = _validate_base_model(
        _required(
            cmeie_binding,
            "cmeie_base_model",
            label="receipt.cmeie_binding",
        ),
        expected_canonical_path=model_path,
        expected_tree_sha256=model_tree_hash,
    )
    runtime_fingerprint_hash = _validate_runtime(
        _required(receipt, "runtime", label="receipt")
    )
    npu_work = _array(
        _required(receipt, "npu_work_not_performed", label="receipt"),
        label="receipt.npu_work_not_performed",
    )
    if not npu_work:
        raise _error("receipt.npu_work_not_performed", "must not be empty")
    for index, value in enumerate(npu_work):
        _text(value, label=f"receipt.npu_work_not_performed[{index}]")
    if "training" not in npu_work:
        raise _error(
            "receipt.npu_work_not_performed",
            "must confirm that training was not performed during preflight",
        )

    _validate_expected_run_binding(
        expected_run_binding,
        actual=source_run_binding,
    )
    receipt_path_binding = _absolute_canonical_path(
        receipt_binding.get("canonical_path"),
        label="receipt descriptor binding canonical_path",
    )
    receipt_hash_binding = _sha256(
        receipt_binding.get("sha256"),
        label="receipt descriptor binding sha256",
    )
    return {
        "schema_version": 1,
        "purpose": _COMMON_PURPOSE,
        "preflight_receipt_path": receipt_path_binding,
        "preflight_receipt_sha256": receipt_hash_binding,
        "preflight_producer_script_sha256": producer_script_hash,
        "source_tree_sha256": source_tree_hash,
        "source_manifest_sha256": source_manifest_hash,
        "cmeie_train_rows": _CBLUE_ROWS["CMeIE-V2/train.jsonl"],
        "cmeie_train_sha256": verified_train_hash,
        "cmeie_dev_rows": _CBLUE_ROWS["CMeIE-V2/dev.jsonl"],
        "cmeie_dev_sha256": dev_hash,
        "base_model_canonical_path": model_path,
        "base_model_tree_sha256": model_tree_hash,
        "base_model_materialization_receipt_sha256": (
            materialization_receipt_hash
        ),
        "runtime_fingerprint_canonical_sha256": runtime_fingerprint_hash,
        "run_binding": source_run_binding,
    }


__all__ = ["validate_cmeie_npu_preflight_receipt"]
