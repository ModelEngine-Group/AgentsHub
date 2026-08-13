"""Fail-closed, no-model-load preflight for a controlled Ascend NPU host.

This script deliberately does not authenticate, transfer files, instantiate ACL,
load an OM, or start training. It records enough host/data facts to decide whether
later, explicitly authorized NPU experiments may begin.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.evidence_tree import scan_bound_regular_tree

EXPECTED_CBLUE_ROWS = {
    "CMeEE-V2/train.jsonl": 15000,
    "CMeEE-V2/dev.jsonl": 5000,
    "CMeIE-V2/train.jsonl": 14339,
    "CMeIE-V2/dev.jsonl": 3585,
}
EXPECTED_CMEEE_TYPES = (
    "dis", "sym", "dru", "equ", "pro", "bod", "ite", "mic", "dep",
)
SOURCE_TREE_SCHEMA_VERSION = 1
SOURCE_TREE_ALGORITHM = "sha256 of sorted path NUL size NUL file_sha256 LF"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_HEAD_PATTERN = re.compile(r"^[0-9a-f]{40}$")
RID_PATTERN = re.compile(
    r"^(?P<timestamp>[0-9]{8}T[0-9]{6}Z)-"
    r"g(?P<head>[0-9a-f]{7})-t(?P<tree>[0-9a-f]{12})$"
)
_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
_READ_CHUNK_SIZE = 1 << 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _same_regular_file(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        stat.S_ISREG(before.st_mode)
        and stat.S_ISREG(after.st_mode)
        and not _is_link_or_reparse(before)
        and not _is_link_or_reparse(after)
        and before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_mode == after.st_mode
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ctime_ns == after.st_ctime_ns
    )


def _bound_regular_file(path: Path, *, label: str) -> tuple[bytes, os.stat_result]:
    """Read one ordinary file without following a pathname substitution."""
    try:
        expected = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {path}") from exc
    if _is_link_or_reparse(expected):
        raise ValueError(f"{label} must not be a symlink or reparse point: {path}")
    if not stat.S_ISREG(expected.st_mode):
        raise ValueError(f"{label} must be an ordinary file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not _same_regular_file(expected, before):
            raise ValueError(f"{label} changed before it was read: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if not _same_regular_file(before, after):
            raise ValueError(f"{label} changed while it was read: {path}")
    finally:
        os.close(descriptor)

    rebound = os.lstat(path)
    if not _same_regular_file(expected, rebound):
        raise ValueError(f"{label} pathname changed while it was read: {path}")
    return b"".join(chunks), expected


def _bound_file_fact(path: Path, *, label: str) -> dict[str, Any]:
    data, metadata = _bound_regular_file(path, label=label)
    return {
        "size": metadata.st_size,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _case_identity(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _canonical_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty POSIX relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    parts = value.split("/")
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or any(part in {"", ".", ".."} for part in parts)
        or "/".join(parts) != value
    ):
        raise ValueError(f"{label} is not a canonical repository-relative path: {value}")
    return value


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


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"JSON contains a duplicate key: {key}")
        payload[key] = value
    return payload


def _canonical_manifest_entries(raw_entries: object) -> list[dict[str, Any]]:
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("source-tree manifest entries must be a non-empty array")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_casefolded: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "size", "sha256"}:
            raise ValueError(
                "each source-tree entry must contain exactly path, size, and sha256"
            )
        relative = _canonical_relative_path(
            raw["path"], label="source-tree manifest entry path"
        )
        size = raw["size"]
        digest = raw["sha256"]
        if type(size) is not int or size < 0:
            raise ValueError(f"source-tree entry has invalid size: {relative}")
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"source-tree entry has invalid SHA-256: {relative}")
        identity = _case_identity(relative)
        if relative in seen or identity in seen_casefolded:
            raise ValueError(f"source-tree manifest contains a duplicate path: {relative}")
        seen.add(relative)
        seen_casefolded.add(identity)
        entries.append({"path": relative, "size": size, "sha256": digest})
    canonical = sorted(entries, key=lambda entry: entry["path"])
    if entries != canonical:
        raise ValueError("source-tree manifest entries are not sorted canonically")
    return canonical


def _ordinary_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {path}") from exc
    if _is_link_or_reparse(metadata):
        raise ValueError(f"{label} must not be a symlink or reparse point: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be an ordinary directory: {path}")
    return metadata


def _exact_child(parent: Path, component: str, *, relative: str) -> Path:
    aliases: list[str] = []
    with os.scandir(parent) as iterator:
        for child in iterator:
            if _case_identity(child.name) == _case_identity(component):
                aliases.append(child.name)
    if aliases != [component]:
        if not aliases:
            raise ValueError(f"source-tree path does not exist: {relative}")
        raise ValueError(f"source-tree path has a case alias or duplicate: {relative}")
    return parent / component


def _source_entry_fact(root: Path, relative: str) -> dict[str, Any]:
    current = root
    parts = relative.split("/")
    for index, component in enumerate(parts):
        current = _exact_child(current, component, relative=relative)
        metadata = os.lstat(current)
        if _is_link_or_reparse(metadata):
            raise ValueError(
                f"source-tree path contains a symlink or reparse point: {relative}"
            )
        if index < len(parts) - 1:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    f"source-tree path has a non-directory component: {relative}"
                )
        elif not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"source-tree path is not an ordinary file: {relative}")
    root_text = os.path.normcase(os.path.abspath(root))
    path_text = os.path.normcase(os.path.abspath(current))
    if os.path.commonpath((root_text, path_text)) != root_text:
        raise ValueError(f"source-tree path escapes the repository root: {relative}")
    fact = _bound_file_fact(current, label=f"source-tree entry {relative}")
    return {"path": relative, **fact}


def _source_tree_report(
    root: Path,
    manifest_path: Path,
    expected_tree_sha256: str,
) -> tuple[dict[str, Any], bool]:
    """Verify an included-tree manifest against the exact current source bytes."""
    report: dict[str, Any] = {
        "valid": False,
        "requested_source_tree_sha256": expected_tree_sha256,
        "manifest_path": str(Path(os.path.abspath(manifest_path))),
    }
    try:
        if (
            not isinstance(expected_tree_sha256, str)
            or SHA256_PATTERN.fullmatch(expected_tree_sha256) is None
        ):
            raise ValueError("--source-tree-sha256 must be a lowercase SHA-256")
        root = Path(os.path.abspath(root))
        _ordinary_directory(root, label="source-tree root")
        report["root"] = str(root)
        script_fact = _bound_file_fact(Path(__file__), label="preflight script")
        report["preflight_script"] = {
            "path": str(Path(__file__).resolve()),
            **script_fact,
        }
        manifest_bytes, manifest_metadata = _bound_regular_file(
            Path(os.path.abspath(manifest_path)), label="source-tree manifest"
        )
        report.update({
            "manifest_size": manifest_metadata.st_size,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        })
        try:
            manifest = json.loads(
                manifest_bytes.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("source-tree manifest must be UTF-8 JSON") from exc
        required_keys = {
            "schema_version",
            "rid",
            "git_head",
            "algorithm",
            "included_tree_sha256",
            "entries",
        }
        if not isinstance(manifest, Mapping) or set(manifest) != required_keys:
            raise ValueError("source-tree manifest schema is invalid")
        if manifest["schema_version"] != SOURCE_TREE_SCHEMA_VERSION:
            raise ValueError("source-tree manifest schema_version is unsupported")
        if manifest["algorithm"] != SOURCE_TREE_ALGORITHM:
            raise ValueError("source-tree manifest algorithm is unsupported")
        rid = manifest["rid"]
        git_head = manifest["git_head"]
        if not isinstance(rid, str) or (rid_match := RID_PATTERN.fullmatch(rid)) is None:
            raise ValueError("source-tree manifest RID is invalid")
        if not isinstance(git_head, str) or GIT_HEAD_PATTERN.fullmatch(git_head) is None:
            raise ValueError("source-tree manifest git_head is invalid")
        entries = _canonical_manifest_entries(manifest["entries"])
        claimed_tree = manifest["included_tree_sha256"]
        if not isinstance(claimed_tree, str) or SHA256_PATTERN.fullmatch(claimed_tree) is None:
            raise ValueError("source-tree manifest included_tree_sha256 is invalid")
        actual_tree = _tree_sha256(entries)
        if claimed_tree != actual_tree:
            raise ValueError("source-tree manifest hash does not match its entries")
        if expected_tree_sha256 != actual_tree:
            raise ValueError("--source-tree-sha256 does not match the manifest")
        if (
            rid_match.group("head") != git_head[:7]
            or rid_match.group("tree") != actual_tree[:12]
        ):
            raise ValueError("source-tree manifest RID does not bind its HEAD and tree")

        verified_entries: list[dict[str, Any]] = []
        for entry in entries:
            fact = _source_entry_fact(root, entry["path"])
            if fact["size"] != entry["size"] or fact["sha256"] != entry["sha256"]:
                raise ValueError(
                    f"current source file does not match the manifest: {entry['path']}"
                )
            verified_entries.append(fact)

        script_entry = next(
            (
                entry
                for entry in entries
                if entry["path"] == "scripts/remote_npu_preflight.py"
            ),
            None,
        )
        if (
            script_entry is None
            or script_entry["size"] != script_fact["size"]
            or script_entry["sha256"] != script_fact["sha256"]
        ):
            raise ValueError(
                "the executing preflight script is not bound by the source-tree manifest"
            )
        report.update({
            "valid": True,
            "source_tree_sha256": actual_tree,
            "manifest_schema_version": manifest["schema_version"],
            "manifest_rid": rid,
            "manifest_git_head": git_head,
            "manifest_algorithm": manifest["algorithm"],
            "file_count": len(verified_entries),
        })
        return report, True
    except (OSError, TypeError, ValueError) as exc:
        report.update({
            "valid": False,
            "error": type(exc).__name__,
            "message": str(exc),
        })
        return report, False


def _cmeie_base_model_report(model_dir: Path) -> tuple[dict[str, Any], bool]:
    """Create a stable, deterministic manifest of an ordinary-file model tree."""
    try:
        binding = scan_bound_regular_tree(
            model_dir,
            label="CMeIE base-model",
        )
        return {
            "valid": True,
            "model_dir": binding["root"],
            "algorithm": SOURCE_TREE_ALGORITHM,
            "file_count": binding["n_files"],
            "tree_sha256": binding["sha256"],
            "files": binding["entries"],
        }, True
    except (OSError, TypeError, ValueError) as exc:
        return {
            "valid": False,
            "model_dir": str(Path(os.path.abspath(model_dir))),
            "error": type(exc).__name__,
            "message": str(exc),
        }, False


def _cmeie_binding_report(
    *,
    source_root: Path,
    source_tree_sha256: str | None,
    source_tree_manifest: Path | None,
    cmeie_base_model: Path | None,
    strict: bool,
) -> tuple[dict[str, Any] | None, bool]:
    values = (source_tree_sha256, source_tree_manifest, cmeie_base_model)
    if all(value is None for value in values) and not strict:
        return None, True
    missing = [
        flag
        for flag, value in (
            ("--source-tree-sha256", source_tree_sha256),
            ("--source-tree-manifest", source_tree_manifest),
            ("--cmeie-base-model", cmeie_base_model),
        )
        if value is None
    ]
    if missing:
        return {
            "complete_arguments": False,
            "valid": False,
            "missing_arguments": missing,
        }, False
    assert source_tree_sha256 is not None
    assert source_tree_manifest is not None
    assert cmeie_base_model is not None
    source_tree, source_tree_ok = _source_tree_report(
        source_root, source_tree_manifest, source_tree_sha256
    )
    base_model, base_model_ok = _cmeie_base_model_report(cmeie_base_model)
    return {
        "complete_arguments": True,
        "valid": source_tree_ok and base_model_ok,
        "source_root": str(Path(os.path.abspath(source_root))),
        "source_tree": source_tree,
        "cmeie_base_model": base_model,
    }, source_tree_ok and base_model_ok


def _run(command: list[str], timeout: float = 20.0) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command}
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "command": command, "error": type(exc).__name__}
    return {
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "")[-4000:],
        "stderr": (completed.stderr or "")[-1000:],
    }


def _package_version(name: str) -> dict[str, Any]:
    """Discover a package without importing extension modules into preflight."""
    try:
        spec = importlib.util.find_spec(name)
    except (AttributeError, ImportError, ValueError) as exc:
        return {"available": False, "error": type(exc).__name__}
    if spec is None:
        return {"available": False, "error": "ModuleNotFoundError"}
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {"available": True, "version": version}


def _torch_npu_probe() -> dict[str, Any]:
    """Probe torch_npu in a disposable child so extension conflicts stay isolated."""
    command = [
        sys.executable,
        "-c",
        (
            "import json; import torch; import torch_npu; "
            "npu = getattr(torch, 'npu', None); "
            "print(json.dumps({'npu_available': bool(npu is not None and npu.is_available())}))"
        ),
    ]
    result = _run(command)
    if result.get("returncode") != 0:
        return {"npu_available": False, "probe": result}
    for line in reversed(str(result.get("stdout", "")).splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("npu_available"), bool):
            return {"npu_available": payload["npu_available"], "probe": result}
    return {"npu_available": False, "probe": result, "error": "invalid_probe_output"}


def _cblue_report(root: Path) -> tuple[dict[str, Any], bool]:
    data_root = root / "data" / "cblue"
    report = {}
    passed = True
    for relative, expected_rows in EXPECTED_CBLUE_ROWS.items():
        path = data_root / relative
        if not path.is_file():
            report[relative] = {"exists": False, "expected_rows": expected_rows}
            passed = False
            continue
        with path.open("r", encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
        item = {
            "exists": True,
            "rows": rows,
            "expected_rows": expected_rows,
            "sha256": _sha256(path),
        }
        report[relative] = item
        passed = passed and rows == expected_rows
    return report, passed


def _required_paths(paths: list[Path]) -> tuple[list[dict[str, Any]], bool]:
    report = []
    passed = True
    for path in paths:
        exists = path.is_file() or path.is_dir()
        item: dict[str, Any] = {"path": str(path), "exists": exists}
        if exists and path.is_file():
            item.update({"size": path.stat().st_size, "sha256": _sha256(path)})
        report.append(item)
        passed = passed and exists
    return report, passed


def _cmeee_checkpoint_report(model_dir: Path | None) -> tuple[dict[str, Any] | None, bool]:
    """Validate the frozen CMeEE model contract without importing Torch.

    The later rescore CLI needs these exact files and configuration values.  Check
    them here so a malformed or differently labelled checkpoint cannot consume an
    NPU slot before the evaluation command can fail clearly.
    """
    if model_dir is None:
        return None, True
    required = (
        "config.json", "globalpointer.pt", "globalpointer_config.json", "types.json",
    )
    files = []
    errors = []
    if not model_dir.is_dir():
        errors.append("model_directory_missing")
    for relative in required:
        path = model_dir / relative
        exists = path.is_file()
        item: dict[str, Any] = {"path": str(path), "exists": exists}
        if exists:
            item.update({"size": path.stat().st_size, "sha256": _sha256(path)})
        else:
            errors.append(f"missing:{relative}")
        files.append(item)
    if not errors:
        try:
            hf_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            config = json.loads((model_dir / "globalpointer_config.json").read_text(encoding="utf-8"))
            types = json.loads((model_dir / "types.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            errors.append("invalid_checkpoint_json")
        else:
            if not isinstance(hf_config, dict):
                errors.append("invalid:config.json")
            if not isinstance(config, dict):
                errors.append("invalid:globalpointer_config.json")
            else:
                max_len = config.get("max_len")
                inner_dim = config.get("inner_dim")
                threshold = config.get("selected_threshold")
                if type(max_len) is not int or max_len <= 0:
                    errors.append("invalid:max_len")
                if type(inner_dim) is not int or inner_dim <= 0:
                    errors.append("invalid:inner_dim")
                if (
                    isinstance(threshold, bool)
                    or not isinstance(threshold, (int, float))
                    or not math.isfinite(float(threshold))
                ):
                    errors.append("invalid:selected_threshold")
            if types != list(EXPECTED_CMEEE_TYPES):
                errors.append("invalid:types.json")
    passed = not errors
    return {
        "model_dir": str(model_dir),
        "files": files,
        "complete": passed,
        "validation_errors": errors,
    }, passed


def _om_manifest_report(
    manifest_path: Path | None,
    om_path: Path | None,
    tokenizer_path: Path | None,
    labels_path: Path | None,
    checkpoint_path: Path | None = None,
    export_receipt_path: Path | None = None,
    atc_log_path: Path | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Hash-verify an OM bundle without loading ACL or the OM model."""
    provided = (manifest_path, om_path, tokenizer_path, labels_path)
    if all(path is None for path in provided):
        return None, True
    if any(path is None for path in provided):
        return {
            "complete_arguments": False,
            "error": "manifest, OM, tokenizer, and labels paths must be supplied together",
        }, False
    assert manifest_path is not None
    assert om_path is not None
    assert tokenizer_path is not None
    assert labels_path is not None
    try:
        from dki.bench.npu.om_manifest import OmManifest

        manifest = OmManifest.load(manifest_path)
        manifest.verify_artifacts(
            om_path,
            tokenizer_path,
            labels_path,
            checkpoint_path,
            export_receipt_path,
            atc_log_path,
        )
        if (
            manifest.schema_version not in {1, 3}
            or not manifest.model_id
            or not manifest.soc_version
            or not manifest.cann_version
            or not manifest.inputs
            or not manifest.outputs
        ) or (
            manifest.schema_version == 3
            and not all(
                (
                    manifest.checkpoint_sha256,
                    manifest.export_receipt_sha256,
                    manifest.onnx_sha256,
                    manifest.model_family,
                    manifest.atc_log_sha256,
                )
            )
        ):
            raise ValueError("manifest contract is incomplete")
    except BaseException as exc:
        return {"complete_arguments": True, "valid": False, "error": type(exc).__name__}, False
    artifacts = []
    for role, path in (
        ("om", om_path),
        ("tokenizer", tokenizer_path),
        ("labels", labels_path),
    ):
        artifacts.append({
            "role": role,
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        })
    if checkpoint_path is not None:
        if checkpoint_path.is_dir():
            from dki.bench.npu.om_manifest import sha256_tree

            artifacts.append({
                "role": "checkpoint",
                "path": str(checkpoint_path),
                "kind": "directory_tree_v1",
                "file_count": sum(
                    1 for item in checkpoint_path.rglob("*") if item.is_file()
                ),
                "sha256": sha256_tree(checkpoint_path),
            })
        else:
            artifacts.append({
                "role": "checkpoint",
                "path": str(checkpoint_path),
                "kind": "file",
                "size": checkpoint_path.stat().st_size,
                "sha256": _sha256(checkpoint_path),
            })
    if export_receipt_path is not None:
        artifacts.append({
            "role": "export_receipt",
            "path": str(export_receipt_path),
            "size": export_receipt_path.stat().st_size,
            "sha256": _sha256(export_receipt_path),
        })
    if atc_log_path is not None:
        artifacts.append({
            "role": "atc_log",
            "path": str(atc_log_path),
            "size": atc_log_path.stat().st_size,
            "sha256": _sha256(atc_log_path),
        })
    return {
        "complete_arguments": True,
        "valid": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "model_id": manifest.model_id,
        "model_family": manifest.model_family,
        "schema_version": manifest.schema_version,
        "soc_version": manifest.soc_version,
        "cann_version": manifest.cann_version,
        "input_count": len(manifest.inputs),
        "output_count": len(manifest.outputs),
        "artifacts": artifacts,
    }, True


def _runtime_report() -> tuple[dict[str, Any], bool]:
    torch_info = _package_version("torch")
    npu_info = _package_version("torch_npu")
    transformers_info = _package_version("transformers")
    npu_probe = _torch_npu_probe() if torch_info["available"] and npu_info["available"] else {
        "npu_available": False,
        "error": "torch_or_torch_npu_not_discoverable",
    }
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_info,
        "torch_npu": npu_info,
        "transformers": transformers_info,
        "torch_npu_available": npu_probe["npu_available"],
        "torch_npu_probe": npu_probe,
        "npu_smi": _run(["npu-smi", "info"]),
        "atc": _run(["atc", "--version"]),
        "ascend_env_present": {
            key: bool(os.environ.get(key))
            for key in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "LD_LIBRARY_PATH")
        },
    }
    npu_smi_ok = report["npu_smi"].get("returncode") == 0
    return report, bool(
        torch_info["available"]
        and npu_info["available"]
        and npu_probe["npu_available"]
        and npu_smi_ok
    )


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    staging = Path(staging_name)
    try:
        try:
            remaining = memoryview(data)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("failed to write preflight staging file")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="No-model-load controlled-NPU preflight")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-path", type=Path, action="append", default=[])
    parser.add_argument("--cmeee-model", type=Path)
    parser.add_argument("--om-manifest", type=Path)
    parser.add_argument("--om-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--labels-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--export-receipt-path", type=Path)
    parser.add_argument("--atc-log-path", type=Path)
    parser.add_argument("--source-tree-sha256")
    parser.add_argument("--source-tree-manifest", type=Path)
    parser.add_argument("--cmeie-base-model", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if os.path.lexists(args.output):
        raise SystemExit(f"preflight output already exists: {args.output}")

    root = args.root.resolve()
    cblue, cblue_ok = _cblue_report(root)
    required, required_ok = _required_paths(args.required_path)
    cmeee_checkpoint, cmeee_checkpoint_ok = _cmeee_checkpoint_report(args.cmeee_model)
    om_manifest, om_manifest_ok = _om_manifest_report(
        args.om_manifest,
        args.om_path,
        args.tokenizer_path,
        args.labels_path,
        args.checkpoint_path,
        args.export_receipt_path,
        args.atc_log_path,
    )
    cmeie_binding, cmeie_binding_ok = _cmeie_binding_report(
        # ``--root`` is the historical CBLUE checkout used only for data
        # inventory.  The included-tree manifest must instead be verified
        # against the repository that owns the executing script; otherwise a
        # remote command with a separate CBLUE checkout would validate the
        # source manifest against the wrong directory.
        source_root=ROOT,
        source_tree_sha256=args.source_tree_sha256,
        source_tree_manifest=args.source_tree_manifest,
        cmeie_base_model=args.cmeie_base_model,
        strict=args.strict,
    )
    runtime, runtime_ok = _runtime_report()
    report = {
        "schema_version": 2,
        "purpose": "preflight_only_no_acl_model_load",
        "root": str(root),
        "cblue": cblue,
        "required_artifacts": required,
        "cmeee_checkpoint": cmeee_checkpoint,
        "om_manifest": om_manifest,
        "cmeie_binding": cmeie_binding,
        "runtime": runtime,
        "strict": args.strict,
        "ready_for_explicit_npu_runs": (
            cblue_ok
            and required_ok
            and cmeee_checkpoint_ok
            and om_manifest_ok
            and cmeie_binding_ok
            and runtime_ok
        ),
        "npu_work_not_performed": [
            "ACL initialization", "OM model load", "OM inference", "training", "benchmarking",
        ],
    }
    try:
        _write(args.output, report)
    except FileExistsError as exc:
        raise SystemExit(
            f"preflight output already exists: {args.output}"
        ) from exc
    print(json.dumps({
        "ready_for_explicit_npu_runs": report["ready_for_explicit_npu_runs"],
        "cblue_ok": cblue_ok,
        "required_artifacts_ok": required_ok,
        "cmeee_checkpoint_ok": cmeee_checkpoint_ok,
        "om_manifest_ok": om_manifest_ok,
        "cmeie_binding_ok": cmeie_binding_ok,
        "runtime_ok": runtime_ok,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0 if (report["ready_for_explicit_npu_runs"] or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
