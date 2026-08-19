"""Fail-closed isolated origin assertion for a staged NL2SQL bundle.

The future outer coordinator must verify and safely extract a bundle before
starting this script with ``python -I -S -B``.  It must independently verify the
runner/dependency-environment manifest and retain the expected bundle-manifest
SHA-256 from trusted attestations.  This script authenticates every staged
member into memory, then imports registered local source only from that byte
snapshot.  It deliberately adds no dependency paths, so it cannot silently use
live site packages or invoke an evaluator.
"""

from __future__ import annotations

import sys


def _require_isolated_startup() -> None:
    if not sys.flags.isolated:
        raise SystemExit("inner execution-bundle assertion requires python -I")
    if not sys.flags.no_site:
        raise SystemExit("inner execution-bundle assertion requires python -S")
    if not sys.dont_write_bytecode:
        raise SystemExit("inner execution-bundle assertion requires python -B")
    for name in ("site", "sitecustomize", "usercustomize"):
        if name in sys.modules:
            raise SystemExit(f"inner startup unexpectedly loaded {name}")


_require_isolated_startup()

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


BUNDLE_SCHEMA_VERSION = 2
BUNDLE_KIND = "nl2sql_execution_bundle"
BUNDLE_ARCHIVE_FORMAT = "zip-stored-v1"
MANIFEST_NAME = "bundle_manifest.json"
PAYLOAD_ROOT = "payload"
RUNNER_ENVIRONMENT_MANIFEST_SHA256_FIELD = "runner_environment_manifest_sha256"
SOURCE_ROLE = "python_source"
RUNTIME_ASSET_ROLE = "runtime_asset"
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_FILE_ENTRIES = 10_000
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_RELATIVE_PATH_CHARS = 512
MAX_PATH_COMPONENTS = 32
MAX_PATH_COMPONENT_CHARS = 128

_COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BYTECODE_SUFFIXES = (".pyc", ".pyo")
_NATIVE_CODE_SUFFIXES = (".pyd", ".so", ".dll", ".dylib")
_AUTOMATIC_IMPORT_FILENAMES = frozenset({"sitecustomize.py", "usercustomize.py"})
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)
_STDLIB_MODULE_NAMES = frozenset(getattr(sys, "stdlib_module_names", ()))
_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)


@dataclass(frozen=True)
class _DirectoryAnchor:
    path: Path
    components: tuple[tuple[Path, os.stat_result], ...]


@dataclass(frozen=True)
class _SnapshotModule:
    data: bytes
    origin: Path
    is_package: bool


def _path_text(path: str | os.PathLike[str]) -> str:
    value = os.fspath(path)
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError("path must be NUL-free text")
    return value


def _absolute_path(path: str | os.PathLike[str]) -> Path:
    """Make a lexical absolute pathname without resolving links or reparses."""

    return Path(os.path.abspath(_path_text(path)))


def _require_byte_limit(value: int | None, *, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} byte limit must be a non-negative integer")
    return value


def _is_link_or_reparse(metadata: object) -> bool:
    return stat.S_ISLNK(getattr(metadata, "st_mode", 0)) or bool(
        getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _lstat(path: Path) -> os.stat_result:
    return os.lstat(path)


def _reject_link_or_reparse(path: Path, metadata: object, *, label: str) -> None:
    if _is_link_or_reparse(metadata):
        raise ValueError(f"{label} contains a symlink or reparse component: {path}")


def _same_birthtime(before: os.stat_result, after: os.stat_result) -> bool:
    return getattr(before, "st_birthtime_ns", None) == getattr(
        after,
        "st_birthtime_ns",
        None,
    )


def _has_stable_file_identity(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and not _is_link_or_reparse(metadata)
        and getattr(metadata, "st_ino", 0) != 0
    )


def _same_file_object_identity(
    before: os.stat_result,
    after: os.stat_result,
) -> bool:
    return (
        _has_stable_file_identity(before)
        and _has_stable_file_identity(after)
        and before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_mode == after.st_mode
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and _same_birthtime(before, after)
    )


def _same_file_identity(before: os.stat_result, after: os.stat_result) -> bool:
    return _same_file_object_identity(before, after) and (
        os.name == "nt"
        or getattr(before, "st_ctime_ns", None) == getattr(after, "st_ctime_ns", None)
    )


def _same_descriptor_snapshot(
    before: os.stat_result,
    after: os.stat_result,
) -> bool:
    return _same_file_object_identity(before, after) and (
        getattr(before, "st_ctime_ns", None) == getattr(after, "st_ctime_ns", None)
    )


def _has_stable_directory_identity(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not _is_link_or_reparse(metadata)
        and getattr(metadata, "st_ino", 0) != 0
    )


def _same_directory_identity(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        _has_stable_directory_identity(before)
        and _has_stable_directory_identity(after)
        and before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and stat.S_IFMT(before.st_mode) == stat.S_IFMT(after.st_mode)
        and _same_birthtime(before, after)
    )


def _capture_directory_components(
    path: Path,
    *,
    label: str,
) -> tuple[tuple[Path, os.stat_result], ...]:
    candidate = _absolute_path(path)
    current = Path(candidate.anchor)
    metadata = _lstat(current)
    _reject_link_or_reparse(current, metadata, label=label)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} is not a directory: {current}")
    snapshots: list[tuple[Path, os.stat_result]] = [(current, metadata)]
    for component in candidate.parts[1:]:
        current = current / component
        try:
            metadata = _lstat(current)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{label} does not exist: {current}") from exc
        _reject_link_or_reparse(current, metadata, label=label)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} is not a directory: {current}")
        snapshots.append((current, metadata))
    return tuple(snapshots)


def _revalidate_directory_components(
    snapshots: tuple[tuple[Path, os.stat_result], ...],
    *,
    label: str,
) -> None:
    for path, before in snapshots:
        try:
            after = _lstat(path)
        except FileNotFoundError as exc:
            raise ValueError(f"{label} component changed: {path}") from exc
        if not _same_directory_identity(before, after):
            raise ValueError(f"{label} component changed: {path}")


def _pin_directory(path: Path, *, label: str) -> _DirectoryAnchor:
    candidate = _absolute_path(path)
    return _DirectoryAnchor(
        path=candidate,
        components=_capture_directory_components(candidate, label=label),
    )


def _open_bound_directory(
    path: Path,
    snapshots: tuple[tuple[Path, os.stat_result], ...],
    *,
    label: str,
) -> int | None:
    """Open every POSIX directory component relative to the prior descriptor."""

    if os.name == "nt" or os.open not in os.supports_dir_fd:
        return None
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    anchor, anchor_metadata = snapshots[0]
    try:
        descriptor = os.open(anchor, flags)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely: {path}") from exc
    try:
        if not _same_directory_identity(anchor_metadata, os.fstat(descriptor)):
            raise ValueError(f"{label} component changed before use: {anchor}")
        for component, (_, expected) in zip(path.parts[1:], snapshots[1:]):
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise ValueError(f"{label} cannot be opened safely: {path}") from exc
            try:
                if not _same_directory_identity(expected, os.fstat(child)):
                    raise ValueError(f"{label} component changed before use: {path}")
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _parent_entry_lstat(
    parent: Path,
    descriptor: int | None,
    name: str,
) -> os.stat_result:
    if descriptor is None:
        return _lstat(parent / name)
    return os.stat(name, dir_fd=descriptor, follow_symlinks=False)


def _read_checked_descriptor(
    descriptor: int,
    expected: os.stat_result,
    *,
    label: str,
    path: Path,
    maximum_bytes: int | None = None,
) -> bytes:
    maximum_bytes = _require_byte_limit(maximum_bytes, label=label)
    before = os.fstat(descriptor)
    if not _same_file_identity(expected, before):
        raise ValueError(f"{label} changed while being opened: {path}")
    if maximum_bytes is not None and before.st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds the byte limit: {path}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    remaining = before.st_size
    chunks: list[bytes] = []
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    extra = os.read(descriptor, 1)
    data = b"".join(chunks)
    after = os.fstat(descriptor)
    if (
        remaining
        or extra
        or not _same_descriptor_snapshot(before, after)
        or len(data) != before.st_size
    ):
        raise ValueError(f"{label} changed while being read: {path}")
    return data


def _read_stable_bytes(
    path: Path,
    *,
    label: str,
    maximum_bytes: int | None = None,
) -> bytes:
    maximum_bytes = _require_byte_limit(maximum_bytes, label=label)
    candidate = _absolute_path(path)
    parent_snapshots = _capture_directory_components(candidate.parent, label=label)
    try:
        path_stat = _lstat(candidate)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {candidate}") from exc
    _reject_link_or_reparse(candidate, path_stat, label=label)
    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {candidate}")
    if maximum_bytes is not None and path_stat.st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds the byte limit: {candidate}")
    parent_descriptor = _open_bound_directory(
        candidate.parent,
        parent_snapshots,
        label=label,
    )
    descriptor: int | None = None
    try:
        current = _parent_entry_lstat(candidate.parent, parent_descriptor, candidate.name)
        _reject_link_or_reparse(candidate, current, label=label)
        if not _same_file_identity(path_stat, current):
            raise ValueError(f"{label} changed while being opened: {candidate}")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        if os.name != "nt":
            flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            if parent_descriptor is None:
                descriptor = os.open(candidate, flags)
            else:
                descriptor = os.open(candidate.name, flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise ValueError(f"{label} cannot be opened safely: {candidate}") from exc
        data = _read_checked_descriptor(
            descriptor,
            path_stat,
            label=label,
            path=candidate,
            maximum_bytes=maximum_bytes,
        )
        final = _parent_entry_lstat(candidate.parent, parent_descriptor, candidate.name)
        _reject_link_or_reparse(candidate, final, label=label)
        if not _same_file_identity(path_stat, final):
            raise ValueError(f"{label} changed while being read: {candidate}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    _revalidate_directory_components(parent_snapshots, label=label)
    return data


def _relative_components(relative: str, *, label: str) -> tuple[str, tuple[str, ...]]:
    canonical = _canonical_relative_path(relative, label=label)
    return canonical, tuple(canonical.split("/"))


def _open_checked_child_directory(
    *,
    parent_path: Path,
    parent_descriptor: int,
    name: str,
    label: str,
) -> tuple[Path, int]:
    candidate = parent_path / name
    try:
        metadata = _parent_entry_lstat(parent_path, parent_descriptor, name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {candidate}") from exc
    _reject_link_or_reparse(candidate, metadata, label=label)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} has a non-directory component: {candidate}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely: {candidate}") from exc
    try:
        if not _same_directory_identity(metadata, os.fstat(descriptor)):
            raise ValueError(f"{label} changed while being opened: {candidate}")
    except BaseException:
        os.close(descriptor)
        raise
    return candidate, descriptor


def _anchored_relative_metadata_fallback(
    root: _DirectoryAnchor,
    relative: str,
    *,
    label: str,
) -> tuple[Path, os.stat_result] | None:
    _, parts = _relative_components(relative, label=label)
    _revalidate_directory_components(root.components, label=label)
    current = root.path
    for index, component in enumerate(parts):
        current = current / component
        try:
            metadata = _lstat(current)
        except FileNotFoundError:
            return None
        _reject_link_or_reparse(current, metadata, label=label)
        if index < len(parts) - 1:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"{label} has a non-directory component: {current}")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file: {current}")
        _revalidate_directory_components(root.components, label=label)
        return current, metadata
    raise AssertionError("relative-path component walk did not return")


def _read_anchored_relative_bytes(
    root: _DirectoryAnchor,
    relative: str,
    *,
    label: str,
    maximum_bytes: int | None = None,
) -> bytes:
    maximum_bytes = _require_byte_limit(maximum_bytes, label=label)
    canonical, parts = _relative_components(relative, label=label)
    root_descriptor = _open_bound_directory(root.path, root.components, label=label)
    if root_descriptor is None:
        checked = _anchored_relative_metadata_fallback(root, canonical, label=label)
        if checked is None:
            raise ValueError(f"{label} does not exist below {root.path}: {canonical}")
        path, expected = checked
        if maximum_bytes is not None and expected.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds the byte limit: {path}")
        data = _read_stable_bytes(
            path,
            label=label,
            maximum_bytes=maximum_bytes,
        )
        try:
            actual = _lstat(path)
        except FileNotFoundError as exc:
            raise ValueError(f"{label} changed while being read: {path}") from exc
        if not _same_file_identity(expected, actual):
            raise ValueError(f"{label} changed while being read: {path}")
        _revalidate_directory_components(root.components, label=label)
        return data
    current_path = root.path
    descriptor = root_descriptor
    file_descriptor: int | None = None
    try:
        _revalidate_directory_components(root.components, label=label)
        for component in parts[:-1]:
            child_path, child_descriptor = _open_checked_child_directory(
                parent_path=current_path,
                parent_descriptor=descriptor,
                name=component,
                label=label,
            )
            os.close(descriptor)
            current_path = child_path
            descriptor = child_descriptor
        candidate = current_path / parts[-1]
        try:
            expected = _parent_entry_lstat(current_path, descriptor, parts[-1])
        except FileNotFoundError as exc:
            raise ValueError(f"{label} does not exist: {candidate}") from exc
        _reject_link_or_reparse(candidate, expected, label=label)
        if not stat.S_ISREG(expected.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file: {candidate}")
        if maximum_bytes is not None and expected.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds the byte limit: {candidate}")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            file_descriptor = os.open(parts[-1], flags, dir_fd=descriptor)
        except OSError as exc:
            raise ValueError(f"{label} cannot be opened safely: {candidate}") from exc
        data = _read_checked_descriptor(
            file_descriptor,
            expected,
            label=label,
            path=candidate,
            maximum_bytes=maximum_bytes,
        )
        actual = _parent_entry_lstat(current_path, descriptor, parts[-1])
        _reject_link_or_reparse(candidate, actual, label=label)
        if not _same_file_identity(expected, actual):
            raise ValueError(f"{label} changed while being read: {candidate}")
        _revalidate_directory_components(root.components, label=label)
        return data
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(descriptor)


def _module_name(value: Any, *, label: str = "manifest entry module") -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} is invalid")
    parts = value.split(".")
    if any(not part or not part.isascii() or not part.isidentifier() for part in parts):
        raise ValueError(f"{label} is invalid")
    if parts[0] == "__main__":
        raise ValueError(f"{label} cannot use __main__")
    if parts[0] in _STDLIB_MODULE_NAMES:
        raise ValueError(f"{label} shadows the standard library")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_relative_path(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if "\\" in value or "\x00" in value or value.startswith("/"):
        raise ValueError(f"{label} is not a canonical relative POSIX path")
    if len(value) > MAX_RELATIVE_PATH_CHARS:
        raise ValueError(f"{label} exceeds the relative-path length limit")
    parts = value.split("/")
    if len(parts) > MAX_PATH_COMPONENTS:
        raise ValueError(f"{label} exceeds the path-depth limit")
    if any(len(part) > MAX_PATH_COMPONENT_CHARS for part in parts):
        raise ValueError(f"{label} contains an oversized path component")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} is not a canonical relative POSIX path")
    if any(_COMPONENT.fullmatch(part) is None for part in parts):
        raise ValueError(f"{label} contains a non-canonical path component")
    if any(
        part.endswith(".")
        or part.casefold().split(".", 1)[0] in _WINDOWS_RESERVED_BASENAMES
        for part in parts
    ):
        raise ValueError(f"{label} contains an unsafe filesystem path component")
    return value


def _validate_payload_path(
    value: Any,
    *,
    label: str,
    role: str | None = None,
) -> str:
    path = _canonical_relative_path(value, label=label)
    parts = [part.casefold() for part in path.split("/")]
    filename = parts[-1]
    if "__pycache__" in parts or filename.endswith(_BYTECODE_SUFFIXES):
        raise ValueError(f"{label} contains a forbidden Python bytecode path")
    if filename.endswith(".pth") or filename in _AUTOMATIC_IMPORT_FILENAMES:
        raise ValueError(f"{label} contains a forbidden Python startup path")
    if role == SOURCE_ROLE:
        if not filename.endswith(".py"):
            raise ValueError(f"{label} is not a Python source path")
    elif role == RUNTIME_ASSET_ROLE:
        if filename.endswith((".py", *_NATIVE_CODE_SUFFIXES)):
            raise ValueError(f"{label} introduces Python code or native metadata")
    return path


def _validate_payload_layout(paths: list[str] | tuple[str, ...] | set[str]) -> None:
    folded_paths: dict[str, str] = {}
    for path in paths:
        folded = path.casefold()
        if folded in folded_paths:
            raise ValueError("staged execution bundle contains case-colliding paths")
        folded_paths[folded] = path
    for path in folded_paths:
        parent = path
        while "/" in parent:
            parent = parent.rsplit("/", 1)[0]
            if parent in folded_paths:
                raise ValueError("staged execution bundle contains a file/directory path conflict")


def _validate_stage_member_name(value: str) -> str:
    member = _canonical_relative_path(value, label="staged member path")
    if member in {MANIFEST_NAME, PAYLOAD_ROOT}:
        return member
    prefix = f"{PAYLOAD_ROOT}/"
    if not member.startswith(prefix):
        raise ValueError("staged member is outside the execution bundle payload")
    _validate_payload_path(member[len(prefix) :], label="staged member path")
    return member


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(payload))


def _canonical_content_sha256(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(str(entry["path"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["role"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _strict_json_loads(data: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} repeats JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains invalid JSON constant: {value}")

    try:
        manifest = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if type(manifest) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return manifest


def _validate_file_entry(entry: Any) -> dict[str, Any]:
    if type(entry) is not dict or set(entry) != {"path", "role", "sha256", "size"}:
        raise ValueError("execution bundle manifest file entry is invalid")
    role = entry.get("role")
    if role not in {SOURCE_ROLE, RUNTIME_ASSET_ROLE}:
        raise ValueError("execution bundle manifest file role is invalid")
    _validate_payload_path(entry.get("path"), label="manifest file path", role=role)
    size = entry.get("size")
    if type(size) is not int or size < 0 or size > MAX_MEMBER_BYTES:
        raise ValueError("execution bundle manifest file size is invalid")
    _require_sha256(entry.get("sha256"), "execution bundle file SHA-256")
    return entry


def _validate_module_registry(
    manifest: Mapping[str, Any],
    files: list[dict[str, Any]],
) -> None:
    registry = manifest.get("module_registry")
    if not isinstance(registry, list) or not registry:
        raise ValueError("execution bundle manifest lacks a module registry")
    source_files = {
        str(entry["path"]): entry
        for entry in files
        if entry["role"] == SOURCE_ROLE
    }
    if not source_files:
        raise ValueError("execution bundle manifest lacks Python source files")
    modules: list[str] = []
    registry_paths: set[str] = set()
    for entry in registry:
        if type(entry) is not dict or set(entry) != {
            "is_package",
            "module",
            "path",
            "sha256",
        }:
            raise ValueError("execution bundle module registry entry is invalid")
        module = _module_name(entry.get("module"), label="module registry module")
        path = _validate_payload_path(
            entry.get("path"),
            label="module registry path",
            role=SOURCE_ROLE,
        )
        is_package = entry.get("is_package")
        if type(is_package) is not bool:
            raise ValueError("execution bundle module registry package flag is invalid")
        expected_path = module.replace(".", "/")
        expected_path += "/__init__.py" if is_package else ".py"
        if path != expected_path:
            raise ValueError("execution bundle module registry path is not canonical")
        source_file = source_files.get(path)
        if source_file is None or source_file["sha256"] != entry.get("sha256"):
            raise ValueError("execution bundle module registry source binding failed")
        _require_sha256(entry.get("sha256"), "module registry SHA-256")
        modules.append(module)
        registry_paths.add(path)
    if modules != sorted(modules) or len(modules) != len(set(modules)):
        raise ValueError("execution bundle module registry is not uniquely sorted")
    if registry_paths != set(source_files):
        raise ValueError("execution bundle module registry does not cover all source")
    if manifest["entry_module"] not in set(modules):
        raise ValueError("execution bundle entry module is not registered")


def _validate_manifest(manifest: dict[str, Any], raw_manifest: bytes) -> list[dict[str, Any]]:
    if len(raw_manifest) > MAX_MANIFEST_BYTES:
        raise ValueError("execution bundle manifest exceeds the byte limit")
    expected_keys = {
        "archive_format",
        "bundle_content_sha256",
        "entry_module",
        "files",
        "kind",
        "manifest_sha256",
        "module_registry",
        "payload_root",
        RUNNER_ENVIRONMENT_MANIFEST_SHA256_FIELD,
        "runtime_assets",
        "schema_version",
    }
    if set(manifest) != expected_keys:
        raise ValueError("execution bundle manifest has unsupported fields")
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or manifest.get("kind") != BUNDLE_KIND
        or manifest.get("archive_format") != BUNDLE_ARCHIVE_FORMAT
        or manifest.get("payload_root") != PAYLOAD_ROOT
    ):
        raise ValueError("unsupported execution bundle manifest")
    if raw_manifest != _canonical_json_bytes(manifest):
        raise ValueError("execution bundle manifest is not canonical JSON")
    declared = _require_sha256(
        manifest.get("manifest_sha256"),
        "execution bundle manifest SHA-256",
    )
    if declared != _manifest_sha256(manifest):
        raise ValueError("execution bundle manifest SHA-256 is invalid")
    _module_name(manifest.get("entry_module"))
    _require_sha256(
        manifest.get(RUNNER_ENVIRONMENT_MANIFEST_SHA256_FIELD),
        "runner environment manifest SHA-256",
    )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("execution bundle manifest lacks files")
    if len(files) > MAX_FILE_ENTRIES:
        raise ValueError("execution bundle manifest exceeds the file-entry limit")
    parsed_files = [_validate_file_entry(entry) for entry in files]
    paths = [str(entry["path"]) for entry in parsed_files]
    if len(paths) != len(set(paths)) or paths != sorted(paths):
        raise ValueError("execution bundle manifest files are not uniquely sorted")
    _validate_payload_layout(paths)
    if sum(int(entry["size"]) for entry in parsed_files) > MAX_TOTAL_BYTES:
        raise ValueError("execution bundle manifest exceeds the total size limit")
    content_digest = _require_sha256(
        manifest.get("bundle_content_sha256"),
        "execution bundle content SHA-256",
    )
    if content_digest != _canonical_content_sha256(parsed_files):
        raise ValueError("execution bundle manifest content digest is invalid")
    _validate_module_registry(manifest, parsed_files)
    asset_paths = [
        entry["path"] for entry in parsed_files if entry["role"] == RUNTIME_ASSET_ROLE
    ]
    runtime_assets = manifest.get("runtime_assets")
    if (
        not isinstance(runtime_assets, list)
        or any(type(path) is not str for path in runtime_assets)
        or runtime_assets != asset_paths
    ):
        raise ValueError("execution bundle manifest runtime assets are invalid")
    return parsed_files


def _stage_tree_sets_fallback(
    root: _DirectoryAnchor,
    *,
    maximum_files: int,
    maximum_directories: int,
) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()

    def walk(current: Path, prefix: str) -> None:
        try:
            entries = os.scandir(current)
        except OSError as exc:
            raise ValueError(f"cannot enumerate staged execution bundle: {current}") from exc
        with entries:
            for entry in entries:
                name = entry.name
                candidate = current / name
                relative = name if not prefix else f"{prefix}/{name}"
                member = _validate_stage_member_name(relative)
                metadata = _lstat(candidate)
                _reject_link_or_reparse(candidate, metadata, label="staged execution bundle")
                if stat.S_ISDIR(metadata.st_mode):
                    if len(directories) >= maximum_directories:
                        raise ValueError(
                            "staged execution bundle exceeds the directory-entry limit"
                        )
                    directories.add(member)
                    walk(candidate, member)
                elif stat.S_ISREG(metadata.st_mode):
                    if len(files) >= maximum_files:
                        raise ValueError(
                            "staged execution bundle exceeds the file-entry limit"
                        )
                    files.add(member)
                else:
                    raise ValueError("staged execution bundle contains a non-file member")

    _revalidate_directory_components(root.components, label="execution bundle stage root")
    walk(root.path, "")
    _revalidate_directory_components(root.components, label="execution bundle stage root")
    return files, directories


def _stage_tree_sets(
    root: _DirectoryAnchor,
    *,
    maximum_files: int,
    maximum_directories: int,
) -> tuple[set[str], set[str]]:
    if type(maximum_files) is not int or maximum_files < 0:
        raise ValueError("staged file-entry limit must be a non-negative integer")
    if type(maximum_directories) is not int or maximum_directories < 0:
        raise ValueError("staged directory-entry limit must be a non-negative integer")
    root_descriptor = _open_bound_directory(
        root.path,
        root.components,
        label="execution bundle stage root",
    )
    if root_descriptor is None:
        return _stage_tree_sets_fallback(
            root,
            maximum_files=maximum_files,
            maximum_directories=maximum_directories,
        )
    files: set[str] = set()
    directories: set[str] = set()

    def walk(current_path: Path, descriptor: int, prefix: str) -> None:
        try:
            entries = os.scandir(descriptor)
        except OSError as exc:
            raise ValueError(
                f"cannot enumerate staged execution bundle: {current_path}"
            ) from exc
        with entries:
            for entry in entries:
                name = entry.name
                candidate = current_path / name
                relative = name if not prefix else f"{prefix}/{name}"
                member = _validate_stage_member_name(relative)
                try:
                    metadata = _parent_entry_lstat(current_path, descriptor, name)
                except FileNotFoundError as exc:
                    raise ValueError(
                        f"staged execution bundle changed during enumeration: {candidate}"
                    ) from exc
                _reject_link_or_reparse(candidate, metadata, label="staged execution bundle")
                if stat.S_ISDIR(metadata.st_mode):
                    if len(directories) >= maximum_directories:
                        raise ValueError(
                            "staged execution bundle exceeds the directory-entry limit"
                        )
                    directories.add(member)
                    child_path, child_descriptor = _open_checked_child_directory(
                        parent_path=current_path,
                        parent_descriptor=descriptor,
                        name=name,
                        label="staged execution bundle",
                    )
                    try:
                        walk(child_path, child_descriptor, member)
                    finally:
                        os.close(child_descriptor)
                elif stat.S_ISREG(metadata.st_mode):
                    if len(files) >= maximum_files:
                        raise ValueError(
                            "staged execution bundle exceeds the file-entry limit"
                        )
                    files.add(member)
                else:
                    raise ValueError("staged execution bundle contains a non-file member")

    try:
        _revalidate_directory_components(root.components, label="execution bundle stage root")
        walk(root.path, root_descriptor, "")
        _revalidate_directory_components(root.components, label="execution bundle stage root")
    finally:
        os.close(root_descriptor)
    return files, directories


def _expected_stage_directories(files: set[str]) -> set[str]:
    directories: set[str] = set()
    for name in files:
        parts = name.split("/")
        for index in range(1, len(parts)):
            directories.add("/".join(parts[:index]))
    return directories


def _validate_stage_layout(files: set[str], directories: set[str]) -> None:
    folded_names: dict[str, str] = {}
    for name in sorted(files | directories):
        folded = name.casefold()
        previous = folded_names.get(folded)
        if previous is not None and previous != name:
            raise ValueError("staged execution bundle contains case-colliding paths")
        folded_names[folded] = name
    folded_files = {name.casefold() for name in files}
    for file_name in folded_files:
        prefix = f"{file_name}/"
        if any(
            candidate != file_name and candidate.startswith(prefix)
            for candidate in folded_names
        ):
            raise ValueError("staged execution bundle contains a file/directory path conflict")


def _snapshot_stage(
    stage_root: Path,
    *,
    expected_bundle_manifest_sha256: str,
) -> tuple[_DirectoryAnchor, dict[str, Any], dict[str, _SnapshotModule]]:
    try:
        root = _pin_directory(stage_root, label="execution bundle stage root")
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("stage root must be a real directory") from exc
    manifest_bytes = _read_anchored_relative_bytes(
        root,
        MANIFEST_NAME,
        label="staged execution-bundle manifest",
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    manifest = _strict_json_loads(manifest_bytes, label="staged execution-bundle manifest")
    files = _validate_manifest(manifest, manifest_bytes)
    declared_manifest_sha256 = str(manifest["manifest_sha256"])
    if declared_manifest_sha256 != expected_bundle_manifest_sha256:
        raise ValueError("bundle manifest SHA-256 differs from expected")
    expected_files = {
        MANIFEST_NAME,
        *(f"{PAYLOAD_ROOT}/{entry['path']}" for entry in files),
    }
    expected_directories = _expected_stage_directories(expected_files)
    observed_files, observed_directories = _stage_tree_sets(
        root,
        maximum_files=len(expected_files),
        maximum_directories=len(expected_directories),
    )
    _validate_stage_layout(observed_files, observed_directories)
    if observed_files != expected_files:
        raise ValueError("staged execution bundle files differ from manifest")
    if observed_directories != expected_directories:
        raise ValueError("staged execution bundle directories differ from manifest")
    _validate_payload_layout(
        {
            name[len(f"{PAYLOAD_ROOT}/"):]
            for name in observed_files
            if name != MANIFEST_NAME
        }
    )
    member_bytes: dict[str, bytes] = {MANIFEST_NAME: manifest_bytes}
    for entry in files:
        member = f"{PAYLOAD_ROOT}/{entry['path']}"
        data = _read_anchored_relative_bytes(
            root,
            member,
            label="staged execution-bundle member",
            maximum_bytes=int(entry["size"]),
        )
        if len(data) != entry["size"] or _sha256_bytes(data) != entry["sha256"]:
            raise ValueError(f"staged execution bundle member binding failed: {entry['path']}")
        member_bytes[member] = data
    _revalidate_directory_components(root.components, label="execution bundle stage root")
    sources: dict[str, _SnapshotModule] = {}
    for entry in manifest["module_registry"]:
        module = str(entry["module"])
        relative = str(entry["path"])
        sources[module] = _SnapshotModule(
            data=member_bytes[f"{PAYLOAD_ROOT}/{relative}"],
            origin=root.path / PAYLOAD_ROOT / Path(relative),
            is_package=bool(entry["is_package"]),
        )
    return root, manifest, sources


class _SnapshotBundleLoader(importlib.abc.Loader):
    """Execute only authenticated source bytes under a logical staged origin."""

    def __init__(self, module_name: str, source: _SnapshotModule) -> None:
        self.module_name = module_name
        self.source = source

    def create_module(self, spec: object) -> None:
        return None

    def get_filename(self, fullname: str) -> str:
        if fullname != self.module_name:
            raise ImportError(f"unexpected bundle module request: {fullname}")
        return str(self.source.origin)

    def is_package(self, fullname: str) -> bool:
        if fullname != self.module_name:
            raise ImportError(f"unexpected bundle module request: {fullname}")
        return self.source.is_package

    def exec_module(self, module: ModuleType) -> None:
        module.__file__ = str(self.source.origin)
        module.__cached__ = None
        if self.source.is_package:
            # Registered submodules are supplied by the meta-path finder.  An
            # empty package path prevents PathFinder from reopening mutable
            # staged files for an unregistered dynamic submodule import.
            module.__path__ = []
        code = compile(self.source.data, str(self.source.origin), "exec", dont_inherit=True)
        exec(code, module.__dict__)


class _SnapshotBundleFinder(importlib.abc.MetaPathFinder):
    """Resolve only registered local modules from an authenticated byte mapping."""

    def __init__(self, sources: Mapping[str, _SnapshotModule]) -> None:
        self._loaders = {
            name: _SnapshotBundleLoader(name, source)
            for name, source in sources.items()
        }
        self._package_names = tuple(
            name for name, source in sources.items() if source.is_package
        )

    def loader_for(self, module_name: str) -> _SnapshotBundleLoader:
        return self._loaders[module_name]

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> importlib.machinery.ModuleSpec | None:
        loader = self._loaders.get(fullname)
        if loader is not None:
            return importlib.util.spec_from_loader(
                fullname,
                loader,
                origin=str(loader.source.origin),
                is_package=loader.source.is_package,
            )
        if any(fullname.startswith(f"{package}.") for package in self._package_names):
            raise ModuleNotFoundError(f"No module named {fullname!r}")
        return None


def _assert_snapshot_module(
    module: ModuleType,
    *,
    module_name: str,
    finder: _SnapshotBundleFinder,
) -> Path:
    expected_loader = finder.loader_for(module_name)
    if getattr(module, "__loader__", None) is not expected_loader:
        raise ValueError(f"loaded module is not backed by the staged snapshot: {module_name}")
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        raise ValueError(f"loaded module has no file origin: {module_name}")
    expected = expected_loader.source.origin
    if _absolute_path(origin) != expected:
        raise ValueError(f"loaded module origin differs from staged registry: {module_name}")
    return expected


def _is_lexically_below(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _configure_import_path(stage_root: Path) -> None:
    """Retain only interpreter bootstrap paths, never the mutable stage tree."""

    base_prefix = _absolute_path(sys.base_prefix)
    bootstrap_paths: list[str] = []
    seen: set[str] = set()
    for raw_path in sys.path:
        if not raw_path:
            raise ValueError("isolated startup has an empty import path")
        path = _absolute_path(raw_path)
        if not _is_lexically_below(path, base_prefix):
            raise ValueError("isolated startup import path is outside the runner base prefix")
        if _is_lexically_below(path, stage_root):
            raise ValueError("isolated startup path overlaps the staged payload")
        value = str(path)
        if value not in seen:
            seen.add(value)
            bootstrap_paths.append(value)
    sys.path[:] = bootstrap_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--runner-environment-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-runner-environment-manifest-sha256",
        required=True,
    )
    parser.add_argument("--expected-bundle-manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        _require_isolated_startup()
        expected_runner_digest = _require_sha256(
            args.expected_runner_environment_manifest_sha256,
            "expected runner environment manifest SHA-256",
        )
        expected_bundle_digest = _require_sha256(
            args.expected_bundle_manifest_sha256,
            "expected bundle manifest SHA-256",
        )
        root, manifest, sources = _snapshot_stage(
            _absolute_path(args.stage_root),
            expected_bundle_manifest_sha256=expected_bundle_digest,
        )
        bound_runner_digest = _require_sha256(
            manifest.get(RUNNER_ENVIRONMENT_MANIFEST_SHA256_FIELD),
            "bundle runner environment manifest SHA-256",
        )
        if bound_runner_digest != expected_runner_digest:
            raise ValueError("runner environment manifest SHA-256 differs from bundle")
        runner_manifest_bytes = _read_stable_bytes(
            _absolute_path(args.runner_environment_manifest),
            label="runner environment manifest",
            maximum_bytes=MAX_MANIFEST_BYTES,
        )
        if _sha256_bytes(runner_manifest_bytes) != expected_runner_digest:
            raise ValueError("runner environment manifest bytes differ from expected")
        entry_module = _module_name(manifest.get("entry_module"))
        if entry_module not in sources:
            raise ValueError("staged entry module is absent from registry")
        for name in sources:
            if name in sys.modules:
                raise ValueError(f"staged registry module is already loaded: {name}")
        finder = _SnapshotBundleFinder(sources)
        original_meta_path = list(sys.meta_path)
        original_sys_path = list(sys.path)
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            _configure_import_path(root.path)
            sys.meta_path.insert(0, finder)
            importlib.invalidate_caches()
            entry = importlib.import_module(entry_module)
            entry_origin = _assert_snapshot_module(
                entry,
                module_name=entry_module,
                finder=finder,
            )
            for name in sources:
                loaded = sys.modules.get(name)
                if loaded is not None:
                    _assert_snapshot_module(
                        loaded,
                        module_name=name,
                        finder=finder,
                    )
        finally:
            sys.meta_path[:] = original_meta_path
            sys.path[:] = original_sys_path
            sys.dont_write_bytecode = previous_dont_write_bytecode
    except (ImportError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "bundle_manifest_sha256": expected_bundle_digest,
                "entry_module": entry_module,
                "module_origin": str(entry_origin),
                "runner_environment_manifest_sha256": expected_runner_digest,
                "stage_root": str(root.path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
