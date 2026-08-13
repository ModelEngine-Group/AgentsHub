"""Immutable, source-only execution bundles for NL2SQL evaluation.

The bundle format is deliberately narrow.  It snapshots the static local Python
import closure of one entry module together with an explicit allow-list of
runtime assets.  It does not inspect a dataset, open a database, or invoke an
evaluator.

Every bundle also binds the SHA-256 of a separately verified runner/dependency
environment manifest.  That external manifest must identify the exact runner,
inner assertion entrypoint, interpreter libraries, and dependency bytes; package
versions alone are not an identity.  The source-only bundle does not include
those third-party requirements, so its isolated inner assertion deliberately
uses ``python -I -S
-B`` and cannot be treated as an evaluator launch.

A future outer coordinator is expected to verify both attested artifacts,
extract the verified in-memory bundle snapshot into a fresh directory, and
launch the isolated inner origin check before it opens any protected input.
This module intentionally does not wire that coordinator to an evaluator.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import io
import json
import os
import re
import secrets
import stat
import struct
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Iterable, Mapping


BUNDLE_SCHEMA_VERSION = 2
BUNDLE_KIND = "nl2sql_execution_bundle"
BUNDLE_ARCHIVE_FORMAT = "zip-stored-v1"
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
BUNDLE_PAYLOAD_ROOT = "payload"
RUNNER_ENVIRONMENT_MANIFEST_SHA256_FIELD = "runner_environment_manifest_sha256"
SOURCE_ROLE = "python_source"
RUNTIME_ASSET_ROLE = "runtime_asset"
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_FILE_ENTRIES = 10_000
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_RELATIVE_PATH_CHARS = 512
MAX_PATH_COMPONENTS = 32
MAX_PATH_COMPONENT_CHARS = 128
# Stored ZIP members plus bounded metadata leave this deliberately conservative
# allowance above the payload limit while rejecting arbitrarily large archives
# before their bytes are read into memory.
MAX_ARCHIVE_BYTES = MAX_TOTAL_BYTES + 64 * 1024 * 1024

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
_DYNAMIC_IMPORT_CALLS = frozenset(
    {
        "__import__",
        "builtins.__import__",
        "importlib.import_module",
    }
)
_DYNAMIC_LOADING_CALLS = frozenset(
    {
        "builtins.compile",
        "builtins.eval",
        "builtins.exec",
        "compile",
        "eval",
        "exec",
        "imp.load_compiled",
        "imp.load_dynamic",
        "imp.load_module",
        "imp.load_package",
        "imp.load_source",
        "importlib.machinery.ExtensionFileLoader",
        "importlib.machinery.SourceFileLoader",
        "importlib.machinery.SourcelessFileLoader",
        "importlib.util.find_spec",
        "importlib.util.module_from_spec",
        "importlib.util.spec_from_file_location",
        "pkgutil.find_loader",
        "pkgutil.get_loader",
        "pkgutil.resolve_name",
        "runpy.run_module",
        "runpy.run_path",
        "zipimport.zipimporter",
    }
)
_DYNAMIC_LOADING_LEAVES = frozenset(
    {
        "ExtensionFileLoader",
        "SourceFileLoader",
        "SourcelessFileLoader",
        "exec_module",
        "find_loader",
        "find_spec",
        "get_loader",
        "import_module",
        "load_compiled",
        "load_dynamic",
        "load_module",
        "load_package",
        "load_source",
        "module_from_spec",
        "resolve_name",
        "run_module",
        "run_path",
        "spec_from_file_location",
        "zipimporter",
    }
)
_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)


@dataclass(frozen=True)
class RuntimeAsset:
    """One explicit non-code file copied into the bundle payload."""

    destination: str
    source: Path


@dataclass(frozen=True)
class SourceModule:
    """A local module included in the static import closure."""

    name: str
    source_path: Path
    relative_path: str
    is_package: bool


@dataclass(frozen=True)
class BuiltExecutionBundle:
    """Identity returned after a create-once archive publication.

    ``manifest_sha256`` identifies the exact canonical manifest bytes.  The
    manifest's embedded ``manifest_sha256`` field is a separate, self-excluding
    canonical-content check because a JSON document cannot contain its own raw
    byte digest.
    """

    archive_path: Path
    archive_sha256: str
    manifest_sha256: str
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class VerifiedExecutionBundle:
    """A fully verified in-memory archive snapshot.

    ``manifest_sha256`` identifies the captured raw manifest bytes.
    ``member_bytes`` is retained so extraction cannot re-open the archive after
    verification and thereby race a changed on-disk archive.
    """

    archive_path: Path
    archive_sha256: str
    manifest_sha256: str
    manifest: Mapping[str, Any]
    member_bytes: Mapping[str, bytes] = field(repr=False)


@dataclass(frozen=True)
class StagedExecutionBundle:
    """A verified bundle extracted below a newly-created directory.

    ``manifest_sha256`` identifies the captured raw manifest bytes.
    ``member_bytes`` is the byte snapshot authenticated while the stage tree was
    verified.  Consumers must execute only this snapshot rather than reopening
    mutable staged pathnames after verification.
    """

    root: Path
    payload_root: Path
    archive_sha256: str
    manifest_sha256: str
    manifest: Mapping[str, Any]
    member_bytes: Mapping[str, bytes] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
    )


@dataclass(frozen=True)
class _DirectoryAnchor:
    """A link-free directory and immutable identities for its lexical parents."""

    path: Path
    components: tuple[tuple[Path, os.stat_result], ...]


@dataclass(frozen=True)
class _SnapshotModule:
    """Authenticated source bytes and logical origin for one bundled module."""

    data: bytes
    origin: Path
    is_package: bool


@dataclass(frozen=True)
class _SourceSnapshot:
    module: SourceModule
    data: bytes


@dataclass(frozen=True)
class _AssetSnapshot:
    destination: str
    source_path: Path
    data: bytes


def outer_coordinator_contract() -> dict[str, Any]:
    """Return the documented, deliberately unimplemented coordinator boundary.

    The returned data is descriptive only.  In particular, this function never
    imports or starts an evaluator and never opens a runtime asset.  The runner
    environment requirement is intentionally fail-closed: this source-only
    bundle must not inherit live site packages as an implicit dependency source.
    """

    return {
        "schema_version": 2,
        "kind": "nl2sql_execution_bundle_outer_coordinator_contract",
        "implemented": False,
        "evaluator_wired": False,
        "preverification_boundary": {
            "stdlib_only": True,
            "imports_dki_before_verification": False,
        },
        "external_attestation": {
            "required": True,
            "implemented": False,
            "caller_selected_authority_allowed": False,
            "missing_attestation_allows_execution": False,
        },
        "required_sequence": [
            "obtain the expected archive SHA-256 from a trusted attestation",
            "independently verify the exact runner/dependency-environment manifest bytes",
            "require that runner manifest SHA-256 to match the bundle binding",
            "verify the archive before opening protected evaluation inputs",
            "extract the verified in-memory snapshot into a fresh directory",
            "launch the source-origin assertion with python -I -S -B",
            "only a later coordinator may configure verified dependencies and begin protected evaluation work",
        ],
        "runner_environment": {
            "required": True,
            "identity": "SHA-256 of exact runner/dependency-environment manifest bytes",
            "version_only_is_insufficient": True,
            "must_bind": [
                "the Python executable and interpreter library bytes",
                "the isolated inner assertion entrypoint bytes",
                "every dependency import-root and file byte identity",
                "the approved dependency import-path order",
            ],
            "ambient_site_fallback_allowed": False,
            "implemented": False,
        },
        "inner_boundary": {
            "requires_python_isolated_mode": True,
            "requires_no_site_mode": True,
            "requires_no_bytecode_writes": True,
            "ignores_pythonpath": True,
            "blocks_sitecustomize_and_pth_processing": True,
            "requires_staged_entry_module_origin": True,
            "requires_runner_environment_manifest_byte_identity": True,
            "allows_ambient_python_dependencies": False,
        },
    }


def isolated_inner_command(
    *,
    python_executable: str | Path,
    inner_entrypoint: Path,
    stage_root: Path,
    runner_environment_manifest: Path,
    runner_environment_manifest_sha256: str,
    bundle_manifest_sha256: str,
) -> list[str]:
    """Build, but do not execute, a fail-closed inner origin assertion command.

    The caller must independently verify ``runner_environment_manifest``, the
    exact ``inner_entrypoint`` bytes, and retain the authenticated bundle
    manifest digest before launch.  The inner process rejects a replacement
    stage manifest before it reads or executes
    any staged source.  This command intentionally adds no dependency paths:
    ``-S`` prevents site-package, ``.pth``, and ``sitecustomize`` fallback.  A
    later coordinator must add a separately verified dependency environment
    before it can execute real ``dki`` code.
    """

    expected_runner_digest = _require_sha256(
        runner_environment_manifest_sha256,
        "runner environment manifest SHA-256",
    )
    expected_bundle_manifest_digest = _require_sha256(
        bundle_manifest_sha256,
        "bundle manifest SHA-256",
    )
    return [
        str(python_executable),
        "-I",
        "-S",
        "-B",
        str(_absolute_path(inner_entrypoint)),
        "--stage-root",
        str(_absolute_path(stage_root)),
        "--runner-environment-manifest",
        str(_absolute_path(runner_environment_manifest)),
        "--expected-runner-environment-manifest-sha256",
        expected_runner_digest,
        "--expected-bundle-manifest-sha256",
        expected_bundle_manifest_digest,
    ]


def canonical_relative_path(value: str, *, label: str = "path") -> str:
    """Require an ASCII, POSIX, non-traversing payload-relative path."""

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


def _path_text(path: str | os.PathLike[str]) -> str:
    value = os.fspath(path)
    if not isinstance(value, str):
        raise ValueError("path must be text")
    if "\x00" in value:
        raise ValueError("path must not contain a NUL byte")
    return value


def _absolute_path(path: str | os.PathLike[str]) -> Path:
    """Normalise a filesystem path without resolving links or reparse points."""

    return Path(os.path.abspath(_path_text(path)))


def _is_link_or_reparse(metadata: object) -> bool:
    """Recognise symlinks, junctions, and every Windows reparse point."""

    return stat.S_ISLNK(getattr(metadata, "st_mode", 0)) or bool(
        getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _lstat(path: Path) -> os.stat_result:
    return os.lstat(path)


def _reject_link_or_reparse(path: Path, metadata: object, *, label: str) -> None:
    if _is_link_or_reparse(metadata):
        raise ValueError(f"{label} contains a symlink or reparse component: {path}")


def _is_link_like(path: Path) -> bool:
    """Return whether a path is a symlink, junction, or other reparse point."""

    try:
        return _is_link_or_reparse(_lstat(Path(path)))
    except FileNotFoundError:
        return False


def _require_byte_limit(value: int | None, *, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} byte limit must be a non-negative integer")
    return value


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
    """Compare pathname and descriptor file identities across Windows semantics."""

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


def _same_directory_identity(
    before: os.stat_result,
    after: os.stat_result,
) -> bool:
    """Compare directories without treating ordinary child writes as swaps."""

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
    """Pin every existing lexical component of an absolute directory path."""

    candidate = _absolute_path(path)
    if not candidate.is_absolute():
        raise ValueError(f"{label} must be absolute before component validation")
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


def _ensure_directory_components(
    path: Path,
    *,
    label: str,
) -> tuple[tuple[Path, os.stat_result], ...]:
    """Create and validate directories one lexical component at a time."""

    candidate = _absolute_path(path)
    if not candidate.is_absolute():
        raise ValueError(f"{label} must be absolute before component validation")
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
        except FileNotFoundError:
            try:
                os.mkdir(current, mode=0o700)
            except FileExistsError:
                pass
            metadata = _lstat(current)
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
    """Fail closed when a checked parent component is later replaced."""

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
    """Bind POSIX child operations to checked, no-follow directory descriptors.

    Opening the complete pathname after a lexical check would leave a window for
    an ancestor replacement.  POSIX therefore descends one component at a time
    from the filesystem anchor through ``dir_fd`` operations.  Windows has no
    standard-library ``openat`` equivalent; its fallback relies on repeated
    lexical identity checks and rejects every reparse point.
    """

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
                raise ValueError(
                    f"{label} cannot be opened safely: {path}"
                ) from exc
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
    # Read one mutation sentinel only after the authenticated initial size.  A
    # growing file is rejected without allocating its unbounded appended bytes.
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


def read_stable_bytes(
    path: Path,
    *,
    label: str = "file",
    maximum_bytes: int | None = None,
) -> bytes:
    """Snapshot one regular file after validating every lexical path component.

    ``maximum_bytes`` is checked from authenticated metadata before the file is
    opened, so a caller never reads an oversized candidate just to reject it.
    No caller uses a resolved path as an authorization decision.  On POSIX the
    final open is relative to a checked parent descriptor; every platform
    revalidates the link-free parent identities and the final pathname after the
    descriptor read.
    """

    maximum_bytes = _require_byte_limit(maximum_bytes, label=label)
    candidate = _absolute_path(path)
    parent_snapshots = _capture_directory_components(candidate.parent, label=label)
    try:
        path_stat = _lstat(candidate)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {candidate}") from exc
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


def discover_local_import_closure(
    source_root: Path,
    entry_module: str,
    *,
    extra_modules: Iterable[str] = (),
) -> tuple[SourceModule, ...]:
    """Discover the static local import closure without reading external code."""

    snapshots = _snapshot_import_closure(
        source_root=source_root,
        entry_module=entry_module,
        extra_modules=extra_modules,
    )
    return tuple(snapshot.module for snapshot in snapshots)


def build_execution_bundle(
    output_archive: Path,
    *,
    source_root: Path,
    entry_module: str,
    runner_environment_manifest_sha256: str,
    runtime_assets: (
        Mapping[str, Path | str]
        | Iterable[RuntimeAsset | tuple[str, Path | str]]
        | None
    ) = None,
    extra_modules: Iterable[str] = (),
) -> BuiltExecutionBundle:
    """Create a deterministic, create-once execution archive.

    ``entry_module`` and ``extra_modules`` identify local Python modules below
    ``source_root``.  ``runtime_assets`` is an explicit ``destination ->
    source`` mapping (or iterable of :class:`RuntimeAsset` / pairs).  Assets
    cannot be Python code, so all executable files remain in the static module
    registry.  ``runner_environment_manifest_sha256`` binds the archive to the
    exact bytes of a separately verified runner/dependency-environment manifest;
    it is not a package-version claim and does not permit ambient site packages.
    """

    runner_digest = _require_sha256(
        runner_environment_manifest_sha256,
        "runner environment manifest SHA-256",
    )
    snapshots = _snapshot_import_closure(
        source_root=source_root,
        entry_module=entry_module,
        extra_modules=extra_modules,
    )
    source_bytes = sum(len(snapshot.data) for snapshot in snapshots)
    assets = _snapshot_runtime_assets(
        runtime_assets,
        maximum_total_bytes=MAX_TOTAL_BYTES - source_bytes,
        maximum_entries=MAX_FILE_ENTRIES - len(snapshots),
    )
    manifest, payload_bytes = _build_manifest_and_payload(
        entry_module=entry_module,
        runner_environment_manifest_sha256=runner_digest,
        source_snapshots=snapshots,
        asset_snapshots=assets,
    )
    manifest_bytes = _canonical_json_bytes(manifest)
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError("execution bundle manifest exceeds the byte limit")
    members = {BUNDLE_MANIFEST_NAME: manifest_bytes}
    members.update(
        {
            _payload_member_name(path): data
            for path, data in payload_bytes.items()
        }
    )
    archive_bytes = _deterministic_archive_bytes(members)
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("execution bundle archive exceeds the byte limit")
    archive_path = _absolute_path(output_archive)
    _publish_bytes_create_once(archive_path, archive_bytes)
    frozen_manifest = _freeze_json(manifest)
    return BuiltExecutionBundle(
        archive_path=archive_path,
        archive_sha256=_sha256_bytes(archive_bytes),
        manifest_sha256=str(manifest["manifest_sha256"]),
        manifest=frozen_manifest,
    )


def verify_execution_bundle(
    archive_path: Path,
    *,
    expected_archive_sha256: str | None = None,
) -> VerifiedExecutionBundle:
    """Read an archive once, validate its exact layout, and retain its snapshot."""

    path = _absolute_path(archive_path)
    raw_archive = read_stable_bytes(
        path,
        label="execution bundle archive",
        maximum_bytes=MAX_ARCHIVE_BYTES,
    )
    archive_sha256 = _sha256_bytes(raw_archive)
    if expected_archive_sha256 is not None:
        _require_sha256(expected_archive_sha256, "expected archive SHA-256")
        if archive_sha256 != expected_archive_sha256:
            raise ValueError("execution bundle archive SHA-256 differs from expected")
    expected_archive_entries = _validate_archive_directory_record(raw_archive)
    try:
        with zipfile.ZipFile(io.BytesIO(raw_archive), "r") as archive:
            members = _safe_archive_members(
                archive,
                expected_entries=expected_archive_entries,
            )
            manifest_info = members.get(BUNDLE_MANIFEST_NAME)
            if manifest_info is None:
                raise ValueError("execution bundle archive lacks its manifest")
            manifest_bytes = archive.read(manifest_info)
            manifest = _strict_json_loads(
                manifest_bytes,
                label="execution bundle manifest",
            )
            file_entries = _validate_manifest(manifest, manifest_bytes)
            expected_names = [
                BUNDLE_MANIFEST_NAME,
                *[_payload_member_name(str(entry["path"])) for entry in file_entries],
            ]
            actual_names = list(members)
            if actual_names != expected_names:
                raise ValueError("execution bundle archive members differ from manifest")
            member_bytes: dict[str, bytes] = {
                BUNDLE_MANIFEST_NAME: manifest_bytes,
            }
            for entry in file_entries:
                member_name = _payload_member_name(str(entry["path"]))
                data = archive.read(members[member_name])
                if len(data) != entry["size"] or _sha256_bytes(data) != entry["sha256"]:
                    raise ValueError(
                        "execution bundle member binding failed: "
                        f"{entry['path']}"
                    )
                member_bytes[member_name] = data
    except zipfile.BadZipFile as exc:
        raise ValueError("execution bundle archive is not a valid ZIP file") from exc
    frozen_manifest = _freeze_json(manifest)
    return VerifiedExecutionBundle(
        archive_path=path,
        archive_sha256=archive_sha256,
        manifest_sha256=str(manifest["manifest_sha256"]),
        manifest=frozen_manifest,
        member_bytes=MappingProxyType(member_bytes),
    )


def extract_execution_bundle(
    verified: VerifiedExecutionBundle,
    destination: Path,
) -> StagedExecutionBundle:
    """Safely materialize a verified snapshot into a newly-created directory.

    The function accepts only :class:`VerifiedExecutionBundle`, never an
    archive path, so extraction cannot silently re-open a changed archive.  A
    failure leaves the newly-created destination in place rather than replacing
    or deleting a directory owned by another process.
    """

    if not isinstance(verified, VerifiedExecutionBundle):
        raise TypeError("extraction requires a VerifiedExecutionBundle snapshot")
    root = _create_fresh_directory(_absolute_path(destination))
    _write_stage_file(
        root,
        BUNDLE_MANIFEST_NAME,
        verified.member_bytes[BUNDLE_MANIFEST_NAME],
    )
    files = verified.manifest["files"]
    for entry in files:
        path = str(entry["path"])
        _write_stage_file(
            root,
            _payload_member_name(path),
            verified.member_bytes[_payload_member_name(path)],
        )
    staged = verify_staged_execution_tree(root.path)
    return StagedExecutionBundle(
        root=staged.root,
        payload_root=staged.payload_root,
        archive_sha256=verified.archive_sha256,
        manifest_sha256=staged.manifest_sha256,
        manifest=staged.manifest,
        member_bytes=staged.member_bytes,
    )


def verify_staged_execution_tree(stage_root: Path) -> StagedExecutionBundle:
    """Authenticate every staged member into one immutable byte snapshot.

    The returned snapshot, rather than its mutable pathname tree, is the only
    acceptable input to the subsequent source importer.
    """

    candidate = _absolute_path(stage_root)
    try:
        root = _pin_directory(candidate, label="execution bundle stage root")
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("execution bundle stage root must be a real directory") from exc
    manifest_bytes = _read_anchored_relative_bytes(
        root,
        BUNDLE_MANIFEST_NAME,
        label="staged execution bundle manifest",
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    manifest = _strict_json_loads(
        manifest_bytes,
        label="staged execution bundle manifest",
    )
    file_entries = _validate_manifest(manifest, manifest_bytes)
    expected_files = {
        BUNDLE_MANIFEST_NAME,
        *[_payload_member_name(str(entry["path"])) for entry in file_entries],
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
        (
            name[len(f"{BUNDLE_PAYLOAD_ROOT}/"):]
            for name in observed_files
            if name != BUNDLE_MANIFEST_NAME
        ),
        label="staged execution bundle",
    )
    member_bytes: dict[str, bytes] = {BUNDLE_MANIFEST_NAME: manifest_bytes}
    for entry in file_entries:
        member_name = _payload_member_name(str(entry["path"]))
        data = _read_anchored_relative_bytes(
            root,
            member_name,
            label="staged execution bundle member",
            maximum_bytes=int(entry["size"]),
        )
        if len(data) != entry["size"] or _sha256_bytes(data) != entry["sha256"]:
            raise ValueError(
                "staged execution bundle member binding failed: "
                f"{entry['path']}"
            )
        member_bytes[member_name] = data
    _revalidate_directory_components(
        root.components,
        label="execution bundle stage root",
    )
    return StagedExecutionBundle(
        root=root.path,
        payload_root=root.path / BUNDLE_PAYLOAD_ROOT,
        archive_sha256="",
        manifest_sha256=str(manifest["manifest_sha256"]),
        manifest=_freeze_json(manifest),
        member_bytes=MappingProxyType(member_bytes),
    )


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
            # Registered submodules are served by the private finder.  Do not
            # leave a mutable staged directory on package search paths.
            module.__path__ = []
        code = compile(
            self.source.data,
            str(self.source.origin),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)


class _SnapshotBundleFinder(importlib.abc.MetaPathFinder):
    """Resolve registered local imports from one immutable byte mapping."""

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


def _snapshot_module_sources(
    checked: StagedExecutionBundle,
) -> tuple[dict[str, Path], dict[str, _SnapshotModule]]:
    registry = _registry_by_module(checked.manifest)
    package_flags = {
        str(entry["module"]): bool(entry["is_package"])
        for entry in checked.manifest["module_registry"]
    }
    sources: dict[str, _SnapshotModule] = {}
    for name, relative in registry.items():
        member_name = _payload_member_name(relative.as_posix())
        try:
            data = checked.member_bytes[member_name]
        except KeyError as exc:
            raise ValueError(f"staged source snapshot is missing: {relative}") from exc
        sources[name] = _SnapshotModule(
            data=data,
            origin=checked.payload_root / relative,
            is_package=package_flags[name],
        )
    return registry, sources


def _bootstrap_import_paths(stage_root: Path) -> list[str]:
    """Retain only interpreter bootstrap paths while snapshot code is imported."""

    base_prefix = _absolute_path(sys.base_prefix)
    retained: list[str] = []
    seen: set[str] = set()
    for raw_path in sys.path:
        if not raw_path:
            continue
        path = _absolute_path(raw_path)
        try:
            path.relative_to(base_prefix)
        except ValueError:
            continue
        try:
            path.relative_to(stage_root)
        except ValueError:
            pass
        else:
            continue
        value = str(path)
        if value not in seen:
            seen.add(value)
            retained.append(value)
    return retained


def _assert_snapshot_module(
    module: ModuleType,
    *,
    module_name: str,
    finder: _SnapshotBundleFinder,
) -> Path:
    expected_loader = finder.loader_for(module_name)
    if getattr(module, "__loader__", None) is not expected_loader:
        raise ValueError(f"loaded module is not backed by the staged snapshot: {module_name}")
    origin = _module_origin(module)
    if origin != expected_loader.source.origin:
        raise ValueError(f"loaded module origin differs from staged registry: {module_name}")
    return origin


def assert_staged_entry_module_origin(
    staged: StagedExecutionBundle,
) -> ModuleType:
    """Execute only a verified byte snapshot and prove its registered origins.

    The mutable extraction tree is reverified first, then every registered
    source byte is retained in memory.  A private meta-path finder executes
    that immutable mapping; it never reopens a staged module pathname and only
    retains interpreter bootstrap paths as import fallbacks.
    """

    if not isinstance(staged, StagedExecutionBundle):
        raise TypeError("module-origin assertion requires a StagedExecutionBundle")
    checked = verify_staged_execution_tree(staged.root)
    if checked.manifest_sha256 != staged.manifest_sha256:
        raise ValueError("staged execution bundle manifest identity changed")
    entry_module = str(checked.manifest["entry_module"])
    registry, sources = _snapshot_module_sources(checked)
    if entry_module not in registry:
        raise ValueError("staged entry module is absent from registry")
    for name in registry:
        if name in sys.modules:
            raise ValueError(f"staged registry module is already loaded: {name}")
    finder = _SnapshotBundleFinder(sources)
    original_meta_path = list(sys.meta_path)
    original_sys_path = list(sys.path)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sys.path[:] = _bootstrap_import_paths(checked.root)
        sys.meta_path.insert(0, finder)
        importlib.invalidate_caches()
        module = importlib.import_module(entry_module)
        _assert_snapshot_module(
            module,
            module_name=entry_module,
            finder=finder,
        )
        for name in registry:
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
    return module


def _snapshot_import_closure(
    *,
    source_root: Path,
    entry_module: str,
    extra_modules: Iterable[str],
) -> tuple[_SourceSnapshot, ...]:
    source_root_path = _absolute_path(source_root)
    try:
        root = _pin_directory(source_root_path, label="source root")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"source root does not exist: {source_root_path}") from exc
    _validate_bundled_module_name(entry_module, label="entry module")
    pending: set[str] = set()
    snapshots: dict[str, _SourceSnapshot] = {}
    total_bytes = 0

    def enqueue(name: str, *, required: bool) -> None:
        _validate_module_name(name, label="local module")
        if name in snapshots or name in pending:
            return
        source = _module_source(root, name)
        if source is None:
            if required:
                raise ValueError(f"local module is missing from source root: {name}")
            return
        _validate_bundled_module_name(name, label="local module")
        parent = name.rpartition(".")[0]
        if parent:
            parent_source = _module_source(root, parent)
            if parent_source is None or not parent_source[1]:
                raise ValueError(
                    "local module lacks a package initializer for parent: "
                    f"{parent}"
                )
            enqueue(parent, required=True)
        pending.add(name)

    enqueue(entry_module, required=True)
    extra_module_names: set[str] = set()
    for name in extra_modules:
        if type(name) is not str:
            raise ValueError("extra module names must be strings")
        if name not in extra_module_names and len(extra_module_names) >= MAX_FILE_ENTRIES:
            raise ValueError("extra modules exceed the file-entry limit")
        extra_module_names.add(name)
    for name in sorted(extra_module_names):
        enqueue(name, required=True)

    while pending:
        name = min(pending)
        pending.remove(name)
        if name in snapshots:
            continue
        source = _module_source(root, name)
        if source is None:
            raise ValueError(f"local module disappeared from source root: {name}")
        source_path, is_package = source
        relative_path = _module_relative_path(name, is_package=is_package)
        if len(snapshots) >= MAX_FILE_ENTRIES:
            raise ValueError("local import closure exceeds the file-entry limit")
        module = SourceModule(
            name=name,
            source_path=source_path,
            relative_path=relative_path,
            is_package=is_package,
        )
        remaining_bytes = MAX_TOTAL_BYTES - total_bytes
        if remaining_bytes < 0:
            raise ValueError("local import closure exceeds the total byte limit")
        data = _read_anchored_relative_bytes(
            root,
            relative_path,
            label=f"local module {name}",
            maximum_bytes=min(MAX_MEMBER_BYTES, remaining_bytes),
        )
        total_bytes += len(data)
        snapshot = _SourceSnapshot(module=module, data=data)
        snapshots[name] = snapshot
        for target, required in _import_targets(snapshot, root):
            enqueue(target, required=required)
    return tuple(snapshots[name] for name in sorted(snapshots))


def _module_relative_path(module_name: str, *, is_package: bool) -> str:
    module = _validate_module_name(module_name, label="module")
    relative = module.replace(".", "/")
    relative += "/__init__.py" if is_package else ".py"
    return _validate_payload_path(
        relative,
        label="local module path",
        role=SOURCE_ROLE,
    )


def _relative_components(relative: str, *, label: str) -> tuple[str, tuple[str, ...]]:
    canonical = canonical_relative_path(relative, label=label)
    return canonical, tuple(canonical.split("/"))


def _open_checked_child_directory(
    *,
    parent_path: Path,
    parent_descriptor: int,
    name: str,
    label: str,
) -> tuple[Path, int]:
    """Descend one POSIX directory entry without ever reopening its pathname."""

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
    """Windows fallback with repeated lexical checks for every component."""

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


def _anchored_relative_metadata(
    root: _DirectoryAnchor,
    relative: str,
    *,
    label: str,
) -> tuple[Path, os.stat_result] | None:
    """Inspect a regular leaf below a pinned directory without link resolution."""

    canonical, parts = _relative_components(relative, label=label)
    root_descriptor = _open_bound_directory(root.path, root.components, label=label)
    if root_descriptor is None:
        return _anchored_relative_metadata_fallback(root, canonical, label=label)
    current_path = root.path
    descriptor = root_descriptor
    try:
        _revalidate_directory_components(root.components, label=label)
        for component in parts[:-1]:
            try:
                child_path, child_descriptor = _open_checked_child_directory(
                    parent_path=current_path,
                    parent_descriptor=descriptor,
                    name=component,
                    label=label,
                )
            except FileNotFoundError:
                return None
            os.close(descriptor)
            current_path = child_path
            descriptor = child_descriptor
        candidate = current_path / parts[-1]
        try:
            metadata = _parent_entry_lstat(current_path, descriptor, parts[-1])
        except FileNotFoundError:
            return None
        _reject_link_or_reparse(candidate, metadata, label=label)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file: {candidate}")
        _revalidate_directory_components(root.components, label=label)
        return candidate, metadata
    finally:
        os.close(descriptor)


def _read_anchored_relative_bytes(
    root: _DirectoryAnchor,
    relative: str,
    *,
    label: str,
    maximum_bytes: int | None = None,
) -> bytes:
    """Read one source/stage member through the pinned root descriptor.

    The POSIX path never reopens a mutable pathname below ``root`` after its
    root directory descriptor is acquired.  This binds the containment decision
    and byte snapshot to one directory hierarchy rather than merely checking
    containment before a later path-based read.
    """

    maximum_bytes = _require_byte_limit(maximum_bytes, label=label)
    canonical, parts = _relative_components(relative, label=label)
    root_descriptor = _open_bound_directory(root.path, root.components, label=label)
    if root_descriptor is None:
        checked = _anchored_relative_metadata_fallback(root, canonical, label=label)
        if checked is None:
            raise FileNotFoundError(
                f"{label} does not exist below {root.path}: {canonical}"
            )
        path, expected = checked
        if maximum_bytes is not None and expected.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds the byte limit: {path}")
        data = read_stable_bytes(
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
            raise FileNotFoundError(f"{label} does not exist: {candidate}") from exc
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


def _module_source(
    root: _DirectoryAnchor,
    module_name: str,
) -> tuple[Path, bool] | None:
    module = _validate_module_name(module_name, label="module")
    module_relative = _module_relative_path(module, is_package=False)
    package_relative = _module_relative_path(module, is_package=True)
    module_source = _anchored_relative_metadata(
        root,
        module_relative,
        label=f"local module {module}",
    )
    package_source = _anchored_relative_metadata(
        root,
        package_relative,
        label=f"local module {module}",
    )
    if module_source is not None and package_source is not None:
        raise ValueError(f"ambiguous local module source: {module_name}")
    if module_source is not None:
        return module_source[0], False
    if package_source is not None:
        return package_source[0], True
    return None


def _relative_source_path(path: Path, root: Path | _DirectoryAnchor) -> str:
    """Return lexical containment only after component validation has occurred."""

    root_path = root.path if isinstance(root, _DirectoryAnchor) else _absolute_path(root)
    candidate = _absolute_path(path)
    try:
        relative = candidate.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(f"local module is outside source root: {path}") from exc
    return _validate_payload_path(
        relative,
        label="local module path",
        role=SOURCE_ROLE,
    )


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    assignments: dict[str, ast.expr | None] = {}
    nodes = tuple(ast.walk(tree))
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    top_level = alias.name.split(".", 1)[0]
                    aliases[top_level] = top_level
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        else:
            targets: tuple[ast.expr, ...]
            value: ast.expr | None
            if isinstance(node, ast.Assign):
                targets = tuple(node.targets)
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = (node.target,)
                value = node.value
            elif isinstance(node, ast.NamedExpr):
                targets = (node.target,)
                value = node.value
            else:
                continue
            if value is None:
                continue
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id in assignments:
                    assignments[target.id] = None
                else:
                    assignments[target.id] = value

    resolving: set[str] = set()

    def resolve_expression(expression: ast.expr) -> str | None:
        if isinstance(expression, ast.Name):
            return resolve_name(expression.id)
        if isinstance(expression, ast.Attribute):
            base = resolve_expression(expression.value)
            if base:
                return f"{base}.{expression.attr}"
        return None

    def resolve_name(name: str) -> str | None:
        known = aliases.get(name)
        if known is not None:
            return known
        value = assignments.get(name)
        if value is None:
            return name
        if name in resolving:
            return None
        if len(resolving) >= 64:
            raise ValueError("local module alias graph exceeds the static analysis limit")
        resolving.add(name)
        try:
            qualified = resolve_expression(value)
        finally:
            resolving.remove(name)
        if qualified is not None:
            aliases[name] = qualified
        return qualified

    for name in assignments:
        resolve_name(name)
    return aliases


def _qualified_expression_name(
    expression: ast.expr,
    aliases: Mapping[str, str],
) -> str | None:
    if isinstance(expression, ast.Name):
        return aliases.get(expression.id, expression.id)
    if isinstance(expression, ast.Attribute):
        base = _qualified_expression_name(expression.value, aliases)
        if base:
            return f"{base}.{expression.attr}"
    return None


def _call_argument(
    call: ast.Call,
    position: int,
    keyword: str,
) -> ast.expr | None:
    if len(call.args) > position:
        return call.args[position]
    for item in call.keywords:
        if item.arg == keyword:
            return item.value
    return None


def _literal_string_argument(
    call: ast.Call,
    position: int,
    keyword: str,
) -> str | None:
    value = _call_argument(call, position, keyword)
    if isinstance(value, ast.Constant) and type(value.value) is str:
        return value.value
    return None


def _resolved_dynamic_import(
    call: ast.Call,
    *,
    call_name: str,
    current_package: str,
    module_name: str,
) -> str:
    if any(isinstance(argument, ast.Starred) for argument in call.args) or any(
        keyword.arg is None for keyword in call.keywords
    ):
        raise ValueError(
            f"local module has an unresolved dynamic import: {module_name}"
        )
    target = _literal_string_argument(call, 0, "name")
    if target is None:
        raise ValueError(
            f"local module has an unresolved dynamic import: {module_name}"
        )
    if call_name in {"__import__", "builtins.__import__"}:
        fromlist_value = _call_argument(call, 3, "fromlist")
        if fromlist_value is not None and not (
            isinstance(fromlist_value, ast.Constant)
            and fromlist_value.value is None
            or isinstance(fromlist_value, (ast.List, ast.Tuple))
            and not fromlist_value.elts
        ):
            raise ValueError(
                f"local module has an unresolved __import__ fromlist: {module_name}"
            )
        level_value = _call_argument(call, 4, "level")
        if level_value is not None and not (
            isinstance(level_value, ast.Constant)
            and type(level_value.value) is int
            and level_value.value == 0
        ):
            raise ValueError(
                f"local module has an unresolved relative __import__: {module_name}"
            )
        if target.startswith("."):
            raise ValueError(
                f"local module has an unresolved relative __import__: {module_name}"
            )
        return _validate_module_name(target, label="dynamic import target")
    if not target.startswith("."):
        return _validate_module_name(target, label="dynamic import target")
    package_value = _call_argument(call, 1, "package")
    if isinstance(package_value, ast.Name) and package_value.id == "__package__":
        package = current_package
    elif isinstance(package_value, ast.Constant) and type(package_value.value) is str:
        package = package_value.value
    else:
        raise ValueError(
            f"local module has an unresolved relative dynamic import: {module_name}"
        )
    if not package:
        raise ValueError(
            f"local module has an invalid relative dynamic import: {module_name}"
        )
    _validate_module_name(package, label="dynamic import package")
    try:
        resolved = importlib.util.resolve_name(target, package)
    except ImportError as exc:
        raise ValueError(
            f"local module has an invalid relative dynamic import: {module_name}"
        ) from exc
    return _validate_module_name(resolved, label="dynamic import target")


def _reject_or_resolve_dynamic_loading(
    tree: ast.AST,
    *,
    current_package: str,
    module_name: str,
) -> list[str]:
    aliases = _import_aliases(tree)
    nodes = tuple(ast.walk(tree))
    parents = {
        id(child): parent
        for parent in nodes
        for child in ast.iter_child_nodes(parent)
    }
    dangerous_names = _DYNAMIC_LOADING_LEAVES | {
        "__import__",
        "compile",
        "eval",
        "exec",
        "import_module",
    }

    def is_loading_reference(expression: ast.expr) -> bool:
        qualified = _qualified_expression_name(expression, aliases)
        return qualified in (_DYNAMIC_IMPORT_CALLS | _DYNAMIC_LOADING_CALLS) or (
            qualified is not None
            and qualified.rsplit(".", 1)[-1] in dangerous_names
        )

    # Known loader callables may be called directly below, but they may not be
    # returned, registered, wrapped, or stored for a later unresolved call.
    for node in nodes:
        if not isinstance(node, (ast.Name, ast.Attribute)):
            continue
        if not is_loading_reference(node):
            continue
        parent = parents.get(id(node))
        if isinstance(parent, ast.Call) and parent.func is node:
            continue
        raise ValueError(
            f"local module exposes an unresolved dynamic loader: {module_name}"
        )

    resolved_imports: list[str] = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        call_name = _qualified_expression_name(node.func, aliases)
        call_leaf = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else call_name.rsplit(".", 1)[-1]
            if call_name is not None
            else None
        )
        if call_name in _DYNAMIC_IMPORT_CALLS:
            resolved_imports.append(
                _resolved_dynamic_import(
                    node,
                    call_name=call_name,
                    current_package=current_package,
                    module_name=module_name,
                )
            )
            continue
        if call_name in _DYNAMIC_LOADING_CALLS or call_leaf in _DYNAMIC_LOADING_LEAVES:
            raise ValueError(
                f"local module uses unsupported dynamic code loading: {module_name}"
            )
        if call_name in {"getattr", "builtins.getattr"}:
            attribute = _literal_string_argument(node, 1, "name")
            target = _call_argument(node, 0, "object")
            target_name = (
                _qualified_expression_name(target, aliases)
                if isinstance(target, ast.expr)
                else None
            )
            if attribute in dangerous_names or (
                attribute is None
                and target_name is not None
                and target_name.split(".", 1)[0]
                in {"__builtins__", "builtins", "imp", "importlib", "runpy", "zipimport"}
            ):
                raise ValueError(
                    f"local module uses unresolved reflected code loading: {module_name}"
                )
        if call_leaf == "get" and isinstance(node.func, ast.Attribute):
            container_name = _qualified_expression_name(node.func.value, aliases)
            reflective_container = (
                _qualified_expression_name(node.func.value.func, aliases)
                if isinstance(node.func.value, ast.Call)
                else None
            )
            attribute = _literal_string_argument(node, 0, "key")
            if attribute in dangerous_names and (
                (container_name is not None and container_name.endswith(".__dict__"))
                or reflective_container
                in {"globals", "locals", "vars", "builtins.vars"}
            ):
                raise ValueError(
                    f"local module uses unresolved reflected code loading: {module_name}"
                )
    for node in nodes:
        if not isinstance(node, ast.Subscript):
            continue
        key = node.slice
        attribute = (
            key.value
            if isinstance(key, ast.Constant) and type(key.value) is str
            else None
        )
        if attribute not in dangerous_names:
            continue
        container_name = _qualified_expression_name(node.value, aliases)
        reflective_call = (
            _qualified_expression_name(node.value.func, aliases)
            if isinstance(node.value, ast.Call)
            else None
        )
        if (
            container_name in {"__builtins__", "builtins.__dict__"}
            or (container_name is not None and container_name.endswith(".__dict__"))
            or reflective_call in {"globals", "locals", "vars", "builtins.vars"}
        ):
            raise ValueError(
                f"local module uses unresolved reflected code loading: {module_name}"
            )
    return resolved_imports


def _import_targets(
    snapshot: _SourceSnapshot,
    root: _DirectoryAnchor,
) -> list[tuple[str, bool]]:
    try:
        tree = ast.parse(snapshot.data, filename=str(snapshot.module.source_path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"local module is not parseable Python: {snapshot.module.name}"
        ) from exc
    current_package = (
        snapshot.module.name
        if snapshot.module.is_package
        else snapshot.module.name.rpartition(".")[0]
    )
    candidates: dict[str, bool] = {}

    def add(name: str, *, required: bool) -> None:
        if not name:
            return
        _validate_module_name(name, label="import target")
        candidates[name] = candidates.get(name, False) or required

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                add(
                    name,
                    required=_belongs_to_local_top_level(root, name),
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if not current_package:
                    raise ValueError(
                        "top-level local module cannot use a relative import: "
                        f"{snapshot.module.name}"
                    )
                relative_name = "." * node.level + (node.module or "")
                try:
                    base = importlib.util.resolve_name(relative_name, current_package)
                except ImportError as exc:
                    raise ValueError(
                        "local module has an invalid relative import: "
                        f"{snapshot.module.name}"
                    ) from exc
                add(base, required=True)
            else:
                base = node.module or ""
                if base:
                    add(
                        base,
                        required=_belongs_to_local_top_level(root, base),
                    )
            if not base:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                candidate = f"{base}.{alias.name}"
                # Imported names can be attributes defined by a package's
                # __init__.py.  Include a matching local submodule when one
                # exists, but do not reject an ordinary attribute.
                add(candidate, required=False)
    for target in _reject_or_resolve_dynamic_loading(
        tree,
        current_package=current_package,
        module_name=snapshot.module.name,
    ):
        add(target, required=_belongs_to_local_top_level(root, target))
    return sorted(candidates.items())


def _belongs_to_local_top_level(root: _DirectoryAnchor, module_name: str) -> bool:
    top_level = module_name.split(".", 1)[0]
    return _module_source(root, top_level) is not None


def _snapshot_runtime_assets(
    runtime_assets: (
        Mapping[str, Path | str]
        | Iterable[RuntimeAsset | tuple[str, Path | str]]
        | None
    ),
    *,
    maximum_total_bytes: int | None = None,
    maximum_entries: int | None = None,
) -> tuple[_AssetSnapshot, ...]:
    if maximum_total_bytes is None:
        maximum_total_bytes = MAX_TOTAL_BYTES
    maximum_total_bytes = _require_byte_limit(
        maximum_total_bytes,
        label="runtime assets",
    )
    if maximum_entries is None:
        maximum_entries = MAX_FILE_ENTRIES
    if type(maximum_entries) is not int or maximum_entries < 0:
        raise ValueError("runtime asset entry limit must be a non-negative integer")
    assets = _normalise_runtime_assets(
        runtime_assets,
        maximum_entries=maximum_entries,
    )
    seen: set[str] = set()
    snapshots: list[_AssetSnapshot] = []
    total_bytes = 0
    for asset in assets:
        destination = canonical_relative_path(
            asset.destination,
            label="runtime asset destination",
        )
        _validate_asset_destination(destination)
        if destination in seen:
            raise ValueError(f"duplicate runtime asset destination: {destination}")
        seen.add(destination)
        source = _absolute_path(asset.source)
        remaining_bytes = maximum_total_bytes - total_bytes
        data = read_stable_bytes(
            source,
            label=f"runtime asset {destination}",
            maximum_bytes=min(MAX_MEMBER_BYTES, remaining_bytes),
        )
        total_bytes += len(data)
        snapshots.append(
            _AssetSnapshot(
                destination=destination,
                source_path=source,
                data=data,
            )
        )
    return tuple(sorted(snapshots, key=lambda item: item.destination))


def _normalise_runtime_assets(
    runtime_assets: (
        Mapping[str, Path | str]
        | Iterable[RuntimeAsset | tuple[str, Path | str]]
        | None
    ),
    *,
    maximum_entries: int,
) -> list[RuntimeAsset]:
    if runtime_assets is None:
        return []
    items: Iterable[RuntimeAsset | tuple[object, object]]
    if isinstance(runtime_assets, Mapping):
        items = runtime_assets.items()
    else:
        items = runtime_assets
    result: list[RuntimeAsset] = []
    for item in items:
        if len(result) >= maximum_entries:
            raise ValueError("runtime assets exceed the file-entry limit")
        if isinstance(item, RuntimeAsset):
            result.append(item)
            continue
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("runtime assets must be RuntimeAsset values or pairs")
        destination, source = item
        result.append(RuntimeAsset(destination=str(destination), source=Path(source)))
    return result


def _validate_asset_destination(destination: str) -> None:
    _validate_payload_path(
        destination,
        label="runtime asset destination",
        role=RUNTIME_ASSET_ROLE,
    )


def _validate_payload_path(
    value: str,
    *,
    label: str,
    role: str | None = None,
) -> str:
    """Reject payload names that Python startup or bytecode rules can consume."""

    path = canonical_relative_path(value, label=label)
    parts = path.split("/")
    folded_parts = [part.casefold() for part in parts]
    filename = folded_parts[-1]
    if "__pycache__" in folded_parts or filename.endswith(_BYTECODE_SUFFIXES):
        raise ValueError(f"{label} contains a forbidden Python bytecode path")
    if filename.endswith(".pth") or filename in _AUTOMATIC_IMPORT_FILENAMES:
        raise ValueError(f"{label} contains a forbidden Python startup path")
    if role == SOURCE_ROLE:
        if not filename.endswith(".py"):
            raise ValueError(f"{label} is not a Python source path")
    elif role == RUNTIME_ASSET_ROLE:
        if filename.endswith((".py", *_NATIVE_CODE_SUFFIXES)):
            raise ValueError("runtime assets cannot introduce Python code or metadata")
    return path


def _validate_payload_layout(paths: Iterable[str], *, label: str) -> None:
    """Reject layouts that collide on case-insensitive filesystems."""

    folded_paths: dict[str, str] = {}
    for path in paths:
        folded = path.casefold()
        if folded in folded_paths:
            raise ValueError(f"{label} contains case-colliding file paths")
        folded_paths[folded] = path
    for path in folded_paths:
        parent = path
        while "/" in parent:
            parent = parent.rsplit("/", 1)[0]
            if parent in folded_paths:
                raise ValueError(f"{label} contains a file/directory path conflict")


def _validate_bundle_member_name(name: str, *, label: str) -> str:
    """Require the one manifest member or a safe payload member path."""

    member = canonical_relative_path(name, label=label)
    if member == BUNDLE_MANIFEST_NAME:
        return member
    prefix = f"{BUNDLE_PAYLOAD_ROOT}/"
    if not member.startswith(prefix):
        raise ValueError(f"{label} is outside the execution bundle payload")
    _validate_payload_path(
        member[len(prefix) :],
        label=label,
    )
    return member


def _validate_stage_tree_member_name(name: str, *, label: str) -> str:
    """Also permit the implicit payload directory during safe stage traversal."""

    member = canonical_relative_path(name, label=label)
    if member == BUNDLE_PAYLOAD_ROOT:
        return member
    return _validate_bundle_member_name(member, label=label)


def _build_manifest_and_payload(
    *,
    entry_module: str,
    runner_environment_manifest_sha256: str,
    source_snapshots: tuple[_SourceSnapshot, ...],
    asset_snapshots: tuple[_AssetSnapshot, ...],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    _validate_bundled_module_name(entry_module, label="entry module")
    runner_digest = _require_sha256(
        runner_environment_manifest_sha256,
        "runner environment manifest SHA-256",
    )
    payload_bytes: dict[str, bytes] = {}
    files: list[dict[str, Any]] = []
    module_registry: list[dict[str, Any]] = []
    for snapshot in source_snapshots:
        path = snapshot.module.relative_path
        if path in payload_bytes:
            raise ValueError(f"duplicate execution bundle payload path: {path}")
        payload_bytes[path] = snapshot.data
        digest = _sha256_bytes(snapshot.data)
        files.append(
            {
                "path": path,
                "role": SOURCE_ROLE,
                "sha256": digest,
                "size": len(snapshot.data),
            }
        )
        module_registry.append(
            {
                "is_package": snapshot.module.is_package,
                "module": snapshot.module.name,
                "path": path,
                "sha256": digest,
            }
        )
    for snapshot in asset_snapshots:
        if snapshot.destination in payload_bytes:
            raise ValueError(
                "runtime asset collides with local source path: "
                f"{snapshot.destination}"
            )
        payload_bytes[snapshot.destination] = snapshot.data
        files.append(
            {
                "path": snapshot.destination,
                "role": RUNTIME_ASSET_ROLE,
                "sha256": _sha256_bytes(snapshot.data),
                "size": len(snapshot.data),
            }
        )
    _validate_payload_layout(payload_bytes, label="execution bundle payload")
    files.sort(key=lambda item: str(item["path"]))
    module_registry.sort(key=lambda item: str(item["module"]))
    if entry_module not in {entry["module"] for entry in module_registry}:
        raise ValueError("entry module is absent from the local import closure")
    runtime_asset_paths = [
        entry["path"] for entry in files if entry["role"] == RUNTIME_ASSET_ROLE
    ]
    manifest: dict[str, Any] = {
        "archive_format": BUNDLE_ARCHIVE_FORMAT,
        "bundle_content_sha256": _canonical_content_sha256(files),
        "entry_module": entry_module,
        "files": files,
        "kind": BUNDLE_KIND,
        "module_registry": module_registry,
        "payload_root": BUNDLE_PAYLOAD_ROOT,
        RUNNER_ENVIRONMENT_MANIFEST_SHA256_FIELD: runner_digest,
        "runtime_assets": runtime_asset_paths,
        "schema_version": BUNDLE_SCHEMA_VERSION,
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    return manifest, payload_bytes


def _canonical_content_sha256(entries: Iterable[Mapping[str, Any]]) -> str:
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


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(payload))


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


def _deterministic_archive_bytes(members: Mapping[str, bytes]) -> bytes:
    if len(members) > MAX_FILE_ENTRIES + 1:
        raise ValueError("execution bundle archive exceeds the file-entry limit")
    if BUNDLE_MANIFEST_NAME not in members:
        raise ValueError("execution bundle archive lacks its manifest")
    ordered = sorted(members)
    payload_entries = 0
    payload_bytes = 0
    archive_bytes = 22
    for name in ordered:
        _validate_bundle_member_name(name, label="archive member name")
        data = members[name]
        if type(data) is not bytes:
            raise ValueError("execution bundle archive members must be bytes")
        if name == BUNDLE_MANIFEST_NAME:
            if len(data) > MAX_MANIFEST_BYTES:
                raise ValueError("execution bundle archive manifest exceeds the byte limit")
        else:
            payload_entries += 1
            payload_bytes += len(data)
            if len(data) > MAX_MEMBER_BYTES:
                raise ValueError("execution bundle archive member exceeds the byte limit")
            if payload_entries > MAX_FILE_ENTRIES:
                raise ValueError("execution bundle archive exceeds the file-entry limit")
            if payload_bytes > MAX_TOTAL_BYTES:
                raise ValueError("execution bundle archive exceeds the total size limit")
        name_bytes = name.encode("ascii")
        archive_bytes += 30 + 46 + (2 * len(name_bytes)) + len(data)
        if archive_bytes > MAX_ARCHIVE_BYTES:
            raise ValueError("execution bundle archive exceeds the byte limit")
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
    ) as archive:
        archive.comment = b""
        for name in ordered:
            info = zipfile.ZipInfo(name, date_time=ZIP_EPOCH)
            info.create_system = 3
            info.create_version = 20
            info.extract_version = 20
            info.flag_bits = 0
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFREG | 0o444) << 16
            archive.writestr(info, members[name])
    result = stream.getvalue()
    if len(result) != archive_bytes or len(result) > MAX_ARCHIVE_BYTES:
        raise ValueError("execution bundle archive size is not canonical")
    return result


def _capture_output_parent(
    parent: Path,
) -> tuple[tuple[Path, os.stat_result], ...]:
    return _ensure_directory_components(parent, label="execution bundle archive parent")


def _unlink_parent_entry(parent: Path, descriptor: int | None, name: str) -> None:
    if descriptor is None:
        (parent / name).unlink()
    else:
        os.unlink(name, dir_fd=descriptor)


def _require_missing_output(parent: Path, descriptor: int | None, name: str) -> None:
    try:
        metadata = _parent_entry_lstat(parent, descriptor, name)
    except FileNotFoundError:
        return
    _reject_link_or_reparse(parent / name, metadata, label="execution bundle archive")
    raise FileExistsError(f"execution bundle archive already exists: {parent / name}")


def _open_staging_file(
    parent: Path,
    descriptor: int | None,
) -> tuple[str, int]:
    """Create a private temporary archive without trusting its name after open."""

    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    else:
        flags |= getattr(os, "O_TEMPORARY", 0)
    for _ in range(128):
        name = f".nl2sql-bundle-{secrets.token_hex(16)}.tmp"
        try:
            if descriptor is None:
                return name, os.open(parent / name, flags, 0o600)
            return name, os.open(name, flags, 0o600, dir_fd=descriptor)
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a staging archive in {parent}")


def _write_descriptor(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("could not write execution bundle staging archive")
        remaining = remaining[written:]
    os.fsync(descriptor)


def _remove_owned_parent_entry(
    parent: Path,
    descriptor: int | None,
    name: str,
    expected: os.stat_result,
) -> None:
    """Best-effort cleanup that never follows or unlinks a replacement file."""

    if descriptor is None and os.name == "nt":
        # O_TEMPORARY owns deletion after descriptor close.  Avoid a later
        # pathname unlink if the parent namespace is concurrently replaced.
        return
    try:
        actual = _parent_entry_lstat(parent, descriptor, name)
    except OSError:
        return
    if not _same_file_object_identity(expected, actual):
        return
    try:
        _unlink_parent_entry(parent, descriptor, name)
    except OSError:
        return


def _publish_staging_archive(
    *,
    parent: Path,
    parent_descriptor: int | None,
    parent_snapshots: tuple[tuple[Path, os.stat_result], ...],
    output_name: str,
    staging_name: str,
    staging_descriptor: int,
    staging_metadata: os.stat_result,
    payload_sha256: str,
) -> None:
    _revalidate_directory_components(
        parent_snapshots,
        label="execution bundle archive parent",
    )
    named_staging = _parent_entry_lstat(parent, parent_descriptor, staging_name)
    if not _same_file_identity(staging_metadata, named_staging):
        raise ValueError("execution bundle staging archive changed before publication")
    if not _same_file_identity(staging_metadata, os.fstat(staging_descriptor)):
        raise ValueError("execution bundle staging descriptor changed before publication")
    if parent_descriptor is None:
        os.link(parent / staging_name, parent / output_name)
    else:
        os.link(
            staging_name,
            output_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    linked_metadata = os.fstat(staging_descriptor)
    if not _same_file_object_identity(staging_metadata, linked_metadata):
        raise ValueError("execution bundle staging descriptor changed during publication")
    try:
        published = _parent_entry_lstat(parent, parent_descriptor, output_name)
    except FileNotFoundError as exc:
        raise ValueError("execution bundle archive changed during publication") from exc
    if not _same_file_object_identity(linked_metadata, published):
        raise ValueError("execution bundle archive does not match staged descriptor")
    _revalidate_directory_components(
        parent_snapshots,
        label="execution bundle archive parent",
    )
    if (
        _sha256_bytes(
            _read_checked_descriptor(
                staging_descriptor,
                linked_metadata,
                label="execution bundle staging descriptor",
                path=parent / staging_name,
                maximum_bytes=MAX_ARCHIVE_BYTES,
            )
        )
        != payload_sha256
    ):
        raise ValueError("execution bundle staging descriptor content changed during publication")
    final_published = _parent_entry_lstat(parent, parent_descriptor, output_name)
    if not _same_file_object_identity(linked_metadata, final_published):
        raise ValueError("execution bundle archive changed after publication")
    _revalidate_directory_components(
        parent_snapshots,
        label="execution bundle archive parent",
    )


def _publish_bytes_create_once(output: Path, data: bytes) -> None:
    output = _absolute_path(output)
    if not output.name:
        raise ValueError("execution bundle archive must name a file")
    parent_snapshots = _capture_output_parent(output.parent)
    parent_descriptor = _open_bound_directory(
        output.parent,
        parent_snapshots,
        label="execution bundle archive parent",
    )
    staging_name: str | None = None
    staging_descriptor: int | None = None
    staging_metadata: os.stat_result | None = None
    payload_sha256 = _sha256_bytes(data)
    try:
        _revalidate_directory_components(
            parent_snapshots,
            label="execution bundle archive parent",
        )
        _require_missing_output(output.parent, parent_descriptor, output.name)
        staging_name, staging_descriptor = _open_staging_file(
            output.parent,
            parent_descriptor,
        )
        _write_descriptor(staging_descriptor, data)
        staging_metadata = os.fstat(staging_descriptor)
        if (
            not _has_stable_file_identity(staging_metadata)
            or staging_metadata.st_size != len(data)
        ):
            raise ValueError("execution bundle staging archive is invalid")
        if (
            _sha256_bytes(
                _read_checked_descriptor(
                    staging_descriptor,
                    staging_metadata,
                    label="execution bundle staging descriptor",
                    path=output.parent / staging_name,
                    maximum_bytes=len(data),
                )
            )
            != payload_sha256
        ):
            raise ValueError("execution bundle staging archive content differs from payload")
        _publish_staging_archive(
            parent=output.parent,
            parent_descriptor=parent_descriptor,
            parent_snapshots=parent_snapshots,
            output_name=output.name,
            staging_name=staging_name,
            staging_descriptor=staging_descriptor,
            staging_metadata=staging_metadata,
            payload_sha256=payload_sha256,
        )
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if staging_name is not None and staging_metadata is not None:
            _remove_owned_parent_entry(
                output.parent,
                parent_descriptor,
                staging_name,
                staging_metadata,
            )
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _validate_archive_directory_record(raw_archive: bytes) -> int:
    """Bound ZIP entry allocation before ``zipfile`` parses the directory."""

    if len(raw_archive) < 22:
        raise ValueError("execution bundle archive lacks a canonical ZIP directory")
    offset = len(raw_archive) - 22
    (
        signature,
        disk_number,
        directory_disk,
        entries_on_disk,
        total_entries,
        directory_size,
        directory_offset,
        comment_size,
    ) = struct.unpack_from("<4s4H2LH", raw_archive, offset)
    if signature != b"PK\x05\x06" or comment_size != 0:
        raise ValueError("execution bundle archive lacks a canonical ZIP directory")
    if (
        disk_number != 0
        or directory_disk != 0
        or entries_on_disk != total_entries
        or total_entries in {0, 0xFFFF}
    ):
        raise ValueError("execution bundle archive uses an unsupported ZIP directory")
    if total_entries > MAX_FILE_ENTRIES + 1:
        raise ValueError("execution bundle archive exceeds the file-entry limit")
    if directory_offset + directory_size != offset:
        raise ValueError("execution bundle archive ZIP directory is not canonical")
    return total_entries


def _safe_archive_members(
    archive: zipfile.ZipFile,
    *,
    expected_entries: int,
) -> dict[str, zipfile.ZipInfo]:
    if archive.comment:
        raise ValueError("execution bundle archive must not contain a ZIP comment")
    infos = archive.infolist()
    if len(infos) != expected_entries:
        raise ValueError("execution bundle archive entry count differs from its directory")
    result: dict[str, zipfile.ZipInfo] = {}
    payload_size = 0
    payload_entries = 0
    names: list[str] = []
    for info in infos:
        name = _validate_archive_member(info)
        if name in result:
            raise ValueError(f"duplicate execution bundle archive member: {name}")
        result[name] = info
        names.append(name)
        if name == BUNDLE_MANIFEST_NAME:
            if info.file_size > MAX_MANIFEST_BYTES:
                raise ValueError("execution bundle archive manifest exceeds the byte limit")
        else:
            payload_entries += 1
            payload_size += info.file_size
            if payload_entries > MAX_FILE_ENTRIES:
                raise ValueError("execution bundle archive exceeds the file-entry limit")
            if payload_size > MAX_TOTAL_BYTES:
                raise ValueError("execution bundle archive exceeds the total size limit")
    if not result:
        raise ValueError("execution bundle archive is empty")
    if names != sorted(names):
        raise ValueError("execution bundle archive members are not sorted")
    if min(info.header_offset for info in infos) != 0:
        raise ValueError("execution bundle archive contains a prepended payload")
    _validate_payload_layout(
        (
            name[len(f"{BUNDLE_PAYLOAD_ROOT}/"):]
            for name in names
            if name != BUNDLE_MANIFEST_NAME
        ),
        label="execution bundle archive",
    )
    return result


def _validate_archive_member(info: zipfile.ZipInfo) -> str:
    name = _validate_bundle_member_name(
        info.filename,
        label="archive member name",
    )
    if info.is_dir():
        raise ValueError(f"execution bundle archive contains a directory member: {name}")
    if info.flag_bits != 0:
        raise ValueError(f"execution bundle archive member has unsupported flags: {name}")
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"execution bundle archive member is not stored: {name}")
    if info.date_time != ZIP_EPOCH or info.extra or info.comment:
        raise ValueError(f"execution bundle archive member metadata is not canonical: {name}")
    if info.create_system != 3:
        raise ValueError(f"execution bundle archive member lacks POSIX type metadata: {name}")
    member_type = stat.S_IFMT(info.external_attr >> 16)
    if member_type == stat.S_IFLNK:
        raise ValueError(f"execution bundle archive member is a symlink: {name}")
    if member_type != stat.S_IFREG:
        raise ValueError(f"execution bundle archive member is not a regular file: {name}")
    maximum_bytes = (
        MAX_MANIFEST_BYTES if name == BUNDLE_MANIFEST_NAME else MAX_MEMBER_BYTES
    )
    if info.file_size > maximum_bytes:
        if name == BUNDLE_MANIFEST_NAME:
            raise ValueError("execution bundle archive manifest exceeds the byte limit")
        raise ValueError(f"execution bundle archive member size is unsafe: {name}")
    if info.file_size < 0 or info.compress_size != info.file_size:
        raise ValueError(f"execution bundle archive member size is unsafe: {name}")
    return name


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
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if type(payload) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _validate_manifest(
    manifest: dict[str, Any],
    raw_manifest: bytes,
) -> list[dict[str, Any]]:
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
        or manifest.get("payload_root") != BUNDLE_PAYLOAD_ROOT
    ):
        raise ValueError("unsupported execution bundle manifest")
    if raw_manifest != _canonical_json_bytes(manifest):
        raise ValueError("execution bundle manifest is not canonical JSON")
    declared_manifest_sha256 = manifest.get("manifest_sha256")
    _require_sha256(declared_manifest_sha256, "execution bundle manifest SHA-256")
    if declared_manifest_sha256 != _manifest_sha256(manifest):
        raise ValueError("execution bundle manifest SHA-256 is invalid")
    _validate_bundled_module_name(
        manifest.get("entry_module"),
        label="manifest entry module",
    )
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
    if sum(int(entry["size"]) for entry in parsed_files) > MAX_TOTAL_BYTES:
        raise ValueError("execution bundle manifest exceeds the total size limit")
    paths = [str(entry["path"]) for entry in parsed_files]
    if len(paths) != len(set(paths)) or paths != sorted(paths):
        raise ValueError("execution bundle manifest files are not uniquely sorted")
    _validate_payload_layout(paths, label="execution bundle manifest")
    if manifest.get("bundle_content_sha256") != _canonical_content_sha256(parsed_files):
        raise ValueError("execution bundle manifest content digest is invalid")
    _require_sha256(
        manifest.get("bundle_content_sha256"),
        "execution bundle content SHA-256",
    )
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


def _validate_file_entry(entry: Any) -> dict[str, Any]:
    if type(entry) is not dict or set(entry) != {"path", "role", "sha256", "size"}:
        raise ValueError("execution bundle manifest file entry is invalid")
    role = entry.get("role")
    if role not in {SOURCE_ROLE, RUNTIME_ASSET_ROLE}:
        raise ValueError("execution bundle manifest file role is invalid")
    path = _validate_payload_path(
        entry.get("path"),
        label="manifest file path",
        role=role,
    )
    if (
        type(entry.get("size")) is not int
        or entry["size"] < 0
        or entry["size"] > MAX_MEMBER_BYTES
    ):
        raise ValueError("execution bundle manifest file size is invalid")
    _require_sha256(entry.get("sha256"), "execution bundle file SHA-256")
    if role == RUNTIME_ASSET_ROLE:
        _validate_asset_destination(path)
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
        module = entry.get("module")
        _validate_bundled_module_name(module, label="module registry module")
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


def _create_fresh_directory(root: Path) -> _DirectoryAnchor:
    """Create one stage root through a checked parent namespace exactly once."""

    root = _absolute_path(root)
    if not root.name:
        raise ValueError("execution bundle stage must name a directory")
    parent_snapshots = _ensure_directory_components(
        root.parent,
        label="execution bundle stage parent",
    )
    parent_descriptor = _open_bound_directory(
        root.parent,
        parent_snapshots,
        label="execution bundle stage parent",
    )
    try:
        _revalidate_directory_components(
            parent_snapshots,
            label="execution bundle stage parent",
        )
        try:
            existing = _parent_entry_lstat(root.parent, parent_descriptor, root.name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            _reject_link_or_reparse(
                root,
                existing,
                label="execution bundle stage",
            )
            raise FileExistsError(f"execution bundle stage already exists: {root}")
        try:
            if parent_descriptor is None:
                os.mkdir(root, mode=0o700)
            else:
                os.mkdir(root.name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise FileExistsError(f"execution bundle stage already exists: {root}") from exc
        metadata = _parent_entry_lstat(root.parent, parent_descriptor, root.name)
        _reject_link_or_reparse(root, metadata, label="execution bundle stage")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"execution bundle stage is not a directory: {root}")
        if parent_descriptor is not None:
            stage_descriptor: int | None = None
            try:
                _, stage_descriptor = _open_checked_child_directory(
                    parent_path=root.parent,
                    parent_descriptor=parent_descriptor,
                    name=root.name,
                    label="execution bundle stage",
                )
            finally:
                if stage_descriptor is not None:
                    os.close(stage_descriptor)
        _revalidate_directory_components(
            parent_snapshots,
            label="execution bundle stage parent",
        )
    finally:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    return _DirectoryAnchor(
        path=root,
        components=(*parent_snapshots, (root, metadata)),
    )


def _open_or_create_stage_directory(
    *,
    parent_path: Path,
    parent_descriptor: int | None,
    name: str,
) -> tuple[Path, int | None]:
    """Create or bind one link-free directory below the stage root."""

    candidate = parent_path / name
    try:
        metadata = _parent_entry_lstat(parent_path, parent_descriptor, name)
    except FileNotFoundError:
        try:
            if parent_descriptor is None:
                os.mkdir(candidate, mode=0o700)
            else:
                os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            pass
        metadata = _parent_entry_lstat(parent_path, parent_descriptor, name)
    _reject_link_or_reparse(candidate, metadata, label="execution bundle stage")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"execution bundle stage has a non-directory component: {candidate}")
    if parent_descriptor is None:
        return candidate, None
    return _open_checked_child_directory(
        parent_path=parent_path,
        parent_descriptor=parent_descriptor,
        name=name,
        label="execution bundle stage",
    )


def _write_stage_file(root: _DirectoryAnchor, relative: str, data: bytes) -> None:
    """Create a staged member through a no-follow, descriptor-bound walk."""

    canonical, parts = _relative_components(relative, label="staged relative path")
    _validate_bundle_member_name(canonical, label="staged relative path")
    root_descriptor = _open_bound_directory(
        root.path,
        root.components,
        label="execution bundle stage root",
    )
    current_path = root.path
    descriptor = root_descriptor
    file_descriptor: int | None = None
    try:
        _revalidate_directory_components(root.components, label="execution bundle stage root")
        for component in parts[:-1]:
            child_path, child_descriptor = _open_or_create_stage_directory(
                parent_path=current_path,
                parent_descriptor=descriptor,
                name=component,
            )
            if descriptor is not None:
                os.close(descriptor)
            current_path = child_path
            descriptor = child_descriptor
        candidate = current_path / parts[-1]
        try:
            existing = _parent_entry_lstat(current_path, descriptor, parts[-1])
        except FileNotFoundError:
            existing = None
        if existing is not None:
            _reject_link_or_reparse(candidate, existing, label="execution bundle stage")
            raise FileExistsError(f"execution bundle stage file already exists: {candidate}")
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        if os.name != "nt":
            flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            if descriptor is None:
                file_descriptor = os.open(candidate, flags, 0o600)
            else:
                file_descriptor = os.open(parts[-1], flags, 0o600, dir_fd=descriptor)
        except FileExistsError as exc:
            raise FileExistsError(
                f"execution bundle stage file already exists: {candidate}"
            ) from exc
        except OSError as exc:
            raise ValueError(f"execution bundle stage cannot create: {candidate}") from exc
        _write_descriptor(file_descriptor, data)
        expected = os.fstat(file_descriptor)
        if not _has_stable_file_identity(expected) or expected.st_size != len(data):
            raise ValueError(f"execution bundle stage file is invalid: {candidate}")
        if (
            _read_checked_descriptor(
                file_descriptor,
                expected,
                label="execution bundle stage file",
                path=candidate,
                maximum_bytes=len(data),
            )
            != data
        ):
            raise ValueError(f"execution bundle stage file content changed: {candidate}")
        actual = _parent_entry_lstat(current_path, descriptor, parts[-1])
        _reject_link_or_reparse(candidate, actual, label="execution bundle stage")
        if not _same_file_identity(expected, actual):
            raise ValueError(f"execution bundle stage file changed: {candidate}")
        _revalidate_directory_components(root.components, label="execution bundle stage root")
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if descriptor is not None:
            os.close(descriptor)


def _safe_stage_path(root: Path, relative: str) -> Path:
    """Return a validated lexical stage pathname without resolving it."""

    canonical, parts = _relative_components(relative, label="staged relative path")
    _validate_bundle_member_name(canonical, label="staged relative path")
    return _absolute_path(root).joinpath(*parts)


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
                candidate = current / entry.name
                relative = entry.name if not prefix else f"{prefix}/{entry.name}"
                member = _validate_stage_tree_member_name(
                    relative,
                    label="staged member path",
                )
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
    """Enumerate a stage incrementally within authenticated layout limits."""

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
                member = _validate_stage_tree_member_name(
                    relative,
                    label="staged member path",
                )
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


def _expected_stage_directories(files: Iterable[str]) -> set[str]:
    directories: set[str] = set()
    for name in files:
        parts = name.split("/")
        for index in range(1, len(parts)):
            directories.add("/".join(parts[:index]))
    return directories


def _validate_stage_layout(files: set[str], directories: set[str]) -> None:
    """Reject case aliases and file/directory collisions in a stage tree."""

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


def _stage_file_set(root: Path | _DirectoryAnchor) -> set[str]:
    """Retain the private helper while deriving exact limits from the manifest."""

    anchor = root if isinstance(root, _DirectoryAnchor) else _pin_directory(
        _absolute_path(root),
        label="execution bundle stage root",
    )
    manifest_bytes = _read_anchored_relative_bytes(
        anchor,
        BUNDLE_MANIFEST_NAME,
        label="staged execution bundle manifest",
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    manifest = _strict_json_loads(
        manifest_bytes,
        label="staged execution bundle manifest",
    )
    file_entries = _validate_manifest(manifest, manifest_bytes)
    expected_files = {
        BUNDLE_MANIFEST_NAME,
        *[_payload_member_name(str(entry["path"])) for entry in file_entries],
    }
    expected_directories = _expected_stage_directories(expected_files)
    files, directories = _stage_tree_sets(
        anchor,
        maximum_files=len(expected_files),
        maximum_directories=len(expected_directories),
    )
    _validate_stage_layout(files, directories)
    _validate_payload_layout(
        (
            name[len(f"{BUNDLE_PAYLOAD_ROOT}/"):]
            for name in files
            if name != BUNDLE_MANIFEST_NAME
        ),
        label="staged execution bundle",
    )
    return files


def _registry_by_module(manifest: Mapping[str, Any]) -> dict[str, Path]:
    registry = manifest["module_registry"]
    payload_paths: dict[str, Path] = {}
    for entry in registry:
        payload_paths[str(entry["module"])] = Path(str(entry["path"]))
    return payload_paths


def _module_origin(module: ModuleType) -> Path:
    """Return a loader-declared lexical origin without resolving staged paths."""

    raw_origin = getattr(module, "__file__", None)
    if not isinstance(raw_origin, str) or not raw_origin:
        raise ValueError(f"loaded module has no file origin: {module.__name__}")
    return _absolute_path(raw_origin)


def _payload_member_name(path: str) -> str:
    payload_path = _validate_payload_path(path, label="payload path")
    return f"{BUNDLE_PAYLOAD_ROOT}/{payload_path}"


def _validate_module_name(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty module name")
    parts = value.split(".")
    if any(not part or not part.isidentifier() or not part.isascii() for part in parts):
        raise ValueError(f"{label} is not a canonical Python module name")
    return value


def _validate_bundled_module_name(value: Any, *, label: str) -> str:
    """Reject unusable entry names and standard-library shadowing."""

    module = _validate_module_name(value, label=label)
    if module.split(".", 1)[0] == "__main__":
        raise ValueError(f"{label} cannot use __main__")
    if module.split(".", 1)[0] in _STDLIB_MODULE_NAMES:
        raise ValueError(f"{label} cannot shadow a standard-library module")
    return module


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    # Windows reports a creation timestamp through lstat() but a different
    # timestamp through fstat(); device, inode, mode, size, and mtime are the
    # portable descriptor/path identity fields used by this format.
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value
