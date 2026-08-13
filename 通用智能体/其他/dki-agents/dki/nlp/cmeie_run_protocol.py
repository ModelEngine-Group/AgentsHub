"""Authoritative run binding and dataset-access ledger for CMeIE.

The remote run directory is a security boundary rather than a convenient
output prefix.  A run is identified by the exact source manifest installed at
``evidence/source/included_tree.json`` and by the immutable RID encoded in the
run-directory basename.  Every externally supplied dataset may be opened only
after a create-once marker has been published in that bound run.

This module intentionally contains no training or evaluation logic.  It gives
those entry points one shared, fail-closed contract for:

* parsing and validating immutable RIDs;
* binding a canonical remote run layout to its source manifest;
* descriptor-reading every installed source file and rejecting any tree drift;
* deriving every CMeIE artifact path from that binding; and
* atomically publishing a dataset-level access marker whose identity does not
  depend on a selected checkpoint.

Formal commands that import code from the frozen source tree must start Python
with ``PYTHONDONTWRITEBYTECODE=1`` (or the equivalent ``-B`` flag).  A generated
``__pycache__`` directory is an unmanifested source-tree mutation and is
therefore rejected rather than silently ignored.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from dki.evidence_io import load_bound_json, read_bound_bytes


SOURCE_MANIFEST_SCHEMA_VERSION = 1
SOURCE_MANIFEST_ALGORITHM = (
    "sha256 of sorted path NUL size NUL file_sha256 LF"
)
SOURCE_MANIFEST_RELATIVE = Path("evidence") / "source" / "included_tree.json"
SOURCE_ROOT_RELATIVE = Path("source") / "dki-agents"
CMEIE_EVIDENCE_RELATIVE = Path("evidence") / "cmeie"

OUTER_DATASET_ROLE = "outer"
OFFICIAL_DEV_DATASET_ROLE = "official_dev_development_confirmation"
DATASET_ROLES = frozenset(
    {OUTER_DATASET_ROLE, OFFICIAL_DEV_DATASET_ROLE}
)

_RID_PATTERN = re.compile(
    r"^(?P<timestamp>[0-9]{8}T[0-9]{6}Z)"
    r"-g(?P<head>[0-9a-f]{7})"
    r"-t(?P<tree>[0-9a-f]{12})$"
)
_GIT_HEAD_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REPARSE_POINT_ATTRIBUTE = getattr(
    stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400
)
_SOURCE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "rid",
        "git_head",
        "algorithm",
        "included_tree_sha256",
        "entries",
    }
)
_PATH_FIELD_NAMES = frozenset(
    {
        "checkpoint",
        "candidate_directory",
        "checkpoint_directory",
        "checkpoint_bundle_manifest",
    }
)
_BOOLEAN_SUFFIXES = (
    "_accessed",
    "_completed",
    "_enabled",
    "_frozen",
    "_passed",
    "_started",
    "_used",
    "_verified",
)


@dataclass(frozen=True, slots=True)
class RidParts:
    """Canonical components encoded in one immutable run identifier."""

    rid: str
    timestamp: str
    head_prefix: str
    tree_prefix: str


@dataclass(frozen=True, slots=True)
class RunBinding:
    """Descriptor-bound identity of one canonical CMeIE run root."""

    rid: str
    timestamp: str
    git_head: str
    included_tree_sha256: str
    run_root: Path
    source_manifest: Path
    source_root: Path
    source_manifest_sha256: str
    source_manifest_size: int
    source_manifest_device: int
    source_manifest_inode: int

    @property
    def head(self) -> str:
        return self.git_head

    @property
    def tree(self) -> str:
        return self.included_tree_sha256


@dataclass(frozen=True, slots=True)
class _SourceTreeSnapshot:
    """One fail-closed metadata snapshot of the installed source tree."""

    root_identity: tuple[int | None, ...]
    directories: tuple[tuple[str, tuple[int | None, ...]], ...]
    files: tuple[tuple[str, tuple[int | None, ...]], ...]


@dataclass(frozen=True, slots=True)
class CmeieRunPaths:
    """All mutable CMeIE outputs derived from a :class:`RunBinding`."""

    run_root: Path
    source_manifest: Path
    source_root: Path
    evidence_dir: Path
    split_dir: Path
    preflight_receipt: Path
    overfit: Path
    base_model: Path
    seed_models_root: Path
    outer_started: Path
    outer_completion: Path
    outer_gate: Path
    official_dev_marker: Path
    official_dev_completion: Path

    @property
    def overfit_receipt(self) -> Path:
        return self.overfit

    @property
    def official_dev_started(self) -> Path:
        return self.official_dev_marker

    def seed_model(self, seed: int) -> Path:
        """Return the sole canonical model directory for a declared seed."""

        if isinstance(seed, bool) or type(seed) is not int:
            raise TypeError("CMeIE seed must be a non-boolean integer")
        if seed not in {42, 43, 44}:
            raise ValueError("CMeIE seed must be one of 42, 43, or 44")
        return self.seed_models_root / f"seed-{seed}"

    def training_attestation(self, seed: int) -> Path:
        """Return the sole external training attestation for one seed."""

        self.seed_model(seed)
        return self.evidence_dir / f"seed-{seed}-training.json"

    def dataset_started(self, role: str) -> Path:
        """Return the fixed dataset-access marker for ``role``."""

        role = _normalise_dataset_role(role)
        if role == OUTER_DATASET_ROLE:
            return self.outer_started
        return self.official_dev_marker

    def dataset_completion(self, role: str) -> Path:
        """Return the fixed dataset-completion receipt for ``role``."""

        role = _normalise_dataset_role(role)
        if role == OUTER_DATASET_ROLE:
            return self.outer_completion
        return self.official_dev_completion


def parse_rid(value: str) -> RidParts:
    """Parse one canonical ``timestamp-gHEAD7-tTREE12`` run identifier."""

    if type(value) is not str:
        raise TypeError("RID must be text")
    match = _RID_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(
            "RID must match YYYYMMDDTHHMMSSZ-g<7 lowercase hex>"
            "-t<12 lowercase hex>"
        )
    timestamp = match.group("timestamp")
    try:
        datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise ValueError("RID timestamp is not a valid UTC calendar value") from exc
    return RidParts(
        rid=value,
        timestamp=timestamp,
        head_prefix=match.group("head"),
        tree_prefix=match.group("tree"),
    )


def _path_text(path: str | os.PathLike[str], *, label: str) -> str:
    try:
        value = os.fspath(path)
    except TypeError as exc:
        raise TypeError(f"{label} path must be text or path-like") from exc
    if type(value) is not str:
        raise TypeError(f"{label} path must resolve to text")
    if not value:
        raise ValueError(f"{label} path must not be empty")
    if "\0" in value:
        raise ValueError(f"{label} path must not contain a NUL byte")
    return value


def _canonical_absolute_path(
    path: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    raw = _path_text(path, label=label)
    windows = PureWindowsPath(raw)
    if windows.drive and not windows.is_absolute():
        raise ValueError(f"{label} path must not be drive-relative")
    components = raw.replace("\\", "/").split("/")
    if any(component in {".", ".."} for component in components):
        raise ValueError(f"{label} path must not contain lexical traversal")
    supplied = Path(raw)
    if not supplied.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    canonical = Path(os.path.abspath(raw))
    if not canonical.is_absolute() or not canonical.anchor:
        raise ValueError(f"{label} path is not canonical")
    return canonical


def _is_link_or_reparse(metadata: object) -> bool:
    return stat.S_ISLNK(int(getattr(metadata, "st_mode", 0))) or bool(
        int(getattr(metadata, "st_file_attributes", 0))
        & _REPARSE_POINT_ATTRIBUTE
    )


def _validate_existing_path(
    path: Path,
    *,
    label: str,
    expected: str | None = None,
) -> os.stat_result:
    """Reject link/reparse components and return the leaf metadata."""

    canonical = _canonical_absolute_path(path, label=label)
    anchor = Path(canonical.anchor)
    current = anchor
    leaf: os.stat_result | None = None
    for index, component in enumerate(canonical.parts):
        if index:
            current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{label} path component does not exist: {current}"
            ) from exc
        if _is_link_or_reparse(metadata):
            raise ValueError(
                f"{label} path contains a symlink or reparse component: "
                f"{current}"
            )
        if index < len(canonical.parts) - 1 and not stat.S_ISDIR(
            metadata.st_mode
        ):
            raise ValueError(
                f"{label} path component is not a directory: {current}"
            )
        leaf = metadata
    if leaf is None:
        raise ValueError(f"{label} path has no filesystem identity")
    if expected == "directory" and not stat.S_ISDIR(leaf.st_mode):
        raise ValueError(f"{label} path must name a directory")
    if expected == "file" and not stat.S_ISREG(leaf.st_mode):
        raise ValueError(f"{label} path must name a regular file")
    return leaf


def _assert_descendant(root: Path, candidate: Path, *, label: str) -> None:
    try:
        common = os.path.commonpath((str(root), str(candidate)))
    except ValueError as exc:
        raise ValueError(f"{label} path is outside the run root") from exc
    if os.path.normcase(common) != os.path.normcase(str(root)):
        raise ValueError(f"{label} path escapes the run root")


def _canonical_manifest_entry_path(value: Any) -> str:
    if type(value) is not str or not value or "\0" in value:
        raise ValueError("source manifest entry path must be non-empty text")
    if "\\" in value:
        raise ValueError("source manifest entry path must use '/' separators")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
    ):
        raise ValueError("source manifest entry path must be repository-relative")
    parts = value.split("/")
    if any(not part for part in parts):
        raise ValueError(
            "source manifest entry path contains an empty component"
        )
    if any(part in {".", ".."} for part in parts):
        raise ValueError(
            "source manifest entry path contains lexical traversal"
        )
    if "/".join(parts) != value:
        raise ValueError("source manifest entry path is not canonical")
    return value


def included_tree_sha256(entries: Any) -> str:
    """Validate schema-v1 entries and return their canonical tree digest."""

    if type(entries) is not list:
        raise ValueError("source manifest entries must be an array")
    canonical: list[tuple[str, int, str]] = []
    paths: set[str] = set()
    casefold_paths: set[str] = set()
    for raw in entries:
        if type(raw) is not dict or set(raw) != {"path", "size", "sha256"}:
            raise ValueError(
                "source manifest entry must contain exactly path, size, sha256"
            )
        relative = _canonical_manifest_entry_path(raw["path"])
        size = raw["size"]
        digest = raw["sha256"]
        if isinstance(size, bool) or type(size) is not int or size < 0:
            raise ValueError(
                f"source manifest entry size is invalid: {relative}"
            )
        if type(digest) is not str or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(
                f"source manifest entry SHA-256 is invalid: {relative}"
            )
        identity = relative.casefold()
        if relative in paths:
            raise ValueError("source manifest contains duplicate paths")
        if identity in casefold_paths:
            raise ValueError("source manifest contains casefold-alias paths")
        paths.add(relative)
        casefold_paths.add(identity)
        canonical.append((relative, size, digest))

    hasher = hashlib.sha256()
    for relative, size, digest in sorted(canonical):
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(size).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(digest.encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _validated_source_manifest(
    payload: Any,
    *,
    rid: RidParts,
) -> tuple[str, str]:
    if type(payload) is not dict:
        raise ValueError("source manifest root must be an object")
    if set(payload) != _SOURCE_MANIFEST_FIELDS:
        missing = sorted(_SOURCE_MANIFEST_FIELDS - set(payload))
        extra = sorted(set(payload) - _SOURCE_MANIFEST_FIELDS)
        raise ValueError(
            "source manifest fields differ from schema v1; "
            f"missing={missing}, extra={extra}"
        )
    schema_version = payload["schema_version"]
    if (
        isinstance(schema_version, bool)
        or type(schema_version) is not int
        or schema_version != SOURCE_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("source manifest schema_version must be exactly 1")
    if type(payload["rid"]) is not str or payload["rid"] != rid.rid:
        raise ValueError("source manifest RID differs from the run root")
    head = payload["git_head"]
    if type(head) is not str or _GIT_HEAD_PATTERN.fullmatch(head) is None:
        raise ValueError("source manifest git_head must be 40 lowercase hex")
    if not head.startswith(rid.head_prefix):
        raise ValueError("source manifest git_head differs from the RID")
    if payload["algorithm"] != SOURCE_MANIFEST_ALGORITHM:
        raise ValueError("source manifest tree algorithm is not schema v1")
    tree = payload["included_tree_sha256"]
    if type(tree) is not str or _SHA256_PATTERN.fullmatch(tree) is None:
        raise ValueError(
            "source manifest included_tree_sha256 must be lowercase SHA-256"
        )
    actual_tree = included_tree_sha256(payload["entries"])
    if tree != actual_tree:
        raise ValueError(
            "source manifest included_tree_sha256 differs from its entries"
        )
    if not tree.startswith(rid.tree_prefix):
        raise ValueError(
            "source manifest included_tree_sha256 differs from the RID"
        )
    return head, tree


def _birthtime_ns(metadata: object) -> int | None:
    value = getattr(metadata, "st_birthtime_ns", None)
    if value is not None:
        return int(value)
    seconds = getattr(metadata, "st_birthtime", None)
    if seconds is None:
        return None
    return int(float(seconds) * 1_000_000_000)


def _source_entry_identity(metadata: object) -> tuple[int | None, ...]:
    """Return metadata that changes when a scanned filesystem object changes."""

    return (
        int(getattr(metadata, "st_dev")),
        int(getattr(metadata, "st_ino")),
        int(getattr(metadata, "st_mode")),
        int(getattr(metadata, "st_nlink")),
        int(getattr(metadata, "st_size")),
        int(getattr(metadata, "st_mtime_ns")),
        int(getattr(metadata, "st_ctime_ns")),
        _birthtime_ns(metadata),
    )


def _stable_source_directory(metadata: object) -> bool:
    return (
        stat.S_ISDIR(int(getattr(metadata, "st_mode", 0)))
        and not _is_link_or_reparse(metadata)
        and int(getattr(metadata, "st_ino", 0)) != 0
    )


def _stable_source_file(metadata: object) -> bool:
    return (
        stat.S_ISREG(int(getattr(metadata, "st_mode", 0)))
        and not _is_link_or_reparse(metadata)
        and int(getattr(metadata, "st_ino", 0)) != 0
    )


def _lstat_source_path(path: Path, *, relative: str) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError(
            f"source tree changed while it was scanned: {relative or '.'}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"source tree could not be scanned safely: {relative or '.'}"
        ) from exc
    if _is_link_or_reparse(metadata):
        raise ValueError(
            "source tree contains a symlink or reparse point: "
            f"{relative or '.'}"
        )
    return metadata


def _scan_exact_source_tree(source_root: Path) -> _SourceTreeSnapshot:
    """Scan a source root without following links and bind every directory.

    Each directory is checked immediately before and after its entries and
    descendants are scanned.  A second complete scan after all descriptor
    reads gives the caller a whole-tree before/after identity comparison.
    """

    root_metadata = _lstat_source_path(source_root, relative="")
    if not _stable_source_directory(root_metadata):
        raise ValueError("source root must be a stable directory")

    directories: dict[str, tuple[int | None, ...]] = {}
    files: dict[str, tuple[int | None, ...]] = {}
    casefold_entries: dict[str, str] = {}
    file_objects: dict[tuple[int, int], str] = {}

    def register(relative: str) -> None:
        identity = relative.casefold()
        previous = casefold_entries.get(identity)
        if previous is not None:
            if previous == relative:
                raise ValueError(
                    f"source tree contains a duplicate path: {relative}"
                )
            raise ValueError(
                "source tree contains casefold-alias paths: "
                f"{previous}, {relative}"
            )
        casefold_entries[identity] = relative

    def scan_directory(directory: Path, relative: str) -> None:
        before = _lstat_source_path(directory, relative=relative)
        if not _stable_source_directory(before):
            raise ValueError(
                "source tree entry is not a stable directory: "
                f"{relative or '.'}"
            )
        before_identity = _source_entry_identity(before)
        if relative:
            register(relative)
            directories[relative] = before_identity

        try:
            with os.scandir(directory) as iterator:
                names = sorted(entry.name for entry in iterator)
        except OSError as exc:
            raise ValueError(
                "source tree changed while it was scanned: "
                f"{relative or '.'}"
            ) from exc

        if len(names) != len(set(names)):
            raise ValueError(
                "source tree directory scan returned duplicate names: "
                f"{relative or '.'}"
            )
        folded_names: dict[str, str] = {}
        for name in names:
            folded = name.casefold()
            previous_name = folded_names.get(folded)
            if previous_name is not None and previous_name != name:
                raise ValueError(
                    "source tree directory contains casefold aliases: "
                    f"{previous_name}, {name}"
                )
            folded_names[folded] = name

            child_relative = f"{relative}/{name}" if relative else name
            canonical_relative = _canonical_manifest_entry_path(
                child_relative
            )
            child = directory / name
            metadata = _lstat_source_path(
                child,
                relative=canonical_relative,
            )
            if _stable_source_directory(metadata):
                scan_directory(child, canonical_relative)
                continue
            if _stable_source_file(metadata):
                if int(getattr(metadata, "st_nlink", 0)) != 1:
                    raise ValueError(
                        "source tree file has multiple hard links: "
                        f"{canonical_relative}"
                    )
                register(canonical_relative)
                object_identity = (
                    int(getattr(metadata, "st_dev")),
                    int(getattr(metadata, "st_ino")),
                )
                previous_path = file_objects.get(object_identity)
                if previous_path is not None:
                    raise ValueError(
                        "source tree files share one filesystem identity: "
                        f"{previous_path}, {canonical_relative}"
                    )
                file_objects[object_identity] = canonical_relative
                files[canonical_relative] = _source_entry_identity(metadata)
                continue
            raise ValueError(
                "source tree contains a special or unstable entry: "
                f"{canonical_relative}"
            )

        after = _lstat_source_path(directory, relative=relative)
        if (
            not _stable_source_directory(after)
            or _source_entry_identity(after) != before_identity
        ):
            raise ValueError(
                "source tree directory changed while it was scanned: "
                f"{relative or '.'}"
            )

    scan_directory(source_root, "")
    root_after = _lstat_source_path(source_root, relative="")
    root_identity = _source_entry_identity(root_metadata)
    if (
        not _stable_source_directory(root_after)
        or _source_entry_identity(root_after) != root_identity
    ):
        raise ValueError("source root changed while it was scanned")
    return _SourceTreeSnapshot(
        root_identity=root_identity,
        directories=tuple(sorted(directories.items())),
        files=tuple(sorted(files.items())),
    )


def _expected_source_layout(
    entries: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    expected_files: dict[str, dict[str, Any]] = {}
    expected_directories: set[str] = set()
    all_paths: dict[str, str] = {}
    for raw in entries:
        relative = _canonical_manifest_entry_path(raw["path"])
        expected_files[relative] = raw
        parts = relative.split("/")
        for index in range(1, len(parts)):
            expected_directories.add("/".join(parts[:index]))

    collisions = sorted(set(expected_files) & expected_directories)
    if collisions:
        raise ValueError(
            "source manifest maps one path as both a file and directory: "
            f"{collisions[0]}"
        )
    for relative in sorted(set(expected_files) | expected_directories):
        folded = relative.casefold()
        previous = all_paths.get(folded)
        if previous is not None:
            if previous == relative:
                raise ValueError(
                    "source manifest maps one path as both a file and directory: "
                    f"{relative}"
                )
            raise ValueError(
                "source manifest contains a file/directory casefold alias: "
                f"{previous}, {relative}"
            )
        all_paths[folded] = relative
    return expected_files, expected_directories


def _assert_snapshot_layout(
    snapshot: _SourceTreeSnapshot,
    *,
    expected_files: set[str],
    expected_directories: set[str],
) -> None:
    actual_files = {relative for relative, _ in snapshot.files}
    actual_directories = {
        relative for relative, _ in snapshot.directories
    }
    missing_files = sorted(expected_files - actual_files)
    extra_files = sorted(actual_files - expected_files)
    missing_directories = sorted(expected_directories - actual_directories)
    extra_directories = sorted(actual_directories - expected_directories)
    if not (
        missing_files
        or extra_files
        or missing_directories
        or extra_directories
    ):
        return

    pycache_paths = [
        path
        for path in (*extra_files, *extra_directories)
        if "__pycache__" in path.split("/") or path.endswith(".pyc")
    ]
    suffix = ""
    if pycache_paths:
        suffix = (
            "; frozen-source commands must set "
            "PYTHONDONTWRITEBYTECODE=1 before Python starts"
        )
    raise ValueError(
        "source root differs from the exact manifest tree; "
        f"missing_files={missing_files}, extra_files={extra_files}, "
        f"missing_directories={missing_directories}, "
        f"extra_directories={extra_directories}{suffix}"
    )


def _validate_installed_source_tree(
    source_root: Path,
    entries: list[dict[str, Any]],
) -> None:
    expected, expected_directories = _expected_source_layout(entries)
    first = _scan_exact_source_tree(source_root)
    _assert_snapshot_layout(
        first,
        expected_files=set(expected),
        expected_directories=expected_directories,
    )

    descriptor_identities: dict[tuple[int, int], str] = {}
    for relative in sorted(expected):
        path = source_root.joinpath(*relative.split("/"))
        _assert_descendant(source_root, path, label="source manifest entry")
        raw, binding = read_bound_bytes(
            path,
            f"CMeIE run source file {relative}",
            reject_multiple_links=True,
        )
        if binding["canonical_path"] != str(path):
            raise ValueError(
                f"source file is not installed at its canonical path: {relative}"
            )
        manifest_entry = expected[relative]
        if len(raw) != manifest_entry["size"]:
            raise ValueError(
                f"source file size differs from the manifest: {relative}"
            )
        if binding["sha256"] != manifest_entry["sha256"]:
            raise ValueError(
                f"source file SHA-256 differs from the manifest: {relative}"
            )
        object_identity = (
            int(binding["device"]),
            int(binding["inode"]),
        )
        previous = descriptor_identities.get(object_identity)
        if previous is not None:
            raise ValueError(
                "source files share one descriptor identity: "
                f"{previous}, {relative}"
            )
        descriptor_identities[object_identity] = relative

    second = _scan_exact_source_tree(source_root)
    _assert_snapshot_layout(
        second,
        expected_files=set(expected),
        expected_directories=expected_directories,
    )
    if second != first:
        raise ValueError(
            "source tree changed while manifest entries were verified"
        )


def load_run_binding(
    run_root: str | os.PathLike[str],
) -> RunBinding:
    """Validate the exact installed source tree, then return its run binding.

    Formal callers must have disabled Python bytecode writes before interpreter
    startup; this function deliberately treats ``__pycache__`` like every other
    unmanifested file or directory.
    """

    root = _canonical_absolute_path(run_root, label="run root")
    prefix = "dki-agents-"
    if not root.name.startswith(prefix):
        raise ValueError(
            "run root basename must be exactly dki-agents-<RID>"
        )
    rid = parse_rid(root.name[len(prefix) :])
    if root.name != f"{prefix}{rid.rid}":
        raise ValueError(
            "run root basename must be exactly dki-agents-<RID>"
        )
    _validate_existing_path(root, label="run root", expected="directory")

    source_manifest = root / SOURCE_MANIFEST_RELATIVE
    source_root = root / SOURCE_ROOT_RELATIVE
    _assert_descendant(root, source_manifest, label="source manifest")
    _assert_descendant(root, source_root, label="source root")

    payload, manifest_binding = load_bound_json(
        source_manifest,
        "CMeIE run source manifest",
        reject_multiple_links=True,
        require_object=True,
    )
    head, tree = _validated_source_manifest(payload, rid=rid)
    _validate_existing_path(
        source_root,
        label="source root",
        expected="directory",
    )
    if manifest_binding["canonical_path"] != str(source_manifest):
        raise ValueError("source manifest is not installed at its canonical path")
    _validate_installed_source_tree(source_root, payload["entries"])

    # The source walk may be large.  Rebind the manifest only after that walk
    # so neither an in-place edit nor an identical-byte pathname replacement
    # can slip between manifest acceptance and the returned RunBinding.
    manifest_raw, manifest_rebound = read_bound_bytes(
        source_manifest,
        "CMeIE run source manifest rebound",
        reject_multiple_links=True,
    )
    if (
        hashlib.sha256(manifest_raw).hexdigest()
        != manifest_binding["sha256"]
        or manifest_rebound != manifest_binding
    ):
        raise ValueError(
            "source manifest changed while the source tree was verified"
        )

    return RunBinding(
        rid=rid.rid,
        timestamp=rid.timestamp,
        git_head=head,
        included_tree_sha256=tree,
        run_root=root,
        source_manifest=source_manifest,
        source_root=source_root,
        source_manifest_sha256=str(manifest_binding["sha256"]),
        source_manifest_size=int(manifest_binding["size"]),
        source_manifest_device=int(manifest_binding["device"]),
        source_manifest_inode=int(manifest_binding["inode"]),
    )


def _validate_binding_shape(binding: RunBinding) -> None:
    if type(binding) is not RunBinding:
        raise TypeError("run_binding must be a RunBinding")
    rid = parse_rid(binding.rid)
    if binding.timestamp != rid.timestamp:
        raise ValueError("run binding timestamp differs from its RID")
    if (
        type(binding.git_head) is not str
        or _GIT_HEAD_PATTERN.fullmatch(binding.git_head) is None
        or not binding.git_head.startswith(rid.head_prefix)
    ):
        raise ValueError("run binding Git HEAD differs from its RID")
    if (
        type(binding.included_tree_sha256) is not str
        or _SHA256_PATTERN.fullmatch(binding.included_tree_sha256) is None
        or not binding.included_tree_sha256.startswith(rid.tree_prefix)
    ):
        raise ValueError("run binding included tree differs from its RID")
    root = _canonical_absolute_path(binding.run_root, label="run binding root")
    if root != binding.run_root:
        raise ValueError("run binding root is not canonical")
    if root.name != f"dki-agents-{binding.rid}":
        raise ValueError("run binding root basename differs from its RID")
    if binding.source_manifest != root / SOURCE_MANIFEST_RELATIVE:
        raise ValueError("run binding source manifest path is not canonical")
    if binding.source_root != root / SOURCE_ROOT_RELATIVE:
        raise ValueError("run binding source root path is not canonical")
    if (
        type(binding.source_manifest_sha256) is not str
        or _SHA256_PATTERN.fullmatch(binding.source_manifest_sha256) is None
    ):
        raise ValueError("run binding source manifest SHA-256 is invalid")
    for name, value in (
        ("source_manifest_size", binding.source_manifest_size),
        ("source_manifest_device", binding.source_manifest_device),
        ("source_manifest_inode", binding.source_manifest_inode),
    ):
        if isinstance(value, bool) or type(value) is not int or value < 0:
            raise ValueError(f"run binding {name} is invalid")


def canonical_run_paths(binding: RunBinding) -> CmeieRunPaths:
    """Derive the sole admissible path for each CMeIE run artifact."""

    _validate_binding_shape(binding)
    root = binding.run_root
    evidence = root / CMEIE_EVIDENCE_RELATIVE
    paths = CmeieRunPaths(
        run_root=root,
        source_manifest=binding.source_manifest,
        source_root=binding.source_root,
        evidence_dir=evidence,
        split_dir=evidence / "split",
        preflight_receipt=evidence / "preflight.json",
        overfit=evidence / "overfit.json",
        base_model=root / "models" / "cmeie_base_model",
        seed_models_root=root / "models" / "cmeie",
        outer_started=evidence / "outer_access_started.json",
        outer_completion=evidence / "outer_completion.json",
        outer_gate=evidence / "outer_gate.json",
        official_dev_marker=(
            evidence / "official_dev_confirmation_started.json"
        ),
        official_dev_completion=(
            evidence / "official_dev_confirmation_completed.json"
        ),
    )
    for field_name in (
        "evidence_dir",
        "split_dir",
        "preflight_receipt",
        "overfit",
        "base_model",
        "seed_models_root",
        "outer_started",
        "outer_completion",
        "outer_gate",
        "official_dev_marker",
        "official_dev_completion",
    ):
        _assert_descendant(
            root,
            getattr(paths, field_name),
            label=f"canonical {field_name}",
        )
    return paths


canonical_paths = canonical_run_paths


def _normalise_dataset_role(role: str) -> str:
    if type(role) is not str:
        raise TypeError("dataset role must be text")
    if role not in DATASET_ROLES:
        raise ValueError(
            "dataset role must be exactly 'outer' or "
            "'official_dev_development_confirmation'"
        )
    return role


def _normalise_sha256(value: str, *, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def dataset_key_material(
    role: str,
    dataset_sha: str,
    split_sha: str,
) -> bytes:
    """Return the exact checkpoint-independent dataset ledger key material."""

    role = _normalise_dataset_role(role)
    dataset_sha = _normalise_sha256(dataset_sha, label="dataset SHA-256")
    split_sha = _normalise_sha256(split_sha, label="split SHA-256")
    return f"{role}\0{dataset_sha}\0{split_sha}".encode("utf-8")


def dataset_key_sha256(
    role: str,
    dataset_sha: str,
    split_sha: str,
) -> str:
    return hashlib.sha256(
        dataset_key_material(role, dataset_sha, split_sha)
    ).hexdigest()


def _is_path_field(name: str) -> bool:
    return (
        name in _PATH_FIELD_NAMES
        or name.endswith("_path")
        or name.endswith("_directory")
        or name.endswith("_dir")
        or name.endswith("_root")
    )


def _validate_evidence_path(value: Any, *, field: str) -> None:
    if type(value) is not str:
        raise ValueError(f"input model evidence {field} must be a path string")
    path = _canonical_absolute_path(value, label=f"input model evidence {field}")
    _validate_existing_path(path, label=f"input model evidence {field}")


def _validate_evidence_json(
    value: Any,
    *,
    path: str,
    field_name: str | None = None,
) -> None:
    if value is None or type(value) in {str, bool, int}:
        pass
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
    elif type(value) is list:
        for index, item in enumerate(value):
            _validate_evidence_json(item, path=f"{path}[{index}]")
    elif type(value) is dict:
        for key, item in value.items():
            if type(key) is not str or not key or "\0" in key:
                raise ValueError(
                    f"{path} contains an invalid JSON object key"
                )
            _validate_evidence_json(
                item,
                path=f"{path}.{key}",
                field_name=key,
            )
    else:
        raise ValueError(
            f"{path} contains a non-canonical JSON value of type "
            f"{type(value).__name__}"
        )

    if field_name is None:
        return
    if field_name == "sha256" or field_name.endswith("_sha256"):
        _normalise_sha256(value, label=f"input model evidence {field_name}")
    if _is_path_field(field_name):
        _validate_evidence_path(value, field=field_name)
    if field_name.endswith(_BOOLEAN_SUFFIXES) and type(value) is not bool:
        raise ValueError(
            f"input model evidence {field_name} must be a JSON boolean"
        )


def _canonical_json_bytes(value: Any) -> bytes:
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


def _validate_attestation_json(
    value: Any,
    *,
    path: str,
    field_name: str | None = None,
) -> None:
    """Validate attestation JSON without treating bound filenames as paths."""

    if value is None or type(value) in {str, bool, int}:
        pass
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
    elif type(value) is list:
        for index, item in enumerate(value):
            _validate_attestation_json(item, path=f"{path}[{index}]")
    elif type(value) is dict:
        for key, item in value.items():
            if type(key) is not str or not key or "\0" in key:
                raise ValueError(
                    f"{path} contains an invalid JSON object key"
                )
            _validate_attestation_json(
                item,
                path=f"{path}.{key}",
                field_name=key,
            )
    else:
        raise ValueError(
            f"{path} contains a non-canonical JSON value of type "
            f"{type(value).__name__}"
        )
    if field_name is None:
        return
    if field_name == "sha256" or field_name.endswith("_sha256"):
        _normalise_sha256(value, label=f"training attestation {field_name}")
    if field_name.endswith(_BOOLEAN_SUFFIXES) and type(value) is not bool:
        raise ValueError(
            f"training attestation {field_name} must be a JSON boolean"
        )


def _same_binding(left: RunBinding, right: RunBinding) -> bool:
    return left == right


def _ensure_evidence_directory(paths: CmeieRunPaths) -> None:
    _validate_existing_path(
        paths.run_root / "evidence",
        label="run evidence root",
        expected="directory",
    )
    try:
        os.mkdir(paths.evidence_dir)
    except FileExistsError:
        pass
    _validate_existing_path(
        paths.evidence_dir,
        label="CMeIE evidence directory",
        expected="directory",
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _publish_hardlink_once(path: Path, encoded: bytes) -> None:
    """Publish ``encoded`` by a same-directory hard link without replacement."""

    if os.path.lexists(path):
        raise FileExistsError(f"dataset access marker already exists: {path}")
    _validate_existing_path(
        path.parent,
        label="dataset marker parent",
        expected="directory",
    )
    stage = path.parent / (
        f".{path.name}.stage-{os.getpid()}-{secrets.token_hex(12)}"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
    )
    descriptor = os.open(stage, flags, 0o600)
    try:
        total = 0
        while total < len(encoded):
            written = os.write(descriptor, encoded[total:])
            if written <= 0:
                raise OSError("failed to write dataset marker staging file")
            total += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    linked = False
    try:
        try:
            os.link(stage, path, follow_symlinks=False)
            linked = True
        except FileExistsError as exc:
            raise FileExistsError(
                f"dataset access marker already exists: {path}"
            ) from exc
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(stage)
        except FileNotFoundError:
            pass
        if linked:
            _fsync_directory(path.parent)

    actual, _ = read_bound_bytes(path, "published dataset access marker")
    if actual != encoded:
        raise RuntimeError("published dataset access marker content mismatch")


def dataset_started_path(binding: RunBinding, role: str) -> Path:
    return canonical_run_paths(binding).dataset_started(role)


def dataset_completion_path(binding: RunBinding, role: str) -> Path:
    return canonical_run_paths(binding).dataset_completion(role)


def publish_training_attestation_once(
    run_binding: RunBinding,
    seed: int,
    evidence_payload: dict[str, Any],
) -> Path:
    """Publish the immutable external attestation for one trained candidate.

    Candidate publication happens before this call.  Therefore any failure
    leaves the candidate directory in place while the absent attestation keeps
    it ineligible for downstream selection.
    """

    _validate_binding_shape(run_binding)
    paths = canonical_run_paths(run_binding)
    candidate = paths.seed_model(seed)
    attestation = paths.training_attestation(seed)
    if os.path.lexists(attestation):
        raise FileExistsError(
            f"training attestation already exists: {attestation}"
        )
    if type(evidence_payload) is not dict:
        raise TypeError("evidence_payload must be a JSON object")
    reserved = {
        "schema_version",
        "kind",
        "attested",
        "rid",
        "head",
        "tree",
        "run_root",
        "source_manifest_path",
        "source_manifest_sha256",
        "seed",
        "candidate_directory",
    }
    collisions = sorted(reserved & set(evidence_payload))
    if collisions:
        raise ValueError(
            "training attestation evidence overrides reserved fields: "
            + ", ".join(collisions)
        )
    required = {
        "training_receipt_sha256",
        "epoch_bindings",
        "preflight_common",
        "source_binding",
        "data_binding",
        "configuration",
    }
    missing = sorted(required - set(evidence_payload))
    if missing:
        raise ValueError(
            "training attestation evidence lacks: " + ", ".join(missing)
        )
    _validate_attestation_json(
        evidence_payload,
        path="training_attestation_evidence",
    )

    live_binding = load_run_binding(run_binding.run_root)
    if not _same_binding(live_binding, run_binding):
        raise ValueError(
            "run binding changed since the source manifest was loaded"
        )
    paths = canonical_run_paths(live_binding)
    candidate = paths.seed_model(seed)
    _validate_existing_path(
        candidate,
        label="training candidate directory",
        expected="directory",
    )
    receipt_path = candidate / "training_receipt.json"
    receipt_bytes, receipt_binding = read_bound_bytes(
        receipt_path,
        "published CMeIE training receipt",
        reject_multiple_links=True,
    )
    expected_receipt_sha256 = _normalise_sha256(
        evidence_payload["training_receipt_sha256"],
        label="training receipt SHA-256",
    )
    if str(receipt_binding["sha256"]) != expected_receipt_sha256:
        raise ValueError(
            "training receipt SHA-256 differs from attestation evidence"
        )
    if hashlib.sha256(receipt_bytes).hexdigest() != expected_receipt_sha256:
        raise RuntimeError("training receipt descriptor digest is inconsistent")

    attestation = paths.training_attestation(seed)
    if os.path.lexists(attestation):
        raise FileExistsError(
            f"training attestation already exists: {attestation}"
        )
    _ensure_evidence_directory(paths)
    evidence_bytes = _canonical_json_bytes(evidence_payload)
    payload = {
        "schema_version": 1,
        "kind": "cmeie_seed_training_attestation",
        "attested": True,
        "rid": live_binding.rid,
        "head": live_binding.git_head,
        "tree": live_binding.included_tree_sha256,
        "run_root": str(live_binding.run_root),
        "source_manifest_path": str(live_binding.source_manifest),
        "source_manifest_sha256": live_binding.source_manifest_sha256,
        "seed": seed,
        "candidate_directory": str(candidate),
        **evidence_payload,
        "training_evidence_sha256": hashlib.sha256(
            evidence_bytes
        ).hexdigest(),
    }
    _publish_hardlink_once(attestation, _canonical_json_bytes(payload))
    return attestation


def publish_dataset_started_once(
    run_binding: RunBinding,
    role: str,
    dataset_sha: str,
    split_sha: str,
    evidence_payload: dict[str, Any],
) -> Path:
    """Publish the sole access marker for one run and one dataset role.

    The ledger key is exactly ``role NUL dataset_sha NUL split_sha``.  Selected
    model/checkpoint evidence is persisted and hashed, but it is deliberately
    excluded from that key.  Consequently copying a candidate or changing a
    selection can never authorize another access to the same dataset in the
    same RID.
    """

    _validate_binding_shape(run_binding)
    role = _normalise_dataset_role(role)
    paths = canonical_run_paths(run_binding)
    marker = paths.dataset_started(role)

    # Existing output is terminal regardless of a caller's new model evidence
    # or of later damage to other run artifacts.  Never reopen or replace it.
    if os.path.lexists(marker):
        raise FileExistsError(
            f"dataset access marker already exists: {marker}"
        )

    dataset_sha = _normalise_sha256(dataset_sha, label="dataset SHA-256")
    split_sha = _normalise_sha256(split_sha, label="split SHA-256")
    if type(evidence_payload) is not dict:
        raise TypeError("evidence_payload must be a JSON object")
    _validate_evidence_json(evidence_payload, path="evidence_payload")

    live_binding = load_run_binding(run_binding.run_root)
    if not _same_binding(live_binding, run_binding):
        raise ValueError(
            "run binding changed since the source manifest was loaded"
        )
    paths = canonical_run_paths(live_binding)
    marker = paths.dataset_started(role)
    if os.path.lexists(marker):
        raise FileExistsError(
            f"dataset access marker already exists: {marker}"
        )
    _ensure_evidence_directory(paths)

    key_material = dataset_key_material(role, dataset_sha, split_sha)
    input_model_bytes = _canonical_json_bytes(evidence_payload)
    payload = {
        "schema_version": 1,
        "kind": "cmeie_dataset_access_started",
        "started": True,
        "completion_recorded": False,
        "rid": live_binding.rid,
        "head": live_binding.git_head,
        "tree": live_binding.included_tree_sha256,
        "run_root": str(live_binding.run_root),
        "source_manifest_path": str(live_binding.source_manifest),
        "source_manifest_sha256": live_binding.source_manifest_sha256,
        "dataset_role": role,
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "dataset_key": key_material.decode("utf-8"),
        "dataset_key_sha256": hashlib.sha256(key_material).hexdigest(),
        "input_model_evidence": evidence_payload,
        "input_model_evidence_sha256": hashlib.sha256(
            input_model_bytes
        ).hexdigest(),
    }
    encoded = _canonical_json_bytes(payload)
    _publish_hardlink_once(marker, encoded)
    return marker


__all__ = [
    "CMEIE_EVIDENCE_RELATIVE",
    "DATASET_ROLES",
    "OFFICIAL_DEV_DATASET_ROLE",
    "OUTER_DATASET_ROLE",
    "SOURCE_MANIFEST_RELATIVE",
    "SOURCE_ROOT_RELATIVE",
    "CmeieRunPaths",
    "RidParts",
    "RunBinding",
    "canonical_paths",
    "canonical_run_paths",
    "dataset_completion_path",
    "dataset_key_material",
    "dataset_key_sha256",
    "dataset_started_path",
    "included_tree_sha256",
    "load_run_binding",
    "parse_rid",
    "publish_dataset_started_once",
    "publish_training_attestation_once",
]
