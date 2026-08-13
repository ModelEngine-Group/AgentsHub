"""Build a deterministic, create-once source archive from included_tree.json.

The manifest is the complete allow-list.  No filesystem discovery is used to
decide what enters the archive.
"""

from __future__ import annotations

import argparse
import errno
import gzip
import hashlib
import json
import os
import re
import stat
import tarfile
import tempfile
import unicodedata
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO


ARCHIVE_ROOT = "dki-agents"
FIXED_MTIME = 0
MANIFEST_SCHEMA_VERSION = 1
TREE_ALGORITHM = "sha256 of sorted path NUL size NUL file_sha256 LF"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
HEAD_PATTERN = re.compile(r"^[0-9a-f]{40}$")
RID_PATTERN = re.compile(
    r"^\d{8}T\d{6}Z-g(?P<head>[0-9a-f]{7})-t(?P<tree>[0-9a-f]{12})$"
)
_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
_READ_CHUNK_SIZE = 1024 * 1024
_SPOOL_MEMORY_LIMIT = 8 * 1024 * 1024


def _path_text(path: str | os.PathLike[str], *, label: str) -> str:
    value = os.fspath(path)
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    if "\0" in value:
        raise ValueError(f"{label} must not contain a NUL byte")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _absolute_path(path: str | os.PathLike[str], *, label: str) -> Path:
    raw = _path_text(path, label=label)
    windows_path = PureWindowsPath(raw)
    if windows_path.drive and not windows_path.is_absolute():
        raise ValueError(f"{label} must not be drive-relative")
    absolute = Path(os.path.abspath(raw))
    if not absolute.is_absolute():
        raise ValueError(f"{label} must resolve to an absolute path")
    return absolute


def _case_identity(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _is_link_or_reparse(metadata: object) -> bool:
    return stat.S_ISLNK(getattr(metadata, "st_mode", 0)) or bool(
        getattr(metadata, "st_file_attributes", 0)
        & _REPARSE_POINT_ATTRIBUTE
    )


def _reject_link_or_reparse(
    path: Path, metadata: object, *, label: str
) -> None:
    if _is_link_or_reparse(metadata):
        raise ValueError(f"{label} contains a symlink or reparse component: {path}")


def _reject_hardlink(metadata: object, *, label: str) -> None:
    if getattr(metadata, "st_nlink", 1) != 1:
        raise ValueError(f"{label} must not be a hard link")


def _has_stable_file_identity(metadata: object) -> bool:
    return (
        stat.S_ISREG(getattr(metadata, "st_mode", 0))
        and not _is_link_or_reparse(metadata)
        and getattr(metadata, "st_ino", 0) != 0
    )


def _has_stable_directory_identity(metadata: object) -> bool:
    return (
        stat.S_ISDIR(getattr(metadata, "st_mode", 0))
        and not _is_link_or_reparse(metadata)
        and getattr(metadata, "st_ino", 0) != 0
    )


def _same_birthtime(before: object, after: object) -> bool:
    return getattr(before, "st_birthtime_ns", None) == getattr(
        after, "st_birthtime_ns", None
    )


def _same_file_object(before: object, after: object) -> bool:
    return (
        _has_stable_file_identity(before)
        and _has_stable_file_identity(after)
        and getattr(before, "st_dev", None) == getattr(after, "st_dev", None)
        and getattr(before, "st_ino", None) == getattr(after, "st_ino", None)
        and getattr(before, "st_mode", None) == getattr(after, "st_mode", None)
        and getattr(before, "st_size", None) == getattr(after, "st_size", None)
        and getattr(before, "st_mtime_ns", None)
        == getattr(after, "st_mtime_ns", None)
        and _same_birthtime(before, after)
    )


def _same_path_file(before: object, after: object) -> bool:
    return _same_file_object(before, after) and (
        os.name == "nt"
        or getattr(before, "st_ctime_ns", None)
        == getattr(after, "st_ctime_ns", None)
    )


def _same_descriptor_file(before: object, after: object) -> bool:
    return _same_file_object(before, after) and getattr(
        before, "st_ctime_ns", None
    ) == getattr(after, "st_ctime_ns", None)


def _same_directory(before: object, after: object) -> bool:
    return (
        _has_stable_directory_identity(before)
        and _has_stable_directory_identity(after)
        and getattr(before, "st_dev", None) == getattr(after, "st_dev", None)
        and getattr(before, "st_ino", None) == getattr(after, "st_ino", None)
        and stat.S_IFMT(getattr(before, "st_mode", 0))
        == stat.S_IFMT(getattr(after, "st_mode", 0))
        and getattr(before, "st_uid", None) == getattr(after, "st_uid", None)
        and getattr(before, "st_gid", None) == getattr(after, "st_gid", None)
        and _same_birthtime(before, after)
    )


def _snapshot_directory_chain(
    directory: Path, *, label: str
) -> tuple[tuple[Path, os.stat_result], ...]:
    if not directory.is_absolute():
        raise ValueError(f"{label} must be absolute")
    anchor = Path(directory.anchor)
    if not anchor:
        raise ValueError(f"{label} has no filesystem anchor")
    snapshots: list[tuple[Path, os.stat_result]] = []
    current = anchor
    for index, component in enumerate(directory.parts):
        if index:
            current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{label} does not exist: {current}") from exc
        _reject_link_or_reparse(current, metadata, label=label)
        if not _has_stable_directory_identity(metadata):
            raise ValueError(f"{label} is not a stable directory: {current}")
        snapshots.append((current, metadata))
    return tuple(snapshots)


def _revalidate_directory_chain(
    snapshots: tuple[tuple[Path, os.stat_result], ...], *, label: str
) -> None:
    for path, expected in snapshots:
        try:
            current = os.lstat(path)
        except FileNotFoundError as exc:
            raise ValueError(f"{label} changed during archive creation: {path}") from exc
        _reject_link_or_reparse(path, current, label=label)
        if not _same_directory(expected, current):
            raise ValueError(f"{label} changed during archive creation: {path}")


def _exact_component(parent: Path, component: str, *, relative: str) -> None:
    aliases: list[str] = []
    with os.scandir(parent) as children:
        for child in children:
            if _case_identity(child.name) == _case_identity(component):
                aliases.append(child.name)
    if component not in aliases:
        if aliases:
            raise ValueError(f"manifest path uses a case alias: {relative}")
        raise FileNotFoundError(parent / component)
    if len(aliases) != 1:
        raise ValueError(f"manifest path has a case-ambiguous component: {relative}")


def _canonical_manifest_path(value: object) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, str):
        raise ValueError("manifest entry path must be text")
    if "\0" in value:
        raise ValueError("manifest entry path must not contain a NUL byte")
    if "\\" in value:
        raise ValueError("manifest entry path must use POSIX separators")
    windows_path = PureWindowsPath(value)
    posix_path = PurePosixPath(value)
    if (
        not value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
    ):
        raise ValueError("manifest entry path must be repository-relative")
    parts = tuple(value.split("/"))
    if any(not part for part in parts):
        raise ValueError("manifest entry path must not contain empty components")
    if any(part in {".", ".."} for part in parts):
        raise ValueError("manifest entry path must not contain lexical traversal")
    canonical = "/".join(parts)
    if canonical != value:
        raise ValueError("manifest entry path is not canonical")
    return canonical, parts


def _canonical_entries(entries: object) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        raise ValueError("manifest entries must be an array")
    canonical: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_identities: set[str] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("manifest entry must be an object")
        if set(raw_entry) != {"path", "size", "sha256"}:
            raise ValueError(
                "manifest entry must contain exactly path, size, and sha256"
            )
        relative, _ = _canonical_manifest_path(raw_entry["path"])
        size = raw_entry["size"]
        digest = raw_entry["sha256"]
        if type(size) is not int or size < 0:
            raise ValueError(f"manifest entry has an invalid size: {relative}")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"manifest entry has an invalid SHA-256: {relative}")
        if relative in seen_paths:
            raise ValueError("manifest contains duplicate paths")
        identity = _case_identity(relative)
        if identity in seen_identities:
            raise ValueError("manifest contains casefold-alias paths")
        seen_paths.add(relative)
        seen_identities.add(identity)
        canonical.append({"path": relative, "size": size, "sha256": digest})
    return sorted(canonical, key=lambda entry: entry["path"])


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


def _read_bound_regular_file(path: Path, *, label: str) -> bytes:
    parent_snapshots = _snapshot_directory_chain(path.parent, label=f"{label} parent")
    _revalidate_directory_chain(parent_snapshots, label=f"{label} parent")
    _exact_component(path.parent, path.name, relative=str(path))
    expected = os.lstat(path)
    _reject_link_or_reparse(path, expected, label=label)
    _reject_hardlink(expected, label=label)
    if not _has_stable_file_identity(expected):
        raise ValueError(f"{label} must be a stable regular file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        _reject_hardlink(before, label=label)
        if not _same_path_file(expected, before):
            raise ValueError(f"{label} changed before reading")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        _reject_hardlink(after, label=label)
        if not _same_descriptor_file(before, after):
            raise ValueError(f"{label} changed while reading")
    finally:
        os.close(descriptor)

    _revalidate_directory_chain(parent_snapshots, label=f"{label} parent")
    rebound = os.lstat(path)
    _reject_link_or_reparse(path, rebound, label=label)
    _reject_hardlink(rebound, label=label)
    if not _same_path_file(expected, rebound):
        raise ValueError(f"{label} pathname changed while reading")
    return b"".join(chunks)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"manifest JSON contains a duplicate key: {key}")
        result[key] = value
    return result


def load_manifest(path: str | os.PathLike[str]) -> tuple[list[dict[str, Any]], str]:
    manifest_path = _absolute_path(path, label="manifest path")
    encoded = _read_bound_regular_file(manifest_path, label="manifest")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("manifest must be UTF-8 JSON") from exc
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("manifest must be a JSON object")
    expected_fields = {
        "schema_version",
        "rid",
        "git_head",
        "algorithm",
        "included_tree_sha256",
        "entries",
    }
    if set(payload) != expected_fields:
        raise ValueError("manifest must contain exactly the schema-1 field set")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("manifest schema_version must be 1")
    rid = payload["rid"]
    match = RID_PATTERN.fullmatch(rid) if isinstance(rid, str) else None
    if match is None:
        raise ValueError("manifest RID is invalid")
    try:
        datetime.strptime(rid[:16], "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise ValueError("manifest RID timestamp is invalid") from exc
    git_head = payload["git_head"]
    if not isinstance(git_head, str) or HEAD_PATTERN.fullmatch(git_head) is None:
        raise ValueError("manifest git_head is invalid")
    if payload["algorithm"] != TREE_ALGORITHM:
        raise ValueError("manifest tree algorithm is invalid")
    entries = _canonical_entries(payload["entries"])
    if not entries:
        raise ValueError("manifest entries must not be empty")
    if payload["entries"] != entries:
        raise ValueError("manifest entries must be in canonical sorted order")
    claimed_tree = payload["included_tree_sha256"]
    if not isinstance(claimed_tree, str) or not SHA256_PATTERN.fullmatch(
        claimed_tree
    ):
        raise ValueError("manifest included_tree_sha256 is invalid")
    actual_tree = _tree_sha256(entries)
    if claimed_tree != actual_tree:
        raise ValueError("manifest included_tree_sha256 does not match its entries")
    if match.group("head") != git_head[:7]:
        raise ValueError("manifest RID Git prefix does not match git_head")
    if match.group("tree") != actual_tree[:12]:
        raise ValueError(
            "manifest RID tree prefix does not match included_tree_sha256"
        )
    return entries, actual_tree


def _checked_source_path(
    root: Path,
    root_snapshots: tuple[tuple[Path, os.stat_result], ...],
    relative: str,
) -> tuple[Path, os.stat_result, tuple[tuple[Path, os.stat_result], ...]]:
    _, parts = _canonical_manifest_path(relative)
    _revalidate_directory_chain(root_snapshots, label="repository root")
    directory_snapshots = list(root_snapshots)
    current = root
    file_metadata: os.stat_result | None = None
    for index, component in enumerate(parts):
        _exact_component(current, component, relative=relative)
        current = current / component
        metadata = os.lstat(current)
        _reject_link_or_reparse(current, metadata, label="manifest path")
        if index < len(parts) - 1:
            if not _has_stable_directory_identity(metadata):
                raise ValueError(
                    f"manifest path has a non-directory component: {relative}"
                )
            directory_snapshots.append((current, metadata))
        else:
            _reject_hardlink(metadata, label=f"source {relative}")
            if not _has_stable_file_identity(metadata):
                raise ValueError(
                    f"manifest path must name a stable regular file: {relative}"
                )
            file_metadata = metadata
    if file_metadata is None:
        raise ValueError(f"manifest path does not name a file: {relative}")
    snapshots = tuple(directory_snapshots)
    _revalidate_directory_chain(snapshots, label="manifest path parent")
    return current, file_metadata, snapshots


def _revalidate_source_binding(
    root: Path,
    relative: str,
    directory_snapshots: tuple[tuple[Path, os.stat_result], ...],
    expected_file: os.stat_result,
) -> None:
    _revalidate_directory_chain(directory_snapshots, label="manifest path parent")
    current = root
    parts = relative.split("/")
    for component in parts:
        _exact_component(current, component, relative=relative)
        current = current / component
    current_file = os.lstat(current)
    _reject_link_or_reparse(current, current_file, label="manifest path")
    _reject_hardlink(current_file, label=f"source {relative}")
    if not _same_path_file(expected_file, current_file):
        raise ValueError(f"manifest path changed while reading: {relative}")


def _spool_verified_source(
    root: Path,
    root_snapshots: tuple[tuple[Path, os.stat_result], ...],
    entry: Mapping[str, Any],
) -> tuple[BinaryIO, int]:
    relative = str(entry["path"])
    source, expected, directory_snapshots = _checked_source_path(
        root, root_snapshots, relative
    )
    if expected.st_size != entry["size"]:
        raise ValueError(f"source size does not match manifest: {relative}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    spool: BinaryIO = tempfile.SpooledTemporaryFile(max_size=_SPOOL_MEMORY_LIMIT)
    try:
        before = os.fstat(descriptor)
        _reject_hardlink(before, label=f"source {relative}")
        if not _same_path_file(expected, before):
            raise ValueError(f"source changed before reading: {relative}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            spool.write(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        _reject_hardlink(after, label=f"source {relative}")
        if not _same_descriptor_file(before, after):
            raise ValueError(f"source changed while reading: {relative}")
        if total != entry["size"]:
            raise ValueError(f"source size does not match manifest: {relative}")
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"source SHA-256 does not match manifest: {relative}")
        _revalidate_source_binding(
            root, relative, directory_snapshots, expected
        )
        spool.seek(0)
        return spool, 0o644
    except BaseException:
        spool.close()
        raise
    finally:
        os.close(descriptor)


def _tar_info(
    name: str, *, size: int, mode: int, member_type: bytes
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = member_type
    info.size = size
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = FIXED_MTIME
    info.linkname = ""
    info.devmajor = 0
    info.devminor = 0
    info.pax_headers = {}
    return info


def _write_archive(
    output_file: BinaryIO,
    *,
    root: Path,
    root_snapshots: tuple[tuple[Path, os.stat_result], ...],
    entries: list[dict[str, Any]],
) -> None:
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=output_file,
        mtime=FIXED_MTIME,
    ) as compressed:
        with tarfile.open(
            fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
        ) as archive:
            archive.addfile(
                _tar_info(
                    f"{ARCHIVE_ROOT}/",
                    size=0,
                    mode=0o755,
                    member_type=tarfile.DIRTYPE,
                )
            )
            for entry in entries:
                source, mode = _spool_verified_source(
                    root, root_snapshots, entry
                )
                try:
                    archive.addfile(
                        _tar_info(
                            f"{ARCHIVE_ROOT}/{entry['path']}",
                            size=entry["size"],
                            mode=mode,
                            member_type=tarfile.REGTYPE,
                        ),
                        source,
                    )
                finally:
                    source.close()
    _revalidate_directory_chain(root_snapshots, label="repository root")


def _target_exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _hash_open_file(handle: BinaryIO) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(_READ_CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _hash_bound_regular_file(path: Path, *, label: str) -> tuple[str, int]:
    parent_snapshots = _snapshot_directory_chain(path.parent, label=f"{label} parent")
    _revalidate_directory_chain(parent_snapshots, label=f"{label} parent")
    _exact_component(path.parent, path.name, relative=str(path))
    expected = os.lstat(path)
    _reject_link_or_reparse(path, expected, label=label)
    _reject_hardlink(expected, label=label)
    if not _has_stable_file_identity(expected):
        raise ValueError(f"{label} must be a stable regular file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        _reject_hardlink(before, label=label)
        if not _same_path_file(expected, before):
            raise ValueError(f"{label} changed before hashing")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        _reject_hardlink(after, label=label)
        if not _same_descriptor_file(before, after):
            raise ValueError(f"{label} changed while hashing")
    finally:
        os.close(descriptor)

    _revalidate_directory_chain(parent_snapshots, label=f"{label} parent")
    rebound = os.lstat(path)
    _reject_link_or_reparse(path, rebound, label=label)
    _reject_hardlink(rebound, label=label)
    if not _same_path_file(expected, rebound):
        raise ValueError(f"{label} pathname changed while hashing")
    return digest.hexdigest(), total


def _open_bound_output_parent(
    snapshots: tuple[tuple[Path, os.stat_result], ...],
) -> int:
    if os.name == "nt":
        raise OSError("bound directory descriptors are unavailable on Windows")
    parent, expected = snapshots[-1]
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(parent, flags)
    if not _same_directory(expected, os.fstat(descriptor)):
        os.close(descriptor)
        raise ValueError("archive output parent changed before publication")
    return descriptor


def _publish_link_no_replace(
    *,
    parent_descriptor: int,
    parent: Path,
    staging_name: str,
    output_name: str,
) -> None:
    try:
        os.link(
            staging_name,
            output_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise FileExistsError(
                f"archive output already exists: {parent / output_name}"
            ) from exc
        raise


def _move_no_replace_windows(staging: Path, output: Path) -> None:
    """Publish a complete sibling staging file atomically with write-through."""

    import ctypes
    from ctypes import wintypes

    move_file = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
    move_file.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    move_file.restype = wintypes.BOOL
    if move_file(str(staging), str(output), 0x00000008):
        return
    error = ctypes.get_last_error()
    if error in {80, 183}:
        raise FileExistsError(f"archive output already exists: {output}")
    raise OSError(error, ctypes.FormatError(error), str(output))


def build_archive(
    *,
    root: str | os.PathLike[str],
    manifest: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    output_path = _absolute_path(output, label="archive output")
    if not output_path.name:
        raise ValueError("archive output must name a file")
    output_parent_snapshots = _snapshot_directory_chain(
        output_path.parent, label="archive output parent"
    )
    if _target_exists(output_path):
        raise FileExistsError(f"archive output already exists: {output_path}")

    repository_root = _absolute_path(root, label="repository root")
    root_snapshots = _snapshot_directory_chain(
        repository_root, label="repository root"
    )
    entries, tree_digest = load_manifest(manifest)

    descriptor, staging_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    staging_path = Path(staging_name)
    published = False
    staging_digest = ""
    staging_size = 0
    parent_descriptor: int | None = None
    try:
        with os.fdopen(descriptor, "w+b") as staging:
            _write_archive(
                staging,
                root=repository_root,
                root_snapshots=root_snapshots,
                entries=entries,
            )
            staging.flush()
            os.fsync(staging.fileno())
            staging_digest = _hash_open_file(staging)
            staging_metadata = os.fstat(staging.fileno())
            staging_size = staging_metadata.st_size
            named_staging = os.lstat(staging_path)
            if not _same_path_file(staging_metadata, named_staging):
                raise ValueError("archive staging file changed before publication")

            _revalidate_directory_chain(
                output_parent_snapshots, label="archive output parent"
            )
            if _target_exists(output_path):
                raise FileExistsError(
                    f"archive output already exists: {output_path}"
                )
        if os.name == "nt":
            _revalidate_directory_chain(
                output_parent_snapshots, label="archive output parent"
            )
            _move_no_replace_windows(staging_path, output_path)
            published = True
        else:
            parent_descriptor = _open_bound_output_parent(output_parent_snapshots)
            _revalidate_directory_chain(
                output_parent_snapshots, label="archive output parent"
            )
            _publish_link_no_replace(
                parent_descriptor=parent_descriptor,
                parent=output_path.parent,
                staging_name=staging_path.name,
                output_name=output_path.name,
            )
            os.unlink(staging_path.name, dir_fd=parent_descriptor)
            published = True

        _revalidate_directory_chain(
            output_parent_snapshots, label="archive output parent"
        )
        final_digest, final_size = _hash_bound_regular_file(
            output_path, label="archive output"
        )
        if final_digest != staging_digest or final_size != staging_size:
            raise ValueError("archive final output differs from staging")
        if parent_descriptor is not None:
            os.fsync(parent_descriptor)
        # Rebind after the directory durability barrier.  On Windows the
        # equivalent barrier is MOVEFILE_WRITE_THROUGH above.
        _revalidate_directory_chain(
            output_parent_snapshots, label="archive output parent"
        )
        final_digest, final_size = _hash_bound_regular_file(
            output_path, label="archive output"
        )
        if final_digest != staging_digest or final_size != staging_size:
            raise ValueError("archive final output differs from staging")
    finally:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        try:
            staging_path.unlink()
        except FileNotFoundError:
            pass

    if not published:
        raise RuntimeError("archive publication did not complete")
    return {
        "output": str(output_path),
        "n_files": len(entries),
        "included_tree_sha256": tree_digest,
        "archive_sha256": final_digest,
        "archive_size": final_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic source .tar.gz from included_tree.json"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = build_archive(
            root=args.root,
            manifest=args.manifest,
            output=args.output,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
