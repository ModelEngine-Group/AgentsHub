"""Create-once manifest for the exact files included in an experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import unicodedata
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RID_PATTERN = re.compile(
    r"^\d{8}T\d{6}Z-g(?P<head>[0-9a-f]{7})-t(?P<tree>[0-9a-f]{12})$"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
_HAS_SECURE_DIR_FD = (
    os.name != "nt"
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "geteuid")
    and os.stat in os.supports_follow_symlinks
    and all(
        operation in os.supports_dir_fd
        for operation in (os.mkdir, os.open, os.stat)
    )
)
_HAS_WINDOWS_CREATE_ONCE = os.name == "nt" and hasattr(os, "O_EXCL")
_HAS_SECURE_PUBLICATION = _HAS_SECURE_DIR_FD or _HAS_WINDOWS_CREATE_ONCE


def _path_text(path: str | os.PathLike[str]) -> str:
    value = os.fspath(path)
    if not isinstance(value, str):
        raise ValueError("path must be text")
    if "\0" in value:
        raise ValueError("path must not contain a NUL byte")
    return value


def _case_identity(path: str) -> str:
    """Return a portable identity for detecting case-alias path spellings."""

    return unicodedata.normalize("NFC", path).casefold()


def _canonical_relative_input(path: str | os.PathLike[str]) -> tuple[str, tuple[str, ...]]:
    """Validate a user supplied repository-relative path without resolving it."""

    raw = _path_text(path)
    normalized = raw.replace("\\", "/")
    windows_path = PureWindowsPath(raw)
    posix_path = PurePosixPath(normalized)
    if (
        not raw
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
    ):
        raise ValueError("included path must be repository-relative")

    parts = tuple(normalized.split("/"))
    if any(not part for part in parts):
        raise ValueError("included path must not contain empty components")
    if any(part in {".", ".."} for part in parts):
        raise ValueError("included path must not contain lexical traversal")
    return "/".join(parts), parts


def _canonical_manifest_path(path: object) -> str:
    if not isinstance(path, str):
        raise ValueError("included entry path must be text")
    if "\\" in path:
        raise ValueError("included entry path must use POSIX separators")
    canonical, _ = _canonical_relative_input(path)
    if path != canonical:
        raise ValueError("included entry path is not canonical")
    return canonical


def _is_link_or_reparse(metadata: object) -> bool:
    mode = getattr(metadata, "st_mode", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(mode) or bool(attributes & _REPARSE_POINT_ATTRIBUTE)


def _reject_link_or_reparse(path: Path, metadata: object, *, label: str) -> None:
    if _is_link_or_reparse(metadata):
        raise ValueError(f"{label} contains a symlink or reparse component: {path}")


def _reject_hardlink(metadata: object, *, label: str) -> None:
    if getattr(metadata, "st_nlink", 1) != 1:
        raise ValueError(f"{label} must not be a hard link")


def _lstat(path: Path) -> os.stat_result:
    return os.lstat(path)


def _reject_existing_link_or_reparse_components(path: Path, *, label: str) -> None:
    """Reject existing links while walking an absolute path from its filesystem root."""

    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute before component validation")
    anchor = Path(path.anchor)
    current = anchor
    _reject_link_or_reparse(current, _lstat(current), label=label)
    for part in path.parts[1:]:
        current = current / part
        try:
            metadata = _lstat(current)
        except FileNotFoundError:
            return
        _reject_link_or_reparse(current, metadata, label=label)


def _repository_root() -> Path:
    root = Path(_path_text(ROOT))
    if not root.is_absolute():
        root = Path(os.path.abspath(root))
    _reject_existing_link_or_reparse_components(root, label="repository root")
    metadata = _lstat(root)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"repository root is not a directory: {root}")
    return root


def _directory_entry_name(parent: Path, component: str, *, relative: str) -> str:
    """Require an exact, unambiguous on-disk spelling before descending."""

    aliases: list[str] = []
    with os.scandir(parent) as entries:
        for entry in entries:
            if _case_identity(entry.name) == _case_identity(component):
                aliases.append(entry.name)
    if component not in aliases:
        if aliases:
            raise ValueError(f"included path uses a case alias: {relative}")
        raise FileNotFoundError(parent / component)
    if len(aliases) != 1:
        raise ValueError(f"included path has a case-ambiguous component: {relative}")
    return component


def _checked_source_file(
    path: str | os.PathLike[str],
) -> tuple[str, Path, os.stat_result]:
    """Return a checked regular file without resolving user-controlled components first."""

    relative, parts = _canonical_relative_input(path)
    root = _repository_root()
    candidate = root
    for index, component in enumerate(parts):
        component = _directory_entry_name(candidate, component, relative=relative)
        candidate = candidate / component
        metadata = _lstat(candidate)
        _reject_link_or_reparse(candidate, metadata, label="included path")
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(
                f"included path has a non-directory component: {candidate}"
            )

    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"included path must be a regular non-symlink file: {candidate}")
    _reject_hardlink(metadata, label=f"included file {relative}")

    # All user-controlled components have been checked without resolution.  Resolve
    # only now as a containment backstop for unusual filesystem behavior.
    resolved_root = root.resolve(strict=True)
    resolved_candidate = candidate.resolve(strict=True)
    try:
        resolved_relative = resolved_candidate.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"included file is outside the repository: {candidate}") from exc
    if resolved_relative != relative:
        raise ValueError(f"included path resolved to a noncanonical location: {relative}")
    return relative, candidate, metadata


def _has_stable_file_identity(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and not _is_link_or_reparse(metadata)
        and metadata.st_ino != 0
    )


def _same_birthtime(before: os.stat_result, after: os.stat_result) -> bool:
    before_birthtime = getattr(before, "st_birthtime_ns", None)
    after_birthtime = getattr(after, "st_birthtime_ns", None)
    return before_birthtime == after_birthtime


def _same_file_object_identity(before: os.stat_result, after: os.stat_result) -> bool:
    """Compare stable file-object fields across lstat and fstat."""

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
    """Compare lstat/fstat fields with matching Windows semantics."""

    return _same_file_object_identity(before, after) and (
        os.name == "nt" or before.st_ctime_ns == after.st_ctime_ns
    )


def _same_descriptor_snapshot(before: os.stat_result, after: os.stat_result) -> bool:
    """Detect changes while a single descriptor is being read."""

    return _same_file_object_identity(before, after) and (
        before.st_ctime_ns == after.st_ctime_ns
    )


def _sha256_checked_descriptor(
    descriptor: int, expected: os.stat_result, *, label: str
) -> str:
    """Hash a single descriptor after binding it to a checked file snapshot."""

    before_read = os.fstat(descriptor)
    _reject_hardlink(expected, label=label)
    _reject_hardlink(before_read, label=label)
    if not _same_file_identity(expected, before_read):
        raise ValueError(f"{label} changed before hashing")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    after_read = os.fstat(descriptor)
    _reject_hardlink(after_read, label=label)
    if not _same_descriptor_snapshot(before_read, after_read):
        raise ValueError(f"{label} changed while hashing")
    return digest.hexdigest()


def _sha256_checked_file(path: Path, expected: os.stat_result) -> str:
    """Hash one stable regular-file descriptor bound to the checked pathname state."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        return _sha256_checked_descriptor(
            descriptor,
            expected,
            label=f"included file {path}",
        )
    finally:
        os.close(descriptor)


def _entry(path: str | os.PathLike[str]) -> dict[str, Any]:
    relative, candidate, metadata = _checked_source_file(path)
    digest = _sha256_checked_file(candidate, metadata)
    return {
        "path": relative,
        "size": metadata.st_size,
        "sha256": digest,
    }


def _canonical_entries(
    entries: Iterable[Mapping[str, object]],
    *,
    require_nonempty: bool,
) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_identities: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("included entry must be an object")
        relative = _canonical_manifest_path(entry.get("path"))
        size = entry.get("size")
        digest = entry.get("sha256")
        if type(size) is not int or size < 0:
            raise ValueError(f"included entry has an invalid size: {relative}")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"included entry has an invalid SHA-256: {relative}")
        if relative in seen_paths:
            raise ValueError("included file list contains duplicate root-relative paths")
        identity = _case_identity(relative)
        if identity in seen_identities:
            raise ValueError("included file list contains case-alias paths")
        seen_paths.add(relative)
        seen_identities.add(identity)
        canonical.append({"path": relative, "size": size, "sha256": digest})
    if require_nonempty and not canonical:
        raise ValueError("included file list is empty")
    return sorted(canonical, key=lambda item: item["path"])


def build_entries(paths: Iterable[str | os.PathLike[str]]) -> list[dict[str, Any]]:
    return _canonical_entries((_entry(path) for path in paths), require_nonempty=True)


def _tree_sha256_from_canonical_entries(entries: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def tree_sha256(entries: Iterable[Mapping[str, object]]) -> str:
    canonical = _canonical_entries(entries, require_nonempty=False)
    return _tree_sha256_from_canonical_entries(canonical)


def git_head() -> str:
    """Return the current commit identifier, not a clean-tree attestation."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    head = completed.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("git rev-parse returned an invalid HEAD")
    return head


def _absolute_output_path(output: str | os.PathLike[str]) -> Path:
    raw = _path_text(output)
    if not raw:
        raise ValueError("manifest output must name a file")
    if any(component in {".", ".."} for component in raw.replace("\\", "/").split("/")):
        raise ValueError("manifest output must not contain lexical traversal")
    windows_path = PureWindowsPath(raw)
    if windows_path.drive and not windows_path.is_absolute():
        raise ValueError("manifest output must not be drive-relative")
    absolute = Path(os.path.abspath(raw))
    if not absolute.is_absolute() or not absolute.name:
        raise ValueError("manifest output must name a file")
    return absolute


def _has_stable_directory_identity(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not _is_link_or_reparse(metadata)
        and metadata.st_ino != 0
    )


def _same_directory_identity(before: os.stat_result, after: os.stat_result) -> bool:
    """Compare stable directory identity without volatile child-entry timestamps."""

    return (
        _has_stable_directory_identity(before)
        and _has_stable_directory_identity(after)
        and before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_mode == after.st_mode
        and getattr(before, "st_uid", None) == getattr(after, "st_uid", None)
        and getattr(before, "st_gid", None) == getattr(after, "st_gid", None)
        and _same_birthtime(before, after)
    )


def _require_trusted_output_directory(path: Path, metadata: os.stat_result) -> None:
    """Require a cross-principal-safe namespace for final-name binding.

    Ordinary POSIX APIs cannot distinguish a malicious process sharing the
    effective UID, so that same-UID case remains outside this boundary.
    """

    # Windows does not expose an effective UID or portable ACL inspection
    # through ``os``. Its publication path instead rejects every reparse
    # component, snapshots directory identities, and uses an atomic O_EXCL
    # allocation for the final name. ACL administration remains an environment
    # responsibility, as it is for any same-principal writer on POSIX.
    if os.name == "nt":
        return

    owner = getattr(metadata, "st_uid", None)
    current_uid = os.geteuid()
    if owner not in {0, current_uid}:
        raise ValueError(
            f"manifest output parent has an untrusted owner: {path}"
        )
    mode = stat.S_IMODE(metadata.st_mode)
    writable_by_others = mode & (stat.S_IWGRP | stat.S_IWOTH)
    sticky = mode & getattr(stat, "S_ISVTX", 0)
    if writable_by_others and not sticky:
        raise ValueError(
            f"manifest output parent is writable by an untrusted principal: {path}"
        )


def _require_secure_output_parent_support() -> None:
    """Refuse publication without a supported create-once safety boundary."""

    if not _HAS_SECURE_PUBLICATION:
        raise OSError(
            "secure manifest publication requires bound-parent or Windows "
            "create-once support"
        )


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _ensure_output_parent(parent: Path) -> None:
    """Create output directories through checked, no-follow parent descriptors."""

    if not parent.is_absolute():
        raise ValueError("manifest output parent must be absolute")
    _require_secure_output_parent_support()
    if _HAS_WINDOWS_CREATE_ONCE:
        _ensure_output_parent_windows(parent)
        return

    flags = _directory_open_flags()
    anchor = Path(parent.anchor)
    current = anchor
    metadata = _lstat(current)
    _reject_link_or_reparse(current, metadata, label="manifest output parent")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"manifest output parent is not a directory: {current}")
    _require_trusted_output_directory(current, metadata)
    descriptor = os.open(anchor, flags)
    try:
        if not _same_directory_identity(metadata, os.fstat(descriptor)):
            raise ValueError(f"manifest output parent changed before creation: {anchor}")
        for component in parent.parts[1:]:
            candidate = current / component
            try:
                metadata = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                metadata = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            _reject_link_or_reparse(candidate, metadata, label="manifest output parent")
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"manifest output parent is not a directory: {candidate}")
            _require_trusted_output_directory(candidate, metadata)
            child = os.open(component, flags, dir_fd=descriptor)
            try:
                if not _same_directory_identity(metadata, os.fstat(child)):
                    raise ValueError(
                        f"manifest output parent changed before creation: {candidate}"
                    )
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
            current = candidate
    finally:
        os.close(descriptor)


def _ensure_output_parent_windows(parent: Path) -> None:
    """Create a link-free Windows parent and detect lexical namespace swaps.

    Python on Windows has no ``dir_fd`` support. Each creation is therefore
    bracketed by stable parent-identity checks, and publication later repeats
    the full component validation before and after the atomic final-name open.
    A detected race may leave a newly created directory or partial create-once
    file behind, but it can never be treated as a successfully published
    manifest or overwritten on retry.
    """

    anchor = Path(parent.anchor)
    current = anchor
    metadata = _lstat(current)
    _reject_link_or_reparse(current, metadata, label="manifest output parent")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"manifest output parent is not a directory: {current}")
    _require_trusted_output_directory(current, metadata)

    for component in parent.parts[1:]:
        before_parent = _lstat(current)
        _reject_link_or_reparse(
            current,
            before_parent,
            label="manifest output parent",
        )
        if not stat.S_ISDIR(before_parent.st_mode):
            raise ValueError(f"manifest output parent is not a directory: {current}")

        candidate = current / component
        try:
            candidate_metadata = _lstat(candidate)
        except FileNotFoundError:
            try:
                os.mkdir(candidate, mode=0o700)
            except FileExistsError:
                pass
            candidate_metadata = _lstat(candidate)

        after_parent = _lstat(current)
        if not _same_directory_identity(before_parent, after_parent):
            raise ValueError(
                f"manifest output parent changed before creation: {current}"
            )
        _reject_link_or_reparse(
            candidate,
            candidate_metadata,
            label="manifest output parent",
        )
        if not stat.S_ISDIR(candidate_metadata.st_mode):
            raise ValueError(
                f"manifest output parent is not a directory: {candidate}"
            )
        _require_trusted_output_directory(candidate, candidate_metadata)
        confirmed = _lstat(candidate)
        if not _same_directory_identity(candidate_metadata, confirmed):
            raise ValueError(
                f"manifest output parent changed before creation: {candidate}"
            )
        current = candidate


def _capture_output_parent(
    parent: Path,
) -> tuple[tuple[Path, os.stat_result], ...]:
    """Return link-free directory identities for every lexical parent component."""

    _ensure_output_parent(parent)
    anchor = Path(parent.anchor)
    current = anchor
    snapshots: list[tuple[Path, os.stat_result]] = []
    metadata = _lstat(current)
    _reject_link_or_reparse(current, metadata, label="manifest output parent")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"manifest output parent is not a directory: {current}")
    _require_trusted_output_directory(current, metadata)
    snapshots.append((current, metadata))
    for component in parent.parts[1:]:
        current = current / component
        metadata = _lstat(current)
        _reject_link_or_reparse(current, metadata, label="manifest output parent")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"manifest output parent is not a directory: {current}")
        _require_trusted_output_directory(current, metadata)
        snapshots.append((current, metadata))
    return tuple(snapshots)


def _revalidate_output_parent(
    snapshots: tuple[tuple[Path, os.stat_result], ...],
) -> None:
    """Fail closed if a checked output-parent component is replaced or reparsed."""

    for path, before in snapshots:
        try:
            after = _lstat(path)
        except FileNotFoundError as exc:
            raise ValueError(f"manifest output parent changed during publication: {path}") from exc
        if not _same_directory_identity(before, after):
            raise ValueError(f"manifest output parent changed during publication: {path}")


def _open_bound_output_parent(
    parent: Path,
    snapshots: tuple[tuple[Path, os.stat_result], ...],
) -> int:
    """Bind destination operations to checked, no-follow directory descriptors."""

    _require_secure_output_parent_support()
    flags = _directory_open_flags()
    descriptor = os.open(Path(parent.anchor), flags)
    try:
        if not _same_directory_identity(snapshots[0][1], os.fstat(descriptor)):
            raise ValueError(f"manifest output parent changed before publication: {parent}")
        for component, (_, expected) in zip(parent.parts[1:], snapshots[1:]):
            child = os.open(component, flags, dir_fd=descriptor)
            try:
                if not _same_directory_identity(expected, os.fstat(child)):
                    raise ValueError(
                        f"manifest output parent changed before publication: {parent}"
                    )
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _parent_entry_lstat(parent: Path, descriptor: int, name: str) -> os.stat_result:
    return os.stat(name, dir_fd=descriptor, follow_symlinks=False)


def _open_new_output(parent: Path, descriptor: int, name: str) -> int:
    """Atomically create the final output without reusing a staging pathname."""

    # O_RDWR permits an exact-byte check through this same descriptor after the
    # write. The output bytes are never reopened by pathname.
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(name, flags, 0o600, dir_fd=descriptor)
    except FileExistsError as exc:
        raise FileExistsError(f"included-tree manifest already exists: {parent / name}") from exc


def _open_new_output_windows(output: Path) -> int:
    """Atomically allocate a Windows final name without reopening it."""

    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        return os.open(output, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"included-tree manifest already exists: {output}"
        ) from exc


def _write_descriptor(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("could not write manifest output file")
        remaining = remaining[written:]
    os.fsync(descriptor)


def _staging_name(output_name: str) -> str:
    return f".{output_name}.{os.getpid()}.{secrets.token_hex(16)}.tmp"


def _create_staging_posix(parent: Path, descriptor: int, output_name: str) -> tuple[str, int]:
    for _ in range(128):
        name = _staging_name(output_name)
        try:
            return name, _open_new_output(parent, descriptor, name)
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate a unique manifest staging file")


def _create_staging_windows(output: Path) -> tuple[Path, int]:
    for _ in range(128):
        staging = output.with_name(_staging_name(output.name))
        try:
            return staging, _open_new_output_windows(staging)
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate a unique manifest staging file")


def _link_no_replace_posix(
    parent: Path,
    descriptor: int,
    staging_name: str,
    output_name: str,
) -> None:
    try:
        os.link(
            staging_name,
            output_name,
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"included-tree manifest already exists: {parent / output_name}"
        ) from exc


def _move_no_replace_windows(staging: Path, output: Path) -> None:
    """Atomically rename a complete sibling staging file with write-through.

    Windows cannot portably fsync a directory handle.  MoveFileExW with
    MOVEFILE_WRITE_THROUGH is the platform durability primitive for publishing
    the already-fsynced staging file; omitting MOVEFILE_REPLACE_EXISTING keeps
    the final name create-once.
    """

    import ctypes
    from ctypes import wintypes

    move_file = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
    move_file.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    move_file.restype = wintypes.BOOL
    movefile_write_through = 0x00000008
    if move_file(str(staging), str(output), movefile_write_through):
        return
    error = ctypes.get_last_error()
    if error in {80, 183}:
        raise FileExistsError(
            f"included-tree manifest already exists: {output}"
        )
    raise OSError(error, ctypes.FormatError(error), str(output))


def _read_bound_final_posix(
    *,
    parent: Path,
    parent_descriptor: int,
    name: str,
    expected_content: bytes,
) -> tuple[str, int]:
    try:
        expected = _parent_entry_lstat(parent, parent_descriptor, name)
    except FileNotFoundError as exc:
        raise ValueError("manifest output changed during publication") from exc
    _reject_link_or_reparse(parent / name, expected, label="manifest output")
    _reject_hardlink(expected, label="manifest output")
    if not _has_stable_file_identity(expected):
        raise ValueError("manifest output is not a stable regular file")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        before = os.fstat(descriptor)
        _reject_hardlink(before, label="manifest output")
        if not _same_file_identity(expected, before):
            raise ValueError("manifest output changed before verification")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        _reject_hardlink(after, label="manifest output")
        if not _same_descriptor_snapshot(before, after):
            raise ValueError("manifest output changed while verifying")
    finally:
        os.close(descriptor)

    actual_content = b"".join(chunks)
    if actual_content != expected_content:
        raise ValueError("manifest output content differs from payload")
    _verify_output_name_binding(parent, parent_descriptor, name, expected)
    return hashlib.sha256(actual_content).hexdigest(), len(actual_content)


def _read_bound_final_windows(
    *,
    output: Path,
    parent_snapshots: tuple[tuple[Path, os.stat_result], ...],
    expected_content: bytes,
) -> tuple[str, int]:
    _revalidate_output_parent(parent_snapshots)
    try:
        expected = _lstat(output)
    except FileNotFoundError as exc:
        raise ValueError("manifest output changed during publication") from exc
    _reject_link_or_reparse(output, expected, label="manifest output")
    _reject_hardlink(expected, label="manifest output")
    if not _has_stable_file_identity(expected):
        raise ValueError("manifest output is not a stable regular file")

    descriptor = os.open(output, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        before = os.fstat(descriptor)
        _reject_hardlink(before, label="manifest output")
        if not _same_file_identity(expected, before):
            raise ValueError("manifest output changed before verification")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        _reject_hardlink(after, label="manifest output")
        if not _same_descriptor_snapshot(before, after):
            raise ValueError("manifest output changed while verifying")
    finally:
        os.close(descriptor)

    actual_content = b"".join(chunks)
    if actual_content != expected_content:
        raise ValueError("manifest output content differs from payload")
    _revalidate_output_parent(parent_snapshots)
    _verify_windows_output_name_binding(output, expected)
    return hashlib.sha256(actual_content).hexdigest(), len(actual_content)


def _prepare_output_destination(
    output: str | os.PathLike[str],
) -> tuple[Path, tuple[tuple[Path, os.stat_result], ...]]:
    absolute = _absolute_output_path(output)
    return absolute, _capture_output_parent(absolute.parent)


def _prepare_output_path(output: str | os.PathLike[str]) -> Path:
    """Retain the legacy private helper's output-path return shape."""

    absolute, _ = _prepare_output_destination(output)
    return absolute


def _verify_output_name_binding(
    parent: Path,
    descriptor: int,
    name: str,
    expected: os.stat_result,
) -> None:
    """Require the final lexical name to still bind to the held descriptor."""

    try:
        named = _parent_entry_lstat(parent, descriptor, name)
    except FileNotFoundError as exc:
        raise ValueError("manifest output changed during publication") from exc
    if not _same_file_object_identity(expected, named):
        raise ValueError("manifest output changed during publication")


def _verify_windows_output_name_binding(
    output: Path,
    expected: os.stat_result,
) -> None:
    """Require a Windows lexical final name to retain descriptor identity."""

    try:
        named = _lstat(output)
    except FileNotFoundError as exc:
        raise ValueError("manifest output changed during publication") from exc
    _reject_link_or_reparse(output, named, label="manifest output")
    if not _same_file_object_identity(expected, named):
        raise ValueError("manifest output changed during publication")


def _publish_manifest_windows(
    *,
    output: Path,
    parent_snapshots: tuple[tuple[Path, os.stat_result], ...],
    encoded_payload: bytes,
    payload_sha256: str,
) -> None:
    """Write a sibling temporary file, then atomically publish it create-once."""

    _revalidate_output_parent(parent_snapshots)
    staging_path: Path | None = None
    staging_descriptor: int | None = None
    moved = False
    try:
        staging_path, staging_descriptor = _create_staging_windows(output)
        _write_descriptor(staging_descriptor, encoded_payload)
        staging_metadata = os.fstat(staging_descriptor)
        if (
            not _has_stable_file_identity(staging_metadata)
            or staging_metadata.st_size != len(encoded_payload)
        ):
            raise ValueError("manifest staging descriptor is invalid")
        if (
            _sha256_checked_descriptor(
                staging_descriptor,
                staging_metadata,
                label="manifest staging descriptor",
            )
            != payload_sha256
        ):
            raise ValueError("manifest staging content differs from payload")

        _revalidate_output_parent(parent_snapshots)
        _verify_windows_output_name_binding(staging_path, staging_metadata)
        os.close(staging_descriptor)
        staging_descriptor = None
        _move_no_replace_windows(staging_path, output)
        moved = True
        _revalidate_output_parent(parent_snapshots)
        digest, size = _read_bound_final_windows(
            output=output,
            parent_snapshots=parent_snapshots,
            expected_content=encoded_payload,
        )
        if digest != payload_sha256 or size != len(encoded_payload):
            raise ValueError("manifest final verification differs from payload")
        # Rebind once more after the write-through publication so the returned
        # success describes the terminal pathname, not the former staging name.
        digest, size = _read_bound_final_windows(
            output=output,
            parent_snapshots=parent_snapshots,
            expected_content=encoded_payload,
        )
        if digest != payload_sha256 or size != len(encoded_payload):
            raise ValueError("manifest final verification differs from payload")
        _revalidate_output_parent(parent_snapshots)
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if staging_path is not None and not moved:
            try:
                staging_path.unlink()
            except FileNotFoundError:
                pass


def publish_manifest(
    *,
    output: Path,
    rid: str,
    entries: list[dict],
    head: str,
) -> dict:
    canonical_entries = _canonical_entries(entries, require_nonempty=True)
    match = RID_PATTERN.fullmatch(rid)
    if not match:
        raise ValueError("RID does not follow the immutable timestamp/head/tree form")
    try:
        datetime.strptime(rid[:16], "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise ValueError("RID timestamp is not a valid UTC calendar value") from exc
    if not isinstance(head, str) or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("current Git HEAD is invalid")
    tree = _tree_sha256_from_canonical_entries(canonical_entries)
    if match.group("head") != head[:7]:
        raise ValueError("RID Git prefix differs from current HEAD")
    if match.group("tree") != tree[:12]:
        raise ValueError("RID tree prefix differs from included files")
    output, parent_snapshots = _prepare_output_destination(output)
    # Entry digests bind the supplied working-tree bytes. Git HEAD is recorded
    # for provenance only; it does not attest that those bytes were tracked or clean.
    payload = {
        "schema_version": 1,
        "rid": rid,
        "git_head": head,
        "algorithm": "sha256 of sorted path NUL size NUL file_sha256 LF",
        "included_tree_sha256": tree,
        "entries": canonical_entries,
    }
    encoded_payload = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    payload_sha256 = hashlib.sha256(encoded_payload).hexdigest()
    if _HAS_WINDOWS_CREATE_ONCE:
        _publish_manifest_windows(
            output=output,
            parent_snapshots=parent_snapshots,
            encoded_payload=encoded_payload,
            payload_sha256=payload_sha256,
        )
        return payload

    parent_descriptor = _open_bound_output_parent(output.parent, parent_snapshots)
    staging_descriptor: int | None = None
    staging_name: str | None = None
    published = False
    try:
        _revalidate_output_parent(parent_snapshots)
        staging_name, staging_descriptor = _create_staging_posix(
            output.parent,
            parent_descriptor,
            output.name,
        )
        _write_descriptor(staging_descriptor, encoded_payload)
        staging_metadata = os.fstat(staging_descriptor)
        if (
            not _has_stable_file_identity(staging_metadata)
            or staging_metadata.st_size != len(encoded_payload)
        ):
            raise ValueError("manifest staging descriptor is invalid")
        if (
            _sha256_checked_descriptor(
                staging_descriptor,
                staging_metadata,
                label="manifest staging descriptor",
            )
            != payload_sha256
        ):
            raise ValueError("manifest staging content differs from payload")
        _revalidate_output_parent(parent_snapshots)
        _verify_output_name_binding(
            output.parent,
            parent_descriptor,
            staging_name,
            staging_metadata,
        )
        _link_no_replace_posix(
            output.parent,
            parent_descriptor,
            staging_name,
            output.name,
        )
        os.unlink(staging_name, dir_fd=parent_descriptor)
        staging_name = None
        published = True
        _revalidate_output_parent(parent_snapshots)
        digest, size = _read_bound_final_posix(
            parent=output.parent,
            parent_descriptor=parent_descriptor,
            name=output.name,
            expected_content=encoded_payload,
        )
        if digest != payload_sha256 or size != len(encoded_payload):
            raise ValueError("manifest final verification differs from payload")
        os.fsync(parent_descriptor)
        _revalidate_output_parent(parent_snapshots)
        digest, size = _read_bound_final_posix(
            parent=output.parent,
            parent_descriptor=parent_descriptor,
            name=output.name,
            expected_content=encoded_payload,
        )
        if digest != payload_sha256 or size != len(encoded_payload):
            raise ValueError("manifest final verification differs from payload")
        _verify_output_name_binding(
            output.parent,
            parent_descriptor,
            output.name,
            _parent_entry_lstat(output.parent, parent_descriptor, output.name),
        )
        _revalidate_output_parent(parent_snapshots)
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if staging_name is not None:
            try:
                os.unlink(staging_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)
    if not published:
        raise RuntimeError("manifest publication did not complete")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="append", required=True)
    parser.add_argument("--hash-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rid")
    args = parser.parse_args()
    try:
        entries = build_entries(args.path)
        tree = tree_sha256(entries)
        head = git_head()
        if args.hash_only:
            print(
                json.dumps(
                    {
                        "git_head": head,
                        "included_tree_sha256": tree,
                        "n_files": len(entries),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.output is None or args.rid is None:
            raise ValueError("--output and --rid are required unless --hash-only")
        payload = publish_manifest(
            output=args.output,
            rid=args.rid,
            entries=entries,
            head=head,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "rid": payload["rid"],
                "included_tree_sha256": payload["included_tree_sha256"],
                "n_files": len(payload["entries"]),
                "output": str(_absolute_output_path(args.output)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
