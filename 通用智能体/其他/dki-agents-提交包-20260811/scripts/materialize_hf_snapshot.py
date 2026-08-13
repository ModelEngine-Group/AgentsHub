"""Materialize a Hugging Face snapshot without publishing symlinks.

Hugging Face cache snapshots normally contain relative symlinks into a sibling
``blobs`` directory.  The strict remote preflight intentionally rejects links,
so this utility copies the dereferenced bytes into a new create-once directory
and binds the operation with a deterministic receipt.

The source snapshot is treated as hostile, mutable input:

* every path component is inspected with ``lstat``;
* directory symlinks and non-symlink reparse points are rejected;
* file symlinks may resolve only to regular files below ``--allowed-root``;
* source and target bindings are checked around descriptor-based reads; and
* the complete source inventory is rescanned immediately before publication.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, BinaryIO


RECEIPT_NAME = "materialization_receipt.json"
TREE_HASH_ALGORITHM = (
    "sha256 of sorted UTF-8 relative_path NUL decimal-size NUL "
    "lowercase-sha256(file bytes) LF"
)
_READ_CHUNK_SIZE = 1024 * 1024
_REPARSE_POINT_ATTRIBUTE = getattr(
    stat,
    "FILE_ATTRIBUTE_REPARSE_POINT",
    0x0400,
)
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_WINDOWS_ERROR_FILE_EXISTS = {80, 183}
_UNSUPPORTED_PUBLICATION_ERRNOS = {
    errno.ENOSYS,
    errno.ENOTSUP,
    getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
}


@dataclass(frozen=True)
class _MetadataBinding:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    birthtime_ns: int | None
    file_attributes: int


@dataclass(frozen=True)
class _DirectoryBinding:
    path: Path
    metadata: _MetadataBinding


@dataclass(frozen=True)
class _LinkBinding:
    path: Path
    metadata: _MetadataBinding
    target: str
    parent_chain: tuple[_DirectoryBinding, ...]


@dataclass(frozen=True)
class _SourceEntry:
    relative_path: str
    origin: Path
    kind: str
    link_target: str | None
    origin_metadata: _MetadataBinding
    origin_parent_chain: tuple[_DirectoryBinding, ...]
    resolved_source: Path
    resolved_metadata: _MetadataBinding
    resolved_parent_chain: tuple[_DirectoryBinding, ...]
    resolution_links: tuple[_LinkBinding, ...]


@dataclass(frozen=True)
class _Inventory:
    directories: tuple[tuple[str, _MetadataBinding], ...]
    entries: tuple[_SourceEntry, ...]


def _path_text(path: str | os.PathLike[str], *, label: str) -> str:
    value = os.fspath(path)
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    if not value:
        raise ValueError(f"{label} must not be empty")
    if "\0" in value:
        raise ValueError(f"{label} must not contain a NUL byte")
    return value


def _absolute_path(path: str | os.PathLike[str], *, label: str) -> Path:
    raw = _path_text(path, label=label)
    windows = PureWindowsPath(raw)
    if windows.drive and not windows.is_absolute():
        raise ValueError(f"{label} must not be drive-relative")
    absolute = Path(os.path.abspath(raw))
    if not absolute.is_absolute():
        raise ValueError(f"{label} must resolve to an absolute path")
    return absolute


def _case_identity(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _portable_component(component: str, *, label: str) -> None:
    if (
        not component
        or component in {".", ".."}
        or component != unicodedata.normalize("NFC", component)
        or "/" in component
        or "\\" in component
        or "\0" in component
    ):
        raise ValueError(f"{label} contains a noncanonical path component")
    try:
        component.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} contains a non-UTF-8 path component") from exc
    windows = PureWindowsPath(component)
    if (
        ":" in component
        or component.endswith((" ", "."))
        or windows.is_reserved()
    ):
        raise ValueError(f"{label} is not a portable canonical path")


def _is_reparse(metadata: os.stat_result | Any) -> bool:
    return bool(
        getattr(metadata, "st_file_attributes", 0)
        & _REPARSE_POINT_ATTRIBUTE
    )


def _is_symlink(metadata: os.stat_result | Any) -> bool:
    return stat.S_ISLNK(getattr(metadata, "st_mode", 0))


def _reject_non_symlink_reparse(
    path: Path,
    metadata: os.stat_result | Any,
    *,
    label: str,
) -> None:
    if _is_reparse(metadata) and not _is_symlink(metadata):
        raise ValueError(f"{label} contains a non-symlink reparse point: {path}")


def _metadata_binding(
    metadata: os.stat_result | Any,
    *,
    label: str,
) -> _MetadataBinding:
    inode = int(getattr(metadata, "st_ino", 0))
    if inode == 0:
        raise ValueError(f"{label} has no stable filesystem identity")
    return _MetadataBinding(
        device=int(getattr(metadata, "st_dev", 0)),
        inode=inode,
        mode=int(getattr(metadata, "st_mode", 0)),
        size=int(getattr(metadata, "st_size", 0)),
        mtime_ns=int(getattr(metadata, "st_mtime_ns", 0)),
        ctime_ns=int(getattr(metadata, "st_ctime_ns", 0)),
        birthtime_ns=getattr(metadata, "st_birthtime_ns", None),
        file_attributes=int(getattr(metadata, "st_file_attributes", 0)),
    )


def _same_file_binding(
    expected: _MetadataBinding,
    metadata: os.stat_result | Any,
) -> bool:
    return expected == _metadata_binding(metadata, label="revalidated file")


def _same_directory_identity(
    expected: _MetadataBinding,
    metadata: os.stat_result | Any,
) -> bool:
    current = _metadata_binding(metadata, label="revalidated directory")
    return (
        stat.S_ISDIR(expected.mode)
        and stat.S_ISDIR(current.mode)
        and expected.device == current.device
        and expected.inode == current.inode
        and stat.S_IFMT(expected.mode) == stat.S_IFMT(current.mode)
        and expected.birthtime_ns == current.birthtime_ns
        and expected.file_attributes == current.file_attributes
    )


def _exact_component(
    parent: Path,
    component: str,
    *,
    label: str,
) -> None:
    aliases: list[str] = []
    with os.scandir(parent) as children:
        for child in children:
            if _case_identity(child.name) == _case_identity(component):
                aliases.append(child.name)
    if component not in aliases:
        if aliases:
            raise ValueError(f"{label} uses a casefold alias: {component}")
        raise FileNotFoundError(parent / component)
    if len(aliases) != 1:
        raise ValueError(f"{label} has a casefold-ambiguous component: {component}")


def _snapshot_real_directory_chain(
    directory: Path,
    *,
    label: str,
) -> tuple[_DirectoryBinding, ...]:
    if not directory.is_absolute():
        raise ValueError(f"{label} must be absolute")
    anchor = Path(directory.anchor)
    if not anchor:
        raise ValueError(f"{label} has no filesystem anchor")
    bindings: list[_DirectoryBinding] = []
    current = anchor
    anchor_metadata = os.lstat(current)
    if _is_symlink(anchor_metadata) or _is_reparse(anchor_metadata):
        raise ValueError(f"{label} has a linked filesystem anchor")
    if not stat.S_ISDIR(anchor_metadata.st_mode):
        raise ValueError(f"{label} anchor is not a directory")
    bindings.append(
        _DirectoryBinding(
            current,
            _metadata_binding(anchor_metadata, label=f"{label} anchor"),
        )
    )
    for component in directory.parts[1:]:
        _portable_component(component, label=label)
        _exact_component(current, component, label=label)
        current = current / component
        metadata = os.lstat(current)
        if _is_symlink(metadata) or _is_reparse(metadata):
            raise ValueError(f"{label} contains a symlink or reparse point: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} contains a non-directory component: {current}")
        bindings.append(
            _DirectoryBinding(
                current,
                _metadata_binding(metadata, label=label),
            )
        )
    return tuple(bindings)


def _revalidate_directory_chain(
    chain: tuple[_DirectoryBinding, ...],
    *,
    label: str,
) -> None:
    for expected in chain:
        try:
            metadata = os.lstat(expected.path)
        except FileNotFoundError as exc:
            raise ValueError(f"{label} changed: {expected.path}") from exc
        if _is_symlink(metadata) or _is_reparse(metadata):
            raise ValueError(f"{label} became a symlink or reparse point")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} became a non-directory")
        if not _same_directory_identity(expected.metadata, metadata):
            raise ValueError(f"{label} changed: {expected.path}")


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _relative_parts_within(path: Path, root: Path, *, label: str) -> tuple[str, ...]:
    if not _path_is_within(path, root):
        raise ValueError(f"{label} escapes the allowed root: {path}")
    relative = os.path.relpath(path, root)
    if relative == ".":
        return ()
    parts = tuple(Path(relative).parts)
    for component in parts:
        _portable_component(component, label=label)
    return parts


def _chain_to_real_directory_within(
    directory: Path,
    *,
    allowed_root: Path,
    allowed_root_chain: tuple[_DirectoryBinding, ...],
    label: str,
) -> tuple[_DirectoryBinding, ...]:
    parts = _relative_parts_within(directory, allowed_root, label=label)
    chain = list(allowed_root_chain)
    current = allowed_root
    for component in parts:
        _exact_component(current, component, label=label)
        current = current / component
        metadata = os.lstat(current)
        if _is_symlink(metadata) or _is_reparse(metadata):
            raise ValueError(f"{label} contains a directory link: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} contains a non-directory: {current}")
        chain.append(
            _DirectoryBinding(
                current,
                _metadata_binding(metadata, label=label),
            )
        )
    return tuple(chain)


def _read_stable_link(
    path: Path,
    expected: _MetadataBinding,
    *,
    label: str,
) -> str:
    before = os.lstat(path)
    if not _is_symlink(before):
        raise ValueError(f"{label} is no longer a symlink: {path}")
    if not _same_file_binding(expected, before):
        raise ValueError(f"{label} changed before readlink: {path}")
    target = os.readlink(path)
    after = os.lstat(path)
    if not _is_symlink(after) or not _same_file_binding(expected, after):
        raise ValueError(f"{label} changed during readlink: {path}")
    if not isinstance(target, str) or not target or "\0" in target:
        raise ValueError(f"{label} has an invalid link target: {path}")
    return target


def _lexical_link_target(link: Path, target: str) -> Path:
    windows = PureWindowsPath(target)
    if windows.drive and not windows.is_absolute():
        raise ValueError(f"symlink has a drive-relative target: {link}")
    if Path(target).is_absolute():
        return Path(os.path.abspath(target))
    return Path(os.path.abspath(os.path.join(link.parent, target)))


def _resolve_file_link(
    link: Path,
    link_metadata: _MetadataBinding,
    *,
    allowed_root: Path,
    allowed_root_chain: tuple[_DirectoryBinding, ...],
) -> tuple[
    str,
    Path,
    _MetadataBinding,
    tuple[_DirectoryBinding, ...],
    tuple[_LinkBinding, ...],
]:
    first_target = _read_stable_link(
        link,
        link_metadata,
        label="source symlink",
    )
    candidate = _lexical_link_target(link, first_target)
    seen: set[tuple[int, int]] = {
        (link_metadata.device, link_metadata.inode)
    }
    resolution_links: list[_LinkBinding] = []

    while True:
        parts = _relative_parts_within(
            candidate,
            allowed_root,
            label="symlink target",
        )
        if not parts:
            raise ValueError(f"symlink resolves to the allowed-root directory: {link}")
        chain = list(allowed_root_chain)
        current = allowed_root
        restart = False
        for index, component in enumerate(parts):
            _exact_component(current, component, label="symlink target")
            current = current / component
            metadata = os.lstat(current)
            _reject_non_symlink_reparse(
                current,
                metadata,
                label="symlink target",
            )
            last = index == len(parts) - 1
            if _is_symlink(metadata):
                binding = _metadata_binding(
                    metadata,
                    label="symlink target",
                )
                if not last:
                    raise ValueError(
                        f"symlink target contains a directory link: {current}"
                    )
                identity = (binding.device, binding.inode)
                if identity in seen:
                    raise ValueError(f"symlink cycle detected at: {current}")
                seen.add(identity)
                target = _read_stable_link(
                    current,
                    binding,
                    label="symlink target",
                )
                resolution_links.append(
                    _LinkBinding(
                        path=current,
                        metadata=binding,
                        target=target,
                        parent_chain=tuple(chain),
                    )
                )
                candidate = _lexical_link_target(current, target)
                restart = True
                break
            if last:
                if stat.S_ISDIR(metadata.st_mode):
                    raise ValueError(f"directory symlink is forbidden: {link}")
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(
                        f"symlink does not resolve to a regular file: {link}"
                    )
                return (
                    first_target,
                    current,
                    _metadata_binding(metadata, label="resolved source"),
                    tuple(chain),
                    tuple(resolution_links),
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    f"symlink target has a non-directory component: {current}"
                )
            chain.append(
                _DirectoryBinding(
                    current,
                    _metadata_binding(metadata, label="symlink target directory"),
                )
            )
        if not restart:
            raise AssertionError("symlink resolution ended without a file")


def _directory_names(directory: Path, *, relative: str) -> list[str]:
    names: list[str] = []
    aliases: dict[str, str] = {}
    with os.scandir(directory) as children:
        for child in children:
            name = child.name
            _portable_component(name, label=f"source entry {relative or '.'}")
            identity = _case_identity(name)
            previous = aliases.get(identity)
            if previous is not None:
                raise ValueError(
                    "source directory contains casefold-alias entries: "
                    f"{previous!r}, {name!r}"
                )
            aliases[identity] = name
            names.append(name)
    return sorted(names)


def _inventory_source(
    *,
    source_snapshot: Path,
    source_chain: tuple[_DirectoryBinding, ...],
    allowed_root: Path,
    allowed_root_chain: tuple[_DirectoryBinding, ...],
) -> _Inventory:
    _revalidate_directory_chain(source_chain, label="source snapshot")
    directories: list[tuple[str, _MetadataBinding]] = []
    entries: list[_SourceEntry] = []
    seen_paths: set[str] = set()
    seen_casefold_paths: set[str] = set()

    def walk(
        directory: Path,
        relative_directory: str,
        parent_chain: tuple[_DirectoryBinding, ...],
    ) -> None:
        before = os.lstat(directory)
        if _is_symlink(before) or _is_reparse(before):
            raise ValueError(f"source contains a directory link: {directory}")
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError(f"source entry is not a directory: {directory}")
        directory_binding = _metadata_binding(
            before,
            label="source directory",
        )
        directories.append((relative_directory, directory_binding))
        names = _directory_names(directory, relative=relative_directory)
        for name in names:
            relative = (
                name
                if not relative_directory
                else f"{relative_directory}/{name}"
            )
            if _case_identity(relative) == _case_identity(RECEIPT_NAME):
                raise ValueError(
                    f"source reserves the receipt pathname: {RECEIPT_NAME}"
                )
            if relative in seen_paths:
                raise ValueError(f"source contains a duplicate path: {relative}")
            identity = _case_identity(relative)
            if identity in seen_casefold_paths:
                raise ValueError(
                    f"source contains a casefold-alias path: {relative}"
                )
            seen_paths.add(relative)
            seen_casefold_paths.add(identity)
            child = directory / name
            metadata = os.lstat(child)
            _reject_non_symlink_reparse(
                child,
                metadata,
                label="source",
            )
            if stat.S_ISDIR(metadata.st_mode) and not _is_symlink(metadata):
                child_binding = _DirectoryBinding(
                    child,
                    _metadata_binding(metadata, label="source directory"),
                )
                walk(
                    child,
                    relative,
                    parent_chain + (child_binding,),
                )
                continue
            origin_binding = _metadata_binding(metadata, label="source entry")
            if _is_symlink(metadata):
                (
                    link_target,
                    resolved_source,
                    resolved_binding,
                    resolved_parent_chain,
                    resolution_links,
                ) = _resolve_file_link(
                    child,
                    origin_binding,
                    allowed_root=allowed_root,
                    allowed_root_chain=allowed_root_chain,
                )
                entries.append(
                    _SourceEntry(
                        relative_path=relative,
                        origin=child,
                        kind="symlink",
                        link_target=link_target,
                        origin_metadata=origin_binding,
                        origin_parent_chain=parent_chain,
                        resolved_source=resolved_source,
                        resolved_metadata=resolved_binding,
                        resolved_parent_chain=resolved_parent_chain,
                        resolution_links=resolution_links,
                    )
                )
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"source contains a special file: {child}")
            resolved_parent_chain = _chain_to_real_directory_within(
                child.parent,
                allowed_root=allowed_root,
                allowed_root_chain=allowed_root_chain,
                label="regular source parent",
            )
            entries.append(
                _SourceEntry(
                    relative_path=relative,
                    origin=child,
                    kind="regular_file",
                    link_target=None,
                    origin_metadata=origin_binding,
                    origin_parent_chain=parent_chain,
                    resolved_source=child,
                    resolved_metadata=origin_binding,
                    resolved_parent_chain=resolved_parent_chain,
                    resolution_links=(),
                )
            )
        after = os.lstat(directory)
        if (
            _is_symlink(after)
            or _is_reparse(after)
            or not _same_file_binding(directory_binding, after)
        ):
            raise ValueError(f"source directory changed while scanning: {directory}")

    walk(source_snapshot, "", source_chain)
    entries.sort(key=lambda entry: entry.relative_path)
    directories.sort(key=lambda item: item[0])
    return _Inventory(tuple(directories), tuple(entries))


def _entry_signature(entry: _SourceEntry) -> tuple[Any, ...]:
    return (
        entry.relative_path,
        entry.origin,
        entry.kind,
        entry.link_target,
        entry.origin_metadata,
        entry.origin_parent_chain,
        entry.resolved_source,
        entry.resolved_metadata,
        entry.resolved_parent_chain,
        entry.resolution_links,
    )


def _assert_same_inventory(before: _Inventory, after: _Inventory) -> None:
    if before.directories != after.directories:
        raise ValueError("source directory inventory changed before publication")
    if tuple(map(_entry_signature, before.entries)) != tuple(
        map(_entry_signature, after.entries)
    ):
        raise ValueError("source file inventory changed before publication")


def _revalidate_entry(entry: _SourceEntry, *, allowed_root: Path) -> None:
    _revalidate_directory_chain(
        entry.origin_parent_chain,
        label=f"source parent for {entry.relative_path}",
    )
    _exact_component(
        entry.origin.parent,
        entry.origin.name,
        label=f"source path {entry.relative_path}",
    )
    current = os.lstat(entry.origin)
    if not _same_file_binding(entry.origin_metadata, current):
        raise ValueError(f"source entry changed: {entry.relative_path}")
    if entry.kind == "symlink":
        if not _is_symlink(current):
            raise ValueError(f"source link changed: {entry.relative_path}")
        current_target = _read_stable_link(
            entry.origin,
            entry.origin_metadata,
            label="source symlink",
        )
        if current_target != entry.link_target:
            raise ValueError(f"source link target changed: {entry.relative_path}")
    elif _is_symlink(current) or _is_reparse(current):
        raise ValueError(f"regular source became a link: {entry.relative_path}")
    for resolution_link in entry.resolution_links:
        _revalidate_directory_chain(
            resolution_link.parent_chain,
            label=f"link-chain parent for {entry.relative_path}",
        )
        _exact_component(
            resolution_link.path.parent,
            resolution_link.path.name,
            label=f"link chain for {entry.relative_path}",
        )
        link_metadata = os.lstat(resolution_link.path)
        if (
            not _is_symlink(link_metadata)
            or not _same_file_binding(
                resolution_link.metadata,
                link_metadata,
            )
        ):
            raise ValueError(
                f"link chain changed: {entry.relative_path}"
            )
        current_target = _read_stable_link(
            resolution_link.path,
            resolution_link.metadata,
            label="link chain",
        )
        if current_target != resolution_link.target:
            raise ValueError(
                f"link-chain target changed: {entry.relative_path}"
            )
    _revalidate_directory_chain(
        entry.resolved_parent_chain,
        label=f"resolved source parent for {entry.relative_path}",
    )
    if not _path_is_within(entry.resolved_source, allowed_root):
        raise ValueError(f"resolved source escaped allowed root: {entry.relative_path}")
    resolved = os.lstat(entry.resolved_source)
    if _is_symlink(resolved) or _is_reparse(resolved):
        raise ValueError(f"resolved source became a link: {entry.relative_path}")
    if not stat.S_ISREG(resolved.st_mode):
        raise ValueError(f"resolved source is not regular: {entry.relative_path}")
    if not _same_file_binding(entry.resolved_metadata, resolved):
        raise ValueError(f"resolved source changed: {entry.relative_path}")


def _open_source_descriptor(entry: _SourceEntry) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(entry.resolved_source, flags)


def _copy_entry_to_staging(
    entry: _SourceEntry,
    *,
    staging: Path,
    allowed_root: Path,
) -> dict[str, Any]:
    _revalidate_entry(entry, allowed_root=allowed_root)
    destination = staging.joinpath(*entry.relative_path.split("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_descriptor = _open_source_descriptor(entry)
    output_descriptor: int | None = None
    try:
        source_before = os.fstat(source_descriptor)
        if not _same_file_binding(entry.resolved_metadata, source_before):
            raise ValueError(
                f"resolved source changed before reading: {entry.relative_path}"
            )
        output_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
        )
        output_descriptor = os.open(destination, output_flags, 0o644)
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(source_descriptor, _READ_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(output_descriptor, view)
                if written <= 0:
                    raise OSError("short write while materializing snapshot")
                view = view[written:]
        if hasattr(os, "fchmod"):
            os.fchmod(output_descriptor, 0o644)
        os.fsync(output_descriptor)
        output_metadata = os.fstat(output_descriptor)
        if not stat.S_ISREG(output_metadata.st_mode):
            raise ValueError("materialized output is not a regular file")
        if output_metadata.st_size != total:
            raise ValueError("materialized output size changed during write")
        source_after = os.fstat(source_descriptor)
        if not _same_file_binding(entry.resolved_metadata, source_after):
            raise ValueError(
                f"resolved source changed while reading: {entry.relative_path}"
            )
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)
        os.close(source_descriptor)
    _revalidate_entry(entry, allowed_root=allowed_root)
    result: dict[str, Any] = {
        "relative_path": entry.relative_path,
        "kind": entry.kind,
        "resolved_source": str(entry.resolved_source),
        "size": total,
        "sha256": digest.hexdigest(),
    }
    if entry.link_target is not None:
        result["link_target"] = entry.link_target
    return result


def _hash_bound_entry(
    entry: _SourceEntry,
    *,
    allowed_root: Path,
) -> tuple[int, str]:
    _revalidate_entry(entry, allowed_root=allowed_root)
    descriptor = _open_source_descriptor(entry)
    try:
        before = os.fstat(descriptor)
        if not _same_file_binding(entry.resolved_metadata, before):
            raise ValueError(f"source changed before final hash: {entry.relative_path}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        if not _same_file_binding(entry.resolved_metadata, after):
            raise ValueError(f"source changed during final hash: {entry.relative_path}")
    finally:
        os.close(descriptor)
    _revalidate_entry(entry, allowed_root=allowed_root)
    return total, digest.hexdigest()


def _tree_sha256(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["relative_path"]):
        digest.update(entry["relative_path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_READ_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_create_once(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
    )
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating materialization receipt")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_stable_regular_file(path: Path) -> tuple[int, str]:
    parent_chain = _snapshot_real_directory_chain(
        path.parent,
        label="staging file parent",
    )
    _exact_component(path.parent, path.name, label="staging file")
    expected = os.lstat(path)
    if (
        _is_symlink(expected)
        or _is_reparse(expected)
        or not stat.S_ISREG(expected.st_mode)
    ):
        raise ValueError(f"staging contains a non-regular file: {path}")
    binding = _metadata_binding(expected, label="staging file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not _same_file_binding(binding, before):
            raise ValueError(f"staging file changed before reading: {path}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        if not _same_file_binding(binding, after):
            raise ValueError(f"staging file changed while reading: {path}")
    finally:
        os.close(descriptor)
    _revalidate_directory_chain(parent_chain, label="staging file parent")
    rebound = os.lstat(path)
    if not _same_file_binding(binding, rebound):
        raise ValueError(f"staging pathname changed while reading: {path}")
    return total, digest.hexdigest()


def _verify_staging_tree(
    staging: Path,
    *,
    receipt_entries: list[dict[str, Any]],
    receipt_bytes: bytes,
) -> None:
    expected = {
        entry["relative_path"]: (
            int(entry["size"]),
            str(entry["sha256"]),
        )
        for entry in receipt_entries
    }
    expected[RECEIPT_NAME] = (
        len(receipt_bytes),
        hashlib.sha256(receipt_bytes).hexdigest(),
    )
    found: set[str] = set()

    def walk(directory: Path, relative_directory: str) -> None:
        before = os.lstat(directory)
        if (
            _is_symlink(before)
            or _is_reparse(before)
            or not stat.S_ISDIR(before.st_mode)
        ):
            raise ValueError(f"staging contains a linked directory: {directory}")
        binding = _metadata_binding(before, label="staging directory")
        for name in _directory_names(directory, relative=relative_directory):
            relative = (
                name
                if not relative_directory
                else f"{relative_directory}/{name}"
            )
            child = directory / name
            metadata = os.lstat(child)
            if (
                stat.S_ISDIR(metadata.st_mode)
                and not _is_symlink(metadata)
                and not _is_reparse(metadata)
            ):
                walk(child, relative)
                continue
            if relative not in expected:
                raise ValueError(f"staging contains an unexpected file: {relative}")
            size, digest = _hash_stable_regular_file(child)
            if (size, digest) != expected[relative]:
                raise ValueError(f"staging content changed: {relative}")
            found.add(relative)
        after = os.lstat(directory)
        if not _same_file_binding(binding, after):
            raise ValueError(f"staging directory changed while scanning: {directory}")

    walk(staging, "")
    if found != set(expected):
        missing = sorted(set(expected) - found)
        raise ValueError(
            "staging is missing expected files: " + ", ".join(missing)
        )


def _fsync_directory_windows(path: Path) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    create_file.restype = ctypes.c_void_p
    flush_file_buffers = kernel32.FlushFileBuffers
    flush_file_buffers.argtypes = (ctypes.c_void_p,)
    flush_file_buffers.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int

    generic_write = 0x40000000
    share_read_write_delete = 0x00000001 | 0x00000002 | 0x00000004
    open_existing = 3
    backup_semantics = 0x02000000
    handle = create_file(
        str(path),
        generic_write,
        share_read_write_delete,
        None,
        open_existing,
        backup_semantics,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not flush_file_buffers(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        close_handle(handle)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        _fsync_directory_windows(path)
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_staging_tree(staging: Path) -> None:
    directories = [staging]
    for root, child_directories, _ in os.walk(staging, topdown=True):
        root_path = Path(root)
        for name in child_directories:
            directories.append(root_path / name)
    for directory in sorted(
        directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        metadata = os.lstat(directory)
        if _is_symlink(metadata) or _is_reparse(metadata):
            raise ValueError(f"staging contains a directory link: {directory}")
        _fsync_directory(directory)


def _target_exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _publish_directory_create_once(staging: Path, destination: Path) -> None:
    staging = Path(os.path.abspath(staging))
    destination = Path(os.path.abspath(destination))
    destination_parent = destination.parent
    staging_metadata = os.lstat(staging)
    if (
        _is_symlink(staging_metadata)
        or _is_reparse(staging_metadata)
        or not stat.S_ISDIR(staging_metadata.st_mode)
    ):
        raise ValueError("publication staging must be a real directory")
    if staging.parent != destination_parent:
        raise ValueError(
            "publication staging and destination must share one directory"
        )

    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file = kernel32.MoveFileExW
        move_file.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
        )
        move_file.restype = ctypes.c_int
        if move_file(str(staging), str(destination), 0):
            return
        error = ctypes.get_last_error()
        if _target_exists(destination) or error in _WINDOWS_ERROR_FILE_EXISTS:
            raise FileExistsError(
                errno.EEXIST,
                "materialization output already exists",
                str(destination),
            )
        raise ctypes.WinError(error)

    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename_no_replace = libc.renameat2
        except AttributeError as exc:
            raise OSError(
                errno.ENOSYS,
                "renameat2 is required for create-once directory publication",
                str(destination),
            ) from exc
        rename_no_replace.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename_no_replace.restype = ctypes.c_int
        ctypes.set_errno(0)
        if (
            rename_no_replace(
                _AT_FDCWD,
                os.fsencode(staging),
                _AT_FDCWD,
                os.fsencode(destination),
                _RENAME_NOREPLACE,
            )
            == 0
        ):
            return
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error,
                "materialization output already exists",
                str(destination),
            )
        raise OSError(error, os.strerror(error), str(destination))

    raise OSError(
        errno.ENOTSUP,
        "create-once directory publication is unsupported on this platform",
        str(destination),
    )


def _safe_remove_staging(staging: Path, output_parent: Path, prefix: str) -> None:
    if not _target_exists(staging):
        return
    absolute = Path(os.path.abspath(staging))
    if absolute.parent != output_parent or not absolute.name.startswith(prefix):
        raise RuntimeError("refusing to remove an unexpected staging directory")
    metadata = os.lstat(absolute)
    if (
        _is_symlink(metadata)
        or _is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise RuntimeError("refusing to remove a linked staging path")
    shutil.rmtree(absolute)


def materialize_snapshot(
    *,
    source_snapshot: str | os.PathLike[str],
    allowed_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    source = _absolute_path(source_snapshot, label="source snapshot")
    allowed = _absolute_path(allowed_root, label="allowed root")
    output_path = _absolute_path(output, label="materialization output")
    if not output_path.name:
        raise ValueError("materialization output must name a directory")
    _portable_component(output_path.name, label="materialization output")

    output_parent_chain = _snapshot_real_directory_chain(
        output_path.parent,
        label="materialization output parent",
    )
    if _target_exists(output_path):
        raise FileExistsError(
            errno.EEXIST,
            "materialization output already exists",
            str(output_path),
        )
    allowed_chain = _snapshot_real_directory_chain(
        allowed,
        label="allowed root",
    )
    source_chain = _snapshot_real_directory_chain(
        source,
        label="source snapshot",
    )
    if not _path_is_within(source, allowed):
        raise ValueError("source snapshot must be inside allowed root")
    if _path_is_within(output_path, source):
        raise ValueError("materialization output must not be inside source snapshot")

    inventory = _inventory_source(
        source_snapshot=source,
        source_chain=source_chain,
        allowed_root=allowed,
        allowed_root_chain=allowed_chain,
    )
    if not inventory.entries:
        raise ValueError("source snapshot contains no files")

    staging_prefix = f".{output_path.name}.materializing-"
    staging = Path(
        tempfile.mkdtemp(
            dir=output_path.parent,
            prefix=staging_prefix,
        )
    )
    published = False
    preserve_staging = False
    receipt_bytes = b""
    receipt_entries: list[dict[str, Any]] = []
    try:
        for entry in inventory.entries:
            receipt_entries.append(
                _copy_entry_to_staging(
                    entry,
                    staging=staging,
                    allowed_root=allowed,
                )
            )

        rescanned = _inventory_source(
            source_snapshot=source,
            source_chain=source_chain,
            allowed_root=allowed,
            allowed_root_chain=allowed_chain,
        )
        _assert_same_inventory(inventory, rescanned)
        receipt_by_path = {
            entry["relative_path"]: entry for entry in receipt_entries
        }
        for entry in rescanned.entries:
            size, digest = _hash_bound_entry(entry, allowed_root=allowed)
            copied = receipt_by_path[entry.relative_path]
            if size != copied["size"] or digest != copied["sha256"]:
                raise ValueError(
                    f"source content changed before publication: "
                    f"{entry.relative_path}"
                )

        receipt_entries.sort(key=lambda item: item["relative_path"])
        tree_digest = _tree_sha256(receipt_entries)
        script_path = Path(__file__).resolve(strict=True)
        receipt = {
            "schema_version": 1,
            "operation": "huggingface_snapshot_materialization",
            "source_snapshot": str(source),
            "allowed_root": str(allowed),
            "materializer_script_sha256": _read_file_sha256(script_path),
            "content_tree": {
                "algorithm": TREE_HASH_ALGORITHM,
                "sha256": tree_digest,
                "n_files": len(receipt_entries),
                "total_bytes": sum(
                    int(entry["size"]) for entry in receipt_entries
                ),
                "receipt_excluded": True,
            },
            "entries": receipt_entries,
        }
        receipt_bytes = (
            json.dumps(
                receipt,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        _write_create_once(staging / RECEIPT_NAME, receipt_bytes)
        _fsync_staging_tree(staging)
        _fsync_directory(output_path.parent)

        last_scan = _inventory_source(
            source_snapshot=source,
            source_chain=source_chain,
            allowed_root=allowed,
            allowed_root_chain=allowed_chain,
        )
        _assert_same_inventory(inventory, last_scan)
        _verify_staging_tree(
            staging,
            receipt_entries=receipt_entries,
            receipt_bytes=receipt_bytes,
        )
        _revalidate_directory_chain(
            output_parent_chain,
            label="materialization output parent",
        )
        if _target_exists(output_path):
            raise FileExistsError(
                errno.EEXIST,
                "materialization output already exists",
                str(output_path),
            )
        try:
            _publish_directory_create_once(staging, output_path)
        except OSError as exc:
            if exc.errno in _UNSUPPORTED_PUBLICATION_ERRNOS:
                preserve_staging = True
            raise
        published = True
        _fsync_directory(output_path.parent)
    finally:
        if not published and not preserve_staging:
            _safe_remove_staging(
                staging,
                output_path.parent,
                staging_prefix,
            )

    return {
        "output": str(output_path),
        "n_files": len(receipt_entries),
        "total_bytes": sum(int(entry["size"]) for entry in receipt_entries),
        "content_tree_sha256": _tree_sha256(receipt_entries),
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Safely dereference one Hugging Face snapshot into a "
            "create-once regular-file tree"
        )
    )
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = materialize_snapshot(
            source_snapshot=args.source_snapshot,
            allowed_root=args.allowed_root,
            output=args.output,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
