"""Descriptor-bound hashing for immutable ordinary-file trees.

The scanner inventories the complete directory tree before and after a single
descriptor read of every file.  It rejects namespace aliases and non-ordinary
objects, binds every pathname to the descriptor that supplied its size and
SHA-256, and fails closed if any directory or file identity changes.
"""

from __future__ import annotations

import hashlib
import os
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from dki.evidence_io import hash_bound_file


TREE_HASH_ALGORITHM = (
    "sha256 of sorted UTF-8 path NUL decimal-size NUL "
    "lowercase-sha256(file bytes) LF"
)
_REPARSE_POINT_ATTRIBUTE = getattr(
    stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400
)


@dataclass(frozen=True, slots=True)
class _Inventory:
    directories: tuple[tuple[str, tuple[int | None, ...]], ...]
    files: tuple[tuple[str, tuple[int | None, ...]], ...]


def _birthtime_ns(metadata: object) -> int | None:
    value = getattr(metadata, "st_birthtime_ns", None)
    if value is not None:
        return int(value)
    seconds = getattr(metadata, "st_birthtime", None)
    if seconds is None:
        return None
    return int(float(seconds) * 1_000_000_000)


def _is_link_or_reparse(metadata: object) -> bool:
    return stat.S_ISLNK(int(getattr(metadata, "st_mode", 0))) or bool(
        int(getattr(metadata, "st_file_attributes", 0))
        & _REPARSE_POINT_ATTRIBUTE
    )


def _stable_directory(metadata: object) -> bool:
    return (
        stat.S_ISDIR(int(getattr(metadata, "st_mode", 0)))
        and not _is_link_or_reparse(metadata)
        and int(getattr(metadata, "st_ino", 0)) != 0
    )


def _stable_regular(metadata: object) -> bool:
    return (
        stat.S_ISREG(int(getattr(metadata, "st_mode", 0)))
        and not _is_link_or_reparse(metadata)
        and int(getattr(metadata, "st_ino", 0)) != 0
    )


def _directory_identity(metadata: object) -> tuple[int | None, ...]:
    return (
        int(getattr(metadata, "st_dev")),
        int(getattr(metadata, "st_ino")),
        int(getattr(metadata, "st_mode")),
        int(getattr(metadata, "st_nlink")),
        _birthtime_ns(metadata),
    )


def _file_identity(metadata: object) -> tuple[int | None, ...]:
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


def _binding_file_identity(binding: dict[str, Any]) -> tuple[int | None, ...]:
    return (
        int(binding["device"]),
        int(binding["inode"]),
        int(binding["mode"]),
        int(binding["nlink"]),
        int(binding["size"]),
        int(binding["mtime_ns"]),
        int(binding["path_ctime_ns"]),
        (
            None
            if binding["birthtime_ns"] is None
            else int(binding["birthtime_ns"])
        ),
    )


def _case_identity(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _canonical_root(
    root: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    value = os.fspath(root)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} root must be a non-empty text path")
    if "\0" in value:
        raise ValueError(f"{label} root must not contain a NUL byte")
    windows = PureWindowsPath(value)
    if windows.drive and not windows.is_absolute():
        raise ValueError(f"{label} root must not be drive-relative")
    if any(
        part in {".", ".."}
        for part in value.replace("\\", "/").split("/")
    ):
        raise ValueError(f"{label} root must not contain lexical traversal")
    absolute = Path(os.path.abspath(value))
    if not absolute.is_absolute() or not absolute.anchor:
        raise ValueError(f"{label} root must resolve to an absolute path")
    return absolute


def _snapshot_directory_chain(
    directory: Path,
    *,
    label: str,
) -> tuple[tuple[Path, tuple[int | None, ...]], ...]:
    anchor = Path(directory.anchor)
    if not anchor:
        raise ValueError(f"{label} root has no filesystem anchor")
    current = anchor
    snapshots: list[tuple[Path, tuple[int | None, ...]]] = []
    for index, component in enumerate(directory.parts):
        if index:
            current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{label} directory does not exist: {current}"
            ) from exc
        if _is_link_or_reparse(metadata):
            raise ValueError(
                f"{label} path contains a symlink or reparse point: {current}"
            )
        if not _stable_directory(metadata):
            raise ValueError(
                f"{label} path component is not an ordinary directory: "
                f"{current}"
            )
        snapshots.append((current, _directory_identity(metadata)))
    return tuple(snapshots)


def _rebind_directory_chain(
    snapshots: tuple[tuple[Path, tuple[int | None, ...]], ...],
    *,
    label: str,
) -> None:
    for path, expected in snapshots:
        try:
            metadata = os.lstat(path)
        except FileNotFoundError as exc:
            raise ValueError(
                f"{label} directory path changed while the tree was scanned: "
                f"{path}"
            ) from exc
        if (
            not _stable_directory(metadata)
            or _directory_identity(metadata) != expected
        ):
            raise ValueError(
                f"{label} directory path changed while the tree was scanned: "
                f"{path}"
            )


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _directory_entries(
    directory: Path,
    expected: object,
    *,
    label: str,
) -> list[tuple[str, os.stat_result]]:
    """Return one bound, sorted no-follow directory inventory."""

    if os.name != "nt" and hasattr(os, "O_DIRECTORY"):
        descriptor = os.open(directory, _directory_open_flags())
        try:
            before = os.fstat(descriptor)
            if (
                not _stable_directory(before)
                or _directory_identity(before)
                != _directory_identity(expected)
            ):
                raise ValueError(
                    f"{label} directory changed before inventory: {directory}"
                )
            with os.scandir(descriptor) as iterator:
                names = sorted(entry.name for entry in iterator)
            result = [
                (
                    name,
                    os.stat(
                        name,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    ),
                )
                for name in names
            ]
            after = os.fstat(descriptor)
            if (
                not _stable_directory(after)
                or _directory_identity(before)
                != _directory_identity(after)
            ):
                raise ValueError(
                    f"{label} directory changed during inventory: {directory}"
                )
            return result
        finally:
            os.close(descriptor)

    with os.scandir(directory) as iterator:
        names = sorted(entry.name for entry in iterator)
    result = [(name, os.lstat(directory / name)) for name in names]
    after = os.lstat(directory)
    if (
        not _stable_directory(after)
        or _directory_identity(after) != _directory_identity(expected)
    ):
        raise ValueError(
            f"{label} directory changed during inventory: {directory}"
        )
    return result


def _inventory_once(root: Path, *, label: str) -> _Inventory:
    directories: dict[str, tuple[int | None, ...]] = {}
    files: dict[str, tuple[int | None, ...]] = {}
    path_identities: dict[str, str] = {}
    object_identities: dict[tuple[int, int], str] = {}
    pending: list[tuple[Path, str]] = [(root, "")]

    while pending:
        directory, prefix = pending.pop()
        try:
            before = os.lstat(directory)
        except FileNotFoundError as exc:
            raise ValueError(
                f"{label} directory disappeared during inventory: {directory}"
            ) from exc
        if not _stable_directory(before):
            raise ValueError(
                f"{label} contains a non-ordinary directory: {directory}"
            )
        directories[prefix] = _directory_identity(before)
        children = _directory_entries(
            directory,
            before,
            label=label,
        )

        child_names: set[str] = set()
        child_directories: list[tuple[Path, str]] = []
        for name, metadata in children:
            identity = _case_identity(name)
            if identity in child_names:
                raise ValueError(
                    f"{label} contains case-alias names in {directory}"
                )
            child_names.add(identity)
            relative = f"{prefix}/{name}" if prefix else name
            if (
                not relative
                or "\\" in relative
                or any(part in {"", ".", ".."} for part in relative.split("/"))
            ):
                raise ValueError(
                    f"{label} contains a non-canonical path: {relative!r}"
                )
            folded = _case_identity(relative)
            previous_path = path_identities.get(folded)
            if previous_path is not None:
                raise ValueError(
                    f"{label} contains duplicate or case-alias paths: "
                    f"{previous_path}, {relative}"
                )
            path_identities[folded] = relative
            path = directory / name
            if _is_link_or_reparse(metadata):
                raise ValueError(
                    f"{label} contains a symlink or reparse point: {relative}"
                )
            if _stable_directory(metadata):
                child_directories.append((path, relative))
                continue
            if not _stable_regular(metadata):
                raise ValueError(
                    f"{label} contains a non-ordinary file: {relative}"
                )
            if int(getattr(metadata, "st_nlink", 0)) != 1:
                raise ValueError(
                    f"{label} file has multiple hard links: {relative}"
                )
            object_identity = (
                int(getattr(metadata, "st_dev")),
                int(getattr(metadata, "st_ino")),
            )
            previous_object = object_identities.get(object_identity)
            if previous_object is not None:
                raise ValueError(
                    f"{label} files share one filesystem object: "
                    f"{previous_object}, {relative}"
                )
            object_identities[object_identity] = relative
            files[relative] = _file_identity(metadata)
        pending.extend(reversed(child_directories))

        rebound = os.lstat(directory)
        if (
            not _stable_directory(rebound)
            or _directory_identity(rebound) != _directory_identity(before)
        ):
            raise ValueError(
                f"{label} directory changed during inventory: {directory}"
            )

    return _Inventory(
        directories=tuple(sorted(directories.items())),
        files=tuple(sorted(files.items())),
    )


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


def scan_bound_regular_tree(
    root: str | os.PathLike[str],
    *,
    label: str = "evidence tree",
) -> dict[str, Any]:
    """Return a deterministic binding for an exact ordinary-file tree."""

    if not isinstance(label, str) or not label:
        raise ValueError("tree label must be non-empty text")
    canonical = _canonical_root(root, label=label)
    chain = _snapshot_directory_chain(canonical, label=label)
    _rebind_directory_chain(chain, label=label)
    first = _inventory_once(canonical, label=label)
    if not first.files:
        raise ValueError(f"{label} contains no ordinary files")

    first_files = dict(first.files)
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for relative in sorted(first_files):
        path = canonical.joinpath(*relative.split("/"))
        binding = hash_bound_file(
            path,
            f"{label} file {relative}",
            reject_multiple_links=True,
        )
        if _binding_file_identity(binding) != first_files[relative]:
            raise ValueError(
                f"{label} file changed between inventory and hashing: "
                f"{relative}"
            )
        size = int(binding["size"])
        digest = str(binding["sha256"])
        entries.append(
            {
                "path": relative,
                "size": size,
                "sha256": digest,
            }
        )
        total_bytes += size

    second = _inventory_once(canonical, label=label)
    if second != first:
        raise ValueError(
            f"{label} tree changed while it was inventoried and hashed"
        )
    _rebind_directory_chain(chain, label=label)
    return {
        "root": str(canonical),
        "algorithm": TREE_HASH_ALGORITHM,
        "sha256": _tree_sha256(entries),
        "n_files": len(entries),
        "total_bytes": total_bytes,
        "entries": entries,
    }


__all__ = ["TREE_HASH_ALGORITHM", "scan_bound_regular_tree"]
