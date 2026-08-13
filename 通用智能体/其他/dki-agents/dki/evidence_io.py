"""Descriptor-bound reads for immutable evidence inputs.

The public helpers in this module deliberately return both the payload and the
metadata binding produced by the *same* descriptor read.  Callers must persist
that binding instead of reopening the pathname to compute a digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from collections.abc import Callable, Mapping
from pathlib import Path, PureWindowsPath
from typing import Any, TypeAlias


_READ_CHUNK_SIZE = 1024 * 1024
# Some Linux overlay filesystems expose mutable timestamps at a coarse clock
# tick.  A file created and rewritten inside the same tick can therefore
# retain identical fstat metadata.  Delay only very recent evidence until the
# next conservative tick before the descriptor read begins; old immutable
# artifacts pay no delay.
_RECENT_METADATA_FENCE_NS = 10_000_000
_REPARSE_POINT_ATTRIBUTE = getattr(
    stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400
)

BindingValue: TypeAlias = str | int | None
EvidenceBinding: TypeAlias = dict[str, BindingValue]
JsonSchema: TypeAlias = (
    Mapping[str, object] | Callable[[Any], object | None]
)


def _path_text(path: str | os.PathLike[str], *, label: str) -> str:
    value = os.fspath(path)
    if not isinstance(value, str):
        raise ValueError(f"{label} path must be text")
    if not value:
        raise ValueError(f"{label} path must not be empty")
    if "\0" in value:
        raise ValueError(f"{label} path must not contain a NUL byte")
    return value


def _canonical_path(path: str | os.PathLike[str], *, label: str) -> Path:
    raw = _path_text(path, label=label)
    windows_path = PureWindowsPath(raw)
    if windows_path.drive and not windows_path.is_absolute():
        raise ValueError(f"{label} path must not be drive-relative")

    # Normalising a traversal before checking it would erase evidence that the
    # caller supplied an escaping path.  Reject it lexically on both separator
    # styles before constructing the absolute path.
    normalised_separators = raw.replace("\\", "/")
    if ".." in normalised_separators.split("/"):
        raise ValueError(f"{label} path must not contain lexical traversal")

    absolute = Path(os.path.abspath(raw))
    if not absolute.is_absolute() or not absolute.anchor:
        raise ValueError(f"{label} path must resolve to an absolute path")
    return absolute


def _is_link_or_reparse(metadata: object) -> bool:
    return stat.S_ISLNK(getattr(metadata, "st_mode", 0)) or bool(
        getattr(metadata, "st_file_attributes", 0)
        & _REPARSE_POINT_ATTRIBUTE
    )


def _reject_link_or_reparse(
    path: Path, metadata: object, *, label: str
) -> None:
    if _is_link_or_reparse(metadata):
        raise ValueError(
            f"{label} path contains a symlink or reparse component: {path}"
        )


def _birthtime_ns(metadata: object) -> int | None:
    value = getattr(metadata, "st_birthtime_ns", None)
    if value is not None:
        return int(value)
    seconds = getattr(metadata, "st_birthtime", None)
    if seconds is None:
        return None
    return int(float(seconds) * 1_000_000_000)


def _stable_regular(metadata: object) -> bool:
    return (
        stat.S_ISREG(getattr(metadata, "st_mode", 0))
        and not _is_link_or_reparse(metadata)
        and int(getattr(metadata, "st_ino", 0)) != 0
    )


def _stable_directory(metadata: object) -> bool:
    return (
        stat.S_ISDIR(getattr(metadata, "st_mode", 0))
        and not _is_link_or_reparse(metadata)
        and int(getattr(metadata, "st_ino", 0)) != 0
    )


def _directory_identity(metadata: object) -> tuple[int, int, int, int | None]:
    return (
        int(getattr(metadata, "st_dev")),
        int(getattr(metadata, "st_ino")),
        stat.S_IFMT(int(getattr(metadata, "st_mode"))),
        _birthtime_ns(metadata),
    )


def _snapshot_directory_chain(
    parent: Path, *, label: str
) -> tuple[tuple[Path, os.stat_result], ...]:
    anchor = Path(parent.anchor)
    if not anchor:
        raise ValueError(f"{label} path has no filesystem anchor")

    snapshots: list[tuple[Path, os.stat_result]] = []
    current = anchor
    components = parent.parts
    if not components or components[0] != parent.anchor:
        raise ValueError(f"{label} path is not canonical")

    for index, component in enumerate(components):
        if index:
            current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{label} path component does not exist: {current}"
            ) from exc
        _reject_link_or_reparse(current, metadata, label=label)
        if not _stable_directory(metadata):
            raise ValueError(
                f"{label} path component is not a stable directory: {current}"
            )
        snapshots.append((current, metadata))
    return tuple(snapshots)


def _revalidate_directory_chain(
    snapshots: tuple[tuple[Path, os.stat_result], ...], *, label: str
) -> None:
    for path, expected in snapshots:
        try:
            current = os.lstat(path)
        except FileNotFoundError as exc:
            raise ValueError(
                f"{label} path changed while evidence was read: {path}"
            ) from exc
        _reject_link_or_reparse(path, current, label=label)
        if (
            not _stable_directory(current)
            or _directory_identity(current) != _directory_identity(expected)
        ):
            raise ValueError(
                f"{label} path changed while evidence was read: {path}"
            )


def _path_metadata(metadata: object) -> tuple[int | None, ...]:
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


def _descriptor_metadata(metadata: object) -> tuple[int | None, ...]:
    return _path_metadata(metadata)


def _fence_recent_descriptor(
    descriptor: int,
    metadata: os.stat_result,
    *,
    label: str,
) -> os.stat_result:
    if os.name == "nt":
        return metadata

    newest_timestamp = max(
        int(getattr(metadata, "st_mtime_ns")),
        int(getattr(metadata, "st_ctime_ns")),
    )
    age_ns = time.time_ns() - newest_timestamp
    if age_ns >= _RECENT_METADATA_FENCE_NS:
        return metadata

    # Cap the delay for future-dated files instead of trusting wall-clock
    # synchronisation between filesystems.  The second fstat below still binds
    # the exact descriptor and rejects any mutation during the fence.
    delay_ns = min(
        _RECENT_METADATA_FENCE_NS,
        max(0, _RECENT_METADATA_FENCE_NS - age_ns),
    )
    if delay_ns:
        time.sleep(delay_ns / 1_000_000_000)

    fenced = os.fstat(descriptor)
    if (
        not _stable_regular(fenced)
        or _descriptor_metadata(metadata) != _descriptor_metadata(fenced)
    ):
        raise ValueError(f"{label} evidence file changed before reading")
    return fenced


def _path_matches_descriptor(path_stat: object, descriptor_stat: object) -> bool:
    if not _stable_regular(path_stat) or not _stable_regular(descriptor_stat):
        return False
    common_matches = (
        int(getattr(path_stat, "st_dev"))
        == int(getattr(descriptor_stat, "st_dev"))
        and int(getattr(path_stat, "st_ino"))
        == int(getattr(descriptor_stat, "st_ino"))
        and int(getattr(path_stat, "st_mode"))
        == int(getattr(descriptor_stat, "st_mode"))
        and int(getattr(path_stat, "st_nlink"))
        == int(getattr(descriptor_stat, "st_nlink"))
        and int(getattr(path_stat, "st_size"))
        == int(getattr(descriptor_stat, "st_size"))
        and int(getattr(path_stat, "st_mtime_ns"))
        == int(getattr(descriptor_stat, "st_mtime_ns"))
        and _birthtime_ns(path_stat) == _birthtime_ns(descriptor_stat)
    )
    if not common_matches:
        return False

    # Windows exposes different historical meanings for st_ctime through
    # pathname and CRT descriptor APIs.  Each view is still bound and compared
    # before/after below; only this cross-view comparison omits ctime there.
    return os.name == "nt" or int(getattr(path_stat, "st_ctime_ns")) == int(
        getattr(descriptor_stat, "st_ctime_ns")
    )


def _reject_multiple_links(metadata: object, *, label: str) -> None:
    if int(getattr(metadata, "st_nlink")) > 1:
        raise ValueError(f"{label} evidence file has multiple hard links")


def _binding(
    path: Path,
    descriptor_stat: os.stat_result,
    path_stat: os.stat_result,
    digest: str,
) -> EvidenceBinding:
    return {
        "canonical_path": str(path),
        "size": int(descriptor_stat.st_size),
        "sha256": digest,
        "device": int(descriptor_stat.st_dev),
        "inode": int(descriptor_stat.st_ino),
        "mode": int(descriptor_stat.st_mode),
        "nlink": int(descriptor_stat.st_nlink),
        "mtime_ns": int(descriptor_stat.st_mtime_ns),
        "ctime_ns": int(descriptor_stat.st_ctime_ns),
        "birthtime_ns": _birthtime_ns(descriptor_stat),
        "path_ctime_ns": int(path_stat.st_ctime_ns),
    }


def _read_bound_file(
    path: str | os.PathLike[str],
    label: str,
    *,
    reject_multiple_links: bool,
    capture_bytes: bool,
) -> tuple[bytes | None, EvidenceBinding]:
    if not isinstance(label, str) or not label:
        raise ValueError("evidence label must be non-empty text")
    if type(reject_multiple_links) is not bool:
        raise TypeError("reject_multiple_links must be a bool")
    if type(capture_bytes) is not bool:
        raise TypeError("capture_bytes must be a bool")

    canonical = _canonical_path(path, label=label)
    directories = _snapshot_directory_chain(canonical.parent, label=label)
    _revalidate_directory_chain(directories, label=label)

    try:
        expected = os.lstat(canonical)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} evidence file does not exist") from exc
    _reject_link_or_reparse(canonical, expected, label=label)
    if not _stable_regular(expected):
        raise ValueError(f"{label} evidence path must name a regular file")
    if reject_multiple_links:
        _reject_multiple_links(expected, label=label)

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(canonical, flags)
    try:
        before = os.fstat(descriptor)
        if not _path_matches_descriptor(expected, before):
            raise ValueError(f"{label} evidence file changed before reading")
        if reject_multiple_links:
            _reject_multiple_links(before, label=label)
        before = _fence_recent_descriptor(
            descriptor,
            before,
            label=label,
        )

        chunks: list[bytes] | None = [] if capture_bytes else None
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if chunks is not None:
                chunks.append(chunk)

        after = os.fstat(descriptor)
        if (
            not _stable_regular(after)
            or _descriptor_metadata(before) != _descriptor_metadata(after)
        ):
            raise ValueError(f"{label} evidence file changed while reading")
        if reject_multiple_links:
            _reject_multiple_links(after, label=label)
    finally:
        os.close(descriptor)

    if total != int(before.st_size):
        raise ValueError(f"{label} evidence file size changed while reading")

    _revalidate_directory_chain(directories, label=label)
    try:
        rebound = os.lstat(canonical)
    except FileNotFoundError as exc:
        raise ValueError(
            f"{label} evidence pathname changed while reading"
        ) from exc
    _reject_link_or_reparse(canonical, rebound, label=label)
    if (
        not _stable_regular(rebound)
        or _path_metadata(expected) != _path_metadata(rebound)
        or not _path_matches_descriptor(rebound, after)
    ):
        raise ValueError(f"{label} evidence pathname changed while reading")
    if reject_multiple_links:
        _reject_multiple_links(rebound, label=label)

    payload = b"".join(chunks) if chunks is not None else None
    return payload, _binding(
        canonical,
        after,
        rebound,
        digest.hexdigest(),
    )


def read_bound_bytes(
    path: str | os.PathLike[str],
    label: str,
    reject_multiple_links: bool = False,
) -> tuple[bytes, EvidenceBinding]:
    """Read a regular file once and return bytes plus their identity binding.

    Every path component is checked with ``lstat`` before opening.  The open
    descriptor is bound to that pathname, all mutable metadata is checked after
    the read, and the pathname is rebound to the same file object at the end.
    The SHA-256 digest is calculated from the exact bytes returned to the
    caller.
    """

    payload, binding = _read_bound_file(
        path,
        label,
        reject_multiple_links=reject_multiple_links,
        capture_bytes=True,
    )
    if payload is None:
        raise RuntimeError("bound evidence reader did not capture bytes")
    return payload, binding


def hash_bound_file(
    path: str | os.PathLike[str],
    label: str,
    reject_multiple_links: bool = True,
) -> EvidenceBinding:
    """Stream one descriptor-bound file into a SHA-256/size binding.

    Unlike :func:`read_bound_bytes`, this function never retains the file
    payload in memory.  Size and SHA-256 are derived from the same descriptor
    pass, making it suitable for large immutable model weights.
    """

    payload, binding = _read_bound_file(
        path,
        label,
        reject_multiple_links=reject_multiple_links,
        capture_bytes=False,
    )
    if payload is not None:
        raise RuntimeError("bound evidence hasher unexpectedly captured bytes")
    return binding


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains a duplicate key: {key}")
        result[key] = value
    return result


def _validate_schema(
    payload: Any, schema: JsonSchema, *, label: str
) -> None:
    if callable(schema):
        result = schema(payload)
        if result is False:
            raise ValueError(f"{label} JSON does not satisfy its schema")
        return
    if not isinstance(schema, Mapping):
        raise TypeError("schema must be a mapping or callable")
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object for schema validation")

    for key, expectation in schema.items():
        if not isinstance(key, str):
            raise TypeError("schema keys must be strings")
        if key not in payload:
            raise ValueError(f"{label} JSON is missing required key: {key}")
        value = payload[key]
        if expectation is None:
            continue
        if isinstance(expectation, type) or (
            isinstance(expectation, tuple)
            and expectation
            and all(isinstance(item, type) for item in expectation)
        ):
            if not isinstance(value, expectation):
                raise ValueError(
                    f"{label} JSON key has an invalid type: {key}"
                )
            continue
        if callable(expectation):
            if expectation(value) is False:
                raise ValueError(
                    f"{label} JSON key does not satisfy its schema: {key}"
                )
            continue
        raise TypeError(
            f"schema value for {key!r} must be a type, tuple of types, "
            "callable, or None"
        )


def load_bound_json(
    path: str | os.PathLike[str],
    label: str,
    reject_multiple_links: bool = False,
    *,
    require_object: bool = False,
    schema: JsonSchema | None = None,
) -> tuple[Any, EvidenceBinding]:
    """Load strict UTF-8 JSON from one descriptor-bound read.

    ``require_object`` rejects arrays and scalars.  A mapping ``schema`` maps
    required keys to types, tuples of types, value validators, or ``None``.
    A callable schema receives the complete payload and may raise ``ValueError``
    or return ``False`` to reject it.
    """

    if type(require_object) is not bool:
        raise TypeError("require_object must be a bool")
    encoded, binding = read_bound_bytes(
        path,
        label,
        reject_multiple_links=reject_multiple_links,
    )
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} evidence must be UTF-8 JSON") from exc
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} evidence is not valid JSON: {exc}") from exc
    if require_object and not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object")
    if schema is not None:
        _validate_schema(payload, schema, label=label)
    return payload, binding
