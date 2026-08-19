"""Integrity helpers for immutable CMeIE GPLinker epoch bundles.

Each epoch directory is frozen by a create-once ``checkpoint_bundle.json``.
The manifest covers every ordinary file in the directory except the manifest
itself.  Its tree digest is computed over records in this exact form::

    UTF-8 path NUL decimal-size NUL lowercase-sha256 LF

Callers must persist the returned manifest and tree digests in the training
receipt.  Re-verification then detects a changed, missing, or added encoder,
tokenizer, checkpoint, or metadata file before any model deserialisation.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any


CHECKPOINT_BUNDLE_MANIFEST = "checkpoint_bundle.json"
CHECKPOINT_BUNDLE_SCHEMA_VERSION = 1
CHECKPOINT_TREE_HASH_ALGORITHM = (
    "sha256 of sorted UTF-8 path NUL decimal-size NUL "
    "lowercase-file-sha256 LF"
)
_REPARSE_POINT_ATTRIBUTE = 0x400
_SHA256_HEXDIGEST_LENGTH = 64
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "tree_hash_algorithm",
        "tree_sha256",
        "n_files",
        "total_bytes",
        "entries",
    }
)
_ENTRY_KEYS = frozenset({"path", "size", "sha256"})


class CheckpointBundleError(ValueError):
    """The checkpoint bundle cannot be published or fails verification."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_reparse_point(status: os.stat_result) -> bool:
    return bool(
        getattr(status, "st_file_attributes", 0)
        & _REPARSE_POINT_ATTRIBUTE
    )


def _is_hard_link(status: os.stat_result) -> bool:
    return int(getattr(status, "st_nlink", 1)) > 1


def _normalise_root(directory: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(directory)))
    try:
        status = os.stat(root, follow_symlinks=False)
    except OSError as exc:
        raise CheckpointBundleError(
            f"checkpoint bundle directory is unavailable: {root}"
        ) from exc
    if (
        stat.S_ISLNK(status.st_mode)
        or _is_reparse_point(status)
        or not stat.S_ISDIR(status.st_mode)
    ):
        raise CheckpointBundleError(
            "checkpoint bundle root must be a real directory"
        )
    return root


def _validate_relative_path(relative: str) -> str:
    if not isinstance(relative, str) or not relative:
        raise CheckpointBundleError("bundle entry path must be non-empty")
    if "\x00" in relative or "\\" in relative:
        raise CheckpointBundleError(
            f"ambiguous bundle entry path: {relative!r}"
        )
    if unicodedata.normalize("NFC", relative) != relative:
        raise CheckpointBundleError(
            f"bundle entry path is not NFC-normalised: {relative!r}"
        )
    path = PurePosixPath(relative)
    parts = relative.split("/")
    if (
        path.is_absolute()
        or path.as_posix() != relative
        or any(
            not part
            or part in {".", ".."}
            or ":" in part
            or part.endswith((" ", "."))
            or any(ord(character) < 32 for character in part)
            for part in parts
        )
    ):
        raise CheckpointBundleError(
            f"unsafe or ambiguous bundle entry path: {relative!r}"
        )
    return relative


def _casefold_key(relative: str) -> str:
    return unicodedata.normalize("NFC", relative).casefold()


def _same_open_file(
    before: os.stat_result,
    opened: os.stat_result,
) -> bool:
    if before.st_size != opened.st_size:
        return False
    before_inode = getattr(before, "st_ino", 0)
    opened_inode = getattr(opened, "st_ino", 0)
    if before_inode and opened_inode and before_inode != opened_inode:
        return False
    before_device = getattr(before, "st_dev", 0)
    opened_device = getattr(opened, "st_dev", 0)
    if before_device and opened_device and before_device != opened_device:
        return False
    for name in ("st_mtime_ns", "st_ctime_ns"):
        before_value = getattr(before, name, None)
        opened_value = getattr(opened, name, None)
        if (
            before_value is not None
            and opened_value is not None
            and before_value != opened_value
        ):
            return False
    return True


def _hash_regular_file(
    path: Path,
    before: os.stat_result,
) -> tuple[int, str]:
    if _is_hard_link(before):
        raise CheckpointBundleError(
            f"hard-linked file in checkpoint bundle: {path}"
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if _is_hard_link(opened):
                raise CheckpointBundleError(
                    f"hard-linked file in checkpoint bundle: {path}"
                )
            if (
                not stat.S_ISREG(opened.st_mode)
                or _is_reparse_point(opened)
                or not _same_open_file(before, opened)
            ):
                raise CheckpointBundleError(
                    f"bundle file changed before hashing: {path}"
                )
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after_open = os.fstat(handle.fileno())
    except CheckpointBundleError:
        raise
    except OSError as exc:
        raise CheckpointBundleError(
            f"cannot read checkpoint bundle file: {path}"
        ) from exc
    try:
        after_path = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise CheckpointBundleError(
            f"bundle file disappeared while hashing: {path}"
        ) from exc
    if _is_hard_link(after_open) or _is_hard_link(after_path):
        raise CheckpointBundleError(
            f"hard-linked file in checkpoint bundle: {path}"
        )
    if (
        not _same_open_file(opened, after_open)
        or not _same_open_file(opened, after_path)
        or stat.S_ISLNK(after_path.st_mode)
        or _is_reparse_point(after_path)
    ):
        raise CheckpointBundleError(
            f"bundle file changed while hashing: {path}"
        )
    return opened.st_size, digest.hexdigest()


def _tree_digest(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _scan_regular_file_tree(directory: Path) -> dict[str, Any]:
    root = _normalise_root(directory)
    entries: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}

    def visit(path: Path, prefix: tuple[str, ...]) -> None:
        try:
            with os.scandir(path) as iterator:
                children = sorted(
                    iterator,
                    key=lambda item: item.name,
                )
        except OSError as exc:
            raise CheckpointBundleError(
                f"cannot enumerate checkpoint bundle directory: {path}"
            ) from exc
        for child in children:
            relative = _validate_relative_path(
                "/".join((*prefix, child.name))
            )
            if (
                "/" not in relative
                and relative != CHECKPOINT_BUNDLE_MANIFEST
                and _casefold_key(relative)
                == _casefold_key(CHECKPOINT_BUNDLE_MANIFEST)
            ):
                raise CheckpointBundleError(
                    "casefold alias of reserved checkpoint manifest: "
                    f"{relative!r}"
                )
            alias = _casefold_key(relative)
            previous = aliases.get(alias)
            if previous is not None and previous != relative:
                raise CheckpointBundleError(
                    "casefold alias in checkpoint bundle: "
                    f"{previous!r} and {relative!r}"
                )
            aliases[alias] = relative
            try:
                status = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise CheckpointBundleError(
                    f"cannot stat checkpoint bundle entry: {relative}"
                ) from exc
            if (
                child.is_symlink()
                or stat.S_ISLNK(status.st_mode)
                or _is_reparse_point(status)
            ):
                raise CheckpointBundleError(
                    "symlink or reparse point in checkpoint bundle: "
                    f"{relative}"
                )
            if stat.S_ISDIR(status.st_mode):
                visit(Path(child.path), (*prefix, child.name))
                continue
            if not stat.S_ISREG(status.st_mode):
                raise CheckpointBundleError(
                    "non-regular checkpoint bundle entry: "
                    f"{relative}"
                )
            if relative == CHECKPOINT_BUNDLE_MANIFEST:
                continue
            size, file_sha256 = _hash_regular_file(
                Path(child.path),
                status,
            )
            entries.append(
                {
                    "path": relative,
                    "size": size,
                    "sha256": file_sha256,
                }
            )

    visit(root, ())
    entries.sort(key=lambda entry: entry["path"])
    if not entries:
        raise CheckpointBundleError("checkpoint bundle file tree is empty")
    checkpoint_entries = [
        entry
        for entry in entries
        if entry["path"] == "gplinker.pt"
    ]
    if len(checkpoint_entries) != 1:
        raise CheckpointBundleError(
            "checkpoint bundle must contain exactly one root gplinker.pt"
        )
    return {
        "entries": entries,
        "tree_sha256": _tree_digest(entries),
        "n_files": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "checkpoint_sha256": checkpoint_entries[0]["sha256"],
    }


def _manifest_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_BUNDLE_SCHEMA_VERSION,
        "kind": "cmeie_gplinker_checkpoint_bundle",
        "tree_hash_algorithm": CHECKPOINT_TREE_HASH_ALGORITHM,
        "tree_sha256": report["tree_sha256"],
        "n_files": report["n_files"],
        "total_bytes": report["total_bytes"],
        "entries": report["entries"],
    }


def _manifest_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointBundleError(
                f"duplicate JSON key in checkpoint bundle manifest: {key}"
            )
        result[key] = value
    return result


def _manifest_file(directory: Path) -> tuple[Path, bytes]:
    root = _normalise_root(directory)
    path = root / CHECKPOINT_BUNDLE_MANIFEST
    try:
        status = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise CheckpointBundleError(
            f"checkpoint bundle manifest is missing: {path}"
        ) from exc
    if (
        stat.S_ISLNK(status.st_mode)
        or _is_reparse_point(status)
        or _is_hard_link(status)
        or not stat.S_ISREG(status.st_mode)
    ):
        raise CheckpointBundleError(
            "checkpoint bundle manifest must be a unique regular file"
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CheckpointBundleError(
            f"cannot read checkpoint bundle manifest: {path}"
        ) from exc
    return path, raw


def _validate_manifest_payload(
    payload: Any,
    raw: bytes,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_KEYS:
        raise CheckpointBundleError(
            "checkpoint bundle manifest has an invalid field set"
        )
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version")
        != CHECKPOINT_BUNDLE_SCHEMA_VERSION
    ):
        raise CheckpointBundleError(
            "checkpoint bundle manifest schema is unsupported"
        )
    if payload.get("kind") != "cmeie_gplinker_checkpoint_bundle":
        raise CheckpointBundleError(
            "checkpoint bundle manifest kind is invalid"
        )
    if payload.get("tree_hash_algorithm") != CHECKPOINT_TREE_HASH_ALGORITHM:
        raise CheckpointBundleError(
            "checkpoint bundle tree hash algorithm is invalid"
        )
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CheckpointBundleError(
            "checkpoint bundle manifest has no file entries"
        )
    validated: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
            raise CheckpointBundleError(
                "checkpoint bundle manifest entry has invalid fields"
            )
        relative = _validate_relative_path(entry.get("path"))
        if relative == CHECKPOINT_BUNDLE_MANIFEST:
            raise CheckpointBundleError(
                "checkpoint bundle manifest cannot list itself"
            )
        if (
            "/" not in relative
            and _casefold_key(relative)
            == _casefold_key(CHECKPOINT_BUNDLE_MANIFEST)
        ):
            raise CheckpointBundleError(
                "checkpoint bundle manifest lists an alias of itself"
            )
        alias = _casefold_key(relative)
        previous = aliases.get(alias)
        if previous is not None:
            raise CheckpointBundleError(
                "duplicate or casefold-alias checkpoint bundle path: "
                f"{previous!r} and {relative!r}"
            )
        aliases[alias] = relative
        size = entry.get("size")
        file_sha256 = entry.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(file_sha256, str)
            or len(file_sha256) != _SHA256_HEXDIGEST_LENGTH
            or any(
                character not in "0123456789abcdef"
                for character in file_sha256
            )
        ):
            raise CheckpointBundleError(
                f"invalid checkpoint bundle metadata for {relative!r}"
            )
        validated.append(
            {
                "path": relative,
                "size": size,
                "sha256": file_sha256,
            }
        )
    if validated != sorted(validated, key=lambda entry: entry["path"]):
        raise CheckpointBundleError(
            "checkpoint bundle manifest entries are not path-sorted"
        )
    if len({entry["path"] for entry in validated}) != len(validated):
        raise CheckpointBundleError(
            "checkpoint bundle manifest contains duplicate paths"
        )
    if (
        type(payload.get("n_files")) is not int
        or payload.get("n_files") != len(validated)
    ):
        raise CheckpointBundleError(
            "checkpoint bundle manifest file count is invalid"
        )
    total_bytes = sum(entry["size"] for entry in validated)
    if (
        type(payload.get("total_bytes")) is not int
        or payload.get("total_bytes") != total_bytes
    ):
        raise CheckpointBundleError(
            "checkpoint bundle manifest byte count is invalid"
        )
    tree_sha256 = payload.get("tree_sha256")
    if (
        not isinstance(tree_sha256, str)
        or tree_sha256 != _tree_digest(validated)
    ):
        raise CheckpointBundleError(
            "checkpoint bundle manifest tree digest is invalid"
        )
    if not any(
        entry["path"] == "gplinker.pt"
        for entry in validated
    ):
        raise CheckpointBundleError(
            "checkpoint bundle manifest lacks root gplinker.pt"
        )
    if raw != _manifest_bytes(payload):
        raise CheckpointBundleError(
            "checkpoint bundle manifest is not canonical JSON"
        )
    return payload


def _decode_manifest(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except CheckpointBundleError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointBundleError(
            "checkpoint bundle manifest is not valid UTF-8 JSON"
        ) from exc
    return _validate_manifest_payload(payload, raw)


def load_checkpoint_bundle_manifest(directory: Path) -> dict[str, Any]:
    """Load and structurally validate a canonical checkpoint manifest."""

    _, raw = _manifest_file(directory)
    return _decode_manifest(raw)


def _validated_expected_sha256(
    name: str,
    value: str | None,
) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEXDIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CheckpointBundleError(f"{name} is not a lowercase SHA-256")
    return value


def _binding(
    directory: Path,
    *,
    payload: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    checkpoint_entry = next(
        entry
        for entry in payload["entries"]
        if entry["path"] == "gplinker.pt"
    )
    return {
        "checkpoint_bundle_manifest": CHECKPOINT_BUNDLE_MANIFEST,
        "checkpoint_bundle_manifest_sha256": manifest_sha256,
        "checkpoint_bundle_tree_sha256": payload["tree_sha256"],
        "checkpoint_bundle_n_files": payload["n_files"],
        "checkpoint_bundle_total_bytes": payload["total_bytes"],
        "checkpoint_sha256": checkpoint_entry["sha256"],
    }


def publish_checkpoint_bundle_manifest(
    directory: Path,
) -> dict[str, Any]:
    """Freeze ``directory`` with a create-once manifest and return its binding."""

    root = _normalise_root(directory)
    manifest_path = root / CHECKPOINT_BUNDLE_MANIFEST
    if os.path.lexists(manifest_path):
        raise CheckpointBundleError(
            f"checkpoint bundle manifest already exists: {manifest_path}"
        )
    report = _scan_regular_file_tree(root)
    payload = _manifest_payload(report)
    raw = _manifest_bytes(payload)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{CHECKPOINT_BUNDLE_MANIFEST}.",
            dir=root,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, manifest_path)
        except FileExistsError as exc:
            raise CheckpointBundleError(
                "checkpoint bundle manifest publication lost a "
                "create-once race"
            ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return verify_checkpoint_bundle(
        root,
        expected_manifest_sha256=_sha256_bytes(raw),
        expected_tree_sha256=report["tree_sha256"],
        expected_checkpoint_sha256=report["checkpoint_sha256"],
    )


def verify_checkpoint_bundle(
    directory: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_tree_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the manifest binding and the exact current regular-file tree."""

    expected_manifest_sha256 = _validated_expected_sha256(
        "expected manifest SHA-256",
        expected_manifest_sha256,
    )
    expected_tree_sha256 = _validated_expected_sha256(
        "expected tree SHA-256",
        expected_tree_sha256,
    )
    expected_checkpoint_sha256 = _validated_expected_sha256(
        "expected checkpoint SHA-256",
        expected_checkpoint_sha256,
    )
    _, raw = _manifest_file(directory)
    manifest_sha256 = _sha256_bytes(raw)
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256
    ):
        raise CheckpointBundleError(
            "checkpoint bundle manifest SHA-256 mismatch"
        )
    payload = _decode_manifest(raw)
    report = _scan_regular_file_tree(directory)
    expected_entries = {
        entry["path"]: entry
        for entry in payload["entries"]
    }
    actual_entries = {
        entry["path"]: entry
        for entry in report["entries"]
    }
    missing = sorted(expected_entries.keys() - actual_entries.keys())
    unexpected = sorted(actual_entries.keys() - expected_entries.keys())
    changed = sorted(
        path
        for path in expected_entries.keys() & actual_entries.keys()
        if expected_entries[path] != actual_entries[path]
    )
    if missing or unexpected or changed:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        if changed:
            details.append("changed=" + ",".join(changed))
        raise CheckpointBundleError(
            "checkpoint bundle exact-file verification failed: "
            + "; ".join(details)
        )
    if report["tree_sha256"] != payload["tree_sha256"]:
        raise CheckpointBundleError(
            "checkpoint bundle current tree differs from its manifest"
        )
    if (
        expected_tree_sha256 is not None
        and report["tree_sha256"] != expected_tree_sha256
    ):
        raise CheckpointBundleError(
            "checkpoint bundle tree SHA-256 mismatch"
        )
    if (
        expected_checkpoint_sha256 is not None
        and report["checkpoint_sha256"] != expected_checkpoint_sha256
    ):
        raise CheckpointBundleError(
            "checkpoint payload SHA-256 mismatch"
        )
    return _binding(
        Path(directory),
        payload=payload,
        manifest_sha256=manifest_sha256,
    )
