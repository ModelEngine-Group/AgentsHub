"""Fail-closed provenance, snapshot, and publication guards for NL2SQL development.

The evaluator and diagnostic auditor work only from bytes captured by a
verified context.  Receipt authorization comes from a source-controlled anchor,
not a caller-provided digest.  This protects a trusted reviewed distribution
and ACL-controlled output root against substituted caller artifacts; it does
not defend against hostile modification of the local verifier/runtime, which
requires an external signed bootstrap or isolated runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping

from dki.analysis.nl2sql_official_v2_development_anchor import (
    OfficialV2DevelopmentAnchor,
    official_v2_development_anchor_for_receipt_sha256,
    trusted_evidence_binding as serialize_trusted_evidence_binding,
)
from dki.analysis.nl2sql_sqlite import (
    require_sqlite_sidecars_absent,
    sqlite_readonly_uri,
)


_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_UTC_TIMESTAMP = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_FORBIDDEN_PATH_TOKENS = frozenset(
    {"protected", "blind", "final3", "holdout", "v3"}
)
_ELIGIBLE_RECEIPT_KIND = "official_public_v2_four_table_development_receipt"
_PAIRING_PROTOCOL = "nl2sql_development_private_first_v1"
_MAX_PUBLIC_RECEIPT_BYTES = 1 << 20

_SOURCE_INPUT_KEYS = frozenset(
    {
        "CSS/train",
        "CSS/dev",
        "EHRSQL/eicu/train",
        "EHRSQL/eicu/valid",
        "EHRSQL/mimic_iii/train",
        "EHRSQL/mimic_iii/valid",
    }
)
_SOURCE_FILE_KEYS = _SOURCE_INPUT_KEYS
_OFFICIAL_TEST_FILE_KEYS = frozenset(
    {"CSS/test", "EHRSQL/eicu/test", "EHRSQL/mimic_iii/test"}
)
_RECEIPT_DIFFICULTIES = frozenset({"medium", "hard", "extra", "extra-hard"})
_RECEIPT_CATEGORIES = frozenset(
    {
        "filtering",
        "aggregation_grouping",
        "top_k",
        "multi_table_join",
        "temporal",
        "knowledge_graph",
        "having",
        "nested_correlated",
        "cte",
        "set_operation",
        "window",
    }
)
_RECEIPT_BENCHMARKS = frozenset({"CSS", "EHRSQL-v1.5.1"})


@dataclass(frozen=True)
class DevelopmentEvidence:
    """Snapshot-bound material that a development evaluator is allowed to use."""

    input_bytes: bytes
    database_snapshot_path: Path
    rid: str
    input_sha256: str
    database_sha256: str
    evidence_sha256: str
    source_tree_sha256: str
    trusted_evidence: OfficialV2DevelopmentAnchor

    @property
    def database_path(self) -> Path:
        """Compatibility name for the verified temporary snapshot path."""
        return self.database_snapshot_path

    @property
    def producer_source_tree_sha256(self) -> str:
        """Return the historic source tree that produced the anchored receipt."""
        return self.trusted_evidence.producer_source_tree_sha256

    @property
    def trusted_evidence_binding(self) -> dict[str, Any]:
        """Return the static binding private receipts must carry verbatim."""
        return serialize_trusted_evidence_binding(self.trusted_evidence)


class _StrictJsonError(ValueError):
    """Internal marker for JSON syntax that standard ``json`` would accept."""


def sha256_file(path: Path) -> str:
    """Hash a regular, already validated file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_lowercase_sha256(value: Any, label: str) -> str:
    """Return only canonical lower-case SHA-256 text."""
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase 64-hex SHA-256")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJsonError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise _StrictJsonError(f"non-finite JSON constant is forbidden: {value}")


def strict_json_loads(data: bytes | str, *, label: str) -> Any:
    """Decode JSON while rejecting duplicate keys and non-finite constants."""
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label} is not UTF-8 JSON") from exc
    elif isinstance(data, str):
        text = data
    else:
        raise TypeError("strict JSON input must be bytes or text")
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (json.JSONDecodeError, _StrictJsonError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc


def strict_jsonl_objects(
    input_bytes: bytes,
    *,
    label: str = "development input",
) -> list[dict[str, Any]]:
    """Decode JSONL objects without duplicate keys or non-finite constants."""
    try:
        lines = input_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8 JSONL") from exc
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        item = strict_json_loads(line, label=f"{label} item {index}")
        if type(item) is not dict:
            raise ValueError(f"{label} item {index} must be a JSON object")
        rows.append(item)
    return rows


def strict_development_jsonl_rows(
    input_bytes: bytes,
    *,
    label: str = "development input",
) -> list[dict[str, Any]]:
    """Decode evaluator rows and reject implicit field-type coercion.

    Projection rows may include additional provenance fields, so this validates
    the exact primitive types needed by the evaluator without discarding those
    source-controlled extra fields.
    """
    rows = strict_jsonl_objects(input_bytes, label=label)
    required = ("id", "question", "gold_sql", "difficulty", "category")
    for index, row in enumerate(rows, start=1):
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f"{label} item {index} lacks required fields")
        for field in required:
            if type(row[field]) is not str:
                raise ValueError(
                    f"{label} item {index} field {field} must be a string"
                )
    return rows


def _forbidden_token(parts: tuple[str, ...]) -> str | None:
    """Return a protected token found as a path-component token, if any."""
    for component in parts:
        tokens = tuple(
            token
            for token in re.split(r"[^a-z0-9]+", component.casefold())
            if token
        )
        for token in tokens:
            if token in _FORBIDDEN_PATH_TOKENS:
                return token
    return None


def _lexical_rooted_path(
    path: Path,
    root: Path,
    label: str,
) -> tuple[Path, tuple[str, ...]]:
    """Validate root-relative spelling without opening candidate filesystem data."""
    lexical_root = Path(root)
    if not lexical_root.is_absolute():
        lexical_root = lexical_root.absolute()
    forbidden_root_token = _forbidden_token(lexical_root.parts)
    if forbidden_root_token:
        raise ValueError(
            f"{label} root uses forbidden path token: {forbidden_root_token}"
        )

    supplied = Path(path)
    if not supplied.is_absolute():
        if ".." in supplied.parts:
            raise ValueError(f"{label} must not contain parent traversal")
        supplied = lexical_root / supplied
    if ".." in supplied.parts:
        raise ValueError(f"{label} must not contain parent traversal")
    try:
        raw_relative = supplied.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain below the project root") from exc
    forbidden = _forbidden_token(raw_relative.parts)
    if forbidden:
        raise ValueError(f"{label} uses forbidden path token: {forbidden}")
    return supplied, raw_relative.parts


def _rooted_path(path: Path, root: Path, label: str) -> tuple[Path, tuple[str, ...]]:
    """Resolve a root-relative path and reject traversal and symlink ambiguity."""
    _supplied, lexical_parts = _lexical_rooted_path(path, root, label)
    supplied_root = Path(root)
    if not supplied_root.is_absolute():
        supplied_root = supplied_root.absolute()
    if supplied_root.is_symlink():
        raise ValueError(f"{label} root must not be a symlink")
    resolved_root = supplied_root.resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"{label} root is not a regular directory")

    supplied = resolved_root.joinpath(*lexical_parts)
    current = resolved_root
    for part in lexical_parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} must not use a symlink")

    resolved = supplied.resolve(strict=False)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the project root") from exc
    forbidden = _forbidden_token(relative.parts)
    if forbidden:
        raise ValueError(f"{label} uses forbidden path token: {forbidden}")
    return resolved, relative.parts


def _scoped_lexical_path(
    path: Path,
    *,
    root: Path,
    scope: str,
    label: str,
) -> tuple[Path, tuple[str, ...], str]:
    supplied, parts = _lexical_rooted_path(path, root, label)
    expected = ("outputs", scope.casefold())
    observed = tuple(part.casefold() for part in parts[:2])
    if len(parts) < 4 or observed != expected or not parts[2]:
        raise ValueError(
            f"{label} must be below outputs/{scope}/<RID> with a file name"
        )
    return supplied, parts, parts[2]


def _scoped_path(
    path: Path,
    *,
    root: Path,
    scope: str,
    label: str,
) -> tuple[Path, str]:
    resolved, parts = _rooted_path(path, root, label)
    expected = ("outputs", scope.casefold())
    observed = tuple(part.casefold() for part in parts[:2])
    if len(parts) < 4 or observed != expected:
        raise ValueError(
            f"{label} must be below outputs/{scope}/<RID> with a file name"
        )
    return resolved, parts[2]


def private_development_path(path: Path, *, root: Path) -> tuple[Path, str]:
    """Validate an existing or prospective private development artifact path."""
    return _scoped_path(
        path,
        root=root,
        scope="private",
        label="private development path",
    )


def public_development_path(path: Path, *, root: Path) -> tuple[Path, str]:
    """Validate an existing or prospective public development artifact path."""
    return _scoped_path(
        path,
        root=root,
        scope="remote_runs",
        label="public development path",
    )


def _regular_file(path: Path, *, root: Path, label: str) -> Path:
    resolved, _ = _rooted_path(path, root, label)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} must be an existing regular file")
    return resolved


def require_fresh_outputs(*paths: Path) -> None:
    """Reject an existing or dangling-link destination before publication."""
    existing = [
        str(path)
        for path in paths
        if Path(path).exists() or Path(path).is_symlink()
    ]
    if existing:
        raise FileExistsError(
            "development output already exists; choose a new RID: "
            + ", ".join(existing)
        )


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    state = Path(path).stat()
    return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns


def _read_verified_bytes_once(
    path: Path,
    label: str,
    *,
    maximum_bytes: int | None = None,
) -> bytes:
    """Capture a regular file once and reject replacement during that read."""
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be an existing regular file")
        before = _file_identity(path)
        with path.open("rb") as handle:
            if maximum_bytes is None:
                data = handle.read()
            else:
                data = handle.read(maximum_bytes + 1)
                if len(data) > maximum_bytes:
                    raise ValueError(f"{label} exceeds its bounded size")
        after = _file_identity(path)
    except OSError as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if before != after or path.is_symlink():
        raise ValueError(f"{label} changed while being read")
    return data


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    label: str,
) -> None:
    observed = set(value)
    if observed != set(expected):
        unknown = sorted(observed - set(expected))
        missing = sorted(set(expected) - observed)
        detail = []
        if unknown:
            detail.append(f"unknown={unknown}")
        if missing:
            detail.append(f"missing={missing}")
        raise ValueError(f"{label} has an invalid field set ({', '.join(detail)})")


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_file_bindings(
    value: Any,
    *,
    label: str,
    expected_keys: frozenset[str],
) -> None:
    bindings = _require_object(value, label)
    _require_exact_keys(bindings, expected_keys, label)
    for name, binding in bindings.items():
        item = _require_object(binding, f"{label}.{name}")
        _require_exact_keys(item, {"sha256", "size"}, f"{label}.{name}")
        require_lowercase_sha256(item["sha256"], f"{label}.{name}.sha256")
        _require_int(item["size"], f"{label}.{name}.size")


def _validate_count_map(
    value: Any,
    *,
    label: str,
    allowed_keys: frozenset[str],
) -> None:
    counts = _require_object(value, label)
    if set(counts) - set(allowed_keys):
        raise ValueError(f"{label} contains an unapproved category")
    for name, count in counts.items():
        _require_int(count, f"{label}.{name}")


def _validate_aggregate(value: Any, *, label: str, positive: bool) -> None:
    aggregate = _require_object(value, label)
    _require_exact_keys(
        aggregate,
        {
            "n",
            "n_hard",
            "benchmark_counts",
            "difficulty_counts",
            "category_counts",
            "unique_target_ast_skeletons",
        },
        label,
    )
    n = _require_int(aggregate["n"], f"{label}.n", minimum=1 if positive else 0)
    n_hard = _require_int(aggregate["n_hard"], f"{label}.n_hard")
    skeletons = _require_int(
        aggregate["unique_target_ast_skeletons"],
        f"{label}.unique_target_ast_skeletons",
    )
    if n_hard > n or skeletons > n:
        raise ValueError(f"{label} aggregate counts are inconsistent")
    _validate_count_map(
        aggregate["benchmark_counts"],
        label=f"{label}.benchmark_counts",
        allowed_keys=_RECEIPT_BENCHMARKS,
    )
    _validate_count_map(
        aggregate["difficulty_counts"],
        label=f"{label}.difficulty_counts",
        allowed_keys=_RECEIPT_DIFFICULTIES,
    )
    _validate_count_map(
        aggregate["category_counts"],
        label=f"{label}.category_counts",
        allowed_keys=_RECEIPT_CATEGORIES,
    )


def _validate_public_v2_receipt_schema(
    payload: dict[str, Any],
    *,
    anchor: OfficialV2DevelopmentAnchor,
) -> None:
    """Validate the complete aggregate-only receipt schema before private reads."""
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "kind",
            "dataset_role",
            "created_at",
            "source_tree_sha256",
            "partition",
            "hashes",
            "source_attempts",
            "source_files",
            "projection",
            "development_candidates",
            "development_selected",
            "train_valid_holdout_role_capacity_proxy",
            "official_tests",
            "privacy",
        },
        "development evidence",
    )
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("development evidence schema_version is invalid")
    if payload["kind"] != _ELIGIBLE_RECEIPT_KIND:
        raise ValueError("development evidence kind is invalid")
    if payload["dataset_role"] != "development_selection_used":
        raise ValueError("development evidence dataset_role is invalid")
    created_at = _require_string(payload["created_at"], "development evidence created_at")
    if not _UTC_TIMESTAMP.fullmatch(created_at):
        raise ValueError("development evidence created_at must be canonical UTC")
    if (
        require_lowercase_sha256(
            payload["source_tree_sha256"],
            "development evidence source_tree_sha256",
        )
        != anchor.producer_source_tree_sha256
    ):
        raise ValueError("development evidence producer source tree differs from anchor")

    partition = _require_object(payload["partition"], "development evidence partition")
    _require_exact_keys(
        partition,
        {"version", "seed", "development_holdout_skeleton_intersection"},
        "development evidence partition",
    )
    if (
        partition["version"] != "v2"
        or partition["seed"] != "20260726-public-official-test-v2"
        or _require_int(
            partition["development_holdout_skeleton_intersection"],
            "development evidence partition intersection",
        )
        != 0
    ):
        raise ValueError("development evidence partition is invalid")

    hashes = _require_object(payload["hashes"], "development evidence hashes")
    _require_exact_keys(
        hashes,
        {
            "builder_sha256",
            "css_projection_module_sha256",
            "ehrsql_projection_module_sha256",
            "protocol_sha256",
            "css_freeze_manifest_sha256",
            "ehrsql_audit_sha256",
            "target_db_sha256",
            "private_development_sha256",
        },
        "development evidence hashes",
    )
    for name, value in hashes.items():
        require_lowercase_sha256(value, f"development evidence hashes.{name}")
    if (
        hashes["private_development_sha256"]
        != anchor.private_development_sha256
        or hashes["target_db_sha256"] != anchor.target_db_sha256
    ):
        raise ValueError("development evidence data hashes differ from anchor")

    source_attempts = _require_object(
        payload["source_attempts"], "development evidence source_attempts"
    )
    _require_exact_keys(
        source_attempts,
        _SOURCE_INPUT_KEYS,
        "development evidence source_attempts",
    )
    for name, value in source_attempts.items():
        _require_int(value, f"development evidence source_attempts.{name}")
    _validate_file_bindings(
        payload["source_files"],
        label="development evidence source_files",
        expected_keys=_SOURCE_FILE_KEYS,
    )

    projection = _require_object(payload["projection"], "development evidence projection")
    _require_exact_keys(
        projection,
        {
            "n_before_global_deduplication",
            "n_after_global_deduplication",
            "deduplication",
            "rejection_counts",
        },
        "development evidence projection",
    )
    before = _require_int(
        projection["n_before_global_deduplication"],
        "development evidence projection before count",
    )
    after = _require_int(
        projection["n_after_global_deduplication"],
        "development evidence projection after count",
    )
    if after > before:
        raise ValueError("development evidence projection counts are inconsistent")
    deduplication = _require_object(
        projection["deduplication"], "development evidence deduplication"
    )
    _require_exact_keys(
        deduplication,
        {
            "exact_duplicate_rows_removed",
            "ambiguous_questions_removed",
            "ambiguous_rows_removed",
        },
        "development evidence deduplication",
    )
    for name, value in deduplication.items():
        _require_int(value, f"development evidence deduplication.{name}")
    rejection_counts = _require_object(
        projection["rejection_counts"],
        "development evidence projection rejection_counts",
    )
    for name, value in rejection_counts.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"(?:CSS/(?:train|dev)|EHRSQL/(?:eicu|mimic_iii)/(?:train|valid))/[a-z0-9_:-]+",
            name,
        ):
            raise ValueError(
                "development evidence projection rejection_counts has an invalid key"
            )
        _require_int(
            value,
            f"development evidence projection rejection_counts.{name}",
        )

    _validate_aggregate(
        payload["development_candidates"],
        label="development evidence development_candidates",
        positive=False,
    )
    _validate_aggregate(
        payload["development_selected"],
        label="development evidence development_selected",
        positive=True,
    )
    _validate_aggregate(
        payload["train_valid_holdout_role_capacity_proxy"],
        label="development evidence holdout capacity proxy",
        positive=False,
    )

    official_tests = _require_object(
        payload["official_tests"], "development evidence official_tests"
    )
    _require_exact_keys(
        official_tests,
        {"content_parsed_by_this_builder", "consumed", "files"},
        "development evidence official_tests",
    )
    if (
        official_tests["content_parsed_by_this_builder"] is not False
        or official_tests["consumed"] is not False
    ):
        raise ValueError("development evidence official-test privacy is invalid")
    _validate_file_bindings(
        official_tests["files"],
        label="development evidence official_tests.files",
        expected_keys=_OFFICIAL_TEST_FILE_KEYS,
    )

    privacy = _require_object(payload["privacy"], "development evidence privacy")
    _require_exact_keys(
        privacy,
        {
            "public_receipt_contains_questions",
            "public_receipt_contains_gold_sql",
            "public_receipt_contains_error_samples",
            "private_projection_allowed_in_submission",
        },
        "development evidence privacy",
    )
    if any(value is not False for value in privacy.values()):
        raise ValueError("development evidence privacy assertions are invalid")


def _load_eligible_receipt(
    evidence_bytes: bytes,
    *,
    anchor: OfficialV2DevelopmentAnchor,
) -> tuple[dict[str, Any], str, str]:
    payload = strict_json_loads(evidence_bytes, label="development evidence")
    if type(payload) is not dict:
        raise ValueError("development evidence must be a JSON object")
    _validate_public_v2_receipt_schema(payload, anchor=anchor)
    hashes = payload["hashes"]
    return (
        payload,
        hashes["private_development_sha256"],
        hashes["target_db_sha256"],
    )


def _copy_database_snapshot(
    source_path: Path,
    *,
    expected_sha256: str,
    temporary_directory: Path,
) -> Path:
    """Copy one sidecar-free database while detecting a source replacement."""
    require_sqlite_sidecars_absent(source_path)
    # Validate URI spelling before the snapshot is later opened through SQLite.
    # The returned URI is deliberately unused at this raw-byte snapshot stage.
    sqlite_readonly_uri(source_path)
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("development database must be an existing regular file")
    try:
        before = _file_identity(source_path)
        snapshot_path = temporary_directory / "development.sqlite"
        digest = hashlib.sha256()
        with source_path.open("rb") as source, snapshot_path.open("xb") as snapshot:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                snapshot.write(block)
            snapshot.flush()
            os.fsync(snapshot.fileno())
        after = _file_identity(source_path)
    except OSError as exc:
        raise ValueError("development database is unreadable") from exc
    require_sqlite_sidecars_absent(source_path)
    if before != after or source_path.is_symlink():
        raise ValueError("development database changed while being snapshotted")
    copied_sha256 = digest.hexdigest()
    if copied_sha256 != expected_sha256:
        raise ValueError("development database does not match its approved evidence")
    if sha256_file(snapshot_path) != copied_sha256:
        raise ValueError("development database snapshot verification failed")
    return snapshot_path


def _relative_posix(parts: tuple[str, ...]) -> str:
    return PurePosixPath(*parts).as_posix()


def _require_anchor_path(
    *,
    actual_parts: tuple[str, ...],
    expected_relative: str | None,
    label: str,
) -> None:
    if expected_relative is None:
        return
    if _relative_posix(actual_parts) != expected_relative:
        raise ValueError(f"{label} differs from its source-controlled anchor path")


@contextmanager
def verified_evaluation_context(
    *,
    input_path: Path,
    database_path: Path,
    evidence_path: Path,
    source_tree_sha256: str,
    root: Path,
    expected_evidence_sha256: str | None = None,
) -> Iterator[DevelopmentEvidence]:
    """Yield verified private input bytes and a sidecar-free database snapshot.

    The compatibility-only ``expected_evidence_sha256`` parameter is rejected
    whenever supplied.  A caller cannot select receipt authority at runtime;
    only the static source-controlled allowlist authorizes public receipt bytes.
    """
    source_tree_sha256 = require_lowercase_sha256(
        source_tree_sha256,
        "source-tree-sha256",
    )
    if expected_evidence_sha256 is not None:
        raise ValueError(
            "caller-provided expected evidence SHA-256 is not an authorization path"
        )

    # Do lexical validation only until the exact public receipt bytes have been
    # read and looked up in the static anchor.  In particular, no candidate
    # input/database regularity check, read, copy, or hash happens before this.
    input_lexical, input_parts, input_rid = _scoped_lexical_path(
        input_path,
        root=root,
        scope="private",
        label="private development path",
    )
    evidence_lexical, evidence_parts, evidence_rid = _scoped_lexical_path(
        evidence_path,
        root=root,
        scope="remote_runs",
        label="public development path",
    )
    database_lexical, database_parts = _lexical_rooted_path(
        database_path,
        root,
        "development database",
    )

    verified_evidence_path = _regular_file(
        evidence_lexical,
        root=root,
        label="development evidence",
    )
    evidence_bytes = _read_verified_bytes_once(
        verified_evidence_path,
        "development evidence",
        maximum_bytes=_MAX_PUBLIC_RECEIPT_BYTES,
    )
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    anchor = official_v2_development_anchor_for_receipt_sha256(evidence_sha256)

    if evidence_rid != anchor.rid or input_rid != anchor.rid:
        raise ValueError("development artifact RID differs from its source-controlled anchor")
    _require_anchor_path(
        actual_parts=evidence_parts,
        expected_relative=anchor.expanded_evidence_relative_path(),
        label="development evidence path",
    )
    _require_anchor_path(
        actual_parts=input_parts,
        expected_relative=anchor.expanded_input_relative_path(),
        label="development input path",
    )
    _require_anchor_path(
        actual_parts=database_parts,
        expected_relative=anchor.expanded_database_relative_path(),
        label="development database path",
    )
    _, expected_input_sha256, expected_database_sha256 = _load_eligible_receipt(
        evidence_bytes,
        anchor=anchor,
    )

    verified_input_path = _regular_file(
        input_lexical,
        root=root,
        label="development input",
    )
    input_bytes = _read_verified_bytes_once(verified_input_path, "development input")
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    if input_sha256 != expected_input_sha256:
        raise ValueError("development input does not match its approved evidence")

    verified_database_path = _regular_file(
        database_lexical,
        root=root,
        label="development database",
    )
    require_sqlite_sidecars_absent(verified_database_path)
    sqlite_readonly_uri(verified_database_path)
    with tempfile.TemporaryDirectory(prefix="nl2sql-development-db-") as temporary:
        snapshot_path = _copy_database_snapshot(
            verified_database_path,
            expected_sha256=expected_database_sha256,
            temporary_directory=Path(temporary),
        )
        yield DevelopmentEvidence(
            input_bytes=input_bytes,
            database_snapshot_path=snapshot_path,
            rid=anchor.rid,
            input_sha256=input_sha256,
            database_sha256=expected_database_sha256,
            evidence_sha256=evidence_sha256,
            source_tree_sha256=source_tree_sha256,
            trusted_evidence=anchor,
        )


def validate_evaluation_context(
    *,
    input_path: Path,
    database_path: Path,
    evidence_path: Path,
    source_tree_sha256: str,
    root: Path,
    expected_evidence_sha256: str | None = None,
):
    """Backward-compatible, fail-closed name for the context-manager API."""
    return verified_evaluation_context(
        input_path=input_path,
        database_path=database_path,
        evidence_path=evidence_path,
        source_tree_sha256=source_tree_sha256,
        root=root,
        expected_evidence_sha256=expected_evidence_sha256,
    )


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _direct_scoped_output(
    path: Path,
    *,
    root: Path,
    scope: str,
    rid: str,
    label: str,
) -> Path:
    resolved, parts = _rooted_path(path, root, label)
    expected = ("outputs", scope.casefold())
    observed = tuple(part.casefold() for part in parts[:2])
    if (
        not isinstance(rid, str)
        or len(parts) != 4
        or observed != expected
        or parts[2] != rid
        or not parts[3]
    ):
        raise ValueError(
            f"{label} must be directly below outputs/{scope}/{rid}/<file>"
        )
    return resolved


def private_development_output_path(
    path: Path,
    *,
    root: Path,
    rid: str,
) -> Path:
    """Validate a direct private RID output path before evaluation begins."""
    return _direct_scoped_output(
        path,
        root=root,
        scope="private",
        rid=rid,
        label="private development output",
    )


def public_development_output_path(
    path: Path,
    *,
    root: Path,
    rid: str,
) -> Path:
    """Validate a direct public RID output path before evaluation begins."""
    return _direct_scoped_output(
        path,
        root=root,
        scope="remote_runs",
        rid=rid,
        label="public development output",
    )


def _prepare_scoped_output(
    path: Path,
    *,
    root: Path,
    scope: str,
    rid: str,
    label: str,
) -> Path:
    destination = _direct_scoped_output(
        path,
        root=root,
        scope=scope,
        rid=rid,
        label=label,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Directory creation is a TOCTOU boundary; re-resolve and scope-check it
    # immediately before staging/linking a final artifact.
    destination = _direct_scoped_output(
        destination,
        root=root,
        scope=scope,
        rid=rid,
        label=label,
    )
    require_fresh_outputs(destination)
    return destination


def _stage_bytes(path: Path, data: bytes) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _link_scoped_once(
    temporary: Path,
    path: Path,
    *,
    root: Path,
    scope: str,
    rid: str,
    label: str,
    before_link: Callable[[], None] | None = None,
) -> Path:
    destination = _prepare_scoped_output(
        path,
        root=root,
        scope=scope,
        rid=rid,
        label=label,
    )
    if before_link is not None:
        before_link()
    os.link(temporary, destination)
    return destination


def _publish_scoped_json_once(
    *,
    path: Path,
    payload: Mapping[str, Any],
    root: Path,
    scope: str,
    rid: str,
    label: str,
) -> None:
    destination = _prepare_scoped_output(
        path,
        root=root,
        scope=scope,
        rid=rid,
        label=label,
    )
    temporary = _stage_bytes(destination, _json_bytes(payload))
    try:
        _link_scoped_once(
            temporary,
            destination,
            root=root,
            scope=scope,
            rid=rid,
            label=label,
        )
    finally:
        temporary.unlink(missing_ok=True)


def publish_private_development_json_once(
    *,
    path: Path,
    payload: Mapping[str, Any],
    root: Path,
    rid: str,
) -> None:
    """Write one private development artifact in its direct RID namespace."""
    _require_pair_payload_rid(
        payload,
        rid=rid,
        label="private development payload",
    )
    _publish_scoped_json_once(
        path=path,
        payload=payload,
        root=root,
        scope="private",
        rid=rid,
        label="private development output",
    )


def publish_public_development_json_once(
    *,
    path: Path,
    payload: Mapping[str, Any],
    root: Path,
    rid: str,
) -> None:
    """Write one public development artifact in its direct RID namespace."""
    _require_pair_payload_rid(
        payload,
        rid=rid,
        label="public development payload",
    )
    _publish_scoped_json_once(
        path=path,
        payload=payload,
        root=root,
        scope="remote_runs",
        rid=rid,
        label="public development output",
    )


def _public_pair_document(
    *,
    private_bytes: bytes,
    public_body: Mapping[str, Any],
) -> dict[str, Any]:
    if "pairing" in public_body:
        raise ValueError("public development body must not provide pairing metadata")
    document = dict(public_body)
    document["pairing"] = {
        "protocol": _PAIRING_PROTOCOL,
        "private_sha256": hashlib.sha256(private_bytes).hexdigest(),
        "published_from_private": True,
    }
    return document


def _recoverable_public_from_private(
    *,
    private_path: Path,
    public_path: Path,
    root: Path,
    rid: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    private_path = _direct_scoped_output(
        private_path,
        root=root,
        scope="private",
        rid=rid,
        label="private development output",
    )
    public_path = _direct_scoped_output(
        public_path,
        root=root,
        scope="remote_runs",
        rid=rid,
        label="public development output",
    )
    private_bytes = _read_verified_bytes_once(
        private_path,
        "private development artifact",
    )
    private_document = strict_json_loads(
        private_bytes,
        label="private development artifact",
    )
    if type(private_document) is not dict:
        raise ValueError("private development artifact is not recoverable JSON")
    if private_document.get("rid") != rid:
        raise ValueError("private development artifact RID differs from its path")
    pairing = private_document.get("pairing")
    if not isinstance(pairing, Mapping) or pairing.get("protocol") != _PAIRING_PROTOCOL:
        raise ValueError("private development artifact lacks recoverable pair metadata")
    expected_relative = public_path.relative_to(Path(root).resolve()).as_posix()
    if pairing.get("public_relative_path") != expected_relative:
        raise ValueError("private development artifact names a different public output")
    public_body = pairing.get("public_body")
    if not isinstance(public_body, Mapping):
        raise ValueError("private development artifact lacks a recoverable public body")
    if public_body.get("rid") != rid:
        raise ValueError("public development body RID differs from its path")
    return (
        _public_pair_document(
            private_bytes=private_bytes,
            public_body=public_body,
        ),
        private_bytes,
        private_document,
    )


def _require_pair_identity(
    private_document: Mapping[str, Any],
    expected_identity: Mapping[str, Any] | None,
) -> None:
    if expected_identity is None:
        return
    for field, expected in expected_identity.items():
        if private_document.get(field) != expected:
            raise ValueError(
                f"private development artifact does not match current {field}"
            )


def _require_private_bytes_unchanged(
    private_path: Path,
    expected_bytes: bytes,
) -> None:
    actual = _read_verified_bytes_once(private_path, "private development artifact")
    if actual != expected_bytes:
        raise ValueError("private development artifact changed during pair publication")


def recover_development_json_pair(
    *,
    private_path: Path,
    public_path: Path,
    root: Path,
    rid: str,
    expected_identity: Mapping[str, Any] | None = None,
) -> bool:
    """Finish or verify a private-first pair without replacing either artifact.

    A surviving private half embeds the fixed public body and output-relative
    path.  Before and after a public link, the private bytes are re-read and
    compared exactly.  Any scope, RID, or hash mismatch is retained for audit
    and rejected; this recovery path never deletes an artifact.
    """
    private_destination = _direct_scoped_output(
        private_path,
        root=root,
        scope="private",
        rid=rid,
        label="private development output",
    )
    public_destination = _direct_scoped_output(
        public_path,
        root=root,
        scope="remote_runs",
        rid=rid,
        label="public development output",
    )
    private_exists = private_destination.exists() or private_destination.is_symlink()
    public_exists = public_destination.exists() or public_destination.is_symlink()
    if not private_exists:
        if public_exists:
            raise ValueError("public development output has no private pair")
        return False
    if private_destination.is_symlink() or not private_destination.is_file():
        raise ValueError("private development output must be an existing regular file")

    public_document, private_bytes, private_document = _recoverable_public_from_private(
        private_path=private_destination,
        public_path=public_destination,
        root=root,
        rid=rid,
    )
    _require_pair_identity(private_document, expected_identity)
    public_bytes = _json_bytes(public_document)
    if public_exists:
        if public_destination.is_symlink() or not public_destination.is_file():
            raise ValueError("public development output must be an existing regular file")
        _require_private_bytes_unchanged(private_destination, private_bytes)
        actual_public = _read_verified_bytes_once(
            public_destination,
            "public development artifact",
        )
        _require_private_bytes_unchanged(private_destination, private_bytes)
        if actual_public != public_bytes:
            raise ValueError("public development output does not match its private pair")
        return True

    temporary_destination = _prepare_scoped_output(
        public_destination,
        root=root,
        scope="remote_runs",
        rid=rid,
        label="public development output",
    )
    temporary = _stage_bytes(temporary_destination, public_bytes)
    try:
        _link_scoped_once(
            temporary,
            temporary_destination,
            root=root,
            scope="remote_runs",
            rid=rid,
            label="public development output",
            before_link=lambda: _require_private_bytes_unchanged(
                private_destination,
                private_bytes,
            ),
        )
    finally:
        temporary.unlink(missing_ok=True)
    _require_private_bytes_unchanged(private_destination, private_bytes)
    actual_public = _read_verified_bytes_once(
        public_destination,
        "public development artifact",
    )
    if actual_public != public_bytes:
        raise ValueError("public development output does not match its private pair")
    return True


def _require_pair_payload_rid(
    payload: Mapping[str, Any],
    *,
    rid: str,
    label: str,
) -> None:
    if payload.get("rid") != rid:
        raise ValueError(f"{label} RID must equal its direct output RID")


def publish_development_json_pair_once(
    *,
    private_path: Path,
    private_payload: Mapping[str, Any],
    public_path: Path,
    public_payload: Mapping[str, Any],
    root: Path,
    rid: str,
) -> None:
    """Publish a recoverable private-first pair without overwrite fallback."""
    private_destination = _prepare_scoped_output(
        private_path,
        root=root,
        scope="private",
        rid=rid,
        label="private development output",
    )
    public_destination = _prepare_scoped_output(
        public_path,
        root=root,
        scope="remote_runs",
        rid=rid,
        label="public development output",
    )
    if "pairing" in private_payload or "pairing" in public_payload:
        raise ValueError("development pair payloads must not provide pairing metadata")
    _require_pair_payload_rid(private_payload, rid=rid, label="private development payload")
    _require_pair_payload_rid(public_payload, rid=rid, label="public development payload")
    private_document = dict(private_payload)
    private_document["pairing"] = {
        "protocol": _PAIRING_PROTOCOL,
        "public_relative_path": public_destination.relative_to(
            Path(root).resolve()
        ).as_posix(),
        "public_body": dict(public_payload),
    }
    private_temporary = _stage_bytes(
        private_destination,
        _json_bytes(private_document),
    )
    try:
        _link_scoped_once(
            private_temporary,
            private_destination,
            root=root,
            scope="private",
            rid=rid,
            label="private development output",
        )
    finally:
        private_temporary.unlink(missing_ok=True)

    # Never delete a linked private artifact if public publication is interrupted.
    # Its embedded public body is the durable recovery record.
    recover_development_json_pair(
        private_path=private_destination,
        public_path=public_destination,
        root=root,
        rid=rid,
    )
