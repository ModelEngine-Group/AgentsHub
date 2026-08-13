"""Strict, shared validation for the frozen CMeIE split-manifest protocol.

The split manifest is an executable data-selection boundary: consumers must
not trust any index until the complete v6 payload has been recomputed from the
bound training rows.  This module deliberately depends only on the standard
library so the builder, NPU smoke preflight, and formal trainer all enforce the
same fail-closed contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from dki.evidence_io import (
    EvidenceBinding,
    load_bound_json,
    read_bound_bytes,
)
from dki.nlp.cmeie_run_protocol import (
    RunBinding,
    canonical_run_paths,
    parse_rid,
)

SCHEMA_VERSION = 6
FIXED_SEED = "20260725"
SPLIT_NAMES = ("fit", "calibration", "outer")
SPLIT_RATIOS = {
    "fit": 0.80,
    "calibration": 0.10,
    "outer": 0.10,
}
MANIFEST_FILENAME = "split_manifest.json"
SHARD_FILENAMES = {
    "fit": "fit.jsonl",
    "calibration": "calibration.jsonl",
    "outer": "outer.jsonl",
}
GROUPING_POLICY = (
    "title-before-@ else normalized-leading-32; "
    "normalized duplicates co-grouped"
)
PREDICATE_COVERAGE_POLICY = {
    "minimum_occurrences": 10,
    "minimum_families": 3,
    "required_in_every_split": True,
}
INTERSECTION_PAIRS = (
    ("fit", "calibration"),
    ("fit", "outer"),
    ("calibration", "outer"),
)
INTERSECTION_KEYS = tuple(
    f"{left}:{right}" for left, right in INTERSECTION_PAIRS
)
ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "rid",
        "git_head",
        "canonical_run_root",
        "seed",
        "source_tree_sha256",
        "source_tree_manifest_sha256",
        "source_train",
        "source_train_sha256",
        "split_ratios",
        "grouping",
        "predicate_coverage_policy",
        "splits",
        "intersections",
        "generator_sha256",
    }
)
SPLIT_FIELDS = frozenset(
    {
        "indices",
        "n_rows",
        "target_ratio",
        "realized_ratio",
        "absolute_ratio_error",
        "row_sha256",
        "family_sha256",
        "normalized_text_sha256",
        "predicate_counts",
        "shard",
    }
)
SHARD_FIELDS = frozenset(
    {
        "relative_path",
        "n_rows",
        "size_bytes",
        "sha256",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
_GIT_HEAD_PATTERN = re.compile(r"[0-9a-f]{40}")
BoundSourceSnapshot = tuple[
    list[dict[str, Any]],
    list[bytes],
    bytes,
    EvidenceBinding,
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalise(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"[\s\W_]+", "", value, flags=re.UNICODE).lower()


def family_key(text: str) -> str:
    """Return the frozen title/prefix family key for one source text."""
    title, separator, _ = str(text).partition("@")
    basis = title if separator else str(text)[:32]
    key = normalise(basis)
    return key or sha256_bytes(str(text).encode("utf-8"))[:32]


def row_hash(row: dict[str, Any]) -> str:
    return sha256_bytes(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def hash_list(values: list[str] | set[str]) -> str:
    return sha256_bytes("\n".join(sorted(values)).encode("utf-8"))


def family_assignments(rows: list[dict[str, Any]]) -> dict[int, str]:
    """Return the frozen coalesced family owner for every source row."""
    normalized_owner: dict[str, str] = {}
    near_owner: dict[str, str] = {}
    assignments: dict[int, str] = {}
    for index, row in enumerate(rows):
        text_key = normalise(row.get("text", ""))
        key = family_key(row.get("text", ""))
        if text_key in normalized_owner:
            key = normalized_owner[text_key]
        else:
            normalized_owner[text_key] = key
        near_key = text_key[:16]
        if near_key and near_key in near_owner:
            key = near_owner[near_key]
        elif near_key:
            near_owner[near_key] = key
        assignments[index] = key
    return assignments


def _predicate_counts(
    rows: list[dict[str, Any]],
    indices: list[int],
) -> Counter[str]:
    return Counter(
        str(spo.get("predicate"))
        for index in indices
        for spo in rows[index].get("spo_list", [])
        if spo.get("predicate")
    )


def sufficiently_supported_predicates(
    rows: list[dict[str, Any]],
) -> set[str]:
    assignments = family_assignments(rows)
    occurrence_counts = _predicate_counts(rows, list(range(len(rows))))
    predicate_families: dict[str, set[str]] = {}
    for index, row in enumerate(rows):
        family = assignments[index]
        for spo in row.get("spo_list", []):
            predicate = spo.get("predicate")
            if predicate:
                predicate_families.setdefault(str(predicate), set()).add(
                    family
                )
    return {
        predicate
        for predicate, count in occurrence_counts.items()
        if (
            count
            >= PREDICATE_COVERAGE_POLICY["minimum_occurrences"]
            and len(predicate_families.get(predicate, set()))
            >= PREDICATE_COVERAGE_POLICY["minimum_families"]
        )
    }


def compute_manifest_evidence(
    rows: list[dict[str, Any]],
    split: dict[str, list[int]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], set[str]]:
    """Recompute every split statistic and leakage intersection."""
    assignments = family_assignments(rows)
    text_hashes = {
        index: sha256_bytes(
            normalise(row.get("text", "")).encode("utf-8")
        )
        for index, row in enumerate(rows)
    }
    split_evidence: dict[str, dict[str, Any]] = {}
    family_sets: dict[str, set[str]] = {}
    text_sets: dict[str, set[str]] = {}
    for name in SPLIT_NAMES:
        indices = split[name]
        families = {assignments[index] for index in indices}
        texts = {text_hashes[index] for index in indices}
        family_sets[name] = families
        text_sets[name] = texts
        realized_ratio = len(indices) / max(1, len(rows))
        split_evidence[name] = {
            "indices": indices,
            "n_rows": len(indices),
            "target_ratio": SPLIT_RATIOS[name],
            "realized_ratio": realized_ratio,
            "absolute_ratio_error": abs(
                realized_ratio - SPLIT_RATIOS[name]
            ),
            "row_sha256": hash_list(
                [row_hash(rows[index]) for index in indices]
            ),
            "family_sha256": hash_list(families),
            "normalized_text_sha256": hash_list(texts),
            "predicate_counts": dict(
                sorted(_predicate_counts(rows, indices).items())
            ),
        }
    intersections = {
        "families": {
            f"{left}:{right}": sorted(
                family_sets[left] & family_sets[right]
            )
            for left, right in INTERSECTION_PAIRS
        },
        "normalized_text_hashes": {
            f"{left}:{right}": sorted(
                text_sets[left] & text_sets[right]
            )
            for left, right in INTERSECTION_PAIRS
        },
    }
    return (
        split_evidence,
        intersections,
        sufficiently_supported_predicates(rows),
    )


def build_shard_bytes(
    source_row_bytes: list[bytes],
    split: dict[str, list[int]],
) -> dict[str, bytes]:
    """Select exact source JSONL line bytes in original-index order."""
    return {
        role: b"".join(
            source_row_bytes[index] for index in split[role]
        )
        for role in SPLIT_NAMES
    }


def shard_evidence(role: str, raw: bytes, n_rows: int) -> dict[str, Any]:
    if role not in SPLIT_NAMES:
        raise ValueError(f"unknown CMeIE split role: {role!r}")
    return {
        "relative_path": SHARD_FILENAMES[role],
        "n_rows": n_rows,
        "size_bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _validate_json_value(value: Any, *, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            _validate_json_value(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")


def parse_cmeie_jsonl_bytes(
    raw: bytes,
    *,
    description: str,
) -> tuple[list[dict[str, Any]], list[bytes]]:
    """Parse strict JSONL while preserving every accepted line byte-for-byte."""
    rows: list[dict[str, Any]] = []
    row_bytes: list[bytes] = []
    for line_number, raw_line in enumerate(
        raw.splitlines(keepends=True),
        start=1,
    ):
        if not raw_line.strip():
            raise ValueError(
                f"{description} line {line_number} must not be blank"
            )
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{description} line {line_number} must be UTF-8"
            ) from exc
        try:
            row = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"invalid {description} row at line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"{description} row {line_number} must be an object"
            )
        _validate_json_value(row, path=f"rows[{len(rows)}]")
        rows.append(row)
        row_bytes.append(raw_line)
    return rows, row_bytes


def load_cmeie_source_bound(path: Path) -> BoundSourceSnapshot:
    """Descriptor-bind one source JSONL read, its digest, and parsed rows."""
    raw, binding = read_bound_bytes(
        path,
        "CMeIE source JSONL",
        reject_multiple_links=True,
    )
    rows, row_bytes = parse_cmeie_jsonl_bytes(
        raw,
        description="CMeIE JSONL",
    )
    return rows, row_bytes, raw, binding


def load_cmeie_source(
    path: Path,
) -> tuple[list[dict[str, Any]], list[bytes], bytes]:
    """Load the source JSONL and retain exact non-empty line bytes."""
    rows, row_bytes, raw, _ = load_cmeie_source_bound(path)
    return rows, row_bytes, raw


def load_cmeie_rows(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows with duplicate-key and non-finite-number rejection."""
    rows, _, _ = load_cmeie_source(path)
    return rows


def load_split_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Load one split manifest strictly and bind its exact source bytes."""
    payload, binding = load_bound_json(
        path,
        "CMeIE split manifest",
        reject_multiple_links=True,
        require_object=True,
    )
    _validate_json_value(payload, path="manifest")
    digest = binding["sha256"]
    if not isinstance(digest, str):
        raise ValueError("CMeIE split manifest binding lacks SHA-256")
    return payload, digest


def _absolute_supplied_path(path: Path, *, label: str) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\0" in raw:
        raise ValueError(f"{label} path must be non-empty text")
    if ".." in raw.replace("\\", "/").split("/"):
        raise ValueError(f"{label} path must not contain lexical traversal")
    return Path(os.path.abspath(raw))


def _validate_manifest_location(
    manifest_path: Path,
    *,
    expected_run_binding: RunBinding,
    allow_same_parent_staging: bool,
) -> Path:
    """Check the RID-derived manifest pathname before any file is opened."""
    if type(expected_run_binding) is not RunBinding:
        raise TypeError("expected_run_binding must be a RunBinding")
    if type(allow_same_parent_staging) is not bool:
        raise TypeError("allow_same_parent_staging must be a bool")
    actual_manifest = _absolute_supplied_path(
        manifest_path,
        label="split manifest",
    )
    canonical_manifest = (
        canonical_run_paths(expected_run_binding).split_dir
        / MANIFEST_FILENAME
    )
    canonical_split_dir = canonical_manifest.parent
    staging_dir = actual_manifest.parent
    staging_prefix = f".{canonical_split_dir.name}."
    staging_suffix = ".staging"
    staging_name = staging_dir.name
    staging_token = ""
    if (
        staging_name.startswith(staging_prefix)
        and staging_name.endswith(staging_suffix)
    ):
        staging_token = staging_name[
            len(staging_prefix):-len(staging_suffix)
        ]
    is_expected_staging = (
        allow_same_parent_staging
        and staging_dir.parent == canonical_split_dir.parent
        and actual_manifest.name == MANIFEST_FILENAME
        and bool(staging_token)
        and re.fullmatch(r"[A-Za-z0-9_-]+", staging_token) is not None
    )
    if actual_manifest != canonical_manifest and not is_expected_staging:
        raise ValueError(
            "split manifest path differs from the expected RID "
            "canonical split directory"
        )
    return actual_manifest


def _expect_exact_keys(
    value: Any,
    expected: set[str] | frozenset[str],
    *,
    path: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"{path} fields differ from the v6 schema; "
            f"missing={missing}, extra={extra}"
        )
    return value


def _normalise_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be exactly 64 hexadecimal characters")
    return value.lower()


def _expect_finite_number(
    value: Any,
    *,
    field: str,
) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a non-boolean finite number")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _validate_run_identity(
    manifest: dict[str, Any],
    *,
    expected_run_binding: RunBinding | None,
) -> dict[str, str]:
    rid = manifest["rid"]
    if type(rid) is not str:
        raise ValueError("manifest.rid must be canonical text")
    parts = parse_rid(rid)
    git_head = manifest["git_head"]
    if (
        type(git_head) is not str
        or _GIT_HEAD_PATTERN.fullmatch(git_head) is None
        or not git_head.startswith(parts.head_prefix)
    ):
        raise ValueError("manifest.git_head differs from its RID")
    source_tree = _normalise_sha256(
        manifest["source_tree_sha256"],
        field="source_tree_sha256",
    )
    if not source_tree.startswith(parts.tree_prefix):
        raise ValueError("manifest.source_tree_sha256 differs from its RID")
    canonical_root = manifest["canonical_run_root"]
    if type(canonical_root) is not str or not canonical_root:
        raise ValueError("manifest.canonical_run_root must be absolute text")
    root = Path(canonical_root)
    if (
        not root.is_absolute()
        or str(Path(os.path.abspath(canonical_root))) != canonical_root
        or root.name != f"dki-agents-{rid}"
    ):
        raise ValueError(
            "manifest.canonical_run_root is not canonical for its RID"
        )
    source_manifest_sha256 = _normalise_sha256(
        manifest["source_tree_manifest_sha256"],
        field="source_tree_manifest_sha256",
    )
    identity = {
        "rid": rid,
        "git_head": git_head,
        "canonical_run_root": canonical_root,
        "source_tree_sha256": source_tree,
        "source_tree_manifest_sha256": source_manifest_sha256,
    }
    if expected_run_binding is not None:
        if type(expected_run_binding) is not RunBinding:
            raise TypeError("expected_run_binding must be a RunBinding")
        expected = {
            "rid": expected_run_binding.rid,
            "git_head": expected_run_binding.git_head,
            "canonical_run_root": str(expected_run_binding.run_root),
            "source_tree_sha256": (
                expected_run_binding.included_tree_sha256
            ),
            "source_tree_manifest_sha256": (
                expected_run_binding.source_manifest_sha256
            ),
        }
        if identity != expected:
            differing = sorted(
                field
                for field in identity
                if identity[field] != expected[field]
            )
            raise ValueError(
                "split manifest run binding differs from the expected RID: "
                f"{differing}"
            )
    return identity


def _validate_indices(
    manifest: dict[str, Any],
    n_rows: int,
) -> dict[str, list[int]]:
    splits = _expect_exact_keys(
        manifest["splits"],
        set(SPLIT_NAMES),
        path="manifest.splits",
    )
    validated: dict[str, list[int]] = {}
    owner: dict[int, str] = {}
    for name in SPLIT_NAMES:
        evidence = _expect_exact_keys(
            splits[name],
            SPLIT_FIELDS,
            path=f"manifest.splits.{name}",
        )
        indices = evidence["indices"]
        if not isinstance(indices, list):
            raise ValueError(
                f"manifest.splits.{name}.indices must be an array"
            )
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in indices
        ):
            raise ValueError(
                f"manifest.splits.{name}.indices must contain only "
                "non-boolean integers"
            )
        if any(index < 0 or index >= n_rows for index in indices):
            raise ValueError(
                f"manifest.splits.{name}.indices contain an out-of-range index"
            )
        if indices != sorted(indices):
            raise ValueError(
                f"manifest.splits.{name}.indices must be sorted"
            )
        if len(indices) != len(set(indices)):
            raise ValueError(
                f"manifest.splits.{name}.indices must be unique"
            )
        for index in indices:
            if index in owner:
                raise ValueError(
                    "split indices overlap: "
                    f"row {index} belongs to {owner[index]} and {name}"
                )
            owner[index] = name
        validated[name] = indices
    expected = set(range(n_rows))
    actual = set(owner)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "split indices must exactly cover every source row; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return validated


def validate_split_manifest(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    train_path: Path,
    expected_source_tree_sha256: str,
    expected_source_tree_manifest_sha256: str,
    generator_path: Path,
    split_manifest_sha256: str | None = None,
    expected_run_binding: RunBinding | None = None,
    _bound_source: BoundSourceSnapshot | None = None,
) -> dict[str, Any]:
    """Fully validate a v6 manifest against the original source JSONL.

    The returned object deliberately omits raw indices and corpus content.  A
    consumer may use ``manifest`` only after this function returns.
    """
    _validate_json_value(manifest, path="manifest")
    _expect_exact_keys(manifest, ROOT_FIELDS, path="manifest")
    if (
        isinstance(manifest["schema_version"], bool)
        or not isinstance(manifest["schema_version"], int)
        or manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("split manifest schema is not v6")
    if manifest["seed"] != FIXED_SEED:
        raise ValueError(
            f"split manifest seed must be exactly {FIXED_SEED!r}"
        )
    run_identity = _validate_run_identity(
        manifest,
        expected_run_binding=expected_run_binding,
    )
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("CMeIE split validation requires at least three rows")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"rows[{index}] must be an object")
        _validate_json_value(row, path=f"rows[{index}]")
    actual_train_path = _absolute_supplied_path(
        train_path,
        label="CMeIE source JSONL",
    )
    if manifest["source_train"] != str(actual_train_path):
        raise ValueError(
            "split manifest source_train differs from the current train file"
        )
    if _bound_source is None:
        source_snapshot = load_cmeie_source_bound(train_path)
    else:
        source_snapshot = _bound_source
    file_rows, source_row_bytes, source_raw, source_binding = source_snapshot
    if source_binding.get("canonical_path") != str(actual_train_path):
        raise ValueError(
            "bound CMeIE source path differs from the current train file"
        )
    if (
        source_binding.get("size") != len(source_raw)
        or source_binding.get("sha256") != sha256_bytes(source_raw)
    ):
        raise ValueError("bound CMeIE source bytes differ from their binding")
    if (
        len(file_rows) != len(rows)
        or [row_hash(row) for row in file_rows]
        != [row_hash(row) for row in rows]
    ):
        raise ValueError(
            "rows supplied to the split validator differ from the current "
            "train file"
        )

    train_sha256 = str(source_binding["sha256"])
    actual_train_sha256 = _normalise_sha256(
        manifest["source_train_sha256"],
        field="source_train_sha256",
    )
    if actual_train_sha256 != train_sha256:
        raise ValueError("train hash differs from frozen split manifest")

    expected_tree = _normalise_sha256(
        expected_source_tree_sha256,
        field="expected_source_tree_sha256",
    )
    expected_tree_manifest = _normalise_sha256(
        expected_source_tree_manifest_sha256,
        field="expected_source_tree_manifest_sha256",
    )
    actual_tree = _normalise_sha256(
        manifest["source_tree_sha256"],
        field="source_tree_sha256",
    )
    actual_tree_manifest = _normalise_sha256(
        manifest["source_tree_manifest_sha256"],
        field="source_tree_manifest_sha256",
    )
    if actual_tree != expected_tree:
        raise ValueError(
            "split manifest source tree differs from the candidate source tree"
        )
    if actual_tree_manifest != expected_tree_manifest:
        raise ValueError(
            "split manifest source tree manifest differs from the current file"
        )
    generator_sha256 = sha256_file(generator_path)
    actual_generator_sha256 = _normalise_sha256(
        manifest["generator_sha256"],
        field="generator_sha256",
    )
    if actual_generator_sha256 != generator_sha256:
        raise ValueError(
            "split manifest generator hash differs from the current builder"
        )

    if manifest["grouping"] != GROUPING_POLICY:
        raise ValueError("split manifest grouping policy is not frozen v6")
    ratios = _expect_exact_keys(
        manifest["split_ratios"],
        set(SPLIT_NAMES),
        path="manifest.split_ratios",
    )
    for name in SPLIT_NAMES:
        value = _expect_finite_number(
            ratios[name],
            field=f"manifest.split_ratios.{name}",
        )
        if value != SPLIT_RATIOS[name]:
            raise ValueError(
                f"manifest.split_ratios.{name} is not frozen v6"
            )
    policy = _expect_exact_keys(
        manifest["predicate_coverage_policy"],
        set(PREDICATE_COVERAGE_POLICY),
        path="manifest.predicate_coverage_policy",
    )
    if (
        isinstance(policy["minimum_occurrences"], bool)
        or type(policy["minimum_occurrences"]) is not int
        or isinstance(policy["minimum_families"], bool)
        or type(policy["minimum_families"]) is not int
        or type(policy["required_in_every_split"]) is not bool
        or policy != PREDICATE_COVERAGE_POLICY
    ):
        raise ValueError(
            "split manifest predicate coverage policy is not frozen v6"
        )

    split = _validate_indices(manifest, len(rows))
    expected_splits, expected_intersections, supported = (
        compute_manifest_evidence(rows, split)
    )
    expected_shard_bytes = build_shard_bytes(source_row_bytes, split)
    for name in SPLIT_NAMES:
        actual = manifest["splits"][name]
        for numeric_field in (
            "n_rows",
            "target_ratio",
            "realized_ratio",
            "absolute_ratio_error",
        ):
            _expect_finite_number(
                actual[numeric_field],
                field=f"manifest.splits.{name}.{numeric_field}",
            )
        if (
            isinstance(actual["n_rows"], bool)
            or type(actual["n_rows"]) is not int
        ):
            raise ValueError(
                f"manifest.splits.{name}.n_rows must be an integer"
            )
        predicate_counts = actual["predicate_counts"]
        if not isinstance(predicate_counts, dict) or any(
            not isinstance(predicate, str)
            or not predicate
            or isinstance(count, bool)
            or type(count) is not int
            or count < 1
            for predicate, count in (
                predicate_counts.items()
                if isinstance(predicate_counts, dict)
                else ()
            )
        ):
            raise ValueError(
                f"manifest.splits.{name}.predicate_counts must map "
                "non-empty strings to positive non-boolean integers"
            )
        comparable = {
            field: actual[field]
            for field in SPLIT_FIELDS
            if field != "shard"
        }
        if comparable != expected_splits[name]:
            differing = sorted(
                field
                for field in SPLIT_FIELDS - {"shard"}
                if actual[field] != expected_splits[name][field]
            )
            raise ValueError(
                f"manifest.splits.{name} evidence does not match source rows: "
                f"{differing}"
            )
        actual_shard = _expect_exact_keys(
            actual["shard"],
            SHARD_FIELDS,
            path=f"manifest.splits.{name}.shard",
        )
        expected_shard = shard_evidence(
            name,
            expected_shard_bytes[name],
            len(split[name]),
        )
        if actual_shard != expected_shard:
            raise ValueError(
                f"manifest.splits.{name}.shard evidence does not match "
                "the exact selected source JSONL bytes"
            )

    intersections = _expect_exact_keys(
        manifest["intersections"],
        {"families", "normalized_text_hashes"},
        path="manifest.intersections",
    )
    for kind in ("families", "normalized_text_hashes"):
        values = _expect_exact_keys(
            intersections[kind],
            set(INTERSECTION_KEYS),
            path=f"manifest.intersections.{kind}",
        )
        for pair, items in values.items():
            if not isinstance(items, list) or any(
                not isinstance(item, str) for item in items
            ):
                raise ValueError(
                    f"manifest.intersections.{kind}.{pair} "
                    "must be an array of strings"
                )
            if items != sorted(set(items)):
                raise ValueError(
                    f"manifest.intersections.{kind}.{pair} "
                    "must be sorted and unique"
                )
    if intersections != expected_intersections:
        raise ValueError(
            "split manifest intersections do not match recomputed source rows"
        )
    if any(
        values
        for kind in expected_intersections.values()
        for values in kind.values()
    ):
        raise ValueError(
            "family or normalized-text leakage exists across CMeIE splits"
        )

    missing_coverage = {
        name: sorted(
            predicate
            for predicate in supported
            if expected_splits[name]["predicate_counts"].get(predicate, 0)
            < 1
        )
        for name in SPLIT_NAMES
    }
    if any(missing_coverage.values()):
        raise ValueError(
            "sufficiently supported predicates are absent from a split: "
            f"{missing_coverage}"
        )

    binding: dict[str, Any] = {
        "schema_version": 1,
        "kind": "cmeie_split_manifest_full_validation",
        "validated": True,
        "split_schema_version": SCHEMA_VERSION,
        "seed": FIXED_SEED,
        "rid": run_identity["rid"],
        "git_head": run_identity["git_head"],
        "canonical_run_root": run_identity["canonical_run_root"],
        "n_rows": len(rows),
        "source_train_sha256": train_sha256,
        "source_tree_sha256": actual_tree,
        "source_tree_manifest_sha256": actual_tree_manifest,
        "generator_sha256": generator_sha256,
        "validator_sha256": sha256_file(Path(__file__)),
        "split_ratios": dict(SPLIT_RATIOS),
        "grouping": GROUPING_POLICY,
        "predicate_coverage_policy": dict(
            PREDICATE_COVERAGE_POLICY
        ),
        "split_n_rows": {
            name: len(split[name]) for name in SPLIT_NAMES
        },
        "split_indices_sha256": {
            name: sha256_bytes(
                json.dumps(
                    split[name],
                    separators=(",", ":"),
                ).encode("ascii")
            )
            for name in SPLIT_NAMES
        },
        "shard_sha256": {
            name: manifest["splits"][name]["shard"]["sha256"]
            for name in SPLIT_NAMES
        },
        "sufficiently_supported_predicates": sorted(supported),
    }
    if split_manifest_sha256 is not None:
        binding["split_manifest_sha256"] = _normalise_sha256(
            split_manifest_sha256,
            field="split_manifest_sha256",
        )
    return binding


def load_and_validate_split_manifest(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    train_path: Path,
    expected_source_tree_sha256: str,
    expected_source_tree_manifest_sha256: str,
    generator_path: Path,
    expected_run_binding: RunBinding | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strictly load then fully validate one manifest before index access."""
    if expected_run_binding is not None:
        _validate_manifest_location(
            path,
            expected_run_binding=expected_run_binding,
            allow_same_parent_staging=False,
        )
    manifest, manifest_sha256 = load_split_manifest(path)
    binding = validate_split_manifest(
        manifest,
        rows,
        train_path=train_path,
        expected_source_tree_sha256=expected_source_tree_sha256,
        expected_source_tree_manifest_sha256=(
            expected_source_tree_manifest_sha256
        ),
        generator_path=generator_path,
        split_manifest_sha256=manifest_sha256,
        expected_run_binding=expected_run_binding,
    )
    return manifest, binding


def _validate_manifest_envelope(
    manifest: dict[str, Any],
    *,
    expected_source_tree_sha256: str,
    expected_source_tree_manifest_sha256: str,
    generator_path: Path,
    expected_run_binding: RunBinding | None,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Validate all v6 fields that do not require opening source/shard data."""
    _validate_json_value(manifest, path="manifest")
    _expect_exact_keys(manifest, ROOT_FIELDS, path="manifest")
    if (
        isinstance(manifest["schema_version"], bool)
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("split manifest schema is not v6")
    if manifest["seed"] != FIXED_SEED:
        raise ValueError(
            f"split manifest seed must be exactly {FIXED_SEED!r}"
        )
    run_identity = _validate_run_identity(
        manifest,
        expected_run_binding=expected_run_binding,
    )
    if (
        not isinstance(manifest["source_train"], str)
        or not manifest["source_train"]
    ):
        raise ValueError("manifest.source_train must be a non-empty string")
    source_train_sha256 = _normalise_sha256(
        manifest["source_train_sha256"],
        field="source_train_sha256",
    )
    expected_tree = _normalise_sha256(
        expected_source_tree_sha256,
        field="expected_source_tree_sha256",
    )
    expected_tree_manifest = _normalise_sha256(
        expected_source_tree_manifest_sha256,
        field="expected_source_tree_manifest_sha256",
    )
    actual_tree = _normalise_sha256(
        manifest["source_tree_sha256"],
        field="source_tree_sha256",
    )
    actual_tree_manifest = _normalise_sha256(
        manifest["source_tree_manifest_sha256"],
        field="source_tree_manifest_sha256",
    )
    if actual_tree != expected_tree:
        raise ValueError(
            "split manifest source tree differs from the candidate source tree"
        )
    if actual_tree_manifest != expected_tree_manifest:
        raise ValueError(
            "split manifest source tree manifest differs from the current file"
        )
    generator_sha256 = sha256_file(generator_path)
    if _normalise_sha256(
        manifest["generator_sha256"],
        field="generator_sha256",
    ) != generator_sha256:
        raise ValueError(
            "split manifest generator hash differs from the current builder"
        )
    if manifest["grouping"] != GROUPING_POLICY:
        raise ValueError("split manifest grouping policy is not frozen v6")
    ratios = _expect_exact_keys(
        manifest["split_ratios"],
        set(SPLIT_NAMES),
        path="manifest.split_ratios",
    )
    for role in SPLIT_NAMES:
        if (
            _expect_finite_number(
                ratios[role],
                field=f"manifest.split_ratios.{role}",
            )
            != SPLIT_RATIOS[role]
        ):
            raise ValueError(
                f"manifest.split_ratios.{role} is not frozen v6"
            )
    policy = _expect_exact_keys(
        manifest["predicate_coverage_policy"],
        set(PREDICATE_COVERAGE_POLICY),
        path="manifest.predicate_coverage_policy",
    )
    if (
        type(policy["minimum_occurrences"]) is not int
        or type(policy["minimum_families"]) is not int
        or type(policy["required_in_every_split"]) is not bool
        or policy != PREDICATE_COVERAGE_POLICY
    ):
        raise ValueError(
            "split manifest predicate coverage policy is not frozen v6"
        )

    split_objects = _expect_exact_keys(
        manifest["splits"],
        set(SPLIT_NAMES),
        path="manifest.splits",
    )
    total_rows = 0
    for role in SPLIT_NAMES:
        evidence = _expect_exact_keys(
            split_objects[role],
            SPLIT_FIELDS,
            path=f"manifest.splits.{role}",
        )
        n_rows = evidence["n_rows"]
        if isinstance(n_rows, bool) or type(n_rows) is not int or n_rows < 0:
            raise ValueError(
                f"manifest.splits.{role}.n_rows must be a non-negative integer"
            )
        total_rows += n_rows
    if total_rows < 3:
        raise ValueError("CMeIE split manifest declares fewer than three rows")
    split = _validate_indices(manifest, total_rows)

    for role in SPLIT_NAMES:
        evidence = split_objects[role]
        if evidence["n_rows"] != len(split[role]):
            raise ValueError(
                f"manifest.splits.{role}.n_rows differs from indices"
            )
        expected_numbers = {
            "target_ratio": SPLIT_RATIOS[role],
            "realized_ratio": len(split[role]) / total_rows,
            "absolute_ratio_error": abs(
                len(split[role]) / total_rows - SPLIT_RATIOS[role]
            ),
        }
        for field, expected in expected_numbers.items():
            actual = _expect_finite_number(
                evidence[field],
                field=f"manifest.splits.{role}.{field}",
            )
            if actual != expected:
                raise ValueError(
                    f"manifest.splits.{role}.{field} does not match indices"
                )
        for field in (
            "row_sha256",
            "family_sha256",
            "normalized_text_sha256",
        ):
            _normalise_sha256(
                evidence[field],
                field=f"manifest.splits.{role}.{field}",
            )
        predicate_counts = evidence["predicate_counts"]
        if not isinstance(predicate_counts, dict) or any(
            not isinstance(predicate, str)
            or not predicate
            or isinstance(count, bool)
            or type(count) is not int
            or count < 1
            for predicate, count in (
                predicate_counts.items()
                if isinstance(predicate_counts, dict)
                else ()
            )
        ):
            raise ValueError(
                f"manifest.splits.{role}.predicate_counts must map "
                "non-empty strings to positive non-boolean integers"
            )
        shard = _expect_exact_keys(
            evidence["shard"],
            SHARD_FIELDS,
            path=f"manifest.splits.{role}.shard",
        )
        if shard["relative_path"] != SHARD_FILENAMES[role]:
            raise ValueError(
                f"manifest.splits.{role}.shard.relative_path is not fixed v6"
            )
        if (
            isinstance(shard["n_rows"], bool)
            or type(shard["n_rows"]) is not int
            or shard["n_rows"] != len(split[role])
        ):
            raise ValueError(
                f"manifest.splits.{role}.shard.n_rows differs from indices"
            )
        if (
            isinstance(shard["size_bytes"], bool)
            or type(shard["size_bytes"]) is not int
            or shard["size_bytes"] < 0
        ):
            raise ValueError(
                f"manifest.splits.{role}.shard.size_bytes must be "
                "a non-negative integer"
            )
        _normalise_sha256(
            shard["sha256"],
            field=f"manifest.splits.{role}.shard.sha256",
        )

    intersections = _expect_exact_keys(
        manifest["intersections"],
        {"families", "normalized_text_hashes"},
        path="manifest.intersections",
    )
    for kind in ("families", "normalized_text_hashes"):
        values = _expect_exact_keys(
            intersections[kind],
            set(INTERSECTION_KEYS),
            path=f"manifest.intersections.{kind}",
        )
        for pair, items in values.items():
            if items != []:
                raise ValueError(
                    f"manifest.intersections.{kind}.{pair} must be empty"
                )

    return split, {
        "source_train_sha256": source_train_sha256,
        "source_tree_sha256": actual_tree,
        "source_tree_manifest_sha256": actual_tree_manifest,
        "generator_sha256": generator_sha256,
        "n_rows": total_rows,
        **run_identity,
    }


def _role_row_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indices = list(range(len(rows)))
    assignments = family_assignments(rows)
    families = set(assignments.values())
    text_hashes = {
        sha256_bytes(normalise(row.get("text", "")).encode("utf-8"))
        for row in rows
    }
    return {
        "n_rows": len(rows),
        "row_sha256": hash_list([row_hash(row) for row in rows]),
        "family_sha256": hash_list(families),
        "normalized_text_sha256": hash_list(text_hashes),
        "predicate_counts": dict(
            sorted(_predicate_counts(rows, indices).items())
        ),
    }


def load_and_validate_role_shard(
    manifest_path: Path,
    shard_path: Path,
    *,
    role: str,
    expected_source_tree_sha256: str,
    expected_source_tree_manifest_sha256: str,
    generator_path: Path,
    expected_run_binding: RunBinding | None = None,
    allow_same_parent_staging: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate exactly one physical shard without opening any other data file.

    ``allow_same_parent_staging`` exists only for the builder's pre-publication
    self-check.  The accepted staging name and parent are derived from the
    expected RID's canonical split directory; ordinary consumers retain the
    strict canonical-path check by leaving it false.
    """
    if role not in SPLIT_NAMES:
        raise ValueError(f"unknown CMeIE split role: {role!r}")
    if type(allow_same_parent_staging) is not bool:
        raise TypeError("allow_same_parent_staging must be a bool")
    if manifest_path.name != MANIFEST_FILENAME:
        raise ValueError(
            f"split manifest filename must be {MANIFEST_FILENAME!r}"
        )
    if expected_run_binding is not None:
        absolute_manifest_path = _validate_manifest_location(
            manifest_path,
            expected_run_binding=expected_run_binding,
            allow_same_parent_staging=allow_same_parent_staging,
        )
    else:
        absolute_manifest_path = _absolute_supplied_path(
            manifest_path,
            label="split manifest",
        )
    expected_shard_path = (
        absolute_manifest_path.parent / SHARD_FILENAMES[role]
    )
    absolute_shard_path = _absolute_supplied_path(
        shard_path,
        label=f"CMeIE {role} shard",
    )
    if absolute_shard_path != expected_shard_path:
        raise ValueError(
            f"{role} shard must be the fixed sibling "
            f"{SHARD_FILENAMES[role]!r}"
        )
    manifest, manifest_sha256 = load_split_manifest(manifest_path)
    _, envelope = _validate_manifest_envelope(
        manifest,
        expected_source_tree_sha256=expected_source_tree_sha256,
        expected_source_tree_manifest_sha256=(
            expected_source_tree_manifest_sha256
        ),
        generator_path=generator_path,
        expected_run_binding=expected_run_binding,
    )
    declared_shard = manifest["splits"][role]["shard"]
    raw, shard_binding = read_bound_bytes(
        shard_path,
        f"CMeIE {role} shard",
        reject_multiple_links=True,
    )
    if shard_binding["size"] != declared_shard["size_bytes"]:
        raise ValueError(
            f"{role} shard size differs from split manifest"
        )
    rows, _ = parse_cmeie_jsonl_bytes(
        raw,
        description=f"CMeIE {role} shard",
    )
    declared = manifest["splits"][role]
    actual_role = _role_row_evidence(rows)
    for field, value in actual_role.items():
        if declared[field] != value:
            raise ValueError(
                f"{role} shard {field} differs from split manifest"
            )
    actual_shard = shard_evidence(role, raw, len(rows))
    if declared["shard"] != actual_shard:
        raise ValueError(
            f"{role} shard byte binding differs from split manifest"
        )
    binding = {
        "schema_version": 1,
        "kind": "cmeie_split_role_shard_validation",
        "validated": True,
        "role": role,
        "source_train_accessed": False,
        "outer_accessed": role == "outer",
        "split_schema_version": SCHEMA_VERSION,
        "seed": FIXED_SEED,
        "rid": envelope["rid"],
        "git_head": envelope["git_head"],
        "canonical_run_root": envelope["canonical_run_root"],
        "source_train_sha256": envelope["source_train_sha256"],
        "source_tree_sha256": envelope["source_tree_sha256"],
        "source_tree_manifest_sha256": envelope[
            "source_tree_manifest_sha256"
        ],
        "generator_sha256": envelope["generator_sha256"],
        "validator_sha256": sha256_file(Path(__file__)),
        "split_manifest_sha256": manifest_sha256,
        "shard_relative_path": SHARD_FILENAMES[role],
        "shard_sha256": actual_shard["sha256"],
        "shard_size_bytes": actual_shard["size_bytes"],
        "n_rows": len(rows),
        "row_sha256": actual_role["row_sha256"],
        "family_sha256": actual_role["family_sha256"],
        "normalized_text_sha256": actual_role[
            "normalized_text_sha256"
        ],
        "predicate_counts": actual_role["predicate_counts"],
    }
    return manifest, rows, binding
