"""Canonical, replay-resistant CMeIE checkpoint selection.

The selector is an evidence consumer, not a trusted source of truth.  It
descriptor-binds the schema-v7 training receipt, re-verifies every completed
epoch bundle, re-computes every extraction-contract digest, and derives the
sole admissible calibration selection.  A downstream consumer repeats that
entire computation before accepting a schema-v3 selection receipt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from dki.evidence_io import (
    EvidenceBinding,
    load_bound_json,
    read_bound_bytes,
)
from dki.nlp.cmeie_artifacts import (
    CHECKPOINT_BUNDLE_MANIFEST,
    CheckpointBundleError,
    verify_checkpoint_bundle,
)
from dki.nlp.cmeie_extraction_contract import (
    BINDING_ALGORITHMS,
    EXTRACTION_CONTRACT_FILENAME,
    ExtractionContractError,
    load_and_validate_extraction_contract,
)
from dki.nlp.cmeie_overfit_protocol import (
    load_canonical_overfit_preflight,
    validate_overfit_preflight,
)
from dki.nlp.cmeie_preflight_protocol import (
    validate_cmeie_npu_preflight_receipt,
)
from dki.nlp.cmeie_run_protocol import (
    RunBinding,
    canonical_run_paths,
    load_run_binding,
)
from dki.nlp.cmeie_split_protocol import MANIFEST_FILENAME, SHARD_FILENAMES


TRAINING_RECEIPT_SCHEMA_VERSION = 7
TRAINING_ATTESTATION_SCHEMA_VERSION = 1
TRAINING_ATTESTATION_KIND = "cmeie_seed_training_attestation"
SELECTION_RECEIPT_SCHEMA_VERSION = 3
SELECTION_RECEIPT_KIND = "cmeie_gplinker_checkpoint_selection"
DECLARED_GRID_SIZE = 54
DECLARED_THRESHOLDS = frozenset({-0.5, 0.0, 0.5})
ALLOWED_SEEDS = frozenset({42, 43, 44})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EPOCH_DIRECTORY_RE = re.compile(r"^epoch-([0-9]{2})$")
_TREE_HASH_ALGORITHM = (
    "sha256 of sorted UTF-8 path NUL decimal-size NUL "
    "lowercase-sha256(file bytes) LF"
)
_REPARSE_POINT_ATTRIBUTE = getattr(
    stat,
    "FILE_ATTRIBUTE_REPARSE_POINT",
    0x0400,
)
_BUNDLE_FIELDS = (
    "checkpoint_bundle_manifest",
    "checkpoint_bundle_manifest_sha256",
    "checkpoint_bundle_tree_sha256",
    "checkpoint_bundle_n_files",
    "checkpoint_bundle_total_bytes",
    "checkpoint_sha256",
)
_CONTRACT_FIELDS = (
    "extraction_contract",
    "extraction_contract_sha256",
    "tokenizer_vocabulary_sha256",
    "label_table_sha256",
    "schema_constraints_sha256",
    "semantic_contract_sha256",
)
_ATTESTATION_EVIDENCE_FIELDS = (
    "training_receipt_sha256",
    "epoch_bindings",
    "preflight_common",
    "source_binding",
    "data_binding",
    "configuration",
    "overfit_preflight_sha256",
)
_ATTESTATION_FIELDS = frozenset(
    {
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
        *_ATTESTATION_EVIDENCE_FIELDS,
        "training_evidence_sha256",
    }
)
_PREFLIGHT_COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "preflight_receipt_path",
        "preflight_receipt_sha256",
        "preflight_producer_script_sha256",
        "source_tree_sha256",
        "source_manifest_sha256",
        "cmeie_train_rows",
        "cmeie_train_sha256",
        "cmeie_dev_rows",
        "cmeie_dev_sha256",
        "base_model_canonical_path",
        "base_model_tree_sha256",
        "base_model_materialization_receipt_sha256",
        "runtime_fingerprint_canonical_sha256",
        "run_binding",
    }
)
_CONFIGURATION_FIELDS = frozenset(
    {
        "seed",
        "epochs",
        "batch_size",
        "gradient_accumulation",
        "calibration_batch_size",
        "max_length",
        "fp16",
        "device",
        "pointer_variant",
        "head_reduction",
        "encoder_learning_rate",
        "head_learning_rate",
        "head_warmup_epochs",
        "head_warmup_learning_rate",
        "joint_head_learning_rate_scale",
        "encoder_gradient_clip",
        "head_gradient_clip",
        "encoder_mode",
    }
)
_DATA_FIELDS = (
    "train_sha256",
    "source_train_sha256",
    "fit_shard_sha256",
    "calibration_shard_sha256",
    "split_manifest_sha256",
)
_TRAINING_SOURCE_FIELDS = {
    "training_script_sha256": "scripts/train_spo_cmeie.py",
    "evaluator_sha256": "scripts/eval_spo_cmeie.py",
    "model_module_sha256": "dki/nlp/cmeie_gplinker.py",
    "alignment_module_sha256": "dki/nlp/cmeie_spo.py",
    "globalpointer_module_sha256": "dki/nlp/globalpointer.py",
    "artifact_integrity_module_sha256": "dki/nlp/cmeie_artifacts.py",
    "extraction_contract_module_sha256": (
        "dki/nlp/cmeie_extraction_contract.py"
    ),
}
_SELECTOR_SOURCE_FILES = {
    **_TRAINING_SOURCE_FIELDS,
    "selector_sha256": "scripts/select_spo_cmeie_checkpoint.py",
    "selection_protocol_sha256": "dki/nlp/cmeie_selection_protocol.py",
    "run_protocol_sha256": "dki/nlp/cmeie_run_protocol.py",
    "preflight_protocol_sha256": "dki/nlp/cmeie_preflight_protocol.py",
    "split_protocol_sha256": "dki/nlp/cmeie_split_protocol.py",
    "overfit_protocol_sha256": "dki/nlp/cmeie_overfit_protocol.py",
    "smoke_script_sha256": "scripts/smoke_cmeie_gplinker_overfit.py",
}


def _error(label: str, message: str) -> ValueError:
    return ValueError(f"{label}: {message}")


def _object(value: object, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise _error(label, "must be a JSON object")
    return value


def _array(value: object, *, label: str) -> list[Any]:
    if type(value) is not list:
        raise _error(label, "must be a JSON array")
    return value


def _required(value: Mapping[str, Any], field: str, *, label: str) -> Any:
    if field not in value:
        raise _error(label, f"missing required field {field!r}")
    return value[field]


def _integer(
    value: object,
    *,
    label: str,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise _error(label, "must be an integer (booleans are not integers)")
    if minimum is not None and value < minimum:
        raise _error(label, f"must be at least {minimum}")
    return value


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or "\0" in value:
        raise _error(label, "must be non-empty text without NUL bytes")
    return value


def _sha256(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if _SHA256_RE.fullmatch(text) is None:
        raise _error(label, "must be a lowercase SHA-256")
    return text


def _finite_unit_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise _error(label, "must be a finite number in [0, 1]")
    return float(value)


def _canonical_json_bytes(value: object) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "selection evidence contains a non-JSON or non-finite value"
        ) from exc


def _canonical_absolute(path: str | os.PathLike[str], *, label: str) -> Path:
    try:
        raw = os.fspath(path)
    except TypeError as exc:
        raise TypeError(f"{label} must be a text path") from exc
    if type(raw) is not str or not raw or "\0" in raw:
        raise ValueError(f"{label} must be a non-empty text path")
    if any(part in {".", ".."} for part in raw.replace("\\", "/").split("/")):
        raise ValueError(f"{label} must not contain lexical traversal")
    supplied = Path(raw)
    if not supplied.is_absolute():
        raise ValueError(f"{label} must be absolute")
    canonical = Path(os.path.abspath(raw))
    if not canonical.is_absolute() or not canonical.anchor:
        raise ValueError(f"{label} is not canonical")
    return canonical


def _real_directory(path: Path, *, label: str) -> None:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {path}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or bool(
            getattr(metadata, "st_file_attributes", 0)
            & _REPARSE_POINT_ATTRIBUTE
        )
    ):
        raise ValueError(f"{label} must be a real directory: {path}")


def _live_run_binding(run_binding: RunBinding) -> RunBinding:
    if type(run_binding) is not RunBinding:
        raise TypeError("run_binding must be a RunBinding")
    live = load_run_binding(run_binding.run_root)
    if live != run_binding:
        raise ValueError(
            "run binding changed since its source manifest was loaded"
        )
    return live


def _source_members(run_binding: RunBinding) -> dict[str, dict[str, Any]]:
    manifest, manifest_evidence = load_bound_json(
        run_binding.source_manifest,
        "CMeIE selection source manifest",
        require_object=True,
    )
    if (
        manifest_evidence["sha256"]
        != run_binding.source_manifest_sha256
        or manifest_evidence["size"] != run_binding.source_manifest_size
    ):
        raise ValueError("run source manifest binding changed")
    entries = _array(
        _required(
            _object(manifest, label="source manifest"),
            "entries",
            label="source manifest",
        ),
        label="source manifest.entries",
    )
    by_path: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(entries):
        entry = _object(raw, label=f"source manifest.entries[{index}]")
        relative = _text(
            _required(
                entry,
                "path",
                label=f"source manifest.entries[{index}]",
            ),
            label=f"source manifest.entries[{index}].path",
        )
        if relative in by_path:
            raise ValueError("source manifest contains duplicate paths")
        by_path[relative] = entry

    result: dict[str, dict[str, Any]] = {}
    for field, relative in _SELECTOR_SOURCE_FILES.items():
        if relative not in by_path:
            raise ValueError(
                f"source manifest is missing selector dependency {relative}"
            )
        posix = PurePosixPath(relative)
        if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
            raise ValueError(f"selector source path is not canonical: {relative}")
        expected = by_path[relative]
        expected_size = _integer(
            _required(expected, "size", label=f"source entry {relative}"),
            label=f"source entry {relative}.size",
            minimum=0,
        )
        expected_sha = _sha256(
            _required(expected, "sha256", label=f"source entry {relative}"),
            label=f"source entry {relative}.sha256",
        )
        path = run_binding.source_root.joinpath(*posix.parts)
        raw, evidence = read_bound_bytes(
            path,
            f"CMeIE selector dependency {relative}",
            reject_multiple_links=True,
        )
        actual_sha = hashlib.sha256(raw).hexdigest()
        if (
            evidence["size"] != expected_size
            or evidence["sha256"] != expected_sha
            or actual_sha != expected_sha
        ):
            raise ValueError(
                f"selector dependency differs from source manifest: {relative}"
            )
        result[field] = {
            "relative_path": relative,
            "path": str(path),
            "size": expected_size,
            "sha256": expected_sha,
        }
    return result


def _canonical_candidate(
    candidate_directory: str | os.PathLike[str],
    *,
    run_binding: RunBinding,
) -> tuple[Path, int]:
    candidate = _canonical_absolute(
        candidate_directory,
        label="candidate directory",
    )
    paths = canonical_run_paths(run_binding)
    matches = [
        seed
        for seed in sorted(ALLOWED_SEEDS)
        if candidate == paths.seed_model(seed)
    ]
    if len(matches) != 1:
        raise ValueError(
            "candidate directory must equal run_protocol.seed_model(seed)"
        )
    return candidate, matches[0]


def _path_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _candidate_layout(candidate: Path) -> dict[str, tuple[int, ...]]:
    """Bind the complete candidate namespace and reject aliasing.

    A successful trainer publishes only one receipt plus one directory per
    recorded epoch.  The canonical selection receipt is the sole additional
    path admitted after selection.  Checkpoint bundles themselves define each
    epoch's exact file set, so nested directories and unrelated files are
    never admissible.
    """

    _real_directory(candidate, label="candidate directory")
    result: dict[str, tuple[int, ...]] = {}
    root_names: set[str] = set()
    epoch_names: set[str] = set()
    try:
        root_entries = list(os.scandir(candidate))
    except OSError as exc:
        raise ValueError("candidate directory cannot be enumerated") from exc
    for entry in root_entries:
        name = entry.name
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or name in root_names
        ):
            raise ValueError("candidate directory has an invalid path entry")
        root_names.add(name)
        path = candidate / name
        metadata = os.lstat(path)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or bool(
                getattr(metadata, "st_file_attributes", 0)
                & _REPARSE_POINT_ATTRIBUTE
            )
            or int(metadata.st_ino) == 0
        ):
            raise ValueError(
                f"candidate path is a symlink, reparse point, or unstable: "
                f"{name}"
            )
        relative = name
        if name in {"training_receipt.json", "selection.json"}:
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    f"candidate evidence path must be a regular file: {name}"
                )
            if int(metadata.st_nlink) != 1:
                raise ValueError(
                    f"candidate evidence path has multiple hard links: {name}"
                )
            result[relative] = _path_identity(metadata)
            continue
        if _EPOCH_DIRECTORY_RE.fullmatch(name) is None:
            raise ValueError(f"candidate directory has extra path: {name}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(
                f"candidate epoch path must be a real directory: {name}"
            )
        epoch_names.add(name)
        result[relative + "/"] = _path_identity(metadata)
        try:
            children = list(os.scandir(path))
        except OSError as exc:
            raise ValueError(
                f"candidate epoch directory cannot be enumerated: {name}"
            ) from exc
        child_names: set[str] = set()
        for child in children:
            child_name = child.name
            if (
                not child_name
                or child_name in {".", ".."}
                or "/" in child_name
                or "\\" in child_name
                or child_name in child_names
            ):
                raise ValueError(
                    f"candidate epoch has an invalid path entry: {name}"
                )
            child_names.add(child_name)
            child_path = path / child_name
            child_metadata = os.lstat(child_path)
            if (
                not stat.S_ISREG(child_metadata.st_mode)
                or stat.S_ISLNK(child_metadata.st_mode)
                or bool(
                    getattr(child_metadata, "st_file_attributes", 0)
                    & _REPARSE_POINT_ATTRIBUTE
                )
                or int(child_metadata.st_ino) == 0
            ):
                raise ValueError(
                    "candidate checkpoint path must be a stable regular "
                    f"file: {name}/{child_name}"
                )
            if int(child_metadata.st_nlink) != 1:
                raise ValueError(
                    "candidate checkpoint path has multiple hard links: "
                    f"{name}/{child_name}"
                )
            result[f"{name}/{child_name}"] = _path_identity(child_metadata)
    if "training_receipt.json" not in root_names:
        raise ValueError("candidate directory lacks training_receipt.json")
    if not epoch_names:
        raise ValueError("candidate directory has no epoch directories")
    return result


def _load_training_attestation(
    *,
    run_binding: RunBinding,
    candidate: Path,
    seed: int,
) -> tuple[Path, dict[str, Any], EvidenceBinding]:
    """Load the sole canonical external authorization for a seed candidate."""

    path = canonical_run_paths(run_binding).training_attestation(seed)
    payload, binding = _stable_bound_json(
        path,
        label="CMeIE seed training attestation",
    )
    canonical = _canonical_json_bytes(payload)
    if (
        binding["canonical_path"] != str(path)
        or binding["size"] != len(canonical)
        or binding["sha256"] != hashlib.sha256(canonical).hexdigest()
    ):
        raise ValueError(
            "CMeIE seed training attestation is not canonical JSON at its "
            "canonical run path"
        )
    if set(payload) != _ATTESTATION_FIELDS:
        missing = sorted(_ATTESTATION_FIELDS - set(payload))
        extra = sorted(set(payload) - _ATTESTATION_FIELDS)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise ValueError(
            "CMeIE seed training attestation fields are invalid: "
            + "; ".join(details)
        )
    expected_core = {
        "schema_version": TRAINING_ATTESTATION_SCHEMA_VERSION,
        "kind": TRAINING_ATTESTATION_KIND,
        "attested": True,
        "rid": run_binding.rid,
        "head": run_binding.git_head,
        "tree": run_binding.included_tree_sha256,
        "run_root": str(run_binding.run_root),
        "source_manifest_path": str(run_binding.source_manifest),
        "source_manifest_sha256": run_binding.source_manifest_sha256,
        "seed": seed,
        "candidate_directory": str(candidate),
    }
    mismatches = sorted(
        field
        for field, expected in expected_core.items()
        if payload.get(field) != expected
    )
    if mismatches:
        raise ValueError(
            "CMeIE seed training attestation run binding mismatch: "
            + ", ".join(mismatches)
        )
    evidence = {
        field: copy.deepcopy(payload[field])
        for field in _ATTESTATION_EVIDENCE_FIELDS
    }
    evidence_sha256 = _sha256(
        payload["training_evidence_sha256"],
        label="training attestation.training_evidence_sha256",
    )
    if hashlib.sha256(_canonical_json_bytes(evidence)).hexdigest() != (
        evidence_sha256
    ):
        raise ValueError(
            "CMeIE seed training attestation evidence digest is inconsistent"
        )
    return path, payload, binding


def _load_canonical_overfit(
    run_binding: RunBinding,
) -> tuple[Path, dict[str, Any], EvidenceBinding]:
    path = canonical_run_paths(run_binding).overfit
    payload, binding = load_canonical_overfit_preflight(path)
    if binding["canonical_path"] != str(path):
        raise ValueError(
            "CMeIE overfit preflight is not at the canonical run path"
        )
    return path, payload, binding


def _stable_bound_json(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], EvidenceBinding]:
    payload, binding = load_bound_json(
        path,
        label,
        reject_multiple_links=True,
        require_object=True,
    )
    return _object(payload, label=label), binding


def _assert_bound_json_unchanged(
    path: Path,
    *,
    label: str,
    expected_payload: dict[str, Any],
    expected_binding: EvidenceBinding,
) -> None:
    payload, binding = _stable_bound_json(path, label=label)
    if payload != expected_payload or binding != expected_binding:
        raise ValueError(f"{label} changed during canonical selection")


def _tree_sha256(entries: list[Any], *, label: str) -> str:
    canonical: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        entry = _object(raw, label=f"{label}.entries[{index}]")
        relative = _text(
            _required(
                entry,
                "path",
                label=f"{label}.entries[{index}]",
            ),
            label=f"{label}.entries[{index}].path",
        )
        posix = PurePosixPath(relative)
        if (
            "\\" in relative
            or posix.is_absolute()
            or posix.as_posix() != relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
        ):
            raise _error(
                f"{label}.entries[{index}].path",
                "must be a canonical relative POSIX path",
            )
        if relative in seen:
            raise _error(label, "contains duplicate tree paths")
        seen.add(relative)
        size = _integer(
            _required(
                entry,
                "size",
                label=f"{label}.entries[{index}]",
            ),
            label=f"{label}.entries[{index}].size",
            minimum=0,
        )
        digest = _sha256(
            _required(
                entry,
                "sha256",
                label=f"{label}.entries[{index}]",
            ),
            label=f"{label}.entries[{index}].sha256",
        )
        canonical.append((relative, size, digest))
    digest = hashlib.sha256()
    for relative, size, file_sha in sorted(canonical):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _model_binding(receipt: dict[str, Any]) -> dict[str, Any]:
    base_model = _text(
        _required(receipt, "base_model", label="training receipt"),
        label="training receipt.base_model",
    )
    tree = _object(
        _required(receipt, "base_model_tree", label="training receipt"),
        label="training receipt.base_model_tree",
    )
    if tree.get("root") != base_model:
        raise ValueError(
            "training receipt base_model_tree.root differs from base_model"
        )
    if tree.get("algorithm") != _TREE_HASH_ALGORITHM:
        raise ValueError("training receipt base model tree algorithm is invalid")
    entries = _array(
        _required(tree, "entries", label="training receipt.base_model_tree"),
        label="training receipt.base_model_tree.entries",
    )
    actual_tree = _tree_sha256(
        entries,
        label="training receipt.base_model_tree",
    )
    expected_tree = _sha256(
        _required(
            tree,
            "sha256",
            label="training receipt.base_model_tree",
        ),
        label="training receipt.base_model_tree.sha256",
    )
    if actual_tree != expected_tree:
        raise ValueError(
            "training receipt base model tree SHA-256 is inconsistent"
        )
    n_files = _integer(
        _required(
            tree,
            "n_files",
            label="training receipt.base_model_tree",
        ),
        label="training receipt.base_model_tree.n_files",
        minimum=1,
    )
    total_bytes = _integer(
        _required(
            tree,
            "total_bytes",
            label="training receipt.base_model_tree",
        ),
        label="training receipt.base_model_tree.total_bytes",
        minimum=0,
    )
    if n_files != len(entries):
        raise ValueError("training receipt base model n_files is inconsistent")
    if total_bytes != sum(
        _integer(
            _object(item, label="base model entry").get("size"),
            label="base model entry.size",
            minimum=0,
        )
        for item in entries
    ):
        raise ValueError(
            "training receipt base model total_bytes is inconsistent"
        )
    max_length = _integer(
        _required(receipt, "max_length", label="training receipt"),
        label="training receipt.max_length",
        minimum=1,
    )
    if max_length != 320:
        raise ValueError("training receipt.max_length must be exactly 320")
    parameter_count = _integer(
        _required(receipt, "parameter_count", label="training receipt"),
        label="training receipt.parameter_count",
        minimum=1,
    )
    algorithms = _object(
        _required(receipt, "binding_algorithms", label="training receipt"),
        label="training receipt.binding_algorithms",
    )
    if algorithms != BINDING_ALGORITHMS:
        raise ValueError(
            "training receipt.binding_algorithms differs from the "
            "extraction-contract algorithms"
        )
    return {
        "base_model": base_model,
        "base_model_tree": copy.deepcopy(tree),
        "base_model_tree_sha256": expected_tree,
        "tokenizer_vocabulary_sha256": _sha256(
            _required(
                receipt,
                "tokenizer_vocabulary_sha256",
                label="training receipt",
            ),
            label="training receipt.tokenizer_vocabulary_sha256",
        ),
        "label_table_sha256": _sha256(
            _required(
                receipt,
                "label_table_sha256",
                label="training receipt",
            ),
            label="training receipt.label_table_sha256",
        ),
        "schema_constraints_sha256": _sha256(
            _required(
                receipt,
                "schema_constraints_sha256",
                label="training receipt",
            ),
            label="training receipt.schema_constraints_sha256",
        ),
        "max_length": max_length,
        "parameter_count": parameter_count,
        "binding_algorithms": copy.deepcopy(algorithms),
    }


def _data_binding(receipt: dict[str, Any]) -> dict[str, str]:
    result = {
        field: _sha256(
            _required(receipt, field, label="training receipt"),
            label=f"training receipt.{field}",
        )
        for field in _DATA_FIELDS
    }
    if result["train_sha256"] != result["source_train_sha256"]:
        raise ValueError(
            "training receipt train_sha256 differs from source_train_sha256"
        )
    return result


def _preflight_binding(
    receipt: dict[str, Any],
    *,
    run_binding: RunBinding,
    data: dict[str, str],
    model: dict[str, Any],
) -> dict[str, Any]:
    preflight = _object(
        _required(receipt, "overfit_preflight", label="training receipt"),
        label="training receipt.overfit_preflight",
    )
    _sha256(
        _required(
            preflight,
            "path_sha256",
            label="training receipt.overfit_preflight",
        ),
        label="training receipt.overfit_preflight.path_sha256",
    )
    expected = {
        "source_tree_sha256": run_binding.included_tree_sha256,
        "source_tree_manifest_sha256": run_binding.source_manifest_sha256,
        "train_sha256": data["train_sha256"],
        "source_train_sha256": data["source_train_sha256"],
        "fit_shard_sha256": data["fit_shard_sha256"],
        "split_manifest_sha256": data["split_manifest_sha256"],
        "base_model": model["base_model"],
        "base_model_tree": model["base_model_tree"],
        "tokenizer_vocabulary_sha256": model[
            "tokenizer_vocabulary_sha256"
        ],
        "label_table_sha256": model["label_table_sha256"],
        "schema_constraints_sha256": model[
            "schema_constraints_sha256"
        ],
        "calibration_accessed": False,
        "outer_accessed": False,
    }
    mismatches = [
        field
        for field, expected_value in expected.items()
        if preflight.get(field) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "training receipt overfit_preflight binding mismatch: "
            + ", ".join(sorted(mismatches))
        )
    _canonical_json_bytes(preflight)
    return copy.deepcopy(preflight)


def _strict_preflight_common(
    receipt: dict[str, Any],
    attestation: dict[str, Any],
    *,
    run_binding: RunBinding,
    data: dict[str, str],
    model: dict[str, Any],
) -> tuple[dict[str, Any], EvidenceBinding]:
    receipt_common = _object(
        _required(receipt, "preflight_common", label="training receipt"),
        label="training receipt.preflight_common",
    )
    attested_common = _object(
        _required(
            attestation,
            "preflight_common",
            label="training attestation",
        ),
        label="training attestation.preflight_common",
    )
    if receipt_common != attested_common:
        raise ValueError(
            "training attestation preflight_common differs from the training "
            "receipt"
        )
    if set(attested_common) != _PREFLIGHT_COMMON_FIELDS:
        raise ValueError(
            "training attestation preflight_common fields are invalid"
        )
    paths = canonical_run_paths(run_binding)
    expected_run = {
        "rid": run_binding.rid,
        "git_head": run_binding.git_head,
        "source_tree_sha256": run_binding.included_tree_sha256,
        "source_manifest_sha256": run_binding.source_manifest_sha256,
    }
    expected = {
        "schema_version": 1,
        "purpose": "cmeie_npu_preflight_common_binding",
        "preflight_receipt_path": str(paths.preflight_receipt),
        "source_tree_sha256": run_binding.included_tree_sha256,
        "source_manifest_sha256": run_binding.source_manifest_sha256,
        "cmeie_train_rows": 14339,
        "cmeie_train_sha256": data["source_train_sha256"],
        "cmeie_dev_rows": 3585,
        "base_model_canonical_path": str(paths.base_model),
        "base_model_tree_sha256": model["base_model_tree_sha256"],
        "run_binding": expected_run,
    }
    mismatches = sorted(
        field
        for field, expected_value in expected.items()
        if attested_common.get(field) != expected_value
    )
    if model["base_model"] != str(paths.base_model):
        mismatches.append("training_receipt.base_model")
    if mismatches:
        raise ValueError(
            "training attestation strict preflight binding mismatch: "
            + ", ".join(sorted(set(mismatches)))
        )
    for field in (
        "preflight_receipt_sha256",
        "preflight_producer_script_sha256",
        "cmeie_dev_sha256",
        "base_model_materialization_receipt_sha256",
        "runtime_fingerprint_canonical_sha256",
    ):
        _sha256(
            _required(
                attested_common,
                field,
                label="training attestation.preflight_common",
            ),
            label=f"training attestation.preflight_common.{field}",
        )
    validated_common = validate_cmeie_npu_preflight_receipt(
        str(paths.preflight_receipt),
        expected_source_tree_sha256=run_binding.included_tree_sha256,
        expected_source_manifest_sha256=run_binding.source_manifest_sha256,
        expected_cmeie_train_sha256=data["source_train_sha256"],
        expected_base_model_canonical_path=str(paths.base_model),
        expected_base_model_tree_sha256=model["base_model_tree_sha256"],
        expected_run_binding=expected_run,
    )
    if validated_common != attested_common:
        raise ValueError(
            "training attestation preflight_common differs from full strict "
            "preflight revalidation"
        )
    _, binding = read_bound_bytes(
        paths.preflight_receipt,
        "CMeIE strict NPU preflight receipt for selection",
        reject_multiple_links=True,
    )
    if (
        binding["canonical_path"] != str(paths.preflight_receipt)
        or binding["sha256"]
        != attested_common["preflight_receipt_sha256"]
    ):
        raise ValueError(
            "training attestation does not bind the canonical strict "
            "preflight receipt"
        )
    _canonical_json_bytes(attested_common)
    return copy.deepcopy(attested_common), binding


def _split_artifact_binding(
    *,
    run_binding: RunBinding,
    data: dict[str, str],
) -> dict[str, dict[str, Any]]:
    paths = canonical_run_paths(run_binding)
    expected = {
        "split_manifest": (
            paths.split_dir / MANIFEST_FILENAME,
            data["split_manifest_sha256"],
        ),
        "fit_shard": (
            paths.split_dir / SHARD_FILENAMES["fit"],
            data["fit_shard_sha256"],
        ),
        "calibration_shard": (
            paths.split_dir / SHARD_FILENAMES["calibration"],
            data["calibration_shard_sha256"],
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for role, (path, expected_sha256) in expected.items():
        _, binding = read_bound_bytes(
            path,
            f"CMeIE canonical {role} for selection",
            reject_multiple_links=True,
        )
        if (
            binding["canonical_path"] != str(path)
            or binding["sha256"] != expected_sha256
        ):
            raise ValueError(
                f"training attestation {role} binding differs from the "
                "canonical split artifact"
            )
        result[role] = {
            "path": str(path),
            "size": int(binding["size"]),
            "sha256": str(binding["sha256"]),
        }
    return result


def _training_configuration(
    receipt: dict[str, Any],
    attestation: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    configuration = _object(
        _required(
            attestation,
            "configuration",
            label="training attestation",
        ),
        label="training attestation.configuration",
    )
    if set(configuration) != _CONFIGURATION_FIELDS:
        raise ValueError(
            "training attestation configuration fields are invalid"
        )
    protocol = _object(
        _required(receipt, "training_protocol", label="training receipt"),
        label="training receipt.training_protocol",
    )
    joint = _object(
        _required(
            protocol,
            "joint_training",
            label="training receipt.training_protocol",
        ),
        label="training receipt.training_protocol.joint_training",
    )
    head_warmup = _object(
        _required(
            protocol,
            "head_only_warmup",
            label="training receipt.training_protocol",
        ),
        label="training receipt.training_protocol.head_only_warmup",
    )
    expected = {
        "seed": seed,
        "epochs": _required(receipt, "epochs", label="training receipt"),
        "batch_size": _required(
            receipt,
            "batch_size",
            label="training receipt",
        ),
        "gradient_accumulation": _required(
            receipt,
            "gradient_accumulation",
            label="training receipt",
        ),
        "calibration_batch_size": _required(
            receipt,
            "calibration_batch_size",
            label="training receipt",
        ),
        "max_length": _required(
            receipt,
            "max_length",
            label="training receipt",
        ),
        "fp16": _required(receipt, "fp16", label="training receipt"),
        "device": "npu",
        "pointer_variant": _required(
            protocol,
            "pointer_variant",
            label="training receipt.training_protocol",
        ),
        "head_reduction": _required(
            protocol,
            "head_reduction",
            label="training receipt.training_protocol",
        ),
        "encoder_learning_rate": _required(
            joint,
            "encoder_learning_rate",
            label="training receipt.training_protocol.joint_training",
        ),
        "head_learning_rate": _required(
            joint,
            "head_learning_rate",
            label="training receipt.training_protocol.joint_training",
        ),
        "head_warmup_epochs": _required(
            head_warmup,
            "epochs",
            label="training receipt.training_protocol.head_only_warmup",
        ),
        "head_warmup_learning_rate": _required(
            head_warmup,
            "head_learning_rate",
            label="training receipt.training_protocol.head_only_warmup",
        ),
        "joint_head_learning_rate_scale": _required(
            joint,
            "head_learning_rate_scale",
            label="training receipt.training_protocol.joint_training",
        ),
        "encoder_gradient_clip": _required(
            joint,
            "encoder_gradient_clip",
            label="training receipt.training_protocol.joint_training",
        ),
        "head_gradient_clip": _required(
            joint,
            "head_gradient_clip",
            label="training receipt.training_protocol.joint_training",
        ),
        "encoder_mode": _required(
            joint,
            "encoder_mode",
            label="training receipt.training_protocol.joint_training",
        ),
    }
    if receipt.get("device") != "npu:0":
        raise ValueError(
            "training receipt device must be the canonical Ascend npu:0"
        )
    if configuration != expected:
        raise ValueError(
            "training attestation configuration differs from the training "
            "receipt"
        )
    _canonical_json_bytes(configuration)
    return copy.deepcopy(configuration)


def _epoch_attestation_bindings(
    receipt: dict[str, Any],
    attestation: dict[str, Any],
) -> list[dict[str, Any]]:
    history = _array(
        _required(receipt, "history", label="training receipt"),
        label="training receipt.history",
    )
    expected_fields = frozenset({"epoch", *_BUNDLE_FIELDS, *_CONTRACT_FIELDS})
    expected: list[dict[str, Any]] = []
    for index, raw in enumerate(history):
        item = _object(raw, label=f"training receipt.history[{index}]")
        missing = sorted(expected_fields - set(item))
        if missing:
            raise ValueError(
                "training receipt history lacks attested epoch fields: "
                + ", ".join(missing)
            )
        expected.append(
            {field: copy.deepcopy(item[field]) for field in expected_fields}
        )
    attested = _array(
        _required(
            attestation,
            "epoch_bindings",
            label="training attestation",
        ),
        label="training attestation.epoch_bindings",
    )
    for index, raw in enumerate(attested):
        item = _object(
            raw,
            label=f"training attestation.epoch_bindings[{index}]",
        )
        if set(item) != expected_fields:
            raise ValueError(
                "training attestation epoch binding fields are invalid"
            )
    if attested != expected:
        raise ValueError(
            "training attestation epoch bindings differ from the training "
            "receipt"
        )
    return copy.deepcopy(expected)


def _validate_training_attestation(
    *,
    attestation: dict[str, Any],
    attestation_binding: EvidenceBinding,
    receipt: dict[str, Any],
    receipt_binding: EvidenceBinding,
    run_binding: RunBinding,
    seed: int,
    data: dict[str, str],
    model: dict[str, Any],
) -> dict[str, Any]:
    receipt_sha256 = _sha256(
        attestation["training_receipt_sha256"],
        label="training attestation.training_receipt_sha256",
    )
    if receipt_sha256 != receipt_binding["sha256"]:
        raise ValueError(
            "training attestation training receipt SHA-256 mismatch"
        )
    expected_source = {
        "rid": run_binding.rid,
        "git_head": run_binding.git_head,
        "source_tree_sha256": run_binding.included_tree_sha256,
        "source_manifest_sha256": run_binding.source_manifest_sha256,
        "source_manifest_path": str(run_binding.source_manifest),
    }
    source = _object(
        attestation["source_binding"],
        label="training attestation.source_binding",
    )
    if source != expected_source:
        raise ValueError(
            "training attestation source binding differs from the "
            "authoritative run"
        )
    attested_data = _object(
        attestation["data_binding"],
        label="training attestation.data_binding",
    )
    if set(attested_data) != set(_DATA_FIELDS) or attested_data != data:
        raise ValueError(
            "training attestation split data binding differs from the "
            "training receipt"
        )
    configuration = _training_configuration(
        receipt,
        attestation,
        seed=seed,
    )
    epochs = _epoch_attestation_bindings(receipt, attestation)
    preflight, preflight_binding = _strict_preflight_common(
        receipt,
        attestation,
        run_binding=run_binding,
        data=data,
        model=model,
    )
    split = _split_artifact_binding(run_binding=run_binding, data=data)
    return {
        "path": str(attestation_binding["canonical_path"]),
        "size": int(attestation_binding["size"]),
        "sha256": str(attestation_binding["sha256"]),
        "schema_version": TRAINING_ATTESTATION_SCHEMA_VERSION,
        "training_evidence_sha256": str(
            attestation["training_evidence_sha256"]
        ),
        "configuration": configuration,
        "epoch_bindings": epochs,
        "preflight_common": preflight,
        "preflight_receipt_binding": {
            "path": str(preflight_binding["canonical_path"]),
            "size": int(preflight_binding["size"]),
            "sha256": str(preflight_binding["sha256"]),
        },
        "split_artifact_binding": split,
    }


def _validate_canonical_overfit_binding(
    *,
    payload: dict[str, Any],
    file_binding: EvidenceBinding,
    receipt: dict[str, Any],
    attestation: dict[str, Any],
    run_binding: RunBinding,
    source_members: dict[str, dict[str, Any]],
    data: dict[str, str],
    model: dict[str, Any],
    strict_preflight_common: dict[str, Any],
) -> dict[str, Any]:
    validated = validate_overfit_preflight(
        payload,
        receipt_sha256=str(file_binding["sha256"]),
        source_train_sha256=data["source_train_sha256"],
        fit_shard_sha256=data["fit_shard_sha256"],
        split_manifest_sha256=data["split_manifest_sha256"],
        base_model=model["base_model"],
        base_model_tree=model["base_model_tree"],
        source_tree_manifest_sha256=run_binding.source_manifest_sha256,
        preflight_script_sha256=source_members[
            "smoke_script_sha256"
        ]["sha256"],
        model_module_sha256=source_members[
            "model_module_sha256"
        ]["sha256"],
        alignment_module_sha256=source_members[
            "alignment_module_sha256"
        ]["sha256"],
        globalpointer_module_sha256=source_members[
            "globalpointer_module_sha256"
        ]["sha256"],
        tokenizer_vocabulary_sha256=model[
            "tokenizer_vocabulary_sha256"
        ],
        label_table_sha256=model["label_table_sha256"],
        schema_constraints_sha256=model[
            "schema_constraints_sha256"
        ],
        source_tree_sha256=run_binding.included_tree_sha256,
        preflight_common=strict_preflight_common,
    )
    receipt_binding = _object(
        _required(
            receipt,
            "overfit_preflight",
            label="training receipt",
        ),
        label="training receipt.overfit_preflight",
    )
    attested_sha256 = _sha256(
        _required(
            attestation,
            "overfit_preflight_sha256",
            label="training attestation",
        ),
        label="training attestation.overfit_preflight_sha256",
    )
    if receipt_binding != validated:
        raise ValueError(
            "training receipt overfit_preflight differs from canonical "
            "overfit revalidation"
        )
    if attested_sha256 != validated["path_sha256"]:
        raise ValueError(
            "training attestation overfit preflight SHA-256 differs from "
            "canonical overfit revalidation"
        )
    return {
        "path": str(file_binding["canonical_path"]),
        "size": int(file_binding["size"]),
        "sha256": str(file_binding["sha256"]),
        "validated_binding": copy.deepcopy(validated),
    }


def _metric(value: object, *, epoch: int) -> dict[str, Any]:
    metric = _object(value, label=f"history epoch {epoch} metric")
    expected_fields = {"tp", "fp", "fn", "precision", "recall", "f1"}
    if set(metric) != expected_fields:
        raise _error(
            f"history epoch {epoch} metric",
            "must contain exactly tp, fp, fn, precision, recall, f1",
        )
    counts = {
        field: _integer(
            metric[field],
            label=f"history epoch {epoch} metric.{field}",
            minimum=0,
        )
        for field in ("tp", "fp", "fn")
    }
    stated = {
        field: _finite_unit_float(
            metric[field],
            label=f"history epoch {epoch} metric.{field}",
        )
        for field in ("precision", "recall", "f1")
    }
    precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
    recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    recomputed = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    inconsistent = [
        field
        for field in ("precision", "recall", "f1")
        if not math.isclose(
            stated[field],
            recomputed[field],
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ]
    if inconsistent:
        raise ValueError(
            f"history epoch {epoch} calibration metric disagrees with "
            "tp/fp/fn: " + ", ".join(inconsistent)
        )
    return copy.deepcopy(metric)


def _declared_selection(value: object, *, epoch: int) -> dict[str, Any]:
    selection = _object(
        value,
        label=f"history epoch {epoch} calibration selection",
    )
    if selection.get("status") != "completed":
        raise ValueError(
            f"history epoch {epoch} completed selection status is invalid"
        )
    if selection.get("grid_size") != DECLARED_GRID_SIZE:
        raise ValueError(
            f"history epoch {epoch} did not evaluate the 54-item grid"
        )
    thresholds = _array(
        _required(
            selection,
            "thresholds",
            label=f"history epoch {epoch} calibration selection",
        ),
        label=f"history epoch {epoch} thresholds",
    )
    if (
        len(thresholds) != 3
        or any(
            type(threshold) is not float
            or threshold not in DECLARED_THRESHOLDS
            for threshold in thresholds
        )
    ):
        raise ValueError(
            f"history epoch {epoch} thresholds are outside the declared grid"
        )
    schema_constraint = selection.get("schema_constraint")
    if schema_constraint not in {"off", "on"}:
        raise ValueError(
            f"history epoch {epoch} schema_constraint is invalid"
        )
    metric = _metric(selection.get("metric"), epoch=epoch)
    return {
        "thresholds": copy.deepcopy(thresholds),
        "schema_constraint": schema_constraint,
        "grid_size": DECLARED_GRID_SIZE,
        "metric": metric,
    }


def _tokenizer_vocabulary(
    checkpoint_directory: Path,
) -> tuple[dict[str, int], dict[str, Any]]:
    path = checkpoint_directory / "tokenizer.json"
    payload, evidence = load_bound_json(
        path,
        "CMeIE checkpoint tokenizer",
        reject_multiple_links=True,
        require_object=True,
    )
    tokenizer = _object(payload, label="checkpoint tokenizer")
    model = _object(
        _required(tokenizer, "model", label="checkpoint tokenizer"),
        label="checkpoint tokenizer.model",
    )
    raw_vocabulary = _required(
        model,
        "vocab",
        label="checkpoint tokenizer.model",
    )
    vocabulary: dict[str, int] = {}
    if type(raw_vocabulary) is dict:
        for token, identifier in raw_vocabulary.items():
            text = _text(token, label="checkpoint tokenizer vocabulary token")
            value = _integer(
                identifier,
                label=f"checkpoint tokenizer vocabulary[{text!r}]",
                minimum=0,
            )
            vocabulary[text] = value
    elif type(raw_vocabulary) is list:
        for identifier, raw in enumerate(raw_vocabulary):
            if (
                type(raw) is not list
                or len(raw) < 1
                or type(raw[0]) is not str
            ):
                raise ValueError(
                    "checkpoint tokenizer unigram vocabulary is invalid"
                )
            vocabulary[_text(raw[0], label="tokenizer unigram token")] = (
                identifier
            )
    else:
        raise ValueError("checkpoint tokenizer model.vocab is invalid")

    added = tokenizer.get("added_tokens", [])
    if type(added) is not list:
        raise ValueError("checkpoint tokenizer added_tokens is invalid")
    for index, raw in enumerate(added):
        item = _object(
            raw,
            label=f"checkpoint tokenizer.added_tokens[{index}]",
        )
        token = _text(
            _required(
                item,
                "content",
                label=f"checkpoint tokenizer.added_tokens[{index}]",
            ),
            label=f"checkpoint tokenizer.added_tokens[{index}].content",
        )
        identifier = _integer(
            _required(
                item,
                "id",
                label=f"checkpoint tokenizer.added_tokens[{index}]",
            ),
            label=f"checkpoint tokenizer.added_tokens[{index}].id",
            minimum=0,
        )
        previous = vocabulary.get(token)
        if previous is not None and previous != identifier:
            raise ValueError(
                "checkpoint tokenizer assigns conflicting IDs to one token"
            )
        vocabulary[token] = identifier
    if not vocabulary or len(set(vocabulary.values())) != len(vocabulary):
        raise ValueError(
            "checkpoint tokenizer vocabulary IDs must be unique"
        )
    return vocabulary, {
        "path": str(path),
        "size": int(evidence["size"]),
        "sha256": str(evidence["sha256"]),
    }


def _verified_completed_epoch(
    candidate: Path,
    item: dict[str, Any],
    *,
    epoch: int,
    model: dict[str, Any],
) -> dict[str, Any]:
    for field in _BUNDLE_FIELDS:
        _required(item, field, label=f"history epoch {epoch}")
    if item["checkpoint_bundle_manifest"] != CHECKPOINT_BUNDLE_MANIFEST:
        raise ValueError(
            f"history epoch {epoch} bundle manifest filename is invalid"
        )
    manifest_sha = _sha256(
        item["checkpoint_bundle_manifest_sha256"],
        label=f"history epoch {epoch}.checkpoint_bundle_manifest_sha256",
    )
    tree_sha = _sha256(
        item["checkpoint_bundle_tree_sha256"],
        label=f"history epoch {epoch}.checkpoint_bundle_tree_sha256",
    )
    checkpoint_sha = _sha256(
        item["checkpoint_sha256"],
        label=f"history epoch {epoch}.checkpoint_sha256",
    )
    n_files = _integer(
        item["checkpoint_bundle_n_files"],
        label=f"history epoch {epoch}.checkpoint_bundle_n_files",
        minimum=1,
    )
    total_bytes = _integer(
        item["checkpoint_bundle_total_bytes"],
        label=f"history epoch {epoch}.checkpoint_bundle_total_bytes",
        minimum=0,
    )
    checkpoint_directory = candidate / f"epoch-{epoch:02d}"
    try:
        bundle = verify_checkpoint_bundle(
            checkpoint_directory,
            expected_manifest_sha256=manifest_sha,
            expected_tree_sha256=tree_sha,
            expected_checkpoint_sha256=checkpoint_sha,
        )
    except (CheckpointBundleError, OSError) as exc:
        raise ValueError(
            f"history epoch {epoch} checkpoint bundle verification failed: "
            f"{exc}"
        ) from exc
    expected_bundle = {
        field: item[field]
        for field in _BUNDLE_FIELDS
    }
    if bundle != expected_bundle:
        mismatches = [
            field
            for field in _BUNDLE_FIELDS
            if bundle.get(field) != expected_bundle.get(field)
        ]
        raise ValueError(
            f"history epoch {epoch} bundle binding mismatch: "
            + ", ".join(mismatches)
        )
    if (
        bundle["checkpoint_bundle_n_files"] != n_files
        or bundle["checkpoint_bundle_total_bytes"] != total_bytes
    ):
        raise ValueError(f"history epoch {epoch} bundle counts are invalid")

    for field in _CONTRACT_FIELDS:
        _required(item, field, label=f"history epoch {epoch}")
    if item["extraction_contract"] != EXTRACTION_CONTRACT_FILENAME:
        raise ValueError(
            f"history epoch {epoch} extraction contract filename is invalid"
        )
    contract_expected = {
        "extraction_contract": item["extraction_contract"],
        **{
            field: _sha256(
                item[field],
                label=f"history epoch {epoch}.{field}",
            )
            for field in _CONTRACT_FIELDS
            if field != "extraction_contract"
        },
    }
    for field in (
        "tokenizer_vocabulary_sha256",
        "label_table_sha256",
        "schema_constraints_sha256",
    ):
        if contract_expected[field] != model[field]:
            raise ValueError(
                f"history epoch {epoch} {field} differs from training model "
                "binding"
            )
    vocabulary, tokenizer_binding = _tokenizer_vocabulary(
        checkpoint_directory
    )
    try:
        contract = load_and_validate_extraction_contract(
            checkpoint_directory,
            tokenizer_vocabulary=vocabulary,
            expected_binding=contract_expected,
        )
    except (ExtractionContractError, OSError, ValueError) as exc:
        raise ValueError(
            f"history epoch {epoch} extraction contract verification failed: "
            f"{exc}"
        ) from exc
    if contract != contract_expected:
        raise ValueError(
            f"history epoch {epoch} extraction contract binding mismatch"
        )
    selection = _declared_selection(
        item.get("calibration_declared_grid_selection"),
        epoch=epoch,
    )
    return {
        "epoch": epoch,
        "checkpoint_directory": checkpoint_directory,
        "checkpoint": checkpoint_directory / "gplinker.pt",
        "bundle": bundle,
        "contract": contract,
        "tokenizer": tokenizer_binding,
        "selection": selection,
    }


def _completed_epochs(
    candidate: Path,
    receipt: dict[str, Any],
    *,
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    history = _array(
        _required(receipt, "history", label="training receipt"),
        label="training receipt.history",
    )
    if not history:
        raise ValueError("training receipt.history must not be empty")
    seen: set[int] = set()
    completed: list[dict[str, Any]] = []
    for index, raw in enumerate(history):
        item = _object(raw, label=f"training receipt.history[{index}]")
        epoch = _integer(
            _required(
                item,
                "epoch",
                label=f"training receipt.history[{index}]",
            ),
            label=f"training receipt.history[{index}].epoch",
            minimum=1,
        )
        if epoch > 99 or epoch in seen:
            raise ValueError(
                f"training receipt history epoch is invalid or duplicate: "
                f"{epoch}"
            )
        seen.add(epoch)
        selection = _object(
            _required(
                item,
                "calibration_declared_grid_selection",
                label=f"history epoch {epoch}",
            ),
            label=f"history epoch {epoch} calibration selection",
        )
        status = selection.get("status")
        if status == "completed":
            completed.append(
                _verified_completed_epoch(
                    candidate,
                    item,
                    epoch=epoch,
                    model=model,
                )
            )
        elif (
            status == "deferred_candidate_explosion"
            and selection.get("selection_used") is False
            and selection.get("grid_size") == DECLARED_GRID_SIZE
        ):
            continue
        else:
            raise ValueError(
                f"history epoch {epoch} calibration selection is incomplete"
            )
    actual = {
        path.name
        for path in candidate.iterdir()
        if path.is_dir() and _EPOCH_DIRECTORY_RE.fullmatch(path.name)
    }
    recorded = {f"epoch-{epoch:02d}" for epoch in seen}
    if actual != recorded:
        raise ValueError(
            "candidate epoch directories differ from training receipt history"
        )
    if not completed:
        raise ValueError(
            "training receipt has no completed calibration checkpoint"
        )
    return completed


def _training_source_binding(
    receipt: dict[str, Any],
    source_members: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    for field in _TRAINING_SOURCE_FIELDS:
        expected = _sha256(
            _required(receipt, field, label="training receipt"),
            label=f"training receipt.{field}",
        )
        if source_members[field]["sha256"] != expected:
            raise ValueError(
                f"training receipt {field} differs from the bound run source"
            )
    return copy.deepcopy(source_members)


def _selection_payload(
    *,
    run_binding: RunBinding,
    source_members: dict[str, dict[str, Any]],
    candidate: Path,
    seed: int,
    receipt_path: Path,
    receipt_binding: EvidenceBinding,
    receipt: dict[str, Any],
    attestation: dict[str, Any],
    overfit_evidence: dict[str, Any],
    data: dict[str, str],
    preflight: dict[str, Any],
    model: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    bundle = copy.deepcopy(selected["bundle"])
    contract = copy.deepcopy(selected["contract"])
    declared = selected["selection"]
    source_binding = {
        "source_manifest_path": str(run_binding.source_manifest),
        "source_manifest_sha256": run_binding.source_manifest_sha256,
        "source_manifest_size": run_binding.source_manifest_size,
        "source_root": str(run_binding.source_root),
        "modules": copy.deepcopy(source_members),
    }
    return {
        "schema_version": SELECTION_RECEIPT_SCHEMA_VERSION,
        "kind": SELECTION_RECEIPT_KIND,
        "dataset_role": "calibration_selection",
        "selection_used": True,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "rid": run_binding.rid,
        "head": run_binding.git_head,
        "tree": run_binding.included_tree_sha256,
        "run_root": str(run_binding.run_root),
        "source_manifest_path": str(run_binding.source_manifest),
        "source_manifest_sha256": run_binding.source_manifest_sha256,
        "source_manifest_size": run_binding.source_manifest_size,
        "source_root": str(run_binding.source_root),
        "source_binding": source_binding,
        "candidate_directory": str(candidate),
        "seed": seed,
        "training_receipt_path": str(receipt_path),
        "training_receipt_sha256": str(receipt_binding["sha256"]),
        "training_receipt_size": int(receipt_binding["size"]),
        "training_receipt_schema_version": (
            TRAINING_RECEIPT_SCHEMA_VERSION
        ),
        "training_attestation_path": attestation["path"],
        "training_attestation_sha256": attestation["sha256"],
        "training_attestation_size": attestation["size"],
        "training_attestation_schema_version": attestation[
            "schema_version"
        ],
        "training_evidence_sha256": attestation[
            "training_evidence_sha256"
        ],
        "training_attestation_binding": copy.deepcopy(attestation),
        "canonical_overfit_preflight_path": overfit_evidence["path"],
        "canonical_overfit_preflight_sha256": overfit_evidence["sha256"],
        "canonical_overfit_preflight_size": overfit_evidence["size"],
        "canonical_overfit_validated_binding": copy.deepcopy(
            overfit_evidence["validated_binding"]
        ),
        "selected_epoch": selected["epoch"],
        "checkpoint": str(selected["checkpoint"]),
        "checkpoint_directory": str(selected["checkpoint_directory"]),
        "checkpoint_sha256": bundle["checkpoint_sha256"],
        "checkpoint_bundle_manifest": str(
            selected["checkpoint_directory"]
            / CHECKPOINT_BUNDLE_MANIFEST
        ),
        "checkpoint_bundle_manifest_sha256": bundle[
            "checkpoint_bundle_manifest_sha256"
        ],
        "checkpoint_bundle_tree_sha256": bundle[
            "checkpoint_bundle_tree_sha256"
        ],
        "checkpoint_bundle_n_files": bundle[
            "checkpoint_bundle_n_files"
        ],
        "checkpoint_bundle_total_bytes": bundle[
            "checkpoint_bundle_total_bytes"
        ],
        "bundle_binding": bundle,
        "extraction_contract": str(
            selected["checkpoint_directory"]
            / EXTRACTION_CONTRACT_FILENAME
        ),
        "extraction_contract_sha256": contract[
            "extraction_contract_sha256"
        ],
        "semantic_contract_sha256": contract[
            "semantic_contract_sha256"
        ],
        "contract_binding": contract,
        "tokenizer_binding": copy.deepcopy(selected["tokenizer"]),
        "thresholds": copy.deepcopy(declared["thresholds"]),
        "schema_constraint": declared["schema_constraint"],
        "calibration_metric": copy.deepcopy(declared["metric"]),
        "grid_size": declared["grid_size"],
        "tie_break": ["f1_desc", "precision_desc", "epoch_asc"],
        "source_tree_sha256": run_binding.included_tree_sha256,
        "source_tree_manifest_sha256": (
            run_binding.source_manifest_sha256
        ),
        **copy.deepcopy(data),
        "data_binding": copy.deepcopy(data),
        "overfit_preflight_sha256": preflight["path_sha256"],
        "preflight_binding": copy.deepcopy(preflight),
        "strict_preflight_common": copy.deepcopy(
            attestation["preflight_common"]
        ),
        "split_artifact_binding": copy.deepcopy(
            attestation["split_artifact_binding"]
        ),
        "base_model": model["base_model"],
        "base_model_tree_sha256": model["base_model_tree_sha256"],
        "tokenizer_vocabulary_sha256": model[
            "tokenizer_vocabulary_sha256"
        ],
        "label_table_sha256": model["label_table_sha256"],
        "schema_constraints_sha256": model[
            "schema_constraints_sha256"
        ],
        "max_length": model["max_length"],
        "parameter_count": model["parameter_count"],
        "model_binding": copy.deepcopy(model),
        **{
            field: source_members[field]["sha256"]
            for field in _TRAINING_SOURCE_FIELDS
        },
        "selector_sha256": source_members["selector_sha256"]["sha256"],
        "selection_protocol_sha256": source_members[
            "selection_protocol_sha256"
        ]["sha256"],
        "run_protocol_sha256": source_members[
            "run_protocol_sha256"
        ]["sha256"],
        "preflight_protocol_sha256": source_members[
            "preflight_protocol_sha256"
        ]["sha256"],
        "split_protocol_sha256": source_members[
            "split_protocol_sha256"
        ]["sha256"],
        "overfit_protocol_sha256": source_members[
            "overfit_protocol_sha256"
        ]["sha256"],
        "smoke_script_sha256": source_members[
            "smoke_script_sha256"
        ]["sha256"],
    }


def compute_canonical_selection(
    candidate_directory: str | os.PathLike[str],
    *,
    run_binding: RunBinding,
) -> dict[str, Any]:
    """Recompute the sole admissible calibration selection for one seed."""

    live = _live_run_binding(run_binding)
    candidate, path_seed = _canonical_candidate(
        candidate_directory,
        run_binding=live,
    )
    (
        overfit_path,
        overfit_payload,
        overfit_file_binding,
    ) = _load_canonical_overfit(live)
    (
        attestation_path,
        attestation_payload,
        attestation_file_binding,
    ) = _load_training_attestation(
        run_binding=live,
        candidate=candidate,
        seed=path_seed,
    )
    candidate_layout = _candidate_layout(candidate)
    receipt_path = candidate / "training_receipt.json"
    receipt, receipt_binding = _stable_bound_json(
        receipt_path,
        label="CMeIE training receipt",
    )
    schema = _integer(
        _required(receipt, "schema_version", label="training receipt"),
        label="training receipt.schema_version",
    )
    if schema != TRAINING_RECEIPT_SCHEMA_VERSION:
        raise ValueError(
            "checkpoint selection requires training receipt schema v7"
        )
    seed = _integer(
        _required(receipt, "seed", label="training receipt"),
        label="training receipt.seed",
    )
    if seed not in ALLOWED_SEEDS or seed != path_seed:
        raise ValueError(
            "training receipt seed must be 42, 43, or 44 and match the "
            "canonical candidate directory"
        )
    source_tree = _sha256(
        _required(receipt, "source_tree_sha256", label="training receipt"),
        label="training receipt.source_tree_sha256",
    )
    source_manifest = _sha256(
        _required(
            receipt,
            "source_tree_manifest_sha256",
            label="training receipt",
        ),
        label="training receipt.source_tree_manifest_sha256",
    )
    if source_tree != live.included_tree_sha256:
        raise ValueError(
            "training receipt source tree differs from the canonical RID"
        )
    if source_manifest != live.source_manifest_sha256:
        raise ValueError(
            "training receipt source manifest differs from the canonical RID"
        )

    source_members = _source_members(live)
    _training_source_binding(receipt, source_members)
    data = _data_binding(receipt)
    model = _model_binding(receipt)
    preflight = _preflight_binding(
        receipt,
        run_binding=live,
        data=data,
        model=model,
    )
    attestation = _validate_training_attestation(
        attestation=attestation_payload,
        attestation_binding=attestation_file_binding,
        receipt=receipt,
        receipt_binding=receipt_binding,
        run_binding=live,
        seed=seed,
        data=data,
        model=model,
    )
    overfit_evidence = _validate_canonical_overfit_binding(
        payload=overfit_payload,
        file_binding=overfit_file_binding,
        receipt=receipt,
        attestation=attestation_payload,
        run_binding=live,
        source_members=source_members,
        data=data,
        model=model,
        strict_preflight_common=attestation["preflight_common"],
    )
    completed = _completed_epochs(candidate, receipt, model=model)
    selected = max(
        completed,
        key=lambda item: (
            item["selection"]["metric"]["f1"],
            item["selection"]["metric"]["precision"],
            -item["epoch"],
        ),
    )
    payload = _selection_payload(
        run_binding=live,
        source_members=source_members,
        candidate=candidate,
        seed=seed,
        receipt_path=receipt_path,
        receipt_binding=receipt_binding,
        receipt=receipt,
        attestation=attestation,
        overfit_evidence=overfit_evidence,
        data=data,
        preflight=preflight,
        model=model,
        selected=selected,
    )
    _canonical_json_bytes(payload)
    _assert_bound_json_unchanged(
        receipt_path,
        label="CMeIE training receipt",
        expected_payload=receipt,
        expected_binding=receipt_binding,
    )
    revalidated_attestation = _validate_training_attestation(
        attestation=attestation_payload,
        attestation_binding=attestation_file_binding,
        receipt=receipt,
        receipt_binding=receipt_binding,
        run_binding=live,
        seed=seed,
        data=data,
        model=model,
    )
    if revalidated_attestation != attestation:
        raise ValueError(
            "training attestation dependencies changed during canonical "
            "selection"
        )
    (
        latest_overfit_path,
        latest_overfit_payload,
        latest_overfit_file_binding,
    ) = _load_canonical_overfit(live)
    if (
        latest_overfit_path != overfit_path
        or latest_overfit_payload != overfit_payload
        or latest_overfit_file_binding != overfit_file_binding
    ):
        raise ValueError(
            "canonical overfit preflight changed during selection"
        )
    revalidated_overfit = _validate_canonical_overfit_binding(
        payload=latest_overfit_payload,
        file_binding=latest_overfit_file_binding,
        receipt=receipt,
        attestation=attestation_payload,
        run_binding=live,
        source_members=source_members,
        data=data,
        model=model,
        strict_preflight_common=attestation["preflight_common"],
    )
    if revalidated_overfit != overfit_evidence:
        raise ValueError(
            "canonical overfit dependencies changed during selection"
        )
    _assert_bound_json_unchanged(
        attestation_path,
        label="CMeIE seed training attestation",
        expected_payload=attestation_payload,
        expected_binding=attestation_file_binding,
    )
    if _candidate_layout(candidate) != candidate_layout:
        raise ValueError(
            "candidate directory changed during canonical selection"
        )
    if _live_run_binding(live) != live:
        raise ValueError("run binding changed during canonical selection")
    return payload


def canonical_selection_path(
    run_binding: RunBinding,
    seed: int,
) -> Path:
    """Return the only legal schema-v3 selection pathname for ``seed``."""

    live = _live_run_binding(run_binding)
    if type(seed) is not int or seed not in ALLOWED_SEEDS:
        raise ValueError("selection seed must be exactly 42, 43, or 44")
    return canonical_run_paths(live).seed_model(seed) / "selection.json"


def publish_canonical_selection_once(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    run_binding: RunBinding,
) -> Path:
    """Publish canonical JSON through a same-directory create-once hard link."""

    live = _live_run_binding(run_binding)
    target = _canonical_absolute(path, label="selection output")
    seed = _integer(
        _required(payload, "seed", label="selection payload"),
        label="selection payload.seed",
    )
    if seed not in ALLOWED_SEEDS:
        raise ValueError("selection payload seed must be 42, 43, or 44")
    expected = canonical_selection_path(live, seed)
    if target != expected:
        raise ValueError(
            "selection output must equal the canonical run selection path"
        )
    if os.path.lexists(target):
        raise FileExistsError(f"checkpoint-selection output exists: {target}")
    if not isinstance(payload, Mapping):
        raise TypeError("selection payload must be a mapping")
    encoded = _canonical_json_bytes(dict(payload))
    target.parent.mkdir(parents=True, exist_ok=True)
    _real_directory(target.parent, label="selection output parent")
    stage = target.parent / (
        f".{target.name}.stage-{os.getpid()}-{secrets.token_hex(12)}"
    )
    descriptor = os.open(
        stage,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("failed to write selection staging file")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        try:
            os.link(stage, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"checkpoint-selection output exists: {target}"
            ) from exc
    finally:
        try:
            os.unlink(stage)
        except FileNotFoundError:
            pass
    actual, binding = read_bound_bytes(
        target,
        "published CMeIE selection receipt",
        reject_multiple_links=True,
    )
    if actual != encoded or binding["sha256"] != hashlib.sha256(encoded).hexdigest():
        raise RuntimeError("published selection receipt content mismatch")
    return target


def load_and_validate_selection_receipt(
    path: str | os.PathLike[str],
    *,
    run_binding: RunBinding,
) -> dict[str, Any]:
    """Load a selection and independently recompute every expected field."""

    live = _live_run_binding(run_binding)
    target = _canonical_absolute(path, label="selection receipt")
    path_seeds = [
        seed
        for seed in sorted(ALLOWED_SEEDS)
        if target == canonical_selection_path(live, seed)
    ]
    if len(path_seeds) != 1:
        raise ValueError(
            "selection receipt path must equal one canonical seed selection "
            "path"
        )
    payload, binding = _stable_bound_json(
        target,
        label="CMeIE selection receipt",
    )
    canonical = _canonical_json_bytes(payload)
    if (
        binding["size"] != len(canonical)
        or binding["sha256"] != hashlib.sha256(canonical).hexdigest()
    ):
        raise ValueError("CMeIE selection receipt is not canonical JSON")
    if payload.get("schema_version") != SELECTION_RECEIPT_SCHEMA_VERSION:
        raise ValueError("CMeIE selection receipt schema_version must be 3")
    if payload.get("kind") != SELECTION_RECEIPT_KIND:
        raise ValueError("CMeIE selection receipt kind is invalid")
    seed = _integer(
        _required(payload, "seed", label="selection receipt"),
        label="selection receipt.seed",
    )
    if seed not in ALLOWED_SEEDS:
        raise ValueError("selection receipt seed must be 42, 43, or 44")
    if seed != path_seeds[0]:
        raise ValueError(
            "selection receipt seed differs from its canonical pathname"
        )
    candidate = canonical_run_paths(live).seed_model(seed)
    if payload.get("candidate_directory") != str(candidate):
        raise ValueError(
            "selection receipt points to a non-canonical candidate directory"
        )
    receipt_path = candidate / "training_receipt.json"
    if payload.get("training_receipt_path") != str(receipt_path):
        raise ValueError(
            "selection receipt points to a non-canonical training receipt"
        )
    attestation_path = canonical_run_paths(live).training_attestation(seed)
    if payload.get("training_attestation_path") != str(attestation_path):
        raise ValueError(
            "selection receipt points to a non-canonical training "
            "attestation"
        )
    overfit_path = canonical_run_paths(live).overfit
    if payload.get("canonical_overfit_preflight_path") != str(overfit_path):
        raise ValueError(
            "selection receipt points to a non-canonical overfit preflight"
        )
    expected = compute_canonical_selection(
        candidate,
        run_binding=live,
    )
    missing = sorted(set(expected) - set(payload))
    extra = sorted(set(payload) - set(expected))
    mismatched = sorted(
        field
        for field in set(expected) & set(payload)
        if payload[field] != expected[field]
    )
    if missing or extra or mismatched:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        if mismatched:
            details.append("mismatched=" + ",".join(mismatched))
        raise ValueError(
            "selection receipt differs from canonical recomputation: "
            + "; ".join(details)
        )
    _assert_bound_json_unchanged(
        target,
        label="CMeIE selection receipt",
        expected_payload=payload,
        expected_binding=binding,
    )
    return payload


__all__ = [
    "DECLARED_GRID_SIZE",
    "SELECTION_RECEIPT_KIND",
    "SELECTION_RECEIPT_SCHEMA_VERSION",
    "TRAINING_ATTESTATION_KIND",
    "TRAINING_ATTESTATION_SCHEMA_VERSION",
    "TRAINING_RECEIPT_SCHEMA_VERSION",
    "canonical_selection_path",
    "compute_canonical_selection",
    "load_and_validate_selection_receipt",
    "publish_canonical_selection_once",
]
