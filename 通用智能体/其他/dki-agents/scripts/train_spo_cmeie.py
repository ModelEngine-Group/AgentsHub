"""Train direct CMeIE text-to-SPO GPLinker candidates from a frozen split."""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from datetime import datetime, timezone
from itertools import product
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dki.evidence_tree import (
    TREE_HASH_ALGORITHM,
    scan_bound_regular_tree,
)
from dki.nlp.cmeie_artifacts import publish_checkpoint_bundle_manifest
from dki.nlp.cmeie_extraction_contract import (
    BINDING_ALGORITHMS,
    publish_extraction_contract_once,
)
from dki.nlp.cmeie_overfit_protocol import (
    validate_overfit_preflight as _shared_validate_overfit_preflight,
)
from dki.nlp.cmeie_preflight_protocol import (
    validate_cmeie_npu_preflight_receipt,
)
from dki.nlp.cmeie_run_protocol import (
    CmeieRunPaths,
    RunBinding,
    canonical_run_paths,
    load_run_binding,
    publish_training_attestation_once,
)
from dki.nlp.cmeie_split_protocol import (
    MANIFEST_FILENAME,
    SCHEMA_VERSION as SPLIT_SCHEMA_VERSION,
    SHARD_FILENAMES,
    load_and_validate_role_shard,
)


FIXED_EPOCHS = 8
FIXED_BATCH_SIZE = 8
FIXED_GRADIENT_ACCUMULATION = 4
FIXED_MAX_LENGTH = 320
FIXED_EARLY_STOPPING_PATIENCE = 2
FIXED_ENCODER_LEARNING_RATE = 2e-5
FIXED_HEAD_LEARNING_RATE = 1e-4
FIXED_HEAD_WARMUP_EPOCHS = 2
FIXED_JOINT_TRAINING_START_EPOCH = 3
FIXED_JOINT_HEAD_LEARNING_RATE_SCALE = 0.1
FIXED_PREFLIGHT_HEAD_LEARNING_RATE = 1e-3
FIXED_PREFLIGHT_STEPS = 400
FIXED_SCHEDULER_WARMUP_RATIO = 0.10
FIXED_WEIGHT_DECAY = 0.01
FIXED_GRADIENT_CLIP = 1.0
FIXED_HEAD_GRADIENT_CLIP = 5.0
FIXED_POINTER_VARIANT = "efficient"
FIXED_HEAD_REDUCTION = "balanced"
FIXED_OVERFIT_F1 = 0.95
FIXED_AMP_INITIAL_SCALE = 1024.0
NO_WEIGHT_DECAY_TERMS = ("bias", "layernorm.weight", "layer_norm.weight")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
TRAINING_RECEIPT_SCHEMA_VERSION = 7
CHECKPOINT_BINDING_FIELDS = (
    "checkpoint_bundle_manifest",
    "checkpoint_bundle_manifest_sha256",
    "checkpoint_bundle_tree_sha256",
    "checkpoint_bundle_n_files",
    "checkpoint_bundle_total_bytes",
    "checkpoint_sha256",
)
EXTRACTION_BINDING_FIELDS = (
    "extraction_contract",
    "extraction_contract_sha256",
    "tokenizer_vocabulary_sha256",
    "label_table_sha256",
    "schema_constraints_sha256",
    "semantic_contract_sha256",
)
REQUIRED_TRAINING_SOURCES = frozenset(
    {
        "dki/evidence_io.py",
        "dki/evidence_tree.py",
        "dki/nlp/cmeie_artifacts.py",
        "dki/nlp/cmeie_extraction_contract.py",
        "dki/nlp/cmeie_gplinker.py",
        "dki/nlp/cmeie_joint_evaluation.py",
        "dki/nlp/cmeie_overfit_protocol.py",
        "dki/nlp/cmeie_preflight_protocol.py",
        "dki/nlp/cmeie_run_protocol.py",
        "dki/nlp/cmeie_selection_protocol.py",
        "dki/nlp/cmeie_split_protocol.py",
        "dki/nlp/cmeie_spo.py",
        "dki/nlp/globalpointer.py",
        "pyproject.toml",
        "scripts/build_cmeie_split_manifest.py",
        "scripts/build_included_tree_manifest.py",
        "scripts/eval_spo_cmeie.py",
        "scripts/eval_spo_cmeie_batch.py",
        "scripts/gate_spo_cmeie_seeds.py",
        "scripts/materialize_hf_snapshot.py",
        "scripts/remote_npu_preflight.py",
        "scripts/select_spo_cmeie_checkpoint.py",
        "scripts/smoke_cmeie_gplinker_overfit.py",
        "scripts/train_spo_cmeie.py",
    }
)


def _publish_directory_create_once(
    staging: Path,
    destination: Path,
) -> None:
    """Atomically publish a same-directory tree without replacing a winner.

    ``os.rename`` is not a create-once primitive on POSIX because it may
    replace an empty destination directory.  Linux exposes the required
    no-replace contract through ``renameat2``; Windows exposes it through
    ``MoveFileExW`` when ``MOVEFILE_REPLACE_EXISTING`` is omitted.
    """

    staging = Path(staging).resolve(strict=True)
    destination = Path(destination)
    destination_parent = destination.parent.resolve(strict=True)
    destination = destination_parent / destination.name
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError("publication staging path must be a real directory")
    if staging.parent != destination_parent:
        raise ValueError(
            "publication staging and destination must share one directory"
        )

    if os.name == "nt":
        move_file = ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        ).MoveFileExW
        move_file.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
        )
        move_file.restype = ctypes.c_int
        if move_file(str(staging), str(destination), 0):
            return
        error = ctypes.get_last_error()
        if os.path.lexists(destination) or error in {80, 183}:
            raise FileExistsError(
                errno.EEXIST,
                "publication destination already exists",
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
                "publication destination already exists",
                str(destination),
            )
        raise OSError(error, os.strerror(error), str(destination))

    raise OSError(
        errno.ENOTSUP,
        "create-once directory publication is unsupported on this platform",
        str(destination),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite_grad_norm_max(values: list[float]) -> float | None:
    """Return a JSON-safe epoch maximum, or ``None`` after any overflow."""

    normalized = [float(value) for value in values]
    if any(not math.isfinite(value) for value in normalized):
        return None
    return max(normalized, default=0.0)


def _strict_training_receipt_json(receipt: dict) -> str:
    """Serialize the formal receipt without permitting NaN or infinities."""

    return json.dumps(
        receipt,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


def _accumulation_group_size(
    micro_batch: int,
    micro_batches: int,
    gradient_accumulation: int,
) -> int:
    """Return the actual divisor for one gradient-accumulation group."""

    if (
        type(micro_batch) is not int
        or type(micro_batches) is not int
        or type(gradient_accumulation) is not int
        or micro_batches < 1
        or gradient_accumulation < 1
        or not 1 <= micro_batch <= micro_batches
    ):
        raise ValueError("invalid gradient-accumulation group coordinates")
    group_start = (
        ((micro_batch - 1) // gradient_accumulation)
        * gradient_accumulation
        + 1
    )
    return min(
        gradient_accumulation,
        micro_batches - group_start + 1,
    )


def _optimizer_step_accounting(
    *,
    planned: int,
    attempts: int,
    skipped: int,
    scheduler_steps: int,
) -> dict[str, int]:
    """Validate and expose planned versus actually executed step counts."""

    values = (planned, attempts, skipped, scheduler_steps)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("optimizer step counts must be non-negative integers")
    if attempts > planned or skipped > attempts:
        raise ValueError("optimizer step counts are inconsistent")
    successful = attempts - skipped
    if scheduler_steps != successful:
        raise ValueError(
            "scheduler steps must equal successful optimizer steps"
        )
    return {
        "total_optimizer_steps": planned,
        "optimizer_step_attempts": attempts,
        "actual_scheduler_steps": scheduler_steps,
        "successful_optimizer_steps": successful,
        "skipped_optimizer_steps": skipped,
    }


def _require_canonical_path(
    supplied: str | os.PathLike[str],
    expected: Path,
    *,
    label: str,
) -> Path:
    raw = os.fspath(supplied)
    if type(raw) is not str or not raw or "\0" in raw:
        raise ValueError(f"{label} must be a non-empty text path")
    if any(
        component in {".", ".."}
        for component in raw.replace("\\", "/").split("/")
    ):
        raise ValueError(f"{label} must not contain lexical traversal")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    canonical = Path(os.path.abspath(raw))
    if os.path.normcase(str(canonical)) != os.path.normcase(str(expected)):
        raise ValueError(
            f"{label} is not the canonical run path: expected {expected}"
        )
    return expected


def _run_binding_fields(binding: RunBinding) -> dict[str, str]:
    return {
        "rid": binding.rid,
        "git_head": binding.git_head,
        "source_tree_sha256": binding.included_tree_sha256,
        "source_manifest_sha256": binding.source_manifest_sha256,
    }


def validate_canonical_training_paths(
    args,
    *,
    execution_root: Path = ROOT,
) -> tuple[RunBinding, CmeieRunPaths]:
    """Load the authoritative run and reject every copied input/output path."""

    binding = load_run_binding(args.run_root)
    paths = canonical_run_paths(binding)
    _require_canonical_path(
        execution_root,
        binding.source_root,
        label="executing source root",
    )
    _require_canonical_path(
        args.source_tree_manifest,
        binding.source_manifest,
        label="source tree manifest",
    )
    _require_canonical_path(
        args.fit_data,
        paths.split_dir / SHARD_FILENAMES["fit"],
        label="fit shard",
    )
    _require_canonical_path(
        args.calibration_data,
        paths.split_dir / SHARD_FILENAMES["calibration"],
        label="calibration shard",
    )
    _require_canonical_path(
        args.split_manifest,
        paths.split_dir / MANIFEST_FILENAME,
        label="split manifest",
    )
    _require_canonical_path(
        args.base_model,
        paths.base_model,
        label="base model",
    )
    _require_canonical_path(
        args.overfit_preflight_receipt,
        paths.overfit,
        label="overfit preflight receipt",
    )
    _require_canonical_path(
        args.output,
        paths.seed_model(args.seed),
        label="candidate output",
    )
    if args.source_tree_sha256.casefold() != (
        binding.included_tree_sha256
    ):
        raise ValueError(
            "source tree SHA-256 differs from the authoritative run binding"
        )
    return binding, paths


def validate_strict_npu_preflight(
    *,
    run_binding: RunBinding,
    paths: CmeieRunPaths,
    source_train_sha256: str,
    base_model_tree: dict,
    smoke_script_sha256: str,
) -> dict:
    """Revalidate the canonical remote preflight before any model work."""

    common = validate_cmeie_npu_preflight_receipt(
        str(paths.preflight_receipt),
        expected_source_tree_sha256=run_binding.included_tree_sha256,
        expected_source_manifest_sha256=(
            run_binding.source_manifest_sha256
        ),
        expected_cmeie_train_sha256=source_train_sha256,
        expected_base_model_canonical_path=str(paths.base_model),
        expected_base_model_tree_sha256=base_model_tree["sha256"],
        expected_run_binding=_run_binding_fields(run_binding),
    )
    if common.get("preflight_receipt_path") != str(
        paths.preflight_receipt
    ):
        raise ValueError("strict preflight receipt path is not canonical")
    if common.get("base_model_canonical_path") != str(paths.base_model):
        raise ValueError("strict preflight base model path is not canonical")
    if (
        common.get("preflight_producer_script_sha256")
        == smoke_script_sha256
    ):
        raise ValueError(
            "remote preflight producer must differ from the overfit smoke "
            "script"
        )
    return common


def _publish_checkpoint_binding(checkpoint: Path) -> dict:
    binding = publish_checkpoint_bundle_manifest(checkpoint)
    if not isinstance(binding, dict):
        raise RuntimeError("checkpoint bundle publisher returned no binding")
    missing = [
        field
        for field in CHECKPOINT_BINDING_FIELDS
        if field not in binding
    ]
    if missing:
        raise RuntimeError(
            "checkpoint bundle binding lacks: " + ", ".join(missing)
        )
    result = {
        field: binding[field]
        for field in CHECKPOINT_BINDING_FIELDS
    }
    if result["checkpoint_bundle_manifest"] != "checkpoint_bundle.json":
        raise RuntimeError("checkpoint bundle manifest name is invalid")
    for field in (
        "checkpoint_bundle_manifest_sha256",
        "checkpoint_bundle_tree_sha256",
        "checkpoint_sha256",
    ):
        value = result[field]
        if (
            not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value)
        ):
            raise RuntimeError(
                f"checkpoint bundle binding {field} is invalid"
            )
    for field, minimum in (
        ("checkpoint_bundle_n_files", 1),
        ("checkpoint_bundle_total_bytes", 0),
    ):
        value = result[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
        ):
            raise RuntimeError(
                f"checkpoint bundle binding {field} is invalid"
            )
    return result


def _publish_extraction_binding(
    checkpoint: Path,
    *,
    tokenizer_vocabulary: dict[str, int],
    entity_labels: list[str],
    predicates: list[str],
    schema_constraints,
) -> dict[str, str]:
    binding = publish_extraction_contract_once(
        checkpoint,
        tokenizer_vocabulary=tokenizer_vocabulary,
        entity_labels=entity_labels,
        predicates=predicates,
        schema_constraints=schema_constraints,
        max_length=FIXED_MAX_LENGTH,
        pointer_variant=FIXED_POINTER_VARIANT,
        head_reduction=FIXED_HEAD_REDUCTION,
        threshold_values=(-0.5, 0.0, 0.5),
        grid_size=54,
    )
    if not isinstance(binding, dict):
        raise RuntimeError("extraction contract publisher returned no binding")
    missing = [
        field for field in EXTRACTION_BINDING_FIELDS if field not in binding
    ]
    if missing:
        raise RuntimeError(
            "extraction contract binding lacks: " + ", ".join(missing)
        )
    result = {
        field: binding[field]
        for field in EXTRACTION_BINDING_FIELDS
    }
    if result["extraction_contract"] != "extraction_contract.json":
        raise RuntimeError("extraction contract filename is invalid")
    for field in EXTRACTION_BINDING_FIELDS[1:]:
        value = result[field]
        if (
            not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value)
        ):
            raise RuntimeError(
                f"extraction contract binding {field} is invalid"
            )
    return result


def _training_receipt_integrity_fields(root: Path = ROOT) -> dict:
    root = Path(root)
    return {
        "schema_version": TRAINING_RECEIPT_SCHEMA_VERSION,
        "evaluator_sha256": _sha(
            root / "scripts" / "eval_spo_cmeie.py"
        ),
        "artifact_integrity_module_sha256": _sha(
            root / "dki" / "nlp" / "cmeie_artifacts.py"
        ),
        "extraction_contract_module_sha256": _sha(
            root / "dki" / "nlp" / "cmeie_extraction_contract.py"
        ),
    }


def _tree_report(directory: Path) -> dict:
    return scan_bound_regular_tree(
        directory,
        label="frozen CMeIE base model",
    )


def _artifact_binding_hashes(
    *,
    tokenizer_vocabulary: dict[str, int],
    entity_labels: list[str],
    predicates: list[str],
    schema_constraints,
) -> dict[str, str]:
    vocabulary = json.dumps(
        sorted(
            (str(token), int(identifier))
            for token, identifier in tokenizer_vocabulary.items()
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    label_table = json.dumps(
        {
            "entity_labels": list(entity_labels),
            "predicates": list(predicates),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    schema = json.dumps(
        schema_constraints,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "tokenizer_vocabulary_sha256": _sha_bytes(vocabulary),
        "label_table_sha256": _sha_bytes(label_table),
        "schema_constraints_sha256": _sha_bytes(schema),
    }


def validate_split_source_tree_binding(
    manifest: dict,
    expected_source_tree_sha256: str,
    expected_source_tree_manifest_sha256: str,
) -> dict[str, str]:
    expected = str(expected_source_tree_sha256).casefold()
    expected_manifest = str(
        expected_source_tree_manifest_sha256
    ).casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("expected source tree SHA-256 is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest):
        raise ValueError("expected source tree manifest SHA-256 is invalid")
    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError(
            f"split manifest schema is not v{SPLIT_SCHEMA_VERSION}"
        )
    actual = str(manifest.get("source_tree_sha256", "")).casefold()
    actual_manifest = str(
        manifest.get("source_tree_manifest_sha256", "")
    ).casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", actual):
        raise ValueError("split manifest source tree SHA-256 is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", actual_manifest):
        raise ValueError(
            "split manifest source tree manifest SHA-256 is invalid"
        )
    if actual != expected:
        raise ValueError(
            "split manifest source tree differs from the candidate source tree"
        )
    if actual_manifest != expected_manifest:
        raise ValueError(
            "split manifest source tree manifest differs from the current file"
        )
    return {
        "source_tree_sha256": actual,
        "source_tree_manifest_sha256": actual_manifest,
    }


def validate_source_tree_manifest(
    manifest: dict,
    *,
    expected_tree_sha256: str,
    root: Path,
) -> dict:
    required = REQUIRED_TRAINING_SOURCES
    if manifest.get("schema_version") != 1:
        raise ValueError("included-tree manifest schema is not v1")
    actual_tree = str(manifest.get("included_tree_sha256", ""))
    if actual_tree != expected_tree_sha256.casefold():
        raise ValueError("included-tree manifest hash differs from CLI binding")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("included-tree manifest has no entries")
    seen = set()
    verified_entries = []
    for entry in entries:
        relative = str(entry.get("path", ""))
        relative_path = PurePosixPath(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or ":" in relative_path.parts[0]
            or relative in seen
        ):
            raise ValueError(f"unsafe or duplicate included path: {relative!r}")
        seen.add(relative)
        path = root.joinpath(*relative.split("/"))
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"included source is not a regular file: {relative}")
        if path.stat().st_size != entry.get("size") or _sha(path) != entry.get(
            "sha256"
        ):
            raise ValueError(f"included source hash mismatch: {relative}")
        verified_entries.append(
            {
                "path": relative,
                "size": entry["size"],
                "sha256": entry["sha256"],
            }
        )
    digest = hashlib.sha256()
    for entry in sorted(verified_entries, key=lambda item: item["path"]):
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    if digest.hexdigest() != actual_tree:
        raise ValueError("included-tree entry digest does not match tree hash")
    missing = sorted(required - seen)
    if missing:
        raise ValueError(
            "included-tree manifest lacks required training sources: "
            + ", ".join(missing)
        )
    return {
        "included_tree_sha256": actual_tree,
        "git_head": manifest.get("git_head"),
        "n_files": len(entries),
    }


def load_isolated_training_data(
    *,
    fit_data: Path,
    calibration_data: Path,
    split_manifest: Path,
    source_tree_sha256: str,
    source_tree_manifest_sha256: str,
) -> tuple[list[dict], list[dict], dict]:
    """Load only fit and calibration v6 shards and bind both validations."""
    arguments = {
        "expected_source_tree_sha256": source_tree_sha256,
        "expected_source_tree_manifest_sha256": (
            source_tree_manifest_sha256
        ),
        "generator_path": (
            ROOT / "scripts" / "build_cmeie_split_manifest.py"
        ),
    }
    fit_manifest, fit, fit_binding = load_and_validate_role_shard(
        split_manifest,
        fit_data,
        role="fit",
        **arguments,
    )
    calibration_manifest, calibration, calibration_binding = (
        load_and_validate_role_shard(
            split_manifest,
            calibration_data,
            role="calibration",
            **arguments,
        )
    )
    if (
        fit_manifest != calibration_manifest
        or fit_binding["split_manifest_sha256"]
        != calibration_binding["split_manifest_sha256"]
        or fit_binding["source_train_sha256"]
        != calibration_binding["source_train_sha256"]
    ):
        raise ValueError(
            "fit and calibration shards are not bound to one manifest"
        )
    return fit, calibration, {
        "fit": fit_binding,
        "calibration": calibration_binding,
    }


def isolated_training_receipt_fields(bindings: dict) -> dict:
    """Return the immutable source/manifest/shard bindings for receipts."""
    fit = bindings["fit"]
    calibration = bindings["calibration"]
    if (
        fit["role"] != "fit"
        or calibration["role"] != "calibration"
        or fit["source_train_sha256"]
        != calibration["source_train_sha256"]
        or fit["split_manifest_sha256"]
        != calibration["split_manifest_sha256"]
    ):
        raise ValueError("isolated training shard bindings are inconsistent")
    return {
        "train_sha256": fit["source_train_sha256"],
        "source_train_sha256": fit["source_train_sha256"],
        "fit_shard_sha256": fit["shard_sha256"],
        "calibration_shard_sha256": calibration["shard_sha256"],
        "split_manifest_sha256": fit["split_manifest_sha256"],
    }


def _device(torch, requested: str):
    if requested == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    raise RuntimeError("the frozen CMeIE protocol requires an available Ascend NPU")


def _uses_weight_decay(name: str) -> bool:
    """Match the proven CMeEE AdamW bias/LayerNorm exclusion policy."""

    lowered = str(name).casefold()
    return not any(term in lowered for term in NO_WEIGHT_DECAY_TERMS)


def build_adamw_parameter_groups(
    *,
    encoder_named_parameters,
    head_named_parameters,
) -> tuple[list[dict], list[dict]]:
    """Return optimizer groups and serialisable evidence for their exact split."""

    groups = []
    evidence = []
    for role, named_parameters, learning_rate in (
        (
            "encoder",
            list(encoder_named_parameters),
            FIXED_ENCODER_LEARNING_RATE,
        ),
        (
            "pointer_heads",
            list(head_named_parameters),
            FIXED_HEAD_LEARNING_RATE,
        ),
    ):
        for decay_enabled in (True, False):
            selected = [
                (name, parameter)
                for name, parameter in named_parameters
                if _uses_weight_decay(name) is decay_enabled
            ]
            if not selected:
                continue
            weight_decay = FIXED_WEIGHT_DECAY if decay_enabled else 0.0
            groups.append(
                {
                    "params": [parameter for _, parameter in selected],
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                }
            )
            evidence.append(
                {
                    "role": role,
                    "decay": decay_enabled,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "parameter_tensors": len(selected),
                    "parameters": sum(
                        int(parameter.numel())
                        for _, parameter in selected
                    ),
                }
            )
    return groups, evidence


def phased_linear_scale(
    step: int,
    *,
    start: int,
    length: int,
    warmup: int,
) -> float:
    """Linear warmup/decay whose clock starts at a phase boundary."""

    relative = int(step) - int(start)
    if relative < 0:
        return 0.0
    length = max(1, int(length))
    warmup = max(0, min(int(warmup), length))
    if warmup and relative < warmup:
        return max(1e-3, relative / warmup)
    decay_length = max(1, length - warmup)
    return max(0.0, (length - relative) / decay_length)


def joint_schedule_contract(
    *,
    optimiser_steps_per_epoch: int,
    epochs: int,
) -> dict:
    """Describe the frozen head-warm-up then joint schedule."""

    steps_per_epoch = max(1, int(optimiser_steps_per_epoch))
    total_steps = steps_per_epoch * int(epochs)
    head_only_steps = steps_per_epoch * FIXED_HEAD_WARMUP_EPOCHS
    joint_steps = total_steps - head_only_steps
    if joint_steps <= 0:
        raise ValueError("phased schedule requires joint optimiser steps")
    return {
        "head_only_optimizer_steps": head_only_steps,
        "joint_optimizer_steps": joint_steps,
        "total_optimizer_steps": total_steps,
        "head_scheduler_warmup_steps": int(
            total_steps * FIXED_SCHEDULER_WARMUP_RATIO
        ),
        "encoder_scheduler_warmup_steps": int(
            joint_steps * FIXED_SCHEDULER_WARMUP_RATIO
        ),
    }


def configure_training_phase(model, encoder_params, *, head_only: bool) -> None:
    """Freeze or unfreeze the encoder while keeping its dropout disabled."""

    for parameter in encoder_params:
        parameter.requires_grad_(not head_only)
    model.train()
    model.encoder.eval()
    if not model.training or model.encoder.training:
        raise RuntimeError("CMeIE phased model mode contract was not applied")


def clip_grouped_gradients(
    torch,
    encoder_params,
    head_params,
    *,
    head_only: bool,
):
    """Clip pointer heads independently so encoder scale cannot starve them."""

    encoder_params = list(encoder_params)
    head_params = list(head_params)
    if not encoder_params or not head_params:
        raise ValueError("grouped gradient clipping requires both groups")
    head_norm = torch.nn.utils.clip_grad_norm_(
        head_params,
        FIXED_HEAD_GRADIENT_CLIP,
    )
    encoder_norm = None
    if not head_only:
        encoder_norm = torch.nn.utils.clip_grad_norm_(
            encoder_params,
            FIXED_GRADIENT_CLIP,
        )
    return {
        "encoder": encoder_norm,
        "pointer_heads": head_norm,
    }


def amp_optimizer_step_succeeded(
    scale_before: float,
    scale_after: float,
) -> bool:
    """A decreased GradScaler scale means the optimizer step was skipped."""

    return float(scale_after) >= float(scale_before)


def validate_fixed_training_arguments(args) -> None:
    expected = {
        "epochs": FIXED_EPOCHS,
        "batch_size": FIXED_BATCH_SIZE,
        "gradient_accumulation": FIXED_GRADIENT_ACCUMULATION,
        "max_length": FIXED_MAX_LENGTH,
        "early_stopping_patience": FIXED_EARLY_STOPPING_PATIENCE,
        "device": "npu",
        "fp16": True,
    }
    mismatches = [
        f"{name}={getattr(args, name)!r}, expected {value!r}"
        for name, value in expected.items()
        if getattr(args, name) != value
    ]
    if mismatches:
        raise ValueError(
            "CMeIE fixed training protocol mismatch: " + "; ".join(mismatches)
        )


def validate_overfit_preflight(
    payload: dict,
    *,
    receipt_sha256: str,
    source_train_sha256: str,
    fit_shard_sha256: str,
    split_manifest_sha256: str,
    base_model: str,
    base_model_tree: dict,
    source_tree_manifest_sha256: str,
    preflight_script_sha256: str,
    model_module_sha256: str,
    alignment_module_sha256: str,
    globalpointer_module_sha256: str,
    tokenizer_vocabulary_sha256: str,
    label_table_sha256: str,
    schema_constraints_sha256: str,
    source_tree_sha256: str,
    preflight_common: dict,
) -> dict:
    """Validate the fit-only proof that authorizes full CMeIE training."""

    return _shared_validate_overfit_preflight(
        payload,
        receipt_sha256=receipt_sha256,
        source_train_sha256=source_train_sha256,
        fit_shard_sha256=fit_shard_sha256,
        split_manifest_sha256=split_manifest_sha256,
        base_model=base_model,
        base_model_tree=base_model_tree,
        source_tree_manifest_sha256=source_tree_manifest_sha256,
        preflight_script_sha256=preflight_script_sha256,
        model_module_sha256=model_module_sha256,
        alignment_module_sha256=alignment_module_sha256,
        globalpointer_module_sha256=globalpointer_module_sha256,
        tokenizer_vocabulary_sha256=tokenizer_vocabulary_sha256,
        label_table_sha256=label_table_sha256,
        schema_constraints_sha256=schema_constraints_sha256,
        source_tree_sha256=source_tree_sha256,
        preflight_common=preflight_common,
    )

def training_attestation_evidence(
    *,
    training_receipt_sha256: str,
    history: list[dict],
    preflight_common: dict,
    run_binding: RunBinding,
    isolated_data_binding: dict,
    configuration: dict,
    overfit_preflight: dict,
) -> dict:
    """Build the complete external binding for a published seed model."""

    epoch_fields = (
        "epoch",
        *CHECKPOINT_BINDING_FIELDS,
        *EXTRACTION_BINDING_FIELDS,
    )
    epoch_bindings = []
    for item in history:
        missing = [
            field for field in epoch_fields if field not in item
        ]
        if missing:
            raise ValueError(
                "training history lacks attested epoch fields: "
                + ", ".join(missing)
            )
        epoch_bindings.append(
            {field: item[field] for field in epoch_fields}
        )
    if not epoch_bindings:
        raise ValueError("training history has no epochs to attest")
    return {
        "training_receipt_sha256": training_receipt_sha256,
        "epoch_bindings": epoch_bindings,
        "preflight_common": preflight_common,
        "source_binding": {
            **_run_binding_fields(run_binding),
            "source_manifest_path": str(run_binding.source_manifest),
        },
        "data_binding": dict(isolated_data_binding),
        "configuration": dict(configuration),
        "overfit_preflight_sha256": str(
            overfit_preflight["path_sha256"]
        ),
    }


def advance_early_stopping(best_rank, stale_epochs: int, current_rank):
    """Advance patience without killing a sparse decoder before first signal.

    Exact SPO F1 commonly stays at zero while the pointer heads learn to cross
    the decoding threshold.  A zero plateau is therefore a warm-up state, not
    evidence that a later epoch cannot improve.  Once any positive F1 has been
    observed, the configured patience applies normally.
    """
    if best_rank is None or current_rank > best_rank:
        return current_rank, 0
    if best_rank[0] <= 0.0:
        return best_rank, 0
    return best_rank, stale_epochs + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overfit-preflight-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=[42, 43, 44])
    parser.add_argument(
        "--entity-label-mode",
        choices=("typed", "role_only"),
        default="typed",
    )
    parser.add_argument("--epochs", type=int, default=FIXED_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FIXED_BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=FIXED_GRADIENT_ACCUMULATION,
    )
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=FIXED_MAX_LENGTH)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=FIXED_EARLY_STOPPING_PATIENCE,
    )
    parser.add_argument("--device", choices=["npu"], default="npu")
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument(
        "--source-tree-manifest",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.source_tree_sha256):
        raise SystemExit("--source-tree-sha256 must be a full SHA-256")
    if args.calibration_batch_size < 1:
        raise SystemExit("--calibration-batch-size must be positive")
    try:
        validate_fixed_training_arguments(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        run_binding, run_paths = validate_canonical_training_paths(args)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid canonical CMeIE run binding: {exc}") from exc
    if os.path.lexists(args.output):
        raise SystemExit(f"candidate output already exists: {args.output}")
    try:
        source_tree_payload = json.loads(
            args.source_tree_manifest.read_text(encoding="utf-8")
        )
        source_tree_binding = validate_source_tree_manifest(
            source_tree_payload,
            expected_tree_sha256=run_binding.included_tree_sha256,
            root=run_binding.source_root,
        )
        source_tree_manifest_sha256 = _sha(
            args.source_tree_manifest
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ) as exc:
        raise SystemExit(f"invalid included-tree manifest: {exc}") from exc
    try:
        (
            fit,
            calibration,
            split_manifest_binding,
        ) = load_isolated_training_data(
            fit_data=args.fit_data,
            calibration_data=args.calibration_data,
            split_manifest=args.split_manifest,
            source_tree_sha256=args.source_tree_sha256,
            source_tree_manifest_sha256=source_tree_manifest_sha256,
        )
        isolated_data_binding = isolated_training_receipt_fields(
            split_manifest_binding
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(
            f"invalid isolated CMeIE fit/calibration shards: {exc}"
        ) from exc
    try:
        overfit_payload = json.loads(
            args.overfit_preflight_receipt.read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"cannot read overfit preflight receipt: {exc}") from exc
    try:
        base_model_path = run_paths.base_model.resolve(strict=True)
        if base_model_path != run_paths.base_model:
            raise ValueError(
                "base model resolves away from its canonical run path"
            )
        base_model_tree = _tree_report(base_model_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot bind local base model tree: {exc}") from exc
    if not fit:
        raise SystemExit("frozen fit split is empty")

    smoke_script_sha256 = _sha(
        ROOT / "scripts" / "smoke_cmeie_gplinker_overfit.py"
    )
    try:
        preflight_common = validate_strict_npu_preflight(
            run_binding=run_binding,
            paths=run_paths,
            source_train_sha256=isolated_data_binding[
                "source_train_sha256"
            ],
            base_model_tree=base_model_tree,
            smoke_script_sha256=smoke_script_sha256,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"strict NPU preflight is not admissible: {exc}") from exc

    from transformers import AutoTokenizer
    from dki.nlp.cmeie_spo import (
        labels_from_fit,
        schema_constraints_from_fit,
    )

    entity_labels, predicates = labels_from_fit(
        fit,
        entity_label_mode=args.entity_label_mode,
    )
    schema_constraints = schema_constraints_from_fit(
        fit,
        entity_label_mode=args.entity_label_mode,
    )
    entity2id = {label: index for index, label in enumerate(entity_labels)}
    predicate2id = {label: index for index, label in enumerate(predicates)}
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise SystemExit("GPLinker training requires a fast tokenizer")
    binding_hashes = _artifact_binding_hashes(
        tokenizer_vocabulary=tokenizer.get_vocab(),
        entity_labels=entity_labels,
        predicates=predicates,
        schema_constraints=schema_constraints,
    )
    try:
        overfit_binding = validate_overfit_preflight(
            overfit_payload,
            receipt_sha256=_sha(args.overfit_preflight_receipt),
            source_train_sha256=isolated_data_binding[
                "source_train_sha256"
            ],
            fit_shard_sha256=isolated_data_binding["fit_shard_sha256"],
            split_manifest_sha256=_sha(args.split_manifest),
            base_model=str(base_model_path),
            base_model_tree=base_model_tree,
            source_tree_manifest_sha256=source_tree_manifest_sha256,
            preflight_script_sha256=smoke_script_sha256,
            model_module_sha256=_sha(
                ROOT / "dki" / "nlp" / "cmeie_gplinker.py"
            ),
            alignment_module_sha256=_sha(
                ROOT / "dki" / "nlp" / "cmeie_spo.py"
            ),
            globalpointer_module_sha256=_sha(
                ROOT / "dki" / "nlp" / "globalpointer.py"
            ),
            tokenizer_vocabulary_sha256=binding_hashes[
                "tokenizer_vocabulary_sha256"
            ],
            label_table_sha256=binding_hashes["label_table_sha256"],
            schema_constraints_sha256=binding_hashes[
                "schema_constraints_sha256"
            ],
            source_tree_sha256=args.source_tree_sha256,
            preflight_common=preflight_common,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    from dki.nlp.cmeie_gplinker import CMeIEGPLinker, gplinker_loss
    from dki.nlp.cmeie_spo import (
        build_labels,
        decode_spo_grid_sparse,
        gold_spo_set,
        sparse_join_match_lower_bound,
        sparsify_pointer_logits,
    )

    random.seed(args.seed); torch.manual_seed(args.seed)
    try:
        device = _device(torch, args.device)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    threshold_values = (-0.5, 0.0, 0.5)
    threshold_candidates = [
        (thresholds, schema_on)
        for thresholds in product(threshold_values, repeat=3)
        for schema_on in (False, True)
    ]

    def calibration_selection():
        model.eval()
        join_safety_cap = 100_000
        complexity = {
            "join_safety_cap": join_safety_cap,
            "rows_checked": 0,
            "maximum_match_lower_bound": 0,
        }
        counts = {
            candidate: {"tp": 0, "fp": 0, "fn": 0}
            for candidate in threshold_candidates
        }
        with torch.no_grad():
            for start in range(0, len(calibration), args.calibration_batch_size):
                batch = calibration[start : start + args.calibration_batch_size]
                encoded = tokenizer(
                    [row["text"] for row in batch],
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                )
                offsets = [
                    [tuple(pair) for pair in mapping.tolist()]
                    for mapping in encoded.pop("offset_mapping")
                ]
                inputs = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask", "token_type_ids"}}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16):
                    outputs = model(**inputs)
                for index, row in enumerate(batch):
                    sparse = {
                        name: sparsify_pointer_logits(
                            values[index],
                            threshold_values[0],
                        )
                        for name, values in outputs.items()
                    }
                    estimate = sparse_join_match_lower_bound(
                        offsets[index],
                        sparse["entity"],
                        sparse["relation_head"],
                        sparse["relation_tail"],
                        entity_labels,
                        (
                            threshold_values[0],
                            threshold_values[0],
                            threshold_values[0],
                        ),
                        join_safety_cap,
                    )
                    complexity["rows_checked"] += 1
                    complexity["maximum_match_lower_bound"] = max(
                        complexity["maximum_match_lower_bound"],
                        int(estimate["match_lower_bound"]),
                    )
                    if estimate["capped"]:
                        return {
                            "status": "deferred_candidate_explosion",
                            "grid_size": len(threshold_candidates),
                            "complexity": complexity,
                            "selection_used": False,
                        }
                    decoded = decode_spo_grid_sparse(
                        row["text"],
                        offsets[index],
                        sparse["entity"],
                        sparse["relation_head"],
                        sparse["relation_tail"],
                        entity_labels,
                        predicates,
                        threshold_values,
                        schema_constraints,
                    )
                    gold = gold_spo_set(row)
                    for candidate, predicted in decoded.items():
                        count = counts[candidate]
                        count["tp"] += len(predicted & gold)
                        count["fp"] += len(predicted - gold)
                        count["fn"] += len(gold - predicted)
        scored = []
        for (thresholds, schema_on), count in counts.items():
            precision = count["tp"] / max(1, count["tp"] + count["fp"])
            recall = count["tp"] / max(1, count["tp"] + count["fn"])
            metric = {
                **count,
                "precision": precision,
                "recall": recall,
                "f1": 2
                * precision
                * recall
                / max(1e-12, precision + recall),
            }
            scored.append((metric, thresholds, schema_on))
        metric, thresholds, schema_on = max(
            scored,
            key=lambda item: (
                item[0]["f1"],
                item[0]["precision"],
                sum(item[1]),
                item[2],
            ),
        )
        return {
            "status": "completed",
            "thresholds": list(thresholds),
            "schema_constraint": "on" if schema_on else "off",
            "metric": metric,
            "grid_size": len(threshold_candidates),
            "tie_break": [
                "micro_f1_desc",
                "precision_desc",
                "threshold_sum_desc",
                "schema_on_first",
            ],
            "complexity": complexity,
            "selection_used": True,
        }

    def collate(batch):
        encoded = tokenizer([row["text"] for row in batch], return_offsets_mapping=True, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        labels, dropped = [], 0
        for index, row in enumerate(batch):
            one, lost = build_labels(
                row,
                [
                    tuple(pair)
                    for pair in encoded["offset_mapping"][index].tolist()
                ],
                entity2id,
                predicate2id,
                allow_cross_sentence_fallback=True,
            )
            labels.append(one); dropped += lost
        result = {key: torch.stack([one[key] for one in labels]) for key in labels[0]}
        encoded.pop("offset_mapping")
        return encoded, result, dropped

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        fit,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=loader_generator,
        num_workers=0,
    )
    encoder = AutoModel.from_pretrained(str(base_model_path))
    model = CMeIEGPLinker(
        encoder,
        int(encoder.config.hidden_size),
        len(entity_labels),
        len(predicates),
        pointer_variant=FIXED_POINTER_VARIANT,
    ).to(device)
    encoder_named = list(model.encoder.named_parameters())
    head_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    encoder_params = [parameter for _, parameter in encoder_named]
    head_params = [parameter for _, parameter in head_named]
    parameter_groups, optimiser_group_evidence = (
        build_adamw_parameter_groups(
            encoder_named_parameters=encoder_named,
            head_named_parameters=head_named,
        )
    )
    optimiser = torch.optim.AdamW(parameter_groups)

    # Mirror the successful CMeEE optimisation recipe: establish usable
    # pointer heads first, then unfreeze the encoder with a smaller head LR and
    # independent gradient clipping.  One optimizer preserves AdamW state.
    optimiser_steps_per_epoch = max(
        1,
        (
            len(loader)
            + args.gradient_accumulation
            - 1
        )
        // args.gradient_accumulation,
    )
    schedule_contract = joint_schedule_contract(
        optimiser_steps_per_epoch=optimiser_steps_per_epoch,
        epochs=args.epochs,
    )
    total_steps = schedule_contract["total_optimizer_steps"]
    head_only_steps = schedule_contract["head_only_optimizer_steps"]
    head_warmup_steps = schedule_contract[
        "head_scheduler_warmup_steps"
    ]
    joint_steps = schedule_contract["joint_optimizer_steps"]
    encoder_warmup_steps = schedule_contract[
        "encoder_scheduler_warmup_steps"
    ]
    schedule_lambdas = []
    for group in optimiser_group_evidence:
        if group["role"] == "encoder":
            schedule_lambdas.append(
                lambda step, start=head_only_steps, length=joint_steps,
                warmup=encoder_warmup_steps: phased_linear_scale(
                    step,
                    start=start,
                    length=length,
                    warmup=warmup,
                )
            )
        else:
            schedule_lambdas.append(
                lambda step, length=total_steps, warmup=head_warmup_steps,
                boundary=head_only_steps: (
                    phased_linear_scale(
                        step,
                        start=0,
                        length=length,
                        warmup=warmup,
                    )
                    * (
                        FIXED_JOINT_HEAD_LEARNING_RATE_SCALE
                        if step >= boundary
                        else 1.0
                    )
                )
            )
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule_lambdas)
    if not (
        hasattr(torch, "npu")
        and hasattr(torch.npu, "amp")
        and hasattr(torch.npu.amp, "GradScaler")
    ):
        raise RuntimeError(
            "NPU FP16 protocol requires torch.npu.amp.GradScaler"
        )
    scaler = torch.npu.amp.GradScaler(
        init_scale=FIXED_AMP_INITIAL_SCALE,
        enabled=args.fp16,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    history = []
    dropped_by_epoch = []
    best_calibration_rank = None
    stale_epochs = 0
    total_skipped_optimizer_steps = 0
    total_optimizer_step_attempts = 0
    total_scheduler_steps = 0
    try:
        for epoch in range(1, args.epochs + 1):
            head_only = epoch <= FIXED_HEAD_WARMUP_EPOCHS
            configure_training_phase(
                model,
                encoder_params,
                head_only=head_only,
            )
            phase = "head_only" if head_only else "joint"
            optimiser.zero_grad(set_to_none=True)
            loss_sum = 0.0
            epoch_dropped = 0
            epoch_skipped_optimizer_steps = 0
            epoch_nonfinite_grad_norm_steps = 0
            epoch_optimizer_step_attempts = 0
            encoder_grad_norms = []
            head_grad_norms = []
            scaler_start = float(scaler.get_scale())
            for step, (inputs, labels, dropped) in enumerate(loader, start=1):
                epoch_dropped += dropped
                inputs = {key: value.to(device) for key, value in inputs.items() if key in {"input_ids", "attention_mask", "token_type_ids"}}
                labels = {key: value.to(device) for key, value in labels.items()}
                accumulation_divisor = _accumulation_group_size(
                    step,
                    len(loader),
                    args.gradient_accumulation,
                )
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16):
                    unscaled_loss = gplinker_loss(
                        model(**inputs),
                        labels,
                        head_reduction=FIXED_HEAD_REDUCTION,
                    )
                    loss = unscaled_loss / accumulation_divisor
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite CMeIE loss at epoch={epoch} step={step}"
                    )
                scaler.scale(loss).backward()
                unscaled_loss_value = float(unscaled_loss.detach().cpu())
                loss_sum += unscaled_loss_value
                if step % args.gradient_accumulation == 0 or step == len(loader):
                    scaler.unscale_(optimiser)
                    group_norms = clip_grouped_gradients(
                        torch,
                        encoder_params,
                        head_params,
                        head_only=head_only,
                    )
                    head_norm_value = float(
                        group_norms["pointer_heads"].detach().cpu()
                    )
                    encoder_norm_value = (
                        None
                        if group_norms["encoder"] is None
                        else float(
                            group_norms["encoder"].detach().cpu()
                        )
                    )
                    head_grad_norms.append(head_norm_value)
                    if encoder_norm_value is not None:
                        encoder_grad_norms.append(encoder_norm_value)
                    nonfinite_grad_norm = (
                        not math.isfinite(head_norm_value)
                        or (
                            encoder_norm_value is not None
                            and not math.isfinite(encoder_norm_value)
                        )
                    )
                    if nonfinite_grad_norm:
                        epoch_nonfinite_grad_norm_steps += 1
                    scale_before = float(scaler.get_scale())
                    scaler.step(optimiser)
                    scaler.update()
                    scale_after = float(scaler.get_scale())
                    epoch_optimizer_step_attempts += 1
                    total_optimizer_step_attempts += 1
                    if amp_optimizer_step_succeeded(
                        scale_before,
                        scale_after,
                    ):
                        schedule.step()
                        total_scheduler_steps += 1
                    else:
                        epoch_skipped_optimizer_steps += 1
                        total_skipped_optimizer_steps += 1
                    if (
                        nonfinite_grad_norm
                        and amp_optimizer_step_succeeded(
                            scale_before,
                            scale_after,
                        )
                    ):
                        raise FloatingPointError(
                            "GradScaler did not reject non-finite gradients "
                            f"at epoch={epoch} step={step}"
                        )
                    optimiser.zero_grad(set_to_none=True)
                    if (
                        epoch_optimizer_step_attempts == 1
                        or epoch_optimizer_step_attempts % 100 == 0
                        or step == len(loader)
                    ):
                        print(
                            json.dumps(
                                {
                                    "event": "cmeie_training_progress",
                                    "epoch": epoch,
                                    "phase": phase,
                                    "micro_batch": step,
                                    "micro_batches": len(loader),
                                    "optimizer_step_attempt": (
                                        epoch_optimizer_step_attempts
                                    ),
                                    "loss": unscaled_loss_value,
                                    "gradient_accumulation_divisor": (
                                        accumulation_divisor
                                    ),
                                    "encoder_pre_clip_grad_norm": (
                                        encoder_norm_value
                                    ),
                                    "head_pre_clip_grad_norm": (
                                        head_norm_value
                                    ),
                                    "amp_scale": scale_after,
                                    "optimizer_step_succeeded": (
                                        amp_optimizer_step_succeeded(
                                            scale_before,
                                            scale_after,
                                        )
                                    ),
                                    "learning_rates": [
                                        float(group["lr"])
                                        for group in optimiser.param_groups
                                    ],
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            flush=True,
                        )
            if epoch_optimizer_step_attempts != optimiser_steps_per_epoch:
                raise RuntimeError(
                    "optimizer step count differs from frozen schedule: "
                    f"epoch={epoch} actual={epoch_optimizer_step_attempts} "
                    f"expected={optimiser_steps_per_epoch}"
                )
            checkpoint = staging / f"epoch-{epoch:02d}"
            checkpoint.mkdir()
            model.encoder.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            torch.save(
                {
                    "model": model.state_dict(),
                    "entity_labels": entity_labels,
                    "predicates": predicates,
                    "entity_label_mode": args.entity_label_mode,
                    "pointer_variant": FIXED_POINTER_VARIANT,
                    "head_reduction": FIXED_HEAD_REDUCTION,
                    "max_length": FIXED_MAX_LENGTH,
                },
                checkpoint / "gplinker.pt",
            )
            extraction_binding = _publish_extraction_binding(
                checkpoint,
                tokenizer_vocabulary=tokenizer.get_vocab(),
                entity_labels=entity_labels,
                predicates=predicates,
                schema_constraints=schema_constraints,
            )
            for field in (
                "tokenizer_vocabulary_sha256",
                "label_table_sha256",
                "schema_constraints_sha256",
            ):
                if extraction_binding[field] != binding_hashes[field]:
                    raise RuntimeError(
                        f"epoch extraction contract {field} changed"
                    )
            checkpoint_binding = _publish_checkpoint_binding(checkpoint)
            history.append({
                "epoch": epoch, "mean_train_loss": loss_sum / max(1, len(loader)),
                **extraction_binding,
                **checkpoint_binding,
                "dropped_label_count": epoch_dropped,
                "phase": phase,
                "encoder_mode": "eval_with_gradients" if not head_only else "eval_frozen",
                "encoder_trainable": not head_only,
                "encoder_pre_clip_grad_norm_max": (
                    _finite_grad_norm_max(encoder_grad_norms)
                    if not head_only
                    else None
                ),
                "head_pre_clip_grad_norm_max": (
                    _finite_grad_norm_max(head_grad_norms)
                ),
                "amp_scale_start": scaler_start,
                "amp_scale_end": float(scaler.get_scale()),
                "skipped_optimizer_steps": (
                    epoch_skipped_optimizer_steps
                ),
                "nonfinite_grad_norm_steps": (
                    epoch_nonfinite_grad_norm_steps
                ),
                "optimizer_steps_expected": (
                    optimiser_steps_per_epoch
                ),
                "optimizer_step_attempts": (
                    epoch_optimizer_step_attempts
                ),
                "successful_optimizer_steps": (
                    epoch_optimizer_step_attempts
                    - epoch_skipped_optimizer_steps
                ),
                "learning_rates_end": [
                    float(group["lr"])
                    for group in optimiser.param_groups
                ],
            })
            dropped_by_epoch.append(epoch_dropped)
            if len(set(dropped_by_epoch)) != 1:
                raise RuntimeError(
                    "dropped-label count changed between full fit epochs"
                )
            selection = calibration_selection()
            history[-1]["calibration_declared_grid_selection"] = selection
            if selection["status"] != "completed":
                print(json.dumps(history[-1], ensure_ascii=False), flush=True)
                continue
            metric = selection["metric"]
            rank = (
                metric["f1"],
                metric["precision"],
                sum(selection["thresholds"]),
                selection["schema_constraint"] == "on",
            )
            best_calibration_rank, stale_epochs = advance_early_stopping(
                best_calibration_rank,
                stale_epochs,
                rank,
            )
            print(json.dumps(history[-1], ensure_ascii=False), flush=True)
            if stale_epochs >= args.early_stopping_patience:
                print(
                    json.dumps(
                        {
                            "early_stopping": True,
                            "epoch": epoch,
                            "best_calibration_f1": best_calibration_rank[0],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                break
        if best_calibration_rank is None:
            raise RuntimeError(
                "all calibration attempts were deferred by the exact-grid "
                "candidate-explosion safety gate"
            )
        receipt = {
            **_training_receipt_integrity_fields(),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_role": "fit_training_calibration_selection",
            "outer_accessed": False,
            "official_dev_accessed": False,
            "seed": args.seed,
            "rid": run_binding.rid,
            "git_head": run_binding.git_head,
            "tree": run_binding.included_tree_sha256,
            "run_root": str(run_binding.run_root),
            "source_manifest_path": str(run_binding.source_manifest),
            "source_tree_sha256": run_binding.included_tree_sha256,
            "source_tree_manifest_sha256": (
                run_binding.source_manifest_sha256
            ),
            "source_tree": source_tree_binding,
            **isolated_data_binding,
            "split_manifest_validation": split_manifest_binding,
            "base_model": str(base_model_path),
            "base_model_tree": base_model_tree,
            "max_length": args.max_length, "epochs": args.epochs,
            "batch_size": args.batch_size, "gradient_accumulation": args.gradient_accumulation,
            "calibration_batch_size": args.calibration_batch_size,
            "fp16": args.fp16, "device": str(device),
            "overfit_preflight": overfit_binding,
            "preflight_common": preflight_common,
            "training_protocol": {
                "entity_label_mode": args.entity_label_mode,
                "pointer_variant": FIXED_POINTER_VARIANT,
                "loss": "sparse_multilabel_categorical_crossentropy",
                "head_reduction": FIXED_HEAD_REDUCTION,
                "group_reduction": "equal_mean_entity_relation_head_relation_tail",
                "authorizing_fit_only_preflight": overfit_binding,
                "head_only_warmup": {
                    "enabled": True,
                    "epochs": FIXED_HEAD_WARMUP_EPOCHS,
                    "separate_optimizer": False,
                    "encoder_mode": "eval_frozen",
                    "head_learning_rate": FIXED_HEAD_LEARNING_RATE,
                },
                "joint_training": {
                    "starts_at_epoch": FIXED_JOINT_TRAINING_START_EPOCH,
                    "starts_at_optimizer_step": head_only_steps + 1,
                    "encoder_learning_rate": FIXED_ENCODER_LEARNING_RATE,
                    "head_learning_rate": (
                        FIXED_HEAD_LEARNING_RATE
                        * FIXED_JOINT_HEAD_LEARNING_RATE_SCALE
                    ),
                    "head_learning_rate_scale": (
                        FIXED_JOINT_HEAD_LEARNING_RATE_SCALE
                    ),
                    "weight_decay": FIXED_WEIGHT_DECAY,
                    "weight_decay_policy": (
                        "exclude_bias_and_layernorm"
                    ),
                    "optimizer_parameter_groups": (
                        optimiser_group_evidence
                    ),
                    "gradient_clip": FIXED_GRADIENT_CLIP,
                    "encoder_gradient_clip": FIXED_GRADIENT_CLIP,
                    "head_gradient_clip": FIXED_HEAD_GRADIENT_CLIP,
                    "gradient_clip_scope": "separate_encoder_and_pointer_heads",
                    "scheduler": "linear",
                    "scheduler_warmup_ratio": (
                        FIXED_SCHEDULER_WARMUP_RATIO
                    ),
                    "head_scheduler_warmup_steps": (
                        head_warmup_steps
                    ),
                    "encoder_scheduler_warmup_steps": (
                        encoder_warmup_steps
                    ),
                    "head_only_optimizer_steps": schedule_contract[
                        "head_only_optimizer_steps"
                    ],
                    "joint_optimizer_steps": schedule_contract[
                        "joint_optimizer_steps"
                    ],
                    "optimizer_steps_per_epoch": (
                        optimiser_steps_per_epoch
                    ),
                    **_optimizer_step_accounting(
                        planned=total_steps,
                        attempts=total_optimizer_step_attempts,
                        skipped=total_skipped_optimizer_steps,
                        scheduler_steps=total_scheduler_steps,
                    ),
                    "completed_epochs": len(history),
                    "effective_batch_size": (
                        args.batch_size * args.gradient_accumulation
                    ),
                    "encoder_mode": "eval_with_gradients",
                    "amp": {
                        "autocast": True,
                        "dtype": "float16",
                        "grad_scaler": True,
                        "initial_scale": FIXED_AMP_INITIAL_SCALE,
                        "unscale_before_gradient_clip": True,
                    },
                },
            },
            "early_stopping": {
                "patience": args.early_stopping_patience,
                "zero_f1_warmup_consumes_patience": False,
                "threshold_grid": list(threshold_values),
                "schema_constraints": ["off", "on"],
                "grid_size": len(threshold_candidates),
                "best_calibration_f1": best_calibration_rank[0],
                "candidate_explosion_policy": {
                    "join_match_lower_bound_cap": 100_000,
                    "deferred_epochs_are_not_selection_used": True,
                },
            },
            "alignment_policy": {
                "same_sentence_preferred": True,
                "deterministic_minimum_cost_monotonic_edge_cover": True,
                "closest_pair_only": False,
                "multiple_surface_occurrences_retained": True,
                "cross_sentence_gold_fallback": (
                    "single_closest_pair_only"
                ),
                "overlapping_token_offsets": True,
            },
            "entity_labels": entity_labels,
            "entity_label_mode": args.entity_label_mode,
            "predicates": predicates,
            "binding_algorithms": dict(BINDING_ALGORITHMS),
            "label_table_sha256": binding_hashes[
                "label_table_sha256"
            ],
            "tokenizer_vocabulary_sha256": binding_hashes[
                "tokenizer_vocabulary_sha256"
            ],
            "schema_constraints": schema_constraints,
            "schema_constraints_sha256": binding_hashes[
                "schema_constraints_sha256"
            ],
            "dropped_label_count": dropped_by_epoch[0],
            "dropped_label_counts_by_epoch": dropped_by_epoch,
            "history": history, "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "training_script_sha256": _sha(Path(__file__)),
            "model_module_sha256": _sha(ROOT / "dki" / "nlp" / "cmeie_gplinker.py"),
            "alignment_module_sha256": _sha(
                ROOT / "dki" / "nlp" / "cmeie_spo.py"
            ),
            "globalpointer_module_sha256": _sha(
                ROOT / "dki" / "nlp" / "globalpointer.py"
            ),
        }
        (staging / "training_receipt.json").write_text(
            _strict_training_receipt_json(receipt),
            encoding="utf-8",
        )
        _publish_directory_create_once(staging, args.output)
    except Exception as exc:
        failure = staging / "training_failed.json"
        if staging.is_dir() and not failure.exists():
            failure.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "failed_staging_preserved",
                        "failure_class": type(exc).__name__,
                        "failed_utc": datetime.now(timezone.utc).isoformat(),
                        "rid": run_binding.rid,
                        "git_head": run_binding.git_head,
                        "tree": run_binding.included_tree_sha256,
                        "run_root": str(run_binding.run_root),
                        "source_manifest_path": str(
                            run_binding.source_manifest
                        ),
                        "source_tree_sha256": (
                            run_binding.included_tree_sha256
                        ),
                        "source_tree_manifest_sha256": (
                            run_binding.source_manifest_sha256
                        ),
                        **isolated_data_binding,
                        "preflight_common": preflight_common,
                        "overfit_preflight_sha256": overfit_binding[
                            "path_sha256"
                        ],
                        "outer_accessed": False,
                        "official_dev_accessed": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        raise
    published_receipt = args.output / "training_receipt.json"
    attestation_evidence = training_attestation_evidence(
        training_receipt_sha256=_sha(published_receipt),
        history=history,
        preflight_common=preflight_common,
        run_binding=run_binding,
        isolated_data_binding=isolated_data_binding,
        configuration={
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "calibration_batch_size": args.calibration_batch_size,
            "max_length": args.max_length,
            "fp16": args.fp16,
            "device": args.device,
            "pointer_variant": FIXED_POINTER_VARIANT,
            "head_reduction": FIXED_HEAD_REDUCTION,
            "encoder_learning_rate": FIXED_ENCODER_LEARNING_RATE,
            "head_learning_rate": (
                FIXED_HEAD_LEARNING_RATE
                * FIXED_JOINT_HEAD_LEARNING_RATE_SCALE
            ),
            "head_warmup_epochs": FIXED_HEAD_WARMUP_EPOCHS,
            "head_warmup_learning_rate": FIXED_HEAD_LEARNING_RATE,
            "joint_head_learning_rate_scale": (
                FIXED_JOINT_HEAD_LEARNING_RATE_SCALE
            ),
            "encoder_gradient_clip": FIXED_GRADIENT_CLIP,
            "head_gradient_clip": FIXED_HEAD_GRADIENT_CLIP,
            "encoder_mode": "eval_with_gradients",
        },
        overfit_preflight=overfit_binding,
    )
    publish_training_attestation_once(
        run_binding,
        args.seed,
        attestation_evidence,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "dropped_label_count": dropped_by_epoch[0],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
