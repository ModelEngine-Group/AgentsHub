"""Fail-closed GPLinker overfit preflight on a tiny fit-only CMeIE subset.

The preflight never reads calibration, outer, or official-dev rows.  Its only
purpose is to prove that the tokenizer/alignment/model/loss/optimizer/decoder
loop can recover a handful of fit triples before another full NPU seed is
allowed to run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from collections import Counter
from itertools import product
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.evidence_tree import (
    TREE_HASH_ALGORITHM,
    scan_bound_regular_tree,
)
from dki.nlp.cmeie_preflight_protocol import (
    validate_cmeie_npu_preflight_receipt,
)
from dki.nlp.cmeie_extraction_contract import BINDING_ALGORITHMS
from dki.nlp.cmeie_overfit_protocol import (
    canonical_overfit_json_bytes,
    overfit_gate_passed as _shared_overfit_gate_passed,
)
from dki.nlp.cmeie_run_protocol import (
    CmeieRunPaths,
    RunBinding,
    canonical_run_paths,
    load_run_binding,
)
from dki.nlp.cmeie_split_protocol import (
    MANIFEST_FILENAME,
    SCHEMA_VERSION as SPLIT_SCHEMA_VERSION,
    SHARD_FILENAMES,
    load_and_validate_role_shard,
)

NO_WEIGHT_DECAY_TERMS = ("bias", "layernorm.weight", "layer_norm.weight")
FIXED_PREFLIGHT_HEAD_LEARNING_RATE = 1e-3
FIXED_HEAD_ONLY_WARMUP_STEPS = 300
FIXED_JOINT_HEAD_LEARNING_RATE_SCALE = 0.1
FIXED_ENCODER_GRADIENT_CLIP = 1.0
FIXED_HEAD_GRADIENT_CLIP = 5.0
FIXED_AMP_INITIAL_SCALE = 1024.0
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


def _uses_weight_decay(name: str) -> bool:
    lowered = str(name).casefold()
    return not any(term in lowered for term in NO_WEIGHT_DECAY_TERMS)


def _adamw_groups(named_parameters, learning_rate: float) -> list[dict]:
    named_parameters = list(named_parameters)
    groups = []
    for decay_enabled in (True, False):
        selected = [
            parameter
            for name, parameter in named_parameters
            if _uses_weight_decay(name) is decay_enabled
        ]
        if selected:
            groups.append(
                {
                    "params": selected,
                    "lr": learning_rate,
                    "weight_decay": 0.01 if decay_enabled else 0.0,
                }
            )
    return groups


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def validate_canonical_smoke_paths(
    args,
    *,
    execution_root: Path = ROOT,
) -> tuple[RunBinding, CmeieRunPaths]:
    """Load the run binding and reject copied source/data/model/output paths."""

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
        args.output,
        paths.overfit,
        label="overfit output",
    )
    if args.source_tree_sha256.casefold() != (
        binding.included_tree_sha256
    ):
        raise ValueError(
            "source tree SHA-256 differs from the authoritative run binding"
        )
    return binding, paths


def validate_source_tree_manifest(
    manifest: dict,
    *,
    expected_tree_sha256: str,
    root: Path,
) -> dict:
    """Recompute every included source file and enforce the production set."""

    if manifest.get("schema_version") != 1:
        raise ValueError("included-tree manifest schema is not v1")
    actual_tree = str(manifest.get("included_tree_sha256", ""))
    if actual_tree != expected_tree_sha256.casefold():
        raise ValueError("included-tree manifest hash differs from run binding")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("included-tree manifest has no entries")
    seen: set[str] = set()
    verified_entries: list[dict[str, Any]] = []
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
            raise ValueError(
                f"unsafe or duplicate included path: {relative!r}"
            )
        seen.add(relative)
        path = root.joinpath(*relative.split("/"))
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"included source is not a regular file: {relative}")
        if (
            path.stat().st_size != entry.get("size")
            or _sha256(path) != entry.get("sha256")
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
    missing = sorted(REQUIRED_TRAINING_SOURCES - seen)
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


def validate_strict_npu_preflight(
    *,
    run_binding: RunBinding,
    paths: CmeieRunPaths,
    source_train_sha256: str,
    base_model_tree: dict[str, Any],
    smoke_script_sha256: str,
) -> dict[str, Any]:
    """Authorize smoke work only from the canonical strict remote receipt."""

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


def _tree_report(directory: Path) -> dict[str, Any]:
    return scan_bound_regular_tree(
        directory,
        label="frozen CMeIE base model",
    )


def _artifact_binding_hashes(
    *,
    tokenizer_vocabulary: dict[str, int],
    entity_labels: list[str],
    predicates: list[str],
    schema_constraints: Any,
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
        "tokenizer_vocabulary_sha256": _sha256_bytes(vocabulary),
        "label_table_sha256": _sha256_bytes(label_table),
        "schema_constraints_sha256": _sha256_bytes(schema),
    }


def validate_split_source_tree_binding(
    manifest: dict[str, Any],
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


def load_isolated_fit_data(
    *,
    fit_data: Path,
    split_manifest: Path,
    source_tree_sha256: str,
    source_tree_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load only the v6 fit shard; no source or outer path is accepted."""
    _, rows, binding = load_and_validate_role_shard(
        split_manifest,
        fit_data,
        role="fit",
        expected_source_tree_sha256=source_tree_sha256,
        expected_source_tree_manifest_sha256=(
            source_tree_manifest_sha256
        ),
        generator_path=(
            ROOT / "scripts" / "build_cmeie_split_manifest.py"
        ),
    )
    return rows, binding


def select_fit_smoke_indices(
    rows: list[dict[str, Any]],
    fit_indices: list[int],
    sample_size: int,
) -> list[int]:
    """Greedily cover predicates, then fill deterministically by row index."""
    if sample_size < 1:
        raise ValueError("sample size must be positive")
    candidates = []
    for index in fit_indices:
        predicates = {
            str(spo.get("predicate", ""))
            for spo in rows[index].get("spo_list", [])
            if str(spo.get("predicate", ""))
        }
        if predicates:
            candidates.append((int(index), predicates))
    if len(candidates) < sample_size:
        raise ValueError("fit split has fewer labelled rows than sample size")

    selected: list[int] = []
    covered: set[str] = set()
    remaining = list(candidates)
    while remaining and len(selected) < sample_size:
        position, (index, predicates) = max(
            enumerate(remaining),
            key=lambda item: (
                len(item[1][1] - covered),
                len(item[1][1]),
                -item[1][0],
            ),
        )
        selected.append(index)
        covered.update(predicates)
        remaining.pop(position)
    return selected


def _json_safe_diagnostic_float(value: Any) -> float | None:
    """Convert a diagnostic scalar to finite JSON, using null on overflow."""

    number = float(value)
    return number if math.isfinite(number) else None


def _summary(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "maximum": None,
        }
    converted = [
        _json_safe_diagnostic_float(value)
        for value in values
    ]
    if any(value is None for value in converted):
        return {
            "count": len(converted),
            "minimum": None,
            "median": None,
            "maximum": None,
        }
    ordered = sorted(value for value in converted if value is not None)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": ordered[(len(ordered) - 1) // 2],
        "maximum": ordered[-1],
    }


def _publish_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"GPLinker overfit preflight exists: {path}")
    encoded = canonical_overfit_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _device(torch, requested: str):
    if requested == "npu":
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise RuntimeError("NPU requested but torch_npu is unavailable")
        return torch.device("npu:0")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cpu")


def overfit_gate_passed(
    *,
    metric: dict[str, Any],
    schema_on: bool,
    minimum_train_f1: float,
) -> bool:
    """Apply the same fail-closed gate consumed by formal seed training."""
    return _shared_overfit_gate_passed(
        metric=metric,
        schema_on=schema_on,
        minimum_train_f1=minimum_train_f1,
    )


def validate_fixed_overfit_protocol(
    args: argparse.Namespace,
    *,
    threshold_values: tuple[float, ...],
) -> None:
    """Reject runs that full seed training would later deem inadmissible."""

    expected = {
        "device": "npu",
        "sample_size": 8,
        "steps": 400,
        "max_length": 320,
        "fp16": True,
        "minimum_train_f1": 0.95,
        "pointer_variant": "efficient",
        "head_reduction": "balanced",
        "positive_weight": 1.0,
        "positive_margin": 0.0,
        "encoder_learning_rate": 2e-5,
        "head_learning_rate": FIXED_PREFLIGHT_HEAD_LEARNING_RATE,
        "freeze_encoder": False,
    }
    mismatches = [
        f"{name}={getattr(args, name)!r}, expected {value!r}"
        for name, value in expected.items()
        if getattr(args, name) != value
    ]
    if threshold_values != (-0.5, 0.0, 0.5):
        mismatches.append(
            f"threshold_values={threshold_values!r}, "
            "expected (-0.5, 0.0, 0.5)"
        )
    if mismatches:
        raise ValueError(
            "CMeIE fixed overfit protocol mismatch: "
            + "; ".join(mismatches)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument(
        "--source-tree-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "npu"], default="npu")
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--entity-label-mode",
        choices=("typed", "role_only"),
        default="typed",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--minimum-train-f1", type=float, default=0.95)
    parser.add_argument(
        "--pointer-variant",
        choices=("efficient", "classic"),
        default="efficient",
    )
    parser.add_argument(
        "--head-reduction",
        choices=(
            "balanced",
            "active_margin",
            "active_bce",
            "balanced_bce",
            "global",
        ),
        default="balanced",
    )
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--positive-margin", type=float, default=0.0)
    parser.add_argument(
        "--encoder-learning-rate",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=FIXED_PREFLIGHT_HEAD_LEARNING_RATE,
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--threshold-value",
        type=float,
        action="append",
        dest="threshold_values",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.source_tree_sha256):
        raise SystemExit("--source-tree-sha256 must be a full SHA-256")
    if args.steps < 1 or args.sample_size < 1:
        raise SystemExit("steps and sample size must be positive")
    if not 0.0 < args.minimum_train_f1 <= 1.0:
        raise SystemExit("minimum train F1 must be in (0, 1]")
    if not math.isfinite(args.positive_weight) or args.positive_weight <= 0:
        raise SystemExit("--positive-weight must be finite and positive")
    if not math.isfinite(args.positive_margin) or args.positive_margin < 0:
        raise SystemExit(
            "--positive-margin must be finite and non-negative"
        )
    if (
        not math.isfinite(args.encoder_learning_rate)
        or args.encoder_learning_rate <= 0
        or not math.isfinite(args.head_learning_rate)
        or args.head_learning_rate <= 0
    ):
        raise SystemExit("learning rates must be finite and positive")
    threshold_values = tuple(
        sorted(
            set(
                args.threshold_values
                if args.threshold_values is not None
                else (-0.5, 0.0, 0.5)
            )
        )
    )
    if (
        not threshold_values
        or any(not math.isfinite(value) for value in threshold_values)
    ):
        raise SystemExit("threshold values must be finite and non-empty")
    try:
        validate_fixed_overfit_protocol(
            args,
            threshold_values=threshold_values,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if (
        args.device == "npu"
        and args.positive_weight != int(args.positive_weight)
    ):
        raise SystemExit(
            "NPU preflight requires an integer positive weight"
        )

    try:
        run_binding, run_paths = validate_canonical_smoke_paths(args)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid canonical CMeIE run binding: {exc}") from exc
    if os.path.lexists(args.output):
        raise SystemExit(f"preflight output exists: {args.output}")
    try:
        source_tree_manifest = json.loads(
            args.source_tree_manifest.read_text(encoding="utf-8")
        )
        source_tree_binding = validate_source_tree_manifest(
            source_tree_manifest,
            expected_tree_sha256=run_binding.included_tree_sha256,
            root=run_binding.source_root,
        )
        source_tree_manifest_sha256 = _sha256(
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
        rows, split_manifest_binding = load_isolated_fit_data(
            fit_data=args.fit_data,
            split_manifest=args.split_manifest,
            source_tree_sha256=args.source_tree_sha256,
            source_tree_manifest_sha256=source_tree_manifest_sha256,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"invalid isolated CMeIE fit shard: {exc}") from exc
    try:
        base_model_path = run_paths.base_model.resolve(strict=True)
        if base_model_path != run_paths.base_model:
            raise ValueError(
                "base model resolves away from its canonical run path"
            )
        base_model_tree = _tree_report(base_model_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot bind local base model tree: {exc}") from exc
    smoke_script_sha256 = _sha256(Path(__file__))
    try:
        preflight_common = validate_strict_npu_preflight(
            run_binding=run_binding,
            paths=run_paths,
            source_train_sha256=split_manifest_binding[
                "source_train_sha256"
            ],
            base_model_tree=base_model_tree,
            smoke_script_sha256=smoke_script_sha256,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"strict NPU preflight is not admissible: {exc}") from exc
    selected_indices = select_fit_smoke_indices(
        rows,
        list(range(len(rows))),
        args.sample_size,
    )
    selected = [rows[index] for index in selected_indices]

    import torch
    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_gplinker import (
        CMeIEGPLinker,
        pointer_group_loss,
    )
    from dki.nlp.cmeie_spo import (
        build_labels,
        decode_spo_grid,
        gold_spo_set,
        labels_from_fit,
        schema_constraints_from_fit,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(torch, args.device)
    if args.fp16 and device.type == "cpu":
        raise SystemExit("--fp16 requires NPU or CUDA")

    fit = rows
    entity_labels, predicates = labels_from_fit(
        fit,
        entity_label_mode=args.entity_label_mode,
    )
    schema_constraints = schema_constraints_from_fit(
        fit,
        entity_label_mode=args.entity_label_mode,
    )
    entity2id = {
        label: index for index, label in enumerate(entity_labels)
    }
    predicate2id = {
        label: index for index, label in enumerate(predicates)
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise SystemExit("GPLinker preflight requires a fast tokenizer")
    binding_hashes = _artifact_binding_hashes(
        tokenizer_vocabulary=tokenizer.get_vocab(),
        entity_labels=entity_labels,
        predicates=predicates,
        schema_constraints=schema_constraints,
    )
    encoded = tokenizer(
        [row["text"] for row in selected],
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
    one_labels = []
    dropped = 0
    for index, row in enumerate(selected):
        labels, lost = build_labels(
            row,
            offsets[index],
            entity2id,
            predicate2id,
            allow_cross_sentence_fallback=True,
        )
        one_labels.append(labels)
        dropped += lost
    if dropped:
        raise SystemExit(
            f"fit-only smoke subset contains {dropped} dropped labels"
        )
    labels = {
        name: torch.stack([item[name] for item in one_labels]).to(device)
        for name in one_labels[0]
    }
    inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }

    encoder = AutoModel.from_pretrained(str(base_model_path))
    model = CMeIEGPLinker(
        encoder,
        int(encoder.config.hidden_size),
        len(entity_labels),
        len(predicates),
        pointer_variant=args.pointer_variant,
    ).to(device)
    encoder_named_parameters = list(model.encoder.named_parameters())
    encoder_parameters = [
        parameter for _, parameter in encoder_named_parameters
    ]
    for parameter in encoder_parameters:
        parameter.requires_grad_(False)
    head_named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    head_parameters = [
        parameter for _, parameter in head_named_parameters
    ]
    encoder_parameter_groups = _adamw_groups(
        encoder_named_parameters,
        args.encoder_learning_rate,
    )
    head_parameter_groups = _adamw_groups(
        head_named_parameters,
        args.head_learning_rate,
    )
    for group in encoder_parameter_groups:
        group["lr"] = 0.0
    optimiser = torch.optim.AdamW(
        encoder_parameter_groups + head_parameter_groups
    )
    if device.type == "npu":
        if not (
            hasattr(torch, "npu")
            and hasattr(torch.npu, "amp")
            and hasattr(torch.npu.amp, "GradScaler")
        ):
            raise RuntimeError(
                "NPU FP16 preflight requires torch.npu.amp.GradScaler"
            )
        scaler = torch.npu.amp.GradScaler(
            init_scale=FIXED_AMP_INITIAL_SCALE,
            enabled=args.fp16,
        )
    else:
        scaler = torch.amp.GradScaler(
            device.type,
            init_scale=FIXED_AMP_INITIAL_SCALE,
            enabled=args.fp16,
        )

    candidates = [
        (thresholds, schema_on)
        for thresholds in product(threshold_values, repeat=3)
        for schema_on in (False, True)
    ]

    def score_current_model() -> dict[str, Any]:
        model.eval()
        counts = {
            candidate: Counter(tp=0, fp=0, fn=0)
            for candidate in candidates
        }
        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.fp16,
            ):
                current_outputs = model(**inputs)
            for index, row in enumerate(selected):
                decoded = decode_spo_grid(
                    row["text"],
                    offsets[index],
                    current_outputs["entity"][index].float().cpu(),
                    current_outputs["relation_head"][index].float().cpu(),
                    current_outputs["relation_tail"][index].float().cpu(),
                    entity_labels,
                    predicates,
                    threshold_values,
                    schema_constraints,
                )
                gold = gold_spo_set(row)
                for candidate, predicted in decoded.items():
                    counts[candidate]["tp"] += len(predicted & gold)
                    counts[candidate]["fp"] += len(predicted - gold)
                    counts[candidate]["fn"] += len(gold - predicted)
        scored = []
        for candidate, count in counts.items():
            precision = count["tp"] / max(
                1,
                count["tp"] + count["fp"],
            )
            recall = count["tp"] / max(
                1,
                count["tp"] + count["fn"],
            )
            f1 = (
                2 * precision * recall
                / max(1e-12, precision + recall)
            )
            scored.append(
                (
                    {
                        **dict(count),
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    },
                    candidate[0],
                    candidate[1],
                )
            )
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
            "declared_threshold_values": list(threshold_values),
            "thresholds": list(thresholds),
            "schema_constraint": "on" if schema_on else "off",
            "metric": metric,
            "grid_size": len(candidates),
        }

    snapshot_steps = sorted(
        {
            0,
            1,
            min(10, args.steps),
            max(1, args.steps // 4),
            max(1, args.steps // 2),
            args.steps,
        }
    )
    snapshots: list[dict[str, Any]] = []
    last_encoder_grad_norm = None
    last_head_grad_norm = None
    skipped_optimizer_steps = 0
    phase_grid_results: dict[str, dict[str, Any]] = {}
    model.train()
    model.encoder.eval()
    for step in range(args.steps + 1):
        head_only = step < FIXED_HEAD_ONLY_WARMUP_STEPS
        if step == FIXED_HEAD_ONLY_WARMUP_STEPS:
            phase_grid_results["head_only"] = score_current_model()
            print(
                json.dumps(
                    {
                        "event": "phase_grid_result",
                        "phase": "head_only",
                        **phase_grid_results["head_only"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            for parameter in encoder_parameters:
                parameter.requires_grad_(True)
            for group in encoder_parameter_groups:
                group["lr"] = args.encoder_learning_rate
            for group in head_parameter_groups:
                group["lr"] = (
                    args.head_learning_rate
                    * FIXED_JOINT_HEAD_LEARNING_RATE_SCALE
                )
        model.train()
        # Match the proven CMeEE recipe: deterministic encoder activations,
        # while gradients are enabled after the head-only warm-up.
        model.encoder.eval()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.fp16,
        ):
            outputs = model(**inputs)
            group_losses = {
                name: pointer_group_loss(
                    outputs[name],
                    labels[name],
                    args.head_reduction,
                    args.positive_weight,
                    args.positive_margin,
                )
                for name in labels
            }
            loss = sum(group_losses.values()) / len(group_losses)
        if step in snapshot_steps:
            positive_summaries = {}
            for name in labels:
                values = (
                    outputs[name].float()[labels[name].bool()]
                    .detach()
                    .cpu()
                    .tolist()
                )
                positive_summaries[name] = _summary(values)
            snapshot = {
                "step": step,
                "phase": "head_only" if head_only else "joint",
                "loss": _json_safe_diagnostic_float(
                    loss.detach().cpu()
                ),
                "group_losses": {
                    name: _json_safe_diagnostic_float(
                        value.detach().cpu()
                    )
                    for name, value in group_losses.items()
                },
                "positive_logits": positive_summaries,
                "previous_encoder_pre_clip_grad_norm": (
                    last_encoder_grad_norm
                ),
                "previous_head_pre_clip_grad_norm": last_head_grad_norm,
            }
            snapshots.append(snapshot)
            print(
                json.dumps(
                    {
                        "event": "training_snapshot",
                        **snapshot,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        if step == args.steps:
            break
        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        head_grad_norm = torch.nn.utils.clip_grad_norm_(
            head_parameters,
            FIXED_HEAD_GRADIENT_CLIP,
        )
        last_head_grad_norm = _json_safe_diagnostic_float(
            head_grad_norm.detach().cpu()
        )
        if head_only:
            last_encoder_grad_norm = None
        else:
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(
                encoder_parameters,
                FIXED_ENCODER_GRADIENT_CLIP,
            )
            last_encoder_grad_norm = _json_safe_diagnostic_float(
                encoder_grad_norm.detach().cpu()
            )
        scale_before = float(scaler.get_scale())
        scaler.step(optimiser)
        scaler.update()
        if float(scaler.get_scale()) < scale_before:
            skipped_optimizer_steps += 1

    phase_grid_results["joint"] = score_current_model()
    selected_phase, declared_grid_result = max(
        phase_grid_results.items(),
        key=lambda item: (
            item[1]["metric"]["f1"],
            item[1]["metric"]["precision"],
            sum(item[1]["thresholds"]),
            item[1]["schema_constraint"] == "on",
            item[0] == "joint",
        ),
    )
    metric = declared_grid_result["metric"]
    schema_on = declared_grid_result["schema_constraint"] == "on"
    passed = overfit_gate_passed(
        metric=metric,
        schema_on=schema_on,
        minimum_train_f1=args.minimum_train_f1,
    )
    payload = {
        "schema_version": 1,
        "kind": "cmeie_gplinker_fit_only_overfit_preflight",
        "passed": passed,
        "minimum_train_f1": args.minimum_train_f1,
        "dataset_role": "fit_only_optimization_preflight",
        "fit_only": True,
        "entity_label_mode": args.entity_label_mode,
        "calibration_accessed": False,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "rid": run_binding.rid,
        "git_head": run_binding.git_head,
        "tree": run_binding.included_tree_sha256,
        "run_root": str(run_binding.run_root),
        "source_manifest_path": str(run_binding.source_manifest),
        "source_tree_sha256": args.source_tree_sha256.lower(),
        "source_tree_manifest_sha256": source_tree_manifest_sha256,
        "source_tree": source_tree_binding,
        "preflight_common": preflight_common,
        "train_sha256": split_manifest_binding[
            "source_train_sha256"
        ],
        "source_train_sha256": split_manifest_binding[
            "source_train_sha256"
        ],
        "fit_shard_sha256": split_manifest_binding["shard_sha256"],
        "split_manifest_sha256": _sha256(args.split_manifest),
        "split_manifest_validation": split_manifest_binding,
        "base_model": str(base_model_path),
        "base_model_tree": base_model_tree,
        "binding_algorithms": dict(BINDING_ALGORITHMS),
        **binding_hashes,
        "seed": args.seed,
        "sample_size": len(selected),
        "sample_row_hashes": [
            hashlib.sha256(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for row in selected
        ],
        "steps": args.steps,
        "max_length": args.max_length,
        "fp16": args.fp16,
        "device": str(device),
        "dropped_label_count": dropped,
        "snapshots": snapshots,
        "declared_grid_result": declared_grid_result,
        "phase_grid_results": phase_grid_results,
        "selected_phase": selected_phase,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "pointer_variant": args.pointer_variant,
        "head_reduction": args.head_reduction,
        "positive_weight": args.positive_weight,
        "positive_margin": args.positive_margin,
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "joint_head_learning_rate": (
            args.head_learning_rate
            * FIXED_JOINT_HEAD_LEARNING_RATE_SCALE
        ),
        "head_only_warmup_steps": FIXED_HEAD_ONLY_WARMUP_STEPS,
        "joint_optimizer_steps": (
            args.steps - FIXED_HEAD_ONLY_WARMUP_STEPS
        ),
        "encoder_mode": "eval_with_gradients_after_warmup",
        "encoder_frozen": False,
        "amp_grad_scaler": bool(args.fp16),
        "amp_initial_scale": FIXED_AMP_INITIAL_SCALE,
        "amp_final_scale": _json_safe_diagnostic_float(
            scaler.get_scale()
        ),
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "gradient_clip": FIXED_ENCODER_GRADIENT_CLIP,
        "gradient_clip_scope": "separate_encoder_and_pointer_heads",
        "encoder_gradient_clip": FIXED_ENCODER_GRADIENT_CLIP,
        "head_gradient_clip": FIXED_HEAD_GRADIENT_CLIP,
        "gradient_clip_after_unscale": True,
        "weight_decay": 0.01,
        "weight_decay_policy": "exclude_bias_and_layernorm",
        "positive_weight_implementation": (
            "integer_unrolled_add"
            if args.positive_weight == int(args.positive_weight)
            and 1 <= int(args.positive_weight) <= 16
            else "direct_scalar_multiply"
        ),
        "preflight_script_sha256": smoke_script_sha256,
        "model_module_sha256": _sha256(
            ROOT / "dki" / "nlp" / "cmeie_gplinker.py"
        ),
        "alignment_module_sha256": _sha256(
            ROOT / "dki" / "nlp" / "cmeie_spo.py"
        ),
        "globalpointer_module_sha256": _sha256(
            ROOT / "dki" / "nlp" / "globalpointer.py"
        ),
    }
    _publish_once(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": passed,
                "train_f1": metric["f1"],
                "sample_size": len(selected),
                "calibration_accessed": False,
                "outer_accessed": False,
                "official_dev_accessed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
