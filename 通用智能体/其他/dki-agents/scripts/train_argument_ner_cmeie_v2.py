"""Train one fail-closed native CMeIE argument-NER feasibility candidate.

This v2 entry point is intentionally narrower than the future five-fold OOF
pipeline.  It trains only on the frozen physical ``fit`` shard, selects one
global decode threshold only on the frozen ``calibration`` shard, and never
opens ``outer`` or official dev data.  A successful run is a feasibility
result, not an end-to-end SPO promotion.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
import random
import shutil
import stat
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


POINTER_VARIANT = "efficient"
INNER_DIM = 64
FIXED_TYPES = 11
FIXED_HEAD_REDUCTION = "balanced"
FIXED_HEAD_WARMUP_EPOCHS = 2
TOP_LAYER_ADAPTATION_WARMUP_EPOCHS = 4
TOP_LAYER_ADAPTATION_COUNT = 2
FIXED_ENCODER_MODE = "eval_with_gradients_after_warmup"
FIXED_ENCODER_GRADIENT_CLIP = 1.0
FIXED_HEAD_GRADIENT_CLIP = 5.0
FIXED_SOFTLEXICON_MIN_CHARS = 2
FIXED_SOFTLEXICON_MODE = "add"
REJECTION_REASON = "no_viable_operating_point_in_declared_global_grid"
REJECTED_EXIT_CODE = 2


def argument_precision_profile(
    precision: str,
    *,
    bf16_supported: bool,
) -> dict[str, bool | str]:
    """Validate the declared NPU autocast mode and scaler contract."""

    mode = str(precision).casefold()
    if mode not in {"fp16", "bf16"}:
        raise ValueError("precision must be fp16 or bf16")
    if mode == "bf16" and not bf16_supported:
        raise ValueError("BF16 precision is not supported by this NPU runtime")
    return {
        "precision": mode,
        "uses_grad_scaler": mode == "fp16",
        "bf16_supported": bool(bf16_supported),
    }


def argument_training_strategy(
    *,
    head_warmup_epochs: int,
    epochs: int,
    encoder_scope: str = "all",
) -> str:
    """Return the only two predeclared schedules allowed by this trainer.

    The frozen schedule is a bounded ablation added after formal phased runs
    showed immediate encoder collapse while head-only epochs kept improving.
    """

    if (
        encoder_scope == "all"
        and
        head_warmup_epochs == FIXED_HEAD_WARMUP_EPOCHS
        and epochs > FIXED_HEAD_WARMUP_EPOCHS
    ):
        return "phased_head_then_joint"
    if encoder_scope == "none" and head_warmup_epochs == epochs:
        return "frozen_encoder_ablation"
    if (
        encoder_scope == "last2"
        and head_warmup_epochs == TOP_LAYER_ADAPTATION_WARMUP_EPOCHS
        and epochs == 8
    ):
        return "top2_encoder_low_lr_ablation"
    raise ValueError(
        "encoder schedule must be all+warmup2, none+warmup=epochs, "
        "or last2+warmup4+epochs8"
    )


def softlex_ablation_contract(
    *,
    role: str | None,
    softlexicon: bool,
    seed: int,
    epochs: int,
    head_warmup_epochs: int,
    encoder_scope: str,
) -> str | None:
    """Validate the sole baseline/augmentation comparison admitted here."""
    if role is None:
        if softlexicon:
            raise ValueError(
                "SoftLexicon is allowed only in the declared ablation role"
            )
        return None
    if role not in {"baseline", "softlex"}:
        raise ValueError("ablation role must be baseline or softlex")
    if (
        seed != 43
        or epochs != 4
        or head_warmup_epochs != 4
        or encoder_scope != "none"
    ):
        raise ValueError(
            "SoftLexicon ablation requires seed43, epochs4, warmup4, "
            "and a frozen encoder"
        )
    if softlexicon is not (role == "softlex"):
        raise ValueError("ablation role and SoftLexicon enablement disagree")
    return role


def selected_encoder_parameter_names(
    names: list[str],
    *,
    encoder_scope: str,
) -> frozenset[str]:
    """Resolve the exact encoder parameters allowed to receive gradients."""

    if encoder_scope == "all":
        return frozenset(names)
    if encoder_scope == "none":
        return frozenset()
    if encoder_scope != "last2":
        raise ValueError(f"unsupported encoder scope: {encoder_scope}")
    by_name: dict[str, int] = {}
    layer_ids: set[int] = set()
    for name in names:
        parts = name.split(".")
        layer_id = None
        for index in range(len(parts) - 2):
            if parts[index : index + 2] == ["encoder", "layer"]:
                try:
                    layer_id = int(parts[index + 2])
                except ValueError:
                    layer_id = None
                break
        if layer_id is not None:
            by_name[name] = layer_id
            layer_ids.add(layer_id)
    if len(layer_ids) < TOP_LAYER_ADAPTATION_COUNT:
        raise ValueError("encoder does not expose at least two numbered layers")
    selected_layers = set(sorted(layer_ids)[-TOP_LAYER_ADAPTATION_COUNT:])
    selected = frozenset(
        name for name, layer_id in by_name.items() if layer_id in selected_layers
    )
    if not selected:
        raise ValueError("top-two encoder scope selected no parameters")
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(
    root: Path,
    *,
    excluded_relative_paths: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Return the repository's stable regular-file tree digest."""
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"tree root is not a directory: {root}")
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"tree contains a symbolic link: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded_relative_paths:
            continue
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode):
            raise ValueError(f"tree contains a non-regular file: {path}")
        size = path.stat().st_size
        file_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        total_bytes += size
    return {
        "algorithm": "sha256 of sorted path NUL size NUL file_sha256 LF",
        "sha256": digest.hexdigest(),
        "n_files": file_count,
        "bytes": total_bytes,
    }


def accumulation_group_size(
    micro_batch: int,
    micro_batches: int,
    gradient_accumulation: int,
) -> int:
    """Return the real divisor for the current accumulation group."""
    if (
        isinstance(micro_batch, bool)
        or isinstance(micro_batches, bool)
        or isinstance(gradient_accumulation, bool)
        or not all(
            isinstance(value, int)
            for value in (micro_batch, micro_batches, gradient_accumulation)
        )
        or micro_batches < 1
        or gradient_accumulation < 1
        or not 1 <= micro_batch <= micro_batches
    ):
        raise ValueError("invalid gradient-accumulation coordinates")
    group_start = (
        ((micro_batch - 1) // gradient_accumulation)
        * gradient_accumulation
        + 1
    )
    return min(
        gradient_accumulation,
        micro_batches - group_start + 1,
    )


def optimizer_steps_per_epoch(
    micro_batches: int,
    gradient_accumulation: int,
) -> int:
    if (
        isinstance(micro_batches, bool)
        or isinstance(gradient_accumulation, bool)
        or not isinstance(micro_batches, int)
        or not isinstance(gradient_accumulation, int)
        or micro_batches < 1
        or gradient_accumulation < 1
    ):
        raise ValueError("optimizer step inputs must be positive integers")
    return (micro_batches + gradient_accumulation - 1) // gradient_accumulation


def advance_sparse_early_stopping(
    best_rank: tuple[float, float, float] | None,
    stale_epochs: int,
    current_rank: tuple[float, float, float],
) -> tuple[tuple[float, float, float], int, bool]:
    """Do not spend patience while a sparse decoder has never produced F1."""
    if best_rank is None or current_rank > best_rank:
        return current_rank, 0, True
    if best_rank[0] <= 0.0:
        return best_rank, 0, False
    return best_rank, stale_epochs + 1, False


def calibration_history_fields(
    threshold_sweep: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a complete immutable-by-copy sweep plus its selected point."""
    if not threshold_sweep:
        raise ValueError("calibration threshold sweep must not be empty")
    required_fields = {
        "threshold",
        "aggregate",
        "by_type",
        "predicted_mentions",
        "gold_mentions",
    }
    for operating_point in threshold_sweep:
        missing = required_fields.difference(operating_point)
        if missing:
            raise ValueError(
                "calibration operating point is incomplete: "
                + ", ".join(sorted(missing))
            )
    selected = max(
        threshold_sweep,
        key=lambda item: (
            item["aggregate"]["f1"],
            item["aggregate"]["precision"],
            item["threshold"],
        ),
    )
    return {
        "calibration_sweep": copy.deepcopy(threshold_sweep),
        "selected_calibration": copy.deepcopy(selected),
    }


def candidate_publication_decision(
    selected_calibration: dict[str, Any],
) -> tuple[bool, str | None]:
    """Fail closed when calibration found no usable mention decoder."""
    aggregate_f1 = float(selected_calibration["aggregate"]["f1"])
    predicted_mentions = int(selected_calibration["predicted_mentions"])
    if not math.isfinite(aggregate_f1):
        raise ValueError("selected aggregate F1 must be finite")
    if predicted_mentions < 0:
        raise ValueError("selected predicted mention count must be non-negative")
    if aggregate_f1 <= 0.0 or predicted_mentions == 0:
        return False, REJECTION_REASON
    return True, None


def parse_threshold_grid(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(
            sorted(
                {
                    float(value.strip())
                    for value in str(raw).split(",")
                    if value.strip()
                }
            )
        )
    except ValueError as error:
        raise ValueError("threshold grid must contain numbers") from error
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("threshold grid must contain finite values")
    return values


def linear_phase_learning_rate(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak: float,
) -> float:
    """Deterministic linear warmup/decay for one optimizer phase."""
    if (
        isinstance(step, bool)
        or isinstance(total_steps, bool)
        or isinstance(warmup_steps, bool)
        or not all(
            isinstance(value, int)
            for value in (step, total_steps, warmup_steps)
        )
        or total_steps < 1
        or not 1 <= step <= total_steps
        or not 0 <= warmup_steps < total_steps
        or not math.isfinite(peak)
        or peak <= 0.0
    ):
        raise ValueError("invalid phase learning-rate schedule")
    if warmup_steps and step <= warmup_steps:
        return peak * step / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    return peak * max(0, total_steps - step) / decay_steps


def validate_argument_overfit_preflight(
    payload: dict[str, Any],
    *,
    run_binding,
    fit_binding: dict[str, Any],
    base_model_tree: dict[str, Any],
    pointer_variant: str,
    use_bilstm: bool,
) -> dict[str, Any]:
    """Validate the fit-only pointer-head gate before full training."""
    if payload.get("kind") != "cmeie_argument_pointer_overfit_preflight":
        raise ValueError("unexpected argument overfit preflight kind")
    if payload.get("schema_version") != 2:
        raise ValueError("unexpected argument overfit preflight schema")
    if payload.get("dataset_role") != "fit_training_diagnostic":
        raise ValueError("argument overfit preflight used the wrong data role")
    if payload.get("selection_used") is not False:
        raise ValueError("argument overfit preflight must not be selection-used")
    if payload.get("outer_accessed") is not False:
        raise ValueError("argument overfit preflight accessed outer")
    if payload.get("official_dev_accessed") is not False:
        raise ValueError("argument overfit preflight accessed official dev")
    expected_run = {
        "rid": run_binding.rid,
        "git_head": run_binding.git_head,
        "source_tree_sha256": run_binding.included_tree_sha256,
        "source_manifest_sha256": run_binding.source_manifest_sha256,
    }
    if payload.get("run_binding") != expected_run:
        raise ValueError("argument overfit run binding differs")
    gate = payload.get("gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError("argument overfit preflight did not pass")
    selected = payload.get("selected_operating_point")
    if not isinstance(selected, dict):
        raise ValueError("argument overfit selected operating point is missing")
    metric = selected.get("metric")
    if not isinstance(metric, dict):
        raise ValueError("argument overfit metric is missing")
    if (
        float(metric.get("f1", -1.0)) < 0.95
        or float(metric.get("precision", -1.0)) < 0.95
        or float(metric.get("recall", -1.0)) < 0.95
    ):
        raise ValueError("argument overfit quality floor was not met")
    density = float(gate.get("density_ratio", math.inf))
    if not math.isfinite(density) or density > 2.0:
        raise ValueError("argument overfit density ceiling was not met")
    bound_fit = payload.get("fit_binding")
    if not isinstance(bound_fit, dict) or (
        bound_fit.get("shard_sha256") != fit_binding["shard_sha256"]
        or bound_fit.get("split_manifest_sha256")
        != fit_binding["split_manifest_sha256"]
    ):
        raise ValueError("argument overfit fit binding differs")
    bound_model = payload.get("base_model_tree")
    if not isinstance(bound_model, dict) or (
        bound_model.get("sha256") != base_model_tree["sha256"]
    ):
        raise ValueError("argument overfit base model tree differs")
    if payload.get("sample_size") != 8 or payload.get("steps") != 400:
        raise ValueError("argument overfit fixed work budget differs")
    if payload.get("head_reduction") != FIXED_HEAD_REDUCTION:
        raise ValueError("argument overfit head reduction differs")
    if payload.get("pointer_variant") != pointer_variant:
        raise ValueError("argument overfit pointer variant differs")
    if payload.get("use_bilstm") is not use_bilstm:
        raise ValueError("argument overfit BiLSTM profile differs")
    expected_sources = {
        "overfit_script_sha256": sha256_file(
            ROOT / "scripts" / "smoke_cmeie_argument_overfit.py"
        ),
        "training_script_sha256": sha256_file(Path(__file__)),
        "argument_module_sha256": sha256_file(
            ROOT / "dki" / "nlp" / "cmeie_argument.py"
        ),
        "globalpointer_module_sha256": sha256_file(
            ROOT / "dki" / "nlp" / "globalpointer.py"
        ),
        "balanced_loss_module_sha256": sha256_file(
            ROOT / "dki" / "nlp" / "cmeie_gplinker.py"
        ),
    }
    if payload.get("source_bindings") != expected_sources:
        raise ValueError("argument overfit source bindings differ")
    return {
        "validated": True,
        "metric": {
            key: metric[key]
            for key in ("tp", "fp", "fn", "precision", "recall", "f1")
        },
        "threshold": selected["threshold"],
        "density_ratio": density,
        "pointer_variant": pointer_variant,
        "use_bilstm": use_bilstm,
        "source_bindings": expected_sources,
    }


def publish_json_create_once(path: Path, payload: dict[str, Any]) -> None:
    """Publish a complete UTF-8 JSON file without replacing another writer."""
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            if hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), 0o444)
        os.link(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _canonical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _require_path(actual: Path, expected: Path, label: str) -> Path:
    actual = _canonical(actual)
    expected = _canonical(expected)
    if actual != expected:
        raise ValueError(f"{label} is not the canonical run path: {expected}")
    return expected


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _state_on_cpu(model) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _normalise_inputs(encoded: dict[str, Any], device) -> dict[str, Any]:
    return {
        key: value.to(device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "token_type_ids", "softlex"}
    }


def _publish_model_directory(
    output: Path,
    writer: Callable[[Path], None],
) -> None:
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(f"candidate output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        writer(staging)
        if os.path.lexists(output):
            raise FileExistsError(f"candidate output already exists: {output}")
        _rename_noreplace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one directory without replacement on Linux."""
    if os.name != "posix":
        raise RuntimeError("argument candidate publication requires Linux")
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is unavailable; refusing unsafe publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(destination)
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strict CMeIE native argument-NER feasibility training",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument(
        "--overfit-preflight-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calibration-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--encoder-gradient-clip",
        type=float,
        default=FIXED_ENCODER_GRADIENT_CLIP,
    )
    parser.add_argument(
        "--head-gradient-clip",
        type=float,
        default=FIXED_HEAD_GRADIENT_CLIP,
    )
    parser.add_argument(
        "--head-warmup-epochs",
        type=int,
        default=FIXED_HEAD_WARMUP_EPOCHS,
    )
    parser.add_argument(
        "--encoder-scope",
        choices=["all", "last2", "none"],
        default="all",
    )
    parser.add_argument(
        "--pointer-variant",
        choices=["efficient", "classic"],
        default=POINTER_VARIANT,
    )
    parser.add_argument("--use-bilstm", action="store_true")
    parser.add_argument(
        "--softlexicon",
        action="store_true",
        help=(
            "enable the single predeclared fit-only native-type BMES "
            "SoftLexicon ablation"
        ),
    )
    parser.add_argument(
        "--ablation-role",
        choices=["baseline", "softlex"],
        default=None,
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "bf16"],
        default="fp16",
    )
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--threshold-grid",
        default="-1.0,-0.5,0.0,0.5,1.0",
    )
    parser.add_argument("--device", choices=["npu"], default="npu")
    args = parser.parse_args()

    for field in (
        "epochs",
        "batch_size",
        "calibration_batch_size",
        "gradient_accumulation",
        "max_length",
        "patience",
        "head_warmup_epochs",
    ):
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            parser.error(f"--{field.replace('_', '-')} must be a positive integer")
    if args.seed not in {42, 43, 44}:
        parser.error("candidate seed must be one of 42, 43, 44")
    if not 0.0 <= args.warmup < 1.0:
        parser.error("--warmup must be in [0, 1)")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (
            args.encoder_lr,
            args.head_lr,
            args.encoder_gradient_clip,
            args.head_gradient_clip,
        )
    ):
        parser.error("learning rates and gradient clips must be finite and positive")
    try:
        training_strategy = argument_training_strategy(
            head_warmup_epochs=args.head_warmup_epochs,
            epochs=args.epochs,
            encoder_scope=args.encoder_scope,
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        ablation_role = softlex_ablation_contract(
            role=args.ablation_role,
            softlexicon=args.softlexicon,
            seed=args.seed,
            epochs=args.epochs,
            head_warmup_epochs=args.head_warmup_epochs,
            encoder_scope=args.encoder_scope,
        )
    except ValueError as error:
        parser.error(str(error))
    encoder_mode = (
        "eval_frozen_all_epochs"
        if training_strategy == "frozen_encoder_ablation"
        else FIXED_ENCODER_MODE
    )
    if training_strategy == "top2_encoder_low_lr_ablation":
        encoder_mode = "eval_top2_layers_with_gradients_after_warmup"
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        parser.error("--weight-decay must be finite and non-negative")
    try:
        threshold_grid = parse_threshold_grid(args.threshold_grid)
    except ValueError as error:
        parser.error(str(error))

    from dki.nlp.cmeie_run_protocol import (
        canonical_run_paths,
        load_run_binding,
    )
    from dki.nlp.cmeie_split_protocol import (
        MANIFEST_FILENAME,
        SHARD_FILENAMES,
        load_and_validate_role_shard,
    )

    run_binding = load_run_binding(args.run_root)
    run_paths = canonical_run_paths(run_binding)
    _require_path(ROOT, run_binding.source_root, "executing source root")
    fit_path = _require_path(
        args.fit_data,
        run_paths.split_dir / SHARD_FILENAMES["fit"],
        "fit shard",
    )
    calibration_path = _require_path(
        args.calibration_data,
        run_paths.split_dir / SHARD_FILENAMES["calibration"],
        "calibration shard",
    )
    manifest_path = _require_path(
        args.split_manifest,
        run_paths.split_dir / MANIFEST_FILENAME,
        "split manifest",
    )
    base_model_path = _require_path(
        args.base_model,
        run_paths.base_model,
        "base model",
    )
    overfit_preflight_path = _require_path(
        args.overfit_preflight_receipt,
        run_paths.evidence_dir / "argument_overfit.json",
        "argument overfit preflight receipt",
    )
    if ablation_role is None:
        expected_output = (
            run_binding.run_root
            / "models"
            / "cmeie_argument"
            / f"seed-{args.seed}"
        )
        marker_stem = f"argument_seed_{args.seed}"
    else:
        expected_output = (
            run_binding.run_root
            / "models"
            / "cmeie_argument_ablation"
            / f"seed-{args.seed}-{ablation_role}"
        )
        marker_stem = f"argument_softlex_{ablation_role}_seed_{args.seed}"
    output_path = _require_path(args.output, expected_output, "candidate output")
    started_marker = (
        run_paths.evidence_dir
        / f"{marker_stem}_started.json"
    )
    completed_marker = (
        run_paths.evidence_dir
        / f"{marker_stem}_completed.json"
    )
    rejected_marker = (
        run_paths.evidence_dir
        / f"{marker_stem}_rejected.json"
    )
    if any(
        os.path.lexists(path)
        for path in (
            output_path,
            started_marker,
            completed_marker,
            rejected_marker,
        )
    ):
        parser.error("candidate output or create-once marker already exists")

    loader_arguments = {
        "expected_source_tree_sha256": run_binding.included_tree_sha256,
        "expected_source_tree_manifest_sha256": (
            run_binding.source_manifest_sha256
        ),
        "generator_path": (
            run_binding.source_root
            / "scripts"
            / "build_cmeie_split_manifest.py"
        ),
        "expected_run_binding": run_binding,
    }
    fit_manifest, fit_rows, fit_binding = load_and_validate_role_shard(
        manifest_path,
        fit_path,
        role="fit",
        **loader_arguments,
    )
    calibration_manifest, calibration_rows, calibration_binding = (
        load_and_validate_role_shard(
            manifest_path,
            calibration_path,
            role="calibration",
            **loader_arguments,
        )
    )
    if (
        fit_manifest != calibration_manifest
        or fit_binding["split_manifest_sha256"]
        != calibration_binding["split_manifest_sha256"]
        or fit_binding["source_train_sha256"]
        != calibration_binding["source_train_sha256"]
    ):
        raise RuntimeError("fit and calibration shards are not one frozen split")
    if not fit_rows or not calibration_rows:
        raise RuntimeError("fit and calibration shards must both be non-empty")
    base_model_tree = sha256_tree(base_model_path)
    training_script_sha256 = sha256_file(Path(__file__))
    argument_module_path = ROOT / "dki" / "nlp" / "cmeie_argument.py"
    argument_module_sha256 = sha256_file(argument_module_path)
    balanced_loss_module_path = ROOT / "dki" / "nlp" / "cmeie_gplinker.py"
    balanced_loss_module_sha256 = sha256_file(balanced_loss_module_path)
    softlexicon_module_path = ROOT / "dki" / "nlp" / "softlexicon.py"
    softlexicon_module_sha256 = sha256_file(softlexicon_module_path)
    try:
        overfit_preflight_payload = json.loads(
            overfit_preflight_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read argument overfit preflight") from error
    overfit_preflight_binding = validate_argument_overfit_preflight(
        overfit_preflight_payload,
        run_binding=run_binding,
        fit_binding=fit_binding,
        base_model_tree=base_model_tree,
        pointer_variant=args.pointer_variant,
        use_bilstm=args.use_bilstm,
    )
    overfit_preflight_binding.update(
        {
            "path": str(overfit_preflight_path),
            "sha256": sha256_file(overfit_preflight_path),
            "size": overfit_preflight_path.stat().st_size,
        }
    )

    import torch
    import torch_npu  # noqa: F401
    import transformers
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModel,
        AutoTokenizer,
    )

    from dki.nlp.cmeie_argument import (
        ARGUMENT_BILSTM_WEIGHTS,
        ARGUMENT_HEAD_WEIGHTS,
        ARGUMENT_MODEL_CONFIG,
        ARGUMENT_SOFTLEXICON,
        ARGUMENT_SOFTLEXICON_WEIGHTS,
        argument_model_contract,
        build_labels,
        decode_mentions,
        gold_mentions,
        labels_from_rows,
        mention_metric,
        mention_metric_by_label,
    )
    from dki.nlp.cmeie_gplinker import pointer_group_loss
    from dki.nlp.globalpointer import EncoderGlobalPointer
    from dki.nlp.softlexicon import (
        argument_surface_lexicon_payload,
        build_cmeie_argument_surface_lexicon,
        canonical_surface_lexicon_bytes,
        encode_softlex_batch,
        surface_lexicon_sha256,
    )

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("the argument feasibility run requires Ascend NPU")
    if not (
        hasattr(torch.npu, "amp")
        and hasattr(torch.npu.amp, "GradScaler")
    ):
        raise RuntimeError("NPU mixed-precision training requires GradScaler")
    precision_profile = argument_precision_profile(
        args.precision,
        bf16_supported=bool(torch.npu.is_bf16_supported()),
    )
    autocast_dtype = (
        torch.float16
        if precision_profile["precision"] == "fp16"
        else torch.bfloat16
    )
    visible_device_count = int(torch.npu.device_count())
    current_device = int(torch.npu.current_device())
    if visible_device_count != 1 or current_device != 0:
        raise RuntimeError(
            "argument feasibility requires exactly one visible NPU device 0"
        )
    for variable in (
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_DEVICE_ID",
        "DEVICE_ID",
    ):
        if os.getenv(variable) != "0":
            raise RuntimeError(f"{variable} must be exactly '0'")
    device = torch.device("npu:0")
    runtime_environment = {
        "ASCEND_RT_VISIBLE_DEVICES": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "ASCEND_DEVICE_ID": os.environ["ASCEND_DEVICE_ID"],
        "DEVICE_ID": os.environ["DEVICE_ID"],
        "visible_logical_npu_count": visible_device_count,
        "current_logical_device": current_device,
        "runtime_device_name": str(torch.npu.get_device_name(0)),
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "transformers": str(transformers.__version__),
        "precision": precision_profile,
    }
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)

    labels = labels_from_rows(fit_rows)
    if len(labels) != FIXED_TYPES:
        raise RuntimeError(
            f"fit shard must expose exactly {FIXED_TYPES} native types; "
            f"found {len(labels)}"
        )
    type_to_id = {label: index for index, label in enumerate(labels)}
    softlexicon = None
    softlexicon_payload = None
    softlexicon_bytes = None
    softlexicon_digest = None
    softlexicon_dim = 0
    if args.softlexicon:
        softlexicon = build_cmeie_argument_surface_lexicon(
            fit_rows,
            type_to_id,
            min_chars=FIXED_SOFTLEXICON_MIN_CHARS,
        )
        if not softlexicon:
            raise RuntimeError("fit-only CMeIE SoftLexicon is empty")
        softlexicon_payload = argument_surface_lexicon_payload(
            softlexicon,
            labels,
            min_chars=FIXED_SOFTLEXICON_MIN_CHARS,
        )
        softlexicon_bytes = canonical_surface_lexicon_bytes(
            softlexicon_payload
        )
        softlexicon_digest = surface_lexicon_sha256(softlexicon_payload)
        if hashlib.sha256(softlexicon_bytes).hexdigest() != softlexicon_digest:
            raise RuntimeError("SoftLexicon canonical hash disagreement")
        softlexicon_dim = len(labels) * 4
    softlexicon_binding = {
        "enabled": bool(args.softlexicon),
        "mode": FIXED_SOFTLEXICON_MODE if args.softlexicon else None,
        "min_chars": FIXED_SOFTLEXICON_MIN_CHARS if args.softlexicon else None,
        "feature_dim": softlexicon_dim,
        "surface_count": (
            softlexicon_payload["surface_count"]
            if softlexicon_payload is not None
            else 0
        ),
        "type_assignment_count": (
            softlexicon_payload["type_assignment_count"]
            if softlexicon_payload is not None
            else 0
        ),
        "payload_sha256": softlexicon_digest,
        "module_sha256": softlexicon_module_sha256,
        "source_role": "fit" if args.softlexicon else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        use_fast=True,
    )

    def collate(batch):
        encoded = tokenizer(
            [row["text"] for row in batch],
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = [
            [tuple(pair) for pair in value.tolist()]
            for value in encoded.pop("offset_mapping")
        ]
        if softlexicon is not None:
            encoded["softlex"] = torch.from_numpy(
                encode_softlex_batch(
                    batch,
                    offsets,
                    softlexicon,
                    len(labels),
                    int(encoded["input_ids"].shape[1]),
                )
            )
        dense_labels = []
        dropped = 0
        for row, row_offsets in zip(batch, offsets):
            row_labels, row_dropped = build_labels(
                row,
                row_offsets,
                type_to_id,
            )
            dense_labels.append(row_labels)
            dropped += row_dropped
        return encoded, torch.stack(dense_labels), dropped

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        fit_rows,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
        num_workers=0,
    )
    encoder = AutoModel.from_pretrained(str(base_model_path))
    model = EncoderGlobalPointer(
        encoder,
        int(encoder.config.hidden_size),
        len(labels),
        inner_dim=INNER_DIM,
        pointer=args.pointer_variant,
        softlex_dim=softlexicon_dim,
        softlex_mode=FIXED_SOFTLEXICON_MODE,
        use_bilstm=args.use_bilstm,
    ).to(device)
    encoder_named = list(model.encoder.named_parameters())
    selected_encoder_names = selected_encoder_parameter_names(
        [name for name, _ in encoder_named],
        encoder_scope=args.encoder_scope,
    )
    selected_encoder_parameter_count = sum(
        parameter.numel()
        for name, parameter in encoder_named
        if name in selected_encoder_names
    )
    head_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    no_decay_terms = ("bias", "layernorm.weight", "layer_norm.weight")

    def uses_decay(name: str) -> bool:
        lowered = name.casefold()
        return not any(term in lowered for term in no_decay_terms)

    parameter_groups = []
    for group_name, named_parameters, learning_rate in (
        ("encoder", encoder_named, args.encoder_lr),
        ("pointer_head", head_named, args.head_lr),
    ):
        for decay in (True, False):
            parameters = [
                parameter
                for name, parameter in named_parameters
                if parameter.requires_grad and uses_decay(name) is decay
            ]
            if parameters:
                parameter_groups.append(
                    {
                        "params": parameters,
                        "lr": (
                            0.0
                            if group_name == "encoder"
                            else learning_rate
                        ),
                        "weight_decay": args.weight_decay if decay else 0.0,
                        "group_name": f"{group_name}_{'decay' if decay else 'no_decay'}",
                        "group_kind": group_name,
                        "peak_lr": learning_rate,
                    }
                )
    optimiser = torch.optim.AdamW(parameter_groups)
    steps_per_epoch = optimizer_steps_per_epoch(
        len(train_loader),
        args.gradient_accumulation,
    )
    total_scheduler_steps = steps_per_epoch * args.epochs
    head_warmup_steps = int(total_scheduler_steps * args.warmup)
    joint_total_steps = (
        steps_per_epoch * (args.epochs - args.head_warmup_epochs)
    )
    encoder_warmup_steps = int(joint_total_steps * args.warmup)

    def set_phase_learning_rates(
        *,
        phase: str,
        head_step: int,
        encoder_step: int,
    ) -> None:
        head_lr = linear_phase_learning_rate(
            step=head_step,
            total_steps=total_scheduler_steps,
            warmup_steps=head_warmup_steps,
            peak=args.head_lr,
        )
        encoder_lr = 0.0
        if phase == "joint":
            encoder_lr = linear_phase_learning_rate(
                step=encoder_step,
                total_steps=joint_total_steps,
                warmup_steps=encoder_warmup_steps,
                peak=args.encoder_lr,
            )
        for group in optimiser.param_groups:
            group["lr"] = (
                encoder_lr
                if group["group_kind"] == "encoder"
                else head_lr
            )

    scaler = torch.npu.amp.GradScaler(
        init_scale=1024.0,
        enabled=bool(precision_profile["uses_grad_scaler"]),
    )
    publish_json_create_once(
        started_marker,
        {
            "schema_version": 2,
            "kind": "cmeie_argument_feasibility_started",
            "created_at": _utc_now(),
            "rid": run_binding.rid,
            "git_head": run_binding.git_head,
            "source_tree_sha256": run_binding.included_tree_sha256,
            "source_manifest_sha256": run_binding.source_manifest_sha256,
            "split_manifest_sha256": fit_binding["split_manifest_sha256"],
            "fit_shard_sha256": fit_binding["shard_sha256"],
            "calibration_shard_sha256": calibration_binding["shard_sha256"],
            "base_model_tree_sha256": base_model_tree["sha256"],
            "training_script_sha256": training_script_sha256,
            "argument_module_sha256": argument_module_sha256,
            "balanced_loss_module_sha256": balanced_loss_module_sha256,
            "softlexicon": softlexicon_binding,
            "overfit_preflight": overfit_preflight_binding,
            "head_reduction": FIXED_HEAD_REDUCTION,
            "pointer_variant": args.pointer_variant,
            "use_bilstm": args.use_bilstm,
            "precision": precision_profile,
            "encoder_mode": encoder_mode,
            "encoder_scope": args.encoder_scope,
            "selected_encoder_parameter_count": (
                selected_encoder_parameter_count
            ),
            "head_warmup_epochs": args.head_warmup_epochs,
            "training_strategy": training_strategy,
            "runtime_environment": runtime_environment,
            "seed": args.seed,
            "ablation_role": ablation_role,
            "terminal_markers": {
                "completed": {
                    "path": str(completed_marker),
                    "semantics": (
                        "candidate model directory was published and verified"
                    ),
                },
                "rejected": {
                    "path": str(rejected_marker),
                    "semantics": (
                        "no candidate model directory was published because "
                        "the tested threshold grid had no viable operating point"
                    ),
                    "reason": REJECTION_REASON,
                    "exit_code": REJECTED_EXIT_CODE,
                    "output_published": False,
                    "completed_marker_created": False,
                },
                "mutually_exclusive": True,
            },
            "outer_accessed": False,
            "official_dev_accessed": False,
        },
    )

    def calibration_scores():
        predictions = {threshold: [] for threshold in threshold_grid}
        gold_rows = []
        model.eval()
        with torch.no_grad():
            for start in range(
                0,
                len(calibration_rows),
                args.calibration_batch_size,
            ):
                batch = calibration_rows[
                    start : start + args.calibration_batch_size
                ]
                encoded = tokenizer(
                    [row["text"] for row in batch],
                    truncation=True,
                    max_length=args.max_length,
                    padding=True,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                )
                offsets = [
                    [tuple(pair) for pair in value.tolist()]
                    for value in encoded.pop("offset_mapping")
                ]
                if softlexicon is not None:
                    encoded["softlex"] = torch.from_numpy(
                        encode_softlex_batch(
                            batch,
                            offsets,
                            softlexicon,
                            len(labels),
                            int(encoded["input_ids"].shape[1]),
                        )
                    )
                inputs = _normalise_inputs(encoded, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=True,
                ):
                    device_logits = model(**inputs).float()
                if not torch.isfinite(device_logits).all():
                    raise FloatingPointError(
                        "non-finite argument calibration logits"
                    )
                logits = device_logits.cpu()
                for index, (row, row_offsets) in enumerate(zip(batch, offsets)):
                    for threshold in threshold_grid:
                        predictions[threshold].append(
                            decode_mentions(
                                row["text"],
                                row_offsets,
                                logits[index],
                                labels,
                                threshold,
                            )
                        )
                    gold_rows.append(gold_mentions(row))
        result = []
        for threshold in threshold_grid:
            aggregate = mention_metric(predictions[threshold], gold_rows)
            result.append(
                {
                    "threshold": threshold,
                    "aggregate": aggregate,
                    "by_type": mention_metric_by_label(
                        predictions[threshold],
                        gold_rows,
                        labels,
                    ),
                    "predicted_mentions": sum(
                        len(row) for row in predictions[threshold]
                    ),
                    "gold_mentions": sum(len(row) for row in gold_rows),
                }
            )
        return result

    history: list[dict[str, Any]] = []
    best_rank: tuple[float, float, float] | None = None
    best_epoch: int | None = None
    best_selection: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    total_skipped_steps = 0
    total_scheduler_steps_taken = 0
    joint_scheduler_steps_taken = 0
    dropped_by_epoch: list[int] = []
    encoder_parameters = [parameter for _, parameter in encoder_named]
    head_parameters = [parameter for _, parameter in head_named]

    for epoch in range(1, args.epochs + 1):
        phase = (
            "head_only"
            if epoch <= args.head_warmup_epochs
            else "joint"
        )
        encoder_trainable = phase == "joint"
        for name, parameter in encoder_named:
            parameter.requires_grad_(
                encoder_trainable and name in selected_encoder_names
            )
        model.train()
        model.encoder.eval()
        optimiser.zero_grad(set_to_none=True)
        loss_sum = 0.0
        epoch_dropped = 0
        epoch_step_attempts = 0
        epoch_skipped_steps = 0
        epoch_nonfinite_grad_norm_steps = 0
        epoch_encoder_grad_norm_max = 0.0
        epoch_head_grad_norm_max = 0.0
        scale_start = float(scaler.get_scale())
        for step, (inputs, dense_labels, dropped) in enumerate(
            train_loader,
            start=1,
        ):
            epoch_dropped += dropped
            inputs = _normalise_inputs(inputs, device)
            dense_labels = dense_labels.to(device)
            divisor = accumulation_group_size(
                step,
                len(train_loader),
                args.gradient_accumulation,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=True,
            ):
                unscaled_loss = pointer_group_loss(
                    model(**inputs),
                    dense_labels,
                    head_reduction=FIXED_HEAD_REDUCTION,
                )
                loss = unscaled_loss / divisor
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite argument loss at epoch={epoch} step={step}"
                )
            scaler.scale(loss).backward()
            unscaled_loss_value = float(unscaled_loss.detach().cpu())
            loss_sum += unscaled_loss_value
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimiser)
                head_norm = torch.nn.utils.clip_grad_norm_(
                    head_parameters,
                    args.head_gradient_clip,
                )
                head_norm_value = float(head_norm.detach().cpu())
                encoder_norm_value = None
                if encoder_trainable:
                    encoder_norm = torch.nn.utils.clip_grad_norm_(
                        encoder_parameters,
                        args.encoder_gradient_clip,
                    )
                    encoder_norm_value = float(
                        encoder_norm.detach().cpu()
                    )
                finite_norms = math.isfinite(head_norm_value) and (
                    encoder_norm_value is None
                    or math.isfinite(encoder_norm_value)
                )
                if math.isfinite(head_norm_value):
                    epoch_head_grad_norm_max = max(
                        epoch_head_grad_norm_max,
                        head_norm_value,
                    )
                if (
                    encoder_norm_value is not None
                    and math.isfinite(encoder_norm_value)
                ):
                    epoch_encoder_grad_norm_max = max(
                        epoch_encoder_grad_norm_max,
                        encoder_norm_value,
                    )
                if not finite_norms:
                    epoch_nonfinite_grad_norm_steps += 1
                set_phase_learning_rates(
                    phase=phase,
                    head_step=total_scheduler_steps_taken + 1,
                    encoder_step=(
                        joint_scheduler_steps_taken + 1
                        if encoder_trainable
                        else 1
                    ),
                )
                scale_before = float(scaler.get_scale())
                scaler.step(optimiser)
                scaler.update()
                scale_after = float(scaler.get_scale())
                step_succeeded = scale_after >= scale_before
                epoch_step_attempts += 1
                if step_succeeded:
                    total_scheduler_steps_taken += 1
                    if encoder_trainable:
                        joint_scheduler_steps_taken += 1
                else:
                    epoch_skipped_steps += 1
                    total_skipped_steps += 1
                if not finite_norms and step_succeeded:
                    raise FloatingPointError(
                        "GradScaler accepted non-finite argument gradients"
                    )
                optimiser.zero_grad(set_to_none=True)
                if (
                    epoch_step_attempts == 1
                    or epoch_step_attempts % 100 == 0
                    or step == len(train_loader)
                ):
                    print(
                        json.dumps(
                            {
                                "event": "cmeie_argument_training_progress",
                                "epoch": epoch,
                                "micro_batch": step,
                                "micro_batches": len(train_loader),
                                "optimizer_step_attempt": epoch_step_attempts,
                                "loss": unscaled_loss_value,
                                "gradient_accumulation_divisor": divisor,
                                "phase": phase,
                                "encoder_trainable": encoder_trainable,
                                "encoder_pre_clip_grad_norm": encoder_norm_value,
                                "head_pre_clip_grad_norm": head_norm_value,
                                "amp_scale": scale_after,
                                "optimizer_step_succeeded": step_succeeded,
                                "learning_rates": [
                                    float(group["lr"])
                                    for group in optimiser.param_groups
                                ],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                        ),
                        flush=True,
                    )
        if epoch_step_attempts != steps_per_epoch:
            raise RuntimeError(
                "optimizer step attempts differ from the declared schedule"
            )
        dropped_by_epoch.append(epoch_dropped)
        if len(set(dropped_by_epoch)) != 1:
            raise RuntimeError("dropped label count changed across fit epochs")
        selections = calibration_scores()
        calibration_record = calibration_history_fields(selections)
        selection = calibration_record["selected_calibration"]
        metric = selection["aggregate"]
        current_rank = (
            float(metric["f1"]),
            float(metric["precision"]),
            float(selection["threshold"]),
        )
        best_rank, stale_epochs, improved = advance_sparse_early_stopping(
            best_rank,
            stale_epochs,
            current_rank,
        )
        if improved:
            best_epoch = epoch
            best_selection = selection
            best_state = _state_on_cpu(model)
        record = {
            "epoch": epoch,
            "phase": phase,
            "encoder_scope": args.encoder_scope,
            "selected_encoder_parameter_count": (
                selected_encoder_parameter_count
                if encoder_trainable
                else 0
            ),
            "encoder_mode": (
                "eval_with_gradients"
                if encoder_trainable
                else "eval_frozen"
            ),
            "encoder_trainable": encoder_trainable,
            "mean_train_loss": loss_sum / max(1, len(train_loader)),
            "dropped_label_count": epoch_dropped,
            **calibration_record,
            "threshold_grid": list(threshold_grid),
            "optimizer_step_attempts": epoch_step_attempts,
            "successful_optimizer_steps": (
                epoch_step_attempts - epoch_skipped_steps
            ),
            "skipped_optimizer_steps": epoch_skipped_steps,
            "nonfinite_grad_norm_steps": epoch_nonfinite_grad_norm_steps,
            "encoder_pre_clip_grad_norm_max": (
                epoch_encoder_grad_norm_max
                if encoder_trainable
                else None
            ),
            "head_pre_clip_grad_norm_max": epoch_head_grad_norm_max,
            "amp_scale_start": scale_start,
            "amp_scale_end": float(scaler.get_scale()),
            "learning_rates_end": [
                float(group["lr"]) for group in optimiser.param_groups
            ],
            "stale_epochs": stale_epochs,
        }
        history.append(record)
        print(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        if stale_epochs >= args.patience:
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "epoch": epoch,
                        "best_f1": best_rank[0],
                        },
                        sort_keys=True,
                        allow_nan=False,
                    ),
                flush=True,
            )
            break

    if best_state is None or best_epoch is None or best_selection is None:
        raise RuntimeError("argument training produced no selectable checkpoint")
    dropped_label_count = dropped_by_epoch[0]
    if sha256_file(Path(__file__)) != training_script_sha256:
        raise RuntimeError("training script changed during the run")
    if sha256_file(argument_module_path) != argument_module_sha256:
        raise RuntimeError("argument module changed during the run")
    if sha256_file(balanced_loss_module_path) != balanced_loss_module_sha256:
        raise RuntimeError("balanced loss module changed during the run")
    if sha256_file(softlexicon_module_path) != softlexicon_module_sha256:
        raise RuntimeError("SoftLexicon module changed during the run")
    if sha256_file(overfit_preflight_path) != overfit_preflight_binding["sha256"]:
        raise RuntimeError("argument overfit preflight changed during the run")
    rebound = load_run_binding(args.run_root)
    if rebound != run_binding:
        raise RuntimeError("source run binding changed during training")
    if sha256_tree(base_model_path) != base_model_tree:
        raise RuntimeError("base model tree changed during training")

    publish_candidate, rejection_reason = candidate_publication_decision(
        best_selection
    )
    if not publish_candidate:
        rejection_payload = {
            "schema_version": 2,
            "kind": "cmeie_argument_feasibility_rejected",
            "created_at": _utc_now(),
            "reason": rejection_reason,
            "rid": run_binding.rid,
            "git_head": run_binding.git_head,
            "seed": args.seed,
            "ablation_role": ablation_role,
            "declared_output": str(output_path),
            "source_tree_sha256": run_binding.included_tree_sha256,
            "source_manifest_sha256": run_binding.source_manifest_sha256,
            "split_manifest_sha256": fit_binding["split_manifest_sha256"],
            "fit_shard_sha256": fit_binding["shard_sha256"],
            "calibration_shard_sha256": calibration_binding["shard_sha256"],
            "base_model_tree": base_model_tree,
            "training_script_sha256": training_script_sha256,
            "argument_module_sha256": argument_module_sha256,
            "balanced_loss_module_sha256": balanced_loss_module_sha256,
            "softlexicon": softlexicon_binding,
            "overfit_preflight": overfit_preflight_binding,
            "head_reduction": FIXED_HEAD_REDUCTION,
            "pointer_variant": args.pointer_variant,
            "use_bilstm": args.use_bilstm,
            "precision": precision_profile,
            "encoder_mode": encoder_mode,
            "encoder_scope": args.encoder_scope,
            "selected_encoder_parameter_count": (
                selected_encoder_parameter_count
            ),
            "training_strategy": training_strategy,
            "fit_validation": fit_binding,
            "calibration_validation": calibration_binding,
            "best_epoch": best_epoch,
            "selected_calibration": best_selection,
            "history": history,
            "outer_accessed": False,
            "official_dev_accessed": False,
        }
        publish_json_create_once(rejected_marker, rejection_payload)
        print(
            json.dumps(
                {
                    "error": "argument feasibility candidate rejected",
                    "reason": rejection_reason,
                    "rejected_marker": str(rejected_marker),
                    "output_published": False,
                    "exit_code": REJECTED_EXIT_CODE,
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return REJECTED_EXIT_CODE

    model.load_state_dict(best_state)

    def write_candidate(staging: Path) -> None:
        model.encoder.save_pretrained(staging)
        tokenizer.save_pretrained(staging)
        head_path = staging / ARGUMENT_HEAD_WEIGHTS
        torch.save(model.gp.state_dict(), head_path)
        bilstm_path = None
        if model.bilstm is not None:
            bilstm_path = staging / ARGUMENT_BILSTM_WEIGHTS
            torch.save(model.bilstm.state_dict(), bilstm_path)
        softlexicon_path = None
        softlexicon_weights_path = None
        if model.softlex_proj is not None:
            if softlexicon_bytes is None or softlexicon_digest is None:
                raise RuntimeError("enabled SoftLexicon lacks a bound payload")
            softlexicon_path = staging / ARGUMENT_SOFTLEXICON
            softlexicon_path.write_bytes(softlexicon_bytes)
            if sha256_file(softlexicon_path) != softlexicon_digest:
                raise RuntimeError("published SoftLexicon hash mismatch")
            softlexicon_weights_path = staging / ARGUMENT_SOFTLEXICON_WEIGHTS
            torch.save(model.softlex_proj.state_dict(), softlexicon_weights_path)
        contract = argument_model_contract(
            labels=labels,
            max_length=args.max_length,
            selected_threshold=best_selection["threshold"],
            pointer_variant=args.pointer_variant,
            use_bilstm=args.use_bilstm,
            softlexicon_sha256=softlexicon_digest,
            softlexicon_min_chars=(
                FIXED_SOFTLEXICON_MIN_CHARS
                if args.softlexicon
                else None
            ),
        )
        contract_path = staging / ARGUMENT_MODEL_CONFIG
        contract_path.write_text(
            json.dumps(
                contract,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        candidate_payload_tree = sha256_tree(staging)
        receipt = {
            "schema_version": 6,
            "kind": "cmeie_argument_feasibility_training",
            "dataset_role": "fit_training_calibration_selection",
            "route": "cmeie_native_argument_ner",
            "created_at": _utc_now(),
            "rid": run_binding.rid,
            "git_head": run_binding.git_head,
            "source_tree_sha256": run_binding.included_tree_sha256,
            "source_manifest_sha256": run_binding.source_manifest_sha256,
            "outer_accessed": False,
            "official_dev_accessed": False,
            "seed": args.seed,
            "ablation_role": ablation_role,
            "fit_validation": fit_binding,
            "calibration_validation": calibration_binding,
            "base_model": str(base_model_path),
            "base_model_tree": base_model_tree,
            "runtime_environment": runtime_environment,
            "labels": labels,
            "n_labels": len(labels),
            "best_epoch": best_epoch,
            "selected_calibration": best_selection,
            "history": history,
            "dropped_label_count": dropped_label_count,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "model_contract": contract,
            "model_contract_sha256": sha256_file(contract_path),
            "head_weights_sha256": sha256_file(head_path),
            "bilstm_weights_sha256": (
                sha256_file(bilstm_path)
                if bilstm_path is not None
                else None
            ),
            "softlexicon": softlexicon_binding,
            "softlexicon_weights_sha256": (
                sha256_file(softlexicon_weights_path)
                if softlexicon_weights_path is not None
                else None
            ),
            "candidate_payload_tree": candidate_payload_tree,
            "training_script_sha256": training_script_sha256,
            "argument_module_sha256": argument_module_sha256,
            "balanced_loss_module_sha256": balanced_loss_module_sha256,
            "softlexicon_module_sha256": softlexicon_module_sha256,
            "overfit_preflight": overfit_preflight_binding,
            "training_strategy": training_strategy,
            "encoder_scope": args.encoder_scope,
            "selected_encoder_parameter_count": (
                selected_encoder_parameter_count
            ),
            "configuration": {
                "pointer_variant": args.pointer_variant,
                "use_bilstm": args.use_bilstm,
                "softlexicon": softlexicon_binding,
                "ablation_role": ablation_role,
                "inner_dim": INNER_DIM,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "calibration_batch_size": args.calibration_batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "max_length": args.max_length,
                "encoder_lr": args.encoder_lr,
                "head_lr": args.head_lr,
                "warmup": args.warmup,
                "weight_decay": args.weight_decay,
                "encoder_gradient_clip": args.encoder_gradient_clip,
                "head_gradient_clip": args.head_gradient_clip,
                "head_warmup_epochs": args.head_warmup_epochs,
                "head_reduction": FIXED_HEAD_REDUCTION,
                "encoder_mode": encoder_mode,
                "encoder_scope": args.encoder_scope,
                "training_strategy": training_strategy,
                "patience": args.patience,
                "threshold_grid": list(threshold_grid),
                "device": "npu:0",
                "precision": precision_profile["precision"],
                "fp16": precision_profile["precision"] == "fp16",
                "bf16": precision_profile["precision"] == "bf16",
                "uses_grad_scaler": precision_profile["uses_grad_scaler"],
                "amp_initial_scale": 1024.0,
            },
            "optimizer_steps_per_epoch": steps_per_epoch,
            "declared_scheduler_steps": total_scheduler_steps,
            "successful_scheduler_steps": total_scheduler_steps_taken,
            "successful_joint_scheduler_steps": joint_scheduler_steps_taken,
            "declared_joint_scheduler_steps": joint_total_steps,
            "head_scheduler_warmup_steps": head_warmup_steps,
            "encoder_scheduler_warmup_steps": encoder_warmup_steps,
            "skipped_optimizer_steps": total_skipped_steps,
        }
        receipt_path = staging / "training_receipt.json"
        receipt_path.write_text(
            json.dumps(
                receipt,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    _publish_model_directory(output_path, write_candidate)
    receipt_path = output_path / "training_receipt.json"
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    candidate_payload_tree = sha256_tree(
        output_path,
        excluded_relative_paths=frozenset({"training_receipt.json"}),
    )
    if receipt_payload.get("candidate_payload_tree") != candidate_payload_tree:
        raise RuntimeError("published candidate payload tree differs from receipt")
    publish_json_create_once(
        completed_marker,
        {
            "schema_version": 1,
            "kind": "cmeie_argument_feasibility_completed",
            "created_at": _utc_now(),
            "rid": run_binding.rid,
            "seed": args.seed,
            "output": str(output_path),
            "training_receipt_sha256": sha256_file(receipt_path),
            "candidate_payload_tree": candidate_payload_tree,
            "best_epoch": best_epoch,
            "selected_calibration": best_selection,
            "outer_accessed": False,
            "official_dev_accessed": False,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "best_epoch": best_epoch,
                "selected_calibration": best_selection,
                "completed_marker": str(completed_marker),
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
