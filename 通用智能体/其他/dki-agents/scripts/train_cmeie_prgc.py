"""Train the bounded PRGC fallback for CMeIE direct text-to-SPO extraction.

This entrypoint is intentionally narrower than the historical experimental
trainers.  In formal mode it accepts only the audited stable recipe: four
head-only epochs followed by two top-2 joint epochs on one Ascend NPU.  It
never opens outer or official-dev data, and it writes a failed calibration
receipt rather than improvising a threshold or encoder sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = (-0.5, 0.0, 0.5)
FORMAL_EPOCHS = 6
FORMAL_HEAD_ONLY_EPOCHS = 4


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"PRGC evidence exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_torch_once(path: Path, payload: dict[str, Any], torch: Any) -> str:
    if path.exists():
        raise FileExistsError(f"PRGC checkpoint exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha(path)


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _gold(row: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (
            str(spo.get("subject", "")),
            str(spo.get("predicate", "")),
            str(spo.get("object", "")),
        )
        for spo in row.get("spo_list", [])
    }


def _predicates(rows: Iterable[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(spo.get("predicate", ""))
            for row in rows
            for spo in row.get("spo_list", [])
            if str(spo.get("predicate", ""))
        }
    )


def _tokenize(rows: list[dict[str, Any]], tokenizer: Any, device: Any) -> tuple[dict[str, Any], list[list[tuple[int, int]]]]:
    encoded = tokenizer(
        [str(row["text"]) for row in rows],
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=320,
    )
    offsets = [
        [tuple(pair) for pair in item.tolist()]
        for item in encoded.pop("offset_mapping")
    ]
    inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    return inputs, offsets


def _make_training_batch(
    rows: list[dict[str, Any],],
    tokenizer: Any,
    predicate2id: dict[str, int],
    device: Any,
) -> tuple[dict[str, Any], list[Any], int]:
    from dki.nlp.cmeie_prgc import build_prgc_labels

    inputs, offsets = _tokenize(rows, tokenizer, device)
    labels = [
        build_prgc_labels(row, offsets[index], predicate2id)
        for index, row in enumerate(rows)
    ]
    return inputs, labels, sum(item.dropped_spo for item in labels)


def _evaluate(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    predicates: list[str],
    device: Any,
    *,
    batch_size: int,
    fp16: bool,
) -> dict[str, Any]:
    """Evaluate the predeclared 3x3x3x2 calibration grid only."""

    import torch

    from dki.nlp.cmeie_prgc import PRGCDensityError, decode_prgc

    combinations = tuple(
        (relation, span, correspondence, schema)
        for relation, span, correspondence in itertools.product(THRESHOLDS, repeat=3)
        for schema in (False, True)
    )
    counts = {
        item: {"tp": 0, "fp": 0, "fn": 0}
        for item in combinations
    }
    rejections = {item: 0 for item in combinations}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            inputs, offsets = _tokenize(batch, tokenizer, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=fp16,
            ):
                output = model(**inputs)
            relation = output["relation"].detach().float().cpu()
            channel_span = output["span"].detach().float().cpu().reshape(
                len(batch), len(predicates), -1, 4
            )
            channel_correspondence = output["correspondence"].detach().float().cpu().reshape(
                len(batch), len(predicates), -1, channel_span.shape[2]
            )
            for row_index, row in enumerate(batch):
                expected = _gold(row)
                for combination in combinations:
                    relation_threshold, span_threshold, correspondence_threshold, schema = combination
                    try:
                        predicted = decode_prgc(
                            str(row["text"]),
                            offsets[row_index],
                            relation[row_index],
                            channel_span[row_index],
                            channel_correspondence[row_index],
                            predicates,
                            (relation_threshold, span_threshold, correspondence_threshold),
                            schema_constraint=schema,
                        )
                    except PRGCDensityError:
                        rejections[combination] += 1
                        continue
                    counts[combination]["tp"] += len(predicted & expected)
                    counts[combination]["fp"] += len(predicted - expected)
                    counts[combination]["fn"] += len(expected - predicted)
    scored = []
    for combination in combinations:
        count = counts[combination]
        if rejections[combination]:
            continue
        tp, fp, fn = count["tp"], count["fp"], count["fn"]
        metric = {
            **count,
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        }
        scored.append((metric, combination))
    if not scored:
        metric = {
            "tp": 0,
            "fp": 0,
            "fn": sum(len(_gold(row)) for row in rows),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        selected = (0.5, 0.5, 0.5, True)
        all_rejected = True
    else:
        metric, selected = max(
            scored,
            key=lambda item: (
                item[0]["f1"],
                item[0]["precision"],
                sum(item[1][:3]),
                item[1][3],
            ),
        )
        all_rejected = False
    return {
        "metric": metric,
        "relation_threshold": selected[0],
        "span_threshold": selected[1],
        "correspondence_threshold": selected[2],
        "schema_constraint": "on" if selected[3] else "off",
        "grid_size": len(combinations),
        "safe_grid_size": len(scored),
        "all_grid_combinations_rejected": all_rejected,
        "density_rejections_by_combination": {
            "|".join(
                [str(item[0]), str(item[1]), str(item[2]), "on" if item[3] else "off"]
            ): count
            for item, count in rejections.items()
            if count
        },
        "schema_constraint_note": "type-free self-surface filter only; PRGC has no type head",
    }


def _head_parameters(model: Any) -> list[Any]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]


def _configure_top2_encoder(model: Any, top_layers: int) -> tuple[list[Any], str]:
    if top_layers != 2:
        raise ValueError("PRGC stable formal recipe requires exactly top-2 encoder layers")
    encoder_parameters = list(model.encoder.parameters())
    for parameter in encoder_parameters:
        parameter.requires_grad_(False)
    candidates = (
        getattr(getattr(model.encoder, "encoder", None), "layer", None),
        getattr(getattr(model.encoder, "transformer", None), "layer", None),
        getattr(model.encoder, "layer", None),
    )
    layers = next((candidate for candidate in candidates if candidate is not None), None)
    if layers is None or not hasattr(layers, "__len__") or len(layers) < 2:
        raise ValueError("encoder does not expose at least two transformer layers")
    selected: list[Any] = []
    for layer in list(layers)[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
            selected.append(parameter)
    if not selected:
        raise RuntimeError("top-2 configuration selected no encoder parameters")
    return selected, "top_2_transformer_layers"


def _clip_gradients(model: Any, torch: Any, *, encoder_frozen: bool) -> dict[str, float | None]:
    subject = [
        *model.subject_boundary.parameters(),
        *model.subject_boundary_bias.parameters(),
        *model.correspondence_subject.parameters(),
    ]
    object_and_verifier = [
        parameter
        for module in (
            model.object_boundary,
            model.object_boundary_bias,
            model.boundary_token_projection,
            model.boundary_token_norm,
            model.correspondence_object,
            model.correspondence_predicate,
            model.relation_presence,
            model.predicate_embedding,
            model.predicate_condition,
            model.conditioned_norm,
        )
        for parameter in module.parameters()
    ]
    object_and_verifier.extend(
        [model.span_logit_prior, model.correspondence_logit_prior]
    )
    subject_norm = torch.nn.utils.clip_grad_norm_(subject, 1.0)
    object_norm = torch.nn.utils.clip_grad_norm_(object_and_verifier, 1.0)
    encoder_norm = None
    if not encoder_frozen:
        encoder_norm = torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters()), 1.0)
    return {
        "encoder": None if encoder_norm is None else float(encoder_norm.detach().cpu()),
        "subject": float(subject_norm.detach().cpu()),
        "object_verifier": float(object_norm.detach().cpu()),
    }


def _run_micro_batch(
    model: Any,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    predicate2id: dict[str, int],
    device: Any,
    *,
    fp16: bool,
) -> tuple[Any, int, dict[str, float]]:
    import torch

    from dki.nlp.cmeie_prgc import (
        build_selected_targets,
        prgc_loss,
        select_training_predicates,
    )

    inputs, labels, dropped = _make_training_batch(rows, tokenizer, predicate2id, device)
    relation_labels = torch.stack([item.relation for item in labels]).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
        hidden = model.encode(**inputs)
        relation_logits = model.relation_logits(hidden, inputs["attention_mask"])
        parent_values, predicate_values = select_training_predicates(relation_logits, labels)
        parents = torch.tensor(parent_values, dtype=torch.long, device=device)
        predicate_ids = torch.tensor(predicate_values, dtype=torch.long, device=device)
        outputs = model.score_channels(
            hidden,
            parent_indices=parents,
            predicate_ids=predicate_ids,
        )
        outputs["relation"] = relation_logits
        span_labels, correspondence_labels = build_selected_targets(
            labels,
            parent_values,
            predicate_values,
            int(inputs["input_ids"].shape[1]),
        )
        total, parts = prgc_loss(
            outputs,
            relation_labels,
            span_labels.to(device),
            correspondence_labels.to(device),
            inputs["attention_mask"],
            hard_negative_ratio=8,
            inactive_channel_weight=0.25,
        )
    return total, dropped, {name: float(value.detach().cpu()) for name, value in parts.items()}


def _train_steps(
    model: Any,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    predicate2id: dict[str, int],
    device: Any,
    optimizer: Any,
    scaler: Any,
    torch: Any,
    *,
    batch_size: int,
    accumulation: int,
    fp16: bool,
    seed: int,
    encoder_frozen: bool,
    maximum_optimizer_steps: int | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    if maximum_optimizer_steps is not None:
        required = maximum_optimizer_steps * accumulation * batch_size
        repeats = math.ceil(required / max(1, len(ordered)))
        ordered = (ordered * repeats)[:required]
    batches = [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]
    model.train()
    # Frozen and top-2 joint phases keep encoder dropout off.  Only the top
    # two layers gain gradients after the phase transition.
    model.encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    parts_total = {"relation": 0.0, "span": 0.0, "correspondence": 0.0}
    dropped = 0
    optimizer_steps = attempts = skipped = nonfinite = 0
    maximum_norms = {"encoder": 0.0, "subject": 0.0, "object_verifier": 0.0}
    for index, batch in enumerate(batches, start=1):
        group_start = ((index - 1) // accumulation) * accumulation + 1
        divisor = min(accumulation, len(batches) - group_start + 1)
        total, lost, parts = _run_micro_batch(
            model,
            batch,
            tokenizer,
            predicate2id,
            device,
            fp16=fp16,
        )
        if not torch.isfinite(total):
            raise RuntimeError("non-finite PRGC loss")
        scaler.scale(total / divisor).backward()
        losses.append(float(total.detach().cpu()))
        dropped += lost
        for name, value in parts.items():
            parts_total[name] += value
        if index % accumulation != 0 and index != len(batches):
            continue
        scaler.unscale_(optimizer)
        norms = _clip_gradients(model, torch, encoder_frozen=encoder_frozen)
        bad = any(value is not None and not math.isfinite(value) for value in norms.values())
        if bad:
            nonfinite += 1
        for name, value in norms.items():
            if value is not None:
                maximum_norms[name] = max(maximum_norms[name], float(value))
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        optimizer.zero_grad(set_to_none=True)
        attempts += 1
        if scale_after >= scale_before:
            optimizer_steps += 1
            if scheduler is not None:
                scheduler.step()
        else:
            skipped += 1
        if bad and scale_after >= scale_before:
            raise RuntimeError("GradScaler accepted non-finite PRGC gradients")
        if maximum_optimizer_steps is not None and optimizer_steps >= maximum_optimizer_steps:
            break
    return {
        "mean_loss": sum(losses) / max(1, len(losses)),
        "mean_loss_parts": {name: value / max(1, len(losses)) for name, value in parts_total.items()},
        "micro_batches": len(losses),
        "optimizer_steps": optimizer_steps,
        "optimizer_step_attempts": attempts,
        "skipped_optimizer_steps": skipped,
        "nonfinite_gradient_steps": nonfinite,
        "maximum_pre_clip_gradient_norms": maximum_norms,
        "dropped_spo": dropped,
        "amp_scale": float(scaler.get_scale()),
    }


def _formal_recipe_valid(args: argparse.Namespace) -> bool:
    return (
        args.epochs == FORMAL_EPOCHS
        and args.head_only_epochs == FORMAL_HEAD_ONLY_EPOCHS
        and args.batch_size == 8
        and args.gradient_accumulation == 4
        and args.max_length == 320
        and args.joint_top_encoder_layers == 2
        and math.isclose(args.joint_encoder_lr, 1e-6, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(args.joint_encoder_adam_eps, 1e-4, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(args.head_lr, 5e-5, rel_tol=0.0, abs_tol=1e-12)
        and args.hard_negative_ratio == 8
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("overfit", "train"), required=True)
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--overfit-receipt", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--head-only-epochs", type=int, default=FORMAL_HEAD_ONLY_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--calibration-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--overfit-rows", type=int, default=8)
    parser.add_argument("--overfit-head-only-steps", type=int, default=1200)
    parser.add_argument("--overfit-joint-steps", type=int, default=100)
    parser.add_argument("--joint-top-encoder-layers", type=int, default=2)
    parser.add_argument("--joint-encoder-lr", type=float, default=1e-6)
    parser.add_argument("--joint-encoder-adam-eps", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--hard-negative-ratio", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--minimum-calibration-f1", type=float, default=0.45)
    parser.add_argument("--diagnostic-full-unfreeze", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.fit_data = args.fit_data.resolve()
    args.split_manifest = args.split_manifest.resolve()
    args.base_model = args.base_model.resolve()
    args.output = args.output.resolve()
    if args.calibration_data is not None:
        args.calibration_data = args.calibration_data.resolve()
    if args.overfit_receipt is not None:
        args.overfit_receipt = args.overfit_receipt.resolve()
    if args.output.exists():
        raise SystemExit(f"PRGC output exists: {args.output}")
    if args.max_length != 320:
        raise SystemExit("PRGC requires max_length=320")
    if args.hard_negative_ratio != 8:
        raise SystemExit("PRGC requires hard_negative_ratio=8")
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        raise SystemExit("PRGC batch size and accumulation must be positive")
    if args.mode == "train" and not _formal_recipe_valid(args):
        raise SystemExit(
            "formal PRGC requires 4 head-only + 2 top-2 epochs, batch=8, "
            "accumulation=4, LR=1e-6/5e-5, encoder eps=1e-4 and HNR=8"
        )
    if args.mode == "train" and args.diagnostic_full_unfreeze:
        raise SystemExit("formal PRGC forbids diagnostic full encoder unfreeze")
    if args.mode == "overfit" and args.joint_top_encoder_layers == 0 and not args.diagnostic_full_unfreeze:
        raise SystemExit("full encoder unfreeze is diagnostic-only")
    if args.joint_top_encoder_layers != 2:
        raise SystemExit("PRGC stable route only permits top-2 encoder adaptation")
    if not args.fit_data.is_file() or not args.split_manifest.is_file() or not args.base_model.is_dir():
        raise SystemExit("PRGC fit/split/base-model input is missing")
    if args.mode == "overfit":
        if args.calibration_data is not None:
            raise SystemExit("PRGC overfit forbids calibration data")
        if args.overfit_receipt is not None:
            raise SystemExit("PRGC overfit must not consume an overfit receipt")
    else:
        if args.calibration_data is None or not args.calibration_data.is_file():
            raise SystemExit("formal PRGC requires calibration data")
        if args.overfit_receipt is None or not args.overfit_receipt.is_file():
            raise SystemExit("formal PRGC requires a passing overfit receipt")
        overfit = json.loads(args.overfit_receipt.read_text(encoding="utf-8"))
        if overfit.get("passed") is not True or overfit.get("calibration_accessed") is not False:
            raise SystemExit("PRGC overfit receipt is not promotion eligible")

    import torch
    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_prgc import CMeIEPRGC, PRGC_ARCHITECTURE_VERSION

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("PRGC requires Ascend NPU")
    if not hasattr(torch.npu, "amp") or not hasattr(torch.npu.amp, "GradScaler"):
        raise SystemExit("PRGC requires torch.npu.amp.GradScaler")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("npu:0")
    fit = [row for row in _jsonl(args.fit_data) if row.get("spo_list")]
    predicates = _predicates(fit)
    if len(predicates) != 44:
        raise SystemExit(f"PRGC expected 44 predicates, received {len(predicates)}")
    predicate2id = {value: index for index, value in enumerate(predicates)}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)
    encoder = AutoModel.from_pretrained(args.base_model, local_files_only=True)
    model = CMeIEPRGC(encoder, int(encoder.config.hidden_size), len(predicates)).to(device)
    scaler = torch.npu.amp.GradScaler(init_scale=1024.0, enabled=args.fp16)
    args.output.mkdir(parents=True, exist_ok=False)
    common = {
        "schema_version": 1,
        "model_family": "prgc",
        "architecture_version": PRGC_ARCHITECTURE_VERSION,
        "source_tree_sha256": args.source_tree_sha256,
        "fit_sha256": _sha(args.fit_data),
        "split_manifest_sha256": _sha(args.split_manifest),
        "base_model": str(args.base_model),
        "base_model_config_sha256": _sha(args.base_model / "config.json"),
        "predicates": predicates,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer_recipe": {
            "head_only_epochs": FORMAL_HEAD_ONLY_EPOCHS,
            "joint_encoder_scope": "top_2_transformer_layers",
            "encoder_lr": 1e-6,
            "encoder_adam_eps": 1e-4,
            "encoder_weight_decay": 0.0,
            "head_lr": 5e-5,
            "head_weight_decay": 0.01,
            "hard_negative_ratio": 8,
            "inactive_channel_weight": 0.25,
            "encoder_eval_mode": True,
            "gradient_clip": {"encoder": 1.0, "subject": 1.0, "object_verifier": 1.0},
        },
        "device": "npu:0",
        "fp16": bool(args.fp16),
        "outer_accessed": False,
        "official_dev_accessed": False,
        "script_sha256": _sha(Path(__file__)),
    }

    if args.mode == "overfit":
        rows = fit[: args.overfit_rows]
        if len(rows) != 8 or sum(len(row.get("spo_list", [])) for row in rows) != 44:
            raise SystemExit("PRGC overfit must use exactly the fixed first 8 fit rows / 44 SPOs")
        encoder_parameters = list(model.encoder.parameters())
        for parameter in encoder_parameters:
            parameter.requires_grad_(False)
        head_parameters = _head_parameters(model)
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": 0.0, "eps": 1e-4, "weight_decay": 0.0},
                {"params": head_parameters, "lr": args.head_lr, "eps": 1e-8, "weight_decay": 0.01},
            ]
        )
        head_only = _train_steps(
            model, rows, tokenizer, predicate2id, device, optimizer, scaler, torch,
            batch_size=min(args.batch_size, len(rows)), accumulation=1, fp16=args.fp16,
            seed=args.seed, encoder_frozen=True,
            maximum_optimizer_steps=args.overfit_head_only_steps,
        )
        head_only_evaluation = _evaluate(
            model, tokenizer, rows, predicates, device,
            batch_size=min(args.calibration_batch_size, len(rows)), fp16=args.fp16,
        )
        top2_parameters, scope = _configure_top2_encoder(model, args.joint_top_encoder_layers)
        optimizer.param_groups[0]["lr"] = args.joint_encoder_lr
        joint = _train_steps(
            model, rows, tokenizer, predicate2id, device, optimizer, scaler, torch,
            batch_size=min(args.batch_size, len(rows)), accumulation=1, fp16=args.fp16,
            seed=args.seed + 1, encoder_frozen=False,
            maximum_optimizer_steps=args.overfit_joint_steps,
        )
        joint_evaluation = _evaluate(
            model, tokenizer, rows, predicates, device,
            batch_size=min(args.calibration_batch_size, len(rows)), fp16=args.fp16,
        )
        checkpoint_sha = _save_torch_once(
            args.output / "model.pt",
            {
                "model": _cpu_state_dict(model),
                "predicates": predicates,
                "model_family": "prgc",
                "architecture_version": PRGC_ARCHITECTURE_VERSION,
            },
            torch,
        )
        encoder_norm = float(joint["maximum_pre_clip_gradient_norms"]["encoder"])
        passed = (
            head_only_evaluation["metric"]["f1"] >= 0.98
            and joint_evaluation["metric"]["f1"] >= 0.95
            and head_only["dropped_spo"] == 0
            and joint["dropped_spo"] == 0
            and head_only["nonfinite_gradient_steps"] == 0
            and joint["nonfinite_gradient_steps"] == 0
            and head_only["skipped_optimizer_steps"] == 0
            and joint["skipped_optimizer_steps"] == 0
            and math.isfinite(encoder_norm)
            and encoder_norm <= 10_000.0
        )
        receipt = {
            **common,
            "kind": "cmeie_prgc_overfit_preflight",
            "mode": "overfit",
            "seed": args.seed,
            "rows": len(rows),
            "spo_count": sum(len(row.get("spo_list", [])) for row in rows),
            "head_only_steps": args.overfit_head_only_steps,
            "joint_steps": args.overfit_joint_steps,
            "joint_encoder_scope": scope,
            "joint_trainable_encoder_parameters": sum(parameter.numel() for parameter in top2_parameters),
            "training": {"head_only": head_only, "joint": joint},
            "evaluation": {"head_only": head_only_evaluation, "joint": joint_evaluation},
            "checkpoint_sha256": checkpoint_sha,
            "calibration_accessed": False,
            "passed": passed,
        }
        _write_json_once(args.output / "overfit_receipt.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if passed else 2

    assert args.calibration_data is not None and args.overfit_receipt is not None
    calibration = _jsonl(args.calibration_data)
    encoder_parameters = list(model.encoder.parameters())
    for parameter in encoder_parameters:
        parameter.requires_grad_(False)
    head_parameters = _head_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": 0.0, "eps": 1e-4, "weight_decay": 0.0},
            {"params": head_parameters, "lr": args.head_lr, "eps": 1e-8, "weight_decay": 0.01},
        ]
    )
    history: list[dict[str, Any]] = []
    stale_epochs = 0
    best_rank: tuple[float, float, float, int] | None = None
    scope = "not_run"
    trainable_encoder: list[Any] = []
    scheduler = None
    batches_per_epoch = math.ceil(len(fit) / args.batch_size)
    joint_steps_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation)
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.head_only_epochs:
            for parameter in model.encoder.parameters():
                parameter.requires_grad_(False)
            optimizer.param_groups[0]["lr"] = 0.0
            phase = "head_only"
            encoder_frozen = True
        else:
            if epoch == args.head_only_epochs + 1:
                trainable_encoder, scope = _configure_top2_encoder(model, args.joint_top_encoder_layers)
                optimizer.param_groups[0]["lr"] = args.joint_encoder_lr
                total_joint_steps = joint_steps_per_epoch * (args.epochs - args.head_only_epochs)
                warmup_steps = max(1, round(total_joint_steps * 0.10))
                def schedule(step: int) -> float:
                    if step < warmup_steps:
                        return step / warmup_steps
                    return max(0.0, (total_joint_steps - step) / max(1, total_joint_steps - warmup_steps))
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [schedule, schedule])
            phase = "joint"
            encoder_frozen = False
        training = _train_steps(
            model, fit, tokenizer, predicate2id, device, optimizer, scaler, torch,
            batch_size=args.batch_size, accumulation=args.gradient_accumulation,
            fp16=args.fp16, seed=args.seed * 1000 + epoch,
            encoder_frozen=encoder_frozen, scheduler=scheduler,
        )
        evaluation = _evaluate(
            model, tokenizer, calibration, predicates, device,
            batch_size=args.calibration_batch_size, fp16=args.fp16,
        )
        epoch_dir = args.output / f"epoch-{epoch:02d}"
        epoch_dir.mkdir(exist_ok=False)
        checkpoint_sha = _save_torch_once(
            epoch_dir / "model.pt",
            {
                "model": _cpu_state_dict(model),
                "predicates": predicates,
                "model_family": "prgc",
                "architecture_version": PRGC_ARCHITECTURE_VERSION,
                "epoch": epoch,
            },
            torch,
        )
        record = {
            "epoch": epoch,
            "phase": phase,
            "training": training,
            "calibration": evaluation,
            "checkpoint": str(epoch_dir / "model.pt"),
            "checkpoint_sha256": checkpoint_sha,
        }
        _write_json_once(epoch_dir / "metrics.json", record)
        history.append(record)
        rank = (
            float(evaluation["metric"]["f1"]),
            float(evaluation["metric"]["precision"]),
            float(evaluation["relation_threshold"]) + float(evaluation["span_threshold"]) + float(evaluation["correspondence_threshold"]),
            1 if evaluation["schema_constraint"] == "on" else 0,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            stale_epochs = 0
        elif epoch > args.head_only_epochs:
            stale_epochs += 1
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        if epoch > args.head_only_epochs and stale_epochs >= args.early_stopping_patience:
            break
    best = max(
        history,
        key=lambda item: (
            item["calibration"]["metric"]["f1"],
            item["calibration"]["metric"]["precision"],
            item["calibration"]["relation_threshold"] + item["calibration"]["span_threshold"] + item["calibration"]["correspondence_threshold"],
            item["calibration"]["schema_constraint"] == "on",
        ),
    )
    selection = {
        **common,
        "kind": "cmeie_prgc_calibration_selection",
        "mode": "train",
        "seed": args.seed,
        "calibration_sha256": _sha(args.calibration_data),
        "overfit_receipt_sha256": _sha(args.overfit_receipt),
        "head_only_epochs": args.head_only_epochs,
        "joint_encoder_scope": scope,
        "joint_trainable_encoder_parameters": sum(parameter.numel() for parameter in trainable_encoder),
        "selected_epoch": best["epoch"],
        "checkpoint": best["checkpoint"],
        "checkpoint_sha256": best["checkpoint_sha256"],
        "calibration_metric": best["calibration"]["metric"],
        "thresholds": {
            "relation": best["calibration"]["relation_threshold"],
            "span": best["calibration"]["span_threshold"],
            "correspondence": best["calibration"]["correspondence_threshold"],
            "schema_constraint": best["calibration"]["schema_constraint"],
        },
        "history": history,
        "dataset_role": "fit_training_calibration_selection",
        "selection_used": True,
        "calibration_gate_f1": args.minimum_calibration_f1,
        "calibration_gate_passed": best["calibration"]["metric"]["f1"] >= args.minimum_calibration_f1,
    }
    _write_json_once(args.output / "selection.json", selection)
    if not selection["calibration_gate_passed"]:
        _write_json_once(
            args.output / "FAILED_EVIDENCE.json",
            {
                "kind": "cmeie_prgc_calibration_gate",
                "status": "failed_calibration_gate",
                "observed_f1": selection["calibration_metric"]["f1"],
                "required_f1": args.minimum_calibration_f1,
                "outer_accessed": False,
                "official_dev_accessed": False,
                "selection_sha256": _sha(args.output / "selection.json"),
            },
        )
    print(json.dumps(selection, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if selection["calibration_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
