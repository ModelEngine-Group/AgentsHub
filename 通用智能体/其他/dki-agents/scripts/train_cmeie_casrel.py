"""Train and select a calibration-only CasRel candidate for CMeIE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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
    """Materialize a portable checkpoint without NPU internal-format tensors."""

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


def _valid_subjects(
    spans: list[tuple[int, int, float]],
    offsets: list[tuple[int, int]],
) -> list[tuple[int, int, float]]:
    return [
        span
        for span in spans
        if offsets[span[0]][1] > offsets[span[0]][0]
        and offsets[span[1]][1] > offsets[span[1]][0]
    ]


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
    import torch

    from dki.nlp.cmeie_casrel import decode_casrel, pair_boundary_logits

    thresholds = (-0.5, 0.0, 0.5)
    counts = {
        (subject_threshold, object_threshold): {"tp": 0, "fp": 0, "fn": 0}
        for subject_threshold in thresholds
        for object_threshold in thresholds
    }
    subject_density = {str(threshold): 0 for threshold in thresholds}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                [row["text"] for row in batch],
                return_offsets_mapping=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=320,
            )
            offsets = [
                [tuple(pair) for pair in mapping.tolist()]
                for mapping in encoded.pop("offset_mapping")
            ]
            inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=fp16,
            ):
                hidden = model.encode(**inputs)
                subject_logits = model.subject_logits(hidden)
            for subject_threshold in thresholds:
                by_row: list[list[tuple[int, int, float]]] = []
                parents: list[int] = []
                flat_spans: list[tuple[int, int]] = []
                for index in range(len(batch)):
                    spans = pair_boundary_logits(
                        subject_logits[index, :, 0].float().cpu(),
                        subject_logits[index, :, 1].float().cpu(),
                        subject_threshold,
                    )
                    spans = _valid_subjects(spans, offsets[index])
                    by_row.append(spans)
                    subject_density[str(subject_threshold)] += len(spans)
                    parents.extend([index] * len(spans))
                    flat_spans.extend((item[0], item[1]) for item in spans)
                object_values = None
                if flat_spans:
                    span_tensor = torch.tensor(
                        flat_spans,
                        dtype=torch.long,
                        device=device,
                    )
                    parent_tensor = torch.tensor(
                        parents,
                        dtype=torch.long,
                        device=device,
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=fp16,
                    ):
                        object_values = model.object_logits(
                            hidden,
                            span_tensor,
                            parent_tensor,
                        ).float().cpu()
                cursor = 0
                for index, row in enumerate(batch):
                    spans = by_row[index]
                    gold = _gold(row)
                    for object_threshold in thresholds:
                        predicted = set()
                        if spans and object_values is not None:
                            predicted = decode_casrel(
                                str(row["text"]),
                                offsets[index],
                                spans,
                                object_values[cursor : cursor + len(spans)],
                                predicates,
                                object_threshold,
                            )
                        count = counts[(subject_threshold, object_threshold)]
                        count["tp"] += len(predicted & gold)
                        count["fp"] += len(predicted - gold)
                        count["fn"] += len(gold - predicted)
                    cursor += len(spans)
    scored = []
    for (subject_threshold, object_threshold), count in counts.items():
        tp, fp, fn = count["tp"], count["fp"], count["fn"]
        metric = {
            **count,
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        }
        scored.append((metric, subject_threshold, object_threshold))
    metric, subject_threshold, object_threshold = max(
        scored,
        key=lambda item: (
            item[0]["f1"],
            item[0]["precision"],
            item[1] + item[2],
        ),
    )
    return {
        "metric": metric,
        "subject_threshold": subject_threshold,
        "object_threshold": object_threshold,
        "grid_size": len(counts),
        "subject_density": subject_density,
    }


def _batch(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    predicate2id: dict[str, int],
    device: Any,
) -> tuple[dict[str, Any], Any, Any, Any, Any, int]:
    import torch

    from dki.nlp.cmeie_casrel import build_casrel_labels

    encoded = tokenizer(
        [row["text"] for row in rows],
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=320,
    )
    offsets = [
        [tuple(pair) for pair in mapping.tolist()]
        for mapping in encoded.pop("offset_mapping")
    ]
    labels = [
        build_casrel_labels(row, offsets[index], predicate2id)
        for index, row in enumerate(rows)
    ]
    sequence = int(encoded["input_ids"].shape[1])
    subject_labels = torch.stack([item.subject for item in labels]).to(device)
    spans = []
    parents = []
    object_labels = []
    for index, item in enumerate(labels):
        for span, target in sorted(item.objects.items()):
            spans.append(span)
            parents.append(index)
            object_labels.append(target)
    if not spans:
        raise RuntimeError("training batch has no aligned subject-object pairs")
    inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    span_tensor = torch.tensor(spans, dtype=torch.long, device=device)
    parent_tensor = torch.tensor(parents, dtype=torch.long, device=device)
    object_tensor = torch.stack(object_labels).to(device)
    if object_tensor.shape[1] != sequence:
        raise RuntimeError("CasRel object label sequence mismatch")
    object_mask = inputs["attention_mask"].index_select(0, parent_tensor)
    return (
        inputs,
        subject_labels,
        span_tensor,
        parent_tensor,
        object_tensor,
        object_mask,
        sum(item.dropped_spo for item in labels),
    )


def _train_epoch(
    model: Any,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    predicate2id: dict[str, int],
    device: Any,
    optimizer: Any,
    scaler: Any,
    *,
    batch_size: int,
    accumulation: int,
    fp16: bool,
    seed: int,
    maximum_optimizer_steps: int | None = None,
    scheduler: Any | None = None,
    encoder_frozen: bool = False,
) -> dict[str, Any]:
    import torch

    from dki.nlp.cmeie_casrel import casrel_loss

    rng = random.Random(seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    if maximum_optimizer_steps is not None:
        repeats = math.ceil(
            maximum_optimizer_steps
            * accumulation
            * batch_size
            / max(1, len(ordered))
        )
        ordered = (ordered * repeats)[
            : maximum_optimizer_steps * accumulation * batch_size
        ]
    model.train()
    # Keep encoder dropout disabled in both phases. Gradients still flow during
    # the joint phase, matching the stable CMeEE/GPLinker training recipe.
    model.encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    dropped = 0
    optimizer_steps = 0
    optimizer_step_attempts = 0
    skipped_optimizer_steps = 0
    nonfinite_gradient_steps = 0
    maximum_gradient_norms = {"encoder": 0.0, "subject": 0.0, "object": 0.0}
    batches = [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]
    for step, rows_batch in enumerate(batches, start=1):
        (
            inputs,
            subject_labels,
            spans,
            parents,
            object_labels,
            object_mask,
            lost,
        ) = _batch(rows_batch, tokenizer, predicate2id, device)
        dropped += lost
        group_start = ((step - 1) // accumulation) * accumulation + 1
        divisor = min(accumulation, len(batches) - group_start + 1)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=fp16,
        ):
            hidden = model.encode(**inputs)
            subject_logits = model.subject_logits(hidden)
            object_logits = model.object_logits(hidden, spans, parents)
            unscaled, _parts = casrel_loss(
                subject_logits,
                subject_labels,
                inputs["attention_mask"],
                object_logits,
                object_labels,
                object_mask,
            )
            loss = unscaled / divisor
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite CasRel loss")
        scaler.scale(loss).backward()
        losses.append(float(unscaled.detach().cpu()))
        if step % accumulation == 0 or step == len(batches):
            scaler.unscale_(optimizer)
            gradient_norms = _clip_casrel_gradients(
                model,
                torch,
                encoder_frozen=encoder_frozen,
            )
            for name, value in gradient_norms.items():
                if value is not None:
                    maximum_gradient_norms[name] = max(
                        maximum_gradient_norms[name],
                        float(value),
                    )
            nonfinite = any(
                value is not None and not math.isfinite(float(value))
                for value in gradient_norms.values()
            )
            if nonfinite:
                nonfinite_gradient_steps += 1
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_attempts += 1
            if scale_after >= scale_before:
                optimizer_steps += 1
                if scheduler is not None:
                    scheduler.step()
            else:
                skipped_optimizer_steps += 1
            if nonfinite and scale_after >= scale_before:
                raise RuntimeError(
                    "GradScaler accepted non-finite CasRel gradients"
                )
            if maximum_optimizer_steps is not None and optimizer_steps >= maximum_optimizer_steps:
                break
    return {
        "mean_loss": sum(losses) / max(1, len(losses)),
        "micro_batches": len(losses),
        "optimizer_steps": optimizer_steps,
        "optimizer_step_attempts": optimizer_step_attempts,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "nonfinite_gradient_steps": nonfinite_gradient_steps,
        "maximum_pre_clip_gradient_norms": maximum_gradient_norms,
        "dropped_spo": dropped,
        "amp_scale": float(scaler.get_scale()),
    }


def _head_parameters(model: Any) -> list[Any]:
    modules = (
        model.context_adapter,
        model.subject,
        model.subject_condition,
        model.conditioned_norm,
        model.object,
        model.subject_object_interaction,
        model.span_verifier,
    )
    return [parameter for module in modules for parameter in module.parameters()]


def _configure_encoder_trainability(
    model: Any,
    top_layers: int,
) -> tuple[list[Any], str]:
    """Unfreeze either the full encoder or only its highest transformer layers.

    CasRel's conditional object loss produced extreme full-encoder gradients in
    earlier runs.  A top-layer scope keeps the embedding and lower contextual
    stack fixed, so domain adaptation cannot destroy the entire representation.
    """

    if top_layers < 0:
        raise ValueError("top encoder layers must be non-negative")
    encoder_parameters = list(model.encoder.parameters())
    for parameter in encoder_parameters:
        parameter.requires_grad_(False)
    if top_layers == 0:
        for parameter in encoder_parameters:
            parameter.requires_grad_(True)
        return encoder_parameters, "full"

    candidates = (
        getattr(getattr(model.encoder, "encoder", None), "layer", None),
        getattr(getattr(model.encoder, "transformer", None), "layer", None),
        getattr(model.encoder, "layer", None),
    )
    layers = next(
        (candidate for candidate in candidates if candidate is not None),
        None,
    )
    if layers is None or not hasattr(layers, "__len__"):
        raise ValueError("encoder does not expose a transformer layer stack")
    if top_layers > len(layers):
        raise ValueError(
            f"requested {top_layers} top layers from an encoder with {len(layers)}"
        )
    selected: list[Any] = []
    for layer in list(layers)[-top_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
            selected.append(parameter)
    if not selected:
        raise ValueError("top-layer encoder scope selected no parameters")
    return selected, f"top_{top_layers}_transformer_layers"


def _joint_stability_passed(
    training: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    *,
    maximum_encoder_norm: float = 10_000.0,
) -> bool:
    """Require a joint overfit phase to retain quality without gradient failure."""

    if training is None or evaluation is None:
        return True
    encoder_norm = float(
        training["maximum_pre_clip_gradient_norms"]["encoder"]
    )
    return (
        evaluation["metric"]["f1"] >= 0.95
        and training["nonfinite_gradient_steps"] == 0
        and math.isfinite(encoder_norm)
        and encoder_norm <= maximum_encoder_norm
    )


def _clip_casrel_gradients(
    model: Any,
    torch: Any,
    *,
    encoder_frozen: bool,
) -> dict[str, float | None]:
    """Clip subject, object, and encoder gradients independently."""

    subject_parameters = list(model.subject.parameters())
    object_parameters = [
        parameter
        for module in (
            model.context_adapter,
            model.subject_condition,
            model.conditioned_norm,
            model.object,
            model.subject_object_interaction,
            model.span_verifier,
        )
        for parameter in module.parameters()
    ]
    subject_norm = torch.nn.utils.clip_grad_norm_(subject_parameters, 1.0)
    object_norm = torch.nn.utils.clip_grad_norm_(object_parameters, 1.0)
    encoder_norm = None
    if not encoder_frozen:
        encoder_norm = torch.nn.utils.clip_grad_norm_(
            list(model.encoder.parameters()),
            1.0,
        )
    return {
        "encoder": None if encoder_norm is None else float(encoder_norm.detach().cpu()),
        "subject": float(subject_norm.detach().cpu()),
        "object": float(object_norm.detach().cpu()),
    }


def _train_overfit_joint(
    model: Any,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    predicate2id: dict[str, int],
    predicates: list[str],
    device: Any,
    scaler: Any,
    *,
    warmup_steps: int,
    joint_steps: int,
    encoder_lr: float,
    encoder_adam_eps: float,
    encoder_top_layers: int,
    head_lr: float,
    joint_head_lr: float,
    batch_size: int,
    fp16: bool,
    seed: int,
) -> dict[str, Any]:
    """One-optimizer head-only warmup -> joint loop (GPLinker style).

    The encoder group lives in the same AdamW from the start with lr=0 and
    ``requires_grad=False``; at the warmup boundary it is unfrozen and its lr
    is raised while the head lr drops.  Keeping one optimizer avoids the
    gradient explosion seen when starting a fresh joint optimizer after the
    head-only phase.  Encoder stays in eval mode throughout (CMeEE recipe).
    """

    import torch

    from dki.nlp.cmeie_casrel import casrel_loss

    rng = random.Random(seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    total_steps = warmup_steps + joint_steps
    repeats = math.ceil(total_steps * batch_size / max(1, len(ordered)))
    ordered = (ordered * repeats)[: total_steps * batch_size]
    batches = [
        ordered[index : index + batch_size]
        for index in range(0, len(ordered), batch_size)
    ]
    encoder_parameters = list(model.encoder.parameters())
    head_parameters = _head_parameters(model)
    for parameter in encoder_parameters:
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": 0.0,
                "eps": encoder_adam_eps,
                "weight_decay": 0.0,
            },
            {
                "params": head_parameters,
                "lr": head_lr,
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        ],
    )
    encoder_group = optimizer.param_groups[0]
    head_group = optimizer.param_groups[1]
    model.train()
    model.encoder.eval()
    phase_stats = {
        phase: {
            "losses": [],
            "optimizer_steps": 0,
            "optimizer_step_attempts": 0,
            "skipped_optimizer_steps": 0,
            "nonfinite_gradient_steps": 0,
            "dropped_spo": 0,
            "maximum_gradient_norms": {"encoder": 0.0, "subject": 0.0, "object": 0.0},
        }
        for phase in ("head_only", "joint")
    }
    head_only_evaluation = None
    for step, rows_batch in enumerate(batches, start=1):
        head_only = step <= warmup_steps
        if step == warmup_steps + 1:
            head_only_evaluation = _evaluate(
                model,
                tokenizer,
                rows,
                predicates,
                device,
                batch_size=min(batch_size, len(rows)),
                fp16=fp16,
            )
            trainable_encoder, encoder_scope = _configure_encoder_trainability(
                model,
                encoder_top_layers,
            )
            encoder_group["lr"] = encoder_lr
            head_group["lr"] = joint_head_lr
        (
            inputs,
            subject_labels,
            spans,
            parents,
            object_labels,
            object_mask,
            lost,
        ) = _batch(rows_batch, tokenizer, predicate2id, device)
        stats = phase_stats["head_only" if head_only else "joint"]
        stats["dropped_spo"] += lost
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=fp16,
        ):
            hidden = model.encode(**inputs)
            subject_logits = model.subject_logits(hidden)
            object_logits = model.object_logits(hidden, spans, parents)
            unscaled, _parts = casrel_loss(
                subject_logits,
                subject_labels,
                inputs["attention_mask"],
                object_logits,
                object_labels,
                object_mask,
            )
        if not torch.isfinite(unscaled):
            raise RuntimeError("non-finite CasRel loss")
        scaler.scale(unscaled).backward()
        stats["losses"].append(float(unscaled.detach().cpu()))
        scaler.unscale_(optimizer)
        gradient_norms = _clip_casrel_gradients(
            model,
            torch,
            encoder_frozen=head_only,
        )
        for name, value in gradient_norms.items():
            if value is not None:
                stats["maximum_gradient_norms"][name] = max(
                    stats["maximum_gradient_norms"][name],
                    float(value),
                )
        nonfinite = any(
            value is not None and not math.isfinite(float(value))
            for value in gradient_norms.values()
        )
        if nonfinite:
            stats["nonfinite_gradient_steps"] += 1
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        optimizer.zero_grad(set_to_none=True)
        stats["optimizer_step_attempts"] += 1
        if scale_after >= scale_before:
            stats["optimizer_steps"] += 1
        else:
            stats["skipped_optimizer_steps"] += 1
        if nonfinite and scale_after >= scale_before:
            raise RuntimeError(
                "GradScaler accepted non-finite CasRel gradients"
            )
    joint_evaluation = _evaluate(
        model,
        tokenizer,
        rows,
        predicates,
        device,
        batch_size=min(batch_size, len(rows)),
        fp16=fp16,
    )
    rendered = {}
    for phase in ("head_only", "joint"):
        stats = phase_stats[phase]
        rendered[phase] = {
            "mean_loss": sum(stats["losses"]) / max(1, len(stats["losses"])),
            "micro_batches": len(stats["losses"]),
            "optimizer_steps": stats["optimizer_steps"],
            "optimizer_step_attempts": stats["optimizer_step_attempts"],
            "skipped_optimizer_steps": stats["skipped_optimizer_steps"],
            "nonfinite_gradient_steps": stats["nonfinite_gradient_steps"],
            "maximum_pre_clip_gradient_norms": stats["maximum_gradient_norms"],
            "dropped_spo": stats["dropped_spo"],
            "amp_scale": float(scaler.get_scale()),
        }
    return {
        "head_only": rendered["head_only"],
        "joint": rendered["joint"],
        "head_only_evaluation": head_only_evaluation,
        "joint_evaluation": joint_evaluation,
        "joint_encoder_scope": (
            "none" if joint_steps == 0 else encoder_scope
        ),
        "joint_trainable_encoder_parameters": (
            0
            if joint_steps == 0
            else sum(parameter.numel() for parameter in trainable_encoder)
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("overfit", "train"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overfit-receipt", type=Path)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--head-only-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    parser.add_argument("--overfit-rows", type=int, default=8)
    parser.add_argument("--overfit-steps", type=int, default=1200)
    parser.add_argument("--overfit-joint-steps", type=int, default=100)
    parser.add_argument("--overfit-joint-encoder-lr", type=float, default=1e-6)
    parser.add_argument("--overfit-joint-top-encoder-layers", type=int, default=2)
    parser.add_argument("--joint-encoder-lr", type=float, default=1e-6)
    parser.add_argument("--joint-encoder-adam-eps", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--joint-head-lr", type=float, default=5e-5)
    parser.add_argument("--joint-top-encoder-layers", type=int, default=2)
    parser.add_argument(
        "--diagnostic-full-unfreeze",
        action="store_true",
        help="permit top_layers=0 only for an explicitly diagnostic overfit run",
    )
    parser.add_argument("--fp16", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.run_root = args.run_root.resolve()
    args.fit_data = args.fit_data.resolve()
    args.split_manifest = args.split_manifest.resolve()
    args.base_model = args.base_model.resolve()
    args.output = args.output.resolve()
    if args.epochs < 1:
        raise SystemExit("CasRel epochs must be positive")
    if args.head_only_epochs < 1:
        raise SystemExit("CasRel head-only epochs must be positive")
    if args.joint_top_encoder_layers < 0 or args.overfit_joint_top_encoder_layers < 0:
        raise SystemExit("CasRel top encoder layers must be non-negative")
    if args.joint_encoder_lr <= 0 or args.joint_head_lr <= 0:
        raise SystemExit("CasRel joint learning rates must be positive")
    if args.joint_encoder_adam_eps <= 0:
        raise SystemExit("CasRel encoder Adam epsilon must be positive")
    if args.head_lr <= 0:
        raise SystemExit("CasRel head learning rate must be positive")
    if args.mode == "train":
        if args.epochs != 6 or args.head_only_epochs != 4:
            raise SystemExit("formal CasRel requires 6 epochs with 4 head-only epochs")
        if args.joint_top_encoder_layers != 2:
            raise SystemExit("formal CasRel requires exactly top-2 encoder layers")
        if not math.isclose(args.joint_encoder_lr, 1e-6, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit("formal CasRel requires encoder LR=1e-6")
        if not math.isclose(args.joint_encoder_adam_eps, 1e-4, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit("formal CasRel requires encoder Adam epsilon=1e-4")
        if not math.isclose(args.head_lr, 5e-5, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit("formal CasRel requires head LR=5e-5")
        if not math.isclose(args.joint_head_lr, 5e-5, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit("formal CasRel requires joint head LR=5e-5")
    if (
        args.mode == "overfit"
        and args.overfit_joint_steps > 0
        and args.overfit_joint_top_encoder_layers == 0
        and not args.diagnostic_full_unfreeze
    ):
        raise SystemExit(
            "full encoder unfreeze is diagnostic-only; pass --diagnostic-full-unfreeze"
        )
    if args.output.exists():
        raise SystemExit(f"CasRel output exists: {args.output}")
    if not args.fit_data.is_file() or not args.split_manifest.is_file():
        raise SystemExit("CasRel training data binding is missing")
    if not args.base_model.is_dir():
        raise SystemExit("CasRel base model is missing")
    fit = [row for row in _jsonl(args.fit_data) if row.get("spo_list")]
    predicates = sorted(
        {
            str(spo.get("predicate", ""))
            for row in fit
            for spo in row.get("spo_list", [])
            if spo.get("predicate")
        }
    )
    if len(predicates) != 44:
        raise SystemExit(f"CasRel expected 44 predicates, received {len(predicates)}")
    predicate2id = {value: index for index, value in enumerate(predicates)}
    calibration: list[dict[str, Any]] | None = None
    if args.mode == "train":
        if args.calibration_data is None or not args.calibration_data.resolve().is_file():
            raise SystemExit("formal CasRel training requires calibration")
        if args.overfit_receipt is None or not args.overfit_receipt.resolve().is_file():
            raise SystemExit("formal CasRel training requires overfit receipt")
        overfit = json.loads(args.overfit_receipt.resolve().read_text(encoding="utf-8"))
        if overfit.get("passed") is not True or overfit.get("calibration_accessed") is not False:
            raise SystemExit("CasRel overfit gate did not pass")
        calibration = _jsonl(args.calibration_data.resolve())
    elif args.calibration_data is not None:
        raise SystemExit("overfit mode forbids calibration input")

    import torch
    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_casrel import CMeIECasRel

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("CasRel formal execution requires Ascend NPU")
    device = torch.device("npu:0")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    encoder = AutoModel.from_pretrained(args.base_model)
    model = CMeIECasRel(
        encoder,
        int(encoder.config.hidden_size),
        len(predicates),
    ).to(device)
    if not hasattr(torch.npu, "amp") or not hasattr(torch.npu.amp, "GradScaler"):
        raise SystemExit("CasRel NPU FP16 requires torch.npu.amp.GradScaler")
    scaler = torch.npu.amp.GradScaler(init_scale=1024.0, enabled=args.fp16)
    args.output.mkdir(parents=True, exist_ok=False)
    history = []
    head_parameters = _head_parameters(model)

    if args.mode == "overfit":
        rows = fit[: args.overfit_rows]
        warmup_steps = max(1, args.overfit_steps)
        joint_training = None
        joint_evaluation = None
        selected_phase = "head_only_warmup"
        if args.overfit_joint_steps > 0:
            joint_run = _train_overfit_joint(
                model,
                rows,
                tokenizer,
                predicate2id,
                predicates,
                device,
                scaler,
                warmup_steps=warmup_steps,
                joint_steps=args.overfit_joint_steps,
                encoder_lr=args.overfit_joint_encoder_lr,
                encoder_adam_eps=args.joint_encoder_adam_eps,
                encoder_top_layers=args.overfit_joint_top_encoder_layers,
                head_lr=args.head_lr,
                joint_head_lr=args.joint_head_lr,
                batch_size=min(args.batch_size, len(rows)),
                fp16=args.fp16,
                seed=args.seed,
            )
            warmup_training = joint_run["head_only"]
            joint_training = joint_run["joint"]
            warmup_evaluation = joint_run["head_only_evaluation"]
            joint_evaluation = joint_run["joint_evaluation"]
            # Deterministic best-phase selection, mirroring the shared
            # overfit protocol: prefer higher F1, then precision.
            joint_rank = (
                joint_evaluation["metric"]["f1"],
                joint_evaluation["metric"]["precision"],
            )
            warmup_rank = (
                warmup_evaluation["metric"]["f1"],
                warmup_evaluation["metric"]["precision"],
            )
            if joint_rank > warmup_rank:
                selected_phase = "joint"
                evaluation = joint_evaluation
            else:
                selected_phase = "head_only_warmup"
                evaluation = warmup_evaluation
        else:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False
            warmup_optimizer = torch.optim.AdamW(
                head_parameters,
                lr=args.head_lr,
                weight_decay=0.01,
            )
            warmup_training = _train_epoch(
                model,
                rows,
                tokenizer,
                predicate2id,
                device,
                warmup_optimizer,
                scaler,
                batch_size=min(args.batch_size, len(rows)),
                accumulation=1,
                fp16=args.fp16,
                seed=args.seed,
                maximum_optimizer_steps=warmup_steps,
                encoder_frozen=True,
            )
            warmup_evaluation = _evaluate(
                model,
                tokenizer,
                rows,
                predicates,
                device,
                batch_size=min(args.batch_size, len(rows)),
                fp16=args.fp16,
            )
            evaluation = warmup_evaluation
        training = {
            "head_only_warmup": warmup_training,
            "joint": joint_training,
            "optimizer_steps": warmup_training["optimizer_steps"]
            + (0 if joint_training is None else joint_training["optimizer_steps"]),
            "skipped_optimizer_steps": warmup_training["skipped_optimizer_steps"]
            + (
                0
                if joint_training is None
                else joint_training["skipped_optimizer_steps"]
            ),
            "dropped_spo": warmup_training["dropped_spo"]
            + (0 if joint_training is None else joint_training["dropped_spo"]),
            "head_only_evaluation": warmup_evaluation,
            "joint_evaluation": joint_evaluation,
            "selected_phase": selected_phase,
            "joint_encoder_scope": (
                "none"
                if joint_training is None
                else joint_run["joint_encoder_scope"]
            ),
            "joint_trainable_encoder_parameters": (
                0
                if joint_training is None
                else joint_run["joint_trainable_encoder_parameters"]
            ),
            "joint_encoder_adam_eps": args.joint_encoder_adam_eps,
        }
        checkpoint = args.output / "model.pt"
        checkpoint_sha = _save_torch_once(
            checkpoint,
            {"model": _cpu_state_dict(model), "predicates": predicates},
            torch,
        )
        receipt = {
            "schema_version": 1,
            "kind": "cmeie_casrel_overfit_preflight",
            "mode": "overfit",
            "seed": args.seed,
            "source_tree_sha256": args.source_tree_sha256,
            "fit_sha256": _sha(args.fit_data),
            "split_manifest_sha256": _sha(args.split_manifest),
            "base_model": str(args.base_model),
            "rows": len(rows),
            "steps": args.overfit_steps,
            "optimizer_recipe": {
                "head_lr": args.head_lr,
                "joint_head_lr": args.joint_head_lr,
                "encoder_lr": args.overfit_joint_encoder_lr,
                "encoder_adam_eps": args.joint_encoder_adam_eps,
                "encoder_scope": (
                    "none"
                    if joint_training is None
                    else joint_run["joint_encoder_scope"]
                ),
                "full_unfreeze_diagnostic": args.diagnostic_full_unfreeze,
                "hard_negative_ratio": 8,
                "encoder_eval_mode": True,
            },
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "training": training,
            "evaluation": evaluation,
            "metric": evaluation["metric"],
            "thresholds": {
                "subject": evaluation["subject_threshold"],
                "object": evaluation["object_threshold"],
            },
            "checkpoint_sha256": checkpoint_sha,
            "calibration_accessed": False,
            "outer_accessed": False,
            "official_dev_accessed": False,
            "joint_stability_passed": _joint_stability_passed(
                joint_training,
                joint_evaluation,
            ),
            "passed": (
                warmup_evaluation["metric"]["f1"] >= 0.98
                and warmup_training["dropped_spo"] == 0
                and warmup_training["nonfinite_gradient_steps"] == 0
                and warmup_training["skipped_optimizer_steps"] == 0
                and (
                    joint_training is None
                    or (
                        joint_training["dropped_spo"] == 0
                        and joint_training["nonfinite_gradient_steps"] == 0
                        and joint_training["skipped_optimizer_steps"] == 0
                    )
                )
                and _joint_stability_passed(
                    joint_training,
                    joint_evaluation,
                )
            ),
            "script_sha256": _sha(Path(__file__)),
        }
        _write_json_once(args.output / "overfit_receipt.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0 if receipt["passed"] else 2

    assert calibration is not None
    best_rank = None
    stale = 0
    joint_steps_per_epoch = math.ceil(
        math.ceil(len(fit) / args.batch_size) / args.gradient_accumulation
    )
    head_only_epochs = min(args.head_only_epochs, args.epochs)
    encoder_parameters = list(model.encoder.parameters())
    for parameter in encoder_parameters:
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": 0.0,
                "eps": args.joint_encoder_adam_eps,
                "weight_decay": 0.0,
            },
            {
                "params": head_parameters,
                "lr": args.head_lr,
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        ],
    )
    encoder_group = optimizer.param_groups[0]
    head_group = optimizer.param_groups[1]
    joint_encoder_scope = "not_run"
    joint_trainable_encoder_parameters = 0
    for epoch in range(1, args.epochs + 1):
        if epoch <= head_only_epochs:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False
            scheduler = None
            model.encoder.eval()
            phase = "head_only"
        else:
            if epoch == head_only_epochs + 1:
                trainable_encoder, joint_encoder_scope = (
                    _configure_encoder_trainability(
                        model,
                        args.joint_top_encoder_layers,
                    )
                )
                joint_trainable_encoder_parameters = sum(
                    parameter.numel() for parameter in trainable_encoder
                )
                encoder_group["lr"] = args.joint_encoder_lr
                head_group["lr"] = args.joint_head_lr
                total_steps = joint_steps_per_epoch * (
                    args.epochs - head_only_epochs
                )
                warmup = max(1, round(total_steps * 0.10))
                def schedule(step: int) -> float:
                    if step < warmup:
                        return step / warmup
                    return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    [schedule, schedule],
                )
            phase = "joint"
        training = _train_epoch(
            model,
            fit,
            tokenizer,
            predicate2id,
            device,
            optimizer,
            scaler,
            batch_size=args.batch_size,
            accumulation=args.gradient_accumulation,
            fp16=args.fp16,
            seed=args.seed * 100 + epoch,
            scheduler=scheduler,
            encoder_frozen=epoch <= head_only_epochs,
        )
        evaluation = _evaluate(
            model,
            tokenizer,
            calibration,
            predicates,
            device,
            batch_size=args.calibration_batch_size,
            fp16=args.fp16,
        )
        epoch_dir = args.output / f"epoch-{epoch:02d}"
        epoch_dir.mkdir(exist_ok=False)
        checkpoint = epoch_dir / "model.pt"
        checkpoint_sha = _save_torch_once(
            checkpoint,
            {
                "model": _cpu_state_dict(model),
                "predicates": predicates,
                "epoch": epoch,
            },
            torch,
        )
        record = {
            "epoch": epoch,
            "phase": phase,
            "training": training,
            "calibration": evaluation,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
        }
        _write_json_once(epoch_dir / "metrics.json", record)
        history.append(record)
        rank = (
            evaluation["metric"]["f1"],
            evaluation["metric"]["precision"],
            -epoch,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            stale = 0
        elif epoch >= head_only_epochs + 2:
            stale += 1
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        if epoch >= head_only_epochs + 3 and stale >= 2:
            break
    best = max(
        history,
        key=lambda item: (
            item["calibration"]["metric"]["f1"],
            item["calibration"]["metric"]["precision"],
            -item["epoch"],
        ),
    )
    selection = {
        "schema_version": 1,
        "kind": "cmeie_casrel_calibration_selection",
        "seed": args.seed,
        "source_tree_sha256": args.source_tree_sha256,
        "fit_sha256": _sha(args.fit_data),
        "calibration_sha256": _sha(args.calibration_data.resolve()),
        "split_manifest_sha256": _sha(args.split_manifest),
        "base_model": str(args.base_model),
        "head_only_epochs": head_only_epochs,
        "joint_encoder_scope": joint_encoder_scope,
        "joint_trainable_encoder_parameters": joint_trainable_encoder_parameters,
        "joint_encoder_lr": args.joint_encoder_lr,
        "joint_encoder_adam_eps": args.joint_encoder_adam_eps,
        "joint_head_lr": args.joint_head_lr,
        "head_lr": args.head_lr,
        "hard_negative_ratio": 8,
        "encoder_eval_mode": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "predicates": predicates,
        "selected_epoch": best["epoch"],
        "checkpoint": best["checkpoint"],
        "checkpoint_sha256": best["checkpoint_sha256"],
        "calibration_metric": best["calibration"]["metric"],
        "subject_threshold": best["calibration"]["subject_threshold"],
        "object_threshold": best["calibration"]["object_threshold"],
        "history": history,
        "dataset_role": "fit_training_calibration_selection",
        "selection_used": True,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "script_sha256": _sha(Path(__file__)),
    }
    _write_json_once(args.output / "selection.json", selection)
    print(json.dumps(selection, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
