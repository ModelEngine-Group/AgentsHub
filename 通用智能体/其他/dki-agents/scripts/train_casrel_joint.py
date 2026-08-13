"""Joint CasRel training: boundary losses + verifier BCE with hard negatives.

Goal: push CMeIE full text-to-SPO calibration above 0.45 / toward 0.5.

Based on blocker_attempt_ledger_20260802.md:
- casrel_loss only trains subject/object boundary heads; span_verifier exists but
  is NEVER trained (baseline checkpoint carries no verifier weights).
- FP=4867 >> TP=2154 is the precision hole; verifier is a designed precision stage.

This script reuses the formal CasRel recipe (4 head-only + 2 top-2 joint epochs,
encoder Adam eps=1e-4, head LR 5e-5, batch 8, acc 4, seed 42) and additionally:
- builds verifier candidates per micro-batch:
    positives = gold-aligned (subject_span, predicate_id, object_span)
    negatives = decoded boundary candidates not in gold (predicted hard negatives)
               + predicate-confusion negatives
- adds verifier_bce_loss (weighted) to the boundary loss, so verifier trains
  jointly and boundary heads get precision feedback.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from dki.nlp.cmeie_casrel import (
    build_casrel_labels,
    pair_boundary_logits,
    predicate_object_candidates,
    verifier_bce_loss,
)
from train_cmeie_casrel import (
    _batch,
    _clip_casrel_gradients,
    _configure_encoder_trainability,
    _evaluate,
    _gold,
    _head_parameters,
    _jsonl,
    _valid_subjects,
)


def _build_verifier_candidates(
    model,
    hidden,
    inputs,
    rows_batch,
    tokenizer,
    predicate2id,
    offsets,
    object_logits,
    device,
    *,
    object_threshold: float = 0.0,
    max_negatives_per_row: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (subject_spans, object_spans, predicate_ids, parents, labels)."""
    from dki.nlp.cmeie_casrel import select_mention_pairs, char_span_to_tokens

    cs, co, cp, cr, cl = [], [], [], [], []
    for i, row in enumerate(rows_batch):
        text = str(row["text"])
        labels = build_casrel_labels(row, offsets[i], predicate2id)
        gold_by_subject = labels.object_spans  # (ss,se) -> ((pid, os, oe), ...)
        gold_set = set()
        for (ss, se), values in gold_by_subject.items():
            for (pid, os_, oe) in values:
                gold_set.add((ss, se, pid, os_, oe))
        # positives from gold
        for (ss, se), values in gold_by_subject.items():
            for (pid, os_, oe) in values:
                cs.append((ss, se)); co.append((os_, oe)); cp.append(pid); cr.append(i); cl.append(1.0)
        # negatives: decoded candidates not in gold
        neg_count = 0
        # recompute subject spans from the model's subject head for this row
        with torch.no_grad():
            row_subject_logits = model.subject_logits(hidden[i : i + 1]).float().cpu()
        spans = _valid_subjects(
            pair_boundary_logits(
                row_subject_logits[0, :, 0],
                row_subject_logits[0, :, 1],
                0.5,
            ),
            offsets[i],
        )
        if spans:
            with torch.no_grad():
                obj = model.object_logits(
                    hidden[i : i + 1],
                    torch.tensor([(s[0], s[1]) for s in spans], dtype=torch.long, device=device),
                    torch.zeros(len(spans), dtype=torch.long, device=device),
                ).float().cpu()
            seen = set()
            for si in range(len(spans)):
                try:
                    cands = predicate_object_candidates(
                        obj[si : si + 1], offsets[i], object_threshold,
                        maximum_per_subject_predicate=64, maximum_per_row=16384,
                    )
                except Exception:
                    continue
                for c in cands:
                    pid, os_, oe = c[1], c[2], c[3]
                    key = (spans[si][0], spans[si][1], pid, os_, oe)
                    if key in gold_set or key in seen:
                        continue
                    seen.add(key)
                    cs.append((spans[si][0], spans[si][1]))
                    co.append((os_, oe)); cp.append(pid); cr.append(i); cl.append(0.0)
                    neg_count += 1
                    if neg_count >= max_negatives_per_row:
                        break
                if neg_count >= max_negatives_per_row:
                    break
    if not cs:
        return None
    return (
        torch.tensor(cs, dtype=torch.long, device=device),
        torch.tensor(co, dtype=torch.long, device=device),
        torch.tensor(cp, dtype=torch.long, device=device),
        torch.tensor(cr, dtype=torch.long, device=device),
        torch.tensor(cl, dtype=torch.float32, device=device),
    )


def _subject_spans_from_logits(model, hidden_row, threshold, device):
    return pair_boundary_logits(
        hidden_row[0, :, 0].float().cpu(),
        hidden_row[0, :, 1].float().cpu(),
        threshold,
    )


def _train_epoch_joint(
    model,
    rows,
    tokenizer,
    predicate2id,
    device,
    optimizer,
    scaler,
    *,
    batch_size: int,
    accumulation: int,
    fp16: bool,
    seed: int,
    verifier_weight: float,
    encoder_frozen: bool,
    maximum_optimizer_steps: int | None = None,
    scheduler=None,
):
    rng = random.Random(seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    model.train()
    model.encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    losses, verifier_losses = [], []
    dropped = 0
    optimizer_steps = optimizer_step_attempts = skipped_optimizer_steps = nonfinite_gradient_steps = 0
    maximum_gradient_norms = {"encoder": 0.0, "subject": 0.0, "object": 0.0}
    batches = [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]
    for step, rows_batch in enumerate(batches, start=1):
        (inputs, subject_labels, spans, parents, object_labels, object_mask, lost) = _batch(
            rows_batch, tokenizer, predicate2id, device
        )
        dropped += lost
        group_start = ((step - 1) // accumulation) * accumulation + 1
        divisor = min(accumulation, len(batches) - group_start + 1)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            hidden = model.encode(**inputs)
            subject_logits = model.subject_logits(hidden)
            object_logits = model.object_logits(hidden, spans, parents)
            boundary_loss, _parts = _boundary_loss(
                subject_logits, subject_labels, inputs["attention_mask"],
                object_logits, object_labels, object_mask,
            )
            total = boundary_loss
            vloss = torch.tensor(0.0, device=device)
            if verifier_weight > 0:
                cand = _build_verifier_candidates(
                    model, hidden, inputs, rows_batch, tokenizer, predicate2id,
                    _offsets_for(rows_batch, tokenizer, device),
                    object_logits, device,
                )
                if cand is not None:
                    cs, co, cp, cr, cl = cand
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                        vlogits = model.score_object_spans(hidden, cs, co, cp, cr)
                    try:
                        vloss = verifier_bce_loss(vlogits.float(), cl.float())
                    except ValueError:
                        vloss = torch.tensor(0.0, device=device)
                    total = boundary_loss + verifier_weight * vloss
            loss = total / divisor
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite joint CasRel loss")
        scaler.scale(loss).backward()
        losses.append(float(boundary_loss.detach().cpu()))
        verifier_losses.append(float(vloss.detach().cpu()))
        if step % accumulation == 0 or step == len(batches):
            scaler.unscale_(optimizer)
            gradient_norms = _clip_casrel_gradients(model, torch, encoder_frozen=encoder_frozen)
            for name, value in gradient_norms.items():
                if value is not None:
                    maximum_gradient_norms[name] = max(maximum_gradient_norms[name], float(value))
            nonfinite = any(
                value is not None and not math.isfinite(float(value)) for value in gradient_norms.values()
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
            if maximum_optimizer_steps is not None and optimizer_steps >= maximum_optimizer_steps:
                break
    return {
        "mean_loss": sum(losses) / max(1, len(losses)),
        "mean_verifier_loss": sum(verifier_losses) / max(1, len(verifier_losses)),
        "micro_batches": len(losses),
        "optimizer_steps": optimizer_steps,
        "optimizer_step_attempts": optimizer_step_attempts,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "nonfinite_gradient_steps": nonfinite_gradient_steps,
        "maximum_pre_clip_gradient_norms": maximum_gradient_norms,
        "dropped_spo": dropped,
        "amp_scale": float(scaler.get_scale()),
    }


def _boundary_loss(subject_logits, subject_labels, attention_mask, object_logits, object_labels, object_mask):
    from dki.nlp.cmeie_casrel import casrel_loss
    return casrel_loss(
        subject_logits, subject_labels, attention_mask,
        object_logits, object_labels, object_mask,
    )


def _offsets_for(rows_batch, tokenizer, device):
    encoded = tokenizer(
        [r["text"] for r in rows_batch], return_offsets_mapping=True, return_tensors="pt",
        padding=True, truncation=True, max_length=320,
    )
    return [[tuple(p) for p in m.tolist()] for m in encoded.pop("offset_mapping")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("overfit", "train"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--head-only-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    parser.add_argument("--overfit-rows", type=int, default=8)
    parser.add_argument("--overfit-steps", type=int, default=600)
    parser.add_argument("--overfit-joint-steps", type=int, default=60)
    parser.add_argument("--max-fit", type=int, default=0, help="cap fit rows (0=all)")
    parser.add_argument("--joint-encoder-lr", type=float, default=1e-6)
    parser.add_argument("--joint-encoder-adam-eps", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--joint-head-lr", type=float, default=5e-5)
    parser.add_argument("--joint-top-encoder-layers", type=int, default=2)
    parser.add_argument("--verifier-weight", type=float, default=0.5)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    fit = [r for r in _jsonl(args.fit_data) if r.get("spo_list")]
    predicates = sorted({str(s.get("predicate", "")) for r in fit for s in r.get("spo_list", []) if s.get("predicate")})
    assert len(predicates) == 44, f"expected 44 predicates, got {len(predicates)}"
    predicate2id = {p: i for i, p in enumerate(predicates)}

    from transformers import AutoModel, AutoTokenizer
    from dki.nlp.cmeie_casrel import CMeIECasRel

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise SystemExit("joint CasRel requires Ascend NPU")
    device = torch.device("npu:0")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    encoder = AutoModel.from_pretrained(args.base_model)
    model = CMeIECasRel(encoder, int(encoder.config.hidden_size), len(predicates)).to(device)
    scaler = torch.npu.amp.GradScaler(init_scale=1024.0, enabled=args.fp16)
    args.output.mkdir(parents=True, exist_ok=True)
    head_parameters = _head_parameters(model)

    calibration = None
    if args.mode == "train":
        calibration = _jsonl(args.calibration_data.resolve())

    rows = fit[: args.overfit_rows] if args.mode == "overfit" else fit
    if args.mode == "train" and args.max_fit > 0:
        rows = fit[: args.max_fit]
        print(f"[info] capped fit rows: {len(rows)} (of {len(fit)})", flush=True)
    history = []
    if args.mode == "overfit":
        warmup_optimizer = torch.optim.AdamW(head_parameters, lr=args.head_lr, weight_decay=0.01)
        warmup = _train_epoch_joint(
            model, rows, tokenizer, predicate2id, device, warmup_optimizer, scaler,
            batch_size=min(args.batch_size, len(rows)), accumulation=1, fp16=args.fp16,
            seed=args.seed, verifier_weight=args.verifier_weight, encoder_frozen=True,
            maximum_optimizer_steps=args.overfit_steps,
        )
        warmup_eval = _evaluate(model, tokenizer, rows, predicates, device,
                                batch_size=min(args.batch_size, len(rows)), fp16=args.fp16)
        print(json.dumps({"mode": "overfit", "warmup": warmup, "warmup_eval": warmup_eval}, ensure_ascii=False, indent=2), flush=True)
        torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "predicates": predicates, "joint": False}, args.output / "model.pt")
        print(f"[done] overfit joint checkpoint -> {args.output / 'model.pt'}", flush=True)
        return 0

    # formal: 4 head-only + 2 top-2 joint, verifier loss active throughout
    for epoch in range(1, args.epochs + 1):
        encoder_frozen = epoch <= args.head_only_epochs
        if encoder_frozen:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False
            head_opt = torch.optim.AdamW(head_parameters, lr=args.head_lr, weight_decay=0.01)
        else:
            top_params, scope = _configure_encoder_trainability(model, args.joint_top_encoder_layers)
            head_opt = torch.optim.AdamW(
                head_parameters + top_params,
                lr=args.joint_head_lr,
                weight_decay=0.0,
            )
            for group in head_opt.param_groups[1:]:
                group["lr"] = args.joint_encoder_lr
                group["eps"] = args.joint_encoder_adam_eps
        training = _train_epoch_joint(
            model, fit, tokenizer, predicate2id, device, head_opt, scaler,
            batch_size=args.batch_size, accumulation=args.gradient_accumulation,
            fp16=args.fp16, seed=args.seed * 100 + epoch,
            verifier_weight=args.verifier_weight, encoder_frozen=encoder_frozen,
        )
        evaluation = _evaluate(model, tokenizer, calibration, predicates, device,
                               batch_size=args.calibration_batch_size, fp16=args.fp16)
        epoch_dir = args.output / f"epoch-{epoch:02d}"
        epoch_dir.mkdir(exist_ok=False)
        torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "predicates": predicates, "epoch": epoch}, epoch_dir / "model.pt")
        record = {"epoch": epoch, "phase": "head_only" if encoder_frozen else "joint",
                  "training": training, "calibration": evaluation}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    selected = max(history, key=lambda item: (item["calibration"]["metric"]["f1"], item["calibration"]["metric"]["precision"]))
    print(json.dumps({"mode": "train", "selected_epoch": selected["epoch"], "selected": selected}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
