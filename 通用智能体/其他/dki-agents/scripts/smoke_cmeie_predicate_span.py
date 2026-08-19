"""Fit-only NPU overfit gate for the predicate-conditioned span linker."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dki.nlp.cmeie_predicate_span import (
    CMeIEPredicateSpanLinker,
    LOSS_MODE_HARD_NEGATIVE_BCE,
    PREDICATE_SPAN_LOSS_MODES,
    build_predicate_span_labels,
    decode_predicate_spans,
    predicate_span_loss,
)
from dki.nlp.cmeie_spo import gold_spo_set, labels_from_fit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _select(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row.get("spo_list") and len(str(row.get("text", ""))) <= 180]
    random.Random(seed).shuffle(candidates)
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(candidates, key=lambda item: (len(str(item.get("text", ""))), str(item.get("text", "")))):
        row_predicates = {str(spo.get("predicate", "")) for spo in row.get("spo_list", [])}
        if row_predicates - seen:
            chosen.append(row)
            seen.update(row_predicates)
            if len(chosen) >= size:
                return chosen
    for row in candidates:
        if row not in chosen:
            chosen.append(row)
            if len(chosen) >= size:
                return chosen
    if len(chosen) < size:
        raise ValueError("fit shard has too few usable rows")
    return chosen


def _publish_json_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--source-tree-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--negatives-per-head", type=int, default=32)
    parser.add_argument("--inactive-weight", type=float, default=0.1)
    parser.add_argument(
        "--loss-mode",
        choices=PREDICATE_SPAN_LOSS_MODES,
        default=LOSS_MODE_HARD_NEGATIVE_BCE,
    )
    parser.add_argument("--minimum-f1", type=float, default=0.90)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("npu:0")
    rows = _load(args.fit_data)
    selected = _select(rows, args.sample_size, args.seed)
    _, predicates = labels_from_fit(rows, entity_label_mode="role_only")
    predicate2id = {label: index for index, label in enumerate(predicates)}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)
    encoded = tokenizer(
        [str(row["text"]) for row in selected],
        max_length=args.max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = [[tuple(map(int, pair)) for pair in item] for item in encoded.pop("offset_mapping").tolist()]
    label_rows = []
    dropped = 0
    for row, mapping in zip(selected, offsets):
        labels, row_dropped = build_predicate_span_labels(row, mapping, predicate2id)
        label_rows.append(labels)
        dropped += row_dropped
    labels = {
        name: torch.stack([item[name] for item in label_rows])
        for name in ("subject", "object", "link")
    }

    encoder = AutoModel.from_pretrained(args.base_model, local_files_only=True).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    hidden_batches = []
    with torch.no_grad():
        for start in range(0, len(selected), args.batch_size):
            stop = min(len(selected), start + args.batch_size)
            model_inputs = {
                key: value[start:stop].to(device)
                for key, value in encoded.items()
            }
            with torch.autocast(device_type="npu", dtype=torch.float16, enabled=True):
                result = encoder(**model_inputs)
            hidden_batches.append(result.last_hidden_state.detach().cpu().to(torch.float16))
    hidden = torch.cat(hidden_batches, dim=0)
    del encoder
    torch.npu.empty_cache()

    model = CMeIEPredicateSpanLinker.__new__(CMeIEPredicateSpanLinker)
    torch.nn.Module.__init__(model)
    from dki.nlp.cmeie_gplinker import DirectionalGlobalPointer
    from dki.nlp.globalpointer import GlobalPointer

    hidden_size = int(hidden.shape[-1])
    model.subject = GlobalPointer(hidden_size, len(predicates), 64)
    model.object = GlobalPointer(hidden_size, len(predicates), 64)
    model.link = DirectionalGlobalPointer(hidden_size, len(predicates), 64)
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.npu.amp.GradScaler(init_scale=1024.0, enabled=True)
    snapshots = []
    skipped = 0
    order = list(range(len(selected)))
    for step in range(args.steps):
        start = (step * args.batch_size) % len(order)
        indices = [order[(start + offset) % len(order)] for offset in range(args.batch_size)]
        batch_hidden = hidden[indices].to(device)
        batch_mask = encoded["attention_mask"][indices].to(device)
        batch_labels = {name: value[indices].to(device) for name, value in labels.items()}
        optimiser.zero_grad(set_to_none=True)
        with torch.autocast(device_type="npu", dtype=torch.float16, enabled=True):
            outputs = {
                "subject": model.subject(batch_hidden, batch_mask),
                "object": model.object(batch_hidden, batch_mask),
                "link": model.link(batch_hidden, batch_mask),
            }
            loss = predicate_span_loss(
                outputs,
                batch_labels,
                loss_mode=args.loss_mode,
                negatives_per_head=args.negatives_per_head,
                inactive_weight=args.inactive_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scale_before = float(scaler.get_scale())
        scaler.step(optimiser)
        scaler.update()
        if float(scaler.get_scale()) < scale_before:
            skipped += 1
        if step in {0, 9, 49, 99, 199, 399, 799, args.steps - 1}:
            snapshot = {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()) if torch.isfinite(grad_norm) else None,
                "amp_scale": float(scaler.get_scale()),
            }
            snapshots.append(snapshot)
            print(json.dumps(snapshot, sort_keys=True), flush=True)

    thresholds = (-0.5, 0.0, 0.5, 1.0)
    candidates = list(product(thresholds, repeat=3))
    counts = {candidate: Counter(tp=0, fp=0, fn=0) for candidate in candidates}
    model.eval()
    with torch.no_grad():
        for index, row in enumerate(selected):
            batch_hidden = hidden[index : index + 1].to(device)
            batch_mask = encoded["attention_mask"][index : index + 1].to(device)
            with torch.autocast(device_type="npu", dtype=torch.float16, enabled=True):
                current = {
                    "subject": model.subject(batch_hidden, batch_mask)[0].cpu(),
                    "object": model.object(batch_hidden, batch_mask)[0].cpu(),
                    "link": model.link(batch_hidden, batch_mask)[0].cpu(),
                }
            gold = gold_spo_set(row)
            for candidate in candidates:
                predicted = decode_predicate_spans(
                    str(row["text"]), offsets[index], current, predicates, candidate
                )
                counts[candidate]["tp"] += len(predicted & gold)
                counts[candidate]["fp"] += len(predicted - gold)
                counts[candidate]["fn"] += len(gold - predicted)
    scored = []
    for candidate, count in counts.items():
        precision = count["tp"] / max(1, count["tp"] + count["fp"])
        recall = count["tp"] / max(1, count["tp"] + count["fn"])
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        scored.append((f1, precision, sum(candidate), candidate, dict(count), recall))
    f1, precision, _, best_thresholds, count, recall = max(scored)
    metric = {**count, "precision": precision, "recall": recall, "f1": f1}
    payload = {
        "schema_version": 1,
        "kind": "cmeie_predicate_span_fit_only_overfit",
        "dataset_role": "fit_only_optimization_preflight",
        "fit_only": True,
        "calibration_accessed": False,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "passed": f1 >= args.minimum_f1,
        "minimum_f1": args.minimum_f1,
        "metric": metric,
        "thresholds": list(best_thresholds),
        "threshold_grid": list(thresholds),
        "fit_data_sha256": _sha256(args.fit_data),
        "source_tree_manifest_sha256": _sha256(args.source_tree_manifest),
        "base_model_config_sha256": _sha256(args.base_model / "config.json"),
        "seed": args.seed,
        "sample_size": len(selected),
        "predicates": len(predicates),
        "covered_predicates": len({str(spo.get("predicate", "")) for row in selected for spo in row.get("spo_list", [])}),
        "max_length": args.max_length,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss_mode": args.loss_mode,
        "loss_ablation": args.loss_mode != LOSS_MODE_HARD_NEGATIVE_BCE,
        "negatives_per_head": args.negatives_per_head,
        "negatives_per_head_applied": args.loss_mode == LOSS_MODE_HARD_NEGATIVE_BCE,
        "inactive_weight": args.inactive_weight,
        "dropped_labels": dropped,
        "skipped_optimizer_steps": skipped,
        "snapshots": snapshots,
    }
    _publish_json_once(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
