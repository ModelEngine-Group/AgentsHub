"""Train one create-once native CMeIE argument NER candidate on fit only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def device_for(torch, requested: str):
    if requested == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise SystemExit("NPU candidate requested but no Ascend NPU is available")
        return torch.device("npu:0")
    return torch.device(requested)


def publish_dir_once(path: Path, writer) -> None:
    if path.exists():
        raise SystemExit(f"candidate output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=path.parent))
    try:
        writer(staging)
        os.rename(staging, path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--threshold-grid", default="0.0,0.5,1.0")
    parser.add_argument("--device", choices=["npu", "cpu"], default="npu")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"candidate output exists: {args.output}")
    if args.seed not in {42, 43, 44}:
        raise SystemExit("fallback candidate seed must be one of 42, 43, 44")
    try:
        threshold_grid = tuple(sorted({float(value) for value in args.threshold_grid.split(",")}))
    except ValueError as exc:
        raise SystemExit("threshold-grid must be comma-separated floats") from exc
    if not threshold_grid or any(value < 0 for value in threshold_grid):
        raise SystemExit("fallback NER rejects negative decode thresholds to prevent dense span explosion")
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if manifest.get("source_train_sha256") != sha(args.train):
        raise SystemExit("train hash differs from frozen split manifest")
    all_rows = rows(args.train)
    fit = [all_rows[index] for index in manifest["splits"]["fit"]["indices"]]
    calibration = [all_rows[index] for index in manifest["splits"]["calibration"]["indices"]]
    if not fit or not calibration:
        raise SystemExit("fit/calibration split is empty")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    from dki.nlp.cmeie_argument import build_labels, decode_mentions, gold_mentions, labels_from_rows, mention_metric
    from dki.nlp.globalpointer import EncoderGlobalPointer, gp_loss

    random.seed(args.seed); torch.manual_seed(args.seed)
    if hasattr(torch, "npu") and hasattr(torch.npu, "manual_seed_all"):
        torch.npu.manual_seed_all(args.seed)
    device = device_for(torch, args.device)
    types = labels_from_rows(fit)
    if len(types) != 11:
        raise SystemExit(f"expected 11 native CMeIE types from fit, found {len(types)}")
    type2id = {name: index for index, name in enumerate(types)}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    def collate(batch):
        encoded = tokenizer([row["text"] for row in batch], truncation=True, max_length=args.max_length, padding=True, return_offsets_mapping=True, return_tensors="pt")
        offsets = [[tuple(pair) for pair in value.tolist()] for value in encoded.pop("offset_mapping")]
        dense, dropped = [], 0
        for row, one_offsets in zip(batch, offsets):
            label, lost = build_labels(row, one_offsets, type2id)
            dense.append(label); dropped += lost
        return encoded, torch.stack(dense), dropped

    loader = DataLoader(fit, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    encoder = AutoModel.from_pretrained(args.base_model)
    model = EncoderGlobalPointer(encoder, int(encoder.config.hidden_size), len(types), inner_dim=64, pointer="efficient").to(device)
    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")]
    optimiser = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.encoder_lr},
        {"params": head_parameters, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    steps = max(1, (len(loader) * args.epochs + args.gradient_accumulation - 1) // args.gradient_accumulation)
    schedule = get_linear_schedule_with_warmup(optimiser, int(steps * args.warmup), steps)

    def calibration_scores():
        predictions = {threshold: [] for threshold in threshold_grid}
        gold = []
        model.eval()
        with torch.no_grad():
            for row in calibration:
                encoded = tokenizer(row["text"], truncation=True, max_length=args.max_length, return_offsets_mapping=True, return_tensors="pt")
                offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")[0].tolist()]
                inputs = {key: value.to(device) for key, value in encoded.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "npu"):
                    logits = model(**inputs)[0].float().cpu()
                for threshold in threshold_grid:
                    predictions[threshold].append(decode_mentions(row["text"], offsets, logits, types, threshold))
                gold.append(gold_mentions(row))
        return [(mention_metric(predictions[threshold], gold), threshold) for threshold in threshold_grid]

    history, best, stale, dropped_total = [], None, 0, 0
    checkpoints: dict[int, dict] = {}
    for epoch in range(1, args.epochs + 1):
        model.train(); optimiser.zero_grad(set_to_none=True); loss_sum = 0.0
        for step, (inputs, labels, dropped) in enumerate(loader, start=1):
            dropped_total += dropped
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "npu"):
                loss = gp_loss(model(**inputs), labels) / args.gradient_accumulation
            loss.backward(); loss_sum += float(loss.detach()) * args.gradient_accumulation
            if step % args.gradient_accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimiser.step(); schedule.step(); optimiser.zero_grad(set_to_none=True)
        scores = calibration_scores()
        metric, threshold = max(scores, key=lambda item: (item[0]["f1"], item[0]["precision"], item[1]))
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        checkpoints[epoch] = state
        record = {"epoch": epoch, "mean_train_loss": loss_sum / max(1, len(loader)), "calibration": {"threshold": threshold, **metric}}
        history.append(record); print(json.dumps(record, ensure_ascii=False), flush=True)
        if best is None or (metric["f1"], metric["precision"], threshold) > (best[1]["f1"], best[1]["precision"], best[2]):
            best = (epoch, metric, threshold); stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    assert best is not None
    best_epoch, best_metric, best_threshold = best
    model.load_state_dict(checkpoints[best_epoch])

    def write(staging: Path):
        model.encoder.save_pretrained(staging); tokenizer.save_pretrained(staging)
        torch.save(model.gp.state_dict(), staging / "globalpointer.pt")
        receipt = {
            "schema_version": 1, "dataset_role": "fit_only_training", "route": "cmeie_native_argument_ner",
            "seed": args.seed, "train_sha256": sha(args.train), "split_manifest_sha256": sha(args.split_manifest),
            "types": types, "n_types": len(types), "base_model": args.base_model, "max_length": args.max_length,
            "best_epoch": best_epoch, "selected_calibration": {"threshold": best_threshold, **best_metric},
            "history": history, "dropped_label_count": dropped_total,
            "training_script_sha256": sha(Path(__file__)), "argument_module_sha256": sha(ROOT / "dki" / "nlp" / "cmeie_argument.py"),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "config": {key: getattr(args, key) for key in ("epochs", "batch_size", "gradient_accumulation", "encoder_lr", "head_lr", "warmup", "weight_decay", "gradient_clip", "patience", "device")},
            "threshold_grid": threshold_grid,
        }
        (staging / "training_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")

    publish_dir_once(args.output, write)
    print(json.dumps({"output": str(args.output), "best_epoch": best_epoch, "calibration": best_metric}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
