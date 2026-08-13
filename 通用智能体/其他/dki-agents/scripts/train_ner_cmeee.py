"""Train and evaluate a <1B GlobalPointer NER model on CMeEE-V2.

Model/checkpoint and decode-threshold selection use only a deterministic split of
official train.  Official dev is loaded exactly once after the candidate is
frozen and is never used for early stopping or threshold search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "cblue" / "CMeEE-V2"
OUT_MODEL = ROOT / "models" / "medical_ner_cmeee"
OUT_METRICS = ROOT / "outputs" / "cblue" / "cmeee_metrics.json"
TYPES = ["dis", "sym", "dru", "equ", "pro", "bod", "ite", "mic", "dep"]
TYPE2ID = {t: i for i, t in enumerate(TYPES)}
DEFAULT_THRESHOLDS = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


def _load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _text_digest(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).hexdigest()


def _split_train_internal(rows, ratio: float, seed: int):
    """Deterministic exact-text-group split; duplicate texts cannot cross it."""
    if not 0.0 < ratio < 1.0:
        raise ValueError("internal validation ratio must be between zero and one")
    if len(rows) < 2:
        raise ValueError("at least two training rows are required")
    groups = {}
    for row in rows:
        groups.setdefault(str(row.get("text", "")), []).append(row)
    if len(groups) < 2:
        raise ValueError("at least two distinct training texts are required")
    ordered = sorted(groups.items(), key=lambda item: (_text_digest(item[0], seed), item[0]))
    target = max(1, min(len(rows) - 1, round(len(rows) * ratio)))
    internal, fit = [], []
    count = 0
    for text, group in ordered:
        if count < target:
            internal.extend(group)
            count += len(group)
        else:
            fit.extend(group)
    if not fit:  # one large final group can overshoot; move the last group back.
        text, group = ordered[-1]
        internal = [row for row in internal if row not in group]
        fit.extend(group)
    fit_texts = {row["text"] for row in fit}
    internal_texts = {row["text"] for row in internal}
    if not fit or not internal or fit_texts & internal_texts:
        raise RuntimeError("failed to construct a non-empty text-disjoint internal split")
    return fit, internal


def _encode(tokenizer, rows, max_len):
    enc = tokenizer(
        [row["text"] for row in rows], truncation=True, max_length=max_len,
        padding="max_length", return_offsets_mapping=True, return_tensors="np",
    )
    offsets = enc["offset_mapping"]
    gold_spans = []
    char_gold = []
    for i, row in enumerate(rows):
        starts, ends = {}, {}
        for token_index, (start, end) in enumerate(offsets[i]):
            if start == end:
                continue
            for char_index in range(int(start), int(end)):
                starts.setdefault(char_index, token_index)
                ends[char_index] = token_index
        spans, chars = [], []
        for entity in row["entities"]:
            entity_type = entity["type"]
            if entity_type not in TYPE2ID:
                continue
            char_start, char_end = int(entity["start"]), int(entity["end"])
            chars.append((entity_type, char_start, char_end))
            if char_start in starts and char_end - 1 in ends:
                spans.append((TYPE2ID[entity_type], starts[char_start], ends[char_end - 1]))
        gold_spans.append(spans)
        char_gold.append((chars, offsets[i]))
    return enc, gold_spans, char_gold


def _build_label(gold_spans, batch_indices, heads, sequence_length):
    import torch

    labels = torch.zeros(len(batch_indices), heads, sequence_length, sequence_length)
    for batch_index, sample_index in enumerate(batch_indices):
        for head, start, end in gold_spans[sample_index]:
            if start < sequence_length and end < sequence_length:
                labels[batch_index, head, start, end] = 1.0
    return labels


def _metric_record(tp, fp, fn, threshold, per_type_counts=None):
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    record = {
        "threshold": float(threshold), "micro_f1": f1,
        "precision": precision, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
    }
    if per_type_counts is not None:
        per_type = {
            entity_type: _metric_record(*counts, threshold)
            for entity_type, counts in per_type_counts.items()
        }
        supported = [item for item in per_type.values() if item["tp"] + item["fn"] > 0]
        record["per_type"] = per_type
        record["macro_f1"] = sum(item["micro_f1"] for item in supported) / max(1, len(supported))
        if supported:
            worst_type, worst = min(
                ((name, item) for name, item in per_type.items() if item["tp"] + item["fn"] > 0),
                key=lambda pair: (pair[1]["micro_f1"], pair[0]),
            )
            record["worst_type"] = {
                "type": worst_type, "f1": worst["micro_f1"],
                "support": worst["tp"] + worst["fn"],
            }
    return record


def _select_threshold(records, fixed_threshold):
    """Max F1; deterministic ties prefer fixed proximity, precision, threshold."""
    return max(
        records,
        key=lambda item: (
            item["micro_f1"], -abs(item["threshold"] - fixed_threshold),
            item["precision"], item["threshold"],
        ),
    )


def _is_same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _promote_model_directory(output_model: Path, write_artifacts) -> None:
    """Build a complete multi-file model in staging, then promote it together."""
    output_model.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_model.name}.staging-", dir=output_model.parent))
    backup = output_model.with_name(f".{output_model.name}.backup-{os.getpid()}")
    try:
        write_artifacts(staging)
        if backup.exists():
            shutil.rmtree(backup)
        if output_model.exists():
            os.replace(output_model, backup)
        try:
            os.replace(staging, output_model)
        except BaseException:
            if backup.exists() and not output_model.exists():
                os.replace(backup, output_model)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5, help="encoder learning rate")
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--head-warmup-epochs", type=int, default=2)
    parser.add_argument("--unfreeze-last-layers", type=int, default=0,
                        help="joint phase: unfreeze only the last N encoder blocks; 0 means all")
    parser.add_argument("--joint-head-lr-scale", type=float, default=0.1,
                        help="head LR multiplier after encoder unfreezes")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="encoder gradient clipping norm (legacy-compatible name)")
    parser.add_argument("--head-max-grad-norm", type=float, default=5.0)
    parser.add_argument("--max-len", type=int, default=180)
    parser.add_argument("--inner-dim", type=int, default=64)
    parser.add_argument("--device", default="npu", choices=["npu", "cuda", "cpu"])
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-dev", type=int, default=0, help="official-dev smoke only")
    parser.add_argument("--internal-val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--eval-on-train", action="store_true", help="diagnostic only")
    parser.add_argument("--decode-threshold", type=float, default=0.0)
    parser.add_argument("--no-sched", action="store_true", help="diagnostic only")
    parser.add_argument("--sweep-threshold", action="store_true",
                        help="print train-derived internal threshold sweep")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="train GlobalPointer head only; supported production profile")
    parser.add_argument("--encoder-eval", action="store_true", help="legacy alias: keep encoder eval mode")
    parser.add_argument("--joint-encoder-eval", action=argparse.BooleanOptionalAction, default=True,
                        help="disable encoder dropout during joint fine-tuning (default: true)")
    parser.add_argument(
        "--pointer", default="classic",
        choices=["classic", "efficient", "egp"],
        help="classic GlobalPointer or Efficient GlobalPointer",
    )
    parser.add_argument(
        "--fgm-epsilon", type=float, default=0.0,
        help="FGM epsilon on word embeddings; 0 disables",
    )
    parser.add_argument(
        "--type-loss-weight", default="none",
        choices=["none", "inverse_freq"],
        help="per-type loss weighting on Multilabel CE rows",
    )
    parser.add_argument(
        "--softlexicon", action="store_true",
        help="inject fit-only SoftLexicon BMES features into encoder hidden states",
    )
    parser.add_argument(
        "--softlex-mode", default="gated", choices=["gated", "add"],
        help="SoftLex fusion: gated (zero-init gate) or plain add",
    )
    parser.add_argument(
        "--use-bilstm", action="store_true",
        help="insert a bidirectional LSTM between encoder and pointer",
    )
    parser.add_argument(
        "--init-from", type=Path, default=None,
        help="warm-start from a promoted CMeEE checkpoint directory "
             "(encoder + globalpointer.pt; softlex_proj kept if shapes match)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-model", type=Path, default=OUT_MODEL)
    parser.add_argument("--output-metrics", type=Path, default=OUT_METRICS)
    args = parser.parse_args()
    if args.pointer == "egp":
        args.pointer = "efficient"
    if args.fgm_epsilon < 0:
        parser.error("fgm-epsilon must be non-negative")
    if args.init_from is not None and not args.init_from.is_dir():
        parser.error(f"--init-from directory not found: {args.init_from}")

    if args.unfreeze_last_layers < 0:
        parser.error("unfreeze-last-layers must be non-negative")
    if not 0 < args.joint_head_lr_scale <= 1:
        parser.error("joint-head-lr-scale must be in (0, 1]")
    if args.lr < 0 or args.head_lr <= 0:
        parser.error("encoder lr must be non-negative and head lr must be positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")
    if not args.freeze_encoder and not 0 <= args.head_warmup_epochs < args.epochs:
        parser.error("full fine-tuning requires 0 <= head-warmup-epochs < epochs")
    if args.max_grad_norm <= 0 or args.head_max_grad_norm <= 0:
        parser.error("gradient clipping norms must be positive")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("warmup-ratio must be in [0, 1)")
    diagnostic_run = (
        args.limit_train > 0
        or args.limit_dev > 0
        or args.eval_on_train
        or args.no_sched
    )
    if diagnostic_run and (
        _is_same_path(args.output_model, OUT_MODEL)
        or _is_same_path(args.output_metrics, OUT_METRICS)
    ):
        parser.error(
            "limited/eval-on-train runs require explicit noncanonical "
            "--output-model and --output-metrics paths"
        )

    import math
    import numpy as np
    import torch
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    from transformers import AutoModel, AutoTokenizer
    from dki.nlp.globalpointer import EncoderGlobalPointer, FGM, decode_spans, gp_loss
    from dki.nlp.softlexicon import build_surface_lexicon, encode_softlex_batch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if hasattr(torch, "npu") and hasattr(torch.npu, "manual_seed_all"):
        torch.npu.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "npu" else "npu")
    split_seed = args.seed if args.split_seed is None else args.split_seed

    all_train_rows = _load(DATA / "train.jsonl")
    if args.limit_train > 0:
        all_train_rows = all_train_rows[:args.limit_train]
    train_rows, internal_rows = _split_train_internal(
        all_train_rows, args.internal_val_ratio, split_seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    enc_train, gold_train, _ = _encode(tokenizer, train_rows, args.max_len)
    enc_internal, _, char_internal = _encode(tokenizer, internal_rows, args.max_len)
    aligned = sum(len(spans) for spans in gold_train)
    total_entities = sum(len(row["entities"]) for row in train_rows)
    print(
        f"[split] fit={len(train_rows)} internal={len(internal_rows)} "
        f"text_overlap=0 seed={split_seed}\n"
        f"[align] fit spans aligned={aligned}/{total_entities} "
        f"({100 * aligned / max(1, total_entities):.1f}%)",
        flush=True,
    )

    softlex_dim = 0
    softlex_lexicon = None
    softlex_train = softlex_internal = None
    if args.softlexicon:
        softlex_dim = len(TYPES) * 4
        softlex_lexicon = build_surface_lexicon(train_rows, TYPE2ID)
        softlex_train = encode_softlex_batch(
            train_rows, enc_train["offset_mapping"], softlex_lexicon, len(TYPES), args.max_len,
        )
        softlex_internal = encode_softlex_batch(
            internal_rows, enc_internal["offset_mapping"], softlex_lexicon, len(TYPES), args.max_len,
        )
        coverage = float((softlex_train.max(axis=-1) > 0).mean())
        print(
            f"[softlexicon] surfaces={len(softlex_lexicon)} dim={softlex_dim} "
            f"fit_token_hit_rate={coverage:.3f}",
            flush=True,
        )

    encoder = AutoModel.from_pretrained(args.base_model)
    model = EncoderGlobalPointer(
        encoder, encoder.config.hidden_size, len(TYPES), args.inner_dim,
        pointer=args.pointer, softlex_dim=softlex_dim, softlex_mode=args.softlex_mode,
        use_bilstm=bool(args.use_bilstm),
    ).to(device)
    if args.init_from is not None:
        init_path = args.init_from.resolve()
        print(f"[init-from] {init_path}", flush=True)
        # Prefer local encoder weights; fall back to HF base if incomplete.
        try:
            encoder_loaded = AutoModel.from_pretrained(str(init_path))
            model.encoder.load_state_dict(encoder_loaded.state_dict())
            del encoder_loaded
        except Exception as exc:  # noqa: BLE001 — warm-start best-effort
            print(f"[init-from] encoder load warn: {exc}", flush=True)
        gp_path = init_path / "globalpointer.pt"
        if gp_path.is_file():
            gp_state = torch.load(gp_path, map_location="cpu")
            missing, unexpected = model.gp.load_state_dict(gp_state, strict=False)
            print(f"[init-from] gp missing={list(missing)} unexpected={list(unexpected)}", flush=True)
        softlex_path = init_path / "softlexicon_proj.pt"
        if model.softlex_proj is not None and softlex_path.is_file():
            sl_state = torch.load(softlex_path, map_location="cpu")
            missing, unexpected = model.softlex_proj.load_state_dict(sl_state, strict=False)
            print(
                f"[init-from] softlex_proj missing={list(missing)} unexpected={list(unexpected)}",
                flush=True,
            )
        gate_path = init_path / "softlexicon_gate.pt"
        if model.softlex_gate is not None and gate_path.is_file():
            gate_state = torch.load(gate_path, map_location="cpu")
            model.softlex_gate.data.copy_(gate_state["gate"].reshape_as(model.softlex_gate.data))
            print(f"[init-from] softlex_gate={float(model.softlex_gate.detach().cpu())}", flush=True)
        bilstm_path = init_path / "bilstm.pt"
        if model.bilstm is not None and bilstm_path.is_file():
            bl_state = torch.load(bilstm_path, map_location="cpu")
            missing, unexpected = model.bilstm.load_state_dict(bl_state, strict=False)
            print(f"[init-from] bilstm missing={list(missing)} unexpected={list(unexpected)}", flush=True)
    type_weights = None
    if args.type_loss_weight == "inverse_freq":
        counts = np.zeros(len(TYPES), dtype=np.float64)
        for spans in gold_train:
            for head, _, _ in spans:
                counts[head] += 1.0
        counts = np.maximum(counts, 1.0)
        inv = counts.sum() / counts
        type_weights = torch.tensor(inv / inv.mean(), dtype=torch.float32, device=device)
        print(f"[type_weights] inverse_freq={type_weights.detach().cpu().tolist()}", flush=True)
    fgm = FGM(model, args.fgm_epsilon) if args.fgm_epsilon > 0 else None
    no_decay = ("bias", "LayerNorm.weight", "layer_norm")

    def groups(named_parameters, learning_rate):
        named_parameters = list(named_parameters)
        decay = [p for name, p in named_parameters if not any(key in name for key in no_decay)]
        no_weight_decay = [p for name, p in named_parameters if any(key in name for key in no_decay)]
        return [
            {"params": decay, "lr": learning_rate, "weight_decay": 0.01},
            {"params": no_weight_decay, "lr": learning_rate, "weight_decay": 0.0},
        ]

    head_named = list(model.gp.named_parameters())
    if model.softlex_proj is not None:
        head_named += list(model.softlex_proj.named_parameters())
    if model.softlex_gate is not None:
        head_named += [("softlex_gate", model.softlex_gate)]
    if model.bilstm is not None:
        head_named += list(model.bilstm.named_parameters())
    optimizer = torch.optim.AdamW(
        groups(model.encoder.named_parameters(), args.lr)
        + groups(head_named, args.head_lr)
    )
    ids_train = torch.tensor(enc_train["input_ids"])
    mask_train = torch.tensor(enc_train["attention_mask"])
    tt_train = torch.tensor(enc_train.get("token_type_ids", np.zeros_like(enc_train["input_ids"])))
    softlex_train_t = (
        torch.tensor(softlex_train) if softlex_train is not None else None
    )
    softlex_internal_t = (
        torch.tensor(softlex_internal) if softlex_internal is not None else None
    )
    threshold_candidates = sorted(set(DEFAULT_THRESHOLDS + (float(args.decode_threshold),)))
    steps_per_epoch = (len(train_rows) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    head_only_steps = 0 if args.freeze_encoder else steps_per_epoch * args.head_warmup_epochs
    joint_steps = max(1, total_steps - head_only_steps)
    head_warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    encoder_warmup_steps = max(1, int(joint_steps * args.warmup_ratio))

    def cosine(step, start, length, warmup):
        relative = step - start
        if relative < 0:
            return 0.0
        if relative < warmup:
            return max(1e-3, relative / max(1, warmup))
        progress = min(1.0, (relative - warmup) / max(1, length - warmup))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = None
    if not args.no_sched:
        encoder_lambda = lambda step: cosine(step, head_only_steps, joint_steps, encoder_warmup_steps)

        def head_lambda(step):
            value = cosine(step, 0, total_steps, head_warmup_steps)
            if head_only_steps > 0 and step >= head_only_steps:
                value *= args.joint_head_lr_scale
            return value

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, [encoder_lambda, encoder_lambda, head_lambda, head_lambda],
        )

    def set_encoder_trainable(flag):
        for parameter in model.encoder.parameters():
            parameter.requires_grad = flag
        if flag and args.unfreeze_last_layers > 0:
            # BERT/RoBERTa-compatible block discovery; fail closed for unknown layouts.
            blocks = getattr(getattr(model.encoder, "encoder", None), "layer", None)
            if blocks is None:
                raise RuntimeError("unfreeze-last-layers requires encoder.encoder.layer")
            for parameter in model.encoder.parameters():
                parameter.requires_grad = False
            selected_blocks = list(blocks)[-args.unfreeze_last_layers:]
            if len(selected_blocks) != args.unfreeze_last_layers:
                raise ValueError("unfreeze-last-layers exceeds available encoder blocks")
            for block in selected_blocks:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    def score(encoded, char_gold, thresholds, softlex_features=None, eval_rows=None):
        ids = torch.tensor(encoded["input_ids"])
        mask = torch.tensor(encoded["attention_mask"])
        token_types = torch.tensor(encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"])))
        if softlex_features is None and softlex_dim > 0 and eval_rows is not None:
            softlex_features = encode_softlex_batch(
                eval_rows, encoded["offset_mapping"], softlex_lexicon, len(TYPES), args.max_len,
            )
        softlex_t = torch.tensor(softlex_features) if softlex_features is not None else None
        counts = {threshold: [0, 0, 0] for threshold in thresholds}
        per_type_counts = {
            threshold: {entity_type: [0, 0, 0] for entity_type in TYPES}
            for threshold in thresholds
        }
        model.eval()
        with torch.no_grad():
            for start in range(0, len(char_gold), args.batch_size):
                stop = min(len(char_gold), start + args.batch_size)
                soft_kw = {}
                if softlex_t is not None:
                    soft_kw["softlex"] = softlex_t[start:stop].to(device)
                logits = model(
                    ids[start:stop].to(device), mask[start:stop].to(device),
                    token_types[start:stop].to(device),
                    **soft_kw,
                ).float().cpu()
                for threshold in thresholds:
                    predictions = decode_spans(logits, threshold=threshold)
                    for local_index, spans in enumerate(predictions):
                        gold_chars, offsets = char_gold[start + local_index]
                        gold_set = set(gold_chars)
                        predicted_set = set()
                        for head, token_start, token_end in spans:
                            char_start = int(offsets[token_start][0])
                            char_end = int(offsets[token_end][1])
                            if char_end > char_start:
                                predicted_set.add((TYPES[head], char_start, char_end))
                        counts[threshold][0] += len(gold_set & predicted_set)
                        counts[threshold][1] += len(predicted_set - gold_set)
                        counts[threshold][2] += len(gold_set - predicted_set)
                        for entity_type in TYPES:
                            gold_type = {item for item in gold_set if item[0] == entity_type}
                            predicted_type = {item for item in predicted_set if item[0] == entity_type}
                            type_counts = per_type_counts[threshold][entity_type]
                            type_counts[0] += len(gold_type & predicted_type)
                            type_counts[1] += len(predicted_type - gold_type)
                            type_counts[2] += len(gold_type - predicted_type)
        return [
            _metric_record(
                *counts[threshold], threshold,
                per_type_counts=per_type_counts[threshold],
            )
            for threshold in thresholds
        ]

    best_state = None
    best_epoch = None
    best_phase = None
    best_internal = None
    start_time = time.time()
    for epoch in range(args.epochs):
        head_only = not args.freeze_encoder and epoch < args.head_warmup_epochs
        set_encoder_trainable(False if args.freeze_encoder or head_only else True)
        model.train()
        if args.freeze_encoder or head_only or args.encoder_eval or args.joint_encoder_eval:
            model.encoder.eval()
        permutation = np.random.permutation(len(train_rows))
        loss_total = 0.0
        for offset in range(0, len(train_rows), args.batch_size):
            batch_indices = permutation[offset:offset + args.batch_size].tolist()
            optimizer.zero_grad(set_to_none=True)
            soft_kw = {}
            if softlex_train_t is not None:
                soft_kw["softlex"] = softlex_train_t[batch_indices].to(device)
            logits = model(
                ids_train[batch_indices].to(device), mask_train[batch_indices].to(device),
                tt_train[batch_indices].to(device),
                **soft_kw,
            )
            labels = _build_label(gold_train, batch_indices, len(TYPES), args.max_len).to(device)
            loss = gp_loss(logits, labels, type_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
            loss.backward()
            if fgm is not None and fgm.attack():
                logits_adv = model(
                    ids_train[batch_indices].to(device), mask_train[batch_indices].to(device),
                    tt_train[batch_indices].to(device),
                    **soft_kw,
                )
                loss_adv = gp_loss(logits_adv, labels, type_weights)
                if torch.isfinite(loss_adv):
                    loss_adv.backward()
                fgm.restore()
            head_params = list(model.gp.parameters())
            if model.softlex_proj is not None:
                head_params += list(model.softlex_proj.parameters())
            if model.softlex_gate is not None:
                head_params += [model.softlex_gate]
            if model.bilstm is not None:
                head_params += list(model.bilstm.parameters())
            head_norm = torch.nn.utils.clip_grad_norm_(head_params, args.head_max_grad_norm)
            if not args.freeze_encoder and not head_only:
                encoder_norm = torch.nn.utils.clip_grad_norm_(
                    model.encoder.parameters(), args.max_grad_norm,
                )
                if not torch.isfinite(encoder_norm):
                    raise FloatingPointError(f"non-finite encoder gradient at epoch {epoch + 1}")
            if not torch.isfinite(head_norm):
                raise FloatingPointError(f"non-finite head gradient at epoch {epoch + 1}")
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_total += float(loss.item())

        internal_records = score(
            enc_internal, char_internal, threshold_candidates,
            softlex_features=softlex_internal, eval_rows=internal_rows,
        )
        internal_selected = _select_threshold(internal_records, args.decode_threshold)
        internal_fixed = next(
            item for item in internal_records if item["threshold"] == args.decode_threshold
        )
        phase = "head" if (head_only or args.freeze_encoder) else "joint"
        # Joint checkpoint+threshold selection is confined to the same train-derived
        # internal split. This is robust to harmless logit-scale drift across epochs.
        if best_internal is None or internal_selected["micro_f1"] > best_internal["micro_f1"]:
            best_internal = internal_selected
            best_epoch = epoch + 1
            best_phase = phase
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
        print(
            f"[epoch {epoch + 1}/{args.epochs}][{phase}] "
            f"loss={loss_total / max(1, steps_per_epoch):.4f} "
            f"internal_fixed_f1={internal_fixed['micro_f1']:.4f} "
            f"internal_selected_f1={internal_selected['micro_f1']:.4f} "
            f"threshold={internal_selected['threshold']:+.1f} "
            f"lr_enc={optimizer.param_groups[0]['lr']:.2e} "
            f"lr_head={optimizer.param_groups[2]['lr']:.2e}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("no eligible checkpoint was selected")
    model.load_state_dict(best_state)
    # Re-score the restored state to assert the bound operating point is reproducible.
    internal_sweep = score(
        enc_internal, char_internal, threshold_candidates,
        softlex_features=softlex_internal, eval_rows=internal_rows,
    )
    selected = _select_threshold(internal_sweep, args.decode_threshold)
    if args.sweep_threshold:
        print("== train-derived internal threshold sweep ==", flush=True)
        for item in internal_sweep:
            print(
                f"  thr={item['threshold']:+.1f} F1={item['micro_f1']:.4f} "
                f"P={item['precision']:.4f} R={item['recall']:.4f}", flush=True,
            )

    # Candidate is now frozen. Official dev is loaded only below this line.
    if args.eval_on_train:
        eval_rows = train_rows
        encoded_eval, _, char_eval = _encode(tokenizer, eval_rows, args.max_len)
        eval_split = "train_self_check"
        dataset_role = "diagnostic"
    else:
        eval_rows = _load(DATA / "dev.jsonl")
        if args.limit_dev > 0:
            eval_rows = eval_rows[:args.limit_dev]
        encoded_eval, _, char_eval = _encode(tokenizer, eval_rows, args.max_len)
        eval_split = "official_dev"
        dataset_role = (
            "diagnostic_limited_official_dev"
            if args.limit_dev > 0 else "development_locked_report"
        )
    official_records = score(
        encoded_eval, char_eval,
        sorted(set((float(args.decode_threshold), selected["threshold"]))),
        eval_rows=eval_rows,
    )
    by_threshold = {item["threshold"]: item for item in official_records}
    fixed = by_threshold[float(args.decode_threshold)]
    selected_official = by_threshold[selected["threshold"]]

    metrics = {
        "schema_version": 2,
        "benchmark": "CBLUE CMeEE-V2",
        "dataset_role": dataset_role,
        "eval_split": eval_split,
        "selection_protocol": {
            "checkpoint": "train-derived internal validation, jointly selected with a preregistered threshold grid",
            "threshold": "train-derived internal validation only",
            "official_dev_used_for_selection": False,
            "text_group_disjoint": True,
        },
        "base_model": args.base_model,
        "decoder": (
            "EfficientGlobalPointer" if args.pointer == "efficient" else "GlobalPointer"
        ),
        "pointer_kind": args.pointer,
        "entity_types": TYPES,
        "n_fit": len(train_rows),
        "n_internal_validation": len(internal_rows),
        "n_dev": len(eval_rows),
        "best_epoch": best_epoch,
        "selected_phase": best_phase,
        "joint_finetuning_promoted": best_phase == "joint",
        "primary_operating_point": "internally_selected_threshold",
        "primary_fixed_threshold": {
            **{key: (round(value, 4) if isinstance(value, float) else value)
               for key, value in fixed.items() if key not in {"per_type", "worst_type"}},
            "per_type": fixed.get("per_type"),
            "worst_type": fixed.get("worst_type"),
            "selection_used": False,
        },
        "internally_selected_threshold": {
            "threshold": selected["threshold"],
            "internal_micro_f1": round(selected["micro_f1"], 4),
            "official_report": {
                **{key: (round(value, 4) if isinstance(value, float) else value)
                   for key, value in selected_official.items()
                   if key not in {"per_type", "worst_type"}},
                "per_type": selected_official.get("per_type"),
                "worst_type": selected_official.get("worst_type"),
                "selection_used": True,
            },
        },
        # Top-level aliases describe the operating point loaded by default inference.
        "micro_f1": round(selected_official["micro_f1"], 4),
        "precision": round(selected_official["precision"], 4),
        "recall": round(selected_official["recall"], 4),
        "tp": selected_official["tp"], "fp": selected_official["fp"], "fn": selected_official["fn"],
        "decode_threshold": selected_official["threshold"],
        "train_seconds": round(time.time() - start_time, 1),
        "train_config": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "encoder_lr": args.lr, "head_lr": args.head_lr,
            "warmup_ratio": args.warmup_ratio, "max_len": args.max_len,
            "inner_dim": args.inner_dim, "seed": args.seed,
            "split_seed": split_seed, "internal_val_ratio": args.internal_val_ratio,
            "head_warmup_epochs": args.head_warmup_epochs,
            "unfreeze_last_layers": args.unfreeze_last_layers,
            "joint_head_lr_scale": args.joint_head_lr_scale,
            "joint_encoder_eval": args.joint_encoder_eval,
            "encoder_grad_clip": args.max_grad_norm,
            "head_grad_clip": args.head_max_grad_norm,
            "freeze_encoder": args.freeze_encoder,
            "encoder_eval": args.encoder_eval, "no_sched": args.no_sched,
            "pointer": args.pointer,
            "fgm_epsilon": args.fgm_epsilon,
            "type_loss_weight": args.type_loss_weight,
            "softlexicon": bool(args.softlexicon),
            "softlex_mode": args.softlex_mode if args.softlexicon else None,
            "use_bilstm": bool(args.use_bilstm),
            "init_from": str(args.init_from) if args.init_from else None,
        },
        "cblue_baseline_ref": {
            "GlobalPointer(bert-base, paper)": "~65.98 dev",
            "RoBERTa-wwm-ext-large": "~67", "human": "~67",
            "EfficientGlobalPointer_ref": "Su Jianlin; community EGP~67.9 with RoBERTa-large",
            "SoftLexicon_ref": "Ma et al.; SoftLexicon+EGP community CMeEE often 0.70+",
        },
    }

    output_metrics = args.output_metrics
    output_model = args.output_model
    deployment_config = {
        "schema_version": 1, "types": TYPES, "inner_dim": args.inner_dim,
        "max_len": args.max_len, "fixed_threshold": args.decode_threshold,
        "selected_threshold": selected["threshold"], "best_epoch": best_epoch,
        "selected_phase": best_phase,
        "base_model": args.base_model, "split_seed": split_seed,
        "internal_val_ratio": args.internal_val_ratio,
        "pointer": args.pointer,
        "softlexicon": bool(args.softlexicon),
        "softlex_mode": args.softlex_mode if args.softlexicon else None,
        "softlex_dim": softlex_dim,
        "use_bilstm": bool(args.use_bilstm),
    }

    def write_model(staging: Path) -> None:
        model.encoder.save_pretrained(staging)
        torch.save(model.gp.state_dict(), staging / "globalpointer.pt")
        if model.bilstm is not None:
            torch.save(model.bilstm.state_dict(), staging / "bilstm.pt")
        if model.softlex_proj is not None:
            torch.save(model.softlex_proj.state_dict(), staging / "softlexicon_proj.pt")
            if model.softlex_gate is not None:
                torch.save({"gate": model.softlex_gate.detach().cpu()}, staging / "softlexicon_gate.pt")
            # Persist fit-only surfaces for inference-time SoftLexicon encoding.
            surfaces = {
                surface: sorted(type_ids)
                for surface, type_ids in sorted((softlex_lexicon or {}).items())
            }
            (staging / "softlexicon_surfaces.json").write_text(
                json.dumps(surfaces, ensure_ascii=False), encoding="utf-8",
            )
        tokenizer.save_pretrained(staging)
        (staging / "types.json").write_text(
            json.dumps(TYPES, ensure_ascii=False), encoding="utf-8",
        )
        (staging / "globalpointer_config.json").write_text(
            json.dumps(deployment_config, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    _promote_model_directory(output_model, write_model)
    _write_json_atomic(output_metrics, metrics)
    print(json.dumps({
        "micro_f1": metrics["micro_f1"],
        "decode_threshold": metrics["decode_threshold"],
        "micro_f1_fixed": round(fixed["micro_f1"], 4),
        "precision": metrics["precision"], "recall": metrics["recall"],
        "best_epoch": best_epoch, "train_seconds": metrics["train_seconds"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
