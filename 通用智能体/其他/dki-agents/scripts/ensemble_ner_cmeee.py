"""CMeEE multi-checkpoint ensemble with train-derived fusion selection.

Official dev is scored only after the fusion policy is frozen on the internal split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_ner_cmeee import (  # noqa: E402
    DATA,
    DEFAULT_THRESHOLDS,
    TYPES,
    TYPE2ID,
    _encode,
    _load,
    _metric_record,
    _select_threshold,
    _split_train_internal,
)


def _load_bundle(model_dir: Path, device, batch_size: int):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from dki.nlp.globalpointer import EncoderGlobalPointer, decode_spans
    from dki.nlp.softlexicon import encode_softlex_batch

    config = json.loads((model_dir / "globalpointer_config.json").read_text(encoding="utf-8"))
    types = json.loads((model_dir / "types.json").read_text(encoding="utf-8"))
    if types != TYPES:
        raise ValueError(f"{model_dir}: types mismatch {types}")
    pointer = str(config.get("pointer", "classic"))
    max_len = int(config["max_len"])
    inner_dim = int(config["inner_dim"])
    threshold = float(config["selected_threshold"])
    softlex_dim = int(config.get("softlex_dim", 0) or 0)
    softlex_on = bool(config.get("softlexicon", False)) or softlex_dim > 0
    softlex_mode = str(config.get("softlex_mode") or "gated")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoder = AutoModel.from_pretrained(model_dir)
    model = EncoderGlobalPointer(
        encoder, encoder.config.hidden_size, len(types), inner_dim,
        pointer=pointer, softlex_dim=softlex_dim if softlex_on else 0,
        softlex_mode=softlex_mode,
    )
    try:
        state = torch.load(model_dir / "globalpointer.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_dir / "globalpointer.pt", map_location="cpu")
    model.gp.load_state_dict(state)
    softlex_lexicon = None
    if model.softlex_proj is not None:
        sl_path = model_dir / "softlexicon_proj.pt"
        if not sl_path.is_file():
            raise FileNotFoundError(f"{model_dir}: missing softlexicon_proj.pt")
        try:
            sl_state = torch.load(sl_path, map_location="cpu", weights_only=True)
        except TypeError:
            sl_state = torch.load(sl_path, map_location="cpu")
        model.softlex_proj.load_state_dict(sl_state)
        gate_path = model_dir / "softlexicon_gate.pt"
        if model.softlex_gate is not None and gate_path.is_file():
            try:
                gate_state = torch.load(gate_path, map_location="cpu", weights_only=True)
            except TypeError:
                gate_state = torch.load(gate_path, map_location="cpu")
            model.softlex_gate.data.copy_(gate_state["gate"].reshape_as(model.softlex_gate.data))
        surfaces = json.loads((model_dir / "softlexicon_surfaces.json").read_text(encoding="utf-8"))
        softlex_lexicon = {k: set(v) for k, v in surfaces.items()}
    model = model.to(device).eval()
    return {
        "dir": model_dir,
        "model": model,
        "tokenizer": tokenizer,
        "max_len": max_len,
        "threshold": threshold,
        "pointer": pointer,
        "batch_size": batch_size,
        "decode_spans": decode_spans,
        "softlex_dim": softlex_dim if softlex_on else 0,
        "softlex_lexicon": softlex_lexicon,
        "encode_softlex_batch": encode_softlex_batch,
        "np": np,
        "torch": torch,
    }


def _predict_char_sets(bundle, rows):
    """Return list[set[(type, start, end)]] for each row."""
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    max_len = bundle["max_len"]
    thr = bundle["threshold"]
    batch_size = bundle["batch_size"]
    decode_spans = bundle["decode_spans"]
    torch = bundle["torch"]
    device = next(model.parameters()).device

    enc, _, char_gold = _encode(tokenizer, rows, max_len)
    ids = torch.tensor(enc["input_ids"])
    mask = torch.tensor(enc["attention_mask"])
    token_types = torch.tensor(enc.get("token_type_ids", bundle["np"].zeros_like(enc["input_ids"])))
    softlex_t = None
    if bundle.get("softlex_dim", 0) > 0 and bundle.get("softlex_lexicon") is not None:
        softlex = bundle["encode_softlex_batch"](
            rows, enc["offset_mapping"], bundle["softlex_lexicon"], len(TYPES), max_len,
        )
        softlex_t = torch.tensor(softlex)
    pred_sets = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            stop = min(len(rows), start + batch_size)
            soft_kw = {}
            if softlex_t is not None:
                soft_kw["softlex"] = softlex_t[start:stop].to(device)
            logits = model(
                ids[start:stop].to(device),
                mask[start:stop].to(device),
                token_types[start:stop].to(device),
                **soft_kw,
            ).float().cpu()
            decoded = decode_spans(logits, threshold=thr)
            for local, spans in enumerate(decoded):
                _, offsets = char_gold[start + local]
                predicted = set()
                for head, token_start, token_end in spans:
                    char_start = int(offsets[token_start][0])
                    char_end = int(offsets[token_end][1])
                    if char_end > char_start:
                        predicted.add((TYPES[head], char_start, char_end))
                pred_sets.append(predicted)
    gold_sets = [set(chars) for chars, _ in char_gold]
    return pred_sets, gold_sets


def _fuse(member_preds, mode: str, k: int = 2):
    """member_preds: list[model][sample] -> set. Return list[sample] sets."""
    n_models = len(member_preds)
    n_samples = len(member_preds[0])
    out = []
    for i in range(n_samples):
        counts = {}
        for m in range(n_models):
            for ent in member_preds[m][i]:
                counts[ent] = counts.get(ent, 0) + 1
        if mode == "union":
            chosen = set(counts)
        elif mode == "intersect":
            chosen = {e for e, c in counts.items() if c == n_models}
        elif mode.startswith("vote"):
            chosen = {e for e, c in counts.items() if c >= k}
        else:
            raise ValueError(mode)
        out.append(chosen)
    return out


def _score_sets(pred_sets, gold_sets):
    tp = fp = fn = 0
    per_type = {t: [0, 0, 0] for t in TYPES}
    for pred, gold in zip(pred_sets, gold_sets):
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        for t in TYPES:
            g = {x for x in gold if x[0] == t}
            p = {x for x in pred if x[0] == t}
            per_type[t][0] += len(g & p)
            per_type[t][1] += len(p - g)
            per_type[t][2] += len(g - p)
    return _metric_record(tp, fp, fn, threshold=0.0, per_type_counts=per_type)


def _build_train_lexicon(fit_rows):
    """Self SoftLexicon: surface -> set of types from fit-only entities."""
    lex = {}
    for row in fit_rows:
        text = row["text"]
        for ent in row.get("entities", []):
            t = ent["type"]
            if t not in TYPE2ID:
                continue
            surface = text[int(ent["start"]): int(ent["end"])]
            if not surface:
                continue
            lex.setdefault(surface, set()).add(t)
    return lex


def _lexicon_candidates(text: str, lexicon: dict):
    """All lexicon surface matches in text as (type, start, end)."""
    hits = set()
    # Longer surfaces first to prefer maximal spans.
    surfaces = sorted(lexicon.keys(), key=len, reverse=True)
    occupied = [False] * len(text)
    for surface in surfaces:
        if len(surface) < 2:
            continue
        start = 0
        while True:
            idx = text.find(surface, start)
            if idx < 0:
                break
            end = idx + len(surface)
            if not any(occupied[idx:end]):
                for t in lexicon[surface]:
                    hits.add((t, idx, end))
                for j in range(idx, end):
                    occupied[j] = True
            start = idx + 1
    return hits


def _apply_lexicon_boost(pred_sets, rows, lexicon, mode: str):
    """mode: off | union_lex | vote_or_lex (add lexicon hits only)."""
    if mode == "off":
        return pred_sets
    out = []
    for pred, row in zip(pred_sets, rows):
        hits = _lexicon_candidates(row["text"], lexicon)
        if mode == "union_lex":
            out.append(pred | hits)
        elif mode == "recall_lex":
            # Add lexicon hit only when no overlapping same-type prediction exists.
            merged = set(pred)
            for t, s, e in hits:
                overlap = any(
                    pt == t and not (e <= ps or pe <= s)
                    for pt, ps, pe in pred
                )
                if not overlap:
                    merged.add((t, s, e))
            out.append(merged)
        else:
            raise ValueError(mode)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="CMeEE checkpoint ensemble")
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--device", default="npu", choices=["npu", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--internal-val-ratio", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floor", type=float, default=0.5940)
    parser.add_argument(
        "--min-member-internal-f1", type=float, default=0.50,
        help="drop members whose solo internal F1 is below this (guards against broken loads)",
    )
    args = parser.parse_args()

    import torch
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    device = torch.device(args.device if args.device != "npu" else "npu")

    bundles = [_load_bundle(path, device, args.batch_size) for path in args.models]
    print(f"[ensemble] n_models={len(bundles)}", flush=True)
    for b in bundles:
        print(f"  - {b['dir']} thr={b['threshold']} pointer={b['pointer']}", flush=True)

    all_train = _load(DATA / "train.jsonl")
    fit_rows, internal_rows = _split_train_internal(
        all_train, args.internal_val_ratio, args.split_seed,
    )
    lexicon = _build_train_lexicon(fit_rows)
    print(f"[lexicon] fit surfaces={len(lexicon)}", flush=True)

    member_internal = []
    gold_internal = None
    kept_bundles = []
    for bundle in bundles:
        pred, gold = _predict_char_sets(bundle, internal_rows)
        gold_internal = gold
        solo = _score_sets(pred, gold)
        print(f"[internal solo] {bundle['dir'].name} F1={solo['micro_f1']:.4f}", flush=True)
        if solo["micro_f1"] < args.min_member_internal_f1:
            print(
                f"[drop] {bundle['dir'].name} below min-member-internal-f1="
                f"{args.min_member_internal_f1}",
                flush=True,
            )
            continue
        member_internal.append(pred)
        kept_bundles.append(bundle)
    bundles = kept_bundles
    if len(bundles) < 1:
        raise RuntimeError("no ensemble members survived internal F1 filter")
    print(f"[ensemble] kept_models={len(bundles)}", flush=True)

    policies = []
    n = len(bundles)
    for mode, k in [("union", 1), ("intersect", n)] + [("vote", k) for k in range(2, n + 1)]:
        for lex_mode in ("off", "recall_lex", "union_lex"):
            fused = _fuse(member_internal, mode if mode != "vote" else "vote", k=k)
            fused = _apply_lexicon_boost(fused, internal_rows, lexicon, lex_mode)
            rec = _score_sets(fused, gold_internal)
            policies.append({
                "mode": mode, "k": k, "lex_mode": lex_mode,
                "internal_micro_f1": rec["micro_f1"],
                "record": rec,
            })
            print(
                f"[internal fuse] mode={mode} k={k} lex={lex_mode} "
                f"F1={rec['micro_f1']:.4f}",
                flush=True,
            )

    best = max(policies, key=lambda p: p["internal_micro_f1"])
    print(
        f"[selected] mode={best['mode']} k={best['k']} lex={best['lex_mode']} "
        f"internal_F1={best['internal_micro_f1']:.4f}",
        flush=True,
    )

    # Official dev once.
    dev_rows = _load(DATA / "dev.jsonl")
    member_dev = []
    gold_dev = None
    for bundle in bundles:
        pred, gold = _predict_char_sets(bundle, dev_rows)
        member_dev.append(pred)
        gold_dev = gold
    fused_dev = _fuse(
        member_dev,
        best["mode"] if best["mode"] != "vote" else "vote",
        k=best["k"],
    )
    fused_dev = _apply_lexicon_boost(fused_dev, dev_rows, lexicon, best["lex_mode"])
    official = _score_sets(fused_dev, gold_dev)
    print(
        f"[official_dev] F1={official['micro_f1']:.4f} "
        f"P={official['precision']:.4f} R={official['recall']:.4f}",
        flush=True,
    )

    payload = {
        "schema_version": 2,
        "benchmark": "CBLUE CMeEE-V2",
        "dataset_role": "development_locked_report",
        "eval_split": "official_dev",
        "decoder": "EnsembleGlobalPointer",
        "selection_protocol": {
            "checkpoint": "frozen member checkpoints",
            "fusion": "train-derived internal validation only",
            "official_dev_used_for_selection": False,
            "text_group_disjoint": True,
        },
        "members": [str(b["dir"]) for b in bundles],
        "selected_policy": {
            "mode": best["mode"], "k": best["k"], "lex_mode": best["lex_mode"],
            "internal_micro_f1": round(best["internal_micro_f1"], 4),
        },
        "micro_f1": round(official["micro_f1"], 4),
        "precision": round(official["precision"], 4),
        "recall": round(official["recall"], 4),
        "tp": official["tp"], "fp": official["fp"], "fn": official["fn"],
        "per_type": official.get("per_type"),
        "worst_type": official.get("worst_type"),
        "floor": args.floor,
        "promoted": official["micro_f1"] > args.floor + 1e-9,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output} promoted={payload['promoted']}", flush=True)
    if payload["promoted"]:
        metrics_path = ROOT / "outputs/cblue/cmeee_metrics.json"
        metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"PROMOTED_CMEEE_ENSEMBLE F1={payload['micro_f1']}", flush=True)
    else:
        print(f"KEEP_FLOOR {args.floor}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
