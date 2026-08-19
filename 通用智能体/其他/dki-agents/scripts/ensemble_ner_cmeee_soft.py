"""Soft-logit CMeEE ensemble for same-architecture GlobalPointer checkpoints.

Averages member logits on a shared token grid, selects decode threshold on the
train-derived internal split, then reports official dev once.
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
    _encode,
    _load,
    _metric_record,
    _split_train_internal,
)
from scripts.ensemble_ner_cmeee import (  # noqa: E402
    _apply_lexicon_boost,
    _build_train_lexicon,
    _fuse,
    _load_bundle,
    _predict_char_sets,
    _score_sets,
)


def _logits_to_char_sets(logits, offsets_batch, threshold, decode_spans):
    decoded = decode_spans(logits, threshold=threshold)
    out = []
    for local, spans in enumerate(decoded):
        offsets = offsets_batch[local]
        predicted = set()
        for head, token_start, token_end in spans:
            char_start = int(offsets[token_start][0])
            char_end = int(offsets[token_end][1])
            if char_end > char_start:
                predicted.add((TYPES[head], char_start, char_end))
        out.append(predicted)
    return out


def _collect_logits(bundles, rows, weights):
    """Return list of (n_samples) CPU logit tensors averaged with weights, plus char gold."""
    import torch

    # Shared encode from first classic member.
    ref = bundles[0]
    enc, _, char_gold = _encode(ref["tokenizer"], rows, ref["max_len"])
    ids = torch.tensor(enc["input_ids"])
    mask = torch.tensor(enc["attention_mask"])
    token_types = torch.tensor(
        enc.get("token_type_ids", ref["np"].zeros_like(enc["input_ids"]))
    )
    batch_size = ref["batch_size"]
    device = next(ref["model"].parameters()).device
    wsum = float(sum(weights))
    weights = [float(w) / wsum for w in weights]

    # Precompute softlex features per bundle (fit-only surfaces baked into ckpt).
    softlex_by_bundle = []
    for bundle in bundles:
        if bundle.get("softlex_dim", 0) > 0 and bundle.get("softlex_lexicon") is not None:
            softlex_by_bundle.append(torch.tensor(bundle["encode_softlex_batch"](
                rows, enc["offset_mapping"], bundle["softlex_lexicon"], len(TYPES), bundle["max_len"],
            )))
        else:
            softlex_by_bundle.append(None)

    avg_chunks = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            stop = min(len(rows), start + batch_size)
            acc = None
            for bundle, weight, softlex_t in zip(bundles, weights, softlex_by_bundle):
                model = bundle["model"]
                # Re-encode if tokenizer/max_len differ (should not for classic group).
                if bundle["max_len"] != ref["max_len"] or bundle["pointer"] != ref["pointer"]:
                    raise RuntimeError("soft ensemble requires identical pointer/max_len")
                soft_kw = {}
                if softlex_t is not None:
                    soft_kw["softlex"] = softlex_t[start:stop].to(device)
                logits = model(
                    ids[start:stop].to(device),
                    mask[start:stop].to(device),
                    token_types[start:stop].to(device),
                    **soft_kw,
                ).float().cpu()
                acc = logits * weight if acc is None else acc + logits * weight
            avg_chunks.append(acc)
    avg = torch.cat(avg_chunks, dim=0)
    offsets = [offsets for _, offsets in char_gold]
    gold_sets = [set(chars) for chars, _ in char_gold]
    return avg, offsets, gold_sets, ref["decode_spans"]


def _select_soft_policy(avg_logits, offsets, gold_sets, decode_spans, rows, lexicon):
    thresholds = sorted(set(DEFAULT_THRESHOLDS))
    best = None
    for thr in thresholds:
        for lex_mode in ("off", "recall_lex"):
            pred = _logits_to_char_sets(avg_logits, offsets, thr, decode_spans)
            pred = _apply_lexicon_boost(pred, rows, lexicon, lex_mode)
            rec = _score_sets(pred, gold_sets)
            item = {
                "threshold": thr,
                "lex_mode": lex_mode,
                "internal_micro_f1": rec["micro_f1"],
                "record": rec,
            }
            print(
                f"[soft internal] thr={thr:+.1f} lex={lex_mode} F1={rec['micro_f1']:.4f}",
                flush=True,
            )
            if best is None or rec["micro_f1"] > best["internal_micro_f1"]:
                best = item
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Soft-logit CMeEE ensemble")
    parser.add_argument("--soft-models", nargs="+", type=Path, required=True,
                        help="same-architecture classic GP checkpoints for logit average")
    parser.add_argument("--hard-models", nargs="*", type=Path, default=[],
                        help="optional extra members for hard vote with soft result")
    parser.add_argument("--device", default="npu", choices=["npu", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--internal-val-ratio", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floor", type=float, default=0.6015)
    parser.add_argument("--min-member-internal-f1", type=float, default=0.50)
    parser.add_argument("--weight-by-internal", action="store_true",
                        help="weight soft average by solo internal F1")
    args = parser.parse_args()

    import torch
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    device = torch.device(args.device if args.device != "npu" else "npu")

    soft_bundles = [_load_bundle(p, device, args.batch_size) for p in args.soft_models]
    hard_bundles = [_load_bundle(p, device, args.batch_size) for p in args.hard_models]

    all_train = _load(DATA / "train.jsonl")
    fit_rows, internal_rows = _split_train_internal(
        all_train, args.internal_val_ratio, args.split_seed,
    )
    lexicon = _build_train_lexicon(fit_rows)

    # Filter soft members by solo internal F1; collect weights.
    kept_soft, weights = [], []
    for bundle in soft_bundles:
        pred, gold = _predict_char_sets(bundle, internal_rows)
        solo = _score_sets(pred, gold)
        print(f"[soft solo] {bundle['dir'].name} F1={solo['micro_f1']:.4f}", flush=True)
        if solo["micro_f1"] < args.min_member_internal_f1:
            print(f"[drop soft] {bundle['dir'].name}", flush=True)
            continue
        kept_soft.append(bundle)
        weights.append(solo["micro_f1"] if args.weight_by_internal else 1.0)
    if len(kept_soft) < 1:
        raise RuntimeError("no soft members survived filter")
    if not args.weight_by_internal:
        weights = [1.0] * len(kept_soft)

    avg_int, offsets_int, gold_int, decode_spans = _collect_logits(
        kept_soft, internal_rows, weights,
    )
    soft_policy = _select_soft_policy(
        avg_int, offsets_int, gold_int, decode_spans, internal_rows, lexicon,
    )
    print(
        f"[selected soft] thr={soft_policy['threshold']:+.1f} "
        f"lex={soft_policy['lex_mode']} F1={soft_policy['internal_micro_f1']:.4f}",
        flush=True,
    )

    soft_pred_int = _logits_to_char_sets(
        avg_int, offsets_int, soft_policy["threshold"], decode_spans,
    )
    soft_pred_int = _apply_lexicon_boost(
        soft_pred_int, internal_rows, lexicon, soft_policy["lex_mode"],
    )

    # Optional hard members + soft as one vote member.
    hard_pred_int = [soft_pred_int]
    kept_hard_bundles = []
    for bundle in hard_bundles:
        pred, _ = _predict_char_sets(bundle, internal_rows)
        solo = _score_sets(pred, gold_int)
        print(f"[hard solo] {bundle['dir'].name} F1={solo['micro_f1']:.4f}", flush=True)
        if solo["micro_f1"] < args.min_member_internal_f1:
            print(f"[drop hard] {bundle['dir'].name}", flush=True)
            continue
        hard_pred_int.append(pred)
        kept_hard_bundles.append(bundle)

    best_hard = {
        "mode": "soft_only", "k": 1,
        "internal_micro_f1": soft_policy["internal_micro_f1"],
    }
    n = len(hard_pred_int)
    if n >= 2:
        for mode, k in [("union", 1), ("intersect", n)] + [("vote", k) for k in range(2, n + 1)]:
            fused = _fuse(hard_pred_int, mode if mode != "vote" else "vote", k=k)
            rec = _score_sets(fused, gold_int)
            print(
                f"[hard+soft internal] mode={mode} k={k} F1={rec['micro_f1']:.4f}",
                flush=True,
            )
            if rec["micro_f1"] > best_hard["internal_micro_f1"]:
                best_hard = {
                    "mode": mode, "k": k,
                    "internal_micro_f1": rec["micro_f1"],
                }

    print(f"[selected policy] soft+{best_hard}", flush=True)

    # Official dev once.
    dev_rows = _load(DATA / "dev.jsonl")
    avg_dev, offsets_dev, gold_dev, decode_spans = _collect_logits(
        kept_soft, dev_rows, weights,
    )
    soft_pred_dev = _logits_to_char_sets(
        avg_dev, offsets_dev, soft_policy["threshold"], decode_spans,
    )
    soft_pred_dev = _apply_lexicon_boost(
        soft_pred_dev, dev_rows, lexicon, soft_policy["lex_mode"],
    )

    if best_hard["mode"] == "soft_only" or not kept_hard_bundles:
        final_pred = soft_pred_dev
    else:
        member_dev = [soft_pred_dev]
        for bundle in kept_hard_bundles:
            pred, _ = _predict_char_sets(bundle, dev_rows)
            member_dev.append(pred)
        final_pred = _fuse(
            member_dev,
            best_hard["mode"] if best_hard["mode"] != "vote" else "vote",
            k=best_hard["k"],
        )

    official = _score_sets(final_pred, gold_dev)
    print(
        f"[official_dev] F1={official['micro_f1']:.4f} "
        f"P={official['precision']:.4f} R={official['recall']:.4f}",
        flush=True,
    )

    thr = float(soft_policy["threshold"])
    payload = {
        "schema_version": 2,
        "benchmark": "CBLUE CMeEE-V2",
        "dataset_role": "development_locked_report",
        "eval_split": "official_dev",
        "n_dev": len(dev_rows),
        "decoder": "SoftLogitEnsembleGlobalPointer",
        "primary_operating_point": "internally_selected_threshold",
        "selected_phase": "soft_ensemble",
        "decode_threshold": thr,
        "selection_protocol": {
            "checkpoint": "train-derived internal validation soft-logit average",
            "threshold": "train-derived internal validation only",
            "soft_threshold": "train-derived internal validation only",
            "hard_fusion": "train-derived internal validation only",
            "official_dev_used_for_selection": False,
            "text_group_disjoint": True,
        },
        "soft_members": [str(b["dir"]) for b in kept_soft],
        "soft_weights": weights,
        "hard_members": [str(b["dir"]) for b in kept_hard_bundles],
        "selected_policy": {
            "soft_threshold": thr,
            "soft_lex_mode": soft_policy["lex_mode"],
            "soft_internal_micro_f1": round(soft_policy["internal_micro_f1"], 4),
            "hard": best_hard,
        },
        "internally_selected_threshold": {
            "threshold": thr,
            "internal_micro_f1": round(soft_policy["internal_micro_f1"], 4),
            "official_report": {
                "threshold": thr,
                "micro_f1": round(official["micro_f1"], 4),
                "precision": round(official["precision"], 4),
                "recall": round(official["recall"], 4),
                "tp": official["tp"], "fp": official["fp"], "fn": official["fn"],
                "per_type": official.get("per_type"),
                "worst_type": official.get("worst_type"),
                "selection_used": True,
            },
        },
        "micro_f1": round(official["micro_f1"], 4),
        "precision": round(official["precision"], 4),
        "recall": round(official["recall"], 4),
        "tp": official["tp"], "fp": official["fp"], "fn": official["fn"],
        "per_type": official.get("per_type"),
        "worst_type": official.get("worst_type"),
        "floor": args.floor,
        "promoted": official["micro_f1"] > args.floor + 1e-9,
        "evidence_triad": {
            "synthetic_pipeline_correctness": "outputs/eval/kg_extraction_eval.json",
            "finetune_same_distribution": "outputs/ner_finetune/metrics.json",
            "cblue_authoritative": "outputs/cblue/cmeee_metrics.json",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output} promoted={payload['promoted']}", flush=True)
    if payload["promoted"]:
        metrics_path = ROOT / "outputs/cblue/cmeee_metrics.json"
        metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"PROMOTED_CMEEE_SOFT F1={payload['micro_f1']}", flush=True)
    else:
        print(f"KEEP_FLOOR {args.floor}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
