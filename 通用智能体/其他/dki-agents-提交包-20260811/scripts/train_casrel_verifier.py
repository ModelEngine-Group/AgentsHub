"""Train the CasRel span_verifier head on top of a frozen CasRel backbone.

Strategy (based on blocker ledger):
- Baseline CasRel calibration F1=0.3268 is limited by FP=4867 >> TP=2154.
- The PredicateAwareSpanVerifier exists in the model but was NEVER trained
  (casrel_loss only contains subject+object boundary losses; the baseline
  checkpoint carries no span_verifier weights).
- Freeze encoder + boundary heads, train only span_verifier with:
    positives = gold-aligned (subject, predicate, object) candidates
    negatives = predicate-confusion + boundary-offset hard negatives
- Then decode_verified_casrel with verifier_threshold prunes FP.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _gold_spans(row: dict, offsets, predicate2id):
    """Return aligned (subject_tokens, predicate_id, object_tokens) triples."""
    from dki.nlp.cmeie_casrel import build_casrel_labels

    labels = build_casrel_labels(row, offsets, predicate2id)
    result = []
    for (ss, se), values in labels.object_spans.items():
        for (pid, os_, oe) in values:
            result.append(((ss, se), pid, (os_, oe)))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu", choices=("npu", "cpu"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_casrel import CMeIECasRel

    fit_all = [r for r in _jsonl(args.fit_data) if r.get("spo_list")]
    val = [r for r in _jsonl(args.val_data) if r.get("spo_list")]
    predicates = sorted({str(s.get("predicate", "")) for r in fit_all for s in r.get("spo_list", []) if s.get("predicate")})
    assert len(predicates) == 44, f"expected 44 predicates, got {len(predicates)}"
    fit = fit_all
    if args.max_examples > 0:
        fit = fit_all[: args.max_examples]
        val = val[: max(1, args.max_examples // 4)]
    predicate2id = {p: i for i, p in enumerate(predicates)}
    p2label = {i: p for p, i in predicate2id.items()}

    device = torch.device("npu:0" if args.device == "npu" and torch.npu.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    encoder = AutoModel.from_pretrained(args.base_model)
    model = CMeIECasRel(encoder, int(encoder.config.hidden_size), len(predicates)).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)  # verifier weights absent
    model.eval()

    # Freeze everything except span_verifier
    for name, param in model.named_parameters():
        param.requires_grad = "span_verifier" in name
    verifier_params = [p for n, p in model.named_parameters() if p.requires_grad]
    assert verifier_params, "no verifier parameters to train"
    print(f"[init] trainable verifier params: {sum(p.numel() for p in verifier_params)}", flush=True)
    optimizer = torch.optim.AdamW(verifier_params, lr=args.lr, weight_decay=1e-4)

    def encode_rows(rows, batch_size):
        """Yield (inputs, candidates) where candidates = aligned gold triples."""
        from dki.nlp.cmeie_casrel import char_span_to_tokens, select_mention_pairs

        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = tokenizer(
                [r["text"] for r in batch], return_offsets_mapping=True, return_tensors="pt",
                padding=True, truncation=True, max_length=320,
            )
            offsets = [[tuple(p) for p in m.tolist()] for m in encoded.pop("offset_mapping")]
            inputs = {k: v.to(device) for k, v in encoded.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            cands = []
            for index, row in enumerate(batch):
                gold = _gold_spans(row, offsets[index], predicate2id)
                text = str(row["text"])
                for (ss, se), pid, (os_, oe) in gold:
                    cands.append({"row": index, "s": (ss, se), "p": pid, "o": (os_, oe), "label": 1.0})
                    # predicate-confusion negatives: same spans, wrong predicates
                    others = [q for q in range(len(predicates)) if q != pid]
                    for q in random.sample(others, min(args.negatives_per_positive // 2, len(others))):
                        cands.append({"row": index, "s": (ss, se), "p": q, "o": (os_, oe), "label": 0.0})
                    # boundary-offset negatives: shift object span by one token
                    for shift in (1, -1):
                        no, ne = os_ + shift, oe + shift
                        if 0 <= no <= ne < len(offsets[index]) and (no, ne) != (os_, oe):
                            cands.append({"row": index, "s": (ss, se), "p": pid, "o": (no, ne), "label": 0.0})
            yield inputs, cands, offsets

    def score_batch(model, inputs, cands, device):
        with torch.no_grad():
            hidden = model.encode(**inputs)
        s_idx = torch.tensor([c["s"] for c in cands], dtype=torch.long, device=device)
        o_idx = torch.tensor([c["o"] for c in cands], dtype=torch.long, device=device)
        p_ids = torch.tensor([c["p"] for c in cands], dtype=torch.long, device=device)
        parents = torch.tensor([c["row"] for c in cands], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16):
            logits = model.score_object_spans(hidden, s_idx, o_idx, p_ids, parents)
        return logits.float().cpu()

    for epoch in range(1, args.epochs + 1):
        random.seed(args.seed + epoch)
        random.shuffle(fit)
        model.train()
        total_loss, total_pos, total_neg = 0.0, 0, 0
        steps = 0
        for inputs, cands, offsets in encode_rows(fit, args.batch_size):
            if len(cands) < 2:
                continue
            logits = score_batch(model, inputs, cands, device)
            labels = torch.tensor([c["label"] for c in cands], dtype=torch.float32)
            pos = labels > 0.5
            neg = ~pos
            if not pos.any() or not neg.any():
                continue
            loss = F.softplus(-logits[pos]).mean() + F.softplus(logits[neg]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(verifier_params, 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            total_pos += int(pos.sum())
            total_neg += int(neg.sum())
            steps += 1
        # validation on val split
        model.eval()
        v_tp = v_fp = v_fn = 0
        v_positive = v_negative = 0
        for inputs, cands, offsets in encode_rows(val, args.batch_size):
            if len(cands) < 2:
                continue
            logits = score_batch(model, inputs, cands, device)
            labels = torch.tensor([c["label"] for c in cands], dtype=torch.float32)
            pos = labels > 0.5
            v_positive += int(pos.sum())
            v_negative += int((~pos).sum())
            # threshold sweep on verifier logits
            preds = (logits > 0.0).float()
            tp = int(((preds == 1) & pos).sum())
            fp = int(((preds == 1) & (~pos)).sum())
            fn = int(((preds == 0) & pos).sum())
            v_tp += tp; v_fp += fp; v_fn += fn
        v_p = v_tp / max(1, v_tp + v_fp)
        v_r = v_tp / max(1, v_tp + v_fn)
        v_f1 = 2 * v_tp / max(1, 2 * v_tp + v_fp + v_fn)
        print(f"[epoch {epoch}] loss={total_loss/max(1,steps):.4f} pos={total_pos} neg={total_neg} "
              f"val P={v_p:.4f} R={v_r:.4f} F1={v_f1:.4f} (tp={v_tp} fp={v_fp} fn={v_fn})", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
         "predicates": predicates, "epochs": args.epochs, "lr": args.lr},
        args.output,
    )
    print(f"[done] verifier checkpoint -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
