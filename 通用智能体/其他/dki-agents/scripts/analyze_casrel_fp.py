"""Analyze CasRel FP/FN distribution on calibration data (diagnostic only)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from dki.nlp.cmeie_casrel import CMeIECasRel, decode_casrel, pair_boundary_logits


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _gold(row: dict) -> set[tuple[str, str, str]]:
    return {(str(s.get("subject", "")), str(s.get("predicate", "")), str(s.get("object", ""))) for s in row.get("spo_list", [])}


def _valid_subjects(spans, offsets):
    return [s for s in spans if offsets[s[0]][1] > offsets[s[0]][0] and offsets[s[1]][1] > offsets[s[1]][0]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--device", default="npu", choices=("npu", "cpu"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--subject-threshold", type=float, default=0.9)
    parser.add_argument("--object-threshold", type=float, default=-0.1)
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    fit = [r for r in _jsonl(args.fit_data) if r.get("spo_list")]
    predicates = sorted({str(s.get("predicate", "")) for r in fit for s in r.get("spo_list", []) if s.get("predicate")})
    rows = _jsonl(args.calibration_data)

    device = torch.device("npu:0" if args.device == "npu" and torch.npu.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    encoder = AutoModel.from_pretrained(args.base_model)
    model = CMeIECasRel(encoder, int(encoder.config.hidden_size), len(predicates)).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    st, ot = args.subject_threshold, args.object_threshold
    fp_pred, fn_pred, tp_pred = Counter(), Counter(), Counter()
    fp_subject_type = Counter()
    gold_pred = Counter()
    n_pred, n_fp, n_fn, n_tp = 0, 0, 0, 0
    examples = []

    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer(
                [r["text"] for r in batch], return_offsets_mapping=True, return_tensors="pt",
                padding=True, truncation=True, max_length=320,
            )
            offsets = [[tuple(p) for p in m.tolist()] for m in encoded.pop("offset_mapping")]
            inputs = {k: v.to(device) for k, v in encoded.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                hidden = model.encode(**inputs)
                subject_logits = model.subject_logits(hidden)
            by_row, parents, flat = [], [], []
            for index in range(len(batch)):
                spans = _valid_subjects(
                    pair_boundary_logits(subject_logits[index, :, 0].float().cpu(), subject_logits[index, :, 1].float().cpu(), st),
                    offsets[index],
                )
                by_row.append(spans)
                parents.extend([index] * len(spans))
                flat.extend((i[0], i[1]) for i in spans)
            object_values = None
            if flat:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                    object_values = model.object_logits(
                        hidden,
                        torch.tensor(flat, dtype=torch.long, device=device),
                        torch.tensor(parents, dtype=torch.long, device=device),
                    ).float().cpu()
            cursor = 0
            for index, row in enumerate(batch):
                spans = by_row[index]
                gold = _gold(row)
                for (s, p, o) in gold:
                    gold_pred[p] += 1
                predicted = set()
                if spans and object_values is not None:
                    predicted = decode_casrel(str(row["text"]), offsets[index], spans,
                                              object_values[cursor : cursor + len(spans)], predicates, ot)
                tp = predicted & gold
                fp = predicted - gold
                fn = gold - predicted
                n_tp += len(tp); n_fp += len(fp); n_fn += len(fn)
                for (s, p, o) in tp: tp_pred[p] += 1
                for (s, p, o) in fp:
                    fp_pred[p] += 1
                    if len(examples) < 120:
                        examples.append({"type": "FP", "predicate": p, "subject": s[:20], "object": o[:20], "text": row["text"][:60]})
                for (s, p, o) in fn:
                    fn_pred[p] += 1
                    if len(examples) < 120:
                        examples.append({"type": "FN", "predicate": p, "subject": s[:20], "object": o[:20], "text": row["text"][:60]})
                cursor += len(spans)

    p = n_tp / max(1, n_tp + n_fp)
    r = n_tp / max(1, n_tp + n_fn)
    f1 = 2 * n_tp / max(1, 2 * n_tp + n_fp + n_fn)
    top_fp = fp_pred.most_common(12)
    top_fn = fn_pred.most_common(12)
    active = sorted(tp_pred.keys())
    print(json.dumps({
        "st": st, "ot": ot,
        "metric": {"f1": f1, "precision": p, "recall": r, "tp": n_tp, "fp": n_fp, "fn": n_fn},
        "gold_triples": sum(gold_pred.values()),
        "active_predicates_tp": len(active),
        "top_fp_predicates": [[k, v] for k, v in top_fp],
        "top_fn_predicates": [[k, v] for k, v in top_fn],
        "fp_per_gold_ratio": round(n_fp / max(1, sum(gold_pred.values())), 3),
        "examples": examples,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
