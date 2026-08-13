"""Evaluate full text-to-SPO with verifier-gated decoding (clean per-row design).

- Store per-row hidden (own length) and subject logits in a first pass.
- For each (subject_threshold, object_threshold): build candidates via
  predicate_object_candidates (4D object logits), score them with the trained
  span_verifier (aligned per candidate), then decode with verifier_threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dki.nlp.cmeie_casrel import (
    CMeIECasRel,
    decode_verified_casrel,
    pair_boundary_logits,
    predicate_object_candidates,
)


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
    parser.add_argument("--preset", type=str, default="verified", help="baseline|verified")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    if args.preset == "verified":
        subject_thresholds = (0.5, 0.7, 0.9)
        object_thresholds = (-0.1, 0.0, 0.1)
        verifier_thresholds = (-1.0, -0.5, 0.0, 0.5, 1.0)
    else:  # baseline reproduction: no verifier gating
        subject_thresholds = (0.5,)
        object_thresholds = (0.0,)
        verifier_thresholds = (-100.0,)

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
    has_verifier = any("span_verifier" in k for k in state["model"])
    model.eval()
    print(f"[info] checkpoint has trained span_verifier: {has_verifier}", flush=True)

    # Pass 1: per-row hidden + subject logits + offsets
    rows_data = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer(
                [r["text"] for r in batch], return_offsets_mapping=True, return_tensors="pt",
                padding=True, truncation=True, max_length=320,
            )
            offsets = [[tuple(p) for p in m.tolist()] for m in encoded.pop("offset_mapping")]
            inputs = {k: v.to(device) for k, v in encoded.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16):
                hidden = model.encode(**inputs)
                subject_logits = model.subject_logits(hidden)
            for i in range(len(batch)):
                rows_data.append({
                    "text": str(batch[i]["text"]),
                    "offsets": offsets[i],
                    "gold": _gold(batch[i]),
                    "hidden": hidden[i : i + 1].float().cpu(),
                    "subject_logits": subject_logits[i : i + 1].float().cpu(),
                })

    # Pass 2: for each (st, ot): object logits per row (st-dependent), candidates (ot), verifier scores
    # Cache object logits per subject threshold once.
    object_cache = {}
    with torch.no_grad():
        for st in subject_thresholds:
            per_row_obj = []
            per_row_spans = []
            for data in rows_data:
                spans = _valid_subjects(
                    pair_boundary_logits(data["subject_logits"][0, :, 0], data["subject_logits"][0, :, 1], st),
                    data["offsets"],
                )
                per_row_spans.append(spans)
                if spans:
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16):
                        obj = model.object_logits(
                            data["hidden"].to(device),
                            torch.tensor([(s[0], s[1]) for s in spans], dtype=torch.long, device=device),
                            torch.zeros(len(spans), dtype=torch.long, device=device),
                        ).float().cpu()
                else:
                    obj = None
                per_row_obj.append(obj)
            object_cache[st] = (per_row_spans, per_row_obj)

    # Candidate + verifier score cache per (st, ot)
    cand_cache = {}
    if has_verifier:
        with torch.no_grad():
            for st in subject_thresholds:
                per_row_spans, per_row_obj = object_cache[st]
                for ot in object_thresholds:
                    scores_by_row = []
                    for ri, data in enumerate(rows_data):
                        spans = per_row_spans[ri]
                        obj = per_row_obj[ri]
                        if not spans or obj is None:
                            scores_by_row.append(None)
                            continue
                        cands = predicate_object_candidates(
                            obj, data["offsets"], ot,
                            maximum_per_subject_predicate=32, maximum_per_row=8192,
                        )
                        if not cands:
                            scores_by_row.append(None)
                            continue
                        cs = torch.tensor([(spans[c[0]][0], spans[c[0]][1]) for c in cands], dtype=torch.long, device=device)
                        co = torch.tensor([(c[2], c[3]) for c in cands], dtype=torch.long, device=device)
                        cp = torch.tensor([c[1] for c in cands], dtype=torch.long, device=device)
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16):
                            ver = model.score_object_spans(
                                data["hidden"].to(device), cs, co, cp,
                                torch.zeros(len(cands), dtype=torch.long, device=device),
                            ).float().cpu()
                        scores_by_row.append((cands, ver))
                    cand_cache[(st, ot)] = scores_by_row

    scored = []
    for st in subject_thresholds:
        per_row_spans, per_row_obj = object_cache[st]
        for ot in object_thresholds:
            for vt in verifier_thresholds:
                tp = fp = fn = 0
                for ri, data in enumerate(rows_data):
                    spans = per_row_spans[ri]
                    obj = per_row_obj[ri]
                    gold = data["gold"]
                    predicted = set()
                    if spans and obj is not None:
                        if has_verifier:
                            entry = cand_cache[(st, ot)][ri]
                            if entry is not None:
                                cands, ver = entry
                                predicted = decode_verified_casrel(
                                    data["text"], data["offsets"], spans, obj, predicates, ot,
                                    ver, vt,
                                    maximum_per_subject_predicate=32, maximum_per_row=8192,
                                )
                        else:
                            from dki.nlp.cmeie_casrel import decode_casrel
                            predicted = decode_casrel(data["text"], data["offsets"], spans, obj, predicates, ot)
                    tp += len(predicted & gold)
                    fp += len(predicted - gold)
                    fn += len(gold - predicted)
                p = tp / max(1, tp + fp)
                r = tp / max(1, tp + fn)
                f1 = 2 * tp / max(1, 2 * tp + fp + fn)
                scored.append({"f1": f1, "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn,
                               "subject_threshold": st, "object_threshold": ot, "verifier_threshold": vt})
    best = max(scored, key=lambda i: (i["f1"], i["precision"]))
    top = sorted(scored, key=lambda i: -i["f1"])[:12]
    print(json.dumps({"has_verifier": has_verifier, "preset": args.preset, "best": best,
                      "top": top, "n_grid": len(scored)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
