"""Reproduce the CasRel calibration grid for a saved checkpoint (diagnostic only)."""
from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--subject-thresholds", type=str, default="-0.5,0.0,0.5")
    parser.add_argument("--object-thresholds", type=str, default="-0.5,0.0,0.5")
    parser.add_argument("--fine-grid", action="store_true")
    args = parser.parse_args()

    if args.fine_grid:
        # Subject threshold drives an object_logits forward pass per value;
        # object thresholds only re-decode cached logits, so scan object finely.
        args.subject_thresholds = "-0.7,-0.3,0.1,0.5,0.9"
        args.object_thresholds = "-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5"

    from transformers import AutoModel, AutoTokenizer

    fit = [r for r in _jsonl(args.fit_data) if r.get("spo_list")]
    predicates = sorted({str(s.get("predicate", "")) for r in fit for s in r.get("spo_list", []) if s.get("predicate")})
    assert len(predicates) == 44, f"expected 44 predicates, got {len(predicates)}"
    rows = _jsonl(args.calibration_data)

    device = torch.device("npu:0" if args.device == "npu" and torch.npu.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    encoder = AutoModel.from_pretrained(args.base_model)
    model = CMeIECasRel(encoder, int(encoder.config.hidden_size), len(predicates)).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    # The 0.322919 baseline checkpoint predates span_verifier; calibration uses
    # decode_casrel (no verifier), so missing verifier keys are tolerated.
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    thresholds = tuple(float(x) for x in args.subject_thresholds.split(","))
    object_thresholds = tuple(float(x) for x in args.object_thresholds.split(","))
    counts = {(s, o): {"tp": 0, "fp": 0, "fn": 0} for s in thresholds for o in object_thresholds}
    subject_density = {str(t): 0 for t in thresholds}

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
            for st in thresholds:
                by_row, parents, flat = [], [], []
                for index in range(len(batch)):
                    spans = _valid_subjects(
                        pair_boundary_logits(subject_logits[index, :, 0].float().cpu(), subject_logits[index, :, 1].float().cpu(), st),
                        offsets[index],
                    )
                    by_row.append(spans)
                    subject_density[str(st)] += len(spans)
                    parents.extend([index] * len(spans))
                    flat.extend((i[0], i[1]) for i in spans)
                object_values = None
                if flat:
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.fp16):
                        object_values = model.object_logits(
                            hidden,
                            torch.tensor(flat, dtype=torch.long, device=device),
                            torch.tensor(parents, dtype=torch.long, device=device),
                        ).float().cpu()
                cursor = 0
                for index, row in enumerate(batch):
                    spans = by_row[index]
                    gold = _gold(row)
                    for ot in object_thresholds:
                        predicted = set()
                        if spans and object_values is not None:
                            predicted = decode_casrel(str(row["text"]), offsets[index], spans,
                                                      object_values[cursor : cursor + len(spans)], predicates, ot)
                        c = counts[(st, ot)]
                        c["tp"] += len(predicted & gold)
                        c["fp"] += len(predicted - gold)
                        c["fn"] += len(gold - predicted)
                    cursor += len(spans)

    scored = []
    for (st, ot), c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        scored.append({"f1": f1, "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn,
                       "subject_threshold": st, "object_threshold": ot})
    best = max(scored, key=lambda i: (i["f1"], i["precision"], i["subject_threshold"] + i["object_threshold"]))
    print(json.dumps({"best": best, "grid": scored, "subject_density": subject_density}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
