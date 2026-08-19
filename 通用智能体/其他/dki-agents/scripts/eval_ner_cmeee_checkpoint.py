"""Rescore a frozen CMeEE GlobalPointer checkpoint without retraining.

This is the first remote action for the promoted max_len=256 candidate: it adds
per-type, macro and worst-type metrics while keeping official dev evaluation-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_ner_cmeee import DATA, TYPES, _encode, _load, _metric_record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_manifest(model_dir: Path) -> list[dict]:
    return [
        {
            "path": path.relative_to(model_dir).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(model_dir.rglob("*"))
        if path.is_file()
    ]


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
    parser = argparse.ArgumentParser(description="Rescore a frozen CMeEE checkpoint")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu", choices=["npu", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--limit-dev", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.limit_dev < 0:
        parser.error("batch-size must be positive and limit-dev non-negative")

    required = [
        args.model / "globalpointer.pt",
        args.model / "globalpointer_config.json",
        args.model / "types.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error(f"checkpoint is incomplete: {missing}")

    import numpy as np
    import torch
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    from transformers import AutoModel, AutoTokenizer
    from dki.nlp.globalpointer import EncoderGlobalPointer, decode_spans

    config = json.loads((args.model / "globalpointer_config.json").read_text(encoding="utf-8"))
    types = json.loads((args.model / "types.json").read_text(encoding="utf-8"))
    if types != TYPES:
        parser.error(f"checkpoint types do not match CMeEE-V2 contract: {types!r}")
    threshold = float(config["selected_threshold"] if args.threshold is None else args.threshold)
    max_len = int(config["max_len"])
    inner_dim = int(config["inner_dim"])
    device = torch.device(args.device if args.device != "npu" else "npu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model)
    pointer = str(config.get("pointer", "classic"))
    model = EncoderGlobalPointer(
        encoder, encoder.config.hidden_size, len(types), inner_dim, pointer=pointer,
    )
    try:
        state = torch.load(args.model / "globalpointer.pt", map_location="cpu", weights_only=True)
    except TypeError:  # older torch on some CANN images
        state = torch.load(args.model / "globalpointer.pt", map_location="cpu")
    model.gp.load_state_dict(state)
    model = model.to(device).eval()

    rows = _load(DATA / "dev.jsonl")
    if args.limit_dev:
        rows = rows[:args.limit_dev]
    encoded, _, char_gold = _encode(tokenizer, rows, max_len)
    ids = torch.tensor(encoded["input_ids"])
    mask = torch.tensor(encoded["attention_mask"])
    token_types = torch.tensor(encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"])))
    counts = [0, 0, 0]
    per_type_counts = {entity_type: [0, 0, 0] for entity_type in types}

    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            stop = min(len(rows), start + args.batch_size)
            logits = model(
                ids[start:stop].to(device),
                mask[start:stop].to(device),
                token_types[start:stop].to(device),
            ).float().cpu()
            predictions = decode_spans(logits, threshold=threshold)
            for local_index, spans in enumerate(predictions):
                gold_chars, offsets = char_gold[start + local_index]
                gold_set = set(gold_chars)
                predicted_set = set()
                for head, token_start, token_end in spans:
                    char_start = int(offsets[token_start][0])
                    char_end = int(offsets[token_end][1])
                    if char_end > char_start:
                        predicted_set.add((types[head], char_start, char_end))
                counts[0] += len(gold_set & predicted_set)
                counts[1] += len(predicted_set - gold_set)
                counts[2] += len(gold_set - predicted_set)
                for entity_type in types:
                    gold_type = {item for item in gold_set if item[0] == entity_type}
                    predicted_type = {item for item in predicted_set if item[0] == entity_type}
                    target = per_type_counts[entity_type]
                    target[0] += len(gold_type & predicted_type)
                    target[1] += len(predicted_type - gold_type)
                    target[2] += len(gold_type - predicted_type)

    metric = _metric_record(
        *counts, threshold, per_type_counts=per_type_counts,
    )
    payload = {
        "schema_version": 1,
        "benchmark": "CBLUE CMeEE-V2 frozen-checkpoint rescore",
        "dataset_role": (
            "diagnostic_limited_official_dev" if args.limit_dev
            else "development_locked_report"
        ),
        "eval_split": "official_dev",
        "selection_used": False,
        "evaluation_scope": "exact_span_type",
        "n_dev": len(rows),
        "model_dir": str(args.model),
        "model_artifacts": _model_manifest(args.model),
        "max_len": max_len,
        "threshold": threshold,
        **metric,
    }
    _write_json_atomic(args.output, payload)
    print(json.dumps({
        key: payload.get(key) for key in
        ("n_dev", "micro_f1", "precision", "recall", "macro_f1", "worst_type")
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
