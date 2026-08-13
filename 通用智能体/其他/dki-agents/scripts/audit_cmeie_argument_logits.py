"""Aggregate-only logit audit for a CMeIE argument-NER checkpoint.

The audit is deliberately limited to the frozen ``fit`` or ``calibration``
shard.  It never reads ``outer`` or official dev data and never persists text,
gold mentions, logits, or decoded predictions.  Its purpose is to determine
whether a previously rejected checkpoint merely missed a viable negative
global threshold, without silently turning the diagnostic grid into a formal
selection result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALLOWED_SPLITS = ("fit", "calibration")
DIAGNOSTIC_THRESHOLDS = (
    -8.0,
    -6.0,
    -5.0,
    -4.0,
    -3.0,
    -2.5,
    -2.0,
    -1.5,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
)
HISTOGRAM_EDGES = (
    -16.0,
    -12.0,
    -10.0,
    -8.0,
    -6.0,
    -5.0,
    -4.0,
    -3.0,
    -2.0,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    12.0,
    16.0,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_threshold_grid(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(
            sorted(
                {
                    float(value.strip())
                    for value in str(raw).split(",")
                    if value.strip()
                }
            )
        )
    except ValueError as error:
        raise ValueError("threshold grid must contain numbers") from error
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("threshold grid must contain finite values")
    return values


def publish_once(path: Path, payload: dict[str, Any]) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"CMeIE argument logit audit exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _device(torch, requested: str):
    if requested == "npu":
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise RuntimeError("NPU argument audit requested but unavailable")
        if int(torch.npu.device_count()) != 1 or int(torch.npu.current_device()) != 0:
            raise RuntimeError("argument audit requires one visible logical NPU 0")
        return torch.device("npu:0")
    return torch.device("cpu")


def _valid_span_mask(offsets, torch):
    """Return the exact token-pair mask accepted by ``decode_mentions``."""
    valid = (offsets[:, 1] > offsets[:, 0])
    return valid[:, None] & valid[None, :] & torch.ones(
        (len(offsets), len(offsets)), dtype=torch.bool
    ).triu()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-generator", type=Path, required=True)
    parser.add_argument("--auditor-source-tree-sha256", required=True)
    parser.add_argument("--split", choices=ALLOWED_SPLITS, default="calibration")
    parser.add_argument("--device", choices=["cpu", "npu"], default="npu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--threshold-grid", default=",".join(map(str, DIAGNOSTIC_THRESHOLDS)))
    args = parser.parse_args()
    if os.path.lexists(args.output):
        raise SystemExit(f"CMeIE argument audit output exists: {args.output}")
    if not re.fullmatch(
        r"[0-9a-fA-F]{64}", args.auditor_source_tree_sha256
    ):
        raise SystemExit(
            "--auditor-source-tree-sha256 must be a full SHA-256"
        )
    if args.batch_size < 1 or args.max_length < 1:
        raise SystemExit("batch size and max length must be positive")
    try:
        threshold_grid = parse_threshold_grid(args.threshold_grid)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    from dki.nlp.cmeie_split_protocol import load_and_validate_role_shard

    manifest_preview = json.loads(
        args.split_manifest.read_text(encoding="utf-8")
    )
    manifest, rows, shard_binding = load_and_validate_role_shard(
        args.split_manifest,
        args.shard,
        role=args.split,
        expected_source_tree_sha256=str(
            manifest_preview["source_tree_sha256"]
        ),
        expected_source_tree_manifest_sha256=str(
            manifest_preview["source_tree_manifest_sha256"]
        ),
        generator_path=args.split_generator,
    )
    del manifest

    import torch
    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_argument import (
        ARGUMENT_BILSTM_WEIGHTS,
        ARGUMENT_HEAD_WEIGHTS,
        build_labels,
        load_argument_model_contract,
    )
    from dki.nlp.globalpointer import EncoderGlobalPointer

    contract = load_argument_model_contract(args.checkpoint)
    labels = list(contract["labels"])
    type_to_id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    encoder = AutoModel.from_pretrained(args.checkpoint)
    device = _device(torch, args.device)
    model = EncoderGlobalPointer(
        encoder,
        int(encoder.config.hidden_size),
        len(labels),
        inner_dim=int(contract["inner_dim"]),
        pointer=str(contract["pointer_variant"]),
        use_bilstm=bool(contract.get("use_bilstm", False)),
    ).to(device)
    head_state = torch.load(
        args.checkpoint / ARGUMENT_HEAD_WEIGHTS,
        map_location="cpu",
        weights_only=True,
    )
    model.gp.load_state_dict(head_state)
    if model.bilstm is not None:
        bilstm_state = torch.load(
            args.checkpoint / ARGUMENT_BILSTM_WEIGHTS,
            map_location="cpu",
            weights_only=True,
        )
        model.bilstm.load_state_dict(bilstm_state)
    model.eval()

    threshold_counts = {threshold: 0 for threshold in threshold_grid}
    gold_above = {threshold: 0 for threshold in threshold_grid}
    histogram = [0 for _ in range(len(HISTOGRAM_EDGES) + 1)]
    gold_histogram = [0 for _ in range(len(HISTOGRAM_EDGES) + 1)]
    valid_logit_count = 0
    gold_logit_count = 0
    dropped_label_count = 0
    logit_minimum = math.inf
    logit_maximum = -math.inf
    gold_minimum = math.inf
    gold_maximum = -math.inf

    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer(
                [row["text"] for row in batch],
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")
            inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "npu",
            ):
                logits = model(**inputs).float().cpu()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite argument audit logits")

            for index, row in enumerate(batch):
                row_offsets = [tuple(pair) for pair in offsets[index].tolist()]
                dense, dropped = build_labels(row, row_offsets, type_to_id)
                dropped_label_count += dropped
                pair_mask = _valid_span_mask(offsets[index], torch)
                valid_scores = logits[index][:, pair_mask].reshape(-1)
                positive_scores = logits[index][dense.bool()]
                valid_logit_count += int(valid_scores.numel())
                gold_logit_count += int(positive_scores.numel())
                if valid_scores.numel():
                    logit_minimum = min(logit_minimum, float(valid_scores.min()))
                    logit_maximum = max(logit_maximum, float(valid_scores.max()))
                    bins = torch.bucketize(
                        valid_scores,
                        torch.tensor(HISTOGRAM_EDGES, dtype=valid_scores.dtype),
                    )
                    one_histogram = torch.bincount(
                        bins,
                        minlength=len(HISTOGRAM_EDGES) + 1,
                    )
                    histogram = [
                        total + int(value)
                        for total, value in zip(histogram, one_histogram.tolist())
                    ]
                if positive_scores.numel():
                    gold_minimum = min(gold_minimum, float(positive_scores.min()))
                    gold_maximum = max(gold_maximum, float(positive_scores.max()))
                    bins = torch.bucketize(
                        positive_scores,
                        torch.tensor(HISTOGRAM_EDGES, dtype=positive_scores.dtype),
                    )
                    one_histogram = torch.bincount(
                        bins,
                        minlength=len(HISTOGRAM_EDGES) + 1,
                    )
                    gold_histogram = [
                        total + int(value)
                        for total, value in zip(gold_histogram, one_histogram.tolist())
                    ]
                for threshold in threshold_grid:
                    threshold_counts[threshold] += int(
                        (valid_scores > threshold).sum()
                    )
                    gold_above[threshold] += int(
                        (positive_scores > threshold).sum()
                    )

    def finite_or_none(value: float) -> float | None:
        return value if math.isfinite(value) else None

    payload = {
        "schema_version": 1,
        "kind": "cmeie_argument_checkpoint_logit_audit",
        "dataset_role": (
            "calibration_diagnostic_selection_used"
            if args.split == "calibration"
            else "fit_training_diagnostic"
        ),
        "selection_used": args.split == "calibration",
        "audited_split": args.split,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "auditor_source_tree_sha256": (
            args.auditor_source_tree_sha256.lower()
        ),
        "split_source_tree_sha256": shard_binding[
            "source_tree_sha256"
        ],
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "shard_binding": shard_binding,
        "checkpoint_path": str(args.checkpoint),
        "argument_model_sha256": sha256_file(args.checkpoint / "argument_model.json"),
        "head_weights_sha256": sha256_file(args.checkpoint / ARGUMENT_HEAD_WEIGHTS),
        "bilstm_weights_sha256": (
            sha256_file(args.checkpoint / ARGUMENT_BILSTM_WEIGHTS)
            if contract.get("use_bilstm", False)
            else None
        ),
        "rows": len(rows),
        "max_length": args.max_length,
        "labels": labels,
        "dropped_label_count": dropped_label_count,
        "valid_logit_count": valid_logit_count,
        "gold_logit_count": gold_logit_count,
        "logit_minimum": finite_or_none(logit_minimum),
        "logit_maximum": finite_or_none(logit_maximum),
        "gold_logit_minimum": finite_or_none(gold_minimum),
        "gold_logit_maximum": finite_or_none(gold_maximum),
        "histogram_edges": list(HISTOGRAM_EDGES),
        "all_valid_logit_histogram": histogram,
        "gold_logit_histogram": gold_histogram,
        "diagnostic_thresholds": list(threshold_grid),
        "candidate_counts_above_threshold": {
            str(threshold): threshold_counts[threshold]
            for threshold in threshold_grid
        },
        "gold_links_above_threshold": {
            str(threshold): {
                "count": gold_above[threshold],
                "recall_ceiling": (
                    gold_above[threshold] / gold_logit_count
                    if gold_logit_count
                    else 0.0
                ),
            }
            for threshold in threshold_grid
        },
        "threshold_policy": (
            "diagnostic only; no threshold is selected or promoted"
        ),
        "device": str(device),
        "contains_text_gold_or_predictions": False,
        "auditor_sha256": sha256_file(Path(__file__)),
        "argument_module_sha256": sha256_file(
            ROOT / "dki" / "nlp" / "cmeie_argument.py"
        ),
        "globalpointer_module_sha256": sha256_file(
            ROOT / "dki" / "nlp" / "globalpointer.py"
        ),
    }
    publish_once(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(rows),
                "valid_logit_count": valid_logit_count,
                "gold_logit_count": gold_logit_count,
                "logit_range": [
                    finite_or_none(logit_minimum),
                    finite_or_none(logit_maximum),
                ],
                "gold_logit_range": [
                    finite_or_none(gold_minimum),
                    finite_or_none(gold_maximum),
                ],
                "candidate_counts_above_threshold": payload[
                    "candidate_counts_above_threshold"
                ],
                "gold_links_above_threshold": payload[
                    "gold_links_above_threshold"
                ],
                "outer_accessed": False,
                "official_dev_accessed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
