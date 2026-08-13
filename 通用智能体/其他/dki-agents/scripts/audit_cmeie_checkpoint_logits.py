"""Aggregate-only gold-logit audit for a staged CMeIE GPLinker checkpoint.

This diagnostic does not tune or select a threshold.  It explains whether the
predeclared calibration grid can possibly recover gold entity/head/tail links,
without storing source text, labels, logits, or decoded predictions.
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

DIAGNOSTIC_THRESHOLDS = (-5.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5)
DECLARED_GRID = (-0.5, 0.0, 0.5)
ALLOWED_SPLITS = ("fit", "calibration")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarise_scores(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "maximum": None,
        }
    ordered = sorted(float(value) for value in values)

    def percentile(probability: float) -> float:
        index = max(0, int(math.ceil(probability * len(ordered))) - 1)
        return ordered[index]

    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "p10": percentile(0.10),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "maximum": ordered[-1],
    }


def publish_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"CMeIE logit audit already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def select_split_rows(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    split: str,
    sample_limit: int,
) -> list[dict[str, Any]]:
    """Select only fit/calibration rows without creating an outer access path."""

    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"logit audit split must be one of {ALLOWED_SPLITS}, got {split!r}"
        )
    if sample_limit < 0:
        raise ValueError("sample limit must be non-negative")
    indices = manifest["splits"][split]["indices"]
    selected = [rows[index] for index in indices]
    return selected[:sample_limit] if sample_limit else selected


def _device(torch, requested: str):
    if requested == "npu":
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise RuntimeError("NPU logit audit requested but torch_npu is unavailable")
        return torch.device("npu:0")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA logit audit requested but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "npu"], default="npu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument(
        "--split",
        choices=ALLOWED_SPLITS,
        default="calibration",
        help="Only fit/calibration are admissible; outer is deliberately absent.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="0 audits the complete selected split",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"CMeIE logit audit output exists: {args.output}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.source_tree_sha256):
        raise SystemExit("--source-tree-sha256 must be a full SHA-256")
    if args.batch_size < 1 or args.sample_limit < 0:
        raise SystemExit("batch size must be positive and sample limit non-negative")
    if not (args.checkpoint / "gplinker.pt").is_file():
        raise SystemExit("checkpoint lacks gplinker.pt")

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if manifest.get("source_train_sha256") != sha256(args.train):
        raise SystemExit("train hash differs from split manifest")
    rows = [
        json.loads(line)
        for line in args.train.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = select_split_rows(
        rows,
        manifest,
        args.split,
        args.sample_limit,
    )

    import torch
    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_gplinker import CMeIEGPLinker
    from dki.nlp.cmeie_spo import build_labels

    saved = torch.load(
        args.checkpoint / "gplinker.pt",
        map_location="cpu",
        weights_only=False,
    )
    entity_labels = list(saved["entity_labels"])
    predicates = list(saved["predicates"])
    entity2id = {label: index for index, label in enumerate(entity_labels)}
    predicate2id = {label: index for index, label in enumerate(predicates)}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    encoder = AutoModel.from_pretrained(args.checkpoint)
    device = _device(torch, args.device)
    model = CMeIEGPLinker(
        encoder,
        int(encoder.config.hidden_size),
        len(entity_labels),
        len(predicates),
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    heads = ("entity", "relation_head", "relation_tail")
    positive_scores: dict[str, list[float]] = {name: [] for name in heads}
    gold_above = {
        name: {threshold: 0 for threshold in DIAGNOSTIC_THRESHOLDS}
        for name in heads
    }
    candidate_above = {
        name: {threshold: 0 for threshold in DIAGNOSTIC_THRESHOLDS}
        for name in heads
    }
    dropped = 0
    with torch.no_grad():
        for start in range(0, len(selected), args.batch_size):
            batch = selected[start : start + args.batch_size]
            encoded = tokenizer(
                [row["text"] for row in batch],
                return_offsets_mapping=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            labels = []
            for index, row in enumerate(batch):
                one, lost = build_labels(
                    row,
                    [
                        tuple(pair)
                        for pair in encoded["offset_mapping"][index].tolist()
                    ],
                    entity2id,
                    predicate2id,
                    allow_cross_sentence_fallback=True,
                )
                labels.append(one)
                dropped += lost
            encoded.pop("offset_mapping")
            inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type in {"npu", "cuda"},
            ):
                outputs = model(**inputs)
            for name in heads:
                dense_labels = torch.stack(
                    [one[name] for one in labels]
                ).to(device)
                scores = outputs[name].float()
                positives = scores[dense_labels.bool()]
                positive_scores[name].extend(positives.cpu().tolist())
                for threshold in DIAGNOSTIC_THRESHOLDS:
                    gold_above[name][threshold] += int(
                        (positives > threshold).sum().item()
                    )
                    candidate_above[name][threshold] += int(
                        (scores > threshold).sum().item()
                    )

    head_reports = {}
    for name in heads:
        total = len(positive_scores[name])
        head_reports[name] = {
            "gold_logit_summary": summarise_scores(positive_scores[name]),
            "gold_links_above_threshold": {
                str(threshold): {
                    "count": gold_above[name][threshold],
                    "recall_ceiling": (
                        gold_above[name][threshold] / total if total else 0.0
                    ),
                }
                for threshold in DIAGNOSTIC_THRESHOLDS
            },
            "all_candidates_above_threshold": {
                str(threshold): candidate_above[name][threshold]
                for threshold in DIAGNOSTIC_THRESHOLDS
            },
        }
    payload = {
        "schema_version": 1,
        "kind": "cmeie_checkpoint_gold_logit_audit",
        "dataset_role": (
            "calibration_diagnostic_selection_used"
            if args.split == "calibration"
            else "fit_training_diagnostic"
        ),
        "selection_used": args.split == "calibration",
        "audited_split": args.split,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "source_tree_sha256": args.source_tree_sha256.lower(),
        "train_sha256": sha256(args.train),
        "split_manifest_sha256": sha256(args.split_manifest),
        "checkpoint_sha256": sha256(args.checkpoint / "gplinker.pt"),
        "checkpoint_path": str(args.checkpoint),
        "rows": len(selected),
        "sample_limit": args.sample_limit,
        "max_length": args.max_length,
        "dropped_label_count": dropped,
        "declared_selection_grid": list(DECLARED_GRID),
        "diagnostic_thresholds": list(DIAGNOSTIC_THRESHOLDS),
        "threshold_policy": (
            "diagnostic thresholds do not change or select the declared grid"
        ),
        "heads": head_reports,
        "device": str(device),
        "contains_text_or_predictions": False,
        "auditor_sha256": sha256(Path(__file__)),
        "model_module_sha256": sha256(
            ROOT / "dki" / "nlp" / "cmeie_gplinker.py"
        ),
        "alignment_module_sha256": sha256(
            ROOT / "dki" / "nlp" / "cmeie_spo.py"
        ),
    }
    publish_once(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(selected),
                "dropped_label_count": dropped,
                "heads": {
                    name: report["gold_logit_summary"]
                    for name, report in head_reports.items()
                },
                "outer_accessed": False,
                "official_dev_accessed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
