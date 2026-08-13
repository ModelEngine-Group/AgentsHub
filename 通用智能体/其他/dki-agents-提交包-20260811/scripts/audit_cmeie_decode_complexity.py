"""Aggregate-only CMeIE checkpoint logit/candidate audit.

The audit never emits source text, labels, predictions, or outer/dev results.
It measures why a declared threshold grid is expensive before changing the
exact decoder implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"decode audit output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _device(torch):
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise SystemExit("decode complexity audit requires NPU/CUDA; CPU fallback refused")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip-grid-decode", action="store_true")
    parser.add_argument(
        "--tail-threshold-floor",
        type=float,
        choices=(-0.5, 0.0, 0.5),
        default=-0.5,
    )
    args = parser.parse_args()
    if len(args.source_tree_sha256) != 64:
        raise SystemExit("--source-tree-sha256 must be a full SHA-256")
    if args.sample_count < 1:
        raise SystemExit("--sample-count must be positive")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_gplinker import CMeIEGPLinker
    from dki.nlp.cmeie_spo import (
        decode_spo_grid_sparse_axes,
        schema_constraints_from_fit,
        sparsify_pointer_logits,
    )

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.train.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = list(manifest["splits"]["calibration"]["indices"])[: args.sample_count]
    sample = [rows[index] for index in indices]
    fit = [
        rows[index]
        for index in manifest["splits"]["fit"]["indices"]
    ]
    schema_constraints = schema_constraints_from_fit(fit)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    entity_labels = list(payload["entity_labels"])
    predicates = list(payload["predicates"])
    checkpoint_dir = args.checkpoint.parent
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    encoder = AutoModel.from_pretrained(checkpoint_dir)
    model = CMeIEGPLinker(
        encoder,
        int(encoder.config.hidden_size),
        len(entity_labels),
        len(predicates),
    )
    model.load_state_dict(payload["model"])
    device = _device(torch)
    model = model.to(device).eval()

    def collate(batch):
        return tokenizer(
            [row["text"] for row in batch],
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )

    loader = DataLoader(sample, batch_size=1, shuffle=False, collate_fn=collate)
    thresholds = (-0.5, 0.0, 0.5)
    totals = {
        head: {
            str(threshold): []
            for threshold in thresholds
        }
        for head in ("entity", "relation_head", "relation_tail")
    }
    inference_seconds = []
    transfer_seconds = []
    decode_seconds = []
    maximum_grid_prediction_counts = []
    join_match_lower_bounds = []
    join_estimate_capped = []
    token_lengths = []
    with torch.no_grad():
        for row, encoded in zip(sample, loader):
            token_lengths.append(int(encoded["attention_mask"].sum()))
            offsets = [
                tuple(pair)
                for pair in encoded.pop("offset_mapping")[0].tolist()
            ]
            inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            started = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.fp16,
            ):
                outputs = model(**inputs)
            if device.type == "npu":
                torch.npu.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds.append(time.perf_counter() - started)
            started = time.perf_counter()
            cpu_outputs = {
                name: value[0].float().cpu()
                for name, value in outputs.items()
            }
            transfer_seconds.append(time.perf_counter() - started)
            for head, logits in cpu_outputs.items():
                for threshold in thresholds:
                    totals[head][str(threshold)].append(
                        int((logits > threshold).sum().item())
                    )
            sparse = {
                name: sparsify_pointer_logits(
                    outputs[name][0],
                    (
                        args.tail_threshold_floor
                        if name == "relation_tail"
                        else thresholds[0]
                    ),
                )
                for name in ("entity", "relation_head", "relation_tail")
            }
            subject_by_start = defaultdict(list)
            object_by_span = defaultdict(int)
            entity_indices, entity_scores = sparse["entity"]
            for (label_id, start, end), score in zip(
                entity_indices.tolist(),
                entity_scores.tolist(),
            ):
                if (
                    score <= thresholds[0]
                    or start > end
                    or offsets[start][1] <= offsets[start][0]
                    or offsets[end][1] <= offsets[end][0]
                ):
                    continue
                role, _ = entity_labels[label_id].split(":", 1)
                if role == "subject":
                    subject_by_start[start].append(end)
                elif role == "object":
                    object_by_span[(start, end)] += 1
            tail_by_subject_end = defaultdict(list)
            tail_indices, _ = sparse["relation_tail"]
            for predicate_id, subject_end, object_end in tail_indices.tolist():
                tail_by_subject_end[(predicate_id, subject_end)].append(object_end)
            join_count = 0
            capped = False
            head_indices, _ = sparse["relation_head"]
            for predicate_id, subject_start, object_start in head_indices.tolist():
                for subject_end in subject_by_start.get(subject_start, ()):
                    for object_end in tail_by_subject_end.get(
                        (predicate_id, subject_end),
                        (),
                    ):
                        join_count += object_by_span.get(
                            (object_start, object_end),
                            0,
                        )
                        if join_count >= 1_000_000:
                            capped = True
                            break
                    if capped:
                        break
                if capped:
                    break
            join_match_lower_bounds.append(join_count)
            join_estimate_capped.append(capped)
            if not args.skip_grid_decode:
                started = time.perf_counter()
                tail_thresholds = tuple(
                    value
                    for value in thresholds
                    if value >= args.tail_threshold_floor
                )
                decoded = decode_spo_grid_sparse_axes(
                    row["text"],
                    offsets,
                    sparse["entity"],
                    sparse["relation_head"],
                    sparse["relation_tail"],
                    entity_labels,
                    predicates,
                    thresholds,
                    thresholds,
                    tail_thresholds,
                    schema_constraints,
                )
                decode_seconds.append(time.perf_counter() - started)
                maximum_grid_prediction_counts.append(
                    max((len(predictions) for predictions in decoded.values()), default=0)
                )

    summary = defaultdict(dict)
    for head, threshold_counts in totals.items():
        for threshold, counts in threshold_counts.items():
            summary[head][threshold] = {
                "mean": sum(counts) / max(1, len(counts)),
                "max": max(counts, default=0),
                "min": min(counts, default=0),
            }
    result = {
        "schema_version": 1,
        "kind": "aggregate_cmeie_decode_complexity_audit",
        "source_tree_sha256": args.source_tree_sha256,
        "train_sha256": sha256(args.train),
        "split_manifest_sha256": sha256(args.split_manifest),
        "split_role": "calibration",
        "outer_accessed": False,
        "official_dev_accessed": False,
        "checkpoint_sha256": sha256(args.checkpoint),
        "script_sha256": sha256(Path(__file__)),
        "sample_count": len(sample),
        "max_length": args.max_length,
        "fp16": args.fp16,
        "device_type": device.type,
        "token_length": {
            "mean": sum(token_lengths) / max(1, len(token_lengths)),
            "max": max(token_lengths, default=0),
            "min": min(token_lengths, default=0),
        },
        "mean_inference_seconds": sum(inference_seconds) / max(1, len(inference_seconds)),
        "mean_dense_transfer_seconds_for_diagnostic_only": (
            sum(transfer_seconds) / max(1, len(transfer_seconds))
        ),
        "grid_decode_skipped": args.skip_grid_decode,
        "tail_threshold_floor": args.tail_threshold_floor,
        "mean_sparse_grid_seconds": (
            sum(decode_seconds) / len(decode_seconds)
            if decode_seconds
            else None
        ),
        "max_grid_prediction_count": (
            max(maximum_grid_prediction_counts)
            if maximum_grid_prediction_counts
            else None
        ),
        "join_match_lower_bound": {
            "maximum": max(join_match_lower_bounds, default=0),
            "cap": 1_000_000,
            "any_capped": any(join_estimate_capped),
        },
        "candidate_counts": summary,
        "contains_source_text_or_predictions": False,
    }
    create_once(args.output, result)
    print(
        json.dumps(
            {
                "sample_count": result["sample_count"],
                "candidate_counts": result["candidate_counts"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
