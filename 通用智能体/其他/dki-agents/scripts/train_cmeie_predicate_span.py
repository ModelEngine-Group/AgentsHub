"""Train the predicate-conditioned CMeIE span linker on Ascend NPU.

This is a calibration-selection entry point.  It reads only the frozen fit and
calibration shards, uses three global thresholds, and atomically publishes one
checkpoint bundle.  Outer and official-dev shards are intentionally absent
from the interface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dki.nlp.cmeie_predicate_span import (
    CMeIEPredicateSpanLinker,
    LOSS_MODE_BALANCED_PER_HEAD_SPARSE_LOGSUMEXP,
    LOSS_MODE_HARD_NEGATIVE_BCE,
    PREDICATE_SPAN_LOSS_MODES,
    PredicateSpanThresholdGridDecode,
    build_predicate_span_labels,
    decode_predicate_span_threshold_grid,
    predicate_span_loss,
)
from dki.nlp.cmeie_spo import gold_spo_set, labels_from_fit
from scripts.train_spo_cmeie import _publish_directory_create_once


DECLARED_THRESHOLD_VALUES = (-0.5, 0.0, 0.5)
HIGH_THRESHOLD_ABLATION_VALUES = (0.0, 0.5, 1.0)
THRESHOLD_GRID_MODES = {
    "declared": DECLARED_THRESHOLD_VALUES,
    "high_threshold_ablation": HIGH_THRESHOLD_ABLATION_VALUES,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _row_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _select_fit(
    rows: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    chosen: list[int] = []
    seen: set[str] = set()
    # Preserve rare-predicate coverage in a bounded feasibility run.
    for index in order:
        predicates = {
            str(spo.get("predicate", ""))
            for spo in rows[index].get("spo_list", [])
        }
        if predicates - seen:
            chosen.append(index)
            seen.update(predicates)
            if len(chosen) == limit:
                break
    chosen_set = set(chosen)
    for index in order:
        if len(chosen) == limit:
            break
        if index not in chosen_set:
            chosen.append(index)
            chosen_set.add(index)
    return [rows[index] for index in chosen]


def _select_calibration(
    rows: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    order = list(range(len(rows)))
    random.Random(seed ^ 0xC0FFEE).shuffle(order)
    return [rows[index] for index in order[:limit]]


def _length_bucket_rows(
    rows: list[dict[str, Any]], *, batch_size: int, seed: int
) -> list[dict[str, Any]]:
    """Return deterministic shuffled batches with similar text lengths."""

    rng = random.Random(seed)
    indices = sorted(
        range(len(rows)),
        key=lambda index: (len(str(rows[index].get("text", ""))), index),
    )
    bucket_size = max(batch_size, batch_size * 32)
    batches: list[list[int]] = []
    for start in range(0, len(indices), bucket_size):
        bucket = indices[start : start + bucket_size]
        rng.shuffle(bucket)
        batches.extend(
            bucket[offset : offset + batch_size]
            for offset in range(0, len(bucket), batch_size)
        )
    rng.shuffle(batches)
    return [rows[index] for batch in batches for index in batch]


def _collate(
    rows: list[dict[str, Any]],
    *,
    tokenizer,
    predicate2id: dict[str, int],
    max_length: int,
    include_labels: bool,
):
    encoded = tokenizer(
        [str(row["text"]) for row in rows],
        max_length=max_length,
        truncation=True,
        padding=True,
        # A bounded set of shapes prevents the host/NPU allocators from
        # retaining hundreds of differently sized dense pointer grids.
        pad_to_multiple_of=16,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = [
        [tuple(map(int, pair)) for pair in item]
        for item in encoded.pop("offset_mapping").tolist()
    ]
    labels = None
    dropped = 0
    if include_labels:
        label_rows = []
        for row, mapping in zip(rows, offsets):
            current, row_dropped = build_predicate_span_labels(
                row,
                mapping,
                predicate2id,
                allow_cross_sentence_fallback=True,
            )
            label_rows.append(current)
            dropped += row_dropped
        labels = {
            name: __import__("torch").stack(
                [item[name] for item in label_rows]
            )
            for name in ("subject", "object", "link")
        }
    return encoded, offsets, labels, dropped


def _metric(count: Counter) -> dict[str, float | int]:
    tp, fp, fn = count["tp"], count["fp"], count["fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class _CalibrationGrid:
    """Accumulate only threshold combinations safe for every observed row."""

    def __init__(
        self, combinations: list[tuple[float, float, float]]
    ) -> None:
        if not combinations:
            raise ValueError("calibration threshold combinations must not be empty")
        self.declared_combinations = tuple(combinations)
        self.counts = {
            thresholds: Counter(tp=0, fp=0, fn=0)
            for thresholds in combinations
        }
        self.rejected_combinations: set[
            tuple[float, float, float]
        ] = set()
        self.rejected_thresholds: dict[
            tuple[str, float], dict[str, int | float | str]
        ] = {}
        self.rejected_join_work: dict[
            tuple[float, float, float], dict[str, Any]
        ] = {}
        self.maximum_predictions_by_combination = {
            thresholds: 0 for thresholds in combinations
        }
        self.maximum_active_by_combination = {
            thresholds: [0, 0, 0] for thresholds in combinations
        }
        self.maximum_join_work_by_combination = {
            thresholds: 0 for thresholds in combinations
        }

    @property
    def valid_combinations(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(self.counts)

    def observe(
        self,
        result: PredicateSpanThresholdGridDecode,
        gold: set[tuple[str, str, str]],
        *,
        calibration_row: int,
    ) -> None:
        for rejection in result.density_rejections:
            key = (rejection.head, rejection.threshold)
            existing = self.rejected_thresholds.get(key)
            if existing is None:
                self.rejected_thresholds[key] = {
                    "head": rejection.head,
                    "threshold": rejection.threshold,
                    "maximum_active_candidates_observed": (
                        rejection.active_candidates
                    ),
                    "first_rejected_calibration_row": calibration_row,
                }
            else:
                existing["maximum_active_candidates_observed"] = max(
                    int(existing["maximum_active_candidates_observed"]),
                    rejection.active_candidates,
                )

        for rejection in result.combination_density_rejections:
            existing = self.rejected_join_work.get(rejection.thresholds)
            if existing is None:
                self.rejected_join_work[rejection.thresholds] = {
                    "thresholds": list(rejection.thresholds),
                    "maximum_join_work_observed": rejection.join_work,
                    "maximum_join_work_allowed": rejection.maximum_join_work,
                    "first_rejected_calibration_row": calibration_row,
                }
            else:
                existing["maximum_join_work_observed"] = max(
                    int(existing["maximum_join_work_observed"]),
                    rejection.join_work,
                )

        newly_rejected = set(result.rejected_combinations).intersection(
            self.counts
        )
        self.rejected_combinations.update(newly_rejected)
        for thresholds in newly_rejected:
            # Discard prior rows too: calibration selects one global threshold
            # combination that must be safe for every row in the dataset.
            del self.counts[thresholds]
            del self.maximum_predictions_by_combination[thresholds]
            del self.maximum_active_by_combination[thresholds]
            del self.maximum_join_work_by_combination[thresholds]
        if not self.counts:
            raise OverflowError(
                "candidate density rejected every declared threshold "
                "combination across calibration rows"
            )
        if set(result.decoded) != set(self.counts):
            raise RuntimeError(
                "threshold-grid decoder did not return exactly the globally "
                "valid calibration combinations"
            )
        if set(result.join_work_by_combination) != set(self.counts):
            raise RuntimeError(
                "threshold-grid decoder did not report join work for exactly "
                "the globally valid calibration combinations"
            )
        if set(result.active_candidates_by_combination) != set(self.counts):
            raise RuntimeError(
                "threshold-grid decoder did not report active candidates for "
                "exactly the globally valid calibration combinations"
            )

        for thresholds, predicted in result.decoded.items():
            self.maximum_predictions_by_combination[thresholds] = max(
                self.maximum_predictions_by_combination[thresholds],
                len(predicted),
            )
            active = result.active_candidates_by_combination[thresholds]
            self.maximum_active_by_combination[thresholds] = [
                max(previous, observed)
                for previous, observed in zip(
                    self.maximum_active_by_combination[thresholds],
                    active,
                )
            ]
            self.maximum_join_work_by_combination[thresholds] = max(
                self.maximum_join_work_by_combination[thresholds],
                result.join_work_by_combination[thresholds],
            )
            count = self.counts[thresholds]
            count["tp"] += len(predicted & gold)
            count["fp"] += len(predicted - gold)
            count["fn"] += len(gold - predicted)

    def density_report(
        self,
        *,
        selected_thresholds: tuple[float, float, float] | None = None,
    ) -> dict[str, Any]:
        rejected = sorted(self.rejected_combinations)
        rejected_thresholds = sorted(
            self.rejected_thresholds.values(),
            key=lambda item: (str(item["head"]), float(item["threshold"])),
        )
        report = {
            "declared_threshold_combination_count": len(
                self.declared_combinations
            ),
            "density_rejected_combination_count": len(rejected),
            "density_rejected_combinations": [list(item) for item in rejected],
            "density_rejected_thresholds": rejected_thresholds,
            "join_work_rejected_combinations": sorted(
                self.rejected_join_work.values(),
                key=lambda item: tuple(item["thresholds"]),
            ),
            "remaining_threshold_combination_count": len(self.counts),
            "grid_maximum_predictions_per_example": max(
                self.maximum_predictions_by_combination.values(),
                default=0,
            ),
            "grid_maximum_active_candidates_per_head": {
                name: max(
                    (
                        values[index]
                        for values in self.maximum_active_by_combination.values()
                    ),
                    default=0,
                )
                for index, name in enumerate(("subject", "object", "link"))
            },
            "grid_maximum_join_work_per_example": max(
                self.maximum_join_work_by_combination.values(),
                default=0,
            ),
            "per_combination_density": [
                {
                    "thresholds": list(thresholds),
                    "maximum_predictions_per_example": (
                        self.maximum_predictions_by_combination[thresholds]
                    ),
                    "maximum_active_candidates_per_head": {
                        name: self.maximum_active_by_combination[thresholds][
                            index
                        ]
                        for index, name in enumerate(
                            ("subject", "object", "link")
                        )
                    },
                    "maximum_join_work_per_example": (
                        self.maximum_join_work_by_combination[thresholds]
                    ),
                }
                for thresholds in sorted(self.counts)
            ],
        }
        if selected_thresholds is not None:
            if selected_thresholds not in self.counts:
                raise ValueError("selected thresholds are not calibration-safe")
            report.update(
                {
                    "selected_maximum_predictions_per_example": (
                        self.maximum_predictions_by_combination[
                            selected_thresholds
                        ]
                    ),
                    "selected_maximum_active_candidates_per_head": {
                        name: self.maximum_active_by_combination[
                            selected_thresholds
                        ][index]
                        for index, name in enumerate(
                            ("subject", "object", "link")
                        )
                    },
                    "selected_maximum_join_work_per_example": (
                        self.maximum_join_work_by_combination[
                            selected_thresholds
                        ]
                    ),
                }
            )
        return report


def _evaluate(
    *,
    model,
    rows: list[dict[str, Any]],
    tokenizer,
    predicate2id: dict[str, int],
    predicates: list[str],
    max_length: int,
    batch_size: int,
    device,
    threshold_values: tuple[float, ...],
    maximum_active_candidates: int,
    maximum_join_work: int,
) -> tuple[
    tuple[float, float, float],
    dict[str, float | int],
    dict[str, Any],
]:
    import torch

    combinations = [
        (subject, obj, link)
        for subject in threshold_values
        for obj in threshold_values
        for link in threshold_values
    ]
    calibration_grid = _CalibrationGrid(combinations)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            encoded, offsets, _, _ = _collate(
                batch_rows,
                tokenizer=tokenizer,
                predicate2id=predicate2id,
                max_length=max_length,
                include_labels=False,
            )
            inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(
                device_type="npu", dtype=torch.float16, enabled=True
            ):
                outputs = model(**inputs)
            for row_index, (row, mapping) in enumerate(
                zip(batch_rows, offsets)
            ):
                current = {
                    name: tensor[row_index].detach().cpu()
                    for name, tensor in outputs.items()
                }
                result = decode_predicate_span_threshold_grid(
                    str(row["text"]),
                    mapping,
                    current,
                    predicates,
                    threshold_values,
                    maximum_active_candidates=maximum_active_candidates,
                    maximum_join_work=maximum_join_work,
                    threshold_combinations=(
                        calibration_grid.valid_combinations
                    ),
                )
                gold = gold_spo_set(row)
                calibration_grid.observe(
                    result,
                    gold,
                    calibration_row=start + row_index + 1,
                )
    scored = [
        (thresholds, _metric(count))
        for thresholds, count in calibration_grid.counts.items()
    ]
    thresholds, metric = max(
        scored,
        key=lambda item: (
            item[1]["f1"],
            item[1]["precision"],
            sum(item[0]),
            item[0],
        ),
    )
    return (
        thresholds,
        metric,
        calibration_grid.density_report(selected_thresholds=thresholds),
    )


def _cpu_state_dict(model) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _set_training_mode(model, *, encoder_trainable: bool) -> None:
    """Train pointer heads while disabling frozen encoder dropout."""

    model.train()
    if not encoder_trainable:
        model.encoder.eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--source-tree-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rescore-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Evaluate one immutable checkpoint on the exact declared grid "
            "without training; the resulting diagnostic is never promotion "
            "eligible"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--head-warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--negatives-per-head", type=int, default=32)
    parser.add_argument("--inactive-weight", type=float, default=0.10)
    parser.add_argument(
        "--loss-mode",
        choices=PREDICATE_SPAN_LOSS_MODES,
        default=LOSS_MODE_HARD_NEGATIVE_BCE,
    )
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--minimum-calibration-f1", type=float, default=0.45)
    parser.add_argument(
        "--threshold-grid-mode",
        choices=tuple(THRESHOLD_GRID_MODES),
        default="declared",
    )
    parser.add_argument(
        "--threshold-values",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument("--maximum-active-candidates", type=int, default=20000)
    parser.add_argument("--maximum-join-work", type=int, default=20000)
    parser.add_argument("--limit-fit", type=int, default=0)
    parser.add_argument("--limit-calibration", type=int, default=0)
    args = parser.parse_args()
    loss_ablation = (
        args.loss_mode == LOSS_MODE_BALANCED_PER_HEAD_SPARSE_LOGSUMEXP
    )
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    if not 0 <= args.head_warmup_epochs <= args.epochs:
        raise ValueError("head-warmup-epochs must not exceed epochs")
    if args.maximum_active_candidates < 1:
        raise ValueError("maximum-active-candidates must be positive")
    if args.maximum_join_work < 1:
        raise ValueError("maximum-join-work must be positive")
    available_threshold_values = THRESHOLD_GRID_MODES[
        args.threshold_grid_mode
    ]
    threshold_values = (
        available_threshold_values
        if args.threshold_values is None
        else tuple(float(value) for value in args.threshold_values)
    )
    if (
        not threshold_values
        or len(set(threshold_values)) != len(threshold_values)
        or any(
            value not in available_threshold_values
            for value in threshold_values
        )
    ):
        raise ValueError(
            "threshold-values must be a unique subset of the selected grid mode"
        )
    if args.rescore_checkpoint_dir is not None and (
        args.threshold_grid_mode != "declared"
        or threshold_values != DECLARED_THRESHOLD_VALUES
    ):
        raise ValueError(
            "checkpoint rescoring requires the exact declared threshold grid"
        )

    import torch
    import torch_npu  # noqa: F401
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModel,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("npu:0")
    full_fit = _load(args.fit_data)
    full_calibration = _load(args.calibration_data)
    fit_rows = _select_fit(full_fit, args.limit_fit, args.seed)
    calibration_rows = _select_calibration(
        full_calibration, args.limit_calibration, args.seed
    )
    _, predicates = labels_from_fit(full_fit, entity_label_mode="role_only")
    predicate2id = {
        predicate: index for index, predicate in enumerate(predicates)
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True
    )
    encoder = AutoModel.from_pretrained(
        args.base_model, local_files_only=True
    )
    hidden_size = int(encoder.config.hidden_size)
    model = CMeIEPredicateSpanLinker(
        encoder, hidden_size, len(predicates), inner_dim=64
    ).to(device)

    if args.rescore_checkpoint_dir is not None:
        checkpoint_dir = args.rescore_checkpoint_dir
        checkpoint_state = checkpoint_dir / "model_state.pt"
        checkpoint_receipt_path = checkpoint_dir / "candidate_receipt.json"
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"checkpoint directory does not exist: {checkpoint_dir}"
            )
        if not checkpoint_state.is_file():
            raise FileNotFoundError(
                f"checkpoint state does not exist: {checkpoint_state}"
            )
        if not checkpoint_receipt_path.is_file():
            raise FileNotFoundError(
                "checkpoint candidate receipt does not exist: "
                f"{checkpoint_receipt_path}"
            )
        checkpoint_receipt = json.loads(
            checkpoint_receipt_path.read_text(encoding="utf-8")
        )
        if not isinstance(checkpoint_receipt, dict):
            raise ValueError("checkpoint candidate receipt must be an object")

        expected_bindings = {
            "fit_data_sha256": _sha256(args.fit_data),
            "calibration_data_sha256": _sha256(args.calibration_data),
            "base_model_config_sha256": _sha256(
                args.base_model / "config.json"
            ),
            "fit_selection_sha256": _row_digest(fit_rows),
            "calibration_selection_sha256": _row_digest(calibration_rows),
        }
        for key, expected in expected_bindings.items():
            if checkpoint_receipt.get(key) != expected:
                raise ValueError(
                    f"checkpoint receipt {key} differs from rescore input"
                )
        if checkpoint_receipt.get("predicates") != predicates:
            raise ValueError(
                "checkpoint predicate label order differs from rescore input"
            )

        state = torch.load(
            checkpoint_state,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        started = time.perf_counter()
        thresholds, metric, density_report = _evaluate(
            model=model,
            rows=calibration_rows,
            tokenizer=tokenizer,
            predicate2id=predicate2id,
            predicates=predicates,
            max_length=args.max_length,
            batch_size=args.evaluation_batch_size,
            device=device,
            threshold_values=threshold_values,
            maximum_active_candidates=args.maximum_active_candidates,
            maximum_join_work=args.maximum_join_work,
        )
        elapsed = time.perf_counter() - started
        diagnostic_gate_passed = (
            metric["f1"] >= args.minimum_calibration_f1
        )
        payload = {
            "schema_version": 1,
            "kind": "cmeie_predicate_span_posthoc_grid_rescore",
            "dataset_role": "calibration_selection_used",
            "posthoc_declared_grid_rescore": True,
            "fit_only_training": True,
            "training_performed": False,
            "calibration_accessed": True,
            "outer_accessed": False,
            "official_dev_accessed": False,
            "passed": diagnostic_gate_passed,
            "artifact_status": (
                "posthoc_quality_observed"
                if diagnostic_gate_passed
                else "failed_evidence"
            ),
            "promotion_eligible": False,
            "minimum_calibration_f1": args.minimum_calibration_f1,
            "thresholds": list(thresholds),
            "declared_threshold_grid": list(DECLARED_THRESHOLD_VALUES),
            "threshold_grid_mode": "declared",
            "threshold_grid_role": "posthoc_declared_grid_rescore",
            "evaluated_threshold_grid": list(threshold_values),
            "maximum_active_candidates": args.maximum_active_candidates,
            "maximum_join_work": args.maximum_join_work,
            **density_report,
            "metric": metric,
            "seed": args.seed,
            "fit_rows": len(fit_rows),
            "fit_rows_available": len(full_fit),
            "calibration_rows": len(calibration_rows),
            "calibration_rows_available": len(full_calibration),
            **expected_bindings,
            "source_tree_manifest_sha256": _sha256(
                args.source_tree_manifest
            ),
            "evaluator_sha256": _sha256(Path(__file__)),
            "checkpoint_model_state_sha256": _sha256(checkpoint_state),
            "checkpoint_candidate_receipt_sha256": _sha256(
                checkpoint_receipt_path
            ),
            "checkpoint_source_tree_manifest_sha256": (
                checkpoint_receipt.get("source_tree_manifest_sha256")
            ),
            "checkpoint_loss_mode": checkpoint_receipt.get("loss_mode"),
            "checkpoint_best_epoch": checkpoint_receipt.get("best_epoch"),
            "checkpoint_threshold_grid_mode": checkpoint_receipt.get(
                "threshold_grid_mode"
            ),
            "predicates": predicates,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "evaluation_batch_size": args.evaluation_batch_size,
            "max_length": args.max_length,
            "elapsed_seconds": elapsed,
            "device": "npu:0",
        }
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{args.output_dir.name}.",
                dir=args.output_dir.parent,
            )
        )
        try:
            (staging / "rescore_receipt.json").write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            if not diagnostic_gate_passed:
                (staging / "FAILED_EVIDENCE.json").write_text(
                    json.dumps(
                        {
                            "artifact_status": "failed_evidence",
                            "passed": False,
                            "promotion_eligible": False,
                            "minimum_calibration_f1": (
                                args.minimum_calibration_f1
                            ),
                            "observed_calibration_f1": metric["f1"],
                        },
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            _publish_directory_create_once(staging, args.output_dir)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if diagnostic_gate_passed else 2

    encoder_parameters = list(model.encoder.parameters())
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    optimiser = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": args.encoder_learning_rate,
            },
            {
                "params": head_parameters,
                "lr": args.head_learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
    )
    batches_per_epoch = math.ceil(len(fit_rows) / args.batch_size)
    optimiser_steps_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation
    )
    total_steps = optimiser_steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=round(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.npu.amp.GradScaler(init_scale=1024.0, enabled=True)
    started = time.perf_counter()
    best_state = None
    best_epoch = 0
    best_thresholds = (0.0, 0.0, 0.0)
    best_metric: dict[str, float | int] = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    best_density_report: dict[str, Any] | None = None
    history = []
    dropped_labels = 0
    skipped_steps = 0
    successful_steps = 0
    optimiser.zero_grad(set_to_none=True)
    stale_epochs = 0
    for epoch in range(args.epochs):
        epoch_rows = _length_bucket_rows(
            fit_rows,
            batch_size=args.batch_size,
            seed=args.seed + epoch,
        )
        loader = DataLoader(
            epoch_rows,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda rows: _collate(
                rows,
                tokenizer=tokenizer,
                predicate2id=predicate2id,
                max_length=args.max_length,
                include_labels=True,
            ),
        )
        encoder_trainable = epoch >= args.head_warmup_epochs
        for parameter in encoder_parameters:
            parameter.requires_grad_(encoder_trainable)
        _set_training_mode(model, encoder_trainable=encoder_trainable)
        losses: list[float] = []
        epoch_dropped = 0
        for batch_index, (encoded, _, labels, dropped) in enumerate(loader):
            inputs = {key: value.to(device) for key, value in encoded.items()}
            npu_labels = {
                name: value.to(device) for name, value in labels.items()
            }
            group_start = (
                batch_index // args.gradient_accumulation
            ) * args.gradient_accumulation
            group_size = min(
                args.gradient_accumulation,
                len(loader) - group_start,
            )
            with torch.autocast(
                device_type="npu", dtype=torch.float16, enabled=True
            ):
                outputs = model(**inputs)
                loss = predicate_span_loss(
                    outputs,
                    npu_labels,
                    negatives_per_head=args.negatives_per_head,
                    inactive_weight=args.inactive_weight,
                    loss_mode=args.loss_mode,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch={epoch + 1} "
                    f"batch={batch_index + 1}"
                )
            scaler.scale(loss / group_size).backward()
            losses.append(float(loss.detach().cpu()))
            epoch_dropped += dropped
            should_step = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                scaler.unscale_(optimiser)
                encoder_norm = torch.nn.utils.clip_grad_norm_(
                    encoder_parameters, 1.0
                )
                head_norm = torch.nn.utils.clip_grad_norm_(
                    head_parameters, 5.0
                )
                scale_before = float(scaler.get_scale())
                scaler.step(optimiser)
                scaler.update()
                if float(scaler.get_scale()) < scale_before:
                    skipped_steps += 1
                else:
                    scheduler.step()
                    successful_steps += 1
                optimiser.zero_grad(set_to_none=True)
                if successful_steps % 25 == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "optimizer_step": successful_steps,
                                "loss": losses[-1],
                                "encoder_grad_norm": (
                                    float(encoder_norm.detach().cpu())
                                    if torch.isfinite(encoder_norm)
                                    else None
                                ),
                                "head_grad_norm": (
                                    float(head_norm.detach().cpu())
                                    if torch.isfinite(head_norm)
                                    else None
                                ),
                                "amp_scale": float(scaler.get_scale()),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        dropped_labels += epoch_dropped
        thresholds, metric, density_report = _evaluate(
            model=model,
            rows=calibration_rows,
            tokenizer=tokenizer,
            predicate2id=predicate2id,
            predicates=predicates,
            max_length=args.max_length,
            batch_size=args.evaluation_batch_size,
            device=device,
            threshold_values=threshold_values,
            maximum_active_candidates=args.maximum_active_candidates,
            maximum_join_work=args.maximum_join_work,
        )
        epoch_receipt = {
            "epoch": epoch + 1,
            "encoder_trainable": encoder_trainable,
            "loss_mode": args.loss_mode,
            "loss_ablation": loss_ablation,
            "mean_loss": sum(losses) / max(1, len(losses)),
            "thresholds": list(thresholds),
            "metric": metric,
            "dropped_labels": epoch_dropped,
            **density_report,
        }
        history.append(epoch_receipt)
        print(json.dumps(epoch_receipt, sort_keys=True), flush=True)
        improvement_key = (
            metric["f1"],
            metric["precision"],
            sum(thresholds),
            thresholds,
        )
        best_key = (
            best_metric["f1"],
            best_metric["precision"],
            sum(best_thresholds),
            best_thresholds,
        )
        if best_state is None or improvement_key > best_key:
            best_epoch = epoch + 1
            best_thresholds = thresholds
            best_metric = metric
            best_density_report = density_report
            best_state = _cpu_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.early_stopping_patience:
            break

    if best_state is None:
        best_state = _cpu_state_dict(model)
    if best_density_report is None:
        raise RuntimeError("training completed without a calibration evaluation")
    elapsed = time.perf_counter() - started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    passed = best_metric["f1"] >= args.minimum_calibration_f1
    payload = {
        "schema_version": 1,
        "kind": "cmeie_predicate_span_calibration_candidate",
        "dataset_role": "calibration_selection_used",
        "fit_only_training": True,
        "calibration_accessed": True,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "passed": passed,
        "artifact_status": (
            "loss_ablation_quality_gate_passed"
            if passed and loss_ablation
            else "quality_gate_passed"
            if passed
            else "failed_evidence"
        ),
        "promotion_eligible": passed and not loss_ablation,
        "loss_mode": args.loss_mode,
        "loss_role": (
            "balanced_per_head_sparse_logsumexp_ablation"
            if loss_ablation
            else "formal_default"
        ),
        "loss_ablation": loss_ablation,
        "minimum_calibration_f1": args.minimum_calibration_f1,
        "best_epoch": best_epoch,
        "thresholds": list(best_thresholds),
        "declared_threshold_grid": list(DECLARED_THRESHOLD_VALUES),
        "threshold_grid_mode": args.threshold_grid_mode,
        "threshold_grid_role": (
            "formal_declared"
            if args.threshold_grid_mode == "declared"
            else "evidence_driven_ablation"
        ),
        "available_threshold_grid_for_mode": list(
            available_threshold_values
        ),
        "evaluated_threshold_grid": list(threshold_values),
        "maximum_active_candidates": args.maximum_active_candidates,
        "maximum_join_work": args.maximum_join_work,
        "cross_sentence_pair_policy": (
            "same_sentence_or_unique_deterministic_nearest_fallback"
        ),
        **best_density_report,
        "metric": best_metric,
        "history": history,
        "seed": args.seed,
        "fit_rows": len(fit_rows),
        "fit_rows_available": len(full_fit),
        "calibration_rows": len(calibration_rows),
        "calibration_rows_available": len(full_calibration),
        "fit_selection_sha256": _row_digest(fit_rows),
        "calibration_selection_sha256": _row_digest(calibration_rows),
        "fit_data_sha256": _sha256(args.fit_data),
        "calibration_data_sha256": _sha256(args.calibration_data),
        "source_tree_manifest_sha256": _sha256(
            args.source_tree_manifest
        ),
        "base_model_config_sha256": _sha256(
            args.base_model / "config.json"
        ),
        "predicates": predicates,
        "parameter_count": parameter_count,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "head_warmup_epochs": args.head_warmup_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation
        ),
        "max_length": args.max_length,
        "pad_to_multiple_of": 16,
        "length_bucket_size": args.batch_size * 32,
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "negatives_per_head": args.negatives_per_head,
        "negatives_per_head_applied": not loss_ablation,
        "inactive_weight": args.inactive_weight,
        "early_stopping_patience": args.early_stopping_patience,
        "successful_optimizer_steps": successful_steps,
        "skipped_optimizer_steps": skipped_steps,
        "dropped_label_observations": dropped_labels,
        "elapsed_seconds": elapsed,
        "device": "npu:0",
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.",
            dir=args.output_dir.parent,
        )
    )
    try:
        torch.save(best_state, staging / "model_state.pt")
        tokenizer.save_pretrained(staging / "tokenizer")
        (staging / "candidate_receipt.json").write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if not passed:
            (staging / "FAILED_EVIDENCE.json").write_text(
                json.dumps(
                    {
                        "artifact_status": "failed_evidence",
                        "passed": False,
                        "loss_mode": args.loss_mode,
                        "loss_ablation": loss_ablation,
                        "minimum_calibration_f1": args.minimum_calibration_f1,
                        "observed_calibration_f1": best_metric["f1"],
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        _publish_directory_create_once(staging, args.output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
