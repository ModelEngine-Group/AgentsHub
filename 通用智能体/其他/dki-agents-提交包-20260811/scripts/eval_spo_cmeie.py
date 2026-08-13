"""Evaluate one GPLinker epoch on the calibration shard only.

Outer evaluation and official development confirmation are intentionally not
implemented here.  They are available solely through
``scripts/eval_spo_cmeie_batch.py``, whose RID-level dataset ledger prevents
per-candidate replay.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dki.evidence_io import load_bound_json, read_bound_bytes
from dki.nlp.cmeie_artifacts import (
    CheckpointBundleError,
    verify_checkpoint_bundle,
)
from dki.nlp.cmeie_extraction_contract import (
    load_and_validate_extraction_contract,
)
from dki.nlp.cmeie_joint_evaluation import publish_json_once
from dki.nlp.cmeie_run_protocol import (
    canonical_run_paths,
    load_run_binding,
)
from dki.nlp.cmeie_split_protocol import (
    SHARD_FILENAMES,
    load_and_validate_role_shard,
)


SUPPORTED_TRAINING_RECEIPT_SCHEMAS = frozenset({6})
EVALUATION_RECEIPT_SCHEMA_VERSION = 4
DECLARED_THRESHOLDS = (-0.5, 0.0, 0.5)


def resolve_checkpoint_bundle(
    checkpoint: Path,
) -> tuple[Path, Path, Path]:
    """Resolve an epoch directory or its canonical ``gplinker.pt``."""

    if checkpoint.is_dir():
        directory = checkpoint
        payload = directory / "gplinker.pt"
    elif checkpoint.is_file() and checkpoint.name == "gplinker.pt":
        directory = checkpoint.parent
        payload = checkpoint
    else:
        raise SystemExit(
            "checkpoint must be an epoch directory or its gplinker.pt file"
        )
    if re.fullmatch(r"epoch-[0-9]{2}", directory.name) is None:
        raise SystemExit("checkpoint directory must be an epoch-NN bundle")
    if not payload.is_file() or payload.is_symlink():
        raise SystemExit("checkpoint payload must be a regular non-link file")
    receipt = directory.parent / "training_receipt.json"
    if not receipt.is_file() or receipt.is_symlink():
        raise SystemExit("candidate directory lacks training_receipt.json")
    return directory, payload, receipt


def verify_history_bundle(
    checkpoint_directory: Path,
    checkpoint_file: Path,
    receipt: dict[str, Any],
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    match = re.fullmatch(r"epoch-([0-9]{2})", checkpoint_directory.name)
    if match is None:
        raise SystemExit("checkpoint directory is not an epoch bundle")
    epoch = int(match.group(1))
    matching = [
        item
        for item in receipt.get("history", [])
        if type(item) is dict and item.get("epoch") == epoch
    ]
    if len(matching) != 1:
        raise SystemExit(
            "checkpoint epoch is not uniquely recorded in training history"
        )
    history = matching[0]
    try:
        bundle = verify_checkpoint_bundle(
            checkpoint_directory,
            expected_manifest_sha256=history.get(
                "checkpoint_bundle_manifest_sha256"
            ),
            expected_tree_sha256=history.get(
                "checkpoint_bundle_tree_sha256"
            ),
            expected_checkpoint_sha256=history.get("checkpoint_sha256"),
        )
    except (CheckpointBundleError, OSError) as exc:
        raise SystemExit(
            f"checkpoint bundle verification failed: {exc}"
        ) from exc
    if bundle["checkpoint_sha256"] != history.get("checkpoint_sha256"):
        raise SystemExit("checkpoint payload differs from training history")
    if checkpoint_file != checkpoint_directory / "gplinker.pt":
        raise SystemExit("checkpoint payload path is not canonical")
    return epoch, history, bundle


def _load_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, binding = load_bound_json(
        path,
        "CMeIE training receipt",
        reject_multiple_links=True,
        require_object=True,
    )
    if payload.get("schema_version") not in (
        SUPPORTED_TRAINING_RECEIPT_SCHEMAS
    ):
        raise SystemExit("calibration requires training receipt schema v6")
    if payload.get("seed") not in {42, 43, 44}:
        raise SystemExit("training receipt seed must be 42, 43, or 44")
    return payload, binding


def _load_model(
    checkpoint_directory: Path,
    checkpoint_file: Path,
    receipt: dict[str, Any],
    history: dict[str, Any],
    *,
    device_name: str,
) -> tuple[Any, Any, dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_gplinker import CMeIEGPLinker

    bundle_before = verify_checkpoint_bundle(
        checkpoint_directory,
        expected_manifest_sha256=history[
            "checkpoint_bundle_manifest_sha256"
        ],
        expected_tree_sha256=history["checkpoint_bundle_tree_sha256"],
        expected_checkpoint_sha256=history["checkpoint_sha256"],
    )
    raw, checkpoint_binding = read_bound_bytes(
        checkpoint_file,
        "CMeIE calibration checkpoint",
        reject_multiple_links=True,
    )
    saved = torch.load(
        io.BytesIO(raw),
        map_location="cpu",
        weights_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_directory,
        use_fast=True,
    )
    contract_before = load_and_validate_extraction_contract(
        checkpoint_directory,
        tokenizer_vocabulary=tokenizer.get_vocab(),
        expected_binding=history,
    )
    encoder = AutoModel.from_pretrained(checkpoint_directory)
    if device_name == "npu" and not (
        hasattr(torch, "npu") and torch.npu.is_available()
    ):
        raise SystemExit("NPU calibration requested but unavailable")
    device = torch.device("npu:0" if device_name == "npu" else "cpu")
    model = CMeIEGPLinker(
        encoder,
        int(encoder.config.hidden_size),
        len(saved["entity_labels"]),
        len(saved["predicates"]),
        pointer_variant=saved.get("pointer_variant", "efficient"),
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    bundle_after = verify_checkpoint_bundle(
        checkpoint_directory,
        expected_manifest_sha256=history[
            "checkpoint_bundle_manifest_sha256"
        ],
        expected_tree_sha256=history["checkpoint_bundle_tree_sha256"],
        expected_checkpoint_sha256=history["checkpoint_sha256"],
    )
    contract_after = load_and_validate_extraction_contract(
        checkpoint_directory,
        tokenizer_vocabulary=tokenizer.get_vocab(),
        expected_binding=history,
    )
    if bundle_before != bundle_after or contract_before != contract_after:
        raise SystemExit("checkpoint evidence changed while loading")
    return (
        model,
        tokenizer,
        saved,
        device,
        bundle_after,
        {
            "checkpoint": checkpoint_binding,
            "contract": contract_after,
        },
    )


def _evaluate(
    rows: list[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    saved: dict[str, Any],
    device: Any,
    receipt: dict[str, Any],
    search_grid: bool,
    thresholds: tuple[float, float, float] | None,
    schema_on: bool,
) -> tuple[dict[str, Any], tuple[float, float, float], bool, int]:
    import torch

    from dki.nlp.cmeie_spo import (
        decode_spo,
        decode_spo_grid_sparse,
        gold_spo_set,
        sparsify_pointer_logits,
    )

    schema_constraints = {
        predicate: [tuple(pair) for pair in pairs]
        for predicate, pairs in receipt["schema_constraints"].items()
    }
    candidates = (
        [
            (candidate, enabled)
            for candidate in product(DECLARED_THRESHOLDS, repeat=3)
            for enabled in (False, True)
        ]
        if search_grid
        else [(thresholds, schema_on)]
    )
    counts = {
        candidate: {"tp": 0, "fp": 0, "fn": 0}
        for candidate in candidates
    }
    with torch.no_grad():
        for row in rows:
            encoded = tokenizer(
                row["text"],
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
                max_length=int(receipt["max_length"]),
            )
            offsets = [
                tuple(item)
                for item in encoded.pop("offset_mapping")[0].tolist()
            ]
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
                output = model(**inputs)
            gold = gold_spo_set(row)
            decoded_grid = (
                decode_spo_grid_sparse(
                    row["text"],
                    offsets,
                    sparsify_pointer_logits(
                        output["entity"][0],
                        DECLARED_THRESHOLDS[0],
                    ),
                    sparsify_pointer_logits(
                        output["relation_head"][0],
                        DECLARED_THRESHOLDS[0],
                    ),
                    sparsify_pointer_logits(
                        output["relation_tail"][0],
                        DECLARED_THRESHOLDS[0],
                    ),
                    saved["entity_labels"],
                    saved["predicates"],
                    DECLARED_THRESHOLDS,
                    schema_constraints,
                )
                if search_grid
                else None
            )
            for candidate, enabled in candidates:
                predicted = (
                    decoded_grid[(candidate, enabled)]
                    if decoded_grid is not None
                    else decode_spo(
                        row["text"],
                        offsets,
                        output["entity"][0].float().cpu(),
                        output["relation_head"][0].float().cpu(),
                        output["relation_tail"][0].float().cpu(),
                        saved["entity_labels"],
                        saved["predicates"],
                        candidate,
                        schema_constraints if enabled else None,
                    )
                )
                counts[(candidate, enabled)]["tp"] += len(predicted & gold)
                counts[(candidate, enabled)]["fp"] += len(predicted - gold)
                counts[(candidate, enabled)]["fn"] += len(gold - predicted)
    scored = []
    for (candidate, enabled), count in counts.items():
        tp, fp, fn = count["tp"], count["fp"], count["fn"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        scored.append(
            (
                {
                    **count,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                },
                candidate,
                enabled,
            )
        )
    metric, chosen, chosen_schema = max(
        scored,
        key=lambda item: (
            item[0]["f1"],
            item[0]["precision"],
            sum(item[1]),
            item[2],
        ),
    )
    return metric, chosen, chosen_schema, len(candidates)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CMeIE GPLinker calibration-shard evaluator"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration",), required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--calibration-shard", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", nargs=3, type=float)
    parser.add_argument("--search-declared-grid", action="store_true")
    parser.add_argument("--schema-constraint", choices=("off", "on"))
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.search_declared_grid:
        if args.thresholds is not None or args.schema_constraint is not None:
            raise SystemExit(
                "declared-grid search forbids manual thresholds/schema"
            )
        thresholds = None
        schema_on = False
    else:
        if args.thresholds is None or args.schema_constraint is None:
            raise SystemExit(
                "fixed calibration requires thresholds and schema constraint"
            )
        thresholds = tuple(args.thresholds)
        if any(value not in DECLARED_THRESHOLDS for value in thresholds):
            raise SystemExit("thresholds must come from the declared grid")
        schema_on = args.schema_constraint == "on"

    run_binding = load_run_binding(args.run_root)
    paths = canonical_run_paths(run_binding)
    manifest_path = Path(os.path.abspath(os.fspath(args.split_manifest)))
    shard_path = Path(os.path.abspath(os.fspath(args.calibration_shard)))
    if manifest_path != paths.split_dir / "split_manifest.json":
        raise SystemExit("split manifest is not the canonical run path")
    if shard_path != paths.split_dir / SHARD_FILENAMES["calibration"]:
        raise SystemExit("calibration shard is not the canonical run path")
    checkpoint_directory, checkpoint_file, receipt_path = (
        resolve_checkpoint_bundle(args.checkpoint)
    )
    if checkpoint_directory.parent not in {
        paths.seed_model(seed) for seed in (42, 43, 44)
    }:
        raise SystemExit("checkpoint is outside the canonical run models")
    output = Path(os.path.abspath(os.fspath(args.output)))
    if output != checkpoint_directory / "calibration_evaluation.json":
        raise SystemExit(
            "output must be the canonical epoch calibration receipt"
        )
    if os.path.lexists(output):
        raise SystemExit(f"calibration output already exists: {output}")

    receipt, receipt_binding = _load_receipt(receipt_path)
    if (
        receipt.get("source_tree_sha256")
        != run_binding.included_tree_sha256
        or receipt.get("source_tree_manifest_sha256")
        != run_binding.source_manifest_sha256
    ):
        raise SystemExit("training receipt differs from the canonical RID")
    _, rows, shard_binding = load_and_validate_role_shard(
        manifest_path,
        shard_path,
        role="calibration",
        expected_source_tree_sha256=run_binding.included_tree_sha256,
        expected_source_tree_manifest_sha256=(
            run_binding.source_manifest_sha256
        ),
        generator_path=(
            run_binding.source_root
            / "scripts"
            / "build_cmeie_split_manifest.py"
        ),
        expected_run_binding=run_binding,
    )
    if receipt.get("split_manifest_sha256") != shard_binding.get(
        "split_manifest_sha256"
    ):
        raise SystemExit("training receipt is bound to a different split")
    epoch, history, _ = verify_history_bundle(
        checkpoint_directory,
        checkpoint_file,
        receipt,
    )
    model, tokenizer, saved, device, bundle, load_evidence = _load_model(
        checkpoint_directory,
        checkpoint_file,
        receipt,
        history,
        device_name=args.device,
    )
    metric, chosen, chosen_schema, grid_size = _evaluate(
        rows,
        model=model,
        tokenizer=tokenizer,
        saved=saved,
        device=device,
        receipt=receipt,
        search_grid=args.search_declared_grid,
        thresholds=thresholds,
        schema_on=schema_on,
    )
    payload = {
        "schema_version": EVALUATION_RECEIPT_SCHEMA_VERSION,
        "kind": "cmeie_gplinker_calibration_evaluation",
        "dataset_role": "calibration",
        "selection_used": True,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "rid": run_binding.rid,
        "head": run_binding.git_head,
        "tree": run_binding.included_tree_sha256,
        "run_root": str(run_binding.run_root),
        "seed": receipt["seed"],
        "selected_epoch": epoch,
        "training_receipt_path": str(receipt_path),
        "training_receipt_sha256": receipt_binding["sha256"],
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": bundle["checkpoint_sha256"],
        "bundle_binding": bundle,
        "checkpoint_descriptor": load_evidence["checkpoint"],
        "contract_binding": load_evidence["contract"],
        "split_binding": shard_binding,
        "thresholds": list(chosen),
        "schema_constraint": "on" if chosen_schema else "off",
        "metric": metric,
        "grid_size": grid_size,
        "device": str(device),
    }
    publish_json_once(
        output,
        payload,
        label="CMeIE calibration evaluation",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
