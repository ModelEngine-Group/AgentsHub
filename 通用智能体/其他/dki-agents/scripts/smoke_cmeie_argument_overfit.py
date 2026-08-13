"""Strict fit-only overfit gate for the CMeIE argument pointer head."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXED_SAMPLE_SIZE = 8
FIXED_STEPS = 400
FIXED_HEAD_LR = 1e-3
FIXED_HEAD_CLIP = 5.0
FIXED_THRESHOLDS = (-0.5, 0.0, 0.5)
FIXED_F1_FLOOR = 0.95
FIXED_PRECISION_FLOOR = 0.95
FIXED_RECALL_FLOOR = 0.95
FIXED_DENSITY_CEILING = 2.0
FIXED_HEAD_REDUCTION = "balanced"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_sha256(row: dict[str, Any]) -> str:
    encoded = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_operating_point(
    sweep: list[dict[str, Any]],
) -> dict[str, Any]:
    if not sweep:
        raise ValueError("argument overfit threshold sweep is empty")
    return max(
        sweep,
        key=lambda item: (
            float(item["metric"]["f1"]),
            float(item["metric"]["precision"]),
            float(item["threshold"]),
        ),
    )


def overfit_gate(selection: dict[str, Any]) -> dict[str, Any]:
    metric = selection["metric"]
    gold_mentions = int(selection["gold_mentions"])
    predicted_mentions = int(selection["predicted_mentions"])
    density_ratio = predicted_mentions / max(1, gold_mentions)
    checks = {
        "f1": float(metric["f1"]) >= FIXED_F1_FLOOR,
        "precision": (
            float(metric["precision"]) >= FIXED_PRECISION_FLOOR
        ),
        "recall": float(metric["recall"]) >= FIXED_RECALL_FLOOR,
        "candidate_density": density_ratio <= FIXED_DENSITY_CEILING,
        "nonempty_gold": gold_mentions > 0,
        "nonempty_predictions": predicted_mentions > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "density_ratio": density_ratio,
        "floors": {
            "f1": FIXED_F1_FLOOR,
            "precision": FIXED_PRECISION_FLOOR,
            "recall": FIXED_RECALL_FLOOR,
            "candidate_density_ceiling": FIXED_DENSITY_CEILING,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fit-data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--source-tree-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=FIXED_SAMPLE_SIZE)
    parser.add_argument("--steps", type=int, default=FIXED_STEPS)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument(
        "--pointer-variant",
        choices=["efficient", "classic"],
        default="efficient",
    )
    parser.add_argument("--use-bilstm", action="store_true")
    parser.add_argument("--device", choices=["npu"], default="npu")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.source_tree_sha256):
        raise SystemExit("--source-tree-sha256 must be a full SHA-256")
    if args.seed != 42:
        raise SystemExit("argument overfit gate uses fixed seed 42")
    if args.sample_size != FIXED_SAMPLE_SIZE or args.steps != FIXED_STEPS:
        raise SystemExit("argument overfit sample size and steps are fixed")
    if args.max_length != 320:
        raise SystemExit("argument overfit max length is fixed at 320")

    from dki.nlp.cmeie_run_protocol import (
        canonical_run_paths,
        load_run_binding,
    )
    from dki.nlp.cmeie_split_protocol import (
        MANIFEST_FILENAME,
        SHARD_FILENAMES,
        load_and_validate_role_shard,
    )
    from scripts.train_argument_ner_cmeie_v2 import (
        _normalise_inputs,
        publish_json_create_once,
        sha256_tree,
    )

    run_binding = load_run_binding(args.run_root)
    run_paths = canonical_run_paths(run_binding)
    if ROOT.resolve(strict=True) != run_binding.source_root:
        raise SystemExit("argument overfit must execute from canonical source")
    if args.source_tree_sha256.lower() != run_binding.included_tree_sha256:
        raise SystemExit("argument overfit source tree differs from RID")
    if Path(os.path.abspath(args.source_tree_manifest)) != (
        run_binding.source_manifest
    ):
        raise SystemExit(
            "argument overfit source manifest is not canonical"
        )
    if args.fit_data.resolve(strict=True) != (
        run_paths.split_dir / SHARD_FILENAMES["fit"]
    ):
        raise SystemExit("argument overfit fit shard is not canonical")
    if args.split_manifest.resolve(strict=True) != (
        run_paths.split_dir / MANIFEST_FILENAME
    ):
        raise SystemExit("argument overfit split manifest is not canonical")
    if args.base_model.resolve(strict=True) != run_paths.base_model:
        raise SystemExit("argument overfit base model is not canonical")
    expected_output = run_paths.evidence_dir / "argument_overfit.json"
    if Path(os.path.abspath(args.output)) != expected_output:
        raise SystemExit("argument overfit output is not canonical")
    if os.path.lexists(args.output):
        raise SystemExit("argument overfit output already exists")
    # ``load_run_binding`` above has already re-read and descriptor-verified the
    # canonical manifest and every installed source file.  Do not weaken that
    # check by importing a second, non-existent verifier here.
    source_manifest_sha256 = sha256_file(args.source_tree_manifest)
    if source_manifest_sha256 != run_binding.source_manifest_sha256:
        raise SystemExit("argument overfit source manifest differs from RID")

    _, fit_rows, fit_binding = load_and_validate_role_shard(
        args.split_manifest,
        args.fit_data,
        role="fit",
        expected_source_tree_sha256=run_binding.included_tree_sha256,
        expected_source_tree_manifest_sha256=(
            run_binding.source_manifest_sha256
        ),
        generator_path=ROOT / "scripts" / "build_cmeie_split_manifest.py",
        expected_run_binding=run_binding,
    )
    if len(fit_rows) < FIXED_SAMPLE_SIZE:
        raise RuntimeError("fit shard is too small for argument overfit")
    sample = fit_rows[:FIXED_SAMPLE_SIZE]
    base_model_tree = sha256_tree(args.base_model)

    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModel, AutoTokenizer

    from dki.nlp.cmeie_argument import (
        build_labels,
        decode_mentions,
        gold_mentions,
        labels_from_rows,
        mention_metric,
    )
    from dki.nlp.cmeie_gplinker import pointer_group_loss
    from dki.nlp.globalpointer import EncoderGlobalPointer

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("argument overfit requires Ascend NPU")
    if int(torch.npu.device_count()) != 1 or int(torch.npu.current_device()) != 0:
        raise RuntimeError("argument overfit requires one visible logical NPU 0")
    for variable in (
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_DEVICE_ID",
        "DEVICE_ID",
    ):
        if os.getenv(variable) != "0":
            raise RuntimeError(f"{variable} must be exactly '0'")
    device = torch.device("npu:0")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)

    labels = labels_from_rows(fit_rows)
    if len(labels) != 11:
        raise RuntimeError("argument overfit requires exactly 11 fit types")
    type_to_id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    encoded = tokenizer(
        [row["text"] for row in sample],
        truncation=True,
        max_length=args.max_length,
        padding=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = [
        [tuple(pair) for pair in value.tolist()]
        for value in encoded.pop("offset_mapping")
    ]
    dense_labels = []
    dropped = 0
    for row, row_offsets in zip(sample, offsets):
        dense, lost = build_labels(row, row_offsets, type_to_id)
        dense_labels.append(dense)
        dropped += lost
    if dropped:
        raise RuntimeError("argument overfit sample contains dropped labels")
    dense_labels = torch.stack(dense_labels).to(device)
    inputs = _normalise_inputs(encoded, device)

    encoder = AutoModel.from_pretrained(args.base_model).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        output = encoder(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **(
                {"token_type_ids": inputs["token_type_ids"]}
                if "token_type_ids" in inputs
                else {}
            ),
        )
        hidden = (
            output.last_hidden_state
            if hasattr(output, "last_hidden_state")
            else output[0]
        ).detach()
    del encoder, output

    span_model = EncoderGlobalPointer(
        torch.nn.Identity(),
        int(hidden.shape[-1]),
        len(labels),
        inner_dim=64,
        pointer=args.pointer_variant,
        use_bilstm=args.use_bilstm,
    ).to(device)
    head_parameters = list(span_model.gp.parameters())
    if span_model.bilstm is not None:
        head_parameters.extend(span_model.bilstm.parameters())
    optimiser = torch.optim.AdamW(
        head_parameters,
        lr=FIXED_HEAD_LR,
        weight_decay=0.01,
    )
    scaler = torch.npu.amp.GradScaler(init_scale=1024.0, enabled=True)
    skipped_steps = 0
    nonfinite_grad_norm_steps = 0
    maximum_grad_norm = 0.0
    final_loss = None
    span_model.train()

    def head_logits():
        features = hidden
        if span_model.bilstm is not None:
            features, _ = span_model.bilstm(features)
            features = features * inputs["attention_mask"].unsqueeze(-1).to(
                dtype=features.dtype
            )
        return span_model.gp(features, inputs["attention_mask"])

    for step in range(1, FIXED_STEPS + 1):
        optimiser.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="npu",
            dtype=torch.float16,
            enabled=True,
        ):
            logits = head_logits()
            loss = pointer_group_loss(
                logits,
                dense_labels,
                head_reduction=FIXED_HEAD_REDUCTION,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite argument overfit loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            head_parameters,
            FIXED_HEAD_CLIP,
        )
        grad_norm_value = float(grad_norm.detach().cpu())
        if math.isfinite(grad_norm_value):
            maximum_grad_norm = max(maximum_grad_norm, grad_norm_value)
        else:
            nonfinite_grad_norm_steps += 1
        scale_before = float(scaler.get_scale())
        scaler.step(optimiser)
        scaler.update()
        if float(scaler.get_scale()) < scale_before:
            skipped_steps += 1
        final_loss = float(loss.detach().cpu())
        if step in {1, 100, 200, 300, 400}:
            print(
                json.dumps(
                    {
                        "event": "cmeie_argument_overfit_progress",
                        "step": step,
                        "steps": FIXED_STEPS,
                        "loss": final_loss,
                        "head_pre_clip_grad_norm": grad_norm_value,
                        "amp_scale": float(scaler.get_scale()),
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )

    span_model.eval()
    with torch.no_grad(), torch.autocast(
        device_type="npu",
        dtype=torch.float16,
        enabled=True,
    ):
        logits = head_logits().float().cpu()
    gold = [gold_mentions(row) for row in sample]
    sweep = []
    for threshold in FIXED_THRESHOLDS:
        predicted = [
            decode_mentions(
                row["text"],
                row_offsets,
                logits[index],
                labels,
                threshold,
            )
            for index, (row, row_offsets) in enumerate(zip(sample, offsets))
        ]
        sweep.append(
            {
                "threshold": threshold,
                "metric": mention_metric(predicted, gold),
                "predicted_mentions": sum(len(value) for value in predicted),
                "gold_mentions": sum(len(value) for value in gold),
            }
        )
    selected = select_operating_point(sweep)
    gate = overfit_gate(selected)
    if skipped_steps or nonfinite_grad_norm_steps:
        gate["passed"] = False
        gate["checks"]["optimizer_integrity"] = False
    else:
        gate["checks"]["optimizer_integrity"] = True

    payload = {
        "schema_version": 2,
        "kind": "cmeie_argument_pointer_overfit_preflight",
        "dataset_role": "fit_training_diagnostic",
        "selection_used": False,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "run_binding": {
            "rid": run_binding.rid,
            "git_head": run_binding.git_head,
            "source_tree_sha256": run_binding.included_tree_sha256,
            "source_manifest_sha256": run_binding.source_manifest_sha256,
        },
        "fit_binding": fit_binding,
        "base_model_tree": base_model_tree,
        "sample_size": FIXED_SAMPLE_SIZE,
        "sample_row_sha256": [row_sha256(row) for row in sample],
        "steps": FIXED_STEPS,
        "max_length": args.max_length,
        "labels": labels,
        "n_labels": len(labels),
        "dropped_label_count": dropped,
        "head_reduction": FIXED_HEAD_REDUCTION,
        "pointer_variant": args.pointer_variant,
        "use_bilstm": args.use_bilstm,
        "head_learning_rate": FIXED_HEAD_LR,
        "head_gradient_clip": FIXED_HEAD_CLIP,
        "threshold_sweep": sweep,
        "selected_operating_point": selected,
        "gate": gate,
        "final_loss": final_loss,
        "maximum_head_pre_clip_grad_norm": maximum_grad_norm,
        "skipped_optimizer_steps": skipped_steps,
        "nonfinite_grad_norm_steps": nonfinite_grad_norm_steps,
        "runtime": {
            "device": str(device),
            "device_name": str(torch.npu.get_device_name(0)),
            "visible_device_count": int(torch.npu.device_count()),
            "current_device": int(torch.npu.current_device()),
            "amp_initial_scale": 1024.0,
            "amp_final_scale": float(scaler.get_scale()),
        },
        "source_bindings": {
            "overfit_script_sha256": sha256_file(Path(__file__)),
            "training_script_sha256": sha256_file(
                ROOT / "scripts" / "train_argument_ner_cmeie_v2.py"
            ),
            "argument_module_sha256": sha256_file(
                ROOT / "dki" / "nlp" / "cmeie_argument.py"
            ),
            "globalpointer_module_sha256": sha256_file(
                ROOT / "dki" / "nlp" / "globalpointer.py"
            ),
            "balanced_loss_module_sha256": sha256_file(
                ROOT / "dki" / "nlp" / "cmeie_gplinker.py"
            ),
        },
        "contains_text_gold_or_predictions": False,
    }
    publish_json_create_once(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": gate["passed"],
                "selected_operating_point": selected,
                "outer_accessed": False,
                "official_dev_accessed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
