"""Run the only authorized CMeIE outer and official-dev evaluations.

The ``outer`` command consumes all three frozen calibration selections and
opens the physical outer shard exactly once after the RID-level create-once
access marker is durable.  The ``official-dev-confirmation`` command consumes
the passed outer gate and opens the expected official development file once
after its own marker is durable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dki.evidence_io import EvidenceBinding, load_bound_json, read_bound_bytes
from dki.nlp.cmeie_joint_evaluation import (
    ALL_SEEDS,
    JointEvaluationError,
    OFFICIAL_CONFIRMATION_KIND,
    OFFICIAL_CONFIRMATION_SCHEMA_VERSION,
    OFFICIAL_DEV_EXPECTED_ROWS,
    OUTER_BATCH_KIND,
    OUTER_BATCH_SCHEMA_VERSION,
    OUTER_MEDIAN_F1_FLOOR,
    OUTER_PRECISION_RECALL_FLOOR,
    OUTER_WORST_F1_FLOOR,
    evaluate_one_model,
    evaluate_rows_once,
    exact_outer_gate_observation,
    load_and_validate_started_marker,
    load_canonical_completion,
    load_predictor,
    official_metric_pass,
    outer_metrics_pass,
    publish_json_once,
    ratio_payload,
    selection_evidence,
    source_member_binding,
    validate_metric,
    validate_selection_set,
)
from dki.nlp.cmeie_preflight_protocol import (
    validate_cmeie_npu_preflight_receipt,
)
from dki.nlp.cmeie_run_protocol import (
    OFFICIAL_DEV_DATASET_ROLE,
    OUTER_DATASET_ROLE,
    RunBinding,
    canonical_run_paths,
    load_run_binding,
    publish_dataset_started_once,
)
from dki.nlp.cmeie_selection_protocol import (
    canonical_selection_path,
    load_and_validate_selection_receipt,
)
from dki.nlp.cmeie_split_protocol import (
    SCHEMA_VERSION as SPLIT_SCHEMA_VERSION,
    SHARD_FILENAMES,
    _validate_manifest_envelope,
    load_and_validate_role_shard,
    parse_cmeie_jsonl_bytes,
)
from dki.nlp.cmeie_spo import gold_spo_set


OUTER_GATE_SCHEMA_VERSION = 2
OUTER_GATE_KIND = "cmeie_gplinker_seed_outer_gate"
_SOURCE_MEMBERS = (
    "dki/nlp/cmeie_joint_evaluation.py",
    "dki/nlp/cmeie_split_protocol.py",
    "scripts/build_cmeie_split_manifest.py",
    "scripts/eval_spo_cmeie_batch.py",
)


def _absolute(path: Path, *, label: str) -> Path:
    raw = os.fspath(path)
    if not raw or "\0" in raw:
        raise JointEvaluationError(f"{label} must be a non-empty path")
    if any(part in {".", ".."} for part in raw.replace("\\", "/").split("/")):
        raise JointEvaluationError(
            f"{label} must not contain lexical traversal"
        )
    result = Path(os.path.abspath(raw))
    if not result.is_absolute():
        raise JointEvaluationError(f"{label} must be absolute")
    return result


def _require_canonical(
    supplied: Path,
    expected: Path,
    *,
    label: str,
) -> Path:
    actual = _absolute(supplied, label=label)
    if actual != expected:
        raise JointEvaluationError(
            f"{label} must equal the run_protocol canonical path {expected}"
        )
    return actual


def _selection_inputs(
    run_binding: RunBinding,
    paths: Sequence[Path],
) -> tuple[
    dict[int, Mapping[str, Any]],
    dict[int, Path],
    dict[int, EvidenceBinding],
    dict[str, Any],
]:
    if len(paths) != len(ALL_SEEDS):
        raise JointEvaluationError(
            "outer requires exactly three selection receipts"
        )
    payloads: list[Mapping[str, Any]] = []
    by_path: dict[int, Path] = {}
    bindings: dict[int, EvidenceBinding] = {}
    for index, supplied in enumerate(paths):
        candidates = [
            seed
            for seed in ALL_SEEDS
            if _absolute(supplied, label=f"selection[{index}]")
            == canonical_selection_path(run_binding, seed)
        ]
        if len(candidates) != 1:
            raise JointEvaluationError(
                f"selection[{index}] is not a canonical seed selection"
            )
        seed = candidates[0]
        path = canonical_selection_path(run_binding, seed)
        payload = load_and_validate_selection_receipt(
            path,
            run_binding=run_binding,
        )
        rebound, binding = load_bound_json(
            path,
            f"CMeIE seed {seed} selection",
            reject_multiple_links=True,
            require_object=True,
        )
        if rebound != payload:
            raise JointEvaluationError(
                f"seed {seed} selection changed after canonical validation"
            )
        payloads.append(payload)
        by_path[seed] = path
        bindings[seed] = binding
    selections, common = validate_selection_set(payloads)
    return selections, by_path, bindings, common


def _expected_outer_manifest(
    run_binding: RunBinding,
    manifest_path: Path,
) -> tuple[dict[str, Any], EvidenceBinding, dict[str, Any]]:
    manifest, binding = load_bound_json(
        manifest_path,
        "CMeIE outer split manifest",
        reject_multiple_links=True,
        require_object=True,
    )
    _, envelope = _validate_manifest_envelope(
        manifest,
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
    outer = manifest.get("splits", {}).get("outer")
    if type(outer) is not dict or type(outer.get("shard")) is not dict:
        raise JointEvaluationError(
            "split manifest lacks the declared outer shard"
        )
    return manifest, binding, envelope


def _preflight_binding(
    run_binding: RunBinding,
    common: Mapping[str, Any],
) -> dict[str, Any]:
    paths = canonical_run_paths(run_binding)
    return validate_cmeie_npu_preflight_receipt(
        str(paths.preflight_receipt),
        expected_source_tree_sha256=run_binding.included_tree_sha256,
        expected_source_manifest_sha256=run_binding.source_manifest_sha256,
        expected_cmeie_train_sha256=str(common["train_sha256"]),
        expected_base_model_canonical_path=str(common["base_model"]),
        expected_base_model_tree_sha256=str(
            common["base_model_tree_sha256"]
        ),
        expected_run_binding={
            "rid": run_binding.rid,
            "git_head": run_binding.git_head,
            "source_tree_sha256": run_binding.included_tree_sha256,
            "source_manifest_sha256": run_binding.source_manifest_sha256,
        },
    )


def _source_bindings(run_binding: RunBinding) -> dict[str, Any]:
    return {
        item["relative_path"]: item
        for item in (
            source_member_binding(run_binding, relative)
            for relative in _SOURCE_MEMBERS
        )
    }


def _model_evidence(
    selections: Mapping[int, Mapping[str, Any]],
    predictors: Sequence[Any],
    metrics: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_seed = {predictor.seed: predictor for predictor in predictors}
    return [
        {
            "seed": seed,
            "selected_epoch": selections[seed]["selected_epoch"],
            "checkpoint": selections[seed]["checkpoint"],
            "checkpoint_sha256": selections[seed]["checkpoint_sha256"],
            "checkpoint_descriptor": dict(
                by_seed[seed].checkpoint_binding
            ),
            "checkpoint_bundle_manifest_sha256": selections[seed][
                "checkpoint_bundle_manifest_sha256"
            ],
            "checkpoint_bundle_tree_sha256": selections[seed][
                "checkpoint_bundle_tree_sha256"
            ],
            "bundle_binding": by_seed[seed].bundle_binding,
            "extraction_contract_sha256": selections[seed][
                "extraction_contract_sha256"
            ],
            "semantic_contract_sha256": selections[seed][
                "semantic_contract_sha256"
            ],
            "contract_binding": by_seed[seed].contract_binding,
            "thresholds": selections[seed]["thresholds"],
            "schema_constraint": selections[seed]["schema_constraint"],
            "metric": metrics[seed],
        }
        for seed in ALL_SEEDS
    ]


def run_outer(
    *,
    run_root: Path,
    selection_paths: Sequence[Path],
    split_manifest_path: Path,
    outer_shard_path: Path,
    device: str,
    predictor_loader: Callable[..., Any] = load_predictor,
    role_loader: Callable[..., Any] = load_and_validate_role_shard,
) -> tuple[dict[str, Any], EvidenceBinding]:
    """Perform one marker-guarded, three-seed outer evaluation."""

    run_binding = load_run_binding(run_root)
    paths = canonical_run_paths(run_binding)
    manifest_path = _require_canonical(
        split_manifest_path,
        paths.split_dir / "split_manifest.json",
        label="split manifest",
    )
    shard_path = _require_canonical(
        outer_shard_path,
        paths.split_dir / SHARD_FILENAMES["outer"],
        label="outer shard",
    )
    if os.path.lexists(paths.outer_completion):
        raise FileExistsError(
            f"outer completion already exists: {paths.outer_completion}"
        )

    selections, selected_paths, selected_bindings, common = (
        _selection_inputs(run_binding, selection_paths)
    )
    if common["split_manifest_sha256"] is None:
        raise JointEvaluationError(
            "selection common binding lacks split manifest SHA-256"
        )
    _, manifest_binding, _ = _expected_outer_manifest(
        run_binding,
        manifest_path,
    )
    if (
        manifest_binding["sha256"]
        != common["split_manifest_sha256"]
    ):
        raise JointEvaluationError(
            "split manifest differs from the three selections"
        )
    declared_shard = load_bound_json(
        manifest_path,
        "CMeIE outer split manifest declaration",
        reject_multiple_links=True,
        require_object=True,
    )[0]["splits"]["outer"]["shard"]
    dataset_sha = str(declared_shard["sha256"])

    # Model/environment failures are resolved before consuming the outer shard.
    predictors = [
        predictor_loader(selections[seed], device_name=device)
        for seed in ALL_SEEDS
    ]
    selection_entries = selection_evidence(
        selections,
        selected_paths,
        selected_bindings,
    )
    marker_evidence = {
        "schema_version": 1,
        "kind": "cmeie_outer_batch_inputs",
        "run_root": str(run_binding.run_root),
        "split_manifest_path": str(manifest_path),
        "split_manifest_sha256": manifest_binding["sha256"],
        "outer_shard_path": str(shard_path),
        "outer_shard_sha256": dataset_sha,
        "selection_receipts": selection_entries,
    }
    publish_dataset_started_once(
        run_binding,
        OUTER_DATASET_ROLE,
        dataset_sha,
        str(manifest_binding["sha256"]),
        marker_evidence,
    )
    marker, marker_binding = load_and_validate_started_marker(
        run_binding,
        role=OUTER_DATASET_ROLE,
        dataset_sha256=dataset_sha,
        split_sha256=str(manifest_binding["sha256"]),
    )

    # This role validator performs the first and only open/read of the shard.
    _, rows, shard_binding = role_loader(
        manifest_path,
        shard_path,
        role="outer",
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
    if (
        shard_binding.get("shard_sha256") != dataset_sha
        or shard_binding.get("split_manifest_sha256")
        != manifest_binding["sha256"]
    ):
        raise JointEvaluationError(
            "validated outer shard differs from its marker"
        )
    metrics = evaluate_rows_once(
        rows,
        predictors,
        gold_fn=gold_spo_set,
    )
    payload = {
        "schema_version": OUTER_BATCH_SCHEMA_VERSION,
        "kind": OUTER_BATCH_KIND,
        "dataset_role": OUTER_DATASET_ROLE,
        "outer_accessed": True,
        "official_dev_accessed": False,
        "selection_used": False,
        "rid": run_binding.rid,
        "head": run_binding.git_head,
        "tree": run_binding.included_tree_sha256,
        "run_root": str(run_binding.run_root),
        "source_manifest_path": str(run_binding.source_manifest),
        "source_manifest_sha256": run_binding.source_manifest_sha256,
        "split_manifest_path": str(manifest_path),
        "split_manifest_sha256": manifest_binding["sha256"],
        "dataset": shard_binding,
        "dataset_access_marker": {
            "path": str(paths.outer_started),
            "sha256": marker_binding["sha256"],
            "dataset_key_sha256": marker["dataset_key_sha256"],
        },
        "selection_receipts": selection_entries,
        "common_bindings": common,
        "models": _model_evidence(
            selections,
            predictors,
            metrics,
        ),
        "source_bindings": _source_bindings(run_binding),
        "device": device,
        "single_dataset_read": True,
        "same_row_loop_for_all_seeds": True,
        "completed": True,
    }
    binding = publish_json_once(
        paths.outer_completion,
        payload,
        label="CMeIE outer batch completion",
    )
    return payload, binding


def _selection_calibration_metric(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    """Recompute one calibration metric from its immutable integer counts."""

    if type(value) is not dict:
        raise JointEvaluationError(f"{label} must be an object")
    counts: dict[str, int] = {}
    for field in ("tp", "fp", "fn"):
        item = value.get(field)
        if type(item) is not int or item < 0:
            raise JointEvaluationError(
                f"{label}.{field} must be a non-negative integer"
            )
        counts[field] = item
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    exact = {
        "precision": Fraction(tp, tp + fp) if tp + fp else Fraction(0),
        "recall": Fraction(tp, tp + fn) if tp + fn else Fraction(0),
        "f1": (
            Fraction(2 * tp, 2 * tp + fp + fn)
            if 2 * tp + fp + fn
            else Fraction(0)
        ),
    }
    for field, expected in exact.items():
        reported = value.get(field)
        if (
            isinstance(reported, bool)
            or not isinstance(reported, (int, float))
            or not math.isfinite(float(reported))
            or abs(Fraction(str(reported)) - expected)
            > Fraction(1, 10**12)
        ):
            raise JointEvaluationError(
                f"{label}.{field} disagrees with integer counts"
            )
    return {**counts, **exact}


def _calibration_metrics(
    selections: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {
        seed: _selection_calibration_metric(
            selections[seed].get("calibration_metric"),
            label=f"selection seed {seed} calibration metric",
        )
        for seed in ALL_SEEDS
    }


def _calibration_metric_evidence(
    metrics: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "counts": {
                field: metrics[seed][field]
                for field in ("tp", "fp", "fn")
            },
            "precision": ratio_payload(metrics[seed]["precision"]),
            "recall": ratio_payload(metrics[seed]["recall"]),
            "f1": ratio_payload(metrics[seed]["f1"]),
        }
        for seed in ALL_SEEDS
    ]


def _calibration_choice(
    metrics: Mapping[int, Mapping[str, Any]],
) -> int:
    """Use calibration F1, precision, then ascending seed only."""

    return max(
        ALL_SEEDS,
        key=lambda seed: (
            metrics[seed]["f1"],
            metrics[seed]["precision"],
            -seed,
        ),
    )


def _batch_outer_metrics(
    batch: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    models = batch.get("models")
    if type(models) is not list or len(models) != len(ALL_SEEDS):
        raise JointEvaluationError(
            "canonical outer batch must contain exactly three model metrics"
        )
    metrics: dict[int, dict[str, Any]] = {}
    for index, model in enumerate(models):
        if type(model) is not dict:
            raise JointEvaluationError(
                f"canonical outer batch model[{index}] is invalid"
            )
        seed = model.get("seed")
        if type(seed) is not int or seed not in ALL_SEEDS or seed in metrics:
            raise JointEvaluationError(
                f"canonical outer batch model[{index}] has an invalid seed"
            )
        metrics[seed] = validate_metric(
            model.get("metric"),
            label=f"canonical outer batch seed {seed} metric",
        )
    if tuple(sorted(metrics)) != ALL_SEEDS:
        raise JointEvaluationError(
            "canonical outer batch metrics must be seeds 42, 43, and 44"
        )
    return metrics


def _validate_outer_batch_for_official(
    run_binding: RunBinding,
    *,
    selections: Mapping[int, Mapping[str, Any]],
    selection_paths: Mapping[int, Path],
    selection_bindings: Mapping[int, EvidenceBinding],
    common: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    EvidenceBinding,
    dict[int, dict[str, Any]],
    dict[str, Any],
    EvidenceBinding,
]:
    """Reopen and semantically verify the canonical outer completion."""

    paths = canonical_run_paths(run_binding)
    batch, batch_binding = load_canonical_completion(
        run_binding,
        role=OUTER_DATASET_ROLE,
        expected_kind=OUTER_BATCH_KIND,
        expected_schema_version=OUTER_BATCH_SCHEMA_VERSION,
    )
    expected_selections = selection_evidence(
        selections,
        selection_paths,
        selection_bindings,
    )
    if (
        batch.get("outer_accessed") is not True
        or batch.get("official_dev_accessed") is not False
        or batch.get("selection_used") is not False
        or batch.get("single_dataset_read") is not True
        or batch.get("same_row_loop_for_all_seeds") is not True
        or batch.get("completed") is not True
        or batch.get("common_bindings") != common
        or batch.get("selection_receipts") != expected_selections
    ):
        raise JointEvaluationError(
            "canonical outer batch common or selection binding is invalid"
        )

    dataset = batch.get("dataset")
    if type(dataset) is not dict:
        raise JointEvaluationError(
            "canonical outer batch dataset binding is invalid"
        )
    dataset_sha = dataset.get("shard_sha256")
    split_sha = batch.get("split_manifest_sha256")
    if (
        type(dataset_sha) is not str
        or type(split_sha) is not str
        or split_sha != common.get("split_manifest_sha256")
        or dataset.get("validated") is not True
        or dataset.get("role") != "outer"
        or dataset.get("outer_accessed") is not True
        or dataset.get("source_train_accessed") is not False
        or dataset.get("split_schema_version") != SPLIT_SCHEMA_VERSION
        or dataset.get("rid") != run_binding.rid
        or dataset.get("git_head") != run_binding.git_head
        or dataset.get("canonical_run_root") != str(run_binding.run_root)
        or dataset.get("source_tree_sha256")
        != run_binding.included_tree_sha256
        or dataset.get("source_tree_manifest_sha256")
        != run_binding.source_manifest_sha256
        or dataset.get("source_train_sha256") != common.get("train_sha256")
        or dataset.get("shard_relative_path")
        != SHARD_FILENAMES["outer"]
        or batch.get("split_manifest_path")
        != str(paths.split_dir / "split_manifest.json")
    ):
        raise JointEvaluationError(
            "canonical outer batch dataset identity is invalid"
        )

    marker, marker_binding = load_and_validate_started_marker(
        run_binding,
        role=OUTER_DATASET_ROLE,
        dataset_sha256=dataset_sha,
        split_sha256=split_sha,
    )
    expected_marker_evidence = {
        "schema_version": 1,
        "kind": "cmeie_outer_batch_inputs",
        "run_root": str(run_binding.run_root),
        "split_manifest_path": batch["split_manifest_path"],
        "split_manifest_sha256": split_sha,
        "outer_shard_path": str(
            paths.split_dir / SHARD_FILENAMES["outer"]
        ),
        "outer_shard_sha256": dataset_sha,
        "selection_receipts": expected_selections,
    }
    if marker.get("input_model_evidence") != expected_marker_evidence:
        raise JointEvaluationError(
            "canonical outer batch access marker evidence is invalid"
        )
    recorded_marker = batch.get("dataset_access_marker")
    if (
        type(recorded_marker) is not dict
        or recorded_marker.get("path") != str(paths.outer_started)
        or recorded_marker.get("sha256") != marker_binding.get("sha256")
        or recorded_marker.get("dataset_key_sha256")
        != marker.get("dataset_key_sha256")
    ):
        raise JointEvaluationError(
            "canonical outer batch access marker binding is invalid"
        )

    metrics = _batch_outer_metrics(batch)
    model_entries = {
        model["seed"]: model
        for model in batch["models"]
        if type(model) is dict and model.get("seed") in ALL_SEEDS
    }
    for seed in ALL_SEEDS:
        model = model_entries.get(seed)
        selection = selections[seed]
        if (
            type(model) is not dict
            or model.get("selected_epoch") != selection["selected_epoch"]
            or model.get("checkpoint") != selection["checkpoint"]
            or model.get("checkpoint_sha256")
            != selection["checkpoint_sha256"]
            or model.get("checkpoint_bundle_manifest_sha256")
            != selection["checkpoint_bundle_manifest_sha256"]
            or model.get("checkpoint_bundle_tree_sha256")
            != selection["checkpoint_bundle_tree_sha256"]
            or model.get("extraction_contract_sha256")
            != selection["extraction_contract_sha256"]
            or model.get("semantic_contract_sha256")
            != selection["semantic_contract_sha256"]
            or model.get("thresholds") != selection["thresholds"]
            or model.get("schema_constraint")
            != selection["schema_constraint"]
            or model.get("bundle_binding") != selection["bundle_binding"]
            or model.get("contract_binding")
            != selection["contract_binding"]
        ):
            raise JointEvaluationError(
                f"canonical outer batch seed {seed} model binding is invalid"
            )

    expected_sources = _source_bindings(run_binding)
    if batch.get("source_bindings") != expected_sources:
        raise JointEvaluationError(
            "canonical outer batch evaluator source binding is invalid"
        )
    if (
        dataset.get("validator_sha256")
        != expected_sources["dki/nlp/cmeie_split_protocol.py"]["sha256"]
        or dataset.get("generator_sha256")
        != expected_sources["scripts/build_cmeie_split_manifest.py"][
            "sha256"
        ]
    ):
        raise JointEvaluationError(
            "canonical outer batch split source binding is invalid"
        )
    return batch, batch_binding, metrics, marker, marker_binding


def _revalidate_selection_snapshots(
    selections: Mapping[int, Mapping[str, Any]],
    selection_paths: Mapping[int, Path],
    selection_bindings: Mapping[int, EvidenceBinding],
) -> None:
    for seed in ALL_SEEDS:
        rebound, binding = load_bound_json(
            selection_paths[seed],
            f"CMeIE seed {seed} selection final rebind",
            reject_multiple_links=True,
            require_object=True,
        )
        if rebound != selections[seed] or binding != selection_bindings[seed]:
            raise JointEvaluationError(
                f"seed {seed} selection changed during outer gate validation"
            )


def _load_outer_gate(
    run_binding: RunBinding,
    *,
    selections: Mapping[int, Mapping[str, Any]],
    selection_paths: Mapping[int, Path],
    selection_bindings: Mapping[int, EvidenceBinding],
    common: Mapping[str, Any],
    requested_seed: int,
    preflight: Mapping[str, Any],
) -> tuple[dict[str, Any], EvidenceBinding]:
    """Recompute every gate decision from canonical upstream evidence."""

    paths = canonical_run_paths(run_binding)
    path = paths.outer_gate
    gate, gate_binding = load_bound_json(
        path,
        "CMeIE outer gate",
        reject_multiple_links=True,
        require_object=True,
    )
    batch, batch_binding, outer_metrics, marker, marker_binding = (
        _validate_outer_batch_for_official(
            run_binding,
            selections=selections,
            selection_paths=selection_paths,
            selection_bindings=selection_bindings,
            common=common,
        )
    )
    if not outer_metrics_pass(outer_metrics):
        raise JointEvaluationError(
            "canonical outer batch does not pass the exact outer gate"
        )

    calibration_metrics = _calibration_metrics(selections)
    chosen_seed = _calibration_choice(calibration_metrics)
    if requested_seed != chosen_seed:
        raise JointEvaluationError(
            "requested official candidate differs from the calibration choice"
        )
    chosen_selection = selections[chosen_seed]
    chosen_binding = selection_bindings[chosen_seed]
    expected_choice = {
        "seed": chosen_seed,
        "tie_break": [
            "calibration_f1_desc",
            "calibration_precision_desc",
            "seed_asc",
        ],
        "selection_receipt_sha256": chosen_binding["sha256"],
        "checkpoint_sha256": chosen_selection["checkpoint_sha256"],
        "checkpoint_bundle_manifest_sha256": chosen_selection[
            "checkpoint_bundle_manifest_sha256"
        ],
        "checkpoint_bundle_tree_sha256": chosen_selection[
            "checkpoint_bundle_tree_sha256"
        ],
    }
    expected_selection_evidence = selection_evidence(
        selections,
        selection_paths,
        selection_bindings,
    )
    expected_outer_metrics = [
        {"seed": seed, **outer_metrics[seed]}
        for seed in ALL_SEEDS
    ]
    expected_calibration_metrics = _calibration_metric_evidence(
        calibration_metrics
    )
    expected_thresholds = {
        "median_f1_minimum": ratio_payload(OUTER_MEDIAN_F1_FLOOR),
        "worst_f1_minimum": ratio_payload(OUTER_WORST_F1_FLOOR),
        "per_seed_precision_minimum": ratio_payload(
            OUTER_PRECISION_RECALL_FLOOR
        ),
        "per_seed_recall_minimum": ratio_payload(
            OUTER_PRECISION_RECALL_FLOOR
        ),
    }
    observation = exact_outer_gate_observation(outer_metrics)
    expected_observed = {
        "median_f1": observation["median_f1"],
        "worst_f1": observation["worst_f1"],
    }
    early_path = paths.evidence_dir / "early_gate.json"
    early, early_binding = load_bound_json(
        early_path,
        "CMeIE early gate reference",
        reject_multiple_links=True,
        require_object=True,
    )
    expected_early_reference = {
        "path": str(early_path),
        "sha256": early_binding["sha256"],
        "schema_version": early.get("schema_version"),
    }
    expected_batch_reference = {
        "path": str(paths.outer_completion),
        "sha256": batch_binding["sha256"],
        "schema_version": batch["schema_version"],
    }
    expected_marker_reference = {
        "path": str(paths.outer_started),
        "sha256": marker_binding["sha256"],
        "dataset_key_sha256": marker["dataset_key_sha256"],
    }
    gate_source = source_member_binding(
        run_binding,
        "scripts/gate_spo_cmeie_seeds.py",
    )
    checks = {
        "schema_version": gate.get("schema_version")
        == OUTER_GATE_SCHEMA_VERSION,
        "kind": gate.get("kind") == OUTER_GATE_KIND,
        "dataset_role": gate.get("dataset_role") == "outer_gate",
        "selection_used": gate.get("selection_used") is False,
        "outer_accessed": gate.get("outer_accessed") is True,
        "official_dev_accessed": gate.get("official_dev_accessed") is False,
        "rid": gate.get("rid") == run_binding.rid,
        "head": gate.get("head") == run_binding.git_head,
        "tree": gate.get("tree") == run_binding.included_tree_sha256,
        "run_root": gate.get("run_root") == str(run_binding.run_root),
        "expected_seeds": gate.get("expected_seeds") == list(ALL_SEEDS),
        "early_gate_receipt": gate.get("early_gate_receipt")
        == expected_early_reference,
        "selection_receipts": gate.get("selection_receipts")
        == expected_selection_evidence,
        "outer_batch_receipt": gate.get("outer_batch_receipt")
        == expected_batch_reference,
        "outer_access_marker": gate.get("outer_access_marker")
        == expected_marker_reference,
        "preflight_receipt_path": gate.get("preflight_receipt_path")
        == preflight.get("preflight_receipt_path"),
        "preflight_receipt_sha256": gate.get("preflight_receipt_sha256")
        == preflight.get("preflight_receipt_sha256"),
        "preflight_binding": gate.get("preflight_binding") == preflight,
        "expected_official_dev_sha256": gate.get(
            "expected_official_dev_sha256"
        )
        == preflight.get("cmeie_dev_sha256"),
        "expected_official_dev_rows": gate.get(
            "expected_official_dev_rows"
        )
        == preflight.get("cmeie_dev_rows")
        == OFFICIAL_DEV_EXPECTED_ROWS,
        "common_bindings": gate.get("common_bindings") == common,
        "calibration_metrics": gate.get("calibration_metrics")
        == expected_calibration_metrics,
        "outer_metrics": gate.get("outer_metrics")
        == expected_outer_metrics,
        "thresholds": gate.get("thresholds") == expected_thresholds,
        "observed": gate.get("observed") == expected_observed,
        "calibration_choice": gate.get("calibration_choice")
        == expected_choice,
        "passed": gate.get("passed") is True,
        "decision": gate.get("decision") == "official_dev_authorized",
        "official_dev_authorized": gate.get(
            "official_dev_authorized"
        )
        is True,
        "authorized_candidate": gate.get("authorized_candidate")
        == expected_choice,
        "gate_script_sha256": gate.get("gate_script_sha256")
        == gate_source["sha256"],
    }
    mismatches = sorted(
        field for field, accepted in checks.items() if not accepted
    )
    if mismatches:
        raise JointEvaluationError(
            "outer gate disagrees with canonical evidence: "
            + ", ".join(mismatches)
        )
    _revalidate_selection_snapshots(
        selections,
        selection_paths,
        selection_bindings,
    )
    return gate, gate_binding


def _parse_official_dev_once(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], EvidenceBinding]:
    raw, binding = read_bound_bytes(
        path,
        "CMeIE official development dataset",
        reject_multiple_links=True,
    )
    if binding.get("sha256") != expected_sha256:
        raise JointEvaluationError(
            "official development dataset SHA-256 differs from preflight"
        )
    rows, _ = parse_cmeie_jsonl_bytes(
        raw,
        description="CMeIE official development dataset",
    )
    if len(rows) != OFFICIAL_DEV_EXPECTED_ROWS:
        raise JointEvaluationError(
            "official development dataset must contain exactly "
            f"{OFFICIAL_DEV_EXPECTED_ROWS} rows"
        )
    return rows, binding


def run_official_dev_confirmation(
    *,
    run_root: Path,
    selection_path: Path,
    preflight_receipt_path: Path,
    outer_gate_receipt_path: Path,
    official_dev_path: Path,
    device: str,
    predictor_loader: Callable[..., Any] = load_predictor,
    dataset_loader: Callable[..., Any] = _parse_official_dev_once,
) -> tuple[dict[str, Any], EvidenceBinding, bool]:
    """Perform the one allowed official development confirmation."""

    run_binding = load_run_binding(run_root)
    paths = canonical_run_paths(run_binding)
    _require_canonical(
        preflight_receipt_path,
        paths.preflight_receipt,
        label="preflight receipt",
    )
    _require_canonical(
        outer_gate_receipt_path,
        paths.outer_gate,
        label="outer gate receipt",
    )
    if os.path.lexists(paths.official_dev_completion):
        raise FileExistsError(
            "official development confirmation completion already exists"
        )

    supplied_selection = _absolute(
        selection_path,
        label="chosen selection",
    )
    seed_matches = [
        seed
        for seed in ALL_SEEDS
        if supplied_selection == canonical_selection_path(run_binding, seed)
    ]
    if len(seed_matches) != 1:
        raise JointEvaluationError(
            "chosen selection is not a canonical seed selection"
        )
    requested_seed = seed_matches[0]
    (
        selections,
        selection_paths,
        selection_bindings,
        common,
    ) = _selection_inputs(
        run_binding,
        [
            canonical_selection_path(run_binding, seed)
            for seed in ALL_SEEDS
        ],
    )
    if selection_paths[requested_seed] != supplied_selection:
        raise JointEvaluationError(
            "chosen selection differs from the canonical selection set"
        )
    selection = selections[requested_seed]
    selection_binding = selection_bindings[requested_seed]
    preflight = _preflight_binding(run_binding, common)
    gate, gate_binding = _load_outer_gate(
        run_binding,
        selections=selections,
        selection_paths=selection_paths,
        selection_bindings=selection_bindings,
        common=common,
        requested_seed=requested_seed,
        preflight=preflight,
    )
    predictor = predictor_loader(selection, device_name=device)
    dev_path = _absolute(
        official_dev_path,
        label="official development dataset",
    )
    if not dev_path.is_file() or dev_path.is_symlink():
        raise JointEvaluationError(
            "official development dataset must be a regular non-link file"
        )
    expected_dev_sha = str(preflight["cmeie_dev_sha256"])
    marker_evidence = {
        "schema_version": 1,
        "kind": "cmeie_official_dev_confirmation_inputs",
        "run_root": str(run_binding.run_root),
        "selection_receipt_path": str(supplied_selection),
        "selection_receipt_sha256": selection_binding["sha256"],
        "preflight_receipt_path": str(paths.preflight_receipt),
        "preflight_receipt_sha256": preflight[
            "preflight_receipt_sha256"
        ],
        "outer_gate_receipt_path": str(paths.outer_gate),
        "outer_gate_receipt_sha256": gate_binding["sha256"],
        "official_dev_path": str(dev_path),
        "official_dev_sha256": expected_dev_sha,
    }
    publish_dataset_started_once(
        run_binding,
        OFFICIAL_DEV_DATASET_ROLE,
        expected_dev_sha,
        str(selection["split_manifest_sha256"]),
        marker_evidence,
    )
    marker, marker_binding = load_and_validate_started_marker(
        run_binding,
        role=OFFICIAL_DEV_DATASET_ROLE,
        dataset_sha256=expected_dev_sha,
        split_sha256=str(selection["split_manifest_sha256"]),
    )
    rows, dataset_binding = dataset_loader(
        dev_path,
        expected_sha256=expected_dev_sha,
    )
    metric = evaluate_one_model(
        rows,
        predictor,
        gold_fn=gold_spo_set,
    )
    passed = official_metric_pass(metric)
    payload = {
        "schema_version": OFFICIAL_CONFIRMATION_SCHEMA_VERSION,
        "kind": OFFICIAL_CONFIRMATION_KIND,
        "dataset_role": OFFICIAL_DEV_DATASET_ROLE,
        "development_confirmation_only": True,
        "outer_accessed": False,
        "official_dev_accessed": True,
        "selection_used": False,
        "rid": run_binding.rid,
        "head": run_binding.git_head,
        "tree": run_binding.included_tree_sha256,
        "run_root": str(run_binding.run_root),
        "source_manifest_path": str(run_binding.source_manifest),
        "source_manifest_sha256": run_binding.source_manifest_sha256,
        "selection_receipt": {
            "seed": selection["seed"],
            "path": str(supplied_selection),
            "sha256": selection_binding["sha256"],
        },
        "preflight_binding": preflight,
        "outer_gate_receipt": {
            "path": str(paths.outer_gate),
            "sha256": gate_binding["sha256"],
            "schema_version": gate["schema_version"],
        },
        "dataset": {
            "path": str(dev_path),
            "sha256": dataset_binding["sha256"],
            "size": dataset_binding["size"],
            "n_rows": len(rows),
        },
        "dataset_access_marker": {
            "path": str(paths.official_dev_marker),
            "sha256": marker_binding["sha256"],
            "dataset_key_sha256": marker["dataset_key_sha256"],
        },
        "model": {
            "seed": predictor.seed,
            "selected_epoch": selection["selected_epoch"],
            "checkpoint": selection["checkpoint"],
            "checkpoint_sha256": selection["checkpoint_sha256"],
            "checkpoint_descriptor": dict(
                predictor.checkpoint_binding
            ),
            "bundle_binding": predictor.bundle_binding,
            "contract_binding": predictor.contract_binding,
            "thresholds": selection["thresholds"],
            "schema_constraint": selection["schema_constraint"],
        },
        "metric": metric,
        "f1_minimum": {"numerator": 1, "denominator": 2},
        "source_bindings": _source_bindings(run_binding),
        "device": device,
        "passed": passed,
        "completed": True,
    }
    binding = publish_json_once(
        paths.official_dev_completion,
        payload,
        label="CMeIE official development confirmation completion",
    )
    return payload, binding, passed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Marker-guarded CMeIE batch evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    outer = subparsers.add_parser("outer")
    outer.add_argument("--run-root", type=Path, required=True)
    outer.add_argument(
        "--selection",
        action="append",
        type=Path,
        required=True,
        help="repeat exactly for canonical seeds 42, 43, and 44",
    )
    outer.add_argument("--split-manifest", type=Path, required=True)
    outer.add_argument("--outer-shard", type=Path, required=True)
    outer.add_argument("--device", choices=("cpu", "npu"), default="npu")

    official = subparsers.add_parser("official-dev-confirmation")
    official.add_argument("--run-root", type=Path, required=True)
    official.add_argument("--selection", type=Path, required=True)
    official.add_argument("--preflight-receipt", type=Path, required=True)
    official.add_argument("--outer-gate-receipt", type=Path, required=True)
    official.add_argument("--official-dev", type=Path, required=True)
    official.add_argument(
        "--device",
        choices=("cpu", "npu"),
        default="npu",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "outer":
            payload, _ = run_outer(
                run_root=args.run_root,
                selection_paths=args.selection,
                split_manifest_path=args.split_manifest,
                outer_shard_path=args.outer_shard,
                device=args.device,
            )
            passed = True
        else:
            payload, _, passed = run_official_dev_confirmation(
                run_root=args.run_root,
                selection_path=args.selection,
                preflight_receipt_path=args.preflight_receipt,
                outer_gate_receipt_path=args.outer_gate_receipt,
                official_dev_path=args.official_dev,
                device=args.device,
            )
    except (
        FileExistsError,
        JointEvaluationError,
        OSError,
        ValueError,
    ) as exc:
        raise SystemExit(f"CMeIE batch evaluation refused: {exc}") from exc
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
