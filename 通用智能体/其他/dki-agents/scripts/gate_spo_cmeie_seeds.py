"""Gate CMeIE seeds using canonical selections and one outer batch receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dki.evidence_io import EvidenceBinding, load_bound_json
from dki.nlp.cmeie_joint_evaluation import (
    ALL_SEEDS,
    JointEvaluationError,
    OUTER_BATCH_KIND,
    OUTER_BATCH_SCHEMA_VERSION,
    OUTER_MEDIAN_F1_FLOOR,
    OUTER_PRECISION_RECALL_FLOOR,
    OUTER_WORST_F1_FLOOR,
    exact_outer_gate_observation,
    load_and_validate_started_marker,
    load_canonical_completion,
    metric_ratio,
    outer_metrics_pass,
    publish_json_once,
    ratio_payload,
    source_member_binding,
    validate_metric,
    validate_selection_set,
)
from dki.nlp.cmeie_preflight_protocol import (
    validate_cmeie_npu_preflight_receipt,
)
from dki.nlp.cmeie_run_protocol import (
    OUTER_DATASET_ROLE,
    RunBinding,
    canonical_run_paths,
    load_run_binding,
)
from dki.nlp.cmeie_selection_protocol import (
    canonical_selection_path,
    load_and_validate_selection_receipt,
)
from dki.nlp.cmeie_split_protocol import SCHEMA_VERSION as SPLIT_SCHEMA_VERSION


EARLY_SEEDS = (42, 43)
EARLY_F1_FLOOR = Fraction(9, 20)
GATE_SCHEMA_VERSION = 2
EARLY_GATE_KIND = "cmeie_gplinker_seed_early_gate"
OUTER_GATE_KIND = "cmeie_gplinker_seed_outer_gate"
OFFICIAL_DEV_EXPECTED_ROWS = 3585


class GateInputError(JointEvaluationError):
    """Evidence cannot authorize the requested seed transition."""


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _absolute(path: Path, *, label: str) -> Path:
    raw = os.fspath(path)
    if not raw or "\0" in raw:
        raise GateInputError(f"{label} must be a non-empty path")
    if any(part in {".", ".."} for part in raw.replace("\\", "/").split("/")):
        raise GateInputError(f"{label} must not contain lexical traversal")
    value = Path(os.path.abspath(raw))
    if not value.is_absolute():
        raise GateInputError(f"{label} must be absolute")
    return value


def _calibration_metric(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise GateInputError(f"{label} must be an object")
    counts: dict[str, int] = {}
    for field in ("tp", "fp", "fn"):
        item = value.get(field)
        if type(item) is not int or item < 0:
            raise GateInputError(
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
            or abs(Fraction(str(reported)) - expected) > Fraction(1, 10**12)
        ):
            raise GateInputError(
                f"{label}.{field} disagrees with integer counts"
            )
    return {**counts, **exact}


def _metric_evidence(
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
        for seed in sorted(metrics)
    ]


def _selection_inputs(
    run_binding: RunBinding,
    supplied_paths: Sequence[Path],
    *,
    expected_seeds: tuple[int, ...],
) -> tuple[
    dict[int, Mapping[str, Any]],
    dict[int, Path],
    dict[int, EvidenceBinding],
    dict[int, dict[str, Any]],
    dict[str, Any],
]:
    if len(supplied_paths) != len(expected_seeds):
        raise GateInputError(
            f"expected exactly {len(expected_seeds)} selections"
        )
    payloads: list[Mapping[str, Any]] = []
    paths: dict[int, Path] = {}
    bindings: dict[int, EvidenceBinding] = {}
    metrics: dict[int, dict[str, Any]] = {}
    for index, supplied in enumerate(supplied_paths):
        actual = _absolute(supplied, label=f"selection[{index}]")
        matching = [
            seed
            for seed in expected_seeds
            if actual == canonical_selection_path(run_binding, seed)
        ]
        if len(matching) != 1:
            raise GateInputError(
                f"selection[{index}] is not a canonical expected seed path"
            )
        seed = matching[0]
        if seed in paths:
            raise GateInputError(f"duplicate selection seed {seed}")
        payload = load_and_validate_selection_receipt(
            actual,
            run_binding=run_binding,
        )
        rebound, binding = load_bound_json(
            actual,
            f"CMeIE seed {seed} selection gate input",
            reject_multiple_links=True,
            require_object=True,
        )
        if rebound != payload:
            raise GateInputError(
                f"seed {seed} selection changed during gate validation"
            )
        payloads.append(payload)
        paths[seed] = actual
        bindings[seed] = binding
        metrics[seed] = _calibration_metric(
            payload.get("calibration_metric"),
            label=f"selection seed {seed} calibration metric",
        )
    if tuple(sorted(paths)) != expected_seeds:
        raise GateInputError(
            f"selection seeds must be exactly {list(expected_seeds)}"
        )

    if expected_seeds == ALL_SEEDS:
        selections, common = validate_selection_set(payloads)
    else:
        # The public set validator intentionally requires all three seeds.
        selections = {int(item["seed"]): item for item in payloads}
        common = {}
        first = selections[expected_seeds[0]]
        common_fields = (
            "rid",
            "head",
            "tree",
            "run_root",
            "source_manifest_sha256",
            "source_tree_sha256",
            "source_tree_manifest_sha256",
            "train_sha256",
            "split_manifest_sha256",
            "base_model",
            "base_model_tree_sha256",
            "tokenizer_vocabulary_sha256",
            "label_table_sha256",
            "schema_constraints_sha256",
            "max_length",
        )
        for field in common_fields:
            expected = first.get(field)
            if any(
                selections[seed].get(field) != expected
                for seed in expected_seeds
            ):
                raise GateInputError(
                    f"early selections differ on common binding {field}"
                )
            common[field] = expected
    return selections, paths, bindings, metrics, common


def _selection_evidence(
    selections: Mapping[int, Mapping[str, Any]],
    paths: Mapping[int, Path],
    bindings: Mapping[int, EvidenceBinding],
) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "path": str(paths[seed]),
            "sha256": bindings[seed]["sha256"],
            "size": bindings[seed]["size"],
            "training_receipt_sha256": selections[seed][
                "training_receipt_sha256"
            ],
            "checkpoint_sha256": selections[seed]["checkpoint_sha256"],
            "checkpoint_bundle_manifest_sha256": selections[seed][
                "checkpoint_bundle_manifest_sha256"
            ],
            "checkpoint_bundle_tree_sha256": selections[seed][
                "checkpoint_bundle_tree_sha256"
            ],
            "extraction_contract_sha256": selections[seed][
                "extraction_contract_sha256"
            ],
            "semantic_contract_sha256": selections[seed][
                "semantic_contract_sha256"
            ],
        }
        for seed in sorted(selections)
    ]


def _early_path(run_binding: RunBinding) -> Path:
    return canonical_run_paths(run_binding).evidence_dir / "early_gate.json"


def _build_early_receipt(
    *,
    run_binding: RunBinding,
    selection_paths: Sequence[Path],
) -> tuple[dict[str, Any], bool]:
    selections, paths, bindings, metrics, common = _selection_inputs(
        run_binding,
        selection_paths,
        expected_seeds=EARLY_SEEDS,
    )
    stop = all(metrics[seed]["f1"] < EARLY_F1_FLOOR for seed in EARLY_SEEDS)
    payload = {
        "schema_version": GATE_SCHEMA_VERSION,
        "kind": EARLY_GATE_KIND,
        "dataset_role": "calibration_seed_gate",
        "selection_used": True,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "rid": run_binding.rid,
        "head": run_binding.git_head,
        "tree": run_binding.included_tree_sha256,
        "run_root": str(run_binding.run_root),
        "expected_seeds": list(EARLY_SEEDS),
        "selection_receipts": _selection_evidence(
            selections,
            paths,
            bindings,
        ),
        "common_bindings": common,
        "calibration_metrics": _metric_evidence(metrics),
        "strict_stop_f1_floor": ratio_payload(EARLY_F1_FLOOR),
        "decision": "stop_joint_route" if stop else "advance_to_seed44",
        "stop_joint_route": stop,
        "advance_to_seed44": not stop,
        "gate_script_sha256": _script_sha256(),
    }
    return payload, not stop


def _validate_early_receipt(
    run_binding: RunBinding,
    *,
    selections: Mapping[int, Mapping[str, Any]],
    paths: Mapping[int, Path],
    bindings: Mapping[int, EvidenceBinding],
    metrics: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], EvidenceBinding]:
    path = _early_path(run_binding)
    payload, binding = load_bound_json(
        path,
        "CMeIE early seed gate",
        reject_multiple_links=True,
        require_object=True,
    )
    expected_selections = {
        seed: selections[seed] for seed in EARLY_SEEDS
    }
    expected_paths = {seed: paths[seed] for seed in EARLY_SEEDS}
    expected_bindings = {seed: bindings[seed] for seed in EARLY_SEEDS}
    expected_metrics = {seed: metrics[seed] for seed in EARLY_SEEDS}
    if (
        payload.get("schema_version") != GATE_SCHEMA_VERSION
        or payload.get("kind") != EARLY_GATE_KIND
        or payload.get("dataset_role") != "calibration_seed_gate"
        or payload.get("selection_used") is not True
        or payload.get("outer_accessed") is not False
        or payload.get("official_dev_accessed") is not False
        or payload.get("rid") != run_binding.rid
        or payload.get("head") != run_binding.git_head
        or payload.get("tree") != run_binding.included_tree_sha256
        or payload.get("run_root") != str(run_binding.run_root)
        or payload.get("expected_seeds") != list(EARLY_SEEDS)
        or payload.get("selection_receipts")
        != _selection_evidence(
            expected_selections,
            expected_paths,
            expected_bindings,
        )
        or payload.get("calibration_metrics")
        != _metric_evidence(expected_metrics)
        or payload.get("strict_stop_f1_floor")
        != ratio_payload(EARLY_F1_FLOOR)
        or payload.get("advance_to_seed44") is not True
        or payload.get("stop_joint_route") is not False
        or payload.get("decision") != "advance_to_seed44"
        or payload.get("gate_script_sha256") != _script_sha256()
    ):
        raise GateInputError(
            "early gate does not authorize the frozen three-seed transition"
        )
    if all(
        metrics[seed]["f1"] < EARLY_F1_FLOOR for seed in EARLY_SEEDS
    ):
        raise GateInputError(
            "seed 42 and seed 43 are both below the strict early floor"
        )
    return payload, binding


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


def _batch_outer_metrics(
    batch: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    models = batch.get("models")
    if type(models) is not list or len(models) != len(ALL_SEEDS):
        raise GateInputError(
            "outer batch must contain exactly three model metrics"
        )
    result: dict[int, dict[str, Any]] = {}
    for index, model in enumerate(models):
        if type(model) is not dict:
            raise GateInputError(f"outer batch model[{index}] is invalid")
        seed = model.get("seed")
        if type(seed) is not int or seed not in ALL_SEEDS or seed in result:
            raise GateInputError(
                f"outer batch model[{index}] has an invalid seed"
            )
        try:
            result[seed] = validate_metric(
                model.get("metric"),
                label=f"outer batch seed {seed} metric",
            )
        except JointEvaluationError as exc:
            raise GateInputError(str(exc)) from exc
    if tuple(sorted(result)) != ALL_SEEDS:
        raise GateInputError(
            "outer batch metrics must be seeds 42, 43, and 44"
        )
    return result


def _validate_outer_batch(
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
    batch, batch_binding = load_canonical_completion(
        run_binding,
        role=OUTER_DATASET_ROLE,
        expected_kind=OUTER_BATCH_KIND,
        expected_schema_version=OUTER_BATCH_SCHEMA_VERSION,
    )
    if (
        batch.get("outer_accessed") is not True
        or batch.get("official_dev_accessed") is not False
        or batch.get("selection_used") is not False
        or batch.get("single_dataset_read") is not True
        or batch.get("same_row_loop_for_all_seeds") is not True
        or batch.get("completed") is not True
        or batch.get("common_bindings") != common
        or batch.get("selection_receipts")
        != _selection_evidence(
            selections,
            selection_paths,
            selection_bindings,
        )
    ):
        raise GateInputError(
            "outer batch common or selection binding is invalid"
        )
    dataset = batch.get("dataset")
    if type(dataset) is not dict:
        raise GateInputError("outer batch dataset binding is invalid")
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
        or dataset.get("shard_relative_path") != "outer.jsonl"
        or batch.get("split_manifest_path")
        != str(
            canonical_run_paths(run_binding).split_dir
            / "split_manifest.json"
        )
    ):
        raise GateInputError("outer batch dataset identity is invalid")
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
            canonical_run_paths(run_binding).split_dir / "outer.jsonl"
        ),
        "outer_shard_sha256": dataset_sha,
        "selection_receipts": batch["selection_receipts"],
    }
    if marker.get("input_model_evidence") != expected_marker_evidence:
        raise GateInputError(
            "outer batch access marker input evidence is invalid"
        )
    recorded_marker = batch.get("dataset_access_marker")
    if (
        type(recorded_marker) is not dict
        or recorded_marker.get("path")
        != str(canonical_run_paths(run_binding).outer_started)
        or recorded_marker.get("sha256") != marker_binding.get("sha256")
        or recorded_marker.get("dataset_key_sha256")
        != marker.get("dataset_key_sha256")
    ):
        raise GateInputError(
            "outer batch access marker binding is invalid"
        )

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
            or model.get("bundle_binding")
            != selection["bundle_binding"]
            or model.get("contract_binding")
            != selection["contract_binding"]
        ):
            raise GateInputError(
                f"outer batch seed {seed} model binding is invalid"
            )
    expected_sources = {
        relative: source_member_binding(run_binding, relative)
        for relative in (
            "dki/nlp/cmeie_joint_evaluation.py",
            "dki/nlp/cmeie_split_protocol.py",
            "scripts/build_cmeie_split_manifest.py",
            "scripts/eval_spo_cmeie_batch.py",
        )
    }
    if batch.get("source_bindings") != expected_sources:
        raise GateInputError(
            "outer batch evaluator source binding is invalid"
        )
    if (
        dataset.get("validator_sha256")
        != expected_sources["dki/nlp/cmeie_split_protocol.py"]["sha256"]
        or dataset.get("generator_sha256")
        != expected_sources["scripts/build_cmeie_split_manifest.py"][
            "sha256"
        ]
    ):
        raise GateInputError(
            "outer batch split producer/validator binding is invalid"
        )
    return (
        batch,
        batch_binding,
        _batch_outer_metrics(batch),
        marker,
        marker_binding,
    )


def _calibration_choice(
    metrics: Mapping[int, Mapping[str, Any]],
) -> int:
    return max(
        ALL_SEEDS,
        key=lambda seed: (
            metrics[seed]["f1"],
            metrics[seed]["precision"],
            -seed,
        ),
    )


def _build_outer_receipt(
    *,
    run_binding: RunBinding,
    selection_paths: Sequence[Path],
) -> tuple[dict[str, Any], bool]:
    selections, paths, bindings, calibration_metrics, common = (
        _selection_inputs(
            run_binding,
            selection_paths,
            expected_seeds=ALL_SEEDS,
        )
    )
    early, early_binding = _validate_early_receipt(
        run_binding,
        selections=selections,
        paths=paths,
        bindings=bindings,
        metrics=calibration_metrics,
    )
    preflight = _preflight_binding(run_binding, common)
    batch, batch_binding, outer_metrics, marker, marker_binding = (
        _validate_outer_batch(
            run_binding,
            selections=selections,
            selection_paths=paths,
            selection_bindings=bindings,
            common=common,
        )
    )
    passed = outer_metrics_pass(outer_metrics)
    chosen_seed = _calibration_choice(calibration_metrics)
    observation = exact_outer_gate_observation(outer_metrics)
    payload: dict[str, Any] = {
        "schema_version": GATE_SCHEMA_VERSION,
        "kind": OUTER_GATE_KIND,
        "dataset_role": "outer_gate",
        "selection_used": False,
        "outer_accessed": True,
        "official_dev_accessed": False,
        "rid": run_binding.rid,
        "head": run_binding.git_head,
        "tree": run_binding.included_tree_sha256,
        "run_root": str(run_binding.run_root),
        "expected_seeds": list(ALL_SEEDS),
        "early_gate_receipt": {
            "path": str(_early_path(run_binding)),
            "sha256": early_binding["sha256"],
            "schema_version": early["schema_version"],
        },
        "selection_receipts": _selection_evidence(
            selections,
            paths,
            bindings,
        ),
        "outer_batch_receipt": {
            "path": str(canonical_run_paths(run_binding).outer_completion),
            "sha256": batch_binding["sha256"],
            "schema_version": batch["schema_version"],
        },
        "outer_access_marker": {
            "path": str(canonical_run_paths(run_binding).outer_started),
            "sha256": marker_binding["sha256"],
            "dataset_key_sha256": marker["dataset_key_sha256"],
        },
        "preflight_receipt_path": preflight["preflight_receipt_path"],
        "preflight_receipt_sha256": preflight[
            "preflight_receipt_sha256"
        ],
        "preflight_binding": preflight,
        "expected_official_dev_sha256": preflight["cmeie_dev_sha256"],
        "expected_official_dev_rows": preflight["cmeie_dev_rows"],
        "common_bindings": common,
        "calibration_metrics": _metric_evidence(calibration_metrics),
        "outer_metrics": [
            {"seed": seed, **outer_metrics[seed]}
            for seed in ALL_SEEDS
        ],
        "thresholds": {
            "median_f1_minimum": ratio_payload(
                OUTER_MEDIAN_F1_FLOOR
            ),
            "worst_f1_minimum": ratio_payload(
                OUTER_WORST_F1_FLOOR
            ),
            "per_seed_precision_minimum": ratio_payload(
                OUTER_PRECISION_RECALL_FLOOR
            ),
            "per_seed_recall_minimum": ratio_payload(
                OUTER_PRECISION_RECALL_FLOOR
            ),
        },
        "observed": {
            "median_f1": observation["median_f1"],
            "worst_f1": observation["worst_f1"],
        },
        "calibration_choice": {
            "seed": chosen_seed,
            "tie_break": [
                "calibration_f1_desc",
                "calibration_precision_desc",
                "seed_asc",
            ],
            "selection_receipt_sha256": bindings[chosen_seed]["sha256"],
            "checkpoint_sha256": selections[chosen_seed][
                "checkpoint_sha256"
            ],
            "checkpoint_bundle_manifest_sha256": selections[chosen_seed][
                "checkpoint_bundle_manifest_sha256"
            ],
            "checkpoint_bundle_tree_sha256": selections[chosen_seed][
                "checkpoint_bundle_tree_sha256"
            ],
        },
        "passed": passed,
        "decision": (
            "official_dev_authorized" if passed else "outer_gate_failed"
        ),
        "official_dev_authorized": passed,
        "gate_script_sha256": _script_sha256(),
    }
    if passed:
        payload["authorized_candidate"] = dict(
            payload["calibration_choice"]
        )
    return payload, passed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hash-bound CMeIE seed gates"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    early = subparsers.add_parser("early")
    early.add_argument("--run-root", type=Path, required=True)
    early.add_argument(
        "--selection",
        action="append",
        type=Path,
        required=True,
        help="repeat exactly for canonical seeds 42 and 43",
    )
    early.add_argument("--output", type=Path, required=True)

    outer = subparsers.add_parser("outer")
    outer.add_argument("--run-root", type=Path, required=True)
    outer.add_argument(
        "--selection",
        action="append",
        type=Path,
        required=True,
        help="repeat exactly for canonical seeds 42, 43, and 44",
    )
    outer.add_argument("--outer-batch-receipt", type=Path, required=True)
    outer.add_argument("--preflight-receipt", type=Path, required=True)
    outer.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_binding = load_run_binding(args.run_root)
        paths = canonical_run_paths(run_binding)
        output = _absolute(args.output, label="gate output")
        if args.command == "early":
            if output != _early_path(run_binding):
                raise GateInputError(
                    "early output must be the canonical RID early gate path"
                )
            payload, passed = _build_early_receipt(
                run_binding=run_binding,
                selection_paths=args.selection,
            )
        else:
            if _absolute(
                args.outer_batch_receipt,
                label="outer batch receipt",
            ) != paths.outer_completion:
                raise GateInputError(
                    "outer batch receipt must be the canonical completion"
                )
            if _absolute(
                args.preflight_receipt,
                label="preflight receipt",
            ) != paths.preflight_receipt:
                raise GateInputError(
                    "preflight receipt must be the canonical run receipt"
                )
            if output != paths.outer_gate:
                raise GateInputError(
                    "outer output must be the canonical RID gate path"
                )
            payload, passed = _build_outer_receipt(
                run_binding=run_binding,
                selection_paths=args.selection,
            )
        publish_json_once(output, payload, label="CMeIE seed gate")
    except (
        FileExistsError,
        GateInputError,
        JointEvaluationError,
        OSError,
        ValueError,
    ) as exc:
        raise SystemExit(f"CMeIE seed gate rejected evidence: {exc}") from exc
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
