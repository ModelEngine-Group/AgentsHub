"""Shared validation for the canonical CMeIE fit-only overfit proof.

The smoke producer, formal trainer, and checkpoint selector must use the same
count-derived metric contract.  No consumer may authorize training or model
selection from self-reported floating-point precision, recall, or F1 alone.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from dki.evidence_io import EvidenceBinding, load_bound_json
from dki.nlp.cmeie_extraction_contract import BINDING_ALGORITHMS


OVERFIT_RECEIPT_SCHEMA_VERSION = 1
OVERFIT_RECEIPT_KIND = "cmeie_gplinker_fit_only_overfit_preflight"
FIXED_PREFLIGHT_STEPS = 400
FIXED_MAX_LENGTH = 320
FIXED_OVERFIT_F1 = 0.95
FIXED_POINTER_VARIANT = "efficient"
FIXED_HEAD_REDUCTION = "balanced"
FIXED_ENCODER_LEARNING_RATE = 2e-5
FIXED_HEAD_LEARNING_RATE = 1e-3
FIXED_GRADIENT_CLIP = 1.0
FIXED_HEAD_GRADIENT_CLIP = 5.0
FIXED_WEIGHT_DECAY = 0.01
FIXED_HEAD_ONLY_WARMUP_STEPS = 300
FIXED_JOINT_HEAD_LEARNING_RATE_SCALE = 0.1
FIXED_AMP_INITIAL_SCALE = 1024.0

_METRIC_FIELDS = frozenset(
    {"tp", "fp", "fn", "precision", "recall", "f1"}
)
_DECLARED_GRID_FIELDS = frozenset(
    {
        "declared_threshold_values",
        "thresholds",
        "schema_constraint",
        "grid_size",
        "metric",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_overfit_json_bytes(value: object) -> bytes:
    """Return the one byte representation emitted by the smoke producer."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "CMeIE overfit receipt contains a non-JSON or non-finite value"
        ) from exc


def load_canonical_overfit_preflight(
    path: str | os.PathLike[str],
) -> tuple[dict[str, Any], EvidenceBinding]:
    """Descriptor-read one canonical, single-link overfit receipt."""

    payload, binding = load_bound_json(
        path,
        "canonical CMeIE overfit preflight",
        reject_multiple_links=True,
        require_object=True,
    )
    if type(payload) is not dict:
        raise ValueError("CMeIE overfit receipt must be a JSON object")
    canonical = canonical_overfit_json_bytes(payload)
    if (
        binding["size"] != len(canonical)
        or binding["sha256"] != hashlib.sha256(canonical).hexdigest()
    ):
        raise ValueError(
            "CMeIE overfit receipt is not canonical JSON"
        )
    return payload, binding


def recompute_overfit_metric(metric: object) -> dict[str, int | float]:
    """Validate the exact metric schema and derive P/R/F1 from counts."""

    if type(metric) is not dict:
        raise ValueError("overfit metric must be a JSON object")
    if set(metric) != _METRIC_FIELDS:
        missing = sorted(_METRIC_FIELDS - set(metric))
        extra = sorted(set(metric) - _METRIC_FIELDS)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise ValueError(
            "overfit metric fields are invalid: " + "; ".join(details)
        )
    counts: dict[str, int] = {}
    for field in ("tp", "fp", "fn"):
        value = metric[field]
        if type(value) is not int or value < 0:
            raise ValueError(
                f"overfit metric {field} must be a non-negative integer"
            )
        counts[field] = value

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = Fraction(tp, tp + fp) if tp + fp else Fraction(0, 1)
    recall = Fraction(tp, tp + fn) if tp + fn else Fraction(0, 1)
    f1 = (
        Fraction(2 * tp, 2 * tp + fp + fn)
        if 2 * tp + fp + fn
        else Fraction(0, 1)
    )
    exact = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    for field, fraction in exact.items():
        stated = metric[field]
        if (
            isinstance(stated, bool)
            or not isinstance(stated, (int, float))
            or not math.isfinite(float(stated))
            or not 0.0 <= float(stated) <= 1.0
            or not math.isclose(
                float(stated),
                float(fraction),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(
                f"overfit metric {field} disagrees with tp/fp/fn"
            )
    return {
        **counts,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def overfit_gate_passed(
    *,
    metric: dict[str, Any],
    schema_on: bool,
    minimum_train_f1: float,
) -> bool:
    """Gate only on count-derived exact F1 and the schema-on contract."""

    if (
        type(schema_on) is not bool
        or isinstance(minimum_train_f1, bool)
        or not isinstance(minimum_train_f1, (int, float))
        or not math.isfinite(float(minimum_train_f1))
        or not 0.0 <= float(minimum_train_f1) <= 1.0
    ):
        raise ValueError("overfit gate inputs are invalid")
    recomputed = recompute_overfit_metric(metric)
    tp = int(recomputed["tp"])
    fp = int(recomputed["fp"])
    fn = int(recomputed["fn"])
    exact_f1 = (
        Fraction(2 * tp, 2 * tp + fp + fn)
        if 2 * tp + fp + fn
        else Fraction(0, 1)
    )
    threshold = Fraction(str(float(minimum_train_f1)))
    return schema_on and exact_f1 >= threshold


def validate_overfit_preflight(
    payload: dict,
    *,
    receipt_sha256: str,
    source_train_sha256: str,
    fit_shard_sha256: str,
    split_manifest_sha256: str,
    base_model: str,
    base_model_tree: dict,
    source_tree_manifest_sha256: str,
    preflight_script_sha256: str,
    model_module_sha256: str,
    alignment_module_sha256: str,
    globalpointer_module_sha256: str,
    tokenizer_vocabulary_sha256: str,
    label_table_sha256: str,
    schema_constraints_sha256: str,
    source_tree_sha256: str,
    preflight_common: dict,
) -> dict:
    """Validate the fit-only proof that authorizes formal CMeIE work."""

    if type(payload) is not dict:
        raise ValueError(
            "CMeIE overfit preflight is not admissible: payload is not an "
            "object"
        )
    declared = payload.get("declared_grid_result")
    mismatches: list[str] = []
    if type(declared) is not dict:
        mismatches.append("declared_grid_result is not an object")
        declared = {}
    elif set(declared) != _DECLARED_GRID_FIELDS:
        mismatches.append(
            "declared_grid_result fields are not the strict contract"
        )
    try:
        metric = recompute_overfit_metric(declared.get("metric"))
    except ValueError as exc:
        mismatches.append(str(exc))
        metric = {}

    phase_grid_results = payload.get("phase_grid_results")
    validated_phase_results: dict[str, tuple[dict, dict]] = {}
    if (
        type(phase_grid_results) is not dict
        or set(phase_grid_results) != {"head_only", "joint"}
    ):
        mismatches.append(
            "phase_grid_results must contain exactly head_only and joint"
        )
    else:
        for phase in ("head_only", "joint"):
            result = phase_grid_results[phase]
            if type(result) is not dict or set(result) != _DECLARED_GRID_FIELDS:
                mismatches.append(
                    f"phase_grid_results.{phase} fields are invalid"
                )
                continue
            try:
                phase_metric = recompute_overfit_metric(
                    result.get("metric")
                )
            except ValueError as exc:
                mismatches.append(f"{phase}: {exc}")
                continue
            thresholds = result.get("thresholds")
            if (
                result.get("declared_threshold_values")
                != [-0.5, 0.0, 0.5]
                or result.get("grid_size") != 54
                or result.get("schema_constraint") not in {"off", "on"}
                or not isinstance(thresholds, list)
                or len(thresholds) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or float(value) not in {-0.5, 0.0, 0.5}
                    for value in thresholds
                )
            ):
                mismatches.append(
                    f"phase_grid_results.{phase} grid contract is invalid"
                )
                continue
            validated_phase_results[phase] = (result, phase_metric)
    selected_phase = payload.get("selected_phase")
    if len(validated_phase_results) == 2:
        expected_phase = max(
            validated_phase_results,
            key=lambda phase: (
                validated_phase_results[phase][1]["f1"],
                validated_phase_results[phase][1]["precision"],
                sum(validated_phase_results[phase][0]["thresholds"]),
                validated_phase_results[phase][0]["schema_constraint"]
                == "on",
                phase == "joint",
            ),
        )
        if selected_phase != expected_phase:
            mismatches.append(
                "selected_phase is not the deterministic best phase"
            )
        elif declared != validated_phase_results[expected_phase][0]:
            mismatches.append(
                "declared_grid_result differs from selected phase result"
            )
    elif selected_phase not in {"head_only", "joint"}:
        mismatches.append("selected_phase is invalid")

    schema_on = declared.get("schema_constraint") == "on"
    exact_passed = False
    if metric:
        exact_passed = overfit_gate_passed(
            metric=metric,
            schema_on=schema_on,
            minimum_train_f1=FIXED_OVERFIT_F1,
        )
    expected = {
        "schema_version": OVERFIT_RECEIPT_SCHEMA_VERSION,
        "kind": OVERFIT_RECEIPT_KIND,
        "passed": exact_passed,
        "fit_only": True,
        "calibration_accessed": False,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "train_sha256": source_train_sha256,
        "source_train_sha256": source_train_sha256,
        "fit_shard_sha256": fit_shard_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "base_model": base_model,
        "base_model_tree": base_model_tree,
        "binding_algorithms": BINDING_ALGORITHMS,
        "source_tree_manifest_sha256": source_tree_manifest_sha256,
        "preflight_script_sha256": preflight_script_sha256,
        "sample_size": 8,
        "steps": FIXED_PREFLIGHT_STEPS,
        "max_length": FIXED_MAX_LENGTH,
        "minimum_train_f1": FIXED_OVERFIT_F1,
        "fp16": True,
        "pointer_variant": FIXED_POINTER_VARIANT,
        "head_reduction": FIXED_HEAD_REDUCTION,
        "encoder_frozen": False,
        "positive_weight": 1.0,
        "positive_margin": 0.0,
        "encoder_learning_rate": FIXED_ENCODER_LEARNING_RATE,
        "head_learning_rate": FIXED_HEAD_LEARNING_RATE,
        "joint_head_learning_rate": (
            FIXED_HEAD_LEARNING_RATE
            * FIXED_JOINT_HEAD_LEARNING_RATE_SCALE
        ),
        "head_only_warmup_steps": FIXED_HEAD_ONLY_WARMUP_STEPS,
        "joint_optimizer_steps": (
            FIXED_PREFLIGHT_STEPS - FIXED_HEAD_ONLY_WARMUP_STEPS
        ),
        "encoder_mode": "eval_with_gradients_after_warmup",
        "amp_grad_scaler": True,
        "amp_initial_scale": FIXED_AMP_INITIAL_SCALE,
        "dropped_label_count": 0,
        "gradient_clip": FIXED_GRADIENT_CLIP,
        "gradient_clip_scope": "separate_encoder_and_pointer_heads",
        "encoder_gradient_clip": FIXED_GRADIENT_CLIP,
        "head_gradient_clip": FIXED_HEAD_GRADIENT_CLIP,
        "gradient_clip_after_unscale": True,
        "weight_decay": FIXED_WEIGHT_DECAY,
        "weight_decay_policy": "exclude_bias_and_layernorm",
        "positive_weight_implementation": "integer_unrolled_add",
        "model_module_sha256": model_module_sha256,
        "alignment_module_sha256": alignment_module_sha256,
        "globalpointer_module_sha256": globalpointer_module_sha256,
        "tokenizer_vocabulary_sha256": tokenizer_vocabulary_sha256,
        "label_table_sha256": label_table_sha256,
        "schema_constraints_sha256": schema_constraints_sha256,
        "preflight_common": preflight_common,
    }
    mismatches.extend(
        f"{name}={payload.get(name)!r}, expected {value!r}"
        for name, value in expected.items()
        if payload.get(name) != value
    )
    if not re.fullmatch(
        r"npu(?::[0-9]+)?",
        str(payload.get("device", "")),
    ):
        mismatches.append("device is not an Ascend NPU")
    if declared.get("declared_threshold_values") != [-0.5, 0.0, 0.5]:
        mismatches.append(
            "overfit proof did not use the full predeclared threshold grid"
        )
    selected_thresholds = declared.get("thresholds")
    if (
        not isinstance(selected_thresholds, list)
        or len(selected_thresholds) != 3
        or any(
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(threshold) not in {-0.5, 0.0, 0.5}
            for threshold in selected_thresholds
        )
    ):
        mismatches.append(
            "overfit proof selected thresholds outside the declared grid"
        )
    if not schema_on:
        mismatches.append(
            "overfit proof did not pass with schema constraint on"
        )
    if declared.get("grid_size") != 54:
        mismatches.append(
            "overfit proof did not evaluate all 54 candidates"
        )
    if metric and not exact_passed:
        mismatches.append(
            f"count-derived overfit F1={metric['f1']!r} is below "
            f"{FIXED_OVERFIT_F1}"
        )
    source_tree = str(payload.get("source_tree_sha256", "")).casefold()
    expected_source_tree = str(source_tree_sha256).casefold()
    if _SHA256_RE.fullmatch(source_tree) is None:
        mismatches.append("overfit source tree SHA-256 is invalid")
    if _SHA256_RE.fullmatch(expected_source_tree) is None:
        mismatches.append("expected source tree SHA-256 is invalid")
    elif (
        _SHA256_RE.fullmatch(source_tree) is not None
        and source_tree != expected_source_tree
    ):
        mismatches.append(
            "overfit source tree SHA-256 differs from the candidate source tree"
        )
    if _SHA256_RE.fullmatch(receipt_sha256) is None:
        mismatches.append("overfit receipt SHA-256 is invalid")
    if mismatches:
        raise ValueError(
            "CMeIE overfit preflight is not admissible: "
            + "; ".join(mismatches)
        )
    return {
        "path_sha256": receipt_sha256,
        "source_tree_sha256": source_tree,
        "source_tree_manifest_sha256": payload[
            "source_tree_manifest_sha256"
        ],
        "train_sha256": payload["train_sha256"],
        "source_train_sha256": payload["source_train_sha256"],
        "fit_shard_sha256": payload["fit_shard_sha256"],
        "split_manifest_sha256": payload["split_manifest_sha256"],
        "base_model": payload["base_model"],
        "binding_algorithms": copy.deepcopy(payload["binding_algorithms"]),
        "preflight_script_sha256": payload["preflight_script_sha256"],
        "base_model_tree": copy.deepcopy(payload["base_model_tree"]),
        "model_module_sha256": payload["model_module_sha256"],
        "alignment_module_sha256": payload["alignment_module_sha256"],
        "tokenizer_vocabulary_sha256": payload[
            "tokenizer_vocabulary_sha256"
        ],
        "label_table_sha256": payload["label_table_sha256"],
        "schema_constraints_sha256": payload[
            "schema_constraints_sha256"
        ],
        "globalpointer_module_sha256": payload[
            "globalpointer_module_sha256"
        ],
        "preflight_common": copy.deepcopy(payload["preflight_common"]),
        "metric": metric,
        "sample_size": payload["sample_size"],
        "steps": payload["steps"],
        "head_reduction": payload["head_reduction"],
        "head_learning_rate": payload["head_learning_rate"],
        "joint_head_learning_rate": payload[
            "joint_head_learning_rate"
        ],
        "head_only_warmup_steps": payload[
            "head_only_warmup_steps"
        ],
        "joint_optimizer_steps": payload["joint_optimizer_steps"],
        "encoder_mode": payload["encoder_mode"],
        "selected_phase": payload["selected_phase"],
        "phase_grid_results": copy.deepcopy(
            payload["phase_grid_results"]
        ),
        "encoder_frozen": payload["encoder_frozen"],
        "amp_grad_scaler": payload["amp_grad_scaler"],
        "gradient_clip_after_unscale": payload[
            "gradient_clip_after_unscale"
        ],
        "gradient_clip_scope": payload["gradient_clip_scope"],
        "encoder_gradient_clip": payload["encoder_gradient_clip"],
        "head_gradient_clip": payload["head_gradient_clip"],
        "weight_decay_policy": payload["weight_decay_policy"],
        "calibration_accessed": False,
        "outer_accessed": False,
        "official_dev_accessed": False,
    }


__all__ = [
    "FIXED_OVERFIT_F1",
    "OVERFIT_RECEIPT_KIND",
    "OVERFIT_RECEIPT_SCHEMA_VERSION",
    "canonical_overfit_json_bytes",
    "load_canonical_overfit_preflight",
    "overfit_gate_passed",
    "recompute_overfit_metric",
    "validate_overfit_preflight",
]
