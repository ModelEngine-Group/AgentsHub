"""Deterministic uncertainty estimates for NL2SQL evaluation.

These helpers operate only on already-scored item records.  They never invoke an
NL2SQL engine, and they fail closed when the population or family partition is
not suitable for an auditable estimate.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from statistics import NormalDist
from typing import Any, TypedDict

WILSON_POLICY = "wilson_score_interval_v1"
FAMILY_CLUSTER_BOOTSTRAP_POLICY = (
    "family_cluster_item_weighted_percentile_bootstrap_v1"
)
DEFAULT_FAMILY_KEY = "partition_family_id"
MAX_BOOTSTRAP_ITERATIONS = 1_000_000
_UNSET = object()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


WILSON_POLICY_HASH = _canonical_hash(
    {
        "policy": WILSON_POLICY,
        "center": "wilson_score",
        "tail": "two_sided_normal",
        "population": "items",
    }
)
FAMILY_CLUSTER_BOOTSTRAP_POLICY_HASH = _canonical_hash(
    {
        "policy": FAMILY_CLUSTER_BOOTSTRAP_POLICY,
        "sampling_unit": "family",
        "families_per_resample": "observed_family_count",
        "within_family_sampling": "all_items",
        "estimator": "item_weighted_accuracy",
        "interval": "two_sided_linear_percentile",
        "family_normalization": "NFKC_strip_nonempty_no_controls",
    }
)


class WilsonInterval(TypedDict):
    method: str
    policy: str
    policy_hash: str
    estimate: float
    point: float
    lower: float
    upper: float
    confidence_level: float
    confidence: float
    correct: int
    total: int
    n: int


class FamilyClusterBootstrapInterval(TypedDict):
    method: str
    policy: str
    policy_hash: str
    estimate: float
    point: float
    lower: float
    upper: float
    confidence_level: float
    confidence: float
    n_resamples: int
    iterations: int
    seed: int
    correct: int
    total: int
    n: int
    family_key: str
    n_families: int
    n_clusters: int
    min_family_size: int
    max_family_size: int
    min_cluster_size: int
    max_cluster_size: int


def _validate_confidence(confidence_level: float) -> float:
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, Real):
        raise TypeError("confidence_level must be a finite real number")
    value = float(confidence_level)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    return value


def _validate_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def wilson_interval(
    correct: int,
    total: int | None = None,
    *,
    n: int | None = None,
    confidence_level: float | object = _UNSET,
    confidence: float | None = None,
) -> WilsonInterval:
    """Return a deterministic two-sided Wilson score interval.

    ``total == 0`` is rejected rather than represented as an artificial
    zero-width interval: there is no estimable accuracy without a population.
    """

    correct_value = _validate_integer(correct, name="correct")
    total_value: int
    if total is None:
        if n is None:
            raise TypeError("total (or compatibility alias n) is required")
        total_value = _validate_integer(n, name="n")
    else:
        total_value = _validate_integer(total, name="total")
        if n is not None:
            n_value = _validate_integer(n, name="n")
            if n_value != total_value:
                raise ValueError("total and n must agree when both are supplied")
    if confidence_level is _UNSET:
        confidence_value = _validate_confidence(
            0.95 if confidence is None else confidence
        )
    else:
        confidence_value = _validate_confidence(confidence_level)
        if confidence is not None:
            confidence_alias = _validate_confidence(confidence)
            if confidence_alias != confidence_value:
                raise ValueError(
                    "confidence_level and confidence must agree when both are supplied"
                )
    if total_value <= 0:
        raise ValueError("total must be greater than zero")
    if correct_value < 0 or correct_value > total_value:
        raise ValueError("correct must be between zero and total")

    estimate = correct_value / total_value
    z = NormalDist().inv_cdf(0.5 + confidence_value / 2.0)
    if not math.isfinite(z):
        raise ValueError("confidence_level produces a non-finite normal quantile")
    z_squared = z * z
    denominator = 1.0 + z_squared / total_value
    center = (estimate + z_squared / (2.0 * total_value)) / denominator
    half_width = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / total_value
            + z_squared / (4.0 * total_value * total_value)
        )
        / denominator
    )
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return {
        "method": "wilson",
        "policy": WILSON_POLICY,
        "policy_hash": WILSON_POLICY_HASH,
        "estimate": estimate,
        "point": estimate,
        "lower": lower,
        "upper": upper,
        "confidence_level": confidence_value,
        "confidence": confidence_value,
        "correct": correct_value,
        "total": total_value,
        "n": total_value,
    }


def canonical_family_id(value: Any) -> str:
    """Return a canonical family identifier or reject an unsafe value.

    Family identifiers are strings only.  Coercing numbers or arbitrary objects
    can silently merge clusters (for example ``1`` and ``"1"``), invalidating
    the resampling unit.
    """

    if not isinstance(value, str):
        raise ValueError("family ID must be a string")
    canonical = unicodedata.normalize("NFKC", value).strip()
    if not canonical:
        raise ValueError("family ID must not be empty")
    if any(unicodedata.category(char).startswith("C") for char in canonical):
        raise ValueError("family ID must not contain control characters")
    return canonical


def _linear_percentile(ordered: Sequence[float], probability: float) -> float:
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if len(ordered) == 1:
        return float(ordered[0])
    position = probability * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(ordered[lower_index])
    fraction = position - lower_index
    return (
        float(ordered[lower_index]) * (1.0 - fraction)
        + float(ordered[upper_index]) * fraction
    )


def family_cluster_bootstrap(
    details: Sequence[Mapping[str, Any]],
    *,
    family_key: str = DEFAULT_FAMILY_KEY,
    correct_key: str = "correct",
    confidence_level: float | object = _UNSET,
    confidence: float | None = None,
    iterations: int | object = _UNSET,
    n_resamples: int | None = None,
    seed: int,
) -> FamilyClusterBootstrapInterval:
    """Bootstrap item accuracy by resampling complete families.

    Each replicate draws the observed number of families with replacement.  A
    selected family contributes *all* of its items, and replicate accuracy is
    total correct items divided by total sampled items.  It is therefore
    item-weighted rather than a macro-average of family accuracies.
    """

    if isinstance(details, (str, bytes)) or not isinstance(details, Sequence):
        raise TypeError("details must be a concrete sequence of item mappings")
    if not details:
        raise ValueError("family cluster bootstrap requires at least one item")
    if not isinstance(family_key, str) or not family_key.strip():
        raise ValueError("family_key must be a non-empty string")
    if not isinstance(correct_key, str) or not correct_key.strip():
        raise ValueError("correct_key must be a non-empty string")
    if family_key == correct_key:
        raise ValueError("family_key and correct_key must be distinct")
    if confidence_level is _UNSET:
        confidence_value = _validate_confidence(
            0.95 if confidence is None else confidence
        )
    else:
        confidence_value = _validate_confidence(confidence_level)
        if confidence is not None:
            confidence_alias = _validate_confidence(confidence)
            if confidence_alias != confidence_value:
                raise ValueError(
                    "confidence_level and confidence must agree when both are supplied"
                )
    if iterations is _UNSET:
        iterations_value = _validate_integer(
            10_000 if n_resamples is None else n_resamples,
            name="n_resamples" if n_resamples is not None else "iterations",
        )
    else:
        iterations_value = _validate_integer(iterations, name="iterations")
        if n_resamples is not None:
            resamples_value = _validate_integer(n_resamples, name="n_resamples")
            if resamples_value != iterations_value:
                raise ValueError(
                    "iterations and n_resamples must agree when both are supplied"
                )
    if iterations_value < 2:
        raise ValueError("iterations must be at least two")
    if iterations_value > MAX_BOOTSTRAP_ITERATIONS:
        raise ValueError(
            f"iterations must not exceed {MAX_BOOTSTRAP_ITERATIONS}"
        )
    seed_value = _validate_integer(seed, name="seed")

    clusters: dict[str, list[bool]] = defaultdict(list)
    total_correct = 0
    for index, detail in enumerate(details):
        if not isinstance(detail, Mapping):
            raise TypeError(f"details[{index}] must be a mapping")
        if family_key not in detail:
            raise ValueError(
                f"details[{index}] is missing required family key {family_key!r}"
            )
        try:
            family_id = canonical_family_id(detail[family_key])
        except ValueError as exc:
            raise ValueError(f"details[{index}] has invalid family ID: {exc}") from exc
        if correct_key not in detail:
            raise ValueError(
                f"details[{index}] is missing correctness key {correct_key!r}"
            )
        correct = detail[correct_key]
        if not isinstance(correct, bool):
            raise ValueError(f"details[{index}] correctness must be boolean")
        clusters[family_id].append(correct)
        total_correct += int(correct)

    family_ids = tuple(sorted(clusters))
    if not family_ids:
        raise ValueError("family cluster bootstrap requires a valid family")
    family_counts = tuple(
        (sum(clusters[family_id]), len(clusters[family_id]))
        for family_id in family_ids
    )
    family_sizes = [size for _, size in family_counts]

    n_families = len(family_counts)
    if n_families < 2:
        raise ValueError(
            "family cluster bootstrap requires at least two distinct families"
        )
    rng = random.Random(seed_value)
    bootstrap_accuracies: list[float] = []
    for _ in range(iterations_value):
        sampled_correct = 0
        sampled_total = 0
        for _ in range(n_families):
            family_correct, family_total = family_counts[rng.randrange(n_families)]
            sampled_correct += family_correct
            sampled_total += family_total
        if sampled_total <= 0:  # defensive: impossible after validation above
            raise RuntimeError("bootstrap replicate has an empty item population")
        bootstrap_accuracies.append(sampled_correct / sampled_total)

    bootstrap_accuracies.sort()
    alpha = (1.0 - confidence_value) / 2.0
    total_items = len(details)
    estimate = total_correct / total_items
    minimum_size = min(family_sizes)
    maximum_size = max(family_sizes)
    return {
        "method": "family_cluster_bootstrap",
        "policy": FAMILY_CLUSTER_BOOTSTRAP_POLICY,
        "policy_hash": FAMILY_CLUSTER_BOOTSTRAP_POLICY_HASH,
        "estimate": estimate,
        "point": estimate,
        "lower": _linear_percentile(bootstrap_accuracies, alpha),
        "upper": _linear_percentile(bootstrap_accuracies, 1.0 - alpha),
        "confidence_level": confidence_value,
        "confidence": confidence_value,
        "n_resamples": iterations_value,
        "iterations": iterations_value,
        "seed": seed_value,
        "correct": total_correct,
        "total": total_items,
        "n": total_items,
        "family_key": family_key,
        "n_families": n_families,
        "n_clusters": n_families,
        "min_family_size": minimum_size,
        "max_family_size": maximum_size,
        "min_cluster_size": minimum_size,
        "max_cluster_size": maximum_size,
    }


# Compatibility synonym for callers that name the estimator rather than method.
cluster_bootstrap_accuracy = family_cluster_bootstrap


__all__ = [
    "DEFAULT_FAMILY_KEY",
    "FAMILY_CLUSTER_BOOTSTRAP_POLICY",
    "FAMILY_CLUSTER_BOOTSTRAP_POLICY_HASH",
    "MAX_BOOTSTRAP_ITERATIONS",
    "WILSON_POLICY",
    "WILSON_POLICY_HASH",
    "FamilyClusterBootstrapInterval",
    "WilsonInterval",
    "canonical_family_id",
    "cluster_bootstrap_accuracy",
    "family_cluster_bootstrap",
    "wilson_interval",
]
