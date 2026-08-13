"""Treatment-sentinel acceptance gates for mixed-precision NER profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SentinelThresholds:
    treatment_recall: float = 0.90
    worst_type_f1: float = 0.90
    max_macro_f1_drop: float = 0.01
    min_overall_f1: float = 0.9490
    min_throughput_retention: float = 0.90


def assess_profile(
    candidate: Mapping[str, object],
    fp16_reference: Mapping[str, object],
    fast_reference: Mapping[str, object],
    thresholds: SentinelThresholds = SentinelThresholds(),
) -> dict:
    by_type = candidate.get("by_type", {})
    treatment = by_type.get("Treatment", {})
    support = int(treatment.get("support", int(treatment.get("tp", 0)) + int(treatment.get("fn", 0))))
    candidate_recall = float(treatment.get("recall", 0.0))
    fp16_recall = float(fp16_reference.get("by_type", {}).get("Treatment", {}).get("recall", 0.0))
    throughput = float(candidate.get("throughput_samples_s", 0.0))
    fast_throughput = float(fast_reference.get("throughput_samples_s", 0.0))
    checks = {
        "treatment_support": support > 0,
        "treatment_recall": candidate_recall >= thresholds.treatment_recall,
        "treatment_noninferiority": candidate_recall >= fp16_recall - 0.02,
        "worst_type_f1": float(candidate.get("worst_type_f1", 0.0)) >= thresholds.worst_type_f1,
        "macro_f1": float(candidate.get("macro_f1", 0.0)) >= float(fp16_reference.get("macro_f1", 0.0)) - thresholds.max_macro_f1_drop,
        "overall_f1": float(candidate.get("f1", 0.0)) >= thresholds.min_overall_f1,
        "throughput_retention": fast_throughput > 0 and throughput / fast_throughput >= thresholds.min_throughput_retention,
    }
    return {
        "promotion": all(checks.values()),
        "checks": checks,
        "treatment_support": support,
        "throughput_retention": 0.0 if not fast_throughput else round(throughput / fast_throughput, 4),
    }
