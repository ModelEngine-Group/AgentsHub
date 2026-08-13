"""Manifest-bound, budgeted adaptive NER routing decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping


class Route(str, Enum):
    DICT = "dict"
    INT8 = "bert_int8_om"
    QUALITY = "quality"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class RoutingPolicy:
    confidence_threshold: float = 0.80
    max_upgrade_rate: float = 0.20
    treatment_type: str = "Treatment"


class AdaptiveRouter:
    def __init__(self, policy: RoutingPolicy, *, calibration_valid: bool, promotion_valid: bool):
        self.policy = policy
        self.enabled = calibration_valid and promotion_valid
        self.total = 0
        self.upgraded = 0

    def decide(
        self,
        entities: Iterable[Mapping[str, object]],
        *,
        calibrated_confidence: float | None,
        kg_conflict: bool = False,
        window_conflict: bool = False,
    ) -> dict:
        self.total += 1
        if not self.enabled:
            return {"route": Route.INT8.value, "reason": "adaptive_not_promoted"}
        entity_list = list(entities)
        sentinel = any(entity.get("type") == self.policy.treatment_type for entity in entity_list)
        confidence_valid = (
            calibrated_confidence is not None
            and math.isfinite(calibrated_confidence)
            and 0.0 <= calibrated_confidence <= 1.0
        )
        uncertain = not confidence_valid or calibrated_confidence < self.policy.confidence_threshold
        requested = sentinel or uncertain or kg_conflict or window_conflict
        budget = max(1, int(self.total * self.policy.max_upgrade_rate + 0.999999))
        if requested and self.upgraded < budget:
            self.upgraded += 1
            reasons = [name for name, active in (
                ("treatment_sentinel", sentinel), ("low_confidence", uncertain),
                ("kg_conflict", kg_conflict), ("window_conflict", window_conflict),
            ) if active]
            return {"route": Route.QUALITY.value, "reason": "+".join(reasons)}
        if requested:
            return {"route": Route.INT8.value, "reason": "upgrade_budget_exhausted"}
        return {"route": Route.INT8.value, "reason": "calibrated_accept"}
