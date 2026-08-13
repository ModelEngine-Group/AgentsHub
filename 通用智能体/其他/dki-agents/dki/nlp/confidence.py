"""Lightweight confidence calibration metrics for adaptive NER routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float

    def predict(self, logits: Sequence[float]) -> list[float]:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        return [sigmoid(float(logit) / self.temperature) for logit in logits]


def binary_nll(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    eps = 1e-12
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("probabilities and labels must be non-empty and aligned")
    return sum(
        -(int(label) * math.log(min(1 - eps, max(eps, probability)))
          + (1 - int(label)) * math.log(min(1 - eps, max(eps, 1 - probability))))
        for probability, label in zip(probabilities, labels)
    ) / len(labels)


def fit_temperature(logits: Sequence[float], labels: Sequence[int]) -> TemperatureCalibrator:
    """Fit one pooled temperature by deterministic log-space grid refinement."""
    if len(logits) != len(labels) or len(logits) < 2 or len(set(labels)) < 2:
        raise ValueError("calibration needs aligned positive and negative events")
    best_t = 1.0
    for step in (0.2, 0.05, 0.01):
        candidates = [max(0.05, best_t + step * offset) for offset in range(-10, 11)]
        best_t = min(candidates, key=lambda value: binary_nll(TemperatureCalibrator(value).predict(logits), labels))
    return TemperatureCalibrator(round(best_t, 4))


def calibration_metrics(probabilities: Sequence[float], labels: Sequence[int], bins: int = 10) -> dict:
    if len(probabilities) != len(labels) or not labels:
        raise ValueError("probabilities and labels must be non-empty and aligned")
    n = len(labels)
    brier = sum((float(p) - int(y)) ** 2 for p, y in zip(probabilities, labels)) / n
    ece = 0.0
    reliability = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, p in enumerate(probabilities) if low <= p < high or (index == bins - 1 and p == 1)]
        if not members:
            reliability.append({"low": low, "high": high, "n": 0, "confidence": None, "accuracy": None})
            continue
        confidence = sum(probabilities[i] for i in members) / len(members)
        accuracy = sum(labels[i] for i in members) / len(members)
        ece += len(members) / n * abs(confidence - accuracy)
        reliability.append({"low": low, "high": high, "n": len(members), "confidence": round(confidence, 6), "accuracy": round(accuracy, 6)})
    return {"nll": round(binary_nll(probabilities, labels), 6), "brier": round(brier, 6), "ece": round(ece, 6), "reliability": reliability}
