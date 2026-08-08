"""Dependency-free confidence calibration and reliability diagnostics."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


def _clip(value: float, eps: float = 1e-6) -> float:
    return min(1.0 - eps, max(eps, float(value)))


def _logit(probability: float) -> float:
    p = _clip(probability)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def reliability_bins(
    confidences: Sequence[float],
    labels: Sequence[int | bool],
    n_bins: int = 10,
) -> list[dict]:
    """Return data for a reliability diagram.

    Empty bins are retained so two experiment reports are directly comparable.
    """
    if len(confidences) != len(labels):
        raise ValueError("confidences and labels must have equal length")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for confidence, label in zip(confidences, labels):
        conf = min(1.0, max(0.0, float(confidence)))
        index = min(n_bins - 1, int(conf * n_bins))
        buckets[index].append((conf, int(bool(label))))
    result = []
    for index, bucket in enumerate(buckets):
        lower, upper = index / n_bins, (index + 1) / n_bins
        count = len(bucket)
        result.append(
            {
                "lower": round(lower, 4),
                "upper": round(upper, 4),
                "count": count,
                "mean_confidence": round(sum(v[0] for v in bucket) / count, 6) if count else None,
                "accuracy": round(sum(v[1] for v in bucket) / count, 6) if count else None,
            }
        )
    return result


def expected_calibration_error(
    confidences: Sequence[float],
    labels: Sequence[int | bool],
    n_bins: int = 10,
) -> float:
    """Compute the standard weighted Expected Calibration Error (ECE)."""
    total = len(confidences)
    if not total:
        return 0.0
    error = 0.0
    for bucket in reliability_bins(confidences, labels, n_bins=n_bins):
        if not bucket["count"]:
            continue
        error += (bucket["count"] / total) * abs(bucket["accuracy"] - bucket["mean_confidence"])
    return round(error, 6)


@dataclass
class TemperatureCalibrator:
    """Binary temperature scaling with an optional calibration bias.

    The bias is important when a candidate generator's base precision is below
    50%; temperature-only scaling cannot move a probability across 0.5.  Fitting
    uses deterministic two-parameter Newton updates (equivalent to constrained
    Platt scaling) and requires no ML framework.
    """

    temperature: float = 1.0
    bias: float = 0.0
    fitted_samples: int = 0
    nll_before: float | None = None
    nll_after: float | None = None

    @staticmethod
    def _nll(
        confidences: Sequence[float],
        labels: Sequence[int | bool],
        temperature: float,
        bias: float = 0.0,
    ) -> float:
        losses = []
        for confidence, label in zip(confidences, labels):
            calibrated = _sigmoid(_logit(float(confidence)) / temperature + bias)
            y = int(bool(label))
            losses.append(-(y * math.log(_clip(calibrated)) + (1 - y) * math.log(_clip(1 - calibrated))))
        return sum(losses) / len(losses) if losses else 0.0

    def fit(
        self,
        confidences: Sequence[float],
        labels: Sequence[int | bool],
        min_temperature: float = 0.2,
        max_temperature: float = 5.0,
        steps: int = 60,
    ) -> "TemperatureCalibrator":
        if len(confidences) != len(labels):
            raise ValueError("confidences and labels must have equal length")
        if not confidences:
            self.temperature = 1.0
            self.bias = 0.0
            self.fitted_samples = 0
            self.nll_before = self.nll_after = 0.0
            return self
        if min_temperature <= 0 or max_temperature <= min_temperature or steps < 2:
            raise ValueError("invalid temperature search range")
        logits = [_logit(value) for value in confidences]
        targets = [int(bool(value)) for value in labels]
        slope, bias = 1.0, 0.0
        best = (self._nll(confidences, labels, 1.0, 0.0), slope, bias)
        for _ in range(steps):
            grad_slope = grad_bias = 0.0
            h_ss = h_sb = h_bb = 0.0
            for logit_value, target in zip(logits, targets):
                predicted = _sigmoid(slope * logit_value + bias)
                residual = predicted - target
                weight = max(1e-8, predicted * (1.0 - predicted))
                grad_slope += residual * logit_value
                grad_bias += residual
                h_ss += weight * logit_value * logit_value
                h_sb += weight * logit_value
                h_bb += weight
            # Tiny L2 term prevents singular Hessians on very small sets.
            h_ss += 1e-6
            h_bb += 1e-6
            determinant = h_ss * h_bb - h_sb * h_sb
            if abs(determinant) < 1e-12:
                break
            delta_slope = (h_bb * grad_slope - h_sb * grad_bias) / determinant
            delta_bias = (-h_sb * grad_slope + h_ss * grad_bias) / determinant
            candidate_slope = min(
                1.0 / min_temperature,
                max(1.0 / max_temperature, slope - delta_slope),
            )
            candidate_bias = min(10.0, max(-10.0, bias - delta_bias))
            candidate_temperature = 1.0 / candidate_slope
            loss = self._nll(confidences, labels, candidate_temperature, candidate_bias)
            if loss > best[0] and (abs(delta_slope) > 1e-6 or abs(delta_bias) > 1e-6):
                # Damped step when the full Newton update overshoots.
                candidate_slope = (slope + candidate_slope) / 2
                candidate_bias = (bias + candidate_bias) / 2
                candidate_temperature = 1.0 / candidate_slope
                loss = self._nll(confidences, labels, candidate_temperature, candidate_bias)
            slope, bias = candidate_slope, candidate_bias
            if loss < best[0]:
                best = (loss, slope, bias)
            if abs(delta_slope) < 1e-8 and abs(delta_bias) < 1e-8:
                break
        self.temperature = round(1.0 / best[1], 8)
        self.bias = round(best[2], 8)
        self.fitted_samples = len(confidences)
        self.nll_before = round(self._nll(confidences, labels, 1.0, 0.0), 8)
        self.nll_after = round(self._nll(confidences, labels, self.temperature, self.bias), 8)
        return self

    def transform_one(self, confidence: float) -> float:
        return round(
            _sigmoid(_logit(confidence) / max(self.temperature, 1e-6) + self.bias),
            6,
        )

    def transform(self, confidences: Iterable[float]) -> list[float]:
        return [self.transform_one(value) for value in confidences]

    def to_dict(self) -> dict:
        return {
            "method": "temperature_scaling",
            "temperature": self.temperature,
            "bias": self.bias,
            "fitted_samples": self.fitted_samples,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
        }

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return str(target)

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureCalibrator":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            temperature=float(data.get("temperature", 1.0)),
            bias=float(data.get("bias", 0.0)),
            fitted_samples=int(data.get("fitted_samples", 0)),
            nll_before=data.get("nll_before"),
            nll_after=data.get("nll_after"),
        )
