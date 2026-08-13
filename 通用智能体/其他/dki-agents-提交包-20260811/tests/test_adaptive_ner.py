import pytest

from dki.nlp.adaptive_router import AdaptiveRouter, RoutingPolicy
from dki.nlp.confidence import TemperatureCalibrator, calibration_metrics, fit_temperature


def test_temperature_requires_both_outcomes():
    with pytest.raises(ValueError):
        fit_temperature([1, 2], [1, 1])


def test_temperature_fit_does_not_worsen_training_nll():
    logits = [5.0, 4.0, -1.0, -0.5]
    labels = [1, 0, 0, 1]
    before = calibration_metrics(TemperatureCalibrator(1).predict(logits), labels)
    calibrator = fit_temperature(logits, labels)
    after = calibration_metrics(calibrator.predict(logits), labels)
    assert after["nll"] <= before["nll"]
    assert len(after["reliability"]) == 10


def test_router_stays_off_without_both_gates():
    router = AdaptiveRouter(RoutingPolicy(), calibration_valid=True, promotion_valid=False)
    assert router.decide([{"type": "Treatment"}], calibrated_confidence=0.1)["reason"] == "adaptive_not_promoted"


def test_router_uses_sentinel_and_respects_budget():
    router = AdaptiveRouter(RoutingPolicy(max_upgrade_rate=0.1), calibration_valid=True, promotion_valid=True)
    first = router.decide([{"type": "Treatment"}], calibrated_confidence=0.99)
    assert first["route"] == "quality"
    decisions = [router.decide([], calibrated_confidence=0.1) for _ in range(9)]
    assert any(item["reason"] == "upgrade_budget_exhausted" for item in decisions)


def test_nonfinite_or_out_of_range_confidence_fails_closed():
    for confidence in (float("nan"), float("inf"), -0.1, 1.1):
        router = AdaptiveRouter(RoutingPolicy(max_upgrade_rate=1), calibration_valid=True, promotion_valid=True)
        decision = router.decide([], calibrated_confidence=confidence)
        assert decision == {"route": "quality", "reason": "low_confidence"}


def test_kg_conflict_triggers_upgrade():
    router = AdaptiveRouter(RoutingPolicy(max_upgrade_rate=1), calibration_valid=True, promotion_valid=True)
    decision = router.decide([], calibrated_confidence=0.99, kg_conflict=True)
    assert decision == {"route": "quality", "reason": "kg_conflict"}
