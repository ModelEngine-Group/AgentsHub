import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gate_cmeie_softlex_ablation",
    ROOT / "scripts" / "gate_cmeie_softlex_ablation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _metric(*, precision, recall, f1, predicted_mentions):
    return {
        "tp": 10,
        "fp": 2,
        "fn": 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_mentions": predicted_mentions,
    }


def test_softlex_ablation_requires_relative_and_absolute_gates():
    baseline = _metric(
        precision=0.72,
        recall=0.62,
        f1=0.66,
        predicted_mentions=100,
    )
    augmented = _metric(
        precision=0.71,
        recall=0.89,
        f1=0.79,
        predicted_mentions=180,
    )
    decision = MODULE.ablation_decision(baseline, augmented)
    assert decision["passed"] is True
    augmented["recall"] = 0.87
    assert MODULE.ablation_decision(baseline, augmented)["passed"] is False


def test_softlex_ablation_rejects_candidate_explosion_and_precision_loss():
    baseline = _metric(
        precision=0.75,
        recall=0.60,
        f1=0.66,
        predicted_mentions=100,
    )
    augmented = _metric(
        precision=0.70,
        recall=0.90,
        f1=0.79,
        predicted_mentions=201,
    )
    decision = MODULE.ablation_decision(baseline, augmented)
    assert decision["passed"] is False
    assert decision["gates"]["candidate_ratio_at_most_2"] is False
    assert decision["gates"]["precision_drop_at_most_0_03"] is False
