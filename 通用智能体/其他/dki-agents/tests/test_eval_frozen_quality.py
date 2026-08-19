"""冻结 quality checkpoint 一次性评测的逐类型统计测试。"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


@pytest.fixture(autouse=True)
def _scripts_on_path(monkeypatch):
    import sys

    monkeypatch.syspath_prepend(str(SCRIPTS))
    sys.modules.pop("eval_frozen_quality", None)


def test_frozen_quality_errors_are_counted_by_type():
    from eval_frozen_quality import _errors

    gold = {"d": [(0, 2, "Symptom"), (4, 7, "Disease"), (9, 11, "Exam")]}
    pred = {"d": [(1, 2, "Symptom"), (4, 7, "Drug"), (20, 22, "Exam")]}
    result = _errors(pred, gold)
    assert result["Symptom"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "tp": 1,
        "fp": 0,
        "fn": 0,
    }
    assert result["Disease"]["fn"] == 1
    assert result["Drug"]["fp"] == 1
    assert result["Exam"]["fp"] == 1
    assert result["Exam"]["fn"] == 1
