import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_cmeie_checkpoint_logits.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_cmeie_checkpoint_logits",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_score_summary_is_deterministic_and_handles_empty():
    assert MODULE.summarise_scores([]) == {
        "count": 0,
        "minimum": None,
        "p10": None,
        "p50": None,
        "p90": None,
        "maximum": None,
    }
    assert MODULE.summarise_scores([-2.0, -1.0, 0.0, 3.0]) == {
        "count": 4,
        "minimum": -2.0,
        "p10": -2.0,
        "p50": -1.0,
        "p90": 3.0,
        "maximum": 3.0,
    }


def test_logit_audit_publish_is_create_once(tmp_path: Path):
    output = tmp_path / "audit.json"
    payload = {
        "contains_text_or_predictions": False,
        "heads": {},
    }
    MODULE.publish_once(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        MODULE.publish_once(output, payload)


def test_logit_audit_selects_only_explicit_fit_or_calibration():
    rows = [{"id": index} for index in range(6)]
    manifest = {
        "splits": {
            "fit": {"indices": [0, 2, 4]},
            "calibration": {"indices": [1, 3]},
            "outer": {"indices": [5]},
        }
    }

    assert MODULE.select_split_rows(rows, manifest, "fit", 2) == [
        {"id": 0},
        {"id": 2},
    ]
    assert MODULE.select_split_rows(
        rows,
        manifest,
        "calibration",
        0,
    ) == [{"id": 1}, {"id": 3}]
    with pytest.raises(ValueError, match="fit.*calibration"):
        MODULE.select_split_rows(rows, manifest, "outer", 0)
    with pytest.raises(ValueError, match="non-negative"):
        MODULE.select_split_rows(rows, manifest, "fit", -1)
