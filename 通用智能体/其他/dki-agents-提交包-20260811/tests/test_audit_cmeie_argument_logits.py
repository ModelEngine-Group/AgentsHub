import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_cmeie_argument_logits.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_cmeie_argument_logits",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_argument_audit_threshold_grid_accepts_negative_finite_values():
    assert MODULE.parse_threshold_grid("0,-1.0,-0.5,-1") == (
        -1.0,
        -0.5,
        0.0,
    )
    with pytest.raises(ValueError, match="finite"):
        MODULE.parse_threshold_grid("-1,nan")
    with pytest.raises(ValueError, match="numbers"):
        MODULE.parse_threshold_grid("-1,nope")


def test_argument_audit_valid_mask_matches_decoder_contract():
    offsets = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 2],
            [0, 0],
        ]
    )
    mask = MODULE._valid_span_mask(offsets, torch)
    assert mask.nonzero(as_tuple=False).tolist() == [
        [1, 1],
        [1, 2],
        [2, 2],
    ]


def test_argument_audit_publish_is_create_once(tmp_path: Path):
    output = tmp_path / "audit.json"
    payload = {"contains_text_gold_or_predictions": False}
    MODULE.publish_once(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        MODULE.publish_once(output, payload)
