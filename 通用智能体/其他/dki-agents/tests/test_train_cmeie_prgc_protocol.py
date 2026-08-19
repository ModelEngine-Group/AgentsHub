from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_cmeie_prgc.py"
SPEC = importlib.util.spec_from_file_location("train_cmeie_prgc", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_formal_prgc_refuses_full_unfreeze_before_opening_data(tmp_path):
    with pytest.raises(SystemExit, match="formal PRGC requires"):
        MODULE.main(
            [
                "--mode",
                "train",
                "--fit-data",
                str(tmp_path / "missing-fit.jsonl"),
                "--calibration-data",
                str(tmp_path / "missing-calibration.jsonl"),
                "--split-manifest",
                str(tmp_path / "missing-split.json"),
                "--base-model",
                str(tmp_path / "missing-model"),
                "--output",
                str(tmp_path / "output"),
                "--source-tree-sha256",
                "a" * 64,
                "--joint-top-encoder-layers",
                "0",
            ]
        )
