from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gate_cmeie_prgc.py"
SPEC = importlib.util.spec_from_file_location("gate_cmeie_prgc", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _selection(seed: int, f1: float) -> dict:
    return {
        "kind": "cmeie_prgc_calibration_selection",
        "model_family": "prgc",
        "seed": seed,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "fit_sha256": "a" * 64,
        "calibration_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
        "predicates": ["p"],
        "calibration_metric": {"f1": f1, "precision": 0.3, "recall": 0.4},
    }


def test_prgc_early_gate_marks_two_failed_seeds_blocked(tmp_path):
    left = tmp_path / "seed42.json"
    right = tmp_path / "seed43.json"
    left.write_text(json.dumps(_selection(42, 0.44)), encoding="utf-8")
    right.write_text(json.dumps(_selection(43, 0.10)), encoding="utf-8")
    output = tmp_path / "gate.json"
    result = MODULE.main(
        [
            "--selection",
            str(left),
            "--selection",
            str(right),
            "--output",
            str(output),
            "--source-tree-sha256",
            "d" * 64,
        ]
    )
    assert result == 2
    assert json.loads(output.read_text(encoding="utf-8"))["decision"] == "BLOCKED_MODEL_QUALITY"
