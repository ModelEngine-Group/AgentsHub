from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "attest_cmeie_prgc_blocked.py"
SPEC = importlib.util.spec_from_file_location("attest_cmeie_prgc_blocked", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_failed_prgc_preflight_produces_no_outer_block_receipt(tmp_path):
    audit = tmp_path / "audit.json"
    overfit = tmp_path / "overfit.json"
    audit.write_text(
        json.dumps(
            {
                "kind": "cmeie_casrel_predicate_aware_candidate_audit",
                "outer_accessed": False,
                "official_dev_accessed": False,
                "candidate_recall": 0.38,
                "density": {"rejections": 1},
                "gate": {"passed": False},
            }
        ),
        encoding="utf-8",
    )
    overfit.write_text(
        json.dumps(
            {
                "kind": "cmeie_prgc_overfit_preflight",
                "model_family": "prgc",
                "passed": False,
                "calibration_accessed": False,
                "outer_accessed": False,
                "official_dev_accessed": False,
                "evaluation": {
                    "head_only": {"metric": {"f1": 0.17}},
                    "joint": {"metric": {"f1": 0.18}},
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "blocked.json"
    assert MODULE.main(
        [
            "--candidate-audit",
            str(audit),
            "--overfit-receipt",
            str(overfit),
            "--output",
            str(output),
            "--source-tree-sha256",
            "a" * 64,
        ]
    ) == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED_MODEL_QUALITY"
    assert payload["outer_accessed"] is False
