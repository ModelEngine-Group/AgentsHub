"""Create the fail-closed CMeIE status receipt after PRGC preflight failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"CMeIE block receipt exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--overfit-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    args = parser.parse_args(argv)
    audit_path = args.candidate_audit.resolve()
    overfit_path = args.overfit_receipt.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"CMeIE block receipt exists: {output}")
    if not audit_path.is_file() or not overfit_path.is_file():
        raise SystemExit("candidate audit or PRGC overfit receipt is missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    overfit = json.loads(overfit_path.read_text(encoding="utf-8"))
    if (
        audit.get("kind") != "cmeie_casrel_predicate_aware_candidate_audit"
        or audit.get("outer_accessed") is not False
        or audit.get("official_dev_accessed") is not False
        or audit.get("gate", {}).get("passed") is not False
    ):
        raise SystemExit("candidate audit does not justify the PRGC fallback")
    if (
        overfit.get("kind") != "cmeie_prgc_overfit_preflight"
        or overfit.get("model_family") != "prgc"
        or overfit.get("passed") is not False
        or overfit.get("calibration_accessed") is not False
        or overfit.get("outer_accessed") is not False
        or overfit.get("official_dev_accessed") is not False
    ):
        raise SystemExit("PRGC preflight is not a valid failed-only receipt")
    payload = {
        "schema_version": 1,
        "kind": "cmeie_model_quality_block",
        "status": "BLOCKED_MODEL_QUALITY",
        "block_phase": "prgc_overfit_preflight",
        "reason": "fixed_8_row_44_spo_overfit_did_not_reach_required_exact_spo_f1",
        "candidate_audit": {
            "path": str(audit_path),
            "sha256": _sha(audit_path),
            "candidate_recall": audit.get("candidate_recall"),
            "density_rejections": audit.get("density", {}).get("rejections"),
        },
        "prgc_overfit": {
            "path": str(overfit_path),
            "sha256": _sha(overfit_path),
            "head_only_f1": overfit.get("evaluation", {}).get("head_only", {}).get("metric", {}).get("f1"),
            "joint_f1": overfit.get("evaluation", {}).get("joint", {}).get("metric", {}).get("f1"),
            "required_head_only_f1": 0.98,
            "required_joint_f1": 0.95,
        },
        "source_tree_sha256": args.source_tree_sha256,
        "calibration_accessed": False,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "seed42_or_seed43_formal_training_started": False,
        "permitted_next_action": "new_architecture_requires_a_new_approved_plan",
        "default_rule_pipeline_unchanged": True,
        "script_sha256": _sha(Path(__file__)),
    }
    _write_once(output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
