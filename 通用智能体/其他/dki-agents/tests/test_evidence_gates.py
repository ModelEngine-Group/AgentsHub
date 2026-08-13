from pathlib import Path

import pytest

from dki.evidence.schema import (
    EvidenceValidationError,
    reject_forbidden_file,
    sha256_file,
    validate_claim,
    validate_claim_files,
    verify_protected_manifest,
)


def _claim(path: Path, digest: str):
    return {
        "claim_id": "test-claim",
        "dataset_role": "development",
        "metric_kind": "exact_span_type",
        "runtime_kind": "shadow",
        "temperature_kind": "not_applicable",
        "evidence_level": "probe",
        "promotion_state": "candidate",
        "selection_used": False,
        "artifacts": [{"path": path.name, "sha256": digest}],
    }


def test_claim_hash_validation(tmp_path):
    artifact = tmp_path / "raw.json"
    artifact.write_text("{}", encoding="utf-8")
    validate_claim_files(_claim(artifact, sha256_file(artifact)), tmp_path)
    with pytest.raises(EvidenceValidationError, match="hash mismatch"):
        validate_claim_files(_claim(artifact, "0" * 64), tmp_path)


def test_blind_claim_cannot_select(tmp_path):
    artifact = tmp_path / "raw.json"
    artifact.write_text("{}", encoding="utf-8")
    claim = _claim(artifact, sha256_file(artifact))
    claim["dataset_role"] = "prospective_blind"
    claim["selection_used"] = True
    with pytest.raises(EvidenceValidationError, match="cannot be used"):
        validate_claim(claim)


def test_invalid_semantics_rejected(tmp_path):
    artifact = tmp_path / "raw.json"
    artifact.write_text("{}", encoding="utf-8")
    claim = _claim(artifact, sha256_file(artifact))
    claim["metric_kind"] = "span-ish"
    with pytest.raises(EvidenceValidationError, match="metric_kind"):
        validate_claim(claim)


def test_protected_manifest_and_content_based_forbidden_gate(tmp_path):
    protected = tmp_path / "final.jsonl"
    protected.write_text('{"secret":"historical blind"}\n', encoding="utf-8")
    digest = sha256_file(protected)
    manifest = {
        "protected": [{
            "path": protected.name,
            "sha256": digest,
            "size": protected.stat().st_size,
            "forbidden_for": ["calibration"],
        }]
    }
    verify_protected_manifest(manifest, tmp_path)
    renamed = tmp_path / "innocent-development-name.jsonl"
    renamed.write_bytes(protected.read_bytes())
    with pytest.raises(EvidenceValidationError, match="forbidden"):
        reject_forbidden_file(renamed, manifest, "calibration")


def test_artifact_cannot_escape_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    claim = _claim(outside, sha256_file(outside))
    claim["artifacts"][0]["path"] = "../outside.json"
    with pytest.raises(EvidenceValidationError, match="escapes"):
        validate_claim_files(claim, root)
