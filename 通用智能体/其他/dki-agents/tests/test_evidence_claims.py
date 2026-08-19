"""Typed-claim loader: validator-passing round-trip and fail-closed on tamper."""

import json

import pytest

from dki.evidence import (
    EvidenceValidationError,
    load_validated_claims,
    sha256_file,
    validate_claim,
)


def _claim(artifact_rel, sha):
    return {
        "claim_id": "unit_metric",
        "dataset_role": "internal_acceptance",
        "metric_kind": "throughput",
        "runtime_kind": "acl_om_npu",
        "temperature_kind": "warm",
        "evidence_level": "formal",
        "promotion_state": "candidate",
        "selection_used": False,
        "device_claim": "npu",
        "artifacts": [{"path": artifact_rel, "sha256": sha}],
        "metric": {"name": "throughput_samples_per_s", "value": 3048.28, "unit": "samples/s"},
    }


def test_load_validated_claims_roundtrip(tmp_path):
    artifact = tmp_path / "evidence.json"
    artifact.write_text('{"throughput": 3048.28}', encoding="utf-8")
    claims_file = tmp_path / "claims.json"
    claims_file.write_text(json.dumps([_claim("evidence.json", sha256_file(artifact))]), encoding="utf-8")
    claims = load_validated_claims(claims_file, tmp_path)
    assert len(claims) == 1 and claims[0]["claim_id"] == "unit_metric"


def test_load_fails_closed_on_tampered_artifact(tmp_path):
    artifact = tmp_path / "evidence.json"
    artifact.write_text('{"throughput": 3048.28}', encoding="utf-8")
    claims_file = tmp_path / "claims.json"
    claims_file.write_text(json.dumps([_claim("evidence.json", sha256_file(artifact))]), encoding="utf-8")
    artifact.write_text('{"throughput": 9999}', encoding="utf-8")  # tamper after hashing
    with pytest.raises(EvidenceValidationError, match="hash mismatch"):
        load_validated_claims(claims_file, tmp_path)


def test_blind_role_cannot_be_used_for_selection():
    claim = _claim("evidence.json", "0" * 64)
    claim["dataset_role"] = "prospective_blind"
    claim["selection_used"] = True
    with pytest.raises(EvidenceValidationError, match="selection"):
        validate_claim(claim)


def test_cpu_cannot_claim_npu_throughput():
    claim = _claim("evidence.json", "0" * 64)
    claim["runtime_kind"] = "cpu"
    claim["device_claim"] = "npu"
    with pytest.raises(EvidenceValidationError, match="CPU evidence cannot claim NPU"):
        validate_claim(claim)


def test_generated_headline_claims_validate_if_present():
    """If the repo already has generated claims, they must all validate."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    claims_file = root / "outputs" / "claims" / "headline_claims.json"
    if not claims_file.is_file():
        pytest.skip("headline claims not generated in this checkout")
    claims = load_validated_claims(claims_file, root)
    assert claims and all(c.get("claim_id") for c in claims)
