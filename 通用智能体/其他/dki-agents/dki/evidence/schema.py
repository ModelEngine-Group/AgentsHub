"""Typed, machine-verifiable evidence claims for new optimization results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_ALLOWED = {
    "dataset_role": {"train", "development", "internal_acceptance", "prospective_blind", "historical_blind"},
    "metric_kind": {"exact_span_type", "relaxed_overlap_type", "throughput", "latency", "energy"},
    "runtime_kind": {"cpu", "pytorch_npu", "acl_om_npu", "shadow"},
    "temperature_kind": {"cold", "warm", "mixed", "not_applicable"},
    "evidence_level": {"probe", "formal", "historical"},
    "promotion_state": {"candidate", "shadow", "promoted", "rejected", "historical"},
}


class EvidenceValidationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_claim(claim: Mapping[str, Any]) -> None:
    for field, allowed in _ALLOWED.items():
        value = claim.get(field)
        if value not in allowed:
            raise EvidenceValidationError(f"{field} must be one of {sorted(allowed)}, got {value!r}")
    if not isinstance(claim.get("claim_id"), str) or not claim["claim_id"]:
        raise EvidenceValidationError("claim_id is required")
    artifacts = claim.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise EvidenceValidationError("at least one artifact is required")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise EvidenceValidationError("artifact entries must be objects")
        if not artifact.get("path") or not artifact.get("sha256"):
            raise EvidenceValidationError("artifact path and sha256 are required")
    if claim["dataset_role"] in {"prospective_blind", "historical_blind"} and claim.get("selection_used"):
        raise EvidenceValidationError("blind evidence cannot be used for selection")
    if claim["runtime_kind"] == "cpu" and claim["metric_kind"] in {"throughput", "latency", "energy"}:
        if claim.get("device_claim") == "npu":
            raise EvidenceValidationError("CPU evidence cannot claim NPU runtime")


def verify_protected_manifest(manifest: Mapping[str, Any], root: Path) -> None:
    entries = manifest.get("protected")
    if not isinstance(entries, list) or not entries:
        raise EvidenceValidationError("protected manifest has no entries")
    for entry in entries:
        path = (root / entry["path"]).resolve()
        if not path.is_file():
            raise EvidenceValidationError(f"protected artifact missing: {entry['path']}")
        if path.stat().st_size != int(entry["size"]):
            raise EvidenceValidationError(f"protected size mismatch: {entry['path']}")
        if sha256_file(path) != entry["sha256"]:
            raise EvidenceValidationError(f"protected hash mismatch: {entry['path']}")


def forbidden_hashes(manifest: Mapping[str, Any], purpose: str) -> set[str]:
    return {
        entry["sha256"] for entry in manifest.get("protected", [])
        if purpose in entry.get("forbidden_for", [])
    }


def reject_forbidden_file(path: Path, manifest: Mapping[str, Any], purpose: str) -> None:
    digest = sha256_file(path)
    if digest in forbidden_hashes(manifest, purpose):
        raise EvidenceValidationError(
            f"{path} is protected and forbidden for {purpose} (sha256={digest})"
        )


def validate_claim_files(claim: Mapping[str, Any], root: Path) -> None:
    validate_claim(claim)
    for artifact in claim["artifacts"]:
        path = (root / artifact["path"]).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise EvidenceValidationError(f"artifact escapes repository: {path}") from exc
        if not path.is_file():
            raise EvidenceValidationError(f"artifact does not exist: {path}")
        actual = sha256_file(path)
        if actual != artifact["sha256"]:
            raise EvidenceValidationError(
                f"artifact hash mismatch for {artifact['path']}: {actual} != {artifact['sha256']}"
            )


def load_validated_claims(claims_path: Path, root: Path) -> list[dict]:
    """Load a claims JSON and return only claims whose schema + artifact hashes
    validate. Fail closed: any invalid claim raises. Use this instead of reading
    raw historical JSON so displayed numbers are always machine-verified."""
    payload = json.loads(Path(claims_path).read_text(encoding="utf-8"))
    claims = payload if isinstance(payload, list) else [payload]
    for claim in claims:
        validate_claim_files(claim, root)
    return claims
