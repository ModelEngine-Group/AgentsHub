"""Source-controlled anchors for official-v2 development provenance.

The allowlist in this module accepts only reviewed public-receipt bytes under a
trusted, reviewed distribution and an ACL-controlled output root.  It stops a
caller from substituting a self-consistent receipt, but it does not protect
against a party that can modify this verifier or its runtime.  That stronger
threat model needs an external signed bootstrap or isolated runtime and is
outside this repository-local mechanism.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping


_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_RID = re.compile(r"\A\d{8}T\d{6}Z-g[0-9a-f]{7}-t[0-9a-f]{12}\Z")
_BINDING_SCHEMA_VERSION = 1
_BINDING_AUTHORITY = "source_controlled_official_v2_development_anchor"


@dataclass(frozen=True)
class OfficialV2DevelopmentAnchor:
    """One reviewed official-v2 receipt and its approved private bindings."""

    rid: str
    receipt_sha256: str
    producer_source_tree_sha256: str
    private_development_sha256: str
    target_db_sha256: str
    evidence_relative_path: str
    input_relative_path: str | None
    database_relative_path: str | None

    def expanded_evidence_relative_path(self) -> str:
        """Return the root-relative receipt path after substituting ``<RID>``."""
        return _expand_rid_path(
            self.evidence_relative_path,
            rid=self.rid,
            label="evidence_relative_path",
        )

    def expanded_input_relative_path(self) -> str | None:
        """Return an optional exact input path for a later reviewed anchor."""
        return _expand_optional_rid_path(
            self.input_relative_path,
            rid=self.rid,
            label="input_relative_path",
        )

    def expanded_database_relative_path(self) -> str | None:
        """Return an optional exact database path for a later reviewed anchor."""
        return _expand_optional_rid_path(
            self.database_relative_path,
            rid=self.rid,
            label="database_relative_path",
        )

    def trusted_binding(self) -> dict[str, Any]:
        """Return the exact non-user-controlled binding carried by receipts."""
        return {
            "schema_version": _BINDING_SCHEMA_VERSION,
            "authority": _BINDING_AUTHORITY,
            **asdict(self),
        }


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase 64-hex SHA-256")
    return value


def _expand_rid_path(value: str, *, rid: str, label: str) -> str:
    if not isinstance(value, str) or value.count("<RID>") != 1:
        raise ValueError(f"{label} must contain exactly one <RID> placeholder")
    expanded = value.replace("<RID>", rid)
    path = PurePosixPath(expanded)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[:3] != ("outputs", "remote_runs", rid)
    ):
        raise ValueError(f"{label} is not a safe public RID-relative path")
    return path.as_posix()


def _expand_optional_rid_path(
    value: str | None,
    *,
    rid: str,
    label: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be null or a non-empty relative path")
    expanded = value.replace("<RID>", rid)
    path = PurePosixPath(expanded)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} is not a safe relative path")
    return path.as_posix()


def _validate_anchor(anchor: OfficialV2DevelopmentAnchor) -> None:
    if not isinstance(anchor, OfficialV2DevelopmentAnchor):
        raise TypeError("official development anchor has an invalid type")
    if not _RID.fullmatch(anchor.rid):
        raise ValueError("official development anchor RID is malformed")
    for label in (
        "receipt_sha256",
        "producer_source_tree_sha256",
        "private_development_sha256",
        "target_db_sha256",
    ):
        _require_sha256(getattr(anchor, label), f"official development anchor {label}")
    anchor.expanded_evidence_relative_path()
    anchor.expanded_input_relative_path()
    anchor.expanded_database_relative_path()


# Add later anchors only by editing this source-controlled tuple in a reviewed
# distribution.  The initial historical record intentionally has no inferred
# private input/database pathname; their bytes are bound by the approved hashes.
OFFICIAL_V2_DEVELOPMENT_ANCHORS: tuple[OfficialV2DevelopmentAnchor, ...] = (
    OfficialV2DevelopmentAnchor(
        rid="20260726T122144Z-g30fce24-t963fd4161f4f",
        receipt_sha256=(
            "af888a1fec92eb26b19dff9aee931d17cb53a41a05d1f4f7a985547ade5aace8"
        ),
        producer_source_tree_sha256=(
            "963fd4161f4f232d781bdb2c8731ddf4d2c935c469ab28d04dc06bdd173ea7fe"
        ),
        private_development_sha256=(
            "d464cccf889ef3fd389c3286a8b198b306124665b3ed4e9c19feaa3e63d9df67"
        ),
        target_db_sha256=(
            "e7721fbaf49a81cc157af453728f004edb4a08aed0f3c449e1e40036562455d8"
        ),
        evidence_relative_path=(
            "outputs/remote_runs/<RID>/nl2sql/development_projection_receipt.json"
        ),
        input_relative_path=None,
        database_relative_path=None,
    ),
)


def _index_anchors(
    anchors: tuple[OfficialV2DevelopmentAnchor, ...],
) -> dict[str, OfficialV2DevelopmentAnchor]:
    by_digest: dict[str, OfficialV2DevelopmentAnchor] = {}
    rids: set[str] = set()
    for anchor in anchors:
        _validate_anchor(anchor)
        if anchor.rid in rids:
            raise RuntimeError("official development anchors contain a duplicate RID")
        if anchor.receipt_sha256 in by_digest:
            raise RuntimeError(
                "official development anchors contain a duplicate receipt digest"
            )
        rids.add(anchor.rid)
        by_digest[anchor.receipt_sha256] = anchor
    if not by_digest:
        raise RuntimeError("official development anchor allowlist must not be empty")
    return by_digest


_ANCHORS_BY_RECEIPT_SHA256 = _index_anchors(OFFICIAL_V2_DEVELOPMENT_ANCHORS)


def official_v2_development_anchor_for_receipt_sha256(
    receipt_sha256: str,
) -> OfficialV2DevelopmentAnchor:
    """Return the one source-controlled anchor for approved receipt bytes."""
    digest = _require_sha256(receipt_sha256, "development evidence SHA-256")
    try:
        return _ANCHORS_BY_RECEIPT_SHA256[digest]
    except KeyError as exc:
        raise ValueError(
            "development evidence SHA-256 is not in the source-controlled allowlist"
        ) from exc


def trusted_evidence_binding(
    anchor: OfficialV2DevelopmentAnchor,
) -> dict[str, Any]:
    """Serialize a reviewed anchor for a private evaluator or auditor receipt."""
    _validate_anchor(anchor)
    return anchor.trusted_binding()


def anchor_from_trusted_evidence_binding(
    binding: Mapping[str, Any],
) -> OfficialV2DevelopmentAnchor:
    """Reject every binding except an exact serialization of a static anchor."""
    if not isinstance(binding, Mapping):
        raise ValueError("trusted development evidence binding must be an object")
    expected_keys = {
        "schema_version",
        "authority",
        "rid",
        "receipt_sha256",
        "producer_source_tree_sha256",
        "private_development_sha256",
        "target_db_sha256",
        "evidence_relative_path",
        "input_relative_path",
        "database_relative_path",
    }
    if set(binding) != expected_keys:
        raise ValueError("trusted development evidence binding has unknown or missing fields")
    if type(binding["schema_version"]) is not int or binding["schema_version"] != _BINDING_SCHEMA_VERSION:
        raise ValueError("trusted development evidence binding schema is invalid")
    if binding["authority"] != _BINDING_AUTHORITY:
        raise ValueError("trusted development evidence binding authority is invalid")
    anchor = official_v2_development_anchor_for_receipt_sha256(
        binding["receipt_sha256"]
    )
    if dict(binding) != anchor.trusted_binding():
        raise ValueError("trusted development evidence binding differs from its static anchor")
    return anchor
