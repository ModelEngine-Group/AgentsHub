"""Fail-closed NL2SQL rule ablation and provenance primitives.

The module is independent of the NL2SQL router and compiler implementations.  It
provides stable cluster/rule identities, immutable policy configuration, public-
safe route provenance, paired strict comparisons, and measured admission gates.
All item-level evaluation remains outside this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any


ABLATION_CONFIG_SCHEMA_VERSION = 2
PROVENANCE_SCHEMA_VERSION = 1

RULE_CLUSTER_CLOSED_STRUCTURAL_OWNERS = "closed_structural_owners"
RULE_CLUSTER_VERSIONED_COMPILERS = "versioned_compilers"
RULE_CLUSTER_TEMPORAL_COHORT_WINDOW_TOPK = "temporal_cohort_window_topk"
RULE_CLUSTER_KG = "kg"
RULE_CLUSTER_EXACT_LEXICAL_CURATED = "exact_lexical_curated"
RULE_CLUSTER_OUTPUT_FORMATTING = "output_formatting"

KNOWN_RULE_CLUSTER_IDS: tuple[str, ...] = (
    RULE_CLUSTER_CLOSED_STRUCTURAL_OWNERS,
    RULE_CLUSTER_VERSIONED_COMPILERS,
    RULE_CLUSTER_TEMPORAL_COHORT_WINDOW_TOPK,
    RULE_CLUSTER_KG,
    RULE_CLUSTER_EXACT_LEXICAL_CURATED,
    RULE_CLUSTER_OUTPUT_FORMATTING,
)
RULE_CLUSTER_IDS = frozenset(KNOWN_RULE_CLUSTER_IDS)

KNOWN_ERROR_CATEGORIES: tuple[str, ...] = (
    "semantic_reasoning",
    "abstention",
    "execution_resource",
    "row_order",
    "tie",
    "rounding",
    "column_order",
    "column_count",
)

_STABLE_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._:-][a-z0-9]+)*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(payload: Any) -> str:
    """Return the SHA-256 of one canonical, finite JSON encoding."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_sha256(value: str, *, field_name: str = "sha256") -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def validate_stable_id(value: str, *, field_name: str = "stable_id") -> str:
    if (
        not isinstance(value, str)
        or len(value) > 96
        or not _STABLE_ID_RE.fullmatch(value)
    ):
        raise ValueError(
            f"{field_name} must be a lowercase stable identifier of at most 96 characters"
        )
    return value


def validate_rule_cluster_id(cluster_id: str) -> str:
    if not isinstance(cluster_id, str) or cluster_id not in RULE_CLUSTER_IDS:
        raise ValueError(
            f"unknown NL2SQL rule cluster {cluster_id!r}; "
            f"known={KNOWN_RULE_CLUSTER_IDS!r}"
        )
    return cluster_id


def validate_rule_id(rule_id: str) -> str:
    return validate_stable_id(rule_id, field_name="rule_id")


def _normalise_reason_codes(values: Iterable[str]) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("reason_codes must be an iterable of stable IDs")
    result: set[str] = set()
    for value in values:
        reason = validate_stable_id(value, field_name="reason_code")
        if reason in result:
            raise ValueError(f"duplicate reason code: {reason}")
        result.add(reason)
    return tuple(sorted(result))


def _alias_value(
    payload: Mapping[str, Any],
    names: Sequence[str],
    *,
    field_name: str,
    required: bool = False,
) -> Any:
    present = [(name, payload[name]) for name in names if name in payload]
    if not present:
        if required:
            raise ValueError(f"missing {field_name}")
        return None
    first = present[0][1]
    if any(value != first for _, value in present[1:]):
        raise ValueError(f"conflicting aliases for {field_name}: {present!r}")
    return first


@dataclass(frozen=True)
class RuleRegistration:
    """One stable rule-to-cluster binding."""

    rule_id: str
    cluster_id: str
    description: str = ""

    def __post_init__(self) -> None:
        validate_rule_id(self.rule_id)
        validate_rule_cluster_id(self.cluster_id)
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")

    def binding_payload(self) -> dict[str, str]:
        return {"rule_id": self.rule_id, "cluster_id": self.cluster_id}


@dataclass(frozen=True)
class RuleRegistrySnapshot:
    """Immutable rule registry used to validate configs and provenance."""

    registrations: tuple[RuleRegistration, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.registrations, tuple):
            object.__setattr__(self, "registrations", tuple(self.registrations))
        seen: set[str] = set()
        ordered: list[RuleRegistration] = []
        for registration in self.registrations:
            if not isinstance(registration, RuleRegistration):
                raise TypeError("registry entries must be RuleRegistration instances")
            if registration.rule_id in seen:
                raise ValueError(f"duplicate rule_id in registry: {registration.rule_id}")
            seen.add(registration.rule_id)
            ordered.append(registration)
        if not ordered:
            raise ValueError("rule registry must not be empty")
        object.__setattr__(
            self,
            "registrations",
            tuple(sorted(ordered, key=lambda item: item.rule_id)),
        )

    @classmethod
    def from_payload(cls, payload: Sequence[Mapping[str, Any]]) -> "RuleRegistrySnapshot":
        if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
            raise TypeError("registry payload must be a sequence")
        registrations: list[RuleRegistration] = []
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping):
                raise TypeError(f"registry entry {index} must be an object")
            unknown = set(item) - {"rule_id", "cluster_id", "description"}
            if unknown:
                raise ValueError(f"registry entry {index} has unknown fields: {sorted(unknown)!r}")
            registrations.append(
                RuleRegistration(
                    rule_id=item.get("rule_id"),
                    cluster_id=item.get("cluster_id"),
                    description=item.get("description", ""),
                )
            )
        return cls(tuple(registrations))

    @property
    def registry_hash(self) -> str:
        return canonical_sha256(self.canonical_payload())

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "registrations": [
                registration.binding_payload() for registration in self.registrations
            ],
        }

    def cluster_for(self, rule_id: str) -> str:
        validate_rule_id(rule_id)
        for registration in self.registrations:
            if registration.rule_id == rule_id:
                return registration.cluster_id
        raise ValueError(f"unknown rule_id in registry snapshot: {rule_id}")

    def validate_membership(self, rule_id: str, cluster_id: str) -> None:
        validate_rule_cluster_id(cluster_id)
        registered = self.cluster_for(rule_id)
        if registered != cluster_id:
            raise ValueError(
                f"rule/cluster mismatch for {rule_id}: registered={registered}, supplied={cluster_id}"
            )


def validate_rule_registry(
    registrations: Sequence[RuleRegistration],
) -> tuple[RuleRegistration, ...]:
    return RuleRegistrySnapshot(tuple(registrations)).registrations


DEFAULT_RULE_REGISTRY = RuleRegistrySnapshot(
    (
        RuleRegistration("closed.medical_analytics", RULE_CLUSTER_CLOSED_STRUCTURAL_OWNERS),
        RuleRegistration("closed.shared_tail_peer", RULE_CLUSTER_CLOSED_STRUCTURAL_OWNERS),
        RuleRegistration("closed.compositional", RULE_CLUSTER_CLOSED_STRUCTURAL_OWNERS),
        RuleRegistration("structural.closure", RULE_CLUSTER_CLOSED_STRUCTURAL_OWNERS),
        RuleRegistration("structural.explicit_four_table", RULE_CLUSTER_CLOSED_STRUCTURAL_OWNERS),
        RuleRegistration("compiler.v23", RULE_CLUSTER_VERSIONED_COMPILERS),
        RuleRegistration("compiler.v21", RULE_CLUSTER_VERSIONED_COMPILERS),
        RuleRegistration("compiler.structured", RULE_CLUSTER_VERSIONED_COMPILERS),
        RuleRegistration("compiler.v11", RULE_CLUSTER_VERSIONED_COMPILERS),
        RuleRegistration("compiler.v10", RULE_CLUSTER_VERSIONED_COMPILERS),
        RuleRegistration("compiler.v9", RULE_CLUSTER_VERSIONED_COMPILERS),
        RuleRegistration("compiler.v8", RULE_CLUSTER_VERSIONED_COMPILERS),
        RuleRegistration("compiler.v7", RULE_CLUSTER_VERSIONED_COMPILERS),
        RuleRegistration("temporal.cohort_window_topk", RULE_CLUSTER_TEMPORAL_COHORT_WINDOW_TOPK),
        RuleRegistration("kg.endpoint_join", RULE_CLUSTER_KG),
        RuleRegistration("kg.entity_range", RULE_CLUSTER_KG),
        RuleRegistration("kg.relation", RULE_CLUSTER_KG),
        RuleRegistration("kg.rule", RULE_CLUSTER_KG),
        RuleRegistration("lexical.exact_curated", RULE_CLUSTER_EXACT_LEXICAL_CURATED),
        RuleRegistration("lexical.clause_composer_v28", RULE_CLUSTER_EXACT_LEXICAL_CURATED),
        RuleRegistration("format.output_complete_structural", RULE_CLUSTER_OUTPUT_FORMATTING),
        RuleRegistration("format.relation_count", RULE_CLUSTER_OUTPUT_FORMATTING),
        RuleRegistration("format.relation_average", RULE_CLUSTER_OUTPUT_FORMATTING),
    )
)


def _normalise_clusters(
    values: Iterable[str] | None,
    *,
    default_all: bool,
    field_name: str,
) -> tuple[str, ...]:
    if values is None:
        return KNOWN_RULE_CLUSTER_IDS if default_all else ()
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of cluster IDs")
    seen: set[str] = set()
    for value in values:
        cluster = validate_rule_cluster_id(value)
        if cluster in seen:
            raise ValueError(f"duplicate cluster in {field_name}: {cluster}")
        seen.add(cluster)
    return tuple(cluster for cluster in KNOWN_RULE_CLUSTER_IDS if cluster in seen)


@dataclass(frozen=True)
class NL2SQLAblationConfig:
    """Immutable profile with separate effective-policy and profile identities."""

    profile_name: str = "baseline"
    enabled_clusters: tuple[str, ...] | None = None
    disabled_clusters: tuple[str, ...] = ()
    disabled_rule_ids: tuple[str, ...] = ()
    registry: RuleRegistrySnapshot = DEFAULT_RULE_REGISTRY
    schema_version: int = field(
        default=ABLATION_CONFIG_SCHEMA_VERSION,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_name, field_name="profile_name")
        if not isinstance(self.registry, RuleRegistrySnapshot):
            raise TypeError("registry must be a RuleRegistrySnapshot")
        requested = _normalise_clusters(
            self.enabled_clusters,
            default_all=True,
            field_name="enabled_clusters",
        )
        explicitly_disabled = set(
            _normalise_clusters(
                self.disabled_clusters,
                default_all=False,
                field_name="disabled_clusters",
            )
        )
        effective_clusters = tuple(
            cluster for cluster in requested if cluster not in explicitly_disabled
        )
        disabled_clusters = tuple(
            cluster for cluster in KNOWN_RULE_CLUSTER_IDS if cluster not in effective_clusters
        )
        if isinstance(self.disabled_rule_ids, (str, bytes)):
            raise TypeError("disabled_rule_ids must be an iterable of rule IDs")
        seen_rules: set[str] = set()
        effective_disabled_rules: list[str] = []
        for value in self.disabled_rule_ids:
            rule_id = validate_rule_id(value)
            if rule_id in seen_rules:
                raise ValueError(f"duplicate rule ID in disabled_rule_ids: {rule_id}")
            seen_rules.add(rule_id)
            cluster = self.registry.cluster_for(rule_id)
            if cluster in effective_clusters:
                effective_disabled_rules.append(rule_id)
        object.__setattr__(self, "enabled_clusters", effective_clusters)
        object.__setattr__(self, "disabled_clusters", disabled_clusters)
        object.__setattr__(self, "disabled_rule_ids", tuple(sorted(effective_disabled_rules)))

    @classmethod
    def from_manifest_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        registry: RuleRegistrySnapshot = DEFAULT_RULE_REGISTRY,
    ) -> "NL2SQLAblationConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("ablation config manifest must be an object")
        allowed = {
            "profile_name",
            "enabled_clusters",
            "disabled_clusters",
            "disabled_rule_ids",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(
                f"ablation config manifest has unknown fields: {sorted(unknown)!r}"
            )
        if "profile_name" not in payload:
            raise ValueError("ablation config manifest is missing profile_name")
        return cls(
            profile_name=payload["profile_name"],
            enabled_clusters=(
                tuple(payload["enabled_clusters"])
                if "enabled_clusters" in payload
                else None
            ),
            disabled_clusters=tuple(payload.get("disabled_clusters", ())),
            disabled_rule_ids=tuple(payload.get("disabled_rule_ids", ())),
            registry=registry,
        )

    @classmethod
    def baseline(
        cls,
        *,
        profile_name: str = "baseline",
        registry: RuleRegistrySnapshot = DEFAULT_RULE_REGISTRY,
    ) -> "NL2SQLAblationConfig":
        return cls(profile_name=profile_name, registry=registry)

    @classmethod
    def only(
        cls,
        *cluster_ids: str,
        profile_name: str = "selected_clusters_only",
        disabled_rule_ids: Iterable[str] = (),
        registry: RuleRegistrySnapshot = DEFAULT_RULE_REGISTRY,
    ) -> "NL2SQLAblationConfig":
        return cls(
            profile_name=profile_name,
            enabled_clusters=tuple(cluster_ids),
            disabled_rule_ids=tuple(disabled_rule_ids),
            registry=registry,
        )

    @classmethod
    def without(
        cls,
        *cluster_ids: str,
        profile_name: str = "baseline_minus_cluster",
        disabled_rule_ids: Iterable[str] = (),
        registry: RuleRegistrySnapshot = DEFAULT_RULE_REGISTRY,
    ) -> "NL2SQLAblationConfig":
        return cls(
            profile_name=profile_name,
            disabled_clusters=tuple(cluster_ids),
            disabled_rule_ids=tuple(disabled_rule_ids),
            registry=registry,
        )

    def resolve(self, cluster_id: str, rule_id: str | None = None) -> "RuleResolution":
        validate_rule_cluster_id(cluster_id)
        if rule_id is not None:
            self.registry.validate_membership(rule_id, cluster_id)
        if cluster_id not in self.enabled_clusters:
            return RuleResolution(
                cluster_id,
                rule_id,
                False,
                RuleActivationReason.CLUSTER_DISABLED,
            )
        if rule_id is not None and rule_id in self.disabled_rule_ids:
            return RuleResolution(
                cluster_id,
                rule_id,
                False,
                RuleActivationReason.RULE_DISABLED,
            )
        return RuleResolution(
            cluster_id,
            rule_id,
            True,
            RuleActivationReason.ENABLED,
        )

    def is_enabled(self, cluster_id: str, rule_id: str | None = None) -> bool:
        return self.resolve(cluster_id, rule_id).enabled

    def policy_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_hash": self.registry.registry_hash,
            "enabled_clusters": list(self.enabled_clusters),
            "disabled_clusters": list(self.disabled_clusters),
            "disabled_rule_ids": list(self.disabled_rule_ids),
        }

    def profile_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_name": self.profile_name,
            "policy_hash": self.policy_hash,
        }

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "enabled_clusters": list(self.enabled_clusters),
            "disabled_clusters": list(self.disabled_clusters),
            "disabled_rule_ids": list(self.disabled_rule_ids),
        }

    def canonical_payload(self) -> dict[str, Any]:
        """Canonical effective-policy payload retained for hash consumers."""

        return self.policy_payload()

    @property
    def policy_hash(self) -> str:
        """Hash only effective behavior and its registry snapshot."""

        return canonical_sha256(self.policy_payload())

    @property
    def behavior_hash(self) -> str:
        return self.policy_hash

    @property
    def profile_hash(self) -> str:
        """Hash the human profile identity bound to the effective policy."""

        return canonical_sha256(self.profile_payload())

    @property
    def canonical_hash(self) -> str:
        """Canonical hash of effective behavior; profile identity is separate."""

        return self.policy_hash


class RuleActivationReason(str, Enum):
    ENABLED = "enabled"
    CLUSTER_DISABLED = "cluster_disabled"
    RULE_DISABLED = "rule_disabled"


@dataclass(frozen=True)
class RuleResolution:
    cluster_id: str
    rule_id: str | None
    enabled: bool
    reason: RuleActivationReason

    def __post_init__(self) -> None:
        validate_rule_cluster_id(self.cluster_id)
        if self.rule_id is not None:
            validate_rule_id(self.rule_id)
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        if not isinstance(self.reason, RuleActivationReason):
            raise TypeError("reason must be a RuleActivationReason")
        if self.enabled != (self.reason is RuleActivationReason.ENABLED):
            raise ValueError("enabled and reason disagree")

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "rule_id": self.rule_id,
            "enabled": self.enabled,
            "reason": self.reason.value,
        }


class RuleDecisionKind(str, Enum):
    SQL = "sql"
    NOT_APPLICABLE = "not_applicable"
    BLOCKED = "blocked"
    DISABLED = "disabled"
    REJECTED = "rejected"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class RuleDecision:
    rule_id: str
    cluster_id: str
    decision: RuleDecisionKind
    reason_codes: tuple[str, ...] = ()
    registry: RuleRegistrySnapshot = field(default=DEFAULT_RULE_REGISTRY, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.registry, RuleRegistrySnapshot):
            raise TypeError("registry must be a RuleRegistrySnapshot")
        self.registry.validate_membership(self.rule_id, self.cluster_id)
        if not isinstance(self.decision, RuleDecisionKind):
            raise TypeError("decision must be a RuleDecisionKind")
        object.__setattr__(self, "reason_codes", _normalise_reason_codes(self.reason_codes))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        registry: RuleRegistrySnapshot = DEFAULT_RULE_REGISTRY,
    ) -> "RuleDecision":
        if not isinstance(payload, Mapping):
            raise TypeError("rule decision must be an object")
        allowed = {
            "rule_id",
            "cluster_id",
            "rule_cluster",
            "decision",
            "reason_codes",
            "registry_hash",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"rule decision has unknown fields: {sorted(unknown)!r}")
        supplied_registry_hash = payload.get("registry_hash")
        if supplied_registry_hash is not None:
            validate_sha256(supplied_registry_hash, field_name="registry_hash")
            if supplied_registry_hash != registry.registry_hash:
                raise ValueError("rule decision registry_hash does not match supplied registry")
        rule_id = _alias_value(payload, ("rule_id",), field_name="rule_id", required=True)
        cluster_id = _alias_value(
            payload,
            ("cluster_id", "rule_cluster"),
            field_name="cluster_id",
            required=True,
        )
        raw_decision = _alias_value(
            payload,
            ("decision",),
            field_name="decision",
            required=True,
        )
        try:
            decision = RuleDecisionKind(raw_decision)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown rule decision: {raw_decision!r}") from exc
        reasons = payload.get("reason_codes", ())
        if isinstance(reasons, list):
            reasons = tuple(reasons)
        return cls(rule_id, cluster_id, decision, reasons, registry)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "cluster_id": self.cluster_id,
            "decision": self.decision.value,
            "reason_codes": list(self.reason_codes),
            "registry_hash": self.registry.registry_hash,
        }


@dataclass(frozen=True)
class RouteProvenance:
    """Canonical additive provenance with a round-trippable versioned schema."""

    route: str
    policy_hash: str
    profile_hash: str
    source_hash: str
    decisions: tuple[RuleDecision, ...] = ()
    rule_id: str | None = None
    cluster_id: str | None = None
    reason_codes: tuple[str, ...] = ()
    registry: RuleRegistrySnapshot = field(default=DEFAULT_RULE_REGISTRY, repr=False)

    def __post_init__(self) -> None:
        validate_stable_id(self.route, field_name="route")
        validate_sha256(self.policy_hash, field_name="policy_hash")
        validate_sha256(self.profile_hash, field_name="profile_hash")
        validate_sha256(self.source_hash, field_name="source_hash")
        if not isinstance(self.registry, RuleRegistrySnapshot):
            raise TypeError("registry must be a RuleRegistrySnapshot")
        if not isinstance(self.decisions, tuple):
            object.__setattr__(self, "decisions", tuple(self.decisions))
        for decision in self.decisions:
            if not isinstance(decision, RuleDecision):
                raise TypeError("decisions must contain RuleDecision values")
            if decision.registry.registry_hash != self.registry.registry_hash:
                raise ValueError("decision registry does not match route provenance registry")
        if (self.rule_id is None) != (self.cluster_id is None):
            raise ValueError("rule_id and cluster_id must be supplied together")
        if self.rule_id is not None:
            self.registry.validate_membership(self.rule_id, self.cluster_id or "")
            matching = [
                decision
                for decision in self.decisions
                if decision.rule_id == self.rule_id
                and decision.cluster_id == self.cluster_id
                and decision.decision is RuleDecisionKind.SQL
            ]
            if not matching:
                raise ValueError("selected rule requires an SQL decision in the trace")
        elif any(
            decision.decision is RuleDecisionKind.SQL for decision in self.decisions
        ):
            raise ValueError("an SQL decision requires selected rule provenance")
        object.__setattr__(self, "reason_codes", _normalise_reason_codes(self.reason_codes))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        registry: RuleRegistrySnapshot = DEFAULT_RULE_REGISTRY,
    ) -> "RouteProvenance":
        if not isinstance(payload, Mapping):
            raise TypeError("route provenance must be an object")
        allowed = {
            "schema_version",
            "route",
            "registry_hash",
            "policy_hash",
            "ablation_config_hash",
            "profile_hash",
            "source_hash",
            "rule_id",
            "selected_rule_id",
            "cluster_id",
            "rule_cluster",
            "selected_rule_cluster",
            "reason_codes",
            "decisions",
            "rule_decisions",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"route provenance has unknown fields: {sorted(unknown)!r}")
        schema_version = payload.get("schema_version", PROVENANCE_SCHEMA_VERSION)
        if schema_version != PROVENANCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported provenance schema_version: {schema_version!r}")
        supplied_registry_hash = payload.get("registry_hash")
        if supplied_registry_hash is not None:
            validate_sha256(supplied_registry_hash, field_name="registry_hash")
            if supplied_registry_hash != registry.registry_hash:
                raise ValueError("provenance registry_hash does not match supplied registry")
        rule_id = _alias_value(
            payload,
            ("rule_id", "selected_rule_id"),
            field_name="rule_id",
        )
        cluster_id = _alias_value(
            payload,
            ("cluster_id", "rule_cluster", "selected_rule_cluster"),
            field_name="cluster_id",
        )
        policy_hash = _alias_value(
            payload,
            ("policy_hash", "ablation_config_hash"),
            field_name="policy_hash",
            required=True,
        )
        raw_decisions = _alias_value(
            payload,
            ("decisions", "rule_decisions"),
            field_name="decisions",
        )
        if raw_decisions is None:
            raw_decisions = []
        if not isinstance(raw_decisions, Sequence) or isinstance(raw_decisions, (str, bytes)):
            raise TypeError("decisions must be a sequence")
        decisions = tuple(
            RuleDecision.from_dict(item, registry=registry) for item in raw_decisions
        )
        reasons = payload.get("reason_codes", ())
        if isinstance(reasons, list):
            reasons = tuple(reasons)
        return cls(
            route=payload.get("route"),
            policy_hash=policy_hash,
            profile_hash=payload.get("profile_hash"),
            source_hash=payload.get("source_hash"),
            decisions=decisions,
            rule_id=rule_id,
            cluster_id=cluster_id,
            reason_codes=reasons,
            registry=registry,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "route": self.route,
            "registry_hash": self.registry.registry_hash,
            "policy_hash": self.policy_hash,
            "profile_hash": self.profile_hash,
            "source_hash": self.source_hash,
            "rule_id": self.rule_id,
            "cluster_id": self.cluster_id,
            "reason_codes": list(self.reason_codes),
            "decisions": [decision.as_dict() for decision in self.decisions],
        }


@dataclass(frozen=True)
class StrictCounts:
    n: int
    n_correct: int

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n <= 0:
            raise ValueError("n must be a positive integer")
        if (
            isinstance(self.n_correct, bool)
            or not isinstance(self.n_correct, int)
            or not 0 <= self.n_correct <= self.n
        ):
            raise ValueError("n_correct must be an integer in [0, n]")

    @property
    def n_incorrect(self) -> int:
        return self.n - self.n_correct

    @property
    def accuracy_fraction(self) -> Fraction:
        return Fraction(self.n_correct, self.n)

    @property
    def accuracy(self) -> float:
        return float(self.accuracy_fraction)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
            "accuracy": _report_float(self.accuracy_fraction),
        }


@dataclass(frozen=True)
class PairedStrictComparison:
    baseline: StrictCounts
    variant: StrictCounts
    n_improved: int
    n_regressed: int
    n_unchanged_correct: int
    n_unchanged_incorrect: int

    def __post_init__(self) -> None:
        if self.baseline.n != self.variant.n:
            raise ValueError("paired comparisons require equal denominators")
        transitions = (
            self.n_improved,
            self.n_regressed,
            self.n_unchanged_correct,
            self.n_unchanged_incorrect,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in transitions
        ):
            raise ValueError("transition counts must be nonnegative integers")
        if sum(transitions) != self.baseline.n:
            raise ValueError("transition counts must sum to n")
        if self.variant.n_correct != (
            self.baseline.n_correct + self.n_improved - self.n_regressed
        ):
            raise ValueError("transition counts disagree with strict counts")

    @property
    def delta_n_correct(self) -> int:
        return self.variant.n_correct - self.baseline.n_correct

    @property
    def delta_fraction(self) -> Fraction:
        return self.variant.accuracy_fraction - self.baseline.accuracy_fraction

    @property
    def delta(self) -> float:
        return float(self.delta_fraction)

    @property
    def loss_fraction(self) -> Fraction:
        return max(Fraction(0), -self.delta_fraction)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.baseline.n,
            "baseline": self.baseline.as_dict(),
            "variant": self.variant.as_dict(),
            "delta_n_correct": self.delta_n_correct,
            "delta_accuracy": _report_float(self.delta_fraction),
            "transitions": {
                "improved": self.n_improved,
                "regressed": self.n_regressed,
                "unchanged_correct": self.n_unchanged_correct,
                "unchanged_incorrect": self.n_unchanged_incorrect,
            },
        }


def _validate_correctness_map(
    values: Mapping[str, bool],
    *,
    label: str,
) -> dict[str, bool]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{label} correctness must be a nonempty mapping")
    result: dict[str, bool] = {}
    for item_id, correct in values.items():
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{label} contains an invalid item ID")
        if not isinstance(correct, bool):
            raise TypeError(f"{label}[{item_id!r}] must be a bool")
        result[item_id] = correct
    return result


def compare_paired_correctness(
    baseline: Mapping[str, bool],
    variant: Mapping[str, bool],
) -> PairedStrictComparison:
    baseline_map = _validate_correctness_map(baseline, label="baseline")
    variant_map = _validate_correctness_map(variant, label="variant")
    if baseline_map.keys() != variant_map.keys():
        missing = sorted(baseline_map.keys() - variant_map.keys())
        extra = sorted(variant_map.keys() - baseline_map.keys())
        raise ValueError(f"unpaired result IDs: missing={missing!r}, extra={extra!r}")
    improved = regressed = unchanged_correct = unchanged_incorrect = 0
    for item_id in sorted(baseline_map):
        before = baseline_map[item_id]
        after = variant_map[item_id]
        if not before and after:
            improved += 1
        elif before and not after:
            regressed += 1
        elif before:
            unchanged_correct += 1
        else:
            unchanged_incorrect += 1
    n = len(baseline_map)
    return PairedStrictComparison(
        baseline=StrictCounts(n, sum(baseline_map.values())),
        variant=StrictCounts(n, sum(variant_map.values())),
        n_improved=improved,
        n_regressed=regressed,
        n_unchanged_correct=unchanged_correct,
        n_unchanged_incorrect=unchanged_incorrect,
    )


def _report_float(value: Fraction | float) -> float:
    result = round(float(value), 12)
    return 0.0 if result == 0 else result


@dataclass(frozen=True)
class FamilyStabilitySummary:
    family_id: str
    runs: tuple[tuple[str, StrictCounts], ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.family_id, field_name="family_id")
        if not self.runs:
            raise ValueError("family stability requires at least one run")
        run_ids = [run_id for run_id, _ in self.runs]
        if run_ids != sorted(run_ids) or len(set(run_ids)) != len(run_ids):
            raise ValueError("stability runs must be unique and sorted")
        if any(not isinstance(counts, StrictCounts) for _, counts in self.runs):
            raise TypeError("stability run values must be StrictCounts")
        if len({counts.n for _, counts in self.runs}) != 1:
            raise ValueError("family stability runs require equal denominators")

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def accuracy_range_fraction(self) -> Fraction:
        values = [counts.accuracy_fraction for _, counts in self.runs]
        return max(values) - min(values)

    @property
    def accuracy_range(self) -> float:
        return float(self.accuracy_range_fraction)

    @property
    def accuracy_stddev(self) -> float:
        values = [counts.accuracy for _, counts in self.runs]
        return statistics.pstdev(values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_runs": self.n_runs,
            "n_items_per_run": self.runs[0][1].n,
            "accuracy_range": _report_float(self.accuracy_range_fraction),
            "accuracy_stddev": _report_float(self.accuracy_stddev),
            "by_run": {run_id: counts.as_dict() for run_id, counts in self.runs},
        }


@dataclass(frozen=True)
class StabilitySummary:
    families: tuple[FamilyStabilitySummary, ...]

    def __post_init__(self) -> None:
        if not self.families:
            raise ValueError("stability requires at least one family")
        ids = [family.family_id for family in self.families]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("stability families must be unique and sorted")

    @property
    def n_families(self) -> int:
        return len(self.families)

    @property
    def min_runs(self) -> int:
        return min(family.n_runs for family in self.families)

    @property
    def max_accuracy_range(self) -> float:
        return max(family.accuracy_range for family in self.families)

    @property
    def max_accuracy_stddev(self) -> float:
        return max(family.accuracy_stddev for family in self.families)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_families": self.n_families,
            "min_runs": self.min_runs,
            "max_accuracy_range": _report_float(self.max_accuracy_range),
            "max_accuracy_stddev": _report_float(self.max_accuracy_stddev),
            "by_family": {
                family.family_id: family.as_dict() for family in self.families
            },
        }


@dataclass(frozen=True)
class StabilityComparison:
    baseline: StabilitySummary
    variant: StabilitySummary

    def __post_init__(self) -> None:
        baseline = {family.family_id: family for family in self.baseline.families}
        variant = {family.family_id: family for family in self.variant.families}
        if baseline.keys() != variant.keys():
            raise ValueError("baseline/variant stability families differ")
        for family_id in baseline:
            before = baseline[family_id]
            after = variant[family_id]
            if [run for run, _ in before.runs] != [run for run, _ in after.runs]:
                raise ValueError(f"stability run IDs differ for family {family_id}")
            if [counts.n for _, counts in before.runs] != [
                counts.n for _, counts in after.runs
            ]:
                raise ValueError(f"stability populations differ for family {family_id}")

    @property
    def range_not_worse(self) -> bool:
        before = {family.family_id: family for family in self.baseline.families}
        return all(
            family.accuracy_range_fraction
            <= before[family.family_id].accuracy_range_fraction
            for family in self.variant.families
        )

    @property
    def stddev_not_worse(self) -> bool:
        before = {family.family_id: family for family in self.baseline.families}
        return all(
            family.accuracy_stddev
            <= before[family.family_id].accuracy_stddev + 1e-15
            for family in self.variant.families
        )

    def as_dict(self) -> dict[str, Any]:
        before = {family.family_id: family for family in self.baseline.families}
        return {
            "range_not_worse": self.range_not_worse,
            "stddev_not_worse": self.stddev_not_worse,
            "baseline": self.baseline.as_dict(),
            "variant": self.variant.as_dict(),
            "by_family_delta": {
                family.family_id: {
                    "range_delta": _report_float(
                        family.accuracy_range
                        - before[family.family_id].accuracy_range
                    ),
                    "stddev_delta": _report_float(
                        family.accuracy_stddev
                        - before[family.family_id].accuracy_stddev
                    ),
                }
                for family in self.variant.families
            },
        }


def measure_family_stability(
    families: Mapping[str, Mapping[str, Mapping[str, bool]]],
) -> StabilitySummary:
    if not isinstance(families, Mapping) or not families:
        raise ValueError("stability families must be a nonempty mapping")
    summaries: list[FamilyStabilitySummary] = []
    for family_id in sorted(families):
        validate_stable_id(family_id, field_name="family_id")
        runs = families[family_id]
        if not isinstance(runs, Mapping) or not runs:
            raise ValueError(f"family {family_id} has no runs")
        canonical_ids: set[str] | None = None
        run_counts: list[tuple[str, StrictCounts]] = []
        for run_id in sorted(runs):
            validate_stable_id(run_id, field_name="run_id")
            values = _validate_correctness_map(
                runs[run_id],
                label=f"family {family_id} run {run_id}",
            )
            item_ids = set(values)
            if canonical_ids is None:
                canonical_ids = item_ids
            elif item_ids != canonical_ids:
                raise ValueError(
                    f"family {family_id} runs do not contain identical item IDs"
                )
            run_counts.append(
                (run_id, StrictCounts(len(values), sum(values.values())))
            )
        summaries.append(FamilyStabilitySummary(family_id, tuple(run_counts)))
    return StabilitySummary(tuple(summaries))


def compare_stability_populations(
    baseline: Mapping[str, Mapping[str, Mapping[str, bool]]],
    variant: Mapping[str, Mapping[str, Mapping[str, bool]]],
) -> StabilityComparison:
    if baseline.keys() != variant.keys():
        raise ValueError("baseline/variant stability family populations differ")
    for family_id in baseline:
        if baseline[family_id].keys() != variant[family_id].keys():
            raise ValueError(f"stability run populations differ for family {family_id}")
        for run_id in baseline[family_id]:
            if baseline[family_id][run_id].keys() != variant[family_id][run_id].keys():
                raise ValueError(
                    f"stability item population differs for family {family_id}, run {run_id}"
                )
    return StabilityComparison(
        measure_family_stability(baseline),
        measure_family_stability(variant),
    )


def measure_stability(
    runs: Mapping[str, Mapping[str, bool]],
) -> StabilitySummary:
    """Compatibility helper for one explicitly named aggregate family."""

    return measure_family_stability({"aggregate": runs})


@dataclass(frozen=True)
class AblationAcceptanceCriteria:
    """Single protocol policy shared by library and CLI."""

    min_pooled_delta: float = 0.0
    max_shadow_loss: float = 0.01
    min_shadow_groups: int = 3
    min_alternate_dbs: int = 3
    max_stability_range: float = 0.02
    max_stability_stddev: float = 0.02
    min_stability_runs: int = 2

    def __post_init__(self) -> None:
        for name in (
            "min_pooled_delta",
            "max_shadow_loss",
            "max_stability_range",
            "max_stability_stddev",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "min_shadow_groups",
            "min_alternate_dbs",
            "min_stability_runs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_pooled_delta": float(self.min_pooled_delta),
            "pooled_delta_must_be_strictly_positive": True,
            "max_shadow_loss": float(self.max_shadow_loss),
            "min_shadow_groups": self.min_shadow_groups,
            "alternate_db_no_loss": True,
            "min_alternate_dbs": self.min_alternate_dbs,
            "max_stability_range": float(self.max_stability_range),
            "max_stability_stddev": float(self.max_stability_stddev),
            "stability_range_must_not_worsen": True,
            "stability_stddev_must_not_worsen": True,
            "min_stability_runs": self.min_stability_runs,
        }


DEFAULT_ABLATION_ACCEPTANCE_CRITERIA = AblationAcceptanceCriteria()


@dataclass(frozen=True)
class AcceptanceGate:
    name: str
    passed: bool
    measured: Any
    requirement: str
    evidence_count: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "measured": self.measured,
            "requirement": self.requirement,
            "evidence_count": self.evidence_count,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AblationAcceptanceDecision:
    criteria: AblationAcceptanceCriteria
    pooled: PairedStrictComparison
    shadow: tuple[tuple[str, PairedStrictComparison], ...]
    alternate_db: tuple[tuple[str, PairedStrictComparison], ...]
    stability: StabilityComparison | None
    gates: tuple[AcceptanceGate, ...]

    @property
    def accepted(self) -> bool:
        return all(gate.passed for gate in self.gates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "criteria": self.criteria.as_dict(),
            "gates": {gate.name: gate.as_dict() for gate in self.gates},
            "measurements": {
                "pooled": self.pooled.as_dict(),
                "shadow_by_slice_family": {
                    name: comparison.as_dict() for name, comparison in self.shadow
                },
                "alternate_db": {
                    name: comparison.as_dict()
                    for name, comparison in self.alternate_db
                },
                "stability": self.stability.as_dict() if self.stability else None,
            },
        }


def assess_ablation_acceptance(
    *,
    pooled: PairedStrictComparison,
    shadow: Mapping[str, PairedStrictComparison],
    alternate_db: Mapping[str, PairedStrictComparison],
    stability: StabilityComparison | None,
    criteria: AblationAcceptanceCriteria | None = None,
) -> AblationAcceptanceDecision:
    if not isinstance(pooled, PairedStrictComparison):
        raise TypeError("pooled must be a PairedStrictComparison")
    criteria = criteria or DEFAULT_ABLATION_ACCEPTANCE_CRITERIA
    if not isinstance(criteria, AblationAcceptanceCriteria):
        raise TypeError("criteria must be AblationAcceptanceCriteria")
    shadow_items = tuple(sorted(shadow.items()))
    alternate_items = tuple(sorted(alternate_db.items()))
    if any(not isinstance(value, PairedStrictComparison) for _, value in shadow_items):
        raise TypeError("shadow values must be PairedStrictComparison instances")
    if any(
        not isinstance(value, PairedStrictComparison) for _, value in alternate_items
    ):
        raise TypeError("alternate_db values must be PairedStrictComparison instances")
    if stability is not None and not isinstance(stability, StabilityComparison):
        raise TypeError("stability must be StabilityComparison or None")

    min_delta = Fraction(str(float(criteria.min_pooled_delta)))
    pooled_passed = pooled.delta_fraction > 0 and pooled.delta_fraction >= min_delta
    pooled_gate = AcceptanceGate(
        "pooled_positive_delta",
        pooled_passed,
        _report_float(pooled.delta_fraction),
        f"strict delta > 0 and >= {float(criteria.min_pooled_delta):.12g}",
        pooled.baseline.n,
        "passed" if pooled_passed else "pooled strict accuracy did not improve",
    )

    worst_shadow_loss = max(
        (comparison.loss_fraction for _, comparison in shadow_items),
        default=None,
    )
    shadow_passed = (
        len(shadow_items) >= criteria.min_shadow_groups
        and worst_shadow_loss is not None
        and worst_shadow_loss <= Fraction(str(float(criteria.max_shadow_loss)))
    )
    shadow_gate = AcceptanceGate(
        "shadow_worst_family_loss",
        shadow_passed,
        _report_float(worst_shadow_loss) if worst_shadow_loss is not None else None,
        (
            f"at least {criteria.min_shadow_groups} fixed slice+family groups; "
            f"worst loss <= {float(criteria.max_shadow_loss):.12g}"
        ),
        len(shadow_items),
        (
            "passed"
            if shadow_passed
            else "missing fixed shadow families or worst-family loss exceeded the limit"
        ),
    )

    alternate_max_loss = max(
        (comparison.loss_fraction for _, comparison in alternate_items),
        default=None,
    )
    alternate_passed = (
        len(alternate_items) >= criteria.min_alternate_dbs
        and all(comparison.delta_n_correct >= 0 for _, comparison in alternate_items)
    )
    alternate_gate = AcceptanceGate(
        "alternate_db_no_loss",
        alternate_passed,
        _report_float(alternate_max_loss) if alternate_max_loss is not None else None,
        (
            f"at least {criteria.min_alternate_dbs} manifest-bound alternate DBs; "
            "variant correct count >= baseline for every DB"
        ),
        len(alternate_items),
        (
            "passed"
            if alternate_passed
            else "missing alternate-DB evidence or at least one fixed DB regressed"
        ),
    )

    stability_passed = (
        stability is not None
        and stability.baseline.min_runs >= criteria.min_stability_runs
        and stability.variant.min_runs >= criteria.min_stability_runs
        and stability.variant.max_accuracy_range <= criteria.max_stability_range + 1e-15
        and stability.variant.max_accuracy_stddev
        <= criteria.max_stability_stddev + 1e-15
        and stability.range_not_worse
        and stability.stddev_not_worse
    )
    stability_measured = None
    evidence_count = 0
    if stability is not None:
        evidence_count = stability.variant.n_families
        stability_measured = {
            "baseline_max_range": _report_float(
                stability.baseline.max_accuracy_range
            ),
            "variant_max_range": _report_float(stability.variant.max_accuracy_range),
            "baseline_max_stddev": _report_float(
                stability.baseline.max_accuracy_stddev
            ),
            "variant_max_stddev": _report_float(
                stability.variant.max_accuracy_stddev
            ),
            "range_not_worse": stability.range_not_worse,
            "stddev_not_worse": stability.stddev_not_worse,
        }
    stability_gate = AcceptanceGate(
        "stability_no_worsening",
        stability_passed,
        stability_measured,
        (
            f"at least {criteria.min_stability_runs} identical-population runs per family; "
            f"variant range <= {criteria.max_stability_range:.12g}, variant stddev <= "
            f"{criteria.max_stability_stddev:.12g}, and neither metric worsens"
        ),
        evidence_count,
        (
            "passed"
            if stability_passed
            else "missing stability evidence, absolute dispersion exceeded, or dispersion worsened"
        ),
    )

    return AblationAcceptanceDecision(
        criteria,
        pooled,
        shadow_items,
        alternate_items,
        stability,
        (pooled_gate, shadow_gate, alternate_gate, stability_gate),
    )
