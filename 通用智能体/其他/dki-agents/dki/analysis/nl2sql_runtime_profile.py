"""Immutable, hash-addressed runtime configuration for NL2SQL pipelines."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


NL2SQL_RUNTIME_PROFILE_VERSION = "nl2sql-runtime-profile-v1"
NL2SQL_RUNTIME_MODES = frozenset(
    {"legacy_sql", "query_plan", "query_plan_v2", "query_plan_v3"}
)
NL2SQL_RULE_POLICIES = frozenset({"rule_first", "plan_first", "model_first"})
NL2SQL_MODE_RULE_POLICIES = MappingProxyType(
    {
        "legacy_sql": frozenset({"rule_first", "model_first"}),
        "query_plan": frozenset({"rule_first", "plan_first"}),
        "query_plan_v2": frozenset({"rule_first", "plan_first"}),
        "query_plan_v3": frozenset({"rule_first", "plan_first"}),
    }
)
NL2SQL_REASONING_MODES = frozenset({"disabled", "adaptive"})
MIN_TOKEN_BUDGET = 1
MAX_TOKEN_BUDGET = 131_072


def _require_nonempty_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: str, label: str) -> str:
    _require_nonempty_text(value, label)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class NL2SQLRuntimeProfile:
    """Complete immutable identity for one supported NL2SQL runtime policy."""

    runtime_mode: str
    rule_policy: str
    llm_samples: int
    rounds: int
    reasoning_mode: str
    token_budget: int
    fail_closed: bool
    selector_version: str
    selector_hash: str
    scorer_version: str
    scorer_hash: str
    counterfactual_version: str
    counterfactual_hash: str
    executor_version: str
    executor_hash: str
    profile_version: str = NL2SQL_RUNTIME_PROFILE_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_text(self.profile_version, "profile_version")
        if self.runtime_mode not in NL2SQL_RUNTIME_MODES:
            raise ValueError(
                "runtime_mode must be one of "
                f"{sorted(NL2SQL_RUNTIME_MODES)}"
            )
        if self.rule_policy not in NL2SQL_RULE_POLICIES:
            raise ValueError(
                "rule_policy must be one of "
                f"{sorted(NL2SQL_RULE_POLICIES)}"
            )
        compatible = NL2SQL_MODE_RULE_POLICIES[self.runtime_mode]
        if self.rule_policy not in compatible:
            raise ValueError(
                f"rule_policy {self.rule_policy!r} is incompatible with "
                f"runtime_mode {self.runtime_mode!r}; expected one of "
                f"{sorted(compatible)}"
            )
        _require_positive_integer(self.llm_samples, "llm_samples")
        _require_positive_integer(self.rounds, "rounds")
        if self.reasoning_mode not in NL2SQL_REASONING_MODES:
            raise ValueError(
                "reasoning_mode must be one of "
                f"{sorted(NL2SQL_REASONING_MODES)}"
            )
        if (
            isinstance(self.token_budget, bool)
            or not isinstance(self.token_budget, int)
            or not MIN_TOKEN_BUDGET <= self.token_budget <= MAX_TOKEN_BUDGET
        ):
            raise ValueError(
                "token_budget must be an integer between "
                f"{MIN_TOKEN_BUDGET} and {MAX_TOKEN_BUDGET}"
            )
        if type(self.fail_closed) is not bool:
            raise ValueError("fail_closed must be a boolean")

        for component in ("selector", "scorer", "counterfactual", "executor"):
            _require_nonempty_text(
                getattr(self, f"{component}_version"),
                f"{component}_version",
            )
            _require_sha256(
                getattr(self, f"{component}_hash"),
                f"{component}_hash",
            )

    def canonical_payload(self) -> Mapping[str, object]:
        """Return an immutable, deterministically serializable profile payload."""
        return MappingProxyType(asdict(self))

    @property
    def canonical_sha256(self) -> str:
        encoded = json.dumps(
            dict(self.canonical_payload()),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def canonical_hash(self) -> str:
        """Compatibility alias for the profile's canonical SHA-256."""
        return self.canonical_sha256

    def nl2sql_constructor_kwargs(self) -> Mapping[str, object]:
        """Map profile vocabulary to the modern ``NL2SQL`` constructor."""
        return MappingProxyType(
            {
                "llm_samples": self.llm_samples,
                "llm_rounds": self.rounds,
                "llm_strategy": self.runtime_mode,
                "agent_rule_policy": self.rule_policy,
                "llm_reasoning_mode": self.reasoning_mode,
                "llm_completion_tokens": self.token_budget,
                "llm_fail_closed": self.fail_closed,
            }
        )


def construct_nl2sql(
    profile: NL2SQLRuntimeProfile,
    db_path: Path,
    **constructor_options: Any,
) -> Any:
    """Lazily import and construct ``NL2SQL`` from one validated profile.

    Extra options support constructor settings outside this profile version, but
    cannot replace fields whose identity is covered by the canonical hash.
    """
    if not isinstance(profile, NL2SQLRuntimeProfile):
        raise TypeError("profile must be an NL2SQLRuntimeProfile")
    profile_options = dict(profile.nl2sql_constructor_kwargs())
    conflicts = sorted(profile_options.keys() & constructor_options.keys())
    if conflicts:
        raise ValueError(
            "constructor_options cannot override profile-controlled fields: "
            + ", ".join(conflicts)
        )

    from dki.analysis.nl2sql import NL2SQL

    return NL2SQL(
        db_path=Path(db_path),
        **profile_options,
        **constructor_options,
    )


build_nl2sql_from_profile = construct_nl2sql


__all__ = [
    "NL2SQL_RUNTIME_PROFILE_VERSION",
    "NL2SQL_RUNTIME_MODES",
    "NL2SQL_RULE_POLICIES",
    "NL2SQL_MODE_RULE_POLICIES",
    "NL2SQL_REASONING_MODES",
    "MIN_TOKEN_BUDGET",
    "MAX_TOKEN_BUDGET",
    "NL2SQLRuntimeProfile",
    "construct_nl2sql",
    "build_nl2sql_from_profile",
]
