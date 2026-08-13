"""Canonical NL2SQL runtime profiles shared by CLI, Agent tools, and evaluators.

Every production or evaluation engine must be constructed through this module so
that the CLI, the Agent ``Nl2SqlTool``, and the evaluation scripts run one and
the same hashed configuration.  Component hashes bind the profile to the actual
selector policy and to the source bytes of the scorer, counterfactual, and
complete-execution modules, so any code drift changes the profile identity.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from dki.analysis.nl2sql_candidate_selector import SELECTOR_VERSION, SelectorConfig
from dki.analysis.nl2sql_counterfactual import COUNTERFACTUAL_VERSION
from dki.analysis.nl2sql_protocol import RESULT_FINGERPRINT_VERSION
from dki.analysis.nl2sql_runtime_profile import (
    NL2SQLRuntimeProfile,
    construct_nl2sql,
)
from dki.analysis.nl2sql_semantic_scorer import SCORER_VERSION


def _module_source_sha256(module_name: str) -> str:
    import importlib

    module = importlib.import_module(module_name)
    source = Path(module.__file__)
    return hashlib.sha256(source.read_bytes()).hexdigest()


def component_identities() -> dict[str, str]:
    """Return the live component versions and hashes bound into every profile."""

    return {
        "selector_version": SELECTOR_VERSION,
        "selector_hash": SelectorConfig().canonical_sha256,
        "scorer_version": SCORER_VERSION,
        "scorer_hash": _module_source_sha256("dki.analysis.nl2sql_semantic_scorer"),
        "counterfactual_version": COUNTERFACTUAL_VERSION,
        "counterfactual_hash": _module_source_sha256(
            "dki.analysis.nl2sql_counterfactual"
        ),
        "executor_version": RESULT_FINGERPRINT_VERSION,
        "executor_hash": _module_source_sha256("dki.analysis.nl2sql_protocol"),
    }


def default_runtime_profile() -> NL2SQLRuntimeProfile:
    """Current shipped product behaviour (unchanged until gates pass)."""

    return NL2SQLRuntimeProfile(
        runtime_mode="legacy_sql",
        rule_policy="rule_first",
        llm_samples=1,
        rounds=2,
        reasoning_mode="disabled",
        token_budget=1024,
        fail_closed=False,
        **component_identities(),
    )


def candidate_query_plan_v3_profile() -> NL2SQLRuntimeProfile:
    """QueryPlan v3 multi-candidate profile under gate evaluation.

    This profile only becomes the product default after the functional,
    stability, and shadow gates in the release plan all pass.
    """

    return NL2SQLRuntimeProfile(
        runtime_mode="query_plan_v3",
        rule_policy="plan_first",
        llm_samples=1,
        rounds=2,
        reasoning_mode="adaptive",
        token_budget=8192,
        fail_closed=False,
        **component_identities(),
    )


NAMED_PROFILES = {
    "default": default_runtime_profile,
    "query_plan_v3_candidate": candidate_query_plan_v3_profile,
}


def resolve_profile(name: str) -> NL2SQLRuntimeProfile:
    if name not in NAMED_PROFILES:
        raise ValueError(
            f"unknown NL2SQL profile {name!r}; expected one of "
            f"{sorted(NAMED_PROFILES)}"
        )
    return NAMED_PROFILES[name]()


def build_engine(db_path: Path, profile_name: str = "default", **options: Any):
    """Construct the shared ``NL2SQL`` engine for one named canonical profile."""

    profile = resolve_profile(profile_name)
    return construct_nl2sql(profile, Path(db_path), **options)


__all__ = [
    "NAMED_PROFILES",
    "build_engine",
    "candidate_query_plan_v3_profile",
    "component_identities",
    "default_runtime_profile",
    "resolve_profile",
]
