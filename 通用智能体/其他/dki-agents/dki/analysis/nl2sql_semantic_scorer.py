"""Gold-free semantic scoring for QueryPlan v3 candidates.

The scorer turns the existing contract, qualifier-obligation, schema-graph and
QueryPlan-v3 validators into deterministic selector evidence.  It never executes
benchmark gold SQL and never treats result-bucket size as correctness evidence.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dki.analysis.nl2sql_capabilities import classify_nl2sql_capabilities
from dki.analysis.nl2sql_candidate_selector import (
    CandidateProvenance,
    CandidateRecord,
    ExecutionFingerprint,
    IntentPlanHashes,
    StaticValidation,
)
from dki.analysis.nl2sql_contract import (
    ContractStatus,
    extract_semantic_contract,
    validate_candidate,
)
from dki.analysis.nl2sql_query_plan_v3 import (
    QueryPlanV3Error,
    canonical_plan_sha256_v3,
    parse_query_plan_v3,
    validate_query_plan_v3,
)
from dki.analysis.nl2sql_semantic_obligations import (
    ObligationDisposition,
    extract_semantic_obligations,
    validate_candidate_obligations,
)


SCORER_VERSION = "nl2sql-semantic-scorer-v1"


@dataclass(frozen=True, slots=True)
class SemanticScore:
    score: float
    hard_failure: bool
    reason_codes: tuple[str, ...]
    capability_coverage: tuple[str, ...]
    contract_complete: bool
    contract_valid: bool
    obligations_renderable: bool
    obligations_valid: bool
    plan_valid: bool


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_codes(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def score_candidate_semantics(
    question: str,
    plan: Mapping[str, Any],
    sql: str,
    *,
    domains: Mapping[str, Sequence[str]] | None = None,
) -> SemanticScore:
    """Score one v3 plan/SQL pair without gold or result-bucket voting."""

    reasons: list[str] = []
    plan_valid = False
    try:
        parsed = parse_query_plan_v3(plan)
        validate_query_plan_v3(parsed, question, domains=domains)
        plan_valid = True
    except (QueryPlanV3Error, ValueError) as exc:
        reasons.append(f"query_plan_v3:{exc}")

    contract = extract_semantic_contract(question)
    contract_complete = contract.status is ContractStatus.COMPLETE
    contract_result = validate_candidate(
        contract,
        sql,
        strict=contract_complete,
    )
    contract_valid = bool(contract_result.accepted)
    if contract_complete and not contract_valid:
        reasons.extend(f"semantic_contract:{reason}" for reason in contract_result.reasons)

    obligations = extract_semantic_obligations(question, domains or {})
    obligations_renderable = (
        obligations.disposition is ObligationDisposition.RENDERABLE
    )
    obligations_valid = True
    if obligations.disposition is ObligationDisposition.TERMINAL_ABSTAIN:
        obligations_valid = False
        reasons.extend(
            f"semantic_obligation:{reason}"
            for reason in (
                obligations.unsupported_reasons
                or ("terminal semantic obligation",)
            )
        )
    elif obligations_renderable:
        obligation_result = validate_candidate_obligations(obligations, sql)
        obligations_valid = bool(obligation_result.accepted)
        if not obligations_valid:
            reasons.extend(
                f"semantic_obligation:{reason}"
                for reason in obligation_result.reasons
            )

    hard_failure = bool(
        not plan_valid
        or (contract_complete and not contract_valid)
        or not obligations_valid
    )

    capabilities = classify_nl2sql_capabilities(question, sql)
    score = 0.0
    if plan_valid:
        score += 0.55
    if contract_complete:
        score += 0.20 if contract_valid else 0.0
    else:
        score += 0.10 if contract_valid else 0.0
    if obligations_renderable:
        score += 0.20 if obligations_valid else 0.0
    elif obligations.disposition is ObligationDisposition.NOT_APPLICABLE:
        score += 0.10
    if capabilities:
        # Capability discovery is evidence that the candidate expresses the
        # question's structure, but it is deliberately a small bounded signal.
        score += min(0.05, 0.005 * len(capabilities))
    if hard_failure:
        score = min(score, 0.49)

    return SemanticScore(
        score=round(min(1.0, score), 6),
        hard_failure=hard_failure,
        reason_codes=_normalise_codes(reasons),
        capability_coverage=tuple(capabilities),
        contract_complete=contract_complete,
        contract_valid=contract_valid,
        obligations_renderable=obligations_renderable,
        obligations_valid=obligations_valid,
        plan_valid=plan_valid,
    )


def candidate_record_from_v3(
    *,
    question: str,
    plan: Mapping[str, Any],
    sql: str,
    generator: str,
    generation_index: int,
    execution_result_sha256: str,
    counterfactual_score: float,
    complexity: int,
    domains: Mapping[str, Sequence[str]] | None = None,
    judge_score: float | None = None,
) -> CandidateRecord:
    """Build selector input from one compiled v3 candidate and its evidence."""

    semantic = score_candidate_semantics(
        question,
        plan,
        sql,
        domains=domains,
    )
    try:
        parsed = parse_query_plan_v3(plan)
        plan_hash = canonical_plan_sha256_v3(parsed)
        intent_hash = _canonical_hash(parsed["intent"])
    except Exception:
        # CandidateRecord still receives stable identities so a failed candidate
        # can be retained in private diagnostics and rejected by the selector.
        plan_hash = _canonical_hash(plan)
        intent_hash = _canonical_hash(plan.get("intent", {}) if isinstance(plan, Mapping) else {})

    validation = StaticValidation(
        syntax_valid=semantic.plan_valid,
        read_only=semantic.plan_valid,
        schema_valid=semantic.plan_valid,
        grounded=not any(code.startswith("query_plan_v3:plan literal") for code in semantic.reason_codes),
        complete=not semantic.hard_failure,
        reason_codes=semantic.reason_codes,
    )
    candidate_id = hashlib.sha256(
        f"{generator}\0{generation_index}\0{plan_hash}".encode("utf-8")
    ).hexdigest()[:24]
    return CandidateRecord(
        provenance=CandidateProvenance(
            candidate_id=candidate_id,
            generator=generator,
            generation_index=generation_index,
            strategy="query_plan_v3",
        ),
        hashes=IntentPlanHashes(
            intent_sha256=intent_hash,
            plan_sha256=plan_hash,
        ),
        sql=sql,
        static_validation=validation,
        execution=ExecutionFingerprint(
            result_sha256=execution_result_sha256,
            executed=True,
            deterministic=True,
        ),
        counterfactual_score=counterfactual_score,
        semantic_score=semantic.score,
        complexity=complexity,
        judge_score=judge_score,
    )


__all__ = [
    "SCORER_VERSION",
    "SemanticScore",
    "candidate_record_from_v3",
    "score_candidate_semantics",
]
