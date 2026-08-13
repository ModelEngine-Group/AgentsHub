"""Deterministic, fail-closed ranking for structured NL2SQL candidates.

The selector consumes evidence produced by candidate generation and validation; it
never executes SQL or inspects benchmark gold.  Candidates with any hard
static, execution, or counterfactual failure are removed before scores, judge
evidence, or execution consensus are considered.  Execution agreement is deliberately only a small capped bonus, not
a majority-vote selection rule.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping, Sequence


SELECTOR_VERSION = "nl2sql-candidate-selector-v1"
MAX_ALLOWED_CONSENSUS_BONUS = 0.05


def _require_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_hash(value: str, label: str) -> str:
    value = _require_text(value, label)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_unit_score(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number in [0, 1]")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    return score


@dataclass(frozen=True, slots=True)
class CandidateProvenance:
    """Stable identity of one independently generated candidate."""

    candidate_id: str
    generator: str
    generation_index: int
    strategy: str = "unspecified"

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.generator, "generator")
        _require_text(self.strategy, "strategy")
        if isinstance(self.generation_index, bool) or not isinstance(
            self.generation_index, int
        ) or self.generation_index < 0:
            raise ValueError("generation_index must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class IntentPlanHashes:
    """Canonical intent and structured-plan identities supplied by generation."""

    intent_sha256: str
    plan_sha256: str

    def __post_init__(self) -> None:
        _require_hash(self.intent_sha256, "intent_sha256")
        _require_hash(self.plan_sha256, "plan_sha256")


@dataclass(frozen=True, slots=True)
class StaticValidation:
    """Fail-closed static validation outcome and machine-readable reason codes."""

    syntax_valid: bool = True
    read_only: bool = True
    schema_valid: bool = True
    grounded: bool = True
    complete: bool = True
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        flags = (
            self.syntax_valid,
            self.read_only,
            self.schema_valid,
            self.grounded,
            self.complete,
        )
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("static validation flags must be booleans")
        codes = tuple(self.reason_codes)
        for code in codes:
            _require_text(code, "static validation reason code")
        if len(set(codes)) != len(codes):
            raise ValueError("static validation reason codes must be unique")
        object.__setattr__(self, "reason_codes", codes)

    @property
    def hard_failure(self) -> bool:
        return not (
            self.syntax_valid
            and self.read_only
            and self.schema_valid
            and self.grounded
            and self.complete
        )


@dataclass(frozen=True, slots=True)
class ExecutionFingerprint:
    """Gold-free identity of a successful, bounded execution result."""

    result_sha256: str | None
    executed: bool = True
    deterministic: bool = True
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.executed) is not bool or type(self.deterministic) is not bool:
            raise ValueError("execution flags must be booleans")
        if self.result_sha256 is not None:
            _require_hash(self.result_sha256, "result_sha256")
        codes = tuple(self.reason_codes)
        for code in codes:
            _require_text(code, "execution reason code")
        if len(set(codes)) != len(codes):
            raise ValueError("execution reason codes must be unique")
        object.__setattr__(self, "reason_codes", codes)
        if self.executed and self.deterministic and self.result_sha256 is None:
            raise ValueError(
                "successful deterministic execution requires result_sha256"
            )

    @property
    def hard_failure(self) -> bool:
        return not (
            self.executed
            and self.deterministic
            and self.result_sha256 is not None
        )

    @property
    def consensus_key(self) -> str | None:
        if not self.hard_failure:
            return self.result_sha256
        return None


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """Structured evidence for one QueryPlan v3 candidate."""

    provenance: CandidateProvenance
    hashes: IntentPlanHashes
    sql: str
    static_validation: StaticValidation
    execution: ExecutionFingerprint
    counterfactual_score: float
    semantic_score: float
    complexity: int
    judge_score: float | None = None
    counterfactual_hard_failure: bool = False

    def __post_init__(self) -> None:
        _require_text(self.sql, "sql")
        if type(self.counterfactual_hard_failure) is not bool:
            raise ValueError("counterfactual_hard_failure must be a boolean")
        object.__setattr__(
            self,
            "counterfactual_score",
            _require_unit_score(
                self.counterfactual_score, "counterfactual_score"
            ),
        )
        object.__setattr__(
            self,
            "semantic_score",
            _require_unit_score(self.semantic_score, "semantic_score"),
        )
        object.__setattr__(
            self,
            "judge_score",
            _require_unit_score(self.judge_score, "judge_score"),
        )
        if isinstance(self.complexity, bool) or not isinstance(self.complexity, int):
            raise ValueError("complexity must be a non-negative integer")
        if self.complexity < 0:
            raise ValueError("complexity must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    """Versioned immutable selector policy."""

    version: str = SELECTOR_VERSION
    semantic_weight: float = 0.55
    counterfactual_weight: float = 0.35
    judge_weight: float = 0.10
    missing_judge_score: float = 0.50
    complexity_penalty_per_unit: float = 0.001
    max_complexity_penalty: float = 0.10
    consensus_bonus_per_peer: float = 0.01
    max_consensus_bonus: float = 0.03
    selection_threshold: float = 0.65
    severe_conflict_score_margin: float = 0.02
    severe_conflict_semantic_floor: float = 0.80

    def __post_init__(self) -> None:
        _require_text(self.version, "selector config version")
        numeric = {
            "semantic_weight": self.semantic_weight,
            "counterfactual_weight": self.counterfactual_weight,
            "judge_weight": self.judge_weight,
            "missing_judge_score": self.missing_judge_score,
            "complexity_penalty_per_unit": self.complexity_penalty_per_unit,
            "max_complexity_penalty": self.max_complexity_penalty,
            "consensus_bonus_per_peer": self.consensus_bonus_per_peer,
            "max_consensus_bonus": self.max_consensus_bonus,
            "selection_threshold": self.selection_threshold,
            "severe_conflict_score_margin": self.severe_conflict_score_margin,
            "severe_conflict_semantic_floor": self.severe_conflict_semantic_floor,
        }
        for label, value in numeric.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be a finite non-negative number")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{label} must be a finite non-negative number")
        if not math.isclose(
            self.semantic_weight
            + self.counterfactual_weight
            + self.judge_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("selector score weights must sum to 1")
        if self.max_consensus_bonus > MAX_ALLOWED_CONSENSUS_BONUS:
            raise ValueError(
                "max_consensus_bonus exceeds the selector safety cap of "
                f"{MAX_ALLOWED_CONSENSUS_BONUS}"
            )
        if self.consensus_bonus_per_peer > self.max_consensus_bonus:
            raise ValueError(
                "consensus_bonus_per_peer cannot exceed max_consensus_bonus"
            )
        for label in (
            "missing_judge_score",
            "max_complexity_penalty",
            "max_consensus_bonus",
            "selection_threshold",
            "severe_conflict_score_margin",
            "severe_conflict_semantic_floor",
        ):
            if getattr(self, label) > 1:
                raise ValueError(f"{label} must be in [0, 1]")

    def canonical_payload(self) -> Mapping[str, str | float]:
        payload = asdict(self)
        return MappingProxyType(payload)

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


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: CandidateRecord
    base_score: float
    consensus_size: int
    consensus_bonus: float
    complexity_penalty: float
    final_score: float


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected: CandidateRecord | None
    abstained: bool
    reason_code: str
    ranking: tuple[RankedCandidate, ...]
    rejected_hard_failures: tuple[CandidateRecord, ...]
    config_version: str
    config_sha256: str


@dataclass(frozen=True, slots=True)
class OracleSelectorSummary:
    """Explicit offline diagnostic; correct flags must come from an oracle."""

    selected_candidate_id: str | None
    abstained: bool
    selected_correct: bool | None
    eligible_correct: int
    eligible_total: int
    hard_failure_correct: int
    hard_failure_total: int


def canonical_config_sha256(config: SelectorConfig) -> str:
    """Return the canonical SHA-256 identity of a selector configuration."""

    return config.canonical_sha256


def _base_score(candidate: CandidateRecord, config: SelectorConfig) -> float:
    judge_score = (
        config.missing_judge_score
        if candidate.judge_score is None
        else candidate.judge_score
    )
    return (
        config.semantic_weight * candidate.semantic_score
        + config.counterfactual_weight * candidate.counterfactual_score
        + config.judge_weight * judge_score
    )


def _has_severe_conflict(
    top: RankedCandidate,
    ranking: Sequence[RankedCandidate],
    config: SelectorConfig,
) -> bool:
    if top.candidate.semantic_score < config.severe_conflict_semantic_floor:
        return False
    conflicting_intents = {top.candidate.hashes.intent_sha256}
    conflicting_results = {top.candidate.execution.consensus_key}
    for item in ranking[1:]:
        if top.final_score - item.final_score > config.severe_conflict_score_margin:
            break
        if item.candidate.semantic_score < config.severe_conflict_semantic_floor:
            continue
        conflicting_intents.add(item.candidate.hashes.intent_sha256)
        conflicting_results.add(item.candidate.execution.consensus_key)
    return len(conflicting_intents) > 1 and len(conflicting_results) > 1

def select_candidate(
    candidates: Sequence[CandidateRecord],
    config: SelectorConfig = SelectorConfig(),
) -> SelectionResult:
    """Rank candidates deterministically or abstain.

    Hard static, execution, and counterfactual failures are partitioned out before
    any score or consensus bucket is computed.  The ordering key ends in stable
    content/provenance identities, so input ordering cannot change the result and
    raw result-bucket majority is never used as a selector.
    """

    if not isinstance(config, SelectorConfig):
        raise TypeError("config must be a SelectorConfig")
    records = tuple(candidates)
    if any(not isinstance(record, CandidateRecord) for record in records):
        raise TypeError("candidates must contain only CandidateRecord values")
    identifiers = [record.provenance.candidate_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate_id values must be unique")

    rejected = tuple(
        sorted(
            (
                record
                for record in records
                if record.static_validation.hard_failure
                or record.execution.hard_failure
                or record.counterfactual_hard_failure
            ),
            key=lambda record: record.provenance.candidate_id,
        )
    )
    eligible = tuple(
        record
        for record in records
        if not record.static_validation.hard_failure
        and not record.execution.hard_failure
        and not record.counterfactual_hard_failure
    )
    if not eligible:
        return SelectionResult(
            selected=None,
            abstained=True,
            reason_code=("no_candidates" if not records else "no_eligible_candidates"),
            ranking=(),
            rejected_hard_failures=rejected,
            config_version=config.version,
            config_sha256=config.canonical_sha256,
        )

    consensus_counts: dict[str, int] = {}
    for record in eligible:
        key = record.execution.consensus_key
        if key is not None:
            consensus_counts[key] = consensus_counts.get(key, 0) + 1

    ranked: list[RankedCandidate] = []
    for record in eligible:
        key = record.execution.consensus_key
        consensus_size = consensus_counts.get(key, 0) if key is not None else 0
        bonus = min(
            config.max_consensus_bonus,
            max(0, consensus_size - 1) * config.consensus_bonus_per_peer,
        )
        penalty = min(
            config.max_complexity_penalty,
            record.complexity * config.complexity_penalty_per_unit,
        )
        base_score = _base_score(record, config)
        ranked.append(
            RankedCandidate(
                candidate=record,
                base_score=base_score,
                consensus_size=consensus_size,
                consensus_bonus=bonus,
                complexity_penalty=penalty,
                final_score=min(
                    1.0,
                    max(0.0, base_score + bonus - penalty),
                ),
            )
        )

    ranking = tuple(
        sorted(
            ranked,
            key=lambda item: (
                -item.final_score,
                -item.candidate.semantic_score,
                -item.candidate.counterfactual_score,
                item.complexity_penalty,
                item.candidate.hashes.intent_sha256,
                item.candidate.hashes.plan_sha256,
                item.candidate.provenance.generator,
                item.candidate.provenance.generation_index,
                item.candidate.provenance.candidate_id,
                item.candidate.sql,
            ),
        )
    )
    top = ranking[0]
    if top.final_score < config.selection_threshold:
        return SelectionResult(
            selected=None,
            abstained=True,
            reason_code="below_selection_threshold",
            ranking=ranking,
            rejected_hard_failures=rejected,
            config_version=config.version,
            config_sha256=config.canonical_sha256,
        )
    if _has_severe_conflict(top, ranking, config):
        return SelectionResult(
            selected=None,
            abstained=True,
            reason_code="unresolved_severe_conflict",
            ranking=ranking,
            rejected_hard_failures=rejected,
            config_version=config.version,
            config_sha256=config.canonical_sha256,
        )
    return SelectionResult(
        selected=top.candidate,
        abstained=False,
        reason_code="selected",
        ranking=ranking,
        rejected_hard_failures=rejected,
        config_version=config.version,
        config_sha256=config.canonical_sha256,
    )


def oracle_selector_summary(
    result: SelectionResult,
    correct_flags: Mapping[str, bool],
) -> OracleSelectorSummary:
    """Summarise selector quality using externally supplied oracle labels.

    This function is intentionally separate from selection.  Runtime selection
    has no parameter or code path for gold labels.
    """

    records = tuple(item.candidate for item in result.ranking) + tuple(
        result.rejected_hard_failures
    )
    identifiers = [record.provenance.candidate_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("selection result contains duplicate candidate IDs")
    expected = set(identifiers)
    supplied = set(correct_flags)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(
            f"correct_flags must exactly cover candidates; missing={missing}, extra={extra}"
        )
    if any(type(flag) is not bool for flag in correct_flags.values()):
        raise ValueError("correct_flags values must be booleans")

    eligible_ids = [
        item.candidate.provenance.candidate_id for item in result.ranking
    ]
    rejected_ids = [
        record.provenance.candidate_id for record in result.rejected_hard_failures
    ]
    selected_id = (
        None if result.selected is None else result.selected.provenance.candidate_id
    )
    return OracleSelectorSummary(
        selected_candidate_id=selected_id,
        abstained=result.abstained,
        selected_correct=(
            None if selected_id is None else correct_flags[selected_id]
        ),
        eligible_correct=sum(correct_flags[candidate_id] for candidate_id in eligible_ids),
        eligible_total=len(eligible_ids),
        hard_failure_correct=sum(
            correct_flags[candidate_id] for candidate_id in rejected_ids
        ),
        hard_failure_total=len(rejected_ids),
    )


__all__ = [
    "SELECTOR_VERSION",
    "MAX_ALLOWED_CONSENSUS_BONUS",
    "CandidateProvenance",
    "IntentPlanHashes",
    "StaticValidation",
    "ExecutionFingerprint",
    "CandidateRecord",
    "SelectorConfig",
    "RankedCandidate",
    "SelectionResult",
    "OracleSelectorSummary",
    "canonical_config_sha256",
    "select_candidate",
    "oracle_selector_summary",
]
