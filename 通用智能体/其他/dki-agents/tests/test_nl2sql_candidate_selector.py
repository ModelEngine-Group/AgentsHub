from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dki.analysis.nl2sql_candidate_selector import (
    CandidateProvenance,
    CandidateRecord,
    ExecutionFingerprint,
    IntentPlanHashes,
    SelectorConfig,
    StaticValidation,
    canonical_config_sha256,
    oracle_selector_summary,
    select_candidate,
)


def _sha(character: str) -> str:
    return character * 64


def _candidate(
    candidate_id: str,
    *,
    semantic: float,
    counterfactual: float,
    result: str,
    intent: str,
    plan: str,
    complexity: int = 10,
    judge: float | None = None,
    static_validation: StaticValidation | None = None,
    generation_index: int = 0,
) -> CandidateRecord:
    return CandidateRecord(
        provenance=CandidateProvenance(
            candidate_id=candidate_id,
            generator="queryplan-v3",
            generation_index=generation_index,
            strategy="independent",
        ),
        hashes=IntentPlanHashes(
            intent_sha256=_sha(intent),
            plan_sha256=_sha(plan),
        ),
        sql=f"SELECT '{candidate_id}'",
        static_validation=static_validation or StaticValidation(),
        execution=ExecutionFingerprint(result_sha256=_sha(result)),
        counterfactual_score=counterfactual,
        semantic_score=semantic,
        complexity=complexity,
        judge_score=judge,
    )


def test_three_vote_incomplete_bucket_loses_to_semantically_valid_candidate() -> None:
    incomplete_bucket = [
        _candidate(
            f"incomplete-{index}",
            semantic=0.50,
            counterfactual=0.45,
            result="a",
            intent="b",
            plan=character,
            complexity=4,
            judge=0.95,
            generation_index=index,
        )
        for index, character in enumerate(("c", "d", "e"))
    ]
    valid = _candidate(
        "semantically-valid",
        semantic=0.96,
        counterfactual=0.94,
        result="f",
        intent="f",
        plan="f",
        complexity=8,
        judge=0.70,
    )

    result = select_candidate([*incomplete_bucket, valid])

    assert result.selected == valid
    assert result.ranking[0].consensus_size == 1
    assert max(item.consensus_bonus for item in result.ranking) == 0.02
    assert {
        item.candidate.provenance.candidate_id
        for item in result.ranking[1:]
    } == {"incomplete-0", "incomplete-1", "incomplete-2"}


def test_hard_failures_cannot_be_rescued_by_consensus_or_judge() -> None:
    failure = StaticValidation(
        complete=False,
        reason_codes=("missing_required_predicate",),
    )
    hard_failures = [
        _candidate(
            f"hard-failure-{index}",
            semantic=1.0,
            counterfactual=1.0,
            result="a",
            intent="a",
            plan=character,
            judge=1.0,
            static_validation=failure,
            generation_index=index,
        )
        for index, character in enumerate(("b", "c", "d"))
    ]
    eligible = _candidate(
        "eligible",
        semantic=0.90,
        counterfactual=0.90,
        result="e",
        intent="e",
        plan="e",
        judge=0.0,
    )

    result = select_candidate([*hard_failures, eligible])

    assert result.selected == eligible
    assert [
        item.candidate.provenance.candidate_id for item in result.ranking
    ] == ["eligible"]
    assert result.ranking[0].consensus_bonus == 0.0
    assert {
        candidate.provenance.candidate_id
        for candidate in result.rejected_hard_failures
    } == {"hard-failure-0", "hard-failure-1", "hard-failure-2"}


def _execution_failure_candidate() -> CandidateRecord:
    return CandidateRecord(
        provenance=CandidateProvenance(
            candidate_id="execution-failure",
            generator="queryplan-v3",
            generation_index=0,
            strategy="independent",
        ),
        hashes=IntentPlanHashes(
            intent_sha256=_sha("a"),
            plan_sha256=_sha("a"),
        ),
        sql="SELECT 1",
        static_validation=StaticValidation(),
        execution=ExecutionFingerprint(
            result_sha256=None,
            executed=False,
            reason_codes=("execution_error",),
        ),
        counterfactual_score=1.0,
        semantic_score=1.0,
        complexity=0,
        judge_score=1.0,
    )


def test_execution_failure_is_eliminated_before_scoring() -> None:
    failed = _execution_failure_candidate()
    eligible = _candidate(
        "eligible-execution",
        semantic=0.90,
        counterfactual=0.90,
        result="e",
        intent="e",
        plan="e",
        judge=0.0,
    )

    result = select_candidate([failed, eligible])

    assert result.selected == eligible
    assert [item.candidate for item in result.ranking] == [eligible]
    assert result.rejected_hard_failures == (failed,)


def test_tie_breaking_and_config_hash_are_deterministic_and_immutable() -> None:
    first = _candidate(
        "first",
        semantic=0.90,
        counterfactual=0.90,
        result="a",
        intent="b",
        plan="b",
        complexity=5,
        judge=0.50,
        generation_index=9,
    )
    second = _candidate(
        "second",
        semantic=0.90,
        counterfactual=0.90,
        result="c",
        intent="a",
        plan="a",
        complexity=5,
        judge=0.50,
        generation_index=1,
    )
    config = SelectorConfig(
        severe_conflict_score_margin=0.0,
        severe_conflict_semantic_floor=1.0,
    )

    forward = select_candidate([first, second], config)
    reverse = select_candidate([second, first], config)

    assert forward.selected == second
    assert reverse.selected == second
    assert [
        item.candidate.provenance.candidate_id for item in forward.ranking
    ] == [
        item.candidate.provenance.candidate_id for item in reverse.ranking
    ]
    assert canonical_config_sha256(config) == config.canonical_sha256
    assert forward.config_sha256 == reverse.config_sha256
    assert len(forward.config_sha256) == 64
    with pytest.raises(FrozenInstanceError):
        config.selection_threshold = 0.0
    with pytest.raises(TypeError):
        config.canonical_payload()["selection_threshold"] = 0.0
    with pytest.raises(ValueError, match="safety cap"):
        SelectorConfig(max_consensus_bonus=0.051)


def test_abstains_below_threshold() -> None:
    candidate = _candidate(
        "weak",
        semantic=0.60,
        counterfactual=0.60,
        result="a",
        intent="a",
        plan="a",
        complexity=0,
        judge=0.60,
    )

    result = select_candidate(
        [candidate],
        SelectorConfig(selection_threshold=0.61),
    )

    assert result.abstained is True
    assert result.selected is None
    assert result.reason_code == "below_selection_threshold"
    assert result.ranking[0].final_score == pytest.approx(0.60)


def test_unresolved_severe_conflict_abstains_and_oracle_is_explicit() -> None:
    first = _candidate(
        "first",
        semantic=0.91,
        counterfactual=0.90,
        result="a",
        intent="a",
        plan="a",
        complexity=0,
        judge=0.90,
    )
    second = _candidate(
        "second",
        semantic=0.90,
        counterfactual=0.91,
        result="b",
        intent="b",
        plan="b",
        complexity=0,
        judge=0.90,
    )

    third = _candidate(
        "third",
        semantic=0.90,
        counterfactual=0.90,
        result="c",
        intent="c",
        plan="c",
        complexity=0,
        judge=0.90,
    )

    result = select_candidate([first, second, third])

    assert result.abstained is True
    assert result.reason_code == "unresolved_severe_conflict"
    diagnostic = oracle_selector_summary(
        result,
        {"first": True, "second": False, "third": False},
    )
    assert diagnostic.selected_correct is None
    assert diagnostic.eligible_correct == 1
    with pytest.raises(ValueError, match="exactly cover"):
        oracle_selector_summary(result, {"first": True})
