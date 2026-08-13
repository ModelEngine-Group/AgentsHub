from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from dki.analysis.nl2sql_candidate_selector import (
    CandidateProvenance,
    CandidateRecord,
    ExecutionFingerprint,
    IntentPlanHashes,
    SelectorConfig,
    StaticValidation,
    select_candidate,
)
from dki.analysis.nl2sql_query_plan_v3 import compile_query_plan_v3
from dki.analysis.nl2sql_query_plan_v3_pipeline import (
    SEMANTIC_LANES,
    QueryPlanV3PipelineConfig,
    assess_static_validation,
    build_lane_prompt,
    plan_counterfactual_probes,
    run_query_plan_v3_pipeline,
)


def _column(table: str, column: str) -> dict[str, str]:
    return {"type": "column", "table": table, "column": column}


def _valid_plan() -> dict:
    aggregate = {
        "type": "aggregate",
        "function": "count",
        "argument": _column("v", "patient_id"),
        "distinct": True,
    }
    return {
        "plan_version": 3,
        "intent": {
            "output": {
                "columns": ["department", "COUNT(DISTINCT patient_id)"],
                "row_set_mode": "preserve",
            },
            "grain": {"kind": "groups", "keys": ["department"]},
            "joins": [],
            "entity_preservation": {
                "mode": "none",
                "alias": None,
                "include_zero_match": False,
            },
            "aggregation": {
                "stage": "per_group",
                "distinct_scope": "aggregate_arguments",
            },
            "set_semantics": {
                "mode": "none",
                "preserve_duplicates": False,
            },
            "null_semantics": {"mode": "unspecified"},
            "tie_policy": {
                "mode": "deterministic",
                "keys": ["department"],
            },
            "ordering": [
                {
                    "key": "COUNT(DISTINCT patient_id)",
                    "direction": "desc",
                    "nulls": "unspecified",
                },
                {
                    "key": "department",
                    "direction": "asc",
                    "nulls": "unspecified",
                },
            ],
            "precision": {"mode": "exact", "digits": None},
        },
        "query": {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "select": [
                {"expr": _column("v", "department")},
                {"expr": aggregate, "alias": "n"},
            ],
            "group_by": [_column("v", "department")],
            "order_by": [
                {"expr": aggregate, "direction": "desc"},
                {"expr": _column("v", "department"), "direction": "asc"},
            ],
        },
    }


QUESTION = "按科室返回不同患者数，按人数降序、科室升序列出。"
LIVE_SCHEMA = (
    "CREATE TABLE visits (visit_id TEXT, patient_id TEXT, visit_date TEXT, "
    "department TEXT, disease TEXT, drug_prescribed TEXT, exam TEXT, "
    "cost REAL, los_days INTEGER);"
)
RESULT_HASH = hashlib.sha256(b"complete-result").hexdigest()


def _successful_executor(sql: str, db_path: Path) -> dict:
    assert sql.startswith("SELECT")
    assert isinstance(db_path, Path)
    return {
        "executed": True,
        "complete": True,
        "readonly": True,
        "deterministic": True,
        "result_sha256": RESULT_HASH,
        "row_count": 2,
    }


def _passing_counterfactual(db_path: Path, sql: str, probes) -> dict:
    assert isinstance(db_path, Path)
    assert sql.startswith("SELECT")
    return {
        "version": "mock-counterfactual-v1",
        "score": 1.0,
        "passed": len(probes),
        "failed": 0,
        "skipped": 0,
        "outcomes": tuple(
            {
                "probe": spec["probe"],
                "applicable": True,
                "passed": True,
                "reason_code": "passed",
            }
            for spec in probes
        ),
    }


def test_lane_prompts_are_fixed_deterministic_and_feedback_is_allowlisted() -> None:
    config = QueryPlanV3PipelineConfig(max_feedback_items=2)
    prompts = [
        build_lane_prompt(
            question=QUESTION,
            live_schema_description=LIVE_SCHEMA,
            lane=lane,
            validation_feedback=(
                "json_schema",
                "execution_failed: SELECT secret FROM gold",
                "schema_validation",
                "question_grounding",
            ),
            config=config,
        )
        for lane in SEMANTIC_LANES
    ]

    assert SEMANTIC_LANES == (
        "grain/entity-preservation",
        "join/cardinality",
        "set-null-boundary",
        "ordering-precision-conservative",
    )
    assert len(set(prompts)) == 4
    for lane, prompt in zip(SEMANTIC_LANES, prompts):
        assert lane in prompt
        assert QUESTION in prompt
        assert LIVE_SCHEMA in prompt
        assert '"$id":"https://modelengine.example/schemas/query-plan-v3.json"' in prompt
        assert '["json_schema","schema_validation"]' in prompt
        assert "SELECT secret" not in prompt
        assert "execution_failed" not in prompt
        assert prompt == build_lane_prompt(
            question=QUESTION,
            live_schema_description=LIVE_SCHEMA,
            lane=lane,
            validation_feedback=(
                "json_schema",
                "execution_failed: SELECT secret FROM gold",
                "schema_validation",
                "question_grounding",
            ),
            config=config,
        )


def test_pipeline_repairs_parse_failures_without_raw_candidate_feedback(tmp_path) -> None:
    generation_prompts: list[str] = []
    repair_prompts: list[str] = []
    invalid_output = "NOT JSON: SELECT * FROM hidden_gold"

    def generate(prompt: str) -> str:
        generation_prompts.append(prompt)
        return invalid_output

    def repair(prompt: str) -> str:
        repair_prompts.append(prompt)
        return json.dumps(_valid_plan(), ensure_ascii=False)

    result = run_query_plan_v3_pipeline(
        question=QUESTION,
        live_schema_description=LIVE_SCHEMA,
        db_path=tmp_path / "unused.sqlite",
        generate_candidate=generate,
        repair_candidate=repair,
        executor=_successful_executor,
        counterfactual_runner=_passing_counterfactual,
        config=QueryPlanV3PipelineConfig(max_attempts_per_lane=2),
    )

    assert result["abstained"] is False
    assert result["attempts"] == 8
    assert result["rounds"] == 2
    assert result["n_candidates"] == 4
    assert len(generation_prompts) == len(repair_prompts) == 4
    assert all("BOUNDED VALIDATION FEEDBACK\n[]" in prompt for prompt in generation_prompts)
    assert all('["json_schema"]' in prompt for prompt in repair_prompts)
    assert all(invalid_output not in prompt for prompt in repair_prompts)
    assert all("hidden_gold" not in prompt for prompt in repair_prompts)
    assert [item["stage"] for item in result["attempt_diagnostics"]].count(
        "json_schema"
    ) == 4


def test_static_validation_flags_are_independent() -> None:
    plan = _valid_plan()
    invalid_sql = (
        'SELECT "v"."missing_column" FROM "visits" AS "v" '
        'ORDER BY "v"."missing_column" ASC'
    )

    validation, semantic = assess_static_validation(
        question=QUESTION,
        plan=plan,
        sql=invalid_sql,
    )

    assert validation.syntax_valid is True
    assert validation.read_only is True
    assert validation.schema_valid is False
    assert validation.grounded is True
    assert validation.complete is False
    assert validation.hard_failure is True
    assert semantic.hard_failure is True
    assert any(code.startswith("schema:") for code in validation.reason_codes)


def test_probe_planning_is_deterministic_and_uses_declared_distinct_intent() -> None:
    plan = _valid_plan()

    first = plan_counterfactual_probes(plan, question=QUESTION)
    second = plan_counterfactual_probes(plan, question=QUESTION)

    assert first == second
    assert first == (
        {
            "probe": "row_count_vs_distinct",
            "expected_relation": "same_result",
        },
    )


def test_counterfactual_failure_is_a_hard_reject_even_with_perfect_score(tmp_path) -> None:
    def generate(_: str) -> dict:
        return _valid_plan()

    def failing_counterfactual(db_path: Path, sql: str, probes) -> dict:
        assert probes
        return {
            "version": "mock-counterfactual-v1",
            "score": 1.0,
            "passed": 0,
            "failed": 1,
            "skipped": 0,
            "outcomes": (
                {
                    "probe": probes[0]["probe"],
                    "applicable": True,
                    "passed": False,
                    "reason_code": "invariant_failed",
                },
            ),
        }

    result = run_query_plan_v3_pipeline(
        question=QUESTION,
        live_schema_description=LIVE_SCHEMA,
        db_path=tmp_path / "unused.sqlite",
        generate_candidate=generate,
        executor=_successful_executor,
        counterfactual_runner=failing_counterfactual,
        config=QueryPlanV3PipelineConfig(max_attempts_per_lane=1),
    )

    assert result["abstained"] is True
    assert result["sql"] is None
    assert result["confidence"] == 0.0
    assert result["selector_reason"] == "no_eligible_candidates"
    assert result["selector_metadata"]["hard_rejected_candidates"] == 4
    assert all(
        item["counterfactual_hard_failure"] is True
        and item["hard_rejected"] is True
        for item in result["candidate_diagnostics"]
    )


def _record(
    candidate_id: str,
    *,
    intent_character: str,
    plan_character: str,
    result_character: str,
    counterfactual_hard_failure: bool = False,
) -> CandidateRecord:
    return CandidateRecord(
        provenance=CandidateProvenance(
            candidate_id=candidate_id,
            generator="test",
            generation_index=0,
            strategy="query_plan_v3",
        ),
        hashes=IntentPlanHashes(
            intent_sha256=intent_character * 64,
            plan_sha256=plan_character * 64,
        ),
        sql=f"SELECT '{candidate_id}'",
        static_validation=StaticValidation(),
        execution=ExecutionFingerprint(result_sha256=result_character * 64),
        counterfactual_score=1.0,
        semantic_score=0.9,
        complexity=1,
        judge_score=None,
        counterfactual_hard_failure=counterfactual_hard_failure,
    )


def test_selector_is_input_order_deterministic_and_eliminates_probe_failures() -> None:
    first = _record(
        "first",
        intent_character="b",
        plan_character="b",
        result_character="b",
    )
    second = _record(
        "second",
        intent_character="a",
        plan_character="a",
        result_character="a",
    )
    rejected = _record(
        "rejected",
        intent_character="c",
        plan_character="c",
        result_character="c",
        counterfactual_hard_failure=True,
    )
    config = SelectorConfig(
        severe_conflict_score_margin=0.0,
        severe_conflict_semantic_floor=1.0,
    )

    forward = select_candidate([first, rejected, second], config)
    reverse = select_candidate([second, rejected, first], config)

    assert forward.selected == reverse.selected == second
    assert rejected in forward.rejected_hard_failures
    assert [item.candidate.provenance.candidate_id for item in forward.ranking] == [
        item.candidate.provenance.candidate_id for item in reverse.ranking
    ]


def test_public_pipeline_surface_has_no_oracle_or_gold_parameters() -> None:
    forbidden = ("gold", "oracle", "reference", "correct", "judge")
    for callable_ in (
        run_query_plan_v3_pipeline,
        build_lane_prompt,
        assess_static_validation,
        plan_counterfactual_probes,
        QueryPlanV3PipelineConfig,
    ):
        parameters = inspect.signature(callable_).parameters
        assert not any(
            token in name.casefold()
            for name in parameters
            for token in forbidden
        )


def test_selected_result_contains_legacy_and_selector_metadata(tmp_path) -> None:
    counterfactual_calls: list[tuple[dict, ...]] = []

    def generate(_: str) -> dict:
        return _valid_plan()

    def counterfactual(db_path: Path, sql: str, probes) -> dict:
        counterfactual_calls.append(tuple(dict(item) for item in probes))
        return _passing_counterfactual(db_path, sql, probes)

    result = run_query_plan_v3_pipeline(
        question=QUESTION,
        live_schema_description=LIVE_SCHEMA,
        db_path=tmp_path / "unused.sqlite",
        generate_candidate=generate,
        executor=_successful_executor,
        counterfactual_runner=counterfactual,
        config=QueryPlanV3PipelineConfig(max_attempts_per_lane=1),
    )

    assert result["sql"] == compile_query_plan_v3(_valid_plan())
    assert isinstance(result["confidence"], float)
    assert result["confidence"] > 0.65
    assert result["strategy"] == "query_plan_v3"
    assert result["plan_version"] == 3
    assert len(result["plan_sha256"]) == 64
    assert result["n_candidates"] == 4
    assert result["n_agree"] == 4
    assert result["rounds"] == 1
    assert result["attempts"] == 4
    assert result["abstained"] is False
    assert result["execution_validated"] is True
    assert result["semantic_validated"] is True
    assert result["selector_reason"] == "selected"
    assert result["selector_metadata"]["selected_candidate_id"] == result[
        "selected_candidate_id"
    ]
    assert len(result["selector_metadata"]["ranking"]) == 4
    assert len(result["candidate_diagnostics"]) == 4
    assert sum(item["selected"] is True for item in result["candidate_diagnostics"]) == 1
    assert all(item["judge_score"] is None for item in result["candidate_diagnostics"])
    assert all("sql" not in item for item in result["candidate_diagnostics"])
    assert result["failure_stages"] == ()
    assert counterfactual_calls == [
        (
            {
                "probe": "row_count_vs_distinct",
                "expected_relation": "same_result",
            },
        )
    ] * 4


def test_format_example_plan_parses_compiles_and_is_embedded() -> None:
    from dki.analysis.nl2sql_query_plan_v3 import parse_query_plan_v3
    from dki.analysis.nl2sql_query_plan_v3_pipeline import (
        _FORMAT_EXAMPLE_PLAN,
        _INTENT_GRAMMAR_NOTE,
    )

    parsed = parse_query_plan_v3(json.dumps(_FORMAT_EXAMPLE_PLAN, ensure_ascii=False))
    sql = compile_query_plan_v3(parsed)
    assert sql.startswith("SELECT")

    prompt = build_lane_prompt(
        question="按科室统计患者数",
        live_schema_description="table visits(...)",
        lane=SEMANTIC_LANES[0],
    )
    assert "FORMAT EXAMPLE" in prompt
    assert "INTENT FIELD GRAMMAR" in prompt
    assert _INTENT_GRAMMAR_NOTE in prompt
    # The example must never leak into feedback or claim question semantics.
    assert "shape only" in prompt
