from __future__ import annotations

import hashlib

from dki.analysis.nl2sql_candidate_selector import select_candidate
from dki.analysis.nl2sql_semantic_scorer import (
    candidate_record_from_v3,
    score_candidate_semantics,
)


def _column(table: str, column: str) -> dict[str, str]:
    return {"type": "column", "table": table, "column": column}


def _plan(*, distinct_count: bool) -> dict:
    aggregate = {
        "type": "aggregate",
        "function": "count",
        "argument": _column("v", "patient_id"),
        "distinct": distinct_count,
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
                "distinct_scope": (
                    "aggregate_arguments" if distinct_count else "none"
                ),
            },
            "set_semantics": {"mode": "none", "preserve_duplicates": False},
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


def test_semantic_scorer_hard_fails_count_distinct_contradiction() -> None:
    question = "按科室返回不同患者数，按人数降序、科室升序列出。"
    valid_plan = _plan(distinct_count=True)
    invalid_plan = _plan(distinct_count=False)
    valid_sql = (
        'SELECT "v"."department", COUNT(DISTINCT "v"."patient_id") AS "n" '
        'FROM "visits" AS "v" GROUP BY "v"."department" '
        'ORDER BY COUNT(DISTINCT "v"."patient_id") DESC, "v"."department" ASC'
    )
    invalid_sql = valid_sql.replace("COUNT(DISTINCT ", "COUNT(")

    valid = score_candidate_semantics(question, valid_plan, valid_sql)
    invalid = score_candidate_semantics(question, invalid_plan, invalid_sql)

    assert valid.hard_failure is False
    assert valid.plan_valid is True
    assert valid.contract_complete is True
    assert valid.contract_valid is True
    assert invalid.hard_failure is True
    assert invalid.score <= 0.49
    assert any("distinct" in code.casefold() for code in invalid.reason_codes)


def test_candidate_record_feeds_fail_closed_selector() -> None:
    question = "按科室返回不同患者数，按人数降序、科室升序列出。"
    valid_plan = _plan(distinct_count=True)
    invalid_plan = _plan(distinct_count=False)
    valid_sql = (
        'SELECT "v"."department", COUNT(DISTINCT "v"."patient_id") AS "n" '
        'FROM "visits" AS "v" GROUP BY "v"."department" '
        'ORDER BY COUNT(DISTINCT "v"."patient_id") DESC, "v"."department" ASC'
    )
    invalid_sql = valid_sql.replace("COUNT(DISTINCT ", "COUNT(")

    valid = candidate_record_from_v3(
        question=question,
        plan=valid_plan,
        sql=valid_sql,
        generator="semantic_full",
        generation_index=0,
        execution_result_sha256=hashlib.sha256(b"valid").hexdigest(),
        counterfactual_score=0.95,
        complexity=8,
    )
    invalid = candidate_record_from_v3(
        question=question,
        plan=invalid_plan,
        sql=invalid_sql,
        generator="grain_cardinality",
        generation_index=1,
        execution_result_sha256=hashlib.sha256(b"invalid").hexdigest(),
        counterfactual_score=0.20,
        complexity=7,
        judge_score=1.0,
    )

    result = select_candidate([invalid, valid])

    assert result.selected == valid
    assert invalid in result.rejected_hard_failures
