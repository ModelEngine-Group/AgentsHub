from __future__ import annotations

import pytest

from dki.analysis.nl2sql_query_plan_v3 import (
    QueryPlanV3Error,
    canonical_plan_sha256_v3,
    compile_query_plan_v3,
    parse_query_plan_v3,
    query_plan_v3_schema,
    validate_query_plan_v3,
)


def _column(table: str, column: str) -> dict[str, str]:
    return {"type": "column", "table": table, "column": column}


def _intent(
    *,
    outputs: list[str],
    grain_kind: str = "rows",
    grain_keys: list[str] | None = None,
    joins: list[dict] | None = None,
    aggregation_stage: str = "none",
    distinct_scope: str = "none",
    row_set_mode: str = "preserve",
    preserve_alias: str | None = None,
    include_zero: bool = False,
    set_mode: str = "none",
    preserve_duplicates: bool = False,
    tie_mode: str = "unspecified",
    tie_keys: list[str] | None = None,
    ordering: list[dict] | None = None,
    precision_mode: str = "exact",
    precision_digits: int | None = None,
) -> dict:
    return {
        "output": {"columns": outputs, "row_set_mode": row_set_mode},
        "grain": {"kind": grain_kind, "keys": grain_keys or []},
        "joins": joins or [],
        "entity_preservation": {
            "mode": "preserve" if preserve_alias else "none",
            "alias": preserve_alias,
            "include_zero_match": include_zero,
        },
        "aggregation": {
            "stage": aggregation_stage,
            "distinct_scope": distinct_scope,
        },
        "set_semantics": {
            "mode": set_mode,
            "preserve_duplicates": preserve_duplicates,
        },
        "null_semantics": {"mode": "unspecified"},
        "tie_policy": {"mode": tie_mode, "keys": tie_keys or []},
        "ordering": ordering or [],
        "precision": {"mode": precision_mode, "digits": precision_digits},
    }


def _plan(intent: dict, query: dict) -> dict:
    return {"plan_version": 3, "intent": intent, "query": query}


def test_v3_compiles_through_bounded_v2_and_hashes_canonically() -> None:
    plan = _plan(
        _intent(
            outputs=["department", "COUNT(*)"],
            grain_kind="groups",
            grain_keys=["department"],
            aggregation_stage="per_group",
            ordering=[
                {"key": "COUNT(*)", "direction": "desc", "nulls": "unspecified"},
                {"key": "department", "direction": "asc", "nulls": "unspecified"},
            ],
            tie_mode="deterministic",
            tie_keys=["department"],
        ),
        {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "select": [
                {"expr": _column("v", "department")},
                {
                    "expr": {
                        "type": "aggregate",
                        "function": "count",
                        "argument": None,
                    },
                    "alias": "n",
                },
            ],
            "group_by": [_column("v", "department")],
            "order_by": [
                {
                    "expr": {
                        "type": "aggregate",
                        "function": "count",
                        "argument": None,
                    },
                    "direction": "desc",
                },
                {"expr": _column("v", "department"), "direction": "asc"},
            ],
        },
    )

    parsed = parse_query_plan_v3(plan)
    sql = compile_query_plan_v3(parsed)

    assert parsed["plan_version"] == 3
    assert "GROUP BY" in sql
    assert "COUNT(*)" in sql
    assert canonical_plan_sha256_v3(plan) == canonical_plan_sha256_v3(parsed)
    assert len(canonical_plan_sha256_v3(plan)) == 64
    validate_query_plan_v3(plan, "按科室统计就诊次数，按次数降序、科室升序返回")


def test_v3_rejects_raw_sql_unknown_fields_and_duplicate_json_keys() -> None:
    query = {
        "type": "select",
        "from": {"type": "table", "table": "patients", "alias": "p"},
        "select": [{"expr": _column("p", "patient_id")}],
    }
    plan = _plan(_intent(outputs=["patient_id"]), query)
    plan["raw_sql"] = "SELECT 1"
    with pytest.raises(QueryPlanV3Error, match="unsupported fields"):
        parse_query_plan_v3(plan)

    raw = '{"plan_version":3,"plan_version":3,"intent":{},"query":{}}'
    with pytest.raises(QueryPlanV3Error, match="duplicate JSON key"):
        parse_query_plan_v3(raw)


def test_v3_rejects_group_grain_without_matching_group_by() -> None:
    plan = _plan(
        _intent(
            outputs=["department", "COUNT(*)"],
            grain_kind="groups",
            grain_keys=["department"],
            aggregation_stage="per_group",
        ),
        {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "select": [
                {"expr": _column("v", "department")},
                {
                    "expr": {
                        "type": "aggregate",
                        "function": "count",
                        "argument": None,
                    }
                },
            ],
        },
    )

    with pytest.raises(QueryPlanV3Error, match="grain|group"):
        parse_query_plan_v3(plan)


def test_v3_rejects_zero_match_preservation_with_inner_join() -> None:
    join_intent = {
        "edge": "patients_visits",
        "left_alias": "p",
        "right_alias": "v",
        "kind": "inner",
        "preserve": "none",
    }
    plan = _plan(
        _intent(
            outputs=["patient_id", "COUNT(*)"],
            grain_kind="groups",
            grain_keys=["patient_id"],
            joins=[join_intent],
            aggregation_stage="per_group",
            preserve_alias="p",
            include_zero=True,
        ),
        {
            "type": "select",
            "from": {"type": "table", "table": "patients", "alias": "p"},
            "joins": [
                {
                    "kind": "inner",
                    "source": {"type": "table", "table": "visits", "alias": "v"},
                    "on": [
                        {
                            "type": "binary",
                            "operator": "eq",
                            "left": _column("p", "patient_id"),
                            "right": _column("v", "patient_id"),
                        }
                    ],
                }
            ],
            "select": [
                {"expr": _column("p", "patient_id")},
                {
                    "expr": {
                        "type": "aggregate",
                        "function": "count",
                        "argument": None,
                    }
                },
            ],
            "group_by": [_column("p", "patient_id")],
        },
    )

    with pytest.raises(QueryPlanV3Error, match="zero|preserv|left"):
        parse_query_plan_v3(plan)


def test_v3_rejects_many_to_many_row_count_without_explicit_distinct_protection() -> None:
    join_intent = {
        "edge": "visits_disease_triples",
        "left_alias": "v",
        "right_alias": "t",
        "kind": "inner",
        "preserve": "none",
    }
    plan = _plan(
        _intent(
            outputs=["COUNT(*)"],
            grain_kind="scalar",
            joins=[join_intent],
            aggregation_stage="after_join",
        ),
        {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "joins": [
                {
                    "source": {"type": "table", "table": "kg_triples", "alias": "t"},
                    "on": [
                        {
                            "type": "binary",
                            "operator": "eq",
                            "left": _column("v", "disease"),
                            "right": _column("t", "head"),
                        }
                    ],
                }
            ],
            "select": [
                {
                    "expr": {
                        "type": "aggregate",
                        "function": "count",
                        "argument": None,
                    }
                }
            ],
        },
    )

    with pytest.raises(QueryPlanV3Error, match="schema guard|many-to-many|amplif|distinct"):
        parse_query_plan_v3(plan)


def test_v3_model_schema_exposes_intent_and_embedded_v2_contract() -> None:
    contract = query_plan_v3_schema()

    assert contract["properties"]["plan_version"] == {"const": 3}
    assert "intent" in contract["required"]
    assert "query" in contract["required"]
    assert contract["schema_graph_sha256"]
    assert contract["query_plan_v2"] == query_plan_v3_schema()["query_plan_v2"]
    assert contract["query_plan_v2"]["properties"]["plan_version"] == {"const": 2}


def test_v3_parse_rejects_output_intent_mismatch() -> None:
    plan = _plan(
        _intent(outputs=["patient_id"], grain_keys=["patient_id"]),
        {
            "type": "select",
            "from": {"type": "table", "table": "patients", "alias": "p"},
            "select": [{"expr": _column("p", "name")}],
        },
    )

    with pytest.raises(QueryPlanV3Error, match="projection semantic mismatch"):
        parse_query_plan_v3(plan)


def test_v3_accepts_schema_edge_in_reverse_join_order() -> None:
    plan = _plan(
        _intent(
            outputs=["visit_id"],
            grain_keys=["visit_id"],
            joins=[
                {
                    "edge": "patients_visits",
                    "left_alias": "p",
                    "right_alias": "v",
                    "kind": "inner",
                    "preserve": "none",
                }
            ],
        ),
        {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "joins": [
                {
                    "source": {"type": "table", "table": "patients", "alias": "p"},
                    "on": [
                        {
                            "type": "binary",
                            "operator": "eq",
                            "left": _column("v", "patient_id"),
                            "right": _column("p", "patient_id"),
                        }
                    ],
                }
            ],
            "select": [{"expr": _column("v", "visit_id")}],
        },
    )

    assert parse_query_plan_v3(plan)["intent"]["joins"][0]["edge"] == "patients_visits"


def test_v3_rejects_schema_guard_buried_under_or() -> None:
    plan = _plan(
        _intent(
            outputs=["visit_id", "type"],
            grain_keys=["visit_id"],
            joins=[
                {
                    "edge": "visits_disease_entities",
                    "left_alias": "v",
                    "right_alias": "e",
                    "kind": "inner",
                    "preserve": "none",
                }
            ],
        ),
        {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "joins": [
                {
                    "source": {"type": "table", "table": "kg_entities", "alias": "e"},
                    "on": [
                        {
                            "type": "binary",
                            "operator": "eq",
                            "left": _column("v", "disease"),
                            "right": _column("e", "name"),
                        }
                    ],
                }
            ],
            "select": [
                {"expr": _column("v", "visit_id")},
                {"expr": _column("e", "type")},
            ],
            "where": {
                "type": "binary",
                "operator": "or",
                "left": {
                    "type": "binary",
                    "operator": "eq",
                    "left": _column("e", "type"),
                    "right": {"type": "literal", "value": "Disease"},
                },
                "right": {
                    "type": "binary",
                    "operator": "eq",
                    "left": _column("v", "department"),
                    "right": {"type": "literal", "value": "Cardiology"},
                },
            },
        },
    )

    with pytest.raises(QueryPlanV3Error, match="schema guard"):
        parse_query_plan_v3(plan)


def test_v3_rejects_count_star_for_zero_match_groups() -> None:
    plan = _plan(
        _intent(
            outputs=["patient_id", "COUNT(*)"],
            grain_kind="groups",
            grain_keys=["patient_id"],
            joins=[
                {
                    "edge": "patients_visits",
                    "left_alias": "p",
                    "right_alias": "v",
                    "kind": "left",
                    "preserve": "left",
                }
            ],
            aggregation_stage="per_group",
            preserve_alias="p",
            include_zero=True,
        ),
        {
            "type": "select",
            "from": {"type": "table", "table": "patients", "alias": "p"},
            "joins": [
                {
                    "kind": "left",
                    "source": {"type": "table", "table": "visits", "alias": "v"},
                    "on": [
                        {
                            "type": "binary",
                            "operator": "eq",
                            "left": _column("p", "patient_id"),
                            "right": _column("v", "patient_id"),
                        }
                    ],
                }
            ],
            "select": [
                {"expr": _column("p", "patient_id")},
                {"expr": {"type": "aggregate", "function": "count", "argument": None}},
            ],
            "group_by": [_column("p", "patient_id")],
        },
    )

    with pytest.raises(QueryPlanV3Error, match=r"COUNT\(\*\)"):
        parse_query_plan_v3(plan)


def test_v3_rejects_undeclared_nested_set_operation() -> None:
    def branch(alias: str) -> dict:
        return {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": alias},
            "select": [{"expr": _column(alias, "patient_id")}],
        }

    plan = _plan(
        _intent(outputs=["patient_id"], grain_keys=["patient_id"]),
        {
            "type": "select",
            "from": {"type": "table", "table": "patients", "alias": "p"},
            "select": [{"expr": _column("p", "patient_id")}],
            "where": {
                "type": "in",
                "value": _column("p", "patient_id"),
                "query": {
                    "type": "set_op",
                    "operator": "union",
                    "left": branch("v1"),
                    "right": branch("v2"),
                },
            },
        },
    )

    with pytest.raises(QueryPlanV3Error, match="not declared"):
        parse_query_plan_v3(plan)


def test_v3_rejects_contradictory_null_and_unbounded_order_claims() -> None:
    null_intent = _intent(outputs=["visit_id"], grain_keys=["visit_id"])
    null_intent["null_semantics"] = {"mode": "only"}
    null_plan = _plan(
        null_intent,
        {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "select": [{"expr": _column("v", "visit_id")}],
            "where": {
                "type": "binary",
                "operator": "and",
                "left": {
                    "type": "binary",
                    "operator": "is",
                    "left": _column("v", "exam"),
                    "right": {"type": "literal", "value": None},
                },
                "right": {
                    "type": "binary",
                    "operator": "is_not",
                    "left": _column("v", "department"),
                    "right": {"type": "literal", "value": None},
                },
            },
        },
    )
    with pytest.raises(QueryPlanV3Error, match="null-only"):
        parse_query_plan_v3(null_plan)

    order_plan = _plan(
        _intent(
            outputs=["name"],
            grain_keys=["name"],
            ordering=[{"key": "name", "direction": "asc", "nulls": "unspecified"}],
        ),
        {
            "type": "select",
            "from": {"type": "table", "table": "patients", "alias": "p"},
            "select": [{"expr": _column("p", "name")}],
            "order_by": [
                {
                    "expr": {
                        "type": "function",
                        "function": "lower",
                        "args": [_column("p", "name")],
                    },
                    "direction": "asc",
                }
            ],
        },
    )
    with pytest.raises(QueryPlanV3Error, match="bounded semantic key"):
        parse_query_plan_v3(order_plan)


def test_v3_resolves_intent_ordering_through_a_select_alias() -> None:
    def plan(order_entry: dict) -> dict:
        return _plan(
            _intent(
                outputs=["department", "COUNT(*)"],
                grain_kind="groups",
                grain_keys=["department"],
                aggregation_stage="per_group",
                ordering=[
                    {"key": "COUNT(*)", "direction": "desc", "nulls": "unspecified"}
                ],
            ),
            {
                "type": "select",
                "from": {"type": "table", "table": "visits", "alias": "v"},
                "select": [
                    {"expr": _column("v", "department"), "alias": "department"},
                    {
                        "expr": {
                            "type": "aggregate",
                            "function": "count",
                            "argument": None,
                        },
                        "alias": "visit_count",
                    },
                ],
                "group_by": [_column("v", "department")],
                "order_by": [order_entry],
            },
        )

    sql = compile_query_plan_v3(plan({"alias": "visit_count", "direction": "desc"}))
    assert sql.endswith("ORDER BY COUNT(*) DESC")
    # The alias resolves to the expression it names, so a declared-but-wrong
    # alias still contradicts the intent.
    with pytest.raises(QueryPlanV3Error, match="ordering key contradicts intent"):
        parse_query_plan_v3(plan({"alias": "department", "direction": "desc"}))


def test_v3_checks_intent_ordering_on_a_bare_set_operation() -> None:
    def plan(order_by: list[dict]) -> dict:
        branch = lambda threshold: {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "select": [{"expr": _column("v", "department"), "alias": "department"}],
            "where": {
                "type": "binary",
                "operator": "gt",
                "left": _column("v", "cost"),
                "right": {"type": "literal", "value": threshold},
            },
            "distinct": True,
        }
        query = {
            "type": "set_op",
            "operator": "union",
            "left": branch(100),
            "right": branch(200),
        }
        if order_by:
            query["order_by"] = order_by
        return _plan(
            _intent(
                outputs=["department"],
                set_mode="any_of",
                row_set_mode="distinct",
                ordering=[
                    {"key": "department", "direction": "asc", "nulls": "unspecified"}
                ],
            ),
            query,
        )

    sql = compile_query_plan_v3(plan([{"alias": "department", "direction": "asc"}]))
    assert sql.endswith('ORDER BY "department" ASC')
    with pytest.raises(QueryPlanV3Error, match="declared ordering is missing"):
        parse_query_plan_v3(plan([]))
