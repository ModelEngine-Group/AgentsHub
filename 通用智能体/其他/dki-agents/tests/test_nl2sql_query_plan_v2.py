from __future__ import annotations

import pytest

from dki.analysis.nl2sql_query_plan_v2 import (
    QueryPlanV2Error,
    compile_query_plan_v2,
    parse_query_plan_v2,
    query_plan_v2_schema,
    validate_plan_v2_grounding,
)


def _column(table: str, column: str) -> dict[str, str]:
    return {"type": "column", "table": table, "column": column}


def test_v2_compiles_join_group_having_without_raw_sql() -> None:
    plan = {
        "plan_version": 2,
        "query": {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "joins": [{
                "source": {"type": "table", "table": "patients", "alias": "p"},
                "on": [{"type": "binary", "operator": "eq", "left": _column("v", "patient_id"), "right": _column("p", "patient_id")}],
            }],
            "select": [
                {"expr": _column("p", "region"), "alias": "region"},
                {"expr": {"type": "aggregate", "function": "count", "argument": None}, "alias": "n"},
            ],
            "where": {"type": "binary", "operator": "gte", "left": _column("v", "cost"), "right": {"type": "literal", "value": 100}},
            "group_by": [_column("p", "region")],
            "having": {"type": "binary", "operator": "gte", "left": {"type": "aggregate", "function": "count", "argument": None}, "right": {"type": "literal", "value": 1}},
            "order_by": [{"expr": {"type": "aggregate", "function": "count", "argument": None}, "direction": "desc"}],
            "limit": 5,
        },
    }
    sql = compile_query_plan_v2(plan)
    assert "SELECT" in sql.upper()
    assert "raw_sql" not in sql
    validate_plan_v2_grounding(plan, "每个地区费用至少100元，至少1次，返回前5名")


def test_v2_compiles_cte_window_and_rejects_unknown_column() -> None:
    plan = {
        "plan_version": 2,
        "query": {
            "type": "select",
            "ctes": [{
                "name": "grouped",
                "query": {
                    "type": "select",
                    "from": {"type": "table", "table": "visits", "alias": "v"},
                    "select": [{"expr": _column("v", "department"), "alias": "department"}, {"expr": {"type": "aggregate", "function": "count", "argument": None}, "alias": "n"}],
                    "group_by": [_column("v", "department")],
                },
            }],
            "from": {"type": "cte", "name": "grouped", "alias": "g"},
            "select": [{"expr": _column("g", "department")}, {"expr": {"type": "window", "function": "row_number", "argument": None, "order_by": [{"expr": _column("g", "n"), "direction": "desc"}]}, "alias": "rank"}],
            "order_by": [{"expr": _column("g", "n"), "direction": "desc"}],
            "limit": 10,
        },
    }
    sql = compile_query_plan_v2(plan)
    assert "WITH" in sql.upper() and "ROW_NUMBER" in sql.upper()
    plan["query"]["select"][0]["expr"]["column"] = "secret"
    with pytest.raises(QueryPlanV2Error):
        compile_query_plan_v2(plan)


def test_v2_rejects_raw_sql_and_unsafe_limit() -> None:
    with pytest.raises(QueryPlanV2Error):
        parse_query_plan_v2({"plan_version": 2, "sql": "SELECT 1", "query": {}})
    with pytest.raises(QueryPlanV2Error):
        parse_query_plan_v2({"plan_version": 2, "query": {"type": "select", "from": {"type": "table", "table": "visits"}, "select": [{"expr": _column("visits", "visit_id")}], "limit": 501}})


def test_v2_window_frame_and_correlated_exists_compile() -> None:
    inner = {
        "type": "select",
        "from": {"type": "table", "table": "visits", "alias": "v2"},
        "select": [{"expr": {"type": "literal", "value": 1}}],
        "where": {
            "type": "binary",
            "operator": "eq",
            "left": _column("v2", "patient_id"),
            "right": _column("v", "patient_id"),
        },
    }
    plan = {
        "plan_version": 2,
        "query": {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "select": [
                {"expr": _column("v", "patient_id")},
                {
                    "alias": "running_cost",
                    "expr": {
                        "type": "window",
                        "function": "sum",
                        "argument": _column("v", "cost"),
                        "partition_by": [_column("v", "patient_id")],
                        "order_by": [
                            {"expr": _column("v", "visit_date"), "direction": "asc"}
                        ],
                        "frame": {
                            "unit": "rows",
                            "start": "unbounded_preceding",
                            "end": "current_row",
                        },
                    },
                },
            ],
            "where": {"type": "exists", "query": inner},
        },
    }
    sql = compile_query_plan_v2(plan)
    assert "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW" in sql
    assert '"v2"."patient_id" = "v"."patient_id"' in sql


def test_v2_rejects_unallowlisted_direct_join_path() -> None:
    plan = {
        "plan_version": 2,
        "query": {
            "type": "select",
            "from": {"type": "table", "table": "patients", "alias": "p"},
            "joins": [
                {
                    "source": {"type": "table", "table": "kg_entities", "alias": "e"},
                    "on": [
                        {
                            "type": "binary",
                            "operator": "eq",
                            "left": _column("p", "name"),
                            "right": _column("e", "name"),
                        }
                    ],
                }
            ],
            "select": [{"expr": _column("p", "patient_id")}],
        },
    }
    with pytest.raises(QueryPlanV2Error, match="join path"):
        parse_query_plan_v2(plan)


def test_v2_grounding_allows_format_controls_and_db_domain_but_not_invented_literal() -> None:
    plan = {
        "plan_version": 2,
        "query": {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "select": [
                {
                    "expr": {
                        "type": "function",
                        "function": "strftime",
                        "args": [
                            {"type": "literal", "value": "%Y-%m"},
                            _column("v", "visit_date"),
                        ],
                    }
                }
            ],
            "where": {
                "type": "binary",
                "operator": "eq",
                "left": _column("v", "department"),
                "right": {"type": "literal", "value": "Cardiology"},
            },
        },
    }
    validate_plan_v2_grounding(
        plan,
        "show visits by month",
        allowed_domain_values=("Cardiology",),
    )
    plan["query"]["where"]["right"]["value"] = "Invented"
    with pytest.raises(QueryPlanV2Error, match="grounded"):
        validate_plan_v2_grounding(plan, "show visits by month")


def test_v2_model_contract_is_complete_enough_to_generate_nodes() -> None:
    contract = query_plan_v2_schema()
    assert "grammar" in contract
    assert set(contract["grammar"]["query"]) == {"select", "set_op"}
    assert len(contract["examples"]) >= 7
    assert "window" in contract["grammar"]["expression"]


def test_v2_all_model_facing_examples_parse_and_compile() -> None:
    contract = query_plan_v2_schema()
    for example in contract["examples"]:
        parsed = parse_query_plan_v2(example["plan"])
        sql = compile_query_plan_v2(parsed)
        assert sql.startswith(("SELECT ", "WITH "))
        assert parse_query_plan_v2(parsed) == parsed


@pytest.mark.parametrize(
    ("function", "argument"),
    [
        ("lead", _column("v", "visit_date")),
        ("last_value", _column("v", "cost")),
        ("ntile", {"type": "literal", "value": 4}),
    ],
)
def test_v2_extended_windows_are_bounded_and_compile(
    function: str, argument: dict[str, object]
) -> None:
    plan = {
        "plan_version": 2,
        "query": {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "select": [
                {"expr": _column("v", "patient_id")},
                {
                    "alias": "window_value",
                    "expr": {
                        "type": "window",
                        "function": function,
                        "argument": argument,
                        "partition_by": [_column("v", "patient_id")],
                        "order_by": [
                            {
                                "expr": _column("v", "visit_date"),
                                "direction": "asc",
                            }
                        ],
                        "frame": {
                            "unit": "rows",
                            "start": "unbounded_preceding",
                            "end": "unbounded_following",
                        },
                    },
                },
            ],
        },
    }
    sql = compile_query_plan_v2(plan)
    assert function.upper() in sql


def test_v2_set_operation_uses_sqlite_safe_derived_branches() -> None:
    branch = lambda table, column, alias: {
        "type": "select",
        "from": {"type": "table", "table": table, "alias": alias},
        "select": [{"expr": _column(alias, column), "alias": "name"}],
        "distinct": True,
    }
    plan = {
        "plan_version": 2,
        "query": {
            "type": "set_op",
            "operator": "union",
            "left": branch("visits", "department", "v"),
            "right": branch("kg_entities", "name", "e"),
        },
    }
    sql = compile_query_plan_v2(plan)
    # Branch columns are projected by name; a wildcard would break downstream
    # lineage analysis, which rejects "SELECT *" outright.
    assert "*" not in sql
    assert '"_set_left"."name" FROM (SELECT DISTINCT' in sql
    assert " UNION " in sql


def test_v2_set_operation_branches_must_align_and_name_columns_uniquely() -> None:
    branch = lambda columns: {
        "type": "select",
        "from": {"type": "table", "table": "visits", "alias": "v"},
        "select": columns,
    }
    mismatched = {
        "plan_version": 2,
        "query": {
            "type": "set_op",
            "operator": "union",
            "left": branch([{"expr": _column("v", "department"), "alias": "a"}]),
            "right": branch(
                [
                    {"expr": _column("v", "department"), "alias": "a"},
                    {"expr": _column("v", "disease"), "alias": "b"},
                ]
            ),
        },
    }
    with pytest.raises(QueryPlanV2Error, match="same number of columns"):
        parse_query_plan_v2(mismatched)
    duplicated = {
        "plan_version": 2,
        "query": {
            "type": "set_op",
            "operator": "union",
            "left": branch(
                [
                    {"expr": _column("v", "department"), "alias": "a"},
                    {"expr": _column("v", "disease"), "alias": "a"},
                ]
            ),
            "right": branch(
                [
                    {"expr": _column("v", "department"), "alias": "a"},
                    {"expr": _column("v", "disease"), "alias": "b"},
                ]
            ),
        },
    }
    with pytest.raises(QueryPlanV2Error, match="unique name"):
        parse_query_plan_v2(duplicated)


def test_v2_offset_compiles_and_requires_limit() -> None:
    base = {
        "type": "select",
        "from": {"type": "table", "table": "kg_triples", "alias": "t"},
        "select": [{"expr": _column("t", "head")}, {"expr": _column("t", "tail")}],
        "order_by": [{"expr": _column("t", "count"), "direction": "desc"}],
        "limit": 3,
        "offset": 3,
    }
    sql = compile_query_plan_v2({"plan_version": 2, "query": base})
    assert sql.endswith("LIMIT 3 OFFSET 3")
    no_limit = {k: v for k, v in base.items() if k != "limit"}
    with pytest.raises(QueryPlanV2Error):
        parse_query_plan_v2({"plan_version": 2, "query": no_limit})
    negative = dict(base, offset=-1)
    with pytest.raises(QueryPlanV2Error):
        parse_query_plan_v2({"plan_version": 2, "query": negative})
    boolean = dict(base, offset=True)
    with pytest.raises(QueryPlanV2Error):
        parse_query_plan_v2({"plan_version": 2, "query": boolean})


def test_v2_visits_self_join_on_patient_id_only() -> None:
    def plan(join_column: str) -> dict:
        return {
            "plan_version": 2,
            "query": {
                "type": "select",
                "from": {"type": "table", "table": "visits", "alias": "v1"},
                "joins": [{
                    "source": {"type": "table", "table": "visits", "alias": "v2"},
                    "on": [{
                        "type": "binary", "operator": "eq",
                        "left": _column("v1", join_column),
                        "right": _column("v2", join_column),
                    }],
                }],
                "select": [
                    {"expr": _column("v1", "patient_id")},
                    {"expr": _column("v1", "cost"), "alias": "first_cost"},
                    {"expr": _column("v2", "cost"), "alias": "second_cost"},
                ],
                "where": {
                    "type": "binary", "operator": "gt",
                    "left": _column("v2", "cost"),
                    "right": {
                        "type": "binary", "operator": "mul",
                        "left": _column("v1", "cost"),
                        "right": {"type": "literal", "value": 1.5},
                    },
                },
                "order_by": [{"expr": _column("v1", "patient_id"), "direction": "asc"}],
            },
        }

    sql = compile_query_plan_v2(plan("patient_id"))
    assert sql.count('"visits"') == 2
    with pytest.raises(QueryPlanV2Error):
        compile_query_plan_v2(plan("department"))


def _join_plan(*on: dict) -> dict:
    return {
        "plan_version": 2,
        "query": {
            "type": "select",
            "from": {"type": "table", "table": "kg_triples", "alias": "t"},
            "joins": [
                {
                    "source": {"type": "table", "table": "kg_entities", "alias": "e"},
                    "on": list(on),
                }
            ],
            "select": [{"expr": _column("t", "tail")}],
        },
    }


_ENTITY_EQUALITY = {
    "type": "binary",
    "operator": "eq",
    "left": _column("e", "name"),
    "right": _column("t", "tail"),
}
_TYPE_FILTER = {
    "type": "binary",
    "operator": "eq",
    "left": _column("e", "type"),
    "right": {"type": "literal", "value": "Drug"},
}


def test_v2_join_on_allows_literal_refinement_beside_an_allowlisted_equality() -> None:
    sql = compile_query_plan_v2(_join_plan(_ENTITY_EQUALITY, _TYPE_FILTER))
    assert '"e"."name" = "t"."tail"' in sql
    assert '"e"."type" = \'Drug\'' in sql


def test_v2_join_on_refinement_alone_is_rejected() -> None:
    with pytest.raises(QueryPlanV2Error, match="allowlisted column equality"):
        parse_query_plan_v2(_join_plan(_TYPE_FILTER))


def test_v2_join_on_cannot_smuggle_an_unallowlisted_path_or_subquery() -> None:
    unallowlisted = {
        "type": "binary",
        "operator": "eq",
        "left": _column("e", "type"),
        "right": _column("t", "relation"),
    }
    with pytest.raises(QueryPlanV2Error, match="join path"):
        parse_query_plan_v2(_join_plan(unallowlisted))
    subquery_operand = {
        "type": "binary",
        "operator": "eq",
        "left": _column("e", "name"),
        "right": {
            "type": "scalar_subquery",
            "query": {
                "type": "select",
                "from": {"type": "table", "table": "visits", "alias": "v"},
                "select": [{"expr": _column("v", "disease")}],
            },
        },
    }
    with pytest.raises(QueryPlanV2Error, match="columns and literals"):
        parse_query_plan_v2(_join_plan(_ENTITY_EQUALITY, subquery_operand))
    with pytest.raises(QueryPlanV2Error, match="comparison between"):
        parse_query_plan_v2(
            _join_plan(
                _ENTITY_EQUALITY,
                {"type": "unary", "operator": "not", "argument": _TYPE_FILTER},
            )
        )


def test_v2_drug_head_join_path_is_allowlisted_but_exam_head_is_not() -> None:
    def plan(visit_column: str) -> dict:
        return {
            "plan_version": 2,
            "query": {
                "type": "select",
                "from": {"type": "table", "table": "visits", "alias": "v"},
                "joins": [
                    {
                        "source": {"type": "table", "table": "kg_triples", "alias": "t"},
                        "on": [
                            {
                                "type": "binary",
                                "operator": "eq",
                                "left": _column("t", "head"),
                                "right": _column("v", visit_column),
                            }
                        ],
                    }
                ],
                "select": [{"expr": _column("v", visit_column)}],
            },
        }

    assert "INNER JOIN" in compile_query_plan_v2(plan("drug_prescribed"))
    # kg_triples heads are only Disease or Drug, so an exam name has no
    # unambiguous head-side meaning and the path stays closed.
    with pytest.raises(QueryPlanV2Error, match="join path"):
        parse_query_plan_v2(plan("exam"))


def _grouped_plan(order_entry: dict) -> dict:
    return {
        "plan_version": 2,
        "query": {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "select": [
                {"expr": _column("v", "department")},
                {
                    "expr": {"type": "aggregate", "function": "count", "argument": None},
                    "alias": "visit_count",
                },
            ],
            "group_by": [_column("v", "department")],
            "order_by": [order_entry],
        },
    }


def test_v2_order_by_select_alias_compiles_and_stays_bounded() -> None:
    sql = compile_query_plan_v2(_grouped_plan({"alias": "visit_count", "direction": "desc"}))
    # The alias is authoring sugar: the compiler substitutes the expression it
    # names so the emitted SQL matches the classic repeat-the-expression form.
    assert sql.endswith("ORDER BY COUNT(*) DESC")
    with pytest.raises(QueryPlanV2Error, match="declared select alias"):
        parse_query_plan_v2(_grouped_plan({"alias": "missing", "direction": "desc"}))
    with pytest.raises(QueryPlanV2Error, match="expr or alias"):
        parse_query_plan_v2(
            _grouped_plan(
                {"alias": "visit_count", "expr": _column("v", "department"), "direction": "asc"}
            )
        )


def test_v2_window_order_by_rejects_the_alias_form() -> None:
    plan = {
        "plan_version": 2,
        "query": {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "select": [
                {"expr": _column("v", "cost"), "alias": "cost"},
                {
                    "alias": "rn",
                    "expr": {
                        "type": "window",
                        "function": "row_number",
                        "argument": None,
                        "order_by": [{"alias": "cost", "direction": "desc"}],
                    },
                },
            ],
        },
    }
    with pytest.raises(QueryPlanV2Error, match="not allowed in this ORDER BY"):
        parse_query_plan_v2(plan)


def _union_plan(**modifiers: object) -> dict:
    branch = lambda table, column, alias: {
        "type": "select",
        "from": {"type": "table", "table": table, "alias": alias},
        "select": [{"expr": _column(alias, column), "alias": "name"}],
        "distinct": True,
    }
    query = {
        "type": "set_op",
        "operator": "union",
        "left": branch("visits", "department", "v"),
        "right": branch("kg_entities", "name", "e"),
    }
    query.update(modifiers)
    return {"plan_version": 2, "query": query}


def test_v2_set_operation_carries_order_and_limit_without_a_wrapper() -> None:
    sql = compile_query_plan_v2(
        _union_plan(order_by=[{"alias": "name", "direction": "asc"}], limit=5)
    )
    assert sql.endswith('ORDER BY "name" ASC LIMIT 5')
    assert " UNION " in sql


def test_v2_set_operation_order_by_is_restricted_to_branch_aliases() -> None:
    with pytest.raises(QueryPlanV2Error, match="declared select alias"):
        parse_query_plan_v2(_union_plan(order_by=[{"alias": "cost", "direction": "asc"}]))
    with pytest.raises(QueryPlanV2Error, match="branch output alias"):
        parse_query_plan_v2(
            _union_plan(order_by=[{"expr": _column("v", "department"), "direction": "asc"}])
        )
    with pytest.raises(QueryPlanV2Error, match="offset requires limit"):
        parse_query_plan_v2(_union_plan(offset=2))


def _date_plan(*args: object) -> dict:
    return {
        "plan_version": 2,
        "query": {
            "type": "select",
            "from": {"type": "table", "table": "visits", "alias": "v"},
            "select": [{"expr": {"type": "aggregate", "function": "count", "argument": None}}],
            "where": {
                "type": "binary",
                "operator": "gte",
                "left": _column("v", "visit_date"),
                "right": {"type": "function", "function": "date_sub", "args": list(args)},
            },
        },
    }


def test_v2_date_offset_emits_a_compiler_built_modifier() -> None:
    plan = _date_plan(
        _column("v", "visit_date"), {"type": "literal", "value": 6}, {"type": "literal", "value": "month"}
    )
    sql = compile_query_plan_v2(plan)
    assert "DATE(\"v\".\"visit_date\", '-6 months')" in sql
    # The unit is a compiler control; the amount still has to come from the question.
    validate_plan_v2_grounding(plan, "统计近6个月内的就诊次数")
    with pytest.raises(QueryPlanV2Error, match="grounded"):
        validate_plan_v2_grounding(plan, "统计近一年内的就诊次数")


def test_v2_date_offset_rejects_free_form_modifiers_and_bad_units() -> None:
    column = _column("v", "visit_date")
    with pytest.raises(QueryPlanV2Error, match="expression, amount, unit"):
        parse_query_plan_v2(_date_plan(column, {"type": "literal", "value": "-6 months"}))
    with pytest.raises(QueryPlanV2Error, match="args\\[2\\]"):
        parse_query_plan_v2(
            _date_plan(column, {"type": "literal", "value": 6}, {"type": "literal", "value": "fortnight"})
        )
    with pytest.raises(QueryPlanV2Error, match="args\\[2\\]"):
        parse_query_plan_v2(
            _date_plan(column, {"type": "literal", "value": 6}, {"type": "literal", "value": "-6 months"})
        )
    with pytest.raises(QueryPlanV2Error, match="args\\[1\\]"):
        parse_query_plan_v2(
            _date_plan(column, {"type": "literal", "value": -6}, {"type": "literal", "value": "month"})
        )
    with pytest.raises(QueryPlanV2Error, match="args\\[1\\]"):
        parse_query_plan_v2(
            _date_plan(column, _column("v", "los_days"), {"type": "literal", "value": "day"})
        )


def test_v2_derived_table_columns_are_emitted_under_their_declared_names() -> None:
    plan = {
        "plan_version": 2,
        "query": {
            "type": "select",
            "from": {
                "type": "subquery",
                "alias": "d",
                "query": {
                    "type": "select",
                    "from": {"type": "table", "table": "visits", "alias": "v"},
                    "select": [
                        {"expr": _column("v", "patient_id")},
                        {"expr": {"type": "aggregate", "function": "count", "argument": None}},
                    ],
                    "group_by": [_column("v", "patient_id")],
                },
            },
            "select": [{"expr": _column("d", "patient_id")}, {"expr": _column("d", "_expr_1")}],
        },
    }
    sql = compile_query_plan_v2(plan)
    assert 'COUNT(*) AS "_expr_1"' in sql
    assert '"d"."_expr_1"' in sql
