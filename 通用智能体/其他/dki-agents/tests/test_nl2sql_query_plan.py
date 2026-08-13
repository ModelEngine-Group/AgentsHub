from __future__ import annotations

import copy
import json
import re

import pytest

import dki.analysis.nl2sql_query_plan as query_plan_module
from dki.analysis.nl2sql_protocol import validate_gold_sql
from dki.analysis.nl2sql_query_plan import (
    JOIN_PATHS,
    MAX_LIMIT,
    QueryPlanCompileError,
    QueryPlanGroundingError,
    QueryPlanJSONError,
    QueryPlanValidationError,
    canonical_plan_json,
    canonical_plan_sha256,
    compile_query_plan,
    parse_query_plan,
    query_plan_schema,
    validate_plan_grounding,
)


def _column(alias: str, column: str) -> dict[str, object]:
    return {"type": "column", "table_alias": alias, "column": column}


def _literal(value_type: str, value: object) -> dict[str, object]:
    return {"type": "literal", "value_type": value_type, "value": value}


def _aggregate(
    function: str,
    argument: dict[str, object] | None,
    *,
    distinct: bool = False,
) -> dict[str, object]:
    return {
        "type": "aggregate",
        "function": function,
        "argument": argument,
        "distinct": distinct,
    }


def _comparison(
    operator: str,
    left: dict[str, object],
    right: dict[str, object],
) -> dict[str, object]:
    return {
        "type": "comparison",
        "operator": operator,
        "left": left,
        "right": right,
    }


def _base_plan() -> dict[str, object]:
    return {
        "plan_version": 1,
        "from": {"table": "visits", "alias": "v"},
        "select": [{"expr": _column("v", "visit_id")}],
    }


def _raw(plan: dict[str, object]) -> str:
    return json.dumps(plan, ensure_ascii=False, separators=(",", ":"))


def test_minimal_plan_is_normalised_with_stable_defaults() -> None:
    parsed = parse_query_plan(_raw(_base_plan()))

    assert parsed["plan_version"] == 1
    assert parsed["joins"] == []
    assert parsed["distinct"] is False
    assert parsed["where"] is None
    assert parsed["group_by"] == []
    assert parsed["having"] is None
    assert parsed["order_by"] == []
    assert parsed["limit"] is None


def test_parser_accepts_utf8_bytes_but_not_a_bom() -> None:
    raw = _raw(_base_plan()).encode()
    assert parse_query_plan(raw)["from"]["table"] == "visits"
    with pytest.raises(QueryPlanJSONError, match="BOM"):
        parse_query_plan(b"\xef\xbb\xbf" + raw)


@pytest.mark.parametrize(
    "raw",
    [
        "SELECT * FROM visits",
        "```json\n{}\n```",
        "null",
        "[]",
        '{"plan_version":1,"plan_version":1}',
        '{"plan_version":1,"from":NaN}',
        '{"plan_version":1,"from":Infinity}',
        "{} trailing text",
    ],
)
def test_parser_rejects_non_strict_json_and_sql_text(raw: str) -> None:
    with pytest.raises(QueryPlanJSONError):
        parse_query_plan(raw)


def test_parser_rejects_wrong_input_type_and_oversized_raw_json() -> None:
    with pytest.raises(QueryPlanJSONError):
        parse_query_plan(_base_plan())  # type: ignore[arg-type]
    with pytest.raises(QueryPlanJSONError, match="exceeds"):
        parse_query_plan(" " * (64 * 1024 + 1))


@pytest.mark.parametrize("version", [0, 2, "1", "v1", True, None])
def test_version_is_exact_integer_one(version: object) -> None:
    plan = _base_plan()
    plan["plan_version"] = version
    with pytest.raises(QueryPlanValidationError, match="plan_version"):
        parse_query_plan(_raw(plan))


def test_unknown_top_level_or_nested_fields_are_rejected() -> None:
    top = _base_plan()
    top["raw_sql"] = "SELECT * FROM visits"
    with pytest.raises(QueryPlanValidationError, match="unknown field"):
        parse_query_plan(_raw(top))

    nested = _base_plan()
    nested["select"][0]["sql"] = "visit_id"  # type: ignore[index]
    with pytest.raises(QueryPlanValidationError, match="unknown field"):
        parse_query_plan(_raw(nested))


def test_schema_is_versioned_closed_and_returned_as_a_copy() -> None:
    first = query_plan_schema()
    second = query_plan_schema()

    assert first["properties"]["plan_version"] == {"const": 1}
    assert first["additionalProperties"] is False
    assert set(first["properties"]["from"]["$ref"].split("/")) >= {"$defs"}
    assert set(first["$defs"]["join"]["properties"]["path"]["enum"]) == set(
        JOIN_PATHS
    )
    first["title"] = "mutated"
    assert second["title"] == "QueryPlan v1"


def test_canonical_json_and_sha_ignore_key_order_and_explicit_defaults() -> None:
    minimal = _base_plan()
    explicit = {
        "limit": None,
        "order_by": [],
        "having": None,
        "group_by": [],
        "where": None,
        "distinct": False,
        "select": [{"expr": _column("v", "visit_id")}],
        "joins": [],
        "from": {"alias": "v", "table": "visits"},
        "plan_version": 1,
    }

    assert canonical_plan_json(minimal) == canonical_plan_json(explicit)
    first = canonical_plan_sha256(minimal)
    assert first == canonical_plan_sha256(_raw(explicit))
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_single_table_compilation_is_ast_built_and_finally_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _base_plan()
    calls: list[str] = []
    monkeypatch.setattr(
        query_plan_module,
        "validate_gold_sql",
        lambda sql: calls.append(sql) or "SELECT SKELETON",
    )

    sql = compile_query_plan(plan)

    assert calls == [sql]
    assert sql.startswith('SELECT "v"."visit_id" FROM "visits" AS "v"')
    assert ";" not in sql


def test_complex_select_where_group_having_order_limit_and_distinct() -> None:
    count_all = _aggregate("count", None)
    plan: dict[str, object] = {
        "plan_version": 1,
        "from": {"table": "visits", "alias": "v"},
        "select": [
            {"expr": _column("v", "disease")},
            {"expr": count_all, "alias": "visit_count"},
        ],
        "distinct": True,
        "where": {
            "type": "boolean",
            "operator": "and",
            "operands": [
                _comparison(
                    "eq", _column("v", "department"), _literal("string", "呼吸科")
                ),
                _comparison(
                    "gte", _column("v", "cost"), _literal("number", 800)
                ),
            ],
        },
        "group_by": [_column("v", "disease")],
        "having": _comparison("gte", count_all, _literal("integer", 2)),
        "order_by": [{"expr": count_all, "direction": "desc"}],
        "limit": 5,
    }

    sql = compile_query_plan(plan)

    assert "SELECT DISTINCT" in sql
    assert '"v"."department" = \'呼吸科\'' in sql
    assert '"v"."cost" >= 800' in sql
    assert 'GROUP BY "v"."disease"' in sql
    assert "HAVING COUNT(*) >= 2" in sql
    assert "ORDER BY COUNT(*) DESC" in sql
    assert sql.endswith("LIMIT 5")
    validate_gold_sql(sql)


def test_all_predicate_shapes_compile_without_raw_fragments() -> None:
    plan = _base_plan()
    plan["where"] = {
        "type": "boolean",
        "operator": "or",
        "operands": [
            {
                "type": "in",
                "expr": _column("v", "department"),
                "values": [_literal("string", "内科"), _literal("string", "儿科")],
            },
            {
                "type": "between",
                "expr": _column("v", "cost"),
                "lower": _literal("number", 300),
                "upper": _literal("number", 900),
                "negated": True,
            },
            {"type": "is_null", "expr": _column("v", "exam"), "negated": True},
            _comparison(
                "not_like",
                _column("v", "disease"),
                _literal("string", "%感冒%"),
            ),
        ],
    }

    sql = compile_query_plan(plan)

    assert " IN (" in sql
    assert "NOT " in sql
    assert re.search(r"BETWEEN 300(?:\.0)? AND 900(?:\.0)?", sql)
    assert " IS NULL" in sql
    assert " LIKE '%感冒%'" in sql


def test_literal_quotes_are_data_not_sql_syntax() -> None:
    plan = _base_plan()
    plan["where"] = _comparison(
        "eq",
        _column("v", "department"),
        _literal("string", "x' OR 1=1 --"),
    )

    sql = compile_query_plan(plan)

    assert "x'' OR 1=1 --" in sql
    validate_gold_sql(sql)


def test_patients_visits_join_is_fixed_and_can_be_entered_from_either_side() -> None:
    forward: dict[str, object] = {
        "plan_version": 1,
        "from": {"table": "patients", "alias": "p"},
        "joins": [
            {"path": "patients_visits", "left_alias": "p", "right_alias": "v"}
        ],
        "select": [
            {"expr": _column("p", "name")},
            {"expr": _column("v", "cost")},
        ],
    }
    reverse = copy.deepcopy(forward)
    reverse["from"] = {"table": "visits", "alias": "v"}

    forward_sql = compile_query_plan(forward)
    reverse_sql = compile_query_plan(reverse)

    for sql in (forward_sql, reverse_sql):
        assert "INNER JOIN" in sql
        assert '"p"."patient_id" = "v"."patient_id"' in sql
        validate_gold_sql(sql)


def test_graph_head_and_tail_paths_can_share_the_triple_alias() -> None:
    plan: dict[str, object] = {
        "plan_version": 1,
        "from": {"table": "kg_triples", "alias": "t"},
        "joins": [
            {
                "path": "kg_entities_triples_head",
                "left_alias": "h",
                "right_alias": "t",
            },
            {
                "path": "kg_entities_triples_tail",
                "left_alias": "e",
                "right_alias": "t",
            },
        ],
        "select": [
            {"expr": _column("h", "name"), "alias": "head_name"},
            {"expr": _column("e", "name"), "alias": "tail_name"},
        ],
    }

    sql = compile_query_plan(plan)

    assert sql.count("INNER JOIN") == 2
    assert '"h"."type" = "t"."head_type"' in sql
    assert '"e"."type" = "t"."tail_type"' in sql


def test_fixed_relation_constant_is_compiled_but_needs_no_question_grounding() -> None:
    plan: dict[str, object] = {
        "plan_version": 1,
        "from": {"table": "visits", "alias": "v"},
        "joins": [
            {
                "path": "visits_belongs_to_department_triples",
                "left_alias": "v",
                "right_alias": "t",
            }
        ],
        "select": [{"expr": _column("v", "visit_id")}],
    }

    validate_plan_grounding(plan, "核对实际疾病和科室是否与图谱一致")
    sql = compile_query_plan(plan)
    assert '"t"."relation" = \'BELONGS_TO_DEPT\'' in sql


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda p: p["from"].update(table="secrets"), "from.table"),
        (lambda p: p["select"][0]["expr"].update(column="password"), "unknown column"),
        (
            lambda p: p.update(
                joins=[
                    {"path": "made_up", "left_alias": "v", "right_alias": "x"}
                ]
            ),
            "path is unknown",
        ),
        (
            lambda p: p.update(
                where=_comparison(
                    "regexp", _column("v", "disease"), _literal("string", "x")
                )
            ),
            "operator is unknown",
        ),
        (
            lambda p: p["select"][0].update(
                expr=_aggregate("median", _column("v", "cost"))
            ),
            "function is unknown",
        ),
    ],
)
def test_unknown_tables_columns_join_paths_operators_and_functions_fail_closed(
    mutator,
    message: str,
) -> None:
    plan = _base_plan()
    mutator(plan)
    with pytest.raises(QueryPlanValidationError, match=message):
        compile_query_plan(plan)


def test_join_must_connect_and_introduce_exactly_one_alias() -> None:
    disconnected = _base_plan()
    disconnected["joins"] = [
        {"path": "patients_visits", "left_alias": "p", "right_alias": "x"}
    ]
    with pytest.raises(QueryPlanValidationError, match="disconnected"):
        compile_query_plan(disconnected)

    cycle: dict[str, object] = {
        "plan_version": 1,
        "from": {"table": "kg_triples", "alias": "t"},
        "joins": [
            {
                "path": "kg_entities_triples_head",
                "left_alias": "e",
                "right_alias": "t",
            },
            {
                "path": "kg_entities_triples_tail",
                "left_alias": "e",
                "right_alias": "t",
            },
        ],
        "select": [{"expr": _column("e", "name")}],
    }
    with pytest.raises(QueryPlanValidationError, match="cycle/redundant"):
        compile_query_plan(cycle)


def test_duplicate_table_and_select_aliases_are_rejected_case_insensitively() -> None:
    duplicate_table = {
        "plan_version": 1,
        "from": {"table": "patients", "alias": "p"},
        "joins": [
            {"path": "patients_visits", "left_alias": "p", "right_alias": "P"}
        ],
        "select": [{"expr": _column("p", "name")}],
    }
    with pytest.raises(QueryPlanValidationError, match="same alias|duplicate alias"):
        compile_query_plan(duplicate_table)

    duplicate_select = _base_plan()
    duplicate_select["select"] = [
        {"expr": _column("v", "visit_id"), "alias": "ID"},
        {"expr": _column("v", "patient_id"), "alias": "id"},
    ]
    with pytest.raises(QueryPlanValidationError, match="duplicate SELECT alias"):
        compile_query_plan(duplicate_select)


def test_aggregate_type_and_phase_rules_fail_closed() -> None:
    sum_text = _base_plan()
    sum_text["select"] = [
        {"expr": _aggregate("sum", _column("v", "disease")), "alias": "bad"}
    ]
    with pytest.raises(QueryPlanValidationError, match="sum requires"):
        compile_query_plan(sum_text)

    aggregate_where = _base_plan()
    aggregate_where["where"] = _comparison(
        "gt", _aggregate("count", None), _literal("integer", 1)
    )
    with pytest.raises(QueryPlanValidationError, match="WHERE"):
        compile_query_plan(aggregate_where)

    bare_column = _base_plan()
    bare_column["select"] = [
        {"expr": _column("v", "disease")},
        {"expr": _aggregate("count", None), "alias": "n"},
    ]
    with pytest.raises(QueryPlanValidationError, match="GROUP BY"):
        compile_query_plan(bare_column)


def test_invalid_literal_shapes_and_type_mismatches_are_rejected() -> None:
    cases = [
        _literal("integer", True),
        _literal("integer", 1.5),
        _literal("number", float("nan")),
        _literal("date", "2024-02-30"),
        _literal("date", "20240229"),
        _literal("string", "line\nbreak"),
        _literal("string", "\ud800"),
        _literal("null", "null"),
        _literal("unknown", "x"),
    ]
    for literal in cases:
        plan = _base_plan()
        plan["where"] = _comparison("eq", _column("v", "cost"), literal)
        with pytest.raises(QueryPlanValidationError):
            compile_query_plan(plan)

    mismatch = _base_plan()
    mismatch["where"] = _comparison(
        "eq", _column("v", "cost"), _literal("string", "expensive")
    )
    with pytest.raises(QueryPlanValidationError, match="incompatible"):
        compile_query_plan(mismatch)


def test_null_must_use_is_null_and_like_requires_string_literal() -> None:
    null_eq = _base_plan()
    null_eq["where"] = _comparison(
        "eq", _column("v", "exam"), _literal("null", None)
    )
    with pytest.raises(QueryPlanValidationError, match="is_null"):
        compile_query_plan(null_eq)

    numeric_like = _base_plan()
    numeric_like["where"] = _comparison(
        "like", _column("v", "exam"), _literal("integer", 3)
    )
    with pytest.raises(QueryPlanValidationError, match="string literal"):
        compile_query_plan(numeric_like)


def test_distinct_count_star_duplicate_group_and_having_rules() -> None:
    distinct_star = _base_plan()
    distinct_star["select"] = [
        {"expr": _aggregate("count", None, distinct=True), "alias": "n"}
    ]
    with pytest.raises(QueryPlanValidationError, match=r"count\(\*\)"):
        compile_query_plan(distinct_star)

    duplicate_group = _base_plan()
    duplicate_group["group_by"] = [
        _column("v", "disease"),
        _column("v", "disease"),
    ]
    with pytest.raises(QueryPlanValidationError, match="duplicate GROUP BY"):
        compile_query_plan(duplicate_group)

    invalid_having = _base_plan()
    invalid_having["having"] = _comparison(
        "eq", _column("v", "department"), _literal("string", "儿科")
    )
    with pytest.raises(QueryPlanValidationError, match="HAVING"):
        compile_query_plan(invalid_having)


def test_limits_list_sizes_and_depth_are_bounded() -> None:
    plan = _base_plan()
    plan["limit"] = MAX_LIMIT + 1
    with pytest.raises(QueryPlanValidationError, match="limit"):
        compile_query_plan(plan)

    too_many_selects = _base_plan()
    too_many_selects["select"] = [
        {"expr": _column("v", "visit_id")} for _ in range(33)
    ]
    with pytest.raises(QueryPlanValidationError, match="select"):
        compile_query_plan(too_many_selects)

    leaf = _comparison(
        "eq", _column("v", "visit_id"), _literal("string", "v1")
    )
    deeply_nested = leaf
    for _ in range(20):
        deeply_nested = {
            "type": "boolean",
            "operator": "and",
            "operands": [deeply_nested, copy.deepcopy(leaf)],
        }
    deep_plan = _base_plan()
    deep_plan["where"] = deeply_nested
    with pytest.raises(QueryPlanValidationError, match="depth|node count"):
        compile_query_plan(deep_plan)


def test_grounding_accepts_normalised_string_number_date_percent_and_chinese_limit() -> None:
    plan = _base_plan()
    plan["where"] = {
        "type": "boolean",
        "operator": "and",
        "operands": [
            _comparison(
                "eq", _column("v", "department"), _literal("string", "呼吸科")
            ),
            _comparison(
                "gte", _column("v", "cost"), _literal("number", 800.0)
            ),
            _comparison(
                "gte",
                _column("v", "cost"),
                _literal("number", 0.8),
            ),
            _comparison(
                "gte",
                _column("v", "visit_date"),
                _literal("date", "2024-03-01"),
            ),
        ],
    }
    plan["limit"] = 5

    validate_plan_grounding(
        plan,
        "查询呼吸科费用至少800元、比例80%，日期从2024年3月1日起，取前五条。",
    )


def test_like_wildcards_do_not_need_to_appear_in_the_question() -> None:
    plan = _base_plan()
    plan["where"] = _comparison(
        "like", _column("v", "disease"), _literal("string", "%感冒%")
    )
    validate_plan_grounding(plan, "查询疾病名称包含感冒的就诊")


def test_database_domain_membership_never_bypasses_question_grounding() -> None:
    plan = _base_plan()
    plan["where"] = _comparison(
        "eq", _column("v", "department"), _literal("string", "虚构科")
    )

    with pytest.raises(QueryPlanGroundingError, match="虚构科"):
        validate_plan_grounding(
            plan,
            "查询某个科室的就诊",
            allowed_domain_values=("虚构科",),
        )


def test_hallucinated_limit_is_rejected_even_without_predicate_literals() -> None:
    plan = _base_plan()
    plan["limit"] = 7

    with pytest.raises(QueryPlanGroundingError, match="limit"):
        validate_plan_grounding(plan, "返回一些就诊记录")
    validate_plan_grounding(plan, "返回前7条就诊记录")


def test_chinese_digit_inside_an_ordinary_word_is_not_numeric_grounding() -> None:
    plan = _base_plan()
    plan["limit"] = 1

    with pytest.raises(QueryPlanGroundingError, match="limit"):
        validate_plan_grounding(plan, "核对疾病和科室是否一致")


def test_all_string_and_numeric_literals_must_be_grounded_not_just_one() -> None:
    plan = _base_plan()
    plan["where"] = {
        "type": "boolean",
        "operator": "and",
        "operands": [
            _comparison(
                "eq", _column("v", "department"), _literal("string", "儿科")
            ),
            _comparison(
                "gte", _column("v", "cost"), _literal("number", 999.0)
            ),
        ],
    }

    with pytest.raises(QueryPlanGroundingError, match="999"):
        validate_plan_grounding(plan, "查询儿科的高费用就诊")


def test_mutating_a_parsed_plan_cannot_bypass_compile_revalidation() -> None:
    parsed = parse_query_plan(_raw(_base_plan()))
    parsed["select"][0]["expr"]["column"] = "secret"

    with pytest.raises(QueryPlanValidationError, match="unknown column"):
        compile_query_plan(parsed)


def test_sqlglot_render_or_shared_gate_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        query_plan_module,
        "validate_gold_sql",
        lambda _sql: (_ for _ in ()).throw(ValueError("gate rejected")),
    )

    with pytest.raises(QueryPlanCompileError, match="gate rejected"):
        compile_query_plan(_base_plan())
