"""Bounded QueryPlan v2 compiler.

QueryPlan v2 is a small relational-algebra language, not a SQL template.  It
accepts only structured nodes and lowers them to SQLite SQL after validating
all identifiers, literals, resource limits and read-only semantics.  There is
deliberately no ``sql``, ``raw_sql`` or free-form expression escape hatch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from dki.analysis.nl2sql_protocol import ALLOWED_TABLES, validate_gold_sql
from dki.analysis.nl2sql_query_plan import TABLE_COLUMNS


PLAN_VERSION = 2
MAX_DEPTH = 16
MAX_NODES = 512
MAX_SELECT = 32
MAX_JOINS = 8
MAX_CTES = 4
MAX_SET_OPS = 4
MAX_SUBQUERIES = 4
MAX_LIMIT = 500
MAX_LITERAL = 512
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,31}\Z")
ALLOWED_AGGREGATES = {"count", "sum", "avg", "min", "max"}
ALLOWED_FUNCTIONS = {
    "round",
    "coalesce",
    "lower",
    "upper",
    "strftime",
    "julianday",
    "abs",
    "substr",
    "date_add",
    "date_sub",
}
# ``date_add``/``date_sub`` are the only way to do calendar arithmetic.  They
# take exactly (date expression, non-negative integer literal, unit literal) and
# the compiler builds the SQLite modifier itself, so no model-authored modifier
# string ever reaches SQL.
ALLOWED_DATE_UNITS = {"day", "month", "year"}
MAX_DATE_OFFSET = 3660
DATE_OFFSET_FUNCTIONS = {"date_add": "+", "date_sub": "-"}
# Arguments that are compiler controls rather than semantic predicates, keyed by
# function name -> positional indices that must not require question grounding.
STRUCTURAL_ARGUMENT_POSITIONS = {
    "strftime": {0},
    "round": {1},
    "substr": {1, 2},
    # The unit word is an enum the compiler renders; the amount stays grounded.
    "date_add": {2},
    "date_sub": {2},
}
ALLOWED_WINDOWS = {
    "row_number",
    "rank",
    "dense_rank",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "lead",
    "lag",
    "first_value",
    "last_value",
    "ntile",
}
ALLOWED_BINOPS = {
    "eq", "neq", "gt", "gte", "lt", "lte", "and", "or",
    "add", "sub", "mul", "div", "like", "is", "is_not",
}
ALLOWED_WINDOW_FRAMES = {
    "unbounded_preceding",
    "current_row",
    "unbounded_following",
}
# Conjuncts allowed inside a JOIN ... ON list.  An ``eq`` between two columns is
# a join path and is checked against the allowlist below; every other comparison
# is a refinement (``AND e.type = 'Drug'``, ``AND v2.visit_id > v1.visit_id``)
# that narrows an already-allowlisted join instead of creating a new one.
ALLOWED_JOIN_REFINEMENT_OPS = {
    "eq", "neq", "gt", "gte", "lt", "lte", "like", "is", "is_not",
}
ALLOWED_DIRECT_JOIN_PATHS = {
    frozenset({("patients", "patient_id"), ("visits", "patient_id")}),
    # visits self-join pairs rows of the same patient (前后两次就诊 comparisons);
    # the equality key is patient_id, inequality refinements belong in ON or WHERE.
    frozenset({("visits", "patient_id")}),
    # Every visit attribute that names a knowledge-graph entity may join it.
    # Each pair below resolves to exactly one entity type in the sealed DB
    # (disease->Disease, drug_prescribed->Drug, exam->Exam, department->Department).
    frozenset({("visits", "disease"), ("kg_entities", "name")}),
    frozenset({("visits", "drug_prescribed"), ("kg_entities", "name")}),
    frozenset({("visits", "exam"), ("kg_entities", "name")}),
    frozenset({("visits", "department"), ("kg_entities", "name")}),
    frozenset({("visits", "disease"), ("kg_triples", "head")}),
    # Complication/comorbidity triples carry the related disease on the tail
    # side; every tail matching a visit disease is head_type-consistent and
    # resolves to tail_type='Disease' only, so the pair stays unambiguous.
    frozenset({("visits", "disease"), ("kg_triples", "tail")}),
    frozenset({("visits", "department"), ("kg_triples", "tail")}),
    frozenset({("visits", "drug_prescribed"), ("kg_triples", "tail")}),
    # kg_triples heads are Disease or Drug only, and no drug name is also a
    # disease name in the sealed DB, so a prescribed drug matched against a
    # head resolves to head_type='Drug' alone (用药禁忌 CONTRAINDICATED_FOR).
    frozenset({("visits", "drug_prescribed"), ("kg_triples", "head")}),
    frozenset({("visits", "exam"), ("kg_triples", "tail")}),
    frozenset({("kg_entities", "name"), ("kg_triples", "head")}),
    frozenset({("kg_entities", "name"), ("kg_triples", "tail")}),
}


class QueryPlanV2Error(ValueError):
    """Base class for fail-closed v2 errors."""


def _reject_unknown(obj: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(obj) - allowed
    if unknown:
        raise QueryPlanV2Error(f"{path} contains unsupported fields: {sorted(unknown)}")


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise QueryPlanV2Error(f"{path} must be a safe identifier")
    return value


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _literal(value: Any, path: str) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > 10**12:
            raise QueryPlanV2Error(f"{path} integer is outside safe range")
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > 10**12:
            raise QueryPlanV2Error(f"{path} number is not finite or is too large")
        return format(value, ".15g")
    if isinstance(value, str):
        if not 1 <= len(value) <= MAX_LITERAL:
            raise QueryPlanV2Error(f"{path} string literal length is invalid")
        return "'" + value.replace("'", "''") + "'"
    raise QueryPlanV2Error(f"{path} has unsupported literal type")


def _walk(value: Any, *, depth: int = 0, counter: list[int] | None = None) -> None:
    counter = counter or [0]
    if depth > MAX_DEPTH:
        raise QueryPlanV2Error(f"plan depth exceeds {MAX_DEPTH}")
    counter[0] += 1
    if counter[0] > MAX_NODES:
        raise QueryPlanV2Error(f"plan nodes exceed {MAX_NODES}")
    if isinstance(value, Mapping):
        for child in value.values():
            _walk(child, depth=depth + 1, counter=counter)
    elif isinstance(value, list):
        for child in value:
            _walk(child, depth=depth + 1, counter=counter)


def _source(source: Any, path: str) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise QueryPlanV2Error(f"{path} must be an object")
    typ = source.get("type")
    _reject_unknown(source, {"type", "table", "alias", "query", "name"}, path)
    if typ == "table":
        table = source.get("table")
        if table not in ALLOWED_TABLES:
            raise QueryPlanV2Error(f"{path}.table is not allowlisted")
        alias = _identifier(source.get("alias", table), f"{path}.alias")
        return {"type": "table", "table": table, "alias": alias}
    if typ == "subquery":
        alias = _identifier(source.get("alias"), f"{path}.alias")
        return {"type": "subquery", "alias": alias, "query": _query(source.get("query"), f"{path}.query")}
    if typ == "cte":
        name = _identifier(source.get("name"), f"{path}.name")
        alias = _identifier(source.get("alias", name), f"{path}.alias")
        return {"type": "cte", "name": name, "alias": alias}
    raise QueryPlanV2Error(f"{path}.type must be table or subquery")


def _expr(value: Any, path: str = "$.expr") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QueryPlanV2Error(f"{path} must be an object")
    typ = value.get("type")
    if typ == "column":
        _reject_unknown(value, {"type", "table", "column"}, path)
        return {"type": typ, "table": _identifier(value.get("table"), f"{path}.table"),
                "column": _identifier(value.get("column"), f"{path}.column")}
    if typ == "literal":
        _reject_unknown(value, {"type", "value"}, path)
        _literal(value.get("value"), f"{path}.value")
        return {"type": typ, "value": value.get("value")}
    if typ == "aggregate":
        _reject_unknown(value, {"type", "function", "argument", "distinct"}, path)
        function = value.get("function")
        if function not in ALLOWED_AGGREGATES:
            raise QueryPlanV2Error(f"{path}.function is not allowlisted")
        argument = None if value.get("argument") is None else _expr(value["argument"], f"{path}.argument")
        return {"type": typ, "function": function, "argument": argument, "distinct": bool(value.get("distinct", False))}
    if typ == "function":
        _reject_unknown(value, {"type", "function", "args"}, path)
        function = value.get("function")
        if function not in ALLOWED_FUNCTIONS:
            raise QueryPlanV2Error(f"{path}.function is not allowlisted")
        args = value.get("args")
        if not isinstance(args, list) or not 1 <= len(args) <= 8:
            raise QueryPlanV2Error(f"{path}.args has invalid size")
        parsed_args = [_expr(item, f"{path}.args[{i}]") for i, item in enumerate(args)]
        if function in DATE_OFFSET_FUNCTIONS:
            _validate_date_offset_args(parsed_args, path)
        return {"type": typ, "function": function, "args": parsed_args}
    if typ == "binary":
        _reject_unknown(value, {"type", "operator", "left", "right"}, path)
        operator = value.get("operator")
        if operator not in ALLOWED_BINOPS:
            raise QueryPlanV2Error(f"{path}.operator is not allowlisted")
        return {"type": typ, "operator": operator,
                "left": _expr(value.get("left"), f"{path}.left"),
                "right": _expr(value.get("right"), f"{path}.right")}
    if typ == "unary":
        _reject_unknown(value, {"type", "operator", "argument"}, path)
        if value.get("operator") != "not":
            raise QueryPlanV2Error(f"{path}.operator must be not")
        return {"type": typ, "operator": "not", "argument": _expr(value.get("argument"), f"{path}.argument")}
    if typ == "in":
        _reject_unknown(value, {"type", "value", "values", "query"}, path)
        item = _expr(value.get("value"), f"{path}.value")
        values = value.get("values")
        if values is not None:
            if not isinstance(values, list) or not values or len(values) > 64:
                raise QueryPlanV2Error(f"{path}.values has invalid size")
            return {"type": typ, "value": item, "values": [_expr(v, f"{path}.values[{i}]") for i, v in enumerate(values)]}
        return {"type": typ, "value": item, "query": _query(value.get("query"), f"{path}.query")}
    if typ in {"exists", "scalar_subquery"}:
        _reject_unknown(value, {"type", "query"}, path)
        return {"type": typ, "query": _query(value.get("query"), f"{path}.query")}
    if typ == "case":
        _reject_unknown(value, {"type", "when", "else"}, path)
        when = value.get("when")
        if not isinstance(when, list) or not when or len(when) > 16:
            raise QueryPlanV2Error(f"{path}.when has invalid size")
        pairs = []
        for i, pair in enumerate(when):
            if not isinstance(pair, Mapping):
                raise QueryPlanV2Error(f"{path}.when[{i}] must be an object")
            _reject_unknown(pair, {"when", "then"}, f"{path}.when[{i}]")
            pairs.append({"when": _expr(pair.get("when"), f"{path}.when[{i}].when"),
                          "then": _expr(pair.get("then"), f"{path}.when[{i}].then")})
        return {"type": typ, "when": pairs, "else": _expr(value.get("else"), f"{path}.else")}
    if typ == "window":
        _reject_unknown(
            value,
            {"type", "function", "argument", "partition_by", "order_by", "frame"},
            path,
        )
        function = value.get("function")
        if function not in ALLOWED_WINDOWS:
            raise QueryPlanV2Error(f"{path}.function is not allowlisted")
        argument = None if value.get("argument") is None else _expr(value["argument"], f"{path}.argument")
        if function in {"row_number", "rank", "dense_rank"} and argument is not None:
            raise QueryPlanV2Error(f"{path}.argument must be null for ranking windows")
        if function in {
            "sum",
            "avg",
            "min",
            "max",
            "lead",
            "lag",
            "first_value",
            "last_value",
            "ntile",
        } and argument is None:
            raise QueryPlanV2Error(f"{path}.argument is required for {function}")
        if function == "ntile" and not (
            argument is not None
            and argument.get("type") == "literal"
            and isinstance(argument.get("value"), int)
            and not isinstance(argument.get("value"), bool)
            and 1 <= argument["value"] <= 100
        ):
            raise QueryPlanV2Error(
                f"{path}.argument must be a literal integer in [1, 100] for ntile"
            )
        partition = value.get("partition_by", [])
        order = value.get("order_by", [])
        if not isinstance(partition, list) or len(partition) > 16 or not isinstance(order, list) or len(order) > 16:
            raise QueryPlanV2Error(f"{path} window clauses are too large")
        frame = value.get("frame")
        parsed_frame = None
        if frame is not None:
            if not isinstance(frame, Mapping):
                raise QueryPlanV2Error(f"{path}.frame must be an object")
            _reject_unknown(frame, {"unit", "start", "end"}, f"{path}.frame")
            if frame.get("unit") != "rows":
                raise QueryPlanV2Error(f"{path}.frame.unit must be rows")
            if (
                frame.get("start") not in ALLOWED_WINDOW_FRAMES
                or frame.get("end") not in ALLOWED_WINDOW_FRAMES
            ):
                raise QueryPlanV2Error(f"{path}.frame boundary is invalid")
            parsed_frame = {
                "unit": "rows",
                "start": frame["start"],
                "end": frame["end"],
            }
        return {"type": typ, "function": function, "argument": argument,
                "partition_by": [_expr(v, f"{path}.partition_by[{i}]") for i, v in enumerate(partition)],
                "order_by": [_order(v, f"{path}.order_by[{i}]") for i, v in enumerate(order)],
                "frame": parsed_frame}
    raise QueryPlanV2Error(f"{path}.type is unsupported")


def _validate_date_offset_args(args: list[dict[str, Any]], path: str) -> None:
    """Bound ``date_add``/``date_sub`` to (expression, amount, unit) triples.

    The amount stays a plain integer literal so the grounding gate can still
    require the question to mention it, while the unit is a closed enum.  The
    SQLite modifier string is assembled by the compiler, never by the model.
    """

    if len(args) != 3:
        raise QueryPlanV2Error(f"{path}.args must be (expression, amount, unit)")
    amount = args[1]
    if (
        amount.get("type") != "literal"
        or isinstance(amount.get("value"), bool)
        or not isinstance(amount.get("value"), int)
        or not 0 <= amount["value"] <= MAX_DATE_OFFSET
    ):
        raise QueryPlanV2Error(
            f"{path}.args[1] must be a literal integer in [0, {MAX_DATE_OFFSET}]"
        )
    unit = args[2]
    if unit.get("type") != "literal" or unit.get("value") not in ALLOWED_DATE_UNITS:
        raise QueryPlanV2Error(
            f"{path}.args[2] must be one of {sorted(ALLOWED_DATE_UNITS)}"
        )


def _order(
    value: Any,
    path: str,
    *,
    output_aliases: frozenset[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QueryPlanV2Error(f"{path} must be an object")
    _reject_unknown(value, {"expr", "alias", "direction"}, path)
    direction = value.get("direction", "asc")
    if direction not in {"asc", "desc"}:
        raise QueryPlanV2Error(f"{path}.direction is invalid")
    alias = value.get("alias")
    if alias is not None:
        # Ordering by a select alias is only meaningful where the alias is an
        # output column of this very query, so it is rejected inside window
        # ``OVER (ORDER BY ...)`` clauses, which have no output list.
        if value.get("expr") is not None:
            raise QueryPlanV2Error(f"{path} must set either expr or alias, not both")
        if output_aliases is None:
            raise QueryPlanV2Error(f"{path}.alias is not allowed in this ORDER BY")
        name = _identifier(alias, f"{path}.alias")
        if name not in output_aliases:
            raise QueryPlanV2Error(f"{path}.alias is not a declared select alias")
        return {"alias": name, "direction": direction}
    return {"expr": _expr(value.get("expr"), f"{path}.expr"), "direction": direction}


def _limit_offset(value: Mapping[str, Any], path: str) -> tuple[int | None, int | None]:
    limit = value.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT):
        raise QueryPlanV2Error(f"{path}.limit is invalid")
    offset = value.get("offset")
    if offset is not None:
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= MAX_LIMIT:
            raise QueryPlanV2Error(f"{path}.offset is invalid")
        if limit is None:
            raise QueryPlanV2Error(f"{path}.offset requires limit")
    return limit, offset


def _query(value: Any, path: str = "$.query") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QueryPlanV2Error(f"{path} must be an object")
    if value.get("type") == "set_op":
        _reject_unknown(
            value,
            {"type", "operator", "left", "right", "all", "order_by", "limit", "offset"},
            path,
        )
        if value.get("operator") not in {"union", "intersect", "except"}:
            raise QueryPlanV2Error(f"{path}.operator is invalid")
        left = _query(value.get("left"), f"{path}.left")
        right = _query(value.get("right"), f"{path}.right")
        left_names = _query_output_aliases(left)
        right_names = _query_output_aliases(right)
        if len(left_names) != len(right_names):
            raise QueryPlanV2Error(
                f"{path} branches must project the same number of columns"
            )
        # The lowering references each branch's outputs by name, and a compound
        # ORDER BY resolves against them, so duplicates would be ambiguous.
        for names, side in ((left_names, "left"), (right_names, "right")):
            if len(set(names)) != len(names):
                raise QueryPlanV2Error(
                    f"{path}.{side} must give every projected column a unique name"
                )
        order_by = value.get("order_by", [])
        if not isinstance(order_by, list) or len(order_by) > MAX_SELECT:
            raise QueryPlanV2Error(f"{path}.order_by is too large")
        # SQLite resolves a compound ORDER BY against the output columns of the
        # first branch only, so the alias form is the only sound one here.
        branch_aliases = frozenset(_query_output_aliases(left))
        parsed_order = []
        for index, item in enumerate(order_by):
            entry = _order(item, f"{path}.order_by[{index}]", output_aliases=branch_aliases)
            if "alias" not in entry:
                raise QueryPlanV2Error(
                    f"{path}.order_by[{index}] must order by a branch output alias"
                )
            parsed_order.append(entry)
        limit, offset = _limit_offset(value, path)
        return {"type": "set_op", "operator": value["operator"], "all": bool(value.get("all", False)),
                "left": left, "right": right,
                "order_by": parsed_order, "limit": limit, "offset": offset}
    _reject_unknown(value, {"type", "from", "joins", "select", "where", "group_by", "having", "order_by", "limit", "offset", "distinct", "ctes"}, path)
    if value.get("type", "select") != "select":
        raise QueryPlanV2Error(f"{path}.type must be select or set_op")
    source = _source(value.get("from"), f"{path}.from")
    joins = value.get("joins", [])
    if not isinstance(joins, list) or len(joins) > MAX_JOINS:
        raise QueryPlanV2Error(f"{path}.joins has invalid size")
    parsed_joins = []
    for i, join in enumerate(joins):
        if not isinstance(join, Mapping):
            raise QueryPlanV2Error(f"{path}.joins[{i}] must be an object")
        _reject_unknown(join, {"type", "source", "on", "kind"}, f"{path}.joins[{i}]")
        if join.get("type", "join") != "join" or join.get("kind", "inner") not in {"inner", "left"}:
            raise QueryPlanV2Error(f"{path}.joins[{i}] has invalid type/kind")
        on = join.get("on")
        if not isinstance(on, list) or not on or len(on) > 8:
            raise QueryPlanV2Error(f"{path}.joins[{i}].on has invalid size")
        parsed_joins.append({"source": _source(join.get("source"), f"{path}.joins[{i}].source"),
                             "on": [_expr(item, f"{path}.joins[{i}].on[{j}]") for j, item in enumerate(on)],
                             "kind": join.get("kind", "inner")})
    select = value.get("select")
    if not isinstance(select, list) or not 1 <= len(select) <= MAX_SELECT:
        raise QueryPlanV2Error(f"{path}.select has invalid size")
    parsed_select = []
    for i, item in enumerate(select):
        if not isinstance(item, Mapping):
            raise QueryPlanV2Error(f"{path}.select[{i}] must be an object")
        _reject_unknown(item, {"expr", "alias"}, f"{path}.select[{i}]")
        alias = item.get("alias")
        parsed_select.append({"expr": _expr(item.get("expr"), f"{path}.select[{i}].expr"),
                              "alias": None if alias is None else _identifier(alias, f"{path}.select[{i}].alias")})
    ctes = value.get("ctes", [])
    if not isinstance(ctes, list) or len(ctes) > MAX_CTES:
        raise QueryPlanV2Error(f"{path}.ctes has invalid size")
    parsed_ctes = []
    for i, cte in enumerate(ctes):
        if not isinstance(cte, Mapping):
            raise QueryPlanV2Error(f"{path}.ctes[{i}] must be an object")
        _reject_unknown(cte, {"name", "query"}, f"{path}.ctes[{i}]")
        parsed_ctes.append({"name": _identifier(cte.get("name"), f"{path}.ctes[{i}].name"),
                            "query": _query(cte.get("query"), f"{path}.ctes[{i}].query")})
    group_by = value.get("group_by", [])
    order_by = value.get("order_by", [])
    if not isinstance(group_by, list) or len(group_by) > MAX_SELECT or not isinstance(order_by, list) or len(order_by) > MAX_SELECT:
        raise QueryPlanV2Error(f"{path} group/order clauses are too large")
    declared_aliases = frozenset(
        item["alias"] for item in parsed_select if item["alias"]
    )
    limit, offset = _limit_offset(value, path)
    return {"type": "select", "from": source, "joins": parsed_joins, "select": parsed_select,
            "where": None if value.get("where") is None else _expr(value["where"], f"{path}.where"),
            "group_by": [_expr(v, f"{path}.group_by[{i}]") for i, v in enumerate(group_by)],
            "having": None if value.get("having") is None else _expr(value["having"], f"{path}.having"),
            "order_by": [
                _order(v, f"{path}.order_by[{i}]", output_aliases=declared_aliases)
                for i, v in enumerate(order_by)
            ],
            "limit": limit, "offset": offset,
            "distinct": bool(value.get("distinct", False)), "ctes": parsed_ctes}


def _validate_join_paths(query: Mapping[str, Any], path: str = "$.query") -> None:
    """Fail closed on non-equality and unknown direct base-table joins."""

    if query["type"] == "set_op":
        _validate_join_paths(query["left"], f"{path}.left")
        _validate_join_paths(query["right"], f"{path}.right")
        return
    for index, cte in enumerate(query["ctes"]):
        _validate_join_paths(cte["query"], f"{path}.ctes[{index}].query")
    source = query["from"]
    aliases: dict[str, str | None] = {
        source["alias"]: source["table"] if source["type"] == "table" else None
    }
    if source["type"] == "subquery":
        _validate_join_paths(source["query"], f"{path}.from.query")
    for index, join in enumerate(query["joins"]):
        joined = join["source"]
        if joined["type"] == "subquery":
            _validate_join_paths(joined["query"], f"{path}.joins[{index}].source.query")
        aliases[joined["alias"]] = (
            joined["table"] if joined["type"] == "table" else None
        )
        anchored = False
        for clause_index, clause in enumerate(join["on"]):
            clause_path = f"{path}.joins[{index}].on[{clause_index}]"
            if (
                clause.get("type") != "binary"
                or clause.get("operator") not in ALLOWED_JOIN_REFINEMENT_OPS
            ):
                raise QueryPlanV2Error(
                    f"{clause_path} must be a comparison between columns and literals"
                )
            left = clause["left"]
            right = clause["right"]
            if left.get("type") not in {"column", "literal"} or right.get(
                "type"
            ) not in {"column", "literal"}:
                raise QueryPlanV2Error(
                    f"{clause_path} may only compare columns and literals"
                )
            if (
                clause["operator"] != "eq"
                or left["type"] != "column"
                or right["type"] != "column"
            ):
                # A refinement narrows the join that the anchoring equality
                # already authorised; it cannot introduce a new join path.
                continue
            left_table = aliases.get(left["table"])
            right_table = aliases.get(right["table"])
            if left_table is None or right_table is None:
                # One side is a derived table or CTE, which has no base-table
                # path to authorise; the equality still anchors the join.
                anchored = True
                continue
            if left_table == right_table == "kg_triples":
                if left["column"] not in {"head", "tail"} or right["column"] not in {
                    "head", "tail"
                }:
                    raise QueryPlanV2Error(
                        f"{clause_path} is not an allowlisted kg_triples self-join"
                    )
                anchored = True
                continue
            direct_path = frozenset(
                {
                    (left_table, left["column"]),
                    (right_table, right["column"]),
                }
            )
            if direct_path not in ALLOWED_DIRECT_JOIN_PATHS:
                raise QueryPlanV2Error(
                    f"{clause_path} is not an allowlisted direct join path"
                )
            anchored = True
        if not anchored:
            raise QueryPlanV2Error(
                f"{path}.joins[{index}].on needs one allowlisted column equality"
            )


def parse_query_plan_v2(raw: str | bytes | bytearray | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        obj = dict(raw)
    else:
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        # Model outputs frequently wrap JSON in Markdown code fences (```json
        # ... ```) or prepend a short reasoning/scratch block.  Strip the
        # fences and, when a bare json.loads still fails, recover the first
        # balanced top-level JSON object so a valid plan is not lost to
        # formatting.  This is a pure robustness fix: the plan is then held to
        # the same strict schema/grounding gates as before.
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I).strip()
        try:
            obj = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end <= start:
                raise QueryPlanV2Error("invalid QueryPlan v2 JSON")
            try:
                obj = json.loads(text[start : end + 1])
            except Exception as exc:
                raise QueryPlanV2Error("invalid QueryPlan v2 JSON") from exc
    if not isinstance(obj, Mapping) or obj.get("plan_version") != PLAN_VERSION:
        raise QueryPlanV2Error("plan_version must be 2")
    _reject_unknown(obj, {"plan_version", "query"}, "$")
    _walk(obj)
    query = _query(obj.get("query"))
    _validate_join_paths(query)
    return {"plan_version": PLAN_VERSION, "query": query}


def _column(value: Mapping[str, Any], scopes: dict[str, set[str]]) -> str:
    table = value["table"]
    column = value["column"]
    if table not in scopes:
        raise QueryPlanV2Error(f"unknown table alias {table!r}")
    allowed = scopes[table]
    if column not in allowed:
        raise QueryPlanV2Error(f"column {table}.{column} is not allowlisted")
    return f"{_quote_identifier(table)}.{_quote_identifier(column)}"


def _expr_sql(value: Mapping[str, Any], scopes: dict[str, set[str]], ctes: dict[str, set[str]]) -> str:
    typ = value["type"]
    if typ == "column":
        return _column(value, scopes)
    if typ == "literal":
        return _literal(value["value"], "literal")
    if typ == "aggregate":
        function = value["function"].upper()
        arg = "*" if value["argument"] is None else _expr_sql(value["argument"], scopes, ctes)
        distinct = "DISTINCT " if value.get("distinct") else ""
        return f"{function}({distinct}{arg})"
    if typ == "function":
        if value["function"] in DATE_OFFSET_FUNCTIONS:
            # Build the SQLite modifier from the validated amount and unit so no
            # model-authored string can reach the emitted SQL.
            sign = DATE_OFFSET_FUNCTIONS[value["function"]]
            amount = value["args"][1]["value"]
            unit = value["args"][2]["value"]
            modifier = _literal(f"{sign}{amount} {unit}s", "date modifier")
            target = _expr_sql(value["args"][0], scopes, ctes)
            return f"DATE({target}, {modifier})"
        return f"{value['function'].upper()}({', '.join(_expr_sql(arg, scopes, ctes) for arg in value['args'])})"
    if typ == "binary":
        operator = {
            "eq": "=", "neq": "<>", "gt": ">", "gte": ">=", "lt": "<",
            "lte": "<=", "and": "AND", "or": "OR", "add": "+", "sub": "-",
            "mul": "*", "div": "/", "like": "LIKE", "is": "IS",
            "is_not": "IS NOT",
        }[value["operator"]]
        return f"({_expr_sql(value['left'], scopes, ctes)} {operator} {_expr_sql(value['right'], scopes, ctes)})"
    if typ == "unary":
        return f"(NOT {_expr_sql(value['argument'], scopes, ctes)})"
    if typ == "in":
        left = _expr_sql(value["value"], scopes, ctes)
        if "values" in value:
            return f"({left} IN ({', '.join(_expr_sql(item, scopes, ctes) for item in value['values'])}))"
        return f"({left} IN ({_query_sql(value['query'], ctes, scopes)}))"
    if typ == "exists":
        return f"EXISTS ({_query_sql(value['query'], ctes, scopes)})"
    if typ == "scalar_subquery":
        return f"({_query_sql(value['query'], ctes, scopes)})"
    if typ == "case":
        pairs = " ".join(f"WHEN {_expr_sql(item['when'], scopes, ctes)} THEN {_expr_sql(item['then'], scopes, ctes)}" for item in value["when"])
        return f"(CASE {pairs} ELSE {_expr_sql(value['else'], scopes, ctes)} END)"
    if typ == "window":
        if value["argument"] is None:
            arguments = "*" if value["function"] == "count" else ""
        else:
            arguments = _expr_sql(value["argument"], scopes, ctes)
        over = []
        if value["partition_by"]:
            over.append("PARTITION BY " + ", ".join(_expr_sql(item, scopes, ctes) for item in value["partition_by"]))
        if value["order_by"]:
            over.append("ORDER BY " + ", ".join(_order_sql(item, scopes, ctes) for item in value["order_by"]))
        if value.get("frame") is not None:
            frame = value["frame"]
            render = {
                "unbounded_preceding": "UNBOUNDED PRECEDING",
                "current_row": "CURRENT ROW",
                "unbounded_following": "UNBOUNDED FOLLOWING",
            }
            over.append(
                "ROWS BETWEEN "
                + render[frame["start"]]
                + " AND "
                + render[frame["end"]]
            )
        return f"{value['function'].upper()}({arguments}) OVER ({' '.join(over)})"
    raise QueryPlanV2Error(f"unsupported expression {typ!r}")


def _order_sql(
    value: Mapping[str, Any],
    scopes: dict[str, set[str]],
    ctes: dict[str, set[str]],
    select: list[Mapping[str, Any]] | None = None,
) -> str:
    suffix = " DESC" if value["direction"] == "desc" else " ASC"
    if "alias" in value:
        # Inside a plain SELECT the alias is only an authoring convenience: it is
        # replaced by the expression it names, so the emitted SQL is identical to
        # the classic form and downstream lineage analysis is unaffected.  A
        # compound query has no such select list, so the output name is emitted.
        for item in select or ():
            if item["alias"] == value["alias"]:
                return _expr_sql(item["expr"], scopes, ctes) + suffix
        return _quote_identifier(value["alias"]) + suffix
    return _expr_sql(value["expr"], scopes, ctes) + suffix


def _source_sql(source: Mapping[str, Any], ctes: dict[str, set[str]]) -> tuple[str, str, set[str]]:
    alias = source["alias"]
    if source["type"] == "table":
        return f"{_quote_identifier(source['table'])} AS {_quote_identifier(alias)}", alias, set(TABLE_COLUMNS[source["table"]])
    if source["type"] == "cte":
        if source["name"] not in ctes:
            raise QueryPlanV2Error(f"unknown CTE source {source['name']!r}")
        return f"{_quote_identifier(source['name'])} AS {_quote_identifier(alias)}", alias, set(ctes[source["name"]])
    return f"({_query_sql(source['query'], ctes, name_outputs=True)}) AS {_quote_identifier(alias)}", alias, set(_query_output_aliases(source["query"]))


def _query_output_aliases(query: Mapping[str, Any]) -> list[str]:
    if query["type"] == "set_op":
        return _query_output_aliases(query["left"])
    aliases = []
    for index, item in enumerate(query["select"]):
        aliases.append(item["alias"] or (item["expr"].get("column") if item["expr"].get("type") == "column" else f"_expr_{index}"))
    return aliases


def _query_sql(
    query: Mapping[str, Any],
    parent_ctes: dict[str, set[str]] | None = None,
    parent_scopes: dict[str, set[str]] | None = None,
    *,
    name_outputs: bool = False,
) -> str:
    """Lower a validated query node.

    ``name_outputs`` makes every projected expression carry the alias that
    :func:`_query_output_aliases` predicts.  Derived tables, CTE bodies and set
    operation branches are consumed by name, so their declared and emitted
    column names must not drift apart.
    """

    ctes = dict(parent_ctes or {})
    if query["type"] == "set_op":
        operator = query["operator"].upper() + (" ALL" if query.get("all") and query["operator"] == "union" else "")
        # SQLite rejects parenthesised bare SELECT terms in a compound query.
        # Wrapping each branch as a derived table preserves the declared binary
        # grouping while remaining valid for nested set operations.  The branch
        # columns are projected by name rather than with ``*`` so downstream
        # lineage analysis never sees a wildcard.
        left = _query_sql(query["left"], ctes, parent_scopes, name_outputs=True)
        right = _query_sql(query["right"], ctes, parent_scopes, name_outputs=True)
        left_projection = ", ".join(
            f'"_set_left".{_quote_identifier(name)}'
            for name in _query_output_aliases(query["left"])
        )
        right_projection = ", ".join(
            f'"_set_right".{_quote_identifier(name)}'
            for name in _query_output_aliases(query["right"])
        )
        sql = (
            f"SELECT {left_projection} FROM ({left}) AS \"_set_left\" {operator} "
            f"SELECT {right_projection} FROM ({right}) AS \"_set_right\""
        )
        # A compound ORDER BY/LIMIT binds to the whole set operation and may
        # only name output columns, which is why parsing restricts it to the
        # alias form.
        if query.get("order_by"):
            sql += " ORDER BY " + ", ".join(
                _order_sql(item, {}, ctes) for item in query["order_by"]
            )
        if query.get("limit") is not None:
            sql += f" LIMIT {query['limit']}"
            if query.get("offset") is not None:
                sql += f" OFFSET {query['offset']}"
        return sql
    cte_sql = []
    for cte in query["ctes"]:
        body = _query_sql(cte["query"], ctes, name_outputs=True)
        aliases = set(_query_output_aliases(cte["query"]))
        ctes[cte["name"]] = aliases
        cte_sql.append(f"{_quote_identifier(cte['name'])} AS ({body})")
    source_sql, alias, columns = _source_sql(query["from"], ctes)
    scopes = dict(parent_scopes or {})
    if alias in scopes:
        raise QueryPlanV2Error(f"query alias shadows outer scope {alias!r}")
    scopes[alias] = columns
    joins = []
    for join in query["joins"]:
        joined_sql, joined_alias, joined_columns = _source_sql(join["source"], ctes)
        if joined_alias in scopes:
            raise QueryPlanV2Error(f"duplicate or shadowing join alias {joined_alias!r}")
        scopes[joined_alias] = joined_columns
        on_sql = " AND ".join(_expr_sql(item, scopes, ctes) for item in join["on"])
        joins.append(f"{join['kind'].upper()} JOIN {joined_sql} ON {on_sql}")
    select_sql = []
    projected = _query_output_aliases(query) if name_outputs else None
    for index, item in enumerate(query["select"]):
        expression = _expr_sql(item["expr"], scopes, ctes)
        alias = item["alias"] or (projected[index] if projected else None)
        if alias:
            expression += f" AS {_quote_identifier(alias)}"
        select_sql.append(expression)
    sql = "SELECT " + ("DISTINCT " if query["distinct"] else "") + ", ".join(select_sql)
    sql += f" FROM {source_sql}" + (" " + " ".join(joins) if joins else "")
    if query["where"] is not None:
        sql += " WHERE " + _expr_sql(query["where"], scopes, ctes)
    if query["group_by"]:
        sql += " GROUP BY " + ", ".join(_expr_sql(item, scopes, ctes) for item in query["group_by"])
    if query["having"] is not None:
        sql += " HAVING " + _expr_sql(query["having"], scopes, ctes)
    if query["order_by"]:
        sql += " ORDER BY " + ", ".join(
            _order_sql(item, scopes, ctes, query["select"])
            for item in query["order_by"]
        )
    if query["limit"] is not None:
        sql += f" LIMIT {query['limit']}"
        if query.get("offset") is not None:
            sql += f" OFFSET {query['offset']}"
    if cte_sql:
        sql = "WITH " + ", ".join(cte_sql) + " " + sql
    return sql


def compile_query_plan_v2(plan: Any) -> str:
    parsed = parse_query_plan_v2(plan)
    sql = _query_sql(parsed["query"])
    try:
        validate_gold_sql(sql)
    except Exception as exc:
        raise QueryPlanV2Error(f"compiled QueryPlan v2 failed read-only gate: {exc}") from exc
    return sql


def validate_plan_v2_grounding(
    plan: Any,
    question: str,
    *,
    allowed_domain_values: tuple[str, ...] = (),
) -> None:
    """Ground semantic literals in the question or sealed DB value domain.

    SQL-format arguments such as ``strftime('%Y-%m', ...)`` and
    ``round(value, 2)`` are compiler controls rather than predicates and are
    checked structurally instead of requiring unnatural words in the question.
    """

    parsed = parse_query_plan_v2(plan)
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", question).casefold())
    domain = {
        re.sub(r"\s+", "", unicodedata.normalize("NFKC", value).casefold())
        for value in allowed_domain_values
        if isinstance(value, str) and value
    }

    def visit(
        value: Any,
        *,
        parent_function: str | None = None,
        argument_index: int | None = None,
    ) -> None:
        if isinstance(value, Mapping):
            if value.get("type") == "literal":
                literal = value.get("value")
                is_format_control = (
                    parent_function in STRUCTURAL_ARGUMENT_POSITIONS
                    and argument_index in STRUCTURAL_ARGUMENT_POSITIONS[parent_function]
                )
                if is_format_control:
                    return
                if isinstance(literal, str):
                    grounded = re.sub(
                        r"\s+",
                        "",
                        unicodedata.normalize("NFKC", literal.strip("%_")).casefold(),
                    )
                    if grounded and grounded not in compact and grounded not in domain:
                        raise QueryPlanV2Error("plan literal is not grounded in question")
                elif isinstance(literal, (int, float)) and str(literal).casefold() not in compact:
                    raise QueryPlanV2Error("plan numeric literal is not grounded in question")
            if value.get("type") == "function":
                function = str(value.get("function", ""))
                for index, child in enumerate(value.get("args", [])):
                    visit(
                        child,
                        parent_function=function,
                        argument_index=index,
                    )
                return
            if value.get("type") in {"exists", "scalar_subquery"}:
                # ``EXISTS (SELECT 1 ...)`` uses a constant projection purely as an
                # existence probe; that constant is a SQL idiom, not a predicate the
                # question must mention.  Every other part of the subquery is still
                # grounded normally.
                subquery = value.get("query")
                if isinstance(subquery, Mapping):
                    for key, child in subquery.items():
                        if key == "select" and value.get("type") == "exists":
                            continue
                        visit(child)
                    return
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(parsed["query"])


def canonical_plan_sha256_v2(plan: Any) -> str:
    parsed = parse_query_plan_v2(plan)
    encoded = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def query_plan_v2_schema() -> dict[str, Any]:
    """Return the complete model-facing grammar and representative examples."""

    return {
        "type": "object",
        "required": ["plan_version", "query"],
        "properties": {
            "plan_version": {"const": 2},
            "query": {"$ref": "grammar.query"},
        },
        "additionalProperties": False,
        "grammar": {
            "query": {
                "select": {
                    "type": "select",
                    "from": "source",
                    "joins": "list[join] optional",
                    "select": "list[{expr: expression, alias?: identifier}]",
                    "where": "expression optional",
                    "group_by": "list[expression] optional",
                    "having": "expression optional",
                    "order_by": "list[order] optional",
                    "limit": "integer 1..500 optional",
                    "offset": "integer 0..500 optional; requires limit; emits LIMIT n OFFSET m",
                    "distinct": "boolean optional",
                    "ctes": "list[{name: identifier, query: query}] optional",
                },
                "set_op": {
                    "type": "set_op",
                    "operator": "union|intersect|except",
                    "left": "query",
                    "right": "query",
                    "all": "boolean optional; only union supports true",
                    "order_by": "list[order] optional; alias form only, naming a left-branch output alias",
                    "limit": "integer 1..500 optional",
                    "offset": "integer 0..500 optional; requires limit",
                },
            },
            "source": {
                "table": {
                    "type": "table",
                    "table": sorted(ALLOWED_TABLES),
                    "alias": "identifier optional",
                },
                "subquery": {
                    "type": "subquery",
                    "query": "query",
                    "alias": "identifier required",
                },
                "cte": {
                    "type": "cte",
                    "name": "identifier",
                    "alias": "identifier optional",
                },
            },
            "join": {
                "type": "join",
                "kind": "inner|left",
                "source": "source",
                "on": (
                    "non-empty list of conjuncts; at least one must be an eq "
                    "between two columns on an allowlisted join path, and every "
                    "other conjunct is a comparison whose two sides are each a "
                    "column or a literal (e.g. e.type = 'Drug')"
                ),
            },
            "order": {
                "expr": "expression; omit when using alias",
                "alias": "one of this query's select aliases; not allowed inside a window OVER",
                "direction": "asc|desc",
            },
            "expression": {
                "column": {"type": "column", "table": "alias", "column": "allowlisted column"},
                "literal": {"type": "literal", "value": "string|number|boolean|null"},
                "aggregate": {
                    "type": "aggregate",
                    "function": sorted(ALLOWED_AGGREGATES),
                    "argument": "expression or null for count(*)",
                    "distinct": "boolean optional",
                },
                "function": {
                    "type": "function",
                    "function": sorted(ALLOWED_FUNCTIONS),
                    "args": "list[expression]",
                    "date_add/date_sub": (
                        "exactly [date expression, literal integer 0..3660, "
                        f"literal unit in {sorted(ALLOWED_DATE_UNITS)}]; the "
                        "compiler emits DATE(expr, '-6 months') itself, so never "
                        "write a modifier string"
                    ),
                },
                "binary": {
                    "type": "binary",
                    "operator": sorted(ALLOWED_BINOPS),
                    "left": "expression",
                    "right": "expression",
                },
                "unary": {"type": "unary", "operator": "not", "argument": "expression"},
                "in_values": {"type": "in", "value": "expression", "values": "non-empty list[expression]"},
                "in_query": {"type": "in", "value": "expression", "query": "query"},
                "exists": {"type": "exists", "query": "query"},
                "scalar_subquery": {"type": "scalar_subquery", "query": "query"},
                "case": {
                    "type": "case",
                    "when": "list[{when: expression, then: expression}]",
                    "else": "expression",
                },
                "window": {
                    "type": "window",
                    "function": sorted(ALLOWED_WINDOWS),
                    "argument": "expression or null",
                    "partition_by": "list[expression]",
                    "order_by": "list[order]",
                    "frame": {
                        "unit": "rows",
                        "start": sorted(ALLOWED_WINDOW_FRAMES),
                        "end": sorted(ALLOWED_WINDOW_FRAMES),
                    },
                },
            },
            "rules": [
                "Use source aliases in every column expression.",
                "Use aggregate argument null for COUNT(*).",
                "Do not emit SQL, raw_sql, or free-form expression fields.",
                "Put semantic constants in literal nodes; they must appear in the question or DB domain.",
                "Order by a computed select item with {\"alias\": \"<select alias>\"};"
                " repeating the expression in order_by also works.",
                "Use is/is_not with literal null for NULL predicates.",
                "There is no between node: write and(gte, lte) over the same expression.",
                "Give every subquery its own source alias; reusing an outer alias is rejected.",
                "Filters that belong to a join may stay in join.on as column/literal"
                " comparisons; comparisons over computed expressions go in where.",
                "Attach ORDER BY/LIMIT of a union/intersect/except to the set_op node"
                " itself, using the alias form against the left branch's aliases.",
                "For relative dates use date_add/date_sub with a unit enum, never a"
                " modifier string.",
            ],
        },
        "examples": [
            {
                "purpose": "filter and deterministic ordering",
                "plan": {
                    "plan_version": 2,
                    "query": {
                        "type": "select",
                        "from": {"type": "table", "table": "visits", "alias": "v"},
                        "select": [
                            {"expr": {"type": "column", "table": "v", "column": "visit_id"}},
                            {"expr": {"type": "column", "table": "v", "column": "cost"}},
                        ],
                        "where": {
                            "type": "binary",
                            "operator": "gte",
                            "left": {"type": "column", "table": "v", "column": "cost"},
                            "right": {"type": "literal", "value": 1000},
                        },
                        "order_by": [
                            {"expr": {"type": "column", "table": "v", "column": "cost"}, "direction": "desc"},
                            {"expr": {"type": "column", "table": "v", "column": "visit_id"}, "direction": "asc"},
                        ],
                    },
                },
            },
            {
                "purpose": "group, having, aggregate and top-k",
                "plan": {
                    "plan_version": 2,
                    "query": {
                        "type": "select",
                        "from": {"type": "table", "table": "visits", "alias": "v"},
                        "select": [
                            {"expr": {"type": "column", "table": "v", "column": "department"}},
                            {"expr": {"type": "aggregate", "function": "count", "argument": None}, "alias": "visit_count"},
                        ],
                        "group_by": [{"type": "column", "table": "v", "column": "department"}],
                        "having": {
                            "type": "binary",
                            "operator": "gte",
                            "left": {"type": "aggregate", "function": "count", "argument": None},
                            "right": {"type": "literal", "value": 5},
                        },
                        "order_by": [
                            {"expr": {"type": "aggregate", "function": "count", "argument": None}, "direction": "desc"},
                            {"expr": {"type": "column", "table": "v", "column": "department"}, "direction": "asc"},
                        ],
                        "limit": 3,
                    },
                },
            },
            {
                "purpose": "allowlisted join",
                "plan": {
                    "plan_version": 2,
                    "query": {
                        "type": "select",
                        "from": {"type": "table", "table": "patients", "alias": "p"},
                        "joins": [
                            {
                                "type": "join",
                                "kind": "inner",
                                "source": {"type": "table", "table": "visits", "alias": "v"},
                                "on": [
                                    {
                                        "type": "binary",
                                        "operator": "eq",
                                        "left": {"type": "column", "table": "p", "column": "patient_id"},
                                        "right": {"type": "column", "table": "v", "column": "patient_id"},
                                    }
                                ],
                            }
                        ],
                        "select": [
                            {"expr": {"type": "column", "table": "p", "column": "name"}},
                            {"expr": {"type": "column", "table": "v", "column": "disease"}},
                        ],
                    },
                },
            },
            {
                "purpose": "correlated exists subquery",
                "plan": {
                    "plan_version": 2,
                    "query": {
                        "type": "select",
                        "from": {"type": "table", "table": "kg_entities", "alias": "e"},
                        "select": [
                            {"expr": {"type": "column", "table": "e", "column": "name"}}
                        ],
                        "where": {
                            "type": "exists",
                            "query": {
                                "type": "select",
                                "from": {"type": "table", "table": "kg_triples", "alias": "t"},
                                "select": [
                                    {"expr": {"type": "literal", "value": 1}}
                                ],
                                "where": {
                                    "type": "binary",
                                    "operator": "and",
                                    "left": {
                                        "type": "binary",
                                        "operator": "eq",
                                        "left": {"type": "column", "table": "t", "column": "head"},
                                        "right": {"type": "column", "table": "e", "column": "name"},
                                    },
                                    "right": {
                                        "type": "binary",
                                        "operator": "eq",
                                        "left": {"type": "column", "table": "t", "column": "relation"},
                                        "right": {"type": "literal", "value": "HAS_COMPLICATION"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            {
                "purpose": "cte aggregate consumed by outer query",
                "plan": {
                    "plan_version": 2,
                    "query": {
                        "type": "select",
                        "ctes": [
                            {
                                "name": "department_cost",
                                "query": {
                                    "type": "select",
                                    "from": {"type": "table", "table": "visits", "alias": "v"},
                                    "select": [
                                        {
                                            "expr": {"type": "column", "table": "v", "column": "department"},
                                            "alias": "department",
                                        },
                                        {
                                            "expr": {
                                                "type": "aggregate",
                                                "function": "avg",
                                                "argument": {"type": "column", "table": "v", "column": "cost"},
                                            },
                                            "alias": "avg_cost",
                                        },
                                    ],
                                    "group_by": [
                                        {"type": "column", "table": "v", "column": "department"}
                                    ],
                                },
                            }
                        ],
                        "from": {"type": "cte", "name": "department_cost", "alias": "d"},
                        "select": [
                            {"expr": {"type": "column", "table": "d", "column": "department"}},
                            {"expr": {"type": "column", "table": "d", "column": "avg_cost"}},
                        ],
                        "order_by": [
                            {
                                "expr": {"type": "column", "table": "d", "column": "avg_cost"},
                                "direction": "desc",
                            }
                        ],
                    },
                },
            },
            {
                "purpose": "set operation nested as an ordered source",
                "plan": {
                    "plan_version": 2,
                    "query": {
                        "type": "select",
                        "from": {
                            "type": "subquery",
                            "alias": "names",
                            "query": {
                                "type": "set_op",
                                "operator": "union",
                                "left": {
                                    "type": "select",
                                    "from": {"type": "table", "table": "visits", "alias": "v"},
                                    "select": [
                                        {
                                            "expr": {"type": "column", "table": "v", "column": "department"},
                                            "alias": "name",
                                        }
                                    ],
                                    "distinct": True,
                                },
                                "right": {
                                    "type": "select",
                                    "from": {"type": "table", "table": "kg_entities", "alias": "e"},
                                    "select": [
                                        {
                                            "expr": {"type": "column", "table": "e", "column": "name"},
                                            "alias": "name",
                                        }
                                    ],
                                    "distinct": True,
                                },
                            },
                        },
                        "select": [
                            {"expr": {"type": "column", "table": "names", "column": "name"}}
                        ],
                        "order_by": [
                            {
                                "expr": {"type": "column", "table": "names", "column": "name"},
                                "direction": "asc",
                            }
                        ],
                    },
                },
            },
            {
                "purpose": "full-frame last-value window",
                "plan": {
                    "plan_version": 2,
                    "query": {
                        "type": "select",
                        "from": {"type": "table", "table": "visits", "alias": "v"},
                        "select": [
                            {"expr": {"type": "column", "table": "v", "column": "patient_id"}},
                            {
                                "alias": "last_cost",
                                "expr": {
                                    "type": "window",
                                    "function": "last_value",
                                    "argument": {"type": "column", "table": "v", "column": "cost"},
                                    "partition_by": [
                                        {"type": "column", "table": "v", "column": "patient_id"}
                                    ],
                                    "order_by": [
                                        {
                                            "expr": {"type": "column", "table": "v", "column": "visit_date"},
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
                },
            },
            {
                "purpose": "join filtered by an entity type inside ON",
                "plan": {
                    "plan_version": 2,
                    "query": {
                        "type": "select",
                        "from": {"type": "table", "table": "kg_triples", "alias": "t"},
                        "joins": [
                            {
                                "type": "join",
                                "kind": "inner",
                                "source": {"type": "table", "table": "kg_entities", "alias": "e"},
                                "on": [
                                    {
                                        "type": "binary",
                                        "operator": "eq",
                                        "left": {"type": "column", "table": "e", "column": "name"},
                                        "right": {"type": "column", "table": "t", "column": "tail"},
                                    },
                                    {
                                        "type": "binary",
                                        "operator": "eq",
                                        "left": {"type": "column", "table": "e", "column": "type"},
                                        "right": {"type": "literal", "value": "Drug"},
                                    },
                                ],
                            }
                        ],
                        "select": [
                            {"expr": {"type": "column", "table": "t", "column": "tail"}},
                            {"expr": {"type": "column", "table": "e", "column": "mention_count"}},
                        ],
                        "distinct": True,
                    },
                },
            },
            {
                "purpose": "aggregate ordered by its own select alias",
                "plan": {
                    "plan_version": 2,
                    "query": {
                        "type": "select",
                        "from": {"type": "table", "table": "visits", "alias": "v"},
                        "select": [
                            {"expr": {"type": "column", "table": "v", "column": "department"}},
                            {
                                "expr": {"type": "aggregate", "function": "count", "argument": None},
                                "alias": "visit_count",
                            },
                        ],
                        "group_by": [{"type": "column", "table": "v", "column": "department"}],
                        "order_by": [
                            {"alias": "visit_count", "direction": "desc"},
                            {"expr": {"type": "column", "table": "v", "column": "department"}, "direction": "asc"},
                        ],
                        "limit": 5,
                    },
                },
            },
            {
                "purpose": "ordered set operation without a wrapper select",
                "plan": {
                    "plan_version": 2,
                    "query": {
                        "type": "set_op",
                        "operator": "intersect",
                        "left": {
                            "type": "select",
                            "from": {"type": "table", "table": "visits", "alias": "v"},
                            "select": [
                                {
                                    "expr": {"type": "column", "table": "v", "column": "drug_prescribed"},
                                    "alias": "drug",
                                }
                            ],
                            "distinct": True,
                        },
                        "right": {
                            "type": "select",
                            "from": {"type": "table", "table": "kg_entities", "alias": "e"},
                            "select": [
                                {"expr": {"type": "column", "table": "e", "column": "name"}, "alias": "drug"}
                            ],
                            "where": {
                                "type": "binary",
                                "operator": "eq",
                                "left": {"type": "column", "table": "e", "column": "type"},
                                "right": {"type": "literal", "value": "Drug"},
                            },
                        },
                        "order_by": [{"alias": "drug", "direction": "asc"}],
                    },
                },
            },
            {
                "purpose": "relative date window without a modifier string",
                "plan": {
                    "plan_version": 2,
                    "query": {
                        "type": "select",
                        "from": {"type": "table", "table": "visits", "alias": "v"},
                        "select": [
                            {"expr": {"type": "aggregate", "function": "count", "argument": None}, "alias": "visit_count"}
                        ],
                        "where": {
                            "type": "binary",
                            "operator": "gte",
                            "left": {"type": "column", "table": "v", "column": "visit_date"},
                            "right": {
                                "type": "function",
                                "function": "date_sub",
                                "args": [
                                    {
                                        "type": "scalar_subquery",
                                        "query": {
                                            "type": "select",
                                            "from": {"type": "table", "table": "visits", "alias": "v2"},
                                            "select": [
                                                {
                                                    "expr": {
                                                        "type": "aggregate",
                                                        "function": "max",
                                                        "argument": {"type": "column", "table": "v2", "column": "visit_date"},
                                                    }
                                                }
                                            ],
                                        },
                                    },
                                    {"type": "literal", "value": 6},
                                    {"type": "literal", "value": "month"},
                                ],
                            },
                        },
                    },
                },
            },
        ],
    }


__all__ = ["PLAN_VERSION", "QueryPlanV2Error", "canonical_plan_sha256_v2", "compile_query_plan_v2", "parse_query_plan_v2", "query_plan_v2_schema", "validate_plan_v2_grounding"]
