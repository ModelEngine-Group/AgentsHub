"""Strict, versioned QueryPlan v1 to deterministic SQLite compilation.

This module is intentionally a much smaller language than SQL.  Callers may
provide table and column identifiers only from the sealed four-table schema,
and joins only by selecting a named relationship from :data:`JOIN_PATHS`.
There is no escape hatch for SQL fragments.

The public boundary is split deliberately:

* :func:`parse_query_plan` handles untrusted JSON and rejects duplicate keys,
  non-finite numbers, unknown fields, and resource-exhaustion shapes.
* :func:`validate_plan_grounding` proves that every user-supplied literal is
  present in the natural-language question.  Database-domain membership never
  substitutes for that proof.
* :func:`compile_query_plan` revalidates even an already parsed mapping, builds
  a SQLGlot AST without parsing caller strings, and finally invokes the shared
  read-only SQL gate.

QueryPlan v1 intentionally supports one SELECT query only.  Subqueries, CTEs,
set operations, scalar functions, arbitrary join predicates, and raw SQL are
outside the language rather than being accepted and filtered after the fact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from functools import reduce
from typing import Any, Iterable, Mapping, Sequence

from dki.analysis.nl2sql_protocol import validate_gold_sql


PLAN_VERSION = 1
MAX_RAW_JSON_BYTES = 64 * 1024
MAX_PLAN_DEPTH = 16
MAX_PLAN_NODES = 256
MAX_SELECT_ITEMS = 32
MAX_JOINS = 8
MAX_PREDICATE_OPERANDS = 32
MAX_IN_VALUES = 64
MAX_GROUP_ITEMS = 32
MAX_ORDER_ITEMS = 32
MAX_LIMIT = 500
MAX_LITERAL_LENGTH = 512

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,31}\Z")


class QueryPlanError(ValueError):
    """Base class for all fail-closed QueryPlan rejections."""


class QueryPlanJSONError(QueryPlanError):
    """The untrusted input is not strict JSON."""


class QueryPlanValidationError(QueryPlanError):
    """The decoded object is not a valid QueryPlan v1."""


class QueryPlanGroundingError(QueryPlanError):
    """A plan literal is not evidenced by the source question."""


class QueryPlanCompileError(QueryPlanError):
    """A valid-looking plan cannot be lowered safely."""


@dataclass(frozen=True)
class _JoinPath:
    left_table: str
    right_table: str
    column_pairs: tuple[tuple[str, str], ...]
    # (side, column, literal kind, literal value).  These constants are part of
    # the trusted relationship definition, not model-supplied literals.
    constants: tuple[tuple[str, str, str, Any], ...] = ()


TABLE_COLUMNS: dict[str, dict[str, str]] = {
    "patients": {
        "patient_id": "text",
        "name": "text",
        "gender": "text",
        "age": "integer",
        "region": "text",
    },
    "visits": {
        "visit_id": "text",
        "patient_id": "text",
        "visit_date": "date",
        "department": "text",
        "disease": "text",
        "drug_prescribed": "text",
        "exam": "text",
        "cost": "number",
        "los_days": "integer",
    },
    "kg_entities": {
        "name": "text",
        "type": "text",
        "mention_count": "integer",
    },
    "kg_triples": {
        "head": "text",
        "head_type": "text",
        "relation": "text",
        "relation_zh": "text",
        "tail": "text",
        "tail_type": "text",
        "confidence": "number",
        "count": "integer",
    },
}


# Every accepted join has a complete ON clause.  A join entry must introduce
# exactly one new alias, which makes a disconnected component or cartesian
# product unrepresentable.
JOIN_PATHS: dict[str, _JoinPath] = {
    "patients_visits": _JoinPath(
        "patients",
        "visits",
        (("patient_id", "patient_id"),),
    ),
    "kg_entities_triples_head": _JoinPath(
        "kg_entities",
        "kg_triples",
        (("name", "head"), ("type", "head_type")),
    ),
    "kg_entities_triples_tail": _JoinPath(
        "kg_entities",
        "kg_triples",
        (("name", "tail"), ("type", "tail_type")),
    ),
    "visits_disease_entities": _JoinPath(
        "visits",
        "kg_entities",
        (("disease", "name"),),
        (("right", "type", "string", "Disease"),),
    ),
    "visits_drug_entities": _JoinPath(
        "visits",
        "kg_entities",
        (("drug_prescribed", "name"),),
        (("right", "type", "string", "Drug"),),
    ),
    "visits_exam_entities": _JoinPath(
        "visits",
        "kg_entities",
        (("exam", "name"),),
        (("right", "type", "string", "Exam"),),
    ),
    "visits_department_entities": _JoinPath(
        "visits",
        "kg_entities",
        (("department", "name"),),
        (("right", "type", "string", "Department"),),
    ),
    "visits_belongs_to_department_triples": _JoinPath(
        "visits",
        "kg_triples",
        (("disease", "head"), ("department", "tail")),
        (("right", "relation", "string", "BELONGS_TO_DEPT"),),
    ),
}

_AGGREGATES = {"count", "sum", "avg", "min", "max"}
_COMPARISON_OPERATORS = {
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "like",
    "not_like",
}
_NUMERIC_TYPES = {"integer", "number"}


def _schema_expression() -> dict[str, Any]:
    return {
        "oneOf": [
            {"$ref": "#/$defs/column"},
            {"$ref": "#/$defs/literal"},
            {"$ref": "#/$defs/aggregate"},
        ]
    }


_QUERY_PLAN_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://modelengine.example/schemas/query-plan-v1.json",
    "title": "QueryPlan v1",
    "type": "object",
    "additionalProperties": False,
    "required": ["plan_version", "from", "select"],
    "properties": {
        "plan_version": {"const": PLAN_VERSION},
        "from": {"$ref": "#/$defs/table_ref"},
        "joins": {
            "type": "array",
            "maxItems": MAX_JOINS,
            "default": [],
            "items": {"$ref": "#/$defs/join"},
        },
        "select": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_SELECT_ITEMS,
            "items": {"$ref": "#/$defs/select_item"},
        },
        "distinct": {"type": "boolean", "default": False},
        "where": {
            "oneOf": [{"type": "null"}, {"$ref": "#/$defs/predicate"}],
            "default": None,
        },
        "group_by": {
            "type": "array",
            "maxItems": MAX_GROUP_ITEMS,
            "default": [],
            "items": {"$ref": "#/$defs/column"},
        },
        "having": {
            "oneOf": [{"type": "null"}, {"$ref": "#/$defs/predicate"}],
            "default": None,
        },
        "order_by": {
            "type": "array",
            "maxItems": MAX_ORDER_ITEMS,
            "default": [],
            "items": {"$ref": "#/$defs/order_item"},
        },
        "limit": {
            "oneOf": [
                {"type": "null"},
                {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT},
            ],
            "default": None,
        },
    },
    "$defs": {
        "identifier": {
            "type": "string",
            "pattern": _IDENTIFIER_RE.pattern,
        },
        "table_ref": {
            "type": "object",
            "additionalProperties": False,
            "required": ["table", "alias"],
            "properties": {
                "table": {"enum": list(TABLE_COLUMNS)},
                "alias": {"$ref": "#/$defs/identifier"},
            },
        },
        "join": {
            "type": "object",
            "additionalProperties": False,
            "required": ["path", "left_alias", "right_alias"],
            "properties": {
                "path": {"enum": list(JOIN_PATHS)},
                "left_alias": {"$ref": "#/$defs/identifier"},
                "right_alias": {"$ref": "#/$defs/identifier"},
            },
        },
        "column": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "table_alias", "column"],
            "properties": {
                "type": {"const": "column"},
                "table_alias": {"$ref": "#/$defs/identifier"},
                "column": {"type": "string"},
            },
        },
        "literal": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "value_type", "value"],
            "properties": {
                "type": {"const": "literal"},
                "value_type": {
                    "enum": ["string", "integer", "number", "boolean", "null", "date"]
                },
                "value": {},
            },
        },
        "aggregate": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "function", "argument"],
            "properties": {
                "type": {"const": "aggregate"},
                "function": {"enum": sorted(_AGGREGATES)},
                "argument": {
                    "oneOf": [{"type": "null"}, {"$ref": "#/$defs/column"}]
                },
                "distinct": {"type": "boolean", "default": False},
            },
        },
        "expression": _schema_expression(),
        "select_item": {
            "type": "object",
            "additionalProperties": False,
            "required": ["expr"],
            "properties": {
                "expr": {"$ref": "#/$defs/expression"},
                "alias": {"$ref": "#/$defs/identifier"},
            },
        },
        "order_item": {
            "type": "object",
            "additionalProperties": False,
            "required": ["expr", "direction"],
            "properties": {
                "expr": {"$ref": "#/$defs/expression"},
                "direction": {"enum": ["asc", "desc"]},
            },
        },
        "predicate": {
            "oneOf": [
                {"$ref": "#/$defs/boolean_predicate"},
                {"$ref": "#/$defs/comparison_predicate"},
                {"$ref": "#/$defs/in_predicate"},
                {"$ref": "#/$defs/between_predicate"},
                {"$ref": "#/$defs/null_predicate"},
            ]
        },
        "boolean_predicate": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "operator", "operands"],
            "properties": {
                "type": {"const": "boolean"},
                "operator": {"enum": ["and", "or"]},
                "operands": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": MAX_PREDICATE_OPERANDS,
                    "items": {"$ref": "#/$defs/predicate"},
                },
            },
        },
        "comparison_predicate": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "operator", "left", "right"],
            "properties": {
                "type": {"const": "comparison"},
                "operator": {"enum": sorted(_COMPARISON_OPERATORS)},
                "left": {"$ref": "#/$defs/expression"},
                "right": {"$ref": "#/$defs/expression"},
            },
        },
        "in_predicate": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "expr", "values"],
            "properties": {
                "type": {"const": "in"},
                "expr": {"$ref": "#/$defs/expression"},
                "values": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_IN_VALUES,
                    "items": {"$ref": "#/$defs/literal"},
                },
                "negated": {"type": "boolean", "default": False},
            },
        },
        "between_predicate": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "expr", "lower", "upper"],
            "properties": {
                "type": {"const": "between"},
                "expr": {"$ref": "#/$defs/expression"},
                "lower": {"$ref": "#/$defs/literal"},
                "upper": {"$ref": "#/$defs/literal"},
                "negated": {"type": "boolean", "default": False},
            },
        },
        "null_predicate": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "expr"],
            "properties": {
                "type": {"const": "is_null"},
                "expr": {"$ref": "#/$defs/expression"},
                "negated": {"type": "boolean", "default": False},
            },
        },
    },
}


def query_plan_schema() -> dict[str, Any]:
    """Return an isolated JSON Schema document for QueryPlan v1."""

    return copy.deepcopy(_QUERY_PLAN_SCHEMA)


def _reject_json_constant(token: str) -> Any:
    raise QueryPlanJSONError(f"non-finite JSON number is forbidden: {token}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QueryPlanJSONError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _budget_check(value: Any, *, depth: int = 0, count: list[int] | None = None) -> None:
    if count is None:
        count = [0]
    if depth > MAX_PLAN_DEPTH:
        raise QueryPlanValidationError(
            f"plan nesting exceeds maximum depth {MAX_PLAN_DEPTH}"
        )
    if isinstance(value, Mapping):
        count[0] += 1
        if count[0] > MAX_PLAN_NODES:
            raise QueryPlanValidationError(
                f"plan exceeds maximum node count {MAX_PLAN_NODES}"
            )
        for key, child in value.items():
            if not isinstance(key, str):
                raise QueryPlanValidationError("all plan object keys must be strings")
            _budget_check(child, depth=depth + 1, count=count)
    elif isinstance(value, (list, tuple)):
        count[0] += 1
        if count[0] > MAX_PLAN_NODES:
            raise QueryPlanValidationError(
                f"plan exceeds maximum node count {MAX_PLAN_NODES}"
            )
        for child in value:
            _budget_check(child, depth=depth + 1, count=count)


def _expect_object(
    value: Any,
    *,
    path: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QueryPlanValidationError(f"{path} must be an object")
    required_set = set(required)
    allowed = required_set | set(optional)
    keys = set(value)
    missing = required_set - keys
    extra = keys - allowed
    if missing:
        raise QueryPlanValidationError(
            f"{path} is missing required field(s): {sorted(missing)!r}"
        )
    if extra:
        raise QueryPlanValidationError(
            f"{path} has unknown field(s): {sorted(map(str, extra))!r}"
        )
    return value


def _expect_list(
    value: Any,
    *,
    path: str,
    minimum: int = 0,
    maximum: int,
) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise QueryPlanValidationError(f"{path} must be an array")
    if not minimum <= len(value) <= maximum:
        raise QueryPlanValidationError(
            f"{path} must contain between {minimum} and {maximum} item(s)"
        )
    return value


def _identifier(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise QueryPlanValidationError(
            f"{path} must match {_IDENTIFIER_RE.pattern!r}"
        )
    return value


def _normalise_literal(value: Any, *, path: str) -> dict[str, Any]:
    obj = _expect_object(
        value,
        path=path,
        required=("type", "value_type", "value"),
    )
    if obj["type"] != "literal":
        raise QueryPlanValidationError(f"{path}.type must be 'literal'")
    kind = obj["value_type"]
    if not isinstance(kind, str):
        raise QueryPlanValidationError(f"{path}.value_type must be a string")
    raw = obj["value"]
    if kind in {"string", "date"}:
        if not isinstance(raw, str):
            raise QueryPlanValidationError(f"{path}.value must be a string for {kind}")
        if (
            not raw
            or len(raw) > MAX_LITERAL_LENGTH
            or any(ord(character) < 0x20 for character in raw)
            or any(0xD800 <= ord(character) <= 0xDFFF for character in raw)
        ):
            raise QueryPlanValidationError(
                f"{path}.value must contain 1..{MAX_LITERAL_LENGTH} valid, "
                "non-control Unicode characters"
            )
        if kind == "date":
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw) is None:
                raise QueryPlanValidationError(
                    f"{path}.value must be an ISO date (YYYY-MM-DD)"
                )
            try:
                parsed = date.fromisoformat(raw)
            except ValueError as exc:
                raise QueryPlanValidationError(
                    f"{path}.value must be an ISO date (YYYY-MM-DD)"
                ) from exc
            raw = parsed.isoformat()
    elif kind == "integer":
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise QueryPlanValidationError(f"{path}.value must be an integer")
        if abs(raw) > 10**15:
            raise QueryPlanValidationError(f"{path}.value is outside the numeric bound")
    elif kind == "number":
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise QueryPlanValidationError(f"{path}.value must be a number")
        if not math.isfinite(float(raw)) or abs(float(raw)) > 10**15:
            raise QueryPlanValidationError(f"{path}.value must be finite and bounded")
        raw = float(raw)
        if raw == 0:
            raw = 0.0
    elif kind == "boolean":
        if not isinstance(raw, bool):
            raise QueryPlanValidationError(f"{path}.value must be a boolean")
    elif kind == "null":
        if raw is not None:
            raise QueryPlanValidationError(f"{path}.value must be null")
    else:
        raise QueryPlanValidationError(f"{path}.value_type is unknown: {kind!r}")
    return {"type": "literal", "value_type": kind, "value": raw}


def _normalise_column(value: Any, *, path: str) -> dict[str, Any]:
    obj = _expect_object(
        value,
        path=path,
        required=("type", "table_alias", "column"),
    )
    if obj["type"] != "column":
        raise QueryPlanValidationError(f"{path}.type must be 'column'")
    column = obj["column"]
    if not isinstance(column, str):
        raise QueryPlanValidationError(f"{path}.column must be a string")
    return {
        "type": "column",
        "table_alias": _identifier(obj["table_alias"], path=f"{path}.table_alias"),
        "column": column,
    }


def _normalise_aggregate(value: Any, *, path: str) -> dict[str, Any]:
    obj = _expect_object(
        value,
        path=path,
        required=("type", "function", "argument"),
        optional=("distinct",),
    )
    if obj["type"] != "aggregate":
        raise QueryPlanValidationError(f"{path}.type must be 'aggregate'")
    function = obj["function"]
    if not isinstance(function, str) or function not in _AGGREGATES:
        raise QueryPlanValidationError(f"{path}.function is unknown: {function!r}")
    argument = obj["argument"]
    if argument is not None:
        argument = _normalise_column(argument, path=f"{path}.argument")
    if argument is None and function != "count":
        raise QueryPlanValidationError(
            f"{path}.argument may be null only for count(*)"
        )
    distinct = obj.get("distinct", False)
    if not isinstance(distinct, bool):
        raise QueryPlanValidationError(f"{path}.distinct must be a boolean")
    if distinct and argument is None:
        raise QueryPlanValidationError(f"{path} cannot use DISTINCT with count(*)")
    return {
        "type": "aggregate",
        "function": function,
        "argument": argument,
        "distinct": distinct,
    }


def _normalise_expression(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QueryPlanValidationError(f"{path} must be a typed expression object")
    kind = value.get("type")
    if kind == "column":
        return _normalise_column(value, path=path)
    if kind == "literal":
        return _normalise_literal(value, path=path)
    if kind == "aggregate":
        return _normalise_aggregate(value, path=path)
    raise QueryPlanValidationError(f"{path}.type is unknown: {kind!r}")


def _normalise_predicate(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QueryPlanValidationError(f"{path} must be a predicate object")
    kind = value.get("type")
    if kind == "boolean":
        obj = _expect_object(
            value,
            path=path,
            required=("type", "operator", "operands"),
        )
        operator = obj["operator"]
        if not isinstance(operator, str) or operator not in {"and", "or"}:
            raise QueryPlanValidationError(f"{path}.operator is unknown: {operator!r}")
        operands = _expect_list(
            obj["operands"],
            path=f"{path}.operands",
            minimum=2,
            maximum=MAX_PREDICATE_OPERANDS,
        )
        return {
            "type": "boolean",
            "operator": operator,
            "operands": [
                _normalise_predicate(item, path=f"{path}.operands[{index}]")
                for index, item in enumerate(operands)
            ],
        }
    if kind == "comparison":
        obj = _expect_object(
            value,
            path=path,
            required=("type", "operator", "left", "right"),
        )
        operator = obj["operator"]
        if not isinstance(operator, str) or operator not in _COMPARISON_OPERATORS:
            raise QueryPlanValidationError(f"{path}.operator is unknown: {operator!r}")
        left = _normalise_expression(obj["left"], path=f"{path}.left")
        right = _normalise_expression(obj["right"], path=f"{path}.right")
        if left["type"] == "literal" and right["type"] == "literal":
            raise QueryPlanValidationError(
                f"{path} must compare at least one column or aggregate"
            )
        if left["type"] == "literal" and right["type"] != "literal":
            raise QueryPlanValidationError(
                f"{path}.left cannot be a literal; put the literal on the right"
            )
        if operator in {"like", "not_like"}:
            if right["type"] != "literal" or right["value_type"] != "string":
                raise QueryPlanValidationError(
                    f"{path}.{operator} requires a string literal on the right"
                )
        if right["type"] == "literal" and right["value_type"] == "null":
            raise QueryPlanValidationError(
                f"{path} must use an is_null predicate for NULL"
            )
        return {
            "type": "comparison",
            "operator": operator,
            "left": left,
            "right": right,
        }
    if kind == "in":
        obj = _expect_object(
            value,
            path=path,
            required=("type", "expr", "values"),
            optional=("negated",),
        )
        values = _expect_list(
            obj["values"],
            path=f"{path}.values",
            minimum=1,
            maximum=MAX_IN_VALUES,
        )
        negated = obj.get("negated", False)
        if not isinstance(negated, bool):
            raise QueryPlanValidationError(f"{path}.negated must be a boolean")
        expression = _normalise_expression(obj["expr"], path=f"{path}.expr")
        if expression["type"] == "literal":
            raise QueryPlanValidationError(f"{path}.expr cannot be a literal")
        return {
            "type": "in",
            "expr": expression,
            "values": [
                _normalise_literal(item, path=f"{path}.values[{index}]")
                for index, item in enumerate(values)
            ],
            "negated": negated,
        }
    if kind == "between":
        obj = _expect_object(
            value,
            path=path,
            required=("type", "expr", "lower", "upper"),
            optional=("negated",),
        )
        negated = obj.get("negated", False)
        if not isinstance(negated, bool):
            raise QueryPlanValidationError(f"{path}.negated must be a boolean")
        expression = _normalise_expression(obj["expr"], path=f"{path}.expr")
        if expression["type"] == "literal":
            raise QueryPlanValidationError(f"{path}.expr cannot be a literal")
        return {
            "type": "between",
            "expr": expression,
            "lower": _normalise_literal(obj["lower"], path=f"{path}.lower"),
            "upper": _normalise_literal(obj["upper"], path=f"{path}.upper"),
            "negated": negated,
        }
    if kind == "is_null":
        obj = _expect_object(
            value,
            path=path,
            required=("type", "expr"),
            optional=("negated",),
        )
        negated = obj.get("negated", False)
        if not isinstance(negated, bool):
            raise QueryPlanValidationError(f"{path}.negated must be a boolean")
        expression = _normalise_expression(obj["expr"], path=f"{path}.expr")
        if expression["type"] == "literal":
            raise QueryPlanValidationError(f"{path}.expr cannot be a literal")
        return {
            "type": "is_null",
            "expr": expression,
            "negated": negated,
        }
    raise QueryPlanValidationError(f"{path}.type is unknown: {kind!r}")


def _resolve_aliases(
    source: Mapping[str, Any], joins: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    aliases = {source["alias"]: source["table"]}
    aliases_folded = {source["alias"].casefold(): source["alias"]}
    for index, join in enumerate(joins):
        path = JOIN_PATHS[join["path"]]
        left_alias = join["left_alias"]
        right_alias = join["right_alias"]
        if left_alias.casefold() == right_alias.casefold():
            raise QueryPlanValidationError(
                f"$.joins[{index}] uses the same alias on both sides"
            )
        left_known = left_alias in aliases
        right_known = right_alias in aliases
        if left_known == right_known:
            reason = "cycle/redundant join" if left_known else "disconnected join"
            raise QueryPlanValidationError(
                f"$.joins[{index}] is a {reason}; each join must introduce exactly one alias"
            )
        if left_known:
            if aliases[left_alias] != path.left_table:
                raise QueryPlanValidationError(
                    f"$.joins[{index}].left_alias is not a {path.left_table} alias"
                )
            new_alias = right_alias
            new_table = path.right_table
        else:
            if aliases[right_alias] != path.right_table:
                raise QueryPlanValidationError(
                    f"$.joins[{index}].right_alias is not a {path.right_table} alias"
                )
            new_alias = left_alias
            new_table = path.left_table
        folded = new_alias.casefold()
        if folded in aliases_folded:
            raise QueryPlanValidationError(
                f"$.joins[{index}] introduces a duplicate alias: {new_alias!r}"
            )
        aliases[new_alias] = new_table
        aliases_folded[folded] = new_alias
    return aliases


def _column_type(expression: Mapping[str, Any], aliases: Mapping[str, str]) -> str:
    alias = expression["table_alias"]
    if alias not in aliases:
        raise QueryPlanValidationError(f"unknown table alias: {alias!r}")
    table = aliases[alias]
    column = expression["column"]
    if column not in TABLE_COLUMNS[table]:
        raise QueryPlanValidationError(
            f"unknown column {column!r} for table {table!r} (alias {alias!r})"
        )
    return TABLE_COLUMNS[table][column]


def _expression_type(expression: Mapping[str, Any], aliases: Mapping[str, str]) -> str:
    kind = expression["type"]
    if kind == "column":
        return _column_type(expression, aliases)
    if kind == "literal":
        literal_type = expression["value_type"]
        return "text" if literal_type in {"string", "date"} else literal_type
    if kind == "aggregate":
        argument = expression["argument"]
        argument_type = _column_type(argument, aliases) if argument is not None else None
        function = expression["function"]
        if function in {"sum", "avg"} and argument_type not in _NUMERIC_TYPES:
            raise QueryPlanValidationError(
                f"{function} requires an integer or number column"
            )
        if function == "count":
            return "integer"
        if function == "avg":
            return "number"
        assert argument_type is not None
        return argument_type
    raise AssertionError(f"unreachable expression kind: {kind!r}")


def _contains_aggregate(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("type") == "aggregate":
            return True
        return any(_contains_aggregate(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_aggregate(child) for child in value)
    return False


def _bare_column_keys(value: Any) -> Iterable[tuple[str, str]]:
    """Yield column references that are not arguments of an aggregate."""

    if isinstance(value, Mapping):
        kind = value.get("type")
        if kind == "aggregate":
            return
        if kind == "column":
            yield (value["table_alias"], value["column"])
            return
        for child in value.values():
            yield from _bare_column_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _bare_column_keys(child)


def _walk_expressions(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if value.get("type") in {"column", "literal", "aggregate"}:
            yield value
            if value.get("type") == "aggregate" and value.get("argument") is not None:
                yield from _walk_expressions(value["argument"])
            return
        for child in value.values():
            yield from _walk_expressions(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_expressions(child)


def _compatible_types(left: str, right: str, *, operator: str) -> bool:
    if operator in {"like", "not_like"}:
        return left in {"text", "date"} and right == "text"
    if "null" in {left, right}:
        return False
    if left in _NUMERIC_TYPES and right in _NUMERIC_TYPES:
        return True
    if left in {"text", "date"} and right in {"text", "date"}:
        return True
    return left == right


def _validate_predicate_types(
    predicate: Mapping[str, Any], aliases: Mapping[str, str], *, phase: str
) -> None:
    if phase == "where" and _contains_aggregate(predicate):
        raise QueryPlanValidationError("WHERE cannot contain an aggregate")
    kind = predicate["type"]
    if kind == "boolean":
        for child in predicate["operands"]:
            _validate_predicate_types(child, aliases, phase=phase)
        return
    if kind == "comparison":
        left = _expression_type(predicate["left"], aliases)
        right = _expression_type(predicate["right"], aliases)
        if not _compatible_types(left, right, operator=predicate["operator"]):
            raise QueryPlanValidationError(
                f"incompatible comparison types: {left!r} and {right!r}"
            )
        return
    if kind == "in":
        target = _expression_type(predicate["expr"], aliases)
        for literal in predicate["values"]:
            value_type = _expression_type(literal, aliases)
            if not _compatible_types(target, value_type, operator="eq"):
                raise QueryPlanValidationError(
                    f"IN literal type {value_type!r} is incompatible with {target!r}"
                )
        return
    if kind == "between":
        target = _expression_type(predicate["expr"], aliases)
        for bound in (predicate["lower"], predicate["upper"]):
            bound_type = _expression_type(bound, aliases)
            if not _compatible_types(target, bound_type, operator="gte"):
                raise QueryPlanValidationError(
                    f"BETWEEN bound type {bound_type!r} is incompatible with {target!r}"
                )
        return
    if kind == "is_null":
        _expression_type(predicate["expr"], aliases)
        return
    raise AssertionError(f"unreachable predicate kind: {kind!r}")


def _normalise_plan(value: Any) -> dict[str, Any]:
    _budget_check(value)
    obj = _expect_object(
        value,
        path="$",
        required=("plan_version", "from", "select"),
        optional=(
            "joins",
            "distinct",
            "where",
            "group_by",
            "having",
            "order_by",
            "limit",
        ),
    )
    version = obj["plan_version"]
    if isinstance(version, bool) or version != PLAN_VERSION:
        raise QueryPlanValidationError(
            f"$.plan_version must be the integer {PLAN_VERSION}"
        )

    source_obj = _expect_object(
        obj["from"], path="$.from", required=("table", "alias")
    )
    table = source_obj["table"]
    if not isinstance(table, str) or table not in TABLE_COLUMNS:
        raise QueryPlanValidationError(f"$.from.table is unknown: {table!r}")
    source = {
        "table": table,
        "alias": _identifier(source_obj["alias"], path="$.from.alias"),
    }

    joins_raw = _expect_list(
        obj.get("joins", []), path="$.joins", maximum=MAX_JOINS
    )
    joins: list[dict[str, str]] = []
    for index, raw_join in enumerate(joins_raw):
        join_obj = _expect_object(
            raw_join,
            path=f"$.joins[{index}]",
            required=("path", "left_alias", "right_alias"),
        )
        join_path = join_obj["path"]
        if not isinstance(join_path, str) or join_path not in JOIN_PATHS:
            raise QueryPlanValidationError(
                f"$.joins[{index}].path is unknown: {join_path!r}"
            )
        joins.append(
            {
                "path": join_path,
                "left_alias": _identifier(
                    join_obj["left_alias"], path=f"$.joins[{index}].left_alias"
                ),
                "right_alias": _identifier(
                    join_obj["right_alias"], path=f"$.joins[{index}].right_alias"
                ),
            }
        )
    aliases = _resolve_aliases(source, joins)

    select_raw = _expect_list(
        obj["select"],
        path="$.select",
        minimum=1,
        maximum=MAX_SELECT_ITEMS,
    )
    select: list[dict[str, Any]] = []
    output_aliases: set[str] = set()
    for index, raw_item in enumerate(select_raw):
        item = _expect_object(
            raw_item,
            path=f"$.select[{index}]",
            required=("expr",),
            optional=("alias",),
        )
        normalised_item: dict[str, Any] = {
            "expr": _normalise_expression(
                item["expr"], path=f"$.select[{index}].expr"
            )
        }
        if "alias" in item:
            alias = _identifier(item["alias"], path=f"$.select[{index}].alias")
            folded = alias.casefold()
            if folded in output_aliases:
                raise QueryPlanValidationError(
                    f"duplicate SELECT alias is forbidden: {alias!r}"
                )
            output_aliases.add(folded)
            normalised_item["alias"] = alias
        select.append(normalised_item)

    distinct = obj.get("distinct", False)
    if not isinstance(distinct, bool):
        raise QueryPlanValidationError("$.distinct must be a boolean")

    where_raw = obj.get("where")
    where = (
        None
        if where_raw is None
        else _normalise_predicate(where_raw, path="$.where")
    )
    group_raw = _expect_list(
        obj.get("group_by", []), path="$.group_by", maximum=MAX_GROUP_ITEMS
    )
    group_by = [
        _normalise_column(item, path=f"$.group_by[{index}]")
        for index, item in enumerate(group_raw)
    ]
    group_keys = {
        (item["table_alias"], item["column"])
        for item in group_by
    }
    if len(group_keys) != len(group_by):
        raise QueryPlanValidationError("duplicate GROUP BY expression is forbidden")

    having_raw = obj.get("having")
    having = (
        None
        if having_raw is None
        else _normalise_predicate(having_raw, path="$.having")
    )

    order_raw = _expect_list(
        obj.get("order_by", []), path="$.order_by", maximum=MAX_ORDER_ITEMS
    )
    order_by: list[dict[str, Any]] = []
    for index, raw_item in enumerate(order_raw):
        item = _expect_object(
            raw_item,
            path=f"$.order_by[{index}]",
            required=("expr", "direction"),
        )
        direction = item["direction"]
        if not isinstance(direction, str) or direction not in {"asc", "desc"}:
            raise QueryPlanValidationError(
                f"$.order_by[{index}].direction is unknown: {direction!r}"
            )
        order_by.append(
            {
                "expr": _normalise_expression(
                    item["expr"], path=f"$.order_by[{index}].expr"
                ),
                "direction": direction,
            }
        )

    limit = obj.get("limit")
    if limit is not None:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_LIMIT
        ):
            raise QueryPlanValidationError(
                f"$.limit must be null or an integer from 1 through {MAX_LIMIT}"
            )

    normalised = {
        "plan_version": PLAN_VERSION,
        "from": source,
        "joins": joins,
        "select": select,
        "distinct": distinct,
        "where": where,
        "group_by": group_by,
        "having": having,
        "order_by": order_by,
        "limit": limit,
    }

    for expression in _walk_expressions(normalised):
        _expression_type(expression, aliases)
    if where is not None:
        _validate_predicate_types(where, aliases, phase="where")
    if having is not None:
        _validate_predicate_types(having, aliases, phase="having")
        if not group_by and not _contains_aggregate(having):
            raise QueryPlanValidationError(
                "HAVING requires GROUP BY or an aggregate expression"
            )

    aggregate_query = any(
        _contains_aggregate(item["expr"]) for item in select
    ) or _contains_aggregate(having) or any(
        _contains_aggregate(item["expr"]) for item in order_by
    )
    if group_by or aggregate_query:
        grouped_surfaces = [item["expr"] for item in select]
        grouped_surfaces.extend(item["expr"] for item in order_by)
        if having is not None:
            grouped_surfaces.append(having)
        for surface in grouped_surfaces:
            for key in _bare_column_keys(surface):
                if key not in group_keys:
                    raise QueryPlanValidationError(
                        "every non-aggregate SELECT/HAVING/ORDER column must appear "
                        "in GROUP BY"
                    )
    return normalised


def parse_query_plan(raw: str | bytes | bytearray) -> dict[str, Any]:
    """Parse untrusted strict JSON and return a normalized QueryPlan v1.

    Unlike ``json.loads`` defaults, this rejects duplicate object keys and the
    JavaScript extensions ``NaN``/``Infinity``.  It accepts no Mapping input so
    callers cannot accidentally skip the raw-JSON boundary; use
    :func:`compile_query_plan` for a programmatically built mapping.
    """

    if isinstance(raw, (bytes, bytearray)):
        if len(raw) > MAX_RAW_JSON_BYTES:
            raise QueryPlanJSONError(
                f"query plan JSON exceeds {MAX_RAW_JSON_BYTES} bytes"
            )
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise QueryPlanJSONError("query plan must be UTF-8 JSON") from exc
    elif isinstance(raw, str):
        text = raw
        if len(text.encode("utf-8")) > MAX_RAW_JSON_BYTES:
            raise QueryPlanJSONError(
                f"query plan JSON exceeds {MAX_RAW_JSON_BYTES} bytes"
            )
    else:
        raise QueryPlanJSONError("raw query plan must be a JSON string or UTF-8 bytes")
    if text.startswith("\ufeff"):
        raise QueryPlanJSONError("a UTF-8 BOM is not permitted in query plan JSON")
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except QueryPlanJSONError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise QueryPlanJSONError(f"invalid strict query plan JSON: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise QueryPlanJSONError("query plan JSON must decode to one object")
    return _normalise_plan(decoded)


def _coerce_plan(plan: Any) -> dict[str, Any]:
    if isinstance(plan, (str, bytes, bytearray)):
        return parse_query_plan(plan)
    return _normalise_plan(plan)


def canonical_plan_json(plan: Any) -> str:
    """Return canonical UTF-8 JSON text after full QueryPlan validation."""

    normalised = _coerce_plan(plan)
    return json.dumps(
        normalised,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_plan_sha256(plan: Any) -> str:
    """Return the SHA-256 of canonical normalized QueryPlan JSON."""

    return hashlib.sha256(canonical_plan_json(plan).encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


_ARABIC_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z0-9_.])"
)
_DATE_RE = re.compile(
    r"(?<!\d)(\d{4})\s*(?:年|[-/.])\s*(\d{1,2})\s*(?:月|[-/.])\s*(\d{1,2})\s*日?(?!\d)"
)
_CHINESE_NUMBER_RE = re.compile(r"[负零〇一二两三四五六七八九十百千]+")
_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000}


def _parse_chinese_integer(token: str) -> int | None:
    negative = token.startswith("负")
    if negative:
        token = token[1:]
    if not token:
        return None
    if all(char in _CHINESE_DIGITS for char in token):
        digits = "".join(str(_CHINESE_DIGITS[char]) for char in token)
        result = int(digits)
        return -result if negative else result
    total = 0
    current = 0
    for char in token:
        if char in _CHINESE_DIGITS:
            current = _CHINESE_DIGITS[char]
        elif char in _CHINESE_UNITS:
            unit = _CHINESE_UNITS[char]
            total += (current or 1) * unit
            current = 0
        else:
            return None
    result = total + current
    return -result if negative else result


def _question_dates(question: str) -> set[date]:
    result: set[date] = set()
    for match in _DATE_RE.finditer(question):
        try:
            result.add(date(*(int(part) for part in match.groups())))
        except ValueError:
            continue
    return result


def _question_numbers(question: str) -> set[Decimal]:
    result: set[Decimal] = set()
    for match in _ARABIC_NUMBER_RE.finditer(question):
        parsed = _decimal(match.group(0))
        if parsed is not None:
            result.add(parsed)
            suffix = question[match.end() : match.end() + 1]
            if suffix in {"%", "％"}:
                result.add(parsed / Decimal(100))
    for match in _CHINESE_NUMBER_RE.finditer(question):
        token = match.group(0)
        before = question[max(0, match.start() - 2) : match.start()]
        after = question[match.end() : match.end() + 1]
        has_unit = any(character in _CHINESE_UNITS for character in token)
        prefixed = before.endswith(
            ("前", "第", "约", "近", "为", "共", "取", "限", "至少", "至多")
        )
        suffix_context = after in {
            "个",
            "条",
            "位",
            "名",
            "次",
            "天",
            "岁",
            "元",
            "例",
            "种",
            "项",
            "家",
            "份",
            "人",
            "遍",
            "级",
            "页",
            "章",
            "年",
            "月",
            "日",
            "%",
            "％",
        }
        terminated_unit = has_unit and (
            not after or after.isspace() or after in "，。；：、,.;:!?！？)）]】"
        )
        # A bare Chinese digit in an ordinary word (for example ``一致`` or
        # ``唯一``) is not numeric evidence.  Require a count/rank context.
        if not (prefixed or suffix_context or terminated_unit):
            continue
        parsed = _parse_chinese_integer(token)
        if parsed is not None:
            result.add(Decimal(parsed))
    return result


def _string_grounded(
    raw: str,
    *,
    compact_question: str,
    numbers: set[Decimal],
    dates: set[date],
) -> bool:
    candidate = re.sub(
        r"\s+", "", unicodedata.normalize("NFKC", raw).casefold()
    )
    # SQL LIKE wildcards describe matching behaviour, not words the user had
    # to utter.  Backslash is not accepted as a custom escape language in v1.
    candidate = candidate.replace("%", "").replace("_", "")
    if not candidate:
        return False
    try:
        parsed_date = date.fromisoformat(candidate)
    except ValueError:
        parsed_date = None
    if parsed_date is not None:
        return parsed_date in dates
    parsed_number = _decimal(candidate)
    if parsed_number is not None:
        return parsed_number in numbers
    return candidate in compact_question


def _iter_literals(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if value.get("type") == "literal":
            yield value
            return
        for child in value.values():
            yield from _iter_literals(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_literals(child)


def validate_plan_grounding(
    plan: Any,
    question: str,
    *,
    allowed_domain_values: Iterable[Any] = (),
) -> None:
    """Require every string/number literal and LIMIT to occur in ``question``.

    ``allowed_domain_values`` is accepted so integration code can pass its
    existing domain catalogue, but it is intentionally *not* an authorization
    source.  A value found only in the database still fails this check.  The
    trusted constants embedded in fixed join paths do not appear as plan
    literals and therefore need no grounding exemption at runtime.
    """

    normalised = _coerce_plan(plan)
    if not isinstance(question, str) or not question.strip():
        raise QueryPlanGroundingError("question must be a non-empty string")
    # Explicitly consume the argument without consulting its values.  Keeping
    # it out of the decision is the security property this API promises.
    del allowed_domain_values
    normalised_question = unicodedata.normalize("NFKC", question).casefold()
    compact_question = re.sub(r"\s+", "", normalised_question)
    numbers = _question_numbers(normalised_question)
    dates = _question_dates(normalised_question)
    missing: list[str] = []
    seen: set[tuple[str, str]] = set()
    for literal in _iter_literals(normalised):
        kind = literal["value_type"]
        raw = literal["value"]
        if kind in {"null", "boolean"}:
            continue
        identity = (kind, repr(raw))
        if identity in seen:
            continue
        seen.add(identity)
        if kind in _NUMERIC_TYPES:
            expected = _decimal(raw)
            grounded = expected is not None and expected in numbers
        else:
            grounded = _string_grounded(
                str(raw),
                compact_question=compact_question,
                numbers=numbers,
                dates=dates,
            )
        if not grounded:
            missing.append(f"{kind}:{raw!r}")
    if normalised["limit"] is not None:
        limit_identity = ("limit", str(normalised["limit"]))
        if limit_identity not in seen and Decimal(normalised["limit"]) not in numbers:
            missing.append(f"limit:{normalised['limit']!r}")
    if missing:
        raise QueryPlanGroundingError(
            "query plan literal(s) are not grounded in the question: "
            + ", ".join(missing)
        )


def _sqlglot_import():
    try:
        from sqlglot import exp
    except ImportError as exc:  # never degrade to string interpolation
        raise RuntimeError("sqlglot is required for QueryPlan compilation") from exc
    return exp


def _ast_identifier(exp: Any, name: str) -> Any:
    return exp.Identifier(this=name, quoted=True)


def _ast_column(exp: Any, expression: Mapping[str, Any]) -> Any:
    return exp.Column(
        this=_ast_identifier(exp, expression["column"]),
        table=_ast_identifier(exp, expression["table_alias"]),
    )


def _ast_literal(exp: Any, literal: Mapping[str, Any]) -> Any:
    kind = literal["value_type"]
    value = literal["value"]
    if kind in {"string", "date"}:
        return exp.Literal.string(value)
    if kind == "integer":
        return exp.Literal.number(str(value))
    if kind == "number":
        return exp.Literal.number(repr(value))
    if kind == "boolean":
        return exp.Boolean(this=value)
    if kind == "null":
        return exp.Null()
    raise AssertionError(f"unreachable literal kind: {kind!r}")


def _ast_expression(exp: Any, expression: Mapping[str, Any]) -> Any:
    kind = expression["type"]
    if kind == "column":
        return _ast_column(exp, expression)
    if kind == "literal":
        return _ast_literal(exp, expression)
    function = expression["function"]
    argument = expression["argument"]
    argument_ast = exp.Star() if argument is None else _ast_column(exp, argument)
    if expression["distinct"]:
        argument_ast = exp.Distinct(expressions=[argument_ast])
    function_class = {
        "count": exp.Count,
        "sum": exp.Sum,
        "avg": exp.Avg,
        "min": exp.Min,
        "max": exp.Max,
    }[function]
    return function_class(this=argument_ast)


def _combine_boolean(exp: Any, operator: str, operands: Sequence[Any]) -> Any:
    node = exp.And if operator == "and" else exp.Or
    return reduce(lambda left, right: node(this=left, expression=right), operands)


def _ast_predicate(exp: Any, predicate: Mapping[str, Any]) -> Any:
    kind = predicate["type"]
    if kind == "boolean":
        return _combine_boolean(
            exp,
            predicate["operator"],
            [_ast_predicate(exp, item) for item in predicate["operands"]],
        )
    if kind == "comparison":
        left = _ast_expression(exp, predicate["left"])
        right = _ast_expression(exp, predicate["right"])
        operator = predicate["operator"]
        if operator == "not_like":
            return exp.Not(this=exp.Like(this=left, expression=right))
        operator_class = {
            "eq": exp.EQ,
            "neq": exp.NEQ,
            "gt": exp.GT,
            "gte": exp.GTE,
            "lt": exp.LT,
            "lte": exp.LTE,
            "like": exp.Like,
        }[operator]
        return operator_class(this=left, expression=right)
    if kind == "in":
        result = exp.In(
            this=_ast_expression(exp, predicate["expr"]),
            expressions=[_ast_literal(exp, value) for value in predicate["values"]],
        )
        return exp.Not(this=result) if predicate["negated"] else result
    if kind == "between":
        result = exp.Between(
            this=_ast_expression(exp, predicate["expr"]),
            low=_ast_literal(exp, predicate["lower"]),
            high=_ast_literal(exp, predicate["upper"]),
        )
        return exp.Not(this=result) if predicate["negated"] else result
    if kind == "is_null":
        result = exp.Is(
            this=_ast_expression(exp, predicate["expr"]), expression=exp.Null()
        )
        return exp.Not(this=result) if predicate["negated"] else result
    raise AssertionError(f"unreachable predicate kind: {kind!r}")


def _ast_table(exp: Any, table: str, alias: str) -> Any:
    return exp.Table(
        this=_ast_identifier(exp, table),
        alias=exp.TableAlias(this=_ast_identifier(exp, alias)),
    )


def _join_on(exp: Any, join: Mapping[str, Any]) -> Any:
    path = JOIN_PATHS[join["path"]]
    left_alias = join["left_alias"]
    right_alias = join["right_alias"]
    predicates = [
        exp.EQ(
            this=_ast_column(
                exp,
                {"table_alias": left_alias, "column": left_column},
            ),
            expression=_ast_column(
                exp,
                {"table_alias": right_alias, "column": right_column},
            ),
        )
        for left_column, right_column in path.column_pairs
    ]
    for side, column, kind, value in path.constants:
        predicates.append(
            exp.EQ(
                this=_ast_column(
                    exp,
                    {
                        "table_alias": left_alias if side == "left" else right_alias,
                        "column": column,
                    },
                ),
                expression=_ast_literal(
                    exp,
                    {"value_type": kind, "value": value},
                ),
            )
        )
    return _combine_boolean(exp, "and", predicates) if len(predicates) > 1 else predicates[0]


def compile_query_plan(plan: Any) -> str:
    """Compile a QueryPlan v1 mapping (or raw JSON) to one SQLite SELECT.

    Validation is intentionally repeated here so mutating the dict returned by
    :func:`parse_query_plan` cannot bypass any invariant.
    """

    normalised = _coerce_plan(plan)
    exp = _sqlglot_import()
    aliases = {normalised["from"]["alias"]: normalised["from"]["table"]}
    projections = []
    for item in normalised["select"]:
        expression = _ast_expression(exp, item["expr"])
        if "alias" in item:
            expression = exp.Alias(
                this=expression,
                alias=_ast_identifier(exp, item["alias"]),
            )
        projections.append(expression)
    tree = exp.Select(expressions=projections)
    if normalised["distinct"]:
        tree.set("distinct", exp.Distinct())
    tree.set(
        "from_",
        exp.From(
            this=_ast_table(
                exp,
                normalised["from"]["table"],
                normalised["from"]["alias"],
            )
        ),
    )

    joins_ast = []
    for join in normalised["joins"]:
        path = JOIN_PATHS[join["path"]]
        if join["left_alias"] in aliases:
            new_alias = join["right_alias"]
            new_table = path.right_table
        else:
            new_alias = join["left_alias"]
            new_table = path.left_table
        joins_ast.append(
            exp.Join(
                this=_ast_table(exp, new_table, new_alias),
                on=_join_on(exp, join),
                kind="INNER",
            )
        )
        aliases[new_alias] = new_table
    if joins_ast:
        tree.set("joins", joins_ast)
    if normalised["where"] is not None:
        tree.set("where", exp.Where(this=_ast_predicate(exp, normalised["where"])))
    if normalised["group_by"]:
        tree.set(
            "group",
            exp.Group(
                expressions=[
                    _ast_column(exp, expression)
                    for expression in normalised["group_by"]
                ]
            ),
        )
    if normalised["having"] is not None:
        tree.set(
            "having",
            exp.Having(this=_ast_predicate(exp, normalised["having"])),
        )
    if normalised["order_by"]:
        tree.set(
            "order",
            exp.Order(
                expressions=[
                    exp.Ordered(
                        this=_ast_expression(exp, item["expr"]),
                        desc=item["direction"] == "desc",
                    )
                    for item in normalised["order_by"]
                ]
            ),
        )
    if normalised["limit"] is not None:
        tree.set(
            "limit",
            exp.Limit(expression=exp.Literal.number(str(normalised["limit"]))),
        )
    try:
        sql = tree.sql(dialect="sqlite", identify=True, pretty=False).strip()
        if not sql:
            raise QueryPlanCompileError("SQLGlot rendered an empty query")
        validate_gold_sql(sql)
    except QueryPlanError:
        raise
    except Exception as exc:
        raise QueryPlanCompileError(f"failed to compile QueryPlan v1: {exc}") from exc
    return sql


__all__ = [
    "JOIN_PATHS",
    "MAX_LIMIT",
    "PLAN_VERSION",
    "QueryPlanCompileError",
    "QueryPlanError",
    "QueryPlanGroundingError",
    "QueryPlanJSONError",
    "QueryPlanValidationError",
    "TABLE_COLUMNS",
    "canonical_plan_json",
    "canonical_plan_sha256",
    "compile_query_plan",
    "parse_query_plan",
    "query_plan_schema",
    "validate_plan_grounding",
]
