"""Bounded semantic QueryPlan v3 for the four-table medical schema.

V3 wraps the proven QueryPlan v2 relational algebra with a required semantic
intent. The intent is executable policy, not advisory metadata: compilation
fails when the query contradicts an explicit claim. There is deliberately no
SQL, raw-SQL, or free-form expression escape hatch.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from dki.analysis.nl2sql_contract import (
    ContractStatus,
    RowSetMode,
    SemanticContract,
    extract_semantic_contract,
    validate_candidate,
)
from dki.analysis.nl2sql_query_plan_v2 import (
    QueryPlanV2Error,
    compile_query_plan_v2,
    parse_query_plan_v2,
    query_plan_v2_schema,
    validate_plan_v2_grounding,
)
from dki.analysis.nl2sql_schema_graph import (
    canonical_schema_graph_sha256,
    schema_graph,
)
from dki.analysis.nl2sql_semantic_obligations import (
    ObligationDisposition,
    SemanticObligations,
    extract_semantic_obligations,
    validate_candidate_obligations,
)

PLAN_VERSION = 3
MAX_RAW_JSON_BYTES = 64 * 1024
MAX_LIST = 32
MAX_TEXT = 128
MAX_PRECISION = 12
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,31}\Z")
_INTENT_FIELDS = {
    "output", "grain", "joins", "entity_preservation", "aggregation",
    "set_semantics", "null_semantics", "tie_policy", "ordering", "precision",
}


class QueryPlanV3Error(ValueError):
    """Base class for fail-closed QueryPlan v3 errors."""


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise QueryPlanV3Error(f"{path} contains unsupported fields: {sorted(unknown)}")


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QueryPlanV3Error(f"{path} must be an object")
    return value


def _list(value: Any, path: str, *, minimum: int = 0, maximum: int = MAX_LIST) -> list[Any]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise QueryPlanV3Error(f"{path} must contain {minimum}..{maximum} items")
    return value


def _enum(value: Any, choices: set[str], path: str) -> str:
    if value not in choices:
        raise QueryPlanV3Error(f"{path} must be one of {sorted(choices)}")
    return str(value)


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise QueryPlanV3Error(f"{path} must be a safe identifier")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= MAX_TEXT:
        raise QueryPlanV3Error(f"{path} must be a bounded non-empty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise QueryPlanV3Error(f"{path} must be a boolean")
    return value


def _strict_json(raw: str | bytes | bytearray) -> Any:
    if isinstance(raw, (bytes, bytearray)):
        if len(raw) > MAX_RAW_JSON_BYTES:
            raise QueryPlanV3Error("QueryPlan v3 JSON is too large")
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise QueryPlanV3Error("QueryPlan v3 JSON must be UTF-8") from exc
    else:
        text = str(raw)
        if len(text.encode("utf-8")) > MAX_RAW_JSON_BYTES:
            raise QueryPlanV3Error("QueryPlan v3 JSON is too large")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I).strip()

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise QueryPlanV3Error(f"duplicate JSON key is forbidden: {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> Any:
        raise QueryPlanV3Error(f"non-finite JSON number is forbidden: {token}")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except QueryPlanV3Error:
        raise
    except Exception as exc:
        raise QueryPlanV3Error("invalid QueryPlan v3 JSON") from exc


def _parse_output(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    _reject_unknown(obj, {"columns", "row_set_mode"}, path)
    columns = [
        _text(item, f"{path}.columns[{index}]")
        for index, item in enumerate(_list(obj.get("columns"), f"{path}.columns", minimum=1))
    ]
    if len(set(columns)) != len(columns):
        raise QueryPlanV3Error(f"{path}.columns contains duplicates")
    return {
        "columns": columns,
        "row_set_mode": _enum(
            obj.get("row_set_mode"), {"preserve", "distinct"}, f"{path}.row_set_mode"
        ),
    }


def _parse_grain(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    _reject_unknown(obj, {"kind", "keys"}, path)
    kind = _enum(obj.get("kind"), {"scalar", "rows", "groups"}, f"{path}.kind")
    keys = [
        _text(item, f"{path}.keys[{index}]")
        for index, item in enumerate(_list(obj.get("keys"), f"{path}.keys", maximum=16))
    ]
    if len(set(keys)) != len(keys):
        raise QueryPlanV3Error(f"{path}.keys contains duplicates")
    if kind == "scalar" and keys:
        raise QueryPlanV3Error(f"{path}.keys must be empty for scalar grain")
    if kind == "groups" and not keys:
        raise QueryPlanV3Error(f"{path}.keys is required for group grain")
    return {"kind": kind, "keys": keys}


def _parse_joins(value: Any, path: str) -> list[dict[str, Any]]:
    graph_edges = schema_graph()["edges"]
    result: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for index, item in enumerate(_list(value, path, maximum=8)):
        item_path = f"{path}[{index}]"
        obj = _object(item, item_path)
        _reject_unknown(
            obj, {"edge", "left_alias", "right_alias", "kind", "preserve"}, item_path
        )
        edge = obj.get("edge")
        if edge not in graph_edges:
            raise QueryPlanV3Error(f"{item_path}.edge is not in the schema graph")
        parsed = {
            "edge": edge,
            "left_alias": _identifier(obj.get("left_alias"), f"{item_path}.left_alias"),
            "right_alias": _identifier(obj.get("right_alias"), f"{item_path}.right_alias"),
            "kind": _enum(obj.get("kind"), {"inner", "left"}, f"{item_path}.kind"),
            "preserve": _enum(obj.get("preserve"), {"none", "left"}, f"{item_path}.preserve"),
        }
        if parsed["kind"] == "inner" and parsed["preserve"] != "none":
            raise QueryPlanV3Error(f"{item_path}.preserve must be none for inner joins")
        if parsed["kind"] == "left" and parsed["preserve"] != "left":
            raise QueryPlanV3Error(f"{item_path}.preserve must be left for left joins")
        identity = (parsed["edge"], parsed["left_alias"], parsed["right_alias"])
        if identity in identities:
            raise QueryPlanV3Error(f"{path} contains a duplicate join")
        identities.add(identity)
        result.append(parsed)
    return result


def _parse_entity_preservation(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    _reject_unknown(obj, {"mode", "alias", "include_zero_match"}, path)
    mode = _enum(obj.get("mode"), {"none", "preserve"}, f"{path}.mode")
    include_zero = _boolean(obj.get("include_zero_match"), f"{path}.include_zero_match")
    alias = obj.get("alias")
    if mode == "none":
        if alias is not None or include_zero:
            raise QueryPlanV3Error(f"{path} cannot preserve an entity in none mode")
        parsed_alias = None
    else:
        parsed_alias = _identifier(alias, f"{path}.alias")
    return {"mode": mode, "alias": parsed_alias, "include_zero_match": include_zero}


def _parse_aggregation(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    _reject_unknown(obj, {"stage", "distinct_scope"}, path)
    return {
        "stage": _enum(
            obj.get("stage"), {"none", "base_rows", "after_join", "per_group"},
            f"{path}.stage",
        ),
        "distinct_scope": _enum(
            obj.get("distinct_scope"), {"none", "rows", "aggregate_arguments"},
            f"{path}.distinct_scope",
        ),
    }


def _parse_set_semantics(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    _reject_unknown(obj, {"mode", "preserve_duplicates"}, path)
    mode = _enum(
        obj.get("mode"), {"none", "any_of", "all_of", "difference"}, f"{path}.mode"
    )
    preserve = _boolean(obj.get("preserve_duplicates"), f"{path}.preserve_duplicates")
    if mode != "any_of" and preserve:
        raise QueryPlanV3Error(f"{path}.preserve_duplicates is valid only for any_of")
    return {"mode": mode, "preserve_duplicates": preserve}


def _parse_tie_policy(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    _reject_unknown(obj, {"mode", "keys"}, path)
    mode = _enum(
        obj.get("mode"), {"unspecified", "deterministic", "include_ties"}, f"{path}.mode"
    )
    keys = [
        _text(item, f"{path}.keys[{index}]")
        for index, item in enumerate(_list(obj.get("keys"), f"{path}.keys"))
    ]
    if len(set(keys)) != len(keys):
        raise QueryPlanV3Error(f"{path}.keys contains duplicates")
    if mode == "deterministic" and not keys:
        raise QueryPlanV3Error(f"{path}.keys is required for deterministic ties")
    if mode != "deterministic" and keys:
        raise QueryPlanV3Error(f"{path}.keys is valid only for deterministic ties")
    return {"mode": mode, "keys": keys}


def _parse_ordering(value: Any, path: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(_list(value, path)):
        item_path = f"{path}[{index}]"
        obj = _object(item, item_path)
        _reject_unknown(obj, {"key", "direction", "nulls"}, item_path)
        result.append(
            {
                "key": _text(obj.get("key"), f"{item_path}.key"),
                "direction": _enum(
                    obj.get("direction"), {"asc", "desc"}, f"{item_path}.direction"
                ),
                "nulls": _enum(
                    obj.get("nulls"), {"unspecified", "first", "last"},
                    f"{item_path}.nulls",
                ),
            }
        )
    return result


def _parse_precision(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    _reject_unknown(obj, {"mode", "digits"}, path)
    mode = _enum(obj.get("mode"), {"exact", "round"}, f"{path}.mode")
    digits = obj.get("digits")
    if mode == "exact":
        if digits is not None:
            raise QueryPlanV3Error(f"{path}.digits must be null for exact precision")
        parsed_digits = None
    else:
        if (
            isinstance(digits, bool) or not isinstance(digits, int)
            or not 0 <= digits <= MAX_PRECISION
        ):
            raise QueryPlanV3Error(
                f"{path}.digits must be an integer in [0, {MAX_PRECISION}]"
            )
        parsed_digits = digits
    return {"mode": mode, "digits": parsed_digits}


def _parse_intent(value: Any, path: str = "$.intent") -> dict[str, Any]:
    obj = _object(value, path)
    _reject_unknown(obj, _INTENT_FIELDS, path)
    missing = _INTENT_FIELDS - set(obj)
    if missing:
        raise QueryPlanV3Error(f"{path} is missing required fields: {sorted(missing)}")
    nulls = _object(obj["null_semantics"], f"{path}.null_semantics")
    _reject_unknown(nulls, {"mode"}, f"{path}.null_semantics")
    return {
        "output": _parse_output(obj["output"], f"{path}.output"),
        "grain": _parse_grain(obj["grain"], f"{path}.grain"),
        "joins": _parse_joins(obj["joins"], f"{path}.joins"),
        "entity_preservation": _parse_entity_preservation(
            obj["entity_preservation"], f"{path}.entity_preservation"
        ),
        "aggregation": _parse_aggregation(obj["aggregation"], f"{path}.aggregation"),
        "set_semantics": _parse_set_semantics(obj["set_semantics"], f"{path}.set_semantics"),
        "null_semantics": {
            "mode": _enum(
                nulls.get("mode"), {"unspecified", "include", "exclude", "only"},
                f"{path}.null_semantics.mode",
            )
        },
        "tie_policy": _parse_tie_policy(obj["tie_policy"], f"{path}.tie_policy"),
        "ordering": _parse_ordering(obj["ordering"], f"{path}.ordering"),
        "precision": _parse_precision(obj["precision"], f"{path}.precision"),
    }


def parse_query_plan_v3(raw: str | bytes | bytearray | Mapping[str, Any]) -> dict[str, Any]:
    """Parse untrusted QueryPlan v3 JSON into one canonical mapping."""

    obj = dict(raw) if isinstance(raw, Mapping) else _strict_json(raw)
    if not isinstance(obj, Mapping) or obj.get("plan_version") != PLAN_VERSION:
        raise QueryPlanV3Error("plan_version must be 3")
    _reject_unknown(obj, {"plan_version", "intent", "query"}, "$")
    if set(obj) != {"plan_version", "intent", "query"}:
        raise QueryPlanV3Error("QueryPlan v3 requires plan_version, intent, and query")
    intent = _parse_intent(obj["intent"])
    query = _object(obj["query"], "$.query")
    try:
        parsed_v2 = parse_query_plan_v2({"plan_version": 2, "query": query})
    except QueryPlanV2Error as exc:
        raise QueryPlanV3Error(f"invalid QueryPlan v3 query: {exc}") from exc
    parsed = {"plan_version": PLAN_VERSION, "intent": intent, "query": parsed_v2["query"]}
    _validate_query_intent_structure(parsed)
    try:
        sql = compile_query_plan_v2(parsed_v2)
    except QueryPlanV2Error as exc:
        raise QueryPlanV3Error(str(exc)) from exc
    result = validate_candidate(_intent_contract(intent), sql)
    if result.reasons:
        raise QueryPlanV3Error("; ".join(result.reasons))
    return parsed


def _visit(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _visit(child)
    elif isinstance(value, list):
        for child in value:
            yield from _visit(child)


def _select_queries(query: Mapping[str, Any]):
    if query["type"] == "set_op":
        yield from _select_queries(query["left"])
        yield from _select_queries(query["right"])
        return
    yield query
    for cte in query["ctes"]:
        yield from _select_queries(cte["query"])
    if query["from"]["type"] == "subquery":
        yield from _select_queries(query["from"]["query"])
    for join in query["joins"]:
        if join["source"]["type"] == "subquery":
            yield from _select_queries(join["source"]["query"])


def _query_has(query: Mapping[str, Any], node_type: str) -> bool:
    return any(node.get("type") == node_type for node in _visit(query))


def _query_has_aggregate_distinct(query: Mapping[str, Any]) -> bool:
    return any(
        node.get("type") == "aggregate" and node.get("distinct")
        for node in _visit(query)
    )


def _null_predicates(query: Mapping[str, Any]) -> tuple[bool, bool]:
    is_null = is_not_null = False
    for node in _visit(query):
        if node.get("type") != "binary" or node.get("operator") not in {"is", "is_not"}:
            continue
        sides = (node.get("left"), node.get("right"))
        if not any(
            isinstance(side, Mapping)
            and side.get("type") == "literal"
            and side.get("value") is None
            for side in sides
        ):
            continue
        is_null |= node["operator"] == "is"
        is_not_null |= node["operator"] == "is_not"
    return is_null, is_not_null


def _round_precisions(query: Mapping[str, Any]) -> list[int]:
    values: list[int] = []
    for node in _visit(query):
        if node.get("type") != "function" or node.get("function") != "round":
            continue
        args = node.get("args", [])
        if (
            len(args) == 2 and args[1].get("type") == "literal"
            and isinstance(args[1].get("value"), int)
            and not isinstance(args[1].get("value"), bool)
        ):
            values.append(args[1]["value"])
        else:
            values.append(0)
    return values


def _normalise_semantic(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).casefold()


def _expr_semantic(expr: Mapping[str, Any]) -> str | None:
    if expr.get("type") == "column":
        return str(expr["column"]).casefold()
    if expr.get("type") == "aggregate":
        argument = expr.get("argument")
        argument_text = "*" if argument is None else _expr_semantic(argument)
        if argument_text is None:
            return None
        distinct = "distinct" if expr.get("distinct") else ""
        return _normalise_semantic(f"{expr['function']}({distinct}{argument_text})")
    if (
        expr.get("type") == "function" and expr.get("function") == "round"
        and expr.get("args")
    ):
        return _expr_semantic(expr["args"][0])
    return None


def _leftmost_select(query: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the branch whose select list names a compound query's outputs."""

    while query["type"] == "set_op":
        query = query["left"]
    return query


def _order_semantic(
    entry: Mapping[str, Any], select_items: Sequence[Mapping[str, Any]]
) -> str | None:
    """Semantic key of one ORDER BY entry in either expression or alias form."""

    if "alias" in entry:
        for item in select_items:
            if item.get("alias") == entry["alias"]:
                return _expr_semantic(item["expr"])
        return None
    return _expr_semantic(entry["expr"])


def _aliases(query: Mapping[str, Any]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    source = query["from"]
    result[source["alias"]] = source.get("table") if source["type"] == "table" else None
    for join in query["joins"]:
        source = join["source"]
        result[source["alias"]] = source.get("table") if source["type"] == "table" else None
    return result


def _eq_pairs(on: Sequence[Mapping[str, Any]]) -> set[frozenset[tuple[str, str]]]:
    result: set[frozenset[tuple[str, str]]] = set()
    for clause in on:
        if (
            clause.get("type") == "binary" and clause.get("operator") == "eq"
            and clause.get("left", {}).get("type") == "column"
            and clause.get("right", {}).get("type") == "column"
        ):
            result.add(
                frozenset({
                    (clause["left"]["table"], clause["left"]["column"]),
                    (clause["right"]["table"], clause["right"]["column"]),
                })
            )
    return result


def _predicate_atoms(value: Any, *, conjunctive: bool = True):
    if not isinstance(value, Mapping):
        return
    if value.get("type") == "binary" and value.get("operator") in {"and", "or"}:
        next_conjunctive = conjunctive and value["operator"] == "and"
        yield from _predicate_atoms(value.get("left"), conjunctive=next_conjunctive)
        yield from _predicate_atoms(value.get("right"), conjunctive=next_conjunctive)
        return
    yield value, conjunctive


def _literal_equalities(query: Mapping[str, Any]) -> set[tuple[str, str, object]]:
    result: set[tuple[str, str, object]] = set()
    predicates = [query.get("where")]
    predicates.extend(clause for join in query["joins"] for clause in join["on"])
    for predicate in predicates:
        for node, conjunctive in _predicate_atoms(predicate):
            if not conjunctive or node.get("type") != "binary" or node.get("operator") != "eq":
                continue
            left, right = node.get("left"), node.get("right")
            if left.get("type") == "column" and right.get("type") == "literal":
                result.add((left["table"], left["column"], right.get("value")))
            if right.get("type") == "column" and left.get("type") == "literal":
                result.add((right["table"], right["column"], left.get("value")))
    return result


def _join_matches(query: Mapping[str, Any], declaration: Mapping[str, Any]) -> bool:
    edge = schema_graph()["edges"][declaration["edge"]]
    aliases = _aliases(query)
    declared_left = declaration["left_alias"]
    declared_right = declaration["right_alias"]
    if (
        aliases.get(declared_left) == edge["left_table"]
        and aliases.get(declared_right) == edge["right_table"]
    ):
        left_alias, right_alias = declared_left, declared_right
    elif (
        aliases.get(declared_left) == edge["right_table"]
        and aliases.get(declared_right) == edge["left_table"]
    ):
        left_alias, right_alias = declared_right, declared_left
    else:
        return False
    introduced_alias = next(
        (
            alias
            for alias in (declared_left, declared_right)
            if any(item["source"]["alias"] == alias for item in query["joins"])
        ),
        None,
    )
    join = next(
        (
            item for item in query["joins"]
            if item["source"]["alias"] == introduced_alias
            and item["kind"] == declaration["kind"]
        ),
        None,
    )
    if join is None:
        return False
    pairs = _eq_pairs(join["on"])
    for pair in edge["join_columns"]:
        if frozenset(
            {(left_alias, pair["left"]), (right_alias, pair["right"])}
        ) not in pairs:
            return False
    literals = _literal_equalities(query)
    for guard in edge["guards"]:
        if guard["kind"] == "column_equality":
            if frozenset(
                {(left_alias, guard["left"]), (right_alias, guard["right"])}
            ) not in pairs:
                return False
        else:
            alias = left_alias if guard["side"] == "left" else right_alias
            if (alias, guard["column"], guard["value"]) not in literals:
                return False
    return True


def _validate_query_intent_structure(plan: Mapping[str, Any]) -> None:
    intent, query = plan["intent"], plan["query"]
    root = query if query["type"] == "select" else None
    declarations = intent["joins"]
    graph_edges = schema_graph()["edges"]

    for declaration in declarations:
        edge = graph_edges[declaration["edge"]]
        if (
            edge["left_to_right"] == "many" and edge["right_to_left"] == "many"
            and intent["aggregation"]["stage"] == "after_join"
            and intent["aggregation"]["distinct_scope"] == "none"
            and any(
                node.get("type") == "aggregate"
                and node.get("function") == "count"
                and node.get("argument") is None
                for node in _visit(query)
            )
        ):
            raise QueryPlanV3Error(
                "many-to-many join can amplify COUNT(*) without explicit distinct protection"
            )

    for declaration in declarations:
        if not any(_join_matches(select, declaration) for select in _select_queries(query)):
            raise QueryPlanV3Error(
                f"declared join {declaration['edge']!r} is missing or lacks a schema guard"
            )

    preservation = intent["entity_preservation"]
    if preservation["mode"] == "preserve" and not any(
        item["kind"] == "left" and item["left_alias"] == preservation["alias"]
        and item["preserve"] == "left" for item in declarations
    ):
        raise QueryPlanV3Error(
            "entity preservation requires a declared left-preserving join"
        )
    if preservation["include_zero_match"]:
        if root is None or not any(join["kind"] == "left" for join in root["joins"]):
            raise QueryPlanV3Error("zero-match preservation requires LEFT JOIN")
        nullable = {
            join["source"]["alias"] for join in root["joins"] if join["kind"] == "left"
        }
        if root["where"] is not None and any(
            node.get("type") == "column" and node.get("table") in nullable
            for node in _visit(root["where"])
        ):
            raise QueryPlanV3Error(
                "WHERE on a nullable join side destroys zero-match rows"
            )
        for node in _visit(root["select"]):
            if (
                node.get("type") == "aggregate"
                and node.get("function") == "count"
                and node.get("argument") is None
            ):
                raise QueryPlanV3Error(
                    "zero-match counts must count a nullable-side non-null key, not COUNT(*)"
                )

    grain_aliases = {
        key.split(".", 1)[0] for key in intent["grain"]["keys"] if "." in key
    }
    for declaration in declarations:
        edge = graph_edges[declaration["edge"]]
        if (
            edge["left_to_right"] == "many" and declaration["kind"] == "inner"
            and declaration["left_alias"] in grain_aliases
            and intent["aggregation"]["stage"] == "none"
            and intent["output"]["row_set_mode"] == "preserve"
        ):
            raise QueryPlanV3Error(
                "one-to-many join can amplify the declared preserved grain"
            )

    aggregation = intent["aggregation"]
    root_has_aggregate = (
        root is not None
        and any(
            node.get("type") == "aggregate"
            for node in _visit(
                {
                    "select": root["select"],
                    "group_by": root["group_by"],
                    "having": root["having"],
                    "order_by": root["order_by"],
                }
            )
        )
    )
    if aggregation["stage"] == "none" and root_has_aggregate:
        raise QueryPlanV3Error("aggregation stage none contradicts outer aggregate expressions")
    if aggregation["stage"] != "none" and not _query_has(query, "aggregate"):
        raise QueryPlanV3Error("declared aggregation stage has no aggregate")
    if aggregation["stage"] == "base_rows" and declarations:
        raise QueryPlanV3Error(
            "base_rows aggregation must be isolated before joins in a derived query"
        )
    grouped_selects = [select for select in _select_queries(query) if select["group_by"]]
    if aggregation["stage"] == "per_group" and not grouped_selects:
        raise QueryPlanV3Error("per_group aggregation requires GROUP BY")
    if intent["grain"]["kind"] == "groups":
        if not grouped_selects:
            raise QueryPlanV3Error("group grain requires GROUP BY")
        declared_grain = {
            _normalise_semantic(key.split(".", 1)[-1])
            for key in intent["grain"]["keys"]
        }
        actual_grouping = {
            semantic
            for select in grouped_selects
            for expression in select["group_by"]
            if (semantic := _expr_semantic(expression)) is not None
        }
        if not declared_grain <= actual_grouping:
            raise QueryPlanV3Error(
                "declared group grain keys are missing from GROUP BY"
            )
    elif intent["grain"]["kind"] == "scalar" and root is not None:
        if root["group_by"]:
            raise QueryPlanV3Error("scalar grain contradicts outer GROUP BY")
    if aggregation["distinct_scope"] == "rows":
        if root is None or not root["distinct"]:
            raise QueryPlanV3Error("row distinct scope requires outer SELECT DISTINCT")
    elif aggregation["distinct_scope"] == "aggregate_arguments":
        if not _query_has_aggregate_distinct(query):
            raise QueryPlanV3Error(
                "aggregate distinct scope requires a DISTINCT aggregate argument"
            )
    elif (root is not None and root["distinct"]) or _query_has_aggregate_distinct(query):
        raise QueryPlanV3Error("query uses DISTINCT without declaring its scope")

    set_nodes = [
        node for node in _visit(query)
        if node.get("type") == "set_op"
    ]
    set_intent = intent["set_semantics"]
    expected = {
        "any_of": "union", "all_of": "intersect", "difference": "except"
    }.get(set_intent["mode"])
    if expected is None:
        if set_nodes:
            raise QueryPlanV3Error("set operation is not declared in intent")
    elif query["type"] != "set_op" or query["operator"] != expected:
        raise QueryPlanV3Error(
            f"{set_intent['mode']} intent requires {expected} query structure"
        )
    elif any(node["operator"] != expected for node in set_nodes):
        raise QueryPlanV3Error("nested set operation contradicts declared set semantics")
    elif bool(query.get("all")) != set_intent["preserve_duplicates"]:
        raise QueryPlanV3Error("set duplicate policy does not match the query")

    is_null, is_not_null = _null_predicates(query)
    null_mode = intent["null_semantics"]["mode"]
    if null_mode == "only" and (not is_null or is_not_null):
        raise QueryPlanV3Error("null-only intent requires only IS NULL predicates")
    if null_mode == "exclude" and (not is_not_null or is_null):
        raise QueryPlanV3Error("null-exclude intent requires only IS NOT NULL predicates")
    if null_mode == "include" and (is_null or is_not_null):
        raise QueryPlanV3Error("null-include intent must not filter NULL membership")

    # A compound query now carries its own ORDER BY/LIMIT, so intent ordering is
    # checked there too instead of only on a wrapping select.
    ordering_root = root if root is not None else query
    if ordering_root["type"] in {"select", "set_op"}:
        order_entries = ordering_root["order_by"]
        alias_source = _leftmost_select(ordering_root)["select"]
        declared_order = intent["ordering"]
        if len(order_entries) < len(declared_order):
            raise QueryPlanV3Error("declared ordering is missing from the query")
        for index, declared in enumerate(declared_order):
            actual = order_entries[index]
            if actual["direction"] != declared["direction"]:
                raise QueryPlanV3Error("ordering direction contradicts intent")
            semantic = _order_semantic(actual, alias_source)
            if semantic is None:
                raise QueryPlanV3Error("ordering expression has no bounded semantic key")
            if semantic != _normalise_semantic(declared["key"]):
                raise QueryPlanV3Error("ordering key contradicts intent")
            if declared["nulls"] != "unspecified":
                raise QueryPlanV3Error(
                    "explicit NULLS FIRST/LAST is outside the v2 lowering"
                )
        tie = intent["tie_policy"]
        if tie["mode"] == "deterministic":
            actual_keys = [
                _order_semantic(item, alias_source) for item in order_entries
            ]
            expected_keys = [_normalise_semantic(item) for item in tie["keys"]]
            if (
                len(actual_keys) < len(expected_keys)
                or actual_keys[-len(expected_keys):] != expected_keys
            ):
                raise QueryPlanV3Error(
                    "deterministic tie keys must be final ordering keys"
                )
        if tie["mode"] == "include_ties":
            if ordering_root["limit"] is not None:
                raise QueryPlanV3Error("include_ties cannot use a hard LIMIT")
            if not order_entries or not any(
                node.get("type") == "window"
                and node.get("function") in {"rank", "dense_rank"}
                for node in _visit(ordering_root)
            ):
                raise QueryPlanV3Error(
                    "include_ties requires ordered RANK/DENSE_RANK structure"
                )

    precision = intent["precision"]
    round_values = _round_precisions(query)
    if precision["mode"] == "round":
        if precision["digits"] not in round_values:
            raise QueryPlanV3Error("declared precision is not implemented by ROUND")
    elif round_values:
        raise QueryPlanV3Error("ROUND changes exact precision without intent")


def _intent_contract(intent: Mapping[str, Any]) -> SemanticContract:
    outputs = intent["output"]["columns"]
    aggregates = [
        item for item in outputs
        if re.match(r"(?i)^(count|sum|avg|min|max)\s*\(", item)
    ]
    projections = [item for item in outputs if item not in aggregates]
    grouping = intent["grain"]["keys"] if intent["grain"]["kind"] == "groups" else []
    ordering = [
        item["key"] + " " + item["direction"].upper()
        + (" NULLS " + item["nulls"].upper() if item["nulls"] != "unspecified" else "")
        for item in intent["ordering"]
    ]
    set_mode = intent["set_semantics"]["mode"]
    return SemanticContract(
        projections=tuple(projections),
        aggregates=tuple(aggregates),
        output_slots=tuple(outputs),
        grouping=tuple(grouping),
        ordering=tuple(ordering),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
        set_semantics=None if set_mode == "none" else set_mode,
        row_set_mode=RowSetMode(intent["output"]["row_set_mode"]),
    )


def _domain_literals(domains: Mapping[str, Sequence[str]] | None) -> tuple[str, ...]:
    if not isinstance(domains, Mapping):
        return ()
    return tuple(
        str(value) for key in sorted(domains) for value in domains[key]
        if isinstance(value, str) and value
    )


def validate_query_plan_v3(
    plan: Any,
    question: str | None = None,
    *,
    domains: Mapping[str, Sequence[str]] | None = None,
    obligations: SemanticObligations | None = None,
) -> str:
    """Validate semantic invariants and return deterministic SQLite SQL.

    The explicit v3 intent is always enforced. Question-derived output shape is
    enforced only when the existing conservative extractor reports COMPLETE.
    Row obligations are enforced when extraction proves them or when supplied.
    """

    parsed = parse_query_plan_v3(plan)
    v2 = {"plan_version": 2, "query": parsed["query"]}
    if question is not None:
        try:
            validate_plan_v2_grounding(
                v2, question, allowed_domain_values=_domain_literals(domains)
            )
        except QueryPlanV2Error as exc:
            raise QueryPlanV3Error(str(exc)) from exc
    try:
        sql = compile_query_plan_v2(v2)
    except QueryPlanV2Error as exc:
        raise QueryPlanV3Error(str(exc)) from exc

    reasons: list[str] = []
    intent_result = validate_candidate(_intent_contract(parsed["intent"]), sql)
    reasons.extend(intent_result.reasons)

    if question is not None:
        extracted = extract_semantic_contract(question)
        if extracted.status is ContractStatus.COMPLETE:
            result = validate_candidate(extracted, sql)
            reasons.extend(f"question contract: {reason}" for reason in result.reasons)
        if obligations is None:
            obligations = extract_semantic_obligations(question, domains or {})

    if obligations is not None:
        if obligations.disposition is ObligationDisposition.TERMINAL_ABSTAIN:
            reasons.extend(
                obligations.unsupported_reasons or ("terminal semantic obligation",)
            )
        elif obligations.disposition is ObligationDisposition.RENDERABLE:
            result = validate_candidate_obligations(obligations, sql)
            reasons.extend(f"semantic obligation: {reason}" for reason in result.reasons)

    reasons = list(dict.fromkeys(reasons))
    if reasons:
        raise QueryPlanV3Error("; ".join(reasons))
    return sql


def compile_query_plan_v3(
    plan: Any,
    question: str | None = None,
    *,
    domains: Mapping[str, Sequence[str]] | None = None,
    obligations: SemanticObligations | None = None,
) -> str:
    """Compile a valid QueryPlan v3 without a raw SQL escape hatch."""

    return validate_query_plan_v3(
        plan, question, domains=domains, obligations=obligations
    )


def canonical_plan_sha256_v3(plan: Any) -> str:
    """Hash the canonical parsed semantic plan."""

    parsed = parse_query_plan_v3(plan)
    encoded = json.dumps(
        parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def query_plan_v3_schema() -> dict[str, Any]:
    """Return the model-facing v3 contract and embedded v2 grammar."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://modelengine.example/schemas/query-plan-v3.json",
        "title": "Semantic QueryPlan v3",
        "type": "object",
        "additionalProperties": False,
        "required": ["plan_version", "intent", "query"],
        "properties": {
            "plan_version": {"const": 3},
            "intent": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_INTENT_FIELDS),
                "properties": {
                    "output": "{columns:list, row_set_mode:preserve|distinct}",
                    "grain": "{kind:scalar|rows|groups, keys:list}",
                    "joins": "list[{edge,left_alias,right_alias,kind,preserve}]",
                    "entity_preservation": "{mode:none|preserve,alias,include_zero_match}",
                    "aggregation": "{stage,distinct_scope}",
                    "set_semantics": "{mode,preserve_duplicates}",
                    "null_semantics": "{mode:unspecified|include|exclude|only}",
                    "tie_policy": "{mode:unspecified|deterministic|include_ties,keys}",
                    "ordering": "list[{key,direction,nulls}]",
                    "precision": "{mode:exact|round,digits:null|0..12}",
                },
            },
            "query": {"$ref": "query_plan_v2.grammar.query"},
        },
        "schema_graph_version": schema_graph()["graph_version"],
        "schema_graph_sha256": canonical_schema_graph_sha256(),
        "schema_graph": schema_graph(),
        "query_plan_v2": query_plan_v2_schema(),
        "rules": [
            "Emit every intent field and one structured query.",
            "Never emit sql, raw_sql, fragments, or free-form expressions.",
            "Declare each base-table join by schema_graph edge and include all guards.",
            "Use LEFT JOIN only for explicitly preserved zero-match entities.",
            "Declare aggregation/DISTINCT, set, NULL, tie, order, and precision semantics.",
            "Do not claim semantics the question does not explicitly make.",
        ],
    }


validate_plan_v3 = validate_query_plan_v3
query_plan_v3_model_schema = query_plan_v3_schema

__all__ = [
    "PLAN_VERSION",
    "QueryPlanV3Error",
    "canonical_plan_sha256_v3",
    "compile_query_plan_v3",
    "parse_query_plan_v3",
    "query_plan_v3_model_schema",
    "query_plan_v3_schema",
    "validate_plan_v3",
    "validate_query_plan_v3",
]
