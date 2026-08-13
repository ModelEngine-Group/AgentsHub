"""Versioned semantic schema graph for the sealed four-table NL2SQL database.

The graph records logical keys and named join relationships independently of the
physical SQLite DDL.  In particular, knowledge-graph keys and relation/type
guards originate in :mod:`dki.kg.store` and :mod:`dki.kg.schema`; they are not
SQL constraints and therefore must remain explicit when a query relies on them.

All public representations are deterministic.  Live verification is optional
and uses the project's hardened read-only SQLite connection.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dki.analysis.nl2sql_query_plan import TABLE_COLUMNS
from dki.analysis.nl2sql_sqlite import bounded_execution, connect_readonly
from dki.kg.schema import ENTITY_TYPES, RELATION_TYPES


SCHEMA_GRAPH_VERSION = 1
_CARDINALITIES = {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}
_MULTIPLICITIES = {"one", "many"}


class SchemaGraphError(ValueError):
    """Raised when the static graph or a supplied live database is inconsistent."""


def _table(name: str, *, keys: list[dict[str, Any]], grain: list[str]) -> dict[str, Any]:
    return {
        "columns": dict(TABLE_COLUMNS[name]),
        "keys": keys,
        "grain": grain,
    }


def _edge(
    left_table: str,
    right_table: str,
    join_columns: list[tuple[str, str]],
    *,
    cardinality: str,
    left_to_right: str,
    right_to_left: str,
    guards: list[dict[str, Any]] | None = None,
    right_requires_left: bool = False,
    description: str,
) -> dict[str, Any]:
    return {
        "left_table": left_table,
        "right_table": right_table,
        "join_columns": [
            {"left": left, "right": right} for left, right in join_columns
        ],
        "guards": list(guards or ()),
        "cardinality": cardinality,
        "left_to_right": left_to_right,
        "right_to_left": right_to_left,
        "right_requires_left": right_requires_left,
        "description": description,
    }


def _constant(side: str, column: str, value: str) -> dict[str, Any]:
    return {
        "kind": "constant",
        "side": side,
        "column": column,
        "operator": "eq",
        "value": value,
    }


def _column_equality(left: str, right: str) -> dict[str, Any]:
    return {"kind": "column_equality", "left": left, "right": right}


_SCHEMA_GRAPH: dict[str, Any] = {
    "graph_version": SCHEMA_GRAPH_VERSION,
    "dialect": "sqlite",
    "tables": {
        "patients": _table(
            "patients",
            keys=[
                {
                    "name": "patients_primary_key",
                    "kind": "primary",
                    "columns": ["patient_id"],
                    "enforced": True,
                }
            ],
            grain=["patient_id"],
        ),
        "visits": _table(
            "visits",
            keys=[
                {
                    "name": "visits_primary_key",
                    "kind": "primary",
                    "columns": ["visit_id"],
                    "enforced": True,
                }
            ],
            grain=["visit_id"],
        ),
        "kg_entities": _table(
            "kg_entities",
            keys=[
                {
                    "name": "kg_entities_logical_name_key",
                    "kind": "logical",
                    "columns": ["name"],
                    "enforced": False,
                },
                {
                    "name": "kg_entities_typed_key",
                    "kind": "candidate",
                    "columns": ["name", "type"],
                    "enforced": False,
                },
            ],
            grain=["name"],
        ),
        "kg_triples": _table(
            "kg_triples",
            keys=[
                {
                    "name": "kg_triples_logical_key",
                    "kind": "logical",
                    "columns": ["head", "relation", "tail"],
                    "enforced": False,
                }
            ],
            grain=["head", "relation", "tail"],
        ),
    },
    "edges": {
        "patients_visits": _edge(
            "patients",
            "visits",
            [("patient_id", "patient_id")],
            cardinality="one_to_many",
            left_to_right="many",
            right_to_left="one",
            right_requires_left=True,
            description="One patient may own many visits; each visit has one patient.",
        ),
        "kg_entities_triples_head": _edge(
            "kg_entities",
            "kg_triples",
            [("name", "head")],
            guards=[_column_equality("type", "head_type")],
            cardinality="one_to_many",
            left_to_right="many",
            right_to_left="one",
            right_requires_left=True,
            description="A typed entity may be the head of many triples.",
        ),
        "kg_entities_triples_tail": _edge(
            "kg_entities",
            "kg_triples",
            [("name", "tail")],
            guards=[_column_equality("type", "tail_type")],
            cardinality="one_to_many",
            left_to_right="many",
            right_to_left="one",
            right_requires_left=True,
            description="A typed entity may be the tail of many triples.",
        ),
        "visits_disease_entities": _edge(
            "visits",
            "kg_entities",
            [("disease", "name")],
            guards=[_constant("right", "type", "Disease")],
            cardinality="many_to_one",
            left_to_right="one",
            right_to_left="many",
            description="A visit disease resolves to at most one Disease entity.",
        ),
        "visits_drug_entities": _edge(
            "visits",
            "kg_entities",
            [("drug_prescribed", "name")],
            guards=[_constant("right", "type", "Drug")],
            cardinality="many_to_one",
            left_to_right="one",
            right_to_left="many",
            description="A prescribed drug resolves to at most one Drug entity.",
        ),
        "visits_exam_entities": _edge(
            "visits",
            "kg_entities",
            [("exam", "name")],
            guards=[_constant("right", "type", "Exam")],
            cardinality="many_to_one",
            left_to_right="one",
            right_to_left="many",
            description="A visit exam resolves to at most one Exam entity.",
        ),
        "visits_department_entities": _edge(
            "visits",
            "kg_entities",
            [("department", "name")],
            guards=[_constant("right", "type", "Department")],
            cardinality="many_to_one",
            left_to_right="one",
            right_to_left="many",
            description="A visit department resolves to at most one Department entity.",
        ),
        "visits_disease_triples": _edge(
            "visits",
            "kg_triples",
            [("disease", "head")],
            guards=[_constant("right", "head_type", "Disease")],
            cardinality="many_to_many",
            left_to_right="many",
            right_to_left="many",
            description="Joining visits to all outgoing disease triples may amplify rows.",
        ),
        "visits_drug_triples": _edge(
            "visits",
            "kg_triples",
            [("drug_prescribed", "tail")],
            guards=[
                _constant("right", "relation", "TREATED_BY_DRUG"),
                _constant("right", "tail_type", "Drug"),
            ],
            cardinality="many_to_many",
            left_to_right="many",
            right_to_left="many",
            description="A drug can occur in many visits and treatment triples.",
        ),
        "visits_exam_triples": _edge(
            "visits",
            "kg_triples",
            [("exam", "tail")],
            guards=[
                _constant("right", "relation", "DIAGNOSED_BY"),
                _constant("right", "tail_type", "Exam"),
            ],
            cardinality="many_to_many",
            left_to_right="many",
            right_to_left="many",
            description="An exam can occur in many visits and diagnosis triples.",
        ),
        "visits_department_triples": _edge(
            "visits",
            "kg_triples",
            [("department", "tail")],
            guards=[
                _constant("right", "relation", "BELONGS_TO_DEPT"),
                _constant("right", "tail_type", "Department"),
            ],
            cardinality="many_to_many",
            left_to_right="many",
            right_to_left="many",
            description="A department can match many visits and disease-department triples.",
        ),
        "visits_belongs_to_department_triples": _edge(
            "visits",
            "kg_triples",
            [("disease", "head"), ("department", "tail")],
            guards=[
                _constant("right", "relation", "BELONGS_TO_DEPT"),
                _constant("right", "head_type", "Disease"),
                _constant("right", "tail_type", "Department"),
            ],
            cardinality="many_to_one",
            left_to_right="one",
            right_to_left="many",
            description="A visit's disease/department pair resolves to one guarded KG edge.",
        ),
    },
    "type_guards": {
        "entity_types": sorted(ENTITY_TYPES),
        "relations": [
            {
                "relation": name,
                "relation_zh": relation.zh,
                "head_type": relation.head_type,
                "tail_type": relation.tail_type,
            }
            for name, relation in sorted(RELATION_TYPES.items())
        ],
    },
}


def _validate_static_graph(graph: Mapping[str, Any]) -> None:
    if graph.get("graph_version") != SCHEMA_GRAPH_VERSION:
        raise SchemaGraphError("schema graph version is invalid")
    tables = graph.get("tables")
    edges = graph.get("edges")
    if not isinstance(tables, Mapping) or set(tables) != set(TABLE_COLUMNS):
        raise SchemaGraphError("schema graph must model exactly the four NL2SQL tables")
    if not isinstance(edges, Mapping):
        raise SchemaGraphError("schema graph edges must be an object")

    for table_name, table in tables.items():
        if not isinstance(table, Mapping):
            raise SchemaGraphError(f"table {table_name!r} must be an object")
        if table.get("columns") != TABLE_COLUMNS[table_name]:
            raise SchemaGraphError(f"table {table_name!r} columns drifted from TABLE_COLUMNS")
        columns = set(TABLE_COLUMNS[table_name])
        grain = table.get("grain")
        if not isinstance(grain, list) or not grain or not set(grain) <= columns:
            raise SchemaGraphError(f"table {table_name!r} has an invalid grain")
        keys = table.get("keys")
        if not isinstance(keys, list) or not keys:
            raise SchemaGraphError(f"table {table_name!r} must declare at least one key")
        for key in keys:
            key_columns = key.get("columns") if isinstance(key, Mapping) else None
            if not isinstance(key_columns, list) or not key_columns or not set(key_columns) <= columns:
                raise SchemaGraphError(f"table {table_name!r} has an invalid key")

    expected_multiplicity = {
        "one_to_one": ("one", "one"),
        "one_to_many": ("many", "one"),
        "many_to_one": ("one", "many"),
        "many_to_many": ("many", "many"),
    }
    for edge_name, edge in edges.items():
        if not isinstance(edge, Mapping):
            raise SchemaGraphError(f"edge {edge_name!r} must be an object")
        left_table = edge.get("left_table")
        right_table = edge.get("right_table")
        if left_table not in tables or right_table not in tables:
            raise SchemaGraphError(f"edge {edge_name!r} references an unknown table")
        cardinality = edge.get("cardinality")
        if cardinality not in _CARDINALITIES:
            raise SchemaGraphError(f"edge {edge_name!r} has invalid cardinality")
        multiplicity = (edge.get("left_to_right"), edge.get("right_to_left"))
        if any(value not in _MULTIPLICITIES for value in multiplicity):
            raise SchemaGraphError(f"edge {edge_name!r} has invalid multiplicity")
        if multiplicity != expected_multiplicity[cardinality]:
            raise SchemaGraphError(f"edge {edge_name!r} cardinality and multiplicity disagree")
        joins = edge.get("join_columns")
        if not isinstance(joins, list) or not joins:
            raise SchemaGraphError(f"edge {edge_name!r} needs join columns")
        for pair in joins:
            if (
                not isinstance(pair, Mapping)
                or pair.get("left") not in TABLE_COLUMNS[left_table]
                or pair.get("right") not in TABLE_COLUMNS[right_table]
            ):
                raise SchemaGraphError(f"edge {edge_name!r} has invalid join columns")
        for guard in edge.get("guards", ()):  # static trusted structure
            if not isinstance(guard, Mapping):
                raise SchemaGraphError(f"edge {edge_name!r} has an invalid guard")
            if guard.get("kind") == "constant":
                side = guard.get("side")
                table = left_table if side == "left" else right_table if side == "right" else None
                if table is None or guard.get("column") not in TABLE_COLUMNS[table]:
                    raise SchemaGraphError(f"edge {edge_name!r} has an invalid constant guard")
            elif guard.get("kind") == "column_equality":
                if (
                    guard.get("left") not in TABLE_COLUMNS[left_table]
                    or guard.get("right") not in TABLE_COLUMNS[right_table]
                ):
                    raise SchemaGraphError(f"edge {edge_name!r} has an invalid column guard")
            else:
                raise SchemaGraphError(f"edge {edge_name!r} has an unknown guard kind")


_validate_static_graph(_SCHEMA_GRAPH)


def schema_graph() -> dict[str, Any]:
    """Return an isolated copy of the versioned four-table schema graph."""

    return copy.deepcopy(_SCHEMA_GRAPH)


def get_schema_graph() -> dict[str, Any]:
    """Compatibility alias for :func:`schema_graph`."""

    return schema_graph()


def canonical_schema_graph_json(graph: Mapping[str, Any] | None = None) -> str:
    """Return deterministic UTF-8 JSON text for a valid schema graph."""

    candidate = _SCHEMA_GRAPH if graph is None else graph
    _validate_static_graph(candidate)
    return json.dumps(
        candidate,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_schema_graph_sha256(graph: Mapping[str, Any] | None = None) -> str:
    """Hash the canonical schema graph representation with SHA-256."""

    return hashlib.sha256(canonical_schema_graph_json(graph).encode("utf-8")).hexdigest()


# Short aliases are useful for callers that already know this module contains a
# schema graph rather than a query plan.
canonical_schema_json = canonical_schema_graph_json
canonical_schema_sha256 = canonical_schema_graph_sha256


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _scalar(connection: Any, sql: str) -> int:
    with bounded_execution(connection):
        row = connection.execute(sql).fetchone()
    return int(row[0] or 0) if row else 0


def _edge_condition(edge: Mapping[str, Any], left_alias: str, right_alias: str) -> str:
    clauses = [
        f"{left_alias}.{_quote_identifier(pair['left'])} = "
        f"{right_alias}.{_quote_identifier(pair['right'])}"
        for pair in edge["join_columns"]
    ]
    for guard in edge["guards"]:
        if guard["kind"] == "constant":
            alias = left_alias if guard["side"] == "left" else right_alias
            clauses.append(
                f"{alias}.{_quote_identifier(guard['column'])} = "
                f"{_quote_literal(guard['value'])}"
            )
        else:
            clauses.append(
                f"{left_alias}.{_quote_identifier(guard['left'])} = "
                f"{right_alias}.{_quote_identifier(guard['right'])}"
            )
    return " AND ".join(clauses)


def verify_schema_graph(db_path: Path) -> dict[str, Any]:
    """Verify logical keys, guarded cardinalities and KG type rules live.

    The report is deterministic for a fixed database.  Empty tables satisfy
    uniqueness and cardinality vacuously, while missing tables/columns fail.
    No schema mutation, temporary table, or writable connection is used.
    """

    graph = _SCHEMA_GRAPH
    violations: list[str] = []
    table_reports: dict[str, Any] = {}
    edge_reports: dict[str, Any] = {}
    guard_report: dict[str, Any] = {}

    connection = connect_readonly(Path(db_path))
    try:
        available_tables: set[str] = set()
        for table_name, table in graph["tables"].items():
            quoted_table = _quote_identifier(table_name)
            with bounded_execution(connection):
                info = connection.execute(f"PRAGMA table_info({quoted_table})").fetchall()
            live_columns = [str(row[1]) for row in info]
            expected_columns = list(table["columns"])
            missing_columns = sorted(set(expected_columns) - set(live_columns))
            extra_columns = sorted(set(live_columns) - set(expected_columns))
            if not info:
                violations.append(f"missing table: {table_name}")
                table_reports[table_name] = {
                    "present": False,
                    "columns": [],
                    "missing_columns": expected_columns,
                    "extra_columns": [],
                    "row_count": None,
                    "keys": [],
                }
                continue
            if missing_columns:
                violations.append(
                    f"table {table_name} missing columns: {','.join(missing_columns)}"
                )
                table_reports[table_name] = {
                    "present": True,
                    "columns": live_columns,
                    "missing_columns": missing_columns,
                    "extra_columns": extra_columns,
                    "row_count": None,
                    "keys": [],
                }
                continue
            available_tables.add(table_name)
            row_count = _scalar(connection, f"SELECT COUNT(*) FROM {quoted_table}")
            key_reports: list[dict[str, Any]] = []
            for key in table["keys"]:
                columns = key["columns"]
                rendered = ", ".join(_quote_identifier(column) for column in columns)
                null_predicate = " OR ".join(
                    f"{_quote_identifier(column)} IS NULL" for column in columns
                )
                duplicate_groups = _scalar(
                    connection,
                    "SELECT COUNT(*) FROM (SELECT "
                    + rendered
                    + f" FROM {quoted_table} GROUP BY {rendered} HAVING COUNT(*) > 1)",
                )
                null_rows = _scalar(
                    connection,
                    f"SELECT COUNT(*) FROM {quoted_table} WHERE {null_predicate}",
                )
                valid = duplicate_groups == 0 and null_rows == 0
                if not valid:
                    violations.append(f"key violation: {table_name}.{key['name']}")
                key_reports.append(
                    {
                        "name": key["name"],
                        "columns": list(columns),
                        "duplicate_groups": duplicate_groups,
                        "null_rows": null_rows,
                        "valid": valid,
                    }
                )
            table_reports[table_name] = {
                "present": True,
                "columns": live_columns,
                "missing_columns": missing_columns,
                "extra_columns": extra_columns,
                "row_count": row_count,
                "keys": key_reports,
            }

        for edge_name, edge in graph["edges"].items():
            left_table = edge["left_table"]
            right_table = edge["right_table"]
            if left_table not in available_tables or right_table not in available_tables:
                edge_reports[edge_name] = {
                    "verified": False,
                    "valid": False,
                    "reason": "missing endpoint table",
                }
                continue
            left_columns = set(graph["tables"][left_table]["columns"])
            right_columns = set(graph["tables"][right_table]["columns"])
            left_keys = graph["tables"][left_table]["keys"]
            right_keys = graph["tables"][right_table]["keys"]
            left_identity = next(
                (key["columns"] for key in left_keys if set(key["columns"]) <= left_columns),
                None,
            )
            right_identity = next(
                (key["columns"] for key in right_keys if set(key["columns"]) <= right_columns),
                None,
            )
            if left_identity is None or right_identity is None:
                edge_reports[edge_name] = {
                    "verified": False,
                    "valid": False,
                    "reason": "missing endpoint identity key",
                }
                violations.append(f"unverifiable edge identity: {edge_name}")
                continue
            left = _quote_identifier(left_table)
            right = _quote_identifier(right_table)
            condition = _edge_condition(edge, "l", "r")
            left_group = ", ".join(
                f"l.{_quote_identifier(column)}" for column in left_identity
            )
            right_group = ", ".join(
                f"r.{_quote_identifier(column)}" for column in right_identity
            )
            left_match_column = f"r.{_quote_identifier(right_identity[0])}"
            right_match_column = f"l.{_quote_identifier(left_identity[0])}"
            left_max = _scalar(
                connection,
                "SELECT COALESCE(MAX(match_count), 0) FROM ("
                f"SELECT {left_group}, COUNT({left_match_column}) AS match_count "
                f"FROM {left} AS l LEFT JOIN {right} AS r ON {condition} "
                f"GROUP BY {left_group})",
            )
            right_max = _scalar(
                connection,
                "SELECT COALESCE(MAX(match_count), 0) FROM ("
                f"SELECT {right_group}, COUNT({right_match_column}) AS match_count "
                f"FROM {right} AS r LEFT JOIN {left} AS l ON {condition} "
                f"GROUP BY {right_group})",
            )
            cardinality_valid = True
            if edge["left_to_right"] == "one" and left_max > 1:
                cardinality_valid = False
            if edge["right_to_left"] == "one" and right_max > 1:
                cardinality_valid = False
            orphan_right_rows = 0
            if edge["right_requires_left"]:
                orphan_right_rows = _scalar(
                    connection,
                    f"SELECT COUNT(*) FROM {right} AS r WHERE NOT EXISTS ("
                    f"SELECT 1 FROM {left} AS l WHERE {condition})",
                )
            valid = cardinality_valid and orphan_right_rows == 0
            if not cardinality_valid:
                violations.append(f"cardinality violation: {edge_name}")
            if orphan_right_rows:
                violations.append(f"referential violation: {edge_name}")
            edge_reports[edge_name] = {
                "verified": True,
                "cardinality": edge["cardinality"],
                "max_right_matches_per_left_row": left_max,
                "max_left_matches_per_right_row": right_max,
                "orphan_right_rows": orphan_right_rows,
                "valid": valid,
            }

        if "kg_entities" in available_tables:
            allowed_types = ", ".join(
                _quote_literal(value) for value in graph["type_guards"]["entity_types"]
            )
            invalid_entity_types = _scalar(
                connection,
                "SELECT COUNT(*) FROM \"kg_entities\" "
                f"WHERE type IS NULL OR type NOT IN ({allowed_types})",
            )
        else:
            invalid_entity_types = None

        relation_checks: list[dict[str, Any]] = []
        if "kg_triples" in available_tables:
            allowed_relations = ", ".join(
                _quote_literal(item["relation"])
                for item in graph["type_guards"]["relations"]
            )
            unknown_relations = _scalar(
                connection,
                "SELECT COUNT(*) FROM \"kg_triples\" "
                f"WHERE relation IS NULL OR relation NOT IN ({allowed_relations})",
            )
            for item in graph["type_guards"]["relations"]:
                mismatches = _scalar(
                    connection,
                    "SELECT COUNT(*) FROM \"kg_triples\" WHERE relation = "
                    + _quote_literal(item["relation"])
                    + " AND (head_type != "
                    + _quote_literal(item["head_type"])
                    + " OR head_type IS NULL OR tail_type != "
                    + _quote_literal(item["tail_type"])
                    + " OR tail_type IS NULL OR relation_zh != "
                    + _quote_literal(item["relation_zh"])
                    + " OR relation_zh IS NULL)",
                )
                relation_checks.append(
                    {"relation": item["relation"], "mismatches": mismatches}
                )
                if mismatches:
                    violations.append(f"relation/type guard violation: {item['relation']}")
        else:
            unknown_relations = None

        if invalid_entity_types:
            violations.append("unknown kg entity type")
        if unknown_relations:
            violations.append("unknown kg relation")
        guard_report = {
            "invalid_entity_type_rows": invalid_entity_types,
            "unknown_relation_rows": unknown_relations,
            "relation_checks": relation_checks,
        }
    finally:
        connection.close()

    return {
        "graph_version": SCHEMA_GRAPH_VERSION,
        "schema_graph_sha256": canonical_schema_graph_sha256(),
        "valid": not violations,
        "tables": table_reports,
        "edges": edge_reports,
        "type_guards": guard_report,
        "violations": list(dict.fromkeys(violations)),
    }


def verify_live_schema_graph(db_path: Path) -> dict[str, Any]:
    """Compatibility alias for :func:`verify_schema_graph`."""

    return verify_schema_graph(db_path)


__all__ = [
    "SCHEMA_GRAPH_VERSION",
    "SchemaGraphError",
    "canonical_schema_graph_json",
    "canonical_schema_graph_sha256",
    "canonical_schema_json",
    "canonical_schema_sha256",
    "get_schema_graph",
    "schema_graph",
    "verify_live_schema_graph",
    "verify_schema_graph",
]
