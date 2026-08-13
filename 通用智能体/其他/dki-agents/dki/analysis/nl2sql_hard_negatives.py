"""Execution-distinguishable, single-mutation hard negatives for Track C."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from dki.analysis.nl2sql_protocol import validate_gold_sql
from dki.analysis.nl2sql_semantic_corpus import (
    REQUIRED_SCHEMA,
    _sha256_bytes,
    _sha256_file,
    _sha256_text,
    _typed_result_value,
    build_ast_metadata,
    execute_with_fingerprint,
    inspect_live_schema,
)


NEGATIVE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _Mutation:
    label: str
    description: str
    apply: Callable[[str, Mapping[str, Any]], str]
    perturbation: str | None = None


def _replace_once(sql: str, old: str, new: str, *, label: str) -> str:
    occurrences = sql.count(old)
    if occurrences != 1:
        raise ValueError(
            f"mutation {label} expected one occurrence of {old!r}; found {occurrences}"
        )
    return sql.replace(old, new, 1)


def _remove_distinct(sql: str, _: Mapping[str, Any]) -> str:
    return _replace_once(
        sql,
        "COUNT(DISTINCT ",
        "COUNT(",
        label="count_distinct_to_count",
    )


def _left_join_to_inner_patient(sql: str, _: Mapping[str, Any]) -> str:
    return _replace_once(
        sql,
        "LEFT JOIN visits AS v",
        "JOIN visits AS v",
        label="left_join_to_inner_patient",
    )


def _left_join_to_inner_kg(sql: str, _: Mapping[str, Any]) -> str:
    return _replace_once(
        sql,
        "LEFT JOIN kg_triples AS k",
        "JOIN kg_triples AS k",
        label="left_join_to_inner_kg",
    )


def _having_boundary_strictened(sql: str, bindings: Mapping[str, Any]) -> str:
    threshold = int(bindings["threshold"])
    return _replace_once(
        sql,
        f"HAVING COUNT(*) >= {threshold}",
        f"HAVING COUNT(*) > {threshold}",
        label="having_gte_to_gt",
    )


def _drop_cte_filter(sql: str, bindings: Mapping[str, Any]) -> str:
    department = str(bindings["department"])
    literal = "'" + department.replace("'", "''") + "'"
    return _replace_once(
        sql,
        f" FROM visits WHERE department = {literal})",
        " FROM visits)",
        label="drop_cte_filter",
    )


def _union_all_to_union(sql: str, _: Mapping[str, Any]) -> str:
    return _replace_once(sql, " UNION ALL ", " UNION ", label="union_all_to_union")


def _union_to_union_all(sql: str, _: Mapping[str, Any]) -> str:
    if " UNION ALL " in sql:
        raise ValueError("union_to_union_all requires a distinct UNION gold query")
    return _replace_once(sql, " UNION ", " UNION ALL ", label="union_to_union_all")


def _rank_to_row_number(sql: str, _: Mapping[str, Any]) -> str:
    return _replace_once(sql, "RANK() OVER", "ROW_NUMBER() OVER", label="rank_to_row_number")


def _exclude_date_upper_bound(sql: str, _: Mapping[str, Any]) -> str:
    return _replace_once(sql, "AND visit_date <=", "AND visit_date <", label="date_lte_to_lt")


def _not_exists_to_not_in(sql: str, _: Mapping[str, Any]) -> str:
    pattern = re.compile(
        r"WHERE NOT EXISTS \(SELECT 1 FROM visits AS v "
        r"WHERE v\.patient_id = p\.patient_id\)"
    )
    replacement = "WHERE p.patient_id NOT IN (SELECT v.patient_id FROM visits AS v)"
    mutated, count = pattern.subn(replacement, sql, count=1)
    if count != 1:
        raise ValueError("mutation not_exists_to_not_in did not match its gold query")
    return mutated


def _reduce_round_precision(sql: str, bindings: Mapping[str, Any]) -> str:
    precision = int(bindings.get("precision", 2))
    if precision <= 0:
        raise ValueError("cannot reduce non-positive rounding precision")
    return _replace_once(
        sql,
        f"ROUND(AVG(v.cost), {precision})",
        f"ROUND(AVG(v.cost), {precision - 1})",
        label="round_precision_reduced",
    )


def _swap_projection_order(sql: str, _: Mapping[str, Any]) -> str:
    old = (
        "SELECT p.region AS region, ROUND(AVG(v.cost), 2) AS average_cost, "
        "COUNT(DISTINCT p.patient_id) AS distinct_patients"
    )
    new = (
        "SELECT ROUND(AVG(v.cost), 2) AS average_cost, p.region AS region, "
        "COUNT(DISTINCT p.patient_id) AS distinct_patients"
    )
    return _replace_once(sql, old, new, label="projection_order_swapped")


def _reverse_primary_sort(sql: str, _: Mapping[str, Any]) -> str:
    return _replace_once(
        sql,
        "ORDER BY average_cost DESC, p.region ASC",
        "ORDER BY average_cost ASC, p.region ASC",
        label="primary_sort_reversed",
    )


_MUTATIONS: dict[str, _Mutation] = {
    "remove_distinct": _Mutation(
        "count_distinct_to_count",
        "Replace one COUNT(DISTINCT key) with COUNT(key).",
        _remove_distinct,
        "duplicate_visit_for_count",
    ),
    "left_join_to_inner_patient": _Mutation(
        "left_join_to_inner_patient",
        "Replace one patient-to-visit LEFT JOIN with an INNER JOIN.",
        _left_join_to_inner_patient,
        "orphan_patient",
    ),
    "left_join_to_inner_kg": _Mutation(
        "left_join_to_inner_kg",
        "Replace one entity-to-triple LEFT JOIN with an INNER JOIN.",
        _left_join_to_inner_kg,
        "unmatched_kg_entity",
    ),
    "having_boundary_strictened": _Mutation(
        "having_gte_to_gt",
        "Change one inclusive HAVING count boundary to an exclusive boundary.",
        _having_boundary_strictened,
        "having_boundary_group",
    ),
    "drop_cte_filter": _Mutation(
        "drop_cte_filter",
        "Remove the single row filter from the source CTE.",
        _drop_cte_filter,
        "outside_cte_filter_row",
    ),
    "union_all_to_union": _Mutation(
        "union_all_to_union",
        "Change bag union into duplicate-eliminating UNION.",
        _union_all_to_union,
        "set_duplicate_witness",
    ),
    "union_to_union_all": _Mutation(
        "union_to_union_all",
        "Change duplicate-eliminating UNION into bag union.",
        _union_to_union_all,
        "set_duplicate_witness",
    ),
    "rank_to_row_number": _Mutation(
        "rank_to_row_number",
        "Replace tie-preserving RANK with tie-breaking ROW_NUMBER.",
        _rank_to_row_number,
        "window_tie",
    ),
    "exclude_date_upper_bound": _Mutation(
        "date_lte_to_lt",
        "Change the inclusive upper date boundary to an exclusive boundary.",
        _exclude_date_upper_bound,
        "upper_date_witness",
    ),
    "not_exists_to_not_in": _Mutation(
        "not_exists_to_not_in",
        "Replace NULL-safe NOT EXISTS anti-join with NOT IN.",
        _not_exists_to_not_in,
        "null_visit_patient_id",
    ),
    "reduce_round_precision": _Mutation(
        "round_precision_reduced",
        "Reduce the requested rounding precision by one decimal place.",
        _reduce_round_precision,
        "precision_witness",
    ),
    "swap_projection_order": _Mutation(
        "projection_order_swapped",
        "Swap the first two requested output columns.",
        _swap_projection_order,
    ),
    "reverse_primary_sort": _Mutation(
        "primary_sort_reversed",
        "Reverse the requested primary result ordering.",
        _reverse_primary_sort,
        "order_witness",
    ),
}


def mutation_labels() -> tuple[str, ...]:
    return tuple(sorted(mutation.label for mutation in _MUTATIONS.values()))


def _validate_single_ast_mutation(gold_sql: str, negative_sql: str, label: str) -> None:
    """Assert that one allowlisted semantic AST delta occurred and nothing else."""

    try:
        from sqlglot import exp, parse_one
    except ImportError as exc:
        raise RuntimeError("sqlglot is required for Track C hard negatives") from exc

    gold = parse_one(gold_sql, read="sqlite")
    negative = parse_one(negative_sql, read="sqlite")

    def count(kind, tree) -> int:
        return sum(1 for _ in tree.find_all(kind))

    if label == "count_distinct_to_count":
        if count(exp.Count, gold) != count(exp.Count, negative):
            raise ValueError("COUNT mutation changed aggregate count")
        delta = count(exp.Distinct, gold) - count(exp.Distinct, negative)
        if delta != 1:
            raise ValueError("COUNT mutation must remove exactly one DISTINCT")
    elif label in {"left_join_to_inner_patient", "left_join_to_inner_kg"}:
        gold_joins = list(gold.find_all(exp.Join))
        negative_joins = list(negative.find_all(exp.Join))
        if len(gold_joins) != len(negative_joins):
            raise ValueError("join mutation changed the join count")
        changed = 0
        for left, right in zip(gold_joins, negative_joins):
            left_side = str(left.args.get("side") or "").casefold()
            right_side = str(right.args.get("side") or "").casefold()
            if left_side != right_side:
                if left_side != "left" or right_side not in {"", "inner"}:
                    raise ValueError("join mutation is not LEFT-to-INNER")
                changed += 1
        if changed != 1:
            raise ValueError("join mutation must change exactly one join")
    elif label == "having_gte_to_gt":
        if count(exp.GTE, gold) - count(exp.GTE, negative) != 1:
            raise ValueError("HAVING mutation must remove one GTE")
        if count(exp.GT, negative) - count(exp.GT, gold) != 1:
            raise ValueError("HAVING mutation must add one GT")
    elif label == "drop_cte_filter":
        if count(exp.Where, gold) - count(exp.Where, negative) != 1:
            raise ValueError("CTE mutation must remove exactly one WHERE")
        if count(exp.CTE, gold) != count(exp.CTE, negative):
            raise ValueError("CTE mutation changed the CTE count")
    elif label in {"union_all_to_union", "union_to_union_all"}:
        if not isinstance(gold, exp.Union) or not isinstance(negative, exp.Union):
            raise ValueError("set duplicate mutation requires UNION roots")
        gold_distinct = gold.args.get("distinct") is not False
        negative_distinct = negative.args.get("distinct") is not False
        expected = (
            (False, True)
            if label == "union_all_to_union"
            else (True, False)
        )
        if (gold_distinct, negative_distinct) != expected:
            raise ValueError("UNION duplicate semantics did not flip as labelled")
    elif label == "rank_to_row_number":
        if count(exp.Rank, gold) - count(exp.Rank, negative) != 1:
            raise ValueError("window mutation must remove one RANK")
        if count(exp.RowNumber, negative) - count(exp.RowNumber, gold) != 1:
            raise ValueError("window mutation must add one ROW_NUMBER")
    elif label == "date_lte_to_lt":
        if count(exp.LTE, gold) - count(exp.LTE, negative) != 1:
            raise ValueError("date mutation must remove one LTE")
        if count(exp.LT, negative) - count(exp.LT, gold) != 1:
            raise ValueError("date mutation must add one LT")
    elif label == "not_exists_to_not_in":
        if count(exp.Exists, gold) - count(exp.Exists, negative) != 1:
            raise ValueError("anti-join mutation must remove one EXISTS")
        if count(exp.In, negative) - count(exp.In, gold) != 1:
            raise ValueError("anti-join mutation must add one IN")
    elif label == "round_precision_reduced":
        gold_round = next(gold.find_all(exp.Round), None)
        negative_round = next(negative.find_all(exp.Round), None)
        if gold_round is None or negative_round is None:
            raise ValueError("precision mutation requires ROUND in both queries")
        gold_decimals = gold_round.args.get("decimals")
        negative_decimals = negative_round.args.get("decimals")
        if not isinstance(gold_decimals, exp.Literal) or not isinstance(
            negative_decimals, exp.Literal
        ):
            raise ValueError("ROUND precision must be literal")
        if int(gold_decimals.this) - int(negative_decimals.this) != 1:
            raise ValueError("ROUND precision must decrease by exactly one")
    elif label == "projection_order_swapped":
        gold_select = next(gold.find_all(exp.Select), None)
        negative_select = next(negative.find_all(exp.Select), None)
        if gold_select is None or negative_select is None:
            raise ValueError("projection mutation requires SELECT roots")
        gold_expressions = [item.sql(dialect="sqlite") for item in gold_select.expressions]
        negative_expressions = [
            item.sql(dialect="sqlite") for item in negative_select.expressions
        ]
        expected = list(gold_expressions)
        expected[0], expected[1] = expected[1], expected[0]
        if negative_expressions != expected:
            raise ValueError("projection mutation must swap only the first two outputs")
    elif label == "primary_sort_reversed":
        gold_order = next(gold.find_all(exp.Order), None)
        negative_order = next(negative.find_all(exp.Order), None)
        if gold_order is None or negative_order is None:
            raise ValueError("sort mutation requires ORDER BY")
        gold_keys = list(gold_order.expressions)
        negative_keys = list(negative_order.expressions)
        if len(gold_keys) != len(negative_keys):
            raise ValueError("sort mutation changed the number of keys")
        changed = 0
        for left, right in zip(gold_keys, negative_keys):
            if left.this.sql(dialect="sqlite") != right.this.sql(dialect="sqlite"):
                raise ValueError("sort mutation changed an ordering expression")
            if bool(left.args.get("desc")) != bool(right.args.get("desc")):
                changed += 1
        if changed != 1:
            raise ValueError("sort mutation must reverse exactly one direction")
    else:
        raise ValueError(f"unknown mutation label: {label}")


def _execute_on_connection(connection: sqlite3.Connection, sql: str) -> dict[str, Any]:
    cursor = connection.execute(sql.strip().rstrip(";"))
    columns = [str(column[0]) for column in cursor.description]
    rows = [tuple(row) for row in cursor.fetchall()]
    payload = {
        "columns": columns,
        "rows": [[_typed_result_value(value) for value in row] for row in rows],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "columns": columns,
        "row_count": len(rows),
        "result_sha256": _sha256_bytes(encoded),
    }


def _copy_four_table_database(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(source)
    destination_connection = sqlite3.connect(destination)
    try:
        for table in REQUIRED_SCHEMA:
            row = source_connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if row is None or not row[0]:
                raise ValueError(f"required table is missing while copying: {table}")
            destination_connection.execute(str(row[0]))
            columns = REQUIRED_SCHEMA[table]
            select_columns = ", ".join(f'"{column}"' for column in columns)
            rows = source_connection.execute(
                f'SELECT {select_columns} FROM "{table}"'
            ).fetchall()
            placeholders = ", ".join("?" for _ in columns)
            destination_connection.executemany(
                f'INSERT INTO "{table}" ({select_columns}) VALUES ({placeholders})',
                rows,
            )
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()


def _unique_text(connection: sqlite3.Connection, table: str, column: str, prefix: str) -> str:
    candidate = prefix
    suffix = 0
    while connection.execute(
        f'SELECT 1 FROM "{table}" WHERE "{column}" = ? LIMIT 1',
        (candidate,),
    ).fetchone():
        suffix += 1
        candidate = f"{prefix}_{suffix}"
    return candidate


def _existing_patient(connection: sqlite3.Connection, *, region: str | None = None) -> tuple[Any, ...]:
    if region is None:
        row = connection.execute(
            "SELECT patient_id, name, gender, age, region FROM patients "
            "ORDER BY patient_id LIMIT 1"
        ).fetchone()
    else:
        row = connection.execute(
            "SELECT patient_id, name, gender, age, region FROM patients "
            "WHERE region = ? ORDER BY patient_id LIMIT 1",
            (region,),
        ).fetchone()
    if row is None:
        raise ValueError("controlled perturbation needs at least one patient")
    return tuple(row)


def _existing_visit(connection: sqlite3.Connection, *, department: str | None = None) -> tuple[Any, ...]:
    if department is None:
        row = connection.execute(
            "SELECT visit_id, patient_id, visit_date, department, disease, "
            "drug_prescribed, exam, cost, los_days FROM visits "
            "ORDER BY visit_id LIMIT 1"
        ).fetchone()
    else:
        row = connection.execute(
            "SELECT visit_id, patient_id, visit_date, department, disease, "
            "drug_prescribed, exam, cost, los_days FROM visits "
            "WHERE department = ? ORDER BY visit_id LIMIT 1",
            (department,),
        ).fetchone()
    if row is None:
        raise ValueError("controlled perturbation needs at least one visit")
    return tuple(row)


def _insert_visit(
    connection: sqlite3.Connection,
    source: Sequence[Any],
    *,
    patient_id: Any = ...,
    visit_date: Any = ...,
    department: Any = ...,
    disease: Any = ...,
    cost: Any = ...,
) -> None:
    values = list(source)
    values[0] = _unique_text(connection, "visits", "visit_id", "__track_c_visit__")
    if patient_id is not ...:
        values[1] = patient_id
    if visit_date is not ...:
        values[2] = visit_date
    if department is not ...:
        values[3] = department
    if disease is not ...:
        values[4] = disease
    if cost is not ...:
        values[7] = cost
    connection.execute(
        "INSERT INTO visits (visit_id, patient_id, visit_date, department, disease, "
        "drug_prescribed, exam, cost, los_days) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        tuple(values),
    )


def _apply_perturbation(
    connection: sqlite3.Connection,
    kind: str,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if kind == "duplicate_visit_for_count":
        department = bindings.get("department")
        region = bindings.get("region")
        if department is not None:
            source = _existing_visit(connection, department=str(department))
            patient_id = source[1]
            if patient_id is None:
                patient_id = _existing_patient(connection)[0]
            _insert_visit(connection, source, patient_id=patient_id)
            return {"kind": kind, "patient_id": str(patient_id)}
        if region is not None:
            candidates = connection.execute(
                "SELECT p.patient_id FROM patients AS p JOIN visits AS v "
                "ON v.patient_id = p.patient_id WHERE p.region = ? "
                "GROUP BY p.patient_id ORDER BY p.patient_id",
                (str(region),),
            ).fetchall()
            if candidates:
                patient_id = candidates[0][0]
                row = connection.execute(
                    "SELECT visit_id, patient_id, visit_date, department, disease, "
                    "drug_prescribed, exam, cost, los_days FROM visits "
                    "WHERE patient_id = ? ORDER BY visit_id LIMIT 1",
                    (patient_id,),
                ).fetchone()
                if row is None:
                    raise ValueError("joined region witness lost its source visit")
                source = tuple(row)
            else:
                template_patient = _existing_patient(connection)
                patient_id = _unique_text(
                    connection,
                    "patients",
                    "patient_id",
                    "__track_c_join_patient__",
                )
                connection.execute(
                    "INSERT INTO patients (patient_id, name, gender, age, region) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        patient_id,
                        "Track C joined patient",
                        template_patient[2],
                        template_patient[3],
                        str(region),
                    ),
                )
                source = _existing_visit(connection)
                _insert_visit(connection, source, patient_id=patient_id)
            _insert_visit(connection, source, patient_id=patient_id)
            return {"kind": kind, "patient_id": str(patient_id)}
        source = _existing_visit(connection)
        patient_id = source[1]
        if patient_id is None:
            patient_id = _existing_patient(connection)[0]
        _insert_visit(connection, source, patient_id=patient_id)
        return {"kind": kind, "patient_id": str(patient_id)}

    if kind == "orphan_patient":
        region = str(bindings["region"])
        source = _existing_patient(connection, region=region)
        patient_id = _unique_text(connection, "patients", "patient_id", "__track_c_patient__")
        connection.execute(
            "INSERT INTO patients (patient_id, name, gender, age, region) "
            "VALUES (?, ?, ?, ?, ?)",
            (patient_id, "Track C orphan", source[2], source[3], region),
        )
        return {"kind": kind, "patient_id": patient_id, "region": region}

    if kind == "unmatched_kg_entity":
        entity_type = str(bindings["entity_type"])
        name = _unique_text(connection, "kg_entities", "name", "__track_c_entity__")
        connection.execute(
            "INSERT INTO kg_entities (name, type, mention_count) VALUES (?, ?, ?)",
            (name, entity_type, 1),
        )
        return {"kind": kind, "name": name, "entity_type": entity_type}

    if kind == "having_boundary_group":
        department = str(bindings["department"])
        threshold = int(bindings["threshold"])
        source = _existing_visit(connection, department=department)
        disease = _unique_text(connection, "visits", "disease", "__track_c_boundary__")
        for _ in range(threshold):
            _insert_visit(connection, source, department=department, disease=disease)
        return {"kind": kind, "department": department, "disease": disease, "count": threshold}

    if kind == "outside_cte_filter_row":
        department = str(bindings["department"])
        source = _existing_visit(connection)
        alternative = connection.execute(
            "SELECT department FROM visits WHERE department IS NOT NULL AND department <> ? "
            "ORDER BY department LIMIT 1",
            (department,),
        ).fetchone()
        outside_department = (
            str(alternative[0])
            if alternative is not None
            else _unique_text(connection, "visits", "department", "__track_c_other_dept__")
        )
        _insert_visit(
            connection,
            source,
            department=outside_department,
            cost=9_999_999.91,
        )
        return {"kind": kind, "department": outside_department}

    if kind == "set_duplicate_witness":
        left_value = str(bindings["left_value"])
        right_value = str(bindings["right_value"])
        source = _existing_visit(connection, department=left_value)
        patient_id = source[1]
        if patient_id is None:
            patient_id = _existing_patient(connection)[0]
        _insert_visit(connection, source, patient_id=patient_id, department=left_value)
        _insert_visit(connection, source, patient_id=patient_id, department=right_value)
        return {"kind": kind, "patient_id": str(patient_id)}

    if kind == "window_tie":
        department = str(bindings["department"])
        source = _existing_visit(connection, department=department)
        maximum = connection.execute(
            "SELECT MAX(cost) FROM visits WHERE department = ?",
            (department,),
        ).fetchone()[0]
        tie_cost = float(maximum if maximum is not None else source[7] or 1.0)
        _insert_visit(connection, source, department=department, cost=tie_cost)
        _insert_visit(connection, source, department=department, cost=tie_cost)
        return {"kind": kind, "department": department, "cost": tie_cost}

    if kind == "upper_date_witness":
        end_date = str(bindings["end_date"])
        source = _existing_visit(connection)
        _insert_visit(connection, source, visit_date=end_date)
        return {"kind": kind, "visit_date": end_date}

    if kind == "null_visit_patient_id":
        source = _existing_visit(connection)
        _insert_visit(connection, source, patient_id=None)
        return {"kind": kind}

    if kind == "precision_witness":
        gender = str(bindings["gender"])
        patient = connection.execute(
            "SELECT patient_id, name, gender, age, region FROM patients "
            "WHERE gender = ? ORDER BY patient_id LIMIT 1",
            (gender,),
        ).fetchone()
        if patient is None:
            raise ValueError(f"no patient is available for gender {gender!r}")
        source = _existing_visit(connection)
        for cost in (1.01, 1.02, 1.04):
            _insert_visit(connection, source, patient_id=patient[0], cost=cost)
        return {"kind": kind, "patient_id": str(patient[0])}

    if kind == "order_witness":
        gender = str(bindings["gender"])
        patients = connection.execute(
            "SELECT patient_id, region FROM patients WHERE gender = ? "
            "ORDER BY patient_id",
            (gender,),
        ).fetchall()
        regions = []
        for patient_id, region in patients:
            if region not in regions:
                regions.append(region)
            if len(regions) >= 2:
                break
        if len(regions) < 2:
            raise ValueError("sort perturbation needs two regions for the selected gender")
        source = _existing_visit(connection)
        selected = []
        for patient_id, region in patients:
            if region in regions and region not in {item[1] for item in selected}:
                selected.append((patient_id, region))
            if len(selected) == 2:
                break
        _insert_visit(connection, source, patient_id=selected[0][0], cost=1.11)
        _insert_visit(connection, source, patient_id=selected[1][0], cost=999_999.99)
        return {"kind": kind, "regions": [str(item[1]) for item in selected]}

    raise ValueError(f"unknown controlled perturbation: {kind}")


def _controlled_distinguishability(
    db_path: Path,
    gold_sql: str,
    negative_sql: str,
    *,
    kind: str,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dki-track-c-negative-") as directory:
        copied = Path(directory) / "perturbed.sqlite"
        _copy_four_table_database(Path(db_path), copied)
        connection = sqlite3.connect(copied)
        try:
            witness = _apply_perturbation(connection, kind, bindings)
            connection.commit()
            gold_execution = _execute_on_connection(connection, gold_sql)
            negative_execution = _execute_on_connection(connection, negative_sql)
        finally:
            connection.close()
    if gold_execution["result_sha256"] == negative_execution["result_sha256"]:
        raise ValueError(
            f"controlled perturbation {kind} did not distinguish the mutation"
        )
    return {
        "source": "controlled_temporary_sqlite",
        "perturbation": witness,
        "gold_execution": gold_execution,
        "negative_execution": negative_execution,
    }


def _candidate_for_profile(
    db_path: Path,
    item: Mapping[str, Any],
    profile: str,
) -> dict[str, Any]:
    if profile not in _MUTATIONS:
        raise ValueError(f"unknown mutation profile {profile!r} in {item.get('id')}")
    mutation = _MUTATIONS[profile]
    gold_sql = str(item["gold_sql"])
    bindings = dict(item["family_metadata"].get("domain_bindings", {}))
    negative_sql = mutation.apply(gold_sql, bindings)
    if negative_sql == gold_sql:
        raise ValueError(f"mutation {profile} did not change SQL for {item['id']}")
    validate_gold_sql(negative_sql)
    _validate_single_ast_mutation(gold_sql, negative_sql, mutation.label)

    gold_execution = execute_with_fingerprint(gold_sql, db_path)
    negative_execution = execute_with_fingerprint(negative_sql, db_path)
    if gold_execution["result_sha256"] != negative_execution["result_sha256"]:
        witness = {
            "source": "base_database",
            "gold_execution": gold_execution,
            "negative_execution": negative_execution,
        }
    else:
        if mutation.perturbation is None:
            raise ValueError(
                f"mutation {mutation.label} is indistinguishable on the base database "
                "and has no controlled perturbation"
            )
        witness = _controlled_distinguishability(
            db_path,
            gold_sql,
            negative_sql,
            kind=mutation.perturbation,
            bindings=bindings,
        )

    negative_ast = build_ast_metadata(negative_sql)
    negative_id = f"{item['id']}--{mutation.label}"
    return {
        "id": negative_id,
        "source_id": item["id"],
        "split": item.get("split"),
        "question": item["question"],
        "gold_sql": gold_sql,
        "negative_sql": negative_sql,
        "semantic_family": item["semantic_family"],
        "join_path": list(item.get("join_path", ())),
        "gold_ast": item["ast"],
        "negative_ast": negative_ast,
        "mutation": {
            "profile": profile,
            "label": mutation.label,
            "description": mutation.description,
            "single_semantic_mutation": True,
        },
        "readonly_executable": True,
        "distinguishability": witness,
    }


def build_hard_negatives(
    db_path: Path,
    corpus_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build only executable hard negatives with an observed semantic witness."""

    database = Path(db_path).resolve()
    if not database.is_file():
        raise FileNotFoundError(f"SQLite database does not exist: {database}")
    inspect_live_schema(database)
    negatives: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in sorted(corpus_rows, key=lambda row: str(row["id"])):
        profiles = item["family_metadata"].get("mutation_profiles", ())
        for profile in profiles:
            negative = _candidate_for_profile(database, item, str(profile))
            if negative["id"] in seen_ids:
                raise ValueError(f"duplicate hard-negative id: {negative['id']}")
            seen_ids.add(negative["id"])
            negatives.append(negative)
    if not negatives:
        raise ValueError("corpus rows expose no Track C mutation profiles")
    negatives.sort(key=lambda row: str(row["id"]))
    labels = sorted({row["mutation"]["label"] for row in negatives})
    output_bytes = (
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in negatives
        )
    ).encode("utf-8")
    manifest = {
        "schema_version": NEGATIVE_SCHEMA_VERSION,
        "kind": "nl2sql_single_mutation_hard_negatives",
        "database_sha256": _sha256_file(database),
        "row_count": len(negatives),
        "source_row_count": len(corpus_rows),
        "mutation_labels": labels,
        "base_database_witnesses": sum(
            row["distinguishability"]["source"] == "base_database" for row in negatives
        ),
        "controlled_temporary_sqlite_witnesses": sum(
            row["distinguishability"]["source"] == "controlled_temporary_sqlite"
            for row in negatives
        ),
        "output_sha256": _sha256_bytes(output_bytes),
    }
    return {"negatives": negatives, "manifest": manifest, "output_bytes": output_bytes}


def _atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_hard_negatives(
    db_path: Path,
    corpus_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Build and atomically write hard negatives plus a deterministic manifest."""

    result = build_hard_negatives(db_path, corpus_rows)
    destination = Path(output_path)
    manifest_destination = (
        Path(manifest_path)
        if manifest_path is not None
        else destination.with_suffix(destination.suffix + ".manifest.json")
    )
    manifest_bytes = (
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    _atomic_write(destination, result["output_bytes"])
    _atomic_write(manifest_destination, manifest_bytes)
    return {
        "output_path": destination,
        "manifest_path": manifest_destination,
        "row_count": len(result["negatives"]),
        "manifest": result["manifest"],
    }
