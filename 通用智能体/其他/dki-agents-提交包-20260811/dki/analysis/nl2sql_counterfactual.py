"""Controlled, source-isolated counterfactual probes for NL2SQL candidates.

The public API copies the four-table database into a temporary SQLite file,
mutates only that copy, and returns typed fingerprints/reason codes.  Database
rows are never included in the returned evidence and are never suitable as LLM
feedback.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dki.analysis.nl2sql_protocol import validate_gold_sql
from dki.analysis.nl2sql_sqlite import bounded_execution


COUNTERFACTUAL_VERSION = "nl2sql-counterfactual-v1"


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    probe: str
    applicable: bool
    passed: bool | None
    reason_code: str
    before_sha256: str | None = None
    after_sha256: str | None = None
    expected_relation: str | None = None


@dataclass(frozen=True, slots=True)
class CounterfactualReport:
    version: str
    score: float
    passed: int
    failed: int
    skipped: int
    outcomes: tuple[ProbeOutcome, ...]


def _typed(value: Any) -> tuple[str, str]:
    if value is None:
        return ("null", "")
    if isinstance(value, bool):
        return ("number", "1" if value else "0")
    if isinstance(value, (int, float)):
        rendered = f"{float(value):.8f}".rstrip("0").rstrip(".")
        return ("number", rendered or "0")
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    return ("text", str(value))


def _execute(connection: sqlite3.Connection, sql: str) -> dict[str, Any]:
    validate_gold_sql(sql)
    with bounded_execution(connection):
        cursor = connection.execute(sql.strip().rstrip(";"))
        columns = [description[0] for description in cursor.description or ()]
        rows = cursor.fetchall()
    canonical = {
        "column_count": len(columns),
        "rows": sorted(
            [[_typed(value) for value in row] for row in rows],
            key=repr,
        ),
    }
    digest = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "sha256": digest,
        "row_count": len(rows),
        "column_count": len(columns),
        "rows": rows,
    }


def _copy_database(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()


def _unique(connection: sqlite3.Connection, table: str, column: str, prefix: str) -> str:
    value = prefix
    index = 0
    while connection.execute(
        f'SELECT 1 FROM "{table}" WHERE "{column}"=? LIMIT 1',
        (value,),
    ).fetchone():
        index += 1
        value = f"{prefix}_{index}"
    return value


def _visit(connection: sqlite3.Connection) -> list[Any]:
    row = connection.execute(
        "SELECT visit_id,patient_id,visit_date,department,disease,"
        "drug_prescribed,exam,cost,los_days FROM visits ORDER BY visit_id LIMIT 1"
    ).fetchone()
    if row is None:
        raise LookupError("missing_visit")
    return list(row)


def _patient(connection: sqlite3.Connection) -> list[Any]:
    row = connection.execute(
        "SELECT patient_id,name,gender,age,region FROM patients ORDER BY patient_id LIMIT 1"
    ).fetchone()
    if row is None:
        raise LookupError("missing_patient")
    return list(row)


def _insert_visit(connection: sqlite3.Connection, values: Sequence[Any]) -> str:
    row = list(values)
    row[0] = _unique(connection, "visits", "visit_id", "__cf_visit__")
    connection.execute("INSERT INTO visits VALUES (?,?,?,?,?,?,?,?,?)", row)
    return str(row[0])


def _duplicate_visit(connection: sqlite3.Connection, _: Mapping[str, Any]) -> str:
    _insert_visit(connection, _visit(connection))
    return "duplicate_visit"


def _second_kg_edge(connection: sqlite3.Connection, _: Mapping[str, Any]) -> str:
    visit = _visit(connection)
    disease = str(visit[4])
    row = connection.execute(
        "SELECT head_type,relation,relation_zh,tail,tail_type,confidence,count "
        "FROM kg_triples WHERE head=? ORDER BY relation,tail LIMIT 1",
        (disease,),
    ).fetchone()
    if row is None:
        raise LookupError("missing_disease_kg_edge")
    tail = _unique(connection, "kg_entities", "name", "__cf_tail__")
    connection.execute(
        "INSERT INTO kg_entities VALUES (?,?,?)",
        (tail, row[4], 1),
    )
    connection.execute(
        "INSERT INTO kg_triples VALUES (?,?,?,?,?,?,?,?)",
        (disease, row[0], row[1], row[2], tail, row[4], row[5], row[6]),
    )
    return "second_kg_edge"


def _zero_match_patient(connection: sqlite3.Connection, _: Mapping[str, Any]) -> str:
    patient = _patient(connection)
    patient[0] = _unique(connection, "patients", "patient_id", "__cf_patient__")
    patient[1] = "Counterfactual zero-match patient"
    connection.execute("INSERT INTO patients VALUES (?,?,?,?,?)", patient)
    return "zero_match_patient"


def _null_visit_patient(connection: sqlite3.Connection, _: Mapping[str, Any]) -> str:
    visit = _visit(connection)
    visit[1] = None
    _insert_visit(connection, visit)
    return "null_visit_patient_id"


def _set_duplicate(connection: sqlite3.Connection, _: Mapping[str, Any]) -> str:
    visit = _visit(connection)
    _insert_visit(connection, visit)
    return "set_duplicate"


def _topk_tie(connection: sqlite3.Connection, bindings: Mapping[str, Any]) -> str:
    visit = _visit(connection)
    if "cost" in bindings:
        tie_cost = float(bindings["cost"])
    else:
        row = connection.execute("SELECT MAX(cost) FROM visits").fetchone()
        tie_cost = float(row[0] if row and row[0] is not None else visit[7] or 1.0)
    visit[7] = tie_cost
    _insert_visit(connection, visit)
    _insert_visit(connection, visit)
    return "topk_tie"


def _date_boundary(connection: sqlite3.Connection, bindings: Mapping[str, Any]) -> str:
    boundary = str(bindings.get("boundary") or "")
    if not boundary:
        raise LookupError("missing_date_boundary")
    visit = _visit(connection)
    visit[2] = boundary
    _insert_visit(connection, visit)
    return "date_boundary"


_MUTATIONS: dict[str, Callable[[sqlite3.Connection, Mapping[str, Any]], str]] = {
    "row_count_vs_distinct": _duplicate_visit,
    "join_amplification": _second_kg_edge,
    "zero_match_preservation": _zero_match_patient,
    "null_anti_join": _null_visit_patient,
    "set_duplicates": _set_duplicate,
    "topk_tie": _topk_tie,
    "date_boundary": _date_boundary,
}


def available_probes() -> tuple[str, ...]:
    return tuple(_MUTATIONS)


def _evaluate_relation(before: dict[str, Any], after: dict[str, Any], relation: str) -> bool:
    if relation == "same_result":
        return before["sha256"] == after["sha256"]
    if relation == "changed_result":
        return before["sha256"] != after["sha256"]
    if relation == "row_count_increases":
        return after["row_count"] > before["row_count"]
    if relation == "row_count_same":
        return after["row_count"] == before["row_count"]
    raise ValueError(f"unsupported counterfactual relation: {relation}")


def run_counterfactual_probe(
    db_path: Path,
    sql: str,
    *,
    probe: str,
    expected_relation: str,
    bindings: Mapping[str, Any] | None = None,
) -> ProbeOutcome:
    """Run one candidate against an isolated controlled database mutation."""

    if probe not in _MUTATIONS:
        raise ValueError(f"unsupported counterfactual probe: {probe}")
    source = Path(db_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with tempfile.TemporaryDirectory(prefix="dki-nl2sql-counterfactual-") as directory:
        copied = Path(directory) / "probe.sqlite"
        _copy_database(source, copied)
        connection = sqlite3.connect(copied)
        try:
            before = _execute(connection, sql)
            try:
                _MUTATIONS[probe](connection, bindings or {})
            except LookupError as exc:
                return ProbeOutcome(
                    probe=probe,
                    applicable=False,
                    passed=None,
                    reason_code=str(exc),
                    before_sha256=before["sha256"],
                    expected_relation=expected_relation,
                )
            connection.commit()
            after = _execute(connection, sql)
        finally:
            connection.close()
    passed = _evaluate_relation(before, after, expected_relation)
    return ProbeOutcome(
        probe=probe,
        applicable=True,
        passed=passed,
        reason_code="passed" if passed else "invariant_failed",
        before_sha256=before["sha256"],
        after_sha256=after["sha256"],
        expected_relation=expected_relation,
    )


def run_counterfactual_suite(
    db_path: Path,
    sql: str,
    probes: Sequence[Mapping[str, Any]],
) -> CounterfactualReport:
    outcomes = tuple(
        run_counterfactual_probe(
            db_path,
            sql,
            probe=str(spec["probe"]),
            expected_relation=str(spec["expected_relation"]),
            bindings=spec.get("bindings") if isinstance(spec, Mapping) else None,
        )
        for spec in probes
    )
    passed = sum(outcome.passed is True for outcome in outcomes)
    failed = sum(outcome.passed is False for outcome in outcomes)
    skipped = sum(not outcome.applicable for outcome in outcomes)
    applicable = passed + failed
    score = 0.5 if applicable == 0 else passed / applicable
    return CounterfactualReport(
        version=COUNTERFACTUAL_VERSION,
        score=round(score, 6),
        passed=passed,
        failed=failed,
        skipped=skipped,
        outcomes=outcomes,
    )


__all__ = [
    "COUNTERFACTUAL_VERSION",
    "CounterfactualReport",
    "ProbeOutcome",
    "available_probes",
    "run_counterfactual_probe",
    "run_counterfactual_suite",
]
