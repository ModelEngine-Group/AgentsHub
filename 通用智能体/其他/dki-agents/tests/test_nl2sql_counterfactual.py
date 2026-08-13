from __future__ import annotations

import hashlib
import sqlite3

from dki.analysis.nl2sql_counterfactual import (
    available_probes,
    run_counterfactual_probe,
    run_counterfactual_suite,
)


def _database(path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE patients (
            patient_id TEXT PRIMARY KEY,
            name TEXT,
            gender TEXT,
            age INTEGER,
            region TEXT
        );
        CREATE TABLE visits (
            visit_id TEXT PRIMARY KEY,
            patient_id TEXT,
            visit_date TEXT,
            department TEXT,
            disease TEXT,
            drug_prescribed TEXT,
            exam TEXT,
            cost REAL,
            los_days INTEGER
        );
        CREATE TABLE kg_entities (name TEXT, type TEXT, mention_count INTEGER);
        CREATE TABLE kg_triples (
            head TEXT, head_type TEXT, relation TEXT, relation_zh TEXT,
            tail TEXT, tail_type TEXT, confidence REAL, count INTEGER
        );
        """
    )
    connection.executemany(
        "INSERT INTO patients VALUES (?,?,?,?,?)",
        [
            ("p1", "A", "女", 40, "北京"),
            ("p2", "B", "男", 50, "上海"),
        ],
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?,?,?,?,?,?,?,?,?)",
        [
            ("v1", "p1", "2025-01-01", "心内科", "高血压", "药A", "检查A", 100.0, 0),
            ("v2", "p2", "2025-02-01", "内分泌科", "糖尿病", "药B", "检查B", 200.0, 1),
        ],
    )
    connection.executemany(
        "INSERT INTO kg_entities VALUES (?,?,?)",
        [
            ("高血压", "Disease", 2),
            ("症状A", "Symptom", 1),
        ],
    )
    connection.execute(
        "INSERT INTO kg_triples VALUES (?,?,?,?,?,?,?,?)",
        ("高血压", "Disease", "HAS_SYMPTOM", "临床表现", "症状A", "Symptom", 0.9, 1),
    )
    connection.commit()
    connection.close()


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_counterfactual_copy_never_mutates_source_and_distinguishes_count_semantics(tmp_path) -> None:
    database = tmp_path / "analysis.sqlite"
    _database(database)
    before_hash = _sha256(database)
    before_rows = sqlite3.connect(database).execute("SELECT COUNT(*) FROM visits").fetchone()[0]

    row_count = run_counterfactual_probe(
        database,
        "SELECT COUNT(*) FROM visits",
        probe="row_count_vs_distinct",
        expected_relation="changed_result",
    )
    distinct_patients = run_counterfactual_probe(
        database,
        "SELECT COUNT(DISTINCT patient_id) FROM visits",
        probe="row_count_vs_distinct",
        expected_relation="same_result",
    )

    assert row_count.applicable is True and row_count.passed is True
    assert distinct_patients.applicable is True and distinct_patients.passed is True
    assert _sha256(database) == before_hash
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == before_rows
    finally:
        connection.close()


def test_zero_match_and_join_amplification_probes_distinguish_structures(tmp_path) -> None:
    database = tmp_path / "analysis.sqlite"
    _database(database)

    preserved = run_counterfactual_probe(
        database,
        "SELECT p.patient_id, COUNT(v.visit_id) FROM patients p "
        "LEFT JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.patient_id",
        probe="zero_match_preservation",
        expected_relation="row_count_increases",
    )
    dropped = run_counterfactual_probe(
        database,
        "SELECT p.patient_id, COUNT(v.visit_id) FROM patients p "
        "JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.patient_id",
        probe="zero_match_preservation",
        expected_relation="row_count_same",
    )
    amplified = run_counterfactual_probe(
        database,
        "SELECT COUNT(*) FROM visits v JOIN kg_triples t ON t.head=v.disease",
        probe="join_amplification",
        expected_relation="changed_result",
    )
    protected = run_counterfactual_probe(
        database,
        "SELECT COUNT(DISTINCT v.visit_id) FROM visits v "
        "JOIN kg_triples t ON t.head=v.disease",
        probe="join_amplification",
        expected_relation="same_result",
    )

    assert preserved.passed is True
    assert dropped.passed is True
    assert amplified.passed is True
    assert protected.passed is True


def test_null_date_set_and_tie_suite_returns_only_fingerprints(tmp_path) -> None:
    database = tmp_path / "analysis.sqlite"
    _database(database)
    report = run_counterfactual_suite(
        database,
        "SELECT visit_id FROM visits WHERE visit_date <= '2025-02-01'",
        [
            {
                "probe": "date_boundary",
                "expected_relation": "changed_result",
                "bindings": {"boundary": "2025-02-01"},
            },
            {
                "probe": "set_duplicates",
                "expected_relation": "changed_result",
            },
            {
                "probe": "topk_tie",
                "expected_relation": "changed_result",
                "bindings": {"cost": 200.0},
            },
            {
                "probe": "null_anti_join",
                "expected_relation": "same_result",
            },
        ],
    )

    assert set(available_probes()) == {
        "row_count_vs_distinct",
        "join_amplification",
        "zero_match_preservation",
        "null_anti_join",
        "set_duplicates",
        "topk_tie",
        "date_boundary",
    }
    assert report.passed + report.failed + report.skipped == 4
    assert 0.0 <= report.score <= 1.0
    for outcome in report.outcomes:
        assert not hasattr(outcome, "rows")
        if outcome.before_sha256 is not None:
            assert len(outcome.before_sha256) == 64
