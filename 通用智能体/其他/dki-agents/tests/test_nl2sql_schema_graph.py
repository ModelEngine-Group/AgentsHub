from __future__ import annotations

import json
import sqlite3

from dki.analysis.nl2sql_schema_graph import (
    SCHEMA_GRAPH_VERSION,
    canonical_schema_graph_json,
    canonical_schema_graph_sha256,
    schema_graph,
    verify_schema_graph,
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
        CREATE TABLE kg_entities (
            name TEXT,
            type TEXT,
            mention_count INTEGER
        );
        CREATE TABLE kg_triples (
            head TEXT,
            head_type TEXT,
            relation TEXT,
            relation_zh TEXT,
            tail TEXT,
            tail_type TEXT,
            confidence REAL,
            count INTEGER
        );
        """
    )
    connection.execute(
        "INSERT INTO patients VALUES ('p1','A','女',40,'北京')"
    )
    connection.execute(
        "INSERT INTO visits VALUES "
        "('v1','p1','2025-01-01','心内科','高血压','药A','检查A',100,0)"
    )
    connection.executemany(
        "INSERT INTO kg_entities VALUES (?,?,?)",
        [
            ("高血压", "Disease", 3),
            ("心内科", "Department", 2),
            ("药A", "Drug", 2),
            ("检查A", "Exam", 2),
        ],
    )
    connection.executemany(
        "INSERT INTO kg_triples VALUES (?,?,?,?,?,?,?,?)",
        [
            (
                "高血压",
                "Disease",
                "BELONGS_TO_DEPT",
                "所属科室",
                "心内科",
                "Department",
                0.9,
                1,
            ),
            (
                "高血压",
                "Disease",
                "TREATED_BY_DRUG",
                "治疗药物",
                "药A",
                "Drug",
                0.8,
                1,
            ),
            (
                "高血压",
                "Disease",
                "DIAGNOSED_BY",
                "诊断检查",
                "检查A",
                "Exam",
                0.8,
                1,
            ),
        ],
    )
    connection.commit()
    connection.close()


def test_schema_graph_is_versioned_canonical_and_isolated() -> None:
    first = schema_graph()
    second = schema_graph()

    assert first == second
    assert first is not second
    assert first["graph_version"] == SCHEMA_GRAPH_VERSION
    assert set(first["tables"]) == {
        "patients",
        "visits",
        "kg_entities",
        "kg_triples",
    }
    assert first["edges"]["patients_visits"]["cardinality"] == "one_to_many"
    assert first["edges"]["visits_disease_triples"]["cardinality"] == "many_to_many"
    assert first["edges"]["visits_belongs_to_department_triples"]["guards"]

    first["tables"]["patients"]["grain"].append("name")
    assert second["tables"]["patients"]["grain"] == ["patient_id"]

    payload = canonical_schema_graph_json()
    assert json.loads(payload)["graph_version"] == SCHEMA_GRAPH_VERSION
    digest = canonical_schema_graph_sha256()
    assert len(digest) == 64
    int(digest, 16)


def test_live_schema_graph_verifies_keys_cardinality_and_relation_guards(tmp_path) -> None:
    database = tmp_path / "analysis.sqlite"
    _database(database)

    report = verify_schema_graph(database)

    assert report["valid"] is True
    assert report["violations"] == []
    assert report["schema_graph_sha256"] == canonical_schema_graph_sha256()
    assert report["tables"]["patients"]["row_count"] == 1
    assert report["edges"]["patients_visits"]["valid"] is True
    assert report["type_guards"]["unknown_relation_rows"] == 0


def test_live_schema_graph_rejects_duplicate_logical_entity_key(tmp_path) -> None:
    database = tmp_path / "invalid.sqlite"
    _database(database)
    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO kg_entities VALUES ('高血压','Disease',4)")
    connection.commit()
    connection.close()

    report = verify_schema_graph(database)

    assert report["valid"] is False
    assert any("kg_entities" in reason for reason in report["violations"])


def test_live_schema_graph_reports_missing_columns_without_crashing(tmp_path) -> None:
    database = tmp_path / "missing-column.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE patients (patient_id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE visits (
            visit_id TEXT PRIMARY KEY, patient_id TEXT, visit_date TEXT,
            department TEXT, disease TEXT, drug_prescribed TEXT, exam TEXT,
            cost REAL, los_days INTEGER
        );
        CREATE TABLE kg_entities (name TEXT, type TEXT, mention_count INTEGER);
        CREATE TABLE kg_triples (
            head TEXT, head_type TEXT, relation TEXT, relation_zh TEXT,
            tail TEXT, tail_type TEXT, confidence REAL, count INTEGER
        );
        """
    )
    connection.commit()
    connection.close()

    report = verify_schema_graph(database)

    assert report["valid"] is False
    assert report["tables"]["patients"]["missing_columns"] == [
        "age",
        "gender",
        "region",
    ]
    assert report["edges"]["patients_visits"]["verified"] is False
    assert report["edges"]["patients_visits"]["valid"] is False


def test_live_schema_graph_supports_without_rowid_tables(tmp_path) -> None:
    database = tmp_path / "without-rowid.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE patients (
            patient_id TEXT PRIMARY KEY, name TEXT, gender TEXT,
            age INTEGER, region TEXT
        ) WITHOUT ROWID;
        CREATE TABLE visits (
            visit_id TEXT PRIMARY KEY, patient_id TEXT, visit_date TEXT,
            department TEXT, disease TEXT, drug_prescribed TEXT, exam TEXT,
            cost REAL, los_days INTEGER
        ) WITHOUT ROWID;
        CREATE TABLE kg_entities (
            name TEXT PRIMARY KEY, type TEXT, mention_count INTEGER
        ) WITHOUT ROWID;
        CREATE TABLE kg_triples (
            head TEXT, head_type TEXT, relation TEXT, relation_zh TEXT,
            tail TEXT, tail_type TEXT, confidence REAL, count INTEGER,
            PRIMARY KEY (head, relation, tail)
        ) WITHOUT ROWID;
        """
    )
    connection.commit()
    connection.close()

    report = verify_schema_graph(database)

    assert report["valid"] is True
    assert all(edge["verified"] for edge in report["edges"].values())
