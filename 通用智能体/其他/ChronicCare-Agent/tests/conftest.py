from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def sample_graph() -> dict:
    return json.loads((FIXTURES_DIR / "sample_graph.json").read_text(encoding="utf-8"))


@pytest.fixture
def open_sql_cases() -> list[dict]:
    return json.loads((FIXTURES_DIR / "open_sql_cases.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_sqlite_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "sample_chroniccare.db"
    table_dir = FIXTURES_DIR / "sample_tables"
    schemas = {
        "patient_profile": (
            "CREATE TABLE patient_profile ("
            "patient_id TEXT PRIMARY KEY, age INTEGER, gender TEXT, "
            "disease_tags TEXT, bmi REAL)"
        ),
        "followup_plan": (
            "CREATE TABLE followup_plan ("
            "plan_id TEXT PRIMARY KEY, patient_id TEXT, followup_date TEXT, "
            "priority TEXT, status TEXT)"
        ),
        "lab_result": (
            "CREATE TABLE lab_result ("
            "lab_id TEXT PRIMARY KEY, patient_id TEXT, item_name TEXT, "
            "value REAL, item_value REAL, abnormal_flag TEXT, test_date TEXT)"
        ),
    }
    with sqlite3.connect(db_path) as connection:
        for table, ddl in schemas.items():
            connection.execute(ddl)
            with (table_dir / f"{table}.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            columns = list(rows[0])
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                [[row[column] for column in columns] for row in rows],
            )
    return db_path


@pytest.fixture
def sample_schema_catalog(sample_sqlite_db: Path) -> dict:
    with sqlite3.connect(sample_sqlite_db) as connection:
        tables = {}
        for table in ("patient_profile", "followup_plan", "lab_result"):
            fields = [{"name": row[1], "type": row[2]} for row in connection.execute(f"PRAGMA table_info({table})")]
            tables[table] = {"name": table, "fields": fields}
    return {
        "status": "success",
        "tables": tables,
        "joins": [
            {
                "left_table": "patient_profile",
                "left_field": "patient_id",
                "right_table": "followup_plan",
                "right_field": "patient_id",
            },
            {
                "left_table": "patient_profile",
                "left_field": "patient_id",
                "right_table": "lab_result",
                "right_field": "patient_id",
            },
        ],
    }
