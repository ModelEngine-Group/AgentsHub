"""
任务二知识图谱数据库结构模块。

该模块创建实体、关系、三元组、来源和质量审计表。
"""

import sqlite3, re, json, time, os, csv, io, subprocess, hashlib, tempfile, shutil
import requests
from typing import Any
from pathlib import Path
from collections import defaultdict
import sqlite3, time, os
from typing import Any
from pathlib import Path

from mcp_server.config import KG_DB

def _task2_sqlite_connect() -> sqlite3.Connection:
    db_path = Path(KG_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _task2_ensure_kg_schema(conn)
    return conn


def _task2_ensure_kg_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS kg_sources (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL UNIQUE,
            source_path TEXT,
            source_type TEXT,
            source_url TEXT,
            record_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS kg_entities (
            entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            description TEXT,
            source_id INTEGER,
            external_id TEXT,
            confidence REAL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            UNIQUE(canonical_name, entity_type)
        );
        CREATE TABLE IF NOT EXISTS kg_relations (
            relation_code TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            subject_type TEXT,
            object_type TEXT,
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS kg_triples (
            triple_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER NOT NULL,
            relation_code TEXT NOT NULL,
            object_id INTEGER NOT NULL,
            source_id INTEGER,
            source_file TEXT,
            source_record_id TEXT,
            evidence TEXT,
            confidence REAL DEFAULT 1.0,
            extractor TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(subject_id, relation_code, object_id, source_id, evidence)
        );
        CREATE TABLE IF NOT EXISTS kg_quality_issues (
            issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER,
            field_name TEXT,
            value TEXT,
            issue_type TEXT,
            evidence TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_kg_entities_name ON kg_entities(canonical_name);
        CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(entity_type);
        CREATE INDEX IF NOT EXISTS idx_kg_triples_relation ON kg_triples(relation_code);
        CREATE INDEX IF NOT EXISTS idx_kg_triples_subject ON kg_triples(subject_id);
        CREATE INDEX IF NOT EXISTS idx_kg_triples_object ON kg_triples(object_id);

        CREATE VIEW IF NOT EXISTS v_entity_stats AS
        SELECT entity_type, COUNT(*) AS entity_count
        FROM kg_entities
        GROUP BY entity_type;

        CREATE VIEW IF NOT EXISTS v_relation_stats AS
        SELECT r.relation_code, r.display_name, COUNT(t.triple_id) AS triple_count
        FROM kg_relations r
        LEFT JOIN kg_triples t ON r.relation_code = t.relation_code
        GROUP BY r.relation_code, r.display_name;

        CREATE VIEW IF NOT EXISTS v_disease_facts AS
        SELECT
            s.canonical_name AS disease,
            s.entity_type AS subject_type,
            r.relation_code,
            r.display_name AS relation_name,
            o.canonical_name AS object,
            o.entity_type AS object_type,
            t.evidence,
            t.confidence,
            src.source_name
        FROM kg_triples t
        JOIN kg_entities s ON t.subject_id = s.entity_id
        JOIN kg_entities o ON t.object_id = o.entity_id
        JOIN kg_relations r ON t.relation_code = r.relation_code
        LEFT JOIN kg_sources src ON t.source_id = src.source_id
        WHERE s.entity_type = 'disease';
        """
    )
    triple_columns = {row[1] for row in conn.execute("PRAGMA table_info(kg_triples)")}
    if "reliability_level" not in triple_columns:
        conn.execute("ALTER TABLE kg_triples ADD COLUMN reliability_level TEXT")
    if "source_file" not in triple_columns:
        conn.execute("ALTER TABLE kg_triples ADD COLUMN source_file TEXT")
    if "source_record_id" not in triple_columns:
        conn.execute("ALTER TABLE kg_triples ADD COLUMN source_record_id TEXT")
    conn.execute("DROP VIEW IF EXISTS v_disease_facts")
    conn.execute(
        """
        CREATE VIEW v_disease_facts AS
        SELECT
            s.canonical_name AS disease,
            s.entity_type AS subject_type,
            r.relation_code,
            r.display_name AS relation_name,
            o.canonical_name AS object,
            o.entity_type AS object_type,
            t.evidence,
            t.confidence,
            t.extractor,
            t.reliability_level,
            t.source_file,
            t.source_record_id,
            src.source_name
        FROM kg_triples t
        JOIN kg_entities s ON t.subject_id = s.entity_id
        JOIN kg_entities o ON t.object_id = o.entity_id
        JOIN kg_relations r ON t.relation_code = r.relation_code
        LEFT JOIN kg_sources src ON t.source_id = src.source_id
        WHERE s.entity_type = 'disease'
        """
    )


def _task2_psql_rows(sql: str) -> list[list[str]]:
    completed = subprocess.run(
        [
            "docker", "exec", "-i", "datamate-database", "psql",
            "-U", "postgres", "-d", "datamate", "-t", "-A", "-F", "\t",
        ],
        input=sql,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return [line.split("\t") for line in completed.stdout.splitlines() if "\t" in line]


def _task2_sql_literal(value: str) -> str:
    return "'" + str(value or "").replace("'", "''") + "'"


_TASK2_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
