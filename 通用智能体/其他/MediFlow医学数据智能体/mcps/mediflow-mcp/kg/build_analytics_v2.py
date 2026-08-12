#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务三分析库构建脚本。

该脚本从任务二知识图谱库聚合疾病、症状、药物、检查和科室等统计表。
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


DDL = """
CREATE TABLE IF NOT EXISTS diseases (
    disease_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    source_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS disease_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    fact_name TEXT NOT NULL,
    evidence TEXT,
    source_name TEXT,
    confidence REAL
);

CREATE TABLE IF NOT EXISTS disease_symptoms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT NOT NULL,
    symptom TEXT NOT NULL,
    confidence REAL,
    source_name TEXT
);

CREATE TABLE IF NOT EXISTS disease_drugs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT NOT NULL,
    drug TEXT NOT NULL,
    confidence REAL,
    source_name TEXT
);

CREATE TABLE IF NOT EXISTS disease_departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT NOT NULL,
    department TEXT NOT NULL,
    confidence REAL,
    source_name TEXT
);

CREATE TABLE IF NOT EXISTS disease_complications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT NOT NULL,
    complication TEXT NOT NULL,
    confidence REAL,
    source_name TEXT
);

CREATE TABLE IF NOT EXISTS disease_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT NOT NULL,
    test TEXT NOT NULL,
    confidence REAL,
    source_name TEXT
);

CREATE TABLE IF NOT EXISTS disease_procedures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT NOT NULL,
    procedure TEXT NOT NULL,
    confidence REAL,
    source_name TEXT
);

CREATE TABLE IF NOT EXISTS disease_causes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT NOT NULL,
    cause TEXT NOT NULL,
    evidence TEXT,
    confidence REAL,
    source_name TEXT
);

CREATE TABLE IF NOT EXISTS disease_preventions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT NOT NULL,
    prevention TEXT NOT NULL,
    evidence TEXT,
    confidence REAL,
    source_name TEXT
);

CREATE TABLE IF NOT EXISTS disease_populations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT NOT NULL,
    population TEXT NOT NULL,
    confidence REAL,
    source_name TEXT
);

CREATE TABLE IF NOT EXISTS entity_stats (
    entity_type TEXT PRIMARY KEY,
    entity_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS relation_stats (
    relation_code TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    triple_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS qa_examples (
    qa_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id TEXT,
    question TEXT,
    answer_id TEXT,
    answer TEXT,
    source_name TEXT
);

CREATE TABLE IF NOT EXISTS nl2sql_query_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT,
    sql TEXT,
    row_count INTEGER,
    error TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_disease_facts_disease ON disease_facts(disease);
CREATE INDEX IF NOT EXISTS idx_disease_facts_type ON disease_facts(fact_type);
CREATE INDEX IF NOT EXISTS idx_disease_symptoms_symptom ON disease_symptoms(symptom);
CREATE INDEX IF NOT EXISTS idx_disease_drugs_drug ON disease_drugs(drug);
CREATE INDEX IF NOT EXISTS idx_disease_departments_department ON disease_departments(department);
CREATE INDEX IF NOT EXISTS idx_disease_tests_test ON disease_tests(test);
CREATE INDEX IF NOT EXISTS idx_disease_procedures_procedure ON disease_procedures(procedure);
CREATE INDEX IF NOT EXISTS idx_disease_causes_disease ON disease_causes(disease);
CREATE INDEX IF NOT EXISTS idx_disease_preventions_disease ON disease_preventions(disease);
CREATE INDEX IF NOT EXISTS idx_disease_populations_population ON disease_populations(population);

CREATE VIEW IF NOT EXISTS v_entity_type_counts AS
SELECT entity_type, entity_count FROM entity_stats ORDER BY entity_count DESC;

CREATE VIEW IF NOT EXISTS v_relation_counts AS
SELECT relation_code, display_name, triple_count FROM relation_stats ORDER BY triple_count DESC;

CREATE VIEW IF NOT EXISTS v_top_symptoms AS
SELECT symptom, COUNT(DISTINCT disease) AS disease_count
FROM disease_symptoms
GROUP BY symptom
ORDER BY disease_count DESC, symptom
LIMIT 100;

CREATE VIEW IF NOT EXISTS v_department_disease_counts AS
SELECT department, COUNT(DISTINCT disease) AS disease_count
FROM disease_departments
GROUP BY department
ORDER BY disease_count DESC, department;

CREATE VIEW IF NOT EXISTS v_drug_disease_counts AS
SELECT drug, COUNT(DISTINCT disease) AS disease_count
FROM disease_drugs
GROUP BY drug
ORDER BY disease_count DESC, drug
LIMIT 100;
"""


def reset_tables(conn: sqlite3.Connection) -> None:
    for table in (
        "diseases",
        "disease_facts",
        "disease_symptoms",
        "disease_drugs",
        "disease_departments",
        "disease_complications",
        "disease_tests",
        "disease_procedures",
        "disease_causes",
        "disease_preventions",
        "disease_populations",
        "entity_stats",
        "relation_stats",
        "qa_examples",
    ):
        conn.execute(f"DELETE FROM {table}")


def build_from_kg(kg_db: Path, analytics_db: Path) -> dict[str, int]:
    analytics_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(analytics_db))
    conn.execute("ATTACH DATABASE ? AS kg", (str(kg_db),))
    conn.executescript(DDL)
    reset_tables(conn)

    conn.execute(
        """
        INSERT OR IGNORE INTO diseases (disease_id, name, description, source_count)
        SELECT e.entity_id, e.canonical_name, e.description, COUNT(DISTINCT t.source_id)
        FROM kg.kg_entities e
        LEFT JOIN kg.kg_triples t ON t.subject_id = e.entity_id OR t.object_id = e.entity_id
        WHERE e.entity_type = 'disease'
        GROUP BY e.entity_id, e.canonical_name, e.description
        """
    )
    conn.execute(
        """
        INSERT INTO disease_facts
            (disease, fact_type, fact_name, evidence, source_name, confidence)
        SELECT disease, relation_code, object, evidence, source_name, confidence
        FROM kg.v_disease_facts
        """
    )

    relation_targets = [
        ("disease_symptoms", "symptom", "has_symptom"),
        ("disease_drugs", "drug", "treated_by_drug"),
        ("disease_departments", "department", "visit_department"),
        ("disease_complications", "complication", "has_complication"),
        ("disease_tests", "test", "requires_test"),
        ("disease_procedures", "procedure", "treated_by_procedure"),
        ("disease_populations", "population", "susceptible_population"),
    ]
    for table, column, relation_code in relation_targets:
        conn.execute(
            f"""
            INSERT INTO {table} (disease, {column}, confidence, source_name)
            SELECT disease, object, confidence, source_name
            FROM kg.v_disease_facts
            WHERE relation_code = ?
            """,
            (relation_code,),
        )

    text_targets = [
        ("disease_causes", "cause", "has_cause"),
        ("disease_preventions", "prevention", "has_prevention"),
    ]
    for table, column, relation_code in text_targets:
        conn.execute(
            f"""
            INSERT INTO {table}
                (disease, {column}, evidence, confidence, source_name)
            SELECT disease, object, evidence, confidence, source_name
            FROM kg.v_disease_facts
            WHERE relation_code = ?
            """,
            (relation_code,),
        )

    conn.execute(
        """
        INSERT INTO entity_stats (entity_type, entity_count)
        SELECT entity_type, entity_count FROM kg.v_entity_stats
        """
    )
    conn.execute(
        """
        INSERT INTO relation_stats (relation_code, display_name, triple_count)
        SELECT relation_code, display_name, triple_count FROM kg.v_relation_stats
        """
    )
    conn.execute(
        """
        DELETE FROM disease_symptoms
        WHERE symptom IN (
            SELECT value
            FROM kg.kg_quality_issues
            WHERE issue_type IN ('suspicious_symptom', 'invalid_text')
        )
        """
    )
    conn.execute(
        """
        DELETE FROM disease_facts
        WHERE fact_type = 'has_symptom'
          AND fact_name IN (
              SELECT value
              FROM kg.kg_quality_issues
              WHERE issue_type IN ('suspicious_symptom', 'invalid_text')
          )
        """
    )
    conn.execute("DELETE FROM disease_facts WHERE fact_type LIKE 'cmeie_%'")
    conn.execute("DELETE FROM relation_stats WHERE relation_code LIKE 'cmeie_%'")
    conn.commit()

    stats = {
        table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in (
            "diseases",
            "disease_facts",
            "disease_symptoms",
            "disease_drugs",
            "disease_departments",
            "disease_complications",
            "disease_tests",
            "disease_procedures",
            "disease_causes",
            "disease_preventions",
            "disease_populations",
            "entity_stats",
            "relation_stats",
            "qa_examples",
        )
    }
    conn.close()
    return stats


def detect_encoding(path: Path) -> str:
    raw = path.read_bytes()[:65536]
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def import_cmedqa(
    analytics_db: Path,
    question_csv: Path,
    answer_csv: Path,
    limit: int | None = 1000,
) -> int:
    if not question_csv.exists() or not answer_csv.exists():
        return 0

    q_encoding = detect_encoding(question_csv)
    a_encoding = detect_encoding(answer_csv)
    questions: dict[str, str] = {}
    with question_csv.open("r", encoding=q_encoding, errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            qid = row.get("question_id") or row.get("qid") or ""
            content = row.get("content") or row.get("question") or ""
            if qid and content:
                questions[qid] = content
            if limit is not None and len(questions) >= limit * 5:
                break

    inserted = 0
    conn = sqlite3.connect(str(analytics_db))
    with answer_csv.open("r", encoding=a_encoding, errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            qid = row.get("question_id") or row.get("qid") or ""
            if qid not in questions:
                continue
            conn.execute(
                """
                INSERT INTO qa_examples
                    (question_id, question, answer_id, answer, source_name)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    qid,
                    questions[qid],
                    row.get("ans_id") or row.get("answer_id") or "",
                    row.get("content") or row.get("answer") or "",
                    "cMedQA2",
                ),
            )
            inserted += 1
            if limit is not None and inserted >= limit:
                break
    conn.commit()
    conn.close()
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-db", default="data/task2_medical_kg.db")
    parser.add_argument("--analytics-db", default="data/task3_analytics.db")
    parser.add_argument("--cmedqa-question", default=None)
    parser.add_argument("--cmedqa-answer", default=None)
    parser.add_argument("--qa-limit", type=int, default=1000)
    args = parser.parse_args()

    stats = build_from_kg(Path(args.kg_db), Path(args.analytics_db))
    if args.cmedqa_question and args.cmedqa_answer:
        stats["qa_examples"] = import_cmedqa(
            Path(args.analytics_db),
            Path(args.cmedqa_question),
            Path(args.cmedqa_answer),
            args.qa_limit,
        )
    print(f"analytics_db={args.analytics_db}")
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
