from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def semantic_four_table_db(tmp_path: Path) -> Path:
    path = tmp_path / "semantic_four_table.sqlite"
    connection = sqlite3.connect(path)
    try:
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
        connection.executemany(
            "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
            [
                ("p1", "甲", "女", 30, "南京"),
                ("p2", "乙", "男", 45, "南京"),
                ("p3", "丙", "女", 52, "北京"),
                ("p4", "丁", "男", 27, "上海"),
                ("p5", "戊", "女", 61, "上海"),
                ("p6", "己", "男", 39, "北京"),
            ],
        )
        connection.executemany(
            "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("v1", "p1", "2024-01-01", "内科", "高血压", "药甲", "血检", 100.111, 0),
                ("v2", "p1", "2024-01-02", "内科", "高血压", "药甲", "血检", 100.111, 2),
                ("v3", "p2", "2024-01-03", "内科", "糖尿病", "药乙", "CT", 200.225, 1),
                ("v4", "p3", "2024-02-01", "外科", "骨折", "药丙", "X光", 300.335, 5),
                ("v5", "p3", "2024-02-02", "外科", "骨折", "药丙", "X光", 300.335, 4),
                ("v6", "p5", "2024-03-01", "外科", "阑尾炎", "药丁", "超声", 450.445, 3),
                ("v7", "p6", "2024-03-02", "儿科", "感冒", "药戊", "血检", 80.125, 0),
                ("v8", "p1", "2024-03-03", "外科", "高血压", "药甲", "CT", 600.665, 1),
            ],
        )
        connection.executemany(
            "INSERT INTO kg_entities VALUES (?, ?, ?)",
            [
                ("高血压", "Disease", 4),
                ("骨折", "Disease", 2),
                ("孤立疾病", "Disease", 1),
                ("头痛", "Symptom", 3),
                ("发热", "Symptom", 2),
                ("孤立症状", "Symptom", 1),
            ],
        )
        connection.executemany(
            "INSERT INTO kg_triples VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("高血压", "Disease", "HAS_SYMPTOM", "临床表现", "头痛", "Symptom", 0.95, 3),
                ("高血压", "Disease", "HAS_SYMPTOM", "临床表现", "发热", "Symptom", 0.85, 1),
                ("骨折", "Disease", "DIAGNOSED_BY", "诊断检查", "X光", "Exam", 0.90, 2),
                ("骨折", "Disease", "HAS_SYMPTOM", "临床表现", "疼痛", "Symptom", 0.88, 2),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return path
