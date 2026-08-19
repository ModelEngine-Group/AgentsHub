"""NL2SQL MVP operators for task 3."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseOperator


NL2SQL_CASES: list[dict[str, str]] = [
    {
        "question": "一共有多少条医疗记录？",
        "sql": "SELECT COUNT(*) AS count FROM medical_records",
    },
    {
        "question": "每个科室有多少患者？",
        "sql": "SELECT department, COUNT(*) AS patient_count FROM medical_records GROUP BY department ORDER BY patient_count DESC",
    },
    {
        "question": "肺炎患者用了什么药？",
        "sql": "SELECT patient_id, medicine FROM medical_records WHERE diagnosis = '肺炎'",
    },
    {
        "question": "年龄最大的患者是谁？",
        "sql": "SELECT patient_id, age, diagnosis FROM medical_records ORDER BY age DESC LIMIT 1",
    },
    {
        "question": "心血管内科有哪些诊断？",
        "sql": "SELECT patient_id, diagnosis FROM medical_records WHERE department = '心血管内科'",
    },
    {
        "question": "使用阿莫西林的患者有哪些？",
        "sql": "SELECT patient_id, diagnosis FROM medical_records WHERE medicine LIKE '%阿莫西林%'",
    },
]


class BuildMedicalSqliteOperator(BaseOperator):
    """Build a SQLite database from cleaned medical records CSV."""

    name = "BuildMedicalSqliteOperator"
    description = "将清洗后的医疗 CSV 导入 SQLite medical_records 表"
    input_schema = {"csv_path": "str", "db_path": "str"}
    output_schema = {"result": "dict"}

    def run(self, csv_path: str, db_path: str) -> dict[str, Any]:
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(csv_file, encoding="utf-8-sig")
        with sqlite3.connect(db_file) as conn:
            df.to_sql("medical_records", conn, if_exists="replace", index=False)
        return {"db_path": str(db_file), "table_name": "medical_records", "row_count": len(df)}


class NL2SQLOperator(BaseOperator):
    """Map supported Chinese questions to SQL templates."""

    name = "NL2SQLOperator"
    description = "将自然语言问题映射为 SQLite SQL"
    input_schema = {"question": "str"}
    output_schema = {"sql": "str"}

    def run(self, question: str) -> str:
        text = question.strip()
        if not text:
            raise ValueError("问题不能为空")

        for case in NL2SQL_CASES:
            if text == case["question"]:
                return case["sql"]

        if "多少条" in text or "总共有多少" in text or "一共有多少" in text:
            return "SELECT COUNT(*) AS count FROM medical_records"
        if "每个科室" in text or "科室有多少" in text:
            return "SELECT department, COUNT(*) AS patient_count FROM medical_records GROUP BY department ORDER BY patient_count DESC"
        if "年龄最大" in text:
            return "SELECT patient_id, age, diagnosis FROM medical_records ORDER BY age DESC LIMIT 1"
        if "肺炎" in text and ("药" in text or "用" in text):
            return "SELECT patient_id, medicine FROM medical_records WHERE diagnosis = '肺炎'"
        if "心血管内科" in text and "诊断" in text:
            return "SELECT patient_id, diagnosis FROM medical_records WHERE department = '心血管内科'"
        if "阿莫西林" in text:
            return "SELECT patient_id, diagnosis FROM medical_records WHERE medicine LIKE '%阿莫西林%'"
        return ""


class ExecuteSQLOperator(BaseOperator):
    """Execute SQL against the medical SQLite database."""

    name = "ExecuteSQLOperator"
    description = "执行 SQL 并返回 dict 行列表"
    input_schema = {"db_path": "str", "sql": "str"}
    output_schema = {"rows": "list[dict]"}

    def run(self, db_path: str, sql: str) -> list[dict[str, Any]]:
        if not sql.strip():
            return []
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql).fetchall()
        return [dict(row) for row in rows]


class EvaluateNL2SQLOperator(BaseOperator):
    """Evaluate template NL2SQL mapping accuracy."""

    name = "EvaluateNL2SQLOperator"
    description = "评估自然语言到 SQL 模板映射准确率"
    input_schema = {"cases": "list[dict] | None"}
    output_schema = {"metrics": "dict"}

    def run(self, cases: list[dict[str, str]] | None = None) -> dict[str, Any]:
        evaluation_cases = cases or NL2SQL_CASES
        mapper = NL2SQLOperator()
        details: list[dict[str, Any]] = []
        correct = 0
        for case in evaluation_cases:
            predicted = mapper.execute(case["question"])
            expected = case["sql"]
            matched = predicted == expected
            if matched:
                correct += 1
            details.append(
                {
                    "question": case["question"],
                    "expected_sql": expected,
                    "predicted_sql": predicted,
                    "matched": matched,
                }
            )

        total = len(evaluation_cases)
        accuracy = correct / total if total else 0.0
        return {"total": total, "correct": correct, "accuracy": accuracy, "details": details}
