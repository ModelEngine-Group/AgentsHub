import sqlite3
from pathlib import Path

from operators import (
    BuildMedicalSqliteOperator,
    EvaluateNL2SQLOperator,
    ExecuteSQLOperator,
    NL2SQLOperator,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CSV_PATH = OUTPUTS_DIR / "cleaned_records.csv"


def test_build_medical_sqlite_operator_creates_database_table(tmp_path):
    db_path = tmp_path / "medical_records.db"

    result = BuildMedicalSqliteOperator().execute(str(CSV_PATH), str(db_path))

    assert result["db_path"] == str(db_path)
    assert result["table_name"] == "medical_records"
    assert result["row_count"] == 15
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM medical_records").fetchone()[0]
    assert count == 15


def test_nl2sql_operator_maps_chinese_questions_to_sql():
    operator = NL2SQLOperator()

    assert operator.execute("一共有多少条医疗记录？") == "SELECT COUNT(*) AS count FROM medical_records"
    assert operator.execute("每个科室有多少患者？") == (
        "SELECT department, COUNT(*) AS patient_count FROM medical_records GROUP BY department ORDER BY patient_count DESC"
    )
    assert operator.execute("肺炎患者用了什么药？") == (
        "SELECT patient_id, medicine FROM medical_records WHERE diagnosis = '肺炎'"
    )


def test_execute_sql_operator_returns_query_rows(tmp_path):
    db_path = tmp_path / "medical_records.db"
    BuildMedicalSqliteOperator().execute(str(CSV_PATH), str(db_path))

    rows = ExecuteSQLOperator().execute(
        str(db_path),
        "SELECT diagnosis, medicine FROM medical_records WHERE diagnosis = '肺炎'",
    )

    assert rows == [{"diagnosis": "肺炎", "medicine": "阿莫西林"}]


def test_evaluate_nl2sql_operator_reaches_target_accuracy():
    result = EvaluateNL2SQLOperator().execute()

    assert result["total"] >= 5
    assert result["accuracy"] >= 0.85
    assert result["correct"] == result["total"]
