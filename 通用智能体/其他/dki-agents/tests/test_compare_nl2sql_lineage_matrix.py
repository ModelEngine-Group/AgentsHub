from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "compare_nl2sql_lineage_matrix.py"


def _engine(root: Path, sql: str) -> None:
    package = root / "dki" / "analysis"
    package.mkdir(parents=True)
    (root / "dki" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "nl2sql.py").write_text(
        "class NL2SQL:\n"
        "    def __init__(self, db_path):\n"
        "        self.db_path = db_path\n"
        "    def rule_sql(self, question):\n"
        f"        return {sql!r}\n",
        encoding="utf-8",
    )


def _database(path: Path, values: list[int]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE t(value INTEGER)")
        connection.executemany("INSERT INTO t(value) VALUES (?)", [(v,) for v in values])
        connection.commit()
    finally:
        connection.close()


def test_matrix_uses_a_fresh_engine_process_for_every_cell(tmp_path: Path):
    historical_engine = tmp_path / "historical_engine"
    current_engine = tmp_path / "current_engine"
    _engine(
        historical_engine,
        "SELECT value FROM t WHERE value = 1 ORDER BY value",
    )
    _engine(current_engine, "SELECT value FROM t ORDER BY value")
    historical_db = tmp_path / "historical.db"
    current_db = tmp_path / "current.db"
    _database(historical_db, [1])
    _database(current_db, [1, 2])
    development = tmp_path / "development.jsonl"
    development.write_text(
        json.dumps(
            {
                "id": "dev-1",
                "question": "all values",
                "gold_sql": "SELECT value FROM t ORDER BY value",
                "difficulty": "hard",
                "category": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "matrix.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--historical-engine-root",
        str(historical_engine),
        "--current-engine-root",
        str(current_engine),
        "--historical-db",
        str(historical_db),
        "--current-db",
        str(current_db),
        "--dataset",
        f"development={development}",
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    cells = json.loads(output.read_text(encoding="utf-8"))["matrix"]["development"]
    assert cells["historical_engine__historical_db"]["correct"] == 1
    assert cells["historical_engine__current_db"]["correct"] == 0
    assert cells["current_engine__historical_db"]["correct"] == 1
    assert cells["current_engine__current_db"]["correct"] == 1
    assert "historical_engine" in cells["historical_engine__current_db"]["engine_module"]
    assert "current_engine" in cells["current_engine__current_db"]["engine_module"]

    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert second.returncode != 0
    assert "refusing to overwrite existing receipt" in second.stderr
