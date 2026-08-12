"""任务三只读 SQL 校验与受限执行。"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Sequence


_UNSAFE_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|"
    r"pragma|vacuum|reindex|analyze|load_extension)\b",
    re.IGNORECASE,
)
_COMMENT_RE = re.compile(r"(--|/\*)")
_ALLOWED_RELATIONS = {
    "diseases",
    "disease_symptoms",
    "disease_drugs",
    "disease_complications",
    "disease_departments",
    "disease_tests",
    "disease_procedures",
    "disease_populations",
    "disease_causes",
    "disease_preventions",
    "disease_facts",
    "entity_stats",
    "relation_stats",
    "qa_examples",
    "v_department_disease_counts",
    "v_drug_disease_counts",
    "v_top_symptoms",
}


class SqlSafetyError(ValueError):
    """SQL 不满足只读执行约束。"""


def validate_readonly_sql(sql: str) -> str:
    statement = str(sql or "").strip().rstrip(";").strip()
    if not statement:
        raise SqlSafetyError("SQL 为空")
    if ";" in statement:
        raise SqlSafetyError("只允许执行一条 SQL")
    if _COMMENT_RE.search(statement):
        raise SqlSafetyError("SQL 不允许包含注释")
    if _UNSAFE_RE.search(statement):
        raise SqlSafetyError("SQL 包含写入或结构变更语句")
    if not re.match(r"^(select|with)\b", statement, flags=re.IGNORECASE):
        raise SqlSafetyError("只允许 SELECT/WITH 查询")
    cte_names = {
        name.lower()
        for name in re.findall(
            r"(?:\bwith\b|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s+as\s*\(",
            statement,
            flags=re.IGNORECASE,
        )
    }
    relations = {
        name.lower()
        for name in re.findall(
            r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)",
            statement,
            flags=re.IGNORECASE,
        )
    }
    unknown = relations - _ALLOWED_RELATIONS - cte_names
    if unknown:
        raise SqlSafetyError(f"SQL 引用了未授权的数据表：{', '.join(sorted(unknown))}")
    return statement


def execute_readonly(
    db_path: str | Path,
    sql: str,
    params: Sequence[Any] = (),
    *,
    max_rows: int = 200,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """在只读连接中执行 SQL，并限制执行时间与返回行数。"""

    statement = validate_readonly_sql(sql)
    path = Path(db_path).resolve()
    started = time.perf_counter()
    deadline = started + timeout_seconds
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 1000)
    try:
        cursor = conn.execute(statement, tuple(params))
        fetched = cursor.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows
        rows = [dict(row) for row in fetched[:max_rows]]
        columns = [item[0] for item in cursor.description] if cursor.description else []
    finally:
        conn.close()
    return {
        "sql": statement,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }
