from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict

from sqlglot import exp, parse_one

from analysis.open_sql.ast_guard import ALLOWED_FUNCTIONS
from analysis.open_sql.schema_catalog import allowed_fields, allowed_tables, db_path, get_schema_catalog

DENIED_ACTIONS = {
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_ANALYZE,
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_INDEX,
    sqlite3.SQLITE_DROP_TEMP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_TRIGGER,
    sqlite3.SQLITE_DROP_TEMP_VIEW,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_PRAGMA,
    sqlite3.SQLITE_REINDEX,
    sqlite3.SQLITE_TRANSACTION,
    sqlite3.SQLITE_UPDATE,
}


def execute_sql(
    sql: str,
    max_rows: int = 500,
    timeout_seconds: float = 5.0,
    max_vm_steps: int = 2_000_000,
) -> Dict[str, Any]:
    started = time.perf_counter()
    deadline = started + max(0.1, float(timeout_seconds))
    catalog = get_schema_catalog()
    table_set = allowed_tables(catalog)
    field_map = allowed_fields(catalog)
    try:
        parsed = parse_one(sql, read="sqlite")
        cte_names = {cte.alias_or_name for cte in parsed.find_all(exp.CTE)}
    except Exception:
        cte_names = set()
    progress = {"steps": 0, "limit": max(10_000, int(max_vm_steps))}

    def authorizer(action: int, arg1: str | None, arg2: str | None, _database: str | None, _source: str | None) -> int:
        if action in DENIED_ACTIONS:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ:
            table, field = str(arg1 or ""), str(arg2 or "")
            if table in cte_names and not field:
                return sqlite3.SQLITE_OK
            if table not in table_set or (field and field not in field_map.get(table, set())):
                return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            function_name = str(arg2 or arg1 or "").upper()
            if function_name and function_name not in ALLOWED_FUNCTIONS:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def progress_handler() -> int:
        progress["steps"] += 1000
        if progress["steps"] > progress["limit"] or time.perf_counter() >= deadline:
            return 1
        return 0

    uri = f"file:{db_path()}?mode=ro&immutable=1"
    try:
        with sqlite3.connect(uri, uri=True, timeout=timeout_seconds) as con:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA query_only = ON")
            con.set_authorizer(authorizer)
            con.set_progress_handler(progress_handler, 1000)
            rows = con.execute(sql).fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        result_rows = [dict(row) for row in rows[:max_rows]]
        return {
            "status": "success",
            "rows": result_rows,
            "row_count": len(result_rows),
            "truncated": truncated,
            "duration_seconds": round(time.perf_counter() - started, 4),
            "empty": len(result_rows) == 0,
            "vm_steps": progress["steps"],
            "resource_limits": {
                "max_rows": max_rows,
                "timeout_seconds": timeout_seconds,
                "max_vm_steps": max_vm_steps,
            },
        }
    except sqlite3.DatabaseError as exc:
        message = str(exc).lower()
        if "interrupted" in message:
            code, public_error = "query_resource_limit", "Query exceeded the configured execution budget."
        elif "not authorized" in message or "authorization denied" in message:
            code, public_error = "sqlite_authorizer_denied", "Query requested an operation outside the read-only policy."
        else:
            code, public_error = "query_execution_failed", "Query could not be executed safely."
        return {
            "status": "failed",
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "duration_seconds": round(time.perf_counter() - started, 4),
            "empty": True,
            "error_code": code,
            "error": public_error,
            "vm_steps": progress["steps"],
        }
