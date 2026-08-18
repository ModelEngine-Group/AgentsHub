from __future__ import annotations

from typing import List, Tuple

from analysis.open_sql.ast_guard import validate_sql as validate_sql_ast


def validate_sql(sql: str | None) -> Tuple[bool, List[str], str | None, List[str]]:
    result = validate_sql_ast(sql, max_result_rows=200)
    errors = [str(item.get("message") or item.get("code")) for item in result.get("errors", [])]
    return bool(result.get("safe")), errors, result.get("normalized_sql"), list(result.get("warnings") or [])
