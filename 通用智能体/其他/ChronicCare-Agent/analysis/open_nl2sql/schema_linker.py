from __future__ import annotations

from typing import Dict, List

from analysis.open_nl2sql.schema_registry import get_schema_registry


def build_schema_links(question: str) -> Dict[str, List[str]]:
    matched_tables: List[str] = []
    matched_columns: List[str] = []
    lowered = question.lower()
    for item in get_schema_registry():
        aliases = [str(alias).lower() for alias in item.get("chinese_alias", [])]
        if any(alias in lowered for alias in aliases):
            table = str(item["table"])
            field = str(item["field"])
            if table not in matched_tables:
                matched_tables.append(table)
            if field not in matched_columns:
                matched_columns.append(field)
    return {
        "tables": matched_tables,
        "columns": matched_columns,
    }
