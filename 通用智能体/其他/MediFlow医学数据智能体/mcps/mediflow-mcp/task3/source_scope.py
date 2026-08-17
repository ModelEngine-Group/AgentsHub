"""任务三分析库的数据来源口径。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def build_analysis_scope(kg_db_path: str | Path) -> dict[str, Any]:
    """返回分析库快照的登记来源，避免把汇总查询误述为单源查询。"""

    path = Path(kg_db_path)
    if not path.is_file():
        return {
            "mode": "aggregate_snapshot",
            "status": "unavailable",
            "registered_source_count": None,
            "sources": [],
            "statement": "本轮查询基于当前分析库快照；来源登记库当前不可用。",
        }
    try:
        with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT source_id, source_name, source_type, record_count, created_at
                FROM kg_sources
                ORDER BY source_id
                """
            ).fetchall()
    except sqlite3.Error:
        return {
            "mode": "aggregate_snapshot",
            "status": "unavailable",
            "registered_source_count": None,
            "sources": [],
            "statement": "本轮查询基于当前分析库快照；来源登记信息暂不可读。",
        }
    sources = [dict(row) for row in rows]
    count = len(sources)
    return {
        "mode": "aggregate_snapshot",
        "status": "ready",
        "registered_source_count": count,
        "sources": sources,
        "statement": (
            f"本轮查询基于当前分析库汇总快照；该快照登记了 {count} 个数据来源，"
            "结果未按单一来源过滤。"
        ),
    }
