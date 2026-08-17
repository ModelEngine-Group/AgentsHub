"""
分析库刷新模块。

该模块从任务二知识图谱库刷新任务三分析库。
"""

from __future__ import annotations

import time
from pathlib import Path

from kg.build_analytics_v2 import build_from_kg
from mcp_server.config import ANALYTICS_DB, KG_DB


def refresh_task3_analytics(
    kg_db: str | Path = KG_DB,
    analytics_db: str | Path = ANALYTICS_DB,
) -> dict:
    start = time.time()
    kg_path = Path(kg_db)
    analytics_path = Path(analytics_db)
    if not kg_path.exists():
        return {
            "status": "error",
            "error": f"KG database does not exist: {kg_path}",
            "kg_db": str(kg_path),
            "analytics_db": str(analytics_path),
        }

    stats = build_from_kg(kg_path, analytics_path)
    return {
        "status": "success",
        "kg_db": str(kg_path),
        "analytics_db": str(analytics_path),
        "elapsed_seconds": round(time.time() - start, 3),
        "stats": stats,
    }
