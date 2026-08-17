"""
任务一异步任务状态模块。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TASK1_ASYNC_STATUS_ROOT = Path(
    os.environ.get("CCF_TASK1_STATUS_ROOT", ROOT / "data" / "task1_mixed_agent_runs")
)


def task1_async_status_path(run_id: str) -> Path:
    return TASK1_ASYNC_STATUS_ROOT / run_id / "status.json"


def write_task1_async_status(run_id: str, payload: dict[str, Any]) -> None:
    path = task1_async_status_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = dict(payload)
    normalized.setdefault("run_id", run_id)
    normalized["updated_at"] = int(time.time())
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_task1_async_status(run_id: str) -> dict[str, Any]:
    path = task1_async_status_path(run_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_task1_async_status(run_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    payload = read_task1_async_status(run_id)
    payload.update(updates)
    write_task1_async_status(run_id, payload)
    return payload
