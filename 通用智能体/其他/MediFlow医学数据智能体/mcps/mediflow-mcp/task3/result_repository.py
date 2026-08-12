"""任务三分析结果的持久化存储。"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


class AnalysisResultRepository:
    """按分析编号原子保存结果，使页面与报告复用同一份数据。"""

    def __init__(self, root: str | Path, max_records: int = 128):
        self.root = Path(root)
        self.max_records = max(1, int(max_records))
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalise_id(analysis_id: str) -> str:
        return str(uuid.UUID(str(analysis_id or "")))

    def save(self, result: dict[str, Any]) -> None:
        analysis_id = self._normalise_id(str(result.get("analysis_id") or ""))
        target = self.root / f"{analysis_id}.json"
        temporary = self.root / f".{analysis_id}.{os.getpid()}.tmp"
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, target)
        self._prune()

    def load(self, analysis_id: str) -> dict[str, Any] | None:
        try:
            target = self.root / f"{self._normalise_id(analysis_id)}.json"
        except (TypeError, ValueError, AttributeError):
            return None
        if not target.is_file():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _prune(self) -> None:
        files = sorted(
            self.root.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in files[self.max_records :]:
            try:
                path.unlink()
            except OSError:
                continue
