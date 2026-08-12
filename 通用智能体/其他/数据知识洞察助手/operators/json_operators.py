"""JSON operator implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseOperator


class SaveJsonOperator(BaseOperator):
    """Save Python dict/list data as UTF-8 JSON."""

    name = "SaveJsonOperator"
    description = "将 dict/list 保存为 UTF-8 JSON 文件"
    input_schema = {"data": "dict | list", "file_path": "str"}
    output_schema = {"saved_path": "str"}

    def run(self, data: dict[str, Any] | list[Any], file_path: str) -> str:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[SaveJsonOperator] saved: {path}")
        return str(path)
