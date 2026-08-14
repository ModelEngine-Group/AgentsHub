from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def relative_to_project(path: str | Path) -> str:
    resolved = resolve_path(path)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_directory(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_missing_values(rows: Iterable[Dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        for value in row.values():
            if value is None:
                total += 1
    return total


def build_result(
    *,
    task_id: str,
    operator: str,
    status: str,
    input_path: str | Path,
    output_path: str | Path,
    metrics: Dict[str, Any] | None = None,
    errors: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "operator": operator,
        "status": status,
        "input_path": relative_to_project(input_path),
        "output_path": relative_to_project(output_path),
        "metrics": metrics or {},
        "errors": errors or [],
    }


def payload_with_defaults(
    *,
    task_id: str,
    input_path: str | Path,
    export_path: str | Path,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "input_path": relative_to_project(input_path),
        "export_path": relative_to_project(export_path),
        "params": {
            "encoding": "utf-8",
            "overwrite": True,
            **(params or {}),
        },
    }


class Timer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return round(time.perf_counter() - self._start, 4)
