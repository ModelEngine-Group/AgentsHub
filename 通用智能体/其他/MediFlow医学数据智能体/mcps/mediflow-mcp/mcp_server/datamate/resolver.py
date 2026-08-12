"""
DataMate 数据集解析模块。

该模块把数据集名称或 UUID 解析为实际数据集，并读取文件清单和文件内容。
"""

from pathlib import Path
import re
import subprocess
from typing import Any

from mcp_server.config import DATASET_VOLUME
from mcp_server.datamate.client import _sudo_command
from mcp_server.kg.schema import _TASK2_UUID_RE, _task2_psql_rows, _task2_sql_literal

_DATASET_VOLUME = DATASET_VOLUME

def normalize_identifier(value):
    return re.sub(r'[\s\-_]+', '', str(value).strip().lower())

def _task2_dataset_from_row(row: list[str], requested: str = "", resolved_by: str = "") -> dict[str, Any]:
    return {
        "id": row[0],
        "name": row[1],
        "format": row[2],
        "dataset_type": row[3],
        "status": row[4],
        "requested_identifier": requested,
        "resolved_by": resolved_by,
    }


def _task2_normalize_dataset_identifier(value: str) -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "清理": "清洗",
        "保留": "保持",
        "来源": "源",
        "结果集": "结果",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return "".join(ch for ch in text if ch.isalnum())


def _task2_resolve_datamate_dataset(identifier: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """解析任务一和任务二共用的 DataMate 数据集标识。

    The user may provide a UUID, exact dataset name, partial name, or a name with
    minor wording differences while preserving a unique timestamp suffix.
    """
    requested = (identifier or "").strip()
    if not requested:
        raise ValueError("dataset_id is required")

    select_cols = "id, name, coalesce(format,''), coalesce(dataset_type,''), coalesce(status,'')"
    candidates: list[dict[str, str]] = []

    def rows_to_candidates(rows: list[list[str]]) -> list[dict[str, str]]:
        return [{"id": row[0], "name": row[1], "status": row[4]} for row in rows if len(row) >= 5]

    if _TASK2_UUID_RE.match(requested):
        rows = _task2_psql_rows(
            f"select {select_cols} from t_dm_datasets where id={_task2_sql_literal(requested)} limit 1;"
        )
        if rows:
            return _task2_dataset_from_row(rows[0], requested, "id"), []

    rows = _task2_psql_rows(
        f"select {select_cols} from t_dm_datasets "
        f"where name={_task2_sql_literal(requested)} "
        "order by created_at desc limit 5;"
    )
    if rows:
        candidates = rows_to_candidates(rows)
        return _task2_dataset_from_row(rows[0], requested, "exact_name"), candidates

    if len(requested) >= 4:
        rows = _task2_psql_rows(
            f"select {select_cols} from t_dm_datasets "
            f"where name like {_task2_sql_literal('%' + requested + '%')} "
            "order by created_at desc limit 10;"
        )
        if rows:
            candidates = rows_to_candidates(rows)
            return _task2_dataset_from_row(rows[0], requested, "name_contains"), candidates

    digit_groups = sorted(re.findall(r"\d{6,}", requested), key=len, reverse=True)
    for digits in digit_groups[:2]:
        rows = _task2_psql_rows(
            f"select {select_cols} from t_dm_datasets "
            f"where name like {_task2_sql_literal('%' + digits + '%')} "
            "order by created_at desc limit 10;"
        )
        if rows:
            candidates = rows_to_candidates(rows)
            return _task2_dataset_from_row(rows[0], requested, f"timestamp_digits:{digits}"), candidates

    wanted_norm = _task2_normalize_dataset_identifier(requested)
    if wanted_norm:
        rows = _task2_psql_rows(
            f"select {select_cols} from t_dm_datasets "
            "where status in ('ACTIVE','PROCESSING','PUBLISHED','DRAFT') "
            "order by created_at desc limit 200;"
        )
        scored: list[tuple[int, list[str]]] = []
        for row in rows:
            name_norm = _task2_normalize_dataset_identifier(row[1])
            if not name_norm:
                continue
            score = 0
            if wanted_norm in name_norm or name_norm in wanted_norm:
                score += min(len(wanted_norm), len(name_norm))
            common_digits = set(re.findall(r"\d{6,}", wanted_norm)) & set(re.findall(r"\d{6,}", name_norm))
            if common_digits:
                score += 100
            if score:
                scored.append((score, row))
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            rows = [row for _, row in scored[:10]]
            candidates = rows_to_candidates(rows)
            return _task2_dataset_from_row(rows[0], requested, "relaxed_name"), candidates

    raise ValueError(f"DataMate dataset not found: {requested}")


def _read_dataset_file_text(file_path: Path) -> str | None:
    try:
        return file_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except PermissionError:
        completed = _sudo_command(["cat", str(file_path)])
        if completed.returncode == 0:
            return completed.stdout
        return None


def _task2_read_datamate_dataset(dataset_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset, candidates = _task2_resolve_datamate_dataset(dataset_id)
    did = dataset["id"]
    if candidates:
        dataset["resolution_candidates"] = candidates[:5]
    ds_rows = _task2_psql_rows(
        "select id, name, coalesce(format,''), coalesce(dataset_type,''), coalesce(status,'') "
        f"from t_dm_datasets where id={_task2_sql_literal(did)} limit 1;"
    )
    if not ds_rows:
        raise ValueError(f"DataMate dataset not found: {did}")
    dataset.update({"format": ds_rows[0][2], "dataset_type": ds_rows[0][3], "status": ds_rows[0][4]})
    rows = _task2_psql_rows(
        "select file_name, coalesce(file_path,''), coalesce(file_type,''), coalesce(file_size,0)::text "
        "from t_dm_dataset_files "
        f"where dataset_id={_task2_sql_literal(did)} and status in ('ACTIVE','COMPLETED') "
        "order by file_name;"
    )
    dataset_root = Path(_DATASET_VOLUME).resolve()
    dataset_dir = (dataset_root / did).resolve()
    files = []
    for file_name, stored_file_path, file_type, file_size in rows:
        normalized = (stored_file_path or "").replace("\\", "/")
        marker = "/dataset/"
        if marker in normalized:
            relative_path = normalized.split(marker, 1)[1].lstrip("/") or f"{did}/{file_name}"
            file_path = (dataset_root / relative_path).resolve()
        else:
            file_path = (dataset_dir / file_name).resolve()
        if not str(file_path).startswith(str(dataset_root)):
            continue
        text = _read_dataset_file_text(file_path)
        if text is None:
            continue
        files.append(
            {
                "file_name": file_name,
                "file_type": file_type,
                "file_size": int(file_size or 0),
                "path": str(file_path),
                "text": text,
            }
        )
    return dataset, files
