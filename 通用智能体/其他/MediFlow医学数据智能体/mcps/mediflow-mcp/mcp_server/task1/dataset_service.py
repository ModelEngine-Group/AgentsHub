"""
任务一 DataMate 数据集服务模块。
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp_server.config import DATAMATE_BASE, DATASET_VOLUME
from mcp_server.datamate.client import _sudo_command
from mcp_server.datamate.resolver import _task2_resolve_datamate_dataset
from mcp_server.task1.inspection import build_preview_samples, recommend_chain, summarize_file_types
from mcp_server.task1.pdf_support import inspect_pdf_capability


Completed = subprocess.CompletedProcess[str]
Resolver = Callable[[str], tuple[dict[str, Any], list[dict[str, Any]]]]
Runner = Callable[[list[str]], Completed]
SqlRunner = Callable[[str], Completed]


def _psql_query(sql: str) -> Completed:
    return subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            "datamate-database",
            "psql",
            "-U",
            "postgres",
            "-d",
            "datamate",
            "-t",
            "-A",
            "-F",
            "\t",
        ],
        input=sql,
        capture_output=True,
        text=True,
    )


def _psql_exec(sql: str) -> Completed:
    return subprocess.run(
        ["docker", "exec", "-i", "datamate-database", "psql", "-U", "postgres", "-d", "datamate"],
        input=sql,
        capture_output=True,
        text=True,
    )


def _sql(value: object) -> str:
    return "'" + str(value or "").replace("'", "''") + "'"


def inspect_datamate_dataset(
    dataset_id: str,
    *,
    dataset_volume: str = DATASET_VOLUME,
    resolver: Resolver = _task2_resolve_datamate_dataset,
    sudo_runner: Runner = _sudo_command,
    psql_runner: SqlRunner = _psql_query,
) -> dict[str, Any]:
    """探查 DataMate 数据集并推荐任务一清洗链。"""
    did = (dataset_id or "").strip()
    if not did:
        return {"status": "error", "error": "dataset_id 不能为空"}
    try:
        resolved_dataset, resolution_candidates = resolver(did)
        requested_identifier = did
        did = resolved_dataset["id"]
    except Exception as exc:
        return {"status": "error", "error": str(exc), "requested_identifier": did}

    query = (
        "SELECT file_name, file_path, file_type, file_size FROM t_dm_dataset_files "
        f"WHERE dataset_id='{did}' AND status IN ('ACTIVE','COMPLETED');"
    )
    result = psql_runner(query)
    rows = [line.split("\t") for line in (result.stdout or "").strip().splitlines() if "\t" in line]
    if not rows:
        return {"status": "error", "error": f"数据集 {did} 无文件或不存在"}

    type_dist = summarize_file_types(rows)

    def read_file(host_path: str) -> str:
        return sudo_runner(["cat", host_path]).stdout or ""

    samples, multi_hint = build_preview_samples(rows, dataset_volume, did, read_file)
    recommendation, advice = recommend_chain(set(type_dist.keys()), multi_hint)

    payload = {
        "status": "ok",
        "dataset_id": did,
        "dataset_name": resolved_dataset["name"],
        "requested_identifier": requested_identifier,
        "resolved_by": resolved_dataset.get("resolved_by", ""),
        "resolution_candidates": resolution_candidates[:5],
        "file_count": len(rows),
        "type_distribution": dict(type_dist),
        "samples": samples,
        "multi_record_hint": multi_hint,
        "recommendation": recommendation,
        "advice": advice,
    }
    if "pdf" in {value.lower() for value in type_dist}:
        payload["pdf_capability"] = inspect_pdf_capability(DATAMATE_BASE)
    return payload


def upload_text_dataset(
    text: str,
    name: str = "user_input",
    *,
    dataset_volume: str = DATASET_VOLUME,
    sudo_runner: Runner = _sudo_command,
    psql_runner: SqlRunner = _psql_exec,
    uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4,
) -> dict[str, Any]:
    """将用户文本注册为真实的 DataMate TEXT 数据集。"""
    if not (text or "").strip():
        return {"status": "error", "error": "text cannot be empty"}

    dataset_id = str(uuid_factory())
    safe_name = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", (name or "用户上传医疗文本"), flags=re.UNICODE).strip("_")
    safe_name = (safe_name or "用户上传医疗文本")[:80]
    file_name = f"{safe_name}.txt"
    file_size = len(text.encode("utf-8"))
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()

    try:
        dataset_dir = f"{dataset_volume}/{dataset_id}"
        sudo_runner(["mkdir", "-p", dataset_dir])
        tmp_path = str(Path(tempfile.gettempdir()) / f"dm_upload_{dataset_id}.txt")
        Path(tmp_path).write_text(text, encoding="utf-8")
        try:
            sudo_runner(["cp", tmp_path, f"{dataset_dir}/{file_name}"])
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass

        sql = (
            f"INSERT INTO t_dm_datasets (id, name, description, dataset_type, path, format, "
            f"size_bytes, file_count, status, is_public, version) VALUES ("
            f"{_sql(dataset_id)}, {_sql(safe_name)}, {_sql('由 Nexent Agent 上传')}, 'TEXT', "
            f"{_sql('/dataset/' + dataset_id)}, 'txt', {file_size}, 1, 'ACTIVE', false, 0);"
            f"INSERT INTO t_dm_dataset_files (id, dataset_id, file_name, file_path, "
            f"file_type, file_size, check_sum, status) VALUES ("
            f"{_sql(str(uuid_factory()))}, {_sql(dataset_id)}, {_sql(file_name)}, "
            f"{_sql('/dataset/' + dataset_id + '/' + file_name)}, 'txt', {file_size}, {_sql(checksum)}, 'ACTIVE');"
        )
        result = psql_runner(sql)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip()[:500] or result.stdout.strip()[:500])

        return {
            "status": "saved",
            "dataset_id": dataset_id,
            "dataset_name": safe_name,
            "char_count": len(text),
            "file_name": file_name,
            "message": f"文本已注册为 DataMate 数据集（{len(text)}字）。请调用 inspect_dataset(dataset_id='{dataset_id}') 或 run_task1_mixed_cleaning(dataset_id='{dataset_id}') 继续处理。",
        }
    except Exception as exc:
        try:
            sudo_runner(["rm", "-rf", f"{dataset_volume}/{dataset_id}"])
        except Exception:
            pass
        return {"status": "error", "error": str(exc), "dataset_id": None, "dataset_name": safe_name}
