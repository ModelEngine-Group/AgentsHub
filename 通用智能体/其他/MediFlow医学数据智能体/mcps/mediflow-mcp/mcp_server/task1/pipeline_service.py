"""
任务一 DataMate 清洗流水线服务。
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

import requests

from mcp_server.config import DATAMATE_BASE
from mcp_server.datamate.resolver import _task2_resolve_datamate_dataset
from mcp_server.task1.chains import DEFAULT_OPERATORS, operator_config


def _operator_id(operator: dict | str) -> str:
    return operator.get("id", "") if isinstance(operator, dict) else str(operator)


def _build_operator_instance(
    operators: Iterable[str] | None,
    with_llm_filter: bool,
    output_jsonl: bool,
) -> list[dict]:
    if operators:
        instance = [operator_config(str(op_id), str(op_id)) for op_id in operators]
    else:
        instance = list(DEFAULT_OPERATORS)

    op_ids = [_operator_id(item) for item in instance]
    if with_llm_filter and "LLMNoiseFilter" not in op_ids:
        instance.append(operator_config("LLMNoiseFilter", "LLM语义噪声过滤"))
        op_ids.append("LLMNoiseFilter")

    if output_jsonl and not any(op_id in {"UnifiedJsonlExporter", "MedicalRecordSplitter"} for op_id in op_ids):
        instance.append(operator_config("UnifiedJsonlExporter", "JSONL统一输出"))

    return instance


def run_datamate_cleaning_pipeline(
    dataset_id: str = "",
    task_name: str = "",
    operators: list | None = None,
    with_llm_filter: bool = False,
    output_jsonl: bool = False,
    *,
    datamate_base: str = DATAMATE_BASE,
    resolver: Callable[[str], tuple[dict, list]] = _task2_resolve_datamate_dataset,
    postprocess: Callable[[str, str], dict] | None = None,
    request_post: Callable = requests.post,
    request_get: Callable = requests.get,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
) -> dict:
    """创建 DataMate 清洗任务并等待完成。"""

    requested_dataset = (dataset_id or "").strip()
    if requested_dataset:
        try:
            resolved_dataset, resolution_candidates = resolver(requested_dataset)
            src_id = resolved_dataset["id"]
            src_ds_name = resolved_dataset["name"]
        except Exception as exc:
            return {"status": "error", "error": str(exc), "requested_identifier": requested_dataset}
    else:
        return {
            "status": "error",
            "error": "dataset_id is required; demo fallback is disabled to avoid false success",
            "hint": "先调用 upload_text_to_datamate 注册用户文本，或显式传入已有 DataMate dataset_id/数据集名称。",
        }

    t0 = now()
    name = (task_name or "").strip() or f"Agent任务-{int(t0)}"

    try:
        listed = request_post(f"{datamate_base}/api/dm/datasets/list", json={"page": 0, "size": 200}, timeout=5)
        if listed.ok:
            items = listed.json().get("data", {}).get("content", [])
            src_ds_name = next((item["name"] for item in items if item.get("id") == src_id), src_ds_name)
    except Exception:
        pass

    instance = _build_operator_instance(operators, with_llm_filter, output_jsonl)

    try:
        created = request_post(
            f"{datamate_base}/api/cleaning/tasks",
            json={
                "name": name,
                "description": "由 Nexent Agent 动态创建的医疗数据处理任务",
                "srcDatasetId": src_id,
                "srcDatasetName": src_ds_name,
                "destDatasetName": f"处理结果-{int(t0)}",
                "destDatasetType": "TEXT",
                "instance": instance,
            },
            timeout=15,
        )
        if not created.ok:
            return {"status": "error", "error": f"创建任务失败: {created.status_code} {created.text[:200]}"}
        task_id = created.json().get("data", {}).get("id")
        if not task_id:
            return {"status": "error", "error": f"未获取到 task_id: {created.text[:200]}"}

        ops_used = [_operator_id(op) for op in instance]
        for _poll_index in range(60):
            sleep(5)
            info = request_get(f"{datamate_base}/api/cleaning/tasks/{task_id}", timeout=10)
            if not info.ok:
                continue
            data = info.json().get("data", {})
            dm_status = data.get("status", "UNKNOWN")
            if dm_status not in {"COMPLETED", "FAILED", "STOPPED"}:
                continue

            progress = data.get("progress", {})
            dest_id = data.get("destDatasetId", "")
            cleanup = {}
            if dm_status == "COMPLETED" and postprocess:
                cleanup = postprocess(dest_id, task_id)
            real_count = cleanup.get("real_count") or progress.get("succeedFileNum", 0)
            elapsed = round(now() - t0, 1)
            throughput = round(real_count / elapsed, 4) if elapsed > 0 else float(real_count)
            return {
                "status": "success" if dm_status == "COMPLETED" else "failed",
                "task_id": task_id,
                "dest_dataset_id": dest_id,
                "task_name": name,
                "source_dataset": {
                    "id": src_id,
                    "name": src_ds_name,
                    "requested_identifier": requested_dataset,
                    "resolved_by": resolved_dataset.get("resolved_by", ""),
                    "resolution_candidates": resolution_candidates[:5],
                },
                "datamate_status": dm_status,
                "operators_used": ops_used,
                "file_count": real_count,
                "success_count": real_count,
                "fail_count": progress.get("failedFileNum", 0),
                "success_rate": f"{progress.get('successRate', 0):.1f}%",
                "before_size_kb": round(data.get("beforeSize", 0) / 1024, 1),
                "after_size_kb": round(data.get("afterSize", 0) / 1024, 1),
                "elapsed_seconds": elapsed,
                "performance": {
                    "elapsed_seconds": elapsed,
                    "processed_files": real_count,
                    "throughput_files_per_second": throughput,
                    "metric_scope": "DataMate cleaning task files",
                },
                "output_cleanup": cleanup,
            }

        return {
            "status": "timeout",
            "task_id": task_id,
            "elapsed_seconds": round(now() - t0, 1),
            "message": "任务仍在运行，可稍后调用 get_datamate_result(task_id) 查询。",
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def get_datamate_cleaning_result(
    task_id: str,
    *,
    datamate_base: str = DATAMATE_BASE,
    request_get: Callable = requests.get,
) -> dict:
    """返回 DataMate 清洗任务结果摘要。"""

    try:
        response = request_get(f"{datamate_base}/api/cleaning/tasks/{task_id}/result", timeout=10)
        if not response.ok:
            return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}
        files = response.json().get("data", [])
        return {
            "task_id": task_id,
            "file_count": len(files),
            "completed_count": sum(1 for item in files if item.get("status") == "COMPLETED"),
            "files": [
                {
                    "name": item.get("srcName", ""),
                    "status": item.get("status", ""),
                    "src_size_kb": round(item.get("srcSize", 0) / 1024, 1),
                    "dest_size_kb": round(item.get("destSize", 0) / 1024, 1),
                }
                for item in files[:10]
            ],
            "total_src_kb": round(sum(item.get("srcSize", 0) for item in files) / 1024, 1),
            "total_dest_kb": round(sum(item.get("destSize", 0) for item in files) / 1024, 1),
        }
    except Exception as exc:
        return {"error": str(exc)}
