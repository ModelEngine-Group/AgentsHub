"""任务一治理元数据登记辅助模块。

当前在线流程会在任务一报告中写入血缘、标签和质量统计。不同 DataMate
版本的治理表结构存在差异，因此本模块采用保守策略：读取清洗报告中的
可观测信息，返回结构化摘要；如果目标环境提供治理表，可在这里扩展写库。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lineage_store import persist_task1_lineage


def _dataset_id(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict) and value.get("id"):
            return str(value["id"])
    return ""


def register_governance(report_path: Path) -> dict[str, Any]:
    """读取任务一报告并返回治理元数据摘要。

    参数:
        report_path: `mixed_cleaning_service.py` 生成的 JSON 报告路径。

    返回:
        可写入最终报告的治理摘要。函数不虚构数据库写入结果；若报告不存在
        或格式异常，会返回可观察的错误状态。
    """
    path = Path(report_path)
    if not path.exists():
        return {"status": "skipped", "reason": "report_not_found", "report_path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "skipped", "reason": f"invalid_report: {exc}", "report_path": str(path)}

    source_dataset_id = _dataset_id(payload, "source_mixed_dataset", "source_dataset")
    final_dataset_id = _dataset_id(
        payload,
        "delivery_dataset",
        "final_delivery_dataset",
        "final_dataset",
    )
    if not source_dataset_id or not final_dataset_id:
        return {
            "status": "skipped",
            "reason": "dataset_ids_missing",
            "source_dataset_id": source_dataset_id,
            "final_dataset_id": final_dataset_id,
            "report_path": str(path),
        }

    lineage = persist_task1_lineage(source_dataset_id, final_dataset_id, payload)
    return {
        "status": "persisted",
        "source_dataset_id": source_dataset_id,
        "final_dataset_id": final_dataset_id,
        "lineage": lineage,
        "quality_report": payload.get("quality_report") or payload.get("reports"),
        "report_path": str(path),
    }
