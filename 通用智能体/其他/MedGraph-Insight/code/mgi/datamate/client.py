"""DataMate 集成客户端。

对齐 DataMate「数据清洗服务」REST API（cleaning tasks）：
  POST /api/cleaning/tasks                     创建清洗任务
  POST /api/cleaning/tasks/{taskId}/execute    执行任务
  GET  /api/cleaning/tasks/{taskId}/log/stream  日志流
  GET  /api/cleaning/tasks/{taskId}/result     结果

设计原则：
- 当配置了可用的 DataMate 服务（base_url + token）时，走真实 REST 调用；
- 未配置时自动回退到「本地内置算子」执行同样的算子流水线，
  保证在无 DataMate 部署的评测/CI 环境下链路依然可跑通、结果可复现。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from .operators import build_text_pipeline


@dataclass
class OperatorInstance:
    """一个算子在流水线中的实例：算子名 + 参数。"""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


class DataMateClient:
    """DataMate 数据清洗能力的统一入口。"""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("DATAMATE_BASE_URL", "")).rstrip("/")
        self.token = token or os.getenv("DATAMATE_TOKEN", "")
        self.timeout = timeout
        self._local_ops = build_text_pipeline_registry()

    @property
    def online(self) -> bool:
        return bool(self.base_url)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def run_cleaning(
        self,
        records: list[str],
        operators: list[OperatorInstance],
        src_dataset_id: str | None = None,
        dest_dataset_id: str | None = None,
    ) -> dict[str, Any]:
        """执行一条清洗流水线，返回 {"output": [...], "trace": [...], "backend": str}。"""
        if self.online:
            return self._run_remote(records, operators, src_dataset_id, dest_dataset_id)
        return self._run_local(records, operators)

    # ---- 远端（真实 DataMate） -----------------------------------------

    def _run_remote(
        self,
        records: list[str],
        operators: list[OperatorInstance],
        src_dataset_id: str | None,
        dest_dataset_id: str | None,
    ) -> dict[str, Any]:
        payload = {
            "src_dataset_id": src_dataset_id,
            "dest_dataset_id": dest_dataset_id,
            "operators": [{"name": o.name, "params": o.params} for o in operators],
        }
        r = requests.post(f"{self.base_url}/api/cleaning/tasks",
                          json=payload, headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        task_id = r.json()["taskId"]
        ex = requests.post(f"{self.base_url}/api/cleaning/tasks/{task_id}/execute",
                           headers=self._headers(), timeout=self.timeout)
        ex.raise_for_status()
        res = requests.get(f"{self.base_url}/api/cleaning/tasks/{task_id}/result",
                           headers=self._headers(), timeout=self.timeout)
        res.raise_for_status()
        body = res.json()
        return {"output": body.get("records", []), "trace": body.get("trace", []),
                "backend": "datamate-remote", "task_id": task_id}

    # ---- 本地回退 -------------------------------------------------------

    def _run_local(self, records: list[str], operators: list[OperatorInstance]) -> dict[str, Any]:
        data = list(records)
        trace: list[dict[str, Any]] = []
        for inst in operators:
            fn = self._local_ops.get(inst.name)
            if fn is None:
                trace.append({"operator": inst.name, "status": "skipped", "reason": "未知算子"})
                continue
            before = len(data)
            data = fn(data, **inst.params)
            trace.append({"operator": inst.name, "in": before, "out": len(data), "status": "ok"})
        return {"output": data, "trace": trace, "backend": "local-fallback"}

    def list_operators(self) -> list[str]:
        return list(self._local_ops)


def build_text_pipeline_registry() -> dict[str, Callable[..., list[str]]]:
    return build_text_pipeline()
