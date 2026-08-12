# -*- coding: utf-8 -*-
"""
任务二知识图谱构建 MCP 工具。
"""

from __future__ import annotations

from mcp_server.task2.pipeline_service import run_kg_pipeline_service
from mcp_server.tools import mcp


@mcp.tool
def run_task2_kg_pipeline(
    dataset_id: str = "",
    task_name: str = "",
    max_records: int = 0,
    dry_run: bool = False,
    persist: bool = True,
    refresh_analytics: bool = False,
    backend: str = "offline",
) -> dict:
    """基于 DataMate 数据集执行任务二知识图谱构建。

    默认使用离线抽取并写入知识图谱；仅在用户明确要求时启用混合增强或
    刷新任务三分析库，避免交互请求被外部模型和全库重建长期阻塞。混合增强本身也有
    缺口句段和候选复核上限，超限时保留离线结果并在指标中报告降级。
    """
    return run_kg_pipeline_service(
        dataset_id=dataset_id,
        task_name=task_name,
        max_records=max_records,
        dry_run=dry_run,
        persist=persist,
        refresh_analytics=refresh_analytics,
        backend=backend,
    )
