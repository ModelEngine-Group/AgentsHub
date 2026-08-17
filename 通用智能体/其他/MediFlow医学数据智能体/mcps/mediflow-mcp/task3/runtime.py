"""任务三共享分析服务的装配入口。

网页、MCP 和其他运行入口只负责协议适配，查询计划、只读执行、证据绑定、
图表和导出数据统一由 :class:`MedicalAnalysisService` 完成。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.llm_client import LLMClient

from .service import MedicalAnalysisService
from .source_scope import build_analysis_scope


def build_analysis_service(
    analytics_db_path: str | Path,
    *,
    kg_db_path: str | Path | None = None,
    llm: LLMClient | None = None,
    scope_provider: Callable[[], dict[str, Any]] | None = None,
) -> MedicalAnalysisService:
    """按统一配置装配任务三分析服务。

    ``kg_db_path`` 只用于生成来源范围说明，不参与统计查询；这样可以让
    Nexent 与可视化平台使用相同的分析库和查询安全边界，同时保留来源追溯。
    调用方如有特殊来源策略，可直接传入 ``scope_provider`` 覆盖默认实现。
    """

    if scope_provider is None and kg_db_path is not None:
        scope_path = Path(kg_db_path)
        scope_provider = lambda: build_analysis_scope(scope_path)
    return MedicalAnalysisService(
        analytics_db_path,
        llm=llm,
        scope_provider=scope_provider,
    )


__all__ = ["build_analysis_service"]
