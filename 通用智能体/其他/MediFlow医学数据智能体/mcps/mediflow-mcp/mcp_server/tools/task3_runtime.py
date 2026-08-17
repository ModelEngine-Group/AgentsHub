"""Nexent MCP 使用的任务三共享服务装配。"""

from __future__ import annotations

import os
from functools import lru_cache

from core.llm_client import LLMClient
from mcp_server.config import (
    ANALYTICS_DB,
    KG_DB,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)
from task3.runtime import build_analysis_service
from task3.service import MedicalAnalysisService


@lru_cache(maxsize=1)
def get_task3_analysis_service() -> MedicalAnalysisService:
    """返回 MCP 与可视化平台共用的分析服务实例。"""

    llm = None
    if LLM_API_KEY:
        llm = LLMClient(
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            timeout=int(os.environ.get('CCF_TASK3_LLM_TIMEOUT', '90')),
        )
    return build_analysis_service(
        ANALYTICS_DB,
        kg_db_path=KG_DB,
        llm=llm,
    )


__all__ = ["get_task3_analysis_service"]
