"""
core —— 核心逻辑层（平台无关）

本层不依赖 DataMate / Nexent 任何平台代码，是纯 Python 算法实现。
被两个适配层复用：
  - operators/   DataMate 算子（在管道内运行）
  - mcp_server/  MCP 工具（被 Nexent Agent 调用）

逻辑写一次，两处复用，体现算子可复用、解耦清晰的架构设计。
"""

from .schemas import Entity, Relation, Triple, QualityResult
from .llm_client import LLMClient
from . import text_preprocessor
from . import nl2sql
from . import medical_fewshot

__all__ = ["Entity", "Relation", "Triple", "QualityResult", "LLMClient",
           "text_preprocessor", "nl2sql", "medical_fewshot"]
