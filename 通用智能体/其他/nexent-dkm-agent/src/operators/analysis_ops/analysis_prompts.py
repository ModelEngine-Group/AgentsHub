"""Canonical prompts for the task-3 analysis local model.

Single source of truth shared by the training-data generator, the QLoRA
fine-tuning script, and the local-model inference paths. Keeping the system
prompt and instruction here guarantees that what a fine-tuned adapter is
trained on matches exactly what it is prompted with at inference time -- a
mismatch silently degrades a fine-tuned model back to base behaviour.

The training user message is built as ``f"{INSTRUCTION}\\n\\n{content}"`` and
inference must reproduce it byte-for-byte, so both sides call
``build_planning_user`` / ``build_nl2sql_user`` instead of formatting locally.
"""

from __future__ import annotations

PLANNING_SYSTEM = "你是图谱数据分析规划助手。"
PLANNING_INSTRUCTION = (
    "你是图谱数据分析规划助手。根据分析需求，从可用算子中选择要执行的算子并按顺序输出。"
    "只输出JSON，包含字段：task_type, operators, intent_keywords, confidence。"
)

NL2SQL_SYSTEM = "你是SQL专家，只输出SQL查询。"
NL2SQL_INSTRUCTION = (
    "你是SQL专家。把下面的图谱分析问题转换为一条只读SELECT语句，"
    "schema为 nodes(id,name,type,mention_count) 与 edges(source,target,predicate,confidence)。只输出SQL。"
)


def build_planning_user(request: str) -> str:
    """Build the planning user message exactly as used during training."""
    return f"{PLANNING_INSTRUCTION}\n\n{request}"


def build_nl2sql_user(question: str) -> str:
    """Build the NL2SQL user message exactly as used during training."""
    return f"{NL2SQL_INSTRUCTION}\n\n{question}"
