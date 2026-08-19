"""Canonical prompts for the task-2 KG extraction local model.

Single source of truth shared by the training-data generator, the QLoRA
fine-tuning script, and the local-model NER inference path. Keeping the system
prompt and instruction here guarantees that what a fine-tuned adapter is
trained on matches exactly what it is prompted with at inference time -- a
mismatch silently degrades a fine-tuned model back to base behaviour.

The training user message is built as ``f"{KG_INSTRUCTION}\\n\\n{text}"`` and
inference must reproduce it, so both sides call ``build_kg_user`` instead of
formatting locally.
"""

from __future__ import annotations

KG_SYSTEM = "你是医疗信息抽取专家。"
KG_INSTRUCTION = "请从以下医疗文本中提取实体和关系。输出JSON格式，包含entities和relations两个字段。"


def build_kg_user(text: str) -> str:
    """Build the KG extraction user message exactly as used during training."""
    return f"{KG_INSTRUCTION}\n\n{text}"
