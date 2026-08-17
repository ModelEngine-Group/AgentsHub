# -*- coding: utf-8 -*-
"""
三元组生成与校验核心逻辑。
对应 DataMate 算子 MedicalTripleGenerator。

被复用于：
  - operators/medical_triple_generator/
  - mcp_server/ (generate_triples 工具)

三元组 = 关系 + 置信度，是入知识图谱的最终形态。

三元组不再独立请求 LLM。它由经过文本证据、关系白名单和实体约束校验的
关系结果生成，避免关系算子与三元组算子产生互相冲突的事实。
"""
from typing import List
from .schemas import Triple
from .llm_client import LLMClient
from .medical_ner import extract_entities
from .medical_re import extract_relations
from .medical_extraction_validation import relations_to_triples


def generate_triples(text: str, llm: LLMClient,
                     min_confidence: float = 0.7) -> List[Triple]:
    """从文本生成知识三元组，过滤低置信度"""
    if not text or not text.strip():
        return []

    entities = extract_entities(text, llm)
    relations = extract_relations(text, llm, entities=entities)
    return relations_to_triples(relations, min_confidence=min_confidence)
