"""dki.kg —— 医疗知识图谱：本体 Schema、三元组存储与查询。"""

from dki.kg.schema import (
    ENTITY_TYPES,
    RELATION_TYPES,
    entity_zh,
    relation_zh,
    schema_description,
    validate_triple,
)
from dki.kg.store import KnowledgeGraph

__all__ = [
    "ENTITY_TYPES",
    "RELATION_TYPES",
    "KnowledgeGraph",
    "entity_zh",
    "relation_zh",
    "schema_description",
    "validate_triple",
]
