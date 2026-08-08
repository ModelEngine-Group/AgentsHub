"""Operator wrapper for CM3KG-backed entity normalization."""
from __future__ import annotations

from medigraph.extraction.cascade import load_entity_linker
from medigraph.operators.base import BaseOperator, OperatorMeta


class EntityLinkerOperator(BaseOperator):
    def __init__(self):
        self.meta = OperatorMeta(
            name="entity_linker",
            version="1.0.0",
            description=(
                "把实体链接到 CM3KG/本地稳定规范 ID，输出 canonical_id、规范名、匹配方法和分数。"
                "输入 {entities}, 输出 {entities,linking_report}。"
            ),
            input_schema={
                "type": "object",
                "properties": {"entities": {"type": "array"}},
                "required": ["entities"],
            },
            output_schema={
                "type": "object",
                "properties": {"entities": {"type": "array"}, "linking_report": {"type": "object"}},
                "required": ["entities", "linking_report"],
            },
        )

    def run(self, inputs: dict, **kwargs) -> dict:
        entities = inputs.get("entities", [])
        entities = entities if isinstance(entities, list) else []
        linker = load_entity_linker()
        linked = linker.link_entities(entities) if linker is not None else entities
        exact = sum(item.get("match_method") in {"exact", "alias"} for item in linked)
        fuzzy = sum(item.get("match_method") == "fuzzy" for item in linked)
        return {
            "entities": linked,
            "linking_report": {
                "entities": len(linked),
                "exact_or_alias": exact,
                "fuzzy": fuzzy,
                "local_id": len(linked) - exact - fuzzy,
                "linked_rate": round((exact + fuzzy) / len(linked), 4) if linked else 0.0,
            },
        }
