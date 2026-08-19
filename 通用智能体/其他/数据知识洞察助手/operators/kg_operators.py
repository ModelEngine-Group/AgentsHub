"""Medical knowledge graph extraction operators.

These rule-based operators are the MVP for task 2. They are deterministic,
easy to test, and can later be replaced by LLM/NPU-backed extractors.
"""

from __future__ import annotations

from typing import Any

from .base import BaseOperator


ENTITY_LEXICON: dict[str, list[str]] = {
    "Disease": [
        "肺炎",
        "冠心病",
        "偏头痛",
        "急性胃肠炎",
        "支气管炎",
        "类风湿性关节炎",
        "腰椎间盘突出",
        "多囊卵巢综合征",
        "2型糖尿病",
        "缺铁性贫血",
        "荨麻疹",
        "急性胆囊炎",
        "尿路感染",
        "急性心肌梗死",
        "急性扁桃体炎",
    ],
    "Symptom": [
        "发热",
        "咳嗽",
        "咳痰",
        "胸痛",
        "呼吸困难",
        "胸闷",
        "心悸",
        "气短",
        "头痛",
        "恶心",
        "呕吐",
        "腹痛",
        "腹泻",
        "关节肿胀",
        "关节疼痛",
        "腰痛",
        "麻木",
        "月经不调",
        "多毛",
        "痤疮",
        "多饮",
        "多尿",
        "多食",
        "体重下降",
        "乏力",
        "头晕",
        "面色苍白",
        "瘙痒",
        "右上腹痛",
        "黄疸",
        "尿频",
        "尿急",
        "尿痛",
        "吞咽困难",
        "咽痛",
    ],
    "Drug": [
        "阿莫西林",
        "头孢类抗生素",
        "左氧氟沙星",
        "阿司匹林",
        "氯吡格雷",
        "布洛芬",
        "曲普坦类药物",
        "诺氟沙星",
        "蒙脱石散",
        "口服补液盐",
        "头孢克洛",
        "阿奇霉素",
        "甲氨蝶呤",
        "来氟米特",
        "炔雌醇环丙孕酮",
        "二甲双胍",
        "硫酸亚铁",
        "富马酸亚铁",
        "氯雷他定",
        "西替利嗪",
        "头孢曲松",
        "哌拉西林",
        "呋喃妥因",
        "青霉素V钾",
    ],
    "Examination": [
        "胸部X线",
        "CT检查",
        "MRI检查",
        "心电图",
        "超声",
    ],
    "Treatment": [
        "呼吸机辅助通气",
        "冠脉支架植入",
        "搭桥手术",
        "物理治疗",
        "牵引",
        "腹腔镜胆囊切除术",
        "急诊PCI",
        "溶栓治疗",
        "扁桃体切除术",
    ],
}

VALID_RELATIONS = {"has_symptom", "treated_by", "requires_exam", "complication_of"}
RELATION_BY_ENTITY_TYPE = {
    "Symptom": "has_symptom",
    "Drug": "treated_by",
    "Examination": "requires_exam",
    "Treatment": "treated_by",
}


class EntityRecognitionOperator(BaseOperator):
    """Recognize medical entities from text using a small lexicon."""

    name = "EntityRecognitionOperator"
    description = "基于词典识别疾病、症状、药物、检查、治疗实体"
    input_schema = {"text": "str"}
    output_schema = {"entities": "list[dict]"}

    def run(self, text: str) -> list[dict[str, str]]:
        entities: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for entity_type, names in ENTITY_LEXICON.items():
            for name in names:
                if name in text and (name, entity_type) not in seen:
                    entities.append({"name": name, "type": entity_type})
                    seen.add((name, entity_type))
        return entities


class RelationExtractionOperator(BaseOperator):
    """Extract typed relations from a medical text block."""

    name = "RelationExtractionOperator"
    description = "基于实体类型抽取疾病到症状、药物、检查、治疗的关系"
    input_schema = {"text": "str"}
    output_schema = {"relations": "list[dict]"}

    def run(self, text: str) -> list[dict[str, str]]:
        entities = EntityRecognitionOperator().execute(text)
        diseases = [item["name"] for item in entities if item["type"] == "Disease"]
        if not diseases:
            return []

        subject = diseases[0]
        relations: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for entity in entities:
            object_type = entity["type"]
            relation = RELATION_BY_ENTITY_TYPE.get(object_type)
            obj = entity["name"]
            if not relation or obj == subject:
                continue
            key = (subject, relation, obj)
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "object_type": object_type,
                }
            )
        return relations


class TripleGenerationOperator(BaseOperator):
    """Generate triples from extracted medical relations."""

    name = "TripleGenerationOperator"
    description = "基于关系抽取结果生成医疗知识三元组"
    input_schema = {"text": "str"}
    output_schema = {"triples": "list[dict]"}

    def run(self, text: str) -> list[dict[str, str]]:
        relations = RelationExtractionOperator().execute(text)
        triples = [
            {"subject": item["subject"], "predicate": item["relation"], "object": item["object"]}
            for item in relations
        ]
        return TripleValidationOperator().execute(triples)


class TripleValidationOperator(BaseOperator):
    """Validate triple shape and relation vocabulary."""

    name = "TripleValidationOperator"
    description = "校验三元组字段完整性和关系类型合法性"
    input_schema = {"triples": "list[dict]"}
    output_schema = {"valid_triples": "list[dict]"}

    def run(self, triples: list[dict[str, Any]]) -> list[dict[str, str]]:
        valid: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for triple in triples:
            subject = str(triple.get("subject", "")).strip()
            predicate = str(triple.get("predicate", "")).strip()
            obj = str(triple.get("object", "")).strip()
            key = (subject, predicate, obj)
            if not subject or not predicate or not obj:
                continue
            if predicate not in VALID_RELATIONS:
                continue
            if key in seen:
                continue
            seen.add(key)
            valid.append({"subject": subject, "predicate": predicate, "object": obj})
        return valid


class GraphBuildOperator(BaseOperator):
    """Build a simple graph dictionary from triples and entities."""

    name = "GraphBuildOperator"
    description = "将三元组和实体列表构建为 nodes/edges 图结构"
    input_schema = {"triples": "list[dict]", "entities": "list[dict]"}
    output_schema = {"graph": "dict(nodes=list, edges=list)"}

    def run(
        self,
        triples: list[dict[str, str]],
        entities: list[dict[str, str]] | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        entity_type_by_name: dict[str, str] = {}
        for entity in entities or []:
            name = entity.get("name")
            entity_type = entity.get("type", "Unknown")
            if name:
                entity_type_by_name[name] = entity_type

        nodes: list[dict[str, str]] = []
        edges: list[dict[str, str]] = []
        seen_nodes: set[str] = set()
        seen_edges: set[tuple[str, str, str]] = set()

        def add_node(name: str) -> None:
            if name in seen_nodes:
                return
            seen_nodes.add(name)
            nodes.append({"id": name, "label": name, "type": entity_type_by_name.get(name, "Unknown")})

        for triple in TripleValidationOperator().execute(triples):
            subject = triple["subject"]
            obj = triple["object"]
            relation = triple["predicate"]
            add_node(subject)
            add_node(obj)
            edge_key = (subject, relation, obj)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append({"source": subject, "target": obj, "relation": relation})

        return {"nodes": nodes, "edges": edges}
