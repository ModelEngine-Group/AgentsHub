# -*- coding: utf-8 -*-
"""
医疗关系抽取核心逻辑（LLM 驱动）。
对应 DataMate 算子 MedicalRelationExtractor。

被复用于：
  - operators/medical_relation_extractor/
  - mcp_server/ (extract_relations 工具)

关系类型对齐 CMeIE 数据集的核心谓词。
"""
import json
from typing import Any, List, Optional
from .schemas import Relation, Entity
from .llm_client import LLMClient
from .medical_extraction_validation import validate_relations
from .medical_fewshot import retrieve_cmeie_examples

# CMeIE 数据集全部 44 种关系类型（严格对齐，用于 F1 评测）
RELATION_TYPES = [
    "临床表现", "传播途径", "侵及周围组织转移的症状",
    "内窥镜检查", "化疗", "发病年龄", "发病性别倾向",
    "发病机制", "发病率", "发病部位", "同义词", "外侵部位",
    "多发地区", "多发季节", "多发群体", "实验室检查",
    "就诊科室", "并发症", "影像学检查", "手术治疗",
    "放射治疗", "死亡率", "治疗后症状", "病史", "病因",
    "病理分型", "病理生理", "相关（导致）", "相关（症状）",
    "相关（转化）", "筛查", "组织学检查", "药物治疗",
    "转移部位", "辅助检查", "辅助治疗", "遗传因素",
    "鉴别诊断", "阶段", "预后状况", "预后生存率",
    "预防", "风险评估因素", "高危因素",
]

_PROMPT = """你是医疗知识图谱专家，熟悉中文医学实体关系抽取。

任务：从下面的医疗文本中抽取实体之间的关系三元组，只输出 JSON，不要任何解释。

支持的关系类型（优先使用）：
{rel_types}

输出格式：{{"relations": [{{"subject": "主体实体", "subject_type": "实体类型", "predicate": "关系类型", "object": "客体实体", "object_type": "实体类型", "confidence": 0.9}}]}}

规则：
1. subject 和 object 必须是文本中出现的实体原文
2. 若文本中无明确关系，输出 {{"relations": []}}
3. predicate 必须逐字使用支持列表中的一个关系类型，不允许自造关系
4. 只输出原文直接表达的关系，不使用医学常识补全
5. 同一 subject-predicate-object 只输出一次

示例：
文本：高血压是心肌梗死的高危因素，患者需长期服用氨氯地平控制血压，定期监测肌钙蛋白。
输出：{{"relations":[{{"subject":"高血压","subject_type":"dis","predicate":"高危因素","object":"心肌梗死","object_type":"dis"}},{{"subject":"心肌梗死","subject_type":"dis","predicate":"药物治疗","object":"氨氯地平","object_type":"dru"}},{{"subject":"心肌梗死","subject_type":"dis","predicate":"实验室检查","object":"肌钙蛋白","object_type":"ite"}}]}}

文本：
{text}

输出 JSON："""


def extract_relations(
    text: str,
    llm: LLMClient,
    entities: Optional[List[Entity]] = None,
    few_shot_examples: Optional[List[dict[str, Any]]] = None,
) -> List[Relation]:
    """从文本抽取医疗关系。entities 可选，传入时可在 prompt 中提示已知实体。"""
    if not text or not text.strip():
        return []

    if few_shot_examples is None:
        few_shot_examples = retrieve_cmeie_examples(text, limit=2)

    entity_hint = ""
    if entities:
        ents_str = "、".join(f"{e.text}({e.type})" for e in entities[:20])
        entity_hint = f"\n\n文本中已识别的实体（供参考）：{ents_str}"

    few_shot = ""
    if few_shot_examples:
        formatted = []
        for example in few_shot_examples[:3]:
            formatted.append(
                "参考文本："
                + str(example.get("text") or "")
                + "\n参考标准输出："
                + json.dumps(
                    {"relations": example.get("relations") or []},
                    ensure_ascii=False,
                )
            )
        few_shot = (
            "\n\n以下是 CMeIE 训练集中的相似标注示例。"
            "学习其实体边界和关系取舍，不要复制未出现在待处理文本中的内容：\n"
            + "\n\n".join(formatted)
        )

    prompt = _PROMPT.format(
        rel_types="、".join(RELATION_TYPES),
        text=text[:1000] + entity_hint,
    )
    if few_shot:
        prompt = few_shot + "\n\n" + prompt
    data = llm.chat_json(prompt)

    if isinstance(data, dict):
        data = data.get("relations", [])
    if not isinstance(data, list):
        return []

    return validate_relations(text, data, entities=entities)
