# -*- coding: utf-8 -*-
"""
医疗实体识别核心逻辑（LLM 驱动）。
对应 DataMate 算子 MedicalEntityExtractor。

被复用于：
  - operators/medical_entity_extractor/
  - mcp_server/ (extract_entities 工具)

实体类型严格对齐 CMeEE-V2 的 9 类标签。
"""
import json
from typing import Any, List, Optional
from .schemas import Entity
from .llm_client import LLMClient
from .medical_extraction_validation import validate_entities
from .medical_fewshot import retrieve_cmeee_examples

ENTITY_TYPES = {
    "dis": "疾病",
    "sym": "症状",
    "dru": "药物",
    "equ": "医疗设备",
    "pro": "医疗程序",
    "bod": "身体部位",
    "ite": "检验项目",
    "mic": "微生物",
    "dep": "科室",
}

_PROMPT = """你是医疗命名实体识别（NER）专家，熟悉中文电子病历和医学文献。

任务：从下面的医疗文本中识别所有医疗实体，严格使用以下 9 种类型，只输出 JSON 数组，不要任何解释。

实体类型：
- dis：疾病（如：2型糖尿病、高血压、心肌梗死、肺炎）
- sym：症状体征（如：发热、头痛、胸闷、水肿、心悸）
- dru：药物（如：二甲双胍、阿司匹林、胰岛素、氨氯地平）
- equ：医疗设备（如：呼吸机、心脏起搏器、血糖仪、CT机）
- pro：医疗操作/手术/治疗方法（如：冠脉造影、心肺复苏、放疗、血液透析）
- bod：身体部位（如：心脏、肝脏、右肺、颈动脉、股骨）
- ite：检验/检查项目（如：血糖、HbA1c、白细胞计数、肌钙蛋白、CT）
- mic：微生物（如：大肠埃希菌、金黄色葡萄球菌、流感病毒）
- dep：科室（如：内分泌科、心内科、急诊科、ICU）

CMeEE 标注约定：
- pro 不仅包含治疗和手术，也包含心电图、胸片、CT、造影、显像等检查程序。
- ite 是可测量的检验/生理指标，如血压、心率、肌张力、白细胞计数。
- bod 可与更长的疾病或症状嵌套，例如“血压升高”同时包含 ite“血压”。
- sym 应保留原文中完整的症状描述，bod/ite 可作为其中的嵌套实体。

输出格式：[{{"text": "实体原文", "type": "类型缩写", "start_idx": 0, "end_idx": 3, "confidence": 0.95}}]

强制规则：
1. start_idx/end_idx 是实体在原文中的字符下标，首字符为 0，end_idx 为闭区间。
2. 只抽取原文中逐字出现的实体，不改写、不翻译、不补充常识。
3. 同一句中重复出现的实体要分别输出；允许嵌套实体。
4. 不确定是否为医学实体时不要输出。

示例1：
文本：患者因2型糖尿病就诊内分泌科，空腹血糖11.2mmol/L，HbA1c 8.5%，给予二甲双胍0.5g tid及胰岛素治疗。
输出：[{{"text":"2型糖尿病","type":"dis"}},{{"text":"内分泌科","type":"dep"}},{{"text":"空腹血糖","type":"ite"}},{{"text":"HbA1c","type":"ite"}},{{"text":"二甲双胍","type":"dru"}},{{"text":"胰岛素","type":"dru"}}]

示例2：
文本：患者男，65岁，因反复胸闷、气短3年，加重伴双下肢水肿1周就诊心内科。心电图示左束支传导阻滞。
输出：[{{"text":"胸闷","type":"sym"}},{{"text":"气短","type":"sym"}},{{"text":"双下肢水肿","type":"sym"}},{{"text":"心内科","type":"dep"}},{{"text":"左束支传导阻滞","type":"dis"}},{{"text":"心电图","type":"ite"}}]

示例3：
文本：由于交感神经亢进，有心率增快、血压升高和瞳孔扩大。
输出：[{{"text":"交感神经亢进","type":"sym"}},{{"text":"交感神经","type":"bod"}},{{"text":"心率增快","type":"sym"}},{{"text":"心率","type":"ite"}},{{"text":"血压升高","type":"sym"}},{{"text":"血压","type":"ite"}},{{"text":"瞳孔扩大","type":"sym"}},{{"text":"瞳孔","type":"bod"}}]

示例4：
文本：房室结消融和起搏器植入可作为室性心动过速的替代疗法。
输出：[{{"text":"房室结消融","type":"pro"}},{{"text":"起搏器植入","type":"pro"}},{{"text":"室性心动过速","type":"dis"}},{{"text":"替代疗法","type":"pro"}}]

文本：
{text}

输出 JSON 数组："""


def extract_entities(
    text: str,
    llm: LLMClient,
    few_shot_examples: Optional[List[dict[str, Any]]] = None,
) -> List[Entity]:
    """从文本抽取医疗实体，类型对齐 CMeEE-V2"""
    if not text or not text.strip():
        return []

    if few_shot_examples is None:
        few_shot_examples = retrieve_cmeee_examples(text, limit=2)

    prompt = _PROMPT.format(text=text[:1200])
    if few_shot_examples:
        formatted = []
        for example in few_shot_examples[:3]:
            formatted.append(
                "参考文本："
                + str(example.get("text") or "")
                + "\n参考标准输出："
                + json.dumps(example.get("entities") or [], ensure_ascii=False)
            )
        prompt = (
            "以下是 CMeEE 训练集中的相似标注示例。"
            "学习其嵌套方式、类型和边界，不要复制未出现在待处理文本中的内容：\n"
            + "\n\n".join(formatted)
            + "\n\n"
            + prompt
        )

    data = llm.chat_json(prompt)
    if not isinstance(data, list):
        # 兼容包裹在 dict 里的情况：{"entities": [...]}
        if isinstance(data, dict):
            data = data.get("entities", data.get("result", []))
        if not isinstance(data, list):
            return []

    type_aliases = {
        "疾病": "dis", "disease": "dis",
        "症状": "sym", "symptom": "sym", "症状体征": "sym",
        "药物": "dru", "drug": "dru",
        "医疗设备": "equ", "equipment": "equ", "设备": "equ",
        "医疗程序": "pro", "procedure": "pro", "手术": "pro", "操作": "pro",
        "身体部位": "bod", "body": "bod", "部位": "bod",
        "检验项目": "ite", "test": "ite", "检查": "ite", "检验": "ite",
        "微生物": "mic", "microorganism": "mic",
        "科室": "dep", "department": "dep",
    }

    normalized = []
    for item in data:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        raw_type = str(copied.get("type", "")).strip()
        copied["type"] = type_aliases.get(raw_type, raw_type)
        normalized.append(copied)
    return validate_entities(text, normalized)
