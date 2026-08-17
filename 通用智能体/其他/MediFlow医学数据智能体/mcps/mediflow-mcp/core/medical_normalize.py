# -*- coding: utf-8 -*-
"""
医疗术语标准化核心逻辑（LLM 驱动）。
把医疗缩写、简称替换为规范全称。如 T2DM→2型糖尿病、AMI→急性心肌梗死。

对应 DataMate 算子 MedicalTermNormalizer。

被复用于：
  - operators/medical_term_normalizer/
  - mcp_server/ (normalize_terms 工具)

采用规则词典优先 + LLM 兜底的混合策略：常见缩写用词典直接替换，罕见术语交由 LLM 处理。
"""
from .llm_client import LLMClient

# 常见缩写词典（规则优先，命中即替换，不走 LLM）
ABBR_DICT = {
    "T2DM": "2型糖尿病", "T1DM": "1型糖尿病", "AMI": "急性心肌梗死",
    "HbA1c": "糖化血红蛋白", "COPD": "慢性阻塞性肺疾病", "HTN": "高血压",
    "CHD": "冠心病", "CKD": "慢性肾脏病", "ARDS": "急性呼吸窘迫综合征",
    "CAP": "社区获得性肺炎", "DIC": "弥散性血管内凝血",
}

_PROMPT = """你是医学文本标准化专家。
请将下面医疗文本中所有医学缩写、简称替换为规范的中文全称，其余内容保持不变。
直接输出替换后的文本，不要解释，不要添加任何标记。

例如：T2DM→2型糖尿病，AMI→急性心肌梗死，HbA1c→糖化血红蛋白，COPD→慢性阻塞性肺疾病。

文本：
{text}"""


def normalize_terms(text: str, llm: LLMClient, use_llm: bool = True) -> str:
    """术语标准化。use_llm=False 时只用词典规则替换（快）"""
    if not text or not text.strip():
        return text

    # 第一步：词典规则替换（快、准、无幻觉）
    result = text
    for abbr, full in ABBR_DICT.items():
        result = result.replace(abbr, full)

    if not use_llm:
        return result

    # 第二步：LLM 处理词典覆盖不到的缩写
    try:
        out = llm.chat(_PROMPT.format(text=result))
        return out if out.strip() else result
    except Exception:
        return result  # LLM 失败时返回词典替换结果，保证不丢数据
