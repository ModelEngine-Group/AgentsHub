"""医疗知识图谱本体定义（Schema）。

实体与关系类型经过裁剪，覆盖赛题要求的「疾病/症状/药物」实体与
「诊断/治疗/并发」关系，同时保持可扩展（新增类型只需登记到下方表中）。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- 实体类型
ENTITY_TYPES: Dict[str, str] = {
    "Disease": "疾病",
    "Symptom": "症状",
    "Drug": "药物",
    "Exam": "检查",
    "Treatment": "治疗方式",
    "Department": "科室",
}


@dataclass(frozen=True)
class RelationType:
    name: str            # 英文标识（图谱内部使用）
    zh: str              # 中文标签（展示使用）
    head_type: str       # 定义域
    tail_type: str       # 值域
    description: str = ""


# ---------------------------------------------------------------- 关系类型
RELATION_TYPES: Dict[str, RelationType] = {
    r.name: r
    for r in [
        RelationType("HAS_SYMPTOM", "临床表现", "Disease", "Symptom",
                     "疾病的典型症状/体征"),
        RelationType("TREATED_BY_DRUG", "治疗药物", "Disease", "Drug",
                     "疾病的常用治疗药物"),
        RelationType("TREATED_BY", "治疗方式", "Disease", "Treatment",
                     "疾病的非药物治疗手段（手术/疗法等）"),
        RelationType("DIAGNOSED_BY", "诊断检查", "Disease", "Exam",
                     "确诊/评估该疾病所依赖的检查项目"),
        RelationType("HAS_COMPLICATION", "并发症", "Disease", "Disease",
                     "疾病可能引起的并发疾病"),
        RelationType("CONTRAINDICATED_FOR", "用药禁忌", "Drug", "Disease",
                     "药物对某类疾病患者禁用/慎用"),
        RelationType("BELONGS_TO_DEPT", "所属科室", "Disease", "Department",
                     "疾病的主要就诊科室"),
    ]
}


def validate_triple(
    head: str, head_type: str, relation: str, tail: str, tail_type: str
) -> Tuple[bool, str]:
    """三元组 Schema 校验：类型合法性、定义域/值域约束、自环检查。"""
    if not head or not tail:
        return False, "头/尾实体为空"
    if head_type not in ENTITY_TYPES:
        return False, f"未知头实体类型: {head_type}"
    if tail_type not in ENTITY_TYPES:
        return False, f"未知尾实体类型: {tail_type}"
    rel = RELATION_TYPES.get(relation)
    if rel is None:
        return False, f"未知关系类型: {relation}"
    if rel.head_type != head_type:
        return False, f"关系 {relation} 要求头实体为 {rel.head_type}，实际 {head_type}"
    if rel.tail_type != tail_type:
        return False, f"关系 {relation} 要求尾实体为 {rel.tail_type}，实际 {tail_type}"
    if head == tail:
        return False, "头尾实体相同（自环）"
    return True, "ok"


def relation_zh(relation: str) -> str:
    r = RELATION_TYPES.get(relation)
    return r.zh if r else relation


def entity_zh(entity_type: str) -> str:
    return ENTITY_TYPES.get(entity_type, entity_type)


def schema_description(lang: str = "zh") -> str:
    """输出给智能体/LLM 的图谱 Schema 描述。"""
    lines = ["实体类型："]
    for et, zh in ENTITY_TYPES.items():
        lines.append(f"  - {et}（{zh}）")
    lines.append("关系类型（头实体 -[关系]-> 尾实体）：")
    for r in RELATION_TYPES.values():
        lines.append(
            f"  - {r.head_type} -[{r.name}/{r.zh}]-> {r.tail_type}：{r.description}"
        )
    return "\n".join(lines)
