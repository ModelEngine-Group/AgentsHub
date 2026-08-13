"""BIO 标签空间定义（医疗 NER 微调）。

6 类实体（与 dki.kg.schema.ENTITY_TYPES / 词典类别一一对应）：
    Disease 疾病 / Symptom 症状 / Drug 药物 / Exam 检查 / Treatment 治疗方式 / Department 科室

BIO 方案：每类实体一个 B- 与 I-，加一个 O，共 6*2+1 = 13 个标签。
标签顺序**确定性**（O 固定为 0，其余按 ENTITY_ORDER 的 B-,I- 交替），
保证 id2label/label2id 在任何机器上一致——训练、导出、OM 解码三处口径统一。
"""

from typing import Dict, List

# 实体类型顺序（决定标签 id，勿随意改动，改动即需重训与重导 OM）
ENTITY_ORDER: List[str] = ["Disease", "Symptom", "Drug", "Exam", "Treatment", "Department"]

OUTSIDE = "O"


def _build_labels() -> List[str]:
    labels = [OUTSIDE]
    for etype in ENTITY_ORDER:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")
    return labels


# 13 个标签：['O', 'B-Disease', 'I-Disease', 'B-Symptom', 'I-Symptom', ...]
LABELS: List[str] = _build_labels()
NUM_LABELS: int = len(LABELS)

id2label: Dict[int, str] = {i: lab for i, lab in enumerate(LABELS)}
label2id: Dict[str, int] = {lab: i for i, lab in enumerate(LABELS)}

# 已知实体类型集合（供后端类型解析用：真实标签优先，占位映射兜底）
KNOWN_TYPES = set(ENTITY_ORDER)


def bio_label(prefix: str, etype: str) -> str:
    """构造 B-/I- 标签字符串，并校验类型合法。"""
    if prefix not in ("B", "I"):
        raise ValueError(f"prefix 必须是 B 或 I，得到 {prefix!r}")
    if etype not in KNOWN_TYPES:
        raise ValueError(f"未知实体类型 {etype!r}，合法：{sorted(KNOWN_TYPES)}")
    return f"{prefix}-{etype}"
