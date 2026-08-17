# -*- coding: utf-8 -*-
"""
文本质量过滤核心逻辑（纯规则，无需 LLM）。
评分维度：医疗术语密度(0.35) + 信息完整度(0.30) + 长度适宜度(0.20) + 洁净度(0.15)。

被复用于：
  - operators/medical_text_quality_filter/   DataMate 算子
  - mcp_server/                              质量过滤工具（可选）
"""
import re
from .schemas import QualityResult

MEDICAL_TERMS = {
    # 疾病/诊断
    "疾病", "诊断", "综合征", "炎症", "感染", "肿瘤", "癌", "高血压", "糖尿病",
    "肺炎", "心肌梗死", "脑卒中", "骨折", "贫血", "哮喘", "肝炎", "肾炎",
    "ARDS", "DIC", "CAP", "T2DM", "HTN", "CHD", "CKD", "AMI", "PAP",
    # 典型躯体症状
    "症状", "发热", "咳嗽", "呼吸困难", "头痛", "头晕", "胸痛", "腹痛",
    "恶心", "呕吐", "乏力", "水肿", "紫绀", "出血", "抽搐", "昏迷",
    "SpO2", "PaO2", "血压", "心率", "体温",
    # 常见主诉/非结构化描述（扩充，覆盖情绪/睡眠/疼痛类）
    "疼痛", "不适", "不舒服", "难受", "酸痛", "肿胀", "麻木", "瘙痒",
    "失眠", "睡眠", "入睡", "早醒", "嗜睡", "乏力", "疲劳", "疲倦",
    "情绪", "焦虑", "抑郁", "烦躁", "紧张", "恐惧", "心情", "心理",
    "食欲", "体重", "消瘦", "肥胖", "多饮", "多尿", "多食", "口渴",
    "心悸", "气短", "胸闷", "憋气", "喘息", "出汗", "盗汗", "潮热",
    "眩晕", "耳鸣", "视力", "视物", "腹泻", "便秘", "腹胀", "便血",
    "月经", "白带", "妊娠", "怀孕",
    # 检查/检验
    "CT", "MRI", "X线", "超声", "心电图", "血常规", "尿常规", "血气",
    "HbA1c", "血糖", "血脂", "肌酐", "转氨酶", "白细胞", "血小板",
    "活检", "培养", "ELISA", "PCR",
    # 药物/治疗
    "治疗", "手术", "用药", "口服", "静脉", "肌注", "雾化", "透析",
    "抗生素", "激素", "胰岛素", "化疗", "放疗", "输血", "吸氧",
    "二甲双胍", "阿托伐他汀", "阿司匹林", "华法林", "肝素",
    # 就诊场景
    "入院", "出院", "主诉", "现病史", "既往史", "查体", "医嘱",
    "门诊", "急诊", "住院", "随访", "手术记录", "病历",
    "医师", "护士", "科室", "床号", "患者", "病人",
}

KEY_ELEMENTS = [
    r"主诉|chief\s*complaint",
    r"诊断|diagnosis|impression",
    r"治疗|处理|医嘱|处方|treatment|prescription",
    r"症状|symptom|complaint",
    r"检查|examination|test|检验",
    r"病史|history|既往",
    r"用药|medication|drug|药物",
    r"随访|follow.?up|复诊",
]

NOISE_PATTERNS = [
    r"<[^>]{1,200}>",
    r"&[a-zA-Z]+;",
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
    r"(.)\1{5,}",
    r"https?://\S+",
]


def score_quality(text: str, threshold: float = 0.3) -> QualityResult:
    """对文本综合评分，返回 QualityResult"""
    if not text or not text.strip():
        return QualityResult(score=0.0, detail={"total": 0.0}, passed=False)

    scores = {
        "term_density": _score_term_density(text) * 0.35,
        "completeness": _score_completeness(text) * 0.30,
        "length": _score_length(text) * 0.20,
        "cleanliness": _score_cleanliness(text) * 0.15,
    }
    total = sum(scores.values())
    detail = {k: round(v, 3) for k, v in scores.items()}
    detail["total"] = round(total, 3)
    return QualityResult(score=total, detail=detail, passed=(total >= threshold))


def _score_term_density(text: str) -> float:
    clean = re.sub(r"<[^>]{1,200}>", "", text)
    if not clean:
        return 0.0
    hit = sum(1 for t in MEDICAL_TERMS if t.lower() in clean.lower())
    return min(hit / 15.0, 1.0)


def _score_completeness(text: str) -> float:
    hit = sum(1 for p in KEY_ELEMENTS if re.search(p, text, re.IGNORECASE))
    return min(hit / len(KEY_ELEMENTS), 1.0)


def _score_length(text: str) -> float:
    n = len(text)
    if n < 50:
        return 0.0
    if n < 200:
        return (n - 50) / 150.0 * 0.5
    if n <= 10000:
        return 1.0
    if n <= 50000:
        return 1.0 - (n - 10000) / 80000.0
    return 0.3


def _score_cleanliness(text: str) -> float:
    total = len(text)
    if total == 0:
        return 0.0
    noise = sum(len(m.group()) for p in NOISE_PATTERNS for m in re.finditer(p, text))
    ratio = min(noise / total, 1.0)
    if ratio < 0.05:
        return 1.0
    if ratio > 0.50:
        return 0.0
    return 1.0 - (ratio - 0.05) / 0.45
