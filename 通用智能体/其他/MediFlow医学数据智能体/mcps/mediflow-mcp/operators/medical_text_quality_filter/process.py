# -*- coding: utf-8 -*-
"""
医疗文本质量过滤算子
对输入文本综合评分（0~1），低于阈值的文档被过滤掉。

评分维度（总分1.0）：
  1. 医疗术语密度（0.35）——含医疗专业词汇越多，质量越高
  2. 信息完整度（0.30）  ——含诊断/症状/治疗等关键要素越全面，质量越高
  3. 文本长度适宜度（0.20）——过短（<50字）或过长（>50000字）降分
  4. 噪声比率（0.15）     ——HTML标签、乱码、特殊字符等噪声占比，越少越好
"""
import re
import time
from typing import Dict, Any

from loguru import logger
from datamate.core.base_op import Filter


# 医疗专业词汇表（覆盖常见疾病、症状、药物、检查、操作类别）
MEDICAL_TERMS = {
    # 诊断/疾病
    "疾病", "诊断", "综合征", "炎症", "感染", "肿瘤", "癌", "高血压", "糖尿病",
    "肺炎", "心肌梗死", "脑卒中", "骨折", "贫血", "哮喘", "肝炎", "肾炎",
    "ARDS", "DIC", "CAP", "T2DM", "HTN", "CHD", "CKD", "AMI", "PAP",
    # 典型躯体症状
    "症状", "发热", "咳嗽", "呼吸困难", "头痛", "头晕", "胸痛", "腹痛",
    "恶心", "呕吐", "乏力", "水肿", "紫绀", "出血", "抽搐", "昏迷",
    "SpO2", "PaO2", "血压", "心率", "体温",
    # 常见主诉/非结构化描述（情绪/睡眠/疼痛/一般不适）
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

# 关键医疗要素（有越多分越高）
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

# 噪声模式（只用明确的 hex escape，避免嵌入特殊字符）
NOISE_PATTERNS = [
    r"<[^>]{1,200}>",                       # HTML 标签
    r"&[a-zA-Z]+;",                         # HTML 实体
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",  # 控制字符
    r"(.)\1{5,}",                           # 连续重复字符（5次以上）
    r"https?://\S+",                        # URL 链接
]


class MedicalTextQualityFilter(Filter):
    """
    医疗文本质量过滤器
    对医疗文本综合评分，过滤低质量文档（乱码、无关内容、过短文本等）
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = float(kwargs.get("qualityThreshold", 0.3))
        self.add_score_field = str(kwargs.get("addScoreField", "true")).lower() == "true"

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        self.read_file_first(sample)
        text = sample.get(self.text_key, "")

        if not text.strip():
            sample[self.text_key] = ""
            return sample

        score, detail = self._compute_quality_score(text)

        if self.add_score_field:
            sample["quality_score"] = round(score, 4)
            sample["quality_detail"] = detail

        if score < self.threshold:
            logger.info(
                f"fileName: {sample.get(self.filename_key, '?')}, "
                f"quality_score={score:.3f} < threshold={self.threshold}, filtered. "
                f"detail={detail}"
            )
            sample[self.text_key] = ""
        else:
            logger.info(
                f"fileName: {sample.get(self.filename_key, '?')}, "
                f"quality_score={score:.3f}, passed. "
                f"costs {time.time() - start:.3f}s"
            )

        return sample

    def _compute_quality_score(self, text: str):
        scores = {}
        scores["term_density"] = self._score_term_density(text) * 0.35
        scores["completeness"] = self._score_completeness(text) * 0.30
        scores["length"] = self._score_length(text) * 0.20
        scores["cleanliness"] = self._score_cleanliness(text) * 0.15
        total = sum(scores.values())
        detail = {k: round(v, 3) for k, v in scores.items()}
        detail["total"] = round(total, 3)
        return total, detail

    def _score_term_density(self, text: str) -> float:
        clean = re.sub(r"<[^>]{1,200}>", "", text)
        if not clean:
            return 0.0
        hit = sum(1 for term in MEDICAL_TERMS if term.lower() in clean.lower())
        return min(hit / 15.0, 1.0)

    def _score_completeness(self, text: str) -> float:
        hit = sum(1 for pat in KEY_ELEMENTS if re.search(pat, text, re.IGNORECASE))
        return min(hit / len(KEY_ELEMENTS), 1.0)

    def _score_length(self, text: str) -> float:
        length = len(text)
        if length < 50:
            return 0.0
        elif length < 200:
            return (length - 50) / 150.0 * 0.5
        elif length <= 10000:
            return 1.0
        elif length <= 50000:
            return 1.0 - (length - 10000) / 80000.0
        else:
            return 0.3

    def _score_cleanliness(self, text: str) -> float:
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        noise_chars = sum(len(m.group()) for pat in NOISE_PATTERNS for m in re.finditer(pat, text))
        noise_ratio = min(noise_chars / total_chars, 1.0)
        if noise_ratio < 0.05:
            return 1.0
        elif noise_ratio > 0.50:
            return 0.0
        else:
            return 1.0 - (noise_ratio - 0.05) / 0.45
