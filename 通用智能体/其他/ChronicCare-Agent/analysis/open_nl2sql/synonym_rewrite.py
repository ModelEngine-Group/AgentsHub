from __future__ import annotations

import re
from typing import Any, Dict

PRONOUN_DISEASE_TOKENS = ("疾病类型", "疾病分布", "患病类型")

CHINESE_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

FOLLOWUP_CHART_HINTS = ("图", "图表", "可视化", "趋势", "折线图", "饼图", "饼状图", "人数图", "人数图表")
FOLLOWUP_COUNT_HINTS = ("随访人数", "随访患者数", "随访患者数量", "随访数量", "随访总数", "需要随访的患者数", "待随访人数")
DISEASE_INVENTORY_HINTS = (
    "几种病",
    "多少种病",
    "常见病",
    "常见疾病",
    "当前常见疾病",
    "常见疾病有什么",
    "常见疾病有哪些",
    "常见的疾病类型",
    "常见的疾病类型有什么",
    "常见的疾病类型有哪些",
    "疾病类型有什么",
    "疾病类型有哪些",
    "当前常见疾病有什么",
    "当前常见疾病有哪些",
    "当前常见病有什么",
    "疾病总数",
    "疾病名称",
    "疾病类型总数",
    "疾病种类",
    "疾病种类总数",
    "有哪些病",
    "有什么病",
    "有啥病",
    "都有什么病",
    "都有哪些病",
    "有什么疾病",
    "有哪些疾病",
    "病都有哪些",
    "病种",
    "病种分布",
    "疾病分布",
)
SUBGRAPH_HINTS = ("图谱", "子图", "知识图谱", "关系图", "关联图", "实体关系图")
SUBGRAPH_COHORT_HINTS = (
    "糖尿病",
    "高血压",
    "高脂血症",
    "冠心病",
    "慢性肾病",
    "脂肪肝",
    "慢阻肺",
    "哮喘",
    "骨关节炎",
    "痛风",
    "慢性心力衰竭",
    "糖尿病肾病",
    "睡眠呼吸暂停",
    "脑血管病",
    "房颤",
    "慢性肝炎",
    "甲减",
    "高风险",
    "中风险",
    "低风险",
    "患者",
    "群体",
    "队列",
    "病人",
)


def _parse_chinese_number(text: str) -> int | None:
    value = text.strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    if value == "十":
        return 10
    if "十" in value:
        left, _, right = value.partition("十")
        tens = CHINESE_DIGITS.get(left, 1 if left == "" else None)
        ones = CHINESE_DIGITS.get(right, 0 if right == "" else None)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    if value in CHINESE_DIGITS:
        return CHINESE_DIGITS[value]
    return None


def extract_future_window_days(question: str, default: int = 30) -> int:
    normalized = " ".join(str(question).strip().split())
    patterns = [
        r"(?:未来|接下来|后续)\s*([0-9零一二两三四五六七八九十]+)\s*(?:天|日)",
        r"([0-9零一二两三四五六七八九十]+)\s*(?:天|日)\s*(?:内)?\s*(?:随访|需要随访|待随访)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        parsed = _parse_chinese_number(match.group(1))
        if parsed is not None and parsed > 0:
            return parsed
    return default


def _is_future_followup_request(normalized: str) -> bool:
    return "未来" in normalized and "随访" in normalized


def _is_disease_inventory_request(normalized: str) -> bool:
    if "风险等级" in normalized or ("风险" in normalized and "分布" in normalized):
        return False
    return any(token in normalized for token in DISEASE_INVENTORY_HINTS)


def _is_subgraph_request(normalized: str) -> bool:
    return any(token in normalized for token in SUBGRAPH_HINTS) and any(token in normalized for token in SUBGRAPH_COHORT_HINTS)


def _build_future_followup_bundle_question(window_days: int) -> Dict[str, Any]:
    return {
        "question": f"根据未来 {window_days} 天随访人数，绘制折线图，饼状图",
        "canonical_id": "future_followup_chart_bundle",
        "reason": "future_followup_dynamic_question",
        "window_days": window_days,
    }


def rewrite_question(question: str) -> Dict[str, Any]:
    normalized = " ".join(str(question).strip().split())
    lowered = normalized.lower()
    window_days = extract_future_window_days(normalized, default=30)
    if normalized == "未来 30 天需要随访的高风险患者的疾病类型分布是什么？":
        return {
            "question": normalized,
            "canonical_id": "future_30d_high_risk_followup_disease_distribution",
            "reason": "canonical_graph_driven_question",
        }
    if normalized == "高盐饮食患者的血压异常比例是多少？":
        return {
            "question": normalized,
            "canonical_id": "high_salt_bp_abnormal_rate",
            "reason": "canonical_graph_driven_question",
        }
    if normalized == "高血压合并糖尿病患者的 HbA1c、血压和 LDL-C 异常情况如何？":
        return {
            "question": normalized,
            "canonical_id": "hypertension_diabetes_multi_indicator",
            "reason": "canonical_graph_driven_question",
        }
    if normalized == "根据未来随访人数，绘制折线图，饼状图":
        return {
            "question": normalized,
            "canonical_id": "future_followup_chart_bundle",
            "reason": "canonical_chart_bundle_question",
            "window_days": 30,
        }
    if _is_disease_inventory_request(normalized):
        return {
            "question": normalized,
            "canonical_id": "kg_disease_inventory",
            "reason": "disease_inventory_rule",
            "window_days": window_days,
        }
    if _is_subgraph_request(normalized):
        return {
            "question": normalized,
            "canonical_id": "dynamic_subgraph_render",
            "reason": "dynamic_subgraph_rule",
            "window_days": window_days,
        }
    if normalized in {
        "未来 30 天需要随访的患者有多少？",
        "未来 30 天需要随访的患者总数是多少？",
        "未来 30 天需要随访的人数是多少？",
        "未来随访人数是多少？",
    }:
        return {
            "question": "根据未来随访人数，绘制折线图，饼状图",
            "canonical_id": "future_followup_chart_bundle",
            "reason": "canonical_future_followup_count_question",
            "window_days": 30,
        }
    if any(token in normalized for token in ("他们", "该群体", "这些患者")) and any(token in normalized for token in PRONOUN_DISEASE_TOKENS):
        return {
            "question": "未来 30 天需要随访的高风险患者的疾病类型分布是什么？",
            "canonical_id": "future_30d_high_risk_followup_disease_distribution",
            "reason": "pronoun_disease_distribution_rule",
        }
    if "高盐" in normalized and ("血压" in normalized or "异常比例" in normalized):
        return {
            "question": "高盐饮食患者的血压异常比例是多少？",
            "canonical_id": "high_salt_bp_abnormal_rate",
            "reason": "lifestyle_bp_rule",
        }
    if "高血压" in normalized and "糖尿病" in normalized and any(token in lowered for token in ("hba1c", "ldl", "ldl-c", "血压", "异常情况")):
        return {
            "question": "高血压合并糖尿病患者的 HbA1c、血压和 LDL-C 异常情况如何？",
            "canonical_id": "hypertension_diabetes_multi_indicator",
            "reason": "multi_indicator_rule",
        }
    if (
        _is_future_followup_request(normalized)
        and any(token in normalized for token in FOLLOWUP_CHART_HINTS)
    ) or (
        any(token in normalized for token in ("绘制", "画", "生成"))
        and any(token in normalized for token in ("折线图", "饼图", "饼状图"))
        and any(token in normalized for token in ("未来", "随访"))
    ):
        payload = _build_future_followup_bundle_question(window_days)
        payload["reason"] = "future_followup_chart_rule"
        return payload
    if _is_future_followup_request(normalized) and any(token in normalized for token in FOLLOWUP_COUNT_HINTS):
        payload = _build_future_followup_bundle_question(window_days)
        payload["reason"] = "future_followup_count_visualized_rule"
        return payload
    if _is_future_followup_request(normalized) and any(token in normalized for token in ("多少", "总数", "人数", "患者数", "患者数量")):
        payload = _build_future_followup_bundle_question(window_days)
        payload["reason"] = "future_followup_count_rule"
        return payload
    return {
        "question": normalized,
        "canonical_id": "standard_or_unknown",
        "reason": "identity",
        "window_days": window_days,
    }
