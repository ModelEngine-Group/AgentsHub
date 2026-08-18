from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from analysis.query_schema import QueryPlan, TimeWindow

DISEASE_ALIASES = {
    "高血压": "hypertension",
    "hypertension": "hypertension",
    "糖尿病": "diabetes",
    "diabetes": "diabetes",
    "高脂血症": "hyperlipidemia",
    "高脂血": "hyperlipidemia",
    "hyperlipidemia": "hyperlipidemia",
    "肥胖": "obesity",
    "高尿酸": "hyperuricemia",
    "高尿酸血症": "hyperuricemia",
    "冠心病": "coronary_risk",
    "冠心病风险": "coronary_risk",
    "慢性肾病": "ckd_risk",
    "肾病": "ckd_risk",
    "脂肪肝": "fatty_liver_risk",
    "代谢综合征": "metabolic_syndrome",
    "脑卒中": "stroke_post",
    "中风": "stroke_post",
    "骨质疏松": "osteoporosis",
    "慢阻肺": "copd",
    "慢性阻塞性肺疾病": "copd",
    "慢病": "chronic_disease",
    "慢病患者": "chronic_disease",
}

RISK_ALIASES = {
    "高风险": "high",
    "中风险": "medium",
    "低风险": "low",
}

CHART_ALIASES = {
    "折线图": "line",
    "趋势图": "line",
    "趋势": "line",
    "饼图": "pie",
    "饼状图": "pie",
    "环形图": "pie",
    "表格": "table",
    "明细表": "table",
    "总览": "overview",
    "总览页": "overview",
    "统计图": "overview",
}


def _normalize_question(question: str) -> str:
    return " ".join(str(question or "").strip().split())


def _parse_chinese_number(token: str) -> Optional[int]:
    digits = {
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
    if not token:
        return None
    if token.isdigit():
        return int(token)
    if token == "十":
        return 10
    if "十" in token:
        left, _, right = token.partition("十")
        tens = digits.get(left, 1 if left == "" else None)
        ones = digits.get(right, 0 if right == "" else None)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    return digits.get(token)


def detect_time_window(question: str) -> Optional[TimeWindow]:
    normalized = _normalize_question(question)
    patterns = [
        (r"未来\s*([0-9零一二两三四五六七八九十]+)\s*天", "future"),
        (r"未来\s*([0-9零一二两三四五六七八九十]+)\s*日", "future"),
        (r"(?:接下来|后续)\s*([0-9零一二两三四五六七八九十]+)\s*(?:天|日)", "future"),
        (r"近\s*([0-9零一二两三四五六七八九十]+)\s*天", "past"),
        (r"近\s*([0-9零一二两三四五六七八九十]+)\s*日", "past"),
    ]
    for pattern, direction in patterns:
        match = re.search(pattern, normalized)
        if match:
            parsed = _parse_chinese_number(match.group(1))
            if parsed and parsed > 0:
                label = f"{'未来' if direction == 'future' else '近'} {parsed} 天"
                return TimeWindow(value=parsed, direction=direction, label=label)
    if "本月" in normalized:
        return TimeWindow(value=30, direction="current", label="本月")
    return None


def detect_diseases(question: str) -> List[str]:
    lowered = _normalize_question(question).lower()
    matched: List[str] = []
    for alias, disease in DISEASE_ALIASES.items():
        if alias.lower() in lowered and disease not in matched:
            matched.append(disease)
    return matched


def detect_risks(question: str) -> List[str]:
    normalized = _normalize_question(question)
    matched: List[str] = []
    for alias, risk in RISK_ALIASES.items():
        if alias in normalized and risk not in matched:
            matched.append(risk)
    return matched


def detect_chart_types(question: str) -> List[str]:
    normalized = _normalize_question(question)
    matched: List[str] = []
    for alias, chart_type in CHART_ALIASES.items():
        if alias in normalized and chart_type not in matched:
            matched.append(chart_type)
    return matched


def _is_graph_question(question: str) -> bool:
    normalized = _normalize_question(question)
    return any(token in normalized for token in ("图谱", "子图", "知识图谱", "关系图"))


def _is_report_question(question: str) -> bool:
    normalized = _normalize_question(question)
    return any(token in normalized for token in ("报告", "图表总览", "图谱入口", "稳定入口"))


def _is_disease_inventory_question(question: str) -> bool:
    normalized = _normalize_question(question)
    return any(
        token in normalized
        for token in (
            "几种病",
            "多少种病",
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
            "常见病",
            "常见疾病",
        )
    )


def _is_followup_question(question: str) -> bool:
    normalized = _normalize_question(question)
    return "随访" in normalized and any(token in normalized for token in ("未来", "近", "本月", "接下来", "后续"))


def _is_distribution_question(question: str) -> bool:
    normalized = _normalize_question(question)
    return any(token in normalized for token in ("分布", "占比", "比例"))


def _is_count_question(question: str) -> bool:
    normalized = _normalize_question(question)
    return any(token in normalized for token in ("多少", "人数", "数量", "统计"))


def _tool_plan_for_intent(intent: str) -> List[str]:
    mapping = {
        "future_followup_chart": ["analysis_query"],
        "risk_distribution": ["analysis_query"],
        "cohort_stats": ["analysis_query"],
        "kg_subgraph": ["kg_subgraph_render"],
        "kg_summary": ["kg_summary"],
        "report_summary": ["report_summary"],
        "nl2sql": ["open_analysis_query"],
        "graph_sql_joint_analysis": ["graph_driven_analysis"],
        "unknown": ["open_analysis_query"],
    }
    return mapping.get(intent, ["open_analysis_query"])


def plan_query(user_question: str, context: Dict[str, Any] | None = None) -> QueryPlan:
    question = _normalize_question(user_question)
    time_window = detect_time_window(question)
    diseases = detect_diseases(question)
    risks = detect_risks(question)
    chart_types = detect_chart_types(question)
    output_preference: List[str] = []
    if chart_types:
        output_preference.extend(["chart", "table"])
    if _is_graph_question(question):
        output_preference.append("graph")
    if not output_preference:
        output_preference.append("text")

    reason_parts: List[str] = []
    if time_window is not None:
        reason_parts.append(f"time_window={time_window.label}")
    if diseases:
        reason_parts.append(f"diseases={','.join(diseases)}")
    if risks:
        reason_parts.append(f"risks={','.join(risks)}")
    if chart_types:
        reason_parts.append(f"chart_types={','.join(chart_types)}")

    if "为什么" in question and "图谱" in question and "患者" in question:
        intent = "graph_sql_joint_analysis"
        route = "graph_driven"
        confidence = 0.95
    elif _is_disease_inventory_question(question):
        intent = "nl2sql"
        route = "analysis"
        confidence = 0.96
    elif _is_report_question(question):
        intent = "report_summary"
        route = "report"
        confidence = 0.96
    elif _is_graph_question(question) and (_is_count_question(question) or diseases or risks or time_window):
        intent = "graph_sql_joint_analysis"
        route = "graph_driven"
        confidence = 0.93
    elif _is_graph_question(question):
        if "质量" in question or "节点" in question or "边数" in question:
            intent = "kg_summary"
            route = "kg"
        else:
            intent = "kg_subgraph"
            route = "kg"
        confidence = 0.92
    elif _is_followup_question(question) and (
        chart_types
        or "图" in question
        or "可视化" in question
        or "随访人数" in question
        or "随访患者数" in question
        or "随访数量" in question
        or _is_count_question(question)
    ):
        intent = "future_followup_chart"
        route = "graph_driven"
        confidence = 0.94
    elif (_is_distribution_question(question) or "风险等级" in question) and (
        "风险" in question or risks or "风险等级" in question
    ):
        intent = "risk_distribution"
        route = "analysis"
        confidence = 0.9
    elif diseases or risks:
        intent = "cohort_stats" if (_is_count_question(question) or "统计" in question) else "nl2sql"
        route = "analysis"
        confidence = 0.87
    elif "sql" in question.lower() or "查询" in question:
        intent = "nl2sql"
        route = "analysis"
        confidence = 0.84
    else:
        intent = "unknown"
        route = "open"
        confidence = 0.45

    return QueryPlan(
        intent=intent,
        time_window=time_window,
        disease_filters=diseases,
        risk_filters=risks,
        chart_types=chart_types,
        output_preference=output_preference,
        tool_plan=_tool_plan_for_intent(intent),
        confidence=confidence,
        reason="; ".join(reason_parts) if reason_parts else "generic_fallback",
        canonical_question=context.get("canonical_question") if context else None,
        route=route,
    )
