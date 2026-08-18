from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, List

from analysis.open_nl2sql.disease_alias import normalize_diseases
from analysis.open_nl2sql.indicator_alias import normalize_indicators
from analysis.open_nl2sql.synonym_rewrite import extract_future_window_days
from runtime_common.cohort_context import has_pronoun_reference
from runtime_common.common import resolve_path

DICT_DIR = "configs/dict"


def _contains(text: str, tokens: List[str]) -> bool:
    return any(token in text for token in tokens)


def _contains_any_phrase(text: str, phrases: List[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _load_json(name: str) -> Dict[str, Any]:
    path = resolve_path(f"{DICT_DIR}/{name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_lines(name: str) -> List[str]:
    path = resolve_path(f"{DICT_DIR}/{name}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache(maxsize=1)
def _keyword_map() -> Dict[str, List[str]]:
    payload = _load_json("question_type_keywords.json")
    return {key: [str(item) for item in value] for key, value in payload.items()}


@lru_cache(maxsize=1)
def _deny_words() -> List[str]:
    return _load_lines("deny_words.txt")


def _has_negation(query: str) -> bool:
    text = str(query or "").strip()
    for token in _deny_words():
        if len(token) <= 1:
            continue
        if token in text:
            return True
    return bool(re.search(r"(^|[\s，。；、,])(?:否|非|不|无)(?:[\s，。；、,]|$)", text))


def _normalized_entities(query: str, last_context: Dict[str, Any]) -> Dict[str, Any]:
    days = extract_future_window_days(query, default=None) if "天" in query or "日" in query else None
    cohort = None
    if has_pronoun_reference(query):
        cohort = (last_context or {}).get("cohort_label") or (last_context or {}).get("name")
    return {
        "diseases": normalize_diseases(query),
        "indicators": normalize_indicators(query),
        "days": days,
        "cohort": cohort,
        "has_negation": _has_negation(query),
    }


def classify_question(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = str(payload.get("query") or "").strip()
    last_context = payload.get("last_context") or {}
    lowered = query.lower()
    entities = _normalized_entities(query, last_context)
    keywords = _keyword_map()

    if entities["has_negation"] and (
        entities["diseases"] or entities["indicators"] or "排除" in query or "不包含" in query
    ):
        return {
            "intent": "unsupported_negation_query",
            "normalized_entities": entities,
            "confidence": 0.98,
            "reason": "识别到否定条件查询，当前按不支持问题处理。",
        }
    if (
        _contains(query, ["现在 ChronicCare 支持哪些算子", "支持哪些 CPU 算子", "CPU 算子", "主线算子", "通用算子"])
        and "npu" not in lowered
    ):
        return {
            "intent": "datamate_pipelines",
            "normalized_entities": entities,
            "confidence": 0.98,
            "reason": "用户询问 ChronicCare DataMate CPU/通用主线算子。",
        }
    if "npu" in lowered or "昇腾" in query or "ascend" in lowered:
        if _contains(query, ["哪些", "支持", "覆盖", "算子"]) and not _contains(query, ["跑", "运行", "执行", "benchmark", "性能", "耗时", "对比"]):
            return {
                "intent": "npu_supported_operators",
                "normalized_entities": entities,
                "confidence": 0.98,
                "reason": "用户询问 NPU 增强覆盖的算子。",
            }
        if _contains(query, ["流水线", "pipeline", "datamate", "全流程", "运行", "执行", "重跑", "处理"]):
            return {
                "intent": "datamate_pipeline_run_npu",
                "normalized_entities": entities,
                "confidence": 0.98,
                "reason": "用户要求启用 NPU 增强运行 DataMate 流水线。",
            }
        if _contains(query, ["benchmark", "性能", "耗时", "对比", "加速效果", "跑一下", "测试"]):
            return {
                "intent": "npu_operator_benchmark",
                "normalized_entities": entities,
                "confidence": 0.98,
                "reason": "用户要求运行或查看 NPU 算子性能测试。",
            }
        return {
            "intent": "npu_readiness_query",
            "normalized_entities": entities,
            "confidence": 0.97,
            "reason": "用户询问 NPU runtime/readiness。",
        }
    if _contains(query, keywords["datamate_pipeline_run"]):
        return {
            "intent": "datamate_pipeline_run",
            "normalized_entities": entities,
            "confidence": 0.98,
            "reason": "用户要求运行 DataMate 全流程。",
        }
    if _contains(query, keywords["datamate_pipeline_status"]):
        return {
            "intent": "datamate_pipeline_status",
            "normalized_entities": entities,
            "confidence": 0.97,
            "reason": "用户询问 DataMate 流水线状态。",
        }
    if _contains(query, keywords["system_status"]):
        return {
            "intent": "system_status",
            "normalized_entities": entities,
            "confidence": 0.97,
            "reason": "用户询问系统状态。",
        }
    if (
        _contains(query, ["当前数据规模", "数据规模是多少", "多少患者", "多少随访记录", "多少检验记录", "多少用药记录"])
        and not _contains(query, ["图谱", "节点", "边", "质量评分"])
    ):
        return {
            "intent": "data_summary",
            "normalized_entities": entities,
            "confidence": 0.97,
            "reason": "用户询问当前患者、随访、检验和用药等数据规模。",
        }
    if (
        _contains(query, keywords["report_summary"])
        and not entities["diseases"]
        and not _contains(query, ["节点", "边", "图谱质量", "质量评分", "图谱规模", "多少节点", "多少边"])
    ):
        return {
            "intent": "report_summary",
            "normalized_entities": entities,
            "confidence": 0.95,
            "reason": "用户询问入口和报告。",
        }
    if _contains(query, keywords["kg_summary"]):
        return {
            "intent": "kg_summary",
            "normalized_entities": entities,
            "confidence": 0.96,
            "reason": "用户询问数据规模或图谱概览。",
        }
    if _contains(query, keywords["capability_examples"]):
        return {
            "intent": "capability_examples",
            "normalized_entities": entities,
            "confidence": 0.97,
            "reason": "用户询问当前分析能力和示例。",
        }
    if _contains(query, ["某个患者", "患者路径", "未来有哪些随访计划", "有哪些风险事件"]) and ("患者" in query or "P0" in query):
        return {
            "intent": "kg_patient_path_query",
            "normalized_entities": entities,
            "confidence": 0.94,
            "reason": "用户询问患者路径、风险事件或随访计划。",
        }
    if _contains(query, keywords["kg_relation_query"]):
        return {
            "intent": "kg_relation_query",
            "normalized_entities": entities,
            "confidence": 0.94,
            "reason": "用户询问实体关系。",
        }
    if _contains(query, keywords["kg_entity_query"]):
        return {
            "intent": "kg_entity_query",
            "normalized_entities": entities,
            "confidence": 0.94,
            "reason": "用户询问疾病实体关联。",
        }
    future_followup_count_hints = [
        "有多少",
        "多少人",
        "多少患者",
        "患者有多少",
        "患者数",
        "人数",
        "数量",
        "统计图",
        "趋势图",
        "折线图",
        "饼图",
        "可视化",
        "画出来",
        "画图",
    ]
    if "未来 N 天" in query and _contains(query, ["高风险", "随访"]):
        return {
            "intent": "future_n_days_high_risk_followup",
            "normalized_entities": entities,
            "confidence": 0.9,
            "reason": "用户使用 N 作为天数占位，前端应提示替换为具体天数或按工具默认处理。",
        }
    if (
        entities["days"]
        and _contains(query, ["高风险", "随访", "疾病类型", "慢病"])
        and _contains(query, ["疾病类型", "慢病", "疾病最多", "疾病分布"])
    ):
        return {
            "intent": "cohort_disease_distribution",
            "normalized_entities": entities,
            "confidence": 0.96,
            "reason": "用户询问未来高风险随访队列的疾病类型分布。",
        }
    if _contains(query, ["高风险患者中哪些疾病最多"]):
        return {
            "intent": "cohort_disease_distribution",
            "normalized_entities": entities,
            "confidence": 0.94,
            "reason": "用户询问高风险患者队列的疾病类型分布。",
        }
    if (
        entities["days"]
        and _contains(query, ["高风险"])
        and (_contains(query, ["随访"]) or _contains_any_phrase(query, future_followup_count_hints))
    ):
        if has_pronoun_reference(query) and entities["cohort"]:
            return {
                "intent": "cohort_disease_distribution",
                "normalized_entities": entities,
                "confidence": 0.93,
                "reason": "用户使用群体指代，需继承上一轮 cohort。",
            }
        return {
            "intent": "future_n_days_high_risk_followup",
            "normalized_entities": entities,
            "confidence": 0.97,
            "reason": "用户询问未来 N 天高风险随访。",
        }
    if (
        entities["days"]
        and (
            _contains(query, ["随访"])
            or ("未来" in query and _contains_any_phrase(query, future_followup_count_hints))
        )
    ):
        return {
            "intent": "future_n_days_followup",
            "normalized_entities": entities,
            "confidence": 0.97,
            "reason": "用户询问未来 N 天随访。",
        }
    if has_pronoun_reference(query):
        return {
            "intent": "cohort_disease_distribution",
            "normalized_entities": entities,
            "confidence": 0.92,
            "reason": "用户使用他们/该群体等指代，应继承上一轮 cohort。",
        }
    if _contains(query, ["高风险患者有多少", "中风险患者有多少", "低风险患者有多少"]):
        return {
            "intent": "risk_level_distribution",
            "normalized_entities": entities,
            "confidence": 0.95,
            "reason": "用户询问特定风险等级人数，应走风险分层统计。",
        }
    if _contains(query, keywords["risk_level_distribution"]) or ("风险等级" in query and "分布" in query):
        return {
            "intent": "risk_level_distribution",
            "normalized_entities": entities,
            "confidence": 0.95,
            "reason": "用户询问风险分层统计。",
        }
    if _contains(query, keywords["disease_combination_distribution"]) and _contains(query, ["分布", "多少"]):
        return {
            "intent": "disease_combination_distribution",
            "normalized_entities": entities,
            "confidence": 0.95,
            "reason": "用户询问疾病组合或共病分布。",
        }
    if entities["diseases"] and _contains(query, ["有多少", "占比是多少", "人数分布"]) and not _contains(query, ["高风险", "中风险", "低风险"]):
        return {
            "intent": "disease_distribution",
            "normalized_entities": entities,
            "confidence": 0.95,
            "reason": "用户围绕具体疾病询问人数、占比或分布。",
        }
    if _contains(query, keywords["disease_distribution"]):
        return {
            "intent": "disease_distribution",
            "normalized_entities": entities,
            "confidence": 0.96,
            "reason": "用户询问疾病分布，应统计疾病类型人数，不应走风险等级分布。",
        }
    if _contains(query, keywords["kg_subgraph_render"]) or (_contains(query, ["图谱", "子图"]) and not _contains(query, ["哪些检查指标", "哪些药物", "风险事件", "患者路径"])):
        return {
            "intent": "kg_subgraph_render",
            "normalized_entities": entities,
            "confidence": 0.95,
            "reason": "用户要求图谱或子图可视化。",
        }
    if entities["indicators"] or _contains(query, keywords["indicator_analysis"]):
        return {
            "intent": "indicator_analysis",
            "normalized_entities": entities,
            "confidence": 0.91,
            "reason": "用户询问指标统计、异常比例或趋势分析。",
        }
    if "npu" in lowered:
        return {
            "intent": "npu_readiness_query",
            "normalized_entities": entities,
            "confidence": 0.9,
            "reason": "用户询问 NPU 相关信息。",
        }
    if _contains(query, ["性能", "吞吐", "压测"]) or any(token in query for token in ("并发量", "并发能力", "并发测试")):
        return {
            "intent": "performance_query",
            "normalized_entities": entities,
            "confidence": 0.9,
            "reason": "用户询问性能类信息。",
        }
    return {
        "intent": "open_sql_analysis",
        "normalized_entities": entities,
        "confidence": 0.75,
        "reason": "归入开放式分析兜底路由。",
    }
