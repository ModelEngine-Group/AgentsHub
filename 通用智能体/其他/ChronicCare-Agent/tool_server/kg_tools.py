from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict, deque
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None
    ImageDraw = None
    ImageFont = None

from kg.graph_io import adjacency_indexes, load_graph_json
from runtime_common.analysis_context import AnalysisContext, attach_analysis_context
from runtime_common.cohort_context import has_pronoun_reference, resolve_active_cohort
from runtime_common.common import read_json, relative_to_project, resolve_path
from tool_server.utils import (
    fetch_one,
    fetch_rows,
    load_current_metrics,
    load_server_config,
    public_artifact_url,
    read_optional_json,
    safety_note,
    service_artifact_url,
)


def _with_analysis_context(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        payload = func(*args, **kwargs)
        if isinstance(payload, dict):
            return attach_analysis_context(payload, AnalysisContext.current())
        return payload

    return wrapped


ENTITY_ALIASES = {
    "高血压": "Disease::hypertension",
    "hypertension": "Disease::hypertension",
    "糖尿病": "Disease::diabetes",
    "diabetes": "Disease::diabetes",
    "高脂血症": "Disease::hyperlipidemia",
    "hyperlipidemia": "Disease::hyperlipidemia",
    "肥胖": "Disease::obesity",
    "obesity": "Disease::obesity",
    "高尿酸": "Disease::hyperuricemia",
    "高尿酸血症": "Disease::hyperuricemia",
    "冠心病": "Disease::coronary_heart_disease",
    "冠心病风险": "Disease::coronary_risk",
    "慢性肾病": "Disease::chronic_kidney_disease",
    "慢性肾病风险": "Disease::ckd_risk",
    "脂肪肝": "Disease::fatty_liver_disease",
    "脂肪肝风险": "Disease::fatty_liver_risk",
    "哮喘": "Disease::asthma",
    "骨关节炎": "Disease::osteoarthritis",
    "痛风": "Disease::gout",
    "慢性心力衰竭": "Disease::chronic_heart_failure",
    "糖尿病肾病": "Disease::diabetic_kidney_disease",
    "阻塞性睡眠呼吸暂停": "Disease::obstructive_sleep_apnea",
    "脑血管病": "Disease::cerebrovascular_disease",
    "心房颤动": "Disease::atrial_fibrillation",
    "房颤": "Disease::atrial_fibrillation",
    "慢性肝炎": "Disease::chronic_hepatitis",
    "甲状腺功能减退": "Disease::hypothyroidism",
    "hba1c": "Indicator::hba1c",
    "糖化血红蛋白": "Indicator::hba1c",
    "ldl-c": "Indicator::ldl_c",
    "ldl": "Indicator::ldl_c",
    "空腹血糖": "Indicator::fasting_glucose",
    "收缩压": "Indicator::systolic_bp",
    "舒张压": "Indicator::diastolic_bp",
    "高盐饮食": "RiskFactor::high_salt_diet",
}

TYPE_COLORS = {
    "Patient": "#2457a5",
    "Disease": "#d1495b",
    "Indicator": "#2e7d32",
    "Drug": "#7b5ea7",
    "RiskEvent": "#e07a1f",
    "FollowupPlan": "#00838f",
    "LifestyleRecord": "#5c6b73",
    "DoctorAdvice": "#a23b72",
    "RiskScore": "#6d4c41",
    "LabResult": "#607d8b",
    "DrugCategory": "#546e7a",
    "RiskFactor": "#8d6e63",
}
TYPE_DISPLAY_NAMES = {
    "Patient": "患者",
    "Disease": "疾病",
    "Indicator": "指标",
    "Drug": "药物",
    "RiskEvent": "风险事件",
    "FollowupPlan": "随访计划",
    "LifestyleRecord": "生活方式",
    "DoctorAdvice": "医生建议",
    "RiskScore": "风险评分",
    "LabResult": "检验结果",
    "DrugCategory": "药物类别",
    "RiskFactor": "风险因素",
}
DISEASE_LABELS = {
    "hypertension": "高血压",
    "diabetes": "糖尿病",
    "hyperlipidemia": "高脂血症",
    "obesity": "肥胖",
    "hyperuricemia": "高尿酸血症",
    "coronary": "冠心病风险",
    "coronary_risk": "冠心病风险",
    "ckd_risk": "慢性肾病风险",
    "fatty_liver_risk": "脂肪肝风险",
    "heart_failure": "心力衰竭风险",
    "metabolic_syndrome": "代谢综合征风险",
    "sleep_apnea_risk": "睡眠呼吸暂停风险",
    "stroke_post": "合成脑卒中既往史",
    "diabetic_kidney_risk": "糖尿病肾病风险",
    "copd": "慢性阻塞性肺疾病（COPD）",
    "osteoporosis": "骨质疏松",
    "chronic_kidney_disease": "慢性肾病",
    "coronary_heart_disease": "冠心病",
    "fatty_liver_disease": "脂肪肝",
    "asthma": "哮喘",
    "osteoarthritis": "骨关节炎",
    "gout": "痛风",
    "chronic_heart_failure": "慢性心力衰竭",
    "diabetic_kidney_disease": "糖尿病肾病",
    "obstructive_sleep_apnea": "阻塞性睡眠呼吸暂停",
    "cerebrovascular_disease": "脑血管病",
    "atrial_fibrillation": "心房颤动",
    "chronic_hepatitis": "慢性肝炎",
    "hypothyroidism": "甲状腺功能减退",
}
INDICATOR_LABELS = {
    "hba1c": "糖化血红蛋白",
    "ldl_c": "低密度脂蛋白胆固醇",
    "fasting_glucose": "空腹血糖",
    "systolic_bp": "收缩压",
    "diastolic_bp": "舒张压",
    "egfr": "估算肾小球滤过率",
    "uacr": "尿白蛋白肌酐比",
    "bmi": "体重指数",
    "uric_acid": "尿酸",
    "hdl_c": "高密度脂蛋白胆固醇",
    "total_cholesterol": "总胆固醇",
    "triglyceride": "甘油三酯",
    "creatinine": "肌酐",
    "alt": "谷丙转氨酶",
    "ast": "谷草转氨酶",
}
RISK_EVENT_LABELS = {
    "bmi_high": "BMI 偏高",
    "glucose_high": "血糖偏高",
    "lipid_abnormal": "血脂异常",
    "blood_pressure_high": "血压偏高",
    "renal_risk": "肾脏风险",
    "uric_acid_high": "尿酸偏高",
    "liver_risk": "肝脏风险",
}
RISK_FACTOR_LABELS = {
    "high_salt_diet": "高盐饮食",
}
RISK_SCORE_LABELS = {
    "high_risk_group": "高风险患者群体",
    "medium_risk_group": "中风险患者群体",
    "low_risk_group": "低风险患者群体",
}
DRUG_LABELS = {
    "atorvastatin": "阿托伐他汀",
    "metformin": "二甲双胍",
    "dapagliflozin": "达格列净",
    "valsartan": "缬沙坦",
    "amlodipine": "氨氯地平",
    "lifestyle_intervention": "生活方式干预",
    "febuxostat": "非布司他",
    "acarbose": "阿卡波糖",
}
LEGEND_TYPES = [
    "Patient",
    "Disease",
    "Indicator",
    "Drug",
    "RiskEvent",
    "FollowupPlan",
    "LifestyleRecord",
    "DoctorAdvice",
    "RiskScore",
]
PATIENT_SAMPLE_LIMIT = 60
RUNTIME_SUBGRAPH_DIR = "outputs/runtime_generated/subgraphs"
LOCAL_SUBGRAPH_DIR = "outputs/local_runtime/subgraphs"


def _topic_from_query(query: str) -> str:
    text = str(query or "").strip()
    text = re.sub(r"^(请|帮我|给我|生成|画出|画一下|查看|查询)", "", text).strip()
    text = re.sub(r"(的)?(知识图谱)?(子图|关系图|关联图|图谱)$", "", text).strip()
    text = text.strip("。！？? ")
    return text or "当前问题相关主题"


def _synthetic_query_subgraph_payload(query: str) -> Dict[str, Any]:
    topic = _topic_from_query(query)
    safe_topic = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", "_", topic).strip("_") or "query_topic"
    topic_node_id = f"Disease::query_{safe_topic[:40]}"
    nodes = [
        {
            "id": topic_node_id,
            "type": "Disease",
            "label": topic,
            "display_name": topic,
            "synthetic": True,
        }
    ]
    return {
        "status": "success",
        "query": query,
        "mapped_intent": "query_subgraph_no_direct_entity_match",
        "seed_ids": [topic_node_id],
        "seed_labels": [topic],
        "node_count": len(nodes),
        "edge_count": 0,
        "nodes": nodes,
        "edges": [],
        "cohort_patient_count": 0,
        "display_patient_node_count": 0,
        "semantic_node_count": len(nodes),
        "top_indicators": [],
        "top_risk_events": [],
        "top_drugs": [],
        "graph_scope_explanation": f"当前知识图谱中暂未找到与“{topic}”直接匹配的结构化实体或患者 cohort，已生成该查询主题的可打开局部子图占位页。",
        "summary_text": f"当前图谱未直接命中“{topic}”，已生成可打开的查询主题子图；若后续数据接入该实体，子图会自动扩展真实关系。",
        "signature": "query_" + hashlib.md5(query.encode("utf-8")).hexdigest()[:16],
    }


def _normalize_lookup_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = lowered.replace("（", "(").replace("）", ")")
    lowered = lowered.replace("，", ",").replace("：", ":").replace("；", ";")
    lowered = re.sub(r"[\s_\-]+", "", lowered)
    return lowered


def _build_entity_alias_map(nodes: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for alias, entity_id in ENTITY_ALIASES.items():
        normalized = _normalize_lookup_text(alias)
        if normalized:
            alias_map[normalized] = entity_id

    for node_id, node in nodes.items():
        node_type = str(node.get("type") or "")
        raw_key = str(node_id.split("::", 1)[-1] or "").strip()
        display_name = str(node.get("display_name") or node.get("label") or node.get("name") or "").strip()
        if raw_key.lower() == "nan" or display_name.lower() == "nan":
            continue

        aliases: List[str] = []
        aliases.extend([raw_key, raw_key.replace("_", " ")])
        if display_name:
            aliases.append(display_name)

        if node_type == "Disease":
            cn_label = DISEASE_LABELS.get(raw_key)
            if cn_label:
                aliases.append(cn_label)
            if raw_key.endswith("_risk"):
                aliases.append(raw_key[:-5])
            if raw_key == "coronary_risk":
                aliases.extend(["冠心病", "冠心病风险"])
            if raw_key == "ckd_risk":
                aliases.extend(["慢性肾病", "慢性肾病风险", "ckd"])
            if raw_key == "fatty_liver_risk":
                aliases.extend(["脂肪肝", "脂肪肝风险"])
            if raw_key == "sleep_apnea_risk":
                aliases.extend(["睡眠呼吸暂停", "睡眠呼吸暂停风险"])
            if raw_key == "stroke_post":
                aliases.extend(["脑卒中", "中风", "脑卒中后状态", "中风后状态"])
            if raw_key == "copd":
                aliases.extend(["慢阻肺", "慢性阻塞性肺疾病", "chronicobstructivepulmonarydisease"])
        elif node_type == "Indicator":
            cn_label = INDICATOR_LABELS.get(raw_key)
            if cn_label:
                aliases.append(cn_label)
        elif node_type == "Drug":
            cn_label = DRUG_LABELS.get(raw_key)
            if cn_label:
                aliases.append(cn_label)
        elif node_type == "RiskEvent":
            cn_label = RISK_EVENT_LABELS.get(raw_key)
            if cn_label:
                aliases.append(cn_label)
        elif node_type == "RiskFactor":
            cn_label = RISK_FACTOR_LABELS.get(raw_key)
            if cn_label:
                aliases.append(cn_label)
        elif node_type == "RiskScore":
            cn_label = RISK_SCORE_LABELS.get(raw_key)
            if cn_label:
                aliases.append(cn_label)

        for alias in aliases:
            normalized = _normalize_lookup_text(alias)
            if normalized and normalized not in alias_map:
                alias_map[normalized] = node_id

    return alias_map


def _safe_replace(path: Path) -> Path:
    if path.exists():
        path.unlink()
    return path


def _subgraph_output_dir(subgraph_id: str) -> Path:
    for base_str in [RUNTIME_SUBGRAPH_DIR, LOCAL_SUBGRAPH_DIR, "outputs/subgraphs"]:
        base_dir = resolve_path(base_str)
        base_dir.mkdir(parents=True, exist_ok=True)
        probe = base_dir / f".{subgraph_id}.probe"
        try:
            for suffix in [".html", ".json", ".svg", ".png"]:
                existing = base_dir / f"{subgraph_id}{suffix}"
                if existing.exists():
                    existing.unlink()
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return base_dir
        except OSError:
            continue
    fallback_dir = resolve_path(LOCAL_SUBGRAPH_DIR)
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return fallback_dir


def _load_graph() -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    cfg = load_server_config()
    graph = load_graph_json(resolve_path(cfg["paths"]["graph_json"]))
    nodes = {node["id"]: node for node in graph["nodes"]}
    outgoing, incoming = adjacency_indexes(graph["edges"])
    return graph, nodes, outgoing, incoming


def _graph_summary_dict() -> Dict[str, Any]:
    cfg = load_server_config()
    return read_json(resolve_path(cfg["paths"]["graph_summary"]))


def _label_for(node_id: str, nodes: Dict[str, Dict[str, Any]]) -> str:
    node = nodes.get(node_id, {})
    node_type = str(node.get("type", node_id.split("::", 1)[0]))
    raw = str(node.get("display_name") or node.get("label") or node.get("name") or node_id.split("::")[-1])
    if node_type == "Patient":
        return f"患者 {node_id.split('::', 1)[-1]}"
    if node_type == "Disease":
        return DISEASE_LABELS.get(raw, DISEASE_LABELS.get(node_id.split("::", 1)[-1], raw))
    if node_type == "Indicator":
        normalized = node_id.split("::", 1)[-1]
        return f"指标 {INDICATOR_LABELS.get(raw, INDICATOR_LABELS.get(normalized, raw))}"
    if node_type == "Drug":
        normalized = node_id.split("::", 1)[-1]
        return f"药物 {DRUG_LABELS.get(raw, DRUG_LABELS.get(normalized, raw))}"
    if node_type == "RiskEvent":
        normalized = node_id.split("::", 1)[-1]
        return f"风险事件 {RISK_EVENT_LABELS.get(raw, RISK_EVENT_LABELS.get(normalized, raw))}"
    if node_type == "RiskFactor":
        normalized = node_id.split("::", 1)[-1]
        return RISK_FACTOR_LABELS.get(raw, RISK_FACTOR_LABELS.get(normalized, raw))
    if node_type == "RiskScore":
        normalized = node_id.split("::", 1)[-1]
        return RISK_SCORE_LABELS.get(raw, RISK_SCORE_LABELS.get(normalized, raw))
    return raw


def _normalize_entity_ids(query: str, nodes: Dict[str, Dict[str, Any]]) -> List[str]:
    lowered = _normalize_lookup_text(query)
    alias_map = _build_entity_alias_map(nodes)
    found: List[str] = []
    for alias, entity_id in sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in lowered and entity_id not in found:
            found.append(entity_id)
    patient_match = re.findall(r"\bP\d{4}\b", query, flags=re.IGNORECASE)
    for patient_id in patient_match:
        entity_id = f"Patient::{patient_id.upper()}"
        if entity_id in nodes and entity_id not in found:
            found.append(entity_id)
    return found


def _query_patient_ids_by_diseases(diseases: List[str]) -> List[str]:
    if not diseases:
        return []
    conditions = " AND ".join(["lower(disease_tags) LIKE ?"] * len(diseases))
    params = [f"%{item.lower()}%" for item in diseases]
    rows = fetch_rows(f"SELECT patient_id FROM patient_profile WHERE {conditions}", params)
    return [str(row["patient_id"]) for row in rows]


def _query_patient_ids_by_risk_level(risk_level: str) -> List[str]:
    rows = fetch_rows(
        "SELECT DISTINCT patient_id FROM patient_risk_score WHERE lower(risk_level) = ?",
        [risk_level.lower()],
    )
    return [str(row["patient_id"]) for row in rows]


def _top_indicator_rows(patient_ids: List[str], limit: int | None = 10) -> List[Dict[str, Any]]:
    if not patient_ids:
        return []
    placeholders = ",".join(["?"] * len(patient_ids))
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    sql = f"""
        SELECT
          item_name AS indicator,
          COUNT(DISTINCT patient_id) AS value,
          COUNT(DISTINCT patient_id) AS patient_count,
          COUNT(*) AS record_count
        FROM lab_result
        WHERE patient_id IN ({placeholders})
        GROUP BY item_name
        ORDER BY patient_count DESC, record_count DESC, indicator ASC
        {limit_clause}
    """
    rows = fetch_rows(sql, patient_ids)
    for row in rows:
        key = str(row.get("indicator") or "")
        row["display_name"] = INDICATOR_LABELS.get(key, key)
    return rows


def _top_risk_event_rows(patient_ids: List[str], limit: int | None = 10) -> List[Dict[str, Any]]:
    if not patient_ids:
        return []
    placeholders = ",".join(["?"] * len(patient_ids))
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    sql = f"""
        SELECT
          event_type,
          COUNT(DISTINCT patient_id) AS value,
          COUNT(DISTINCT patient_id) AS patient_count,
          COUNT(*) AS record_count
        FROM risk_event
        WHERE patient_id IN ({placeholders})
        GROUP BY event_type
        ORDER BY patient_count DESC, record_count DESC, event_type ASC
        {limit_clause}
    """
    rows = fetch_rows(sql, patient_ids)
    for row in rows:
        key = str(row.get("event_type") or "")
        row["display_name"] = RISK_EVENT_LABELS.get(key, key)
    return rows


def _top_drug_rows(patient_ids: List[str], limit: int | None = 10) -> List[Dict[str, Any]]:
    if not patient_ids:
        return []
    placeholders = ",".join(["?"] * len(patient_ids))
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    sql = f"""
        SELECT
          drug_name,
          COUNT(DISTINCT patient_id) AS value,
          COUNT(DISTINCT patient_id) AS patient_count,
          COUNT(*) AS record_count
        FROM medication_record
        WHERE patient_id IN ({placeholders})
        GROUP BY drug_name
        ORDER BY patient_count DESC, record_count DESC, drug_name ASC
        {limit_clause}
    """
    rows = fetch_rows(sql, patient_ids)
    for row in rows:
        key = str(row.get("drug_name") or "")
        row["display_name"] = DRUG_LABELS.get(key, key)
    return rows


def _kg_relation_intent(query: str) -> Dict[str, bool]:
    return {
        "indicator": any(token in query for token in ("指标", "检查", "检验", "HbA1c", "hba1c")),
        "drug": any(token in query for token in ("药物", "用药", "药")),
        "risk_event": any(token in query for token in ("风险事件", "风险", "事件")),
    }


def _indicator_table(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    table_rows = [
        {
            "指标名称": row.get("display_name") or row.get("indicator"),
            "指标编码": row.get("indicator"),
            "覆盖患者数": int(row.get("patient_count", row.get("value", 0)) or 0),
            "检验记录数": int(row.get("record_count", 0) or 0),
        }
        for row in rows
    ]
    return {
        "kind": "indicator",
        "row_count": len(table_rows),
        "rows": table_rows,
        "strict_rows_only": True,
    }


def _drug_table(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    table_rows = [
        {
            "药物名称": row.get("display_name") or row.get("drug_name"),
            "药物编码": row.get("drug_name"),
            "覆盖患者数": int(row.get("patient_count", row.get("value", 0)) or 0),
            "用药记录数": int(row.get("record_count", 0) or 0),
        }
        for row in rows
    ]
    return {
        "kind": "drug",
        "row_count": len(table_rows),
        "rows": table_rows,
        "strict_rows_only": True,
        "allowed_names": [str(row.get("药物名称") or "") for row in table_rows if str(row.get("药物名称") or "").strip()],
    }


def _risk_event_table(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    table_rows = [
        {
            "风险事件": row.get("display_name") or row.get("event_type"),
            "事件编码": row.get("event_type"),
            "覆盖患者数": int(row.get("patient_count", row.get("value", 0)) or 0),
            "事件记录数": int(row.get("record_count", 0) or 0),
        }
        for row in rows
    ]
    return {
        "kind": "risk_event",
        "row_count": len(table_rows),
        "rows": table_rows,
        "strict_rows_only": True,
    }


def _combined_detail_table(
    indicators: List[Dict[str, Any]],
    drugs: List[Dict[str, Any]],
    risk_events: List[Dict[str, Any]],
    intent: Dict[str, bool],
) -> Dict[str, Any]:
    table_rows: List[Dict[str, Any]] = []
    if intent["indicator"]:
        table_rows.extend(
            {
                "类别": "检查指标",
                "名称": row.get("display_name") or row.get("indicator"),
                "编码": row.get("indicator"),
                "覆盖患者数": int(row.get("patient_count", row.get("value", 0)) or 0),
                "记录数": int(row.get("record_count", 0) or 0),
            }
            for row in indicators
        )
    if intent["drug"]:
        table_rows.extend(
            {
                "类别": "药物",
                "名称": row.get("display_name") or row.get("drug_name"),
                "编码": row.get("drug_name"),
                "覆盖患者数": int(row.get("patient_count", row.get("value", 0)) or 0),
                "记录数": int(row.get("record_count", 0) or 0),
            }
            for row in drugs
        )
    if intent["risk_event"]:
        table_rows.extend(
            {
                "类别": "风险事件",
                "名称": row.get("display_name") or row.get("event_type"),
                "编码": row.get("event_type"),
                "覆盖患者数": int(row.get("patient_count", row.get("value", 0)) or 0),
                "记录数": int(row.get("record_count", 0) or 0),
            }
            for row in risk_events
        )
    return {
        "kind": "combined",
        "row_count": len(table_rows),
        "rows": table_rows,
        "strict_rows_only": True,
        "category_counts": {
            "检查指标": len(indicators) if intent["indicator"] else 0,
            "药物": len(drugs) if intent["drug"] else 0,
            "风险事件": len(risk_events) if intent["risk_event"] else 0,
        },
    }


def _kg_detail_table_for_intent(query: str, indicators: List[Dict[str, Any]], drugs: List[Dict[str, Any]], risk_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    intent = _kg_relation_intent(query)
    if sum(bool(value) for value in intent.values()) > 1:
        return _combined_detail_table(indicators, drugs, risk_events, intent)
    if intent["drug"] and not intent["indicator"]:
        return _drug_table(drugs)
    if intent["risk_event"] and not intent["indicator"] and not intent["drug"]:
        return _risk_event_table(risk_events)
    return _indicator_table(indicators)


def _cohort_disease_distribution(patient_ids: List[str]) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    if not patient_ids:
        return []
    placeholders = ",".join(["?"] * len(patient_ids))
    rows = fetch_rows(f"SELECT disease_tags FROM patient_profile WHERE patient_id IN ({placeholders})", patient_ids)
    for row in rows:
        raw = str(row.get("disease_tags") or "")
        for tag in [item.strip() for item in raw.split(";") if item.strip() and item.strip() != "nan"]:
            counter[tag] += 1
    return [{"disease": disease, "patient_count": count} for disease, count in counter.most_common(12)]


def _sample_patient_ids(patient_ids: List[str], limit: int = 6) -> List[str]:
    return sorted(patient_ids)[:limit]


def _query_high_salt_bp_abnormal_patient_ids() -> List[str]:
    rows = fetch_rows(
        """
        SELECT DISTINCT s.patient_id
        FROM lifestyle_record s
        JOIN lab_result l
          ON s.patient_id = l.patient_id
         AND s.visit_id = l.visit_id
        WHERE s.salt_intake_level = ?
          AND l.item_name IN ('systolic_bp', 'diastolic_bp')
          AND COALESCE(l.abnormal_flag, '') != 'normal'
        ORDER BY s.patient_id
        """,
        ["high"],
    )
    return [str(row.get("patient_id")) for row in rows if str(row.get("patient_id") or "").strip()]


def _high_salt_bp_relation_subgraph_payload(query: str, nodes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    patient_ids = _query_high_salt_bp_abnormal_patient_ids()
    sample_patients = _sample_patient_ids(patient_ids, limit=min(12, len(patient_ids)))
    indicator_keys = ["systolic_bp", "diastolic_bp"]
    risk_event_keys = ["blood_pressure_high"]
    risk_factor_id = "RiskFactor::high_salt_diet"

    subgraph_nodes: Dict[str, Dict[str, Any]] = {}
    subgraph_edges: List[Dict[str, Any]] = []

    subgraph_nodes[risk_factor_id] = nodes.get(
        risk_factor_id,
        {"id": risk_factor_id, "type": "RiskFactor", "display_name": "high_salt_diet"},
    )

    for indicator_key in indicator_keys:
        node_id = f"Indicator::{indicator_key}"
        subgraph_nodes[node_id] = nodes.get(
            node_id,
            {"id": node_id, "type": "Indicator", "display_name": indicator_key},
        )
        subgraph_edges.append({"source": risk_factor_id, "relation": "associated_indicator", "target": node_id})

    for risk_key in risk_event_keys:
        node_id = f"RiskEvent::{risk_key}"
        subgraph_nodes[node_id] = nodes.get(
            node_id,
            {"id": node_id, "type": "RiskEvent", "display_name": risk_key},
        )
        subgraph_edges.append({"source": risk_factor_id, "relation": "associated_risk_event", "target": node_id})
        for indicator_key in indicator_keys:
            subgraph_edges.append(
                {
                    "source": f"Indicator::{indicator_key}",
                    "relation": "indicator_supports_event",
                    "target": node_id,
                }
            )

    for patient_id in sample_patients:
        node_id = f"Patient::{patient_id}"
        subgraph_nodes[node_id] = {
            "id": node_id,
            "type": "Patient",
            "display_name": patient_id,
        }
        subgraph_edges.append({"source": node_id, "relation": "patient_has_lifestyle_factor", "target": risk_factor_id})
        for risk_key in risk_event_keys:
            subgraph_edges.append(
                {
                    "source": node_id,
                    "relation": "patient_has_risk_event",
                    "target": f"RiskEvent::{risk_key}",
                }
            )

    return {
        "status": "success",
        "query": query,
        "seed_ids": [risk_factor_id, "RiskEvent::blood_pressure_high"],
        "seed_labels": ["高盐饮食", "血压异常"],
        "cohort_patient_count": len(patient_ids),
        "display_patient_node_count": len(sample_patients),
        "semantic_node_count": 1 + len(indicator_keys) + len(risk_event_keys),
        "node_count": len(subgraph_nodes),
        "edge_count": len(subgraph_edges),
        "top_indicators": [{"indicator": key, "value": len(patient_ids)} for key in indicator_keys],
        "top_risk_events": [{"event_type": key, "value": len(patient_ids)} for key in risk_event_keys],
        "top_drugs": [],
        "nodes": list(subgraph_nodes.values()),
        "edges": subgraph_edges,
        "signature": "graph_query_high_salt_bp_abnormal_relation",
        "graph_scope_explanation": (
            f"该关系群体患者共 {len(patient_ids)} 人；页面当前展示其中 {len(sample_patients)} 个示例患者节点，"
            f"并聚焦展示 2 个血压指标与 1 个核心风险事件，用于说明高盐饮食与血压异常之间的直接关联路径。"
        ),
    }


def _cohort_signature(entity_ids: List[str]) -> str:
    suffix = "_".join(sorted(item.split("::", 1)[-1] for item in entity_ids))
    return f"cohort_subgraph_{suffix}" if suffix else "cohort_subgraph"


def _cohort_subgraph_payload(query: str, disease_entity_ids: List[str], nodes: Dict[str, Dict[str, Any]], max_nodes: int = 80) -> Dict[str, Any]:
    disease_keys = [item.split("::", 1)[-1] for item in disease_entity_ids]
    patient_ids = _query_patient_ids_by_diseases(disease_keys)
    indicator_rows = _top_indicator_rows(patient_ids, limit=12)
    risk_rows = _top_risk_event_rows(patient_ids, limit=8)
    drug_rows = _top_drug_rows(patient_ids, limit=8)
    sample_patients = _sample_patient_ids(patient_ids, limit=min(PATIENT_SAMPLE_LIMIT, len(patient_ids)))
    subgraph_nodes: Dict[str, Dict[str, Any]] = {}
    subgraph_edges: List[Dict[str, Any]] = []
    for entity_id in disease_entity_ids:
        if entity_id in nodes:
            subgraph_nodes[entity_id] = nodes[entity_id]
    for row in indicator_rows:
        indicator_key = str(row.get("indicator") or "")
        node_id = f"Indicator::{indicator_key}"
        subgraph_nodes[node_id] = {
            "id": node_id,
            "type": "Indicator",
            "display_name": indicator_key,
        }
        for disease_id in disease_entity_ids:
            subgraph_edges.append({"source": disease_id, "relation": "cohort_core_indicator", "target": node_id})
    for row in risk_rows:
        risk_key = str(row.get("event_type") or "")
        node_id = f"RiskEvent::{risk_key}"
        subgraph_nodes[node_id] = {
            "id": node_id,
            "type": "RiskEvent",
            "display_name": risk_key,
        }
        for disease_id in disease_entity_ids:
            subgraph_edges.append({"source": disease_id, "relation": "cohort_common_risk", "target": node_id})
    for row in drug_rows:
        drug_key = str(row.get("drug_name") or "")
        node_id = f"Drug::{drug_key}"
        subgraph_nodes[node_id] = {
            "id": node_id,
            "type": "Drug",
            "display_name": drug_key,
        }
        for disease_id in disease_entity_ids:
            subgraph_edges.append({"source": disease_id, "relation": "cohort_common_drug", "target": node_id})
    for patient_id in sample_patients:
        node_id = f"Patient::{patient_id}"
        subgraph_nodes[node_id] = {
            "id": node_id,
            "type": "Patient",
            "display_name": patient_id,
        }
        for disease_id in disease_entity_ids:
            subgraph_edges.append({"source": node_id, "relation": "patient_has_disease", "target": disease_id})
    return {
        "status": "success",
        "query": query,
        "seed_ids": disease_entity_ids,
        "seed_labels": [_label_for(entity_id, nodes) for entity_id in disease_entity_ids],
        "cohort_patient_count": len(patient_ids),
        "display_patient_node_count": len(sample_patients),
        "semantic_node_count": len(disease_entity_ids) + len(indicator_rows) + len(risk_rows) + len(drug_rows),
        "node_count": len(subgraph_nodes),
        "edge_count": len(subgraph_edges),
        "top_indicators": indicator_rows[:10],
        "top_risk_events": risk_rows[:10],
        "top_drugs": drug_rows[:10],
        "nodes": list(subgraph_nodes.values()),
        "edges": subgraph_edges,
        "signature": _cohort_signature(disease_entity_ids),
        "graph_scope_explanation": (
            f"该群体实际患者共 {len(patient_ids)} 人；页面当前展示其中 {len(sample_patients)} 个示例患者节点，"
            f"并聚焦展示最常见的 {len(indicator_rows)} 个指标、{len(risk_rows)} 个风险事件和 {len(drug_rows)} 个用药节点。"
        ),
    }


def _risk_level_entity_id(risk_level: str) -> str:
    return f"RiskScore::{risk_level.lower()}"


def _risk_cohort_subgraph_payload(query: str, risk_level: str, nodes: Dict[str, Dict[str, Any]], max_nodes: int = 80) -> Dict[str, Any]:
    patient_ids = _query_patient_ids_by_risk_level(risk_level)
    indicator_rows = _top_indicator_rows(patient_ids, limit=None)
    risk_rows = _top_risk_event_rows(patient_ids, limit=None)
    drug_rows = _top_drug_rows(patient_ids, limit=None)
    risk_node_id = _risk_level_entity_id(risk_level)
    subgraph_nodes: Dict[str, Dict[str, Any]] = {
        risk_node_id: {"id": risk_node_id, "type": "RiskScore", "display_name": f"{risk_level}_risk_group"}
    }
    subgraph_edges: List[Dict[str, Any]] = []
    sample_patients = _sample_patient_ids(patient_ids, limit=min(PATIENT_SAMPLE_LIMIT, len(patient_ids)))
    for row in indicator_rows:
        indicator_key = str(row.get("indicator") or "")
        node_id = f"Indicator::{indicator_key}"
        subgraph_nodes[node_id] = {
            "id": node_id,
            "type": "Indicator",
            "display_name": indicator_key,
        }
        subgraph_edges.append({"source": risk_node_id, "relation": "risk_group_core_indicator", "target": node_id})
    for row in risk_rows:
        risk_key = str(row.get("event_type") or "")
        node_id = f"RiskEvent::{risk_key}"
        subgraph_nodes[node_id] = {
            "id": node_id,
            "type": "RiskEvent",
            "display_name": risk_key,
        }
        subgraph_edges.append({"source": risk_node_id, "relation": "risk_group_common_event", "target": node_id})
    for row in drug_rows:
        drug_key = str(row.get("drug_name") or "")
        node_id = f"Drug::{drug_key}"
        subgraph_nodes[node_id] = {
            "id": node_id,
            "type": "Drug",
            "display_name": drug_key,
        }
        subgraph_edges.append({"source": risk_node_id, "relation": "risk_group_common_drug", "target": node_id})
    for patient_id in sample_patients:
        node_id = f"Patient::{patient_id}"
        subgraph_nodes[node_id] = {
            "id": node_id,
            "type": "Patient",
            "display_name": patient_id,
        }
        subgraph_edges.append({"source": node_id, "relation": "patient_has_risk_level", "target": risk_node_id})
    return {
        "status": "success",
        "query": query,
        "seed_ids": [risk_node_id],
        "seed_labels": [f"{risk_level}_risk_group"],
        "cohort_patient_count": len(patient_ids),
        "display_patient_node_count": len(sample_patients),
        "semantic_node_count": 1 + len(indicator_rows) + len(risk_rows) + len(drug_rows),
        "node_count": len(subgraph_nodes),
        "edge_count": len(subgraph_edges),
        "top_indicators": indicator_rows[:10],
        "top_risk_events": risk_rows[:10],
        "top_drugs": drug_rows[:10],
        "nodes": list(subgraph_nodes.values()),
        "edges": subgraph_edges,
        "signature": f"cohort_subgraph_{risk_level.lower()}_risk",
        "graph_scope_explanation": (
            f"该风险群体实际患者共 {len(patient_ids)} 人；页面当前展示其中 {len(sample_patients)} 个示例患者节点，"
            f"并聚焦展示最常见的 {len(indicator_rows)} 个指标、{len(risk_rows)} 个风险事件和 {len(drug_rows)} 个用药节点。"
        ),
    }


def _cohort_query_signature(query: str, disease_entity_ids: List[str]) -> str:
    query_text = query.lower()
    markers: List[str] = []
    if "合并" in query_text or "同时" in query_text or "共同" in query_text:
        markers.append("cohort")
    if "子图" in query_text or "图谱" in query_text:
        markers.append("subgraph")
    markers.extend(sorted(item.split("::", 1)[-1] for item in disease_entity_ids))
    return "_".join(markers) if markers else query_text


@_with_analysis_context
def kg_summary() -> Dict[str, Any]:
    cfg = load_server_config()
    graph_summary = _graph_summary_dict()
    kg_quality = read_optional_json("data/processed/kg_quality_report.json")
    current_metrics = load_current_metrics()
    entity_type_count = graph_summary.get("entity_type_count", {}) or {}
    relation_type_count = graph_summary.get("relation_type_count", {}) or {}
    top_degree_nodes = kg_quality.get("top_degree_nodes", []) or []
    quality_highlights = {
        "isolated_node_count": kg_quality.get("isolated_node_count"),
        "isolated_node_rate": kg_quality.get("isolated_node_rate"),
        "weakly_connected_components": kg_quality.get("weakly_connected_components"),
        "largest_component_node_count": kg_quality.get("largest_component_node_count"),
        "largest_component_ratio": kg_quality.get("largest_component_ratio"),
        "average_degree": kg_quality.get("average_degree"),
        "duplicate_edge_count": kg_quality.get("duplicate_edge_count"),
        "self_loop_count": kg_quality.get("self_loop_count"),
        "missing_display_name_count": kg_quality.get("missing_display_name_count"),
        "rejected_triples_count": kg_quality.get("rejected_triples_count"),
    }
    if not kg_quality:
        quality_highlights.update(
            {
                "average_degree": round(
                    (float(graph_summary.get("edge_count", 0) or 0) * 2.0)
                    / max(1, float(graph_summary.get("node_count", 0) or 0)),
                    4,
                ),
                "rejected_triples_count": current_metrics.get("rejected_triples_count", 0),
            }
        )
    disease_labels = sorted(
        {
            DISEASE_LABELS.get(tag.strip(), tag.strip())
            for row in fetch_rows("SELECT disease_tags FROM patient_profile WHERE disease_tags IS NOT NULL AND trim(disease_tags) != ''")
            for tag in str(row.get("disease_tags") or "").split(";")
            if tag.strip() and tag.strip().lower() != "nan"
        }
    )
    patient_count = int(current_metrics.get("patient_count") or 0)
    visit_count = int(current_metrics.get("visit_count") or 0)
    lab_result_count = int(current_metrics.get("lab_result_count") or 0)
    medication_record_count = int(current_metrics.get("medication_record_count") or 0)
    node_count = int(graph_summary["node_count"] or 0)
    edge_count = int(graph_summary["edge_count"] or 0)
    data_scale_rows = [
        {"维度": "患者数", "数值": patient_count, "说明": "来自 current_metrics / SQLite 主线指标"},
        {"维度": "随访记录数", "数值": visit_count, "说明": "来自 current_metrics / visit_record"},
        {"维度": "检验记录数", "数值": lab_result_count, "说明": "来自 current_metrics / lab_result"},
        {"维度": "用药记录数", "数值": medication_record_count, "说明": "来自 current_metrics / medication_record"},
        {"维度": "知识图谱节点数", "数值": node_count, "说明": "来自 graph_summary.json"},
        {"维度": "知识图谱边数", "数值": edge_count, "说明": "来自 graph_summary.json"},
    ]
    return {
        "status": "success",
        "data_version": current_metrics.get("data_version"),
        "patient_count": patient_count,
        "visit_count": visit_count,
        "lab_result_count": lab_result_count,
        "medication_record_count": medication_record_count,
        "node_count": node_count,
        "edge_count": edge_count,
        "table": {"rows": data_scale_rows},
        "entity_type_count": entity_type_count,
        "entity_type_total_count": len(entity_type_count),
        "relation_type_count": relation_type_count,
        "relation_type_total_count": len(relation_type_count),
        "top_entity_types": sorted(entity_type_count.items(), key=lambda item: item[1], reverse=True)[:8],
        "top_relation_types": sorted(relation_type_count.items(), key=lambda item: item[1], reverse=True)[:8],
        "quality_highlights": quality_highlights,
        "disease_type_count": len(disease_labels),
        "disease_labels": disease_labels,
        "top_degree_nodes": top_degree_nodes[:10],
        "known_issues": kg_quality.get("known_issues", []) if kg_quality else [],
        "scoring_rule": kg_quality.get("scoring_rule") if kg_quality else "DataMate-only fallback summary",
        "graph_html_path": cfg["paths"]["graph_html"],
        "graph_url": public_artifact_url(cfg, "/artifacts/graph-overview.html"),
        "graph_service_url": service_artifact_url(cfg, "/artifacts/graph-overview.html"),
        "chart_index_url": public_artifact_url(cfg, "/artifacts/charts"),
        "chart_index_service_url": service_artifact_url(cfg, "/artifacts/charts"),
        "report_url": public_artifact_url(cfg, "/artifacts/report"),
        "report_service_url": service_artifact_url(cfg, "/artifacts/report"),
        "text": (
            f"当前合成数据规模为：患者 {patient_count} 人、随访记录 {visit_count} 条、"
            f"检验记录 {lab_result_count} 条、用药记录 {medication_record_count} 条；"
            f"知识图谱节点 {node_count} 个、边 {edge_count} 条；"
            f"实体类型 {len(entity_type_count)} 种、关系类型 {len(relation_type_count)} 种。"
        ),
        "answer_guardrail": "回答图谱概览时，只能使用本工具返回的真实节点数、边数、实体类型、关系类型和疾病类型信息，禁止编造模板示例值或泛化关系名。",
        "safety_note": safety_note(cfg),
    }


@_with_analysis_context
def kg_query(query_type: str, entity_id: str) -> Dict[str, Any]:
    graph, nodes, outgoing, incoming = _load_graph()
    cfg = load_server_config()
    if entity_id not in nodes:
        return {"status": "failed", "query_type": query_type, "entity_id": entity_id, "errors": [f"Entity not found: {entity_id}"], "safety_note": safety_note(cfg)}
    if query_type == "patient_overview":
        patient_id = entity_id.split("::", 1)[-1]
        return kg_patient_path_query(patient_id)
    if query_type in {"disease_profile", "drug_profile", "indicator_profile"}:
        return kg_entity_query(f"请查询 {entity_id} 的图谱关联")
    return {"status": "failed", "query_type": query_type, "entity_id": entity_id, "errors": [f"Unsupported query_type: {query_type}"], "safety_note": safety_note(cfg)}


@_with_analysis_context
def kg_entity_query(query: str) -> Dict[str, Any]:
    cfg = load_server_config()
    graph, nodes, outgoing, incoming = _load_graph()
    entity_ids = _normalize_entity_ids(query, nodes)
    if not entity_ids:
        return {"status": "failed", "query": query, "errors": ["未识别到图谱实体。"], "safety_note": safety_note(cfg)}
    entity_id = entity_ids[0]
    entity = nodes.get(entity_id, {})
    entity_type = entity.get("type", entity_id.split("::", 1)[0])
    label = _label_for(entity_id, nodes)
    if entity_type == "Disease":
        disease_key = entity_id.split("::", 1)[-1]
        patient_ids = _query_patient_ids_by_diseases([disease_key])
        indicators = _top_indicator_rows(patient_ids)
        drugs = _top_drug_rows(patient_ids)
        risk_events = _top_risk_event_rows(patient_ids)
        table = _kg_detail_table_for_intent(query, indicators, drugs, risk_events)
        direct_edges = outgoing.get(entity_id, [])
        direct_neighbors = [
            {"relation": edge["relation"], "target_id": edge["target"], "target_label": _label_for(edge["target"], nodes)}
            for edge in direct_edges
        ]
        return {
            "status": "success",
            "query": query,
            "entity_id": entity_id,
            "entity_label": label,
            "entity_type": entity_type,
            "cohort_patient_count": len(patient_ids),
            "associated_indicators": indicators,
            "associated_drugs": drugs,
            "associated_risk_events": risk_events,
            "direct_neighbors": direct_neighbors,
            "table": table,
            "text": f"{label} 关联检查指标 {len(indicators)} 项、药物 {len(drugs)} 项、风险事件 {len(risk_events)} 项。",
            "final_answer_lock": (
                f"当前问题必须只使用 table.rows 中的 {int(table.get('row_count') or len(table.get('rows', [])))} 行作答；"
                "禁止补充任何 table.rows 之外的指标、药物或风险事件名称。"
            ),
            "answer_guardrail": (
                f"回答 {label} 的关联指标、药物或风险事件时，必须优先使用 table.rows 的真实表格。"
                "覆盖患者数是该疾病群体内至少出现过该项目的去重患者数，记录数是对应检验/用药/事件记录条数。"
                "禁止复用上一轮问题的数字，禁止编造 1200、1500 等模板值；"
                "如果 table.strict_rows_only=true，最终答案不得出现 table.rows 之外的任何名称。"
            ),
            "safety_note": safety_note(cfg),
        }
    if entity_type == "Patient":
        return kg_patient_path_query(entity_id.split("::", 1)[-1])
    if entity_type == "Indicator":
        rows = fetch_rows(
            """
            SELECT p.disease_tags, COUNT(*) AS value
            FROM patient_profile p
            JOIN lab_result l ON p.patient_id = l.patient_id
            WHERE l.item_name = ?
            GROUP BY p.disease_tags
            ORDER BY value DESC
            LIMIT 10
            """,
            [entity_id.split("::", 1)[-1]],
        )
        return {
            "status": "success",
            "query": query,
            "entity_id": entity_id,
            "entity_label": label,
            "entity_type": entity_type,
            "associated_disease_groups": rows,
            "text": f"{label} 主要出现在 {len(rows)} 类疾病组合中。",
            "safety_note": safety_note(cfg),
        }
    return {
        "status": "success",
        "query": query,
        "entity_id": entity_id,
        "entity_label": label,
        "entity_type": entity_type,
        "outgoing_relations": outgoing.get(entity_id, [])[:20],
        "incoming_relations": incoming.get(entity_id, [])[:20],
        "safety_note": safety_note(cfg),
    }


@_with_analysis_context
def kg_relation_query(query: str) -> Dict[str, Any]:
    cfg = load_server_config()
    graph, nodes, outgoing, incoming = _load_graph()
    entity_ids = _normalize_entity_ids(query, nodes)
    diseases = [item.split("::", 1)[-1] for item in entity_ids if item.startswith("Disease::")]
    intent = _kg_relation_intent(query)

    if "共同关联" in query and len(diseases) >= 2 and intent["indicator"] and not intent["drug"] and not intent["risk_event"]:
        patient_ids = _query_patient_ids_by_diseases(diseases[:2])
        indicators = _top_indicator_rows(patient_ids, limit=None)
        risk_events = _top_risk_event_rows(patient_ids, limit=12)
        drugs = _top_drug_rows(patient_ids, limit=12)
        table = _kg_detail_table_for_intent(query, indicators, drugs, risk_events)
        return {
            "status": "success",
            "query": query,
            "mode": "shared_cohort",
            "diseases": diseases[:2],
            "cohort_patient_count": len(patient_ids),
            "shared_indicators": indicators,
            "shared_risk_events": risk_events,
            "shared_drugs": drugs,
            "table": table,
            "text": (
                f"{DISEASE_LABELS.get(diseases[0], diseases[0])} 与 {DISEASE_LABELS.get(diseases[1], diseases[1])}"
                f"共同患者 {len(patient_ids)} 人；表格中的覆盖患者数均为去重患者数，记录数单独列出。"
            ),
            "final_answer_lock": (
                f"当前共同指标问题必须只使用 table.rows 中的 {int(table.get('row_count') or len(table.get('rows', [])))} 行作答；"
                "禁止漏行、截断为 6 项或补充表格外指标；覆盖患者数是共同患病队列内的去重患者数。"
            ),
            "answer_guardrail": "表格默认解释为共同患者群体内的去重覆盖患者数；不要把检验/用药/事件记录数说成患者数，不要复用单病种数字。",
            "safety_note": safety_note(cfg),
        }
    if "共同关联" in query and len(diseases) >= 2 and (intent["drug"] or intent["risk_event"]):
        return {
            "status": "failed",
            "query": query,
            "mode": "ambiguous_shared_relation",
            "diseases": diseases[:2],
            "errors": [
                "该问题混入了“共同关联”和药物/风险事件口径，容易继承上一轮共同患者上下文。请按当前用户原句调用 chroniccare_kg_entity_query，例如“糖尿病关联哪些药物？”或“高血压关联哪些风险事件？”。"
            ],
            "answer_guardrail": "不要返回共同患者 450/1985 作为药物或风险事件问题答案；必须重新按当前用户原句调用实体查询工具。",
            "safety_note": safety_note(cfg),
        }
    if "高盐" in query and ("血压" in query or "异常" in query):
        metric = fetch_one(
            """
            SELECT
              COUNT(DISTINCT s.patient_id) AS patient_count,
              COUNT(DISTINCT CASE WHEN l.abnormal_flag != 'normal' THEN s.patient_id END) AS abnormal_patient_count,
              COUNT(*) AS lab_count,
              ROUND(SUM(CASE WHEN l.abnormal_flag != 'normal' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS abnormal_rate
            FROM lifestyle_record s
            JOIN lab_result l
              ON s.patient_id = l.patient_id AND s.visit_id = l.visit_id
            WHERE s.salt_intake_level = 'high' AND l.item_name IN ('systolic_bp', 'diastolic_bp')
            """
        )
        return {
            "status": "success",
            "query": query,
            "mode": "lifestyle_relation",
            "relation": "high_salt_diet_to_bp_abnormal",
            "patient_count": metric.get("patient_count", 0),
            "abnormal_patient_count": metric.get("abnormal_patient_count", 0),
            "lab_count": metric.get("lab_count", 0),
            "abnormal_rate": metric.get("abnormal_rate"),
            "table": {
                "rows": [
                    {
                        "关系": "高盐饮食 -> 血压异常",
                        "高盐饮食且有血压检验患者数": metric.get("patient_count", 0),
                        "其中血压异常患者数": metric.get("abnormal_patient_count", 0),
                        "血压检验记录数": metric.get("lab_count", 0),
                        "血压异常记录比例": metric.get("abnormal_rate"),
                    }
                ]
            },
            "text": (
                f"高盐饮食且有同次血压检验的患者 {metric.get('patient_count', 0)} 人，其中血压异常患者 "
                f"{metric.get('abnormal_patient_count', 0)} 人；同次血压检验记录 {metric.get('lab_count', 0)} 条，"
                f"异常记录比例为 {metric.get('abnormal_rate')}。"
            ),
            "answer_guardrail": "高盐饮食且有血压检验患者数与其中血压异常患者数均为去重患者数；血压检验记录数及异常记录比例属于记录口径，不能与人数混用。",
            "safety_note": safety_note(cfg),
        }
    if diseases:
        patient_ids = _query_patient_ids_by_diseases([diseases[0]])
        top_indicators = _top_indicator_rows(patient_ids)
        top_risk_events = _top_risk_event_rows(patient_ids)
        top_drugs = _top_drug_rows(patient_ids)
        table = _indicator_table(top_indicators)
        if intent["drug"] and not intent["indicator"]:
            table = _drug_table(top_drugs)
        elif intent["risk_event"] and not intent["indicator"] and not intent["drug"]:
            table = _risk_event_table(top_risk_events)
        return {
            "status": "success",
            "query": query,
            "mode": "single_disease_relation",
            "disease": diseases[0],
            "cohort_patient_count": len(patient_ids),
            "top_indicators": top_indicators,
            "top_risk_events": top_risk_events,
            "top_drugs": top_drugs,
            "table": table,
            "text": (
                f"{DISEASE_LABELS.get(diseases[0], diseases[0])} 相关患者 {len(patient_ids)} 人，"
                f"关联指标 {len(top_indicators)} 项、风险事件 {len(top_risk_events)} 类、药物 {len(top_drugs)} 项；"
                "表格中的覆盖患者数均为去重患者数，记录数单独列出。"
            ),
            "answer_guardrail": "回答必须使用 table.rows 中的覆盖患者数；不要自行生成 1500/1200 等模板值，也不要把记录数当患者数。",
            "safety_note": safety_note(cfg),
        }
    relation_counter = Counter(edge["relation"] for edge in graph["edges"])
    return {
        "status": "success",
        "query": query,
        "mode": "graph_relation_summary",
        "relation_type_topn": [{"relation": relation, "count": count} for relation, count in relation_counter.most_common(12)],
        "safety_note": safety_note(cfg),
    }


@_with_analysis_context
def kg_patient_path_query(patient_id: str, max_hops: int = 3) -> Dict[str, Any]:
    cfg = load_server_config()
    rows = {
        "profile": fetch_one("SELECT * FROM patient_profile WHERE patient_id = ?", [patient_id]),
        "risk_events": fetch_rows("SELECT event_type, event_level, created_at FROM risk_event WHERE patient_id = ? ORDER BY created_at DESC LIMIT 20", [patient_id]),
        "followup_plans": fetch_rows("SELECT plan_type, priority, status, followup_date FROM followup_plan WHERE patient_id = ? ORDER BY followup_date DESC LIMIT 20", [patient_id]),
        "medications": fetch_rows("SELECT drug_name, drug_category, adherence, start_date, end_date FROM medication_record WHERE patient_id = ? ORDER BY start_date DESC LIMIT 20", [patient_id]),
        "visits": fetch_rows("SELECT visit_id, visit_date, chief_complaint, doctor_advice, followup_plan FROM visit_record WHERE patient_id = ? ORDER BY visit_date DESC LIMIT 20", [patient_id]),
    }
    if not rows["profile"]:
        return {"status": "failed", "patient_id": patient_id, "errors": [f"Patient not found: {patient_id}"], "safety_note": safety_note(cfg)}
    diseases = [item for item in str(rows["profile"].get("disease_tags") or "").split(";") if item and item != "nan"]
    return {
        "status": "success",
        "patient_id": patient_id,
        "max_hops": max_hops,
        "diseases": diseases,
        "risk_events": rows["risk_events"],
        "followup_plans": rows["followup_plans"],
        "medications": rows["medications"],
        "visits": rows["visits"],
        "text": f"患者 {patient_id} 关联 {len(diseases)} 个疾病标签、{len(rows['risk_events'])} 条风险事件、{len(rows['followup_plans'])} 条随访计划。",
        "safety_note": safety_note(cfg),
    }


def _node_type(node_id: str, nodes: Dict[str, Dict[str, Any]]) -> str:
    return str(nodes.get(node_id, {}).get("type", node_id.split("::", 1)[0]))


def _edge_priority(edge: Dict[str, Any], current: str, nodes: Dict[str, Dict[str, Any]]) -> Tuple[int, str, str]:
    other = edge["target"] if edge["source"] == current else edge["source"]
    node_type = _node_type(other, nodes)
    priorities = {
        "Disease": 0,
        "Indicator": 1,
        "RiskEvent": 2,
        "Drug": 3,
        "RiskFactor": 4,
        "DoctorAdvice": 5,
        "FollowupPlan": 6,
        "LifestyleRecord": 7,
        "Patient": 8,
        "RiskScore": 9,
        "Visit": 10,
        "LabResult": 11,
    }
    return priorities.get(node_type, 50), node_type, other


def _expand_frontier(
    seed_ids: List[str],
    outgoing: Dict[str, List[Dict[str, Any]]],
    incoming: Dict[str, List[Dict[str, Any]]],
    nodes: Dict[str, Dict[str, Any]],
    max_nodes: int,
) -> Tuple[Set[str], List[Dict[str, Any]]]:
    selected: Set[str] = set(seed_ids)
    selected_edges: List[Dict[str, Any]] = []
    queue = deque(seed_ids)
    type_counts: Counter[str] = Counter(_node_type(node_id, nodes) for node_id in seed_ids)
    type_limits = {"Patient": 6, "Visit": 0, "LabResult": 0}
    while queue and len(selected) < max_nodes:
        current = queue.popleft()
        ranked_edges = sorted(outgoing.get(current, []) + incoming.get(current, []), key=lambda edge: _edge_priority(edge, current, nodes))
        for edge in ranked_edges:
            source = edge["source"]
            target = edge["target"]
            if source in selected and target in selected:
                selected_edges.append(edge)
                continue
            if len(selected) >= max_nodes:
                break
            if source not in selected:
                source_type = _node_type(source, nodes)
                if source_type in type_limits and type_counts[source_type] >= type_limits[source_type]:
                    continue
                selected.add(source)
                type_counts[source_type] += 1
                queue.append(source)
            if len(selected) >= max_nodes:
                break
            if target not in selected:
                target_type = _node_type(target, nodes)
                if target_type in type_limits and type_counts[target_type] >= type_limits[target_type]:
                    continue
                selected.add(target)
                type_counts[target_type] += 1
                queue.append(target)
            selected_edges.append(edge)
    return selected, selected_edges


@_with_analysis_context
def kg_subgraph_query(query: str, max_nodes: int = 80) -> Dict[str, Any]:
    cfg = load_server_config()
    graph, nodes, outgoing, incoming = _load_graph()
    seed_ids = _normalize_entity_ids(query, nodes)
    if "高风险" in query and ("群体" in query or "患者" in query or "关系图" in query or "子图" in query):
        payload = _risk_cohort_subgraph_payload(query, "high", nodes, max_nodes=max_nodes)
        payload["safety_note"] = safety_note(cfg)
        artifact = _materialize_subgraph_artifact(query, payload, cfg)
        payload.update(artifact)
        return payload
    if "中风险" in query and ("群体" in query or "患者" in query or "关系图" in query or "子图" in query):
        payload = _risk_cohort_subgraph_payload(query, "medium", nodes, max_nodes=max_nodes)
        payload["safety_note"] = safety_note(cfg)
        artifact = _materialize_subgraph_artifact(query, payload, cfg)
        payload.update(artifact)
        return payload
    if "低风险" in query and ("群体" in query or "患者" in query or "关系图" in query or "子图" in query):
        payload = _risk_cohort_subgraph_payload(query, "low", nodes, max_nodes=max_nodes)
        payload["safety_note"] = safety_note(cfg)
        artifact = _materialize_subgraph_artifact(query, payload, cfg)
        payload.update(artifact)
        return payload
    if "高盐饮食" in query and ("血压" in query or "异常" in query or "关系图" in query):
        payload = _high_salt_bp_relation_subgraph_payload(query, nodes)
        payload["safety_note"] = safety_note(cfg)
        artifact = _materialize_subgraph_artifact(query, payload, cfg)
        payload.update(artifact)
        return payload
    if not seed_ids and has_pronoun_reference(query) and ("疾病类型" in query or "疾病分布" in query or "患病类型" in query):
        cohort_resolution = resolve_active_cohort(query)
        last_cohort = cohort_resolution.get("cohort")
        if cohort_resolution.get("status") == "needs_clarification" or not last_cohort:
            return {
                "status": "failed",
                "clarification_required": True,
                "query": query,
                "errors": [cohort_resolution.get("question") or "请先说明具体患者群体，当前没有可继承的上一轮 cohort 上下文。"],
                "context_mode": cohort_resolution.get("context_mode"),
                "conversation_id": cohort_resolution.get("conversation_id"),
                "safety_note": safety_note(cfg),
            }
        if "high_risk_followup" in str(last_cohort.get("cohort_label", "")):
            days = int(last_cohort.get("window_days") or 30)
            offset_days = max(0, days - 1)
            distribution = _cohort_disease_distribution(
                [
                    row["patient_id"]
                    for row in fetch_rows(
                        """
                        SELECT DISTINCT patient_id
                        FROM followup_plan
                        WHERE priority = 'high'
                          AND status IN ('pending', 'scheduled')
                          AND date(followup_date) BETWEEN date('now') AND date('now', '+' || ? || ' day')
                        """,
                        [offset_days],
                    )
                ]
            )
            return {
                "status": "success",
                "query": query,
                "mapped_intent": "cohort_disease_distribution",
                "distribution": distribution,
                "text": f"已继承上一轮 cohort，并返回未来 {days} 天高风险随访患者的疾病分布。",
                "cohort_context": last_cohort,
                "safety_note": safety_note(cfg),
            }
        return {
            "status": "failed",
            "query": query,
            "errors": ["当前只支持继承已保存的患者群体上下文；请明确说明具体患者群体。"],
            "cohort_context": last_cohort,
            "safety_note": safety_note(cfg),
        }
    if not seed_ids:
        payload = _synthetic_query_subgraph_payload(query)
        payload["safety_note"] = safety_note(cfg)
        return payload
    disease_seed_ids = [item for item in seed_ids if item.startswith("Disease::")]
    if disease_seed_ids:
        payload = _cohort_subgraph_payload(query, disease_seed_ids, nodes, max_nodes=max_nodes)
        payload["safety_note"] = safety_note(cfg)
        artifact = _materialize_subgraph_artifact(query, payload, cfg)
        payload.update(artifact)
        return payload
    selected_ids, _ = _expand_frontier(seed_ids, outgoing, incoming, nodes, max_nodes=max_nodes)
    selected_nodes = [nodes[node_id] for node_id in selected_ids if node_id in nodes]
    unique_edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for edge in graph["edges"]:
        if edge["source"] in selected_ids and edge["target"] in selected_ids:
            unique_edges[(edge["source"], edge["relation"], edge["target"])] = edge
    selected_edges = list(unique_edges.values())
    payload = {
        "status": "success",
        "query": query,
        "seed_ids": seed_ids,
        "node_count": len(selected_nodes),
        "edge_count": len(selected_edges),
        "nodes": selected_nodes,
        "edges": selected_edges,
        "safety_note": safety_note(cfg),
    }
    artifact = _materialize_subgraph_artifact(query, payload, cfg)
    payload.update(artifact)
    return payload


def _materialize_subgraph_artifact(query: str, payload: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    signature = str(payload.get("signature") or "").strip() or None
    if signature is None and payload.get("seed_ids"):
        disease_seed_ids = [item for item in payload.get("seed_ids", []) if str(item).startswith("Disease::")]
        if disease_seed_ids:
            signature = _cohort_query_signature(query, disease_seed_ids)
    subgraph_id = _subgraph_id_from_query(query, signature=signature)
    subgraph_dir = _subgraph_output_dir(subgraph_id)
    html_path = subgraph_dir / f"{subgraph_id}.html"
    render_nodes = list(payload["nodes"])
    render_nodes.append({"id": "__meta__cohort_patient_total", "type": "Meta", "value": payload.get("cohort_patient_count", 0)})
    render_nodes.append({"id": "__meta__semantic_node_total", "type": "Meta", "value": payload.get("semantic_node_count", 0)})
    _render_subgraph_html(html_path, query, render_nodes, payload["edges"], meta=payload)
    meta_path = subgraph_dir / f"{subgraph_id}.json"
    _safe_replace(meta_path)
    meta_path.write_text(
        json.dumps(
            {
                "query": query,
                "subgraph_id": subgraph_id,
                "seed_labels": payload.get("seed_labels", []),
                "cohort_patient_count": payload.get("cohort_patient_count", 0),
                "display_patient_node_count": payload.get("display_patient_node_count", 0),
                "semantic_node_count": payload.get("semantic_node_count", 0),
                "node_count": payload.get("node_count", 0),
                "edge_count": payload.get("edge_count", 0),
                "top_indicators": payload.get("top_indicators", []),
                "top_risk_events": payload.get("top_risk_events", []),
                "top_drugs": payload.get("top_drugs", []),
                "graph_scope_explanation": payload.get("graph_scope_explanation"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    preview_svg = _subgraph_preview_svg(query, payload)
    preview_path = subgraph_dir / f"{subgraph_id}.svg"
    _safe_replace(preview_path)
    preview_path.write_text(preview_svg, encoding="utf-8")
    preview_png_path = subgraph_dir / f"{subgraph_id}.png"
    _safe_replace(preview_png_path)
    _write_subgraph_preview_png(preview_png_path, query, payload)
    html_route = f"/artifacts/subgraphs/{subgraph_id}.html"
    json_route = f"/artifacts/subgraphs/{subgraph_id}.json"
    preview_svg_route = f"/artifacts/subgraphs/{subgraph_id}.svg"
    preview_png_route = f"/artifacts/subgraphs/{subgraph_id}.png"
    preview_route = preview_png_route if preview_png_path.exists() else preview_svg_route
    html_url = public_artifact_url(cfg, html_route)
    html_service_url = service_artifact_url(cfg, html_route)
    preview_url = public_artifact_url(cfg, preview_route)
    preview_service_url = service_artifact_url(cfg, preview_route)
    return {
        "subgraph_id": subgraph_id,
        "html_path": relative_to_project(html_path),
        "graph_path": relative_to_project(html_path),
        "json_path": relative_to_project(meta_path),
        "preview_path": relative_to_project(preview_path),
        "preview_png_path": relative_to_project(preview_png_path),
        "html_route_path": html_route,
        "graph_route_path": html_route,
        "json_route_path": json_route,
        "html_url": html_url,
        "graph_url": html_url,
        "service_html_url": html_service_url,
        "graph_service_url": html_service_url,
        "json_url": public_artifact_url(cfg, json_route),
        "json_service_url": service_artifact_url(cfg, json_route),
        "preview_route_path": preview_route,
        "preview_url": preview_url,
        "preview_service_url": preview_service_url,
        "preview_svg_route_path": preview_svg_route,
        "preview_svg_url": public_artifact_url(cfg, preview_svg_route),
        "preview_svg_service_url": service_artifact_url(cfg, preview_svg_route),
        "preview_png_route_path": preview_png_route,
        "preview_png_url": public_artifact_url(cfg, preview_png_route),
        "preview_png_service_url": service_artifact_url(cfg, preview_png_route),
    }


def _write_subgraph_preview_png(path: Path, query: str, payload: Dict[str, Any]) -> None:
    if Image is None or ImageDraw is None or ImageFont is None:
        return
    seed_labels = [str(item).strip() for item in payload.get("seed_labels", []) if str(item).strip()]
    subject = "、".join(seed_labels) if seed_labels else _topic_from_query(query)
    subject = (subject or "当前问题")[:28]
    cohort_count = int(payload.get("cohort_patient_count", 0) or 0)
    display_count = int(payload.get("display_patient_node_count", 0) or 0)
    semantic_count = int(payload.get("semantic_node_count", 0) or 0)
    node_count = int(payload.get("node_count", 0) or 0)
    edge_count = int(payload.get("edge_count", 0) or 0)
    image = Image.new("RGB", (1280, 720), "#f4f8fd")
    draw = ImageDraw.Draw(image)
    def load_font(size: int):
        for font_path in [
            "/app/assets/fonts/DroidSansFallback.ttf",
            "/usr/share/fonts/google-droid-fonts/DroidSansFallback.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            try:
                if Path(font_path).exists():
                    return ImageFont.truetype(font_path, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def load_latin_font(size: int):
        for font_path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/app/assets/fonts/DejaVuSans.ttf",
        ]:
            try:
                if Path(font_path).exists():
                    return ImageFont.truetype(font_path, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def draw_mixed_text(x: int, y: int, text: Any, *, cjk_font: Any, latin_font: Any, fill: str) -> None:
        cursor = x
        for chunk in re.findall(r"[\x00-\x7f]+|[^\x00-\x7f]+", str(text)):
            font = latin_font if chunk and all(ord(ch) < 128 for ch in chunk) else cjk_font
            draw.text((cursor, y), chunk, fill=fill, font=font)
            try:
                bbox = draw.textbbox((cursor, y), chunk, font=font)
                cursor += bbox[2] - bbox[0]
            except Exception:
                cursor += len(chunk) * 12

    title_font = load_font(32)
    body_font = load_font(22)
    small_font = load_font(18)
    latin_title_font = load_latin_font(32)
    latin_body_font = load_latin_font(22)
    latin_small_font = load_latin_font(18)
    draw.rounded_rectangle((34, 34, 1246, 686), radius=24, fill="#ffffff", outline="#d9e2ec", width=2)
    draw_mixed_text(70, 70, f"{subject}知识图谱子图", cjk_font=title_font, latin_font=latin_title_font, fill="#102a43")
    draw_mixed_text(70, 118, "当前问题实时生成的局部图谱预览；完整结构请打开 HTML 图谱页面。", cjk_font=small_font, latin_font=latin_small_font, fill="#486581")
    cards = [
        ("群体患者", cohort_count),
        ("示例节点", display_count),
        ("语义节点", semantic_count),
        ("节点/关系", f"{node_count} / {edge_count}"),
    ]
    for idx, (label, value) in enumerate(cards):
        x1 = 70 + idx * 285
        draw.rounded_rectangle((x1, 175, x1 + 250, 295), radius=18, fill="#f8fbff", outline="#d9e2ec", width=2)
        draw_mixed_text(x1 + 25, 205, label, cjk_font=small_font, latin_font=latin_small_font, fill="#486581")
        draw_mixed_text(x1 + 25, 245, str(value), cjk_font=body_font, latin_font=latin_body_font, fill="#102a43")
    for x, y, color, label in [
        (180, 470, "#2457a5", "患者"),
        (505, 470, "#d1495b", "疾病/群体"),
        (815, 425, "#2e7d32", "检查指标"),
        (815, 535, "#e07a1f", "风险事件"),
        (1095, 480, "#7b5ea7", "药物"),
    ]:
        draw.ellipse((x - 24, y - 24, x + 24, y + 24), fill=color)
        draw_mixed_text(x - 45, y + 42, label, cjk_font=small_font, latin_font=latin_small_font, fill="#102a43")
    for xy in [((204, 470), (473, 470)), ((537, 455), (791, 425)), ((537, 485), (791, 535)), ((537, 470), (1068, 480))]:
        draw.line(xy, fill="#c7d0d9", width=4)
    image.save(path, format="PNG")


def _subgraph_preview_svg(query: str, payload: Dict[str, Any]) -> str:
    seed_labels = [str(item).strip() for item in payload.get("seed_labels", []) if str(item).strip()]
    subject = "、".join(seed_labels) if seed_labels else _topic_from_query(query)
    subject = html.escape(subject[:60] or "当前问题")
    scope = html.escape(str(payload.get("graph_scope_explanation") or payload.get("summary_text") or "当前问题驱动的局部知识图谱子图。")[:110])
    cohort_count = int(payload.get("cohort_patient_count", 0) or 0)
    display_count = int(payload.get("display_patient_node_count", 0) or 0)
    semantic_count = int(payload.get("semantic_node_count", 0) or 0)
    node_count = int(payload.get("node_count", 0) or 0)
    edge_count = int(payload.get("edge_count", 0) or 0)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#f7fbff"/>
      <stop offset="100%" stop-color="#edf5ff"/>
    </linearGradient>
  </defs>
  <rect width="1280" height="720" rx="28" fill="url(#bg)"/>
  <rect x="34" y="34" width="1212" height="652" rx="24" fill="#ffffff" stroke="#d9e2ec"/>
  <text x="70" y="105" font-size="42" font-weight="700" fill="#102a43">{subject}知识图谱子图</text>
  <text x="70" y="150" font-size="22" fill="#486581">当前问题实时生成的局部知识图谱预览；完整交互图谱见下方打开入口。</text>
  <rect x="70" y="195" width="250" height="120" rx="18" fill="#f8fbff" stroke="#d9e2ec"/>
  <text x="95" y="235" font-size="20" fill="#486581">群体患者总数</text>
  <text x="95" y="285" font-size="44" font-weight="700" fill="#102a43">{cohort_count}</text>
  <rect x="345" y="195" width="250" height="120" rx="18" fill="#f8fbff" stroke="#d9e2ec"/>
  <text x="370" y="235" font-size="20" fill="#486581">示例患者节点</text>
  <text x="370" y="285" font-size="44" font-weight="700" fill="#102a43">{display_count}</text>
  <rect x="620" y="195" width="250" height="120" rx="18" fill="#f8fbff" stroke="#d9e2ec"/>
  <text x="645" y="235" font-size="20" fill="#486581">当前语义节点</text>
  <text x="645" y="285" font-size="44" font-weight="700" fill="#102a43">{semantic_count}</text>
  <rect x="895" y="195" width="300" height="120" rx="18" fill="#f8fbff" stroke="#d9e2ec"/>
  <text x="920" y="235" font-size="20" fill="#486581">本次子图规模</text>
  <text x="920" y="285" font-size="34" font-weight="700" fill="#102a43">节点 {node_count} / 关系 {edge_count}</text>
  <circle cx="180" cy="470" r="28" fill="#2457a5"/>
  <text x="145" y="525" font-size="20" fill="#102a43">示例患者</text>
  <line x1="208" y1="470" x2="468" y2="470" stroke="#c7d0d9" stroke-width="4"/>
  <circle cx="505" cy="470" r="32" fill="#d1495b"/>
  <text x="445" y="525" font-size="20" fill="#102a43">核心疾病/群体</text>
  <line x1="537" y1="455" x2="790" y2="425" stroke="#c7d0d9" stroke-width="4"/>
  <line x1="537" y1="485" x2="790" y2="535" stroke="#c7d0d9" stroke-width="4"/>
  <circle cx="815" cy="425" r="24" fill="#2e7d32"/>
  <text x="780" y="475" font-size="20" fill="#102a43">检查指标</text>
  <circle cx="815" cy="535" r="24" fill="#e07a1f"/>
  <text x="780" y="585" font-size="20" fill="#102a43">风险事件</text>
  <line x1="839" y1="480" x2="1068" y2="480" stroke="#c7d0d9" stroke-width="4"/>
  <circle cx="1095" cy="480" r="24" fill="#7b5ea7"/>
  <text x="1060" y="530" font-size="20" fill="#102a43">常用药物</text>
  <foreignObject x="70" y="620" width="1140" height="52">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-size:18px;color:#486581;line-height:1.5;">{scope}</div>
  </foreignObject>
</svg>"""
    return svg


def _subgraph_preview_data_uri(query: str, payload: Dict[str, Any]) -> str:
    svg = _subgraph_preview_svg(query, payload)
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _subgraph_id_from_query(query: str, signature: str | None = None) -> str:
    slug_source = signature or query.lower()
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug_source).strip("_")
    if not slug:
        digest = hashlib.md5(query.encode("utf-8")).hexdigest()[:10]
        slug = f"graph_query_{digest}"
    return f"subgraph_{slug[:64]}"


def _render_subgraph_html(path: Path, query: str, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], meta: Dict[str, Any] | None = None) -> None:
    meta = meta or {}
    width = 2380
    height = 1380
    visible_nodes = [node for node in nodes if str(node.get("type", "")) != "Meta"]
    positions: Dict[str, Tuple[float, float]] = {}
    nodes_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for node in visible_nodes:
        nodes_by_type[str(node.get("type", "Unknown"))].append(node)

    patient_nodes = nodes_by_type.get("Patient", [])
    disease_nodes = nodes_by_type.get("Disease", [])
    indicator_nodes = nodes_by_type.get("Indicator", [])
    risk_nodes = nodes_by_type.get("RiskEvent", [])
    drug_nodes = nodes_by_type.get("Drug", [])
    followup_nodes = nodes_by_type.get("FollowupPlan", [])
    lifestyle_nodes = nodes_by_type.get("LifestyleRecord", [])
    advice_nodes = nodes_by_type.get("DoctorAdvice", [])
    score_nodes = nodes_by_type.get("RiskScore", [])
    other_nodes = [
        node
        for node_type, bucket in nodes_by_type.items()
        if node_type not in {"Patient", "Disease", "Indicator", "RiskEvent", "Drug", "FollowupPlan", "LifestyleRecord", "DoctorAdvice", "RiskScore"}
        for node in bucket
    ]

    def place_vertical(bucket: List[Dict[str, Any]], x: float, y_start: float, y_end: float) -> None:
        if not bucket:
            return
        step = (y_end - y_start) / max(1, len(bucket) - 1) if len(bucket) > 1 else 0
        for index, node in enumerate(bucket):
            y = y_start + index * step if len(bucket) > 1 else (y_start + y_end) / 2
            positions[node["id"]] = (x, y)

    def place_grid(bucket: List[Dict[str, Any]], x_start: float, x_end: float, y_start: float, y_end: float, columns: int) -> None:
        if not bucket:
            return
        columns = max(1, columns)
        rows = math.ceil(len(bucket) / columns)
        x_step = (x_end - x_start) / max(1, columns - 1) if columns > 1 else 0
        y_step = (y_end - y_start) / max(1, rows - 1) if rows > 1 else 0
        for index, node in enumerate(bucket):
            row = index // columns
            col = index % columns
            x = x_start + col * x_step if columns > 1 else (x_start + x_end) / 2
            y = y_start + row * y_step if rows > 1 else (y_start + y_end) / 2
            positions[node["id"]] = (x, y)

    place_grid(patient_nodes, 110, 980, 140, height - 140, columns=6)
    place_vertical(disease_nodes, 1180, 430, 930)
    place_vertical(score_nodes, 1360, 500, 860)
    place_grid(indicator_nodes, 1580, 1930, 150, 720, columns=2)
    place_grid(risk_nodes, 1580, 1930, 790, 1210, columns=2)
    place_vertical(drug_nodes, 2180, 220, 1140)
    place_vertical(followup_nodes, 2060, 980, 1160)
    place_vertical(lifestyle_nodes, 2060, 1180, 1260)
    place_vertical(advice_nodes, 2060, 1280, 1340)
    place_vertical(other_nodes, 1460, 1020, 1320)

    edge_svg: List[str] = []
    for edge in edges:
        if edge["source"] not in positions or edge["target"] not in positions:
            continue
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        edge_svg.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='#c7d0d9' stroke-width='1.2' />"
        )

    node_lookup = {item["id"]: item for item in visible_nodes}
    node_svg: List[str] = []
    for node in visible_nodes:
        x, y = positions[node["id"]]
        node_type = str(node.get("type", "Unknown"))
        fill = TYPE_COLORS.get(node_type, "#455a64")
        label = html.escape(_label_for(node["id"], node_lookup))
        title = html.escape(node["id"])
        label_y = y + 34
        font_size = 11 if node_type == "Patient" else 10
        radius = 18 if node_type in {"Disease", "RiskScore"} else 16
        node_svg.append(
            f"<g><title>{title}</title><circle cx='{x:.1f}' cy='{y:.1f}' r='{radius}' fill='{fill}' />"
            f"<text x='{x:.1f}' y='{label_y:.1f}' text-anchor='middle' font-size='{font_size}' fill='#102a43'>{label}</text></g>"
        )

    legend_html = "".join(
        [
            f"<div style='display:flex;align-items:center;gap:8px;'><span style='display:inline-block;width:12px;height:12px;background:{TYPE_COLORS.get(item, '#455a64')};border-radius:50%;'></span>{TYPE_DISPLAY_NAMES.get(item, item)}</div>"
            for item in LEGEND_TYPES
        ]
    )
    table_rows = "".join(
        [
            f"<tr><td>{html.escape(node['id'])}</td><td>{html.escape(TYPE_DISPLAY_NAMES.get(str(node.get('type', '')), str(node.get('type', ''))))}</td><td>{html.escape(_label_for(node['id'], node_lookup))}</td></tr>"
            for node in visible_nodes
        ]
    )
    edge_rows = "".join(
        [
            f"<tr><td>{html.escape(edge['source'])}</td><td>{html.escape(edge['relation'])}</td><td>{html.escape(edge['target'])}</td></tr>"
            for edge in edges[:200]
        ]
    )

    payload_patient_total = 0
    payload_patient_display = len(patient_nodes)
    payload_semantic_total = len(visible_nodes) - len(patient_nodes)
    for node in nodes:
        if node.get("id") == "__meta__cohort_patient_total":
            payload_patient_total = int(node.get("value") or 0)
        if node.get("id") == "__meta__semantic_node_total":
            payload_semantic_total = int(node.get("value") or payload_semantic_total)

    seed_labels = [str(item).strip() for item in meta.get("seed_labels", []) if str(item).strip()]
    top_indicators = meta.get("top_indicators") or []
    top_risk_events = meta.get("top_risk_events") or []
    top_drugs = meta.get("top_drugs") or []
    subject_title = "、".join(seed_labels) if seed_labels else "当前问题相关群体"
    page_title = f"{subject_title}知识图谱子图"

    real_total_text = (
        f"<p><strong>群体患者总人数：</strong>{payload_patient_total} 人；"
        f"<strong>当前展示的示例患者：</strong>{payload_patient_display} 人；"
        f"<strong>当前语义节点数：</strong>{payload_semantic_total} 个。</p>"
        if payload_patient_total
        else ""
    )
    graph_scope_note = (
        f"<p><strong>当前图谱主题：</strong>{html.escape(subject_title)}</p>"
        "<p><strong>阅读方式：</strong>左侧蓝色节点是示例患者，中间红色节点是本次问题的核心疾病或核心群体，"
        "右侧绿色 / 橙色 / 紫色节点分别表示高频检查指标、常见风险事件和常用药物。当前页面不是全局图谱全量展开，而是围绕本次问题抽取的局部关系图。</p>"
    )
    cohort_summary = (
        f"<p><strong>当前页面结构：</strong>疾病 {len(disease_nodes)} 个，风险分层 {len(score_nodes)} 个，指标 {len(indicator_nodes)} 个，风险事件 {len(risk_nodes)} 个，药物 {len(drug_nodes)} 个，示例患者 {len(patient_nodes)} 个。</p>"
    )

    def _tag_list(items: List[Dict[str, Any]], key: str, limit: int = 8) -> str:
        labels = []
        for item in items[:limit]:
            value = str(item.get(key) or "").strip()
            count = int(item.get("value", 0) or 0)
            if value:
                labels.append(
                    f"<span style='display:inline-block;padding:6px 10px;margin:4px;border-radius:999px;background:#eef4ff;color:#1f3c88;font-size:13px;'>{html.escape(value)} · {count}</span>"
                )
        return "".join(labels) if labels else "<span style='color:#829ab1;'>暂无</span>"

    content = f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(page_title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f9fb; color: #102a43; }}
    .panel {{ background: white; border-radius: 16px; padding: 20px; margin-bottom: 20px; box-shadow: 0 12px 30px rgba(16,42,67,0.08); }}
    .legend {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }}
    .cards {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .card {{ background:#f8fbff; border:1px solid #d9e2ec; border-radius:14px; padding:14px 16px; }}
    .card .label {{ font-size:13px; color:#486581; margin-bottom:6px; }}
    .card .value {{ font-size:28px; font-weight:700; color:#102a43; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e6edf3; text-align: left; padding: 8px 10px; font-size: 13px; }}
    .graph-wrap {{ overflow-x: auto; overflow-y: hidden; padding-bottom: 10px; }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>{html.escape(page_title)}</h1>
    <p><strong>查询：</strong>{html.escape(query)}</p>
    <p>这不是全局知识图谱，而是围绕当前问题实时生成的局部子图。</p>
    {real_total_text}
    {graph_scope_note}
    <div class="cards">
      <div class="card"><div class="label">核心主题</div><div class="value" style="font-size:22px;">{html.escape(subject_title)}</div></div>
      <div class="card"><div class="label">群体患者总数</div><div class="value">{payload_patient_total}</div></div>
      <div class="card"><div class="label">示例患者节点</div><div class="value">{payload_patient_display}</div></div>
      <div class="card"><div class="label">当前语义节点</div><div class="value">{payload_semantic_total}</div></div>
    </div>
    <div style="margin-top:16px;">{cohort_summary}</div>
    <p><strong>缩写说明：</strong>`Pxxxx` 表示患者编号，页面中会显示为“患者 Pxxxx”；不同颜色对应不同节点类型。</p>
    <p><strong>交互说明：</strong>当前页面是固定布局的 HTML/SVG 图谱，可通过页面滚动和横向滚动查看完整结构；暂不支持拖拽节点。</p>
    <p>本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。</p>
  </div>
  <div class="panel">
    <h2>结构解读</h2>
    <p>从左到右阅读即可：</p>
    <ol>
      <li>左侧蓝色节点：该群体中的示例患者。</li>
      <li>中间红色节点：本次问题的核心疾病或核心群体，本页应为“{html.escape(subject_title)}”。</li>
      <li>右侧绿色 / 橙色 / 紫色节点：分别表示高频检查指标、常见风险事件、常用药物。</li>
    </ol>
    <p><strong>高频检查指标：</strong>{_tag_list(top_indicators, 'indicator')}</p>
    <p><strong>常见风险事件：</strong>{_tag_list(top_risk_events, 'event_type')}</p>
    <p><strong>常用药物：</strong>{_tag_list(top_drugs, 'drug_name')}</p>
  </div>
  <div class="panel">
    <h2>图例</h2>
    <div class="legend">{legend_html}</div>
  </div>
  <div class="panel">
    <h2>子图视图</h2>
    <p style="color:#486581;margin-top:-4px;">固定布局图谱，可横向滚动查看；当前版本不支持节点拖拽。</p>
    <div class="graph-wrap">
      <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="max-width:none;height:auto;background:#fcfdff;border:1px solid #e6edf3;border-radius:12px;">
        {''.join(edge_svg)}
        {''.join(node_svg)}
      </svg>
    </div>
  </div>
  <div class="panel">
    <h2>节点说明</h2>
    <table><thead><tr><th>Node ID</th><th>Type</th><th>Label</th></tr></thead><tbody>{table_rows}</tbody></table>
  </div>
  <div class="panel">
    <h2>关系说明</h2>
    <table><thead><tr><th>Source</th><th>Relation</th><th>Target</th></tr></thead><tbody>{edge_rows}</tbody></table>
  </div>
</body>
</html>"""
    _safe_replace(path)
    path.write_text(content, encoding="utf-8")


@_with_analysis_context
def kg_subgraph_render(query: str, max_nodes: int = 80) -> Dict[str, Any]:
    cfg = load_server_config()
    payload = kg_subgraph_query(query, max_nodes=max_nodes)
    if payload.get("status") != "success" or "nodes" not in payload:
        return payload
    artifact = _materialize_subgraph_artifact(query, payload, cfg)
    return {
        "status": "success",
        "query": query,
        "subgraph_id": artifact["subgraph_id"],
        "node_count": payload["node_count"],
        "edge_count": payload["edge_count"],
        "seed_labels": payload.get("seed_labels", []),
        "cohort_patient_count": payload.get("cohort_patient_count"),
        "display_patient_node_count": payload.get("display_patient_node_count"),
        "semantic_node_count": payload.get("semantic_node_count"),
        "top_indicators": payload.get("top_indicators", []),
        "top_risk_events": payload.get("top_risk_events", []),
        "top_drugs": payload.get("top_drugs", []),
        "html_path": artifact["html_path"],
        "graph_path": artifact["graph_path"],
        "json_path": artifact["json_path"],
        "preview_path": artifact["preview_path"],
        "preview_png_path": artifact["preview_png_path"],
        "html_route_path": artifact["html_route_path"],
        "graph_route_path": artifact["graph_route_path"],
        "json_route_path": artifact["json_route_path"],
        "html_url": artifact["html_url"],
        "graph_url": artifact["graph_url"],
        "service_html_url": artifact["service_html_url"],
        "graph_service_url": artifact["graph_service_url"],
        "json_url": artifact["json_url"],
        "json_service_url": artifact["json_service_url"],
        "preview_route_path": artifact["preview_route_path"],
        "preview_url": artifact["preview_url"],
        "preview_service_url": artifact["preview_service_url"],
        "preview_svg_route_path": artifact["preview_svg_route_path"],
        "preview_svg_url": artifact["preview_svg_url"],
        "preview_svg_service_url": artifact["preview_svg_service_url"],
        "preview_png_route_path": artifact["preview_png_route_path"],
        "preview_png_url": artifact["preview_png_url"],
        "preview_png_service_url": artifact["preview_png_service_url"],
        "explanation": "已生成可通过 HTTP 访问的问题驱动子图，可用于 Nexent 或浏览器直接查看。",
        "graph_scope_explanation": payload.get("graph_scope_explanation"),
        "safety_note": safety_note(cfg),
    }
