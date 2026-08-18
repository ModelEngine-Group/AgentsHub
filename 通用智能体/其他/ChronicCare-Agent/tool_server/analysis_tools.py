from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None
    ImageDraw = None
    ImageFont = None

from analysis.open_nl2sql.intent_router import route_intent
from analysis.open_nl2sql.schema_linker import build_schema_links
from analysis.open_nl2sql.schema_registry import get_schema_registry
from analysis.open_nl2sql.sql_candidate_builder import build_sql_candidate
from analysis.open_nl2sql.sql_explainer import build_sql_response
from analysis.open_nl2sql.sql_guard import validate_sql
from analysis.open_nl2sql.synonym_rewrite import extract_future_window_days, rewrite_question
from analysis.open_sql.open_sql_service import open_sql_query as run_open_sql_query
from analysis.query_planner import plan_query
from orchestration.intent_router import route_intent as route_agent_intent
from orchestration.question_pipeline import run_question_pipeline
from runtime_common.analysis_context import AnalysisContext, attach_analysis_context
from runtime_common.cohort_context import (
    get_current_conversation_id,
    has_pronoun_reference,
    load_last_cohort,
    resolve_active_cohort,
    save_conversation_cohort,
    save_last_cohort,
)
from runtime_common.common import read_json, resolve_path
from tool_server.pipeline_tools import (
    datamate_pipeline_report_by_run,
    datamate_pipeline_status_by_run,
    run_datamate_pipeline,
)
from tool_server.utils import (
    artifact_exists_for_route,
    artifact_route_path,
    ensure_parent,
    fetch_one,
    fetch_rows,
    load_current_metrics,
    load_server_config,
    public_artifact_url,
    safety_note,
    service_artifact_url,
)

SPECIAL_CANONICALS = {
    "future_30d_high_risk_followup_disease_distribution": "未来 30 天需要随访的高风险患者的疾病类型分布是什么？",
    "high_salt_bp_abnormal_rate": "高盐饮食患者的血压异常比例是多少？",
    "hypertension_diabetes_multi_indicator": "高血压合并糖尿病患者的 HbA1c、血压和 LDL-C 异常情况如何？",
    "future_followup_chart_bundle": "根据未来随访人数，绘制折线图，饼状图",
}

QUESTION_METADATA = {
    "高血压合并糖尿病患者的平均 HbA1c 是多少？": {
        "intent": "cohort_metric",
        "chart_type": "metric_card",
        "metric_name": "avg_hba1c",
        "metric_label": "平均 HbA1c",
        "unit": "%",
    },
    "糖尿病患者的空腹血糖平均值是多少？": {
        "intent": "cohort_metric",
        "chart_type": "metric_card",
        "metric_name": "avg_fasting_glucose",
        "metric_label": "平均空腹血糖",
        "unit": "mmol/L",
    },
    "高脂血症患者的 LDL-C 异常比例是多少？": {
        "intent": "cohort_ratio",
        "chart_type": "metric_card",
        "metric_name": "ldl_abnormal_rate",
        "metric_label": "LDL-C 异常比例",
        "unit": "ratio",
    },
    "BMI 超标患者有多少人？": {
        "intent": "cohort_count",
        "chart_type": "metric_card",
        "metric_name": "overweight_patient_count",
        "metric_label": "BMI 超标患者数",
        "unit": "人",
    },
    "不同风险等级患者的 HbA1c 平均值是多少？": {
        "intent": "risk_group_metric",
        "chart_type": "table",
        "metric_name": "avg_hba1c",
        "metric_label": "不同风险等级 HbA1c 平均值",
        "unit": "%",
    },
    "最近 6 个月 HbA1c 异常人数趋势如何？": {
        "intent": "trend_query",
        "chart_type": "line",
        "metric_name": "abnormal_patient_count",
        "metric_label": "HbA1c 异常人数",
        "unit": "人",
    },
    "最近 6 个月血压异常人数趋势如何？": {
        "intent": "trend_query",
        "chart_type": "line",
        "metric_name": "abnormal_patient_count",
        "metric_label": "血压异常人数",
        "unit": "人",
    },
    "高血压患者最近半年的血压趋势如何？": {
        "intent": "trend_query",
        "chart_type": "line_bundle",
        "metric_name": "blood_pressure_trend",
        "metric_label": "血压趋势",
        "unit": "mmHg",
    },
    "糖尿病患者最近 3 个月 HbA1c 异常比例是多少？": {
        "intent": "cohort_ratio",
        "chart_type": "metric_card",
        "metric_name": "hba1c_abnormal_rate_3m",
        "metric_label": "近 3 个月 HbA1c 异常比例",
        "unit": "ratio",
    },
}

FOLLOWUP_DYNAMIC_VISUAL_HINTS = ("图", "图表", "可视化", "趋势", "折线图", "饼图", "饼状图")
FOLLOWUP_DYNAMIC_COUNT_HINTS = ("多少", "数量", "人数", "统计")
SUBGRAPH_REQUEST_HINTS = ("子图", "图谱", "关系图", "关联图", "知识图谱")
GLOBAL_GRAPH_REQUEST_HINTS = (
    "图谱在哪里看",
    "打开知识图谱",
    "打开图谱",
    "图谱入口",
    "知识图谱入口",
    "知识图谱",
    "图谱总览",
    "图谱概览",
    "看知识图谱",
    "看图谱",
)

FIELD_LABELS = {
    "avg_hba1c": "平均 HbA1c",
    "avg_fasting_glucose": "平均空腹血糖",
    "overweight_patient_count": "BMI 超标患者数",
    "avg_ldl_c": "平均 LDL-C",
    "patient_count": "患者数",
    "row_count": "记录数",
    "hba1c_abnormal_rate": "HbA1c 异常率",
    "bp_abnormal_rate": "血压异常率",
    "ldl_abnormal_rate": "LDL-C 异常率",
    "abnormal_patient_count": "异常患者数",
    "risk_level": "风险等级",
    "month": "月份",
    "avg_systolic_bp": "平均收缩压",
    "avg_diastolic_bp": "平均舒张压",
    "gender": "性别",
    "age": "年龄",
    "name": "姓名",
    "patient_id": "患者编号",
    "disease_tags": "疾病标签",
}

DISEASE_LABELS = {
    "hypertension": "高血压",
    "diabetes": "糖尿病",
    "hyperlipidemia": "高脂血症",
    "obesity": "肥胖",
    "hyperuricemia": "高尿酸血症",
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

GENDER_LABELS = {
    "male": "男",
    "female": "女",
}

PLANNER_LOG_DIR = "outputs/planner_logs"
OPEN_NL2SQL_DIR = "outputs/open_nl2sql"
RUNTIME_CHART_DIR = "outputs/runtime_generated/charts"
RUNTIME_GRAPH_ANALYSIS_DIR = "outputs/runtime_generated/graph_driven_analysis"
LOCAL_CHART_DIR = "outputs/local_runtime/charts"
LOCAL_GRAPH_ANALYSIS_DIR = "outputs/local_runtime/graph_driven_analysis"


def _question_slug(question: str) -> str:
    digest = hashlib.md5(question.encode("utf-8")).hexdigest()[:12]
    return digest


def _safe_replace(path: Path) -> Path:
    if path.exists():
        path.unlink()
    return path


def _first_writable_output_dir(candidates: List[str], filenames: List[str]) -> Tuple[str, Path]:
    last_error: Exception | None = None
    for base in candidates:
        try:
            output_dir = resolve_path(base)
            output_dir.mkdir(parents=True, exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                continue
            paths = [output_dir / filename for filename in filenames]
            if any(path.exists() and not os.access(path, os.W_OK) for path in paths):
                continue
            return base, output_dir
        except OSError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise PermissionError(f"No writable output directory found for {filenames}")


def _normalize_controlled_question(question: str) -> str:
    normalized = " ".join(str(question or "").strip().split())
    compact = normalized.replace(" ", "")
    if "高血压" in compact and "糖尿病" in compact and "hba1c" in compact.lower() and any(token in compact for token in ("平均", "均值")):
        return "高血压合并糖尿病患者的平均 HbA1c 是多少？"
    if "糖尿病" in compact and "空腹血糖" in compact and any(token in compact for token in ("平均", "均值")):
        return "糖尿病患者的空腹血糖平均值是多少？"
    if "高脂血症" in compact and ("ldl-c" in compact.lower() or "ldl" in compact.lower() or "低密度脂蛋白" in compact) and "异常比例" in compact:
        return "高脂血症患者的 LDL-C 异常比例是多少？"
    if "bmi" in compact.lower() and "超标" in compact and any(token in compact for token in ("多少", "人数", "数量")):
        return "BMI 超标患者有多少人？"
    if "不同风险等级" in compact and "hba1c" in compact.lower() and any(token in compact for token in ("平均", "均值")):
        return "不同风险等级患者的 HbA1c 平均值是多少？"
    if "最近6个月" in compact and "hba1c" in compact.lower() and "异常人数趋势" in compact:
        return "最近 6 个月 HbA1c 异常人数趋势如何？"
    if "最近6个月" in compact and "血压异常人数趋势" in compact:
        return "最近 6 个月血压异常人数趋势如何？"
    if "高血压" in compact and "最近半年" in compact and "血压趋势" in compact:
        return "高血压患者最近半年的血压趋势如何？"
    if "糖尿病" in compact and "最近3个月" in compact and "hba1c" in compact.lower() and "异常比例" in compact:
        return "糖尿病患者最近 3 个月 HbA1c 异常比例是多少？"
    return normalized


def _extract_hba1c_abnormal_trend_months(question: str) -> int | None:
    normalized = " ".join(str(question or "").strip().split())
    compact = normalized.replace(" ", "")
    if "hba1c" not in compact.lower() or "异常人数" not in compact or "趋势" not in compact:
        return None
    match = re.search(r"最近\s*(\d{1,2})\s*个?月", normalized)
    if not match:
        match = re.search(r"最近(\d{1,2})个?月", compact)
    if not match:
        if "最近半年" in compact:
            return 6
        return None
    months = int(match.group(1))
    return max(1, min(months, 24))


def _extract_body_fragment(html: str) -> str:
    match = re.search(r"<body[^>]*>(.*)</body>", html, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else html


def _write_planner_log(question: str, planner_payload: Dict[str, Any], extra: Dict[str, Any] | None = None) -> str:
    payload = {"question": question, "planner": planner_payload}
    if extra:
        payload.update(extra)
    candidate_paths = [
        ensure_parent(resolve_path(f"{PLANNER_LOG_DIR}/planner_{_question_slug(question)}.json")),
        Path(f"/tmp/chroniccare_planner_{_question_slug(question)}.json"),
    ]
    for target in candidate_paths:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _safe_replace(target)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return target.as_posix() if str(target).startswith("/tmp/") else f"{PLANNER_LOG_DIR}/{target.name}"
        except PermissionError:
            continue
    return "planner_log_unavailable"


def _write_query_result_artifacts(prefix: str, rows: List[Dict[str, Any]]) -> Dict[str, str]:
    output_dir = resolve_path(OPEN_NL2SQL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    headers = list(rows[0].keys()) if rows else []
    _safe_replace(json_path)
    _safe_replace(csv_path)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return {
        "json_path": f"{OPEN_NL2SQL_DIR}/{json_path.name}",
        "csv_path": f"{OPEN_NL2SQL_DIR}/{csv_path.name}",
    }


def _execute_safe_sql_question(question: str, sql: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_server_config()
    is_safe, errors, safe_sql, warnings = validate_sql(sql)
    if not is_safe or safe_sql is None:
        return {
            "status": "failed",
            "question": question,
            "sql": sql,
            "is_safe": False,
            "warnings": warnings,
            "errors": errors or ["SQL safety validation failed."],
            "safety_note": safety_note(cfg),
        }
    try:
        rows = fetch_rows(safe_sql)
    except Exception as exc:
        return {
            "status": "failed",
            "question": question,
            "sql": safe_sql,
            "is_safe": True,
            "warnings": warnings,
            "errors": [f"SQL execution failed: {exc}"],
            "safety_note": safety_note(cfg),
        }
    response = build_sql_response(question=question, sql=safe_sql, result=rows, filters=filters, warnings=warnings)
    artifacts = _write_query_result_artifacts(f"query_{_question_slug(question)}", rows)
    response.update(
        {
            "status": "success",
            "question": question,
            "table": {"rows": rows[:50], "row_count": len(rows)},
            "result_artifacts": artifacts,
            "result_table_url": public_artifact_url(cfg, f"/artifacts/open-nl2sql/{Path(artifacts['csv_path']).name}"),
            "safety_note": safety_note(cfg),
        }
    )
    return response


def _candidate_for_plan(question: str, planner_payload: Dict[str, Any]) -> str | None:
    diseases = planner_payload.get("disease_filters", [])
    risks = planner_payload.get("risk_filters", [])
    time_window = planner_payload.get("time_window") or {}
    lowered_question = question.lower()
    if ("冠心病" in question or "coronary" in lowered_question) and ("高脂血症" in question or "hyperlipidemia" in lowered_question):
        return (
            "SELECT m.drug_name, re.event_type, COUNT(DISTINCT p.patient_id) AS patient_count "
            "FROM patient_profile p "
            "JOIN medication_record m ON p.patient_id = m.patient_id "
            "JOIN risk_event re ON p.patient_id = re.patient_id "
            "WHERE lower(p.disease_tags) LIKE '%coronary_risk%' "
            "  AND lower(p.disease_tags) LIKE '%hyperlipidemia%' "
            "GROUP BY m.drug_name, re.event_type "
            "ORDER BY patient_count DESC, m.drug_name ASC, re.event_type ASC "
            "LIMIT 20"
        )
    if planner_payload.get("intent") == "risk_distribution":
        if "high" in risks and "占比" in question:
            return (
                "SELECT ROUND(SUM(CASE WHEN prs.risk_level = 'high' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS high_risk_ratio "
                "FROM patient_risk_score prs"
            )
        return (
            "SELECT prs.risk_level, COUNT(DISTINCT prs.patient_id) AS patient_count "
            "FROM patient_risk_score prs GROUP BY prs.risk_level ORDER BY patient_count DESC"
        )
    if "随访" in question and time_window:
        days = int(time_window.get("value", 30) or 30)
        if any(token in question for token in ("多少", "数量", "人数", "统计")) and not any(token in question for token in FOLLOWUP_DYNAMIC_VISUAL_HINTS):
            return (
                "SELECT COUNT(DISTINCT fp.patient_id) AS patient_count "
                "FROM followup_plan fp "
                f"WHERE fp.status IN ('pending','scheduled') AND date(fp.followup_date) BETWEEN date('now','localtime') AND date('now','localtime','+{days} day')"
            )
    if "hypertension" in diseases and "diabetes" in diseases:
        return (
            "SELECT COUNT(DISTINCT p.patient_id) AS patient_count "
            "FROM patient_profile p "
            "WHERE lower(p.disease_tags) LIKE '%hypertension%' AND lower(p.disease_tags) LIKE '%diabetes%'"
        )
    if "hypertension" in diseases:
        return (
            "SELECT COUNT(DISTINCT p.patient_id) AS patient_count "
            "FROM patient_profile p WHERE lower(p.disease_tags) LIKE '%hypertension%'"
        )
    if "diabetes" in diseases:
        return (
            "SELECT COUNT(DISTINCT p.patient_id) AS patient_count "
            "FROM patient_profile p WHERE lower(p.disease_tags) LIKE '%diabetes%'"
        )
    if "高风险" in question or "high" in risks:
        return (
            "SELECT COUNT(DISTINCT prs.patient_id) AS patient_count "
            "FROM patient_risk_score prs WHERE prs.risk_level = 'high'"
        )
    return None


def _read_json_first(paths: List[str]) -> Dict[str, Any]:
    for candidate in paths:
        path = resolve_path(candidate)
        if path.exists():
            return read_json(path)
    return {}


def _looks_like_subgraph_request(question: str, planner: Dict[str, Any]) -> bool:
    if not any(token in question for token in SUBGRAPH_REQUEST_HINTS):
        return False
    return bool(planner.get("disease_filters") or planner.get("risk_filters") or planner.get("intent") == "graph_sql_joint_analysis")


def _looks_like_disease_inventory_question(question: str) -> bool:
    if _looks_like_disease_combination_question(question):
        return False
    if "风险等级" in question or ("风险" in question and "分布" in question):
        return False
    return any(
        token in question
        for token in (
            "几种病",
            "多少种病",
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
            "患者人数分布",
            "患者疾病分布",
            "疾病人数分布",
        )
    )


def _looks_like_disease_combination_question(question: str) -> bool:
    normalized = str(question or "")
    return any(token in normalized for token in ("疾病组合", "共病", "多病", "多种病", "不同疾病组合"))


def _looks_like_cohort_disease_question(question: str) -> bool:
    return any(token in question for token in ("疾病", "慢病", "病种", "患病类型", "疾病类型"))


def _looks_like_cohort_risk_question(question: str) -> bool:
    return "风险等级" in question or ("风险" in question and "分布" in question)


def _contains_explicit_disease_name(question: str) -> bool:
    normalized = str(question or "").lower()
    disease_tokens = set(DISEASE_LABELS.keys()) | set(DISEASE_LABELS.values()) | {
        "中风",
        "脑卒中",
        "冠心病",
        "慢阻肺",
        "copd",
        "脂肪肝",
        "慢性肾病",
        "糖尿病肾病",
        "睡眠呼吸暂停",
    }
    return any(str(token).lower() in normalized for token in disease_tokens if token)


def _display_disease_label(raw: str) -> str:
    value = str(raw or "").strip()
    if not value or value.lower() == "nan":
        return ""
    lowered = value.lower()
    if lowered in DISEASE_LABELS:
        return DISEASE_LABELS[lowered]
    return value.replace("_", " ").strip()


def _collect_disease_inventory_rows() -> List[Dict[str, Any]]:
    rows = fetch_rows(
        """
        SELECT patient_id, disease_tags
        FROM patient_profile
        WHERE disease_tags IS NOT NULL
          AND trim(disease_tags) != ''
        """
    )
    counter: Counter[str] = Counter()
    for row in rows:
        tags = {
            _display_disease_label(item)
            for item in str(row.get("disease_tags") or "").split(";")
            if _display_disease_label(item)
        }
        for tag in tags:
            counter[tag] += 1
    total_patients = fetch_one("SELECT COUNT(DISTINCT patient_id) AS patient_count FROM patient_profile").get("patient_count", 0) or 0
    normalized_name_to_code = {value: key for key, value in DISEASE_LABELS.items()}
    return [
        {
            "疾病名称": name,
            "英文标准Code": normalized_name_to_code.get(name, ""),
            "患者人数": count,
            "占比": round(count / max(1, int(total_patients)), 4),
        }
        for name, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _collect_disease_combination_rows(limit: int | None = 12, min_tags: int = 2) -> List[Dict[str, Any]]:
    rows = fetch_rows(
        """
        SELECT disease_tags, COUNT(DISTINCT patient_id) AS patient_count
        FROM patient_profile
        WHERE disease_tags IS NOT NULL AND trim(disease_tags) != ''
        GROUP BY disease_tags
        ORDER BY patient_count DESC, disease_tags ASC
        """
    )
    result: List[Dict[str, Any]] = []
    for row in rows:
        raw_tags: List[str] = []
        seen: set[str] = set()
        for item in str(row.get("disease_tags") or "").split(";"):
            tag = item.strip()
            if not tag or tag == "nan" or tag in seen:
                continue
            seen.add(tag)
            raw_tags.append(tag)
        if len(raw_tags) < min_tags:
            continue
        tags = [DISEASE_LABELS.get(item, item) for item in raw_tags]
        result.append(
            {
                "疾病组合": " + ".join(tags),
                "患者人数": int(row.get("patient_count", 0) or 0),
                "疾病标签数": len(raw_tags),
                "统计口径": "精确多病组合",
            }
        )
        if limit is not None and len(result) >= limit:
            break
    return result


def _collect_disease_pairwise_combination_rows(limit: int = 20) -> List[Dict[str, Any]]:
    rows = fetch_rows(
        """
        SELECT patient_id, disease_tags
        FROM patient_profile
        WHERE disease_tags IS NOT NULL AND trim(disease_tags) != ''
        """
    )
    counter: Counter[Tuple[str, str]] = Counter()
    for row in rows:
        tags = sorted(
            {
                DISEASE_LABELS.get(item.strip(), item.strip())
                for item in str(row.get("disease_tags") or "").split(";")
                if item.strip() and item.strip() != "nan"
            }
        )
        for index, left in enumerate(tags):
            for right in tags[index + 1 :]:
                counter[(left, right)] += 1
    return [
        {
            "疾病组合": f"{left} + {right}",
            "患者人数": count,
            "统计口径": "两两共现",
        }
        for (left, right), count in counter.most_common(limit)
    ]


def _collect_exact_combo_length_rows() -> List[Dict[str, Any]]:
    rows = fetch_rows(
        """
        SELECT disease_tags, COUNT(DISTINCT patient_id) AS patient_count
        FROM patient_profile
        WHERE disease_tags IS NOT NULL AND trim(disease_tags) != ''
        GROUP BY disease_tags
        """
    )
    counter: Counter[int] = Counter()
    for row in rows:
        tags = [
            item.strip()
            for item in str(row.get("disease_tags") or "").split(";")
            if item.strip() and item.strip() != "nan"
        ]
        if tags:
            counter[len(set(tags))] += int(row.get("patient_count", 0) or 0)
    return [
        {"疾病标签数": tag_count, "患者人数": patient_count}
        for tag_count, patient_count in sorted(counter.items())
    ]


def _persist_cohort_context(payload: Dict[str, Any], *, question: str, cohort_label: str, cohort_type: str) -> None:
    context = AnalysisContext.current()
    window = payload.get("window") if isinstance(payload.get("window"), dict) else {}
    context_payload = {
            "cohort_id": _question_slug(f"{cohort_label}:{question}:{context.data_version}"),
            "source_question": question,
            "source_tool": payload.get("matched_id") or payload.get("intent"),
            "cohort_label": cohort_label,
            "cohort_type": cohort_type,
            "cohort_definition": {
                "type": cohort_type,
                "window_days": payload.get("window_days"),
                "filters": payload.get("filters") or {},
            },
            "filters": payload.get("filters") or {},
            "time_window": window,
            "window_days": payload.get("window_days"),
            "cohort_patient_count": payload.get("cohort_patient_count"),
            "patient_count": payload.get("cohort_patient_count"),
            "data_version": context.data_version,
            "as_of_date": context.as_of_date,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "report_url": payload.get("report_url") or payload.get("graph_url"),
            "graph_url": payload.get("graph_url"),
            "saved_from": payload.get("analysis_id") or payload.get("matched_id") or payload.get("intent"),
        }
    conversation_id = get_current_conversation_id()
    if conversation_id:
        save_conversation_cohort(conversation_id, context_payload)
    else:
        context_payload["context_mode"] = "legacy_global_compatibility"
        save_last_cohort(context_payload)


def _disease_inventory_payload(question: str) -> Dict[str, Any]:
    cfg = load_server_config()
    rows = _collect_disease_inventory_rows()
    patient_total = int(fetch_one("SELECT COUNT(DISTINCT patient_id) AS patient_count FROM patient_profile").get("patient_count", 0) or 0)
    disease_patient_total = sum(int(item.get("患者人数", 0) or 0) for item in rows)
    disease_labels = [str(item["疾病名称"]) for item in rows]
    disease_type_count = len(rows)
    disease_distribution_text = _format_disease_distribution_text(rows)
    chart_rows = [
        {
            **item,
            "疾病覆盖率标签": f"{item['疾病名称']}（{float(item['占比']) * 100:.1f}%）",
        }
        for item in rows
    ]
    combination_rows = _collect_disease_combination_rows()
    prevalence_svg = _bar_chart_svg(
        "慢病患者覆盖率（多标签）",
        f"口径：每个病种患者人数 ÷ 唯一患者总数 {patient_total}；患者可有多种病，各比例可重叠且不要求合计 100%",
        chart_rows,
        "疾病覆盖率标签",
        "患者人数",
    )
    pie_svg_path = _write_svg_chart("disease_inventory_distribution.svg", prevalence_svg)
    disease_rows = [[str(index + 1), item["疾病名称"], item["英文标准Code"], str(item["患者人数"]), f"{float(item['占比']) * 100:.1f}%"] for index, item in enumerate(rows)]
    combo_rows = [[str(index + 1), item["疾病组合"], str(item["患者人数"])] for index, item in enumerate(combination_rows)]
    chart_html = (
        "<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'><title>慢病疾病类型分布</title></head>"
        "<body style='font-family:Arial;padding:24px;background:#f7f9fb;color:#102a43;'>"
        "<h1>当前慢病知识图谱中的疾病类型</h1>"
        "<p>先从患者主表中拆分 disease_tags，再按患者去重统计每类疾病的覆盖人数，并实时生成分布图。</p>"
        f"<p><strong>说明：</strong>当前唯一患者总数为 {patient_total} 人；疾病标签累计覆盖 {disease_patient_total} 人次。一个患者可能同时带有多个疾病标签，因此疾病人数之和可能大于患者总数。</p>"
        f"{prevalence_svg}"
        f"{_html_table(['序号', '疾病名称', '英文标准Code', '患者人数', '占比'], disease_rows)}"
        f"{_html_table(['序号', '常见疾病组合', '患者人数'], combo_rows)}"
        "<p style='margin-top:24px;'>本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。</p>"
        "</body></html>"
    )
    payload = {
        "title": "当前慢病知识图谱中的疾病类型",
        "disease_type_count": disease_type_count,
        "rows": rows,
    }
    json_path, html_path, chart_path = _write_analysis_bundle(
        "analysis_disease_inventory",
        payload,
        chart_html,
        "先从 patient_profile 中拆分 disease_tags，再按患者去重统计每类疾病的覆盖人数，并实时生成疾病分布图。",
        related_links=[
            {
                "label": "疾病分布图",
                "title": "疾病分布状况图",
                "url": public_artifact_url(cfg, "/artifacts/charts/disease_inventory_distribution.svg"),
                "service_url": service_artifact_url(cfg, "/artifacts/charts/disease_inventory_distribution.svg"),
                "route_path": artifact_route_path("/artifacts/charts/disease_inventory_distribution.svg"),
            },
            {
                "label": "完整分析报告",
                "title": "疾病类型分析报告",
                "url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_inventory"),
                "service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_inventory"),
                "route_path": artifact_route_path("/artifacts/graph-driven/analysis_disease_inventory"),
            },
        ],
        embedded_sections=[
            {"title": "疾病覆盖率图", "content": prevalence_svg},
        ],
    )
    return {
        "status": "success",
        "question": question,
        "matched_id": "kg_disease_inventory",
        "intent": "kg_disease_inventory",
        "analysis_id": "analysis_disease_inventory",
        "chart_type": "chart_bundle",
        "metric": {
            "name": "disease_type_count",
            "label": "疾病类型总数",
            "value": disease_type_count,
            "unit": "种",
        },
        "table": {"rows": rows, "detail_rows": rows},
        "patient_count": patient_total,
        "disease_patient_total": disease_patient_total,
        "disease_combination_distribution": combination_rows,
        "charts": [
            {
                "name": "疾病分布状况图",
                "path": pie_svg_path,
                "url": public_artifact_url(cfg, "/artifacts/charts/disease_inventory_distribution.svg"),
                "png_alias_url": public_artifact_url(cfg, "/artifacts/charts/disease_inventory_distribution.png"),
                "service_url": service_artifact_url(cfg, "/artifacts/charts/disease_inventory_distribution.svg"),
                "type": "bar",
            }
        ],
        "insight": f"当前唯一患者总数为 {patient_total} 人，共识别出 {disease_type_count} 种疾病类型：{disease_distribution_text}。",
        "summary_text": f"当前唯一患者总数为 {patient_total} 人，共识别出 {disease_type_count} 种疾病类型：{disease_distribution_text}。疾病标签累计覆盖 {disease_patient_total} 人次；一个患者可能同时属于多种疾病。已生成疾病分布图、疾病组合分布和完整分析页面。",
        "answer": f"当前数据中的 {disease_type_count} 种疾病类型分别为：{disease_distribution_text}。当前唯一患者总数为 {patient_total} 人；由于一个患者可能同时属于多种疾病，因此各病种人数之和可能大于患者总数。",
        "final_answer_lock": (
            f"当前常见病/疾病分布必须按 patient_profile.disease_tags 统计：唯一患者总数 {patient_total} 人，"
            f"疾病类型 {disease_type_count} 种。最终回答必须使用 rows/detail_rows 中的疾病名称和患者人数，"
            "禁止改用知识图谱患者口径、节点数或 DataMate 汇总口径。"
        ),
        "disease_labels": disease_labels,
        "disease_type_count": disease_type_count,
        "analysis_json": json_path,
        "analysis_html": html_path,
        "analysis_chart_html": chart_path,
        "chart_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_inventory_chart"),
        "chart_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_inventory_chart"),
        "report_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_inventory"),
        "report_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_inventory"),
        "graph_url": None,
        "graph_service_url": None,
        "safety_note": safety_note(cfg),
    }


def _normalize_disease_label(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if raw in DISEASE_LABELS.values():
        return raw
    if lowered in DISEASE_LABELS:
        return DISEASE_LABELS[lowered]
    return DISEASE_LABELS.get(raw, raw)


def _format_disease_distribution_text(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "当前未识别到可展示的疾病类型。"
    items = []
    for row in rows:
        name = str(row.get("疾病名称") or "")
        patient_count = int(row.get("患者人数") or 0)
        ratio = float(row.get("占比") or 0.0) * 100
        items.append(f"{name}（{patient_count}人，{ratio:.1f}%）")
    return "；".join(items)


def _disease_distribution_payload(question: str, diseases: List[str] | None = None) -> Dict[str, Any]:
    normalized_diseases = [_normalize_disease_label(item) for item in (diseases or []) if _normalize_disease_label(item)]
    if not normalized_diseases:
        return _disease_inventory_payload(question)
    cfg = load_server_config()
    rows = _collect_disease_inventory_rows()
    matched_rows = [row for row in rows if str(row.get("疾病名称")) in normalized_diseases]
    patient_total = int(fetch_one("SELECT COUNT(DISTINCT patient_id) AS patient_count FROM patient_profile").get("patient_count", 0) or 0)
    combination_rows = _collect_disease_combination_rows()
    matched_distribution_text = _format_disease_distribution_text(matched_rows)
    matched_label = "、".join(str(row.get("疾病名称")) for row in matched_rows if row.get("疾病名称"))
    if len(normalized_diseases) > 1:
        label_to_code = {label: code for code, label in DISEASE_LABELS.items()}
        codes = [label_to_code.get(label, label) for label in normalized_diseases]
        conditions = " AND ".join(
            f"lower(disease_tags) LIKE '%{str(code).lower()}%'" for code in codes
        )
        matched_patient_count = int(fetch_one(
            f"SELECT COUNT(DISTINCT patient_id) AS patient_count FROM patient_profile WHERE {conditions}"
        ).get("patient_count", 0) or 0)
        matched_rows = [{
            "疾病名称": "合并".join(normalized_diseases),
            "英文标准Code": "+".join(codes),
            "患者人数": matched_patient_count,
            "占比": round(matched_patient_count / max(1, patient_total), 4),
            "统计口径": "同时满足全部疾病标签（AND 交集）",
        }]
        matched_distribution_text = _format_disease_distribution_text(matched_rows)
        metric_label = f"{matched_label}共病患者数"
    else:
        matched_patient_count = sum(int(row.get("患者人数", 0) or 0) for row in matched_rows)
        metric_label = f"{matched_label}患者数" if matched_label else "命中疾病患者数"
    return {
        "status": "success",
        "question": question,
        "matched_id": "disease_distribution",
        "intent": "disease_distribution",
        "metric": {
            "name": "matched_disease_patient_count",
            "label": metric_label,
            "value": matched_patient_count,
            "unit": "人",
        },
        "table": {"rows": matched_rows, "detail_rows": matched_rows},
        "rows": matched_rows,
        "disease_labels": [str(item.get("疾病名称")) for item in matched_rows],
        "matched_disease_count": len(matched_rows),
        "matched_patient_count": matched_patient_count,
        "disease_type_count": len(rows),
        "patient_count": patient_total,
        "disease_combination_distribution": combination_rows,
        "summary_text": f"当前命中 {len(matched_rows)} 种疾病：{matched_distribution_text}。已返回对应患者人数、占比和常见疾病组合。",
        "insight": f"患者主表当前总患者数为 {patient_total} 人；一个患者可能同时属于多种疾病，因此疾病人数之间可以重叠。",
        "answer": f"当前命中的疾病类型为：{matched_distribution_text}。",
        "final_answer_lock": (
            f"当前问题是单病/疾病人数问题，必须按 patient_profile.disease_tags 去重统计。"
            f"{matched_label or '命中疾病'}患者数 = {matched_patient_count} 人；"
            f"总患者基数 = {patient_total} 人。禁止回答知识图谱患者数、节点数或其它口径。"
        ),
        "report_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_inventory"),
        "report_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_inventory"),
        "chart_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_inventory_chart"),
        "chart_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_inventory_chart"),
        "graph_url": None,
        "graph_service_url": None,
        "safety_note": safety_note(cfg),
    }


def _disease_combination_distribution_payload(question: str) -> Dict[str, Any]:
    cfg = load_server_config()
    all_combination_rows = _collect_disease_combination_rows(limit=None, min_tags=2)
    top_combination_rows = all_combination_rows[:12]
    pairwise_requested = any(token in str(question or "") for token in ("两两", "任意两种", "两种病", "二病"))
    pairwise_rows = _collect_disease_pairwise_combination_rows() if pairwise_requested else []
    combo_length_rows = _collect_exact_combo_length_rows()
    tag_count_rows = fetch_rows(
        """
        SELECT patient_id, disease_tags
        FROM patient_profile
        """
    )
    tag_count_counter: Counter[int] = Counter()
    for row in tag_count_rows:
        tags = {
            item.strip()
            for item in str(row.get("disease_tags") or "").split(";")
            if item.strip() and item.strip().lower() != "nan"
        }
        tag_count_counter[len(tags)] += 1
    total_row = fetch_one("SELECT COUNT(DISTINCT patient_id) AS patient_count FROM patient_profile")
    total_patients = int(total_row.get("patient_count", 0) or 0)
    multimorbidity_count = sum(count for tag_count, count in tag_count_counter.items() if tag_count >= 2)
    single_disease_count = int(tag_count_counter.get(1, 0) or 0)
    no_tag_patient_count = int(tag_count_counter.get(0, 0) or 0)
    multimorbidity_ratio = round(multimorbidity_count / max(1, total_patients), 4)
    top_patient_sum = sum(int(item.get("患者人数") or 0) for item in top_combination_rows)
    other_patient_count = max(0, multimorbidity_count - top_patient_sum)
    other_combo_count = max(0, len(all_combination_rows) - len(top_combination_rows))
    combination_rows = list(top_combination_rows)
    if other_combo_count:
        combination_rows.append(
            {
                "疾病组合": f"其他多病组合（{other_combo_count} 种）",
                "患者人数": other_patient_count,
                "疾病标签数": "2+",
                "统计口径": "精确多病组合汇总",
            }
        )
    chart_rows = [{"疾病组合": item["疾病组合"], "患者人数": item["患者人数"]} for item in top_combination_rows[:10]]
    chart_html = _bar_chart_html("常见多病组合人数分布", chart_rows, "疾病组合", "患者人数")
    chart_svg = _bar_chart_svg(
        "常见多病组合人数分布",
        "口径：仅展示精确多病组合 Top 10，单病患者不进入本表；总共病人数见指标",
        chart_rows,
        "疾病组合",
        "患者人数",
    )
    chart_svg_path = _write_svg_chart("disease_combination_distribution.svg", chart_svg)
    payload = {
        "title": "疾病组合与共病分布",
        "combination_rows": combination_rows,
        "top_combination_rows": top_combination_rows,
        "all_combination_row_count": len(all_combination_rows),
        "displayed_top_rows_patient_sum": top_patient_sum,
        "other_multimorbidity_combo_count": other_combo_count,
        "other_multimorbidity_patient_count": other_patient_count,
        "pairwise_rows": pairwise_rows,
        "combo_length_rows": combo_length_rows,
        "multimorbidity_count": multimorbidity_count,
        "multimorbidity_ratio": multimorbidity_ratio,
        "single_disease_patient_count": single_disease_count,
        "no_tag_patient_count": no_tag_patient_count,
        "display_scope": "精确多病组合 Top 12 + 其他多病组合汇总（仅包含疾病标签数 >= 2 的患者）",
    }
    json_path, html_path, chart_path = _write_analysis_bundle(
        "analysis_disease_combination_distribution",
        payload,
        chart_html,
        "先从 patient_profile.disease_tags 拆分疾病标签，再按患者去重统计常见疾病组合与多病共病患者规模。",
        related_links=[
            {
                "label": "疾病组合分布图",
                "title": "常见疾病组合人数分布",
                "url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_combination_distribution_chart"),
                "service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_combination_distribution_chart"),
                "route_path": artifact_route_path("/artifacts/graph-driven/analysis_disease_combination_distribution_chart"),
            },
            {
                "label": "完整分析报告",
                "title": "疾病组合与共病分布报告",
                "url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_combination_distribution"),
                "service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_combination_distribution"),
                "route_path": artifact_route_path("/artifacts/graph-driven/analysis_disease_combination_distribution"),
            },
        ],
    )
    return {
        "status": "success",
        "question": question,
        "matched_id": "disease_combination_distribution",
        "intent": "disease_combination_distribution",
        "analysis_id": "analysis_disease_combination_distribution",
        "metric": {
            "name": "multimorbidity_patient_count",
            "label": "多病共病患者数",
            "value": multimorbidity_count,
            "unit": "人",
        },
        "table": {
            "rows": combination_rows,
            "detail_rows": combination_rows,
            "top_rows": top_combination_rows,
            "all_combination_row_count": len(all_combination_rows),
            "displayed_top_rows_patient_sum": top_patient_sum,
            "other_multimorbidity_patient_count": other_patient_count,
            "pairwise_rows": pairwise_rows,
            "combo_length_rows": combo_length_rows,
        },
        "rows": combination_rows,
        "top_combination_rows": top_combination_rows,
        "all_combination_row_count": len(all_combination_rows),
        "displayed_table_patient_sum": sum(int(item.get("患者人数") or 0) for item in combination_rows),
        "displayed_top_rows_patient_sum": top_patient_sum,
        "other_multimorbidity_combo_count": other_combo_count,
        "other_multimorbidity_patient_count": other_patient_count,
        "pairwise_rows": pairwise_rows,
        "combo_length_rows": combo_length_rows,
        "multimorbidity_count": multimorbidity_count,
        "multimorbidity_patient_count": multimorbidity_count,
        "single_disease_patient_count": single_disease_count,
        "no_tag_patient_count": no_tag_patient_count,
        "multimorbidity_ratio": multimorbidity_ratio,
        "analysis_json": json_path,
        "analysis_html": html_path,
        "analysis_chart_html": chart_path,
        "chart_asset_path": chart_svg_path,
        "report_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_combination_distribution"),
        "report_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_combination_distribution"),
        "chart_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_combination_distribution_chart"),
        "chart_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_disease_combination_distribution_chart"),
        "charts": [
            {
                "name": "多病组合分布图",
                "path": chart_svg_path,
                "url": public_artifact_url(cfg, "/artifacts/charts/disease_combination_distribution.svg"),
                "png_alias_url": public_artifact_url(cfg, "/artifacts/charts/disease_combination_distribution.png"),
                "service_url": service_artifact_url(cfg, "/artifacts/charts/disease_combination_distribution.svg"),
                "type": "bar",
            }
        ],
        "summary_text": (
            f"当前多病共病患者共 {multimorbidity_count} 人，占患者总数 {multimorbidity_ratio * 100:.1f}%；"
            f"单病患者 {single_disease_count} 人、无有效疾病标签患者 {no_tag_patient_count} 人已从多病组合表中排除；"
            f"表格展示精确多病组合 Top 12，并用“其他多病组合”汇总剩余 {other_combo_count} 种组合，表内患者数合计为 {multimorbidity_count} 人。"
        ),
        "insight": (
            "精确多病组合表按 patient_profile.disease_tags 的完整标签集合统计，单病患者不会进入本表；"
            "表格与图表均使用同一份精确组合数据。只有用户明确询问两两共现时才返回两两共现表。"
        ),
        "final_answer_lock": (
            "当前问题使用精确疾病组合口径；正文表格必须逐行复述 table.rows，图表使用相同 top_combination_rows。"
            "禁止把 pairwise_rows 的两两共现人数替换进精确组合表。"
        ),
        "safety_note": safety_note(cfg),
    }


def data_summary() -> Dict[str, Any]:
    payload = _kg_summary_payload("现在有多少患者、随访记录、检验记录？")
    payload["matched_id"] = "data_summary"
    payload["intent"] = "data_summary"
    return attach_analysis_context(payload, AnalysisContext.current())


def disease_distribution_query(question: str) -> Dict[str, Any]:
    if _looks_like_disease_combination_question(question):
        payload = _disease_combination_distribution_payload(question)
    elif _looks_like_disease_inventory_question(question) or "慢病类型分布" in str(question or ""):
        payload = _disease_inventory_payload(question)
    else:
        routed = route_agent_intent({"query": question, "last_context": load_last_cohort()})
        diseases = routed.get("normalized_entities", {}).get("diseases") or []
        payload = _disease_distribution_payload(question, diseases) if diseases else _disease_inventory_payload(question)
    return attach_analysis_context(payload, AnalysisContext.current())


def disease_combination_distribution_query(question: str) -> Dict[str, Any]:
    return attach_analysis_context(_disease_combination_distribution_payload(question), AnalysisContext.current())


def risk_level_distribution_query(question: str) -> Dict[str, Any]:
    payload = _risk_level_distribution_payload(question)
    payload["matched_id"] = "risk_level_distribution"
    payload["intent"] = "risk_level_distribution"
    return attach_analysis_context(payload, AnalysisContext.current())


def followup_high_risk_query(question: str) -> Dict[str, Any]:
    days = int(extract_future_window_days(question) or 30)
    return attach_analysis_context(open_analysis_query(question), AnalysisContext.current().with_window(days))


def cohort_disease_distribution_query(question: str) -> Dict[str, Any]:
    return attach_analysis_context(open_analysis_query(question), AnalysisContext.current())


def metric_query(question: str) -> Dict[str, Any]:
    payload = run_open_sql_query(question, prefer_llm=True, allow_chart=True, last_context=load_last_cohort())
    if payload.get("status") == "success":
        payload.update(
            {
                "original_question": question,
                "rewritten_question": payload.get("question", question),
                "canonical_id": "open_sql_query",
                "matched_id": payload.get("template_id") or payload.get("intent"),
                "fallback_used": payload.get("stage") == "fallback",
            }
        )
        return payload
    return analysis_query(question)


def trend_query(question: str) -> Dict[str, Any]:
    payload = run_open_sql_query(question, prefer_llm=True, allow_chart=True, last_context=load_last_cohort())
    if payload.get("status") == "success":
        payload.update(
            {
                "original_question": question,
                "rewritten_question": payload.get("question", question),
                "canonical_id": "open_sql_query",
                "matched_id": payload.get("template_id") or payload.get("intent"),
                "fallback_used": payload.get("stage") == "fallback",
            }
        )
        return payload
    return analysis_query(question)


def _risk_level_distribution_payload(question: str) -> Dict[str, Any]:
    cfg = load_server_config()
    rows = fetch_rows(
        """
        WITH ranked AS (
          SELECT
            patient_id,
            risk_level,
            ROW_NUMBER() OVER (
              PARTITION BY patient_id
              ORDER BY datetime(COALESCE(created_at, '1900-01-01')) DESC, visit_id DESC
            ) AS rn
          FROM patient_risk_score
        )
        SELECT
          risk_level AS 风险等级,
          COUNT(*) AS 患者人数
        FROM ranked
        WHERE rn = 1
        GROUP BY risk_level
        ORDER BY CASE lower(risk_level)
          WHEN 'high' THEN 1
          WHEN 'medium' THEN 2
          WHEN 'low' THEN 3
          ELSE 9
        END
        """
    )
    total = sum(int(row.get("患者人数", 0) or 0) for row in rows)
    normalized_rows = []
    label_map = {"high": "高风险", "medium": "中风险", "low": "低风险"}
    for row in rows:
        raw_level = str(row.get("风险等级") or "").strip().lower()
        count = int(row.get("患者人数", 0) or 0)
        normalized_rows.append(
            {
                "风险等级": label_map.get(raw_level, raw_level or "未知"),
                "患者人数": count,
                "占比": round(count / max(1, total), 4),
            }
        )
    pie_svg = _pie_chart_svg(
        "不同风险等级患者人数分布",
        "口径：按每位患者最新一条 risk score 记录统计高/中/低风险人数，避免同一患者跨多次随访重复计数",
        normalized_rows,
        "风险等级",
        "患者人数",
        center_value=total,
        center_label="唯一患者数",
    )
    pie_svg_path = _write_svg_chart("risk_level_distribution.svg", pie_svg)
    return {
        "status": "success",
        "question": question,
        "matched_id": "risk_level_distribution",
        "intent": "risk_level_distribution",
        "metric": {
            "name": "risk_level_patient_total",
            "label": "风险分层患者总数",
            "value": total,
            "unit": "人",
        },
        "table": {"rows": normalized_rows, "detail_rows": normalized_rows},
        "rows": normalized_rows,
        "charts": [
            {
                "name": "风险等级分布图",
                "path": pie_svg_path,
                "url": public_artifact_url(cfg, "/artifacts/charts/risk_level_distribution.svg"),
                "png_alias_url": public_artifact_url(cfg, "/artifacts/charts/risk_level_distribution.png"),
                "service_url": service_artifact_url(cfg, "/artifacts/charts/risk_level_distribution.svg"),
                "type": "pie",
            }
        ],
        "summary_text": f"当前按最新风险评分统计，唯一患者共 {total} 人，已返回高/中/低风险人数分布。",
        "insight": "该结果按每位患者最新一条风险评分记录聚合，因此总人数应与患者总数口径一致，不会因多次随访重复计数。",
        "metric_definition": f"统计口径为全量 {total} 名患者按每位患者最新一条风险评分记录聚合；这不是任意未来 N 天待随访患者队列，后者必须按用户指定窗口实时计算。",
        "report_url": public_artifact_url(cfg, "/artifacts/charts/risk_level_distribution.svg"),
        "report_service_url": service_artifact_url(cfg, "/artifacts/charts/risk_level_distribution.svg"),
        "chart_url": public_artifact_url(cfg, "/artifacts/charts/risk_level_distribution.svg"),
        "chart_service_url": service_artifact_url(cfg, "/artifacts/charts/risk_level_distribution.svg"),
        "graph_url": public_artifact_url(cfg, "/artifacts/graph.html"),
        "graph_service_url": service_artifact_url(cfg, "/artifacts/graph.html"),
        "safety_note": safety_note(cfg),
    }


def _system_status_payload(question: str) -> Dict[str, Any]:
    cfg = load_server_config()
    metrics = load_current_metrics()
    return {
        "status": "success",
        "question": question,
        "matched_id": "system_status",
        "intent": "system_status",
        "service_status": "ok",
        "pipeline_ready": True,
        "patient_count": metrics.get("patient_count"),
        "visit_count": metrics.get("visit_count"),
        "lab_result_count": metrics.get("lab_result_count"),
        "medication_record_count": metrics.get("medication_record_count"),
        "node_count": metrics.get("node_count"),
        "edge_count": metrics.get("edge_count"),
        "summary_text": "当前 ChronicCare 运行状态正常，分析服务、图表入口和图谱入口均已就绪。",
        "report_url": public_artifact_url(cfg, "/artifacts/report"),
        "report_service_url": service_artifact_url(cfg, "/artifacts/report"),
        "chart_url": public_artifact_url(cfg, "/artifacts/charts"),
        "chart_service_url": service_artifact_url(cfg, "/artifacts/charts"),
        "graph_url": public_artifact_url(cfg, "/artifacts/graph.html"),
        "graph_service_url": service_artifact_url(cfg, "/artifacts/graph.html"),
        "safety_note": safety_note(cfg),
    }


def _kg_summary_payload(question: str) -> Dict[str, Any]:
    from tool_server.kg_tools import kg_summary

    payload = kg_summary()
    summary_text = (
        payload.get("text")
        or (
            f"当前真实数据规模为：患者 {payload.get('patient_count', 'N/A')} 人、"
            f"随访记录 {payload.get('visit_count', 'N/A')} 条、"
            f"检验记录 {payload.get('lab_result_count', 'N/A')} 条、"
            f"用药记录 {payload.get('medication_record_count', 'N/A')} 条；"
            f"图谱节点 {payload.get('node_count', 'N/A')} 个、边 {payload.get('edge_count', 'N/A')} 条；"
            f"实体类型 {payload.get('entity_type_total_count', 'N/A')} 种、"
            f"关系类型 {payload.get('relation_type_total_count', 'N/A')} 种。"
        )
    )
    payload.update(
        {
            "question": question,
            "matched_id": "kg_summary",
            "intent": "kg_summary",
            "chart_url": payload.get("chart_index_url"),
            "chart_service_url": payload.get("chart_index_service_url"),
            "summary_text": summary_text,
        }
    )
    return payload


def _extract_patient_id(text: str) -> str | None:
    match = re.search(r"\bP\d{4}\b", text or "", flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def _dynamic_subgraph_analysis(question: str, planner: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_server_config()
    from tool_server.kg_tools import kg_subgraph_render

    payload = kg_subgraph_render(question, max_nodes=96)
    if payload.get("status") != "success":
        return payload
    return {
        "status": "success",
        "question": question,
        "analysis_id": f"dynamic_subgraph_{_question_slug(question)}",
        "intent": "dynamic_subgraph_render",
        "chart_type": "knowledge_subgraph",
        "graph_url": payload.get("html_url"),
        "graph_service_url": payload.get("service_html_url"),
        "html_url": payload.get("html_url"),
        "service_html_url": payload.get("service_html_url"),
        "html_route_path": payload.get("html_route_path"),
        "preview_url": payload.get("preview_url"),
        "preview_service_url": payload.get("preview_service_url"),
        "preview_route_path": payload.get("preview_route_path"),
        "preview_png_url": payload.get("preview_png_url"),
        "preview_png_service_url": payload.get("preview_png_service_url"),
        "preview_png_route_path": payload.get("preview_png_route_path"),
        "preview_svg_url": payload.get("preview_svg_url"),
        "preview_svg_service_url": payload.get("preview_svg_service_url"),
        "preview_svg_route_path": payload.get("preview_svg_route_path"),
        # Pure subgraph requests should expose only the real-time subgraph page.
        "report_url": None,
        "report_service_url": None,
        "chart_url": None,
        "chart_service_url": None,
        "charts": [
            {
                "name": "知识图谱子图预览",
                "url": payload.get("preview_png_url") or payload.get("preview_url"),
                "service_url": payload.get("preview_png_service_url") or payload.get("preview_service_url"),
                "route_path": payload.get("preview_png_route_path") or payload.get("preview_route_path"),
                "type": "png" if payload.get("preview_png_url") else "svg",
            }
        ] if (payload.get("preview_png_url") or payload.get("preview_url")) else [],
        "subgraph_id": payload.get("subgraph_id"),
        "graph_node_count": payload.get("node_count"),
        "graph_edge_count": payload.get("edge_count"),
        "seed_labels": payload.get("seed_labels", []),
        "cohort_patient_count": payload.get("cohort_patient_count"),
        "display_patient_node_count": payload.get("display_patient_node_count"),
        "semantic_node_count": payload.get("semantic_node_count"),
        "graph_scope": payload.get("graph_scope_explanation"),
        "summary_text": (
            "已根据当前问题实时生成图谱子图，可直接打开当前子图页面。"
        ),
        "insight": "该结果仅返回当前问题对应的实时子图，不附带额外分析报告或其它图表。",
        "planner": planner,
        "safety_note": safety_note(cfg),
    }


def _translate_scalar(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in GENDER_LABELS:
            return GENDER_LABELS[lowered]
    return value


def _translate_disease_tags(raw: str) -> str:
    tags = [item.strip() for item in str(raw or "").split(";") if item.strip() and item.strip() != "nan"]
    return "、".join(DISEASE_LABELS.get(item, item) for item in tags)


def _humanize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    translated = {}
    for key, value in row.items():
        label = FIELD_LABELS.get(str(key), str(key))
        if key == "disease_tags":
            translated[label] = _translate_disease_tags(str(value or ""))
        else:
            translated[label] = _translate_scalar(value)
    return translated


def _first_metric(row: Dict[str, Any]) -> Tuple[str | None, Any]:
    for key, value in row.items():
        if isinstance(value, (int, float)):
            return str(key), value
    return None, None


def _metric_from_question(question: str, row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = QUESTION_METADATA.get(question, {})
    metric_name = metadata.get("metric_name")
    if metric_name and metric_name in row:
        return {
            "name": metric_name,
            "label": metadata.get("metric_label", FIELD_LABELS.get(metric_name, metric_name)),
            "value": row.get(metric_name),
            "unit": metadata.get("unit"),
        }
    fallback_name, fallback_value = _first_metric(row)
    return {
        "name": fallback_name,
        "label": FIELD_LABELS.get(str(fallback_name), str(fallback_name)) if fallback_name else None,
        "value": fallback_value,
        "unit": metadata.get("unit"),
    }


def _write_cohort_exports(analysis_id: str, title: str, rows: List[Dict[str, Any]]) -> Dict[str, str]:
    output_dir = resolve_path(RUNTIME_GRAPH_ANALYSIS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{analysis_id}_patients.csv"
    html_path = output_dir / f"{analysis_id}_patients.html"
    headers = list(rows[0].keys()) if rows else []
    _safe_replace(csv_path)
    _safe_replace(html_path)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    table_header = "".join(f"<th style='padding:10px 12px;border-bottom:1px solid #d9e2ec;text-align:left;'>{header}</th>" for header in headers)
    table_body = "".join(
        "<tr>" + "".join(f"<td style='padding:10px 12px;border-bottom:1px solid #e6edf3;'>{row.get(header, '')}</td>" for header in headers) + "</tr>"
        for row in rows
    )
    html_path.write_text(
        f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>{title}</title></head>
<body style="font-family:Arial;padding:24px;background:#f7f9fb;color:#102a43;">
<h1>{title}</h1>
<p>该页面用于展示疾病群体的全量患者列表，图谱页面负责解释结构，这里负责全量明细与导出。</p>
<p><a href="{csv_path.name}" target="_blank">下载 CSV 导出文件</a></p>
<table style="width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden;">
<thead><tr>{table_header}</tr></thead><tbody>{table_body}</tbody>
</table>
<p style="margin-top:24px;">本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。</p>
</body></html>""",
        encoding="utf-8",
    )
    return {
        "csv_path": f"{RUNTIME_GRAPH_ANALYSIS_DIR}/{csv_path.name}",
        "html_path": f"{RUNTIME_GRAPH_ANALYSIS_DIR}/{html_path.name}",
    }


def _load_indicator_items() -> List[Dict[str, Any]]:
    cfg = load_server_config()
    return _read_json_first([cfg["paths"]["indicator_results"]]).get("items", [])


def _public_question_id(raw_id: Any) -> str:
    raw = str(raw_id or "")
    if raw.startswith("NLQ") and raw[3:].isdigit():
        return f"AQ{raw[3:]}"
    return raw or "AQ000"



def _find_standard_question(question: str) -> Dict[str, Any] | None:
    items = _load_indicator_items()
    exact = next((item for item in items if item["question"] == question), None)
    if exact is not None:
        return exact
    keywords = [keyword for keyword in ["高血压", "糖尿病", "血脂", "HbA1c", "空腹血糖", "趋势", "BMI", "图谱", "关系", "实体", "随访", "高风险", "盐"] if keyword.lower() in question.lower()]
    best_score = -1
    best_item = None
    for item in items:
        score = sum(1 for keyword in keywords if keyword in item["question"])
        if score > best_score:
            best_score = score
            best_item = item
    return best_item if best_score > 0 else None


def _is_high_risk_followup_question(text: str) -> bool:
    normalized = str(text or "").lower().replace(" ", "").replace("-", "")
    return any(token in str(text or "") for token in ("高风险", "高危")) or "highrisk" in normalized


def analysis_query(question: str) -> Dict[str, Any]:
    cfg = load_server_config()
    rewritten = rewrite_question(question)
    planner = plan_query(question, {"canonical_question": rewritten.get("question")}).to_dict()
    planner_log_path = _write_planner_log(question, planner, {"entrypoint": "analysis_query"})
    direct_controlled_question = _normalize_controlled_question(question)
    normalized_question = (
        direct_controlled_question
        if direct_controlled_question in QUESTION_METADATA
        else _normalize_controlled_question(rewritten.get("question", question))
    )
    if _looks_like_disease_inventory_question(question) or rewritten.get("canonical_id") == "kg_disease_inventory":
        payload = _disease_inventory_payload(normalized_question)
        payload.update({"planner": planner, "planner_log_path": planner_log_path})
        return payload
    canonical_id = rewritten.get("canonical_id")
    window_days = int(rewritten.get("window_days") or extract_future_window_days(rewritten.get("question", question), default=30))
    if canonical_id == "future_followup_chart_bundle" or (
        "随访" in question and window_days != 30
    ):
        if (
            canonical_id == "future_n_days_high_risk_followup"
            or _is_high_risk_followup_question(question)
            or _is_high_risk_followup_question(rewritten.get("question", ""))
        ):
            payload = _future_high_risk_followup_count(window_days=window_days)
            payload.update(
                {
                    "matched_id": f"future_high_risk_followup_{window_days}d",
                    "question": normalized_question,
                    "canonical_id": "future_n_days_high_risk_followup",
                    "intent": "future_n_days_high_risk_followup",
                    "chart_type": "chart_bundle",
                    "planner": planner,
                    "planner_log_path": planner_log_path,
                }
            )
        else:
            payload = _future_followup_chart_bundle(window_days=window_days)
            payload.update(
                {
                    "matched_id": f"dynamic_future_followup_{window_days}d",
                    "question": normalized_question,
                    "canonical_id": "future_followup_chart_bundle",
                    "intent": "future_followup_dynamic",
                    "chart_type": "chart_bundle",
                    "planner": planner,
                    "planner_log_path": planner_log_path,
                }
            )
        return payload
    if canonical_id == "dynamic_subgraph_render":
        payload = _dynamic_subgraph_analysis(normalized_question, planner)
        payload.update(
            {
                "matched_id": "dynamic_subgraph_render",
                "intent": "dynamic_subgraph_render",
                "chart_type": "knowledge_subgraph",
                "planner": planner,
                "planner_log_path": planner_log_path,
            }
        )
        return payload
    controlled_payload = _controlled_metric_payload(normalized_question)
    if controlled_payload is not None:
        controlled_payload.update(
            {
                "matched_id": controlled_payload.get("analysis_id", "controlled_metric_query"),
                "intent": QUESTION_METADATA.get(controlled_payload.get("question", ""), {}).get("intent", "controlled_metric_query"),
                "chart_type": QUESTION_METADATA.get(controlled_payload.get("question", ""), {}).get("chart_type", "metric_card"),
                "planner": planner,
                "planner_log_path": planner_log_path,
            }
        )
        return controlled_payload
    if canonical_id in {
        "future_30d_high_risk_followup_disease_distribution",
        "high_salt_bp_abnormal_rate",
        "hypertension_diabetes_multi_indicator",
    }:
        payload = graph_driven_analysis(question)
        payload.update(
            {
                "matched_id": payload.get("analysis_id"),
                "intent": "graph_driven_analysis",
                "chart_type": "graph_driven_bundle",
                "planner": planner,
                "planner_log_path": planner_log_path,
            }
        )
        return payload
    exact = _find_standard_question(normalized_question)
    if exact is None:
        if _looks_like_subgraph_request(normalized_question, planner):
            payload = _dynamic_subgraph_analysis(normalized_question, planner)
            payload.update(
                {
                    "matched_id": "dynamic_subgraph_render",
                    "planner": planner,
                    "planner_log_path": planner_log_path,
                    "schema_registry_size": len(get_schema_registry()),
                }
            )
            return payload
        candidate_sql = _candidate_for_plan(normalized_question, planner)
        if candidate_sql:
            payload = _execute_safe_sql_question(
                normalized_question,
                candidate_sql,
                {
                    "time_window": planner.get("time_window"),
                    "disease_filters": planner.get("disease_filters"),
                    "risk_filters": planner.get("risk_filters"),
                },
            )
            payload.update(
                {
                    "matched_id": "planner_generated_sql",
                    "intent": planner.get("intent"),
                    "chart_type": "table",
                    "planner": planner,
                    "planner_log_path": planner_log_path,
                    "schema_registry_size": len(get_schema_registry()),
                }
            )
            return payload
        open_payload = open_analysis_query(normalized_question)
        if open_payload.get("status") == "success":
            open_payload.setdefault("delegated_from", "analysis_query")
            open_payload.setdefault("planner", planner)
            open_payload.setdefault("planner_log_path", planner_log_path)
            return open_payload
        return {
            "status": "failed",
            "question": question,
            "planner": planner,
            "planner_log_path": planner_log_path,
            "errors": ["No matching stable analysis question found."],
            "safety_note": safety_note(cfg),
        }
    rows = exact.get("rows") or []
    first_row = rows[0] if rows else {}
    metric = _metric_from_question(exact["question"], first_row if isinstance(first_row, dict) else {})
    detail_rows = [_humanize_row(dict(row)) for row in rows if isinstance(row, dict)]
    metadata = QUESTION_METADATA.get(exact["question"], {})
    return {
        "status": "success",
        "matched_id": _public_question_id(exact["id"]),
        "source_id": exact["id"],
        "question": exact["question"],
        "intent": exact.get("intent") or metadata.get("intent") or "standard_analysis",
        "chart_type": exact.get("chart_type") or metadata.get("chart_type") or ("table" if len(rows) > 1 else "metric_card"),
        "metric": metric,
        "table": {"rows": rows, "detail_rows": detail_rows},
        "insight": exact.get("insight") or f"已返回 {exact['question']} 的标准分析结果。",
        "planner": planner,
        "planner_log_path": planner_log_path,
        "safety_note": safety_note(cfg),
    }


def _bar_chart_html(title: str, rows: List[Dict[str, Any]], label_key: str, value_key: str) -> str:
    max_value = max([float(item.get(value_key, 0) or 0) for item in rows] + [1.0])
    bars: List[str] = []
    for item in rows:
        label = str(item.get(label_key))
        value = float(item.get(value_key, 0) or 0)
        width = max(6.0, (value / max_value) * 100.0)
        bars.append(
            f"<div style='margin:10px 0;'><div style='font-size:13px;color:#243b53'>{label} ({value})</div>"
            f"<div style='height:22px;background:linear-gradient(90deg,#1f77b4,#5dade2);width:{width}%;border-radius:999px;'></div></div>"
        )
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>{title}</title></head>
<body style="font-family:Arial;padding:24px;background:#f7f9fb;color:#102a43;">
<h1>{title}</h1>
{''.join(bars)}
<p>本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。</p>
</body></html>"""


def _metric_cards_html(title: str, metrics: List[Dict[str, Any]]) -> str:
    cards = "".join(
        [
            f"<div style='background:white;padding:18px;border-radius:16px;box-shadow:0 12px 30px rgba(16,42,67,0.08);'>"
            f"<div style='font-size:14px;color:#486581;'>{item['label']}</div>"
            f"<div style='font-size:30px;font-weight:700;margin-top:8px;'>{item['value']}</div>"
            f"</div>"
            for item in metrics
        ]
    )
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>{title}</title></head>
<body style="font-family:Arial;padding:24px;background:#f7f9fb;color:#102a43;">
<h1>{title}</h1>
<div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;'>{cards}</div>
<p style='margin-top:24px;'>本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。</p>
</body></html>"""


def _svg_palette() -> List[str]:
    return ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]


def _future_followup_window(days: int = 30) -> Dict[str, Any]:
    exact_days = max(1, min(int(days or 1), 200))
    context = AnalysisContext.current().with_window(exact_days)
    return {
        "start_date": str(context.window_start or context.as_of_date),
        "end_date": str(context.window_end or context.as_of_date),
        "window_days": exact_days,
        "window_inclusive": True,
        "as_of_date": context.as_of_date,
        "timezone": context.timezone,
        "data_version": context.data_version,
        "sqlite_version": context.sqlite_version,
        "graph_version": context.graph_version,
    }


def _fill_daily_rows(rows: List[Dict[str, Any]], start_date: str, end_date: str) -> List[Dict[str, Any]]:
    row_map = {str(item.get("followup_date")): int(item.get("patient_count", 0) or 0) for item in rows}
    cursor = fetch_rows(
        """
        WITH RECURSIVE dates(day) AS (
          SELECT date(?)
          UNION ALL
          SELECT date(day, '+1 day')
          FROM dates
          WHERE day < date(?)
        )
        SELECT day AS followup_date
        FROM dates
        """,
        [start_date, end_date],
    )
    return [
        {
            "followup_date": str(item["followup_date"]),
            "patient_count": row_map.get(str(item["followup_date"]), 0),
        }
        for item in cursor
    ]


def _risk_level_label(value: str) -> str:
    mapping = {
        "high": "高风险",
        "medium": "中风险",
        "low": "低风险",
        "urgent": "紧急",
        "normal": "正常",
    }
    key = str(value or "").strip().lower()
    return mapping.get(key, str(value or "未标注"))


def _html_table(headers: List[str], rows: List[List[str]]) -> str:
    header_html = "".join(
        f"<th style='text-align:left;padding:12px 14px;border-bottom:1px solid #d9e2ec;color:#102a43;background:#f8fbff;'>{header}</th>"
        for header in headers
    )
    body_html = "".join(
        "<tr>"
        + "".join(
            f"<td style='padding:12px 14px;border-bottom:1px solid #e6edf3;color:#243b53;'>{cell}</td>"
            for cell in row
        )
        + "</tr>"
        for row in rows
    )
    return (
        "<div style='overflow:auto;border:1px solid #d9e2ec;border-radius:14px;background:white;'>"
        "<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody></table></div>"
    )


def _month_rows_to_table(rows: List[Dict[str, Any]], value_key: str) -> str:
    return _html_table(
        ["月份", FIELD_LABELS.get(value_key, value_key)],
        [[str(item.get("month") or ""), str(item.get(value_key) or 0)] for item in rows],
    )


def _trend_analysis_bundle(
    *,
    analysis_id: str,
    title: str,
    subtitle: str,
    rows: List[Dict[str, Any]],
    value_key: str,
    summary_cards: List[Dict[str, Any]],
    explanation: str,
) -> Dict[str, Any]:
    cfg = load_server_config()
    line_svg = _line_chart_svg(title, subtitle, rows, "month", value_key)
    chart_file = _write_svg_chart(f"{analysis_id}.svg", line_svg)
    chart_html = _chart_bundle_html(
        title,
        summary_cards,
        [
            {"title": "趋势折线图", "content": line_svg},
            {"title": "趋势明细", "content": _month_rows_to_table(rows, value_key)},
        ],
    )
    payload = {
        "title": title,
        "rows": rows,
        "value_key": value_key,
        "summary_cards": summary_cards,
    }
    json_path, html_path, chart_path = _write_analysis_bundle(
        analysis_id,
        payload,
        chart_html,
        explanation,
    )
    return {
        "analysis_id": analysis_id,
        "analysis_json": json_path,
        "analysis_html": html_path,
        "analysis_chart_html": chart_path,
        "chart_asset_path": chart_file,
        "report_url": public_artifact_url(cfg, f"/artifacts/graph-driven/{analysis_id}"),
        "report_service_url": service_artifact_url(cfg, f"/artifacts/graph-driven/{analysis_id}"),
        "chart_url": public_artifact_url(cfg, f"/artifacts/graph-driven/{analysis_id}_chart"),
        "chart_service_url": service_artifact_url(cfg, f"/artifacts/graph-driven/{analysis_id}_chart"),
        "image_url": public_artifact_url(cfg, f"/artifacts/charts/{analysis_id}.svg"),
        "image_service_url": service_artifact_url(cfg, f"/artifacts/charts/{analysis_id}.svg"),
    }


def _line_chart_svg(title: str, subtitle: str, rows: List[Dict[str, Any]], label_key: str, value_key: str) -> str:
    width = 1440
    height = 880
    margin_left = 110
    margin_right = 72
    margin_top = 108
    margin_bottom = 124
    palette = _svg_palette()
    values = [float(item.get(value_key, 0) or 0) for item in rows]
    max_value = max(values + [1.0])
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    points: List[str] = []
    area_points: List[str] = []
    circles: List[str] = []
    labels: List[str] = []
    guides: List[str] = []
    y_labels: List[str] = []
    point_labels: List[str] = []
    accent_x = margin_left
    accent_y = margin_top + plot_h
    tick_count = 6
    for tick in range(tick_count):
        ratio = tick / max(1, tick_count - 1)
        y = margin_top + plot_h - ratio * plot_h
        guides.append(f"<line x1='{margin_left}' y1='{y:.1f}' x2='{width - margin_right}' y2='{y:.1f}' stroke='#d9e2ec' stroke-dasharray='6 8'/>")
        y_value = max_value * ratio
        y_labels.append(f"<text x='{margin_left - 18}' y='{y + 6:.1f}' font-size='18' text-anchor='end' fill='#486581'>{y_value:.0f}</text>")
    max_index = max(range(len(values)), key=lambda idx: values[idx]) if values else 0
    for index, item in enumerate(rows):
        x = margin_left + (plot_w * index / max(1, len(rows) - 1))
        y = margin_top + plot_h - (float(item.get(value_key, 0) or 0) / max_value) * plot_h
        points.append(f"{x:.1f},{y:.1f}")
        area_points.append(f"{x:.1f},{y:.1f}")
        radius = 11 if index == max_index else 7.5
        circles.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius}' fill='{palette[0]}' stroke='white' stroke-width='3' />")
        raw_label = str(item.get(label_key) or "")
        label = raw_label if len(rows) <= 12 else raw_label[5:]
        if index in {0, len(rows) - 1, max_index} or index % max(1, len(rows) // 5) == 0:
            labels.append(f"<text x='{x:.1f}' y='{height - 38}' font-size='18' text-anchor='middle' fill='#486581'>{label}</text>")
        if float(item.get(value_key, 0) or 0) > 0 or index == max_index:
            point_labels.append(
                f"<g><rect x='{x - 26:.1f}' y='{y - 52:.1f}' rx='12' ry='12' width='52' height='30' fill='white' opacity='0.96'/>"
                f"<text x='{x:.1f}' y='{y - 31:.1f}' font-size='16' font-weight='700' text-anchor='middle' fill='#102a43'>{int(float(item.get(value_key, 0) or 0))}</text></g>"
            )
    area = " ".join([f"{accent_x:.1f},{accent_y:.1f}"] + area_points + [f"{width - margin_right:.1f},{accent_y:.1f}"])
    return f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' preserveAspectRatio='xMidYMid meet'>
<defs>
  <linearGradient id='bgGradient' x1='0' y1='0' x2='0' y2='1'>
    <stop offset='0%' stop-color='#f8fbff'/>
    <stop offset='100%' stop-color='#eef4fb'/>
  </linearGradient>
  <linearGradient id='areaGradient' x1='0' y1='0' x2='0' y2='1'>
    <stop offset='0%' stop-color='{palette[0]}' stop-opacity='0.28'/>
    <stop offset='100%' stop-color='{palette[0]}' stop-opacity='0.03'/>
  </linearGradient>
  <filter id='softShadow' x='-10%' y='-10%' width='120%' height='140%'>
    <feDropShadow dx='0' dy='12' stdDeviation='14' flood-color='#8aa5bf' flood-opacity='0.18'/>
  </filter>
</defs>
<rect width='{width}' height='{height}' rx='28' ry='28' fill='url(#bgGradient)'/>
<text x='{margin_left}' y='54' font-size='44' font-family='Arial' font-weight='700' fill='#102a43'>{title}</text>
<text x='{margin_left}' y='84' font-size='22' font-family='Arial' fill='#486581'>{subtitle}</text>
{''.join(guides)}
<line x1='{margin_left}' y1='{height-margin_bottom}' x2='{width-margin_right}' y2='{height-margin_bottom}' stroke='#9fb3c8' stroke-width='2'/>
<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height-margin_bottom}' stroke='#9fb3c8' stroke-width='2'/>
{''.join(y_labels)}
<path d='M {area}' fill='url(#areaGradient)'/>
<polyline fill='none' stroke='{palette[0]}' stroke-width='6' stroke-linecap='round' stroke-linejoin='round' points='{' '.join(points)}' filter='url(#softShadow)'/>
{''.join(circles)}
{''.join(point_labels)}
{''.join(labels)}
</svg>"""


def _pie_chart_svg(
    title: str,
    subtitle: str,
    rows: List[Dict[str, Any]],
    label_key: str,
    value_key: str,
    *,
    center_value: int | None = None,
    center_label: str = "总患者数",
) -> str:
    width = 1440
    height = 880
    cx = 360
    cy = 470
    radius = 220
    inner_radius = 112
    total = sum(float(item.get(value_key, 0) or 0) for item in rows) or 1.0
    start_angle = 0.0
    palette = _svg_palette()
    slices: List[str] = []
    legends: List[str] = []
    for index, item in enumerate(rows):
        value = float(item.get(value_key, 0) or 0)
        fraction = value / total
        end_angle = start_angle + fraction * 360.0
        color = palette[index % len(palette)]
        if len(rows) == 1:
            slices.append(
                f"<circle cx='{cx}' cy='{cy}' r='{radius}' fill='none' stroke='{color}' stroke-width='{radius - inner_radius}' />"
            )
        else:
            x1 = cx + radius * math.cos(math.radians(start_angle))
            y1 = cy + radius * math.sin(math.radians(start_angle))
            x2 = cx + radius * math.cos(math.radians(end_angle))
            y2 = cy + radius * math.sin(math.radians(end_angle))
            inner_x1 = cx + inner_radius * math.cos(math.radians(start_angle))
            inner_y1 = cy + inner_radius * math.sin(math.radians(start_angle))
            inner_x2 = cx + inner_radius * math.cos(math.radians(end_angle))
            inner_y2 = cy + inner_radius * math.sin(math.radians(end_angle))
            large_arc = 1 if end_angle - start_angle > 180 else 0
            slices.append(
                f"<path d='M {x1:.2f} {y1:.2f} "
                f"A {radius} {radius} 0 {large_arc} 1 {x2:.2f} {y2:.2f} "
                f"L {inner_x2:.2f} {inner_y2:.2f} "
                f"A {inner_radius} {inner_radius} 0 {large_arc} 0 {inner_x1:.2f} {inner_y1:.2f} Z' "
                f"fill='{color}'/>"
            )
        percentage = fraction * 100.0
        legend_label = _risk_level_label(str(item.get(label_key) or ""))
        legends.append(
            f"<rect x='820' y='{220 + index*70}' width='22' height='22' rx='6' ry='6' fill='{color}'/>"
            f"<text x='858' y='{238 + index*70}' font-size='28' font-family='Arial' font-weight='700' fill='#243b53'>{legend_label}</text>"
            f"<text x='858' y='{272 + index*70}' font-size='20' font-family='Arial' fill='#486581'>{int(value)} 人，占比 {percentage:.1f}%</text>"
        )
        start_angle = end_angle
    return f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' preserveAspectRatio='xMidYMid meet'>
<defs>
  <linearGradient id='pieBgGradient' x1='0' y1='0' x2='1' y2='1'>
    <stop offset='0%' stop-color='#f8fbff'/>
    <stop offset='100%' stop-color='#eef4fb'/>
  </linearGradient>
  <filter id='pieShadow' x='-10%' y='-10%' width='120%' height='140%'>
    <feDropShadow dx='0' dy='14' stdDeviation='16' flood-color='#8aa5bf' flood-opacity='0.16'/>
  </filter>
</defs>
<rect width='{width}' height='{height}' rx='28' ry='28' fill='url(#pieBgGradient)'/>
<text x='70' y='56' font-size='44' font-family='Arial' font-weight='700' fill='#102a43'>{title}</text>
<text x='70' y='88' font-size='22' font-family='Arial' fill='#486581'>{subtitle}</text>
<g filter='url(#pieShadow)'>
{''.join(slices)}
</g>
<circle cx='{cx}' cy='{cy}' r='{inner_radius - 8}' fill='white'/>
<text x='{cx}' y='{cy - 10}' text-anchor='middle' font-size='54' font-family='Arial' font-weight='700' fill='#102a43'>{int(center_value if center_value is not None else total)}</text>
<text x='{cx}' y='{cy + 34}' text-anchor='middle' font-size='24' font-family='Arial' fill='#486581'>{center_label}</text>
{''.join(legends)}
</svg>"""


def _bar_chart_svg(title: str, subtitle: str, rows: List[Dict[str, Any]], label_key: str, value_key: str) -> str:
    width = 1800
    row_height = 92
    label_x = 80
    bar_x = 900
    bar_max_width = 620
    height = max(620, 190 + len(rows) * row_height)
    max_value = max([float(item.get(value_key, 0) or 0) for item in rows] + [1.0])
    palette = _svg_palette()
    bars: List[str] = []

    def esc(value: Any) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def compact_label(value: str, max_chars: int = 46) -> List[str]:
        text = value.strip()
        if len(text) <= max_chars:
            return [text]
        separators = [" + ", "，", "、", "/"]
        for sep in separators:
            parts = text.split(sep)
            if len(parts) <= 1:
                continue
            first = ""
            second_parts: List[str] = []
            for part in parts:
                candidate = part if not first else f"{first}{sep}{part}"
                if len(candidate) <= max_chars:
                    first = candidate
                else:
                    second_parts.append(part)
            second = sep.join(second_parts)
            if first and second:
                if len(second) > max_chars:
                    second = second[: max_chars - 1] + "…"
                return [first, second]
        return [text[: max_chars - 1] + "…"]

    for index, item in enumerate(rows):
        label_lines = compact_label(str(item.get(label_key) or ""))
        value = float(item.get(value_key, 0) or 0)
        y = 150 + index * row_height
        bar_width = bar_max_width * (value / max_value)
        color = palette[index % len(palette)]
        label_svg = "".join(
            f"<text x='{label_x}' y='{y + 8 + line_index * 28}' font-size='22' font-family='Arial' "
            f"fill='#243b53'>{esc(line)}</text>"
            for line_index, line in enumerate(label_lines)
        )
        value_x = min(width - 120, bar_x + bar_width + 22)
        bars.append(
            label_svg
            +
            f"<rect x='{bar_x}' y='{y - 20}' width='{bar_width:.1f}' height='34' rx='17' ry='17' fill='{color}' opacity='0.92'/>"
            f"<text x='{value_x:.1f}' y='{y + 5}' font-size='24' font-family='Arial' font-weight='700' fill='#102a43'>{value:.0f}</text>"
        )
    return f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' preserveAspectRatio='xMidYMid meet'>
<defs>
  <linearGradient id='barBgGradient' x1='0' y1='0' x2='1' y2='1'>
    <stop offset='0%' stop-color='#f8fbff'/>
    <stop offset='100%' stop-color='#eef4fb'/>
  </linearGradient>
</defs>
<rect width='{width}' height='{height}' rx='28' ry='28' fill='url(#barBgGradient)'/>
<text x='70' y='56' font-size='44' font-family='Arial' font-weight='700' fill='#102a43'>{title}</text>
<text x='70' y='88' font-size='22' font-family='Arial' fill='#486581'>{subtitle}</text>
{''.join(bars)}
</svg>"""


def _write_svg_chart(filename: str, content: str) -> str:
    base, output_dir = _first_writable_output_dir(
        [RUNTIME_CHART_DIR, LOCAL_CHART_DIR, "outputs/charts"],
        [filename],
    )
    output_path = ensure_parent(output_dir / filename)
    _safe_replace(output_path)
    output_path.write_text(content, encoding="utf-8")
    return f"{base}/{filename}"


def _preferred_chart_font(size: int):
    if ImageFont is None:
        return None
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


def _latin_chart_font(size: int):
    if ImageFont is None:
        return None
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


def _draw_mixed_text(draw: Any, xy: Tuple[int, int], text: Any, *, cjk_font: Any, latin_font: Any, fill: str) -> None:
    x, y = xy
    for chunk in re.findall(r"[\x00-\x7f]+|[^\x00-\x7f]+", str(text)):
        font = latin_font if chunk and all(ord(ch) < 128 for ch in chunk) else cjk_font
        draw.text((x, y), chunk, fill=fill, font=font)
        try:
            bbox = draw.textbbox((x, y), chunk, font=font)
            x += bbox[2] - bbox[0]
        except Exception:
            x += len(chunk) * 12


def _png_img_html(route_path: str, alt: str) -> str:
    return (
        f"<img src='{artifact_route_path(route_path)}' alt='{alt}' "
        "style='display:block;width:100%;max-width:980px;height:auto;margin:0 auto;"
        "border:1px solid #d9e2ec;border-radius:8px;background:#fff;'/>"
    )


def _write_followup_line_png(filename: str, title: str, rows: List[Dict[str, Any]], label_key: str, value_key: str) -> str | None:
    if Image is None or ImageDraw is None or ImageFont is None:
        return None
    try:
        base, output_dir = _first_writable_output_dir(
            [RUNTIME_CHART_DIR, LOCAL_CHART_DIR, "outputs/charts"],
            [filename],
        )
        output_path = ensure_parent(output_dir / filename)
        _safe_replace(output_path)
        width, height = 1280, 720
        image = Image.new("RGB", (width, height), "#f8fafc")
        draw = ImageDraw.Draw(image)
        title_font = _preferred_chart_font(32)
        body_font = _latin_chart_font(20)
        small_font = _latin_chart_font(16)
        cjk_body_font = _preferred_chart_font(20)
        draw.rounded_rectangle((28, 24, width - 28, height - 24), radius=18, fill="#ffffff", outline="#e2e8f0", width=2)
        _draw_mixed_text(draw, (58, 44), title, cjk_font=title_font, latin_font=_latin_chart_font(32), fill="#102a43")
        plot_left, plot_top, plot_right, plot_bottom = 92, 132, 1205, 600
        values = [int(item.get(value_key, 0) or 0) for item in rows]
        max_value = max(values + [1])
        y_axis_max = max_value if max_value <= 10 else int(math.ceil(max_value / 5.0) * 5)
        for index in range(5):
            y = plot_bottom - int((plot_bottom - plot_top) * index / 4)
            draw.line((plot_left, y, plot_right, y), fill="#edf2f7", width=1)
            draw.text((46, y - 10), str(round(y_axis_max * index / 4)), fill="#52606d", font=small_font)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#94a3b8", width=2)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#94a3b8", width=2)
        if rows:
            step = (plot_right - plot_left) / max(1, len(rows) - 1)
            points = []
            for index, item in enumerate(rows):
                x = int(plot_left + step * index)
                value = int(item.get(value_key, 0) or 0)
                y = int(plot_bottom - (plot_bottom - plot_top) * value / max(1, y_axis_max))
                points.append((x, y))
            if len(points) >= 2:
                draw.line(points, fill="#2563eb", width=5, joint="curve")
            for x, y in points:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#2563eb", outline="#ffffff", width=2)
            tick_count = min(8, len(rows))
            tick_indexes = sorted(set(round(i * (len(rows) - 1) / max(1, tick_count - 1)) for i in range(tick_count)))
            for index in tick_indexes:
                x, _ = points[index]
                label = str(rows[index].get(label_key) or "")
                draw.text((x - 46, plot_bottom + 22), label[5:], fill="#52606d", font=small_font)
            _draw_mixed_text(
                draw,
                (plot_left, height - 54),
                f"统计天数 {len(rows)} 天，总人数 {sum(values)} 人，峰值 {max_value} 人/日",
                cjk_font=cjk_body_font,
                latin_font=body_font,
                fill="#243b53",
            )
        image.save(output_path, "PNG")
        return f"{base}/{filename}"
    except Exception:
        return None


def _write_followup_pie_png(filename: str, title: str, rows: List[Dict[str, Any]], label_key: str, value_key: str) -> str | None:
    if Image is None or ImageDraw is None or ImageFont is None:
        return None
    try:
        base, output_dir = _first_writable_output_dir(
            [RUNTIME_CHART_DIR, LOCAL_CHART_DIR, "outputs/charts"],
            [filename],
        )
        output_path = ensure_parent(output_dir / filename)
        _safe_replace(output_path)
        width, height = 1280, 720
        image = Image.new("RGB", (width, height), "#f8fafc")
        draw = ImageDraw.Draw(image)
        title_font = _preferred_chart_font(32)
        body_font = _preferred_chart_font(24)
        small_font = _preferred_chart_font(18)
        latin_title_font = _latin_chart_font(32)
        latin_body_font = _latin_chart_font(24)
        latin_small_font = _latin_chart_font(18)
        draw.rounded_rectangle((28, 24, width - 28, height - 24), radius=18, fill="#ffffff", outline="#e2e8f0", width=2)
        _draw_mixed_text(draw, (58, 44), title, cjk_font=title_font, latin_font=latin_title_font, fill="#102a43")
        total = sum(int(item.get(value_key, 0) or 0) for item in rows) or 1
        colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2"]
        bbox = (120, 150, 570, 600)
        if len(rows) == 1:
            draw.ellipse(bbox, fill="#dbeafe", outline="#93c5fd", width=2)
            draw.arc(bbox, start=-90, end=270, fill=colors[0], width=62)
            draw.text((250, 326), str(total), fill="#102a43", font=_latin_chart_font(54))
            draw.text((250, 386), "人", fill="#486581", font=body_font)
        else:
            start = -90.0
            for index, item in enumerate(rows):
                value = int(item.get(value_key, 0) or 0)
                angle = 360.0 * value / total
                draw.pieslice(bbox, start=start, end=start + angle, fill=colors[index % len(colors)], outline="#ffffff", width=2)
                start += angle
        legend_x, legend_y = 680, 170
        for index, item in enumerate(rows):
            label = _risk_level_label(str(item.get(label_key) or ""))
            value = int(item.get(value_key, 0) or 0)
            ratio = value / max(1, total)
            y = legend_y + index * 72
            draw.rounded_rectangle((legend_x, y, legend_x + 30, y + 30), radius=6, fill=colors[index % len(colors)])
            _draw_mixed_text(draw, (legend_x + 46, y - 4), f"{label}: {value} 人", cjk_font=body_font, latin_font=latin_body_font, fill="#243b53")
            _draw_mixed_text(draw, (legend_x + 46, y + 30), f"占比 {ratio * 100:.1f}%", cjk_font=small_font, latin_font=latin_small_font, fill="#64748b")
        image.save(output_path, "PNG")
        return f"{base}/{filename}"
    except Exception:
        return None


def _hba1c_abnormal_trend_payload(months: int, question: str | None = None) -> Dict[str, Any]:
    bounded_months = max(1, min(int(months or 6), 24))
    offset_months = max(0, bounded_months - 1)
    rows = fetch_rows(
        f"""
        SELECT
          strftime('%Y-%m', COALESCE(test_date, record_time)) AS month,
          COUNT(DISTINCT patient_id) AS abnormal_patient_count
        FROM lab_result
        WHERE item_name = 'hba1c'
          AND abnormal_flag != 'normal'
          AND date(COALESCE(test_date, record_time)) >= date('now','start of month','-{offset_months} months')
        GROUP BY strftime('%Y-%m', COALESCE(test_date, record_time))
        ORDER BY month
        """
    )
    analysis_id = f"analysis_trend_hba1c_abnormal_{bounded_months}m"
    title = f"最近 {bounded_months} 个月 HbA1c 异常人数趋势"
    latest_value = int(rows[-1]["abnormal_patient_count"]) if rows else 0
    latest_month = rows[-1]["month"] if rows else "N/A"
    peak_month = max(rows, key=lambda item: int(item.get("abnormal_patient_count", 0) or 0)).get("month") if rows else "N/A"
    bundle = _trend_analysis_bundle(
        analysis_id=analysis_id,
        title=title,
        subtitle="口径：按自然月统计 HbA1c 异常患者去重人数",
        rows=rows,
        value_key="abnormal_patient_count",
        summary_cards=[
            {"label": "统计月份数", "value": len(rows)},
            {"label": "最新月份", "value": latest_month},
            {"label": "最新异常人数", "value": latest_value},
            {"label": "峰值月份", "value": peak_month},
        ],
        explanation=f"按月份统计最近 {bounded_months} 个月 HbA1c 异常患者人数趋势，便于观察糖控异常波动。",
    )
    return {
        "status": "success",
        "question": question or f"最近 {bounded_months} 个月 HbA1c 异常人数趋势如何？",
        "metric": {"name": "abnormal_patient_count", "label": "HbA1c 异常人数", "value": latest_value, "unit": "人"},
        "table": {"rows": rows, "detail_rows": rows},
        "summary_text": f"最近 {bounded_months} 个月 HbA1c 异常人数趋势已生成，最新月份 {latest_month} 为 {latest_value} 人。",
        "insight": f"已生成最近 {bounded_months} 个月 HbA1c 异常人数折线图和趋势明细。",
        **bundle,
        "charts": [
            {
                "name": f"最近{bounded_months}个月HbA1c异常人数趋势图",
                "url": bundle["image_url"],
                "service_url": bundle["image_service_url"],
                "type": "line",
            }
        ],
    }


def _entry_button(label: str, href: str, *, primary: bool = False) -> str:
    if not href:
        return ""
    background = "linear-gradient(135deg,#1f5fbf,#3478f6)" if primary else "#f8fbff"
    color = "#ffffff" if primary else "#1f5fbf"
    border = "none" if primary else "1px solid #d9e2ec"
    return (
        f"<a href='{href}' target='_blank' style='display:inline-flex;align-items:center;justify-content:center;"
        f"padding:10px 16px;border-radius:12px;text-decoration:none;font-size:14px;font-weight:700;"
        f"background:{background};color:{color};border:{border};'>{label}</a>"
    )


def _entry_meta(label: str, value: str) -> str:
    if not value:
        return ""
    return (
        "<div style='margin-top:8px;padding:10px 12px;background:#f8fbff;border-radius:10px;"
        "font-size:12px;line-height:1.6;color:#52606d;word-break:break-all;'>"
        f"<strong style='color:#334e68'>{label}：</strong>{value}</div>"
    )


def _entry_row(item: Dict[str, str]) -> str:
    browser_href = item.get("route_path") or item.get("url") or ""
    service_href = item.get("service_url") or ""
    if not artifact_exists_for_route(browser_href):
        return ""
    return (
        "<tr>"
        f"<td style='padding:12px 14px;border-bottom:1px solid #e6edf3;font-weight:700;color:#102a43;'>{item['label']}</td>"
        f"<td style='padding:12px 14px;border-bottom:1px solid #e6edf3;color:#486581;'>{item.get('title') or item['label']}</td>"
        f"<td style='padding:12px 14px;border-bottom:1px solid #e6edf3;'>{_entry_button(item.get('browser_label', '浏览器入口'), browser_href, primary=True)}</td>"
        f"<td style='padding:12px 14px;border-bottom:1px solid #e6edf3;'>{_entry_button(item.get('service_label', '服务入口'), service_href, primary=False) if service_href and service_href != browser_href else '<span style=\"color:#9fb3c8;font-size:13px;\">同浏览器入口</span>'}</td>"
        "</tr>"
    )


def _write_analysis_bundle(
    analysis_id: str,
    payload: Dict[str, Any],
    chart_html: str,
    explanation: str,
    related_links: List[Dict[str, str]] | None = None,
    embedded_sections: List[Dict[str, str]] | None = None,
) -> Tuple[str, str, str]:
    base, output_dir = _first_writable_output_dir(
        [RUNTIME_GRAPH_ANALYSIS_DIR, LOCAL_GRAPH_ANALYSIS_DIR, "outputs/graph_driven_analysis"],
        [f"{analysis_id}.json", f"{analysis_id}.html", f"{analysis_id}_chart.html"],
    )
    json_path = ensure_parent(output_dir / f"{analysis_id}.json")
    html_path = ensure_parent(output_dir / f"{analysis_id}.html")
    chart_path = ensure_parent(output_dir / f"{analysis_id}_chart.html")
    _safe_replace(json_path)
    _safe_replace(html_path)
    _safe_replace(chart_path)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    chart_path.write_text(chart_html, encoding="utf-8")
    chart_fragment = _extract_body_fragment(chart_html)
    links_html = ""
    if related_links:
        validated_links = [item for item in related_links if artifact_exists_for_route(item.get("route_path") or item.get("url") or "")]
        items = "".join(_entry_row(item) for item in validated_links)
        detail_blocks = "".join(
            "<div style='margin-top:12px;padding:12px 14px;background:#f8fbff;border-radius:12px;'>"
            f"<div style='font-size:13px;font-weight:700;color:#102a43'>{item['label']}</div>"
            f"{_entry_meta('当前页内相对路径', item.get('route_path') or item.get('url') or '')}"
            f"{_entry_meta('浏览器访问地址', item.get('url') or '')}"
            f"{_entry_meta('服务访问地址', item.get('service_url') or '')}"
            "</div>"
            for item in validated_links
        )
        if validated_links:
            links_html = (
                "<section style='margin-top:24px;background:white;padding:22px;border-radius:20px;"
                "box-shadow:0 12px 30px rgba(16,42,67,0.08);'>"
                "<h2 style='margin:0 0 16px 0;color:#102a43'>相关产物入口</h2>"
                "<p style='margin:0 0 14px 0;color:#486581'>以下入口已在当前运行时完成产物存在性校验，仅展示可实际打开的页面。</p>"
                "<table style='width:100%;border-collapse:collapse;border:1px solid #d9e2ec;border-radius:14px;overflow:hidden;'>"
                "<thead><tr>"
                "<th style='text-align:left;padding:12px 14px;border-bottom:1px solid #d9e2ec;'>类型</th>"
                "<th style='text-align:left;padding:12px 14px;border-bottom:1px solid #d9e2ec;'>说明</th>"
                "<th style='text-align:left;padding:12px 14px;border-bottom:1px solid #d9e2ec;'>浏览器入口</th>"
                "<th style='text-align:left;padding:12px 14px;border-bottom:1px solid #d9e2ec;'>服务入口</th>"
                "</tr></thead>"
                f"<tbody>{items}</tbody></table>"
                f"{detail_blocks}</section>"
            )
    embedded_html = ""
    if embedded_sections:
        embedded_html = "".join(
            f"<section style='margin-top:24px;background:white;padding:22px;border-radius:20px;"
            "box-shadow:0 12px 30px rgba(16,42,67,0.08);'>"
            f"<h2 style='margin:0 0 16px 0;color:#102a43'>{item['title']}</h2>{item['content']}</section>"
            for item in embedded_sections
        )
    html_path.write_text(
        f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>{analysis_id}</title></head>
<body style="font-family:Arial;padding:30px;background:linear-gradient(180deg,#f6f9fd 0%,#edf3fb 100%);color:#102a43;">
<div style="max-width:1680px;margin:0 auto;">
<header style="background:white;border-radius:24px;padding:26px 28px;box-shadow:0 18px 40px rgba(16,42,67,0.08);">
<h1 style="margin:0 0 14px 0;font-size:52px;line-height:1.15;">{payload.get('title', analysis_id)}</h1>
<p style="margin:0;font-size:22px;line-height:1.8;color:#243b53;">{explanation}</p>
</header>
<section style="margin-top:24px;background:white;padding:20px;border-radius:24px;box-shadow:0 16px 36px rgba(16,42,67,0.08);overflow:hidden;">
{chart_fragment}
</section>
{links_html}
{embedded_html}
<p style="margin:28px 0 0 0;font-size:18px;color:#486581;">本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。</p>
</div>
</body></html>""",
        encoding="utf-8",
    )
    return (
        f"{base}/{analysis_id}.json",
        f"{base}/{analysis_id}.html",
        f"{base}/{analysis_id}_chart.html",
    )


def _chart_bundle_html(title: str, summary_cards: List[Dict[str, Any]], sections: List[Dict[str, str]]) -> str:
    cards_html = "".join(
        [
            f"<div style='background:white;border-radius:8px;padding:14px 16px;box-shadow:0 8px 22px rgba(16,42,67,0.07);'>"
            f"<div style='font-size:13px;color:#486581'>{item['label']}</div>"
            f"<div style='font-size:30px;font-weight:700;color:#102a43;margin-top:6px'>{item['value']}</div>"
            f"</div>"
            for item in summary_cards
        ]
    )
    section_html = "".join(
        [
            f"<section style='background:white;border-radius:8px;padding:16px 18px;box-shadow:0 8px 22px rgba(16,42,67,0.07);margin-top:16px;overflow-x:auto;'>"
            f"<h2 style='margin:0 0 12px 0;color:#102a43'>{item['title']}</h2>{item['content']}</section>"
            for item in sections
        ]
    )
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>{title}</title></head>
<body style="font-family:'Noto Sans CJK SC','Microsoft YaHei',Arial,sans-serif;padding:20px;background:#f6f8fb;color:#102a43;">
<main style='max-width:1080px;margin:0 auto;'>
<h1 style='font-size:28px;margin:0 0 16px 0;'>{title}</h1>
<div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;'>{cards_html}</div>
{section_html}
<p style='margin-top:24px;color:#486581;'>本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。</p>
</main>
</body></html>"""


def _future_high_risk_distribution(window_days: int = 30) -> Dict[str, Any]:
    cfg = load_server_config()
    window = _future_followup_window(days=window_days)
    exact_window_days = int(window.get("window_days") or window_days or 30)
    disease_counter: Counter[str] = Counter()
    patient_rows = fetch_rows(
        """
        SELECT DISTINCT p.patient_id, p.disease_tags
        FROM patient_profile p
        JOIN followup_plan f ON p.patient_id = f.patient_id
        WHERE f.priority = 'high'
          AND f.status IN ('pending', 'scheduled')
          AND date(f.followup_date) BETWEEN date(?) AND date(?)
        """,
        [window["start_date"], window["end_date"]],
    )
    for row in patient_rows:
        for tag in [item.strip() for item in str(row.get("disease_tags") or "").split(";") if item.strip() and item.strip() != "nan"]:
            disease_counter[tag] += 1
    rows = [{"disease": disease, "patient_count": count} for disease, count in disease_counter.most_common(12)]
    title = f"未来 {exact_window_days} 天高风险随访患者疾病分布"
    chart_html = _bar_chart_html(title, rows, "disease", "patient_count")
    chart_svg = _bar_chart_svg(
        title,
        f"口径：未来 {exact_window_days} 天 pending/scheduled 且 priority=high 的去重患者疾病标签分布",
        rows,
        "disease",
        "patient_count",
    )
    chart_svg_filename = f"cohort_disease_distribution_{exact_window_days}d.svg"
    chart_svg_path = _write_svg_chart(chart_svg_filename, chart_svg)
    payload = {
        "title": title,
        "window": window,
        "window_days": exact_window_days,
        "cohort_patient_count": len(patient_rows),
        "distribution": rows,
    }
    try:
        graph_payload = _dynamic_subgraph_analysis("请生成高风险患者群体的图谱子图。", {"intent": "dynamic_subgraph_render"})
    except Exception as exc:
        graph_payload = {
            "status": "failed",
            "graph_url": "",
            "graph_service_url": "",
            "errors": [f"高风险患者图谱子图生成失败：{exc}"],
        }
    json_path, html_path, chart_path = _write_analysis_bundle(
        f"analysis_future_{exact_window_days}d_high_risk_followup_disease_distribution",
        payload,
        chart_html,
        f"先通过随访计划筛出未来 {exact_window_days} 天内需要高优先级随访的患者群体，再统计其疾病标签分布。",
        related_links=[
            {
                "label": "高风险患者图谱子图",
                "title": "用于解释未来重点随访高风险患者的关系结构",
                "url": graph_payload.get("graph_url", ""),
                "route_path": graph_payload.get("graph_url", ""),
                "service_url": graph_payload.get("graph_service_url", ""),
            }
        ],
    )
    return {
        "status": "success",
        "question": f"未来 {exact_window_days} 天需要随访的高风险患者的疾病类型分布是什么？",
        "analysis_id": f"analysis_future_{exact_window_days}d_high_risk_followup_disease_distribution",
        "cohort_label": f"future_{exact_window_days}d_high_risk_followup",
        "window": window,
        "window_days": exact_window_days,
        "exact_window_days": exact_window_days,
        "table": {"rows": rows},
        "insight": f"该群体共有 {len(patient_rows)} 名患者，疾病标签以前 12 类分布展示。",
        "analysis_json": json_path,
        "analysis_html": html_path,
        "analysis_chart_html": chart_path,
        "chart_asset_path": chart_svg_path,
        "report_url": public_artifact_url(cfg, f"/artifacts/graph-driven/analysis_future_{exact_window_days}d_high_risk_followup_disease_distribution"),
        "graph_url": graph_payload.get("graph_url"),
        "graph_service_url": graph_payload.get("graph_service_url"),
        "report_service_url": service_artifact_url(cfg, f"/artifacts/graph-driven/analysis_future_{exact_window_days}d_high_risk_followup_disease_distribution"),
        "chart_url": public_artifact_url(cfg, f"/artifacts/graph-driven/analysis_future_{exact_window_days}d_high_risk_followup_disease_distribution_chart"),
        "chart_service_url": service_artifact_url(cfg, f"/artifacts/graph-driven/analysis_future_{exact_window_days}d_high_risk_followup_disease_distribution_chart"),
        "charts": [
            {
                "name": f"未来{exact_window_days}天高风险随访患者疾病分布图",
                "path": chart_svg_path,
                "url": public_artifact_url(cfg, f"/artifacts/charts/{chart_svg_filename}"),
                "png_alias_url": public_artifact_url(cfg, f"/artifacts/charts/cohort_disease_distribution_{exact_window_days}d.png"),
                "service_url": service_artifact_url(cfg, f"/artifacts/charts/{chart_svg_filename}"),
                "type": "bar",
            }
        ],
        "summary_text": f"未来 {exact_window_days} 天高风险随访患者共 {len(patient_rows)} 人，已展示图表、图谱依据和明细表。",
        "answer_guardrail": f"本结果只允许描述未来 {exact_window_days} 天窗口，禁止混入其他天数的图表标题、说明或比较语句。",
    }


def _future_high_risk_followup_count(window_days: int = 30) -> Dict[str, Any]:
    window = _future_followup_window(days=window_days)
    exact_window_days = int(window.get("window_days") or window_days or 30)
    cohort_row = fetch_one(
        """
        SELECT COUNT(DISTINCT patient_id) AS patient_count
        FROM followup_plan
        WHERE priority = 'high'
          AND status IN ('pending', 'scheduled')
          AND date(followup_date) >= date(?)
        """,
        [window["start_date"]],
    )
    row = fetch_one(
        """
        SELECT COUNT(DISTINCT patient_id) AS patient_count
        FROM followup_plan
        WHERE priority = 'high'
          AND status IN ('pending', 'scheduled')
          AND date(followup_date) BETWEEN date(?) AND date(?)
        """,
        [window["start_date"], window["end_date"]],
    )
    future_pool_patient_count = int(cohort_row.get("patient_count", 0) or 0)
    window_patient_count = int(row.get("patient_count", 0) or 0)
    patient_count = window_patient_count
    distribution_payload = _future_high_risk_distribution(window_days=exact_window_days)
    chart_bundle_payload = _future_followup_chart_bundle(
        window_days=exact_window_days,
        priorities=["high"],
        analysis_suffix="high_risk",
        title_prefix="高风险",
    )
    chart_table = chart_bundle_payload.get("table") if isinstance(chart_bundle_payload.get("table"), dict) else {}
    risk_distribution_rows = (
        chart_table.get("risk_distribution_rows") if isinstance(chart_table.get("risk_distribution_rows"), list) else []
    )
    risk_distribution_rows = [{"risk_level": "high", "patient_count": patient_count, "risk_label": "高风险"}]
    trend_rows = chart_table.get("trend_rows") if isinstance(chart_table.get("trend_rows"), list) else []
    return {
        "status": "success",
        "question": f"未来 {exact_window_days} 天需要随访的高风险患者有多少？",
        "intent": "future_n_days_high_risk_followup",
        "analysis_id": f"analysis_future_{exact_window_days}d_high_risk_followup_count",
        "cohort_label": f"future_{exact_window_days}d_high_risk_followup",
        "window": window,
        "window_days": exact_window_days,
        "cohort_patient_count": patient_count,
        "window_scheduled_patient_count": window_patient_count,
        "future_pool_patient_count": future_pool_patient_count,
        "metric": {
            "name": "future_high_risk_followup_patient_count",
            "label": f"未来 {exact_window_days} 天高风险随访患者数",
            "value": patient_count,
            "unit": "人",
        },
        "table": {
            "rows": [
                {"指标": f"未来 {exact_window_days} 天需随访高风险患者总数", "数值": patient_count, "单位": "人"},
                {"指标": "统计窗口", "数值": f"{window['start_date']} 至 {window['end_date']}", "单位": ""},
                {"指标": "从当前日期起所有未来高风险待随访池", "数值": future_pool_patient_count, "单位": "人"},
            ],
            "trend_rows": trend_rows,
            "risk_distribution_rows": risk_distribution_rows,
        },
        "risk_distribution_rows": risk_distribution_rows,
        "summary_text": (
            f"未来 {exact_window_days} 天窗口内需要随访的高风险患者共 {patient_count} 人。"
        ),
        "insight": (
            f"本指标严格按 {window['start_date']} 至 {window['end_date']} 的 {exact_window_days} 天窗口统计，结果为 {patient_count} 人；"
            f"从 {window['start_date']} 起所有未来高风险待随访池共有 {future_pool_patient_count} 人，两个口径不混用。"
        ),
        "final_answer_lock": (
            f"未来 {exact_window_days} 天需要随访的高风险患者数 = {patient_count} 人；"
            f"统计窗口为 {window['start_date']} 至 {window['end_date']}。"
            "最终回答和表格必须使用这个窗口值，禁止改写成全体患者数、风险分层总人数或历史模板值。"
        ),
        "report_route_path": chart_bundle_payload.get("report_route_path"),
        "report_url": chart_bundle_payload.get("report_url"),
        "report_service_url": chart_bundle_payload.get("report_service_url"),
        "chart_route_path": chart_bundle_payload.get("chart_route_path"),
        "chart_url": chart_bundle_payload.get("chart_url"),
        "chart_service_url": chart_bundle_payload.get("chart_service_url"),
        "charts": chart_bundle_payload.get("charts"),
        "table_preview_chart_html": chart_bundle_payload.get("analysis_chart_html"),
        "graph_url": distribution_payload.get("graph_url"),
        "graph_service_url": distribution_payload.get("graph_service_url"),
    }


def _future_followup_chart_bundle(
    window_days: int = 30,
    priorities: List[str] | None = None,
    analysis_suffix: str = "",
    title_prefix: str = "",
) -> Dict[str, Any]:
    cfg = load_server_config()
    window = _future_followup_window(days=window_days)
    exact_window_days = int(window.get("window_days") or window_days or 30)
    normalized_priorities = [str(item).strip().lower() for item in (priorities or []) if str(item).strip()]
    priority_filter_sql = ""
    priority_params: List[Any] = []
    priority_label = ""
    if normalized_priorities:
        placeholders = ",".join(["?"] * len(normalized_priorities))
        priority_filter_sql = f" AND priority IN ({placeholders})"
        priority_params = normalized_priorities
        priority_label = "/".join(_risk_level_label(item) for item in normalized_priorities)
    scope_prefix = f"{title_prefix}随访" if title_prefix else "随访"
    file_prefix = f"{analysis_suffix}_" if analysis_suffix else ""
    analysis_id = f"analysis_future_followup_chart_bundle_{file_prefix}{exact_window_days}d"
    raw_trend_rows = fetch_rows(
        f"""
        SELECT followup_date, COUNT(DISTINCT patient_id) AS patient_count
        FROM followup_plan
        WHERE status IN ('pending', 'scheduled')
          AND date(followup_date) BETWEEN date(?) AND date(?)
          {priority_filter_sql}
        GROUP BY followup_date
        ORDER BY followup_date
        """,
        [window["start_date"], window["end_date"], *priority_params],
    )
    trend_rows = _fill_daily_rows(raw_trend_rows, window["start_date"], window["end_date"])
    pie_rows = fetch_rows(
        f"""
        SELECT priority AS risk_level, COUNT(DISTINCT patient_id) AS patient_count
        FROM followup_plan
        WHERE status IN ('pending', 'scheduled')
          AND date(followup_date) BETWEEN date(?) AND date(?)
          {priority_filter_sql}
        GROUP BY priority
        ORDER BY patient_count DESC, priority ASC
        """,
        [window["start_date"], window["end_date"], *priority_params],
    )
    summary = {
        "scheduled_patient_count": sum(int(item.get("patient_count", 0) or 0) for item in pie_rows),
        "peak_daily_count": max([int(item.get("patient_count", 0) or 0) for item in trend_rows] + [0]),
        "peak_date": max(trend_rows, key=lambda item: int(item.get("patient_count", 0) or 0)).get("followup_date") if trend_rows else window["start_date"],
        "active_days": sum(1 for item in trend_rows if int(item.get("patient_count", 0) or 0) > 0),
    }
    summary["average_daily_count"] = round(
        summary["scheduled_patient_count"] / max(1, summary["active_days"]),
        1,
    )
    line_title = f"未来 {exact_window_days} 天{scope_prefix}人数趋势"
    pie_title = f"未来 {exact_window_days} 天{scope_prefix}患者风险等级分布"
    if priority_label:
        line_subtitle = f"口径：严格统计当前日期起未来 {exact_window_days} 天内 pending/scheduled 且风险等级为 {priority_label} 的每日去重患者数"
        pie_subtitle = f"口径：严格统计当前日期起未来 {exact_window_days} 天内 pending/scheduled 且风险等级为 {priority_label} 的去重患者分布"
    else:
        line_subtitle = f"口径：严格统计当前日期起未来 {exact_window_days} 天内 pending/scheduled 随访计划的每日去重患者数"
        pie_subtitle = f"口径：严格统计当前日期起未来 {exact_window_days} 天内 pending/scheduled 随访计划的去重患者风险等级分布"
    line_svg = _line_chart_svg(line_title, line_subtitle, trend_rows, "followup_date", "patient_count")
    pie_svg = _pie_chart_svg(pie_title, pie_subtitle, pie_rows, "risk_level", "patient_count")
    line_svg_path = _write_svg_chart(
        f"line_followup_trend_{file_prefix}{exact_window_days}d.svg",
        line_svg,
    )
    pie_svg_path = _write_svg_chart(
        f"pie_risk_distribution_{file_prefix}{exact_window_days}d.svg",
        pie_svg,
    )
    line_png_filename = f"line_followup_trend_{file_prefix}{exact_window_days}d.png"
    pie_png_filename = f"pie_risk_distribution_{file_prefix}{exact_window_days}d.png"
    line_png_path = _write_followup_line_png(line_png_filename, line_title, trend_rows, "followup_date", "patient_count")
    pie_png_path = _write_followup_pie_png(pie_png_filename, pie_title, pie_rows, "risk_level", "patient_count")
    trend_table_html = _html_table(
        ["日期", "随访人数"],
        [[str(item.get("followup_date") or ""), str(int(item.get("patient_count", 0) or 0))] for item in trend_rows],
    )
    risk_table_html = _html_table(
        ["风险等级", "患者人数", "占比"],
        [
            [
                _risk_level_label(str(item.get("risk_level") or "")),
                str(int(item.get("patient_count", 0) or 0)),
                f"{(float(item.get('patient_count', 0) or 0) / max(1, summary['scheduled_patient_count'])) * 100:.1f}%",
            ]
            for item in pie_rows
        ],
    )
    access_links_html = (
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;'>"
        f"<a href='{artifact_route_path(f'/artifacts/charts/line_followup_trend_{file_prefix}{exact_window_days}d.svg')}' target='_blank' "
        "style='display:block;background:#f8fbff;border:1px solid #d9e2ec;border-radius:14px;padding:14px 16px;color:#102a43;text-decoration:none;'>"
        "<div style='font-size:13px;color:#486581'>折线图</div><div style='margin-top:6px;font-size:16px;font-weight:700;'>直接查看趋势图</div></a>"
        f"<a href='{artifact_route_path(f'/artifacts/charts/pie_risk_distribution_{file_prefix}{exact_window_days}d.svg')}' target='_blank' "
        "style='display:block;background:#f8fbff;border:1px solid #d9e2ec;border-radius:14px;padding:14px 16px;color:#102a43;text-decoration:none;'>"
        "<div style='font-size:13px;color:#486581'>饼图</div><div style='margin-top:6px;font-size:16px;font-weight:700;'>直接查看风险分布</div></a>"
        f"<a href='{artifact_route_path(f'/artifacts/graph-driven/{analysis_id}')}' target='_blank' "
        "style='display:block;background:#f8fbff;border:1px solid #d9e2ec;border-radius:14px;padding:14px 16px;color:#102a43;text-decoration:none;'>"
        "<div style='font-size:13px;color:#486581'>分析总览</div><div style='margin-top:6px;font-size:16px;font-weight:700;'>打开详情页</div></a>"
        "</div>"
    )
    overview_title = f"未来 {exact_window_days} 天{scope_prefix}图表总览"
    line_content = _png_img_html(f"/artifacts/charts/{line_png_filename}", line_title) if line_png_path else line_svg
    pie_content = _png_img_html(f"/artifacts/charts/{pie_png_filename}", pie_title) if pie_png_path else pie_svg
    chart_html = _chart_bundle_html(
        overview_title,
        [
            {"label": "窗口起始日", "value": window["start_date"]},
            {"label": "窗口结束日", "value": window["end_date"]},
            {"label": f"未来{scope_prefix}患者数", "value": summary["scheduled_patient_count"]},
            {"label": "峰值日患者数", "value": f"{summary['peak_date']} / {summary['peak_daily_count']}"},
        ],
        [
            {"title": "随访趋势折线图", "content": line_content},
            {"title": "风险等级分布", "content": pie_content},
            {"title": "数据明细（每日随访人数）", "content": trend_table_html},
            {"title": "风险等级统计", "content": risk_table_html},
            {"title": "图表入口", "content": access_links_html},
        ],
    )
    payload = {
        "title": overview_title,
        "window": window,
        "window_days": exact_window_days,
        "scheduled_patient_count": summary["scheduled_patient_count"],
        "peak_daily_count": summary["peak_daily_count"],
        "peak_date": summary["peak_date"],
        "active_days": summary["active_days"],
        "average_daily_count": summary["average_daily_count"],
        "trend_rows": trend_rows,
        "risk_distribution_rows": [
            {
                **item,
                "risk_level_label": _risk_level_label(str(item.get("risk_level") or "")),
                "ratio": round(float(item.get("patient_count", 0) or 0) / max(1, summary["scheduled_patient_count"]), 4),
                "ratio_text": f"{(float(item.get('patient_count', 0) or 0) / max(1, summary['scheduled_patient_count'])) * 100:.1f}%",
            }
            for item in pie_rows
        ],
    }
    json_path, html_path, chart_path = _write_analysis_bundle(
        analysis_id,
        payload,
        chart_html,
        (
            f"先定位当前日期起未来 {exact_window_days} 天内仍处于 pending 或 scheduled 状态的随访计划"
            f"{'，并筛选风险等级为 ' + priority_label if priority_label else ''}，再统计每日去重患者数与风险等级分布，并即时生成图表。"
        ),
    )
    payload = {
        "status": "success",
        "question": f"根据未来 {exact_window_days} 天{scope_prefix}人数，绘制折线图，饼状图",
        "analysis_id": analysis_id,
        "window": window,
        "window_days": exact_window_days,
        "exact_window_days": exact_window_days,
        "metric": {
            "name": "future_followup_patient_count",
            "value": summary["scheduled_patient_count"],
            "unit": "人",
        },
        "table": {
            "trend_rows": trend_rows,
            "risk_distribution_rows": payload["risk_distribution_rows"],
        },
        "summary": summary,
        "charts": [
            {
                "name": f"未来{exact_window_days}天{scope_prefix}患者风险等级分布饼状图",
                "path": pie_svg_path,
                "png_path": pie_png_path,
                "route_path": artifact_route_path(f"/artifacts/charts/pie_risk_distribution_{file_prefix}{exact_window_days}d.svg"),
                "png_route_path": artifact_route_path(f"/artifacts/charts/{pie_png_filename}") if pie_png_path else None,
                "url": public_artifact_url(cfg, f"/artifacts/charts/pie_risk_distribution_{file_prefix}{exact_window_days}d.svg"),
                "png_url": public_artifact_url(cfg, f"/artifacts/charts/{pie_png_filename}") if pie_png_path else None,
                "service_url": service_artifact_url(cfg, f"/artifacts/charts/pie_risk_distribution_{file_prefix}{exact_window_days}d.svg"),
                "png_service_url": service_artifact_url(cfg, f"/artifacts/charts/{pie_png_filename}") if pie_png_path else None,
                "type": "pie",
            },
            {
                "name": f"未来{exact_window_days}天{scope_prefix}人数趋势折线图",
                "path": line_svg_path,
                "png_path": line_png_path,
                "route_path": artifact_route_path(f"/artifacts/charts/line_followup_trend_{file_prefix}{exact_window_days}d.svg"),
                "png_route_path": artifact_route_path(f"/artifacts/charts/{line_png_filename}") if line_png_path else None,
                "url": public_artifact_url(cfg, f"/artifacts/charts/line_followup_trend_{file_prefix}{exact_window_days}d.svg"),
                "png_url": public_artifact_url(cfg, f"/artifacts/charts/{line_png_filename}") if line_png_path else None,
                "service_url": service_artifact_url(cfg, f"/artifacts/charts/line_followup_trend_{file_prefix}{exact_window_days}d.svg"),
                "png_service_url": service_artifact_url(cfg, f"/artifacts/charts/{line_png_filename}") if line_png_path else None,
                "type": "line",
            },
        ],
        "analysis_json": json_path,
        "analysis_html": html_path,
        "analysis_chart_html": chart_path,
        "report_route_path": artifact_route_path(f"/artifacts/graph-driven/{analysis_id}"),
        "report_url": public_artifact_url(cfg, f"/artifacts/graph-driven/{analysis_id}"),
        "report_service_url": service_artifact_url(cfg, f"/artifacts/graph-driven/{analysis_id}"),
        "chart_route_path": artifact_route_path(f"/artifacts/graph-driven/{analysis_id}_chart"),
        "chart_url": public_artifact_url(cfg, f"/artifacts/graph-driven/{analysis_id}_chart"),
        "chart_service_url": service_artifact_url(cfg, f"/artifacts/graph-driven/{analysis_id}_chart"),
        "summary_text": f"未来 {exact_window_days} 天共有 {summary['scheduled_patient_count']} 名{scope_prefix}患者，已完成趋势统计并生成图表入口；峰值出现在 {summary['peak_date']}，当天 {summary['peak_daily_count']} 人。",
        "insight": (
            f"已基于 {window['start_date']} 至 {window['end_date']} 的真实未来 {exact_window_days} 天随访计划生成趋势统计图表。"
            f"共有 {summary['scheduled_patient_count']} 名{'指定风险等级' if priority_label else 'pending/scheduled'}患者，峰值日期为 {summary['peak_date']}，当日 {summary['peak_daily_count']} 人。"
        ),
        "answer_guardrail": f"本结果只允许描述未来 {exact_window_days} 天窗口，禁止混入其他天数的图表标题、说明或比较语句。",
    }
    return payload


def _high_salt_bp_abnormal_rate() -> Dict[str, Any]:
    cfg = load_server_config()
    row = fetch_one(
        """
        SELECT
          COUNT(*) AS lab_count,
          ROUND(SUM(CASE WHEN l.abnormal_flag != 'normal' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS abnormal_rate,
          COUNT(DISTINCT s.patient_id) AS patient_count
        FROM lifestyle_record s
        JOIN lab_result l
          ON s.patient_id = l.patient_id AND s.visit_id = l.visit_id
        WHERE s.salt_intake_level = 'high'
          AND l.item_name IN ('systolic_bp', 'diastolic_bp')
        """
    )
    metrics = [
        {"label": "高盐饮食患者数", "value": row.get("patient_count", 0)},
        {"label": "血压检验记录数", "value": row.get("lab_count", 0)},
        {"label": "血压异常比例", "value": row.get("abnormal_rate", 0)},
    ]
    chart_html = _metric_cards_html("高盐饮食与血压异常", metrics)
    payload = {"title": "高盐饮食与血压异常", "metrics": metrics}
    json_path, html_path, chart_path = _write_analysis_bundle(
        "analysis_high_salt_bp_abnormal_rate",
        payload,
        chart_html,
        "先用生活方式记录定位高盐饮食患者，再联接同次随访的血压检验结果，统计异常比例。",
    )
    return {
        "status": "success",
        "question": SPECIAL_CANONICALS["high_salt_bp_abnormal_rate"],
        "analysis_id": "analysis_high_salt_bp_abnormal_rate",
        "metric": {"name": "bp_abnormal_rate", "value": row.get("abnormal_rate"), "unit": "ratio"},
        "table": {"rows": [row]},
        "insight": f"高盐饮食相关患者共有 {row.get('patient_count', 0)} 人，血压异常比例为 {row.get('abnormal_rate')}.",
        "analysis_json": json_path,
        "analysis_html": html_path,
        "analysis_chart_html": chart_path,
        "report_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_high_salt_bp_abnormal_rate"),
        "chart_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_high_salt_bp_abnormal_rate_chart"),
        "summary_text": f"高盐饮食相关患者 {row.get('patient_count', 0)} 人，血压异常比例 {row.get('abnormal_rate')}。",
    }


def _hypertension_diabetes_multi_indicator() -> Dict[str, Any]:
    cfg = load_server_config()
    row = fetch_one(
        """
        SELECT
          COUNT(DISTINCT p.patient_id) AS patient_count,
          ROUND(AVG(CASE WHEN l.item_name = 'hba1c' THEN CAST(l.item_value AS REAL) END), 2) AS avg_hba1c,
          ROUND(AVG(CASE WHEN l.item_name = 'ldl_c' THEN CAST(l.item_value AS REAL) END), 2) AS avg_ldl_c,
          ROUND(SUM(CASE WHEN l.item_name = 'hba1c' AND l.abnormal_flag != 'normal' THEN 1 ELSE 0 END) * 1.0 /
            NULLIF(SUM(CASE WHEN l.item_name = 'hba1c' THEN 1 ELSE 0 END), 0), 4) AS hba1c_abnormal_rate,
          ROUND(SUM(CASE WHEN l.item_name IN ('systolic_bp', 'diastolic_bp') AND l.abnormal_flag != 'normal' THEN 1 ELSE 0 END) * 1.0 /
            NULLIF(SUM(CASE WHEN l.item_name IN ('systolic_bp', 'diastolic_bp') THEN 1 ELSE 0 END), 0), 4) AS bp_abnormal_rate,
          ROUND(SUM(CASE WHEN l.item_name = 'ldl_c' AND l.abnormal_flag != 'normal' THEN 1 ELSE 0 END) * 1.0 /
            NULLIF(SUM(CASE WHEN l.item_name = 'ldl_c' THEN 1 ELSE 0 END), 0), 4) AS ldl_abnormal_rate
        FROM patient_profile p
        JOIN lab_result l ON p.patient_id = l.patient_id
        WHERE lower(p.disease_tags) LIKE '%hypertension%'
          AND lower(p.disease_tags) LIKE '%diabetes%'
        """
    )
    metrics = [
        {"label": "患者数", "value": row.get("patient_count", 0)},
        {"label": "HbA1c 异常率", "value": row.get("hba1c_abnormal_rate", 0)},
        {"label": "血压异常率", "value": row.get("bp_abnormal_rate", 0)},
        {"label": "LDL-C 异常率", "value": row.get("ldl_abnormal_rate", 0)},
        {"label": "平均 HbA1c", "value": row.get("avg_hba1c", 0)},
        {"label": "平均 LDL-C", "value": row.get("avg_ldl_c", 0)},
    ]
    cohort_rows_raw = fetch_rows(
        """
        SELECT
          p.patient_id,
          p.name,
          p.gender,
          p.age,
          p.disease_tags,
          ROUND(AVG(CASE WHEN l.item_name = 'hba1c' THEN CAST(l.item_value AS REAL) END), 2) AS avg_hba1c,
          ROUND(AVG(CASE WHEN l.item_name = 'ldl_c' THEN CAST(l.item_value AS REAL) END), 2) AS avg_ldl_c
        FROM patient_profile p
        LEFT JOIN lab_result l ON p.patient_id = l.patient_id
        WHERE lower(p.disease_tags) LIKE '%hypertension%'
          AND lower(p.disease_tags) LIKE '%diabetes%'
        GROUP BY p.patient_id, p.name, p.gender, p.age, p.disease_tags
        ORDER BY p.patient_id
        """
    )
    cohort_rows = [_humanize_row(dict(item)) for item in cohort_rows_raw]
    export_paths = _write_cohort_exports(
        "analysis_hypertension_diabetes_multi_indicator",
        "高血压合并糖尿病群体全量患者列表",
        cohort_rows,
    )
    chart_html = _metric_cards_html("高血压合并糖尿病群体多指标分析", metrics)
    payload = {"title": "高血压合并糖尿病群体多指标分析", "metrics": metrics}
    from tool_server.kg_tools import kg_subgraph_render

    subgraph = kg_subgraph_render("请生成高血压合并糖尿病群体的图谱子图。", max_nodes=96)
    cohort_preview_rows = cohort_rows[:12]
    cohort_preview_html = _html_table(
        list(cohort_preview_rows[0].keys()) if cohort_preview_rows else ["患者编号", "姓名"],
        [list(map(str, item.values())) for item in cohort_preview_rows] if cohort_preview_rows else [],
    )
    subgraph_route = artifact_route_path(f"/artifacts/subgraphs/{subgraph.get('subgraph_id', '')}") if subgraph.get("subgraph_id") else subgraph.get("html_url", "")
    graph_preview_html = (
        f"<iframe src='{subgraph_route}' style='width:100%;min-height:860px;border:none;border-radius:18px;background:#f8fbff;'></iframe>"
        if subgraph.get("html_url")
        else "<p style='margin:0;color:#486581;'>当前没有可预览的图谱子图，请通过图谱入口单独打开。</p>"
    )
    json_path, html_path, chart_path = _write_analysis_bundle(
        "analysis_hypertension_diabetes_multi_indicator",
        payload,
        chart_html,
        "先用疾病组合圈定高血压合并糖尿病患者，再汇总 HbA1c、血压和 LDL-C 的异常情况。",
        related_links=[
            {
                "label": "图表页",
                "title": "多指标图表总览",
                "url": public_artifact_url(load_server_config(), "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_chart"),
                "service_url": service_artifact_url(load_server_config(), "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_chart"),
                "route_path": artifact_route_path("/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_chart"),
            },
            {
                "label": "图谱子图",
                "title": "实时生成的群体子图",
                "url": subgraph.get("html_url", ""),
                "service_url": subgraph.get("service_html_url", ""),
                "route_path": artifact_route_path(f"/artifacts/subgraphs/{subgraph.get('subgraph_id', '')}") if subgraph.get("subgraph_id") else subgraph.get("html_url", ""),
            },
            {
                "label": "全量患者列表",
                "title": "患者明细表",
                "url": public_artifact_url(load_server_config(), "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_patients"),
                "service_url": service_artifact_url(load_server_config(), "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_patients"),
                "route_path": artifact_route_path("/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_patients"),
            },
            {
                "label": "全量患者 CSV 导出",
                "title": "CSV 导出",
                "url": public_artifact_url(load_server_config(), "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_patients.csv"),
                "service_url": service_artifact_url(load_server_config(), "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_patients.csv"),
                "route_path": artifact_route_path("/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_patients.csv"),
            },
        ],
        embedded_sections=[
            {"title": "图谱预览", "content": graph_preview_html},
            {"title": "全量患者预览（前 12 行）", "content": cohort_preview_html},
        ],
    )
    return {
        "status": "success",
        "question": SPECIAL_CANONICALS["hypertension_diabetes_multi_indicator"],
        "analysis_id": "analysis_hypertension_diabetes_multi_indicator",
        "metric": {"name": "multi_indicator_profile", "value": row},
        "table": {"rows": [row]},
        "insight": "已完成针对高血压合并糖尿病群体的 HbA1c、血压和 LDL-C 指标画像。",
        "analysis_json": json_path,
        "analysis_html": html_path,
        "analysis_chart_html": chart_path,
        "graph_url": subgraph.get("html_url"),
        "graph_service_url": subgraph.get("service_html_url"),
        "subgraph_id": subgraph.get("subgraph_id"),
        "graph_scope": "图谱页面展示的是围绕高血压合并糖尿病群体构建的语义子图，并不会把全部患者节点一次性铺开。",
        "cohort_patient_count": row.get("patient_count", 0),
        "cohort_table_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_patients"),
        "cohort_csv_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_patients.csv"),
        "cohort_table_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_patients"),
        "cohort_csv_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_patients.csv"),
        "report_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator"),
        "report_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator"),
        "chart_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_chart"),
        "chart_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_hypertension_diabetes_multi_indicator_chart"),
        "summary_text": f"高血压合并糖尿病患者统计共 {row.get('patient_count', 0)} 人，已生成图谱预览、患者明细入口和 CSV 导出入口。",
        "cohort_table_path": export_paths["html_path"],
        "cohort_csv_path": export_paths["csv_path"],
    }


def _controlled_metric_payload(question: str) -> Dict[str, Any] | None:
    cfg = load_server_config()
    canonical_question = _normalize_controlled_question(question)
    if canonical_question == "高血压合并糖尿病患者的平均 HbA1c 是多少？":
        row = fetch_one(
            """
            SELECT
              COUNT(DISTINCT p.patient_id) AS patient_count,
              ROUND(AVG(CAST(l.item_value AS REAL)), 2) AS avg_hba1c
            FROM patient_profile p
            JOIN lab_result l ON p.patient_id = l.patient_id
            WHERE lower(p.disease_tags) LIKE '%hypertension%'
              AND lower(p.disease_tags) LIKE '%diabetes%'
              AND l.item_name = 'hba1c'
            """
        )
        metrics = [
            {"label": "患者数", "value": int(row.get("patient_count", 0) or 0)},
            {"label": "平均 HbA1c", "value": row.get("avg_hba1c", 0)},
        ]
        chart_html = _metric_cards_html(canonical_question, metrics)
        payload = {"title": canonical_question, "metrics": metrics}
        json_path, html_path, chart_path = _write_analysis_bundle(
            "analysis_metric_hypertension_diabetes_avg_hba1c",
            payload,
            chart_html,
            "先筛出同时具有高血压和糖尿病标签的患者，再统计其 HbA1c 检验均值。",
        )
        return {
            "status": "success",
            "question": canonical_question,
            "metric": {"name": "avg_hba1c", "label": "平均 HbA1c", "value": row.get("avg_hba1c"), "unit": "%"},
            "table": {"rows": [row], "detail_rows": [_humanize_row(dict(row))]},
            "summary_text": f"高血压合并糖尿病患者共 {int(row.get('patient_count', 0) or 0)} 人，平均 HbA1c 为 {row.get('avg_hba1c')}%。",
            "insight": "该结果来自 patient_profile 与 lab_result 的受控 SQL 联表查询。",
            "analysis_id": "analysis_metric_hypertension_diabetes_avg_hba1c",
            "analysis_json": json_path,
            "analysis_html": html_path,
            "analysis_chart_html": chart_path,
            "report_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_hypertension_diabetes_avg_hba1c"),
            "report_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_hypertension_diabetes_avg_hba1c"),
            "chart_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_hypertension_diabetes_avg_hba1c_chart"),
            "chart_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_hypertension_diabetes_avg_hba1c_chart"),
        }
    if canonical_question == "糖尿病患者的空腹血糖平均值是多少？":
        row = fetch_one(
            """
            SELECT
              COUNT(DISTINCT l.patient_id) AS patient_count,
              COUNT(*) AS row_count,
              ROUND(AVG(CAST(l.item_value AS REAL)), 2) AS avg_fasting_glucose
            FROM lab_result l
            JOIN patient_profile p ON p.patient_id = l.patient_id
            WHERE lower(p.disease_tags) LIKE '%diabetes%'
              AND l.item_name = 'fasting_glucose'
            """
        )
        distribution_rows = fetch_rows(
            """
            SELECT
              CASE
                WHEN CAST(l.item_value AS REAL) < 6.1 THEN '< 6.1 mmol/L'
                WHEN CAST(l.item_value AS REAL) < 7.0 THEN '6.1 - 7.0 mmol/L'
                WHEN CAST(l.item_value AS REAL) < 10.0 THEN '7.0 - 10.0 mmol/L'
                ELSE '>= 10.0 mmol/L'
              END AS glucose_range,
              COUNT(*) AS patient_count
            FROM lab_result l
            JOIN patient_profile p ON p.patient_id = l.patient_id
            WHERE lower(p.disease_tags) LIKE '%diabetes%'
              AND l.item_name = 'fasting_glucose'
            GROUP BY glucose_range
            ORDER BY CASE glucose_range
              WHEN '< 6.1 mmol/L' THEN 1
              WHEN '6.1 - 7.0 mmol/L' THEN 2
              WHEN '7.0 - 10.0 mmol/L' THEN 3
              ELSE 4
            END
            """
        )
        metrics = [
            {"label": "患者数", "value": int(row.get("patient_count", 0) or 0)},
            {"label": "空腹血糖检验记录数", "value": int(row.get("row_count", 0) or 0)},
            {"label": "平均空腹血糖", "value": row.get("avg_fasting_glucose", 0)},
        ]
        distribution_svg = _bar_chart_svg(
            "糖尿病患者空腹血糖分布图",
            "口径：按糖尿病患者 fasting_glucose 检验记录分箱统计",
            distribution_rows,
            "glucose_range",
            "patient_count",
        )
        distribution_svg_path = _write_svg_chart("fasting_glucose_distribution.svg", distribution_svg)
        chart_html = _metric_cards_html(canonical_question, metrics) + distribution_svg
        payload = {"title": canonical_question, "metrics": metrics}
        json_path, html_path, chart_path = _write_analysis_bundle(
            "analysis_metric_diabetes_avg_fpg",
            payload,
            chart_html,
            "先筛出糖尿病患者，再对其空腹血糖检验值求均值。",
        )
        return {
            "status": "success",
            "question": canonical_question,
            "metric": {"name": "avg_fasting_glucose", "label": "平均空腹血糖", "value": row.get("avg_fasting_glucose"), "unit": "mmol/L"},
            "table": {"rows": [row], "detail_rows": [_humanize_row(dict(row))], "distribution_rows": distribution_rows},
            "charts": [
                {
                    "name": "空腹血糖分布图",
                    "path": distribution_svg_path,
                    "url": public_artifact_url(cfg, "/artifacts/charts/fasting_glucose_distribution.svg"),
                    "service_url": service_artifact_url(cfg, "/artifacts/charts/fasting_glucose_distribution.svg"),
                    "type": "bar",
                }
            ],
            "summary_text": f"糖尿病患者共 {int(row.get('patient_count', 0) or 0)} 人，空腹血糖检验 {int(row.get('row_count', 0) or 0)} 条，平均空腹血糖为 {row.get('avg_fasting_glucose')} mmol/L。",
            "insight": "该结果基于糖尿病患者的 fasting_glucose 检验记录计算。",
            "analysis_id": "analysis_metric_diabetes_avg_fpg",
            "analysis_json": json_path,
            "analysis_html": html_path,
            "analysis_chart_html": chart_path,
            "report_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_diabetes_avg_fpg"),
            "report_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_diabetes_avg_fpg"),
            "chart_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_diabetes_avg_fpg_chart"),
            "chart_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_diabetes_avg_fpg_chart"),
        }
    if canonical_question == "高脂血症患者的 LDL-C 异常比例是多少？":
        row = fetch_one(
            """
            SELECT
              COUNT(DISTINCT l.patient_id) AS patient_count,
              COUNT(*) AS row_count,
              ROUND(SUM(CASE WHEN l.abnormal_flag != 'normal' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS ldl_abnormal_rate
            FROM lab_result l
            JOIN patient_profile p ON p.patient_id = l.patient_id
            WHERE lower(p.disease_tags) LIKE '%hyperlipidemia%'
              AND l.item_name = 'ldl_c'
            """
        )
        metrics = [
            {"label": "患者数", "value": int(row.get("patient_count", 0) or 0)},
            {"label": "LDL-C 检验记录数", "value": int(row.get("row_count", 0) or 0)},
            {"label": "LDL-C 异常比例", "value": row.get("ldl_abnormal_rate", 0)},
        ]
        chart_html = _metric_cards_html(canonical_question, metrics)
        payload = {"title": canonical_question, "metrics": metrics}
        json_path, html_path, chart_path = _write_analysis_bundle(
            "analysis_metric_hyperlipidemia_ldl_abnormal_rate",
            payload,
            chart_html,
            "先筛出高脂血症患者，再统计其 LDL-C 检验记录中的异常比例。",
        )
        return {
            "status": "success",
            "question": canonical_question,
            "metric": {"name": "ldl_abnormal_rate", "label": "LDL-C 异常比例", "value": row.get("ldl_abnormal_rate"), "unit": "ratio"},
            "table": {"rows": [row], "detail_rows": [_humanize_row(dict(row))]},
            "summary_text": f"高脂血症患者共 {int(row.get('patient_count', 0) or 0)} 人，LDL-C 检验 {int(row.get('row_count', 0) or 0)} 条，异常比例为 {row.get('ldl_abnormal_rate')}.",
            "insight": "该比例来自高脂血症患者所有 LDL-C 检验记录的异常标记统计。",
            "analysis_id": "analysis_metric_hyperlipidemia_ldl_abnormal_rate",
            "analysis_json": json_path,
            "analysis_html": html_path,
            "analysis_chart_html": chart_path,
            "report_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_hyperlipidemia_ldl_abnormal_rate"),
            "report_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_hyperlipidemia_ldl_abnormal_rate"),
            "chart_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_hyperlipidemia_ldl_abnormal_rate_chart"),
            "chart_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_hyperlipidemia_ldl_abnormal_rate_chart"),
        }
    if canonical_question == "BMI 超标患者有多少人？":
        row = fetch_one(
            """
            SELECT COUNT(DISTINCT patient_id) AS overweight_patient_count
            FROM patient_profile
            WHERE CAST(bmi AS REAL) >= 24
            """
        )
        metrics = [{"label": "BMI 超标患者数", "value": int(row.get("overweight_patient_count", 0) or 0)}]
        chart_html = _metric_cards_html(canonical_question, metrics)
        payload = {"title": canonical_question, "metrics": metrics}
        json_path, html_path, chart_path = _write_analysis_bundle(
            "analysis_metric_bmi_overweight_count",
            payload,
            chart_html,
            "按 patient_profile 主表中的 BMI 字段统计 BMI≥24 的患者人数。",
        )
        return {
            "status": "success",
            "question": canonical_question,
            "metric": {"name": "overweight_patient_count", "label": "BMI 超标患者数", "value": row.get("overweight_patient_count"), "unit": "人"},
            "table": {"rows": [row], "detail_rows": [_humanize_row(dict(row))]},
            "summary_text": f"当前 BMI 超标（BMI≥24）的患者共 {int(row.get('overweight_patient_count', 0) or 0)} 人。",
            "insight": "该结果来自 patient_profile 主表中的 BMI 字段。",
            "analysis_id": "analysis_metric_bmi_overweight_count",
            "analysis_json": json_path,
            "analysis_html": html_path,
            "analysis_chart_html": chart_path,
            "report_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_bmi_overweight_count"),
            "report_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_bmi_overweight_count"),
            "chart_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_bmi_overweight_count_chart"),
            "chart_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_bmi_overweight_count_chart"),
        }
    if canonical_question == "不同风险等级患者的 HbA1c 平均值是多少？":
        rows = fetch_rows(
            """
            SELECT
              prs.risk_level,
              COUNT(DISTINCT prs.patient_id) AS patient_count,
              ROUND(AVG(CAST(l.item_value AS REAL)), 2) AS avg_hba1c
            FROM patient_risk_score prs
            JOIN lab_result l
              ON prs.patient_id = l.patient_id
             AND prs.visit_id = l.visit_id
            WHERE l.item_name = 'hba1c'
            GROUP BY prs.risk_level
            ORDER BY CASE prs.risk_level WHEN 'high' THEN 1 WHEN 'medium' THEN 2 WHEN 'low' THEN 3 ELSE 4 END
            """
        )
        detail_rows = [
            {
                "风险等级": _risk_level_label(str(item.get("risk_level") or "")),
                "患者数": int(item.get("patient_count", 0) or 0),
                "平均 HbA1c": item.get("avg_hba1c"),
            }
            for item in rows
        ]
        chart_html = _html_table(
            ["风险等级", "患者数", "平均 HbA1c"],
            [[str(item["风险等级"]), str(item["患者数"]), str(item["平均 HbA1c"])] for item in detail_rows],
        )
        payload = {"title": canonical_question, "rows": detail_rows}
        json_path, html_path, chart_path = _write_analysis_bundle(
            "analysis_metric_risk_level_avg_hba1c",
            payload,
            chart_html,
            "先按 patient_risk_score 分层，再联接同次 visit 的 HbA1c 检验结果计算均值。",
        )
        return {
            "status": "success",
            "question": canonical_question,
            "metric": {"name": "avg_hba1c", "label": "不同风险等级 HbA1c 平均值", "value": detail_rows},
            "table": {"rows": detail_rows, "detail_rows": detail_rows},
            "summary_text": "已按高/中/低风险分层返回 HbA1c 平均值和对应患者数。",
            "insight": "该结果用于比较不同风险等级患者的 HbA1c 控制情况。",
            "analysis_id": "analysis_metric_risk_level_avg_hba1c",
            "analysis_json": json_path,
            "analysis_html": html_path,
            "analysis_chart_html": chart_path,
            "report_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_risk_level_avg_hba1c"),
            "report_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_risk_level_avg_hba1c"),
            "chart_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_risk_level_avg_hba1c_chart"),
            "chart_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_risk_level_avg_hba1c_chart"),
        }
    if canonical_question == "最近 6 个月 HbA1c 异常人数趋势如何？":
        return _hba1c_abnormal_trend_payload(6, canonical_question)
    if canonical_question == "最近 6 个月血压异常人数趋势如何？":
        rows = fetch_rows(
            """
            SELECT
              strftime('%Y-%m', COALESCE(test_date, record_time)) AS month,
              COUNT(DISTINCT patient_id) AS abnormal_patient_count
            FROM lab_result
            WHERE item_name IN ('systolic_bp', 'diastolic_bp')
              AND abnormal_flag != 'normal'
              AND date(COALESCE(test_date, record_time)) >= date('now','start of month','-5 months')
            GROUP BY strftime('%Y-%m', COALESCE(test_date, record_time))
            ORDER BY month
            """
        )
        bundle = _trend_analysis_bundle(
            analysis_id="analysis_trend_bp_abnormal_6m",
            title="最近 6 个月血压异常人数趋势",
            subtitle="口径：按自然月统计血压异常患者去重人数",
            rows=rows,
            value_key="abnormal_patient_count",
            summary_cards=[
                {"label": "统计月份数", "value": len(rows)},
                {"label": "最新月份", "value": rows[-1]["month"] if rows else "N/A"},
                {"label": "最新异常人数", "value": int(rows[-1]["abnormal_patient_count"]) if rows else 0},
                {"label": "峰值月份", "value": max(rows, key=lambda item: int(item.get("abnormal_patient_count", 0) or 0)).get("month") if rows else "N/A"},
            ],
            explanation="按月份统计血压异常患者人数趋势，便于观察最近半年血压波动情况。",
        )
        return {
            "status": "success",
            "question": canonical_question,
            "metric": {"name": "abnormal_patient_count", "label": "血压异常人数", "value": int(rows[-1]["abnormal_patient_count"]) if rows else 0, "unit": "人"},
            "table": {"rows": rows, "detail_rows": rows},
            "summary_text": f"最近 6 个月血压异常人数趋势已生成，最新月份 {rows[-1]['month'] if rows else 'N/A'} 为 {int(rows[-1]['abnormal_patient_count']) if rows else 0} 人。",
            "insight": "已生成最近 6 个月血压异常人数折线图和趋势明细。",
            **bundle,
            "charts": [{"name": "最近6个月血压异常人数趋势图", "url": bundle["image_url"], "service_url": bundle["image_service_url"], "type": "line"}],
        }
    if canonical_question == "高血压患者最近半年的血压趋势如何？":
        systolic_rows = fetch_rows(
            """
            SELECT
              strftime('%Y-%m', COALESCE(l.test_date, l.record_time)) AS month,
              ROUND(AVG(CAST(l.item_value AS REAL)), 2) AS avg_systolic_bp
            FROM lab_result l
            JOIN patient_profile p ON p.patient_id = l.patient_id
            WHERE lower(p.disease_tags) LIKE '%hypertension%'
              AND l.item_name = 'systolic_bp'
              AND date(COALESCE(l.test_date, l.record_time)) >= date('now','start of month','-5 months')
            GROUP BY strftime('%Y-%m', COALESCE(l.test_date, l.record_time))
            ORDER BY month
            """
        )
        diastolic_rows = fetch_rows(
            """
            SELECT
              strftime('%Y-%m', COALESCE(l.test_date, l.record_time)) AS month,
              ROUND(AVG(CAST(l.item_value AS REAL)), 2) AS avg_diastolic_bp
            FROM lab_result l
            JOIN patient_profile p ON p.patient_id = l.patient_id
            WHERE lower(p.disease_tags) LIKE '%hypertension%'
              AND l.item_name = 'diastolic_bp'
              AND date(COALESCE(l.test_date, l.record_time)) >= date('now','start of month','-5 months')
            GROUP BY strftime('%Y-%m', COALESCE(l.test_date, l.record_time))
            ORDER BY month
            """
        )
        by_month: Dict[str, Dict[str, Any]] = {}
        for row in systolic_rows:
            by_month[str(row.get("month"))] = {"month": row.get("month"), "avg_systolic_bp": row.get("avg_systolic_bp"), "avg_diastolic_bp": None}
        for row in diastolic_rows:
            month = str(row.get("month"))
            by_month.setdefault(month, {"month": month, "avg_systolic_bp": None, "avg_diastolic_bp": None})
            by_month[month]["avg_diastolic_bp"] = row.get("avg_diastolic_bp")
        rows = [by_month[key] for key in sorted(by_month.keys())]
        systolic_svg = _line_chart_svg("最近 6 个月高血压患者平均收缩压趋势", "口径：按自然月统计高血压患者收缩压均值", rows, "month", "avg_systolic_bp")
        diastolic_svg = _line_chart_svg("最近 6 个月高血压患者平均舒张压趋势", "口径：按自然月统计高血压患者舒张压均值", rows, "month", "avg_diastolic_bp")
        _write_svg_chart("analysis_trend_hypertension_systolic_6m.svg", systolic_svg)
        _write_svg_chart("analysis_trend_hypertension_diastolic_6m.svg", diastolic_svg)
        chart_html = _chart_bundle_html(
            "高血压患者最近半年的血压趋势",
            [
                {"label": "统计月份数", "value": len(rows)},
                {"label": "最新月份", "value": rows[-1]["month"] if rows else "N/A"},
                {"label": "最新平均收缩压", "value": rows[-1]["avg_systolic_bp"] if rows else "N/A"},
                {"label": "最新平均舒张压", "value": rows[-1]["avg_diastolic_bp"] if rows else "N/A"},
            ],
            [
                {"title": "平均收缩压趋势", "content": systolic_svg},
                {"title": "平均舒张压趋势", "content": diastolic_svg},
                {"title": "趋势明细", "content": _html_table(["月份", "平均收缩压", "平均舒张压"], [[str(item.get("month")), str(item.get("avg_systolic_bp")), str(item.get("avg_diastolic_bp"))] for item in rows])},
            ],
        )
        payload = {"title": "高血压患者最近半年的血压趋势", "rows": rows}
        json_path, html_path, chart_path = _write_analysis_bundle(
            "analysis_trend_hypertension_bp_6m",
            payload,
            chart_html,
            "按月份统计高血压患者近 6 个月平均收缩压与平均舒张压变化趋势。",
        )
        return {
            "status": "success",
            "question": canonical_question,
            "metric": {"name": "blood_pressure_trend", "label": "血压趋势", "value": rows[-1] if rows else {}, "unit": "mmHg"},
            "table": {"rows": rows, "detail_rows": rows},
            "summary_text": "已生成高血压患者最近 6 个月的平均收缩压与平均舒张压趋势图。",
            "insight": "该趋势用于观察高血压患者近半年血压控制的月度变化。",
            "analysis_id": "analysis_trend_hypertension_bp_6m",
            "analysis_json": json_path,
            "analysis_html": html_path,
            "analysis_chart_html": chart_path,
            "report_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_trend_hypertension_bp_6m"),
            "report_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_trend_hypertension_bp_6m"),
            "chart_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_trend_hypertension_bp_6m_chart"),
            "chart_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_trend_hypertension_bp_6m_chart"),
            "charts": [
                {"name": "高血压患者平均收缩压趋势图", "url": public_artifact_url(cfg, "/artifacts/charts/analysis_trend_hypertension_systolic_6m.svg"), "service_url": service_artifact_url(cfg, "/artifacts/charts/analysis_trend_hypertension_systolic_6m.svg"), "type": "line"},
                {"name": "高血压患者平均舒张压趋势图", "url": public_artifact_url(cfg, "/artifacts/charts/analysis_trend_hypertension_diastolic_6m.svg"), "service_url": service_artifact_url(cfg, "/artifacts/charts/analysis_trend_hypertension_diastolic_6m.svg"), "type": "line"},
            ],
        }
    if canonical_question == "糖尿病患者最近 3 个月 HbA1c 异常比例是多少？":
        row = fetch_one(
            """
            SELECT
              COUNT(DISTINCT l.patient_id) AS patient_count,
              COUNT(*) AS row_count,
              ROUND(SUM(CASE WHEN l.abnormal_flag != 'normal' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS hba1c_abnormal_rate_3m
            FROM lab_result l
            JOIN patient_profile p ON p.patient_id = l.patient_id
            WHERE lower(p.disease_tags) LIKE '%diabetes%'
              AND l.item_name = 'hba1c'
              AND date(COALESCE(l.test_date, l.record_time)) >= date('now','start of month','-2 months')
            """
        )
        metrics = [
            {"label": "患者数", "value": int(row.get("patient_count", 0) or 0)},
            {"label": "HbA1c 检验记录数", "value": int(row.get("row_count", 0) or 0)},
            {"label": "近 3 个月 HbA1c 异常比例", "value": row.get("hba1c_abnormal_rate_3m", 0)},
        ]
        chart_html = _metric_cards_html(canonical_question, metrics)
        payload = {"title": canonical_question, "metrics": metrics}
        json_path, html_path, chart_path = _write_analysis_bundle(
            "analysis_metric_diabetes_hba1c_abnormal_rate_3m",
            payload,
            chart_html,
            "筛出糖尿病患者近 3 个月的 HbA1c 检验记录，统计异常比例。",
        )
        return {
            "status": "success",
            "question": canonical_question,
            "metric": {"name": "hba1c_abnormal_rate_3m", "label": "近 3 个月 HbA1c 异常比例", "value": row.get("hba1c_abnormal_rate_3m"), "unit": "ratio"},
            "table": {"rows": [row], "detail_rows": [_humanize_row(dict(row))]},
            "summary_text": f"糖尿病患者近 3 个月 HbA1c 检验 {int(row.get('row_count', 0) or 0)} 条，异常比例为 {row.get('hba1c_abnormal_rate_3m')}.",
            "insight": "该结果可用于观察糖尿病患者最近 3 个月的糖控异常水平。",
            "analysis_id": "analysis_metric_diabetes_hba1c_abnormal_rate_3m",
            "analysis_json": json_path,
            "analysis_html": html_path,
            "analysis_chart_html": chart_path,
            "report_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_diabetes_hba1c_abnormal_rate_3m"),
            "report_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_diabetes_hba1c_abnormal_rate_3m"),
            "chart_url": public_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_diabetes_hba1c_abnormal_rate_3m_chart"),
            "chart_service_url": service_artifact_url(cfg, "/artifacts/graph-driven/analysis_metric_diabetes_hba1c_abnormal_rate_3m_chart"),
        }
    return None


def graph_sql_joint_analysis(question: str) -> Dict[str, Any]:
    cfg = load_server_config()
    planner = plan_query(question).to_dict()
    planner_log_path = _write_planner_log(question, planner, {"entrypoint": "graph_sql_joint_analysis"})
    if "为什么" in question and "图谱" in question and "患者" in question:
        return {
            "status": "success",
            "question": question,
            "sql_result_summary": "SQL 统计用于全量计数，图谱用于关系结构展示。",
            "sql": None,
            "graph_url": public_artifact_url(cfg, "/artifacts/graph.html"),
            "graph_service_url": service_artifact_url(cfg, "/artifacts/graph.html"),
            "graph_node_count": None,
            "graph_edge_count": None,
            "display_policy": "图谱展示语义核心节点与示例患者节点，全量统计以 SQL 结果和导出表为准。",
            "limitations": ["为避免节点过多影响可读性，图谱不会一次性展开全部患者节点。"],
            "summary": "SQL 统计用于全量患者数，图谱用于解释关系结构。若把全部患者节点同时铺开，页面会失去可读性，因此默认只展示语义核心节点和部分示例患者。",
            "summary_text": "图谱不是全量患者平铺页，而是关系结构解释页；全量患者数量以 SQL 和导出表为准。",
            "planner": planner,
            "planner_log_path": planner_log_path,
            "safety_note": safety_note(cfg),
        }
    candidate_sql = _candidate_for_plan(question, planner)
    sql_payload = (
        _execute_safe_sql_question(
            question,
            candidate_sql,
            {
                "time_window": planner.get("time_window"),
                "disease_filters": planner.get("disease_filters"),
                "risk_filters": planner.get("risk_filters"),
            },
        )
        if candidate_sql
        else {
            "status": "failed",
            "summary": "当前没有可直接执行的统计 SQL。",
            "sql": None,
            "table": {"rows": [], "row_count": 0},
            "warnings": [],
            "errors": ["No safe SQL candidate generated."],
        }
    )
    from tool_server.kg_tools import kg_subgraph_render

    if ("冠心病" in question or "coronary" in question.lower()) and ("高脂血症" in question or "hyperlipidemia" in question.lower()):
        graph_query = "请生成冠心病合并高脂血症群体的图谱子图。"
    elif "hypertension" in planner.get("disease_filters", []) and "diabetes" in planner.get("disease_filters", []):
        graph_query = "请生成高血压合并糖尿病群体的图谱子图。"
    elif "high" in planner.get("risk_filters", []) or "高风险" in question:
        graph_query = "请生成高风险患者群体的图谱子图。"
    elif "hypertension" in planner.get("disease_filters", []):
        graph_query = "请生成高血压患者群体的图谱子图。"
    elif "diabetes" in planner.get("disease_filters", []):
        graph_query = "请生成糖尿病患者群体的图谱子图。"
    else:
        graph_query = question
    subgraph = kg_subgraph_render(graph_query, max_nodes=96)
    return {
        "status": "success" if sql_payload.get("status") == "success" or subgraph.get("status") == "success" else "failed",
        "question": question,
        "sql_result_summary": sql_payload.get("summary"),
        "sql": sql_payload.get("sql"),
        "graph_url": subgraph.get("html_url", public_artifact_url(cfg, "/artifacts/graph.html")),
        "graph_service_url": subgraph.get("service_html_url", service_artifact_url(cfg, "/artifacts/graph.html")),
        "graph_node_count": subgraph.get("node_count"),
        "graph_edge_count": subgraph.get("edge_count"),
        "display_policy": "图谱展示语义核心节点与示例患者节点，全量统计以 SQL 结果和导出表为准。",
        "limitations": ["为避免节点过多影响可读性，图谱不会一次性展开全部患者节点。"],
        "metric_definition": sql_payload.get("metric_definition"),
        "table": sql_payload.get("table", {"rows": [], "row_count": 0}),
        "result_artifacts": sql_payload.get("result_artifacts"),
        "result_table_url": sql_payload.get("result_table_url"),
        "result_table_service_url": sql_payload.get("result_table_url", "").replace(public_artifact_url(cfg, "/artifacts/open-nl2sql/"), service_artifact_url(cfg, "/artifacts/open-nl2sql/")) if sql_payload.get("result_table_url") else None,
        "summary": f"{sql_payload.get('summary', '已完成统计。')} 图谱入口已一并提供，用于解释群体结构和关系。",
        "summary_text": f"{sql_payload.get('summary', '已完成统计。')} 同时已提供图谱入口和结果表入口，便于同时查看全量统计、患者明细和关系结构。",
        "planner": planner,
        "planner_log_path": planner_log_path,
        "safety_note": safety_note(cfg),
    }


def graph_driven_analysis(question: str) -> Dict[str, Any]:
    cfg = load_server_config()
    rewritten = rewrite_question(question)
    planner = plan_query(question, {"canonical_question": rewritten.get("question")}).to_dict()
    planner_log_path = _write_planner_log(question, planner, {"entrypoint": "graph_driven_analysis"})
    canonical_id = rewritten.get("canonical_id")
    window_days = int(rewritten.get("window_days") or extract_future_window_days(rewritten.get("question", question), default=30))
    if "未来" in question and "高风险" in question and any(token in question for token in ("图谱", "图表", "依据")):
        payload = _future_high_risk_distribution(window_days=window_days)
    elif canonical_id == "future_30d_high_risk_followup_disease_distribution":
        payload = _future_high_risk_distribution(window_days=window_days)
    elif canonical_id == "high_salt_bp_abnormal_rate":
        payload = _high_salt_bp_abnormal_rate()
    elif canonical_id == "hypertension_diabetes_multi_indicator":
        payload = _hypertension_diabetes_multi_indicator()
    elif canonical_id == "future_followup_chart_bundle":
        payload = _future_followup_chart_bundle(window_days=window_days)
    elif planner.get("intent") == "future_followup_chart":
        payload = _future_followup_chart_bundle(window_days=int((planner.get("time_window") or {}).get("value", window_days or 30)))
    elif planner.get("intent") == "graph_sql_joint_analysis":
        payload = graph_sql_joint_analysis(question)
    elif (
        ("合并" in question or "共同" in question)
        and any(token in question for token in ("图谱", "关系", "风险事件", "用药", "CSV", "明细"))
    ):
        payload = graph_sql_joint_analysis(question)
    elif _looks_like_subgraph_request(question, planner):
        payload = _dynamic_subgraph_analysis(question, planner)
    else:
        return {
            "status": "failed",
            "question": question,
            "rewritten_question": rewritten.get("question"),
            "errors": ["当前问题未命中稳定的图谱驱动分析或联合分析路由。"],
            "planner": planner,
            "planner_log_path": planner_log_path,
            "safety_note": safety_note(cfg),
        }
    if "analysis_html" in payload:
        payload["report_url"] = public_artifact_url(cfg, f"/artifacts/graph-driven/{Path(payload['analysis_html']).stem}")
    if "analysis_chart_html" in payload:
        payload["chart_url"] = public_artifact_url(cfg, f"/artifacts/graph-driven/{Path(payload['analysis_chart_html']).stem}")
    payload["rewritten_question"] = rewritten.get("question")
    payload["planner"] = planner
    payload["planner_log_path"] = planner_log_path
    payload["safety_note"] = safety_note(cfg)
    if payload.get("cohort_label"):
        _persist_cohort_context(
            payload,
            question=question,
            cohort_label=str(payload.get("cohort_label")),
            cohort_type="graph_driven_cohort",
        )
    return payload


def open_analysis_query(question: str) -> Dict[str, Any]:
    cfg = load_server_config()
    cohort_resolution = resolve_active_cohort(question)
    if cohort_resolution.get("status") == "needs_clarification":
        return {
            "status": "failed",
            "clarification_required": True,
            "question": question,
            "errors": [cohort_resolution.get("question")],
            "context_mode": cohort_resolution.get("context_mode"),
            "conversation_id": cohort_resolution.get("conversation_id"),
            "safety_note": safety_note(cfg),
        }
    last_cohort = cohort_resolution.get("cohort") or {}
    early_hba1c_trend_months = _extract_hba1c_abnormal_trend_months(question)
    if early_hba1c_trend_months is not None:
        normalized_question = _normalize_controlled_question(question)
        planner = plan_query(question, {"canonical_question": normalized_question}).to_dict()
        planner_log_path = _write_planner_log(question, planner, {"entrypoint": "open_analysis_query"})
        schema_links = build_schema_links(normalized_question)
        route = route_intent(normalized_question)
        candidate = build_sql_candidate(normalized_question, route, schema_links)
        routed = route_agent_intent({"query": question, "last_context": last_cohort})
        payload = _hba1c_abnormal_trend_payload(early_hba1c_trend_months, normalized_question)
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": f"analysis_trend_hba1c_abnormal_{early_hba1c_trend_months}m",
                "matched_id": f"dynamic_hba1c_abnormal_trend_{early_hba1c_trend_months}m",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
                "safety_note": safety_note(cfg),
            }
        )
        return payload

    # Keep calibrated ChronicCare business routes ahead of Open SQL.  Open SQL is
    # useful as a broad fallback, but these intents have stricter cohort
    # definitions and generated artifacts that the frontend regression set relies
    # on.
    direct_controlled_question = _normalize_controlled_question(question)
    normalized_question_for_priority = (
        direct_controlled_question
        if direct_controlled_question in QUESTION_METADATA
        else _normalize_controlled_question(question)
    )
    priority_window_days = int(extract_future_window_days(normalized_question_for_priority, default=30))
    priority_route = route_agent_intent({"query": question, "last_context": last_cohort})
    if (
        "chroniccare" in normalized_question_for_priority.lower()
        or "系统" in normalized_question_for_priority
    ) and any(token in normalized_question_for_priority for token in ("运行状态", "系统状态", "健康", "正常", "是否正常")):
        schema_links = build_schema_links(normalized_question_for_priority)
        route = route_intent(normalized_question_for_priority)
        candidate = build_sql_candidate(normalized_question_for_priority, route, schema_links)
        planner = plan_query(question, {"canonical_question": normalized_question_for_priority}).to_dict()
        planner_log_path = _write_planner_log(question, planner, {"entrypoint": "open_analysis_query_priority"})
        payload = _system_status_payload(normalized_question_for_priority)
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question_for_priority,
                "canonical_id": "system_status",
                "matched_id": "system_status",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": priority_route,
                "safety_note": safety_note(cfg),
            }
        )
        return payload
    if _looks_like_disease_combination_question(normalized_question_for_priority):
        schema_links = build_schema_links(normalized_question_for_priority)
        route = route_intent(normalized_question_for_priority)
        candidate = build_sql_candidate(normalized_question_for_priority, route, schema_links)
        planner = plan_query(question, {"canonical_question": normalized_question_for_priority}).to_dict()
        planner_log_path = _write_planner_log(question, planner, {"entrypoint": "open_analysis_query_priority"})
        payload = _disease_combination_distribution_payload(normalized_question_for_priority)
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question_for_priority,
                "canonical_id": "disease_combination_distribution",
                "matched_id": "disease_combination_distribution",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": priority_route,
            }
        )
        _persist_cohort_context(
            payload,
            question=question,
            cohort_label="multimorbidity_distribution",
            cohort_type="disease_combination_distribution",
        )
        return payload
    if (
        "未来" in normalized_question_for_priority
        and "随访" in normalized_question_for_priority
        and (
            "高风险" in normalized_question_for_priority
            or priority_route.get("intent") == "future_n_days_high_risk_followup"
            or not _contains_explicit_disease_name(normalized_question_for_priority)
        )
        and (
            any(token in normalized_question_for_priority for token in FOLLOWUP_DYNAMIC_VISUAL_HINTS)
            or any(token in normalized_question_for_priority for token in FOLLOWUP_DYNAMIC_COUNT_HINTS)
        )
    ):
        schema_links = build_schema_links(normalized_question_for_priority)
        route = route_intent(normalized_question_for_priority)
        candidate = build_sql_candidate(normalized_question_for_priority, route, schema_links)
        planner = plan_query(question, {"canonical_question": normalized_question_for_priority}).to_dict()
        planner_log_path = _write_planner_log(question, planner, {"entrypoint": "open_analysis_query_priority"})
        payload = _future_high_risk_followup_count(window_days=priority_window_days)
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question_for_priority,
                "canonical_id": "future_n_days_high_risk_followup",
                "matched_id": f"future_high_risk_followup_{priority_window_days}d",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": priority_route,
                "safety_note": safety_note(cfg),
            }
        )
        _persist_cohort_context(
            payload,
            question=question,
            cohort_label=str(payload.get("cohort_label")),
            cohort_type="future_high_risk_followup",
        )
        return payload
    if (has_pronoun_reference(question) or "前一个群体" in question) and last_cohort:
        if "high_risk_followup" in str(last_cohort.get("cohort_label", "")):
            exact_window_days = int(last_cohort.get("window_days") or priority_window_days or 30)
            if _looks_like_cohort_risk_question(question):
                patient_count = int(last_cohort.get("cohort_patient_count") or last_cohort.get("patient_count") or 0)
                payload = {
                    "status": "success",
                    "question": question,
                    "cohort_label": str(last_cohort.get("cohort_label")),
                    "cohort_patient_count": patient_count,
                    "window_days": exact_window_days,
                    "table": {"rows": [
                        {"风险等级": "高风险", "患者人数": patient_count, "占比": 1.0},
                        {"风险等级": "中风险", "患者人数": 0, "占比": 0.0},
                        {"风险等级": "低风险", "患者人数": 0, "占比": 0.0},
                    ]},
                    "summary_text": f"这里的群体指同一会话上一轮未来 {exact_window_days} 天高风险随访患者，共 {patient_count} 人；该群体本身即为高风险人群，因此风险等级分布为高风险 100%。",
                    "safety_note": safety_note(cfg),
                }
            else:
                payload = _future_high_risk_distribution(window_days=exact_window_days)
            payload.update({
                "original_question": question,
                "canonical_id": "cohort_disease_distribution",
                "matched_id": "cohort_disease_distribution",
                "cohort_context": last_cohort,
                "context_mode": cohort_resolution.get("context_mode"),
                "conversation_id": cohort_resolution.get("conversation_id"),
                "fallback_used": False,
            })
            return attach_analysis_context(payload, AnalysisContext.current().with_window(exact_window_days))

    open_sql_payload = run_open_sql_query(question, prefer_llm=True, allow_chart=True, last_context=last_cohort)
    if open_sql_payload.get("status") == "success":
        open_sql_payload.update(
            {
                "original_question": question,
                "rewritten_question": open_sql_payload.get("question", question),
                "canonical_id": "open_sql_query",
                "matched_id": open_sql_payload.get("template_id") or open_sql_payload.get("intent"),
                "fallback_used": open_sql_payload.get("stage") == "fallback",
            }
        )
        return open_sql_payload
    direct_rule_payload = run_question_pipeline(question, last_context=last_cohort)
    if direct_rule_payload is not None:
        classified = (direct_rule_payload.get("rule_pipeline") or {}).get("classification") or {}
        routed_plan = (direct_rule_payload.get("rule_pipeline") or {}).get("plan") or {}
        direct_rule_payload.update(
            {
                "original_question": question,
                "rewritten_question": question,
                "canonical_id": direct_rule_payload.get("canonical_id") or routed_plan.get("intent"),
                "matched_id": direct_rule_payload.get("matched_id") or routed_plan.get("intent"),
                "agent_route": {
                    "intent": routed_plan.get("intent"),
                    "tool": routed_plan.get("tool"),
                    "normalized_entities": routed_plan.get("normalized_entities") or {},
                    "confidence": classified.get("confidence"),
                    "reason": classified.get("reason"),
                    "executor": routed_plan.get("executor"),
                },
                "fallback_used": False,
                "safety_note": direct_rule_payload.get("safety_note") or safety_note(cfg),
            }
        )
        return direct_rule_payload
    rewritten = rewrite_question(question)
    planner = plan_query(question, {"canonical_question": rewritten.get("question")}).to_dict()
    planner_log_path = _write_planner_log(question, planner, {"entrypoint": "open_analysis_query"})
    canonical_id = rewritten.get("canonical_id")
    window_days = int(rewritten.get("window_days") or extract_future_window_days(rewritten.get("question", question), default=30))
    normalized_question = (
        direct_controlled_question
        if direct_controlled_question in QUESTION_METADATA
        else _normalize_controlled_question(str(rewritten.get("question", question) or question))
    )
    schema_links = build_schema_links(normalized_question)
    route = route_intent(normalized_question)
    routed = route_agent_intent({"query": question, "last_context": last_cohort})
    candidate = build_sql_candidate(normalized_question, route, schema_links)
    hba1c_trend_months = _extract_hba1c_abnormal_trend_months(question)
    if hba1c_trend_months is not None:
        payload = _hba1c_abnormal_trend_payload(hba1c_trend_months, normalized_question)
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": f"analysis_trend_hba1c_abnormal_{hba1c_trend_months}m",
                "matched_id": f"dynamic_hba1c_abnormal_trend_{hba1c_trend_months}m",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
            }
        )
        return payload
    is_datamate_request = any(
        token in normalized_question
        for token in ("DataMate", "datamate", "pipeline", "算子", "原始数据", "数据处理流程", "重建知识图谱", "刷新图谱")
    )
    if routed.get("intent") == "kg_subgraph_render" and not (
        routed.get("normalized_entities", {}).get("diseases")
        or routed.get("normalized_entities", {}).get("indicators")
        or routed.get("normalized_entities", {}).get("cohort")
    ):
        if not any(token in question for token in GLOBAL_GRAPH_REQUEST_HINTS):
            payload = _dynamic_subgraph_analysis(normalized_question, planner)
            payload.update(
                {
                    "original_question": question,
                    "rewritten_question": normalized_question,
                    "canonical_id": "dynamic_subgraph_render",
                    "matched_id": "dynamic_subgraph_render",
                    "schema_links": schema_links,
                    "sql_candidate": candidate,
                    "fallback_used": False,
                    "planner": planner,
                    "planner_log_path": planner_log_path,
                    "safety_note": safety_note(cfg),
                    "agent_route": routed,
                }
            )
            return payload

        from tool_server.report_tools import reports_summary

        payload = reports_summary()
        payload.update(
            {
                "question": normalized_question,
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": "report_summary",
                "matched_id": "report_summary",
                "intent": "report_summary",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
                "summary_text": "已返回当前可直接打开的知识图谱入口、报告入口和图表入口。",
            }
        )
        return payload
    if _looks_like_disease_combination_question(normalized_question):
        payload = _disease_combination_distribution_payload(normalized_question)
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": "disease_combination_distribution",
                "matched_id": "disease_combination_distribution",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
            }
        )
        _persist_cohort_context(
            payload,
            question=question,
            cohort_label="multimorbidity_distribution",
            cohort_type="disease_combination_distribution",
        )
        return payload
    if _looks_like_disease_inventory_question(normalized_question):
        payload = _disease_inventory_payload(normalized_question)
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": "kg_disease_inventory",
                "matched_id": "kg_disease_inventory",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
            }
        )
        _persist_cohort_context(payload, question=question, cohort_label="all_patients_disease_distribution", cohort_type="disease_distribution")
        return payload
    if routed.get("intent") == "system_status":
        payload = _system_status_payload(normalized_question)
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": "system_status",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
            }
        )
        return payload
    if routed.get("intent") == "kg_summary":
        payload = _kg_summary_payload(normalized_question)
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": "kg_summary",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
            }
        )
        return payload
    if routed.get("intent") == "report_summary":
        from tool_server.report_tools import reports_summary

        payload = reports_summary()
        payload.update(
            {
                "question": normalized_question,
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": "report_summary",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
            }
        )
        return payload
    if routed.get("intent") == "kg_entity_query":
        from tool_server.kg_tools import kg_entity_query

        payload = kg_entity_query(normalized_question)
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": "kg_entity_query",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
                "summary_text": payload.get("text"),
            }
        )
        return payload
    if routed.get("intent") == "kg_relation_query":
        from tool_server.kg_tools import kg_relation_query

        payload = kg_relation_query(normalized_question)
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": "kg_relation_query",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
                "summary_text": payload.get("text") or payload.get("insight"),
            }
        )
        return payload
    if routed.get("intent") == "kg_patient_path_query":
        from tool_server.kg_tools import kg_patient_path_query

        patient_id = _extract_patient_id(normalized_question)
        if not patient_id:
            return {
                "status": "failed",
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": "kg_patient_path_query",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
                "errors": ["请提供明确患者编号，例如 P0001，才能查询该患者的风险事件或未来随访计划。"],
                "safety_note": safety_note(cfg),
            }
        payload = kg_patient_path_query(patient_id)
        payload.update(
            {
                "question": normalized_question,
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": "kg_patient_path_query",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
                "summary_text": payload.get("text"),
            }
        )
        return payload
    controlled_payload = _controlled_metric_payload(normalized_question)
    if controlled_payload is not None:
        controlled_payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": controlled_payload.get("analysis_id", "controlled_metric_query"),
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
            }
        )
        return controlled_payload
    if routed.get("intent") == "risk_level_distribution":
        payload = _risk_level_distribution_payload(normalized_question)
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": "risk_level_distribution",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
            }
        )
        return payload
    if routed.get("intent") == "disease_distribution":
        payload = _disease_distribution_payload(normalized_question, routed.get("normalized_entities", {}).get("diseases") or [])
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": "disease_distribution",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
            }
        )
        if payload.get("disease_labels"):
            _persist_cohort_context(
                payload,
                question=question,
                cohort_label="disease_distribution_" + "_".join(str(item) for item in payload.get("disease_labels", [])[:4]),
                cohort_type="disease_distribution",
            )
        return payload
    if routed.get("intent") == "future_n_days_high_risk_followup":
        payload = _future_high_risk_followup_count(window_days=int(routed.get("normalized_entities", {}).get("days") or window_days or 30))
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": "future_n_days_high_risk_followup",
                "matched_id": f"future_high_risk_followup_{int(routed.get('normalized_entities', {}).get('days') or window_days or 30)}d",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
            }
        )
        _persist_cohort_context(
            payload,
            question=question,
            cohort_label=str(payload.get("cohort_label")),
            cohort_type="future_high_risk_followup",
        )
        return payload
    if routed.get("intent") == "cohort_disease_distribution":
        if not last_cohort:
            return {
                "status": "failed",
                "original_question": question,
                "rewritten_question": normalized_question,
                "errors": ["请先说明具体患者群体，当前没有可继承的上一轮 cohort 上下文。"],
                "agent_route": routed,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "safety_note": safety_note(cfg),
            }
        if "high_risk_followup" in str(last_cohort.get("cohort_label", "")):
            if _looks_like_cohort_risk_question(question):
                patient_count = int(last_cohort.get("cohort_patient_count") or 0)
                exact_window_days = int(last_cohort.get("window_days") or window_days or 30)
                payload = {
                    "status": "success",
                    "question": normalized_question,
                    "cohort_label": str(last_cohort.get("cohort_label")),
                    "window_days": exact_window_days,
                    "table": {
                        "rows": [
                            {"风险等级": "高风险", "患者人数": patient_count, "占比": 1.0},
                            {"风险等级": "中风险", "患者人数": 0, "占比": 0.0},
                            {"风险等级": "低风险", "患者人数": 0, "占比": 0.0},
                        ]
                    },
                    "summary_text": f"这里的“该群体”指上一轮查询得到的未来 {exact_window_days} 天高风险随访患者，共 {patient_count} 人；该群体本身即为高风险人群，因此风险等级分布为高风险 100%。",
                    "report_url": last_cohort.get("report_url"),
                    "graph_url": last_cohort.get("graph_url"),
                    "safety_note": safety_note(cfg),
                }
            else:
                payload = _future_high_risk_distribution(window_days=int(last_cohort.get("window_days") or window_days or 30))
            payload.update(
                {
                    "original_question": question,
                    "rewritten_question": normalized_question,
                    "canonical_id": "cohort_disease_distribution",
                    "matched_id": "cohort_disease_distribution",
                    "schema_links": schema_links,
                    "sql_candidate": candidate,
                    "fallback_used": False,
                    "planner": planner,
                    "planner_log_path": planner_log_path,
                    "agent_route": routed,
                    "cohort_context": last_cohort,
                }
            )
            return payload
        if last_cohort.get("cohort_type") == "disease_distribution" and _looks_like_cohort_disease_question(question):
            payload = _disease_inventory_payload(normalized_question)
            payload.update(
                {
                    "original_question": question,
                    "rewritten_question": normalized_question,
                    "canonical_id": "cohort_disease_distribution",
                    "matched_id": "cohort_disease_distribution",
                    "schema_links": schema_links,
                    "sql_candidate": candidate,
                    "fallback_used": False,
                    "planner": planner,
                    "planner_log_path": planner_log_path,
                    "agent_route": routed,
                    "cohort_context": last_cohort,
                    "summary_text": "这里的“他们”指上一轮查询得到的疾病分布患者群体，已继续返回该群体的疾病目录与人数分布。",
                }
            )
            return payload
    if is_datamate_request:
        if any(token in normalized_question for token in ("状态", "进度", "到哪一步", "哪些算子", "耗时", "多久", "同步")):
            payload = datamate_pipeline_status_by_run("latest")
            payload.update(
                {
                    "original_question": question,
                    "rewritten_question": normalized_question,
                    "question": normalized_question,
                    "canonical_id": "datamate_pipeline_status",
                    "matched_id": "datamate_pipeline_status",
                    "intent": "datamate_pipeline_status",
                    "schema_links": schema_links,
                    "sql_candidate": candidate,
                    "fallback_used": False,
                    "planner": planner,
                    "planner_log_path": planner_log_path,
                    "summary_text": payload.get("summary") or "已返回最近一次 DataMate pipeline 状态和算子明细。",
                    "report_url": payload.get("pipeline_browser_url") or public_artifact_url(cfg, "/artifacts/report"),
                    "chart_url": payload.get("chart_index_url") or public_artifact_url(cfg, "/artifacts/charts"),
                    "graph_url": payload.get("graph_url") or public_artifact_url(cfg, "/artifacts/graph.html"),
                    "agent_route": routed,
                }
            )
            return payload
        if any(token in normalized_question for token in ("运行", "执行", "重跑", "重新", "重建")):
            payload = run_datamate_pipeline(task_id=f"open_datamate_run_{_question_slug(question)}", force=True, safe_run=True)
            payload.update(
                {
                    "original_question": question,
                    "rewritten_question": normalized_question,
                    "question": normalized_question,
                    "canonical_id": "datamate_pipeline_run",
                    "matched_id": "datamate_pipeline_run",
                    "intent": "datamate_pipeline_run",
                    "schema_links": schema_links,
                    "sql_candidate": candidate,
                    "fallback_used": False,
                    "planner": planner,
                    "planner_log_path": planner_log_path,
                    "summary_text": (
                        payload.get("summary")
                        or "已返回 DataMate pipeline 执行摘要、11 个算子状态和最新产物快照。"
                    ),
                    "report_url": payload.get("pipeline_browser_url") or public_artifact_url(cfg, "/artifacts/report"),
                    "chart_url": payload.get("chart_index_url") or public_artifact_url(cfg, "/artifacts/charts"),
                    "graph_url": payload.get("graph_url") or public_artifact_url(cfg, "/artifacts/graph.html"),
                    "agent_route": routed,
                }
            )
            return payload
        payload = datamate_pipeline_report_by_run("latest")
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "question": normalized_question,
                "canonical_id": "datamate_pipeline_report",
                "matched_id": "datamate_pipeline_report",
                "intent": "datamate_pipeline_report",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "summary_text": payload.get("summary") or "已返回最近一次 DataMate pipeline 运行报告与检查报告。",
                "report_url": payload.get("report_browser_url") or public_artifact_url(cfg, "/artifacts/report"),
                "chart_url": payload.get("chart_index_url") or public_artifact_url(cfg, "/artifacts/charts"),
                "graph_url": payload.get("graph_url") or public_artifact_url(cfg, "/artifacts/graph.html"),
                "agent_route": routed,
            }
        )
        return payload
    if (
        "未来" in normalized_question
        and "随访" in normalized_question
        and (
            any(token in normalized_question for token in FOLLOWUP_DYNAMIC_VISUAL_HINTS)
            or any(token in normalized_question for token in FOLLOWUP_DYNAMIC_COUNT_HINTS)
        )
    ):
        is_high_risk_followup = (
            routed.get("intent") == "future_n_days_high_risk_followup"
            or _is_high_risk_followup_question(question)
            or _is_high_risk_followup_question(normalized_question)
        )
        if is_high_risk_followup:
            payload = _future_high_risk_followup_count(window_days=window_days)
            canonical_followup_id = "future_n_days_high_risk_followup"
            matched_followup_id = f"future_high_risk_followup_{window_days}d"
            followup_intent = "future_n_days_high_risk_followup"
            cohort_type = "future_high_risk_followup"
            cohort_label = str(payload.get("cohort_label") or f"future_{window_days}d_high_risk_followup")
        else:
            payload = _future_followup_chart_bundle(window_days=window_days)
            canonical_followup_id = "future_followup_chart_bundle"
            matched_followup_id = f"dynamic_future_followup_{window_days}d"
            followup_intent = "future_followup_dynamic"
            cohort_type = "future_followup"
            cohort_label = f"future_{window_days}d_followup"
        if "analysis_html" in payload:
            payload["report_url"] = public_artifact_url(cfg, f"/artifacts/graph-driven/{Path(payload['analysis_html']).stem}")
        if "analysis_chart_html" in payload:
            payload["chart_url"] = public_artifact_url(cfg, f"/artifacts/graph-driven/{Path(payload['analysis_chart_html']).stem}")
        payload.update(
            {
                "status": "success",
                "original_question": question,
                "rewritten_question": normalized_question,
                "question": normalized_question,
                "canonical_id": canonical_followup_id,
                "matched_id": matched_followup_id,
                "intent": followup_intent,
                "chart_type": "chart_bundle",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "safety_note": safety_note(cfg),
                "agent_route": routed,
            }
        )
        _persist_cohort_context(payload, question=question, cohort_label=cohort_label, cohort_type=cohort_type)
        return payload
    if _looks_like_subgraph_request(normalized_question, planner):
        payload = _dynamic_subgraph_analysis(normalized_question, planner)
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "canonical_id": "dynamic_subgraph_render",
                "matched_id": "dynamic_subgraph_render",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "safety_note": safety_note(cfg),
                "agent_route": routed,
            }
        )
        return payload
    if canonical_id in {
        "future_30d_high_risk_followup_disease_distribution",
        "high_salt_bp_abnormal_rate",
        "hypertension_diabetes_multi_indicator",
        "future_followup_chart_bundle",
    }:
        payload = graph_driven_analysis(rewritten["question"])
        payload.update(
            {
                "original_question": question,
                "rewritten_question": rewritten["question"],
                "canonical_id": canonical_id,
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
            }
        )
        if payload.get("cohort_label"):
            _persist_cohort_context(payload, question=question, cohort_label=str(payload.get("cohort_label")), cohort_type="graph_driven_cohort")
        return payload
    if route["route"] == "graph_driven":
        payload = graph_driven_analysis(rewritten["question"])
        payload.update(
            {
                "original_question": question,
                "rewritten_question": rewritten["question"],
                "canonical_id": rewritten.get("canonical_id"),
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
            }
        )
        return payload
    candidate_sql = candidate.get("sql") or _candidate_for_plan(rewritten["question"], planner)
    if candidate_sql and planner.get("intent") in {"cohort_stats", "risk_distribution", "nl2sql"}:
        payload = _execute_safe_sql_question(
            rewritten["question"],
            candidate_sql,
            {
                "time_window": planner.get("time_window"),
                "disease_filters": planner.get("disease_filters"),
                "risk_filters": planner.get("risk_filters"),
            },
        )
        payload.update(
            {
                "original_question": question,
                "rewritten_question": rewritten["question"],
                "canonical_id": rewritten.get("canonical_id"),
                "schema_links": schema_links,
                "sql_candidate": {"sql": candidate_sql, "executable": True},
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
            }
        )
        return payload
    standard = _find_standard_question(rewritten["question"])
    if standard is not None:
        payload = analysis_query(standard["question"])
        payload.update(
            {
                "original_question": question,
                "rewritten_question": rewritten["question"],
                "canonical_id": rewritten.get("canonical_id"),
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "agent_route": routed,
            }
        )
        return payload
    if rewritten.get("canonical_id") == "future_30d_high_risk_followup_disease_distribution":
        payload = _future_high_risk_distribution(window_days=int(rewritten.get("window_days") or 30))
        payload.update(
            {
                "original_question": question,
                "rewritten_question": rewritten["question"],
                "canonical_id": rewritten.get("canonical_id"),
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
            }
        )
        return payload
    if any(token in normalized_question for token in ("重建知识图谱", "刷新图谱", "重新处理数据")):
        payload = run_datamate_pipeline(task_id=f"open_datamate_run_{_question_slug(question)}", force=True, safe_run=True)
        payload.update(
            {
                "original_question": question,
                "rewritten_question": normalized_question,
                "question": normalized_question,
                "canonical_id": "datamate_pipeline_run",
                "matched_id": "datamate_pipeline_run",
                "intent": "datamate_pipeline_run",
                "schema_links": schema_links,
                "sql_candidate": candidate,
                "fallback_used": False,
                "planner": planner,
                "planner_log_path": planner_log_path,
                "summary_text": payload.get("summary") or "已触发 DataMate pipeline 重跑，并返回知识图谱重建相关状态。",
                "agent_route": routed,
            }
        )
        return payload
    return {
        "status": "failed",
        "original_question": question,
        "rewritten_question": rewritten["question"],
        "canonical_id": rewritten.get("canonical_id"),
        "schema_links": schema_links,
        "sql_candidate": candidate,
        "fallback_used": True,
        "planner": planner,
        "planner_log_path": planner_log_path,
        "agent_route": routed,
        "errors": ["未能为该开放式问题找到稳定的 SQL 或图谱分析路由。"],
        "safety_note": safety_note(cfg),
    }


