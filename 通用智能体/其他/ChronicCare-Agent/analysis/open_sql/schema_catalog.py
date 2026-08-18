from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from runtime_common.common import resolve_path

DB_PATH = "data/sqlite/chroniccare.db"
CATALOG_JSON = "outputs/evaluation/open_sql_schema_catalog.json"
CATALOG_MD = "outputs/evaluation/open_sql_schema_catalog.md"

CORE_TABLES = {
    "patient_profile",
    "visit_record",
    "lab_result",
    "medication_record",
    "followup_plan",
    "risk_event",
    "lifestyle_record",
    "doctor_advice",
    "patient_risk_score",
}

FIELD_LABELS = {
    "patient_id": "患者编号",
    "visit_id": "随访/就诊编号",
    "disease_tags": "疾病标签",
    "risk_level": "风险等级",
    "risk_score": "风险评分",
    "risk_factors": "风险因素",
    "item_name": "指标名称",
    "item_value": "指标值",
    "value": "指标值",
    "abnormal_flag": "异常标记",
    "test_date": "检验日期",
    "record_time": "记录时间",
    "drug_category": "药物类别",
    "drug_name": "药物名称",
    "followup_date": "随访日期",
    "priority": "优先级",
    "status": "状态",
    "event_type": "风险事件类型",
    "salt_intake_level": "盐摄入水平",
    "exercise_minutes_per_week": "每周运动分钟",
    "sleep_hours": "睡眠小时",
    "smoking_status": "吸烟状态",
    "bmi": "BMI",
}

JOIN_RELATIONS = [
    ("patient_profile", "patient_id", "visit_record", "patient_id"),
    ("patient_profile", "patient_id", "lab_result", "patient_id"),
    ("patient_profile", "patient_id", "medication_record", "patient_id"),
    ("patient_profile", "patient_id", "followup_plan", "patient_id"),
    ("patient_profile", "patient_id", "patient_risk_score", "patient_id"),
    ("patient_risk_score", "patient_id", "lab_result", "patient_id"),
    ("patient_profile", "patient_id", "risk_event", "patient_id"),
    ("patient_profile", "patient_id", "lifestyle_record", "patient_id"),
    ("patient_profile", "patient_id", "doctor_advice", "patient_id"),
    ("visit_record", "visit_id", "lab_result", "visit_id"),
    ("visit_record", "visit_id", "medication_record", "visit_id"),
    ("visit_record", "visit_id", "followup_plan", "visit_id"),
    ("visit_record", "visit_id", "risk_event", "visit_id"),
    ("visit_record", "visit_id", "lifestyle_record", "visit_id"),
    ("visit_record", "visit_id", "doctor_advice", "visit_id"),
    ("visit_record", "visit_id", "patient_risk_score", "visit_id"),
]


def _field_policy(name: str, field_type: str) -> Dict[str, Any]:
    lowered = name.lower()
    numeric_like = lowered in {
        "age",
        "bmi",
        "item_value",
        "value",
        "reference_high",
        "reference_low",
        "risk_score",
        "exercise_minutes_per_week",
        "sleep_hours",
    }
    group_like = lowered in {
        "gender",
        "disease_tags",
        "risk_level",
        "priority",
        "status",
        "item_name",
        "abnormal_flag",
        "drug_category",
        "drug_name",
        "event_type",
        "advice_type",
        "salt_intake_level",
        "smoking_status",
        "alcohol_status",
    }
    date_like = lowered.endswith("_date") or lowered in {"created_at", "record_time", "test_date", "followup_date", "visit_date"}
    return {
        "name": name,
        "label": FIELD_LABELS.get(name, name),
        "type": field_type or "TEXT",
        "select": True,
        "where": True,
        "group_by": group_like or date_like,
        "aggregate": numeric_like,
        "question_types": ["count", "distribution"] + (["avg", "trend"] if numeric_like or date_like else []),
    }


def db_path() -> Path:
    return resolve_path(DB_PATH)


def build_schema_catalog(write_files: bool = True) -> Dict[str, Any]:
    path = db_path()
    if not path.exists():
        catalog = {"status": "failed", "error": f"SQLite database not found: {DB_PATH}", "db_path": DB_PATH, "tables": {}, "joins": []}
    else:
        tables: Dict[str, Any] = {}
        with sqlite3.connect(path) as con:
            available = [row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            for table in available:
                if table not in CORE_TABLES:
                    continue
                columns = []
                for col in con.execute(f"PRAGMA table_info({table})"):
                    columns.append(_field_policy(str(col[1]), str(col[2] or "TEXT")))
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                tables[table] = {"name": table, "row_count": count, "fields": columns}
        allowed_join_set = []
        for left_table, left_field, right_table, right_field in JOIN_RELATIONS:
            if left_table in tables and right_table in tables:
                allowed_join_set.append(
                    {
                        "left_table": left_table,
                        "left_field": left_field,
                        "right_table": right_table,
                        "right_field": right_field,
                    }
                )
        catalog = {"status": "success", "db_path": DB_PATH, "tables": tables, "joins": allowed_join_set}
    if write_files:
        write_schema_catalog(catalog)
    return catalog


def get_schema_catalog() -> Dict[str, Any]:
    return build_schema_catalog(write_files=True)


def allowed_tables(catalog: Dict[str, Any] | None = None) -> set[str]:
    catalog = catalog or get_schema_catalog()
    return set((catalog.get("tables") or {}).keys())


def allowed_fields(catalog: Dict[str, Any] | None = None) -> Dict[str, set[str]]:
    catalog = catalog or get_schema_catalog()
    result: Dict[str, set[str]] = {}
    for table, meta in (catalog.get("tables") or {}).items():
        result[table] = {field["name"] for field in meta.get("fields", [])}
    return result


def write_schema_catalog(catalog: Dict[str, Any]) -> None:
    json_path = resolve_path(CATALOG_JSON)
    md_path = resolve_path(CATALOG_MD)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    lines: List[str] = ["# Open SQL Schema Catalog", ""]
    for table, meta in (catalog.get("tables") or {}).items():
        lines.append(f"## {table}")
        lines.append("")
        lines.append(f"- rows: `{meta.get('row_count')}`")
        lines.append("")
        lines.append("| field | label | type | select | where | group_by | aggregate |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for field in meta.get("fields", []):
            lines.append(
                "| {name} | {label} | {type} | {select} | {where} | {group_by} | {aggregate} |".format(**field)
            )
        lines.append("")
    lines.append("## Allowed Joins")
    lines.append("")
    for join in catalog.get("joins", []):
        lines.append(
            f"- `{join['left_table']}.{join['left_field']} = {join['right_table']}.{join['right_field']}`"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
