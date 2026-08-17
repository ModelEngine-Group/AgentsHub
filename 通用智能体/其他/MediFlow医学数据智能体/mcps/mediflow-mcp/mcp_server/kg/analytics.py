"""
任务三分析查询辅助模块。

该模块封装疾病详情、统计摘要和可视化平台需要的分析数据读取逻辑。
"""

import sqlite3, re, json, time, os, csv, io, subprocess, hashlib, tempfile, shutil
import requests
from typing import Any
from pathlib import Path
from collections import defaultdict
import sqlite3, re, json
def _disease_aliases(name: str) -> list[str]:
    cleaned = (name or "").strip()
    if not cleaned:
        return []
    aliases = {cleaned}
    substitutions = [
        ("2型", "Ⅱ型"),
        ("2型", "II型"),
        ("Ⅱ型", "2型"),
        ("II型", "2型"),
        ("1型", "Ⅰ型"),
        ("1型", "I型"),
        ("Ⅰ型", "1型"),
        ("I型", "1型"),
    ]
    for old, new in substitutions:
        if old in cleaned:
            aliases.add(cleaned.replace(old, new))
    return sorted(aliases, key=len, reverse=True)


def _query_disease_aspect(
    conn: sqlite3.Connection,
    disease_terms: list[str],
    table: str,
    value_col: str,
    limit: int = 40,
) -> list[dict]:
    where = " OR ".join(["disease LIKE ?"] * len(disease_terms))
    params = [f"%{term}%" for term in disease_terms]
    sql = (
        f"SELECT disease, {value_col}, "
        f"MAX(confidence) AS confidence, "
        f"COALESCE(MAX(source_name), '') AS source_name "
        f"FROM {table} WHERE {where} "
        f"GROUP BY disease, {value_col} "
        f"ORDER BY confidence DESC, {value_col} LIMIT ?"
    )
    rows = conn.execute(sql, (*params, limit)).fetchall()
    return [
        {
            "disease": row[0],
            "value": row[1],
            "confidence": row[2],
            "source_name": row[3],
        }
        for row in rows
    ]


@mcp.tool
def query_disease_analytics(disease: str, aspect: str = "all") -> dict:
    """查询任务三分析库中的疾病知识。

    适用于特定疾病的精确问答，例如：
    - query_disease_analytics("2型糖尿病", "symptoms")
    - query_disease_analytics("高血压", "drugs")
    - query_disease_analytics("肺炎", "all")

    aspect 可选：
    all, symptoms, drugs, tests, complications, departments, causes,
    preventions, procedures, populations。
    """
    if not Path(ANALYTICS_DB).exists():
        return {"error": "分析数据库未构建，请先运行 kg/build_analytics_v2.py"}
    terms = _disease_aliases(disease)
    if not terms:
        return {"error": "disease 不能为空"}

    aspect_map = {
        "symptoms": ("disease_symptoms", "symptom", "症状"),
        "drugs": ("disease_drugs", "drug", "药物"),
        "tests": ("disease_tests", "test", "检查"),
        "complications": ("disease_complications", "complication", "并发症"),
        "departments": ("disease_departments", "department", "科室"),
        "causes": ("disease_causes", "cause", "病因"),
        "preventions": ("disease_preventions", "prevention", "预防"),
        "procedures": ("disease_procedures", "procedure", "治疗方式"),
        "populations": ("disease_populations", "population", "人群"),
    }
    aliases = {
        "symptom": "symptoms",
        "drug": "drugs",
        "test": "tests",
        "check": "tests",
        "complication": "complications",
        "department": "departments",
        "cause": "causes",
        "prevention": "preventions",
        "procedure": "procedures",
        "treatment": "procedures",
        "population": "populations",
    }
    selected = aliases.get((aspect or "all").strip(), (aspect or "all").strip())
    keys = list(aspect_map) if selected == "all" else [selected]
    unknown = [key for key in keys if key not in aspect_map]
    if unknown:
        return {"error": f"不支持的 aspect: {aspect}", "supported_aspects": ["all", *aspect_map.keys()]}

    conn = sqlite3.connect(ANALYTICS_DB)
    try:
        results = {}
        for key in keys:
            table, value_col, label = aspect_map[key]
            rows = _query_disease_aspect(conn, terms, table, value_col)
            results[key] = {"label": label, "items": rows, "count": len(rows)}
        return {
            "disease": disease,
            "matched_terms": terms,
            "aspect": selected,
            "database": os.path.basename(ANALYTICS_DB),
            "results": results,
        }
    finally:
        conn.close()


@mcp.tool
def ask_medical_analytics(question: str) -> dict:
    """使用任务三规则优先查询引擎回答医疗统计/疾病知识问题。

    该工具优先套用可解释 SQL 模板，用于展示从中文问题到 SQL、再到结构化结果的过程。
    """
    if not Path(ANALYTICS_DB).exists():
        return {"error": "分析数据库未构建，请先运行 kg/build_analytics_v2.py", "columns": [], "rows": [], "row_count": 0}
    return medical_query_engine.ask_medical_db(question, ANALYTICS_DB)
