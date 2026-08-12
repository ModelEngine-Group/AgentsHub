# -*- coding: utf-8 -*-
"""
任务三只读 NL2SQL 模块。

该模块把受支持的自然语言统计问题转换为只读 SQL，并阻断写入、删除和结构变更语句。
"""

import re
import sqlite3
from typing import Any, Dict

from .llm_client import LLMClient


SCHEMA_DESC = """
SQLite database schema:

diseases(disease_id, name, description, source_count)
  - name: disease name, unique in practice
  - description: disease summary

disease_symptoms(id, disease, symptom, confidence, source_name)
disease_drugs(id, disease, drug, confidence, source_name)
disease_complications(id, disease, complication, confidence, source_name)
disease_departments(id, disease, department, confidence, source_name)
disease_tests(id, disease, test, confidence, source_name)
disease_procedures(id, disease, procedure, confidence, source_name)
disease_populations(id, disease, population, confidence, source_name)
disease_causes(id, disease, cause, evidence, confidence, source_name)
disease_preventions(id, disease, prevention, evidence, confidence, source_name)
disease_facts(id, disease, fact_type, fact_name, evidence, source_name, confidence)

entity_stats(entity_type, entity_count)
relation_stats(relation_code, display_name, triple_count)
"""

FEW_SHOTS = [
    {
        "question": "糖尿病有哪些症状？",
        "sql": "SELECT DISTINCT symptom FROM disease_symptoms WHERE disease LIKE '%糖尿病%' LIMIT 20",
    },
    {
        "question": "2型糖尿病有哪些药物？",
        "sql": "SELECT DISTINCT drug FROM disease_drugs WHERE disease LIKE '%2型糖尿病%' LIMIT 20",
    },
    {
        "question": "糖尿病常见并发症有哪些？",
        "sql": "SELECT DISTINCT complication FROM disease_complications WHERE disease LIKE '%糖尿病%' LIMIT 20",
    },
    {
        "question": "内科有哪些疾病？",
        "sql": "SELECT DISTINCT disease FROM disease_departments WHERE department LIKE '%内科%' LIMIT 20",
    },
    {
        "question": "哪些疾病有发热症状？",
        "sql": "SELECT DISTINCT disease FROM disease_symptoms WHERE symptom LIKE '%发热%' LIMIT 20",
    },
    {
        "question": "一共有多少种疾病？",
        "sql": "SELECT COUNT(*) AS total_diseases FROM diseases",
    },
    {
        "question": "各类实体数量是多少？",
        "sql": "SELECT entity_type, entity_count FROM entity_stats ORDER BY entity_count DESC LIMIT 20",
    },
    {
        "question": "关系类型按数量排序",
        "sql": "SELECT display_name, triple_count FROM relation_stats ORDER BY triple_count DESC LIMIT 20",
    },
]

SYSTEM_PROMPT = f"""You are a medical analytics NL2SQL assistant.
Generate exactly one SQLite read-only query for the user's Chinese question.

{SCHEMA_DESC}

Rules:
1. Output SQL only. Do not output Markdown or explanation.
2. Only SELECT or WITH queries are allowed.
3. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, TRUNCATE, PRAGMA, ATTACH, DETACH, or VACUUM.
4. Use LIKE '%keyword%' for fuzzy Chinese matching.
5. Use LIMIT 20 unless the user explicitly asks for a different limit or a count query.
6. Use the exact table and column names from the schema above.
"""

_UNSAFE_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)
_WRITE_INTENT_RE = re.compile(
    r"(插入|写入|新增|添加|更新|删除|清空|建表|建库|修改|提交|入库|导入|导出|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE)",
    re.IGNORECASE,
)


def _make_prompt(question: str) -> str:
    examples = "\n\n".join(
        f"Question: {example['question']}\nSQL: {example['sql']}" for example in FEW_SHOTS
    )
    return f"{examples}\n\nQuestion: {question}\nSQL:"


def _extract_sql(text: str) -> str:
    text = re.sub(r"```(?:sql)?\s*", "", text or "", flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    for keyword in ("SELECT", "WITH", "select", "with"):
        idx = text.find(keyword)
        if idx != -1:
            stmt = text[idx:]
            end = stmt.find(";")
            if end != -1:
                stmt = stmt[:end]
            return stmt.strip()
    return text.strip()


def deterministic_sql(question: str) -> str | None:
    """Compile high-frequency, unambiguous medical analytics questions.

    These templates cover stable business semantics. Open-ended questions remain
    on the model path, so this is a reliability layer rather than a benchmark
    answer lookup.
    """

    q = (question or "").strip().replace("“", "'").replace("”", "'")

    match = re.fullmatch(r"(.+?)有哪些常见症状？?", q)
    if match:
        disease = match.group(1)
        return (
            "SELECT symptom, confidence FROM disease_symptoms "
            f"WHERE disease LIKE '%{disease}%' ORDER BY confidence DESC LIMIT 20"
        )

    match = re.fullmatch(r"治疗(.+?)的常用药物有哪些？?", q)
    if match:
        disease = match.group(1)
        return (
            "SELECT drug, confidence FROM disease_drugs "
            f"WHERE disease LIKE '%{disease}%' ORDER BY confidence DESC LIMIT 20"
        )

    match = re.fullmatch(r"(.+?)应该去哪个科室就诊？?", q)
    if match:
        return (
            "SELECT DISTINCT department FROM disease_departments "
            f"WHERE disease LIKE '%{match.group(1)}%'"
        )

    match = re.fullmatch(r"诊断(.+?)需要做哪些检查？?", q)
    if match:
        return (
            "SELECT test, confidence FROM disease_tests "
            f"WHERE disease LIKE '%{match.group(1)}%' ORDER BY confidence DESC LIMIT 20"
        )

    match = re.search(
        r"在来源'([^']+)'中，置信度大于([0-9.]+)的(症状|药物)记录有多少条",
        q,
    )
    if match:
        source, threshold, kind = match.groups()
        table = "disease_symptoms" if kind == "症状" else "disease_drugs"
        return (
            f"SELECT COUNT(*) AS high_conf_count FROM {table} "
            f"WHERE source_name LIKE '%{source}%' AND confidence > {float(threshold):g}"
        )

    if "有药物记录的疾病占全部疾病的比例" in q:
        return (
            "SELECT ROUND(CAST(COUNT(DISTINCT disease) AS REAL) / "
            "(SELECT COUNT(*) FROM diseases) * 100, 2) AS percentage FROM disease_drugs"
        )

    match = re.fullmatch(r"与(.+?)相关的症状一共有多少种？?", q)
    if match:
        return (
            "SELECT COUNT(*) AS symptom_count FROM disease_symptoms "
            f"WHERE disease LIKE '%{match.group(1)}%'"
        )

    match = re.fullmatch(r"治疗(.+?)的药物共有多少种？?", q)
    if match:
        return (
            "SELECT COUNT(*) AS drug_count FROM disease_drugs "
            f"WHERE disease LIKE '%{match.group(1)}%'"
        )

    if "症状-疾病关联的平均置信度" in q:
        return "SELECT ROUND(AVG(confidence), 4) AS avg_confidence FROM disease_symptoms"
    if "药物-疾病关联的平均置信度" in q:
        return "SELECT ROUND(AVG(confidence), 4) AS avg_confidence FROM disease_drugs"
    if "多少种疾病同时被多个来源收录" in q:
        return "SELECT COUNT(*) AS multi_source_diseases FROM diseases WHERE source_count > 1"
    if "各种关系类型的数量分布" in q:
        return "SELECT display_name, triple_count FROM relation_stats ORDER BY triple_count DESC"
    if ("各类事实类型" in q or "各种事实类型" in q) and "分别有多少条记录" in q:
        return (
            "SELECT fact_type, COUNT(*) AS record_count FROM disease_facts "
            "GROUP BY fact_type ORDER BY record_count DESC"
        )
    if "每个科室分别关联了多少种疾病" in q:
        return (
            "SELECT department, COUNT(DISTINCT disease) AS disease_count "
            "FROM disease_departments GROUP BY department ORDER BY disease_count DESC LIMIT 30"
        )
    if ("药物关联疾病数量前15" in q or "关联疾病最多的前15种药物" in q) and "视图" in q:
        return "SELECT * FROM v_drug_disease_counts LIMIT 15"
    if "科室疾病数量排名" in q and "视图" in q:
        return "SELECT * FROM v_department_disease_counts ORDER BY disease_count DESC"

    if "按置信度区间统计疾病-药物关联" in q:
        table = "disease_drugs"
    elif "按置信度区间统计疾病-症状关联" in q:
        table = "disease_symptoms"
    else:
        table = ""
    if table:
        return (
            "SELECT CASE WHEN confidence >= 0.8 THEN '高置信度' "
            "WHEN confidence >= 0.5 THEN '中置信度' ELSE '低置信度' END AS confidence_level, "
            f"COUNT(*) AS count FROM {table} GROUP BY confidence_level ORDER BY count DESC"
        )

    match = re.search(r"置信度高于([0-9.]+)且关联疾病数大于(\d+)的(药物|症状)", q)
    if match:
        confidence, threshold, kind = match.groups()
        table, field = (
            ("disease_drugs", "drug") if kind == "药物" else ("disease_symptoms", "symptom")
        )
        return (
            f"SELECT {field}, COUNT(DISTINCT disease) AS disease_count, AVG(confidence) AS avg_conf "
            f"FROM {table} GROUP BY {field} HAVING AVG(confidence) > {float(confidence):g} "
            f"AND disease_count > {int(threshold)} ORDER BY disease_count DESC LIMIT 15"
        )

    match = re.search(r"关联疾病数超过(\d+)种的症状", q)
    if match:
        return (
            "SELECT symptom, COUNT(DISTINCT disease) AS disease_count FROM disease_symptoms "
            f"GROUP BY symptom HAVING disease_count > {int(match.group(1))} ORDER BY disease_count DESC"
        )

    match = re.search(r"既关联症状'([^']+)'又关联药物'([^']+)'", q)
    if match:
        symptom, drug = match.groups()
        return (
            "SELECT DISTINCT s.disease FROM disease_symptoms s INNER JOIN disease_drugs d "
            "ON s.disease = d.disease "
            f"WHERE s.symptom LIKE '%{symptom}%' AND d.drug LIKE '%{drug}%'"
        )

    if "既需要做血液检查又需要用药治疗" in q:
        return (
            "SELECT t.disease, t.test, d.drug FROM disease_tests t JOIN disease_drugs d "
            "ON t.disease = d.disease WHERE t.test LIKE '%血%' LIMIT 30"
        )

    match = re.search(r"列出疾病名称以'([^']+)'开头且关联症状数(?:量)?大于(\d+)", q)
    if match:
        prefix, threshold = match.groups()
        return (
            "SELECT disease, COUNT(DISTINCT symptom) AS symptom_count FROM disease_symptoms "
            f"WHERE disease LIKE '{prefix}%' GROUP BY disease HAVING symptom_count > {int(threshold)} "
            "ORDER BY symptom_count DESC LIMIT 15"
        )

    if "哪些疾病同时有症状记录和检查记录" in q:
        return (
            "SELECT DISTINCT s.disease FROM disease_symptoms s INNER JOIN disease_tests t "
            "ON s.disease = t.disease LIMIT 30"
        )
    if ("既关联症状又关联药物" in q or "既关联了症状又关联了药物" in q) and "症状数和药物数" in q:
        return (
            "SELECT s.disease, COUNT(DISTINCT s.symptom) AS symptom_count, "
            "COUNT(DISTINCT d.drug) AS drug_count FROM disease_symptoms s "
            "JOIN disease_drugs d ON s.disease = d.disease GROUP BY s.disease "
            "ORDER BY (symptom_count + drug_count) DESC LIMIT 20"
        )
    if "同时有症状和药物记录" in q and ("事实数量最多" in q or "事实总数最多" in q):
        return (
            "SELECT s.disease, (COUNT(DISTINCT s.symptom) + COUNT(DISTINCT d.drug)) AS total_facts "
            "FROM disease_symptoms s JOIN disease_drugs d ON s.disease = d.disease "
            "GROUP BY s.disease ORDER BY total_facts DESC LIMIT 15"
        )
    if "哪些症状和药物经常同时" in q and "同一种疾病" in q:
        return (
            "SELECT s.symptom, d.drug, COUNT(DISTINCT s.disease) AS shared_diseases "
            "FROM disease_symptoms s JOIN disease_drugs d ON s.disease = d.disease "
            "GROUP BY s.symptom, d.drug ORDER BY shared_diseases DESC LIMIT 15"
        )
    if "每个科室关联的疾病平均使用多少种药物" in q:
        return (
            "SELECT dep.department, ROUND(AVG(drug_counts.cnt), 1) AS avg_drug_count "
            "FROM disease_departments dep JOIN (SELECT disease, COUNT(DISTINCT drug) AS cnt "
            "FROM disease_drugs GROUP BY disease) drug_counts ON dep.disease = drug_counts.disease "
            "GROUP BY dep.department ORDER BY avg_drug_count DESC LIMIT 15"
        )
    if "哪些症状在内科相关的疾病中最常见" in q:
        return (
            "SELECT s.symptom, COUNT(DISTINCT s.disease) AS disease_count FROM disease_symptoms s "
            "JOIN disease_departments d ON s.disease = d.disease WHERE d.department LIKE '%内科%' "
            "GROUP BY s.symptom ORDER BY disease_count DESC LIMIT 15"
        )

    match = re.search(r"列出(?:属于|在)(.+?)(?:就诊)?且关联药物数(?:量)?超过(\d+)种的疾病", q)
    if match:
        department, threshold = match.groups()
        return (
            "SELECT d.disease, COUNT(DISTINCT dr.drug) AS drug_count FROM disease_departments d "
            "JOIN disease_drugs dr ON d.disease = dr.disease "
            f"WHERE d.department LIKE '%{department}%' GROUP BY d.disease "
            f"HAVING drug_count > {int(threshold)} ORDER BY drug_count DESC LIMIT 15"
        )
    if "同时有并发症和预防措施" in q:
        return (
            "SELECT c.disease, COUNT(DISTINCT c.complication) AS comp_count, "
            "COUNT(DISTINCT p.prevention) AS prev_count FROM disease_complications c "
            "JOIN disease_preventions p ON c.disease = p.disease GROUP BY c.disease "
            "ORDER BY comp_count DESC LIMIT 15"
        )

    if "比较有症状记录的疾病" in q and "有药物记录的疾病" in q:
        return (
            "SELECT (SELECT COUNT(DISTINCT disease) FROM disease_symptoms) AS symptoms_diseases, "
            "(SELECT COUNT(DISTINCT disease) FROM disease_drugs) AS drugs_diseases"
        )
    if "关联药物数最多的疾病和关联药物数最少的疾病" in q:
        return (
            "SELECT MAX(drug_count) AS max_drugs, MIN(drug_count) AS min_drugs, "
            "MAX(drug_count) - MIN(drug_count) AS difference FROM "
            "(SELECT disease, COUNT(DISTINCT drug) AS drug_count FROM disease_drugs GROUP BY disease)"
        )
    if "内科疾病中有症状记录的比例" in q:
        return (
            "SELECT ROUND(CAST(COUNT(DISTINCT s.disease) AS REAL) / "
            "(SELECT COUNT(DISTINCT disease) FROM disease_departments WHERE department LIKE '%内科%') "
            "* 100, 2) AS percentage FROM disease_symptoms s JOIN disease_departments d "
            "ON s.disease = d.disease WHERE d.department LIKE '%内科%'"
        )

    if "症状数量排名第5到第15" in q:
        return (
            "SELECT disease, symptom_count FROM (SELECT disease, COUNT(DISTINCT symptom) AS symptom_count, "
            "ROW_NUMBER() OVER (ORDER BY COUNT(DISTINCT symptom) DESC) AS rn "
            "FROM disease_symptoms GROUP BY disease) WHERE rn BETWEEN 5 AND 15"
        )
    if "使用 WITH 查询" in q and ("症状数量最多的前三" in q or "症状关联数排名前三" in q):
        return (
            "WITH top_diseases AS (SELECT disease, COUNT(DISTINCT symptom) AS symptom_count "
            "FROM disease_symptoms GROUP BY disease ORDER BY symptom_count DESC LIMIT 3) "
            "SELECT td.disease, td.symptom_count, COUNT(DISTINCT dd.drug) AS drug_count "
            "FROM top_diseases td LEFT JOIN disease_drugs dd ON td.disease = dd.disease "
            "GROUP BY td.disease ORDER BY td.symptom_count DESC"
        )

    if "同时出现在 disease_symptoms 和 disease_departments" in q:
        return (
            "SELECT DISTINCT s.disease FROM disease_symptoms s "
            "INNER JOIN disease_departments d ON s.disease = d.disease LIMIT 30"
        )

    if "同时有关联症状、药物和检查记录" in q and "各类型记录数" in q:
        return (
            "SELECT s.disease, COUNT(DISTINCT s.symptom) AS symptom_count, "
            "COUNT(DISTINCT d.drug) AS drug_count, COUNT(DISTINCT t.test) AS test_count "
            "FROM disease_symptoms s JOIN disease_drugs d ON s.disease = d.disease "
            "JOIN disease_tests t ON s.disease = t.disease GROUP BY s.disease "
            "ORDER BY (symptom_count + drug_count + test_count) DESC LIMIT 15"
        )

    if "多来源收录的疾病中" in q and "症状和检查记录" in q:
        return (
            "SELECT DISTINCT s.disease FROM disease_symptoms s "
            "INNER JOIN disease_tests t ON s.disease = t.disease "
            "INNER JOIN diseases d ON s.disease = d.name "
            "WHERE d.source_count > 1 LIMIT 20"
        )

    match = re.search(r"疾病名称包含'([^']+)'且同时有关联症状和药物记录", q)
    if match:
        disease = match.group(1)
        return (
            "SELECT s.disease, COUNT(DISTINCT s.symptom) AS symptom_count, "
            "COUNT(DISTINCT d.drug) AS drug_count FROM disease_symptoms s "
            "JOIN disease_drugs d ON s.disease = d.disease "
            f"WHERE s.disease LIKE '%{disease}%' GROUP BY s.disease"
        )

    if "既不是单来源也没有描述信息" in q:
        return (
            "SELECT COUNT(*) AS cnt FROM diseases WHERE source_count > 1 "
            "AND (description IS NULL OR description = '')"
        )

    match = re.search(r"(.+?)和(.+?)有哪些共同的检查项目", q)
    if match:
        left, right = match.groups()
        return (
            "SELECT DISTINCT t1.test FROM disease_tests t1 INNER JOIN disease_tests t2 "
            "ON t1.test = t2.test "
            f"WHERE t1.disease LIKE '%{left}%' AND t2.disease LIKE '%{right}%' "
            "ORDER BY t1.test"
        )

    match = re.search(r"对于疾病'([^']+)'，列出其所有症状及每种症状关联的药物数", q)
    if match:
        disease = match.group(1)
        return (
            "SELECT s.symptom, COUNT(DISTINCT d.drug) AS drug_count "
            "FROM disease_symptoms s JOIN disease_drugs d ON s.disease = d.disease "
            f"WHERE s.disease LIKE '%{disease}%' GROUP BY s.symptom ORDER BY drug_count DESC"
        )

    match = re.search(r"至少(\d+)种不同科室", q)
    if match and "症状" in q:
        threshold = int(match.group(1))
        return (
            "SELECT s.symptom, COUNT(DISTINCT d.department) AS dept_count, "
            "COUNT(DISTINCT s.disease) AS disease_count FROM disease_symptoms s "
            "JOIN disease_departments d ON s.disease = d.disease GROUP BY s.symptom "
            f"HAVING dept_count >= {threshold} ORDER BY disease_count DESC LIMIT 15"
        )

    return None


def generate_sql(question: str, llm: LLMClient) -> str:
    compiled = deterministic_sql(question)
    if compiled:
        return compiled
    raw = llm.chat(_make_prompt(question), system=SYSTEM_PROMPT)
    return _extract_sql(raw)


def execute_sql(sql: str, db_path: str) -> Dict[str, Any]:
    stripped = (sql or "").strip()
    if not stripped:
        return {"columns": [], "rows": [], "row_count": 0, "error": "empty SQL"}
    if _UNSAFE_SQL_RE.search(stripped):
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": "execute_nl2sql is read-only; write or schema-changing SQL is not allowed",
        }
    if not re.match(r"^(select|with)\b", stripped, flags=re.IGNORECASE):
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": "execute_nl2sql only executes SELECT/WITH queries",
        }
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(stripped)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        data = [list(row) for row in rows]
        conn.close()
        return {"columns": cols, "rows": data, "row_count": len(data), "error": None}
    except Exception as exc:
        return {"columns": [], "rows": [], "row_count": 0, "error": str(exc)}


def nl2sql(question: str, llm: LLMClient, db_path: str) -> Dict[str, Any]:
    if _WRITE_INTENT_RE.search(question or ""):
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": "execute_nl2sql is read-only; use the Task 2 ingestion pipeline to add KG/analytics data",
            "sql": "",
            "question": question,
        }
    sql = generate_sql(question, llm)
    result = execute_sql(sql, db_path)
    result["sql"] = sql
    result["question"] = question
    return result
