# -*- coding: utf-8 -*-
"""
任务三医学分析查询引擎。

该模块按问题类型选择只读 SQL 模板，在任务三分析库上查询疾病、症状、药物、检查和科室等信息。
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any


FORBIDDEN_SQL_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)

QUESTION_KEYWORDS = {
    "symptom": ("症状", "表征", "现象", "症候", "表现", "临床表现"),
    "cause": ("病因", "原因", "为什么", "怎么会", "为何", "成因"),
    "complication": ("并发症", "并发", "伴随", "伴有", "合并"),
    "drug": ("药", "药品", "用药", "胶囊", "口服液", "药物治疗", "推荐药"),
    "test": ("检查", "检查项目", "查出", "测出", "检验", "化验"),
    "department": ("科室", "挂什么科", "看哪个科", "就诊", "属于哪个科"),
    "prevention": ("预防", "防治", "避免", "防止", "怎么防"),
    "procedure": ("治疗方式", "怎么治疗", "治疗方法", "手术", "疗法", "治法"),
    "population": ("易感", "好发", "多发人群", "哪些人", "人群"),
    "count": ("统计", "数量", "多少", "排行", "排名", "最多", "Top", "top"),
}


@dataclass
class QueryResult:
    question: str
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    error: str | None = None
    matched_template: str | None = None
    question_types: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "error": self.error,
            "matched_template": self.matched_template,
            "question_types": self.question_types or [],
        }


def classify_question(question: str) -> list[str]:
    q = re.sub(r"\s+", "", question)
    types = [name for name, words in QUESTION_KEYWORDS.items() if any(w in q for w in words)]
    return types or ["general"]


def validate_readonly_sql(sql: str) -> None:
    stripped = sql.strip().rstrip(";")
    if not re.match(r"^(SELECT|WITH)\b", stripped, flags=re.IGNORECASE):
        raise ValueError("Only SELECT/WITH statements are allowed.")
    if FORBIDDEN_SQL_RE.search(stripped):
        raise ValueError("Potentially unsafe SQL keyword detected.")
    if ";" in stripped:
        raise ValueError("Multiple SQL statements are not allowed.")


def _like(value: str) -> str:
    return "%" + value.replace("%", "").replace("_", "") + "%"


def _clean_entity(value: str) -> str:
    value = re.sub(r"^(请问|帮我|查询|查一下|统计|列出|请|这个|该)", "", value)
    value = re.sub(r"(的|是|有|会|需要|应该|可以|能|要|怎么|如何)+$", "", value)
    value = re.sub(r"(症状|临床表现|表现)$", "", value)
    return value.strip("，。？！、：:；; 的")


def _first_entity_match(q: str, patterns: list[str]) -> tuple[str, str] | None:
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            entity = _clean_entity(match.group(1))
            if entity:
                return entity, pattern
    return None


def template_sql(question: str) -> tuple[str | None, tuple[Any, ...], str | None]:
    q = re.sub(r"\s+", "", question).strip("，。？！、：:；;")

    if re.search(r"(总共|共有|一共).*(多少|几).*(疾病|病种)", q):
        return "SELECT COUNT(*) AS total_diseases FROM diseases", (), "count_diseases"

    if "实体" in q and ("统计" in q or "数量" in q or "分布" in q):
        return (
            "SELECT entity_type, entity_count FROM v_entity_type_counts LIMIT 20",
            (),
            "entity_type_counts",
        )

    if "关系" in q and ("统计" in q or "数量" in q or "分布" in q):
        return (
            "SELECT relation_code, display_name, triple_count FROM v_relation_counts LIMIT 30",
            (),
            "relation_counts",
        )

    if "科室" in q and ("统计" in q or "最多" in q or "数量" in q or "排行" in q):
        return (
            "SELECT department, disease_count FROM v_department_disease_counts LIMIT 30",
            (),
            "department_counts",
        )

    if ("症状" in q or "临床表现" in q) and (
        "最多" in q or "高频" in q or "排行" in q or "频率" in q or "最高" in q
    ):
        return (
            "SELECT symptom, disease_count FROM v_top_symptoms LIMIT 30",
            (),
            "top_symptoms",
        )

    if ("药" in q or "药物" in q) and ("最多" in q or "高频" in q or "排行" in q):
        return (
            "SELECT drug, disease_count FROM v_drug_disease_counts LIMIT 30",
            (),
            "top_drugs",
        )

    found = _first_entity_match(
        q,
        [
            r"(.+?)(?:需要做哪些检查|需要做什么检查|要做哪些检查|要做什么检查|检查项目|怎么检查|如何检查)",
            r"(.+?)(?:有哪些检查|有什么检查)",
        ],
    )
    if found:
        disease, _ = found
        return (
            """
            SELECT test, confidence, source_name
            FROM disease_tests
            WHERE disease LIKE ?
            GROUP BY test
            ORDER BY confidence DESC, test
            LIMIT 40
            """,
            (_like(disease),),
            "disease_tests",
        )

    found = _first_entity_match(
        q,
        [
            r"(.+?)(?:的病因是什么|病因是什么|是什么原因|原因是什么|为什么会|为何会|怎么引起|病因)",
        ],
    )
    if found:
        disease, _ = found
        return (
            """
            SELECT cause, confidence, source_name
            FROM disease_causes
            WHERE disease LIKE ?
            GROUP BY cause
            ORDER BY confidence DESC
            LIMIT 10
            """,
            (_like(disease),),
            "disease_causes",
        )

    found = _first_entity_match(
        q,
        [
            r"(.+?)(?:怎么预防|如何预防|预防方法|怎样预防|怎么防|防治)",
        ],
    )
    if found:
        disease, _ = found
        return (
            """
            SELECT prevention, confidence, source_name
            FROM disease_preventions
            WHERE disease LIKE ?
            GROUP BY prevention
            ORDER BY confidence DESC
            LIMIT 10
            """,
            (_like(disease),),
            "disease_preventions",
        )

    found = _first_entity_match(
        q,
        [
            r"(.+?)(?:怎么治疗|如何治疗|治疗方式|治疗方法|有哪些疗法|有什么疗法|手术治疗)",
        ],
    )
    if found:
        disease, _ = found
        return (
            """
            SELECT procedure, confidence, source_name
            FROM disease_procedures
            WHERE disease LIKE ?
            GROUP BY procedure
            ORDER BY confidence DESC, procedure
            LIMIT 40
            """,
            (_like(disease),),
            "disease_procedures",
        )

    found = _first_entity_match(
        q,
        [
            r"(.+?)(?:哪些人容易得|哪些人容易患|好发人群|易感人群|多发人群)",
        ],
    )
    if found:
        disease, _ = found
        return (
            """
            SELECT population, confidence, source_name
            FROM disease_populations
            WHERE disease LIKE ?
            GROUP BY population
            ORDER BY confidence DESC, population
            LIMIT 30
            """,
            (_like(disease),),
            "disease_populations",
        )

    found = _first_entity_match(
        q,
        [
            r"(.+?)(?:有哪些|有什么|的)?(?:症状|临床表现|表现)",
        ],
    )
    if found and "哪些疾病" not in q:
        disease, _ = found
        return (
            """
            SELECT symptom, confidence, source_name
            FROM disease_symptoms
            WHERE disease LIKE ?
            GROUP BY symptom
            ORDER BY confidence DESC, symptom
            LIMIT 40
            """,
            (_like(disease),),
            "disease_symptoms",
        )

    found = _first_entity_match(
        q,
        [
            r"(.+?)(?:用什么药|吃什么药|推荐药|药物治疗|治疗药物|有哪些药|有什么药)",
        ],
    )
    if found:
        disease, _ = found
        return (
            """
            SELECT drug, confidence, source_name
            FROM disease_drugs
            WHERE disease LIKE ?
            GROUP BY drug
            ORDER BY confidence DESC, drug
            LIMIT 50
            """,
            (_like(disease),),
            "disease_drugs",
        )

    found = _first_entity_match(q, [r"(.+?)(?:有哪些|有什么|常见)?(?:并发症|并发|合并症)"])
    if found and "哪些疾病" not in q:
        disease, _ = found
        return (
            """
            SELECT complication, confidence, source_name
            FROM disease_complications
            WHERE disease LIKE ?
            GROUP BY complication
            ORDER BY confidence DESC, complication
            LIMIT 40
            """,
            (_like(disease),),
            "disease_complications",
        )

    found = _first_entity_match(q, [r"(.+?)(?:看哪个科|挂什么科|就诊科室|属于哪个科|去哪个科)"])
    if found:
        disease, _ = found
        return (
            """
            SELECT department, confidence, source_name
            FROM disease_departments
            WHERE disease LIKE ?
            GROUP BY department
            ORDER BY confidence DESC, department
            LIMIT 20
            """,
            (_like(disease),),
            "disease_departments",
        )

    match = re.search(r"哪些疾病.*(?:有|出现|伴有|表现为)(.+?)(?:症状|表现)?$", q)
    if match:
        symptom = _clean_entity(match.group(1))
        if symptom:
            return (
                """
                SELECT disease, symptom, confidence, source_name
                FROM disease_symptoms
                WHERE symptom LIKE ?
                GROUP BY disease, symptom
                ORDER BY confidence DESC, disease
                LIMIT 50
                """,
                (_like(symptom),),
                "symptom_to_diseases",
            )

    match = re.search(r"(.+?)(?:可以治疗|治疗|适用于)(哪些疾病|什么病)", q)
    if match:
        drug = _clean_entity(match.group(1))
        if drug:
            return (
                """
                SELECT disease, drug, confidence, source_name
                FROM disease_drugs
                WHERE drug LIKE ?
                GROUP BY disease, drug
                ORDER BY confidence DESC, disease
                LIMIT 50
                """,
                (_like(drug),),
                "drug_to_diseases",
            )

    match = re.search(r"(.+?)(?:能检查出|可以查出|用于检查)(哪些疾病|什么病)", q)
    if match:
        test = _clean_entity(match.group(1))
        if test:
            return (
                """
                SELECT disease, test, confidence, source_name
                FROM disease_tests
                WHERE test LIKE ?
                GROUP BY disease, test
                ORDER BY confidence DESC, disease
                LIMIT 50
                """,
                (_like(test),),
                "test_to_diseases",
            )

    match = re.search(r"(.+?科)(?:有哪些|有什么|治疗哪些)(?:疾病|病)", q)
    if match:
        department = _clean_entity(match.group(1))
        return (
            """
            SELECT disease, department, confidence, source_name
            FROM disease_departments
            WHERE department LIKE ?
            GROUP BY disease, department
            ORDER BY confidence DESC, disease
            LIMIT 80
            """,
            (_like(department),),
            "department_to_diseases",
        )

    match = re.search(r"(.+?)(?:详细信息|详情|介绍|简介|是什么病|是什么疾病)", q)
    if match:
        disease = _clean_entity(match.group(1))
        if disease:
            return (
                """
                SELECT disease, fact_type, fact_name, confidence, source_name
                FROM disease_facts
                WHERE disease LIKE ?
                ORDER BY confidence DESC, fact_type, fact_name
                LIMIT 80
                """,
                (_like(disease),),
                "disease_fact_summary",
            )

    return None, (), None


def execute_sql(sql: str, db_path: str, params: tuple[Any, ...] = ()) -> dict[str, Any]:
    validate_readonly_sql(sql)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        columns = [item[0] for item in cur.description] if cur.description else []
        data = [list(row) for row in rows]
        return {"columns": columns, "rows": data, "row_count": len(data), "error": None}
    except Exception as exc:
        return {"columns": [], "rows": [], "row_count": 0, "error": str(exc)}
    finally:
        conn.close()


def ask_medical_db(question: str, db_path: str, sql: str | None = None) -> dict[str, Any]:
    question_types = classify_question(question)
    try:
        if sql is None:
            sql, params, template_name = template_sql(question)
            if sql is None:
                return QueryResult(
                    question=question,
                    sql="",
                    columns=[],
                    rows=[],
                    row_count=0,
                    error="No deterministic template matched.",
                    question_types=question_types,
                ).to_dict()
        else:
            params = ()
            template_name = "manual_sql"

        result = execute_sql(sql, db_path, params)
        return QueryResult(
            question=question,
            sql=sql,
            columns=result["columns"],
            rows=result["rows"],
            row_count=result["row_count"],
            error=result["error"],
            matched_template=template_name,
            question_types=question_types,
        ).to_dict()
    except Exception as exc:
        return QueryResult(
            question=question,
            sql=sql or "",
            columns=[],
            rows=[],
            row_count=0,
            error=str(exc),
            question_types=question_types,
        ).to_dict()


def inspect_analytics_db(db_path: str) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        tables = [
            "diseases",
            "disease_facts",
            "disease_symptoms",
            "disease_drugs",
            "disease_tests",
            "disease_departments",
            "disease_complications",
            "entity_stats",
            "relation_stats",
        ]
        counts = {
            table: conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
            for table in tables
        }
        top_relations = [
            dict(row)
            for row in conn.execute(
                "SELECT relation_code, display_name, triple_count FROM v_relation_counts LIMIT 20"
            ).fetchall()
        ]
        top_entities = [
            dict(row)
            for row in conn.execute(
                "SELECT entity_type, entity_count FROM v_entity_type_counts LIMIT 20"
            ).fetchall()
        ]
        return {
            "counts": counts,
            "top_relations": top_relations,
            "top_entities": top_entities,
        }
    finally:
        conn.close()
