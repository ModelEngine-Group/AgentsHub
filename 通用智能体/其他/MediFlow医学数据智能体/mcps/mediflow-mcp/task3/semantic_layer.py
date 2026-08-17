"""任务三医学分析语义层。

本模块集中维护可查询的实体、关系、指标和只读表映射，使智能体、
NL2SQL、证据表、图表与导出报告使用相同的数据口径。
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from .contracts import AnalysisPlan, AnalysisQuery


SCHEMA_DESCRIPTION = """
SQLite 医学分析库：
- diseases(disease_id, name, description, source_count)：疾病知识条目，不是患者记录。
- disease_symptoms(id, disease, symptom, confidence, source_name)
- disease_drugs(id, disease, drug, confidence, source_name)
- disease_complications(id, disease, complication, confidence, source_name)
- disease_departments(id, disease, department, confidence, source_name)
- disease_tests(id, disease, test, confidence, source_name)
- disease_procedures(id, disease, procedure, confidence, source_name)
- disease_populations(id, disease, population, confidence, source_name)
- disease_causes(id, disease, cause, evidence, confidence, source_name)
- disease_preventions(id, disease, prevention, evidence, confidence, source_name)
- entity_stats(entity_type, entity_count)
- relation_stats(relation_code, display_name, triple_count)
- v_department_disease_counts(department, disease_count)
- v_top_symptoms(symptom, disease_count)
- v_drug_disease_counts(drug, disease_count)

数据口径：
1. 当前库没有患者主表、就诊表或病例级唯一患者标识，不能计算患者人数、患病率或处方量。
2. disease_count 表示知识图谱中与某项事实关联的疾病条目数，不表示患者人数。
3. 所有查询必须是只读 SELECT/WITH。
""".strip()


@dataclass(frozen=True, slots=True)
class FactSpec:
    key: str
    keywords: tuple[str, ...]
    table: str
    value_column: str
    label: str


FACT_SPECS = (
    FactSpec("symptom", ("症状", "表现", "体征"), "disease_symptoms", "symptom", "症状"),
    FactSpec("drug", ("药", "药物", "用药", "吃什么"), "disease_drugs", "drug", "药物"),
    FactSpec("complication", ("并发症", "并发", "合并症"), "disease_complications", "complication", "并发症"),
    FactSpec("department", ("科室", "挂什么科", "挂哪科", "就诊"), "disease_departments", "department", "科室"),
    FactSpec("test", ("检查", "化验", "检验", "检测"), "disease_tests", "test", "检查"),
    FactSpec("procedure", ("治疗", "怎么治", "疗法", "手术"), "disease_procedures", "procedure", "治疗方式"),
    FactSpec("population", ("易感", "好发", "人群"), "disease_populations", "population", "易感人群"),
    FactSpec("cause", ("病因", "原因", "为什么"), "disease_causes", "cause", "病因"),
    FactSpec("prevention", ("预防", "避免", "怎么防"), "disease_preventions", "prevention", "预防"),
)

PATIENT_COUNT_RE = re.compile(
    r"(患者|病人|病例).{0,8}(数量|人数|多少|统计)|患病率|发病率"
)
COMPLEX_ANALYSIS_RE = re.compile(
    r"比较|趋势|相关|关联|占比|比例|平均|分组|分别|同时|以及|并且|导出|报告|分析"
)


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", "", str(question or "").strip())


def find_diseases(conn: sqlite3.Connection, question: str, limit: int = 5) -> list[str]:
    """识别问题中明确出现的疾病，优先匹配较长名称。"""

    compact = normalize_question(question)
    rows = conn.execute(
        """
        SELECT name
        FROM diseases
        WHERE ? LIKE '%' || name || '%'
        ORDER BY LENGTH(name) DESC, source_count DESC
        LIMIT ?
        """,
        (compact, limit),
    ).fetchall()
    names: list[str] = []
    for row in rows:
        name = str(row[0])
        if name not in names:
            names.append(name)
    return names


def detect_fact_specs(question: str) -> list[FactSpec]:
    compact = normalize_question(question)
    return [
        spec
        for spec in FACT_SPECS
        if any(keyword in compact for keyword in spec.keywords)
    ]


def _fact_query(spec: FactSpec, disease: str) -> AnalysisQuery:
    return AnalysisQuery(
        title=f"{disease}的{spec.label}",
        purpose=f"查询知识图谱中与{disease}关联的{spec.label}",
        sql=(
            f"SELECT {spec.value_column} AS {spec.label}, "
            "MAX(confidence) AS 置信度, "
            "GROUP_CONCAT(DISTINCT source_name) AS 数据来源 "
            f"FROM {spec.table} "
            "WHERE disease LIKE ? "
            f"GROUP BY {spec.value_column} "
            f"ORDER BY 置信度 DESC, {spec.value_column} "
            "LIMIT 40"
        ),
        params=(f"%{disease}%",),
        chart_type="table",
    )


def _append_unique(plan: AnalysisPlan, query: AnalysisQuery) -> None:
    if query.sql not in {item.sql for item in plan.queries}:
        plan.queries.append(query)


def semantic_plan(conn: sqlite3.Connection, question: str) -> AnalysisPlan:
    """为常见医学知识与统计问题生成确定、可审计的分析计划。"""

    compact = normalize_question(question)
    diseases = find_diseases(conn, compact)
    subject = diseases[0] if diseases else None
    plan = AnalysisPlan(question=question, subject=subject)

    if PATIENT_COUNT_RE.search(compact):
        plan.unsupported.append(
            "当前分析库保存的是疾病知识与关系，不含患者级唯一记录，"
            "因此不能据此计算患者人数、患病率或发病率。"
        )

    fact_specs = detect_fact_specs(compact)
    if subject:
        for spec in fact_specs:
            _append_unique(plan, _fact_query(spec, subject))

    dimension_terms = (
        "科室",
        "部门",
        "药物",
        "症状",
        "关系",
        "实体",
        "并发症",
        "检查",
        "治疗",
    )
    if (
        re.search(r"疾病种类|多少种疾病|疾病总数|疾病数量", compact)
        and not PATIENT_COUNT_RE.search(compact)
        and not any(term in compact for term in dimension_terms)
    ):
        _append_unique(
            plan,
            AnalysisQuery(
                title="疾病知识条目数量",
                purpose="统计分析库中的疾病知识条目数",
                sql="SELECT COUNT(*) AS 疾病知识条目数 FROM diseases",
                chart_type="metric",
            ),
        )

    if any(word in compact for word in ("科室", "部门")) and any(
        word in compact for word in ("统计", "数量", "分布", "排行", "排名", "最多", "关联")
    ):
        _append_unique(
            plan,
            AnalysisQuery(
                title="科室关联疾病分布",
                purpose="按科室统计知识图谱中关联的疾病条目数",
                sql=(
                    "SELECT department AS 科室, disease_count AS 关联疾病条目数 "
                    "FROM v_department_disease_counts "
                    "ORDER BY disease_count DESC LIMIT 30"
                ),
                chart_type="auto",
            ),
        )

    if any(word in compact for word in ("药", "药物")) and any(
        word in compact for word in ("统计", "数量", "分布", "排行", "排名", "最多", "关联")
    ) and not subject:
        _append_unique(
            plan,
            AnalysisQuery(
                title="药物关联疾病分布",
                purpose="按药物统计知识图谱中关联的疾病条目数",
                sql=(
                    "SELECT drug AS 药物, disease_count AS 关联疾病条目数 "
                    "FROM v_drug_disease_counts "
                    "ORDER BY disease_count DESC LIMIT 30"
                ),
                chart_type="auto",
            ),
        )

    if "症状" in compact and any(
        word in compact for word in ("统计", "数量", "分布", "排行", "排名", "最多", "关联")
    ) and not subject:
        _append_unique(
            plan,
            AnalysisQuery(
                title="症状关联疾病分布",
                purpose="按症状统计知识图谱中关联的疾病条目数",
                sql=(
                    "SELECT symptom AS 症状, disease_count AS 关联疾病条目数 "
                    "FROM v_top_symptoms "
                    "ORDER BY disease_count DESC LIMIT 30"
                ),
                chart_type="auto",
            ),
        )

    if "实体" in compact and any(word in compact for word in ("统计", "数量", "分布")):
        _append_unique(
            plan,
            AnalysisQuery(
                title="知识图谱实体分布",
                purpose="统计不同实体类型的数量",
                sql=(
                    "SELECT entity_type AS 实体类型, entity_count AS 数量 "
                    "FROM entity_stats ORDER BY entity_count DESC LIMIT 30"
                ),
                chart_type="auto",
            ),
        )

    if "关系" in compact and any(word in compact for word in ("统计", "数量", "分布")):
        _append_unique(
            plan,
            AnalysisQuery(
                title="知识图谱关系分布",
                purpose="统计不同关系类型的数量",
                sql=(
                    "SELECT display_name AS 关系, triple_count AS 数量 "
                    "FROM relation_stats ORDER BY triple_count DESC LIMIT 30"
                ),
                chart_type="auto",
            ),
        )

    if subject and not fact_specs and not plan.queries:
        _append_unique(
            plan,
            AnalysisQuery(
                title=f"{subject}知识摘要",
                purpose=f"查询{subject}的知识摘要和来源覆盖",
                sql=(
                    "SELECT name AS 疾病, description AS 简介, source_count AS 来源数 "
                    "FROM diseases WHERE name = ? LIMIT 1"
                ),
                params=(subject,),
                chart_type="table",
            ),
        )
    return plan


def needs_llm_planner(question: str, plan: AnalysisPlan) -> bool:
    """判断语义层是否需要 LLM 补充开放式分析计划。"""

    compact = normalize_question(question)
    return bool(COMPLEX_ANALYSIS_RE.search(compact)) or not plan.queries


def allowed_schema_prompt() -> str:
    return SCHEMA_DESCRIPTION
