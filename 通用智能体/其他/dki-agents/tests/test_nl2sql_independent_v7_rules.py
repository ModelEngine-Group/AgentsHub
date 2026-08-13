"""Selection-used v7 development regressions for bounded semantic families."""
from __future__ import annotations

import pytest

from dki.analysis.nl2sql_independent_v7 import compile_independent_v7_sql


DOMAINS = {
    "region": ["广州", "上海", "成都", "杭州", "南京", "西安"],
    "department": ["肾内科", "骨科", "泌尿外科", "消化内科"],
    "disease": ["甲状腺功能亢进症", "甲状腺功能减退症", "冠心病"],
    "exam": ["D-二聚体检测"],
    "kg_entity": ["甲状腺功能减退症"],
    "kg_entity_type": ["Department", "Disease", "Exam", "Symptom", "Treatment"],
    "kg_relation": [
        "BELONGS_TO_DEPT",
        "CONTRAINDICATED_FOR",
        "DIAGNOSED_BY",
        "HAS_COMPLICATION",
        "HAS_SYMPTOM",
        "TREATED_BY_DRUG",
    ],
}


@pytest.mark.parametrize(
    ("category", "question", "expected"),
    [
        (
            "filtering",
            "查找男性、年龄在48至66岁之间且来自广州的患者，返回编号与年龄。",
            "SELECT patient_id, age FROM patients WHERE gender='男' AND region='广州' "
            "AND age BETWEEN 48 AND 66 ORDER BY age, patient_id",
        ),
        (
            "aggregation_grouping",
            "按地区统计患者人数、平均年龄以及最高年龄，并按人数从多到少展示。",
            "SELECT region, COUNT(*) AS patient_count, ROUND(AVG(age),2) AS avg_age, "
            "MAX(age) AS max_age FROM patients GROUP BY region "
            "ORDER BY patient_count DESC, region",
        ),
        (
            "top_k",
            "列出平均就诊费用最高的4个科室，同时报告就诊次数。",
            "SELECT department, COUNT(*) AS visit_count, ROUND(AVG(cost),2) AS avg_cost "
            "FROM visits GROUP BY department ORDER BY avg_cost DESC, department LIMIT 4",
        ),
        (
            "multi_table_join",
            "连接图谱头实体与实体表，汇总各头实体类型的三元组数和平均提及次数。",
            "SELECT e.type, COUNT(*) AS triple_count, ROUND(AVG(e.mention_count),2) "
            "AS avg_head_mentions FROM kg_entities e JOIN kg_triples t "
            "ON t.head=e.name AND t.head_type=e.type GROUP BY e.type "
            "ORDER BY triple_count DESC, e.type",
        ),
        (
            "temporal",
            "按年份和季度统计就诊次数与去重患者数，按时间顺序输出。",
            "SELECT substr(visit_date,1,4) AS year, "
            "((CAST(substr(visit_date,6,2) AS INTEGER)-1)/3)+1 AS quarter, "
            "COUNT(*) AS visit_count, COUNT(DISTINCT patient_id) AS patient_count "
            "FROM visits GROUP BY year, quarter ORDER BY year, quarter",
        ),
        (
            "knowledge_graph",
            "统计每种疾病连接到多少个不同症状，并优先展示症状更丰富的疾病；"
            "关系代码等于HAS_SYMPTOM。",
            "SELECT head AS disease, COUNT(DISTINCT tail) AS symptom_count FROM kg_triples "
            "WHERE relation='HAS_SYMPTOM' GROUP BY head "
            "ORDER BY symptom_count DESC, disease",
        ),
        (
            "having",
            "找出就诊次数超过3次的疾病，并报告患者数和平均费用；"
            "只保留既定排序后的前31行。",
            "SELECT disease, COUNT(*) AS visit_count, COUNT(DISTINCT patient_id) "
            "AS patient_count, ROUND(AVG(cost),2) AS avg_cost FROM visits GROUP BY disease "
            "HAVING COUNT(*)>3 ORDER BY visit_count DESC, disease LIMIT 31",
        ),
        (
            "nested_correlated",
            "找出费用高于其所在科室平均费用的就诊，返回科室、就诊编号和费用。",
            "SELECT v.department, v.visit_id, v.cost FROM visits v WHERE v.cost>"
            "(SELECT AVG(x.cost) FROM visits x WHERE x.department=v.department) "
            "ORDER BY v.department, v.cost DESC, v.visit_id",
        ),
        (
            "cte",
            "先汇总各科室费用，再计算每个科室占全库费用的比例并按比例降序；"
            "只保留既定排序后的前22行。",
            "WITH dept_cost AS (SELECT department, SUM(cost) AS total_cost FROM visits "
            "GROUP BY department), grand AS (SELECT SUM(total_cost) AS all_cost FROM dept_cost) "
            "SELECT d.department, ROUND(d.total_cost,2) AS total_cost, "
            "ROUND(d.total_cost*100.0/g.all_cost,2) AS cost_pct FROM dept_cost d "
            "CROSS JOIN grand g ORDER BY cost_pct DESC, d.department LIMIT 22",
        ),
        (
            "set_operation",
            "求实际处方药与图谱治疗药物的交集，返回两边都出现的药物名称；"
            "关系代码等于TREATED_BY_DRUG。",
            "SELECT drug_prescribed AS drug FROM visits WHERE LENGTH(drug_prescribed) "
            "INTERSECT SELECT tail FROM kg_triples WHERE relation='TREATED_BY_DRUG' "
            "ORDER BY drug",
        ),
        (
            "window",
            "在每个科室内部按费用给就诊记录编号，展示各科室费用最高的前三次就诊；"
            "rn不高于3。",
            "WITH ranked AS (SELECT department, visit_id, cost, ROW_NUMBER() OVER "
            "(PARTITION BY department ORDER BY cost DESC, visit_id) AS rn FROM visits) "
            "SELECT department, visit_id, cost, rn FROM ranked WHERE rn<=3 "
            "ORDER BY department, rn",
        ),
        (
            "multi_table_join_set",
            "找出西安患者的疾病集合与全库其他地区疾病集合的交集。",
            "SELECT v.disease FROM visits v JOIN patients p ON p.patient_id=v.patient_id "
            "WHERE p.region='西安' INTERSECT SELECT v.disease FROM visits v "
            "JOIN patients p ON p.patient_id=v.patient_id WHERE p.region<>'西安' "
            "ORDER BY disease",
        ),
        (
            "cte_ratio",
            "用两个公用表表达式分别计算地区患者数与地区就诊量，再合并展示"
            "南京在内的全部地区指标；只保留既定排序后的前26行。",
            "WITH patient_region AS (SELECT region, COUNT(*) AS patient_count FROM patients "
            "GROUP BY region), visit_region AS (SELECT p.region, COUNT(v.visit_id) AS visit_count "
            "FROM patients p JOIN visits v ON v.patient_id=p.patient_id GROUP BY p.region) "
            "SELECT pr.region, pr.patient_count, vr.visit_count, "
            "ROUND(vr.visit_count*1.0/pr.patient_count,2) AS visits_per_patient "
            "FROM patient_region pr JOIN visit_region vr ON vr.region=pr.region "
            "ORDER BY visits_per_patient DESC, pr.region LIMIT 26",
        ),
    ],
)
def test_v7_operation_family_regressions(category: str, question: str, expected: str):
    assert category
    assert compile_independent_v7_sql(question, DOMAINS) == expected


def test_v7_text_slots_must_be_grounded():
    question = (
        "筛出火星科中费用高于100元的就诊，给出就诊编号、日期、费用和住院天数。"
    )
    assert compile_independent_v7_sql(question, DOMAINS) is None


def test_v7_unknown_or_mutating_request_abstains():
    assert compile_independent_v7_sql("删除全部患者记录。", DOMAINS) is None
