"""Selection-used independent-v5 development regressions.

The fixtures cover parameterised operation families discovered on the
development member only.  The runtime compiler never reads that dataset.
"""
from __future__ import annotations

import pytest

from dki.analysis.nl2sql_sketch import compile_independent_structured_sql
from dki.analysis.nl2sql import NL2SQL
from dki.analysis.nl2sql_compile_result import CompileDisposition


DOMAINS = {
    "region": ["北京", "成都", "西安", "长沙"],
    "department": ["心血管内科", "神经内科", "泌尿外科"],
}


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "列出年龄在20岁至44岁之间(含边界)的患者ID、姓名和年龄,"
            "并按年龄升序、患者ID升序排列。",
            "SELECT patient_id, name, age FROM patients WHERE age BETWEEN 20 AND 44 "
            "ORDER BY age ASC, patient_id ASC",
        ),
        (
            "查询性别为女且地区不是西安的患者姓名、地区和年龄,"
            "结果按地区、姓名依次升序排列。",
            "SELECT name, region, age FROM patients WHERE gender = '女' "
            "AND region <> '西安' ORDER BY region ASC, name ASC",
        ),
        (
            "列出费用高于4900元且住院天数至少为3天的就诊ID、患者ID、"
            "费用和住院天数,按费用降序、就诊ID升序排列。",
            "SELECT visit_id, patient_id, cost, los_days FROM visits "
            "WHERE cost > 4900 AND los_days >= 3 ORDER BY cost DESC, visit_id ASC",
        ),
        (
            "输出年龄除以10后的整数部分等于3的患者ID、姓名与年龄,"
            "按年龄和患者ID升序。",
            "SELECT patient_id, name, age FROM patients "
            "WHERE CAST(age / 10 AS INTEGER) = 3 ORDER BY age ASC, patient_id ASC",
        ),
        (
            "查询地区为北京且姓名长度不少于2个字符的患者姓名、性别和年龄,"
            "按姓名升序。",
            "SELECT name, gender, age FROM patients WHERE region = '北京' "
            "AND LENGTH(name) >= 2 ORDER BY name ASC",
        ),
        (
            "找出患者人数最多的前5个地区,输出地区和患者人数;"
            "人数相同时按地区升序。",
            "SELECT region, COUNT(*) AS patient_count FROM patients GROUP BY region "
            "ORDER BY patient_count DESC, region ASC LIMIT 5",
        ),
        (
            "按就诊次数取前4种疾病,输出疾病及次数;次数相同时按疾病升序。",
            "SELECT disease, COUNT(*) AS visit_count FROM visits GROUP BY disease "
            "ORDER BY visit_count DESC, disease ASC LIMIT 4",
        ),
        (
            "从置信度不低于0.90的三元组中选出观察次数最高的前4条,"
            "输出头实体、关系、尾实体、观察次数和置信度;依次按观察次数降序、"
            "置信度降序、头实体和尾实体升序。",
            "SELECT head, relation, tail, count, confidence FROM kg_triples "
            "WHERE confidence >= 0.9 ORDER BY count DESC, confidence DESC, "
            "head ASC, tail ASC LIMIT 4",
        ),
        (
            "统计每个科室中年龄不少于45岁的不同患者数,输出科室和去重患者数,"
            "按人数降序、科室升序;只统计能匹配患者信息的就诊。",
            "SELECT v.department, COUNT(DISTINCT p.patient_id) AS patient_count "
            "FROM visits AS v JOIN patients AS p ON p.patient_id = v.patient_id "
            "WHERE p.age >= 45 GROUP BY v.department "
            "ORDER BY patient_count DESC, v.department ASC",
        ),
        (
            "统计2025年各季度的就诊次数,输出季度编号1至4和次数,按季度升序;"
            "为消除边界与常量歧义,本查询采用的精确字面量还包括:%m、3、"
            "2025-01-01、2026-01-01。",
            "SELECT CAST((CAST(STRFTIME('%m', visit_date) AS INTEGER) - 1) / 3 "
            "AS INTEGER) + 1 AS quarter_no, COUNT(*) AS visit_count FROM visits "
            "WHERE visit_date >= '2025-01-01' AND visit_date < '2026-01-01' "
            "GROUP BY quarter_no ORDER BY quarter_no ASC",
        ),
        (
            "按科室统计就诊数,只保留就诊数至少为8的科室;输出科室和就诊数,"
            "按就诊数降序、科室升序。",
            "SELECT department, COUNT(*) AS visit_count FROM visits "
            "GROUP BY department HAVING COUNT(*) >= 8 "
            "ORDER BY visit_count DESC, department ASC",
        ),
        (
            "用递归CTE生成0到15的住院天数序列,并左连接就诊表统计每个天数"
            "对应的就诊数;输出天数和次数,按天数升序。",
            "WITH RECURSIVE days(n) AS (SELECT 0 UNION ALL SELECT n + 1 FROM days "
            "WHERE n < 15) SELECT days.n AS los_days, "
            "COUNT(v.visit_id) AS visit_count FROM days LEFT JOIN visits AS v "
            "ON v.los_days = days.n GROUP BY days.n ORDER BY days.n ASC",
        ),
        (
            "合并就诊疾病名称与知识图谱实体名称,使用UNION去重;"
            "输出统一字段name并按名称升序。",
            "SELECT disease AS name FROM visits UNION SELECT name FROM kg_entities "
            "ORDER BY name ASC",
        ),
        (
            "在每个实体类型内按提及次数降序、名称升序编号,只保留每类前3名;"
            "输出类型、名称、提及次数和行号,按类型、行号升序。",
            "SELECT type, name, mention_count, row_no FROM "
            "(SELECT type, name, mention_count, ROW_NUMBER() OVER "
            "(PARTITION BY type ORDER BY mention_count DESC, name ASC) AS row_no "
            "FROM kg_entities) WHERE row_no <= 3 ORDER BY type ASC, row_no ASC",
        ),
    ],
)
def test_parameterised_sketches(question: str, expected: str):
    assert compile_independent_structured_sql(question, DOMAINS) == expected


def test_database_literal_must_be_grounded():
    question = (
        "查询性别为女且地区不是火星的患者姓名、地区和年龄,"
        "结果按地区、姓名依次升序排列。"
    )
    assert compile_independent_structured_sql(question, DOMAINS) is None


def test_unknown_shape_abstains():
    assert compile_independent_structured_sql("删除所有患者。", DOMAINS) is None


def test_complete_temporal_sketch_precedes_partial_temporal_owner():
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS
    question = (
        "统计2025年各季度的就诊次数,输出季度编号1至4和次数,按季度升序;"
        "为消除边界与常量歧义,本查询采用的精确字面量还包括:%m、3、"
        "2025-01-01、2026-01-01。"
    )

    result = compiler._rule_sql_normalized_result(question)

    assert result.disposition is CompileDisposition.SQL
    assert result.sql is not None
    assert "AS quarter_no" in result.sql
