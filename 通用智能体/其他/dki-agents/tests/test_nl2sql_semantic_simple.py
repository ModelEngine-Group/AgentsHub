"""Focused regression coverage for the simple medical SQL compiler."""

import pytest

from dki.analysis.nl2sql_semantic_simple import compile_simple_medical_sql


DOMAINS = {
    "region": ["北京", "上海", "深圳"],
    "department": ["内科", "外科", "心内科"],
    "disease": ["高血压", "糖尿病"],
    "drug": ["阿司匹林"],
    "exam": ["血常规"],
}


def test_grouped_top_k_is_not_reduced_to_a_scalar_metric():
    assert compile_simple_medical_sql("就诊次数最多的前3个科室。", DOMAINS) == (
        "SELECT department, COUNT(*) AS visit_count FROM visits "
        "GROUP BY department ORDER BY visit_count DESC, department ASC LIMIT 3"
    )


def test_grouped_metrics_keep_every_requested_output_and_explicit_order():
    question = (
        "统计各科室就诊次数、不同患者数、费用总额和平均费用，"
        "返回科室、就诊次数、不同患者数、费用总额和平均费用；"
        "按费用总额降序、平均费用升序、科室升序排列。"
    )
    assert compile_simple_medical_sql(question, DOMAINS) == (
        "SELECT department, COUNT(*) AS visit_count, "
        "COUNT(DISTINCT patient_id) AS patient_count, SUM(cost) AS total_cost, "
        "AVG(cost) AS avg_cost FROM visits GROUP BY department "
        "ORDER BY total_cost DESC, avg_cost ASC, department ASC"
    )


def test_each_city_patient_count_and_average_age_uses_patient_group():
    assert compile_simple_medical_sql(
        "每个城市的患者数和平均年龄分别是多少",
        DOMAINS,
    ) == (
        "SELECT region, COUNT(*) AS patient_count, "
        "ROUND(AVG(age), 2) AS avg_age "
        "FROM patients GROUP BY region "
        "ORDER BY patient_count DESC"
    )


def test_top_visit_with_patient_field_uses_join_and_preserves_projection_order():
    question = "费用最高的前3次就诊，返回患者编号、患者年龄、就诊编号和费用。"
    assert compile_simple_medical_sql(question, DOMAINS) == (
        "SELECT p.patient_id, p.age, v.visit_id, v.cost FROM visits AS v "
        "JOIN patients AS p ON p.patient_id = v.patient_id "
        "ORDER BY v.cost DESC, v.visit_date ASC, v.visit_id ASC LIMIT 3"
    )


def test_patient_age_natural_order_keeps_secondary_key():
    question = (
        "查询年龄不低于60岁的患者，返回患者编号、姓名和年龄，"
        "年龄较大者在前，同龄时按姓名升序。"
    )
    assert compile_simple_medical_sql(question, DOMAINS) == (
        "SELECT patient_id, name, age FROM patients WHERE age >= 60 "
        "ORDER BY age DESC, name ASC, patient_id ASC"
    )


def test_explicit_visit_order_overrides_natural_cost_ties():
    question = (
        "查询费用不低于0元的就诊，返回就诊编号、费用和就诊日期，"
        "按费用降序、就诊日期降序。"
    )
    assert compile_simple_medical_sql(question, DOMAINS) == (
        "SELECT visit_id, cost, visit_date FROM visits WHERE cost >= 0 "
        "ORDER BY cost DESC, visit_date DESC, visit_id ASC"
    )


def test_patient_filter_on_visit_detail_requires_join():
    question = "查询年龄至少50岁的患者的就诊记录，返回就诊编号、费用。"
    assert compile_simple_medical_sql(question, DOMAINS) == (
        "SELECT v.visit_id, v.cost FROM patients AS p JOIN visits AS v "
        "ON v.patient_id = p.patient_id WHERE p.age >= 50 "
        "ORDER BY v.visit_date ASC, v.visit_id ASC"
    )


def test_join_preserves_grounded_age_date_and_cost_filters():
    question = (
        "查询上海地区年龄不低于50岁的患者在2025-01-01至2025-01-31期间"
        "费用不低于1000元的就诊记录，返回患者姓名、就诊编号、就诊日期和费用。"
    )
    assert compile_simple_medical_sql(question, DOMAINS) == (
        "SELECT p.name, v.visit_id, v.visit_date, v.cost FROM patients AS p "
        "JOIN visits AS v ON v.patient_id = p.patient_id "
        "WHERE p.age >= 50 AND p.region = '上海' "
        "AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31' "
        "AND v.cost >= 1000 ORDER BY v.visit_date ASC, v.visit_id ASC"
    )


@pytest.mark.parametrize(
    "question",
    [
        "查询费用不低于元的就诊，返回就诊编号。",
        "查询2025-01-01的就诊，返回就诊编号。",
        "查询年龄不低于岁的患者，返回患者编号。",
    ],
)
def test_malformed_date_cost_and_age_filters_fail_closed(question):
    assert compile_simple_medical_sql(question, DOMAINS) is None


def test_simple_compiler_abstains_for_mixed_disjunctive_region_polarity():
    question = "查询深圳患者或北京地区以外患者，返回患者编号。"

    assert compile_simple_medical_sql(question, DOMAINS) is None


def test_grouped_having_request_is_not_reduced_to_scalar_visit_count():
    question = "就诊次数超过20次的疾病有哪些"

    assert compile_simple_medical_sql(question, DOMAINS) is None


def test_peer_average_having_is_not_reduced_to_plain_grouped_count():
    question = "就诊次数超过平均每种疾病就诊次数的疾病有哪些"

    assert compile_simple_medical_sql(question, DOMAINS) is None
