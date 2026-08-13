"""Synthetic SQLite regressions for advanced semantic NL2SQL compilers."""
from __future__ import annotations

import sqlite3

from dki.analysis.nl2sql_age_average_intent import (
    AgeAverageDisposition,
    AgeAverageOrder,
    AgeAverageReason,
    classify_age_average_intent,
)
from dki.analysis.nl2sql_temporal_cohort_intent import (
    TemporalCohortDisposition,
    TemporalCohortKind,
    TemporalCohortReason,
    classify_temporal_cohort_intent,
)
from dki.analysis.nl2sql_compile_result import CompileDisposition
from dki.analysis.nl2sql_semantic_advanced import (
    _compile_department_topk,
    _compile_episode_segmentation,
    _compile_first_last,
    _compile_lag_gap,
    _compile_ntile_quartiles,
    _compile_patient_exists,
    _compile_patient_rollup,
    _compile_patient_total_peer_average,
    _compile_region_topk,
    _compile_required_departments,
    _compile_running_cost,
    compile_advanced_medical_result,
    compile_advanced_medical_sql,
)


DOMAINS = {
    "region": ["北京", "深圳"],
    "department": ["内科", "外科"],
    "disease": ["疾病甲", "疾病乙"],
    "drug": ["药A", "药B", "药C", "药D"],
    "exam": ["检查甲"],
}


def _medical_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE patients (
            patient_id INTEGER PRIMARY KEY,
            name TEXT,
            gender TEXT,
            age INTEGER,
            region TEXT
        );
        CREATE TABLE visits (
            visit_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            visit_date TEXT,
            department TEXT,
            disease TEXT,
            drug_prescribed TEXT,
            exam TEXT,
            cost REAL,
            los_days INTEGER
        );
        """
    )
    return connection


def test_region_topk_uses_unqualified_cte_age_predicate():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 50, "深圳"),
            (2, "乙", "女", 35, "深圳"),
            (3, "丙", "男", 55, "北京"),
        ],
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (2, 2, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 1000.0, 1),
            (3, 3, "2025-01-01", "外科", "疾病乙", "药B", "检查甲", 300.0, 1),
        ],
    )

    sql = compile_advanced_medical_sql(
        "每个地区累计费用排名前1，年龄至少40岁的患者，返回地区、患者编号、总费用和排名。",
        DOMAINS,
    )

    assert sql is not None
    assert "WHERE age >= 40" in sql
    assert "age.age" not in sql
    rows = connection.execute(sql).fetchall()
    assert {row[1] for row in rows} == {1, 3}


def test_department_topk_aliases_drug_in_derived_table():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (11, 1, "2025-01-03", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (12, 1, "2025-01-04", "内科", "疾病甲", "药B", "检查甲", 200.0, 1),
            (13, 2, "2025-01-05", "外科", "疾病乙", "药C", "检查甲", 150.0, 1),
            (14, 2, "2025-01-06", "外科", "疾病乙", "药D", "检查甲", 50.0, 1),
        ],
    )

    sql = compile_advanced_medical_sql(
        "在2025-01-01至2025-12-31期间，每个科室按费用降序取前1，"
        "返回科室、就诊编号、处方药物和排名。",
        DOMAINS,
    )

    assert sql is not None
    assert "v.drug_prescribed AS drug" in sql
    rows = connection.execute(sql).fetchall()
    assert {row[0]: row[1:] for row in rows} == {
        "内科": (12, "药B", 1),
        "外科": (13, "药C", 1),
    }


def test_department_topk_rejects_unavailable_derived_projection():
    assert (
        _compile_department_topk(
            "每个科室按费用降序取前1，返回科室、就诊编号、累计费用和排名。",
            DOMAINS,
        )
        is None
    )


def test_monthly_yoy_uses_prior_calendar_months_not_observed_row_offsets():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "2024-01-10", "内科", "疾病甲", "药A", "检查甲", 10.0, 1),
            (2, 1, "2024-03-10", "内科", "疾病甲", "药A", "检查甲", 30.0, 1),
            (3, 1, "2025-01-10", "内科", "疾病甲", "药A", "检查甲", 25.0, 1),
            (4, 1, "2025-03-10", "内科", "疾病甲", "药A", "检查甲", 45.0, 1),
            (5, 1, "2025-04-10", "内科", "疾病甲", "药A", "检查甲", 80.0, 1),
        ],
    )

    sql = compile_advanced_medical_sql(
        "按月统计2025-01-01至2025-12-31的总费用同比变化，"
        "返回月份、总费用、上年同期和同比变化。",
        DOMAINS,
    )

    assert sql is not None
    assert "JOIN monthly AS prior_month" in sql
    assert "LAG(" not in sql
    assert connection.execute(sql).fetchall() == [
        ("2025-01", 25.0, 10.0, 15.0),
        ("2025-03", 45.0, 30.0, 15.0),
    ]


def test_grouped_having_applies_patient_qualifiers_before_aggregation():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 50, "深圳"),
            (2, "乙", "女", 35, "深圳"),
            (3, "丙", "男", 50, "深圳"),
            (4, "丁", "女", 50, "北京"),
        ],
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (2, 1, "2025-01-02", "内科", "疾病甲", "药A", "检查甲", 200.0, 1),
            (3, 2, "2025-01-03", "内科", "疾病甲", "药A", "检查甲", 1000.0, 1),
            (4, 2, "2025-01-04", "内科", "疾病甲", "药A", "检查甲", 1000.0, 1),
            (5, 3, "2025-01-05", "内科", "疾病甲", "药A", "检查甲", 800.0, 1),
            (6, 3, "2025-01-06", "内科", "疾病甲", "药A", "检查甲", 800.0, 1),
            (7, 4, "2025-01-07", "外科", "疾病乙", "药B", "检查甲", 900.0, 1),
            (8, 4, "2025-01-08", "外科", "疾病乙", "药B", "检查甲", 900.0, 1),
        ],
    )

    sql = compile_advanced_medical_sql(
        "按科室统计2025-01-01至2025-12-31女性、年龄至少40岁、深圳地区患者的"
        "就诊次数和平均费用，筛选就诊次数至少2且平均费用至少100，"
        "返回科室、就诊次数和平均费用。",
        DOMAINS,
    )

    assert sql is not None
    assert "JOIN patients AS p ON p.patient_id = v.patient_id" in sql
    assert "p.age >= 40" in sql
    assert "p.gender = '女'" in sql
    assert "p.region = '深圳'" in sql
    assert connection.execute(sql).fetchall() == [("内科", 2, 150.0)]


def test_age_peer_average_preserves_explicit_positive_region_filter():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 60, "深圳"),
            (2, "乙", "女", 40, "深圳"),
            (3, "丙", "女", 80, "北京"),
            (4, "丁", "女", 20, "北京"),
            (5, "戊", "男", 99, "深圳"),
        ],
    )

    sql = compile_advanced_medical_sql(
        "找出深圳地区年龄高于同地区同性别患者平均年龄的女性患者，"
        "返回患者编号、姓名和年龄。",
        DOMAINS,
    )

    assert sql is not None
    assert "peer.region = p.region" in sql
    assert "peer.gender = p.gender" in sql
    assert "p.region = '深圳'" in sql
    assert "p.gender = '女'" in sql
    assert connection.execute(sql).fetchall() == [(1, "甲", 60)]


def test_age_peer_average_intent_records_exclusive_ownership():
    global_intent = classify_age_average_intent(
        "找出年龄高于全部患者平均年龄的患者，返回患者ID、姓名、年龄和地区；"
        "按年龄降序、患者ID升序排列。",
        DOMAINS,
    )
    peer_intent = classify_age_average_intent(
        "找出年龄高于同地区同性别患者平均年龄的女病人，"
        "返回患者编号和年龄，按年龄降序、患者ID升序排列。",
        DOMAINS,
    )
    filtered_global_intent = classify_age_average_intent(
        "找出深圳地区年龄至少40岁的女性患者，年龄高于全部患者平均年龄，"
        "返回患者ID、姓名、年龄和地区；按年龄降序、患者ID升序排列。",
        DOMAINS,
    )
    global_suffix_intent = classify_age_average_intent(
        "找出年龄高于全部患者平均年龄且在2025年就诊过的女患者，"
        "返回患者编号和年龄，按年龄升序前3名。",
        DOMAINS,
    )
    fixed_peer_gender_intent = classify_age_average_intent(
        "找出年龄高于同地区女性患者平均年龄的患者，返回患者编号和年龄。",
        DOMAINS,
    )
    fixed_peer_age_intent = classify_age_average_intent(
        "找出年龄高于同地区年龄至少40岁患者平均年龄的患者，返回患者编号和年龄。",
        DOMAINS,
    )
    other_patient_peer_intent = classify_age_average_intent(
        "找出年龄高于同地区其他患者平均年龄的患者，返回患者编号和年龄。",
        DOMAINS,
    )
    other_patient_peer_alt_intent = classify_age_average_intent(
        "找出年龄高于同地区其它患者平均年龄的患者，返回患者编号和年龄。",
        DOMAINS,
    )
    derived_peer_average_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号、年龄和同地区平均年龄。",
        DOMAINS,
    )
    chinese_outpatient_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄且在2025年有门诊记录的患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    follow_up_visit_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄且曾复诊过的患者，返回患者编号和年龄。",
        DOMAINS,
    )
    visit_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄且曾在肿瘤科就诊的患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    order_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄，按姓名升序排列。",
        DOMAINS,
    )
    chinese_topk_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄，取前三名。",
        DOMAINS,
    )
    bare_chinese_topk_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄，保留3名。",
        DOMAINS,
    )
    projection_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号、年龄和疾病。",
        DOMAINS,
    )
    aggregate_projection_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和平均年龄。",
        DOMAINS,
    )
    distinct_projection_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回不同的患者编号和年龄。",
        DOMAINS,
    )
    multi_order_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄的患者，返回患者编号和年龄；"
        "按姓名升序排列；按年龄降序、患者ID升序排列。",
        DOMAINS,
    )
    unrenderable_age_intent = classify_age_average_intent(
        "找出年龄40岁的、年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    renderable_age_intent = classify_age_average_intent(
        "找出满40岁且年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    bounded_outer_age_intent = classify_age_average_intent(
        "找出年龄至少40岁且不超过50岁且年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    bounded_outer_age_sql = compile_advanced_medical_sql(
        "找出年龄至少40岁且不超过50岁且年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    unrenderable_region_intent = classify_age_average_intent(
        "找出广州地区年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    unsupported_trailing_region_negation_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄且北京地区以外的患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    post_output_region_negation_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄，北京地区以外。",
        DOMAINS,
    )
    supported_leading_region_negation_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄且排除北京地区的患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    ungrounded_disease_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄的糖尿病患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    negated_gender_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄的非女性患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    disjunctive_gender_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄的女性或男性患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    identity_intent = classify_age_average_intent(
        "找出姓名为甲且年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    trailing_identity_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄，患者姓名为甲。",
        DOMAINS,
    )
    grouping_intent = classify_age_average_intent(
        "按地区分组找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    trailing_grouping_intent = classify_age_average_intent(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄，按地区分组。",
        DOMAINS,
    )
    disjunctive_peer_intent = classify_age_average_intent(
        "找出年龄高于同地区或同性别患者平均年龄的患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    ordinary_average = classify_age_average_intent(
        "按地区统计患者平均年龄，筛选平均年龄高于40，返回地区和平均年龄。",
        DOMAINS,
    )
    other_patient_peer_sql = compile_advanced_medical_sql(
        "找出年龄高于同地区其他患者平均年龄的患者，返回患者编号和年龄。",
        DOMAINS,
    )
    unsupported_trailing_region_negation_sql = compile_advanced_medical_sql(
        "找出年龄高于同地区患者平均年龄且北京地区以外的患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )
    post_output_region_negation_sql = compile_advanced_medical_sql(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄，北京地区以外。",
        DOMAINS,
    )
    trailing_grouping_sql = compile_advanced_medical_sql(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄，按地区分组。",
        DOMAINS,
    )
    trailing_identity_sql = compile_advanced_medical_sql(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄，患者姓名为甲。",
        DOMAINS,
    )
    supported_leading_region_negation_sql = compile_advanced_medical_sql(
        "找出年龄高于同地区患者平均年龄且排除北京地区的患者，"
        "返回患者编号和年龄。",
        DOMAINS,
    )

    assert global_intent.disposition is AgeAverageDisposition.EXPLICIT_GLOBAL
    assert global_intent.projection == ("patient_id", "name", "age", "region")
    assert global_intent.order is AgeAverageOrder.DEFAULT
    assert filtered_global_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.NONCANONICAL_GLOBAL_SURFACE in filtered_global_intent.reasons
    assert AgeAverageReason.UNSUPPORTED_GENDER_FILTER in filtered_global_intent.reasons
    assert AgeAverageReason.UNSUPPORTED_REGION_FILTER in filtered_global_intent.reasons
    assert AgeAverageReason.UNSUPPORTED_AGE_BOUND in filtered_global_intent.reasons
    assert global_suffix_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.VISIT_QUALIFIER in global_suffix_intent.reasons
    assert AgeAverageReason.UNSUPPORTED_GENDER_FILTER in global_suffix_intent.reasons
    assert AgeAverageReason.UNSUPPORTED_ORDER in global_suffix_intent.reasons
    assert fixed_peer_gender_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert (
        AgeAverageReason.UNSUPPORTED_PEER_DEMOGRAPHIC_SCOPE
        in fixed_peer_gender_intent.reasons
    )
    assert fixed_peer_age_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert fixed_peer_age_intent.has_fixed_peer_qualifier_scope
    assert (
        AgeAverageReason.UNSUPPORTED_PEER_DEMOGRAPHIC_SCOPE
        in fixed_peer_age_intent.reasons
    )
    assert other_patient_peer_intent.disposition is AgeAverageDisposition.ADVANCED_PEER
    assert other_patient_peer_intent.excludes_outer_patient_from_peer
    assert other_patient_peer_alt_intent.disposition is AgeAverageDisposition.ADVANCED_PEER
    assert other_patient_peer_alt_intent.excludes_outer_patient_from_peer
    assert other_patient_peer_sql is not None
    assert "peer.patient_id <> p.patient_id" in other_patient_peer_sql
    assert derived_peer_average_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert (
        AgeAverageReason.UNSUPPORTED_DERIVED_PEER_PROJECTION
        in derived_peer_average_intent.reasons
    )
    assert chinese_outpatient_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.VISIT_QUALIFIER in chinese_outpatient_intent.reasons
    assert follow_up_visit_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.VISIT_QUALIFIER in follow_up_visit_intent.reasons
    assert peer_intent.disposition is AgeAverageDisposition.ADVANCED_PEER
    assert peer_intent.peer_by_region and peer_intent.peer_by_gender
    assert peer_intent.outer_gender == "女"
    assert peer_intent.projection == ("patient_id", "age")
    assert peer_intent.order is AgeAverageOrder.DEFAULT
    assert visit_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.VISIT_QUALIFIER in visit_intent.reasons
    assert order_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.UNSUPPORTED_ORDER in order_intent.reasons
    assert chinese_topk_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.UNSUPPORTED_ORDER in chinese_topk_intent.reasons
    assert bare_chinese_topk_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.UNSUPPORTED_ORDER in bare_chinese_topk_intent.reasons
    assert projection_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.UNSUPPORTED_PROJECTION in projection_intent.reasons
    assert aggregate_projection_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert aggregate_projection_intent.has_aggregate_projection
    assert (
        AgeAverageReason.UNSUPPORTED_AGGREGATE_PROJECTION
        in aggregate_projection_intent.reasons
    )
    assert distinct_projection_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.UNSUPPORTED_DISTINCT_PROJECTION in distinct_projection_intent.reasons
    assert multi_order_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.UNSUPPORTED_ORDER in multi_order_intent.reasons
    assert unrenderable_age_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.UNSUPPORTED_AGE_BOUND in unrenderable_age_intent.reasons
    assert renderable_age_intent.disposition is AgeAverageDisposition.ADVANCED_PEER
    assert bounded_outer_age_intent.disposition is AgeAverageDisposition.ADVANCED_PEER
    assert bounded_outer_age_sql is not None
    assert "p.age BETWEEN 40 AND 50" in bounded_outer_age_sql
    assert unrenderable_region_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.UNSUPPORTED_REGION_FILTER in unrenderable_region_intent.reasons
    assert (
        unsupported_trailing_region_negation_intent.disposition
        is AgeAverageDisposition.TERMINAL_ABSTAIN
    )
    assert (
        AgeAverageReason.UNSUPPORTED_REGION_FILTER
        in unsupported_trailing_region_negation_intent.reasons
    )
    assert unsupported_trailing_region_negation_sql is None
    assert (
        post_output_region_negation_intent.disposition
        is AgeAverageDisposition.TERMINAL_ABSTAIN
    )
    assert (
        AgeAverageReason.UNSUPPORTED_REGION_FILTER
        in post_output_region_negation_intent.reasons
    )
    assert post_output_region_negation_sql is None
    assert (
        supported_leading_region_negation_intent.disposition
        is AgeAverageDisposition.ADVANCED_PEER
    )
    assert supported_leading_region_negation_sql is not None
    assert "p.region NOT IN ('北京')" in supported_leading_region_negation_sql
    assert ungrounded_disease_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.VISIT_QUALIFIER in ungrounded_disease_intent.reasons
    assert negated_gender_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.UNSUPPORTED_GENDER_FILTER in negated_gender_intent.reasons
    assert disjunctive_gender_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.UNSUPPORTED_GENDER_FILTER in disjunctive_gender_intent.reasons
    assert identity_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.UNSUPPORTED_IDENTITY_FILTER in identity_intent.reasons
    assert trailing_identity_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.UNSUPPORTED_IDENTITY_FILTER in trailing_identity_intent.reasons
    assert trailing_identity_sql is None
    assert grouping_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.UNSUPPORTED_GROUPING in grouping_intent.reasons
    assert trailing_grouping_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.UNSUPPORTED_GROUPING in trailing_grouping_intent.reasons
    assert trailing_grouping_sql is None
    assert disjunctive_peer_intent.disposition is AgeAverageDisposition.TERMINAL_ABSTAIN
    assert AgeAverageReason.DISJUNCTIVE_PEER_SCOPE in disjunctive_peer_intent.reasons
    assert ordinary_average.disposition is AgeAverageDisposition.NOT_APPLICABLE


def test_age_peer_average_preserves_direct_patient_gender_synonym():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 60, "深圳"),
            (2, "乙", "女", 40, "深圳"),
            (3, "丙", "男", 90, "深圳"),
            (4, "丁", "男", 30, "深圳"),
        ],
    )

    sql = compile_advanced_medical_sql(
        "找出年龄高于同地区同性别患者平均年龄的女病人，"
        "返回患者编号和年龄。",
        DOMAINS,
    )

    assert sql is not None
    assert "peer.region = p.region" in sql
    assert "peer.gender = p.gender" in sql
    assert "p.gender = '女'" in sql
    assert connection.execute(sql).fetchall() == [(1, 60)]


def test_projection_aware_compiler_precedes_family_fallback(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module

    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda _question, _domains, **_kwargs: "SELECT 'family' AS source",
    )

    sql = compile_advanced_medical_sql(
        "找出深圳地区年龄高于同地区同性别患者平均年龄的女性患者，"
        "返回患者编号、姓名和年龄。",
        DOMAINS,
    )

    assert sql is not None
    assert sql != "SELECT 'family' AS source"
    assert "peer.region = p.region" in sql


def test_grouped_having_retains_repeated_bounds_for_one_metric():
    sql = compile_advanced_medical_sql(
        "按科室统计就诊次数，筛选就诊次数至少2且不超过5，"
        "返回科室和就诊次数。",
        DOMAINS,
    )

    assert sql is not None
    assert "HAVING COUNT(*) >= 2 AND COUNT(*) <= 5" in sql


def test_department_topk_joins_patients_for_demographic_filters():
    sql = _compile_department_topk(
        "每个科室按费用降序取前1，年龄至少40岁的女性深圳患者，"
        "返回科室、就诊编号和排名。",
        DOMAINS,
    )

    assert sql is not None
    assert "FROM visits AS v JOIN patients AS p ON p.patient_id = v.patient_id" in sql
    assert "p.age >= 40" in sql
    assert "p.gender = '女'" in sql
    assert "p.region = '深圳'" in sql


def test_region_topk_preserves_an_explicit_positive_region_filter():
    sql = _compile_region_topk(
        "每个地区累计费用排名前1，年龄至少40岁的女性深圳患者，"
        "返回地区、患者编号、总费用和排名。",
        DOMAINS,
    )

    assert sql is not None
    assert "p.age >= 40" in sql
    assert "p.gender = '女'" in sql
    assert "p.region = '深圳'" in sql


def test_patient_total_peer_average_projects_and_filters_demographics():
    sql = _compile_patient_total_peer_average(
        "找出深圳地区年龄至少40岁的女性患者，其累计费用高于同地区平均累计费用，"
        "返回患者编号、姓名和总费用。",
        DOMAINS,
    )

    assert sql is not None
    assert "p.age, p.gender" in sql
    assert "pt.age >= 40" in sql
    assert "pt.gender = '女'" in sql
    assert "pt.region = '深圳'" in sql


def test_first_last_restricts_the_patient_cohort_by_demographics():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 50, "深圳"),
            (2, "乙", "男", 50, "深圳"),
        ],
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (2, 1, "2025-01-20", "外科", "疾病乙", "药B", "检查甲", 200.0, 1),
            (3, 2, "2025-01-02", "内科", "疾病甲", "药A", "检查甲", 300.0, 1),
            (4, 2, "2025-01-22", "外科", "疾病乙", "药B", "检查甲", 400.0, 1),
        ],
    )

    sql = _compile_first_last(
        "找出至少有2次就诊的女性患者，返回患者编号、首次就诊日期和末次就诊日期。",
        DOMAINS,
    )

    assert sql is not None
    assert "p.gender = '女'" in sql
    assert connection.execute(sql).fetchall() == [(1, "2025-01-01", "2025-01-20")]


def test_ntile_restricts_the_ranked_population_by_demographics():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 50, "深圳"),
            (2, "乙", "男", 50, "深圳"),
            (3, "丙", "女", 40, "北京"),
        ],
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (2, 2, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 1000.0, 1),
            (3, 3, "2025-01-01", "外科", "疾病乙", "药B", "检查甲", 300.0, 1),
        ],
    )

    sql = _compile_ntile_quartiles(
        "按累计费用将女性患者分为四分位，返回患者编号、总费用和四分位。",
        DOMAINS,
    )

    assert sql is not None
    assert "p.gender = '女'" in sql
    assert connection.execute(sql).fetchall() == [(1, 100.0, 1), (3, 300.0, 2)]


def test_patient_rollup_applies_domain_and_date_filters_before_aggregation():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 50, "深圳"),
            (2, "乙", "男", 50, "北京"),
            (3, "丙", "女", 30, "深圳"),
        ],
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "2025-01-03", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (2, 1, "2025-01-20", "内科", "疾病甲", "药A", "检查甲", 150.0, 1),
            (3, 1, "2025-02-01", "内科", "疾病甲", "药A", "检查甲", 1000.0, 1),
            (4, 2, "2025-01-10", "外科", "疾病乙", "药B", "检查甲", 1000.0, 1),
            (5, 3, "2025-01-10", "内科", "疾病甲", "药A", "检查甲", 1000.0, 1),
        ],
    )

    sql = _compile_patient_rollup(
        "汇总每位患者，筛选年龄至少40岁、就诊次数至少2、总费用至少200、科室数至少1，"
        "限定2025-01-01至2025-01-31期间内科就诊，"
        "返回患者编号、年龄、就诊次数、总费用和科室数。",
        DOMAINS,
    )

    assert sql is not None
    assert "v.visit_date >= '2025-01-01'" in sql
    assert "v.visit_date <= '2025-01-31'" in sql
    assert "v.department = '内科'" in sql
    assert connection.execute(sql).fetchall() == [(1, 50, 2, 250.0, 1)]


def test_window_compilers_apply_patient_demographics_before_windows():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 50, "深圳"),
            (2, "乙", "男", 50, "深圳"),
        ],
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (2, 1, "2025-01-20", "外科", "疾病乙", "药B", "检查甲", 200.0, 1),
            (3, 2, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 300.0, 1),
            (4, 2, "2025-01-20", "外科", "疾病乙", "药B", "检查甲", 400.0, 1),
        ],
    )
    cases = (
        (
            _compile_running_cost,
            "对2025-01-01至2025-12-31期间的女性患者，按患者和日期计算累计费用，"
            "返回患者编号、就诊日期、费用和累计费用。",
        ),
        (
            _compile_lag_gap,
            "找出女性患者相邻两次就诊间隔超过10天的记录，"
            "返回患者编号、就诊日期、前一次日期和间隔天数。",
        ),
        (
            _compile_episode_segmentation,
            "将女性患者相隔超过10天的就诊划分为新的就诊事件，"
            "返回患者编号、就诊日期和就诊事件。",
        ),
    )

    for compiler, question in cases:
        sql = compiler(question, DOMAINS)
        assert sql is not None
        assert "JOIN patients AS p ON p.patient_id = v.patient_id" in sql
        assert "p.gender = '女'" in sql
        assert {row[0] for row in connection.execute(sql).fetchall()} == {1}


def test_temporal_cohort_intent_records_exclusive_ownership():
    family_retention = classify_temporal_cohort_intent(
        "按首次就诊月份统计年龄至少40岁的女性患者在偏移0至6个月的留存率。",
        DOMAINS,
    )
    evidence_domains = {**DOMAINS, "kg_relation_zh": ["呈现症状"]}
    family_evidence = classify_temporal_cohort_intent(
        "深圳地区年龄至少40岁的女性队列人群漏斗知识覆盖证据，"
        "自2025-01-01以来，疾病拥有呈现症状关系，"
        "关系置信度至少0.8，观察次数至少3。",
        evidence_domains,
    )
    date_only_funnel = classify_temporal_cohort_intent(
        "按首次就诊月统计患者漏斗，在7天和30天内再次就诊人数，返回结果。",
        DOMAINS,
    )
    date_only_retention = classify_temporal_cohort_intent(
        "按首次就诊月统计患者在第1个月和第3个月留存，返回结果。",
        DOMAINS,
    )
    filtered_funnel = classify_temporal_cohort_intent(
        "按首次就诊月统计内科患者漏斗，在7天和30天内再次就诊人数，返回结果。",
        DOMAINS,
    )
    bounded_retention = classify_temporal_cohort_intent(
        "按首诊月统计年龄至少40岁且不超过50岁的女性患者留存率，偏移月0至3。",
        DOMAINS,
    )
    projected_funnel = classify_temporal_cohort_intent(
        "按首次就诊月统计患者漏斗，在7天和30天内再次就诊人数，返回患者编号。",
        DOMAINS,
    )
    grouped_funnel = classify_temporal_cohort_intent(
        "按首次就诊月统计患者漏斗，在7天和30天内再次就诊人数，按地区统计，返回结果。",
        DOMAINS,
    )
    negated_evidence = classify_temporal_cohort_intent(
        "深圳地区以外年龄至少40岁的女性队列人群漏斗知识覆盖证据，"
        "自2025-01-01以来，疾病拥有呈现症状关系，"
        "关系置信度至少0.8，观察次数至少3。",
        evidence_domains,
    )
    visit_qualified_retention = classify_temporal_cohort_intent(
        "按首诊月统计患糖尿病且年龄至少40岁的女性患者留存率，偏移月0至3。",
        DOMAINS,
    )
    visit_qualified_evidence = classify_temporal_cohort_intent(
        "深圳地区患糖尿病且年龄至少40岁的女性队列人群漏斗知识覆盖证据，"
        "自2025-01-01以来，疾病拥有呈现症状关系，"
        "关系置信度至少0.8，观察次数至少3。",
        evidence_domains,
    )
    sparse_reengagement = classify_temporal_cohort_intent(
        "统计2024年首次就诊后30天内再次就诊人数，按月分组。",
        DOMAINS,
    )
    follow_up_reengagement = classify_temporal_cohort_intent(
        "统计首次就诊后30天内回诊人数，按月分组。",
        DOMAINS,
    )
    bare_cohort = classify_temporal_cohort_intent(
        "筛选深圳队列患者，返回患者编号。",
        DOMAINS,
    )
    bare_funnel = classify_temporal_cohort_intent("人群漏斗", DOMAINS)
    repeat_visit = classify_temporal_cohort_intent(
        "列出30天内再次就诊的患者，返回患者编号。",
        DOMAINS,
    )

    assert family_retention.disposition is TemporalCohortDisposition.FAMILY_RETENTION
    assert family_retention.kind is TemporalCohortKind.RETENTION
    assert family_retention.has_complete_family_retention
    assert family_evidence.disposition is TemporalCohortDisposition.FAMILY_EVIDENCE_FUNNEL
    assert family_evidence.kind is TemporalCohortKind.FUNNEL
    assert family_evidence.has_complete_family_evidence
    assert date_only_funnel.disposition is TemporalCohortDisposition.ADVANCED_DATE_ONLY
    assert date_only_funnel.kind is TemporalCohortKind.FUNNEL
    assert date_only_retention.disposition is TemporalCohortDisposition.ADVANCED_DATE_ONLY
    assert date_only_retention.kind is TemporalCohortKind.RETENTION
    assert filtered_funnel.disposition is TemporalCohortDisposition.TERMINAL_ABSTAIN
    assert (
        TemporalCohortReason.UNSUPPORTED_NON_DATE_QUALIFIER
        in filtered_funnel.reasons
    )
    assert bounded_retention.disposition is TemporalCohortDisposition.FAMILY_RETENTION
    assert bounded_retention.has_complete_family_retention
    assert projected_funnel.disposition is TemporalCohortDisposition.TERMINAL_ABSTAIN
    assert TemporalCohortReason.UNSUPPORTED_PROJECTION in projected_funnel.reasons
    assert grouped_funnel.disposition is TemporalCohortDisposition.TERMINAL_ABSTAIN
    assert TemporalCohortReason.UNSUPPORTED_GROUPING in grouped_funnel.reasons
    assert negated_evidence.disposition is TemporalCohortDisposition.TERMINAL_ABSTAIN
    assert TemporalCohortReason.UNSUPPORTED_NEGATED_QUALIFIER in negated_evidence.reasons
    assert visit_qualified_retention.disposition is TemporalCohortDisposition.TERMINAL_ABSTAIN
    assert not visit_qualified_retention.has_complete_family_retention
    assert TemporalCohortReason.UNSUPPORTED_NON_DATE_QUALIFIER in visit_qualified_retention.reasons
    assert visit_qualified_evidence.disposition is TemporalCohortDisposition.TERMINAL_ABSTAIN
    assert not visit_qualified_evidence.has_complete_family_evidence
    assert TemporalCohortReason.UNSUPPORTED_FAMILY_EVIDENCE in visit_qualified_evidence.reasons
    assert sparse_reengagement.disposition is TemporalCohortDisposition.TERMINAL_ABSTAIN
    assert sparse_reengagement.is_exclusive
    assert sparse_reengagement.kind is TemporalCohortKind.REENGAGEMENT
    assert sparse_reengagement.has_first_visit_anchor
    assert sparse_reengagement.has_reengagement
    assert sparse_reengagement.has_relative_reengagement
    assert TemporalCohortReason.UNSUPPORTED_REENGAGEMENT in sparse_reengagement.reasons
    assert follow_up_reengagement.disposition is TemporalCohortDisposition.TERMINAL_ABSTAIN
    assert follow_up_reengagement.kind is TemporalCohortKind.REENGAGEMENT
    assert follow_up_reengagement.has_relative_reengagement
    assert TemporalCohortReason.UNSUPPORTED_REENGAGEMENT in follow_up_reengagement.reasons
    assert bare_cohort.disposition is TemporalCohortDisposition.NOT_APPLICABLE
    assert bare_funnel.disposition is TemporalCohortDisposition.NOT_APPLICABLE
    assert repeat_visit.disposition is TemporalCohortDisposition.NOT_APPLICABLE


def test_temporal_cohort_routes_to_its_designated_lowerer():
    retention_sql = compile_advanced_medical_sql(
        "按首次就诊月份统计年龄至少40岁的女性患者在偏移0至6个月的留存率。",
        DOMAINS,
    )
    assert retention_sql is not None
    assert "WITH eligible AS" in retention_sql
    assert "gender = '女'" in retention_sql
    assert "age >= 40" in retention_sql
    assert "month_offset BETWEEN 0 AND 6" in retention_sql

    bounded_retention_sql = compile_advanced_medical_sql(
        "按首诊月统计年龄至少40岁且不超过50岁的女性患者留存率，偏移月0至3。",
        DOMAINS,
    )
    assert bounded_retention_sql is not None
    assert "age BETWEEN 40 AND 50" in bounded_retention_sql
    assert "month_offset BETWEEN 0 AND 3" in bounded_retention_sql

    evidence_domains = {**DOMAINS, "kg_relation_zh": ["呈现症状"]}
    evidence_sql = compile_advanced_medical_sql(
        "深圳地区年龄至少40岁的女性队列人群漏斗知识覆盖证据，"
        "自2025-01-01以来，疾病拥有呈现症状关系，"
        "关系置信度至少0.8，观察次数至少3。",
        evidence_domains,
    )
    assert evidence_sql is not None
    assert "region = '深圳'" in evidence_sql
    assert "age >= 40" in evidence_sql
    assert "gender = '女'" in evidence_sql
    assert "v.visit_date >= '2025-01-01'" in evidence_sql
    assert "kt.relation_zh = '呈现症状'" in evidence_sql
    assert "kt.confidence >= 0.8" in evidence_sql
    assert "kt.count >= 3" in evidence_sql

    funnel_sql = compile_advanced_medical_sql(
        "按首次就诊月统计患者漏斗，在7天和30天内再次就诊人数，返回结果。",
        DOMAINS,
    )
    retention_date_only_sql = compile_advanced_medical_sql(
        "按首次就诊月统计患者在第1个月和第3个月留存，返回结果。",
        DOMAINS,
    )
    assert funnel_sql is not None
    assert "cohort_funnel" in funnel_sql
    assert retention_date_only_sql is not None
    assert "cohort_retention" in retention_date_only_sql


def test_temporal_cohort_terminal_and_failed_family_routes_do_not_fall_through(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module

    routed_questions: list[str] = []

    def _failed_designated_family(
        question,
        _domains,
        *,
        intent=None,
        obligations=None,
    ):
        routed_questions.append(question)
        return None

    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda _question, _domains, **_kwargs: "SELECT 'broad-family' AS source",
    )
    monkeypatch.setattr(
        families_module,
        "compile_temporal_cohort_family_sql",
        _failed_designated_family,
    )
    family_retention_question = (
        "按首诊月统计年龄至少40岁的女性患者留存率，偏移月0至3。"
    )
    bounded_retention_question = (
        "按首诊月统计年龄至少40岁且不超过50岁的女性患者留存率，偏移月0至3。"
    )
    visit_qualified_evidence_question = (
        "深圳地区患糖尿病且年龄至少40岁的女性队列人群漏斗知识覆盖证据，"
        "自2025-01-01以来，疾病拥有呈现症状关系，"
        "关系置信度至少0.8，观察次数至少3。"
    )
    evidence_domains = {**DOMAINS, "kg_relation_zh": ["呈现症状"]}
    terminal_questions = (
        "统计2024年首次就诊后30天内再次就诊人数，按月分组。",
        "按首次就诊月统计内科患者漏斗，在7天和30天内再次就诊人数，返回结果。",
        "按首次就诊月统计患者漏斗，在7天和30天内再次就诊人数，返回患者编号。",
        "按首次就诊月统计患者漏斗，在7天和30天内再次就诊人数，按地区统计，返回结果。",
        "按首诊月统计患糖尿病且年龄至少40岁的女性患者留存率，偏移月0至3。",
        "按首次就诊月统计患糖尿病患者漏斗，在7天和30天内再次就诊人数，返回结果。",
    )

    assert compile_advanced_medical_sql(family_retention_question, DOMAINS) is None
    assert compile_advanced_medical_sql(bounded_retention_question, DOMAINS) is None
    for terminal_question in terminal_questions:
        assert compile_advanced_medical_sql(terminal_question, DOMAINS) is None
    assert (
        compile_advanced_medical_sql(
            visit_qualified_evidence_question,
            evidence_domains,
        )
        is None
    )
    assert routed_questions == [family_retention_question, bounded_retention_question]


def test_date_only_analytical_compilers_reject_non_date_filters():
    questions = (
        "按月统计2025-01-01至2025-12-31的内科总费用同比变化，"
        "返回月份、总费用、上年同期和同比变化。",
        "按首次就诊月统计内科患者漏斗，在7天和30天内再次就诊人数，返回结果。",
        "按首次就诊月统计女性患者7天和30天留存，返回结果。",
    )

    for question in questions:
        assert compile_advanced_medical_sql(question, DOMAINS) is None


def test_required_departments_preserve_patient_filters_and_reject_visit_constraints():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 50, "深圳"),
            (2, "乙", "男", 50, "深圳"),
            (3, "丙", "女", 35, "北京"),
            (4, "丁", "女", 50, "北京"),
        ],
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (2, 1, "2025-01-02", "外科", "疾病乙", "药B", "检查甲", 100.0, 1),
            (3, 2, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (4, 2, "2025-01-02", "外科", "疾病乙", "药B", "检查甲", 100.0, 1),
            (5, 3, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (6, 3, "2025-01-02", "外科", "疾病乙", "药B", "检查甲", 100.0, 1),
            (7, 4, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
        ],
    )

    sql = _compile_required_departments(
        "列出女性且年龄至少40岁的患者，他们都去过内科和外科，返回患者编号和姓名。",
        DOMAINS,
    )

    assert sql is not None
    assert "p.gender = '女'" in sql
    assert "p.age >= 40" in sql
    assert connection.execute(sql).fetchall() == [(1, "甲")]
    assert (
        compile_advanced_medical_sql(
            "列出都去过内科和外科的患者，限定2025-01-01至2025-12-31，"
            "返回患者编号。",
            DOMAINS,
        )
        is None
    )
    assert (
        compile_advanced_medical_sql(
            "列出都去过内科和外科且患疾病甲的患者，返回患者编号。",
            DOMAINS,
        )
        is None
    )


def test_rollup_preserves_all_aggregate_bounds():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 50, "深圳"),
            (2, "乙", "男", 50, "深圳"),
        ],
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (2, 1, "2025-01-02", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (3, 2, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (4, 2, "2025-01-02", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (5, 2, "2025-01-03", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
        ],
    )

    sql = _compile_patient_rollup(
        "汇总每位患者，筛选年龄至少40岁、就诊次数至少2且不超过2、"
        "总费用至少200且不超过200、科室数至少1且不超过1，"
        "返回患者编号、年龄、就诊次数、总费用和科室数。",
        DOMAINS,
    )

    assert sql is not None
    assert "visit_count >= 2 AND visit_count <= 2" in sql
    assert "total_cost >= 200 AND total_cost <= 200" in sql
    assert "department_count >= 1 AND department_count <= 1" in sql
    assert connection.execute(sql).fetchall() == [(1, 50, 2, 200.0, 1)]


def test_first_last_preserves_reverse_form_upper_visit_count_bound():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 50, "深圳"),
            (2, "乙", "男", 50, "深圳"),
        ],
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (2, 1, "2025-01-02", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (3, 2, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (4, 2, "2025-01-02", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (5, 2, "2025-01-03", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
        ],
    )

    sql = _compile_first_last(
        "找出至少有2次就诊且不超过2次的患者，返回患者编号、"
        "首次就诊日期和末次就诊日期。",
        DOMAINS,
    )

    assert sql is not None
    assert "pvc.visit_count >= 2 AND pvc.visit_count <= 2" in sql
    assert connection.execute(sql).fetchall() == [
        (1, "2025-01-01", "2025-01-02"),
    ]


def test_region_topk_preserves_all_eligibility_bounds():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 50, "深圳"),
            (2, "乙", "男", 50, "深圳"),
            (3, "丙", "女", 50, "北京"),
        ],
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (2, 1, "2025-01-02", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (3, 2, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 1000.0, 1),
            (4, 2, "2025-01-02", "内科", "疾病甲", "药A", "检查甲", 1000.0, 1),
            (5, 2, "2025-01-03", "内科", "疾病甲", "药A", "检查甲", 1000.0, 1),
            (6, 3, "2025-01-01", "外科", "疾病乙", "药B", "检查甲", 300.0, 1),
            (7, 3, "2025-01-02", "外科", "疾病乙", "药B", "检查甲", 300.0, 1),
        ],
    )

    sql = _compile_region_topk(
        "每个地区累计费用排名前1，年龄至少40岁、"
        "就诊次数至少2且不超过2，返回地区、患者编号、总费用和排名。",
        DOMAINS,
    )

    assert sql is not None
    assert "visit_count >= 2 AND visit_count <= 2" in sql
    assert connection.execute(sql).fetchall() == [
        ("北京", 3, 600.0, 1),
        ("深圳", 1, 200.0, 1),
    ]


def test_terminal_advanced_abstentions_do_not_reach_family_fallback(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module

    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda _question, _domains, **_kwargs: "SELECT 'family' AS source",
    )
    questions = (
        "按月统计2025-01-01至2025-12-31的内科总费用同比变化，"
        "返回月份、总费用、上年同期和同比变化。",
        "按首次就诊月统计内科患者漏斗，在7天和30天内再次就诊人数，返回结果。",
        "按首次就诊月统计女性患者7天和30天留存，返回结果。",
        "列出都去过内科和外科的患者，限定2025-01-01至2025-12-31，"
        "返回患者编号。",
        "列出没有都去过内科和外科的患者，返回患者编号。",
    )

    for question in questions:
        assert compile_advanced_medical_sql(question, DOMAINS) is None
    assert _compile_required_departments(
        "列出没有都去过内科和外科的患者，返回患者编号。",
        DOMAINS,
    ) is None


def test_patient_exists_rejects_date_and_numeric_visit_qualifiers_before_fallback(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module

    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda _question, _domains, **_kwargs: "SELECT 'family' AS source",
    )
    questions = (
        "列出既患疾病甲又使用药A的患者，限定2025-01-01至2025-12-31，返回患者编号。",
        "列出既患疾病甲又使用药A且就诊费用超过100元的患者，返回患者编号。",
    )

    for question in questions:
        assert _compile_patient_exists(question, DOMAINS) is None
        assert compile_advanced_medical_sql(question, DOMAINS) is None


def test_date_only_analytics_reject_numeric_visit_and_aggregate_qualifiers_before_fallback(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module

    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda _question, _domains, **_kwargs: "SELECT 'family' AS source",
    )
    questions = (
        "按月统计2025-01-01至2025-12-31总费用至少100的总费用同比变化，"
        "返回月份、总费用、上年同期和同比变化。",
        "按首次就诊月统计每次就诊费用超过100的患者漏斗，"
        "在7天和30天内再次就诊人数，返回结果。",
        "按首次就诊月统计每次就诊费用超过100的患者7天和30天留存，返回结果。",
    )

    for question in questions:
        assert compile_advanced_medical_sql(question, DOMAINS) is None


def test_region_topk_renders_unit_bearing_visit_count_ranges_before_ranking():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 50, "深圳"),
            (2, "乙", "男", 35, "深圳"),
            (3, "丙", "女", 50, "深圳"),
            (4, "丁", "女", 45, "北京"),
            (5, "戊", "男", 45, "北京"),
        ],
    )
    visits = [
        (1, 1, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
        (2, 1, "2025-01-02", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
        (3, 2, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 1000.0, 1),
        (4, 2, "2025-01-02", "内科", "疾病甲", "药A", "检查甲", 1000.0, 1),
        (5, 3, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 2000.0, 1),
        (6, 4, "2025-01-01", "外科", "疾病乙", "药B", "检查甲", 100.0, 1),
        (7, 4, "2025-01-02", "外科", "疾病乙", "药B", "检查甲", 100.0, 1),
        (8, 4, "2025-01-03", "外科", "疾病乙", "药B", "检查甲", 100.0, 1),
    ]
    visits.extend(
        (9 + index, 5, f"2025-02-{index + 1:02d}", "外科", "疾病乙", "药B", "检查甲", 1000.0, 1)
        for index in range(6)
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        visits,
    )

    for range_text in ("2至5次", "2..5次"):
        sql = _compile_region_topk(
            "每个地区累计费用排名前1，年龄不低于40岁、就诊次数为"
            + range_text
            + "，返回地区、患者编号、总费用和排名。",
            DOMAINS,
        )

        assert sql is not None
        assert "age >= 40" in sql
        assert "visit_count >= 2 AND visit_count <= 5" in sql
        assert connection.execute(sql).fetchall() == [
            ("北京", 4, 300.0, 1),
            ("深圳", 1, 200.0, 1),
        ]


def test_region_topk_rejects_unrenderable_visit_count_range_before_fallback(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module

    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda _question, _domains, **_kwargs: "SELECT 'family' AS source",
    )
    question = (
        "每个地区累计费用排名前1，年龄不低于40岁、就诊次数为2至次，"
        "返回地区、患者编号、总费用和排名。"
    )

    assert _compile_region_topk(question, DOMAINS) is None
    assert compile_advanced_medical_sql(question, DOMAINS) is None


def test_required_departments_keeps_positive_not_lower_than_age_bound():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 45, "深圳"),
            (2, "乙", "男", 35, "北京"),
        ],
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (2, 1, "2025-01-02", "外科", "疾病乙", "药B", "检查甲", 100.0, 1),
            (3, 2, "2025-01-01", "内科", "疾病甲", "药A", "检查甲", 100.0, 1),
            (4, 2, "2025-01-02", "外科", "疾病乙", "药B", "检查甲", 100.0, 1),
        ],
    )
    question = "列出年龄不低于40岁的患者，他们都去过内科和外科，返回患者编号。"

    sql = _compile_required_departments(question, DOMAINS)

    assert sql is not None
    assert "p.age >= 40" in sql
    assert connection.execute(sql).fetchall() == [(1,)]


def test_required_department_negative_and_numeric_scopes_are_terminal(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module

    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda _question, _domains, **_kwargs: "SELECT 'family' AS source",
    )
    negative = "列出从未分别去过内科和外科的患者，返回患者编号。"
    action_before_universal = "列出未覆盖所有内科和外科的患者，返回患者编号。"
    lexical_action_before_universal = "列出没有覆盖所有内科和外科的患者，返回患者编号。"
    numeric = "列出都去过内科和外科且就诊费用超过100元的患者，返回患者编号。"

    assert compile_advanced_medical_sql(negative, DOMAINS) is None
    assert _compile_required_departments(action_before_universal, DOMAINS) is None
    assert compile_advanced_medical_sql(action_before_universal, DOMAINS) is None
    assert _compile_required_departments(lexical_action_before_universal, DOMAINS) is None
    assert compile_advanced_medical_sql(lexical_action_before_universal, DOMAINS) is None
    assert _compile_required_departments(numeric, DOMAINS) is None
    assert compile_advanced_medical_sql(numeric, DOMAINS) is None


def test_range_forms_preserve_both_bounds_in_rollup_first_last_and_region_topk():
    for connector in ("至", "到"):
        rollup = _compile_patient_rollup(
            "汇总每位患者，筛选年龄至少40岁、"
            f"就诊次数为2{connector}5次、总费用为200{connector}600元、"
            f"科室数为1{connector}2个，"
            "返回患者编号、年龄、就诊次数、总费用和科室数。",
            DOMAINS,
        )
        first_last = _compile_first_last(
            f"找出就诊次数为2{connector}5次的患者，返回患者编号、"
            "首次就诊日期和末次就诊日期。",
            DOMAINS,
        )
        region_topk = _compile_region_topk(
            "每个地区累计费用排名前1，年龄至少40岁、"
            f"就诊次数为2{connector}5次，返回地区、患者编号、总费用和排名。",
            DOMAINS,
        )

        assert rollup is not None
        assert "visit_count >= 2 AND visit_count <= 5" in rollup
        assert "total_cost >= 200 AND total_cost <= 600" in rollup
        assert "department_count >= 1 AND department_count <= 2" in rollup
        assert first_last is not None
        assert "pvc.visit_count >= 2 AND pvc.visit_count <= 5" in first_last
        assert region_topk is not None
        assert "visit_count >= 2 AND visit_count <= 5" in region_topk


def test_reverse_not_exceeding_bound_does_not_also_emit_exceeding_predicate():
    sql = _compile_first_last(
        "找出不超过2次就诊且至少有1次就诊的患者，返回患者编号、"
        "首次就诊日期和末次就诊日期。",
        DOMAINS,
    )

    assert sql is not None
    assert "pvc.visit_count <= 2" in sql
    assert "pvc.visit_count >= 1" in sql
    assert "pvc.visit_count > 2" not in sql


def test_age_peer_average_uncovered_qualifier_is_terminal_before_family_fallback(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module

    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda _question, _domains, **_kwargs: "SELECT 'family' AS source",
    )
    question = (
        "找出年龄高于同地区患者平均年龄且曾在内科就诊的患者，"
        "返回患者编号和年龄。"
    )
    ungrounded_question = (
        "找出年龄高于同地区患者平均年龄且曾在肿瘤科就诊的患者，"
        "返回患者编号和年龄。"
    )
    ungrounded_disease_question = (
        "找出年龄高于同地区患者平均年龄的糖尿病患者，"
        "返回患者编号和年龄。"
    )
    negated_gender_question = (
        "找出年龄高于同地区患者平均年龄的非女性患者，"
        "返回患者编号和年龄。"
    )
    identity_question = (
        "找出姓名为甲且年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄。"
    )
    grouping_question = (
        "按地区分组找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄。"
    )
    disjunctive_peer_question = (
        "找出年龄高于同地区或同性别患者平均年龄的患者，"
        "返回患者编号和年龄。"
    )
    aggregate_projection_question = (
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和平均年龄。"
    )
    distinct_projection_question = (
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回不同的患者编号和年龄。"
    )

    assert compile_advanced_medical_sql(question, DOMAINS) is None
    assert compile_advanced_medical_sql(ungrounded_question, DOMAINS) is None
    assert compile_advanced_medical_sql(ungrounded_disease_question, DOMAINS) is None
    assert compile_advanced_medical_sql(negated_gender_question, DOMAINS) is None
    assert compile_advanced_medical_sql(identity_question, DOMAINS) is None
    assert compile_advanced_medical_sql(grouping_question, DOMAINS) is None
    assert compile_advanced_medical_sql(disjunctive_peer_question, DOMAINS) is None
    assert compile_advanced_medical_sql(aggregate_projection_question, DOMAINS) is None
    assert compile_advanced_medical_sql(distinct_projection_question, DOMAINS) is None


def test_age_average_intent_defers_legacy_region_scope_to_family(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module

    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda _question, _domains, **_kwargs: "SELECT 'family' AS source",
    )
    question = (
        "找出年龄高于本地区同性别患者平均年龄的患者，"
        "返回患者编号和年龄。"
    )

    intent = classify_age_average_intent(question, DOMAINS)

    assert intent.disposition is AgeAverageDisposition.NOT_APPLICABLE
    assert intent.has_legacy_region_peer_scope
    assert compile_advanced_medical_sql(question, DOMAINS) == "SELECT 'family' AS source"


def test_age_peer_average_unsupported_order_is_terminal_before_family_fallback(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module

    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda _question, _domains, **_kwargs: "SELECT 'family' AS source",
    )

    sql = compile_advanced_medical_sql(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄，按姓名升序排列。",
        DOMAINS,
    )
    chinese_topk_sql = compile_advanced_medical_sql(
        "找出年龄高于同地区患者平均年龄的患者，"
        "返回患者编号和年龄，取前十名。",
        DOMAINS,
    )

    assert sql is None
    assert chinese_topk_sql is None


def test_advanced_dispatch_blocks_terminal_mixed_region_polarity():
    question = (
        "找出深圳患者或北京地区以外患者，不要求发生在同一次就诊，"
        "但历史上既有疾病甲又使用药A。"
    )

    result = compile_advanced_medical_result(question, DOMAINS)

    assert result.disposition is CompileDisposition.BLOCKED
    assert result.sql is None


def test_patient_total_peer_average_does_not_claim_an_age_peer_with_total_projection():
    sql = _compile_patient_total_peer_average(
        "找出年龄高于同地区患者平均年龄的患者，返回患者编号和总费用。",
        DOMAINS,
    )

    assert sql is None


def test_patient_total_peer_average_filters_candidate_totals_after_peer_average():
    sql = _compile_patient_total_peer_average(
        "找出累计费用至少1000元且高于同地区平均累计费用的患者，"
        "返回患者编号和总费用。",
        DOMAINS,
    )

    assert sql is not None
    assert "pt.total_cost >= 1000" in sql
    assert sql.index("JOIN region_averages") < sql.index("pt.total_cost >= 1000")


def test_ntile_filters_eligible_patient_totals_before_assigning_quartiles():
    sql = _compile_ntile_quartiles(
        "按累计费用将女性患者分为四分位，且累计费用至少1000元，"
        "返回患者编号、总费用和四分位。",
        DOMAINS,
    )

    assert sql is not None
    assert "eligible_patient_totals AS (SELECT * FROM patient_totals WHERE total_cost >= 1000)" in sql
    assert "FROM eligible_patient_totals" in sql
    assert sql.index("total_cost >= 1000") < sql.index("NTILE(4)")


def test_running_cost_filters_derived_window_value_before_limit():
    sql = _compile_running_cost(
        "在2024-01-01至2024-12-31期间，按每位患者和日期升序计算累计费用，"
        "仅保留累计费用至少1000元的前2条记录，"
        "返回患者编号、就诊日期、费用和累计费用。",
        DOMAINS,
    )

    assert sql is not None
    assert "WITH running_visits AS" in sql
    assert "WHERE rv.running_cost >= 1000" in sql
    assert sql.index("AS running_cost") < sql.index("WHERE rv.running_cost >= 1000")
    assert sql.index("WHERE rv.running_cost >= 1000") < sql.index("LIMIT 2")


def test_derived_metric_compilers_reject_unowned_numeric_thresholds():
    cases = (
        (
            _compile_patient_total_peer_average,
            "找出累计费用至少1000元、平均费用至少100元且高于同地区平均累计费用的患者，"
            "返回患者编号和总费用。",
        ),
        (
            _compile_ntile_quartiles,
            "按累计费用将女性患者分为四分位，且累计费用至少1000元、"
            "就诊次数至少2次，返回患者编号、总费用和四分位。",
        ),
        (
            _compile_running_cost,
            "在2024-01-01至2024-12-31期间，按每位患者和日期升序计算累计费用，"
            "且单次费用至少100元，返回患者编号、就诊日期、费用和累计费用。",
        ),
        (
            _compile_running_cost,
            "在2024-01-01至2024-12-31期间，按每位患者和日期升序计算累计费用，"
            "且至少100元的单次费用，返回患者编号、就诊日期、费用和累计费用。",
        ),
    )

    for compiler, question in cases:
        assert compiler(question, DOMAINS) is None


def test_derived_metric_owners_block_unrenderable_bounds_before_family_fallback(monkeypatch):
    import dki.analysis.nl2sql_semantic_families as families_module

    monkeypatch.setattr(
        families_module,
        "compile_semantic_family_sql",
        lambda _question, _domains, **_kwargs: "SELECT 'family' AS source",
    )
    questions = (
        "找出累计费用至少元且高于同地区平均累计费用的患者，"
        "返回患者编号和总费用。",
        "按累计费用将女性患者分为四分位，且累计费用至少元，"
        "返回患者编号、总费用和四分位。",
        "在2024-01-01至2024-12-31期间，按每位患者和日期升序计算累计费用，"
        "仅返回累计费用至少元的记录，"
        "返回患者编号、就诊日期、费用和累计费用。",
        "找出累计费用至少1000元、平均费用至少100元且高于同地区平均累计费用的患者，"
        "返回患者编号和总费用。",
        "按累计费用将女性患者分为四分位，且累计费用至少1000元、"
        "就诊次数至少2次，返回患者编号、总费用和四分位。",
        "在2024-01-01至2024-12-31期间，按每位患者和日期升序计算累计费用，"
        "且单次费用至少100元，返回患者编号、就诊日期、费用和累计费用。",
        "在2024-01-01至2024-12-31期间，按每位患者和日期升序计算累计费用，"
        "且至少100元的单次费用，返回患者编号、就诊日期、费用和累计费用。",
    )

    for question in questions:
        result = compile_advanced_medical_result(question, DOMAINS)
        assert result.disposition is CompileDisposition.BLOCKED
        assert result.sql is None


def test_running_cost_rejects_unavailable_derived_projection():
    sql = _compile_running_cost(
        "在2024-01-01至2024-12-31期间，按每位患者和日期升序计算累计费用，"
        "返回患者编号、就诊日期、费用、累计费用和间隔天数。",
        DOMAINS,
    )

    assert sql is None


def test_unbounded_running_cost_retains_the_requested_projection():
    sql = _compile_running_cost(
        "对2024-01-01至2024-12-31期间的女性患者，按患者和日期计算累计费用，"
        "返回患者编号、就诊日期、费用和累计费用。",
        DOMAINS,
    )

    assert sql is not None
    assert "AS running_cost, v.visit_id FROM" not in sql
