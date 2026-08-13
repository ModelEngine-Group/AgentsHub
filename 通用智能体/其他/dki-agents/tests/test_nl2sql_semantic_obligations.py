from __future__ import annotations

import pytest

from dki.analysis.nl2sql_semantic_obligations import (
    DateRole,
    ObligationDisposition,
    PredicateOperator,
    PredicatePhase,
    SourceRole,
    extract_semantic_obligations,
    render_source_predicates,
    validate_candidate_obligations,
)


DOMAINS = {
    "patient_id": ["P001", "2", "30", "40"],
    "patient_name": ["张三", "李四"],
    "visit_id": ["V001", "2", "30", "40"],
    "region": ["北京", "上海", "广州"],
    "department": ["心内科", "神经内科"],
    "disease": ["高血压", "糖尿病"],
    "drug": ["阿司匹林", "二甲双胍"],
    "exam": ["心电图", "血常规"],
}


def _concept(obligations, name):
    return [item for item in obligations.predicates if item.concept == name]


def test_numeric_ids_require_explicit_patient_or_visit_role_context():
    top_k = extract_semantic_obligations(
        "每个科室取费用最高的2条，年龄至少30岁且住院间隔超过40天",
        DOMAINS,
    )
    assert not _concept(top_k, "patient_id")
    assert not _concept(top_k, "visit_id")

    identities = extract_semantic_obligations(
        "患者ID为2且就诊编号为30的记录",
        DOMAINS,
    )
    assert _concept(identities, "patient_id")[0].values == ("2",)
    assert _concept(identities, "visit_id")[0].values == ("30",)


def test_patient_name_requires_role_context():
    unowned = extract_semantic_obligations("张三之后按姓名排序", DOMAINS)
    assert not _concept(unowned, "patient_name")

    owned = extract_semantic_obligations("患者姓名为张三的就诊", DOMAINS)
    assert _concept(owned, "patient_name")[0].values == ("张三",)


def test_graph_only_entity_literal_is_not_recast_as_visit_filter():
    graph = extract_semantic_obligations("高血压有哪些症状", DOMAINS)
    assert graph.disposition is ObligationDisposition.NOT_APPLICABLE
    assert not _concept(graph, "disease")

    diagnostic_graph = extract_semantic_obligations(
        "高血压需要做什么检查",
        DOMAINS,
    )
    assert diagnostic_graph.disposition is ObligationDisposition.NOT_APPLICABLE
    assert not _concept(diagnostic_graph, "disease")

    clinical = extract_semantic_obligations("高血压患者的就诊记录", DOMAINS)
    assert _concept(clinical, "disease")[0].values == ("高血压",)


def test_ratio_qualifier_must_remain_in_conditional_aggregate_numerator():
    obligations = extract_semantic_obligations(
        "高血压就诊占全部就诊的百分比",
        DOMAINS,
    )
    disease = _concept(obligations, "disease")[0]
    assert disease.phase is PredicatePhase.CONDITIONAL_AGGREGATE

    conditional = (
        "SELECT 100.0 * SUM(CASE WHEN disease='高血压' THEN 1 ELSE 0 END) "
        "/ COUNT(*) AS pct FROM visits"
    )
    assert validate_candidate_obligations(obligations, conditional).accepted
    assert not validate_candidate_obligations(
        obligations,
        "SELECT COUNT(*) FROM visits WHERE disease='高血压'",
    ).accepted
    assert not validate_candidate_obligations(
        obligations,
        "SELECT COUNT(*) FROM visits",
    ).accepted

    outpatient = extract_semantic_obligations(
        "门诊（住院天数为0）就诊占比是多少",
        DOMAINS,
    )
    assert _concept(outpatient, "los_days")[0].phase is (
        PredicatePhase.CONDITIONAL_AGGREGATE
    )
    assert validate_candidate_obligations(
        outpatient,
        "SELECT SUM(CASE WHEN los_days=0 THEN 1 ELSE 0 END) * 1.0 "
        "/ COUNT(*) FROM visits",
    ).accepted

    male = extract_semantic_obligations(
        "男性患者占全部患者的百分比",
        DOMAINS,
    )
    assert _concept(male, "gender")[0].phase is (
        PredicatePhase.CONDITIONAL_AGGREGATE
    )
    assert validate_candidate_obligations(
        male,
        "SELECT SUM(CASE WHEN gender='男' THEN 1 ELSE 0 END) * 1.0 "
        "/ COUNT(*) FROM patients",
    ).accepted


def test_positive_and_negative_region_obligations_keep_polarity():
    positive = extract_semantic_obligations("北京或上海地区的患者", DOMAINS)
    item = _concept(positive, "region")[0]
    assert item.operator is PredicateOperator.IN
    assert item.values == ("北京", "上海")

    negative = extract_semantic_obligations("排除北京和上海地区的患者", DOMAINS)
    item = _concept(negative, "region")[0]
    assert item.operator is PredicateOperator.NOT_IN
    assert item.values == ("北京", "上海")

    mixed = extract_semantic_obligations("北京或者排除上海地区的患者", DOMAINS)
    assert mixed.disposition is ObligationDisposition.TERMINAL_ABSTAIN


def test_required_all_values_need_intersect_not_in_or_union():
    obligations = extract_semantic_obligations(
        "既在心内科就诊过、又在神经内科就诊过的患者",
        DOMAINS,
    )
    department = _concept(obligations, "department")[0]
    assert department.operator is PredicateOperator.ALL_OF

    intersect = (
        "SELECT patient_id FROM ("
        "SELECT patient_id FROM visits WHERE department='心内科' "
        "INTERSECT "
        "SELECT patient_id FROM visits WHERE department='神经内科'"
        ") AS matched"
    )
    assert validate_candidate_obligations(obligations, intersect).accepted
    correlated_exists = (
        "SELECT p.patient_id FROM patients p WHERE EXISTS ("
        "SELECT v1.patient_id AS patient_id FROM visits v1 "
        "WHERE v1.patient_id=p.patient_id AND v1.department='心内科'"
        ") AND EXISTS ("
        "SELECT v2.patient_id AS patient_id FROM visits v2 "
        "WHERE v2.patient_id=p.patient_id AND v2.department='神经内科'"
        ")"
    )
    assert validate_candidate_obligations(
        obligations,
        correlated_exists,
    ).accepted

    for unsafe in (
        "SELECT patient_id FROM visits "
        "WHERE department IN ('心内科', '神经内科')",
        "SELECT patient_id FROM ("
        "SELECT patient_id FROM visits WHERE department='心内科' UNION "
        "SELECT patient_id FROM visits WHERE department='神经内科'"
        ") AS matched",
        "SELECT p.patient_id FROM patients p WHERE EXISTS ("
        "SELECT v1.patient_id AS patient_id FROM visits v1 "
        "WHERE v1.patient_id=p.patient_id AND v1.department='心内科'"
        ") OR EXISTS ("
        "SELECT v2.patient_id AS patient_id FROM visits v2 "
        "WHERE v2.patient_id=p.patient_id AND v2.department='神经内科'"
        ")",
    ):
        assert not validate_candidate_obligations(obligations, unsafe).accepted


def test_any_of_values_may_be_lowered_as_distinct_union_only():
    obligations = extract_semantic_obligations(
        "曾在心内科或神经内科就诊过的患者",
        DOMAINS,
    )
    department = _concept(obligations, "department")[0]
    assert department.operator is PredicateOperator.IN

    union = (
        "SELECT patient_id FROM visits WHERE department='心内科' "
        "UNION "
        "SELECT patient_id FROM visits WHERE department='神经内科'"
    )
    assert validate_candidate_obligations(obligations, union).accepted

    for unsafe in (
        union.replace(" UNION ", " UNION ALL "),
        "SELECT patient_id FROM visits WHERE department='心内科' "
        "UNION SELECT patient_id FROM visits",
        "SELECT patient_id FROM visits WHERE department='心内科' "
        "INTERSECT "
        "SELECT patient_id FROM visits WHERE department='神经内科'",
    ):
        assert not validate_candidate_obligations(obligations, unsafe).accepted


def test_difference_and_anti_join_preserve_anti_existence_polarity():
    domains = {
        **DOMAINS,
        "department": ["心血管内科", "内分泌科"],
    }
    difference = extract_semantic_obligations(
        "列出在心血管内科就诊过但没有在内分泌科就诊过的患者编号。",
        domains,
    )
    departments = _concept(difference, "department")
    assert [item.operator for item in departments] == [
        PredicateOperator.EQ,
        PredicateOperator.ABSENT,
    ]
    except_sql = (
        "SELECT patient_id FROM visits WHERE department='心血管内科' "
        "EXCEPT "
        "SELECT patient_id FROM visits WHERE department='内分泌科'"
    )
    assert validate_candidate_obligations(difference, except_sql).accepted
    assert not validate_candidate_obligations(
        difference,
        "SELECT patient_id FROM visits "
        "WHERE department IN ('心血管内科', '内分泌科')",
    ).accepted

    trailing_negative = extract_semantic_obligations(
        "在心内科出现但在神经内科没出现的疾病",
        DOMAINS,
    )
    departments = _concept(trailing_negative, "department")
    assert [item.operator for item in departments] == [
        PredicateOperator.EQ,
        PredicateOperator.ABSENT,
    ]
    assert validate_candidate_obligations(
        trailing_negative,
        "SELECT disease FROM visits WHERE department='心内科' "
        "EXCEPT "
        "SELECT disease FROM visits WHERE department='神经内科'",
    ).accepted

    anti_drug = extract_semantic_obligations(
        "哪些患者从未被开过阿司匹林？列出姓名。",
        DOMAINS,
    )
    drug = _concept(anti_drug, "drug")[0]
    assert drug.operator is PredicateOperator.ABSENT
    correct = (
        "SELECT p.name FROM patients p WHERE NOT EXISTS ("
        "SELECT v.patient_id AS patient_id FROM visits v "
        "WHERE v.patient_id=p.patient_id "
        "AND v.drug_prescribed='阿司匹林')"
    )
    assert validate_candidate_obligations(anti_drug, correct).accepted
    assert not validate_candidate_obligations(
        anti_drug,
        "SELECT DISTINCT p.name FROM patients p JOIN visits v "
        "ON v.patient_id=p.patient_id "
        "WHERE v.drug_prescribed='阿司匹林'",
    ).accepted


def test_shared_anti_clause_qualifiers_must_share_one_not_exists_scope():
    obligations = extract_semantic_obligations(
        "列出从未在神经内科使用阿司匹林的患者编号。",
        DOMAINS,
    )
    department = _concept(obligations, "department")[0]
    drug = _concept(obligations, "drug")[0]
    assert department.operator is PredicateOperator.ABSENT
    assert drug.operator is PredicateOperator.ABSENT
    assert department.scope_id == drug.scope_id

    shared = (
        "SELECT p.patient_id FROM patients p WHERE NOT EXISTS ("
        "SELECT v.patient_id AS patient_id FROM visits v "
        "WHERE v.patient_id=p.patient_id "
        "AND v.department='神经内科' "
        "AND v.drug_prescribed='阿司匹林')"
    )
    assert validate_candidate_obligations(obligations, shared).accepted
    split = (
        "SELECT p.patient_id FROM patients p WHERE NOT EXISTS ("
        "SELECT v1.patient_id AS patient_id FROM visits v1 "
        "WHERE v1.patient_id=p.patient_id AND v1.department='神经内科'"
        ") AND NOT EXISTS ("
        "SELECT v2.patient_id AS patient_id FROM visits v2 "
        "WHERE v2.patient_id=p.patient_id "
        "AND v2.drug_prescribed='阿司匹林')"
    )
    assert not validate_candidate_obligations(obligations, split).accepted


def test_required_qualifiers_cannot_hide_under_or_or_outer_join():
    obligations = extract_semantic_obligations(
        "列出在神经内科就诊且患糖尿病的患者编号",
        DOMAINS,
    )
    unsafe_or = (
        "SELECT patient_id FROM visits "
        "WHERE department='神经内科' OR disease='糖尿病'"
    )
    assert not validate_candidate_obligations(obligations, unsafe_or).accepted

    patient_visit = extract_semantic_obligations(
        "神经内科就诊的女性患者",
        DOMAINS,
    )
    unsafe_left_join = (
        "SELECT p.patient_id FROM patients p LEFT JOIN visits v "
        "ON v.patient_id=p.patient_id AND v.department='神经内科' "
        "WHERE p.gender='女'"
    )
    assert not validate_candidate_obligations(
        patient_visit,
        unsafe_left_join,
    ).accepted
    grouped_count = (
        "SELECT p.patient_id, COUNT(v.visit_id) AS visit_count "
        "FROM patients p LEFT JOIN visits v "
        "ON v.patient_id=p.patient_id AND v.department='神经内科' "
        "WHERE p.gender='女' GROUP BY p.patient_id"
    )
    assert validate_candidate_obligations(
        patient_visit,
        grouped_count,
    ).accepted


def test_date_roles_are_structured_and_three_dates_are_terminal():
    exact = extract_semantic_obligations("就诊日期为2026-01-03", DOMAINS)
    assert exact.dates[0].role is DateRole.EXACT
    assert exact.dates[0].operator is PredicateOperator.EQ

    since = extract_semantic_obligations("自2026-01-03起的就诊", DOMAINS)
    assert since.dates[0].role is DateRole.SINCE
    assert since.dates[0].operator is PredicateOperator.GE

    inclusive_since = extract_semantic_obligations(
        "2026-01-03及以后的就诊",
        DOMAINS,
    )
    assert inclusive_since.dates[0].role is DateRole.SINCE
    assert inclusive_since.dates[0].operator is PredicateOperator.GE

    until = extract_semantic_obligations("截至2026-01-03的就诊", DOMAINS)
    assert until.dates[0].role is DateRole.UNTIL
    assert until.dates[0].operator is PredicateOperator.LE

    date_range = extract_semantic_obligations(
        "2026-01-01到2026-01-31的就诊",
        DOMAINS,
    )
    assert [item.role for item in date_range.dates] == [
        DateRole.RANGE_START,
        DateRole.RANGE_END,
    ]

    unrelated = extract_semantic_obligations(
        "比较2026-01-01、2026-02-01、2026-03-01的就诊",
        DOMAINS,
    )
    assert unrelated.disposition is ObligationDisposition.TERMINAL_ABSTAIN
    assert "unassigned date literals" in unrelated.unsupported_reasons


def test_row_cost_comparison_does_not_invert_negative_operator():
    obligations = extract_semantic_obligations(
        "心内科女性患者单次费用不超过100元的就诊次数",
        DOMAINS,
    )
    cost = _concept(obligations, "cost")[0]
    assert cost.operator is PredicateOperator.LE
    assert cost.values == (100,)


@pytest.mark.parametrize(
    "question",
    (
        "费用在100到200元之间（含边界）的就诊",
        "费用从100元到200元（含首尾）的就诊",
    ),
)
def test_inclusive_cost_range_is_one_fail_closed_between_obligation(question):
    obligations = extract_semantic_obligations(question, DOMAINS)
    costs = _concept(obligations, "cost")

    assert len(costs) == 1
    assert costs[0].source is SourceRole.VISIT
    assert costs[0].column == "cost"
    assert costs[0].operator is PredicateOperator.BETWEEN
    assert costs[0].values == (100, 200)
    assert costs[0].phase is PredicatePhase.BASE_ROWS

    correct = "SELECT visit_id FROM visits WHERE cost BETWEEN 100 AND 200"
    assert validate_candidate_obligations(obligations, correct).accepted
    for unsafe in (
        "SELECT visit_id FROM visits WHERE cost BETWEEN 100 AND 201",
        "SELECT visit_id FROM visits WHERE cost > 100 AND cost < 200",
        "SELECT visit_id FROM visits WHERE cost BETWEEN 200 AND 100",
        "SELECT visit_id FROM visits WHERE ROUND(cost) BETWEEN 100 AND 200",
        "SELECT visit_id FROM visits WHERE cost BETWEEN 100 AND 200 OR 1=1",
    ):
        assert not validate_candidate_obligations(obligations, unsafe).accepted


def test_plain_visit_cost_threshold_is_base_row_but_aggregate_cost_is_not():
    row = extract_semantic_obligations(
        "每位患者最近一次费用至少1000元的就诊",
        DOMAINS,
    )
    cost = _concept(row, "cost")[0]
    assert cost.operator is PredicateOperator.GE
    assert cost.values == (1000,)

    aggregate = extract_semantic_obligations(
        "按科室统计平均费用至少100元的科室",
        DOMAINS,
    )
    assert not _concept(aggregate, "cost")


@pytest.mark.parametrize(
    "question",
    (
        "最长住院天数至少5天",
        "最大住院天数至少5天",
        "平均住院天数至少5天",
        "总住院天数至少5天",
        "累计住院天数至少5天",
        "合计住院天数至少5天",
        "住院天数总和至少5天",
        "住院天数求和至少5天",
    ),
)
def test_derived_los_metrics_are_not_base_row_obligations(question):
    obligations = extract_semantic_obligations(question, DOMAINS)

    assert not _concept(obligations, "los_days")


def test_plain_los_threshold_remains_a_base_row_obligation():
    obligations = extract_semantic_obligations(
        "住院天数至少5天的就诊",
        DOMAINS,
    )
    los_days = _concept(obligations, "los_days")

    assert len(los_days) == 1
    assert los_days[0].operator is PredicateOperator.GE
    assert los_days[0].values == (5,)
    assert los_days[0].phase is PredicatePhase.BASE_ROWS
    assert validate_candidate_obligations(
        obligations,
        "SELECT visit_id FROM visits WHERE los_days>=5",
    ).accepted


@pytest.mark.parametrize("rank_label", ("编号", "序号", "排名", "名次", "行号"))
def test_cost_does_not_capture_rank_metadata_value(rank_label):
    obligations = extract_semantic_obligations(
        f"费用{rank_label}为2的记录",
        DOMAINS,
    )

    assert not _concept(obligations, "cost")


def test_plain_cost_equality_still_binds_the_amount():
    obligations = extract_semantic_obligations("费用为100元的就诊", DOMAINS)
    costs = _concept(obligations, "cost")

    assert len(costs) == 1
    assert costs[0].operator is PredicateOperator.EQ
    assert costs[0].values == (100,)
    assert costs[0].phase is PredicatePhase.BASE_ROWS
    assert validate_candidate_obligations(
        obligations,
        "SELECT visit_id FROM visits WHERE cost=100",
    ).accepted


def test_adjacent_metric_thresholds_bind_to_their_own_labels():
    obligations = extract_semantic_obligations(
        "住院天数至少0天、费用至少3000元的就诊",
        DOMAINS,
    )

    assert _concept(obligations, "los_days")[0].values == (0,)
    assert _concept(obligations, "cost")[0].values == (3000,)


def test_candidate_must_keep_patient_and_visit_qualifiers_and_join():
    obligations = extract_semantic_obligations(
        "就诊ID为V001、科室心内科的女性患者在哪天就诊",
        DOMAINS,
    )
    incomplete = (
        "SELECT v.visit_date FROM visits v "
        "WHERE v.visit_id='V001'"
    )
    result = validate_candidate_obligations(obligations, incomplete)
    assert not result.accepted
    assert "missing gender qualifier" in result.reasons
    assert "missing department qualifier" in result.reasons
    assert "missing patient visit join" in result.reasons

    complete = (
        "SELECT v.visit_date FROM patients p "
        "JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE v.visit_id='V001' AND v.department='心内科' "
        "AND p.gender='女'"
    )
    assert validate_candidate_obligations(obligations, complete).accepted


def test_patient_literal_on_wrong_table_or_column_is_rejected():
    obligations = extract_semantic_obligations(
        "患者ID为P001且科室心内科的就诊",
        DOMAINS,
    )
    wrong = (
        "SELECT v.visit_id FROM visits v "
        "WHERE v.patient_id='P001' AND v.department='心内科'"
    )
    result = validate_candidate_obligations(obligations, wrong)
    assert not result.accepted
    assert "missing patient_id qualifier" in result.reasons
    assert "missing patient visit join" in result.reasons


def test_literal_in_projection_does_not_consume_a_qualifier():
    obligations = extract_semantic_obligations("女性患者", DOMAINS)
    sql = "SELECT '女' AS requested_gender, p.patient_id FROM patients p"
    result = validate_candidate_obligations(obligations, sql)
    assert not result.accepted
    assert result.reasons == ("missing gender qualifier",)


def test_range_may_be_rendered_as_between():
    obligations = extract_semantic_obligations(
        "2026-01-01到2026-01-31的心内科就诊",
        DOMAINS,
    )
    sql = (
        "SELECT v.visit_id FROM visits v "
        "WHERE v.department='心内科' "
        "AND v.visit_date BETWEEN '2026-01-01' AND '2026-01-31'"
    )
    assert validate_candidate_obligations(obligations, sql).accepted


def test_date_filter_after_row_number_does_not_count_as_source_filter():
    obligations = extract_semantic_obligations(
        "2026-01-01到2026-01-31每个科室取费用最高的3条",
        DOMAINS,
    )
    outer_only = (
        "WITH ranked AS ("
        "SELECT v.*, ROW_NUMBER() OVER ("
        "PARTITION BY v.department ORDER BY v.cost DESC"
        ") AS rn FROM visits v"
        ") "
        "SELECT * FROM ranked "
        "WHERE visit_date BETWEEN '2026-01-01' AND '2026-01-31' AND rn<=3"
    )
    result = validate_candidate_obligations(obligations, outer_only)
    assert not result.accepted
    assert "unconsumed date literal" in result.reasons

    source_filtered = (
        "WITH ranked AS ("
        "SELECT v.*, ROW_NUMBER() OVER ("
        "PARTITION BY v.department ORDER BY v.cost DESC"
        ") AS rn FROM visits v "
        "WHERE v.visit_date BETWEEN '2026-01-01' AND '2026-01-31'"
        ") "
        "SELECT * FROM ranked WHERE rn<=3"
    )
    assert validate_candidate_obligations(obligations, source_filtered).accepted


def test_having_predicate_does_not_consume_base_row_qualifier():
    obligations = extract_semantic_obligations(
        "心内科单次费用至少100元的就诊次数",
        DOMAINS,
    )
    late = (
        "SELECT v.department, COUNT(*) FROM visits v "
        "WHERE v.department='心内科' GROUP BY v.department "
        "HAVING v.cost>=100"
    )
    result = validate_candidate_obligations(obligations, late)
    assert not result.accepted
    assert "missing cost qualifier" in result.reasons

    early = (
        "SELECT v.department, COUNT(*) FROM visits v "
        "WHERE v.department='心内科' AND v.cost>=100 "
        "GROUP BY v.department"
    )
    assert validate_candidate_obligations(obligations, early).accepted


def test_filter_after_lag_does_not_count_as_episode_source_filter():
    obligations = extract_semantic_obligations(
        "2026-01-01到2026-01-31的心内科就诊按间隔划分episode",
        DOMAINS,
    )
    late = (
        "WITH ordered AS ("
        "SELECT v.*, LAG(v.visit_date) OVER ("
        "PARTITION BY v.patient_id ORDER BY v.visit_date"
        ") AS previous_date FROM visits v"
        ") "
        "SELECT * FROM ordered WHERE department='心内科' "
        "AND visit_date BETWEEN '2026-01-01' AND '2026-01-31'"
    )
    result = validate_candidate_obligations(obligations, late)
    assert not result.accepted
    assert "missing department qualifier" in result.reasons
    assert "unconsumed date literal" in result.reasons


def test_render_source_predicates_keeps_source_roles_separate():
    obligations = extract_semantic_obligations(
        "女性且地区不是北京的心内科就诊，单次费用至少100元",
        DOMAINS,
    )
    patient = render_source_predicates(
        obligations,
        SourceRole.PATIENT,
        alias="p",
    )
    visit = render_source_predicates(
        obligations,
        SourceRole.VISIT,
        alias="v",
    )
    assert "p.gender='女'" in patient
    assert "p.region!='北京'" in patient
    assert "v.department='心内科'" in visit
    assert "v.cost>=100" in visit


def test_binary_obligation_requires_a_direct_positive_column_comparison():
    obligations = extract_semantic_obligations(
        "年龄大于18岁的患者",
        DOMAINS,
    )

    assert validate_candidate_obligations(
        obligations,
        "SELECT patient_id FROM patients WHERE age > 18",
    ).accepted
    for unsafe in (
        "SELECT patient_id FROM patients WHERE (age > 18) = 0",
        "SELECT patient_id FROM patients WHERE age + 100 > 18",
        "SELECT patient_id FROM patients WHERE CAST(age AS INTEGER) > 18",
        "SELECT patient_id FROM patients WHERE NOT (age > 18)",
        "SELECT patient_id FROM patients WHERE age > 18 OR 1 = 1",
    ):
        result = validate_candidate_obligations(obligations, unsafe)
        assert not result.accepted, unsafe
        assert "missing age qualifier" in result.reasons


def test_patient_visit_join_must_be_a_positive_conjunctive_atom():
    obligations = extract_semantic_obligations(
        "神经内科就诊的女性患者",
        DOMAINS,
    )
    valid = (
        "SELECT p.patient_id FROM patients AS p JOIN visits AS v "
        "ON p.patient_id = v.patient_id "
        "WHERE p.gender = '女' AND v.department = '神经内科'"
    )
    bypassed = (
        "SELECT p.patient_id FROM patients AS p CROSS JOIN visits AS v "
        "WHERE (p.patient_id = v.patient_id OR 1 = 1) "
        "AND p.gender = '女' AND v.department = '神经内科'"
    )

    assert obligations.requires_patient_visit_join
    assert validate_candidate_obligations(obligations, valid).accepted
    result = validate_candidate_obligations(obligations, bypassed)
    assert not result.accepted
    assert "missing patient visit join" in result.reasons


def test_unconsumed_literal_filters_cannot_silently_narrow_the_answer():
    obligations = extract_semantic_obligations(
        "年龄大于18岁的患者",
        DOMAINS,
    )
    for unsafe in (
        "SELECT patient_id FROM patients WHERE age > 18 AND 1 = 0",
        "SELECT patient_id FROM patients WHERE age > 18 AND region = '北京'",
        "SELECT patient_id FROM patients WHERE age > 18 AND age BETWEEN 30 AND 60",
        "SELECT patient_id FROM patients WHERE age > 18 AND region IN ('北京', '上海')",
    ):
        result = validate_candidate_obligations(obligations, unsafe)
        assert not result.accepted, unsafe
        assert any("unconsumed" in reason for reason in result.reasons)


def test_no_obligation_candidate_still_receives_filter_closure_audit():
    obligations = extract_semantic_obligations(
        "返回患者编号",
        DOMAINS,
    )
    assert obligations.disposition is ObligationDisposition.NOT_APPLICABLE

    assert validate_candidate_obligations(
        obligations,
        "SELECT patient_id FROM patients",
    ).accepted
    result = validate_candidate_obligations(
        obligations,
        "SELECT patient_id FROM patients WHERE age = 18",
    )
    assert not result.accepted
    assert "unconsumed patients.age literal predicate" in result.reasons


def test_public_english_age_gender_and_los_phrases_create_obligations():
    younger = extract_semantic_obligations(
        "how many patients aged below 76 years had 高血压?",
        DOMAINS,
    )
    age = _concept(younger, "age")[0]
    assert age.operator is PredicateOperator.LT
    assert age.values == (76,)
    assert validate_candidate_obligations(
        younger,
        "SELECT COUNT(DISTINCT p.patient_id) FROM patients p "
        "JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE p.age<76 AND v.disease='高血压'",
    ).accepted

    above = extract_semantic_obligations(
        "what is the average age of 女 patients who are 40 years or above?",
        DOMAINS,
    )
    assert _concept(above, "gender")[0].values == ("女",)
    assert _concept(above, "age")[0].operator is PredicateOperator.GE
    assert validate_candidate_obligations(
        above,
        "SELECT AVG(p.age) FROM patients p "
        "WHERE p.gender='女' AND p.age>=40",
    ).accepted

    for question, operator, value, sql in (
        (
            "patients were admitted for more than 7 days in hospital",
            PredicateOperator.GT,
            7,
            "SELECT patient_id FROM visits WHERE los_days>7",
        ),
        (
            "patients were hospitalized for 0 days",
            PredicateOperator.EQ,
            0,
            "SELECT patient_id FROM visits WHERE los_days=0",
        ),
    ):
        obligations = extract_semantic_obligations(question, DOMAINS)
        los_days = _concept(obligations, "los_days")[0]
        assert los_days.operator is operator
        assert los_days.values == (value,)
        assert validate_candidate_obligations(obligations, sql).accepted


def test_public_identity_phrases_bind_to_the_intended_source():
    english_subject = extract_semantic_obligations(
        "what is the age of subject id P001?",
        DOMAINS,
    )
    subject_id = _concept(english_subject, "patient_id")[0]
    assert subject_id.source is SourceRole.PATIENT
    assert validate_candidate_obligations(
        english_subject,
        "SELECT age FROM patients WHERE patient_id='P001'",
    ).accepted

    visit_owned_patient = extract_semantic_obligations(
        "Count the number of times patient P001 received 心电图.",
        DOMAINS,
    )
    patient_id = _concept(visit_owned_patient, "patient_id")[0]
    assert patient_id.source is SourceRole.VISIT
    assert validate_candidate_obligations(
        visit_owned_patient,
        "SELECT COUNT(*) FROM visits "
        "WHERE patient_id='P001' AND exam='心电图'",
    ).accepted

    for question in (
        "what is the diagnosis for patient name 张三?",
        "参保人张三在本月结算了多少次？",
        "这位张三的患者有多少次就诊？",
        "在这段时间里病号张三的医疗费有多少钱？",
    ):
        obligations = extract_semantic_obligations(question, DOMAINS)
        name = _concept(obligations, "patient_name")[0]
        assert name.source is SourceRole.PATIENT
        assert name.values == ("张三",)

    chinese_patient_id = extract_semantic_obligations(
        "参保人的编码为P001的结算次数",
        DOMAINS,
    )
    assert _concept(chinese_patient_id, "patient_id")[0].source is SourceRole.PATIENT

    for question in (
        "查询医疗就诊V001所需要的住院时长",
        "编号V001的医疗就诊需要住院多久",
        "在V001这次医疗就诊中需要住院多久",
    ):
        obligations = extract_semantic_obligations(question, DOMAINS)
        visit_id = _concept(obligations, "visit_id")[0]
        assert visit_id.source is SourceRole.VISIT
        assert validate_candidate_obligations(
            obligations,
            "SELECT los_days FROM visits WHERE visit_id='V001'",
        ).accepted


def test_css_amount_within_phrase_is_a_closed_cost_qualifier():
    obligations = extract_semantic_obligations(
        "参保人张三1000元以内小金额结算次数有多少？",
        DOMAINS,
    )
    cost = _concept(obligations, "cost")[0]
    assert cost.operator is PredicateOperator.LE
    assert cost.values == (1000,)

    correct = (
        "SELECT COUNT(*) FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id "
        "WHERE p.name='张三' AND v.cost<=1000"
    )
    assert validate_candidate_obligations(obligations, correct).accepted
    for unsafe in (
        correct + " AND p.region='北京'",
        correct.replace("v.cost<=1000", "CAST(v.cost AS INTEGER)<=1000"),
        correct.replace("v.cost<=1000", "v.cost+1<=1000"),
        correct.replace("v.cost<=1000", "v.cost<=1000 OR 1=1"),
    ):
        assert not validate_candidate_obligations(obligations, unsafe).accepted


@pytest.mark.parametrize(
    ("question", "start", "end"),
    (
        ("2026年上半年就诊了多少次", "2026-01-01", "2026-06-30"),
        ("2026年下半年就诊了多少次", "2026-07-01", "2026-12-31"),
    ),
)
def test_half_year_phrase_binds_both_visit_date_boundaries(
    question,
    start,
    end,
):
    obligations = extract_semantic_obligations(question, DOMAINS)

    assert tuple(item.value for item in obligations.dates) == (start, end)
    assert validate_candidate_obligations(
        obligations,
        "SELECT COUNT(*) FROM visits "
        f"WHERE visit_date>='{start}' AND visit_date<='{end}'",
    ).accepted
    missing_end = validate_candidate_obligations(
        obligations,
        f"SELECT COUNT(*) FROM visits WHERE visit_date>='{start}'",
    )
    assert not missing_end.accepted
    assert "unconsumed date literal" in missing_end.reasons
