"""Primitive slot parsing for the schema-bound NL2SQL compiler."""

from dki.analysis.nl2sql_semantic_common import (
    RegionFilterDisposition,
    age_predicates,
    answer_items,
    classify_region_filter_intent,
    comparison,
    date_predicates,
    gender_predicate,
    limit,
    region_predicates,
)


def test_age_gender_and_region_slots_are_composed():
    question = "找出深圳地区年龄在40至70岁之间的女性患者"
    domains = {"region": ["深圳", "北京"]}
    assert age_predicates(question) == ["age BETWEEN 40 AND 70"]
    assert gender_predicate(question) == "gender = '女'"
    assert region_predicates(question, domains) == ["region = '深圳'"]


def test_region_slot_polarity_is_scoped_to_each_grounded_value():
    domains = {"region": ["北京", "上海", "深圳"]}
    question = "找出深圳女性患者，不要求发生在同一次就诊，但历史上既有疾病甲又使用药A。"

    intent = classify_region_filter_intent(question, domains)

    assert intent.disposition is RegionFilterDisposition.RENDERABLE
    assert intent.positive_values == ("深圳",)
    assert intent.negative_values == ()
    assert region_predicates(question, domains) == ["region = '深圳'"]


def test_region_slot_keeps_local_exclusions_and_rejects_mixed_or_scope():
    domains = {"region": ["北京", "上海", "深圳"]}
    conjunctive = "排除北京、上海地区的深圳患者"
    disjunctive = "查询深圳患者或北京地区以外患者"

    conjunctive_intent = classify_region_filter_intent(conjunctive, domains)
    disjunctive_intent = classify_region_filter_intent(disjunctive, domains)

    assert conjunctive_intent.disposition is RegionFilterDisposition.RENDERABLE
    assert conjunctive_intent.positive_values == ("深圳",)
    assert conjunctive_intent.negative_values == ("北京", "上海")
    assert region_predicates(conjunctive, domains) == [
        "region = '深圳'",
        "region NOT IN ('北京', '上海')",
    ]
    assert disjunctive_intent.disposition is RegionFilterDisposition.TERMINAL_ABSTAIN
    assert region_predicates(disjunctive, domains) == []


def test_age_bounds_parse_split_natural_phrases():
    assert age_predicates("满35岁且不超过75岁的患者") == [
        "age BETWEEN 35 AND 75"
    ]
    assert age_predicates("年龄至少40岁且不超过50岁的患者") == [
        "age BETWEEN 40 AND 50"
    ]
    assert age_predicates("年龄至少40岁且年龄不超过50岁的患者") == [
        "age BETWEEN 40 AND 50"
    ]
    assert age_predicates("未满60岁的患者") == ["age < 60"]
    assert age_predicates("年龄≥55岁的人群") == ["age >= 55"]


def test_bare_age_above_is_strict_but_and_above_is_inclusive():
    assert age_predicates("60岁以上患者") == ["age > 60"]
    assert age_predicates("60岁及以上患者") == ["age >= 60"]


def test_age_bounds_do_not_claim_unlabelled_metric_ranges():
    assert age_predicates("观察次数至少4且不超过6") == []


def test_date_range_and_since_are_distinct():
    assert date_predicates("2025-01-01至2025-12-31的就诊") == [
        "visit_date BETWEEN '2025-01-01' AND '2025-12-31'"
    ]
    assert date_predicates("自2025-01-01以来的就诊", alias="v") == [
        "v.visit_date >= '2025-01-01'"
    ]
    assert date_predicates("查询2025-01-01的记录") == []


def test_answer_items_preserve_requested_projection_order():
    question = "筛选符合条件的患者，返回编号、姓名和年龄；按年龄降序。"
    assert answer_items(question) == ["编号", "姓名", "年龄"]


def test_answer_items_do_not_split_lexical_mention_count():
    question = "返回头实体、尾实体类型、提及次数和关系置信度，按提及次数降序。"
    assert answer_items(question) == ["头实体", "尾实体类型", "提及次数", "关系置信度"]


def test_threshold_and_topk_slots_handle_symbols_and_words():
    assert comparison("记录量≥4且费用均值>8000元", "记录量") == (">=", "4")
    assert comparison("记录量≥4且费用均值>8000元", "费用均值") == (">", "8000")
    assert limit("就诊量排名前5的疾病") == 5
