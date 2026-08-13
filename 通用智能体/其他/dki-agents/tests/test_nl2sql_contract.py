"""Synthetic tests for the semantic NL2SQL candidate contract gate."""

import pytest

from dki.analysis.nl2sql_contract import (
    ContractStatus,
    RowSetMode,
    SemanticContract,
    extract_semantic_contract,
    validate_candidate,
)


def test_extracts_final_output_clause_in_requested_order():
    contract = extract_semantic_contract(
        "筛选患者后，先显示内部说明，再返回患者编号、姓名和年龄；按年龄降序。"
    )
    assert contract.projections == ("patient_id", "name", "age")
    assert contract.ordering == ("age DESC",)
    assert contract.status is ContractStatus.COMPLETE


def test_output_clause_does_not_project_its_filter_prefix():
    contract = extract_semantic_contract(
        "返回头实体类型为Disease、中文关系为临床表现且置信度至少0.5的"
        "头实体、尾实体和观察次数。"
    )

    assert contract.projections == ("head", "tail", "count")
    assert contract.output_slots == ("head", "tail", "count")


def test_preserves_mixed_raw_and_aggregate_output_order():
    contract = extract_semantic_contract("返回就诊次数和科室")

    assert contract.projections == ("department",)
    assert contract.aggregates == ("COUNT(*)",)
    assert contract.output_slots == ("COUNT(*)", "department")
    assert validate_candidate(
        contract,
        "SELECT COUNT(*) AS n, department FROM visits GROUP BY department",
    ).accepted is True

    reversed_result = validate_candidate(
        contract,
        "SELECT department, COUNT(*) AS n FROM visits GROUP BY department",
    )
    assert reversed_result.accepted is False
    assert any("select-list order" in reason for reason in reversed_result.reasons)


def test_ranking_key_repeat_uses_requested_value_occurrence_for_output_order():
    contract = extract_semantic_contract(
        "返回就诊次数最多的前2个科室及其就诊次数。"
    )

    assert contract.projections == ("department",)
    assert contract.aggregates == ("COUNT(*)",)
    assert contract.output_slots == ("department", "COUNT(*)")
    assert validate_candidate(
        contract,
        """
        SELECT department, COUNT(*) AS visit_count
        FROM visits
        GROUP BY department
        ORDER BY visit_count DESC
        LIMIT 2
        """,
    ).accepted is True


def test_per_patient_latest_output_ignores_qualifier_measure_occurrence():
    contract = extract_semantic_contract(
        "截至2025-03-05，返回每位患者最近一次费用至少1000元的"
        "就诊日期、科室和费用。"
    )

    assert contract.projections == (
        "patient_id",
        "visit_date",
        "department",
        "cost",
    )
    assert contract.grouping == ()
    assert validate_candidate(
        contract,
        """
        SELECT v.patient_id, v.visit_date, v.department, v.cost
        FROM visits AS v
        WHERE v.visit_date <= '2025-03-05'
          AND v.cost >= 1000
          AND v.visit_date = (
              SELECT MAX(v2.visit_date) AS max_visit_date
              FROM visits AS v2
              WHERE v2.patient_id = v.patient_id
                AND v2.visit_date <= '2025-03-05'
                AND v2.cost >= 1000
          )
        ORDER BY v.patient_id, v.visit_id
        """,
    ).accepted is True


def test_per_patient_first_visit_date_is_a_minimum_aggregate():
    contract = extract_semantic_contract(
        "返回2025-01-19及以后每位患者诊断为糖尿病肾病的首次就诊日期。"
    )

    assert contract.projections == ("patient_id",)
    assert contract.aggregates == ("MIN(visit_date)",)
    assert contract.grouping == ("patient_id",)
    assert validate_candidate(
        contract,
        """
        SELECT patient_id, MIN(visit_date) AS first_visit_date
        FROM visits
        WHERE visit_date >= '2025-01-19'
          AND disease = '糖尿病肾病'
        GROUP BY patient_id
        ORDER BY patient_id
        """,
    ).accepted is True


def test_per_department_maximum_row_is_not_an_aggregate_output_or_grouping():
    contract = extract_semantic_contract(
        "返回每个科室中住院天数至少0天、费用至少3000元且费用等于"
        "该科室在相同住院天数条件下最高费用的就诊编号、科室和费用。"
    )

    assert contract.projections == ("visit_id", "department", "cost")
    assert contract.aggregates == ()
    assert contract.grouping == ()
    assert validate_candidate(
        contract,
        """
        SELECT v.visit_id, v.department, v.cost
        FROM visits AS v
        WHERE v.los_days >= 0
          AND v.cost >= 3000
          AND v.cost = (
              SELECT MAX(v2.cost) AS max_cost
              FROM visits AS v2
              WHERE v2.department = v.department
                AND v2.los_days >= 0
          )
        ORDER BY v.department, v.visit_id
        """,
    ).accepted is True


def test_limit_only_contract_does_not_invent_an_empty_output_shape():
    contract = extract_semantic_contract("返回前2名及其名次。")

    assert contract.projections == ()
    assert contract.aggregates == ()
    assert contract.limit == 2
    assert validate_candidate(
        contract,
        """
        SELECT visit_id, patient_id, cost,
               ROW_NUMBER() OVER (
                   PARTITION BY department
                   ORDER BY cost DESC, visit_id
               ) AS visit_rank
        FROM visits
        ORDER BY visit_rank
        LIMIT 2
        """,
    ).accepted is True


def test_limit_only_contract_accepts_controlled_deterministic_fee_rank_output():
    contract = extract_semantic_contract("返回前3名及其名次。")

    assert validate_candidate(
        contract,
        "SELECT visit_id, patient_id, cost, "
        "ROW_NUMBER() OVER (PARTITION BY department "
        "ORDER BY cost DESC, visit_id) AS fee_rank "
        "FROM visits ORDER BY fee_rank LIMIT 3",
    ).accepted


@pytest.mark.parametrize(
    ("contract", "sql"),
    [
        (
            SemanticContract(
                projections=("patient_id", "total_rank"),
                aggregates=("SUM(cost)",),
                output_slots=("patient_id", "SUM(cost)", "total_rank"),
                grouping=("patient_id",),
                ordering=("total_rank ASC",),
                limit=3,
                confidence=1.0,
                status=ContractStatus.COMPLETE,
            ),
            "SELECT patient_id, SUM(cost) AS total_cost, "
            "DENSE_RANK() OVER (ORDER BY SUM(cost) DESC) AS total_rank "
            "FROM visits GROUP BY patient_id ORDER BY total_rank LIMIT 3",
        ),
        (
            SemanticContract(
                projections=("head", "observation_rank"),
                aggregates=("SUM(count)",),
                output_slots=("head", "SUM(count)", "observation_rank"),
                grouping=("head",),
                ordering=("observation_rank ASC",),
                limit=3,
                confidence=1.0,
                status=ContractStatus.COMPLETE,
            ),
            "SELECT head, SUM(count) AS observed_count, "
            "ROW_NUMBER() OVER (ORDER BY SUM(count) DESC, head) "
            "AS observation_rank FROM kg_triples GROUP BY head "
            "ORDER BY observation_rank LIMIT 3",
        ),
    ],
)
def test_active_contract_accepts_controlled_global_rank_outputs(contract, sql):
    assert validate_candidate(contract, sql).accepted


def test_semantic_contract_preserves_legacy_positional_fields():
    contract = SemanticContract(
        ("department",),
        ("COUNT(*)",),
        ("department",),
        ("COUNT(*) DESC",),
        3,
        1.0,
        ContractStatus.COMPLETE,
    )

    assert contract.projections == ("department",)
    assert contract.aggregates == ("COUNT(*)",)
    assert contract.grouping == ("department",)
    assert contract.ordering == ("COUNT(*) DESC",)
    assert contract.limit == 3
    assert contract.confidence == 1.0
    assert contract.status is ContractStatus.COMPLETE
    assert contract.output_slots == ()


def test_extracts_visit_and_knowledge_graph_raw_columns():
    visit = extract_semantic_contract("返回就诊编号、就诊日期、科室、疾病、费用和住院天数")
    assert visit.projections == (
        "visit_id",
        "visit_date",
        "department",
        "disease",
        "cost",
        "los_days",
    )

    graph = extract_semantic_contract(
        "输出头实体、中文关系、尾实体、置信度和观察次数"
    )
    assert graph.projections == (
        "head",
        "relation_zh",
        "tail",
        "confidence",
        "count",
    )


def test_extracts_count_distinct_and_numeric_aggregates():
    contract = extract_semantic_contract(
        "返回不同患者数、费用平均值、费用总额、最高费用和最低费用；按费用平均值降序；前5个。"
    )
    assert contract.aggregates == (
        "COUNT(DISTINCT patient_id)",
        "AVG(cost)",
        "SUM(cost)",
        "MAX(cost)",
        "MIN(cost)",
    )
    assert contract.ordering == ("AVG(cost) DESC",)
    assert contract.limit == 5
    assert contract.status is ContractStatus.COMPLETE


@pytest.mark.parametrize(
    "question",
    [
        "返回平均费用（保留2位小数）。",
        "返回平均费用，四舍五入到三位小数。",
        "返回平均费用，精度保留4位。",
    ],
)
def test_decimal_precision_does_not_become_a_result_limit(question):
    contract = extract_semantic_contract(question)

    assert contract.limit is None


def test_real_return_count_survives_an_independent_precision_phrase():
    contract = extract_semantic_contract(
        "返回3条就诊记录，费用保留2位小数。"
    )

    assert contract.limit == 3


def test_explicit_order_direction_is_not_duplicated_by_default_ascending_rule():
    contract = extract_semantic_contract("返回就诊编号和费用；按费用降序排序。")

    assert contract.ordering == ("cost DESC",)


def test_multiple_explicit_order_keys_retain_their_direction_and_sequence():
    contract = extract_semantic_contract(
        "返回患者编号和年龄；按年龄降序、患者编号升序排列。"
    )

    assert contract.ordering == ("age DESC", "patient_id ASC")


def test_undirected_ordering_still_defaults_to_ascending():
    contract = extract_semantic_contract("返回患者编号；按患者编号排序。")

    assert contract.ordering == ("patient_id ASC",)


@pytest.mark.parametrize(
    ("question", "expected_group"),
    [
        ("按英文关系分组统计三元组数量。", "relation"),
        ("按关系英文名分组统计三元组数量。", "relation"),
        ("按 relation 分组统计三元组数量。", "relation"),
        ("按关系分组统计三元组数量。", "relation"),
        ("各关系统计三元组数量。", "relation"),
        ("按中文关系分组统计三元组数量。", "relation_zh"),
        ("按关系中文名分组统计三元组数量。", "relation_zh"),
        ("按 relation_zh 分组统计三元组数量。", "relation_zh"),
    ],
)
def test_relation_group_aliases_keep_english_and_chinese_columns_distinct(
    question,
    expected_group,
):
    contract = extract_semantic_contract(question)

    assert contract.grouping == (expected_group,)


def test_bare_relation_is_canonical_in_projection_output_and_ordering():
    contract = extract_semantic_contract(
        "列出关系、头实体和尾实体；按关系升序排序。"
    )

    assert contract.projections == ("relation", "head", "tail")
    assert contract.output_slots == ("relation", "head", "tail")
    assert contract.ordering == ("relation ASC",)


def test_uppercase_relation_filter_does_not_relabel_local_chinese_output():
    contract = extract_semantic_contract(
        "返回关系为 HAS_SYMPTOM 的中文关系和尾实体。"
    )

    assert contract.projections == ("relation_zh", "tail")
    assert contract.output_slots == ("relation_zh", "tail")


def test_explicit_chinese_relation_output_rebinds_bare_relation_grouping():
    contract = extract_semantic_contract(
        "每种关系的平均置信度，返回关系中文名和平均置信度，"
        "按平均置信度降序排列，前3条。"
    )

    assert contract.projections == ("relation_zh",)
    assert contract.aggregates == ("AVG(confidence)",)
    assert contract.output_slots == ("relation_zh", "AVG(confidence)")
    assert contract.grouping == ("relation_zh",)
    assert contract.ordering == ("AVG(confidence) DESC",)
    assert contract.limit == 3


def test_explicit_code_relation_output_keeps_bare_relation_grouping_canonical():
    contract = extract_semantic_contract(
        "每种关系的平均置信度，返回关系、关系中文名和平均置信度。"
    )

    assert contract.projections == ("relation", "relation_zh")
    assert contract.grouping == ("relation", "relation_zh")


def test_evidence_count_is_the_physical_kg_count_column():
    contract = extract_semantic_contract(
        "返回头实体、尾实体、置信度和证据次数。"
    )

    assert contract.projections == ("head", "tail", "confidence", "count")
    assert contract.aggregates == ()


def test_accumulated_evidence_count_is_sum_of_the_physical_column():
    contract = extract_semantic_contract(
        "按英文关系汇总证据次数，返回英文关系和证据次数合计。"
    )

    assert contract.projections == ("relation",)
    assert contract.aggregates == ("SUM(count)",)
    assert contract.grouping == ("relation",)


def test_entity_type_statistics_bind_row_count_sum_and_average_mentions():
    contract = extract_semantic_contract(
        "按组统计每种实体类型的实体数、提及总次数和平均提及次数"
        "（保留2位小数）。"
    )

    assert contract.aggregates == (
        "COUNT(*)",
        "SUM(mention_count)",
        "AVG(mention_count)",
    )
    assert contract.grouping == ("type",)
    assert contract.limit is None


def test_count_distinct_is_not_confused_with_select_distinct():
    count_contract = extract_semantic_contract("返回不同头实体数。")

    assert count_contract.aggregates == ("COUNT(DISTINCT head)",)
    assert count_contract.row_set_mode is RowSetMode.PRESERVE
    assert validate_candidate(
        count_contract,
        "SELECT COUNT(DISTINCT head) FROM kg_triples",
    ).accepted
    assert not validate_candidate(
        count_contract,
        "SELECT DISTINCT head FROM kg_triples",
    ).accepted

    row_contract = extract_semantic_contract("返回不同头实体。")
    assert row_contract.aggregates == ()
    assert row_contract.projections == ("head",)
    assert row_contract.row_set_mode is RowSetMode.DISTINCT
    assert validate_candidate(
        row_contract,
        "SELECT DISTINCT head FROM kg_triples",
    ).accepted


def test_count_distinct_bare_relation_is_aggregate_not_outer_distinct():
    contract = extract_semantic_contract("统计不同关系数。")

    assert contract.projections == ()
    assert contract.aggregates == ("COUNT(DISTINCT relation)",)
    assert contract.row_set_mode is RowSetMode.PRESERVE
    assert validate_candidate(
        contract,
        "SELECT COUNT(DISTINCT relation) FROM kg_triples",
    ).accepted


@pytest.mark.parametrize(
    "question",
    [
        (
            "在每个地区内按年龄降序给患者做连续排名，"
            "列出地区、患者编号、年龄和排名。"
        ),
        (
            "在每种实体类型内按提及次数降序生成排序编号，"
            "列出实体类型、实体名称、提及次数和编号。"
        ),
        (
            "在每种性别内按年龄计算累计分布，"
            "列出性别、患者编号、年龄和累计分布。"
        ),
    ],
)
def test_window_analysis_language_does_not_invent_grouped_count(question):
    contract = extract_semantic_contract(question)

    assert contract.aggregates == ()


def test_extracts_average_visit_cost_in_explicit_age_group_output():
    contract = extract_semantic_contract("返回各年龄段的平均就诊费用")

    assert contract.projections == ("age",)
    assert contract.aggregates == ("AVG(cost)",)
    assert contract.grouping == ("age",)
    assert contract.status is ContractStatus.COMPLETE


@pytest.mark.parametrize(
    ("question", "direction"),
    [
        ("平均就诊费用最高的年龄段", "DESC"),
        ("平均就诊费用最低的年龄段", "ASC"),
    ],
)
def test_age_group_average_extreme_ranks_the_metric_not_raw_age(
    question,
    direction,
):
    contract = extract_semantic_contract(question)
    sql = (
        "SELECT CASE WHEN p.age <= 18 THEN '0-18' "
        "WHEN p.age <= 40 THEN '19-40' "
        "WHEN p.age <= 60 THEN '41-60' ELSE '60+' END AS age_group, "
        "ROUND(AVG(v.cost), 2) AS avg_value "
        "FROM visits AS v JOIN patients AS p "
        "ON v.patient_id = p.patient_id "
        "GROUP BY CASE WHEN p.age <= 18 THEN '0-18' "
        "WHEN p.age <= 40 THEN '19-40' "
        "WHEN p.age <= 60 THEN '41-60' ELSE '60+' END "
        f"ORDER BY avg_value {direction} LIMIT 1"
    )

    assert contract.projections == ("age",)
    assert contract.aggregates == ("AVG(cost)",)
    assert contract.grouping == ("age",)
    assert contract.ordering == (f"AVG(cost) {direction}",)
    assert contract.limit == 1
    assert validate_candidate(contract, sql).accepted


@pytest.mark.parametrize(
    "question",
    (
        "以费用1000元为界分成较低费用档和较高费用档，各档统计就诊次数。",
        "以费用1000为界分成较低和较高两个费用档，各档汇总就诊次数。",
    ),
)
def test_complete_cost_band_wording_extracts_a_grouped_cost_contract(question):
    contract = extract_semantic_contract(question)

    assert contract.projections == ("cost",)
    assert contract.aggregates == ("COUNT(*)",)
    assert contract.grouping == ("cost",)
    assert validate_candidate(
        contract,
        "SELECT CASE WHEN v.cost < 1000 THEN '较低费用档' "
        "ELSE '较高费用档' END AS cost_band, COUNT(*) AS n "
        "FROM visits AS v GROUP BY cost_band",
    ).accepted


def test_cost_band_repeated_case_grouping_uses_the_same_strict_lineage():
    contract = SemanticContract(
        projections=("cost",),
        aggregates=("COUNT(*)",),
        grouping=("cost",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    case = "CASE WHEN cost < 1000 THEN 'low' ELSE 'high' END"

    result = validate_candidate(
        contract,
        f"SELECT {case} AS cost_band, COUNT(*) FROM visits GROUP BY {case}",
    )

    assert result.accepted is True


@pytest.mark.parametrize(
    "question",
    (
        "分成较低费用档和较高费用档，各档统计就诊次数。",
        "以费用1000元为界分成费用档，各档统计就诊次数。",
        "以费用1000元为界分成较低费用档和较高费用档。",
        "各档统计就诊次数。以费用1000元为界分成较低费用档和较高费用档。",
        "以费用1000元为界。分成较低费用档和较高费用档，各档统计就诊次数。",
    ),
)
def test_incomplete_cost_band_wording_does_not_infer_cost_grouping(question):
    contract = extract_semantic_contract(question)

    assert contract.grouping != ("cost",)


@pytest.mark.parametrize(
    ("question", "projections", "aggregates", "grouping", "ordering", "limit"),
    [
        ("患者总数是多少", (), ("COUNT(*)",), (), (), None),
        (
            "就诊量前5的疾病是哪些",
            ("disease",),
            ("COUNT(*)",),
            ("disease",),
            ("COUNT(*) DESC",),
            5,
        ),
        (
            "就诊次数最多的前三个科室",
            ("department",),
            ("COUNT(*)",),
            ("department",),
            ("COUNT(*) DESC",),
            3,
        ),
        ("按月统计就诊量趋势", ("month",), ("COUNT(*)",), ("month",), (), None),
        ("按年统计就诊量", ("year",), ("COUNT(*)",), ("year",), (), None),
        ("按年龄段统计就诊量", ("age",), ("COUNT(*)",), ("age",), (), None),
        (
            "处方最多的前5种药物",
            ("drug_prescribed",),
            ("COUNT(*)",),
            ("drug_prescribed",),
            ("COUNT(*) DESC",),
            5,
        ),
        (
            "各检查项目的使用次数",
            ("exam",),
            ("COUNT(*)",),
            ("exam",),
            (),
            None,
        ),
        (
            "提及次数最多的疾病有哪些",
            ("name", "mention_count"),
            (),
            (),
            ("mention_count DESC",),
            10,
        ),
    ],
)
def test_compact_public_base_contracts_keep_their_result_grain(
    question,
    projections,
    aggregates,
    grouping,
    ordering,
    limit,
):
    contract = extract_semantic_contract(question)

    assert contract.projections == projections
    assert contract.aggregates == aggregates
    assert contract.grouping == grouping
    assert contract.ordering == ordering
    assert contract.limit == limit
    assert contract.status is ContractStatus.COMPLETE


def test_prescription_top_count_does_not_override_an_explicit_average_metric():
    contract = extract_semantic_contract("处方平均费用最高的前5种药物")

    assert contract.projections == ("drug_prescribed",)
    assert contract.aggregates == ("AVG(cost)",)
    assert contract.grouping == ("drug_prescribed",)
    assert contract.ordering == ("AVG(cost) DESC",)
    assert contract.limit == 5


@pytest.mark.parametrize("entity_noun", ["疾病", "症状", "药物", "检查"])
def test_entity_mention_leaderboards_share_row_level_contract(entity_noun):
    contract = extract_semantic_contract(f"提及次数最多的{entity_noun}有哪些")

    assert contract.projections == ("name", "mention_count")
    assert contract.aggregates == ()
    assert contract.grouping == ()
    assert contract.ordering == ("mention_count DESC",)
    assert contract.limit == 10


def test_grouping_defaults_to_count_for_grouped_statistics():
    contract = extract_semantic_contract("按科室统计就诊次数，返回科室和次数；前3个")
    assert contract.grouping == ("department",)
    assert contract.projections == ("department",)
    assert contract.aggregates == ("COUNT(*)",)
    assert contract.ordering == ("COUNT(*) DESC",)
    assert contract.limit == 3


@pytest.mark.parametrize(
    ("question", "expected_ordering"),
    (
        (
            "按年统计就诊次数；每年内按次数降序排列。",
            ("year ASC", "COUNT(*) DESC"),
        ),
        (
            "按月份统计就诊次数；按月份及次数降序排列。",
            ("month ASC", "COUNT(*) DESC"),
        ),
    ),
)
def test_temporal_count_ordering_keeps_outer_period_before_descending_count(
    question,
    expected_ordering,
):
    contract = extract_semantic_contract(question)

    assert contract.aggregates == ("COUNT(*)",)
    assert contract.ordering == expected_ordering


@pytest.mark.parametrize(
    ("question", "expected_ordering"),
    (
        (
            "按年统计就诊次数；按年份降序、次数降序排列。",
            ("year DESC", "COUNT(*) DESC"),
        ),
        (
            "按月份统计就诊次数；按月份降序、次数降序排列。",
            ("month DESC", "COUNT(*) DESC"),
        ),
        (
            "按年统计平均费用；每年内按次数降序排列。",
            ("AVG(cost) DESC",),
        ),
        (
            "按年统计就诊次数；每年内取第1行，并说明每年内按次数降序。",
            (),
        ),
        (
            "按月份统计就诊次数；每月内取第1行，并说明每月内按次数降序。",
            (),
        ),
        (
            "按年统计就诊次数；每年内按次数降序排名。",
            (),
        ),
        (
            "按月份统计就诊次数；每月内按次数降序取前1行。",
            (),
        ),
    ),
)
def test_temporal_count_ordering_does_not_override_explicit_or_inner_semantics(
    question,
    expected_ordering,
):
    contract = extract_semantic_contract(question)

    assert contract.ordering == expected_ordering


def test_group_argmin_infers_dimension_ascending_count_and_single_limit():
    contract = extract_semantic_contract("找出就诊数量最少的科室。")

    assert contract.projections == ("department",)
    assert contract.aggregates == ("COUNT(*)",)
    assert contract.grouping == ("department",)
    assert contract.ordering == ("COUNT(*) ASC",)
    assert contract.limit == 1
    assert validate_candidate(
        contract,
        """
        SELECT department, COUNT(*) AS visit_count
        FROM visits
        GROUP BY department
        ORDER BY visit_count ASC, department ASC
        LIMIT 1
        """,
    ).accepted is True


def test_which_dimension_metric_superlative_infers_grouped_top_one():
    disease = extract_semantic_contract(
        "哪种疾病涉及的不同患者人数最多？"
    )
    assert disease.projections == ("disease",)
    assert disease.aggregates == ("COUNT(DISTINCT patient_id)",)
    assert disease.grouping == ("disease",)
    assert disease.ordering == ("COUNT(DISTINCT patient_id) DESC",)
    assert disease.limit == 1

    drug = extract_semantic_contract(
        "开药次数最多的是哪种药？"
    )
    assert drug.projections == ("drug_prescribed",)
    assert drug.grouping == ("drug_prescribed",)
    assert drug.limit == 1


def test_patient_demographic_group_uses_patient_row_count():
    contract = extract_semantic_contract(
        "每个城市的患者数和平均年龄分别是多少"
    )

    assert contract.projections == ("region",)
    assert contract.aggregates == ("COUNT(*)", "AVG(age)")
    assert contract.grouping == ("region",)


def test_explicit_patient_attribute_group_does_not_become_patient_entity_grain():
    contract = extract_semantic_contract(
        "对诊断为2型糖尿病且费用不低于0元的就诊，"
        "按患者性别分组统计不同患者数和平均费用。"
    )

    assert contract.projections == ("gender",)
    assert contract.aggregates == (
        "COUNT(DISTINCT patient_id)",
        "AVG(cost)",
    )
    assert contract.grouping == ("gender",)


def test_aggregate_threshold_target_infers_group_key_before_final_metric_output():
    contract = extract_semantic_contract(
        "找出开具药物为丙戊酸钠且开具次数不低于2次的科室，"
        "并显示开具次数。"
    )

    assert contract.projections == ("department",)
    assert contract.aggregates == ("COUNT(*)",)
    assert contract.grouping == ("department",)
    assert not validate_candidate(
        contract,
        "SELECT COUNT(*) FROM visits WHERE drug_prescribed='丙戊酸钠'",
    ).accepted
    assert validate_candidate(
        contract,
        "SELECT department, COUNT(*) AS prescription_count FROM visits "
        "WHERE drug_prescribed='丙戊酸钠' GROUP BY department "
        "HAVING COUNT(*)>=2 ORDER BY prescription_count DESC, department",
    ).accepted


def test_kg_type_transition_count_is_bound_to_the_complete_transition_tuple():
    contract = extract_semantic_contract(
        "知识图谱按头实体类型、关系和尾实体类型统计类型转换次数。"
    )

    assert contract.projections == ("head_type", "relation", "tail_type")
    assert contract.aggregates == ("COUNT(*)",)
    assert contract.grouping == ("head_type", "relation", "tail_type")


@pytest.mark.parametrize(
    ("question", "expected_aggregates"),
    (
        (
            "知识图谱按头实体类型、关系和尾实体类型汇总观察次数合计。",
            ("SUM(count)",),
        ),
        (
            "知识图谱按头实体类型、关系和尾实体类型统计不同类型转换种类数。",
            (),
        ),
        (
            "知识图谱按头实体类型、关系和尾实体类型统计不同类型转换次数。",
            (),
        ),
        ("按科室统计费用总和并比较类型转换次数。", ("SUM(cost)",)),
        ("按科室统计类型转换次数。", ()),
    ),
)
def test_type_transition_words_do_not_broaden_other_aggregate_semantics(
    question,
    expected_aggregates,
):
    contract = extract_semantic_contract(question)

    assert contract.aggregates == expected_aggregates


def test_natural_kg_rank_and_count_contracts_use_graph_roles():
    head_rank = extract_semantic_contract("拥有并发症最多的疾病是哪种")
    assert head_rank.projections == ("head",)
    assert head_rank.aggregates == ("COUNT(*)",)
    assert head_rank.grouping == ("head",)
    assert validate_candidate(
        head_rank,
        "SELECT head AS disease, COUNT(*) AS n FROM kg_triples "
        "WHERE relation='HAS_COMPLICATION' GROUP BY head ORDER BY n DESC LIMIT 1",
    ).accepted

    tail_rank = extract_semantic_contract("被最多疾病列为症状的是哪个症状")
    assert tail_rank.projections == ("tail",)
    assert tail_rank.aggregates == ("COUNT(*)",)
    assert tail_rank.grouping == ("tail",)
    assert validate_candidate(
        tail_rank,
        "SELECT tail AS symptom, COUNT(*) AS n FROM kg_triples "
        "WHERE relation='HAS_SYMPTOM' GROUP BY tail ORDER BY n DESC LIMIT 1",
    ).accepted

    head_count = extract_semantic_contract(
        "各类疾病关联的临床表现数量，降序前3"
    )
    assert head_count.projections == ("head",)
    assert head_count.aggregates == ("COUNT(*)",)
    assert head_count.grouping == ("head",)
    assert validate_candidate(
        head_count,
        "SELECT head AS disease, COUNT(*) AS n FROM kg_triples "
        "WHERE relation='HAS_SYMPTOM' GROUP BY head "
        "ORDER BY n DESC LIMIT 3",
    ).accepted


def test_structured_triple_with_head_mentions_keeps_base_triple_columns():
    contract = extract_semantic_contract(
        "查询头实体类型为Disease、中文关系为临床表现且置信度不低于0.07"
        "的三元组，并返回头实体提及次数。"
    )

    assert contract.projections == (
        "head",
        "relation_zh",
        "tail",
        "mention_count",
    )
    assert validate_candidate(
        contract,
        "SELECT t.head, t.relation_zh, t.tail, e.mention_count "
        "FROM kg_triples t JOIN kg_entities e "
        "ON t.head=e.name AND t.head_type=e.type "
        "WHERE t.head_type='Disease' AND t.relation_zh='临床表现' "
        "AND t.confidence>=0.07 "
        "ORDER BY e.mention_count DESC, t.head, t.tail",
    ).accepted


def test_natural_kg_required_relations_extract_all_of_set_semantics():
    contract = extract_semantic_contract(
        "既有并发症又有用药禁忌记录的疾病"
    )

    assert contract.set_semantics == "all_of"
    assert validate_candidate(
        contract,
        "SELECT head FROM kg_triples WHERE relation='HAS_COMPLICATION' "
        "INTERSECT "
        "SELECT head FROM kg_triples WHERE relation='CONTRAINDICATED_FOR'",
    ).accepted


def test_dimension_occurrence_difference_extracts_matching_set_semantics():
    contract = extract_semantic_contract(
        "在呼吸内科出现但在骨科没出现的疾病"
    )

    assert contract.set_semantics == "difference"
    assert validate_candidate(
        contract,
        "SELECT disease FROM visits WHERE department='呼吸内科' "
        "EXCEPT "
        "SELECT disease FROM visits WHERE department='骨科'",
    ).accepted


def test_group_aggregate_comparisons_keep_group_metric_and_global_baseline_shape():
    average = extract_semantic_contract(
        "平均住院天数低于全体平均住院天数的疾病"
    )
    assert average.projections == ("disease",)
    assert average.aggregates == ("AVG(los_days)",)
    assert average.grouping == ("disease",)
    assert average.ordering == ("AVG(los_days) ASC",)
    assert validate_candidate(
        average,
        "SELECT disease, ROUND(AVG(los_days), 2) AS avg_value "
        "FROM visits GROUP BY disease "
        "HAVING AVG(los_days) < (SELECT AVG(los_days) FROM visits) "
        "ORDER BY avg_value ASC",
    ).accepted

    count = extract_semantic_contract(
        "就诊次数超过平均每种疾病就诊次数的疾病"
    )
    assert count.projections == ("disease",)
    assert count.aggregates == ("COUNT(*)",)
    assert count.grouping == ("disease",)
    assert count.ordering == ("COUNT(*) DESC",)
    assert validate_candidate(
        count,
        "SELECT disease, COUNT(*) AS n FROM visits GROUP BY disease "
        "HAVING COUNT(*) > (SELECT AVG(group_count) FROM "
        "(SELECT COUNT(*) AS group_count FROM visits GROUP BY disease) "
        "AS group_counts) "
        "ORDER BY n DESC",
    ).accepted


def test_patient_aggregate_uses_hidden_entity_key_not_display_name_grain():
    contract = extract_semantic_contract(
        "按患者汇总费用，列出累计费用最高的前3位患者姓名和累计费用。"
    )

    assert contract.projections == ("name",)
    assert contract.aggregates == ("SUM(cost)",)
    assert contract.grouping == ("patient_id", "name")
    assert validate_candidate(
        contract,
        "SELECT p.name, SUM(v.cost) AS total_cost "
        "FROM patients p JOIN visits v ON v.patient_id=p.patient_id "
        "GROUP BY p.patient_id, p.name "
        "ORDER BY total_cost DESC, p.name ASC LIMIT 3",
    ).accepted
    assert not validate_candidate(
        contract,
        "SELECT p.name, SUM(v.cost) AS total_cost "
        "FROM patients p JOIN visits v ON v.patient_id=p.patient_id "
        "GROUP BY p.name ORDER BY total_cost DESC LIMIT 3",
    ).accepted


def test_patient_aggregate_ranking_may_keep_metric_out_of_name_only_output():
    contract = extract_semantic_contract(
        "总花费最高的前三位患者分别叫什么名字？"
    )
    sql = (
        "SELECT p.name FROM patients p "
        "JOIN visits v ON v.patient_id=p.patient_id "
        "GROUP BY p.patient_id, p.name "
        "ORDER BY SUM(v.cost) DESC, p.name ASC LIMIT 3"
    )

    assert contract.projections == ("name",)
    assert contract.aggregates == ()
    assert contract.grouping == ("patient_id", "name")
    assert contract.ordering == ("SUM(cost) DESC",)
    assert validate_candidate(contract, sql).accepted
    assert not validate_candidate(
        contract,
        sql.replace("SELECT p.name", "SELECT p.name, SUM(v.cost) AS total_cost"),
    ).accepted


def test_date_range_filter_before_answer_does_not_become_projection():
    contract = extract_semantic_contract(
        "列出患者P00001就诊结算日期在2025-01-01到2025-01-31内的"
        "所有医疗就诊记录编号"
    )

    assert contract.projections == ("visit_id",)
    assert contract.aggregates == ()


def test_implicit_answer_list_after_peer_average_filter_excludes_filter_metric():
    contract = extract_semantic_contract(
        "查询科室为内分泌科且费用至少比该科室平均费用高0元的"
        "就诊编号、患者编号、诊断和费用。"
    )

    assert contract.projections == (
        "visit_id",
        "patient_id",
        "disease",
        "cost",
    )
    assert contract.aggregates == ()
    assert contract.output_slots == contract.projections
    assert contract.status is ContractStatus.COMPLETE


@pytest.mark.parametrize(
    ("question", "expected", "sql"),
    [
        (
            "对科室内分泌科中费用不低于0元的就诊按费用降序编号，"
            "返回前3名及其名次。",
            (
                ("visit_id", "patient_id", "cost", "fee_rank"),
                (),
                (),
                ("fee_rank ASC",),
                3,
            ),
            "SELECT visit_id, patient_id, cost, "
            "ROW_NUMBER() OVER (PARTITION BY department "
            "ORDER BY cost DESC, visit_id) AS fee_rank "
            "FROM visits WHERE department='内分泌科' AND cost>=0 "
            "ORDER BY fee_rank LIMIT 3",
        ),
        (
            "在科室内分泌科中按患者累计费用排名，筛选累计费用不低于0元的"
            "记录并返回前10名及排名。",
            (
                ("patient_id", "total_rank"),
                ("SUM(cost)",),
                ("patient_id",),
                ("total_rank ASC",),
                10,
            ),
            "SELECT patient_id, SUM(cost) AS total_cost, "
            "DENSE_RANK() OVER (ORDER BY SUM(cost) DESC) AS total_rank "
            "FROM visits WHERE department='内分泌科' "
            "GROUP BY patient_id HAVING SUM(cost)>=0 "
            "ORDER BY total_rank, patient_id LIMIT 10",
        ),
        (
            "对头实体类型为Disease、中文关系为临床表现且置信度不低于0.07"
            "的三元组，按头实体累计观察次数排名并返回前10名。",
            (
                ("head", "observation_rank"),
                ("SUM(count)",),
                ("head",),
                ("observation_rank ASC",),
                10,
            ),
            "SELECT head, SUM(count) AS observed_count, "
            "ROW_NUMBER() OVER (ORDER BY SUM(count) DESC, head) "
            "AS observation_rank FROM kg_triples "
            "WHERE head_type='Disease' AND relation_zh='临床表现' "
            "AND confidence>=0.07 GROUP BY head "
            "ORDER BY observation_rank LIMIT 10",
        ),
    ],
)
def test_controlled_rank_contracts_bind_outputs_grouping_and_window_order(
    question,
    expected,
    sql,
):
    contract = extract_semantic_contract(question)

    assert (
        contract.projections,
        contract.aggregates,
        contract.grouping,
        contract.ordering,
        contract.limit,
    ) == expected
    assert contract.status is ContractStatus.COMPLETE
    assert validate_candidate(contract, sql).accepted


def test_english_number_of_patients_masks_filter_nouns_from_output():
    contract = extract_semantic_contract(
        "give me the number of 高血压 primary disease patients "
        "who have stayed more than 7 days in hospital."
    )

    assert contract.projections == ()
    assert contract.aggregates == ("COUNT(DISTINCT patient_id)",)


def test_english_number_of_days_does_not_cross_relative_clause_into_patient_count():
    contract = extract_semantic_contract(
        "find out the primary disease and number of days for which patient "
        "with patient id P00001 was hospitalized in hospital."
    )

    assert contract.aggregates == ()


def test_english_number_of_times_is_record_count_not_patient_count():
    contract = extract_semantic_contract(
        "Count the number of times patient P00001 has received 腹部CT."
    )

    assert contract.aggregates == ("COUNT(*)",)


def test_row_argmax_question_asks_for_name_not_scalar_maximum():
    contract = extract_semantic_contract(
        "费用最高的那次就诊对应的患者叫什么名字？"
    )

    assert contract.projections == ("name",)
    assert contract.aggregates == ()
    assert contract.ordering == ("cost DESC",)
    assert contract.limit == 1
    assert validate_candidate(
        contract,
        """
        SELECT p.name
        FROM patients AS p
        JOIN visits AS v ON v.patient_id = p.patient_id
        ORDER BY v.cost DESC, v.visit_id ASC
        LIMIT 1
        """,
    ).accepted is True
    assert validate_candidate(
        contract,
        "SELECT MAX(cost) AS max_value FROM visits",
    ).accepted is False


def test_row_argmin_question_asks_for_name_not_scalar_minimum():
    contract = extract_semantic_contract(
        "费用最低的就诊患者姓名是什么？"
    )

    assert contract.projections == ("name",)
    assert contract.aggregates == ()
    assert contract.ordering == ("cost ASC",)
    assert contract.limit == 1
    assert validate_candidate(
        contract,
        "SELECT p.name FROM patients AS p "
        "JOIN visits AS v ON v.patient_id = p.patient_id "
        "ORDER BY v.cost ASC, v.visit_id ASC LIMIT 1",
    ).accepted
    assert not validate_candidate(
        contract,
        "SELECT MIN(cost) AS min_value FROM visits",
    ).accepted


def test_scalar_minimum_with_visit_classifier_does_not_become_top_one():
    contract = extract_semantic_contract(
        "最低的一笔就诊费用是多少？"
    )

    assert contract.projections == ()
    assert contract.aggregates == ("MIN(cost)",)
    assert contract.ordering == ()
    assert contract.limit is None
    assert validate_candidate(
        contract,
        "SELECT MIN(cost) AS min_cost FROM visits",
    ).accepted


def test_top_visit_rows_keep_ranking_out_of_projection_and_bind_bare_id():
    contract = extract_semantic_contract(
        "列出费用最高的3次就诊的编号和费用。"
    )

    assert contract.status is ContractStatus.COMPLETE
    assert contract.projections == ("visit_id", "cost")
    assert contract.aggregates == ()
    assert contract.grouping == ()
    assert contract.ordering == ("cost DESC",)
    assert contract.limit == 3
    assert validate_candidate(
        contract,
        "SELECT visit_id, cost FROM visits "
        "ORDER BY cost DESC, visit_id ASC LIMIT 3",
    ).accepted


def test_compound_union_requires_matching_any_of_semantics_and_branch_outputs():
    contract = extract_semantic_contract(
        "列出曾在心血管内科或内分泌科就诊过的患者编号。"
    )
    union = (
        "SELECT patient_id FROM visits WHERE department='心血管内科' "
        "UNION "
        "SELECT patient_id FROM visits WHERE department='内分泌科' "
        "ORDER BY patient_id ASC"
    )

    assert contract.set_semantics == "any_of"
    assert validate_candidate(contract, union).accepted
    assert not validate_candidate(
        contract,
        union.replace(" UNION ", " INTERSECT "),
    ).accepted
    assert not validate_candidate(
        contract,
        "SELECT patient_id FROM visits WHERE department='心血管内科' "
        "UNION SELECT visit_id FROM visits WHERE department='内分泌科'",
    ).accepted
    assert not validate_candidate(
        contract,
        union.replace(" UNION ", " UNION ALL "),
    ).accepted


def test_difference_contract_rejects_an_any_of_simple_lowering():
    contract = extract_semantic_contract(
        "列出在心血管内科就诊过但没有在内分泌科就诊过的患者编号。"
    )

    assert contract.set_semantics == "difference"
    result = validate_candidate(
        contract,
        "SELECT patient_id FROM visits "
        "WHERE department IN ('心血管内科', '内分泌科')",
    )
    assert not result.accepted
    assert any("difference" in reason for reason in result.reasons)


def test_global_latest_visit_question_binds_temporal_order_and_limit():
    contract = extract_semantic_contract(
        "最近一次就诊的患者叫什么名字？"
    )

    assert contract.status is ContractStatus.COMPLETE
    assert contract.projections == ("name",)
    assert contract.aggregates == ()
    assert contract.ordering == ("visit_date DESC",)
    assert contract.limit == 1
    assert validate_candidate(
        contract,
        "SELECT p.name FROM patients AS p "
        "JOIN visits AS v ON v.patient_id = p.patient_id "
        "ORDER BY v.visit_date DESC, v.visit_id DESC LIMIT 1",
    ).accepted
    assert not validate_candidate(
        contract,
        "SELECT DISTINCT p.name FROM patients AS p "
        "JOIN visits AS v ON v.patient_id = p.patient_id "
        "ORDER BY p.name ASC",
    ).accepted


def test_per_patient_latest_visit_is_not_misread_as_global_top_one():
    contract = extract_semantic_contract(
        "列出每位患者最近一次就诊的患者编号、日期和科室。"
    )

    assert contract.limit is None
    assert contract.ordering == ()


def test_unnamed_scalar_aggregate_subquery_is_not_an_exposed_relation():
    contract = extract_semantic_contract(
        "列出每位患者最近一次就诊的患者编号、日期和科室。"
    )
    sql = (
        "SELECT v.patient_id, v.visit_date, v.department FROM visits AS v "
        "WHERE v.visit_date = ("
        "SELECT MAX(v2.visit_date) FROM visits AS v2 "
        "WHERE v2.patient_id = v.patient_id"
        ") ORDER BY v.patient_id ASC, v.visit_id ASC"
    )

    assert validate_candidate(contract, sql).accepted


def test_per_patient_temporal_interrogatives_keep_row_grain_and_outputs():
    earliest = extract_semantic_contract(
        "每位患者第一次就诊的编号、患者编号和日期分别是什么？"
    )
    assert earliest.projections == ("visit_id", "patient_id", "visit_date")
    assert earliest.grouping == ()
    assert earliest.ordering == ()
    assert earliest.limit is None

    latest = extract_semantic_contract(
        "请返回各患者末次就诊的患者编号、日期和科室。"
    )
    assert latest.projections == ("patient_id", "visit_date", "department")
    assert latest.grouping == ()
    assert latest.ordering == ()
    assert latest.limit is None


@pytest.mark.parametrize(
    ("question", "expected_grouping"),
    [
        (
            "按头实体类型和尾实体类型分组统计三元组数量。",
            ("head_type", "tail_type"),
        ),
        (
            "按头尾实体类型组合统计三元组数量。",
            ("head_type", "tail_type"),
        ),
        (
            "按科室、疾病分组统计就诊数量。",
            ("department", "disease"),
        ),
        (
            "知识图谱中关系DIAGNOSED_BY连接了哪些头尾类型组合？"
            "统计每种组合的三元组数。",
            ("head_type", "tail_type"),
        ),
        (
            "用公共表表达式连接患者与就诊，统计地区和科室组合的就诊次数"
            "并分地区降序展示。",
            ("region", "department"),
        ),
    ],
)
def test_controlled_composite_group_dimensions_keep_declared_order(
    question,
    expected_grouping,
):
    contract = extract_semantic_contract(question)

    assert contract.projections == expected_grouping
    assert contract.aggregates == ("COUNT(*)",)
    assert contract.grouping == expected_grouping
    assert contract.status is ContractStatus.COMPLETE


def test_multiple_role_fields_without_explicit_group_language_remain_rows():
    contract = extract_semantic_contract("返回头实体类型和尾实体类型字段清单。")

    assert contract.aggregates == ()
    assert contract.grouping == ()


@pytest.mark.parametrize(
    ("question", "expected_projections"),
    [
        (
            "统计费用不低于1000元的就诊编号、患者编号和费用字段清单。",
            ("visit_id", "patient_id", "cost"),
        ),
        (
            "汇总就诊日期在2025-01-01至2025-01-31范围内的"
            "患者编号和就诊日期字段清单。",
            ("patient_id", "visit_date"),
        ),
        (
            "请汇总就诊日期从2024-10-29到2026-03-18（含首尾）的"
            "就诊编号、日期和疾病。",
            ("visit_id", "visit_date", "disease"),
        ),
    ],
)
def test_filtered_statistic_field_lists_are_complete_row_outputs(
    question,
    expected_projections,
):
    contract = extract_semantic_contract(question)

    assert contract.projections == expected_projections
    assert contract.output_slots == expected_projections
    assert contract.aggregates == ()
    assert contract.grouping == ()
    assert contract.status is ContractStatus.COMPLETE


@pytest.mark.parametrize(
    "question",
    [
        "统计费用不低于1000元的费用字段清单。",
        "汇总就诊编号和费用字段清单。",
    ],
)
def test_field_list_row_inference_requires_filter_and_two_complete_aliases(question):
    contract = extract_semantic_contract(question)

    assert contract.projections == ()
    assert contract.aggregates == ()
    assert contract.grouping == ()
    assert contract.status is ContractStatus.NONE


@pytest.mark.parametrize(
    ("question", "expected_aggregate"),
    [
        ("最高单次费用是多少？", "MAX(cost)"),
        ("最长住院天数是多少？", "MAX(los_days)"),
        ("证据合计是多少？", "SUM(count)"),
        ("总费用是多少？", "SUM(cost)"),
    ],
)
def test_controlled_scalar_measure_phrases_bind_exact_aggregates(
    question,
    expected_aggregate,
):
    contract = extract_semantic_contract(question)

    assert contract.projections == ()
    assert contract.aggregates == (expected_aggregate,)
    assert contract.grouping == ()
    assert contract.status is ContractStatus.COMPLETE


def test_summary_verb_final_character_does_not_invent_sum():
    contract = extract_semantic_contract("汇总费用字段清单。")

    assert contract.aggregates == ()
    assert contract.status is ContractStatus.NONE


@pytest.mark.parametrize(
    ("question", "expected_projections"),
    [
        (
            "返回每个科室的最近一条就诊记录的就诊编号、日期和费用。",
            ("department", "visit_id", "visit_date", "cost"),
        ),
        (
            "返回每个科室费用最高的就诊编号和费用。",
            ("department", "visit_id", "cost"),
        ),
        (
            "返回每种疾病住院天数最长的就诊编号、疾病和住院天数。",
            ("visit_id", "disease", "los_days"),
        ),
        (
            "返回按科室分区取第1行的就诊编号和科室。",
            ("visit_id", "department"),
        ),
        (
            "对每个科室找出按日期和就诊编号倒序后的第1次就诊，"
            "列出科室、日期和就诊编号。",
            ("department", "visit_date", "visit_id"),
        ),
        (
            "按疾病找出费用等于该疾病最高费用的就诊编号、疾病和费用。",
            ("visit_id", "disease", "cost"),
        ),
        (
            "用公共表表达式给每种疾病的就诊按费用降序编号，"
            "列出编号为1的疾病、就诊编号和费用。",
            ("disease", "visit_id", "cost"),
        ),
    ],
)
def test_partitioned_first_recent_and_argmax_rows_do_not_become_grouped(
    question,
    expected_projections,
):
    contract = extract_semantic_contract(question)

    assert contract.projections == expected_projections
    assert contract.aggregates == ()
    assert contract.grouping == ()
    assert contract.ordering == ()
    assert contract.limit is None


def test_global_latest_row_still_has_global_order_and_limit():
    contract = extract_semantic_contract("返回全局最近一次就诊的就诊编号和日期。")

    assert contract.grouping == ()
    assert contract.ordering == ("visit_date DESC",)
    assert contract.limit == 1


@pytest.mark.parametrize(
    ("question", "expected_projections"),
    [
        (
            "查询关系CODE为HAS_SYMPTOM且头实体类型为Disease的尾实体名称和类型。",
            ("tail", "tail_type"),
        ),
        (
            "查询关系CODE为HAS_CAUSE且尾实体类型为Disease的头实体名称和类型。",
            ("head", "head_type"),
        ),
    ],
)
def test_relation_code_relative_role_filters_return_only_final_entity_fields(
    question,
    expected_projections,
):
    contract = extract_semantic_contract(question)

    assert contract.projections == expected_projections
    assert contract.output_slots == expected_projections
    assert contract.aggregates == ()
    assert contract.grouping == ()
    assert contract.status is ContractStatus.COMPLETE


@pytest.mark.parametrize(
    "question",
    (
        "列出在关系BELONGS_TO_DEPT中担任头实体的实体名称和类型。",
        "列出在关系CONTRAINDICATED_FOR中担任尾实体的实体名称和类型。",
        "列出在关系DIAGNOSED_BY中担任头实体的实体名称和类型。"
        "这项比较用于识别相对整体基准的对象。",
    ),
)
def test_nested_relation_role_is_only_a_filter_for_entity_name_and_type(question):
    contract = extract_semantic_contract(question)

    assert contract.projections == ("name", "type")
    assert contract.output_slots == ("name", "type")
    assert contract.aggregates == ()
    assert contract.grouping == ()
    assert contract.status is ContractStatus.COMPLETE


def test_bare_entity_name_and_type_without_relation_code_keeps_generic_type():
    contract = extract_semantic_contract("返回尾实体名称和类型。")

    assert contract.projections == ("tail", "type")
    assert contract.projections != ("tail", "tail_type")


def test_none_contract_accepts_structurally_safe_select():
    contract = extract_semantic_contract("查询一些记录")
    assert contract.status is ContractStatus.NONE
    result = validate_candidate(contract, "SELECT patient_id FROM patients")
    assert result.accepted is True
    assert result.reasons == ()


def test_rejects_avg_age_when_raw_patient_columns_are_requested():
    contract = extract_semantic_contract("列出患者编号、姓名和年龄")
    result = validate_candidate(
        contract,
        "SELECT patient_id, name, AVG(age) FROM patients",
    )
    assert result.accepted is False
    assert any("projection" in reason for reason in result.reasons)


def test_rejects_wrong_count_distinct_form():
    contract = extract_semantic_contract("返回不同患者数")
    result = validate_candidate(contract, "SELECT COUNT(*) FROM patients")
    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_rejects_aggregate_function_and_measure_mismatch():
    contract = extract_semantic_contract("返回费用平均值")
    result = validate_candidate(contract, "SELECT SUM(age) FROM patients")
    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_accepts_aliases_round_cast_and_outer_select():
    contract = extract_semantic_contract(
        "返回疾病和平均费用；按平均费用降序；前3种。"
    )
    sql = """
        WITH source AS (
            SELECT disease, cost FROM visits
        )
        SELECT
            disease AS diagnosis,
            ROUND(CAST(AVG(cost) AS REAL), 2) AS avg_cost
        FROM source
        GROUP BY disease
        ORDER BY avg_cost DESC
        LIMIT 3
    """
    result = validate_candidate(contract, sql)
    assert result.accepted is True
    assert result.reasons == ()


def test_rejects_missing_group_order_and_limit_clauses():
    contract = extract_semantic_contract("按科室统计就诊次数，返回科室和次数；按次数降序；前3个")
    result = validate_candidate(
        contract,
        "SELECT department, COUNT(*) FROM visits",
    )
    assert result.accepted is False
    assert any("grouping" in reason for reason in result.reasons)
    assert any("ordering" in reason for reason in result.reasons)
    assert any("limit" in reason for reason in result.reasons)


def test_accepts_harmless_deterministic_tie_suffix():
    contract = extract_semantic_contract("按科室统计就诊次数；前3个")
    result = validate_candidate(
        contract,
        """
        SELECT department, COUNT(*) AS n
        FROM visits
        GROUP BY department
        ORDER BY n DESC, department ASC
        LIMIT 3
        """,
    )
    assert result.accepted is True


def test_rejects_projection_count_and_semantics():
    contract = extract_semantic_contract("显示患者编号、姓名和年龄")
    result = validate_candidate(contract, "SELECT patient_id, age FROM patients")
    assert result.accepted is False
    assert any("count mismatch" in reason for reason in result.reasons)
    assert any("semantic mismatch" in reason for reason in result.reasons)


def test_rejects_wrong_grouping_order_direction_and_limit():
    contract = extract_semantic_contract("按科室统计就诊次数；按次数降序；前3个")
    result = validate_candidate(
        contract,
        """
        SELECT disease, COUNT(*) AS n
        FROM visits
        GROUP BY disease
        ORDER BY n ASC
        LIMIT 4
        """,
    )
    assert result.accepted is False
    assert any("grouping" in reason for reason in result.reasons)
    assert any("ordering" in reason for reason in result.reasons)
    assert any("limit" in reason for reason in result.reasons)


def test_rejects_parse_failure_and_non_select_candidates():
    contract = extract_semantic_contract("返回患者编号")
    assert not validate_candidate(contract, "SELECT FROM").accepted
    assert not validate_candidate(contract, "DELETE FROM patients").accepted
    assert not validate_candidate(contract, "not SQL at all").accepted


def test_non_strict_partial_contract_allows_omitted_optional_clauses():
    contract = SemanticContract(
        projections=("department",),
        aggregates=("COUNT(*)",),
        grouping=("department",),
        ordering=("COUNT(*) DESC",),
        limit=3,
        confidence=0.55,
        status=ContractStatus.PARTIAL,
    )
    result = validate_candidate(
        contract,
        "SELECT department, COUNT(*) FROM visits GROUP BY department",
        strict=False,
    )
    assert result.accepted is True


def test_case_bucket_alias_traces_to_its_single_source_column():
    contract = SemanticContract(
        projections=("age",),
        aggregates=("AVG(cost)",),
        grouping=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    sql = """
        SELECT
            CASE
                WHEN p.age <= 18 THEN '0-18'
                WHEN p.age <= 40 THEN '19-40'
                WHEN p.age <= 60 THEN '41-60'
                ELSE '60+'
            END AS age_group,
            ROUND(AVG(v.cost), 2) AS avg_value
        FROM visits AS v
        JOIN patients AS p ON v.patient_id = p.patient_id
        GROUP BY CASE
            WHEN p.age <= 18 THEN '0-18'
            WHEN p.age <= 40 THEN '19-40'
            WHEN p.age <= 60 THEN '41-60'
            ELSE '60+'
        END
        ORDER BY age_group
    """

    result = validate_candidate(contract, sql)
    assert result.accepted is True


def test_age_band_alias_uses_the_same_strict_age_case_lineage():
    contract = SemanticContract(
        projections=("age",),
        aggregates=("COUNT(*)",),
        grouping=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )

    result = validate_candidate(
        contract,
        "SELECT CASE WHEN age < 18 THEN 'minor' ELSE 'adult' END AS age_band, "
        "COUNT(*) AS n FROM patients GROUP BY age_band",
    )

    assert result.accepted is True


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT CASE WHEN cost < 18 THEN 'minor' ELSE 'adult' END AS age_band, "
        "COUNT(*) FROM visits GROUP BY age_band",
        "SELECT CASE WHEN ROUND(age) < 18 THEN 'minor' ELSE 'adult' END "
        "AS age_band, COUNT(*) FROM patients GROUP BY age_band",
        "SELECT CASE WHEN age + 1 < 18 THEN 'minor' ELSE 'adult' END "
        "AS age_band, COUNT(*) FROM patients GROUP BY age_band",
        "SELECT CASE WHEN age < 18 OR age IS NULL THEN 'minor' ELSE 'adult' END "
        "AS age_band, COUNT(*) FROM patients GROUP BY age_band",
        "SELECT CASE WHEN age < 18 THEN 'same' ELSE 'same' END AS age_band, "
        "COUNT(*) FROM patients GROUP BY age_band",
        "SELECT CASE WHEN age < 18 THEN 'minor' ELSE 'adult' END AS age_bucket, "
        "COUNT(*) FROM patients GROUP BY age_bucket",
        "WITH source AS (SELECT age, region AS age_band FROM patients) "
        "SELECT CASE WHEN age < 18 THEN 'minor' ELSE 'adult' END AS age_band, "
        "COUNT(*) FROM source GROUP BY age_band",
    ),
)
def test_age_band_alias_does_not_broaden_the_case_bucket_proof(sql):
    contract = SemanticContract(
        projections=("age",),
        aggregates=("COUNT(*)",),
        grouping=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )

    assert validate_candidate(contract, sql).accepted is False


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT CASE WHEN age < 1000 THEN 'low' ELSE 'high' END AS cost_band, "
        "COUNT(*) FROM patients GROUP BY cost_band",
        "SELECT CASE WHEN los_days < 10 THEN 'low' ELSE 'high' END AS cost_band, "
        "COUNT(*) FROM visits GROUP BY cost_band",
        "SELECT CASE WHEN ROUND(cost) < 1000 THEN 'low' ELSE 'high' END "
        "AS cost_band, COUNT(*) FROM visits GROUP BY cost_band",
        "SELECT CASE WHEN cost + 1 < 1000 THEN 'low' ELSE 'high' END "
        "AS cost_band, COUNT(*) FROM visits GROUP BY cost_band",
        "SELECT CASE WHEN cost < 1000 OR cost IS NULL THEN 'low' ELSE 'high' END "
        "AS cost_band, COUNT(*) FROM visits GROUP BY cost_band",
        "SELECT CASE WHEN cost < 1000 THEN 'same' ELSE 'same' END AS cost_band, "
        "COUNT(*) FROM visits GROUP BY cost_band",
        "SELECT CASE WHEN cost < 1000 THEN 0 ELSE 1 END AS cost_band, "
        "COUNT(*) FROM visits GROUP BY cost_band",
        "SELECT CASE WHEN cost < 1000 THEN 'low' ELSE 'high' END AS expense_band, "
        "COUNT(*) FROM visits GROUP BY expense_band",
        "SELECT CASE WHEN cost < 1000 THEN 'low' ELSE 'high' END, COUNT(*) "
        "FROM visits GROUP BY CASE WHEN cost < 1000 THEN 'low' ELSE 'high' END",
        "WITH source AS (SELECT cost, cost AS cost_band FROM visits) "
        "SELECT CASE WHEN cost < 1000 THEN 'low' ELSE 'high' END AS cost_band, "
        "COUNT(*) FROM source GROUP BY cost_band",
        "WITH source AS (SELECT cost, CASE WHEN cost < 1000 THEN 'low' "
        "ELSE 'high' END AS cost_band FROM visits GROUP BY cost, cost_band) "
        "SELECT cost_band, COUNT(*) FROM source GROUP BY cost",
        "WITH source AS (SELECT CASE WHEN cost < 1000 THEN 'low' ELSE 'high' END "
        "AS cost_band FROM visits GROUP BY cost_band) "
        "SELECT CASE WHEN cost_band < 1 THEN 'low' ELSE 'high' END AS cost_band, "
        "COUNT(*) FROM source GROUP BY cost_band",
    ),
)
def test_cost_band_case_lineage_rejects_non_direct_or_shadowed_shapes(sql):
    contract = SemanticContract(
        projections=("cost",),
        aggregates=("COUNT(*)",),
        grouping=("cost",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )

    assert validate_candidate(contract, sql).accepted is False


def test_ungrouped_cost_band_cannot_impersonate_raw_cost():
    contract = SemanticContract(
        projections=("cost",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )

    result = validate_candidate(
        contract,
        "SELECT CASE WHEN cost < 1000 THEN 'low' ELSE 'high' END AS cost_band "
        "FROM visits",
    )

    assert result.accepted is False
    assert any("cost bucket" in reason for reason in result.reasons)


def test_case_bucket_alias_cannot_spoof_a_different_source_column():
    contract = SemanticContract(
        projections=("age",),
        grouping=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    sql = """
        SELECT
            CASE WHEN v.cost < 100 THEN 'low' ELSE 'high' END AS age_group,
            COUNT(*) AS n
        FROM visits AS v
        GROUP BY CASE WHEN v.cost < 100 THEN 'low' ELSE 'high' END
        ORDER BY age_group
    """

    result = validate_candidate(contract, sql)
    assert result.accepted is False
    assert any(
        "projection" in reason or "grouping" in reason or "source column" in reason
        for reason in result.reasons
    )


def test_case_bucket_rejects_a_tautological_branch_before_age_comparison():
    contract = SemanticContract(
        projections=("age",),
        aggregates=("AVG(cost)",),
        grouping=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT
            CASE
                WHEN 1 = 1 THEN 'all'
                WHEN p.age <= 18 THEN 'child'
                ELSE 'adult'
            END AS age_group,
            AVG(v.cost) AS avg_cost
        FROM visits AS v
        JOIN patients AS p ON v.patient_id = p.patient_id
        GROUP BY CASE
            WHEN 1 = 1 THEN 'all'
            WHEN p.age <= 18 THEN 'child'
            ELSE 'adult'
        END
        """,
    )

    assert result.accepted is False
    assert any("projection" in reason or "grouping" in reason for reason in result.reasons)


def test_case_bucket_rejects_an_age_self_comparison():
    contract = SemanticContract(
        projections=("age",),
        aggregates=("AVG(cost)",),
        grouping=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT
            CASE WHEN p.age IS p.age THEN 'all' ELSE 'other' END AS age_group,
            AVG(v.cost) AS avg_cost
        FROM visits AS v
        JOIN patients AS p ON v.patient_id = p.patient_id
        GROUP BY CASE WHEN p.age IS p.age THEN 'all' ELSE 'other' END
        """,
    )

    assert result.accepted is False
    assert any("projection" in reason or "grouping" in reason for reason in result.reasons)


def test_case_bucket_rejects_a_compound_tautological_age_condition():
    contract = SemanticContract(
        projections=("age",),
        aggregates=("AVG(cost)",),
        grouping=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT
            CASE WHEN p.age <= 18 OR 1 = 1 THEN 'all' ELSE 'other' END AS age_group,
            AVG(v.cost) AS avg_cost
        FROM visits AS v
        JOIN patients AS p ON v.patient_id = p.patient_id
        GROUP BY CASE WHEN p.age <= 18 OR 1 = 1 THEN 'all' ELSE 'other' END
        """,
    )

    assert result.accepted is False
    assert any("projection" in reason or "grouping" in reason for reason in result.reasons)


def test_case_bucket_requires_unique_string_labels():
    contract = SemanticContract(
        projections=("age",),
        aggregates=("AVG(cost)",),
        grouping=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT
            CASE
                WHEN p.age <= 18 THEN 'child'
                WHEN p.age <= 40 THEN 'adult'
                ELSE 'adult'
            END AS age_group,
            AVG(v.cost) AS avg_cost
        FROM visits AS v
        JOIN patients AS p ON v.patient_id = p.patient_id
        GROUP BY CASE
            WHEN p.age <= 18 THEN 'child'
            WHEN p.age <= 40 THEN 'adult'
            ELSE 'adult'
        END
        """,
    )

    assert result.accepted is False
    assert any("projection" in reason or "grouping" in reason for reason in result.reasons)


def test_constant_case_cannot_impersonate_a_raw_age_projection():
    contract = SemanticContract(
        projections=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        "SELECT CASE WHEN age IS NULL THEN 0 ELSE 0 END AS age FROM patients",
    )

    assert result.accepted is False
    assert any("projection" in reason for reason in result.reasons)


def test_numerically_equivalent_case_labels_cannot_create_a_fake_age_bucket():
    contract = SemanticContract(
        projections=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        "SELECT CASE WHEN age IS NULL THEN 0 ELSE 0.0 END AS age_group FROM patients",
    )

    assert result.accepted is False
    assert any("projection" in reason for reason in result.reasons)


def test_constant_case_cannot_impersonate_an_age_group_dimension():
    contract = SemanticContract(
        projections=("age",),
        aggregates=("COUNT(*)",),
        grouping=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT CASE WHEN age IS NULL THEN 'x' ELSE 'x' END AS age_group, COUNT(*) AS n
        FROM patients
        GROUP BY CASE WHEN age IS NULL THEN 'x' ELSE 'x' END
        """,
    )

    assert result.accepted is False
    assert any("projection" in reason or "grouping" in reason for reason in result.reasons)


def test_case_bucket_requires_the_explicit_age_group_alias():
    contract = SemanticContract(
        projections=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT CASE WHEN age <= 18 THEN '0-18' ELSE '19+' END AS age
        FROM patients
        """,
    )

    assert result.accepted is False
    assert any("projection" in reason for reason in result.reasons)


def test_safe_group_by_age_bucket_alias_resolves_to_the_projected_dimension():
    contract = SemanticContract(
        projections=("age",),
        aggregates=("COUNT(*)",),
        grouping=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT CASE WHEN age <= 18 THEN '0-18' ELSE '19+' END AS age_group, COUNT(*) AS n
        FROM patients
        GROUP BY age_group
        """,
    )

    assert result.accepted is True


def test_cte_column_list_names_an_unaliased_age_bucket_before_lineage_validation():
    contract = SemanticContract(
        projections=("age",),
        aggregates=("COUNT(*)",),
        grouping=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH bucket(age_group) AS (
            SELECT CASE WHEN age <= 18 THEN '0-18' ELSE '19+' END
            FROM patients
        )
        SELECT age_group, COUNT(*) AS n
        FROM bucket
        GROUP BY age_group
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_group_by_alias_does_not_override_a_visible_source_column():
    contract = SemanticContract(
        projections=("age",),
        aggregates=("COUNT(*)",),
        grouping=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH source AS (SELECT age, region AS age_group FROM patients)
        SELECT age AS age_group, COUNT(*) AS n
        FROM source
        GROUP BY age_group
        """,
    )

    assert result.accepted is False
    assert any("grouping" in reason for reason in result.reasons)


def test_quoted_dotted_identifier_is_not_a_qualified_source_column():
    contract = SemanticContract(
        projections=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(contract, 'SELECT "patients.age" FROM patients')

    assert result.accepted is False


def test_quoted_dotted_relation_is_not_an_allowed_base_source():
    contract = SemanticContract(
        confidence=0.0,
        status=ContractStatus.NONE,
    )
    result = validate_candidate(
        contract,
        'SELECT patient_id FROM "foreign.patients"',
    )

    assert result.accepted is False
    assert "quoted dotted source relation is not supported" in result.reasons


def test_quoted_dotted_cte_name_does_not_create_a_bare_source_alias():
    result = validate_candidate(
        SemanticContract(confidence=0.0, status=ContractStatus.NONE),
        '''
        WITH "foreign.secret"(patient_id) AS (
            SELECT patient_id FROM patients
        )
        SELECT patient_id FROM secret
        ''',
    )

    assert result.accepted is False
    assert "forbidden source relation: secret" in result.reasons


@pytest.mark.parametrize(
    ("sql", "reason"),
    (
        (
            "SELECT patient_id FROM patientſ",
            "forbidden source relation: patientſ",
        ),
        (
            "SELECT loſ_days FROM visits",
            "unknown or ambiguous source column: loſ_days",
        ),
        (
            "SELECT patients.patient_id FROM patients AS patientſ",
            "unknown or ambiguous source column: patients.patient_id",
        ),
    ),
)
def test_sqlite_identifier_resolution_does_not_use_unicode_casefold(sql, reason):
    result = validate_candidate(
        SemanticContract(confidence=0.0, status=ContractStatus.NONE),
        sql,
    )

    assert result.accepted is False
    assert reason in result.reasons


def test_unicode_cte_name_is_not_an_ascii_base_relation_shadow():
    result = validate_candidate(
        SemanticContract(confidence=0.0, status=ContractStatus.NONE),
        """
        WITH patientſ(patient_id) AS (
            SELECT patient_id FROM patients
        )
        SELECT patient_id FROM patientſ
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_ascii_identifier_case_remains_supported():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(contract, "SELECT PATIENT_ID FROM PATIENTS")

    assert result.accepted is True
    assert result.reasons == ()


@pytest.mark.parametrize(
    ("sql", "reason"),
    (
        (
            'SELECT patient_id FROM "patients "',
            "forbidden source relation: patients ",
        ),
        (
            'SELECT patient_id FROM "patients]"',
            "forbidden source relation: patients]",
        ),
        (
            'SELECT "patient_id " FROM patients',
            "unknown or ambiguous source column: patient_id ",
        ),
        (
            'SELECT patients.patient_id FROM patients AS "patients "',
            "unknown or ambiguous source column: patients.patient_id",
        ),
        (
            '''
            WITH source("patient_id ") AS (
                SELECT patient_id FROM patients
            )
            SELECT patient_id FROM source
            ''',
            "unknown or ambiguous source column: patient_id",
        ),
    ),
)
def test_sqlite_identifier_resolution_preserves_quoted_contents(sql, reason):
    result = validate_candidate(
        SemanticContract(confidence=0.0, status=ContractStatus.NONE),
        sql,
    )

    assert result.accepted is False
    assert reason in result.reasons


def test_quoted_identifiers_preserve_contents_and_ascii_case_matching():
    result = validate_candidate(
        SemanticContract(confidence=0.0, status=ContractStatus.NONE),
        '''
        WITH source("patient_id ") AS (
            SELECT "PATIENT_ID" FROM "PATIENTS"
        )
        SELECT "patient_id " FROM source AS "source"
        ''',
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_raw_transformation_cannot_impersonate_a_requested_column():
    contract = SemanticContract(
        projections=("name",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(contract, "SELECT ROUND(name) FROM patients")

    assert result.accepted is False
    assert any("projection" in reason for reason in result.reasons)


def test_string_literals_cannot_impersonate_group_or_order_ordinals():
    contract = SemanticContract(
        projections=("department",),
        aggregates=("COUNT(*)",),
        grouping=("department",),
        ordering=("COUNT(*) DESC",),
        limit=3,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT department, COUNT(*) AS n
        FROM visits
        GROUP BY '1'
        ORDER BY '2' DESC
        LIMIT 3
        """,
    )

    assert result.accepted is False
    assert any("grouping" in reason for reason in result.reasons)
    assert any("ordering" in reason for reason in result.reasons)


def test_windowed_aggregate_cannot_impersonate_an_ordinary_aggregate():
    contract = SemanticContract(
        aggregates=("COUNT(*)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(contract, "SELECT COUNT(*) OVER () AS n FROM visits")

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_unsupported_calendar_format_cannot_impersonate_a_month_dimension():
    contract = SemanticContract(
        projections=("month",),
        aggregates=("COUNT(*)",),
        grouping=("month",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT strftime('%d', visit_date) AS month, COUNT(*) AS n
        FROM visits
        GROUP BY strftime('%d', visit_date)
        """,
    )

    assert result.accepted is False
    assert any("projection" in reason or "grouping" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    ("expression", "signature"),
    (
        ("SUBSTR(visit_date, 1, 7)", "month"),
        ("SUBSTR(visit_date, 1, 4)", "year"),
    ),
)
def test_exact_substr_calendar_prefix_preserves_calendar_lineage(expression, signature):
    contract = SemanticContract(
        projections=(signature,),
        aggregates=("COUNT(*)",),
        grouping=(signature,),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"SELECT {expression} AS {signature}, COUNT(*) AS n "
        f"FROM visits GROUP BY {expression}",
    )

    assert result.accepted is True
    assert result.reasons == ()


@pytest.mark.parametrize(
    "expression",
    (
        "SUBSTR(visit_date, 2, 7)",
        "SUBSTR(visit_date, 1, 6)",
        "SUBSTR(disease, 1, 7)",
        "SUBSTR(visit_date, 1 + 0, 7)",
    ),
)
def test_noncanonical_substr_cannot_impersonate_a_month_dimension(expression):
    contract = SemanticContract(
        projections=("month",),
        aggregates=("COUNT(*)",),
        grouping=("month",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"SELECT {expression} AS month, COUNT(*) AS n "
        f"FROM visits GROUP BY {expression}",
    )

    assert result.accepted is False
    assert any("projection" in reason or "grouping" in reason for reason in result.reasons)


def test_single_unnamed_derived_source_keeps_unqualified_grouped_lineage():
    contract = SemanticContract(
        projections=("relation",),
        aggregates=("COUNT(*)",),
        grouping=("relation",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT relation, cnt
        FROM (
            SELECT relation, COUNT(*) AS cnt
            FROM kg_triples
            GROUP BY relation
        )
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_multiple_unnamed_derived_sources_remain_rejected():
    result = validate_candidate(
        SemanticContract(confidence=0.0, status=ContractStatus.NONE),
        """
        SELECT patient_id
        FROM (SELECT patient_id FROM patients)
        JOIN (SELECT patient_id FROM visits) USING (patient_id)
        """,
    )

    assert result.accepted is False
    assert "derived source requires an alias" in result.reasons


def test_internal_unnamed_derived_key_is_not_a_visible_sql_qualifier():
    result = validate_candidate(
        SemanticContract(confidence=0.0, status=ContractStatus.NONE),
        'SELECT "unnamed-derived".patient_id '
        "FROM (SELECT patient_id FROM patients)",
    )

    assert result.accepted is False
    assert any("unknown or ambiguous source column" in reason for reason in result.reasons)


def test_aggregate_of_aggregate_cte_cannot_impersonate_a_base_average():
    contract = SemanticContract(
        aggregates=("AVG(cost)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH grouped AS (
            SELECT department, AVG(cost) AS cost
            FROM visits
            GROUP BY department
        )
        SELECT AVG(cost) AS avg_cost FROM grouped
        """,
    )

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_literal_only_cte_cannot_supply_a_count_contract():
    contract = SemanticContract(
        aggregates=("COUNT(*)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        "WITH fake AS (SELECT 0 AS x) SELECT COUNT(*) AS n FROM fake",
    )

    assert result.accepted is False
    assert result.reasons == ("candidate has no allowed base source",)


def test_select_list_order_cannot_be_laundered_by_separate_projection_buckets():
    contract = SemanticContract(
        projections=("department",),
        aggregates=("COUNT(*)",),
        grouping=("department",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        "SELECT COUNT(*) AS n, department FROM visits GROUP BY department",
    )

    assert result.accepted is False
    assert any("select-list order" in reason for reason in result.reasons)


def test_top_k_offset_is_not_equivalent_to_the_requested_prefix():
    contract = SemanticContract(
        projections=("department",),
        aggregates=("COUNT(*)",),
        grouping=("department",),
        ordering=("COUNT(*) DESC",),
        limit=3,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT department, COUNT(*) AS n
        FROM visits
        GROUP BY department
        ORDER BY n DESC
        LIMIT 3 OFFSET 1
        """,
    )

    assert result.accepted is False
    assert any("offset" in reason for reason in result.reasons)


def test_active_contract_rejects_an_outer_offset_without_a_top_k_request():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        "SELECT patient_id FROM patients LIMIT -1 OFFSET 1",
    )

    assert result.accepted is False
    assert any("offset" in reason for reason in result.reasons)


def test_active_contract_rejects_an_unrequested_outer_limit():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(contract, "SELECT patient_id FROM patients LIMIT 1")

    assert result.accepted is False
    assert "unexpected limit clause" in result.reasons


def test_active_contract_rejects_an_unrequested_random_outer_sample():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        "SELECT patient_id FROM patients ORDER BY RANDOM() LIMIT 1",
    )

    assert result.accepted is False
    assert "unexpected limit clause" in result.reasons
    assert "nondeterministic ordering controls a top-k limit" in result.reasons


def test_active_contract_rejects_a_nested_offset():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH skipped AS (
            SELECT patient_id FROM patients LIMIT -1 OFFSET 1
        )
        SELECT patient_id FROM skipped
        """,
    )

    assert result.accepted is False
    assert any("offset" in reason for reason in result.reasons)


def test_active_contract_rejects_a_nested_limit():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH sample AS (
            SELECT patient_id FROM patients LIMIT 10
        )
        SELECT patient_id FROM sample
        """,
    )

    assert result.accepted is False
    assert any("nested limit" in reason for reason in result.reasons)


def test_active_contract_rejects_nondeterministic_nested_sampling():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH sample AS (
            SELECT patient_id FROM patients ORDER BY RANDOM() LIMIT 10
        )
        SELECT patient_id FROM sample
        """,
    )

    assert result.accepted is False
    assert any("nested limit" in reason for reason in result.reasons)
    assert any("nondeterministic" in reason for reason in result.reasons)


def test_active_contract_rejects_volatile_scalar_subquery_ordering():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id = (
            SELECT patient_id
            FROM patients
            ORDER BY RANDOM()
        )
        """,
    )

    assert result.accepted is False
    assert "nondeterministic ordering controls a scalar subquery" in result.reasons


def test_active_contract_rejects_unresolved_scalar_subquery_ordering():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id = (
            SELECT patient_id
            FROM patients
            ORDER BY 2
        )
        """,
    )

    assert result.accepted is False
    assert "ordered scalar-subquery provenance cannot be established" in result.reasons


def test_active_contract_allows_deterministic_scalar_subquery_ordering():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id = (
            SELECT patient_id
            FROM patients
            ORDER BY patient_id ASC
        )
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_active_contract_does_not_treat_an_in_subquery_as_scalar():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id IN (
            SELECT patient_id
            FROM patients
            ORDER BY RANDOM()
        )
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


@pytest.mark.parametrize(
    ("rank_expression", "window_clause"),
    (
        (
            "ROW_NUMBER() OVER (PARTITION BY department ORDER BY RANDOM())",
            "",
        ),
        (
            "ROW_NUMBER() OVER ranked_window",
            "WINDOW ranked_window AS (PARTITION BY department ORDER BY RANDOM())",
        ),
    ),
)
def test_active_contract_rejects_volatile_window_rank_membership(
    rank_expression,
    window_clause,
):
    contract = SemanticContract(
        projections=("patient_id",),
        limit=5,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        WITH ranked AS (
            SELECT patient_id, {rank_expression} AS visit_rank
            FROM visits
            {window_clause}
        )
        SELECT patient_id
        FROM ranked
        WHERE visit_rank <= 5
        LIMIT 5
        """,
    )

    assert result.accepted is False
    assert "nondeterministic window ordering is not permitted for an active semantic contract" in result.reasons


def test_active_contract_rejects_volatile_window_rank_source_alias():
    contract = SemanticContract(
        projections=("patient_id",),
        limit=5,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH seeded AS (
            SELECT patient_id, department, RANDOM() AS rank_tie
            FROM visits
        ), ranked AS (
            SELECT patient_id,
                   ROW_NUMBER() OVER (PARTITION BY department ORDER BY rank_tie) AS visit_rank
            FROM seeded
        )
        SELECT patient_id
        FROM ranked
        WHERE visit_rank <= 5
        LIMIT 5
        """,
    )

    assert result.accepted is False
    assert "nondeterministic window ordering is not permitted for an active semantic contract" in result.reasons


def test_active_contract_allows_deterministic_window_rank_membership():
    contract = SemanticContract(
        projections=("patient_id",),
        limit=5,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH ranked AS (
            SELECT patient_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY department ORDER BY visit_date ASC, visit_id ASC
                   ) AS visit_rank
            FROM visits
        )
        SELECT patient_id
        FROM ranked
        WHERE visit_rank <= 5
        LIMIT 5
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


@pytest.mark.parametrize("operator", ("UNION", "INTERSECT", "EXCEPT"))
def test_active_contract_rejects_set_operation_root_paging(operator):
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT patient_id
        FROM patients AS p
        WHERE p.patient_id IN (
            SELECT a.patient_id FROM patients AS a
            {operator}
            SELECT b.patient_id FROM patients AS b
            ORDER BY 1
            LIMIT 1 OFFSET 1
        )
        """,
    )

    assert result.accepted is False
    assert any("offset" in reason for reason in result.reasons)
    assert any("nested limit" in reason for reason in result.reasons)


@pytest.mark.parametrize("operator", ("UNION", "INTERSECT", "EXCEPT"))
def test_nested_compound_cte_scope_reaches_each_select_leaf(operator):
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id IN (
            WITH scoped(id) AS (
                SELECT patient_id
                FROM patients
            )
            SELECT id
            FROM scoped
            {operator}
            SELECT id
            FROM scoped
        )
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


@pytest.mark.parametrize("operator", ("UNION", "INTERSECT", "EXCEPT"))
def test_nested_compound_cte_cannot_shadow_a_base_relation(operator):
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id IN (
            WITH patients AS (
                SELECT patient_id
                FROM visits
            )
            SELECT patient_id
            FROM patients
            {operator}
            SELECT patient_id
            FROM patients
        )
        """,
    )

    assert result.accepted is False
    assert "CTE shadows base relation: patients" in result.reasons


@pytest.mark.parametrize("operator", ("UNION", "INTERSECT", "EXCEPT"))
def test_nested_compound_pending_cte_name_masks_an_outer_cte(operator):
    contract = SemanticContract(
        confidence=0.0,
        status=ContractStatus.NONE,
    )
    result = validate_candidate(
        contract,
        f"""
        WITH later(id) AS (
            SELECT patient_id
            FROM patients
        )
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id IN (
            WITH earlier(id) AS (
                SELECT id
                FROM later
            ),
            later(id) AS (
                SELECT patient_id
                FROM patients
            )
            SELECT id
            FROM earlier
            {operator}
            SELECT id
            FROM earlier
        )
        """,
    )

    assert result.accepted is False
    assert "forward or cyclic CTE reference: later" in result.reasons


@pytest.mark.parametrize("operator", ("UNION", "INTERSECT", "EXCEPT"))
def test_nested_compound_recursive_ctes_are_rejected_at_the_query_root(operator):
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id IN (
            WITH RECURSIVE scoped(id) AS (
                SELECT patient_id
                FROM patients
            )
            SELECT id
            FROM scoped
            {operator}
            SELECT id
            FROM scoped
        )
        """,
    )

    assert result.accepted is False
    assert "recursive CTEs are not supported" in result.reasons


@pytest.mark.parametrize("operator", ("UNION", "INTERSECT", "EXCEPT"))
@pytest.mark.parametrize(
    "root_clause",
    (
        "ORDER BY (WITH RECURSIVE scoped(id) AS (SELECT patient_id FROM patients) SELECT 1)",
        "LIMIT (WITH RECURSIVE scoped(id) AS (SELECT patient_id FROM patients) SELECT 1)",
        "LIMIT -1 OFFSET (WITH RECURSIVE scoped(id) AS (SELECT patient_id FROM patients) SELECT 1)",
    ),
)
def test_nested_compound_root_clauses_audit_scalar_query_roots(operator, root_clause):
    contract = SemanticContract(
        confidence=0.0,
        status=ContractStatus.NONE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id IN (
            SELECT a.patient_id
            FROM patients AS a
            {operator}
            SELECT b.patient_id
            FROM patients AS b
            {root_clause}
        )
        """,
    )

    assert result.accepted is False
    assert "recursive CTEs are not supported" in result.reasons


@pytest.mark.parametrize("operator", ("UNION", "INTERSECT", "EXCEPT"))
def test_nested_compound_values_rows_audit_scalar_query_roots(operator):
    contract = SemanticContract(
        confidence=0.0,
        status=ContractStatus.NONE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id IN (
            (VALUES ((
                WITH RECURSIVE scoped(id) AS (
                    SELECT patient_id
                    FROM patients
                )
                SELECT 1
            )))
            {operator}
            (SELECT b.patient_id FROM patients AS b)
        )
        """,
    )

    assert result.accepted is False
    assert "recursive CTEs are not supported" in result.reasons


@pytest.mark.parametrize("operator", ("UNION", "INTERSECT", "EXCEPT"))
def test_nested_compound_values_rows_audit_scalar_query_sources(operator):
    contract = SemanticContract(
        confidence=0.0,
        status=ContractStatus.NONE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id IN (
            (VALUES ((SELECT patient_id FROM forbidden_relation)))
            {operator}
            (SELECT b.patient_id FROM patients AS b)
        )
        """,
    )

    assert result.accepted is False
    assert "forbidden source relation: forbidden_relation" in result.reasons


@pytest.mark.parametrize(
    "with_clause",
    (
        "WITH patients(x) AS (SELECT 1 AS x)",
        "WITH RECURSIVE scoped(x) AS (SELECT 1 AS x)",
    ),
)
def test_nested_values_root_cannot_bypass_cte_policy(with_clause):
    result = validate_candidate(
        SemanticContract(confidence=0.0, status=ContractStatus.NONE),
        f"""
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id IN (
            {with_clause}
            VALUES (1)
        )
        """,
    )

    # Current SQLGlot releases reject WITH ... VALUES during parsing.  If parser
    # support changes, the AST walker must still reject the base shadow or
    # recursive CTE instead of silently falling through the Values root.
    assert result.accepted is False


@pytest.mark.parametrize("operator", ("UNION", "INTERSECT", "EXCEPT"))
def test_active_contract_rejects_compound_ordinal_volatile_order_beneath_outer_limit(operator):
    contract = extract_semantic_contract("列出前5名患者编号")
    result = validate_candidate(
        contract,
        f"""
        SELECT p.patient_id
        FROM patients AS p
        WHERE (p.patient_id, 0) IN (
            SELECT a.patient_id, RANDOM() AS tie
            FROM patients AS a
            {operator}
            SELECT b.patient_id, RANDOM() AS tie
            FROM patients AS b
            ORDER BY 2
        )
        LIMIT 5
        """,
    )

    assert result.accepted is False
    assert any("nondeterministic ordering controls an outer limit" in reason for reason in result.reasons)


def test_active_contract_rejects_volatile_inner_order_feeding_outer_limit():
    contract = extract_semantic_contract("列出前5名患者编号")
    result = validate_candidate(
        contract,
        """
        SELECT patient_id
        FROM (
            SELECT patient_id, RANDOM() AS tie
            FROM patients
            ORDER BY tie
        ) AS sample
        LIMIT 5
        """,
    )

    assert result.accepted is False
    assert any("nondeterministic" in reason for reason in result.reasons)


def test_active_contract_rejects_ordinal_volatile_inner_order_feeding_outer_limit():
    contract = extract_semantic_contract("列出前5名患者编号")
    result = validate_candidate(
        contract,
        """
        SELECT patient_id
        FROM (
            SELECT patient_id, RANDOM() AS tie
            FROM patients
            ORDER BY 2
        ) AS sample
        LIMIT 5
        """,
    )

    assert result.accepted is False
    assert any("nondeterministic" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "order_expression",
    (
        "+2",
        "+(2)",
        "+2 COLLATE NOCASE",
        "2 COLLATE NOCASE",
        "(2) COLLATE NOCASE",
        "0x2",
        "0x2 COLLATE NOCASE",
        "-(-2) COLLATE NOCASE",
        "-(-0x2) COLLATE NOCASE",
        "likely(2) COLLATE NOCASE",
        "likely(+2) COLLATE NOCASE",
        "unlikely(2) COLLATE NOCASE",
        "likelihood(2, 0.5) COLLATE NOCASE",
    ),
)
def test_active_contract_rejects_wrapped_or_hex_volatile_inner_ordinal(order_expression):
    contract = extract_semantic_contract("列出前5名患者编号")
    result = validate_candidate(
        contract,
        f"""
        SELECT patient_id
        FROM (
            SELECT patient_id, RANDOM() AS tie
            FROM patients
            ORDER BY {order_expression}
        ) AS sample
        LIMIT 5
        """,
    )

    assert result.accepted is False
    assert any("nondeterministic ordering controls an outer limit" in reason for reason in result.reasons)


def test_large_sqlite_integer_order_term_is_not_a_positional_slot():
    contract = extract_semantic_contract("列出前5名患者编号")
    result = validate_candidate(
        contract,
        """
        SELECT patient_id
        FROM (
            SELECT patient_id, RANDOM() AS tie
            FROM patients
            ORDER BY 2147483648
        ) AS sample
        LIMIT 5
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_out_of_range_numeric_order_slot_is_not_silently_ignored():
    contract = extract_semantic_contract("列出前5名患者编号")
    result = validate_candidate(
        contract,
        """
        SELECT patient_id
        FROM (
            SELECT patient_id, RANDOM() AS tie
            FROM patients
            ORDER BY 0
        ) AS sample
        LIMIT 5
        """,
    )

    assert result.accepted is False
    assert any("ordered paging provenance cannot be established" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    ("group_expression", "order_expression"),
    (
        ("likely(1)", "-(-2) DESC"),
        ("likelihood(1, 0.5)", "unlikely(2) DESC"),
    ),
)
def test_sqlite_transparent_wrapped_ordinals_match_grouping_and_ordering(
    group_expression,
    order_expression,
):
    contract = SemanticContract(
        projections=("department",),
        aggregates=("COUNT(*)",),
        grouping=("department",),
        ordering=("COUNT(*) DESC",),
        limit=3,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT department, COUNT(*) AS total
        FROM visits
        GROUP BY {group_expression}
        ORDER BY {order_expression}
        LIMIT 3
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


@pytest.mark.parametrize("operator", ("UNION", "INTERSECT", "EXCEPT"))
def test_active_contract_checks_the_right_compound_output_slot_for_volatility(operator):
    contract = extract_semantic_contract("列出前5名患者编号")
    result = validate_candidate(
        contract,
        f"""
        SELECT p.patient_id
        FROM patients AS p
        WHERE (p.patient_id, 0) IN (
            SELECT a.patient_id, 0 AS stable_tie
            FROM patients AS a
            {operator}
            SELECT b.patient_id, RANDOM() AS random_tie
            FROM patients AS b
            ORDER BY 2
        )
        LIMIT 5
        """,
    )

    assert result.accepted is False
    assert any("nondeterministic ordering controls an outer limit" in reason for reason in result.reasons)


def test_active_contract_checks_compound_alias_against_every_output_branch():
    contract = extract_semantic_contract("列出前5名患者编号")
    result = validate_candidate(
        contract,
        """
        SELECT p.patient_id
        FROM patients AS p
        WHERE (p.patient_id, 0) IN (
            SELECT a.patient_id, 0 AS stable_tie
            FROM patients AS a
            UNION
            SELECT b.patient_id, RANDOM() AS random_tie
            FROM patients AS b
            ORDER BY stable_tie
        )
        LIMIT 5
        """,
    )

    assert result.accepted is False
    assert any("nondeterministic ordering controls an outer limit" in reason for reason in result.reasons)


def test_active_contract_checks_a_later_compound_alias_for_volatility():
    contract = extract_semantic_contract("列出前5名患者编号")
    result = validate_candidate(
        contract,
        """
        SELECT p.patient_id
        FROM patients AS p
        WHERE (p.patient_id, 0) IN (
            SELECT a.patient_id, 0 AS stable_tie
            FROM patients AS a
            UNION
            SELECT b.patient_id, RANDOM() AS random_tie
            FROM patients AS b
            ORDER BY random_tie
        )
        LIMIT 5
        """,
    )

    assert result.accepted is False
    assert any("nondeterministic ordering controls an outer limit" in reason for reason in result.reasons)
    assert not any("unknown or ambiguous source column: random_tie" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "order_expression",
    (
        "right_id",
        "likely((right_id))",
        "likely(right_id COLLATE NOCASE)",
        "likelihood((right_id COLLATE NOCASE), 0.5)",
    ),
)
def test_active_contract_allows_a_deterministic_later_compound_alias(order_expression):
    contract = extract_semantic_contract("列出前5名患者编号")
    result = validate_candidate(
        contract,
        f"""
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id IN (
            SELECT a.patient_id AS left_id
            FROM patients AS a
            UNION
            SELECT b.patient_id AS right_id
            FROM patients AS b
            ORDER BY {order_expression}
        )
        ORDER BY p.patient_id ASC
        LIMIT 5
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


@pytest.mark.parametrize(
    "order_expression",
    (
        "(id) ASC",
        "likely((id)) ASC",
        "likelihood((id), 0.5) ASC",
    ),
)
def test_active_contract_allows_transparent_wrapped_direct_order_alias(order_expression):
    contract = SemanticContract(
        projections=("patient_id",),
        ordering=("patient_id ASC",),
        limit=5,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT patient_id AS id
        FROM patients
        ORDER BY {order_expression}
        LIMIT 5
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_collated_direct_order_alias_does_not_match_default_collation_contract():
    contract = SemanticContract(
        projections=("patient_id",),
        ordering=("patient_id ASC",),
        limit=5,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT patient_id AS id
        FROM patients
        ORDER BY likely(id) COLLATE NOCASE ASC
        LIMIT 5
        """,
    )

    assert result.accepted is False
    assert any("ordering" in reason for reason in result.reasons)
    assert not any("unknown or ambiguous source column: id" in reason for reason in result.reasons)


def test_compound_output_alias_does_not_inherit_a_branch_source_volatility():
    contract = SemanticContract(
        projections=("patient_id",),
        limit=5,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH source AS (
            SELECT patient_id, RANDOM() AS right_id
            FROM patients
        )
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id IN (
            SELECT a.patient_id AS left_id
            FROM source AS a
            UNION
            SELECT b.patient_id AS right_id
            FROM patients AS b
            ORDER BY right_id
        )
        LIMIT 5
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_compound_order_unknown_output_remains_a_structural_rejection():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id IN (
            SELECT a.patient_id
            FROM patients AS a
            UNION
            SELECT b.patient_id
            FROM patients AS b
            ORDER BY unknown_output
        )
        """,
    )

    assert result.accepted is False
    assert any("unknown or ambiguous source column: unknown_output" in reason for reason in result.reasons)


def test_active_contract_rejects_unknown_compound_order_provenance_under_outer_limit():
    contract = extract_semantic_contract("列出前5名患者编号")
    result = validate_candidate(
        contract,
        """
        SELECT p.patient_id
        FROM patients AS p
        WHERE p.patient_id IN (
            VALUES (1)
            UNION
            SELECT b.patient_id
            FROM patients AS b
            ORDER BY 1
        )
        LIMIT 5
        """,
    )

    assert result.accepted is False
    assert any(
        "ordered paging provenance cannot be established for an outer limit" in reason
        for reason in result.reasons
    )


@pytest.mark.parametrize(
    "group_expression",
    ("(d)", "likely((d))", "likelihood((d), 0.5)"),
)
def test_active_contract_allows_transparent_wrapped_direct_group_alias(group_expression):
    contract = SemanticContract(
        projections=("department",),
        aggregates=("COUNT(*)",),
        grouping=("department",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT department AS d, COUNT(*) AS total
        FROM visits
        GROUP BY {group_expression}
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_collated_direct_group_alias_does_not_match_default_collation_contract():
    contract = SemanticContract(
        projections=("department",),
        aggregates=("COUNT(*)",),
        grouping=("department",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT department AS d, COUNT(*) AS total
        FROM visits
        GROUP BY likely(d) COLLATE NOCASE
        """,
    )

    assert result.accepted is False
    assert any("grouping" in reason for reason in result.reasons)
    assert not any("unknown or ambiguous source column: d" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "group_expression",
    ("1 COLLATE NOCASE", "likely(1) COLLATE NOCASE"),
)
def test_collated_group_ordinal_does_not_match_default_collation_contract(group_expression):
    contract = SemanticContract(
        projections=("department",),
        aggregates=("COUNT(*)",),
        grouping=("department",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT department, COUNT(*) AS total
        FROM visits
        GROUP BY {group_expression}
        """,
    )

    assert result.accepted is False
    assert any("grouping" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "order_expression",
    ("1 COLLATE NOCASE", "likely(1) COLLATE NOCASE"),
)
def test_collated_order_ordinal_does_not_match_default_collation_contract(order_expression):
    contract = SemanticContract(
        projections=("patient_id",),
        ordering=("patient_id ASC",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT patient_id
        FROM patients
        ORDER BY {order_expression}
        """,
    )

    assert result.accepted is False
    assert any("ordering" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    ("contract_order", "candidate_order"),
    (
        ("patient_id ASC", "patient_id ASC NULLS FIRST"),
        ("patient_id DESC", "patient_id DESC NULLS LAST"),
        ("patient_id ASC NULLS LAST", "patient_id ASC NULLS LAST"),
        ("patient_id NULLS LAST", "patient_id NULLS LAST"),
    ),
)
def test_ordering_contract_preserves_effective_null_placement(
    contract_order,
    candidate_order,
):
    contract = SemanticContract(
        projections=("patient_id",),
        ordering=(contract_order,),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT patient_id
        FROM patients
        ORDER BY {candidate_order}
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


@pytest.mark.parametrize(
    ("contract_order", "candidate_order"),
    (
        ("patient_id ASC", "patient_id ASC NULLS LAST"),
        ("patient_id DESC", "patient_id DESC NULLS FIRST"),
        ("patient_id ASC NULLS LAST", "patient_id ASC"),
    ),
)
def test_ordering_contract_rejects_changed_null_placement(
    contract_order,
    candidate_order,
):
    contract = SemanticContract(
        projections=("patient_id",),
        ordering=(contract_order,),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT patient_id
        FROM patients
        ORDER BY {candidate_order}
        """,
    )

    assert result.accepted is False
    assert any(
        "ordering key/direction/null-placement mismatch" in reason
        for reason in result.reasons
    )


@pytest.mark.parametrize(
    "order_expression",
    (
        "strftime('%S', 'now')",
        "strftime('%S')",
        "strftime('%S', 'subsec')",
        "date('now')",
        "date()",
        "date('subsec')",
        "time('now')",
        "time()",
        "datetime('now')",
        "datetime()",
        "datetime('subsec')",
        "julianday('now')",
        "julianday()",
        "unixepoch('now')",
        "unixepoch()",
        "unixepoch('subsecond')",
        "timediff('now', visit_date)",
    ),
)
def test_limit_contract_rejects_sqlite_current_time_ordering(order_expression):
    contract = SemanticContract(
        projections=("visit_date",),
        limit=1,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT visit_date
        FROM visits
        ORDER BY {order_expression}
        LIMIT 1
        """,
    )

    assert result.accepted is False
    assert "nondeterministic ordering controls a top-k limit" in result.reasons


@pytest.mark.parametrize(
    "order_expression",
    (
        "strftime('%S', visit_date)",
        "strftime('now', visit_date)",
        "date(visit_date)",
        "time(visit_date)",
        "datetime(visit_date)",
        "julianday(visit_date)",
        "unixepoch(visit_date)",
        "timediff(visit_date, visit_date)",
    ),
)
def test_limit_contract_allows_deterministic_sqlite_time_column_ordering(order_expression):
    contract = SemanticContract(
        projections=("visit_date",),
        limit=1,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        SELECT visit_date
        FROM visits
        ORDER BY {order_expression}
        LIMIT 1
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_active_contract_allows_nested_deterministic_order_without_paging():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH ordered AS (
            SELECT patient_id FROM patients ORDER BY patient_id
        )
        SELECT patient_id FROM ordered
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_none_contract_preserves_structural_offset_compatibility():
    contract = extract_semantic_contract("查询一些记录")
    result = validate_candidate(
        contract,
        "SELECT patient_id FROM patients LIMIT -1 OFFSET 1",
    )

    assert contract.status is ContractStatus.NONE
    assert result.accepted is True
    assert result.reasons == ()


def test_limit_only_contract_rejects_nondeterministic_outer_ordering():
    contract = extract_semantic_contract("列出前5名患者编号")
    assert contract.projections == ("patient_id",)
    assert contract.limit == 5
    assert contract.ordering == ()

    result = validate_candidate(
        contract,
        "SELECT patient_id FROM patients ORDER BY RANDOM() LIMIT 5",
    )

    assert result.accepted is False
    assert any("nondeterministic" in reason for reason in result.reasons)


def test_limit_only_contract_allows_deterministic_outer_ordering():
    contract = extract_semantic_contract("列出前5名患者编号")
    result = validate_candidate(
        contract,
        "SELECT patient_id FROM patients ORDER BY patient_id ASC LIMIT 5",
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_derived_volatile_tie_key_is_rejected_for_a_limited_result():
    contract = SemanticContract(
        projections=("department",),
        aggregates=("COUNT(*)",),
        grouping=("department",),
        ordering=("COUNT(*) DESC",),
        limit=5,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH ranked AS (
            SELECT department, COUNT(*) AS n, RANDOM() AS tie
            FROM visits
            GROUP BY department
        )
        SELECT department, n
        FROM ranked
        ORDER BY n DESC, tie ASC
        LIMIT 5
        """,
    )

    assert result.accepted is False
    assert any("nondeterministic" in reason for reason in result.reasons)


@pytest.mark.parametrize("sort_expression", ("r + 0", "ABS(r)"))
def test_unsupported_derived_expression_preserves_volatile_lineage(sort_expression):
    contract = SemanticContract(
        projections=("patient_id",),
        limit=5,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        WITH seeded AS (
            SELECT patient_id, RANDOM() AS r
            FROM patients
        ), masked AS (
            SELECT patient_id, {sort_expression} AS sort_key
            FROM seeded
        )
        SELECT patient_id
        FROM masked
        ORDER BY sort_key
        LIMIT 5
        """,
    )

    assert result.accepted is False
    assert "nondeterministic ordering controls a top-k limit" in result.reasons


def test_deterministic_unsupported_derived_expression_is_not_volatile():
    contract = SemanticContract(
        projections=("patient_id",),
        limit=5,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH masked AS (
            SELECT patient_id, patient_id + 0 AS sort_key
            FROM patients
        )
        SELECT patient_id
        FROM masked
        ORDER BY sort_key
        LIMIT 5
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_randomblob_tie_breaker_is_rejected_as_nondeterministic():
    contract = SemanticContract(
        projections=("department",),
        aggregates=("COUNT(*)",),
        grouping=("department",),
        ordering=("COUNT(*) DESC",),
        limit=3,
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT department, COUNT(*) AS n
        FROM visits
        GROUP BY department
        ORDER BY n DESC, RANDOMBLOB(4)
        LIMIT 3
        """,
    )

    assert result.accepted is False
    assert any("nondeterministic" in reason for reason in result.reasons)


def test_transformed_aggregate_input_cannot_impersonate_the_requested_measure():
    contract = SemanticContract(
        aggregates=("AVG(cost)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        "SELECT AVG(CAST(cost AS INTEGER)) AS avg_cost FROM visits",
    )

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_grouped_cte_cannot_launder_a_base_row_count():
    contract = SemanticContract(
        aggregates=("COUNT(*)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH grouped AS (
            SELECT department, COUNT(*) AS n
            FROM visits
            GROUP BY department
        )
        SELECT COUNT(*) AS n FROM grouped
        """,
    )

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_distinct_cte_cannot_launder_an_average_input():
    contract = SemanticContract(
        aggregates=("AVG(cost)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH unique_costs AS (
            SELECT DISTINCT cost FROM visits
        )
        SELECT AVG(cost) AS avg_cost FROM unique_costs
        """,
    )

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_limited_cte_cannot_supply_an_outer_aggregate_input():
    contract = SemanticContract(
        aggregates=("AVG(cost)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH sample AS (
            SELECT cost FROM visits LIMIT 10
        )
        SELECT AVG(cost) AS avg_cost FROM sample
        """,
    )

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_multi_source_cte_cannot_supply_an_outer_aggregate_input():
    contract = SemanticContract(
        aggregates=("AVG(cost)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH joined AS (
            SELECT v.cost
            FROM visits AS v
            JOIN patients AS p ON p.patient_id = v.patient_id
        )
        SELECT AVG(cost) AS avg_cost FROM joined
        """,
    )

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_source_less_aggregate_cte_cannot_impersonate_a_base_count():
    contract = SemanticContract(
        aggregates=("COUNT(*)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH fake AS (
            SELECT COUNT(*) AS n
        )
        SELECT fake.n FROM fake CROSS JOIN patients
        """,
    )

    assert result.accepted is False


def test_grouped_cte_output_cannot_impersonate_a_scalar_aggregate():
    contract = SemanticContract(
        aggregates=("COUNT(*)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH grouped AS (
            SELECT department, COUNT(*) AS n
            FROM visits
            GROUP BY department
        )
        SELECT n FROM grouped
        """,
    )

    assert result.accepted is False
    assert any("grouping" in reason for reason in result.reasons)


def test_grouped_cte_semantics_survive_a_transparent_wrapper():
    contract = SemanticContract(
        aggregates=("COUNT(*)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH grouped AS (
            SELECT department, COUNT(*) AS n
            FROM visits
            GROUP BY department
        ), passthrough AS (
            SELECT n FROM grouped
        )
        SELECT n FROM passthrough
        """,
    )

    assert result.accepted is False
    assert any("grouping" in reason for reason in result.reasons)


def test_grouped_cte_join_cannot_hide_scalar_grouping_semantics():
    contract = SemanticContract(
        aggregates=("COUNT(*)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH grouped AS (
            SELECT department, COUNT(*) AS n
            FROM visits
            GROUP BY department
        )
        SELECT grouped.n
        FROM grouped CROSS JOIN patients
        """,
    )

    assert result.accepted is False
    assert any("grouped derived source" in reason for reason in result.reasons)


def test_grouped_cte_wrapper_join_cannot_hide_scalar_grouping_semantics():
    contract = SemanticContract(
        aggregates=("COUNT(*)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH grouped AS (
            SELECT department, COUNT(*) AS n
            FROM visits
            GROUP BY department
        ), passthrough AS (
            SELECT n
            FROM grouped CROSS JOIN patients
        )
        SELECT n FROM passthrough
        """,
    )

    assert result.accepted is False
    assert any("grouped derived source" in reason for reason in result.reasons)


def test_grouped_cte_intermediate_join_cannot_hide_scalar_grouping_semantics():
    contract = SemanticContract(
        aggregates=("COUNT(*)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH grouped AS (
            SELECT department, COUNT(*) AS n
            FROM visits
            GROUP BY department
        ), joined AS (
            SELECT grouped.n
            FROM grouped CROSS JOIN patients
        )
        SELECT n FROM joined
        """,
    )

    assert result.accepted is False
    assert any("grouped derived source" in reason for reason in result.reasons)


def test_scalar_aggregate_cte_cannot_be_forwarded_through_a_join():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH stats AS (
            SELECT AVG(p.age) AS avg_age
            FROM patients AS p
        )
        SELECT stats.avg_age
        FROM visits AS v
        JOIN patients AS p ON v.patient_id = p.patient_id
        CROSS JOIN stats
        """,
    )

    assert result.accepted is False
    assert any("joined derived aggregate source" in reason for reason in result.reasons)


def test_scalar_aggregate_cte_passthrough_cannot_be_forwarded_through_a_join():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH stats AS (
            SELECT AVG(p.age) AS avg_age
            FROM patients AS p
        ), passthrough AS (
            SELECT avg_age AS value
            FROM stats
        )
        SELECT passthrough.value
        FROM passthrough CROSS JOIN visits AS v
        """,
    )

    assert result.accepted is False
    assert any("joined derived aggregate source" in reason for reason in result.reasons)


def test_declared_scalar_cte_output_cannot_be_forwarded_through_a_join():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH stats(avg_age) AS (
            SELECT AVG(age)
            FROM patients
        ), passthrough AS (
            SELECT avg_age
            FROM stats
        )
        SELECT passthrough.avg_age
        FROM passthrough CROSS JOIN visits AS v
        """,
    )

    assert result.accepted is False
    assert any("joined derived aggregate source" in reason for reason in result.reasons)


def test_scalar_aggregate_cte_join_hidden_in_an_intermediate_cte_is_rejected():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH stats AS (
            SELECT AVG(p.age) AS avg_age
            FROM patients AS p
        ), joined AS (
            SELECT stats.avg_age
            FROM stats CROSS JOIN visits AS v
        )
        SELECT avg_age FROM joined
        """,
    )

    assert result.accepted is False
    assert any("joined derived aggregate source" in reason for reason in result.reasons)


def test_scalar_aggregate_subquery_cannot_be_forwarded_through_a_join():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT stats.avg_age
        FROM (
            SELECT AVG(age) AS avg_age
            FROM patients
        ) AS stats
        CROSS JOIN visits AS v
        """,
    )

    assert result.accepted is False
    assert any("joined derived aggregate source" in reason for reason in result.reasons)


def test_scalar_aggregate_cte_star_passthrough_cannot_hide_a_join():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH stats AS (
            SELECT AVG(age) AS avg_age
            FROM patients
        ), joined AS (
            SELECT stats.*
            FROM stats CROSS JOIN visits AS v
        )
        SELECT avg_age FROM joined
        """,
    )

    assert result.accepted is False
    assert any("joined derived aggregate source" in reason for reason in result.reasons)


def test_single_source_scalar_aggregate_cte_remains_valid():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH stats AS (
            SELECT AVG(age) AS avg_age
            FROM patients
        )
        SELECT avg_age FROM stats
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_scalar_aggregate_cte_can_cross_a_literal_singleton():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH stats AS (
            SELECT AVG(age) AS avg_age FROM patients
        ), one AS (
            SELECT 1 AS marker
        )
        SELECT stats.avg_age
        FROM stats CROSS JOIN one
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_scalar_aggregate_subquery_can_cross_a_literal_singleton():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT stats.avg_age
        FROM (
            SELECT AVG(age) AS avg_age FROM patients
        ) AS stats
        CROSS JOIN (
            SELECT 1 AS marker
        ) AS one
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_singleton_proof_survives_a_direct_aggregate_passthrough():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH stats AS (
            SELECT AVG(age) AS avg_age FROM patients
        ), passthrough AS (
            SELECT avg_age AS value FROM stats
        ), one AS (
            SELECT 1 AS marker
        )
        SELECT passthrough.value
        FROM passthrough CROSS JOIN one
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_unreferenced_valid_aggregate_join_does_not_reject_the_outer_result():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH safe AS (
            SELECT AVG(age) AS avg_age FROM patients
        ), unused_stats AS (
            SELECT AVG(age) AS avg_age FROM patients
        ), unused_joined AS (
            SELECT unused_stats.avg_age
            FROM unused_stats CROSS JOIN patients
        )
        SELECT safe.avg_age FROM safe
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_unreferenced_valid_grouped_join_does_not_reject_the_outer_result():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH safe AS (
            SELECT AVG(age) AS avg_age FROM patients
        ), unused_grouped AS (
            SELECT department, COUNT(*) AS n
            FROM visits
            GROUP BY department
        ), unused_joined AS (
            SELECT unused_grouped.n
            FROM unused_grouped CROSS JOIN patients
        )
        SELECT safe.avg_age FROM safe
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_unreferenced_cte_with_a_forbidden_source_remains_rejected():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH invalid_source AS (
            SELECT item FROM forbidden_relation
        )
        SELECT patient_id FROM patients
        """,
    )

    assert result.accepted is False
    assert any("forbidden source relation" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "join_clause",
    (
        "JOIN one ON 1 = 1",
        "LEFT JOIN one ON 1 = 1",
        "RIGHT JOIN one ON 1 = 1",
        "FULL OUTER JOIN one ON 1 = 1",
    ),
)
def test_singleton_aggregate_proof_rejects_non_cross_joins(join_clause):
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        WITH stats AS (
            SELECT AVG(age) AS avg_age FROM patients
        ), one AS (
            SELECT 1 AS marker
        )
        SELECT stats.avg_age
        FROM stats {join_clause}
        """,
    )

    assert result.accepted is False
    assert any("joined derived aggregate source" in reason for reason in result.reasons)


def test_singleton_aggregate_proof_rejects_a_non_singleton_derived_partner():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH stats AS (
            SELECT AVG(age) AS avg_age FROM patients
        ), many AS (
            SELECT patient_id FROM patients
        )
        SELECT stats.avg_age
        FROM stats CROSS JOIN many
        """,
    )

    assert result.accepted is False
    assert any("joined derived aggregate source" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    ("projection", "suffix"),
    (
        ("stats.avg_age", "WHERE one.marker = 1"),
        ("DISTINCT stats.avg_age", ""),
        ("stats.avg_age", "GROUP BY stats.avg_age"),
        ("stats.avg_age", "HAVING stats.avg_age IS NOT NULL"),
        ("stats.avg_age", "LIMIT 1"),
        ("stats.avg_age", "LIMIT 1 OFFSET 0"),
        ("stats.avg_age, (SELECT 1 AS marker) AS ignored", ""),
        ("stats.*", ""),
    ),
)
def test_singleton_aggregate_proof_rejects_row_shaping_or_non_passthrough_sql(
    projection,
    suffix,
):
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        f"""
        WITH stats AS (
            SELECT AVG(age) AS avg_age FROM patients
        ), one AS (
            SELECT 1 AS marker
        )
        SELECT {projection}
        FROM stats CROSS JOIN one
        {suffix}
        """,
    )

    assert result.accepted is False
    assert any("joined derived aggregate source" in reason for reason in result.reasons)


def test_cross_join_cannot_multiply_a_count_input():
    contract = SemanticContract(
        aggregates=("COUNT(*)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        "SELECT COUNT(*) FROM visits AS v CROSS JOIN patients AS p",
    )

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_cross_join_cannot_multiply_a_numeric_aggregate_input():
    contract = SemanticContract(
        aggregates=("AVG(cost)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        "SELECT AVG(v.cost) FROM visits AS v CROSS JOIN patients AS p",
    )

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_lookup_join_cannot_weight_patient_side_average_by_visit_count():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT AVG(p.age)
        FROM visits AS v
        JOIN patients AS p ON v.patient_id = p.patient_id
        """,
    )

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_lookup_join_preserves_idempotent_patient_side_maximum():
    contract = SemanticContract(
        aggregates=("MAX(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT MAX(p.age)
        FROM patients AS p
        JOIN visits AS v ON p.patient_id = v.patient_id
        WHERE v.disease = '高血压' AND v.los_days = 0
        """,
    )

    assert result.accepted is True


def test_reversed_lookup_join_preserves_visit_row_count():
    contract = SemanticContract(
        aggregates=("COUNT(*)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT COUNT(*)
        FROM patients AS p
        JOIN visits AS v ON p.patient_id = v.patient_id
        """,
    )

    assert result.accepted is True


def test_reversed_lookup_join_preserves_visit_side_sum():
    contract = SemanticContract(
        aggregates=("SUM(cost)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT SUM(v.cost)
        FROM patients AS p
        JOIN visits AS v ON p.patient_id = v.patient_id
        """,
    )

    assert result.accepted is True


def test_reversed_lookup_join_preserves_distinct_patient_count_with_visit_sum():
    contract = SemanticContract(
        projections=("region",),
        aggregates=("COUNT(DISTINCT patient_id)", "SUM(cost)"),
        grouping=("region",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT p.region,
               COUNT(DISTINCT p.patient_id) AS patient_count,
               SUM(v.cost) AS total_cost
        FROM patients AS p
        JOIN visits AS v ON p.patient_id = v.patient_id
        GROUP BY p.region
        """,
    )

    assert result.accepted is True


def test_cross_join_cannot_launder_distinct_patient_count():
    contract = SemanticContract(
        aggregates=("COUNT(DISTINCT patient_id)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT COUNT(DISTINCT p.patient_id)
        FROM patients AS p
        CROSS JOIN visits AS v
        """,
    )

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_reversed_lookup_join_cannot_weight_patient_average_by_visit_count():
    contract = SemanticContract(
        aggregates=("AVG(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT AVG(p.age)
        FROM patients AS p
        JOIN visits AS v ON p.patient_id = v.patient_id
        """,
    )

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_lookup_join_cannot_weight_patient_side_sum_by_visit_count():
    contract = SemanticContract(
        aggregates=("SUM(age)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT SUM(p.age)
        FROM visits AS v
        JOIN patients AS p ON v.patient_id = p.patient_id
        """,
    )

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_lookup_join_preserves_visit_side_count_star_semantics():
    contract = SemanticContract(
        aggregates=("COUNT(*)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT COUNT(*)
        FROM visits AS v
        JOIN patients AS p ON v.patient_id = p.patient_id
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_lookup_join_preserves_visit_side_average_with_custom_aliases():
    contract = SemanticContract(
        aggregates=("AVG(cost)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        SELECT AVG(visit_rows.cost)
        FROM visits AS visit_rows
        JOIN patients AS patient_lookup
            ON visit_rows.patient_id = patient_lookup.patient_id
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_nested_row_shaping_subquery_cannot_supply_an_outer_aggregate_input():
    contract = SemanticContract(
        aggregates=("AVG(cost)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH filtered AS (
            SELECT v.cost AS cost
            FROM visits AS v
            WHERE v.cost IN (
                SELECT v2.cost AS cost
                FROM visits AS v2
                GROUP BY v2.cost
                HAVING COUNT(*) = 1
            )
        )
        SELECT AVG(f.cost) FROM filtered AS f
        """,
    )

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_transparent_single_source_cte_preserves_aggregate_input_safety():
    contract = SemanticContract(
        aggregates=("AVG(cost)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH source AS (
            SELECT disease, cost FROM visits
        )
        SELECT AVG(cost) AS avg_cost FROM source
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_filtered_single_source_cte_preserves_aggregate_input_safety():
    contract = SemanticContract(
        aggregates=("AVG(cost)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )
    result = validate_candidate(
        contract,
        """
        WITH filtered AS (
            SELECT cost FROM visits WHERE cost >= 0
        )
        SELECT AVG(cost) AS avg_cost FROM filtered
        """,
    )

    assert result.accepted is True
    assert result.reasons == ()


def test_unrequested_select_distinct_cannot_change_raw_result_cardinality():
    contract = SemanticContract(
        projections=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )

    result = validate_candidate(contract, "SELECT DISTINCT age FROM patients")

    assert result.accepted is False
    assert "unrequested SELECT DISTINCT changes result cardinality" in result.reasons


@pytest.mark.parametrize(
    "question",
    (
        "按名称顺序列出所有不同的处方药物。",
        "把所有处方药去重后按名称列出来。",
        "列出曾经来院就诊的患者姓名。",
        "哪些患者去过神经内科？列出姓名。",
    ),
)
def test_requested_or_existential_distinct_is_bound_to_the_contract(question):
    contract = extract_semantic_contract(question)

    assert contract.row_set_mode is RowSetMode.DISTINCT


@pytest.mark.parametrize(
    "question",
    (
        "How much does the 腹部CT cost?",
        "What is the diagnosis cost for 胃溃疡?",
        "What is the price of the drug 美托洛尔?",
        "What is the charge for a 腹部CT?",
    ),
)
def test_observed_cost_value_questions_request_a_distinct_row_set(question):
    contract = extract_semantic_contract(question)

    assert contract.row_set_mode is RowSetMode.DISTINCT


@pytest.mark.parametrize(
    "question",
    (
        "列出至少2次就诊记录的患者编号。",
        "列出至少三次就诊记录的患者姓名。",
        "列出至少N次就诊记录的患者编号。",
    ),
)
def test_minimum_occurrence_predicate_never_requests_outer_distinct(question):
    contract = extract_semantic_contract(question)

    assert contract.row_set_mode is RowSetMode.PRESERVE


@pytest.mark.parametrize(
    ("question", "expected_projections"),
    (
        (
            "连接就诊和图谱实体，列出处方药为缬沙坦且关系为 "
            "TREATED_BY_DRUG 时的药物、头实体和关系。",
            ("drug_prescribed", "head", "relation"),
        ),
        (
            "连接患者、就诊和以疾病为头实体的图谱，列出上海中疾病为高血压"
            "对应的地区、疾病、关系和尾实体。",
            ("region", "disease", "relation", "tail"),
        ),
        (
            "连接就诊和图谱实体，列出检查为心电图且关系为 DIAGNOSED_BY "
            "时的检查、头实体和尾实体。",
            ("exam", "head", "tail"),
        ),
        (
            "连接图谱三元组和实体，列出疾病实体的疾病、关系和尾实体。",
            ("disease", "relation", "tail"),
        ),
    ),
)
def test_explicit_cross_table_object_tuple_list_requests_outer_distinct(
    question,
    expected_projections,
):
    contract = extract_semantic_contract(question)

    assert contract.projections == expected_projections
    assert contract.output_slots == expected_projections
    assert contract.row_set_mode is RowSetMode.DISTINCT


def test_cross_table_object_tuple_distinct_is_enforced_only_at_outer_select():
    contract = extract_semantic_contract(
        "连接就诊和图谱实体，列出处方药为缬沙坦且关系为 "
        "TREATED_BY_DRUG 时的药物、头实体和关系。"
    )
    body = (
        "v.drug_prescribed, t.head, t.relation "
        "FROM visits AS v JOIN kg_triples AS t "
        "ON v.drug_prescribed = t.tail "
        "WHERE v.drug_prescribed = '缬沙坦' "
        "AND t.relation = 'TREATED_BY_DRUG'"
    )

    assert validate_candidate(contract, f"SELECT DISTINCT {body}").accepted
    result = validate_candidate(contract, f"SELECT {body}")
    assert result.accepted is False
    assert "requested SELECT DISTINCT is missing" in result.reasons


@pytest.mark.parametrize(
    "question",
    (
        "连接患者和就诊，列出每次就诊的患者编号和费用。",
        "连接图谱三元组和实体，逐条列出每条边的头实体和关系。",
        "连接患者和就诊，列出明细记录并保留重复。",
        "连接患者和就诊，按记录列出患者编号。",
        "连接患者和就诊，按科室统计就诊次数。",
        "连接患者和就诊，列出费用最高的前5条记录。",
        "列出患者编号和姓名。",
    ),
)
def test_non_object_set_list_shapes_preserve_outer_rows(question):
    contract = extract_semantic_contract(question)

    assert contract.row_set_mode is RowSetMode.PRESERVE


@pytest.mark.parametrize(
    "question",
    (
        "连接就诊和图谱实体，列出就诊编号、头实体和关系。",
        "连接图谱三元组和实体，逐条列出头实体、关系和尾实体。",
        "连接图谱三元组和实体，列出头实体、关系、尾实体和观察次数合计。",
        "连接图谱三元组和实体，生成排序编号后列出头实体、关系和尾实体。",
        "连接图谱三元组和实体，列出前5条头实体、关系和尾实体。",
        "连接图谱三元组和实体，偏移10条后列出头实体、关系和尾实体。",
        "连接图谱三元组和实体，合并两类结果后列出头实体、关系和尾实体。",
    ),
)
def test_cross_table_distinct_inference_excludes_record_and_analytical_shapes(
    question,
):
    contract = extract_semantic_contract(question)

    assert contract.row_set_mode is RowSetMode.PRESERVE


def test_requested_distinct_requires_the_outer_select_to_deduplicate():
    contract = SemanticContract(
        projections=("name",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
        row_set_mode=RowSetMode.DISTINCT,
    )

    assert validate_candidate(
        contract,
        "SELECT DISTINCT name FROM patients",
    ).accepted
    result = validate_candidate(contract, "SELECT name FROM patients")
    assert not result.accepted
    assert "requested SELECT DISTINCT is missing" in result.reasons


def test_patient_key_from_base_table_is_inherently_distinct_under_exists():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
        row_set_mode=RowSetMode.DISTINCT,
    )
    candidate = (
        "SELECT p.patient_id FROM patients p "
        "WHERE EXISTS (SELECT 1 FROM visits v "
        "WHERE v.patient_id=p.patient_id AND v.department='神经内科') "
        "ORDER BY p.patient_id"
    )

    assert validate_candidate(contract, candidate).accepted
    joined = validate_candidate(
        contract,
        "SELECT p.patient_id FROM patients p "
        "JOIN visits v ON v.patient_id=p.patient_id",
    )
    assert not joined.accepted
    assert "requested SELECT DISTINCT is missing" in joined.reasons


def test_raw_row_cross_join_requires_a_supported_singleton_proof():
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )

    result = validate_candidate(
        contract,
        "SELECT p.patient_id FROM patients AS p CROSS JOIN visits AS v",
    )

    assert result.accepted is False
    assert any("raw-row CROSS JOIN" in reason for reason in result.reasons)


def test_case_age_bucket_cannot_impersonate_an_ungrouped_raw_age():
    contract = SemanticContract(
        projections=("age",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )

    result = validate_candidate(
        contract,
        """
        SELECT CASE
                   WHEN age < 18 THEN 'minor'
                   ELSE 'adult'
               END AS age_group
        FROM patients
        """,
    )

    assert result.accepted is False
    assert any("age bucket" in reason for reason in result.reasons)


@pytest.mark.parametrize("target_type", ("INTEGER", "TEXT"))
def test_lossy_cast_cannot_impersonate_the_requested_average(target_type):
    contract = SemanticContract(
        aggregates=("AVG(cost)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )

    result = validate_candidate(
        contract,
        f"SELECT CAST(AVG(cost) AS {target_type}) FROM visits",
    )

    assert result.accepted is False
    assert any("aggregate" in reason for reason in result.reasons)


def test_real_cast_and_literal_round_remain_presentation_only():
    contract = SemanticContract(
        aggregates=("AVG(cost)",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )

    result = validate_candidate(
        contract,
        "SELECT ROUND(CAST(AVG(cost) AS REAL), 2) FROM visits",
    )

    assert result.accepted is True
    assert result.reasons == ()


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT patient_id FROM patients "
        "WHERE writefile('unsafe.bin', patient_id) IS NOT NULL",
        "SELECT patient_id FROM patients ORDER BY randomblob(8)",
        "WITH unused AS (SELECT mystery(age) AS value FROM patients) "
        "SELECT patient_id FROM patients",
    ),
)
def test_unapproved_sqlite_functions_are_rejected_across_the_entire_tree(sql):
    contract = SemanticContract(
        projections=("patient_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )

    result = validate_candidate(contract, sql)

    assert result.accepted is False
    assert any("function" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT patient_id FROM patients WHERE random() >= -9223372036854775808",
        "SELECT visit_id FROM visits WHERE visit_date <= CURRENT_TIMESTAMP",
    ),
)
def test_nondeterministic_functions_are_rejected_outside_ordering_too(sql):
    contract = SemanticContract(
        projections=("patient_id",) if "patient_id" in sql else ("visit_id",),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )

    result = validate_candidate(contract, sql)

    assert result.accepted is False
    assert any("nondeterministic" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    ("projection", "sql"),
    (
        ("age", "SELECT age FROM patients ORDER BY ABS(age)"),
        (
            "visit_date",
            "SELECT visit_date FROM visits ORDER BY date(visit_date)",
        ),
        (
            "drug_prescribed",
            "SELECT drug_prescribed FROM visits "
            "WHERE LENGTH(TRIM(drug_prescribed)) > 0",
        ),
    ),
)
def test_public_deterministic_sqlite_functions_remain_approved(projection, sql):
    contract = SemanticContract(
        projections=(projection,),
        confidence=1.0,
        status=ContractStatus.COMPLETE,
    )

    result = validate_candidate(contract, sql)

    assert result.accepted is True
    assert result.reasons == ()
