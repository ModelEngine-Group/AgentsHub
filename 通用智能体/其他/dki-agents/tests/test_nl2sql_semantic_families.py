"""Focused regressions for high-specificity semantic NL2SQL families."""
from __future__ import annotations

import sqlite3

from dki.analysis.nl2sql_semantic import compile_semantic_medical_sql
from dki.analysis.nl2sql_semantic_families import (
    _compile_age_peer_average,
    _compile_department_topk,
    _compile_episode_summary,
    _compile_first_last,
    _compile_lag_gap,
    _compile_monthly_yoy,
    _compile_ntile_quartiles,
    _compile_older_only_diseases,
    _compile_patient_exists,
    _compile_patient_rollup,
    _compile_patient_total_peer_average,
    _compile_region_benchmark_topk,
    _compile_required_departments,
    _compile_running_cost,
    compile_semantic_family_sql,
)


DOMAINS = {
    "region": ["北京", "深圳"],
    "department": ["内科", "外科"],
    "disease": ["疾病甲", "疾病乙"],
    "drug": ["药A", "药B"],
    "exam": ["检查甲"],
    "kg_relation_zh": ["呈现症状", "接受治疗", "并发症", "治疗药物"],
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


def test_department_topk_family_defers_explicit_projection():
    question = (
        "在2025-01-01至2025-12-31期间，每个科室按费用降序取前1，"
        "返回科室、就诊编号、处方药物和排名。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None


def test_required_entity_family_defers_type_projection_to_kg_compiler():
    question = (
        "查询作为头实体的疾病实体，同时具有呈现症状和接受治疗两种关系，"
        "呈现症状关系置信度至少0.7且接受治疗关系置信度至少0.8，"
        "返回实体名称、实体类型和提及次数。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None
    sql = compile_semantic_medical_sql(question, DOMAINS)
    assert sql is not None
    assert "SELECT e.name, e.type, e.mention_count" in sql
    assert "e.type = 'Disease'" in sql
    assert "t1.confidence >= 0.7" in sql
    assert "t2.confidence >= 0.8" in sql


def test_required_entity_family_retains_conjunctive_mention_bounds():
    question = (
        "查询作为头实体的疾病实体，同时具有呈现症状和接受治疗两种关系，"
        "第一条边置信度至少0.7且第二条边置信度至少0.8，"
        "提及次数至少3且不超过5。"
    )

    for sql in (
        compile_semantic_family_sql(question, DOMAINS),
        compile_semantic_medical_sql(question, DOMAINS),
    ):
        assert sql is not None
        assert "ke.mention_count >= 3" in sql
        assert "ke.mention_count <= 5" in sql
        assert "ke.type = 'Disease'" in sql


def test_required_entity_family_abstains_for_mention_disjunction():
    question = (
        "查询作为头实体的疾病实体，同时具有呈现症状和接受治疗两种关系，"
        "第一条边置信度至少0.7且第二条边置信度至少0.8，"
        "提及次数至少3或不超过5。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None
    assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_two_hop_family_keeps_distinct_thresholds_per_hop():
    sql = compile_semantic_family_sql(
        "查询并发症到治疗药物的两跳路径，第一条边置信度至少0.7且第二条边置信度至少0.8，"
        "第一条边观察次数至少2且第二条边观察次数至少4。",
        DOMAINS,
    )

    assert sql is not None
    assert "c.confidence >= 0.7" in sql
    assert "d.confidence >= 0.8" in sql
    assert "c.count >= 2" in sql
    assert "d.count >= 4" in sql


def test_two_hop_family_abstains_for_unshared_single_thresholds():
    assert (
        compile_semantic_family_sql(
            "查询并发症到治疗药物的两跳路径，边置信度至少0.7且观察次数至少2。",
            DOMAINS,
        )
        is None
    )


def test_fixed_kg_family_abstains_for_unowned_result_modifiers():
    base = (
        "查询并发症到治疗药物的两跳路径，"
        "第一条边置信度至少0.7且第二条边置信度至少0.8，"
        "第一条边观察次数至少2且第二条边观察次数至少4，"
    )

    for modifier in ("目标科室从高到低。", "最多2条。"):
        question = base + modifier

        assert compile_semantic_family_sql(question, DOMAINS) is None
        assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_fixed_kg_conditional_counts_abstain_for_unowned_result_modifiers():
    base = (
        "临床表现、治疗药物和诊断检查条件计数覆盖审计，"
        "症状数至少2，治疗药物数至少2，诊断检查数至少2，"
    )

    assert compile_semantic_family_sql(base, DOMAINS) is not None
    for modifier in ("目标字段从高到低？", "最多2条。"):
        question = base + modifier

        assert compile_semantic_family_sql(question, DOMAINS) is None
        assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_kg_conditional_counts_retains_conjunctive_metric_bounds():
    question = (
        "临床表现、治疗药物和诊断检查条件计数覆盖审计，"
        "症状数至少2且不超过4，治疗药物数至少3且不超过5，"
        "诊断检查数至少1且不超过2。"
    )

    for sql in (
        compile_semantic_family_sql(question, DOMAINS),
        compile_semantic_medical_sql(question, DOMAINS),
    ):
        assert sql is not None
        for predicate in (
            "'临床表现' THEN tail END) >= 2",
            "'临床表现' THEN tail END) <= 4",
            "'治疗药物' THEN tail END) >= 3",
            "'治疗药物' THEN tail END) <= 5",
            "'诊断检查' THEN tail END) >= 1",
            "'诊断检查' THEN tail END) <= 2",
        ):
            assert predicate in sql


def test_kg_conditional_counts_public_dispatch_handles_all_grounded_labels():
    domains = {
        **DOMAINS,
        "kg_relation_zh": ["临床表现", "治疗药物", "诊断检查"],
    }
    question = (
        "临床表现、治疗药物和诊断检查条件计数覆盖审计，"
        "症状数至少2且不超过4，治疗药物数至少3且不超过5，"
        "诊断检查数至少1且不超过2。"
    )

    sql = compile_semantic_medical_sql(question, domains)

    assert sql is not None
    for predicate in (
        "'临床表现' THEN tail END) >= 2",
        "'临床表现' THEN tail END) <= 4",
        "'治疗药物' THEN tail END) >= 3",
        "'治疗药物' THEN tail END) <= 5",
        "'诊断检查' THEN tail END) >= 1",
        "'诊断检查' THEN tail END) <= 2",
    ):
        assert predicate in sql


def test_kg_conditional_counts_abstains_for_metric_disjunction():
    question = (
        "临床表现、治疗药物和诊断检查条件计数覆盖审计，"
        "症状数至少2或不超过4，治疗药物数至少3，诊断检查数至少1。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None
    assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_monthly_yoy_family_uses_calendar_key_with_prior_baseline():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "2024-01-10", "内科", "疾病甲", "药A", "检查甲", 10.0, 1),
            (2, 1, "2024-03-10", "内科", "疾病甲", "药A", "检查甲", 30.0, 1),
            (3, 1, "2025-01-10", "内科", "疾病甲", "药A", "检查甲", 25.0, 1),
            (4, 1, "2025-03-10", "内科", "疾病甲", "药A", "检查甲", 45.0, 1),
        ],
    )

    sql = compile_semantic_family_sql(
        "按月统计2025-01-01至2025-12-31的总费用同比。",
        DOMAINS,
    )

    assert sql is not None
    assert "JOIN monthly AS prior_month" in sql
    assert "LAG(" not in sql
    assert connection.execute(sql).fetchall() == [
        ("2025-01", 25.0, 10.0, 15.0),
        ("2025-03", 45.0, 30.0, 15.0),
    ]


def test_grouped_having_family_applies_patient_qualifiers():
    connection = _medical_database()
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
        [
            (1, "甲", "女", 50, "深圳"),
            (2, "乙", "女", 35, "深圳"),
            (3, "丙", "男", 50, "深圳"),
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
        ],
    )

    sql = compile_semantic_family_sql(
        "按科室筛选2025-01-01至2025-12-31女性、年龄至少40岁、深圳地区患者的"
        "就诊次数和平均费用，筛选就诊次数至少2且平均费用至少100。",
        DOMAINS,
    )

    assert sql is not None
    assert "JOIN patients AS p ON p.patient_id = v.patient_id" in sql
    assert "p.age >= 40" in sql
    assert "p.gender = '女'" in sql
    assert "p.region = '深圳'" in sql
    assert connection.execute(sql).fetchall() == [("内科", 2, 150.0, 300.0)]


def test_two_hop_family_retains_each_lower_and_upper_edge_bound():
    sql = compile_semantic_family_sql(
        "查询并发症到治疗药物的两跳路径，"
        "第一条边置信度至少0.7且不超过0.9且第二条边置信度至少0.8且不超过0.95，"
        "第一条边观察次数至少2且不超过4且第二条边观察次数至少3且不超过5。",
        DOMAINS,
    )

    assert sql is not None
    for predicate in (
        "c.confidence >= 0.7",
        "c.confidence <= 0.9",
        "d.confidence >= 0.8",
        "d.confidence <= 0.95",
        "c.count >= 2",
        "c.count <= 4",
        "d.count >= 3",
        "d.count <= 5",
    ):
        assert predicate in sql


def test_two_hop_family_binds_reversed_named_relation_bounds():
    sql = compile_semantic_family_sql(
        "查询并发症到治疗药物的两跳路径，"
        "治疗药物关系置信度至少0.8且观察次数至少5，"
        "并发症关系置信度至少0.7且观察次数至少3。",
        DOMAINS,
    )

    assert sql is not None
    assert "c.confidence >= 0.7" in sql
    assert "c.count >= 3" in sql
    assert "d.confidence >= 0.8" in sql
    assert "d.count >= 5" in sql


def test_disease_pair_family_preserves_distinct_edge_criteria():
    sql = compile_semantic_family_sql(
        "疾病对共同症状重叠，共同症状至少2，"
        "第一条边置信度至少0.7且第二条边置信度至少0.8，"
        "第一条边观察次数至少3且第二条边观察次数至少5。",
        DOMAINS,
    )

    assert sql is not None
    assert "a.confidence >= 0.7" in sql
    assert "b.confidence >= 0.8" in sql
    assert "a.count >= 3" in sql
    assert "b.count >= 5" in sql


def test_disease_pair_family_retains_conjunctive_shared_symptom_bounds():
    question = (
        "疾病对共同症状重叠，共同症状至少2且不超过4，"
        "第一条边置信度至少0.7且第二条边置信度至少0.8，"
        "第一条边观察次数至少3且第二条边观察次数至少5。"
    )

    for sql in (
        compile_semantic_family_sql(question, DOMAINS),
        compile_semantic_medical_sql(question, DOMAINS),
    ):
        assert sql is not None
        assert "COUNT(DISTINCT a.tail) >= 2" in sql
        assert "COUNT(DISTINCT a.tail) <= 4" in sql


def test_disease_pair_family_abstains_for_shared_symptom_disjunction():
    question = (
        "疾病对共同症状重叠，共同症状至少2或不超过4，"
        "第一条边置信度至少0.7且第二条边置信度至少0.8，"
        "第一条边观察次数至少3且第二条边观察次数至少5。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None
    assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_disease_pair_family_abstains_for_unscoped_single_edge_threshold():
    assert (
        compile_semantic_family_sql(
            "疾病对共同症状重叠，共同症状至少2，边置信度至少0.7。",
            DOMAINS,
        )
        is None
    )


def test_disease_pair_family_accepts_compatible_explicit_projection():
    sql = compile_semantic_family_sql(
        "疾病对共同症状重叠，共同症状至少2，"
        "第一条边置信度至少0.7且第二条边置信度至少0.8，"
        "第一条边观察次数至少3且第二条边观察次数至少5，"
        "返回疾病一、疾病二和共享症状数。",
        DOMAINS,
    )

    assert sql is not None
    assert "SELECT a.head AS disease_one, b.head AS disease_two" in sql


def test_disease_pair_family_defers_explicit_projection():
    """Keep the historical node ID while pinning the stricter replacement.

    The old implementation deferred every explicit projection.  The current
    contract accepts only the compatible, fully scoped projection and still
    abstains on ambiguous criteria; retaining this node makes that migration
    visible instead of silently deleting a baseline test.
    """

    test_disease_pair_family_accepts_compatible_explicit_projection()


def test_disease_pair_family_retains_optional_shared_count_range():
    sql = compile_semantic_family_sql(
        "疾病对共同症状重叠，共同症状至少2，"
        "两条边置信度均至少0.7，观察次数均在2至5之间。",
        DOMAINS,
    )

    assert sql is not None
    for predicate in (
        "a.count >= 2",
        "a.count <= 5",
        "b.count >= 2",
        "b.count <= 5",
    ):
        assert predicate in sql


def test_required_relation_family_retains_distinct_bounded_edge_conditions():
    sql = compile_semantic_family_sql(
        "要求集合中的呈现症状和接受治疗两类关系逐一满足，"
        "呈现症状关系置信度至少0.7且不超过0.9且观察次数至少2且不超过4，"
        "接受治疗关系置信度至少0.8且不超过0.95且观察次数至少3且不超过5，"
        "筛选就诊次数至少2且平均费用至少100。",
        DOMAINS,
    )

    assert sql is not None
    assert (
        "(r.relation_zh = '呈现症状' AND kt.confidence >= 0.7 "
        "AND kt.confidence <= 0.9 AND kt.count >= 2 AND kt.count <= 4)"
    ) in sql
    assert (
        "(r.relation_zh = '接受治疗' AND kt.confidence >= 0.8 "
        "AND kt.confidence <= 0.95 AND kt.count >= 3 AND kt.count <= 5)"
    ) in sql
    assert "))) GROUP BY ke.name, ke.mention_count" in sql
    assert "ke.type = 'Disease'" in sql
    assert "AVG(v.cost) >= 100" in sql
    assert sql.count("(") == sql.count(")")


def test_required_relation_family_keeps_visit_bounds_distinct_from_edge_counts():
    question = (
        "要求集合中的呈现症状和接受治疗两类关系逐一满足，"
        "呈现症状关系置信度至少0.7且观察次数至少4，"
        "接受治疗关系置信度至少0.8且观察次数至少3，"
        "筛选就诊次数至少2且平均费用至少100。"
    )

    for sql in (
        compile_semantic_family_sql(question, DOMAINS),
        compile_semantic_medical_sql(question, DOMAINS),
    ):
        assert sql is not None
        assert "ke.type = 'Disease'" in sql
        assert "kt.count >= 4" in sql
        assert "COUNT(v.visit_id) >= 2" in sql
        assert "COUNT(v.visit_id) >= 4" not in sql
        assert "AVG(v.cost) >= 100" in sql


def test_required_relation_family_retains_conjunctive_visit_bounds():
    question = (
        "要求集合中的呈现症状和接受治疗两类关系逐一满足，"
        "呈现症状关系置信度至少0.7且观察次数至少2，"
        "接受治疗关系置信度至少0.8且观察次数至少3，"
        "筛选就诊次数至少2且不超过4且平均费用至少100。"
    )

    for sql in (
        compile_semantic_family_sql(question, DOMAINS),
        compile_semantic_medical_sql(question, DOMAINS),
    ):
        assert sql is not None
        assert "COUNT(v.visit_id) >= 2" in sql
        assert "COUNT(v.visit_id) <= 4" in sql


def test_required_relation_family_retains_since_date_scope():
    question = (
        "要求集合中的呈现症状和接受治疗两类关系逐一满足，"
        "呈现症状关系置信度至少0.7且观察次数至少2，"
        "接受治疗关系置信度至少0.8且观察次数至少3，"
        "自2025-01-01以来筛选就诊次数至少2且平均费用至少100。"
    )

    for sql in (
        compile_semantic_family_sql(question, DOMAINS),
        compile_semantic_medical_sql(question, DOMAINS),
    ):
        assert sql is not None
        assert "v.visit_date >= '2025-01-01'" in sql


def test_required_relation_family_abstains_for_unrenderable_date_scope():
    question = (
        "要求集合中的呈现症状和接受治疗两类关系逐一满足，"
        "呈现症状关系置信度至少0.7且观察次数至少2，"
        "接受治疗关系置信度至少0.8且观察次数至少3，"
        "在2025-01-01筛选就诊次数至少2且平均费用至少100。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None
    assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_required_relation_family_abstains_for_visit_count_disjunction():
    question = (
        "要求集合中的呈现症状和接受治疗两类关系逐一满足，"
        "呈现症状关系置信度至少0.7且观察次数至少2，"
        "接受治疗关系置信度至少0.8且观察次数至少3，"
        "筛选就诊次数至少2或不超过4且平均费用至少100。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None
    assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_required_relation_family_abstains_for_average_cost_disjunction():
    question = (
        "要求集合中的呈现症状和接受治疗两类关系逐一满足，"
        "呈现症状关系置信度至少0.7且观察次数至少2，"
        "接受治疗关系置信度至少0.8且观察次数至少3，"
        "筛选就诊次数至少2且平均费用至少100或不超过50。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None
    assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_cohort_evidence_retains_relation_metrics_and_gender():
    sql = compile_semantic_family_sql(
        "深圳地区年龄至少40岁的女性队列自2025-01-01以来知识覆盖证据，"
        "疾病拥有呈现症状关系，关系置信度至少0.8且不超过0.9，"
        "观察次数至少3且不超过5。",
        DOMAINS,
    )

    assert sql is not None
    assert "gender = '女'" in sql
    assert "kt.relation_zh = '呈现症状'" in sql
    assert "kt.confidence >= 0.8" in sql
    assert "kt.confidence <= 0.9" in sql
    assert "kt.count >= 3" in sql
    assert "kt.count <= 5" in sql


def test_cohort_evidence_abstains_for_collection_ordinal_relation_scope():
    question = (
        "深圳地区年龄至少40岁的女性队列自2025-01-01以来知识覆盖证据，"
        "疾病拥有呈现症状关系集合中的第一条边，"
        "关系置信度至少0.8且观察次数至少3。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None
    assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_family_tail_coverage_retains_relation_metrics():
    sql = compile_semantic_family_sql(
        "按呈现症状关系筛选尾实体，尾实体提及次数高于同类型平均，"
        "至少覆盖2种不同疾病，关系置信度至少0.8且不超过0.9，"
        "观察次数至少3且不超过5。",
        DOMAINS,
    )

    assert sql is not None
    assert "kt.relation_zh = '呈现症状'" in sql
    assert "kt.confidence >= 0.8" in sql
    assert "kt.confidence <= 0.9" in sql
    assert "kt.count >= 3" in sql
    assert "kt.count <= 5" in sql


def test_family_tail_coverage_retains_additive_mention_bound():
    question = (
        "按呈现症状关系筛选尾实体，尾实体提及次数高于同类型平均，"
        "尾实体提及次数至少10，至少覆盖2种不同疾病，"
        "关系置信度至少0.8且观察次数至少3。"
    )

    outputs = {
        "family": compile_semantic_family_sql(question, DOMAINS),
        "semantic": compile_semantic_medical_sql(question, DOMAINS),
    }
    for owner, sql in outputs.items():
        assert sql is not None, owner
        assert "ke.mention_count >= 10" in sql


def test_family_tail_coverage_abstains_for_collection_ordinal_relation_scope():
    question = (
        "按呈现症状关系集合中的第一条边筛选尾实体，"
        "尾实体提及次数高于同类型平均，至少覆盖2种不同疾病，"
        "关系置信度至少0.8且观察次数至少3。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None
    assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_family_tail_coverage_retains_conjunctive_disease_bounds():
    question = (
        "按呈现症状关系筛选尾实体，尾实体提及次数高于同类型平均，"
        "至少覆盖2种不同疾病且不超过4种不同疾病，"
        "关系置信度至少0.8且不超过0.9，观察次数至少3且不超过5。"
    )

    outputs = {
        "family": compile_semantic_family_sql(question, DOMAINS),
        "semantic": compile_semantic_medical_sql(question, DOMAINS),
    }
    for owner, sql in outputs.items():
        assert sql is not None, owner
        assert "COUNT(DISTINCT kt.head) >= 2" in sql
        assert "COUNT(DISTINCT kt.head) <= 4" in sql


def test_family_tail_coverage_abstains_for_disease_count_disjunction():
    question = (
        "按呈现症状关系筛选尾实体，尾实体提及次数高于同类型平均，"
        "至少覆盖2种不同疾病或不超过4种不同疾病，"
        "关系置信度至少0.8且观察次数至少3。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None
    assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_kg_only_families_abstain_for_unowned_patient_cohort_scope():
    questions = (
        (
            "在女性患者所患疾病中进行条件计数覆盖审计："
            "症状至少2个、治疗药物至少2种、诊断检查至少2项。"
        ),
        (
            "在女性患者所患疾病中审计疾病两两组成的疾病对："
            "共同症状至少2个，且两条边的置信度均至少0.8。"
        ),
        (
            "在女性患者所患疾病中审计并发症→治疗药物两跳路径："
            "两条边的置信度均至少0.8，观察次数均至少3次。"
        ),
        (
            "在女性患者所患疾病的呈现症状关系中，找出尾实体："
            "提及量高于同类型均值，至少覆盖3种不同疾病，"
            "置信度至少0.8，观察次数至少5。"
        ),
        (
            "在女性患者所患的疾病实体中，哪些疾病同时具有呈现症状和接受治疗关系，"
            "呈现症状关系置信度至少0.8，接受治疗关系置信度至少0.8，"
            "且提及次数至少10？"
        ),
        (
            "进行条件计数覆盖审计：症状至少2个、治疗药物至少2种、诊断检查至少2项，"
            "且费用至少100。"
        ),
    )

    for question in questions:
        assert compile_semantic_family_sql(question, DOMAINS) is None
        assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_kg_only_families_abstain_for_grounded_visit_field_scope():
    question = (
        "限定内科的临床表现、治疗药物和诊断检查条件计数覆盖审计，"
        "症状数至少2、治疗药物数至少2、诊断检查数至少2。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None
    assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_observed_risk_chain_scopes_observations_to_patient_demographics():
    sql = compile_semantic_family_sql(
        "并发症与用药禁忌风险链零次也保留并统计实际处方，"
        "年龄至少40岁的女性深圳患者自2025-01-01以来，"
        "并发症关系置信度至少0.7且观察次数至少2，"
        "用药禁忌关系置信度至少0.8且观察次数至少3。",
        DOMAINS,
    )

    assert sql is not None
    assert "EXISTS (SELECT 1 FROM patients AS p WHERE p.patient_id = v.patient_id" in sql
    assert "p.age >= 40" in sql
    assert "p.gender = '女'" in sql
    assert "p.region = '深圳'" in sql


def test_observed_risk_chain_retains_observed_prescription_bounds():
    question = (
        "并发症与用药禁忌风险链零次也保留并统计实际处方，"
        "年龄至少40岁的女性深圳患者自2025-01-01以来，"
        "并发症关系置信度至少0.7且观察次数至少2，"
        "用药禁忌关系置信度至少0.8且观察次数至少3，"
        "实际处方次数至少2且不超过4。"
    )

    for sql in (
        compile_semantic_family_sql(question, DOMAINS),
        compile_semantic_medical_sql(question, DOMAINS),
    ):
        assert sql is not None
        assert "HAVING COUNT(v.visit_id) >= 2 AND COUNT(v.visit_id) <= 4" in sql


def test_observed_risk_chain_public_dispatch_handles_all_grounded_metric_labels():
    domains = {
        **DOMAINS,
        "kg_relation_zh": [*DOMAINS["kg_relation_zh"], "用药禁忌"],
    }
    question = (
        "并发症与用药禁忌风险链零次也保留并统计实际处方，"
        "年龄至少40岁的女性深圳患者自2025-01-01以来，"
        "并发症关系置信度至少0.7且观察次数至少2，"
        "用药禁忌关系置信度至少0.8且观察次数至少3，"
        "实际处方次数至少2且不超过4。"
    )

    sql = compile_semantic_medical_sql(question, domains)

    assert sql is not None
    assert "c.confidence >= 0.7" in sql
    assert "x.confidence >= 0.8" in sql
    assert "c.count >= 2" in sql
    assert "x.count >= 3" in sql
    assert "HAVING COUNT(v.visit_id) >= 2 AND COUNT(v.visit_id) <= 4" in sql


def test_observed_risk_chain_retains_row_level_visit_cost_bounds():
    question = (
        "并发症与用药禁忌风险链零次也保留并统计实际处方，"
        "年龄至少40岁的女性深圳患者自2025-01-01以来，"
        "并发症关系置信度至少0.7且观察次数至少2，"
        "用药禁忌关系置信度至少0.8且观察次数至少3，"
        "实际处方次数至少2且费用至少100。"
    )

    for sql in (
        compile_semantic_family_sql(question, DOMAINS),
        compile_semantic_medical_sql(question, DOMAINS),
    ):
        assert sql is not None
        assert "AND v.cost >= 100" in sql


def test_observed_risk_chain_abstains_for_unowned_visit_cost_scope():
    base = (
        "并发症与用药禁忌风险链零次也保留并统计实际处方，"
        "年龄至少40岁的女性深圳患者自2025-01-01以来，"
        "并发症关系置信度至少0.7且观察次数至少2，"
        "用药禁忌关系置信度至少0.8且观察次数至少3，"
        "实际处方次数至少2且"
    )

    for cost_scope in ("平均费用至少100。", "费用至少100或不超过200。"):
        question = base + cost_scope

        assert compile_semantic_family_sql(question, DOMAINS) is None
        assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_observed_risk_chain_abstains_for_prescription_count_disjunction():
    question = (
        "并发症与用药禁忌风险链零次也保留并统计实际处方，"
        "年龄至少40岁的女性深圳患者自2025-01-01以来，"
        "并发症关系置信度至少0.7且观察次数至少2，"
        "用药禁忌关系置信度至少0.8且观察次数至少3，"
        "实际处方次数至少2或不超过4。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None
    assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_patient_kg_union_retains_gender_and_conjunctive_mention_bounds():
    question = "将年龄至少40岁的女性深圳患者和提及次数至少3且不超过5的药物实体合并为统一名称列表。"

    for sql in (
        compile_semantic_family_sql(question, DOMAINS),
        compile_semantic_medical_sql(question, DOMAINS),
    ):
        assert sql is not None
        assert "FROM patients AS p" in sql
        assert "gender = '女'" in sql
        assert "type = 'Drug'" in sql
        assert "mention_count >= 3" in sql
        assert "mention_count <= 5" in sql


def test_patient_kg_union_abstains_for_mention_disjunction():
    question = "将年龄至少40岁的女性深圳患者和提及次数至少3或不超过5的药物实体合并为统一名称列表。"

    assert compile_semantic_family_sql(question, DOMAINS) is None
    assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_patient_kg_union_abstains_for_unowned_temporal_scope():
    question = (
        "将自2025-01-01以来年龄至少40岁的女性深圳患者和"
        "提及次数至少3的药物实体合并为统一名称列表。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None
    assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_disease_having_topk_retains_patient_cohort_qualifiers():
    sql = compile_semantic_family_sql(
        "自2025-01-01以来年龄至少40岁且不超过50岁的女性深圳地区患者中，"
        "就诊次数至少2且平均费用最高的前3种疾病。",
        DOMAINS,
    )

    assert sql is not None
    assert "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id" in sql
    assert "v.visit_date >= '2025-01-01'" in sql
    assert "p.age BETWEEN 40 AND 50" in sql
    assert "p.gender = '女'" in sql
    assert "p.region = '深圳'" in sql


def test_patient_rollup_retains_conjunctive_age_range():
    sql = compile_semantic_family_sql(
        "年龄至少40岁且不超过50岁的患者按患者汇总，总费用至少100且就诊次数至少2。",
        DOMAINS,
    )

    assert sql is not None
    assert "p.age BETWEEN 40 AND 50" in sql


def test_family_templates_retain_conjunctive_age_range():
    cases = (
        (
            "按首诊月统计年龄至少40岁且不超过50岁的女性患者留存率，偏移月0至3。",
            "age BETWEEN 40 AND 50",
        ),
        (
            "深圳地区年龄至少40岁且不超过50岁的女性队列自2025-01-01以来知识覆盖证据，"
            "疾病拥有呈现症状关系，关系置信度至少0.8且观察次数至少3。",
            "age BETWEEN 40 AND 50",
        ),
        (
            "年龄至少40岁且不超过50岁的患者按总费用四分位分组，"
            "筛选就诊次数至少2且编号不超过2。",
            "p.age BETWEEN 40 AND 50",
        ),
        (
            "建立深圳地区年龄至少40岁且不超过50岁的女性患者队列，"
            "自2025-01-01以来按科室统计。",
            "age BETWEEN 40 AND 50",
        ),
        (
            "将年龄至少40岁且不超过50岁的女性深圳患者和提及次数至少3的药物实体"
            "合并为统一名称列表。",
            "age BETWEEN 40 AND 50",
        ),
    )

    for question, predicate in cases:
        sql = compile_semantic_family_sql(question, DOMAINS)
        assert sql is not None
        assert predicate in sql


def test_cohort_evidence_retains_row_level_visit_cost_scope():
    question = (
        "深圳地区年龄至少40岁的女性队列自2025-01-01以来知识覆盖证据，"
        "疾病拥有呈现症状关系，关系置信度至少0.8且观察次数至少3，费用至少100。"
    )

    outputs = {
        "family": compile_semantic_family_sql(question, DOMAINS),
        "semantic": compile_semantic_medical_sql(question, DOMAINS),
    }
    for owner, sql in outputs.items():
        assert sql is not None, owner
        assert "age >= 40" in sql
        assert "p.age >= 40" not in sql
        assert "v.cost >= 100" in sql


def test_cohort_evidence_abstains_for_unowned_visit_cost_scope():
    base = (
        "深圳地区年龄至少40岁的女性队列自2025-01-01以来知识覆盖证据，"
        "疾病拥有呈现症状关系，关系置信度至少0.8且观察次数至少3，"
    )

    for cost_scope in ("平均费用至少100。", "费用至少100或不超过200。"):
        question = base + cost_scope

        assert compile_semantic_family_sql(question, DOMAINS) is None
        assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_older_only_family_abstains_for_conjunctive_age_range():
    assert (
        compile_semantic_family_sql(
            "2025年年龄至少40岁且不超过50岁的患者疾病仅见于该人群，"
            "按知识库热度前3名。",
            DOMAINS,
        )
        is None
    )


def test_required_relation_visits_scopes_patient_demographics():
    sql = compile_semantic_family_sql(
        "要求集合中的呈现症状和接受治疗两类关系逐一满足，"
        "呈现症状关系置信度至少0.7且观察次数至少2，"
        "接受治疗关系置信度至少0.8且观察次数至少3，"
        "女性年龄至少40岁深圳地区患者，"
        "筛选就诊次数至少2且平均费用至少100。",
        DOMAINS,
    )

    assert sql is not None
    assert "JOIN patients AS p ON p.patient_id = v.patient_id" in sql
    assert "p.age >= 40" in sql
    assert "p.gender = '女'" in sql
    assert "p.region = '深圳'" in sql


def test_fixed_tail_coverage_template_defers_to_projection_aware_kg_compiler():
    question = (
        "按关系统计呈现症状的尾实体，尾实体提及次数高于同类型平均，"
        "不同头实体至少2个，返回关系中文名、不同尾实体数和不同头实体数。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None
    sql = compile_semantic_medical_sql(question, DOMAINS)
    assert sql is not None
    assert "SELECT t.relation_zh, COUNT(DISTINCT t.tail) AS distinct_tail_count" in sql


def test_legacy_family_lowerers_retain_static_patient_qualifiers():
    cases = (
        (
            _compile_episode_summary,
            "年龄至少40岁的女性深圳患者相隔超过30天即断开形成新一段，"
            "输出每段起止日期范围和持续天数，至少2次就诊。",
            ("p.age >= 40", "p.gender = '女'", "p.region = '深圳'"),
        ),
        (
            _compile_region_benchmark_topk,
            "每个地区中年龄至少40岁的女性深圳患者，科室覆盖数高于地区平均，"
            "且就诊次数至少2、总费用至少100的前3名。",
            ("p.age >= 40", "p.gender = '女'", "p.region = '深圳'"),
        ),
        (
            _compile_ntile_quartiles,
            "年龄至少40岁的女性深圳患者按总费用四分位分组，"
            "筛选就诊次数至少2且编号不超过2。",
            ("p.age >= 40", "p.gender = '女'", "p.region = '深圳'"),
        ),
        (
            _compile_required_departments,
            "年龄至少40岁的女性深圳患者，内科和外科都至少一次，输出不同科室数量。",
            ("p.age >= 40", "p.gender = '女'", "p.region = '深圳'"),
        ),
        (
            _compile_monthly_yoy,
            "按月统计年龄至少40岁的女性深圳患者2025-01-01至2025-12-31的总费用同比。",
            ("p.age >= 40", "p.gender = '女'", "p.region = '深圳'"),
        ),
        (
            _compile_lag_gap,
            "找出年龄至少40岁的女性深圳患者相邻就诊相隔至少30天的记录。",
            ("p.age >= 40", "p.gender = '女'", "p.region = '深圳'"),
        ),
        (
            _compile_first_last,
            "年龄至少40岁的女性深圳患者的首次和末次疾病，至少2次就诊。",
            ("p.age >= 40", "p.gender = '女'", "p.region = '深圳'"),
        ),
        (
            _compile_running_cost,
            "深圳地区年龄至少40岁的女性患者自2025-01-01以来每次就诊的累计费用。",
            ("p.age >= 40", "p.gender = '女'", "p.region = '深圳'"),
        ),
        (
            _compile_department_topk,
            "在2025-01-01至2025-12-31期间，年龄至少40岁的女性深圳患者"
            "每个科室内费用最高的前2名。",
            ("p.age >= 40", "p.gender = '女'", "p.region = '深圳'"),
        ),
        (
            _compile_patient_rollup,
            "年龄至少40岁的女性深圳患者按患者汇总，总费用至少100且就诊次数至少2。",
            ("p.age >= 40", "p.gender = '女'", "p.region = '深圳'"),
        ),
        (
            _compile_patient_total_peer_average,
            "年龄至少40岁的女性深圳患者个人总费用高于所在地区平均，且就诊次数至少2。",
            ("p.age >= 40", "p.gender = '女'", "p.region = '深圳'"),
        ),
        (
            _compile_patient_exists,
            "年龄至少40岁的女性深圳患者历史记录同时有疾病甲和药A，"
            "可来自不同就诊。",
            ("p.age >= 40", "p.gender = '女'", "p.region = '深圳'"),
        ),
        (
            _compile_age_peer_average,
            "年龄超过80岁的女性患者，排除深圳地区，"
            "找出年龄高于所在地区同性别患者平均年龄的患者。",
            ("p.age > 80", "p.gender = '女'", "p.region <> '深圳'"),
        ),
    )

    for compiler, question, predicates in cases:
        sql = compiler(question, DOMAINS)

        assert sql is not None
        for predicate in predicates:
            assert predicate in sql


def test_older_only_diseases_scopes_gender_and_region_to_both_difference_arms():
    sql = _compile_older_only_diseases(
        "2025年年龄至少40岁的女性深圳患者疾病差集，仅见于该人群且没有出现在低龄患者，"
        "按知识库热度前3名。",
        DOMAINS,
    )

    assert sql is not None
    assert "p.age >= 40" in sql
    assert "p.age < 40" in sql
    assert sql.count("p.gender = '女'") == 3
    assert sql.count("p.region = '深圳'") == 3


def test_older_only_diseases_retains_row_level_visit_cost_scope():
    question = (
        "2025年年龄至少40岁的女性深圳患者疾病差集，仅见于该人群且没有出现在低龄患者，"
        "按知识库热度前3名，费用至少100。"
    )

    for sql in (
        _compile_older_only_diseases(question, DOMAINS),
        compile_semantic_family_sql(question, DOMAINS),
        compile_semantic_medical_sql(question, DOMAINS),
    ):
        assert sql is not None
        assert sql.count("v.cost >= 100") == 3


def test_older_only_diseases_abstains_for_unowned_visit_cost_scope():
    base = (
        "2025年年龄至少40岁的女性深圳患者疾病差集，仅见于该人群且没有出现在低龄患者，"
        "按知识库热度前3名，"
    )

    for cost_scope in ("平均费用至少100。", "费用至少100或不超过200。"):
        question = base + cost_scope

        assert _compile_older_only_diseases(question, DOMAINS) is None
        assert compile_semantic_family_sql(question, DOMAINS) is None
        assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_peer_average_lowerers_keep_candidate_cohorts_outside_peer_baselines():
    total_cost_sql = _compile_patient_total_peer_average(
        "年龄至少40岁的女性深圳患者个人总费用高于所在地区平均，且就诊次数至少2。",
        DOMAINS,
    )
    age_sql = _compile_age_peer_average(
        "年龄超过80岁的女性患者，排除深圳地区，"
        "找出年龄高于所在地区同性别患者平均年龄的患者。",
        DOMAINS,
    )

    assert total_cost_sql is not None
    assert "p.age >= 40" in total_cost_sql
    assert "p.gender = '女'" in total_cost_sql
    assert "p.region = '深圳'" in total_cost_sql
    assert "p2.age >= 40" not in total_cost_sql
    assert "p2.gender = '女'" not in total_cost_sql
    assert "p2.region = '深圳'" not in total_cost_sql

    assert age_sql is not None
    assert "p.age > 80" in age_sql
    assert "p.gender = '女'" in age_sql
    assert "p2.age > 80" not in age_sql


def test_visit_window_lowerers_keep_visits_only_sources_without_patient_qualifiers():
    monthly_sql = _compile_monthly_yoy(
        "按月统计2025-01-01至2025-12-31的总费用同比。",
        DOMAINS,
    )
    lag_sql = _compile_lag_gap(
        "找出相邻就诊相隔至少30天的记录。",
        DOMAINS,
    )
    topk_sql = _compile_department_topk(
        "在2025-01-01至2025-12-31期间，每个科室内费用最高的前2名。",
        DOMAINS,
    )

    for sql in (monthly_sql, lag_sql, topk_sql):
        assert sql is not None
        assert "patients AS p" not in sql


def test_patient_exists_does_not_invert_region_for_unrelated_not_required_clause():
    question = (
        "找出深圳女性患者，不要求发生在同一次就诊，"
        "但历史上既有疾病甲又使用药A。"
    )

    for sql in (
        compile_semantic_family_sql(question, DOMAINS),
        compile_semantic_medical_sql(question, DOMAINS),
    ):
        assert sql is not None
        assert "p.gender = '女'" in sql
        assert "p.region = '深圳'" in sql
        assert "p.region NOT IN ('深圳')" not in sql


def test_mixed_disjunctive_region_polarity_is_terminal_before_fallback():
    question = "查询深圳患者或北京地区以外患者，返回患者编号。"

    assert compile_semantic_family_sql(question, DOMAINS) is None
    assert compile_semantic_medical_sql(question, DOMAINS) is None


def test_direct_family_dispatch_abstains_for_terminal_age_peer_intent():
    question = (
        "找出年龄高于同地区患者平均年龄且不要求发生在同一次就诊，"
        "但历史上既有疾病甲又使用药A的患者，返回患者编号和年龄。"
    )

    # The correlated age comparator cannot preserve the independent visit
    # predicates.  A direct public family call must not drop it and emit only
    # the patient EXISTS branch.
    assert compile_semantic_family_sql(question, DOMAINS) is None


def test_direct_family_dispatch_routes_exclusive_temporal_intents_only_to_designated_owner(
    monkeypatch,
):
    import dki.analysis.nl2sql_semantic_families as families_module

    routed: list[tuple[str, object]] = []

    def _designated_temporal_owner(
        question,
        _domains,
        *,
        intent=None,
        obligations=None,
    ):
        routed.append((question, intent))
        return "SELECT 'designated-temporal' AS source"

    monkeypatch.setattr(
        families_module,
        "compile_temporal_cohort_family_sql",
        _designated_temporal_owner,
    )
    retention_question = (
        "按首诊月统计年龄至少40岁的女性患者留存率，偏移月0至3。"
    )
    evidence_domains = {**DOMAINS, "kg_relation_zh": ["呈现症状"]}
    evidence_question = (
        "深圳地区年龄至少40岁的女性队列人群漏斗知识覆盖证据，"
        "自2025-01-01以来，疾病拥有呈现症状关系，"
        "关系置信度至少0.8，观察次数至少3。"
    )
    advanced_only_question = (
        "按首次就诊月统计患者漏斗，在7天和30天内再次就诊人数，返回结果。"
    )
    terminal_question = (
        "按首次就诊月统计内科患者漏斗，在7天和30天内再次就诊人数，返回结果。"
    )

    assert (
        families_module.compile_semantic_family_sql(retention_question, DOMAINS)
        == "SELECT 'designated-temporal' AS source"
    )
    assert (
        families_module.compile_semantic_family_sql(evidence_question, evidence_domains)
        == "SELECT 'designated-temporal' AS source"
    )
    assert (
        families_module.compile_semantic_family_sql(advanced_only_question, DOMAINS)
        is None
    )
    assert families_module.compile_semantic_family_sql(terminal_question, DOMAINS) is None
    assert [question for question, _ in routed] == [retention_question, evidence_question]
