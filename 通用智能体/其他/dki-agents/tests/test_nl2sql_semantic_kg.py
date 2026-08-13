"""Synthetic contract tests for the schema-bound KG semantic compiler."""

from dki.analysis.nl2sql_compile_result import CompileDisposition
from dki.analysis.nl2sql_semantic_kg import (
    analyze_relation_request,
    compile_kg_medical_sql,
    relation_fallback_disposition,
)


DOMAINS = {
    "kg_relation_zh": ["呈现症状", "接受治疗", "属于科室"],
    "region": ["深圳", "北京"],
    "department": ["内科", "外科"],
    "disease": ["疾病甲", "疾病乙"],
}


def test_relation_fallback_disposition_blocks_declined_structured_requests():
    assert (
        relation_fallback_disposition(
            "针对呈现症状ＯＲ接受治疗关系三元组边，"
            "第一个关系置信度至少0.8且观察次数至少4，返回头实体。",
            DOMAINS,
        )
        is CompileDisposition.BLOCKED
    )
    assert (
        relation_fallback_disposition(
            "针对呈现症状三元组边，置信度至少0.7且观察次数至少4。",
            DOMAINS,
        )
        is CompileDisposition.BLOCKED
    )
    assert (
        relation_fallback_disposition("查询呈现症状关系三元组。", DOMAINS)
        is CompileDisposition.NOT_APPLICABLE
    )


def test_relation_edge_filter_is_grounded_and_projected():
    sql = compile_kg_medical_sql(
        "针对呈现症状三元组边，置信度至少0.75且观察次数至少4，"
        "返回头实体、尾实体、置信度和观察次数。",
        DOMAINS,
    )
    assert sql == (
        "SELECT head, tail, confidence, count FROM kg_triples "
        "WHERE relation_zh = '呈现症状' AND confidence >= 0.75 AND count >= 4 "
        "ORDER BY confidence DESC, count DESC, head ASC, tail ASC"
    )


def test_tail_entity_join_uses_mention_threshold():
    sql = compile_kg_medical_sql(
        "将尾实体连接到kg_entities，尾实体提及次数至少3，"
        "返回实体名称、提及次数、头实体和观察次数。",
        DOMAINS,
    )
    assert sql is not None
    assert "JOIN kg_entities AS e ON e.name = t.tail" in sql
    assert "e.mention_count >= 3" in sql
    assert "SELECT e.name, e.mention_count, t.head, t.count" in sql


def test_per_head_topk_has_both_requested_metrics():
    sql = compile_kg_medical_sql(
        "按每个头实体的接受治疗关系统计，返回头实体、不同尾实体数和观察次数总和，"
        "排名前3。",
        DOMAINS,
    )
    assert sql is not None
    assert "relation_zh = '接受治疗'" in sql
    assert "COUNT(DISTINCT tail) AS distinct_tail_count" in sql
    assert "SUM(count) AS total_observations" in sql
    assert sql.endswith("LIMIT 3")


def test_entity_exists_requires_two_grounded_relations():
    sql = compile_kg_medical_sql(
        "作为头实体且类型为Disease的实体同时具有呈现症状和接受治疗关系，"
        "且两条边置信度均至少0.8，返回实体名称、类型和提及次数。",
        DOMAINS,
    )
    assert sql is not None
    assert sql.count("EXISTS (SELECT 1") == 2
    assert "t1.head = e.name" in sql
    assert "t2.relation_zh = '接受治疗'" in sql
    assert "t2.confidence >= 0.8" in sql


def test_tail_coverage_uses_same_type_average_and_having():
    sql = compile_kg_medical_sql(
        "按关系统计尾实体提及次数高于同类型平均且不同头实体至少2个，"
        "返回关系中文名、不同尾实体数和不同头实体数。",
        DOMAINS,
    )
    assert sql is not None
    assert "e.mention_count > (SELECT AVG(e2.mention_count)" in sql
    assert "WHERE e2.type = e.type" in sql
    assert "HAVING COUNT(DISTINCT t.head) >= 2" in sql


def test_two_hop_chain_has_distinct_edge_thresholds():
    sql = compile_kg_medical_sql(
        "查询两跳关系链，从头实体到尾实体，第一跳为呈现症状、第二跳为属于科室，"
        "第一条边置信度至少0.7且第二条边置信度至少0.8，"
        "返回源疾病、中间实体和目标科室。",
        DOMAINS,
    )
    assert sql is not None
    assert "ON t2.head = t1.tail" in sql
    assert "t1.confidence >= 0.7" in sql
    assert "t2.confidence >= 0.8" in sql
    assert "t1.relation_zh = '呈现症状'" in sql
    assert "t2.relation_zh = '属于科室'" in sql


def test_disease_pairs_use_shared_tail_and_canonical_pair_order():
    sql = compile_kg_medical_sql(
        "疾病头实体共享尾实体的疾病对，两个置信度均至少0.8且共享尾实体数至少2，"
        "返回疾病对、疾病二和共享尾实体数。",
        DOMAINS,
    )
    assert sql is not None
    assert "ON a.tail = b.tail AND a.head < b.head" in sql
    assert "a.confidence >= 0.8" in sql
    assert "b.confidence >= 0.8" in sql
    assert "HAVING COUNT(DISTINCT a.tail) >= 2" in sql


def test_required_relation_set_joins_grouped_visits():
    sql = compile_kg_medical_sql(
        "具有呈现症状和接受治疗关系的疾病头实体患者就诊，按地区分组，"
        "返回地区、就诊次数和不同疾病数。",
        DOMAINS,
    )
    assert sql is not None
    assert "qualified_heads AS" in sql
    assert "COUNT(DISTINCT relation_zh) = 2" in sql
    assert "JOIN qualified_heads AS q ON q.head = v.disease" in sql
    assert "GROUP BY p.region" in sql


def test_entity_type_recognition_uses_tail_position_and_range():
    sql = compile_kg_medical_sql(
        "识别呈现症状关系的尾实体，尾实体类型为Exam，提及次数在2至8之间，"
        "返回实体名称、类型和提及次数。",
        DOMAINS,
    )
    assert sql is not None
    assert "ON t.tail = e.name" in sql
    assert "e.type = 'Exam'" in sql
    assert "e.mention_count BETWEEN 2 AND 8" in sql


def test_entity_type_recognition_abstains_for_collection_ordinal_relation_scope():
    question = (
        "识别呈现症状关系集合中的第一条边的尾实体，尾实体类型为Exam，"
        "提及次数在2至8之间，返回实体名称、类型和提及次数。"
    )
    request = analyze_relation_request(question, DOMAINS)

    assert request.connective == "set_operation"
    assert compile_kg_medical_sql(question, DOMAINS) is None


def test_patient_kg_union_has_cohort_predicates_and_source_labels():
    sql = compile_kg_medical_sql(
        "将年龄至少40岁女性深圳患者与Disease图谱实体UNION成统一列表，"
        "返回名称和来源。",
        DOMAINS,
    )
    assert sql is not None
    assert "p.age >= 40" in sql
    assert "p.gender = '女'" in sql
    assert "p.region = '深圳'" in sql
    assert "'patient' AS source" in sql
    assert "'kg_entity' AS source" in sql
    assert "UNION" in sql


def test_risk_chain_connects_visits_to_two_graph_edges():
    sql = compile_kg_medical_sql(
        "风险链就诊，从头实体到尾实体，呈现症状和接受治疗，"
        "边置信度分别至少0.7和0.8，返回就诊ID、疾病、风险实体和风险结果。",
        DOMAINS,
    )
    assert sql is not None
    assert "t1.head = v.disease" in sql
    assert "t2.head = t1.tail" in sql
    assert "t1.confidence >= 0.7" in sql
    assert "t2.confidence >= 0.8" in sql


def test_kg_to_visits_coverage_has_total_and_covered_counts():
    sql = compile_kg_medical_sql(
        "按地区统计知识图谱覆盖率就诊，疾病头实体，"
        "返回地区、总就诊数、覆盖就诊数和覆盖率。",
        DOMAINS,
    )
    assert sql is not None
    assert "LEFT JOIN kg_triples AS t ON t.head = v.disease" in sql
    assert "AS total_visit_count" in sql
    assert "AS covered_visit_count" in sql
    assert "AS coverage" in sql


def test_kg_coverage_abstains_for_terminal_mixed_region_scope():
    question = (
        "深圳患者或北京地区以外患者，按地区统计知识图谱覆盖率就诊，疾病头实体，"
        "返回地区、总就诊数、覆盖就诊数和覆盖率。"
    )

    assert compile_kg_medical_sql(question, DOMAINS) is None


def test_older_only_enrichment_returns_graph_metrics():
    sql = compile_kg_medical_sql(
        "年龄至少65岁的老年患者疾病头实体图谱富集，"
        "返回疾病、老年患者数、图谱提及次数和不同关系数。",
        DOMAINS,
    )
    assert sql is not None
    assert "p.age >= 65" in sql
    assert "e.type = 'Disease'" in sql
    assert "e.mention_count" in sql
    assert "COUNT(DISTINCT t.relation_zh) AS relation_kind_count" in sql


def test_missing_edge_projection_abstains():
    assert (
        compile_kg_medical_sql(
            "查询呈现症状三元组，置信度至少0.75且观察次数至少4。",
            DOMAINS,
        )
        is None
    )


def test_ungrounded_relation_abstains():
    assert (
        compile_kg_medical_sql(
            "查询未知关系三元组，置信度至少0.75且观察次数至少4，"
            "返回头实体、尾实体、置信度和观察次数。",
            DOMAINS,
        )
        is None
    )


def test_two_hop_missing_second_threshold_abstains():
    assert (
        compile_kg_medical_sql(
            "查询两跳关系链，从头实体到尾实体，第一跳为呈现症状、第二跳为属于科室，"
            "第一条边置信度至少0.7，返回源疾病、中间实体和目标科室。",
            DOMAINS,
        )
        is None
    )


def test_required_relation_visits_enforces_shared_edge_qualifiers():
    sql = compile_kg_medical_sql(
        "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
        "每条关系边置信度至少0.8且观察次数至少4，"
        "按地区分组，返回地区、就诊次数和不同疾病数。",
        DOMAINS,
    )
    assert sql is not None
    assert "(relation_zh = '呈现症状' AND confidence >= 0.8 AND count >= 4)" in sql
    assert "(relation_zh = '接受治疗' AND confidence >= 0.8 AND count >= 4)" in sql



def test_required_relation_visits_enforces_ordered_edge_qualifiers():
    sql = compile_kg_medical_sql(
        "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
        "两条边的置信度分别至少0.7和0.8，观察次数分别至少3和5，"
        "按地区分组，返回地区、就诊次数和不同疾病数。",
        DOMAINS,
    )
    assert sql is not None
    assert "(relation_zh = '呈现症状' AND confidence >= 0.7 AND count >= 3)" in sql
    assert "(relation_zh = '接受治疗' AND confidence >= 0.8 AND count >= 5)" in sql



def test_required_relation_visits_enforces_relation_named_qualifiers():
    sql = compile_kg_medical_sql(
        "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
        "呈现症状关系置信度至少0.7且观察次数至少3，"
        "接受治疗关系置信度至少0.8且观察次数至少5，"
        "按地区分组，返回地区、就诊次数和不同疾病数。",
        DOMAINS,
    )
    assert sql is not None
    assert "(relation_zh = '呈现症状' AND confidence >= 0.7 AND count >= 3)" in sql
    assert "(relation_zh = '接受治疗' AND confidence >= 0.8 AND count >= 5)" in sql


def test_required_relation_visits_abstains_for_unscoped_edge_qualifiers():
    assert (
        compile_kg_medical_sql(
            "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
            "置信度至少0.7和0.8且观察次数至少3和5，"
            "按地区分组，返回地区、就诊次数和不同疾病数。",
            DOMAINS,
        )
        is None
    )


def test_required_relation_visits_abstains_for_malformed_edge_bound():
    assert (
        compile_kg_medical_sql(
            "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
            "每条关系边置信度至少，按地区分组，返回地区和就诊次数。",
            DOMAINS,
        )
        is None
    )


def test_grounded_tail_join_enforces_bounds_and_exact_projection():
    sql = compile_kg_medical_sql(
        "查询呈现症状关系的尾实体，尾实体提及次数至少3，"
        "置信度至少0.8且观察次数至少4，"
        "返回实体名称、提及次数、头实体。",
        DOMAINS,
    )
    assert sql == (
        "SELECT e.name, e.mention_count, t.head FROM kg_triples AS t "
        "JOIN kg_entities AS e ON e.name = t.tail "
        "WHERE t.relation_zh = '呈现症状' AND e.mention_count >= 3 "
        "AND t.confidence >= 0.8 AND t.count >= 4 "
        "ORDER BY e.mention_count DESC, e.name ASC, t.head ASC, t.tail ASC"
    )


def test_grounded_tail_join_abstains_for_ambiguous_edge_bound():
    assert (
        compile_kg_medical_sql(
            "查询呈现症状关系的尾实体，尾实体提及次数至少3，"
            "置信度至少0.7和0.8，返回实体名称、提及次数、头实体。",
            DOMAINS,
        )
        is None
    )


def test_grounded_edge_filter_retains_two_sided_bounds_and_projection_order():
    sql = compile_kg_medical_sql(
        "针对呈现症状三元组边，置信度至少0.7且不超过0.9，"
        "观察次数在2至5之间，返回尾实体、观察次数、头实体。",
        DOMAINS,
    )

    assert sql is not None
    assert sql.startswith("SELECT tail, count, head FROM kg_triples")
    assert "confidence >= 0.7" in sql
    assert "confidence <= 0.9" in sql
    assert "count >= 2" in sql
    assert "count <= 5" in sql


def test_grounded_edge_filter_accepts_one_metric_bound_and_keeps_head_type():
    sql = compile_kg_medical_sql(
        "返回头实体类型为Disease、中文关系为呈现症状且置信度至少0.5的"
        "头实体、尾实体和观察次数。",
        {**DOMAINS, "kg_head_type": ["Disease"]},
    )

    assert sql == (
        "SELECT head, tail, count FROM kg_triples "
        "WHERE relation_zh = '呈现症状' AND head_type = 'Disease' "
        "AND confidence >= 0.5 "
        "ORDER BY count DESC, confidence DESC, head ASC, tail ASC"
    )


def test_entity_required_relations_bind_named_repeated_confidence_and_count_bounds():
    sql = compile_kg_medical_sql(
        "作为头实体且类型为Disease的实体同时具有呈现症状和接受治疗关系，"
        "呈现症状关系置信度至少0.7且不超过0.9且观察次数至少2且不超过5，"
        "接受治疗关系置信度至少0.8且不超过0.95且观察次数至少3且不超过6，"
        "返回实体名称、类型和提及次数。",
        DOMAINS,
    )

    assert sql is not None
    for predicate in (
        "t1.confidence >= 0.7",
        "t1.confidence <= 0.9",
        "t1.count >= 2",
        "t1.count <= 5",
        "t2.confidence >= 0.8",
        "t2.confidence <= 0.95",
        "t2.count >= 3",
        "t2.count <= 6",
    ):
        assert predicate in sql


def test_entity_required_relations_abstains_for_unowned_order_and_top_k():
    assert (
        compile_kg_medical_sql(
            "作为头实体且类型为Disease的实体同时具有呈现症状和接受治疗关系，"
            "呈现症状关系置信度至少0.7且接受治疗关系置信度至少0.8，"
            "返回实体名称，按实体名称降序排列，前2条。",
            DOMAINS,
        )
        is None
    )


def test_required_all_routes_abstain_for_alternate_top_k_phrases():
    questions = (
        (
            "作为头实体且类型为Disease的实体同时具有呈现症状和接受治疗关系，"
            "呈现症状关系置信度至少0.7且接受治疗关系置信度至少0.8，"
            "返回实体名称；最多2条。"
        ),
        (
            "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
            "每条关系边置信度至少0.8且观察次数至少4，按地区分组，"
            "返回地区、就诊次数和不同疾病数；最多2条。"
        ),
    )

    for question in questions:
        assert not analyze_relation_request(question, DOMAINS).top_k_spans
        assert compile_kg_medical_sql(question, DOMAINS) is None


def test_entity_required_relations_abstains_for_duplicate_declaration():
    assert (
        compile_kg_medical_sql(
            "作为头实体且类型为Disease的实体同时具有呈现症状、呈现症状和接受治疗关系，"
            "每条关系边置信度均至少0.8，返回实体名称、类型和提及次数。",
            DOMAINS,
        )
        is None
    )


def test_entity_required_relations_abstains_for_ordinal_chain_with_ordered_bounds():
    question = (
        "查询两跳关系链，从头实体到尾实体，作为头实体且类型为Disease的实体"
        "同时具有呈现症状和接受治疗关系，第一条边和第二条边置信度分别至少0.7和0.8，"
        "返回实体名称、类型和提及次数。"
    )
    request = analyze_relation_request(question, DOMAINS)

    assert request.connective == "ordered"
    assert compile_kg_medical_sql(question, DOMAINS) is None


def test_two_hop_chain_binds_reversed_named_relation_qualifiers_to_roles():
    sql = compile_kg_medical_sql(
        "查询两跳关系链，从头实体到尾实体，第一跳为呈现症状、第二跳为属于科室，"
        "属于科室关系置信度至少0.8且观察次数至少5，"
        "呈现症状关系置信度至少0.7且观察次数至少3，"
        "返回源疾病、中间实体和目标科室。",
        DOMAINS,
    )

    assert sql is not None
    assert "t1.relation_zh = '呈现症状'" in sql
    assert "t1.confidence >= 0.7" in sql
    assert "t1.count >= 3" in sql
    assert "t2.relation_zh = '属于科室'" in sql
    assert "t2.confidence >= 0.8" in sql
    assert "t2.count >= 5" in sql


def test_risk_chain_patient_filters_join_the_patients_table():
    sql = compile_kg_medical_sql(
        "年龄至少40岁的女性深圳患者风险链就诊，从头实体到尾实体，"
        "呈现症状和接受治疗，边置信度分别至少0.7和0.8，"
        "返回就诊ID、疾病、风险实体和风险结果。",
        DOMAINS,
    )

    assert sql is not None
    assert "FROM visits AS v JOIN patients AS p ON p.patient_id = v.patient_id" in sql
    assert "p.age >= 40" in sql
    assert "p.gender = '女'" in sql
    assert "p.region = '深圳'" in sql


def test_patient_kg_union_retains_grounded_visit_filters_in_patient_arm():
    sql = compile_kg_medical_sql(
        "将年龄至少40岁女性深圳患者在2025-01-01至2025-12-31因疾病甲就诊"
        "与Disease图谱实体UNION成统一列表，返回名称和来源。",
        DOMAINS,
    )

    assert sql is not None
    assert "EXISTS (SELECT 1 FROM visits AS v WHERE v.patient_id = p.patient_id" in sql
    assert "v.disease = '疾病甲'" in sql
    assert "v.visit_date >= '2025-01-01'" in sql
    assert "v.visit_date <= '2025-12-31'" in sql


def test_tail_coverage_consumes_relation_bound_metrics():
    sql = compile_kg_medical_sql(
        "按关系统计呈现症状的尾实体，尾实体提及次数高于同类型平均，"
        "不同头实体至少2个，呈现症状关系置信度至少0.8且不超过0.9，"
        "观察次数至少4且不超过6，返回关系中文名、不同尾实体数和不同头实体数。",
        DOMAINS,
    )

    assert sql is not None
    assert "t.relation_zh = '呈现症状'" in sql
    assert "t.confidence >= 0.8" in sql
    assert "t.confidence <= 0.9" in sql
    assert "t.count >= 4" in sql
    assert "t.count <= 6" in sql


def test_older_enrichment_consumes_relation_bound_metrics():
    sql = compile_kg_medical_sql(
        "年龄至少65岁的老年患者疾病头实体图谱富集，"
        "呈现症状关系置信度至少0.8且不超过0.9，"
        "观察次数至少4且不超过6，"
        "返回疾病、老年患者数、图谱提及次数和不同关系数。",
        DOMAINS,
    )

    assert sql is not None
    assert "t.relation_zh = '呈现症状'" in sql
    assert "t.confidence >= 0.8" in sql
    assert "t.confidence <= 0.9" in sql
    assert "t.count >= 4" in sql
    assert "t.count <= 6" in sql


def test_generic_and_bespoke_tail_joins_reject_unmappable_projections():
    assert (
        compile_kg_medical_sql(
            "将尾实体连接到kg_entities，尾实体提及次数至少3，"
            "返回实体名称、提及次数、头实体和未知指标。",
            DOMAINS,
        )
        is None
    )
    assert (
        compile_kg_medical_sql(
            "查询呈现症状关系的尾实体，尾实体提及次数至少3，"
            "置信度至少0.8且观察次数至少4，"
            "返回实体名称、提及次数、头实体和未知指标。",
            DOMAINS,
        )
        is None
    )


def test_generic_raw_edge_union_preserves_bounds_and_top_k():
    sql = compile_kg_medical_sql(
        "针对呈现症状或接受治疗关系三元组边，置信度低于0.8且观察次数不超过4，"
        "返回头实体、尾实体、置信度和观察次数，排名前3。",
        DOMAINS,
    )

    assert sql is not None
    assert "relation_zh IN ('呈现症状', '接受治疗')" in sql
    assert "confidence < 0.8" in sql
    assert "count <= 4" in sql
    assert sql.endswith("LIMIT 3")


def test_generic_tail_join_admits_only_raw_relation_union_span():
    sql = compile_kg_medical_sql(
        "将呈现症状或接受治疗关系的尾实体连接到kg_entities，"
        "尾实体提及次数至少3且不超过5，置信度至少0.8且观察次数至少4，"
        "返回实体名称、提及次数、头实体和观察次数。",
        DOMAINS,
    )

    assert sql is not None
    assert "t.relation_zh IN ('呈现症状', '接受治疗')" in sql
    assert "e.mention_count >= 3" in sql
    assert "e.mention_count <= 5" in sql
    assert "t.confidence >= 0.8" in sql
    assert "t.count >= 4" in sql


def test_generic_per_head_top_k_admits_raw_relation_union_span():
    sql = compile_kg_medical_sql(
        "按每个头实体的呈现症状或接受治疗关系统计，置信度至少0.8且观察次数至少4，"
        "返回头实体、不同尾实体数和观察次数总和，排名前3。",
        DOMAINS,
    )

    assert sql is not None
    assert "relation_zh IN ('呈现症状', '接受治疗')" in sql
    assert "confidence >= 0.8" in sql
    assert "count >= 4" in sql
    assert sql.endswith("LIMIT 3")


def test_generic_flat_relation_union_rejects_ambiguous_ownership():
    questions = (
        "针对呈现症状和接受治疗关系三元组边，置信度至少0.8且观察次数至少4，"
        "返回头实体、尾实体、置信度和观察次数。",
        "针对呈现症状、接受治疗关系三元组边，置信度至少0.8且观察次数至少4，"
        "返回头实体、尾实体、置信度和观察次数。",
        "针对呈现症状或接受治疗关系三元组边，每条关系置信度至少0.8且观察次数至少4，"
        "返回头实体、尾实体、置信度和观察次数。",
        "针对呈现症状或接受治疗关系三元组边，呈现症状关系置信度至少0.7且"
        "接受治疗关系置信度至少0.8且观察次数至少4，"
        "返回头实体、尾实体、置信度和观察次数。",
        "针对呈现症状或接受治疗关系三元组边，置信度至少0.7或不超过0.8且"
        "观察次数至少4，返回头实体、尾实体、置信度和观察次数。",
    )

    for question in questions:
        assert compile_kg_medical_sql(question, DOMAINS) is None


def test_grounded_tail_join_retains_all_mention_bounds():
    sql = compile_kg_medical_sql(
        "查询呈现症状关系的尾实体，尾实体提及次数至少3且不超过5，"
        "置信度至少0.8且观察次数至少4，返回实体名称、提及次数、头实体。",
        DOMAINS,
    )

    assert sql is not None
    assert "e.mention_count >= 3" in sql
    assert "e.mention_count <= 5" in sql


def test_required_relation_visits_retains_all_visit_metric_bounds():
    sql = compile_kg_medical_sql(
        "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
        "每条关系边置信度至少0.8且观察次数至少4，"
        "费用至少100且不超过500，住院天数至少2且不超过5，"
        "按地区分组，返回地区、就诊次数和不同疾病数。",
        DOMAINS,
    )

    assert sql is not None
    for predicate in (
        "v.cost >= 100",
        "v.cost <= 500",
        "v.los_days >= 2",
        "v.los_days <= 5",
    ):
        assert predicate in sql


def test_required_relation_visits_expands_visit_metric_ranges():
    sql = compile_kg_medical_sql(
        "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
        "每条关系边置信度至少0.8且观察次数至少4，"
        "费用在100至500之间，住院天数在2至5之间，"
        "按地区分组，返回地区、就诊次数和不同疾病数。",
        DOMAINS,
    )

    assert sql is not None
    for predicate in (
        "v.cost >= 100",
        "v.cost <= 500",
        "v.los_days >= 2",
        "v.los_days <= 5",
    ):
        assert predicate in sql


def test_required_relation_visits_abstains_for_unowned_order_and_top_k():
    assert (
        compile_kg_medical_sql(
            "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
            "每条关系边置信度至少0.8且观察次数至少4，按地区分组，"
            "返回地区、就诊次数和不同疾病数，按地区降序排列，前2条。",
            DOMAINS,
        )
        is None
    )


def test_required_relation_visits_abstains_for_disjunctive_visit_metric_bound():
    assert (
        compile_kg_medical_sql(
            "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
            "每条关系边置信度至少0.8且观察次数至少4，"
            "费用至少100或不超过500，按地区分组，"
            "返回地区、就诊次数和不同疾病数。",
            DOMAINS,
        )
        is None
    )


def test_grounded_edge_filter_preserves_rank_and_strict_equality_bounds():
    sql = compile_kg_medical_sql(
        "针对呈现症状三元组边，置信度等于0.8且观察次数小于4，"
        "返回头实体、尾实体和置信度，排名前3。",
        DOMAINS,
    )

    assert sql is not None
    assert "confidence = 0.8" in sql
    assert "count < 4" in sql
    assert sql.endswith("LIMIT 3")


def test_tail_joins_retain_range_followed_by_explicit_mention_bound():
    grounded_sql = compile_kg_medical_sql(
        "查询呈现症状关系的尾实体，尾实体提及次数在2至8之间且不超过5，"
        "置信度至少0.8且观察次数至少4，返回实体名称、提及次数、头实体。",
        DOMAINS,
    )
    union_sql = compile_kg_medical_sql(
        "将呈现症状或接受治疗关系的尾实体连接到kg_entities，"
        "尾实体提及次数在2至8之间且不超过5，"
        "置信度至少0.8且观察次数至少4，"
        "返回实体名称、提及次数、头实体和观察次数。",
        DOMAINS,
    )

    for sql in (grounded_sql, union_sql):
        assert sql is not None
        assert "e.mention_count >= 2" in sql
        assert "e.mention_count <= 8" in sql
        assert "e.mention_count <= 5" in sql


def test_entity_type_recognition_retains_complete_mention_range_bounds():
    sql = compile_kg_medical_sql(
        "识别呈现症状关系的尾实体，尾实体类型为Exam，"
        "提及次数在2至8之间且不超过5，返回实体名称、类型和提及次数。",
        DOMAINS,
    )

    assert sql is not None
    assert "e.mention_count BETWEEN 2 AND 8" in sql
    assert "e.mention_count <= 5" in sql


def test_entity_type_recognition_abstains_for_incomplete_mention_range_bounds():
    for suffix in ("或不超过5", "且不超过"):
        assert (
            compile_kg_medical_sql(
                "识别呈现症状关系的尾实体，尾实体类型为Exam，"
                f"提及次数在2至8之间{suffix}，"
                "返回实体名称、类型和提及次数。",
                DOMAINS,
            )
            is None
        )


def test_required_relation_visits_abstains_for_english_edge_metric_or():
    assert (
        compile_kg_medical_sql(
            "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
            "每条关系边置信度至少0.8 OR 不超过0.9且观察次数至少4，"
            "按地区分组，返回地区、就诊次数和不同疾病数。",
            DOMAINS,
        )
        is None
    )


def test_row_level_metric_compilers_reject_aggregate_modifiers():
    edge_questions = (
        "针对呈现症状三元组边，平均置信度至少0.8且观察次数至少4，"
        "返回头实体、尾实体和置信度。",
        "针对呈现症状三元组边，置信度至少0.8且总观察次数至少4，"
        "返回头实体、尾实体和置信度。",
    )
    visit_questions = (
        "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
        "每条关系边置信度至少0.8且观察次数至少4，平均费用至少100，"
        "按地区分组，返回地区、就诊次数和不同疾病数。",
        "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
        "每条关系边置信度至少0.8且观察次数至少4，总费用至少100，"
        "按地区分组，返回地区、就诊次数和不同疾病数。",
    )

    for question in (*edge_questions, *visit_questions):
        assert compile_kg_medical_sql(question, DOMAINS) is None


def test_generic_flat_relation_union_rejects_containment_ownership():
    assert (
        compile_kg_medical_sql(
            "针对包含呈现症状或接受治疗关系三元组边，置信度至少0.8且观察次数至少4，"
            "返回头实体、尾实体、置信度和观察次数。",
            DOMAINS,
        )
        is None
    )


def test_required_relation_visits_rejects_explicit_relation_disjunction():
    assert (
        compile_kg_medical_sql(
            "具有呈现症状或接受治疗关系的疾病头实体患者就诊，"
            "按地区分组，返回地区、就诊次数和不同疾病数。",
            DOMAINS,
        )
        is None
    )


def test_visit_metric_disjunction_is_rejected_across_distant_text():
    distant_context = "这里有一段足够长的附加说明，用于确保费用条件之间相隔超过旧的局部检查窗口。"
    for connector in ("或", " OR "):
        risk_question = (
            "风险链就诊，从头实体到尾实体，呈现症状和接受治疗，"
            "边置信度分别至少0.7和0.8，费用至少100，"
            f"{distant_context}{connector}不超过500，"
            "返回就诊ID、疾病、风险实体和风险结果。"
        )
        union_question = (
            "将年龄至少40岁女性深圳患者在费用至少100，"
            f"{distant_context}{connector}不超过500时就诊"
            "与Disease图谱实体UNION成统一列表，返回名称和来源。"
        )

        # The risk-chain form protects the graph route; the UNION form reaches
        # the visit predicate compiler directly, so neither distant branch may
        # be rewritten as a single lower-bound predicate.
        assert compile_kg_medical_sql(risk_question, DOMAINS) is None
        assert compile_kg_medical_sql(union_question, DOMAINS) is None


def test_row_level_metric_compilers_reject_total_cost_amount_modifier():
    assert (
        compile_kg_medical_sql(
            "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
            "每条关系边置信度至少0.8且观察次数至少4，费用总额至少100，"
            "按地区分组，返回地区、就诊次数和不同疾病数。",
            DOMAINS,
        )
        is None
    )


def test_structured_flat_relation_union_rejects_partial_and_set_ownership():
    suffix = "置信度至少0.8且观察次数至少4，返回头实体、尾实体、置信度和观察次数。"
    questions = (
        f"针对“呈现症状”或接受治疗关系三元组边，{suffix}",
        f"针对呈现症状或接受治疗关系交集三元组边，{suffix}",
        f"针对呈现症状或接受治疗关系并集三元组边，{suffix}",
        f"针对呈现症状或接受治疗关系差集三元组边，{suffix}",
        f"针对呈现症状或接受治疗关系集合三元组边，{suffix}",
        f"针对呈现症状或接受治疗关系三元组边，第一个关系{suffix}",
        f"针对呈现症状或接受治疗关系三元组边，前者{suffix}",
    )

    for question in questions:
        assert compile_kg_medical_sql(question, DOMAINS) is None


def test_structured_flat_relation_union_preserves_range_and_punctuation_bounds():
    question = (
        "针对呈现症状或接受治疗关系三元组边，置信度介于0.7与0.8之间；"
        "观察次数至少4，不超过8，返回头实体、尾实体、置信度和观察次数。"
    )
    request = analyze_relation_request(question, DOMAINS)
    assert request.connective == "literal_or"
    assert [
        (bound.metric, bound.operator, bound.value)
        for bound in request.metric_bounds
    ] == [
        ("confidence", ">=", "0.7"),
        ("confidence", "<=", "0.8"),
        ("count", ">=", "4"),
        ("count", "<=", "8"),
    ]
    assert request.metric_complete
    assert request.errors == ()
    assert request.accepts_flat_predicate()
    sql = compile_kg_medical_sql(question, DOMAINS)

    assert sql is not None
    for predicate in (
        "relation_zh IN ('呈现症状', '接受治疗')",
        "confidence >= 0.7",
        "confidence <= 0.8",
        "count >= 4",
        "count <= 8",
    ):
        assert predicate in sql


def test_grounded_edge_filter_abstains_for_collection_first_edge_request():
    question = (
        "针对呈现症状关系集合中的第一条边，置信度至少0.8且观察次数至少4，"
        "返回头实体、尾实体。"
    )
    request = analyze_relation_request(question, DOMAINS)

    assert request.connective == "set_operation"
    assert compile_kg_medical_sql(question, DOMAINS) is None


def test_flat_edge_filters_abstain_for_unowned_custom_order():
    question = (
        "针对呈现症状三元组边，置信度至少0.8且观察次数至少4，"
        "返回头实体、尾实体，按头实体升序排列。"
    )
    request = analyze_relation_request(question, DOMAINS)

    assert request.ordering_spans
    assert compile_kg_medical_sql(question, DOMAINS) is None


def test_fixed_shape_routes_recognize_unmarked_ordering():
    base = (
        "查询两跳关系链，从头实体到尾实体，第一跳为呈现症状、第二跳为属于科室，"
        "第一条边置信度至少0.7且第二条边置信度至少0.8，"
        "返回源疾病、中间实体和目标科室；"
    )

    for ordering in (
        "目标科室从高到低。",
        "目标科室排序。",
        "目标科室从高到低？",
        "目标科室排序！",
    ):
        question = base + ordering
        request = analyze_relation_request(question, DOMAINS)

        assert request.ordering_spans
        assert compile_kg_medical_sql(question, DOMAINS) is None


def test_fixed_shape_kg_routes_abstain_for_unowned_order_and_top_k():
    cases = (
        (
            "按关系统计尾实体提及次数高于同类型平均且不同头实体至少2个，"
            "返回关系中文名、不同尾实体数和不同头实体数",
            "按关系中文名降序排列，前2条。",
        ),
        (
            "查询两跳关系链，从头实体到尾实体，第一跳为呈现症状、第二跳为属于科室，"
            "第一条边置信度至少0.7且第二条边置信度至少0.8，"
            "返回源疾病、中间实体和目标科室",
            "按目标科室从高到低，前2条。",
        ),
        (
            "疾病头实体共享尾实体的疾病对，两个置信度均至少0.8且共享尾实体数至少2，"
            "返回疾病对、疾病二和共享尾实体数",
            "按疾病二降序排列，前2条。",
        ),
        (
            "按地区统计知识图谱覆盖率就诊，疾病头实体，"
            "返回地区、总就诊数、覆盖就诊数和覆盖率",
            "按地区降序排列，前2条。",
        ),
        (
            "风险链就诊，从头实体到尾实体，呈现症状和接受治疗，"
            "边置信度分别至少0.7和0.8，返回就诊ID、疾病、风险实体和风险结果",
            "按疾病降序排列，前2条。",
        ),
        (
            "年龄至少65岁的老年患者疾病头实体图谱富集，"
            "返回疾病、老年患者数和就诊次数",
            "按疾病降序排列，前2条。",
        ),
        (
            "识别呈现症状关系的尾实体，尾实体类型为Exam，提及次数在2至8之间，"
            "返回实体名称、类型和提及次数",
            "按实体名称降序排列，前2条。",
        ),
        (
            "将年龄至少40岁女性深圳患者与Disease图谱实体UNION成统一列表，"
            "返回名称和来源",
            "按名称降序排列，前2条。",
        ),
    )

    for base, modifiers in cases:
        question = f"{base}，{modifiers}"
        request = analyze_relation_request(question, DOMAINS)

        assert compile_kg_medical_sql(f"{base}。", DOMAINS) is not None
        assert request.ordering_spans
        assert request.top_k_spans
        assert compile_kg_medical_sql(question, DOMAINS) is None


def test_repeated_relation_two_hop_preserves_distinct_edge_roles():
    question = (
        "查询两跳关系链，从头实体到尾实体，第一跳为呈现症状、第二跳为呈现症状，"
        "第一条边置信度至少0.7且第二条边置信度至少0.8，"
        "第一条边观察次数至少3且第二条边观察次数至少5，"
        "返回源疾病、中间实体和目标科室。"
    )
    request = analyze_relation_request(question, DOMAINS)
    sql = compile_kg_medical_sql(question, DOMAINS)

    assert request.values == ("呈现症状", "呈现症状")
    assert request.connective == "ordered"
    assert sql is not None
    for predicate in (
        "t1.relation_zh = '呈现症状'",
        "t2.relation_zh = '呈现症状'",
        "t1.confidence >= 0.7",
        "t2.confidence >= 0.8",
        "t1.count >= 3",
        "t2.count >= 5",
    ):
        assert predicate in sql


def test_full_width_or_is_owned_only_as_a_complete_literal_relation_union():
    question = (
        "针对呈现症状ＯＲ接受治疗关系三元组边，置信度至少0.8且观察次数至少4，"
        "返回头实体、尾实体、置信度和观察次数。"
    )
    request = analyze_relation_request(question, DOMAINS)
    sql = compile_kg_medical_sql(question, DOMAINS)

    assert request.connective == "literal_or"
    assert request.errors == ()
    assert sql is not None
    assert "relation_zh IN ('呈现症状', '接受治疗')" in sql


def test_full_width_metric_or_abstains_instead_of_dropping_one_branch():
    question = (
        "针对呈现症状三元组边，置信度至少0.7 ＯＲ 不超过0.8且观察次数至少4，"
        "返回头实体、尾实体、置信度和观察次数。"
    )
    request = analyze_relation_request(question, DOMAINS)

    assert "unowned disjunction" in request.errors
    assert compile_kg_medical_sql(question, DOMAINS) is None


def test_row_level_edge_filter_abstains_for_post_average_confidence():
    question = (
        "针对呈现症状三元组边，平均后的置信度至少0.8且观察次数至少4，"
        "返回头实体、尾实体、置信度和观察次数。"
    )
    request = analyze_relation_request(question, DOMAINS)

    assert not request.metric_complete
    assert compile_kg_medical_sql(question, DOMAINS) is None


def test_grouped_visit_cost_full_width_or_abstains_instead_of_retaining_one_bound():
    question = (
        "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
        "每条关系边置信度至少0.8且观察次数至少4，费用至少100 ＯＲ 不超过500，"
        "按地区分组，返回地区、就诊次数和不同疾病数。"
    )

    assert compile_kg_medical_sql(question, DOMAINS) is None


def test_tail_coverage_abstains_for_disjunctive_having_threshold():
    question = (
        "按关系统计尾实体提及次数高于同类型平均且不同头实体至少2个ＯＲ不超过5个，"
        "返回关系中文名、不同尾实体数和不同头实体数。"
    )

    assert compile_kg_medical_sql(question, DOMAINS) is None


def test_kg_only_routes_abstain_for_unowned_clinical_scope():
    questions = (
        (
            "在女性患者队列中，按关系统计尾实体提及次数高于同类型平均且不同头实体至少2个，"
            "返回关系中文名、不同尾实体数和不同头实体数。"
        ),
        (
            "在女性患者中查询两跳关系链，从头实体到尾实体，"
            "第一跳为呈现症状、第二跳为属于科室，"
            "第一条边置信度至少0.7且第二条边置信度至少0.8，"
            "返回源疾病、中间实体和目标科室。"
        ),
        (
            "在女性患者队列中，作为头实体且类型为Disease的实体"
            "同时具有呈现症状和接受治疗关系，"
            "两条边置信度均至少0.8，返回实体名称、类型和提及次数。"
        ),
        (
            "针对女性患者的呈现症状三元组边，置信度至少0.8且观察次数至少4，"
            "返回头实体、尾实体、置信度和观察次数。"
        ),
        (
            "针对呈现症状三元组边，置信度至少0.8且观察次数至少4且费用至少100，"
            "返回头实体、尾实体、置信度和观察次数。"
        ),
    )

    for question in questions:
        assert compile_kg_medical_sql(question, DOMAINS) is None


def test_visit_routes_consume_only_complete_date_scope():
    base = (
        "具有呈现症状和接受治疗关系的疾病头实体患者就诊，"
        "每条关系边置信度至少0.8且观察次数至少4，"
        "按地区分组，返回地区、就诊次数和不同疾病数。"
    )
    supported = base.replace("按地区分组", "自2025-01-01以来按地区分组")
    sql = compile_kg_medical_sql(supported, DOMAINS)

    assert sql is not None
    assert "v.visit_date >= '2025-01-01'" in sql

    for date_scope in (
        "在2025-01-01按地区分组",
        "在2025-01-01、2025-02-01和2025-03-01按地区分组",
    ):
        assert compile_kg_medical_sql(base.replace("按地区分组", date_scope), DOMAINS) is None
