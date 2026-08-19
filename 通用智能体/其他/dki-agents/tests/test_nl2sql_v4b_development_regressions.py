"""Selection-used v4b development regressions for deterministic compilers."""
from __future__ import annotations

import pytest

from dki.analysis.nl2sql_independent_v7 import compile_independent_v7_sql
from dki.analysis.nl2sql_semantic import compile_semantic_medical_result
from dki.analysis.nl2sql_semantic_families import compile_semantic_family_sql
from dki.analysis.nl2sql_semantic_simple import compile_simple_medical_sql


DOMAINS = {
    "region": ["深圳", "北京", "成都", "南京", "上海", "西安"],
    "department": ["心血管内科", "消化内科", "骨科", "泌尿外科"],
    "disease": ["胃溃疡", "消化道出血"],
    "drug": ["克拉霉素"],
    "exam": ["胃镜"],
    "kg_entity_type": ["Disease", "Drug", "Symptom", "Exam"],
    "kg_relation_zh": [
        "临床表现",
        "治疗药物",
        "诊断检查",
        "并发症",
        "用药禁忌",
        "所属科室",
        "治疗方式",
    ],
}


def test_patient_age_desc_order_survives_later_tie_clause():
    sql = compile_simple_medical_sql(
        "请按年龄从高到低列出北京男性患者中满35岁且不超过75岁的人员，"
        "并返回编号、姓名、年龄；年龄相同按姓名、编号升序。",
        DOMAINS,
    )

    assert sql is not None
    assert sql.endswith("ORDER BY age DESC, name ASC, patient_id ASC")


def test_patient_age_range_projection_keeps_full_shape_and_desc_order():
    # v4b_0001: 行级患者投影必须完整返回编号/姓名/年龄，且"年龄较大者在前"
    # 等价于年龄降序、同龄按姓名和编号升序稳定排序。此形状曾因投影缺 name
    # 且排序方向错误而被契约校验拦截并整题弃权。
    sql = compile_independent_v7_sql(
        "找出深圳地区年龄在40至70岁之间的女性患者，"
        "给出患者编号、姓名和年龄；先列年龄较大者，同龄时按姓名和编号排序。",
        DOMAINS,
    )

    assert sql is not None
    assert "SELECT patient_id, name, age FROM patients" in sql
    assert "gender='女'" in sql
    assert "region='深圳'" in sql
    assert "age BETWEEN 40 AND 70" in sql
    assert sql.endswith("ORDER BY age DESC, name ASC, patient_id ASC")


def test_patient_age_range_projection_lower_age_first_uses_ascending_order():
    # 反向措辞（"年龄较小者在前"）应切换到年龄升序，同时保留稳定排序。
    sql = compile_independent_v7_sql(
        "找出深圳地区年龄在40至70岁之间的女性患者，"
        "给出患者编号、姓名和年龄；先列年龄较小者，同龄时按姓名和编号排序。",
        DOMAINS,
    )

    assert sql is not None
    assert "SELECT patient_id, name, age FROM patients" in sql
    assert sql.endswith("ORDER BY age ASC, name ASC, patient_id ASC")


def test_patient_group_parses_compact_average_minimum_maximum_age_list():
    sql = compile_simple_medical_sql(
        "年龄达到30岁的患者，按所在地区与性别拆分后各有多少人，"
        "平均、最小和最大年龄是多少？平均值取一位小数，先显示大组。",
        DOMAINS,
    )

    assert sql is not None
    assert "COUNT(*) AS patient_count" in sql
    assert "ROUND(AVG(age), 1) AS avg_age" in sql
    assert "MIN(age) AS min_age" in sql
    assert "MAX(age) AS max_age" in sql
    assert sql.endswith("ORDER BY patient_count DESC, region ASC, gender ASC")


def test_patient_group_retains_each_region_and_each_gender_dimensions():
    sql = compile_simple_medical_sql(
        "每个地区每种性别的患者数和平均年龄分别是多少",
        DOMAINS,
    )

    assert sql is not None
    assert sql.startswith(
        "SELECT region, gender, COUNT(*) AS patient_count, "
        "AVG(age) AS avg_age FROM patients"
    )
    assert "GROUP BY region, gender" in sql
    assert sql.endswith("ORDER BY patient_count DESC, region ASC, gender ASC")


def test_top_visit_detail_phrase_keeps_complete_six_column_shape():
    sql = compile_simple_medical_sql(
        "请给出从2025-07-01到2026-06-30花费排名前10的就诊明细，"
        "需带患者姓名、科室和疾病；最高费用优先，同价时较早记录优先。",
        DOMAINS,
    )

    assert sql is not None
    assert sql.startswith(
        "SELECT v.visit_id, v.visit_date, p.name, v.department, v.disease, v.cost "
    )
    assert sql.endswith(
        "ORDER BY v.cost DESC, v.visit_date ASC, v.visit_id ASC LIMIT 10"
    )


def test_disease_topk_keeps_primary_count_order_before_cost_tie_break():
    sql = compile_simple_medical_sql(
        "统计2025-01-01至2025-12-31就诊次数最多的前7种疾病，"
        "返回疾病、次数、不同患者数和总费用；总费用保留两位，"
        "次数相同按总费用降序。",
        DOMAINS,
    )

    assert sql is not None
    assert sql.endswith(
        "ORDER BY visit_count DESC, total_cost DESC, disease ASC LIMIT 7"
    )


def test_region_exclusion_with_explicit_copula_is_negative():
    sql = compile_simple_medical_sql(
        "请列出50岁及以上的张姓患者，但排除地区为上海者；"
        "显示编号、姓名、年龄、地区，年长者优先。",
        DOMAINS,
    )

    assert sql is not None
    assert "region NOT IN ('上海')" in sql
    assert "region = '上海'" not in sql


def test_cohort_department_and_retention_use_unaliased_age_inside_patient_cte():
    department_sql = compile_semantic_family_sql(
        "先定义深圳、女性且至少50岁的患者队列，再汇总他们自2025-01-01起"
        "在各科室的就诊次数、不同患者数和总费用；金额两位。",
        DOMAINS,
    )
    retention_sql = compile_semantic_family_sql(
        "对50岁及以上女性患者，以个人首次就诊月为队列月，统计首次月后"
        "0至6个月各偏移月仍有就诊的人数、队列规模和留存率。",
        DOMAINS,
    )

    for sql in (department_sql, retention_sql):
        assert sql is not None
        assert "FROM patients WHERE" in sql
        assert "p.age" not in sql
        assert "age >= 50" in sql


@pytest.mark.parametrize(
    ("question", "marker"),
    [
        (
            "请生成2024-08-01—2026-05-31月度费用同比序列：每月与前推12个月的"
            "月总额作差，只显示存在对比月的月份，输出三项金额。",
            "LAG(total_cost, 12)",
        ),
        (
            "对2025-01-01至2025-12-31每个科室分别取费用最高的前2次就诊，"
            "返回科室内名次、编号、日期、疾病、费用。",
            "ROW_NUMBER() OVER (PARTITION BY department",
        ),
        (
            "查找疾病经并发症再到该并发症治疗药物的两跳路径，要求两条边置信度"
            "均至少0.9且计数均不少于1；返回原疾病、并发症、药物和两边置信度。",
            "JOIN kg_triples AS d",
        ),
        (
            "疾病对共同症状重叠，共同症状至少2，两条边置信度均至少0.9，"
            "返回疾病一、疾病二和共享症状数。",
            "a.head < b.head",
        ),
        (
            "临床表现、治疗药物和诊断检查条件计数覆盖审计，症状数至少2，"
            "治疗药物数至少1，诊断检查数至少2，输出疾病及三项去重计数。",
            "COUNT(DISTINCT CASE WHEN relation_zh = '临床表现'",
        ),
    ],
)
def test_fixed_families_accept_only_their_compatible_explicit_shape(question, marker):
    sql = compile_semantic_family_sql(question, DOMAINS)

    assert sql is not None
    assert marker in sql


def test_fixed_department_topk_still_rejects_an_extra_drug_projection():
    question = (
        "在2025-01-01至2025-12-31期间，每个科室按费用降序取前1，"
        "返回科室、就诊编号、处方药物和排名。"
    )

    assert compile_semantic_family_sql(question, DOMAINS) is None


@pytest.mark.parametrize(
    ("question", "marker"),
    [
        (
            "列出知识图谱中“临床表现”关系里置信度至少0.98且出现次数"
            "不少于3的边，返回头实体、尾实体、置信度和次数，按次数、置信度降序。",
            "SELECT head, tail, confidence, count",
        ),
        (
            "查看“临床表现”关系中尾实体提及次数至少10的连接，返回头实体、"
            "尾实体、尾实体类型、提及次数和关系置信度，按提及次数降序。",
            "JOIN kg_entities AS ke",
        ),
        (
            "找出知识图谱中“临床表现”尾实体种类最多的前6个头实体，返回"
            "头实体、不同尾实体数、关系计数合计，按尾实体数降序。",
            "COUNT(DISTINCT tail) AS distinct_tail_count",
        ),
    ],
)
def test_grounded_kg_listing_families(question, marker):
    sql = compile_semantic_family_sql(question, DOMAINS)

    assert sql is not None
    assert marker in sql


@pytest.mark.parametrize(
    ("question", "marker"),
    [
        (
            "请筛选治疗药物关系：置信度下限为0.99、计数下限为2。"
            "输出起点、终点、置信度、计数，计数高者优先。",
            "confidence >= 0.99 AND count >= 2",
        ),
        (
            "知识图谱的诊断检查连接里，哪些尾端实体至少被提及4次？"
            "列出头实体、尾实体、类型、提及次数、置信度。",
            "ke.mention_count >= 4",
        ),
        (
            "按科室汇总2025-07-01到2026-06-30的记录，只保留至少6次且"
            "平均花费超过3000元的组；输出科室、次数、均值、总额，金额两位。",
            "HAVING COUNT(*) >= 6 AND AVG(v.cost) > 3000",
        ),
        (
            "哪些至少就诊4次的患者，个人累计费用高于所在地区有就诊患者的"
            "平均累计费用？返回基本信息、次数和总额。",
            "s.total_cost > (SELECT AVG(region_total)",
        ),
        (
            "疾病实体需同时满足：有置信度≥0.9的并发症边，也有同下限的"
            "治疗方式边。返回名称、提及次数，并按热度从高到低。",
            "t2.relation_zh = '治疗方式'",
        ),
        (
            "在“临床表现”关系中，找出至少连接3种不同疾病、且提及次数高于"
            "同类型实体平均值的尾实体；返回尾实体、类型、疾病数、提及量。",
            "ke2.type = kt.tail_type",
        ),
        (
            "生成统一名称列表，一侧是西安且年龄不低于55岁的患者，另一侧是"
            "至少提及15次的症状实体。输出名称及来源标签，按名称和来源排列；"
            "症状实体按在“临床表现”关系中作为尾实体来识别。",
            "UNION SELECT name, '知识实体' AS source",
        ),
        (
            "哪些患者至少就诊4次？请展示其姓名、编号、最初一次的日期与疾病、"
            "最近一次的日期与疾病，以及记录总数。",
            "AS last_visit_date",
        ),
        (
            "统计2024-07-01至2026-04-30各月总花费，并为有足够历史的月份"
            "给出12个月滞后值和本月减滞后值。",
            "LAG(total_cost, 12)",
        ),
        (
            "请列出图谱中的疾病→并发症→治疗药物链，两个环节都需达到"
            "0.95置信度和1次计数，输出三类实体及各边置信度。",
            "c.count >= 1 AND d.count >= 1",
        ),
        (
            "知识图谱中哪些按名称有序的疾病对拥有不少于3个共同症状，并且"
            "相关边置信度至少0.85？返回疾病一、疾病二、共享数。",
            "a.confidence >= 0.85 AND b.confidence >= 0.85",
        ),
        (
            "哪些疾病同时具备至少1个症状、3个药物和2个检查知识目标？"
            "列出疾病和各类目标数，覆盖广者在前。",
            "drug_count + exam_count",
        ),
        (
            "将年龄≥60且至少就诊1次的人按累计费用降序做四等分编号，"
            "只保留编号≤1的高费用组，输出患者资料和汇总指标。",
            "NTILE(4)",
        ),
        (
            "找出‘原疾病有并发症、某药物又对该并发症禁忌’的图谱链，两条边"
            "置信度均≥0.9；并统计2025-01-01以来原疾病就诊中实际处方该药的"
            "次数，零次也保留；涉及的知识关系按“用药禁忌”标签识别。",
            "LEFT JOIN visits AS v",
        ),
        (
            "30岁以上男性患者从首诊月起，在偏移不超过4个月的各月有多少人"
            "再次或当月就诊？返回队列月、偏移、人数、队列规模、留存率。",
            "AS retention_rate",
        ),
        (
            "疾病必须逐一满足关系清单{并发症、治疗方式}，相应边置信度不低于"
            "0.9，且实际记录至少1条。返回疾病、提及次数、就诊数、平均费用。",
            "WITH required_relation(relation_zh)",
        ),
        (
            "请统计治疗药物关系的每个尾端覆盖多少个疾病，只保留覆盖≥2且热度"
            "超过其实体类型均值者，按疾病覆盖数和提及量降序。",
            "HAVING COUNT(DISTINCT kt.head) >= 2",
        ),
        (
            "哪些所属科室目标同时具备至少2个疾病来源和高于同类均值的提及次数？"
            "返回尾实体、类型、来源疾病数、提及量，高覆盖者优先。",
            "kt.relation_zh = '所属科室'",
        ),
        (
            "请对疾病两两比较图谱症状，只列共同症状达到1个、双方边置信度均≥0.98"
            "的组合，输出两种疾病及共有数量；涉及的知识关系按“临床表现”标签识别。",
            "a.confidence >= 0.98 AND b.confidence >= 0.98",
        ),
        (
            "将图谱关系按疾病做条件计数，筛出拥有不少于4个临床表现、1种治疗药物、"
            "1种诊断检查者，按总覆盖量降序。",
            "symptom_count + drug_count + exam_count",
        ),
        (
            "请构建北京60岁以上人群漏斗：队列规模、2024-07-20后就诊人数、就诊疾病"
            "拥有治疗药物关系的人数、疾病种类数、费用合计。",
            "age >= 60",
        ),
        (
            "知识链若表明某疾病可并发另一疾病，且某药对该并发疾病禁忌，并且两边"
            "置信度≥0.85，则返回链条及2025-07-01后匹配处方次数，零次链也要列出；"
            "涉及的知识关系按“并发症”、“用药禁忌”标签识别。",
            "v.visit_date >= '2025-07-01'",
        ),
        (
            "2025年哪些疾病出现在45岁以上人群却没有出现在45岁以下人群？按知识库"
            "热度取前7，并附该高龄组记录数。",
            "p.age >= 45",
        ),
        (
            "请筛选同时完整覆盖治疗药物与所属科室两种高置信知识关系的疾病，每类"
            "至少一边达到0.98；再保留就诊量≥3者，输出知识热度和费用均值；涉及的"
            "小数结果保留两位小数。",
            "WITH required_relation(relation_zh)",
        ),
    ],
)
def test_selection_used_family_wording_regressions(question, marker):
    result = compile_semantic_medical_result(question, DOMAINS)

    assert result.sql is not None
    assert marker in result.sql
