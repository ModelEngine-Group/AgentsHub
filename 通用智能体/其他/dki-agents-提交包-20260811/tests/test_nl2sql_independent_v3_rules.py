import re

import pytest

import dki.analysis.nl2sql as nl2sql_module
import dki.analysis.nl2sql_explicit as nl2sql_explicit_module
from dki.analysis.nl2sql import NL2SQL
from dki.analysis.nl2sql_explicit import compile_explicit_medical_sql


DOMAINS = {
    "region": ["广州", "北京", "上海"],
    "department": ["心血管内科", "呼吸内科", "内分泌科"],
    "drug": ["丙戊酸钠"],
    "exam": ["胃镜检查", "头颅CT"],
    "kg_entity": ["胃溃疡", "乏力", "呋塞米"],
    # The normalized public rule chain also inspects these empty domains after
    # the explicit compiler abstains.
    "patient_name": [],
    "patient_id": [],
    "visit_id": [],
    "disease": [],
    "kg_head_type": [],
    "kg_relation_zh": [
        "临床表现",
        "所属科室",
        "治疗药物",
        "用药禁忌",
    ],
}


@pytest.mark.parametrize(
    ("category", "question", "expected"),
    [
        (
            "filtering",
            "筛选年龄为偶数并且在64岁及以上的患者，输出患者ID、姓名和年龄；"
            "按年龄升序、患者ID升序排列。这里“偶数”的精确定义是年龄除以2后的余数等于0。",
            "SELECT patient_id, name, age FROM patients WHERE age >= 64 "
            "AND age % 2 = 0 ORDER BY age ASC, patient_id ASC",
        ),
        (
            "aggregation_grouping",
            "按地区统计患者人数，返回地区和患者人数；患者人数降序，"
            "人数相同按地区升序排列。",
            "SELECT region, COUNT(*) AS patient_count FROM patients "
            "GROUP BY region ORDER BY patient_count DESC, region ASC",
        ),
        (
            "multi_table_join",
            "查找广州患者费用至少12000元的就诊，显示患者姓名、年龄、"
            "就诊ID、疾病和费用；按费用降序、患者姓名升序、就诊ID升序排列。",
            "SELECT p.name, p.age, v.visit_id, v.disease, v.cost "
            "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id "
            "WHERE p.region = '广州' AND v.cost >= 12000 "
            "ORDER BY v.cost DESC, p.name ASC, v.visit_id ASC",
        ),
        (
            "knowledge_graph",
            "查询胃溃疡在图谱中的全部关系，返回关系中文名、尾实体、"
            "尾实体类型、置信度和观察次数；按关系中文名、尾实体升序排列。",
            "SELECT relation_zh, tail, tail_type, confidence, count "
            "FROM kg_triples WHERE head = '胃溃疡' "
            "ORDER BY relation_zh ASC, tail ASC",
        ),
        (
            "temporal",
            "列出2023年5月发生的就诊，返回就诊ID、日期、疾病和科室；"
            "按日期升序、就诊ID升序排列。精确日期边界为2023-05-01（含）"
            "至2023-06-01（不含）。",
            "SELECT visit_id, visit_date, disease, department FROM visits "
            "WHERE visit_date >= '2023-05-01' AND visit_date < '2023-06-01' "
            "ORDER BY visit_date ASC, visit_id ASC",
        ),
        (
            "top_k",
            "找出年龄最大的前7名患者，年龄相同按患者ID升序；"
            "返回患者ID、姓名、年龄和地区。",
            "SELECT patient_id, name, age, region FROM patients "
            "ORDER BY age DESC, patient_id ASC LIMIT 7",
        ),
        (
            "nested_correlated",
            "找出年龄高于全部患者平均年龄的患者，返回患者ID、姓名、年龄和地区；"
            "按年龄降序、患者ID升序排列。",
            "SELECT patient_id, name, age, region FROM patients "
            "WHERE age > (SELECT AVG(age) FROM patients) "
            "ORDER BY age DESC, patient_id ASC",
        ),
        (
            "having",
            "按地区统计患者人数，只保留人数至少9人的地区；返回地区和人数，"
            "按人数降序、地区升序排列。",
            "SELECT region, COUNT(*) AS patient_count FROM patients "
            "GROUP BY region HAVING COUNT(*) >= 9 "
            "ORDER BY patient_count DESC, region ASC",
        ),
        (
            "cte",
            "用CTE统计每个科室的就诊次数和平均费用，再返回平均费用高于6000元"
            "的科室；显示科室、次数、平均费用，按平均费用降序、科室升序排列。",
            "WITH department_stats AS (SELECT department, "
            "COUNT(*) AS visit_count, AVG(cost) AS avg_cost "
            "FROM visits GROUP BY department) "
            "SELECT department, visit_count, avg_cost FROM department_stats "
            "WHERE avg_cost > 6000 ORDER BY avg_cost DESC, department ASC",
        ),
        (
            "window",
            "对全部就诊按费用降序计算并列名次，费用相同名次相同且后续名次留空；"
            "返回就诊ID、疾病、费用和费用名次，按名次、就诊ID升序排列。",
            "SELECT visit_id, disease, cost, "
            "RANK() OVER (ORDER BY cost DESC) AS cost_rank "
            "FROM visits ORDER BY cost_rank ASC, visit_id ASC",
        ),
        (
            "set_operation",
            "返回同时作为非空真实处方药、又作为图谱 Drug 类型实体的药物名，"
            "即两者交集；按药物升序排列。其中非空明确指字符长度大于0。",
            "SELECT drug_prescribed AS drug FROM visits "
            "WHERE LENGTH(drug_prescribed) > 0 INTERSECT "
            "SELECT name AS drug FROM kg_entities WHERE type = 'Drug' "
            "ORDER BY drug ASC",
        ),
    ],
)
def test_explicit_medical_analytics_regressions(
    category,
    question,
    expected,
):
    assert category
    assert compile_explicit_medical_sql(question, DOMAINS) == expected


def test_exact_like_literal_is_not_expanded_by_medical_alias_normalization():
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._normalization_literals = {"2型糖尿病"}
    compiler._alias_map = {"糖尿病": "2型糖尿病"}
    compiler._normalization_pattern = re.compile("糖尿病")
    question = (
        "列出主诊断名称中含“糖尿病”的就诊。"
        "疾病名称筛选使用的精确LIKE模式为%糖尿病%。"
    )
    normalized = compiler._normalize(question)
    assert "含“2型糖尿病”" in normalized
    assert "精确LIKE模式为%糖尿病%" in normalized


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "查询年龄不低于55岁且来自广州的女性患者，"
            "返回患者ID、姓名、年龄和地区；按年龄降序、患者ID升序排列。",
            "SELECT patient_id, name, age, region FROM patients "
            "WHERE age >= 55 AND gender = '女' AND region = '广州' "
            "ORDER BY age DESC, patient_id ASC",
        ),
        (
            "查询费用不低于7000元且科室为呼吸内科的就诊，"
            "返回就诊ID、日期、疾病和费用；"
            "按费用降序、日期升序、就诊ID升序排列。",
            "SELECT visit_id, visit_date, disease, cost FROM visits "
            "WHERE cost >= 7000 AND department = '呼吸内科' "
            "ORDER BY cost DESC, visit_date ASC, visit_id ASC",
        ),
        (
            "按科室统计就诊次数、费用总额和最长住院天数，"
            "返回科室、就诊次数、费用总额、最长住院天数；"
            "按费用总额降序、科室升序排列。",
            "SELECT department, COUNT(*) AS visit_count, "
            "SUM(cost) AS total_cost, MAX(los_days) AS max_los_days "
            "FROM visits GROUP BY department "
            "ORDER BY total_cost DESC, department ASC",
        ),
        (
            "按头实体统计边数和观察次数总和，"
            "返回头实体、边数、观察次数总和；"
            "按边数降序、头实体升序排列。",
            "SELECT head, COUNT(*) AS edge_count, "
            "SUM(count) AS total_observations FROM kg_triples "
            "GROUP BY head ORDER BY edge_count DESC, head ASC",
        ),
    ],
)
def test_compositional_fallback_handles_unseen_slot_combinations(
    question,
    expected,
):
    assert compile_explicit_medical_sql(question, DOMAINS) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "连接患者和就诊，筛选年龄不低于50岁的广州患者及费用至少8000元的记录，"
            "返回患者ID、姓名、就诊ID、日期、费用；"
            "按费用降序、患者ID升序、就诊ID升序排列。",
            "SELECT p.patient_id, p.name, v.visit_id, v.visit_date, v.cost "
            "FROM patients AS p JOIN visits AS v ON v.patient_id = p.patient_id "
            "WHERE p.age >= 50 AND p.region = '广州' AND v.cost >= 8000 "
            "ORDER BY v.cost DESC, p.patient_id ASC, v.visit_id ASC",
        ),
        (
            "连接患者与就诊后按患者地区统计就诊次数、不同患者数和平均费用，"
            "返回地区、就诊次数、不同患者数、平均费用；"
            "按就诊次数降序、地区升序排列。",
            "SELECT p.region AS region, COUNT(*) AS visit_count, "
            "COUNT(DISTINCT p.patient_id) AS patient_count, "
            "AVG(v.cost) AS avg_cost FROM patients AS p "
            "JOIN visits AS v ON v.patient_id = p.patient_id "
            "GROUP BY p.region ORDER BY visit_count DESC, region ASC",
        ),
        (
            "按患者统计其就诊次数、累计费用和最长住院天数，"
            "返回患者ID及三项指标；"
            "按就诊次数降序、累计费用降序、患者ID升序排列。",
            "SELECT p.patient_id AS patient_id, COUNT(*) AS visit_count, "
            "SUM(v.cost) AS total_cost, MAX(v.los_days) AS max_los_days "
            "FROM patients AS p JOIN visits AS v "
            "ON v.patient_id = p.patient_id GROUP BY p.patient_id "
            "ORDER BY visit_count DESC, total_cost DESC, patient_id ASC",
        ),
        (
            "按患者统计最早就诊日、最晚就诊日和就诊次数，"
            "返回患者ID及三项信息；"
            "按就诊次数降序、患者ID升序排列。",
            "SELECT p.patient_id AS patient_id, "
            "MIN(v.visit_date) AS first_visit_date, "
            "MAX(v.visit_date) AS last_visit_date, "
            "COUNT(*) AS visit_count FROM patients AS p "
            "JOIN visits AS v ON v.patient_id = p.patient_id "
            "GROUP BY p.patient_id "
            "ORDER BY visit_count DESC, patient_id ASC",
        ),
    ],
)
def test_compositional_patient_visit_join_fallback(question, expected):
    assert compile_explicit_medical_sql(question, DOMAINS) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "统计2024-12-08至2025-03-05之间的就诊，"
            "按月份返回就诊次数和平均费用。",
            "SELECT substr(visit_date, 1, 7) AS visit_month, "
            "COUNT(*) AS visit_count, AVG(cost) AS avg_cost FROM visits "
            "WHERE visit_date >= '2024-12-08' "
            "AND visit_date <= '2025-03-05' "
            "GROUP BY substr(visit_date, 1, 7) "
            "ORDER BY visit_month ASC",
        ),
        (
            "对科室为内分泌科且费用至少1000元的就诊按处方药物分组，"
            "返回处方次数和总费用。",
            "SELECT drug_prescribed, COUNT(*) AS prescription_count, "
            "SUM(cost) AS total_cost FROM visits "
            "WHERE cost >= 1000 AND department = '内分泌科' "
            "GROUP BY drug_prescribed "
            "ORDER BY prescription_count DESC, drug_prescribed ASC",
        ),
        (
            "对费用至少1000元且住院天数至少0天的就诊按科室分组，"
            "返回各科室的就诊次数。",
            "SELECT department, COUNT(*) AS visit_count FROM visits "
            "WHERE cost >= 1000 AND los_days >= 0 GROUP BY department "
            "ORDER BY visit_count DESC, department ASC",
        ),
        (
            "对住院天数至少1天且费用至少1500元的就诊按诊断分组，"
            "返回各诊断的就诊次数和平均费用。",
            "SELECT disease, COUNT(*) AS visit_count, "
            "AVG(cost) AS avg_cost FROM visits "
            "WHERE cost >= 1500 AND los_days >= 1 GROUP BY disease "
            "ORDER BY visit_count DESC, disease ASC",
        ),
        (
            "统计2024-09-14及以后每年的就诊次数和总费用。",
            "SELECT substr(visit_date, 1, 4) AS visit_year, "
            "COUNT(*) AS visit_count, SUM(cost) AS total_cost FROM visits "
            "WHERE visit_date >= '2024-09-14' "
            "GROUP BY substr(visit_date, 1, 4) "
            "ORDER BY visit_year ASC",
        ),
        (
            "统计科室内分泌科在2024-10-20至2025-01-19之间"
            "每月的就诊次数和平均费用。",
            "SELECT substr(visit_date, 1, 7) AS visit_month, "
            "COUNT(*) AS visit_count, AVG(cost) AS avg_cost FROM visits "
            "WHERE visit_date >= '2024-10-20' "
            "AND visit_date <= '2025-01-19' "
            "AND department = '内分泌科' "
            "GROUP BY substr(visit_date, 1, 7) "
            "ORDER BY visit_month ASC",
        ),
        (
            "对中文关系为临床表现且置信度至少0.7的三元组，"
            "按尾实体统计头实体数和观察次数总和。",
            "SELECT tail, COUNT(DISTINCT head) AS distinct_head_count, "
            "SUM(count) AS total_observations FROM kg_triples "
            "WHERE relation_zh = '临床表现' AND confidence >= 0.7 "
            "GROUP BY tail "
            "ORDER BY total_observations DESC, tail ASC",
        ),
    ],
)
def test_public_authoritative_grouping_frames(question, expected):
    assert compile_explicit_medical_sql(question, DOMAINS) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "返回至少有一次在科室内分泌科且费用至少5000元"
            "就诊记录的患者编号和姓名。",
            "SELECT p.patient_id, p.name FROM patients AS p "
            "WHERE EXISTS (SELECT 1 FROM visits AS v "
            "WHERE v.patient_id = p.patient_id AND v.cost >= 5000 "
            "AND v.department = '内分泌科') "
            "ORDER BY p.patient_id ASC",
        ),
        (
            "查询科室内分泌科中费用至少比该科室平均费用高0元的"
            "就诊编号、患者编号、诊断和费用。",
            "SELECT visit_id, patient_id, disease, cost FROM visits AS v "
            "WHERE department = '内分泌科' "
            "AND cost >= (SELECT AVG(cost) + 0 FROM visits "
            "WHERE department = '内分泌科') "
            "ORDER BY cost DESC, visit_id ASC",
        ),
        (
            "返回类型为Symptom、提及次数至少1次且没有作为任何"
            "三元组头实体出现的实体名称。",
            "SELECT e.name FROM kg_entities AS e "
            "WHERE e.type = 'Symptom' AND e.mention_count >= 1 "
            "AND NOT EXISTS (SELECT 1 FROM kg_triples AS t "
            "WHERE t.head = e.name AND t.head_type = e.type) "
            "ORDER BY e.name ASC",
        ),
        (
            "查询头实体为呋塞米且置信度至少0.6的所有关系、"
            "中文关系名和尾实体。",
            "SELECT relation, relation_zh, tail FROM kg_triples "
            "WHERE head = '呋塞米' AND confidence >= 0.6 "
            "ORDER BY relation ASC, tail ASC",
        ),
        (
            "在科室内分泌科中，返回累计费用最高的前4位患者"
            "及其累计费用。",
            "SELECT patient_id, SUM(cost) AS total_cost FROM visits "
            "WHERE department = '内分泌科' GROUP BY patient_id "
            "ORDER BY total_cost DESC, patient_id ASC LIMIT 4",
        ),
        (
            "对2024-10-01及以后的就诊，按患者和日期计算累计费用，"
            "返回前4条记录。",
            "SELECT patient_id, visit_date, cost, "
            "SUM(cost) OVER (PARTITION BY patient_id "
            "ORDER BY visit_date ASC, visit_id ASC "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) "
            "AS running_cost FROM visits "
            "WHERE visit_date >= '2024-10-01' "
            "ORDER BY patient_id ASC, visit_date ASC, visit_id ASC LIMIT 4",
        ),
        (
            "先计算各科室的就诊次数和平均费用，再返回就诊次数至少3次"
            "且平均费用至少5000元的科室。",
            "WITH department_stats AS (SELECT department, "
            "COUNT(*) AS visit_count, AVG(cost) AS avg_cost "
            "FROM visits GROUP BY department) "
            "SELECT department, visit_count, avg_cost "
            "FROM department_stats WHERE visit_count >= 3 "
            "AND avg_cost >= 5000 "
            "ORDER BY visit_count DESC, department ASC",
        ),
        (
            "找出开具药物丙戊酸钠至少1次的科室，并返回处方次数。",
            "SELECT department, COUNT(*) AS prescription_count FROM visits "
            "WHERE drug_prescribed = '丙戊酸钠' GROUP BY department "
            "HAVING COUNT(*) >= 1 "
            "ORDER BY prescription_count DESC, department ASC",
        ),
        (
            "找出至少有2次就诊且平均费用至少5000元的诊断，"
            "并返回就诊次数和平均费用。",
            "SELECT disease, COUNT(*) AS visit_count, "
            "AVG(cost) AS avg_cost FROM visits GROUP BY disease "
            "HAVING COUNT(*) >= 2 AND AVG(cost) >= 5000 "
            "ORDER BY avg_cost DESC, disease ASC",
        ),
        (
            "先汇总2025-10-22至2026-02-02之间每月的就诊次数和总费用，"
            "再返回就诊次数至少3次的月份。",
            "WITH monthly AS (SELECT substr(visit_date, 1, 7) "
            "AS visit_month, COUNT(*) AS visit_count, "
            "SUM(cost) AS total_cost FROM visits "
            "WHERE visit_date >= '2025-10-22' "
            "AND visit_date <= '2026-02-02' "
            "GROUP BY substr(visit_date, 1, 7)) "
            "SELECT visit_month, visit_count, total_cost FROM monthly "
            "WHERE visit_count >= 3 ORDER BY visit_month ASC",
        ),
    ],
)
def test_public_authoritative_advanced_frames(question, expected):
    assert compile_explicit_medical_sql(question, DOMAINS) == expected


def test_explicit_quoted_relation_union_preserves_strict_and_inclusive_bounds():
    sql = compile_explicit_medical_sql(
        "查询“临床表现”或“治疗药物”关系，置信度低于0.8且观察次数不少于4，"
        "返回头实体、关系中文名、尾实体、观察次数。",
        DOMAINS,
    )

    assert sql is not None
    assert "relation_zh IN ('临床表现', '治疗药物')" in sql
    assert "confidence < 0.8" in sql
    assert "count >= 4" in sql


def test_explicit_quoted_relation_union_keeps_equality_relation_filter():
    sql = compile_explicit_medical_sql(
        "查询“临床表现”或“治疗药物”关系，置信度恰好为0.8且观察次数大于4，"
        "返回头实体、关系中文名、尾实体和观察次数。",
        DOMAINS,
    )

    assert sql is not None
    assert "relation_zh IN ('临床表现', '治疗药物')" in sql
    assert "confidence = 0.8" in sql
    assert "count > 4" in sql


def test_explicit_equality_sibling_keeps_single_relation_filter():
    sql = compile_explicit_medical_sql(
        "查询临床表现关系，置信度恰好为0.8且观察次数至少4，"
        "返回头实体、关系中文名、尾实体和观察次数。",
        DOMAINS,
    )

    assert sql is not None
    assert "relation_zh = '临床表现'" in sql
    assert "confidence = 0.8" in sql
    assert "count >= 4" in sql


@pytest.mark.parametrize(
    "question",
    (
        "查询“临床表现”或“临床表现”关系，置信度至少0.8且观察次数至少4，"
        "返回头实体、关系中文名、尾实体、观察次数。",
        "查询“临床表现”和“治疗药物”关系，置信度至少0.8且观察次数至少4，"
        "返回头实体、关系中文名、尾实体、观察次数。",
        "查询“临床表现”或“治疗药物”关系，每条关系置信度至少0.8且观察次数至少4，"
        "返回头实体、关系中文名、尾实体、观察次数。",
        "查询“临床表现”或“治疗药物”关系，临床表现关系置信度至少0.7且"
        "治疗药物关系置信度至少0.8且观察次数至少4，"
        "返回头实体、关系中文名、尾实体、观察次数。",
        "查询“临床表现”或“治疗药物”关系，置信度至少0.7或不超过0.8且"
        "观察次数至少4，返回头实体、关系中文名、尾实体、观察次数。",
    ),
)
def test_explicit_quoted_relation_union_rejects_ambiguous_ownership(question):
    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_explicit_quoted_relation_union_abstains_for_unrendered_metric_conjuncts():
    metric_clauses = (
        "置信度至少0.7且不超过0.8且观察次数至少4",
        "置信度至少0.7，且不超过0.8且观察次数至少4",
        "置信度介于0.7和0.8之间且观察次数至少4",
    )

    for metric_clause in metric_clauses:
        assert (
            compile_explicit_medical_sql(
                "查询“临床表现”或“治疗药物”关系，"
                f"{metric_clause}，返回头实体、关系中文名、尾实体、观察次数，"
                "按关系中文名升序排列。",
                DOMAINS,
            )
            is None
        )


def test_explicit_quoted_relation_union_preserves_symbolic_strict_bounds():
    sql = compile_explicit_medical_sql(
        "查询“临床表现”或“治疗药物”关系，置信度>0.8且观察次数<4，"
        "返回头实体、关系中文名、尾实体、观察次数。",
        DOMAINS,
    )

    assert sql is not None
    assert "confidence > 0.8" in sql
    assert "count < 4" in sql


def test_explicit_quoted_relation_union_preserves_equality_on_both_metrics():
    sql = compile_explicit_medical_sql(
        "查询“临床表现”或“治疗药物”关系，置信度等于0.8且观察次数为4，"
        "返回头实体、关系中文名、尾实体、观察次数。",
        DOMAINS,
    )

    assert sql is not None
    assert "confidence = 0.8" in sql
    assert "count = 4" in sql


def test_public_rule_route_preserves_explicit_quoted_relation_union(monkeypatch):
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda question, domains: None,
    )

    sql = compiler._rule_sql_normalized(
        "查询“临床表现”或“治疗药物”关系，置信度恰好为0.8且观察次数至少4，"
        "返回头实体、关系中文名、尾实体和观察次数。"
    )

    assert sql is not None
    assert "relation_zh IN ('临床表现', '治疗药物')" in sql
    assert "confidence = 0.8" in sql
    assert "count >= 4" in sql


@pytest.mark.parametrize(
    "metric_clause",
    (
        "置信度至少0.7，且不超过0.8且观察次数至少4",
        "置信度至少0.7；不超过0.8且观察次数至少4",
        "置信度至少0.7与不超过0.8且观察次数至少4",
        "置信度介于0.7和0.8之间且观察次数至少4",
        "置信度至少0.7且观察次数至少4，且不超过8",
        "置信度至少0.7且观察次数至少4；不超过8",
        "置信度至少0.7且观察次数至少4与不超过8",
        "置信度至少0.7且观察次数介于4和8之间",
    ),
)
def test_quoted_relation_union_metric_ranges_do_not_reach_public_fallback(
    metric_clause,
    monkeypatch,
):
    question = (
        "查询“临床表现”或“治疗药物”关系，"
        f"{metric_clause}，返回头实体、关系中文名、尾实体、观察次数，"
        "按关系中文名升序排列。"
    )
    assert compile_explicit_medical_sql(question, DOMAINS) is None

    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda value, domains: None,
    )
    assert compiler._rule_sql_normalized(question) is None


@pytest.mark.parametrize(
    ("metric_clause", "predicates"),
    (
        ("置信度>0.8且观察次数<4", ("confidence > 0.8", "count < 4")),
        ("置信度=0.8且观察次数=4", ("confidence = 0.8", "count = 4")),
    ),
)
def test_public_rule_route_preserves_quoted_union_symbolic_operators(
    metric_clause,
    predicates,
    monkeypatch,
):
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda value, domains: None,
    )

    sql = compiler._rule_sql_normalized(
        "查询“临床表现”或“治疗药物”关系，"
        f"{metric_clause}，返回头实体、关系中文名、尾实体、观察次数。"
    )

    assert sql is not None
    assert "relation_zh IN ('临床表现', '治疗药物')" in sql
    for predicate in predicates:
        assert predicate in sql


@pytest.mark.parametrize(
    "question",
    (
        "查询“临床表现”或“治疗药物”关系集合，置信度至少0.8且观察次数至少4，"
        "返回头实体、关系中文名、尾实体、观察次数。",
        "查询包含“临床表现”或“治疗药物”关系，置信度至少0.8且观察次数至少4，"
        "返回头实体、关系中文名、尾实体、观察次数。",
        "查询包括“临床表现”或“治疗药物”关系，置信度至少0.8且观察次数至少4，"
        "返回头实体、关系中文名、尾实体、观察次数。",
        "查询“临床表现”或“治疗药物”关系，第一条关系置信度至少0.8且"
        "观察次数至少4，返回头实体、关系中文名、尾实体、观察次数。",
    ),
)
def test_explicit_quoted_relation_union_rejects_set_and_ordinal_ownership(question):
    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_explicit_filtering_rejects_repeated_single_relation_thresholds():
    question = (
        "查询临床表现关系，临床表现关系置信度至少0.7且"
        "临床表现关系置信度不超过0.8且观察次数至少4，"
        "返回头实体、关系中文名、尾实体、观察次数，按观察次数降序排列。"
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_explicit_generic_filtering_rejects_unquoted_relation_union():
    question = (
        "查询临床表现或治疗药物关系，置信度至少0.8且观察次数至少4，"
        "返回头实体、关系中文名、尾实体、观察次数，按观察次数降序排列。"
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None


@pytest.mark.parametrize("suffix", ("差集", "集合"))
def test_explicit_generic_group_rejects_nonflat_relation_sets(suffix):
    question = (
        "按关系中文名统计三元组数，筛选临床表现或治疗药物关系"
        f"{suffix}，置信度至少0.8且观察次数至少4，"
        "返回关系中文名和三元组数，按三元组数降序排列。"
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_explicit_generic_group_preserves_safe_literal_relation_union():
    sql = compile_explicit_medical_sql(
        "按关系中文名统计三元组数，筛选临床表现或治疗药物关系，"
        "置信度>0.7且观察次数不超过8，"
        "返回关系中文名和三元组数，按三元组数降序排列。",
        DOMAINS,
    )

    assert sql is not None
    assert "relation_zh IN ('临床表现', '治疗药物')" in sql
    assert "confidence > 0.7" in sql
    assert "count <= 8" in sql


def test_explicit_generic_group_rejects_average_edge_confidence_qualifier():
    question = (
        "按关系中文名统计三元组数和平均边置信度，平均边置信度低于0.8，"
        "返回关系中文名、三元组数和平均边置信度，按三元组数降序排列。"
    )

    assert nl2sql_explicit_module._rule_generic_group(question, DOMAINS) is None
    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_explicit_generic_group_rejects_relation_metric_range_terminally(monkeypatch):
    question = (
        "按关系中文名统计三元组数，筛选临床表现关系，"
        "置信度至少0.7且不超过0.8，"
        "返回关系中文名和三元组数，按三元组数降序排列。"
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None

    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda value, domains: None,
    )
    assert compiler._rule_sql_normalized(question) is None


def test_explicit_generic_nested_projection_rejects_relation_range():
    question = (
        "查询头实体为胃溃疡，返回所有关系、中文关系名和尾实体，"
        "临床表现关系置信度至少0.7且不超过0.8。"
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_explicit_direct_kg_filter_rejects_relation_range():
    question = "查询临床表现关系全部疾病与科室，置信度至少0.7且不超过0.8。"

    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_explicit_defers_peer_age_average_to_advanced_owner():
    question = (
        "找出年龄高于同地区同性别患者平均年龄的女病人，"
        "返回患者编号和年龄，按年龄降序、患者ID升序排列。"
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_explicit_relation_fence_rejects_static_type_route_without_relation_owner():
    question = (
        "查询临床表现关系，头实体类型为Disease且尾实体类型为Symptom，"
        "返回头实体、关系中文名、尾实体和置信度，按头实体升序排列。"
    )

    sql = compile_explicit_medical_sql(question, DOMAINS)

    assert sql is not None
    assert "relation_zh = '临床表现'" in sql


def test_explicit_relation_fence_rejects_static_group_route_without_relation_owner():
    question = "按关系中文名统计图谱三元组数量、平均置信度和观察次数总和，筛选临床表现关系。"

    sql = compile_explicit_medical_sql(question, DOMAINS)

    assert sql is not None
    assert "relation_zh = '临床表现'" in sql


def test_explicit_direct_relation_route_rejects_unowned_top_k():
    assert (
        compile_explicit_medical_sql(
            "查询临床表现关系全部疾病与科室，前2条。",
            DOMAINS,
        )
        is None
    )


def test_explicit_quoted_relation_union_rejects_unowned_projection():
    question = (
        "查询“临床表现”或“治疗药物”关系，置信度至少0.8且观察次数至少4，"
        "返回头实体、关系中文名、尾实体、观察次数和置信度。"
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_explicit_generic_group_rejects_unowned_projection_and_ordering():
    projection = (
        "按关系中文名统计三元组数，筛选临床表现关系，"
        "返回关系中文名、三元组数和尾实体，按三元组数降序排列。"
    )
    ordering = (
        "按关系中文名统计三元组数，筛选临床表现关系，"
        "返回关系中文名和三元组数，按三元组数降序排列，"
        "再按关系中文名升序排列。"
    )

    assert compile_explicit_medical_sql(projection, DOMAINS) is None
    assert compile_explicit_medical_sql(ordering, DOMAINS) is None


def test_explicit_relation_fence_rejects_repeated_required_all_occurrence(monkeypatch):
    question = "查询同时具有临床表现和临床表现关系的头实体，返回头实体。"
    monkeypatch.setattr(
        nl2sql_explicit_module,
        "_rule_set_operation",
        lambda value: (
            "SELECT head FROM kg_triples "
            "WHERE relation_zh = '临床表现'"
        ),
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_explicit_relation_fence_rejects_duplicate_required_all_aliases(
    monkeypatch,
):
    question = "查询同时具有临床表现和临床表现关系的头实体，返回头实体。"
    monkeypatch.setattr(
        nl2sql_explicit_module,
        "_rule_set_operation",
        lambda value: (
            "SELECT d.head FROM kg_triples AS d "
            "JOIN kg_triples AS m ON m.head = d.head "
            "WHERE d.relation_zh = '临床表现' "
            "AND m.relation_zh = '临床表现'"
        ),
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_explicit_relation_fence_rejects_chain_only_required_all_join(monkeypatch):
    question = "查询同时具有临床表现和治疗药物关系的头实体，返回头实体。"
    monkeypatch.setattr(
        nl2sql_explicit_module,
        "_rule_set_operation",
        lambda value: (
            "SELECT d.head FROM kg_triples AS d "
            "JOIN kg_triples AS m ON m.head = d.tail "
            "WHERE d.relation_zh = '临床表现' "
            "AND m.relation_zh = '治疗药物'"
        ),
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_explicit_relation_fence_preserves_canonical_required_all_group(
    monkeypatch,
):
    question = "查询同时具有临床表现和治疗药物关系的头实体，返回头实体。"
    expected = (
        "SELECT head FROM kg_triples "
        "WHERE relation_zh IN ('临床表现', '治疗药物') "
        "GROUP BY head HAVING COUNT(DISTINCT relation_zh) = 2"
    )
    monkeypatch.setattr(
        nl2sql_explicit_module,
        "_rule_set_operation",
        lambda value: expected,
    )

    assert compile_explicit_medical_sql(question, DOMAINS) == expected


def test_explicit_relation_fence_preserves_distinct_required_all_aliases(monkeypatch):
    question = "查询同时具有临床表现和治疗药物关系的头实体，返回头实体。"
    expected = (
        "SELECT DISTINCT d.head AS disease FROM kg_triples AS d "
        "JOIN kg_triples AS m ON m.head = d.head "
        "WHERE d.relation_zh = '临床表现' "
        "AND m.relation_zh = '治疗药物'"
    )
    monkeypatch.setattr(
        nl2sql_explicit_module,
        "_rule_set_operation",
        lambda value: expected,
    )

    assert compile_explicit_medical_sql(question, DOMAINS) == expected


@pytest.mark.parametrize("verb", ("返回", "列出", "给出"))
def test_explicit_relation_fence_rejects_unmappable_projection(monkeypatch, verb):
    question = f"查询临床表现关系，{verb}头实体、未知指标。"
    monkeypatch.setattr(
        nl2sql_explicit_module,
        "_rule_set_operation",
        lambda value: (
            "SELECT head FROM kg_triples "
            "WHERE relation_zh = '临床表现'"
        ),
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_explicit_relation_fence_rejects_required_all_or_lowering(monkeypatch):
    question = "查询同时具有临床表现和治疗药物关系的头实体，返回头实体。"
    monkeypatch.setattr(
        nl2sql_explicit_module,
        "_rule_set_operation",
        lambda value: (
            "SELECT d.head FROM kg_triples AS d "
            "JOIN kg_triples AS m ON m.head = d.head "
            "WHERE d.relation_zh = '临床表现' "
            "OR m.relation_zh = '治疗药物'"
        ),
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_explicit_relation_fence_rejects_partial_grouped_projection(monkeypatch):
    question = (
        "按关系中文名统计三元组数，筛选临床表现关系，"
        "返回关系中文名和三元组数，按三元组数降序排列。"
    )
    monkeypatch.setattr(
        nl2sql_explicit_module,
        "_rule_set_operation",
        lambda value: (
            "SELECT relation_zh FROM kg_triples "
            "WHERE relation_zh = '临床表现' "
            "GROUP BY relation_zh ORDER BY relation_zh ASC"
        ),
    )

    sql = compile_explicit_medical_sql(question, DOMAINS)

    assert sql is not None
    assert sql.startswith(
        "SELECT relation_zh, COUNT(*) AS triple_count FROM kg_triples"
    )


def test_explicit_required_all_distinct_projection_remains_supported():
    domains = {
        **DOMAINS,
        "kg_relation_zh": [
            *DOMAINS["kg_relation_zh"],
            "诊断检查",
        ],
    }
    expected = (
        "SELECT DISTINCT d.head AS disease FROM kg_triples AS d "
        "JOIN kg_triples AS m ON m.head = d.head "
        "WHERE d.relation_zh = '诊断检查' "
        "AND m.relation_zh = '治疗药物' ORDER BY disease ASC"
    )

    assert (
        compile_explicit_medical_sql(
            "既有诊断检查边又有治疗药物边，返回疾病。",
            domains,
        )
        == expected
    )


@pytest.mark.parametrize(
    "projection",
    (
        "所有关系、中文关系名和尾实体",
        "关系、关系中文名和尾实体",
    ),
)
def test_explicit_relation_projection_aliases_preserve_exact_columns(
    monkeypatch,
    projection,
):
    question = f"查询临床表现关系，返回{projection}。"
    expected = (
        "SELECT relation, relation_zh, tail FROM kg_triples "
        "WHERE relation_zh = '临床表现'"
    )
    monkeypatch.setattr(
        nl2sql_explicit_module,
        "_rule_set_operation",
        lambda value: expected,
    )

    assert compile_explicit_medical_sql(question, DOMAINS) == expected


def test_required_all_qualifiers_do_not_revive_through_legacy_set_fallback(
    monkeypatch,
):
    domains = {
        **DOMAINS,
        "kg_relation_zh": [
            *DOMAINS["kg_relation_zh"],
            "诊断检查",
        ],
    }
    question = (
        "既有诊断检查边又有治疗药物边，"
        "诊断检查边置信度至少0.8且治疗药物边观察次数至少4，"
        "返回疾病和置信度，按置信度降序排列，前2条。"
    )
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = domains
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda value, domains: None,
    )

    assert compile_explicit_medical_sql(question, domains) is None
    assert compiler._rule_sql_normalized(question) is None


def test_required_all_static_route_abstention_does_not_revive_through_fallback(
    monkeypatch,
):
    question = (
        "作为头实体且类型为Disease的实体同时具有临床表现和治疗药物关系，"
        "临床表现关系置信度至少0.7且治疗药物关系置信度至少0.8，"
        "返回实体名称，按实体名称降序排列，前2条。"
    )
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS
    monkeypatch.setattr(
        nl2sql_module,
        "compile_compositional_medical_sql",
        lambda value, domains: None,
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None
    assert compiler._rule_sql_normalized(question) is None


def test_legacy_relation_average_abstains_for_unowned_having_qualifier():
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS
    question = (
        "每种关系的平均置信度至少0.8，"
        "返回关系中文名和平均置信度，按平均置信度降序排列，前3条。"
    )

    assert compiler._rule_kg(question) is None


def test_legacy_triple_count_abstains_for_aggregate_confidence_qualifier():
    question = "平均置信度低于0.8的三元组有多少？"
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS

    assert compile_explicit_medical_sql(question, DOMAINS) is None
    assert compiler._rule_kg(question) is None


def test_legacy_triple_count_abstains_for_aggregate_insufficient_confidence():
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS

    assert compiler._rule_kg("平均置信度不足0.8的三元组有多少？") is None


def test_legacy_triple_count_abstains_for_second_row_metric():
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS
    question = "置信度低于0.8且观察次数至少4的三元组有多少？"

    assert compiler._rule_kg(question) is None


def test_legacy_triple_count_preserves_one_raw_confidence_predicate():
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS

    assert compiler._rule_kg("置信度低于0.8的三元组有多少？") == (
        "SELECT COUNT(*) AS n_triples FROM kg_triples WHERE confidence < 0.8"
    )


def test_legacy_triple_count_preserves_raw_insufficient_confidence_predicate():
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS

    assert compiler._rule_kg("置信度不足0.8的三元组有多少？") == (
        "SELECT COUNT(*) AS n_triples FROM kg_triples WHERE confidence < 0.8"
    )


def test_legacy_relation_average_abstains_for_unowned_result_shape():
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS
    question = (
        "每种关系的平均置信度，返回平均置信度，"
        "按关系中文名升序排列。"
    )

    assert compiler._rule_kg(question) is None


def test_legacy_relation_average_preserves_its_complete_result_shape():
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS
    question = (
        "每种关系的平均置信度，返回关系中文名和平均置信度，"
        "按平均置信度降序排列，前3条。"
    )
    expected = (
        "SELECT relation_zh, ROUND(AVG(confidence), 3) AS avg_confidence "
        "FROM kg_triples GROUP BY relation_zh ORDER BY avg_confidence DESC LIMIT 3"
    )

    assert compiler._rule_kg(question) == expected


def test_generic_group_abstains_for_metric_cost_disjunction_in_every_route():
    compiler = NL2SQL.__new__(NL2SQL)
    compiler._domains = DOMAINS
    questions = (
        "按科室统计就诊次数，费用至少100 ＯＲ 不超过500，"
        "返回科室和就诊次数，按就诊次数降序排列。",
        "按科室统计就诊次数，费用至少100，ＯＲ 不超过500，"
        "返回科室和就诊次数，按就诊次数降序排列。",
        "按科室统计就诊次数，费用至少100 OR 住院天数至少3，"
        "返回科室和就诊次数，按就诊次数降序排列。",
    )

    for question in questions:
        assert compile_explicit_medical_sql(question, DOMAINS) is None
        assert compiler._rule_visits(question) is None


def test_generic_kg_group_abstains_for_peer_average_and_having_threshold():
    question = (
        "按关系中文名统计不同尾实体数和不同头实体数，"
        "尾实体提及次数高于同类型平均且不同头实体至少8个，"
        "返回关系中文名、不同尾实体数和不同头实体数，"
        "按不同头实体数降序排列。"
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_fixed_head_average_route_abstains_for_unowned_result_shape():
    question = (
        "统计头实体、边数和平均置信度，平均边置信度低于0.8，"
        "返回头实体和平均置信度，按头实体降序排列，前2条。"
    )

    assert compile_explicit_medical_sql(question, DOMAINS) is None


def test_fixed_head_average_route_preserves_its_complete_static_shape():
    question = (
        "平均边置信度低于0.8，返回头实体、边数和平均置信度，"
        "按平均置信度升序、头实体升序排列。"
    )
    expected = (
        "SELECT head, COUNT(*) AS edge_count, AVG(confidence) AS avg_confidence "
        "FROM kg_triples GROUP BY head HAVING AVG(confidence) < 0.8 "
        "ORDER BY avg_confidence ASC, head ASC"
    )

    assert compile_explicit_medical_sql(question, DOMAINS) == expected
