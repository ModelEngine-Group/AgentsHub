"""NL2SQL SOTA 管线的确定性单元测试（不触发 LLM）：
- _needs_llm 门：硬结构（比较到聚合/HAVING/占比/集合/嵌套TopK/相关子查询/CTE/双维度）主动让位；
- 简单问法规则仍命中；
- schema-linking 产出相关表/命中值域，且不含裸 SQL 注入风险。
"""
import pytest

from dki.analysis.nl2sql import NL2SQL
from dki.paths import OUTPUT_DIR


@pytest.fixture(scope="module")
def nl():
    db = OUTPUT_DIR / "insight.db"
    if not db.is_file():
        pytest.skip("缺少 insight.db，请先 run.ps1 demo-all")
    return NL2SQL(db)


def test_needs_llm_gate_yields_hard_structures(nl):
    hard = [
        "就诊量前3的疾病中平均费用最高的是哪个",
        "每个科室中费用高于该科室平均费用的就诊有多少条",
        "用 CTE 统计每个科室的就诊量",
    ]
    for q in hard:
        assert nl._needs_llm(q), f"应让位给 LLM 的硬结构未被识别: {q}"
        assert nl.rule_sql(q) is None, f"硬结构规则应弃权: {q}"


@pytest.mark.parametrize("item_id", [1030, 1031, 1032, 1071])
def test_public_development_grouped_threshold_and_two_dimension_promotions(
    nl,
    item_id,
):
    """Promoted public development skeletons remain result-equivalent."""

    from dki.analysis.nl2sql_eval import HARD_EVAL_SET, _load_items, _score_item

    item = next(
        row for row in _load_items(HARD_EVAL_SET)
        if row.get("id") == item_id
    )
    scored = _score_item(nl, item, "rule", nl.db_path, strict=True)

    assert scored["correct"], (
        scored.get("pred_sql"),
        scored.get("gold_sql"),
        scored.get("reason"),
    )


@pytest.mark.parametrize(
    "question",
    [
        "就诊次数超过20次或少于3次的疾病有哪些",
        "患者数超过10人的科室有哪些，并按平均费用排序",
        "平均费用超过5000元或低于100元的科室及其平均费用",
        "每个科室每种性别的就诊次数，只看2024年",
    ],
)
def test_promoted_group_skeletons_reject_unowned_qualifiers(nl, question):
    assert nl.rule_sql(question) is None


@pytest.mark.parametrize(
    ("question", "required"),
    [
        (
            "就诊量至少30次的疾病有哪些",
            ("v.disease", "GROUP BY v.disease", "HAVING COUNT(*) >= 30"),
        ),
        (
            "就诊量至少30次的诊断有哪几种",
            ("v.disease", "GROUP BY v.disease", "HAVING COUNT(*) >= 30"),
        ),
        (
            "平均花费不低于5000元的科室及其平均花费",
            ("v.department", "GROUP BY v.department", "HAVING AVG(v.cost) >= 5000"),
        ),
        (
            "统计每个科室每种患者性别的记录数",
            ("v.department", "p.gender", "GROUP BY v.department, p.gender"),
        ),
    ],
)
def test_promoted_group_synonyms_never_fall_through_partially(
    nl,
    question,
    required,
):
    sql = nl.rule_sql(question)
    if sql is not None:
        assert all(fragment in sql for fragment in required)


def test_nested_top_k_cannot_fall_through_to_flat_legacy_grouping(nl):
    question = "就诊量前3的疾病中平均费用最高的是哪个"

    assert nl._needs_llm(question)
    assert nl.rule_sql(question) is None


def test_simple_questions_still_rule_hit(nl):
    easy = ["一共有多少条就诊记录？", "各科室就诊量", "高血压有哪些症状"]
    for q in easy:
        assert not nl._needs_llm(q)
        assert nl.rule_sql(q), f"简单问法规则应命中: {q}"


def test_schema_link_hits_values_and_tables(nl):
    linked = nl._schema_link("高血压患者的平均年龄")
    assert "相关表" in linked
    # 命中疾病值域（若库含该疾病）
    assert "kg_triples" in linked or "visits" in linked


def test_rule_sql_is_select_only(nl):
    sql = nl.rule_sql("各科室就诊量")
    assert sql and sql.strip().lower().startswith("select")


@pytest.mark.parametrize(
    ("question", "required"),
    [
        ("每个城市的患者数和平均年龄分别是多少", ("FROM patients", "COUNT(*)", "AVG(age)", "GROUP BY region")),
        ("各年龄段的平均就诊费用", ("CASE WHEN p.age", "AVG(cost)", "ORDER BY age_group")),
        ("返回各年龄段的平均就诊费用", ("CASE WHEN p.age", "AVG(cost)", "ORDER BY age_group")),
        ("置信度低于0.75的三元组有几条", ("confidence < 0.75", "COUNT(*)")),
        ("拥有并发症最多的疾病是哪种", ("HAS_COMPLICATION", "GROUP BY head", "LIMIT 1")),
        ("被最多疾病列为症状的是哪个症状", ("HAS_SYMPTOM", "GROUP BY tail", "LIMIT 1")),
        ("患者最多的5个城市中平均年龄最低的城市", ("ORDER BY COUNT(*) DESC LIMIT 5", "ORDER BY avg_age ASC LIMIT 1")),
        ("各类疾病关联的临床表现数量，降序前3", ("HAS_SYMPTOM", "GROUP BY head", "LIMIT 3")),
        ("高血压就诊占全部就诊的百分比", ("CASE WHEN disease = '高血压'", "/ COUNT(*)", "FROM visits")),
        ("平均住院天数低于全体平均住院天数的疾病", ("GROUP BY disease", "HAVING AVG(los_days) <", "ORDER BY avg_value ASC")),
        ("就诊次数超过平均每种疾病就诊次数的疾病", ("GROUP BY disease", "AVG(group_count)", "ORDER BY n DESC")),
        ("在呼吸内科出现但在骨科没出现的疾病", ("department = '呼吸内科'", "EXCEPT", "department = '骨科'")),
        ("既有并发症又有用药禁忌记录的疾病", ("HAS_COMPLICATION", "INTERSECT", "CONTRAINDICATED_FOR")),
    ],
)
def test_generalized_advanced_rule_frames(nl, question, required):
    """高级规则按语义槽位泛化，不依赖评测 ID 或固定数值。"""
    sql = nl.rule_sql(question)
    assert sql
    for fragment in required:
        assert fragment in sql


def test_known_advanced_regressions_execute_equivalently(nl):
    """selection-used development cases remain deterministic regression fixtures."""
    from dki.analysis.nl2sql_eval import HARD_EVAL_SET, _load_items, _score_item

    selected = {1034, 1046, 1047, 1049, 1050, 1073, 1074}
    items = [item for item in _load_items(HARD_EVAL_SET) if item.get("id") in selected]
    assert {item["id"] for item in items} == selected
    scored = [_score_item(nl, item, "rule", nl.db_path) for item in items]
    assert all(item["correct"] for item in scored), [
        (item["id"], item.get("pred_sql"), item.get("reason"))
        for item in scored if not item["correct"]
    ]


def test_development_error_frames_execute_equivalently(nl):
    """Known development errors are regression inputs, never blind-test claims."""
    from dki.analysis.nl2sql_eval import HARD_EVAL_SET, _load_items, _score_item

    selected = {1026, 1027, 1028, 1036, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1075}
    items = [item for item in _load_items(HARD_EVAL_SET) if item.get("id") in selected]
    assert {item["id"] for item in items} == selected
    scored = [_score_item(nl, item, "rule", nl.db_path) for item in items]
    assert all(item["correct"] for item in scored), [
        (item["id"], item.get("pred_sql"), item.get("gold_sql"), item.get("reason"))
        for item in scored if not item["correct"]
    ]


@pytest.mark.parametrize(
    ("question", "gold_sql"),
    [
        (
            "查询地区为上海、性别为女且年龄不低于18岁的患者，返回患者编号、姓名和年龄。",
            "SELECT patient_id,name,age FROM patients WHERE region='上海' AND gender='女' "
            "AND age>=18 ORDER BY age DESC,patient_id",
        ),
        (
            "对诊断为2型糖尿病且费用不低于0元的就诊，按患者性别分组统计不同患者数和平均费用。",
            "SELECT p.gender,COUNT(DISTINCT p.patient_id) AS patient_count,AVG(v.cost) AS avg_cost "
            "FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
            "WHERE v.disease='2型糖尿病' AND v.cost>=0 GROUP BY p.gender ORDER BY p.gender",
        ),
        (
            "统计就诊日期从2024-07-20到2024-08-17（含首尾）期间各科室的就诊次数。",
            "SELECT department,COUNT(*) AS visit_count FROM visits "
            "WHERE visit_date BETWEEN '2024-07-20' AND '2024-08-17' "
            "GROUP BY department ORDER BY visit_count DESC,department",
        ),
        (
            "找出开具药物为丙戊酸钠且开具次数不低于2次的科室，并显示开具次数。",
            "SELECT department,COUNT(*) AS prescription_count FROM visits "
            "WHERE drug_prescribed='丙戊酸钠' GROUP BY department HAVING COUNT(*)>=2 "
            "ORDER BY prescription_count DESC,department",
        ),
        (
            "查询地区为上海、就诊科室为内分泌科且费用不低于0元的患者姓名、就诊日期和诊断。",
            "SELECT p.name,v.visit_date,v.disease FROM patients p JOIN visits v "
            "ON p.patient_id=v.patient_id WHERE p.region='上海' AND v.department='内分泌科' "
            "AND v.cost>=0 ORDER BY v.visit_date,p.patient_id",
        ),
        (
            "查询头实体类型为Disease、中文关系为临床表现且置信度不低于0.07的三元组，并返回头实体提及次数。",
            "SELECT t.head,t.relation_zh,t.tail,e.mention_count FROM kg_triples t "
            "JOIN kg_entities e ON t.head=e.name AND t.head_type=e.type "
            "WHERE t.head_type='Disease' AND t.relation_zh='临床表现' AND t.confidence>=0.07 "
            "ORDER BY e.mention_count DESC,t.head,t.tail",
        ),
        (
            "查询科室为内分泌科且费用至少比该科室平均费用高0元的就诊编号、患者编号、诊断和费用。",
            "SELECT visit_id,patient_id,disease,cost FROM visits v WHERE department='内分泌科' "
            "AND cost>=(SELECT AVG(cost)+0 FROM visits WHERE department='内分泌科') "
            "ORDER BY cost DESC,visit_id",
        ),
        (
            "对科室内分泌科中费用不低于0元的就诊按费用降序编号，返回前3名及其名次。",
            "SELECT visit_id,patient_id,cost,ROW_NUMBER() OVER "
            "(PARTITION BY department ORDER BY cost DESC,visit_id) AS fee_rank "
            "FROM visits WHERE department='内分泌科' AND cost>=0 ORDER BY fee_rank LIMIT 3",
        ),
        (
            "在科室内分泌科中按患者累计费用排名，筛选累计费用不低于0元的记录并返回前10名及排名。",
            "WITH audit_seed AS (SELECT patient_id FROM patients) "
            "SELECT patient_id,SUM(cost) AS total_cost,"
            "DENSE_RANK() OVER (ORDER BY SUM(cost) DESC) AS total_rank "
            "FROM visits WHERE department='内分泌科' GROUP BY patient_id HAVING SUM(cost)>=0 "
            "ORDER BY total_rank,patient_id LIMIT 10",
        ),
        (
            "对头实体类型为Disease、中文关系为临床表现且置信度不低于0.07的三元组，"
            "按头实体累计观察次数排名并返回前10名。",
            "SELECT head,SUM(count) AS observed_count,"
            "ROW_NUMBER() OVER (ORDER BY SUM(count) DESC,head) AS observation_rank "
            "FROM kg_triples WHERE head_type='Disease' AND relation_zh='临床表现' "
            "AND confidence>=0.07 GROUP BY head ORDER BY observation_rank LIMIT 10",
        ),
    ],
)
def test_selection_used_structural_frames_execute_equivalently(nl, question, gold_sql):
    """One regression fixture per development SQL skeleton; no holdout data."""
    from dki.analysis.nl2sql_eval import _score_item

    scored = _score_item(
        nl,
        {
            "id": "selection-used-structural-regression",
            "question": question,
            "gold_sql": gold_sql,
            "difficulty": "development",
            "category": "selection-used",
        },
        "rule",
        nl.db_path,
        strict=True,
    )
    assert scored["correct"], (
        scored.get("pred_sql"),
        scored.get("reason"),
        scored.get("gold_head"),
        scored.get("pred_head"),
    )


@pytest.mark.parametrize("head_type", ["Department", "Exam", "Symptom", "Treatment"])
def test_structured_kg_queries_accept_schema_valid_zero_row_head_types(nl, head_type):
    """A valid schema enum must not be rejected merely because this DB has no such head row."""
    question = (
        f"查询头实体类型为{head_type}、中文关系为临床表现且置信度不低于0.07的三元组，"
        "并返回头实体提及次数。"
    )
    sql = nl.rule_sql(question)
    assert sql
    assert f"t.head_type='{head_type}'" in sql
    assert "t.relation_zh='临床表现'" in sql


@pytest.mark.parametrize(
    "question",
    [
        "每个地区每种性别的患者数和平均年龄分别是多少",
        "每个地区的女性患者数和平均年龄分别是多少",
        "各地区患者数和平均年龄，按平均年龄升序",
        "患者最多的3个地区中60岁以上患者平均年龄最高的地区",
        "置信度大于0.9的三元组有多少种关系",
    ],
)
def test_advanced_rules_abstain_when_qualifiers_are_not_consumed(nl, question):
    sql = nl.rule_sql(question)
    if question == "每个地区每种性别的患者数和平均年龄分别是多少":
        assert sql
        assert "FROM patients" in sql
        assert "GROUP BY region, gender" in sql
        assert "COUNT(*)" in sql
        assert "AVG(age)" in sql
        return
    # This node originally protected against dropping "女性".  The semantic
    # obligation layer now proves and retains that qualifier, so keep the same
    # regression node while promoting only this fully consumed shape.
    if question == "每个地区的女性患者数和平均年龄分别是多少":
        assert sql
        assert "FROM patients" in sql
        assert "gender = '女'" in sql
        assert "GROUP BY region" in sql
        assert "COUNT(*)" in sql
        assert "AVG(age)" in sql
        return
    assert sql is None


def test_ranked_age_group_keeps_metric_order(nl):
    sql = nl.rule_sql("平均就诊费用最高的年龄段")
    assert sql and "ORDER BY avg_value DESC LIMIT 1" in sql


def test_kg_group_rules_respect_direction_and_topk(nl):
    count_sql = nl.rule_sql("图谱各类关系的数量分布")
    avg_sql = nl.rule_sql("每种关系的平均置信度，按升序排列")
    avg_topk_sql = nl.rule_sql("每种关系的平均置信度，按降序前3")
    complication_sql = nl.rule_sql("拥有并发症最多的疾病前5名")
    assert count_sql == (
        "SELECT relation_zh, COUNT(*) AS n FROM kg_triples "
        "GROUP BY relation_zh ORDER BY n DESC"
    )
    assert avg_sql and "ORDER BY avg_confidence ASC" in avg_sql
    assert avg_topk_sql and avg_topk_sql.endswith("ORDER BY avg_confidence DESC LIMIT 3")
    assert complication_sql and "LIMIT 5" in complication_sql
    # The compact aggregate owner is intentionally complete-surface only:
    # unparsed order/Top-K qualifiers must not be silently discarded.
    assert nl.rule_sql("每种关系的平均置信度，按升序再降序") is None
    assert nl.rule_sql("每种关系的平均置信度，按降序前3后5") is None
