import sqlite3

import pytest

from dki.analysis.nl2sql import NL2SQL
from dki.analysis.nl2sql_closed_families import (
    compile_closed_medical_analytics_sql,
)
from dki.analysis.nl2sql_structural_closure import (
    compile_structural_closure_sql,
)


DOMAINS = {
    "department": ["合成科室甲", "合成科室乙", "合成科室丙"],
    "disease": ["疾病甲", "疾病乙", "疾病丙"],
    "region": ["甲地区", "乙地区"],
    "visit_id": ["visit_demo_9"],
    "kg_entity": ["疾病甲", "检查甲", "药物甲", "科室甲"],
    "kg_entity_type": ["Disease", "Exam", "Drug", "Symptom", "Department"],
    "kg_relation": [
        "HAS_SYMPTOM",
        "HAS_COMPLICATION",
        "DIAGNOSED_BY",
        "TREATED_BY",
        "TREATED_BY_DRUG",
        "BELONGS_TO_DEPT",
    ],
}


@pytest.mark.parametrize(
    "question,expected",
    [
        (
            "为开展合成费用复盘，筛出科室为合成科室甲并且费用大于等于1234元的就诊，"
            "列出日期、患者编号与费用，先按日期后按编号升序。",
            "SELECT visit_date, patient_id, cost FROM visits "
            "WHERE department = '合成科室甲' AND cost >= 1234 "
            "ORDER BY visit_date, visit_id",
        ),
        (
            "为开展合成人群复盘，列出年龄在22至57岁之间且性别为女的"
            "患者编号、姓名与地区，按年龄和姓名升序。",
            "SELECT patient_id, name, region FROM patients "
            "WHERE age BETWEEN 22 AND 57 AND gender = '女' ORDER BY age, name",
        ),
        (
            "为开展合成联表复盘，连接患者与就诊记录，筛选地区为甲地区且"
            "就诊科室为合成科室乙的数据，返回患者编号、姓名、就诊编号、日期与疾病"
            "并按指定业务顺序排列。",
            "SELECT p.patient_id, p.name, v.visit_id, v.visit_date, v.disease "
            "FROM patients p JOIN visits v ON v.patient_id = p.patient_id "
            "WHERE p.region = '甲地区' AND v.department = '合成科室乙' "
            "ORDER BY v.visit_date DESC, p.patient_id",
        ),
        (
            "为开展合成存在性复盘，查询地区为乙地区且至少存在一次合成科室丙就诊的"
            "患者编号、姓名和年龄；存在性判断必须关联当前患者。",
            "SELECT p.patient_id, p.name, p.age FROM patients p "
            "WHERE p.region = '乙地区' AND EXISTS (SELECT 1 FROM visits v "
            "WHERE v.patient_id = p.patient_id AND v.department = '合成科室丙') "
            "ORDER BY p.patient_id",
        ),
        (
            "为开展合成图谱复盘，查找实体类型为Disease且作为头实体时拥有至少2条"
            "关系标识为HAS_SYMPTOM的图谱边的实体，边数由关联当前实体的子查询统计。",
            "SELECT e.name, e.type, e.mention_count FROM kg_entities e "
            "WHERE e.type = 'Disease' AND (SELECT COUNT(*) FROM kg_triples t "
            "WHERE t.head = e.name AND t.relation = 'HAS_SYMPTOM') >= 2 "
            "ORDER BY e.mention_count DESC, e.name",
        ),
        (
            "为开展合成分组复盘，限定科室为合成科室甲或合成科室乙，"
            "形成分组结果时按疾病分别汇总病例数与保留2位小数的费用总额，"
            "并按指定分组键与统计值稳定排序。",
            "SELECT disease, COUNT(*) AS case_count, ROUND(SUM(cost), 2) AS total_cost "
            "FROM visits WHERE department IN ('合成科室甲', '合成科室乙') "
            "GROUP BY disease ORDER BY total_cost DESC, disease",
        ),
    ],
)
def test_operational_audit_compiles_complete_synthetic_families(question, expected):
    assert compile_closed_medical_analytics_sql(question, DOMAINS) == expected


@pytest.mark.parametrize(
    "question",
    [
        "为开展合成科室为合成科室甲的隐藏过滤复盘，筛出科室为合成科室甲并且费用大于等于1元的就诊，列出日期、患者编号与费用，先按日期后按编号升序。",
        "为开展合成人群复盘，列出年龄在57至22岁之间且性别为女的患者编号、姓名与地区，按年龄和姓名升序。",
        "为开展合成图谱复盘，查找实体类型为Unknown且作为头实体时拥有至少2条关系标识为HAS_SYMPTOM的图谱边的实体，边数由关联当前实体的子查询统计。",
    ],
)
def test_operational_audit_rejects_hidden_or_unbound_slots(question):
    assert compile_closed_medical_analytics_sql(question, DOMAINS) is None


@pytest.mark.parametrize(
    "question,expected",
    [
        (
            "围绕检查口径核验，在生成审计清单时，"
            "按主要检查分组，包括空白检查组，仅输出涉及至少2种不同疾病的检查与记录数。",
            "SELECT exam, COUNT(DISTINCT disease) AS disease_count, "
            "COUNT(*) AS visit_count FROM visits GROUP BY exam "
            "HAVING COUNT(DISTINCT disease) >= 2 "
            "ORDER BY disease_count DESC, exam",
        ),
        (
            "按关系标识和头类型组合分组，只保留观察次数总和至少70的组合。",
            "SELECT relation, head_type, COUNT(*) AS edge_count, "
            "SUM(count) AS observations FROM kg_triples "
            "GROUP BY relation, head_type HAVING SUM(count) >= 70 "
            "ORDER BY observations DESC, relation, head_type",
        ),
        (
            "连接Exam实体与DIAGNOSED_BY尾边，返回检查名、提及次数和平均关系置信度。",
            "SELECT e.name AS exam, e.mention_count, "
            "ROUND(AVG(t.confidence), 4) AS avg_confidence "
            "FROM kg_entities e JOIN kg_triples t ON t.tail = e.name "
            "WHERE e.type = 'Exam' AND t.relation = 'DIAGNOSED_BY' "
            "GROUP BY e.name, e.mention_count "
            "ORDER BY e.mention_count DESC, exam",
        ),
        (
            "连接Drug实体与TREATED_BY_DRUG尾边，"
            "按药物统计关联疾病数和观察次数总和。",
            "SELECT e.name AS drug, COUNT(DISTINCT t.head) AS disease_count, "
            "SUM(t.count) AS observations FROM kg_entities e "
            "JOIN kg_triples t ON t.tail = e.name "
            "WHERE e.type = 'Drug' AND t.relation = 'TREATED_BY_DRUG' "
            "GROUP BY e.name ORDER BY disease_count DESC, drug",
        ),
        (
            "列出至少作为TREATED_BY_DRUG尾实体出现的Drug实体名称与提及次数。",
            "SELECT e.name, e.mention_count FROM kg_entities e "
            "WHERE e.type = 'Drug' AND EXISTS (SELECT 1 FROM kg_triples t "
            "WHERE t.tail = e.name AND t.relation = 'TREATED_BY_DRUG') "
            "ORDER BY e.mention_count DESC, e.name",
        ),
        (
            "寻找共享同一TREATED_BY治疗方式的不同疾病对，"
            "按疾病名称小于关系去除镜像。",
            "SELECT a.head AS disease_a, b.head AS disease_b, "
            "a.tail AS shared_tail FROM kg_triples a "
            "JOIN kg_triples b ON b.tail = a.tail AND a.head < b.head "
            "WHERE a.relation = 'TREATED_BY' "
            "AND b.relation = 'TREATED_BY' "
            "ORDER BY shared_tail, disease_a, disease_b",
        ),
        (
            "筛选年龄小于20或大于88的患者，输出编号、姓名、年龄，按年龄排序。",
            "SELECT patient_id, name, age FROM patients "
            "WHERE age < 20 OR age > 88 ORDER BY age, patient_id",
        ),
        (
            "查找疾病属于疾病甲、疾病乙或疾病丙且费用至少8000元的"
            "就诊编号、疾病与费用。",
            "SELECT visit_id, disease, cost FROM visits "
            "WHERE disease IN ('疾病甲', '疾病乙', '疾病丙') "
            "AND cost >= 8000 ORDER BY disease, cost DESC, visit_id",
        ),
        (
            "列出所有就诊住院天数都不超过18天的已就诊患者，"
            "使用NOT EXISTS排除大于18天者。",
            "SELECT p.patient_id, p.name FROM patients p WHERE EXISTS "
            "(SELECT 1 FROM visits v WHERE v.patient_id = p.patient_id) "
            "AND NOT EXISTS (SELECT 1 FROM visits v "
            "WHERE v.patient_id = p.patient_id AND v.los_days > 18) "
            "ORDER BY p.patient_id",
        ),
        (
            "连接费用至少15000元的就诊与疾病图谱边，"
            "按关系中文名统计不同就诊数。",
            "SELECT t.relation_zh, COUNT(DISTINCT v.visit_id) AS visit_count "
            "FROM visits v JOIN kg_triples t ON t.head = v.disease "
            "WHERE v.cost >= 15000 GROUP BY t.relation_zh "
            "ORDER BY visit_count DESC, t.relation_zh",
        ),
        (
            "使用LAG得到同一患者上一笔就诊费用和日期，"
            "返回当前日期、上次日期与费用差。",
            "SELECT patient_id, visit_date, previous_date, cost, "
            "ROUND(cost - previous_cost, 2) AS cost_gap FROM "
            "(SELECT patient_id, visit_date, visit_id, cost, "
            "LAG(visit_date) OVER (PARTITION BY patient_id "
            "ORDER BY visit_date, visit_id) AS previous_date, "
            "LAG(cost) OVER (PARTITION BY patient_id "
            "ORDER BY visit_date, visit_id) AS previous_cost FROM visits) "
            "WHERE previous_date IS NOT NULL "
            "ORDER BY patient_id, visit_date, visit_id",
        ),
        (
            "列出发生在患者首次就诊后0到30天内的记录，"
            "区间端点为0和30天。",
            "SELECT v.visit_id, v.patient_id, v.visit_date FROM visits v "
            "WHERE julianday(v.visit_date) - julianday((SELECT MIN(w.visit_date) "
            "FROM visits w WHERE w.patient_id = v.patient_id)) "
            "BETWEEN 0 AND 30 "
            "ORDER BY v.patient_id, v.visit_date, v.visit_id",
        ),
    ],
)
def test_closed_medical_analytics_compiles_complete_synthetic_families(
    question,
    expected,
):
    assert compile_closed_medical_analytics_sql(question, DOMAINS) == expected


@pytest.mark.parametrize(
    "question",
    [
        (
            "围绕50岁患者复核，在生成审计清单时，"
            "按主要检查分组，包括空白检查组，仅输出涉及至少2种不同疾病的检查与记录数。"
        ),
        "按费用升序分页，跳过最低的3条后取4条就诊；OFFSET为2、LIMIT为4。",
        "筛选年龄小于90或大于20的患者，输出编号、姓名、年龄，按年龄排序。",
        "列出发生在患者首次就诊后30到10天内的记录，区间端点为30和10天。",
        "在类型Unknown中按提及次数取前5个实体，返回名称与次数；数量5。",
    ],
)
def test_closed_medical_analytics_rejects_hidden_or_inconsistent_slots(question):
    assert compile_closed_medical_analytics_sql(question, DOMAINS) is None


@pytest.fixture()
def closed_family_db(tmp_path):
    path = tmp_path / "closed-family.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE patients (
            patient_id TEXT PRIMARY KEY, name TEXT, gender TEXT,
            age INTEGER, region TEXT
        );
        CREATE TABLE visits (
            visit_id TEXT PRIMARY KEY, patient_id TEXT, visit_date TEXT,
            department TEXT, disease TEXT, drug_prescribed TEXT, exam TEXT,
            cost REAL, los_days INTEGER
        );
        CREATE TABLE kg_entities (name TEXT, type TEXT, mention_count INTEGER);
        CREATE TABLE kg_triples (
            head TEXT, head_type TEXT, relation TEXT, relation_zh TEXT,
            tail TEXT, tail_type TEXT, confidence REAL, count INTEGER
        );
        INSERT INTO patients VALUES ('p1', '甲', '女', 31, '甲地区');
        INSERT INTO visits VALUES (
            'visit_demo_9', 'p1', '2026-01-01', '科室甲', '疾病甲',
            '药物甲', '检查甲', 100.0, 2
        );
        INSERT INTO kg_entities VALUES ('疾病甲', 'Disease', 9);
        INSERT INTO kg_entities VALUES ('检查甲', 'Exam', 8);
        INSERT INTO kg_triples VALUES (
            '疾病甲', 'Disease', 'DIAGNOSED_BY', '诊断检查',
            '检查甲', 'Exam', 0.9, 2
        );
        INSERT INTO kg_triples VALUES (
            '疾病甲', 'Disease', 'HAS_COMPLICATION', '并发症',
            '疾病乙', 'Disease', 0.8, 1
        );
        INSERT INTO kg_triples VALUES (
            '疾病甲', 'Disease', 'TREATED_BY', '治疗方式',
            '治疗甲', 'Treatment', 0.8, 1
        );
        INSERT INTO kg_triples VALUES (
            '疾病甲', 'Disease', 'TREATED_BY_DRUG', '治疗药物',
            '药物甲', 'Drug', 0.8, 1
        );
        INSERT INTO kg_triples VALUES (
            '药物甲', 'Drug', 'CONTRAINDICATED_FOR', '禁忌疾病',
            '疾病乙', 'Disease', 0.8, 1
        );
        """
    )
    connection.commit()
    connection.close()
    return path


def test_model_first_runs_closed_family_before_llm(closed_family_db, monkeypatch):
    nl = NL2SQL(
        db_path=closed_family_db,
        llm_strategy="legacy_sql",
        agent_rule_policy="model_first",
    )
    monkeypatch.setattr(
        nl,
        "_llm_pipeline",
        lambda _question: pytest.fail("closed family must run before the model"),
    )
    response = nl.to_sql(
        "按主要检查分组，包括空白检查组，"
        "仅输出涉及至少1种不同疾病的检查与记录数。"
    )
    assert response["engine"] == "rule"
    assert response["plan"]["route"] == "bounded_rule_before_model"


@pytest.mark.parametrize(
    ("question", "expected"),
    (
        (
            "第一个CTE计算每位患者总费用（无就诊按0），第二个计算各地区平均与最大总费用；"
            "返回高于地区平均但低于地区最大值的患者，费用保留2位小数。",
            "WITH patient_totals AS (SELECT p.patient_id, p.region, "
            "COALESCE(SUM(v.cost), 0) AS total_cost FROM patients p "
            "LEFT JOIN visits v ON v.patient_id = p.patient_id "
            "GROUP BY p.patient_id, p.region), region_stats AS (SELECT region, "
            "AVG(total_cost) AS avg_total, MAX(total_cost) AS max_total "
            "FROM patient_totals GROUP BY region) SELECT p.patient_id, p.region, "
            "ROUND(p.total_cost, 2) AS total_cost FROM patient_totals p "
            "JOIN region_stats r ON r.region = p.region "
            "WHERE p.total_cost > r.avg_total AND p.total_cost < r.max_total "
            "ORDER BY p.region, p.total_cost DESC, p.patient_id",
        ),
        (
            "返回同时至少存在一条关系HAS_COMPLICATION出边和一条关系TREATED_BY出边的"
            "头实体名称，按名称排列。",
            "SELECT e.name FROM kg_entities e WHERE EXISTS (SELECT 1 FROM "
            "kg_triples a WHERE a.head = e.name AND a.relation = "
            "'HAS_COMPLICATION') AND EXISTS (SELECT 1 FROM kg_triples b "
            "WHERE b.head = e.name AND b.relation = 'TREATED_BY') "
            "ORDER BY e.name",
        ),
        (
            "按患者分区并按日期、就诊ID升序，在完整分区窗口用LAST_VALUE给每行附上"
            "末条就诊费用。",
            "SELECT patient_id, visit_id, visit_date, cost, LAST_VALUE(cost) "
            "OVER (PARTITION BY patient_id ORDER BY visit_date, visit_id "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) "
            "AS last_cost FROM visits ORDER BY patient_id, visit_date, visit_id",
        ),
        (
            "寻找第一跳关系TREATED_BY_DRUG、第二跳关系CONTRAINDICATED_FOR且第一跳尾等于"
            "第二跳头的两跳路径，返回起点、中间点、终点并依次排列。",
            "SELECT a.head AS start_entity, a.tail AS middle_entity, "
            "b.tail AS end_entity FROM kg_triples a JOIN kg_triples b "
            "ON b.head = a.tail WHERE a.relation = 'TREATED_BY_DRUG' "
            "AND b.relation = 'CONTRAINDICATED_FOR' "
            "ORDER BY start_entity, middle_entity, end_entity",
        ),
        (
            "按患者及日期、就诊ID排序，用LEAD取得后2次日期并计算距后2次的整数天数，"
            "返回当前与后次日期及间隔。",
            "SELECT patient_id, visit_id, visit_date, LEAD(visit_date, 2) "
            "OVER (PARTITION BY patient_id ORDER BY visit_date, visit_id) "
            "AS next_date, CAST(julianday(LEAD(visit_date, 2) OVER "
            "(PARTITION BY patient_id ORDER BY visit_date, visit_id)) - "
            "julianday(visit_date) AS INTEGER) AS days_to_next FROM visits "
            "ORDER BY patient_id, visit_date, visit_id",
        ),
    ),
)
def test_output_complete_structural_contracts_are_whole_question_compilers(
    closed_family_db,
    question,
    expected,
):
    nl = NL2SQL(closed_family_db)
    assert nl._compile_output_complete_structural_sql(nl._normalize(question)) == expected


@pytest.mark.parametrize(
    "question",
    (
        "返回同时至少存在一条关系UNKNOWN_RELATION出边和一条关系TREATED_BY出边的头实体名称，按名称排列。",
        "返回同时至少存在一条关系TREATED_BY出边和一条关系TREATED_BY出边的头实体名称，按名称排列。",
        "按患者分区并按日期、就诊ID升序，在完整分区窗口用LAST_VALUE给每行附上首条就诊费用。",
        "第一个CTE计算每位患者总费用（无就诊按0），第二个计算各地区平均与最大总费用；返回高于地区平均的患者，费用保留2位小数。",
        "寻找第一跳关系UNKNOWN_RELATION、第二跳关系TREATED_BY且第一跳尾等于第二跳头的两跳路径，返回起点、中间点、终点并依次排列。",
        "按患者及日期、就诊ID排序，用LEAD取得后2次日期并计算距后3次的整数天数，返回当前与后次日期及间隔。",
    ),
)
def test_output_complete_structural_contracts_fail_closed(
    closed_family_db,
    question,
):
    nl = NL2SQL(closed_family_db)
    assert nl._compile_output_complete_structural_sql(nl._normalize(question)) is None


@pytest.mark.parametrize(
    ("question", "expected"),
    (
        (
            "对年龄不低于45岁的患者，按地区统计人数及平均年龄（保留2位小数），"
            "按人数降序、地区升序排列。",
            "SELECT region, COUNT(*) AS patient_count, ROUND(AVG(age), 2) "
            "AS avg_age FROM patients WHERE age >= 45 GROUP BY region "
            "ORDER BY patient_count DESC, region ASC",
        ),
        (
            "筛选就诊日期以2024-开头的记录，按科室统计次数并按次数降序、科室升序排列。",
            "SELECT department, COUNT(*) AS visit_count FROM visits "
            "WHERE visit_date LIKE '2024-%' GROUP BY department "
            "ORDER BY visit_count DESC, department ASC",
        ),
        (
            "按科室计算平均费用并保留1位小数，仅返回平均费用大于3500元的科室，"
            "按平均费用降序、科室排列。",
            "SELECT department, ROUND(AVG(cost), 1) AS avg_cost FROM visits "
            "GROUP BY department HAVING AVG(cost) > 3500 "
            "ORDER BY avg_cost DESC, department",
        ),
        (
            "返回不存在费用不少于6000元就诊的患者ID和姓名，按患者ID排列。",
            "SELECT p.patient_id, p.name FROM patients p WHERE NOT EXISTS "
            "(SELECT 1 FROM visits v WHERE v.patient_id = p.patient_id "
            "AND v.cost >= 6000) ORDER BY p.patient_id",
        ),
        (
            "连接甲地区患者及其日期先后的两次就诊，仅保留间隔不超过45天的配对，"
            "返回患者ID、两次就诊ID和间隔天数，按患者ID、前后日期排列。",
            "SELECT p.patient_id, a.visit_id, b.visit_id, "
            "CAST(julianday(b.visit_date) - julianday(a.visit_date) AS INTEGER) "
            "AS gap_days FROM patients p JOIN visits a ON a.patient_id = "
            "p.patient_id JOIN visits b ON b.patient_id = p.patient_id "
            "AND b.visit_date > a.visit_date WHERE p.region = '甲地区' AND "
            "julianday(b.visit_date) - julianday(a.visit_date) <= 45 "
            "ORDER BY p.patient_id, a.visit_date, b.visit_date",
        ),
        (
            "将Exam实体作为尾端连接三元组，仅保留关系DIAGNOSED_BY，返回实体名、"
            "提及次数、头实体和观察次数，按实体名、头实体排列。",
            "SELECT e.name, e.mention_count, t.head, t.count FROM kg_entities e "
            "JOIN kg_triples t ON t.tail = e.name WHERE e.type = 'Exam' "
            "AND t.relation = 'DIAGNOSED_BY' ORDER BY e.name, t.head",
        ),
        (
            "用CTE生成第一跳关系为DIAGNOSED_BY的所有两跳路径，再按起点统计不同终点数，"
            "按终点数降序、起点排列。",
            "WITH paths AS (SELECT a.head AS start_entity, b.tail AS end_entity "
            "FROM kg_triples a JOIN kg_triples b ON b.head = a.tail "
            "WHERE a.relation = 'DIAGNOSED_BY') SELECT start_entity, "
            "COUNT(DISTINCT end_entity) AS reachable_count FROM paths "
            "GROUP BY start_entity ORDER BY reachable_count DESC, start_entity",
        ),
        (
            "用CTE取得就诊疾病与Disease实体的交集，再连接头实体不同邻居数；"
            "没有出边按0处理，按邻居数降序、疾病排列。",
            "WITH visit_diseases AS (SELECT DISTINCT disease FROM visits), "
            "graph_diseases AS (SELECT name FROM kg_entities WHERE type = "
            "'Disease'), matched AS (SELECT disease FROM visit_diseases "
            "INTERSECT SELECT name FROM graph_diseases), edge_counts AS "
            "(SELECT head, COUNT(DISTINCT tail) AS neighbor_count FROM "
            "kg_triples GROUP BY head) SELECT m.disease, "
            "COALESCE(e.neighbor_count, 0) AS neighbor_count FROM matched m "
            "LEFT JOIN edge_counts e ON e.head = m.disease "
            "ORDER BY neighbor_count DESC, m.disease",
        ),
        (
            "按患者分区并按日期、就诊ID排序，计算从分区首条到当前行的累计费用并保留1位小数。",
            "SELECT patient_id, visit_id, visit_date, cost, "
            "ROUND(SUM(cost) OVER (PARTITION BY patient_id ORDER BY visit_date, "
            "visit_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 1) "
            "AS running_cost FROM visits ORDER BY patient_id, visit_date, visit_id",
        ),
        (
            "按科室分区、费用降序和就诊ID升序用ROW_NUMBER编号，"
            "返回每科室序号不超过2的前2条就诊。",
            "WITH ranked AS (SELECT department, visit_id, disease, cost, "
            "ROW_NUMBER() OVER (PARTITION BY department ORDER BY cost DESC, "
            "visit_id) AS rn FROM visits) SELECT department, visit_id, disease, "
            "cost, rn FROM ranked WHERE rn <= 2 ORDER BY department, rn",
        ),
    ),
)
def test_structural_development_families_generalize_with_synthetic_slots(
    closed_family_db,
    question,
    expected,
):
    nl = NL2SQL(closed_family_db)
    assert nl._compile_output_complete_structural_sql(nl._normalize(question)) == expected


@pytest.mark.parametrize(
    "question",
    (
        "连接不存在地区患者及其日期先后的两次就诊，仅保留间隔不超过45天的配对，返回患者ID、两次就诊ID和间隔天数，按患者ID、前后日期排列。",
        "将Exam实体作为尾端连接三元组，仅保留关系UNKNOWN_RELATION，返回实体名、提及次数、头实体和观察次数，按实体名、头实体排列。",
        "按科室分区、费用降序和就诊ID升序用ROW_NUMBER编号，返回每科室序号不超过2的前3条就诊。",
        "对年龄不低于151岁的患者，按地区统计人数及平均年龄（保留2位小数），按人数降序、地区升序排列。",
    ),
)
def test_structural_development_families_reject_unbound_or_inconsistent_slots(
    closed_family_db,
    question,
):
    nl = NL2SQL(closed_family_db)
    assert nl._compile_output_complete_structural_sql(nl._normalize(question)) is None


@pytest.mark.parametrize(
    ("question", "expected"),
    (
        (
            "列出性别为女、年龄在20至60岁（含边界）且地区为甲地区的患者ID、"
            "姓名和年龄，并按年龄升序、患者ID升序排列。",
            "SELECT patient_id, name, age FROM patients WHERE gender = '女' "
            "AND age BETWEEN 20 AND 60 AND region = '甲地区' "
            "ORDER BY age ASC, patient_id ASC",
        ),
        (
            "返回费用不少于2500元且同时高于其科室平均费用和疾病平均费用的"
            "就诊ID、科室、疾病与费用，按费用降序、就诊ID排列。",
            "SELECT v.visit_id, v.department, v.disease, v.cost FROM visits v "
            "WHERE v.cost >= 2500 AND v.cost > (SELECT AVG(x.cost) FROM visits x "
            "WHERE x.department = v.department) AND v.cost > "
            "(SELECT AVG(y.cost) FROM visits y WHERE y.disease = v.disease) "
            "ORDER BY v.cost DESC, v.visit_id",
        ),
        (
            "将年龄低于25岁患者的地区标记为“地区来源”，与费用大于9000元就诊的"
            "科室标记为“科室来源”后做保留重复的纵向合并。",
            "SELECT '地区来源' AS source, region AS value FROM patients "
            "WHERE age < 25 UNION ALL SELECT '科室来源', department FROM visits "
            "WHERE cost > 9000 ORDER BY source, value",
        ),
        (
            "从Exam实体名中减去就诊不同疾病名，返回剩余名称并排列。",
            "SELECT name FROM kg_entities WHERE type = 'Exam' EXCEPT "
            "SELECT DISTINCT disease FROM visits ORDER BY name",
        ),
        (
            "关联患者与就诊，按患者计算末次与首次就诊日期之间的整数天数跨度，"
            "返回患者ID、姓名和跨度，按跨度降序、患者ID升序排列。",
            "SELECT p.patient_id, p.name, CAST(julianday(MAX(v.visit_date)) - "
            "julianday(MIN(v.visit_date)) AS INTEGER) AS observed_span_days "
            "FROM patients p JOIN visits v ON v.patient_id = p.patient_id "
            "GROUP BY p.patient_id, p.name "
            "ORDER BY observed_span_days DESC, p.patient_id ASC",
        ),
        (
            "按科室分区、费用降序和就诊ID升序用ROW_NUMBER编号，"
            "返回每科室序号不超过2的前二条就诊。",
            "WITH ranked AS (SELECT department, visit_id, disease, cost, "
            "ROW_NUMBER() OVER (PARTITION BY department ORDER BY cost DESC, "
            "visit_id) AS rn FROM visits) SELECT department, visit_id, disease, "
            "cost, rn FROM ranked WHERE rn <= 2 ORDER BY department, rn",
        ),
    ),
)
def test_additional_structural_families_preserve_complete_output_contracts(
    closed_family_db,
    question,
    expected,
):
    nl = NL2SQL(closed_family_db)
    assert nl._compile_output_complete_structural_sql(nl._normalize(question)) == expected


@pytest.mark.parametrize(
    "question",
    (
        "列出性别为女、年龄在60至20岁（含边界）且地区为甲地区的患者ID、姓名和年龄，并按年龄升序、患者ID升序排列。",
        "列出性别为女、年龄在20至60岁（含边界）且地区为不存在地区的患者ID、姓名和年龄，并按年龄升序、患者ID升序排列。",
        "从Unknown实体名中减去就诊不同疾病名，返回剩余名称并排列。",
        "按科室分区、费用降序和就诊ID升序用ROW_NUMBER编号，返回每科室序号不超过2的前三条就诊。",
    ),
)
def test_additional_structural_families_fail_closed_on_bad_slots(
    closed_family_db,
    question,
):
    nl = NL2SQL(closed_family_db)
    assert nl._compile_output_complete_structural_sql(nl._normalize(question)) is None


@pytest.mark.parametrize(
    ("question", "expected"),
    (
        (
            "列出实体类型为Drug或Exam且提及次数至多12次的实体，返回name、type、"
            "mention_count，并按第一列升序排列。",
            "SELECT name, type, mention_count FROM kg_entities WHERE type IN "
            "('Drug', 'Exam') AND mention_count <= 12 ORDER BY 1 ASC",
        ),
        (
            "列出日期不早于2024-02-01且早于2024-05-01的就诊ID、科室和费用，"
            "按日期、就诊ID升序排列。",
            "SELECT visit_id, department, cost FROM visits WHERE visit_date >= "
            "'2024-02-01' AND visit_date < '2024-05-01' "
            "ORDER BY visit_date, visit_id",
        ),
        (
            "列出星期编号为0或6的就诊ID、日期和疾病，其中星期编号由strftime的%w计算，"
            "按日期、就诊ID升序排列。",
            "SELECT visit_id, visit_date, disease FROM visits WHERE "
            "strftime('%w', visit_date) IN ('0', '6') ORDER BY visit_date, visit_id",
        ),
        (
            "忽略年份，按strftime的%m转换成整数后的月份编号汇总费用并保留1位小数，"
            "按月份编号升序排列。",
            "SELECT CAST(strftime('%m', visit_date) AS INTEGER) AS "
            "visit_month_number, ROUND(SUM(cost), 1) AS total_cost FROM visits "
            "GROUP BY CAST(strftime('%m', visit_date) AS INTEGER) "
            "ORDER BY visit_month_number",
        ),
        (
            "关联患者与就诊，在2024-01-01至2024-06-30（含边界）内按月份和地区"
            "汇总费用并保留1位小数，按月份、地区升序排列。",
            "SELECT p.region, substr(v.visit_date, 1, 7) AS visit_month, "
            "ROUND(SUM(v.cost), 1) AS total_cost FROM patients p JOIN visits v "
            "ON v.patient_id = p.patient_id WHERE v.visit_date BETWEEN "
            "'2024-01-01' AND '2024-06-30' GROUP BY p.region, "
            "substr(v.visit_date, 1, 7) ORDER BY visit_month, p.region",
        ),
        (
            "以全表最晚就诊日期为基准，统计该日期前90天起各科室的就诊次数，"
            "按次数降序、科室升序排列。",
            "SELECT department, COUNT(*) AS recent_count FROM visits WHERE "
            "visit_date >= date((SELECT MAX(visit_date) FROM visits), '-90 days') "
            "GROUP BY department ORDER BY recent_count DESC, department",
        ),
        (
            "返回每位患者不存在更晚日期就诊的记录之就诊ID、日期和疾病，"
            "若末次日期有多条则全部保留，按患者ID、就诊ID升序排列。",
            "SELECT v.visit_id, v.visit_date, v.disease FROM visits v WHERE NOT "
            "EXISTS (SELECT 1 FROM visits later WHERE later.patient_id = "
            "v.patient_id AND later.visit_date > v.visit_date) "
            "ORDER BY v.patient_id, v.visit_id",
        ),
        (
            "将同一患者不同年份的就诊两两配对，要求后一次就诊年份大于前一次，"
            "按患者统计这种跨年配对数，按配对数降序、患者ID升序排列。",
            "SELECT a.patient_id, COUNT(*) AS cross_year_pairs FROM visits a "
            "JOIN visits b ON b.patient_id = a.patient_id AND "
            "substr(b.visit_date, 1, 4) > substr(a.visit_date, 1, 4) "
            "GROUP BY a.patient_id ORDER BY cross_year_pairs DESC, a.patient_id",
        ),
        (
            "连接患者与就诊，仅统计性别为女且日期不早于2024-03-01的记录，"
            "按地区和科室统计次数及平均费用（保留1位小数），"
            "按地区、次数降序、科室排列。",
            "SELECT p.region, v.department, COUNT(*) AS visit_count, "
            "ROUND(AVG(v.cost), 1) AS avg_cost FROM patients p JOIN visits v "
            "ON v.patient_id = p.patient_id WHERE p.gender = '女' AND "
            "v.visit_date >= '2024-03-01' GROUP BY p.region, v.department "
            "ORDER BY p.region, visit_count DESC, v.department",
        ),
        (
            "把三元组头尾两端连接实体表，仅保留观察次数不少于2次的边，"
            "按关系统计不同头、尾实体数和两端平均提及次数之和（保留1位小数），"
            "按关系排列。",
            "SELECT t.relation, COUNT(DISTINCT t.head) AS head_count, "
            "COUNT(DISTINCT t.tail) AS tail_count, "
            "ROUND(AVG(h.mention_count + e.mention_count), 1) AS "
            "endpoint_mentions FROM kg_triples t JOIN kg_entities h ON "
            "h.name = t.head JOIN kg_entities e ON e.name = t.tail WHERE "
            "t.count >= 2 GROUP BY t.relation ORDER BY t.relation",
        ),
        (
            "连接患者与就诊，仅统计年龄不低于55岁的记录，按地区和科室计算"
            "平均费用并保留1位小数，按平均费用降序、地区、科室排列。",
            "SELECT p.region, v.department, ROUND(AVG(v.cost), 1) AS avg_cost "
            "FROM patients p JOIN visits v ON v.patient_id = p.patient_id "
            "WHERE p.age >= 55 GROUP BY p.region, v.department "
            "ORDER BY avg_cost DESC, p.region, v.department",
        ),
        (
            "把关系HAS_COMPLICATION的三元组头尾两端分别连接实体表，"
            "返回头尾名称、两端提及次数和置信度，按头、尾排列。",
            "SELECT h.name AS head, h.mention_count AS head_mentions, e.name AS "
            "tail, e.mention_count AS tail_mentions, t.confidence FROM "
            "kg_triples t JOIN kg_entities h ON h.name = t.head JOIN "
            "kg_entities e ON e.name = t.tail WHERE t.relation = "
            "'HAS_COMPLICATION' ORDER BY h.name, e.name",
        ),
        (
            "依次连接患者、就诊和疾病头实体三元组，筛选地区甲地区及关系DIAGNOSED_BY，"
            "返回患者ID、疾病和尾实体，按患者ID、疾病、尾实体排列。",
            "SELECT p.patient_id, v.disease, t.tail FROM patients p JOIN visits v "
            "ON v.patient_id = p.patient_id JOIN kg_triples t ON t.head = "
            "v.disease WHERE p.region = '甲地区' AND t.relation = "
            "'DIAGNOSED_BY' ORDER BY p.patient_id, v.disease, t.tail",
        ),
        (
            "连接患者、就诊和与就诊drug_prescribed同名的Drug实体，"
            "仅保留年龄不低于50岁，返回患者ID、年龄、实体名和提及次数，"
            "按患者ID、实体名排列。",
            "SELECT p.patient_id, p.age, e.name, e.mention_count FROM patients p "
            "JOIN visits v ON v.patient_id = p.patient_id JOIN kg_entities e ON "
            "e.name = v.drug_prescribed WHERE p.age >= 50 AND e.type = 'Drug' "
            "ORDER BY p.patient_id, e.name",
        ),
        (
            "对每次就诊，仅保留费用高于同患者全部就诊平均费用的记录，"
            "返回就诊ID、疾病、科室和费用，按费用降序、就诊ID排列。",
            "SELECT v.visit_id, v.disease, v.department, v.cost FROM visits v "
            "WHERE v.cost > (SELECT AVG(x.cost) FROM visits x WHERE "
            "x.patient_id = v.patient_id) ORDER BY v.cost DESC, v.visit_id",
        ),
        (
            "返回既有合成科室甲就诊又有合成科室乙就诊，且不存在住院天数大于7天"
            "就诊的患者ID和姓名，按患者ID排列。",
            "SELECT p.patient_id, p.name FROM patients p WHERE EXISTS "
            "(SELECT 1 FROM visits a WHERE a.patient_id = p.patient_id AND "
            "a.department = '合成科室甲') AND EXISTS (SELECT 1 FROM visits b "
            "WHERE b.patient_id = p.patient_id AND b.department = '合成科室乙') "
            "AND NOT EXISTS (SELECT 1 FROM visits c WHERE c.patient_id = "
            "p.patient_id AND c.los_days > 7) ORDER BY p.patient_id",
        ),
        (
            "先用CTE统计2024-04-01及以后每位患者的就诊次数和总费用，"
            "再连接患者姓名；总费用保留1位小数，按总费用降序、患者ID排列。",
            "WITH pc AS (SELECT patient_id, COUNT(*) AS visit_count, SUM(cost) "
            "AS total_cost FROM visits WHERE visit_date >= '2024-04-01' "
            "GROUP BY patient_id) SELECT p.patient_id, p.name, pc.visit_count, "
            "ROUND(pc.total_cost, 1) AS total_cost FROM patients p JOIN pc ON "
            "pc.patient_id = p.patient_id ORDER BY total_cost DESC, p.patient_id",
        ),
        (
            "用递归CTE从HAS_COMPLICATION边的深度1开始沿尾实体继续扩展，"
            "最多深度2，新尾实体在路径中出现位置等于0以避免环。",
            "WITH RECURSIVE paths(start_entity, current_entity, depth, path) AS "
            "(SELECT head, tail, 1, '|' || head || '|' || tail || '|' FROM "
            "kg_triples WHERE relation = 'HAS_COMPLICATION' UNION ALL SELECT "
            "p.start_entity, t.tail, p.depth + 1, p.path || t.tail || '|' FROM "
            "paths p JOIN kg_triples t ON t.head = p.current_entity WHERE "
            "p.depth < 2 AND instr(p.path, '|' || t.tail || '|') = 0) SELECT "
            "start_entity, current_entity, depth FROM paths "
            "ORDER BY start_entity, depth, current_entity",
        ),
        (
            "按科室分区，计算每次费用占科室总费用的百分比，乘100.0并保留1位小数，"
            "按科室、百分比降序、就诊ID排列。",
            "SELECT department, visit_id, cost, ROUND(100.0 * cost / SUM(cost) "
            "OVER (PARTITION BY department), 1) AS department_cost_percent "
            "FROM visits ORDER BY department, department_cost_percent DESC, visit_id",
        ),
        (
            "把单次费用除以住院天数加2.0作为调整日均费用，按科室求平均值并保留3位小数，"
            "按平均值降序、科室升序排列。",
            "SELECT department, ROUND(AVG(cost / (los_days + 2.0)), 3) "
            "AS avg_cost_per_adjusted_day FROM visits GROUP BY department "
            "ORDER BY avg_cost_per_adjusted_day DESC, department",
        ),
        (
            "将Drug类型实体与其作为头实体的三元组左连接，返回出边数量最多的5个实体"
            "及其提及次数和出边数；先按出边数、再按提及次数降序，最后按名称升序。",
            "SELECT e.name, e.mention_count, COUNT(t.rowid) AS outgoing_count "
            "FROM kg_entities e LEFT JOIN kg_triples t ON t.head = e.name "
            "WHERE e.type = 'Drug' GROUP BY e.name, e.mention_count "
            "ORDER BY outgoing_count DESC, e.mention_count DESC, e.name ASC LIMIT 5",
        ),
        (
            "返回就诊日期中月份两位字符等于08的就诊ID、日期和科室，"
            "按日期、就诊ID升序排列。",
            "SELECT visit_id, visit_date, department FROM visits "
            "WHERE substr(visit_date, 6, 2) = '08' ORDER BY visit_date, visit_id",
        ),
        (
            "对每个患者列出日期等于该患者最早就诊日期的所有就诊，"
            "返回患者ID、就诊ID和日期，按患者ID、就诊ID升序排列。",
            "SELECT v1.patient_id, v1.visit_id, v1.visit_date FROM visits v1 "
            "WHERE v1.visit_date = (SELECT MIN(v2.visit_date) FROM visits v2 "
            "WHERE v2.patient_id = v1.patient_id) "
            "ORDER BY v1.patient_id, v1.visit_id",
        ),
        (
            "从全表最早就诊日期加180天所得日期起，按科室计算平均费用并保留3位小数，"
            "按平均费用降序、科室升序排列。",
            "SELECT department, ROUND(AVG(cost), 3) AS avg_cost FROM visits "
            "WHERE visit_date >= (SELECT date(MIN(visit_date), '+180 days') "
            "FROM visits) GROUP BY department ORDER BY avg_cost DESC, department",
        ),
        (
            "按关系统计边数，仅返回不同头实体数不少于12个的关系，"
            "按边数降序、关系排列。",
            "SELECT relation, COUNT(*) AS edge_count FROM kg_triples "
            "GROUP BY relation HAVING COUNT(DISTINCT head) >= 12 "
            "ORDER BY edge_count DESC, relation",
        ),
        (
            "按实体类型计算平均提及次数并保留3位小数，仅返回平均值大于9.5次的类型，"
            "按平均值降序、类型排列。",
            "SELECT type, ROUND(AVG(mention_count), 3) AS avg_mentions "
            "FROM kg_entities GROUP BY type HAVING AVG(mention_count) > 9.5 "
            "ORDER BY avg_mentions DESC, type",
        ),
        (
            "以提及次数是否不少于18次划分为“重点”和“常规”，统计两档实体数并按档位名称升序排列。",
            "SELECT CASE WHEN mention_count >= 18 THEN '重点' ELSE '常规' END "
            "AS frequency_band, COUNT(*) AS entity_count FROM kg_entities "
            "GROUP BY CASE WHEN mention_count >= 18 THEN '重点' ELSE '常规' END "
            "ORDER BY frequency_band",
        ),
        (
            "计算各疾病平均就诊费用并保留3位小数，返回平均费用最高的4种疾病；并列时疾病升序。",
            "SELECT disease, ROUND(AVG(cost), 3) AS avg_cost FROM visits "
            "GROUP BY disease ORDER BY avg_cost DESC, disease ASC LIMIT 4",
        ),
        (
            "关联患者与就诊，按性别和科室组合计算平均费用并保留1位小数，"
            "返回平均费用最高的8个组合；并列时按性别、科室升序。",
            "SELECT p.gender, v.department, ROUND(AVG(v.cost), 1) AS avg_cost "
            "FROM patients p JOIN visits v ON v.patient_id = p.patient_id "
            "GROUP BY p.gender, v.department ORDER BY avg_cost DESC, "
            "p.gender ASC, v.department ASC LIMIT 8",
        ),
        (
            "把三元组头实体连接到实体表，以置信度乘提及次数作为每条记录的分值，"
            "按关系计算平均分并保留3位小数，返回平均分最高的6种关系；并列时关系标识升序。",
            "SELECT t.relation, ROUND(AVG(t.confidence * e.mention_count), 3) "
            "AS weighted_score FROM kg_triples t JOIN kg_entities e "
            "ON e.name = t.head GROUP BY t.relation "
            "ORDER BY weighted_score DESC, t.relation ASC LIMIT 6",
        ),
        (
            "按日期统计每日就诊次数，返回就诊次数最多的9个日期；并列时日期升序。",
            "SELECT visit_date, COUNT(*) AS daily_visits FROM visits "
            "GROUP BY visit_date ORDER BY daily_visits DESC, visit_date ASC LIMIT 9",
        ),
        (
            "将就诊连接患者，查询2024-03-01及以后主诊断为疾病乙的患者姓名、性别、"
            "日期和检查，按日期、姓名排列。",
            "SELECT p.name, p.gender, v.visit_date, v.exam FROM visits v "
            "JOIN patients p ON p.patient_id = v.patient_id "
            "WHERE v.disease = '疾病乙' AND v.visit_date >= '2024-03-01' "
            "ORDER BY v.visit_date, p.name",
        ),
        (
            "连接患者与就诊，仅考虑年龄不低于47岁的患者，按性别和疾病统计不同患者数，"
            "按人数降序、性别、疾病排列。",
            "SELECT p.gender, v.disease, COUNT(DISTINCT p.patient_id) "
            "AS patient_count FROM patients p JOIN visits v "
            "ON v.patient_id = p.patient_id WHERE p.age >= 47 "
            "GROUP BY p.gender, v.disease "
            "ORDER BY patient_count DESC, p.gender, v.disease",
        ),
        (
            "将不同就诊科室与Department实体名做去重并集，按名称排列。",
            "SELECT DISTINCT department AS name FROM visits UNION "
            "SELECT name FROM kg_entities WHERE type = 'Department' ORDER BY name",
        ),
    ),
)
def test_structural_closure_compiles_synthetic_whole_question_contracts(
    question,
    expected,
):
    assert compile_structural_closure_sql(question, DOMAINS) == expected


@pytest.mark.parametrize(
    "question",
    (
        "列出实体类型为Drug或Drug且提及次数至多12次的实体，返回name、type、mention_count，并按第一列升序排列。",
        "列出日期不早于2024-05-01且早于2024-02-01的就诊ID、科室和费用，按日期、就诊ID升序排列。",
        "把关系UNKNOWN_RELATION的三元组头尾两端分别连接实体表，返回头尾名称、两端提及次数和置信度，按头、尾排列。",
        "返回既有不存在科室就诊又有合成科室乙就诊，且不存在住院天数大于7天就诊的患者ID和姓名，按患者ID排列。",
        "用递归CTE从HAS_COMPLICATION边的深度2开始沿尾实体继续扩展，最多深度3，新尾实体在路径中出现位置等于0以避免环。",
        "连接患者与就诊，仅统计年龄不低于55岁的记录，按地区和科室计算平均费用并保留1位小数，按平均费用降序、地区、科室排列，并额外返回患者姓名。",
        "返回就诊日期中月份两位字符等于13的就诊ID、日期和科室，按日期、就诊ID升序排列。",
        "将Unknown类型实体与其作为头实体的三元组左连接，返回出边数量最多的5个实体及其提及次数和出边数；先按出边数、再按提及次数降序，最后按名称升序。",
        "将Drug类型实体与其作为头实体的三元组左连接，返回出边数量最多的0个实体及其提及次数和出边数；先按出边数、再按提及次数降序，最后按名称升序。",
        "从全表最早就诊日期加0天所得日期起，按科室计算平均费用并保留3位小数，按平均费用降序、科室升序排列。",
        "按关系统计边数，仅返回不同头实体数不少于12个的关系，按边数降序、关系排列，并返回置信度。",
        "计算各疾病平均就诊费用并保留3位小数，返回平均费用最高的0种疾病；并列时疾病升序。",
        "将就诊连接患者，查询2024-03-01及以后主诊断为未知疾病的患者姓名、性别、日期和检查，按日期、姓名排列。",
        "连接患者与就诊，仅考虑年龄不低于151岁的患者，按性别和疾病统计不同患者数，按人数降序、性别、疾病排列。",
        "将不同就诊科室与Unknown实体名做去重并集，按名称排列。",
        "按日期统计每日就诊次数，返回就诊次数最多的9个日期；并列时日期升序，并返回费用。",
    ),
)
def test_structural_closure_rejects_unbound_inconsistent_or_hidden_slots(question):
    assert compile_structural_closure_sql(question, DOMAINS) is None
