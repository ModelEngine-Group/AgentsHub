"""Development-only regression fixtures for the bounded v8 compiler."""

import pytest

from dki.analysis.nl2sql_independent_v8 import compile_independent_v8_sql
from dki.analysis.nl2sql_protocol import validate_gold_sql


DOMAINS = {
    "disease": ["胃溃疡"],
    "department": ["消化内科"],
    "kg_relation": ["HAS_SYMPTOM"],
    "kg_relation_zh": ["所属科室"],
    "kg_entity_type": ["Symptom"],
}


CASES = [
    (
        "filter_nonempty_drug",
        "找出诊断为胃溃疡且处方药字段非空字符串的就诊ID、患者ID、处方药和检查项目,按患者ID及就诊ID升序;本查询其余精确非空常量为:0。",
        "SELECT visit_id, patient_id, drug_prescribed, exam FROM visits WHERE disease = '胃溃疡' AND LENGTH(drug_prescribed) > 0 ORDER BY patient_id ASC, visit_id ASC",
    ),
    (
        "filter_round_cost",
        "列出费用四舍五入到整数后不低于5700元的就诊ID、原始费用和科室,按原始费用降序、就诊ID升序。",
        "SELECT visit_id, cost, department FROM visits WHERE ROUND(cost, 0) >= 5700 ORDER BY cost DESC, visit_id ASC",
    ),
    (
        "group_gender_age",
        "仅考虑年龄不少于18岁的患者,分别按性别计算平均年龄(四舍五入到两位小数)和最小年龄,输出性别、平均年龄、最小年龄,并按性别升序;为消除边界与常量歧义,本查询采用的精确字面量还包括:2。",
        "SELECT gender, ROUND(AVG(age), 2) AS avg_age, MIN(age) AS min_age FROM patients WHERE age >= 18 GROUP BY gender ORDER BY gender ASC",
    ),
    (
        "group_head_tail_type",
        "仅统计置信度不低于0.80的三元组,按头实体类型与尾实体类型的组合计数;输出两种类型和数量,按数量降序,再按头类型、尾类型升序。",
        "SELECT head_type, tail_type, COUNT(*) AS triple_count FROM kg_triples WHERE confidence >= 0.8 GROUP BY head_type, tail_type ORDER BY triple_count DESC, head_type ASC, tail_type ASC",
    ),
    (
        "join_both_entities",
        "把置信度不低于0.85的三元组头、尾同时连接到实体表,输出头实体、头实体表类型、关系、尾实体、尾实体表类型;要求头尾都匹配,按头、关系、尾升序。",
        "SELECT t.head, h.type AS head_entity_type, t.relation, t.tail, e.type AS tail_entity_type FROM kg_triples AS t JOIN kg_entities AS h ON h.name = t.head JOIN kg_entities AS e ON e.name = t.tail WHERE t.confidence >= 0.85 ORDER BY t.head ASC, t.relation ASC, t.tail ASC",
    ),
    (
        "temporal_quarter",
        "列出2024年第二季度(4月1日含至7月1日不含)的就诊ID、日期、科室和疾病,按日期、就诊ID升序;为消除边界与常量歧义,本查询采用的精确字面量还包括:2024-04-01、2024-07-01。",
        "SELECT visit_id, visit_date, department, disease FROM visits WHERE visit_date >= '2024-04-01' AND visit_date < '2024-07-01' ORDER BY visit_date ASC, visit_id ASC",
    ),
    (
        "temporal_first_visit",
        "列出2024年内每名患者最早的就诊日期,输出患者ID和最早日期,按最早日期、患者ID升序;为消除边界与常量歧义,本查询采用的精确字面量还包括:2024-01-01、2025-01-01。",
        "SELECT patient_id, MIN(visit_date) AS first_date FROM visits WHERE visit_date >= '2024-01-01' AND visit_date < '2025-01-01' GROUP BY patient_id ORDER BY first_date ASC, patient_id ASC",
    ),
    (
        "temporal_weekday",
        "统计2024年每个星期几的就诊次数,SQLite星期编号0表示周日、6表示周六;输出星期编号和次数,按编号升序;为消除边界与常量歧义,本查询采用的精确字面量还包括:%w、2024-01-01、2025-01-01。",
        "SELECT STRFTIME('%w', visit_date) AS weekday_no, COUNT(*) AS visit_count FROM visits WHERE visit_date >= '2024-01-01' AND visit_date < '2025-01-01' GROUP BY weekday_no ORDER BY weekday_no ASC",
    ),
    (
        "temporal_month_patients",
        "找出2024年下半年(7月1日含至下一年1月1日不含)按月的不同患者数,输出月份和去重患者数,按月份升序;为消除边界与常量歧义,本查询采用的精确字面量还包括:2024-07-01、2025-01-01。",
        "SELECT SUBSTR(visit_date, 1, 7) AS month, COUNT(DISTINCT patient_id) AS patient_count FROM visits WHERE visit_date >= '2024-07-01' AND visit_date < '2025-01-01' GROUP BY month ORDER BY month ASC",
    ),
    (
        "kg_entity_mentions",
        "统计每种知识图谱实体类型中提及次数大于0的实体数,输出类型和实体数,按实体数降序、类型升序。",
        "SELECT type, COUNT(*) AS entity_count FROM kg_entities WHERE mention_count > 0 GROUP BY type ORDER BY entity_count DESC, type ASC",
    ),
    (
        "kg_relation_zh",
        "查询关系中文名为所属科室且观察次数至少为1的三元组,输出头实体、头类型、尾实体、尾类型和观察次数,按观察次数降序、头尾升序。",
        "SELECT head, head_type, tail, tail_type, count FROM kg_triples WHERE relation_zh = '所属科室' AND count >= 1 ORDER BY count DESC, head ASC, tail ASC",
    ),
    (
        "kg_head_is_tail",
        "列出既作为置信度不低于0.80的三元组头实体、又作为另一条三元组尾实体的实体名,结果去重并按实体名升序。",
        "SELECT DISTINCT a.head AS entity_name FROM kg_triples AS a JOIN kg_triples AS b ON b.tail = a.head WHERE a.confidence >= 0.8 ORDER BY entity_name ASC",
    ),
    (
        "kg_same_type_average",
        "找出提及次数不少于1且达到同类型实体平均值或以上的实体,输出名称、类型、提及次数,按类型升序、提及次数降序、名称升序。",
        "SELECT e.name, e.type, e.mention_count FROM kg_entities AS e WHERE e.mention_count >= 1 AND e.mention_count >= (SELECT AVG(x.mention_count) FROM kg_entities AS x WHERE x.type = e.type) ORDER BY e.type ASC, e.mention_count DESC, e.name ASC",
    ),
    (
        "having_disease_cost",
        "按疾病计算平均费用,只保留平均费用高于5000元的疾病;输出疾病、保留两位小数的平均费用和就诊数,按平均费用降序、疾病升序;为消除边界与常量歧义,本查询采用的精确字面量还包括:2。",
        "SELECT disease, ROUND(AVG(cost), 2) AS avg_cost, COUNT(*) AS visit_count FROM visits GROUP BY disease HAVING AVG(cost) > 5000 ORDER BY avg_cost DESC, disease ASC",
    ),
    (
        "having_relation_counts",
        "按知识图谱关系统计三元组数和观察次数总和,只保留三元组数至少为10且观察次数总和大于20的关系;按三元组数降序、关系升序。",
        "SELECT relation, COUNT(*) AS triple_count, SUM(count) AS observation_total FROM kg_triples GROUP BY relation HAVING COUNT(*) >= 10 AND SUM(count) > 20 ORDER BY triple_count DESC, relation ASC",
    ),
    (
        "nested_exists_cost",
        "列出至少有一次就诊费用高于7000元的患者ID、姓名和地区,按患者ID升序;为消除边界与常量歧义,本查询采用的精确字面量还包括:1。",
        "SELECT p.patient_id, p.name, p.region FROM patients AS p WHERE EXISTS (SELECT 1 FROM visits AS v WHERE v.patient_id = p.patient_id AND v.cost > 7000) ORDER BY p.patient_id ASC",
    ),
    (
        "nested_region_average",
        "找出年龄高于其所在地区患者平均年龄的1.0倍的患者,输出姓名、地区、年龄,按地区升序、年龄降序、姓名升序。",
        "SELECT p.name, p.region, p.age FROM patients AS p WHERE p.age > 1 * (SELECT AVG(q.age) FROM patients AS q WHERE q.region = p.region) ORDER BY p.region ASC, p.age DESC, p.name ASC",
    ),
    (
        "nested_department_average",
        "找出费用高于同科室平均费用的1.0倍的就诊,输出就诊ID、科室和费用,按科室升序、费用降序、就诊ID升序。",
        "SELECT v.visit_id, v.department, v.cost FROM visits AS v WHERE v.cost > 1 * (SELECT AVG(x.cost) FROM visits AS x WHERE x.department = v.department) ORDER BY v.department ASC, v.cost DESC, v.visit_id ASC",
    ),
    (
        "nested_relation_average",
        "找出置信度高于同一关系平均置信度再加0.00的三元组,输出头、关系、尾和置信度,按关系、置信度降序、头尾升序。",
        "SELECT t.head, t.relation, t.tail, t.confidence FROM kg_triples AS t JOIN (SELECT relation, AVG(confidence) AS avg_confidence FROM kg_triples GROUP BY relation) AS a ON a.relation = t.relation WHERE t.confidence > a.avg_confidence + 0 ORDER BY t.relation ASC, t.confidence DESC, t.head ASC, t.tail ASC",
    ),
    (
        "nested_visit_count_average",
        "找出就诊次数高于所有有就诊患者平均就诊次数再加0次的患者ID及其就诊次数,按次数降序、患者ID升序。",
        "SELECT v.patient_id, COUNT(*) AS visit_count FROM visits AS v GROUP BY v.patient_id HAVING COUNT(*) > (SELECT AVG(cnt) FROM (SELECT COUNT(*) AS cnt FROM visits GROUP BY patient_id)) + 0 ORDER BY visit_count DESC, v.patient_id ASC",
    ),
    (
        "nested_department_membership",
        "列出曾在消化内科就诊且年龄高于全体患者平均年龄的患者ID、姓名和年龄,按年龄降序、患者ID升序。",
        "SELECT p.patient_id, p.name, p.age FROM patients AS p WHERE p.age > (SELECT AVG(age) FROM patients) AND p.patient_id IN (SELECT patient_id FROM visits WHERE department = '消化内科') ORDER BY p.age DESC, p.patient_id ASC",
    ),
    (
        "cte_expensive",
        "先用CTE筛选费用至少为4000元的就诊,再按科室统计次数和平均费用(两位小数);输出科室、次数、平均费用,按次数降序、科室升序;为消除边界与常量歧义,本查询采用的精确字面量还包括:2。",
        "WITH expensive AS (SELECT department, cost FROM visits WHERE cost >= 4000) SELECT department, COUNT(*) AS visit_count, ROUND(AVG(cost), 2) AS avg_cost FROM expensive GROUP BY department ORDER BY visit_count DESC, department ASC",
    ),
    (
        "cte_head_stats",
        "用CTE为每个知识图谱头实体汇总三元组数和观察次数,再输出三元组数至少为4的头实体及两项指标,按三元组数降序、头实体升序。",
        "WITH head_stats AS (SELECT head, COUNT(*) AS triple_count, SUM(count) AS observation_total FROM kg_triples GROUP BY head) SELECT head, triple_count, observation_total FROM head_stats WHERE triple_count >= 4 ORDER BY triple_count DESC, head ASC",
    ),
    (
        "cte_type_average",
        "用CTE在提及次数不少于1的实体中计算各类型平均提及次数,再输出同样满足下限且达到本类型平均值的实体名称、类型、次数,按类型、次数降序、名称升序。",
        "WITH type_avg AS (SELECT type, AVG(mention_count) AS avg_mentions FROM kg_entities WHERE mention_count >= 1 GROUP BY type) SELECT e.name, e.type, e.mention_count FROM kg_entities AS e JOIN type_avg AS a ON a.type = e.type WHERE e.mention_count >= 1 AND e.mention_count >= a.avg_mentions ORDER BY e.type ASC, e.mention_count DESC, e.name ASC",
    ),
    (
        "cte_relation_counts",
        "用CTE挑出同时有至少2种不同关系的头实体,再返回这些头实体的名称及不同关系数,按关系数降序、名称升序。",
        "WITH rel_counts AS (SELECT head, COUNT(DISTINCT relation) AS relation_count FROM kg_triples GROUP BY head) SELECT head, relation_count FROM rel_counts WHERE relation_count >= 2 ORDER BY relation_count DESC, head ASC",
    ),
    (
        "set_head_except_tail",
        "返回作为知识图谱头实体出现但从未作为尾实体出现的实体名;使用差集去重并按名称升序。",
        "SELECT head AS entity_name FROM kg_triples EXCEPT SELECT tail AS entity_name FROM kg_triples ORDER BY entity_name ASC",
    ),
    (
        "set_relation_union_type",
        "把关系为HAS_SYMPTOM的尾实体与类型为Symptom的实体名称合并去重,再排除空字符串;输出名称并按升序排列;本查询其余精确非空常量为:0。",
        "SELECT tail AS name FROM kg_triples WHERE relation = 'HAS_SYMPTOM' AND LENGTH(tail) > 0 UNION SELECT name FROM kg_entities WHERE type = 'Symptom' AND LENGTH(name) > 0 ORDER BY name ASC",
    ),
    (
        "window_lag",
        "对每名患者按日期和就诊ID排序,显示上一次就诊日期;输出患者ID、就诊ID、当前日期和前次日期,按患者ID、当前日期、就诊ID升序。",
        "SELECT patient_id, visit_id, visit_date, LAG(visit_date) OVER (PARTITION BY patient_id ORDER BY visit_date ASC, visit_id ASC) AS previous_date FROM visits ORDER BY patient_id ASC, visit_date ASC, visit_id ASC",
    ),
    (
        "window_lead",
        "对每名患者按日期和就诊ID排序,显示下一次就诊的疾病;输出患者ID、就诊ID、当前疾病和下一疾病,按患者ID、日期、就诊ID升序。",
        "SELECT patient_id, visit_id, disease, LEAD(disease) OVER (PARTITION BY patient_id ORDER BY visit_date ASC, visit_id ASC) AS next_disease FROM visits ORDER BY patient_id ASC, visit_date ASC, visit_id ASC",
    ),
    (
        "window_dense_rank",
        "在每种知识图谱关系内按观察次数降序计算稠密名次,只保留前2名;输出关系、头、尾、观察次数和名次,按关系、名次、头、尾升序。",
        "SELECT relation, head, tail, count, obs_rank FROM (SELECT relation, head, tail, count, DENSE_RANK() OVER (PARTITION BY relation ORDER BY count DESC) AS obs_rank FROM kg_triples) WHERE obs_rank <= 2 ORDER BY relation ASC, obs_rank ASC, head ASC, tail ASC",
    ),
    (
        "window_department_percent",
        "计算每次就诊费用占本科室总费用的百分比,保留两位小数;输出就诊ID、科室、费用和百分比,按科室、百分比降序、就诊ID升序;为消除边界与常量歧义,本查询采用的精确字面量还包括:100.0、2。",
        "SELECT visit_id, department, cost, ROUND(100.0 * cost / SUM(cost) OVER (PARTITION BY department), 2) AS dept_pct FROM visits ORDER BY department ASC, dept_pct DESC, visit_id ASC",
    ),
    (
        "window_moving_average",
        "对每个科室按日期和就诊ID排序,计算当前行及前两行的三次移动平均费用并保留两位小数;输出科室、就诊ID、日期、费用和移动平均,按科室、日期、就诊ID升序;为消除边界与常量歧义,本查询采用的精确字面量还包括:2。",
        "SELECT department, visit_id, visit_date, cost, ROUND(AVG(cost) OVER (PARTITION BY department ORDER BY visit_date ASC, visit_id ASC ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS moving_avg_3 FROM visits ORDER BY department ASC, visit_date ASC, visit_id ASC",
    ),
]


@pytest.mark.parametrize(("case_id", "question", "expected"), CASES, ids=[case[0] for case in CASES])
def test_v8_development_operation_families(case_id, question, expected):
    del case_id
    actual = compile_independent_v8_sql(question, DOMAINS)
    assert actual == expected
    validate_gold_sql(actual)


@pytest.mark.parametrize(
    "question",
    [
        "删除所有患者数据",
        "列出费用较高的记录",
        "把关系为UNKNOWN_REL的尾实体与类型为Symptom的名称合并",
        "找出诊断为不存在疾病且处方药字段非空字符串的就诊",
    ],
)
def test_v8_compiler_abstains_on_unsafe_or_ungrounded_requests(question):
    assert compile_independent_v8_sql(question, DOMAINS) is None
