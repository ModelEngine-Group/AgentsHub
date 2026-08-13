import sqlite3

import pytest

from dki.analysis.nl2sql import NL2SQL


@pytest.fixture()
def projection_db(tmp_path):
    path = tmp_path / "projection.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE patients (
            patient_id TEXT PRIMARY KEY,
            name TEXT,
            gender TEXT,
            age INTEGER,
            region TEXT
        );
        CREATE TABLE visits (
            visit_id TEXT PRIMARY KEY,
            patient_id TEXT,
            visit_date TEXT,
            department TEXT,
            disease TEXT,
            drug_prescribed TEXT,
            exam TEXT,
            cost REAL,
            los_days INTEGER
        );
        CREATE TABLE kg_entities (
            name TEXT,
            type TEXT,
            mention_count INTEGER
        );
        CREATE TABLE kg_triples (
            head TEXT,
            head_type TEXT,
            relation TEXT,
            relation_zh TEXT,
            tail TEXT,
            tail_type TEXT,
            confidence REAL,
            count INTEGER
        );
        """
    )
    conn.executemany(
        "INSERT INTO patients VALUES (?,?,?,?,?)",
        [
            ("P1", "甲", "女", 55, "上海"),
            ("P2", "乙", "男", 44, "北京"),
        ],
    )
    conn.executemany(
        "INSERT INTO visits VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (
                "V1",
                "P1",
                "2024-08-01",
                "内分泌科",
                "糖尿病肾病",
                "药物甲",
                "尿培养",
                5000.0,
                3,
            ),
            (
                "V2",
                "P2",
                "2024-09-01",
                "神经内科",
                "慢性胃炎",
                "药物乙",
                "头颅MRI",
                3000.0,
                5,
            ),
            (
                "V3",
                "P1",
                "2025-01-01",
                "神经内科",
                "腰椎间盘突出症",
                "",
                "肝功能检查",
                4000.0,
                3,
            ),
            (
                "V4",
                "P2",
                "2025-02-01",
                "内分泌科",
                "骨质疏松症",
                "",
                "骨密度检测",
                2000.0,
                2,
            ),
        ],
    )
    conn.executemany(
        "INSERT INTO kg_entities VALUES (?,?,?)",
        [
            ("疾病甲", "Disease", 20),
            ("症状甲", "Symptom", 10),
            ("检查甲", "Exam", 8),
        ],
    )
    conn.executemany(
        "INSERT INTO kg_triples VALUES (?,?,?,?,?,?,?,?)",
        [(
            "疾病甲",
            "Disease",
            "HAS_SYMPTOM",
            "临床表现",
            "症状甲",
            "Symptom",
            0.8,
            3,
        ), (
            "疾病甲",
            "Disease",
            "DIAGNOSED_BY",
            "诊断检查",
            "检查甲",
            "Exam",
            0.9,
            2,
        )],
    )
    conn.commit()
    conn.close()
    return path


CASES = [
    (
        "filtering",
        "查询科室为神经内科且费用至少5000元的就诊，"
        "返回就诊编号、患者编号、诊断和费用。",
        "SELECT visit_id, patient_id, disease, cost FROM visits "
        "WHERE department='神经内科' AND cost>=5000 "
        "ORDER BY cost DESC, visit_id",
    ),
    (
        "aggregation_grouping",
        "统计年龄至少45岁且性别为女的患者，"
        "按地区返回不同患者数和就诊总费用。",
        "SELECT p.region, COUNT(DISTINCT p.patient_id) AS patient_count, "
        "SUM(v.cost) AS total_cost FROM patients p "
        "JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE p.age>=45 AND p.gender='女' GROUP BY p.region "
        "ORDER BY patient_count DESC, p.region",
    ),
    (
        "top_k",
        "在2024-07-20至2024-10-16之间，"
        "返回就诊次数最多的前2个科室及其就诊次数。",
        "SELECT department, COUNT(*) AS visit_count FROM visits "
        "WHERE visit_date BETWEEN '2024-07-20' AND '2024-10-16' "
        "GROUP BY department ORDER BY visit_count DESC, department LIMIT 2",
    ),
    (
        "multi_table_join",
        "查询年龄至少50岁且检查为尿培养的患者姓名、年龄、诊断和就诊日期。",
        "SELECT p.name, p.age, v.disease, v.visit_date FROM patients p "
        "JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE p.age>=50 AND v.exam='尿培养' "
        "ORDER BY v.visit_date, p.patient_id",
    ),
    (
        "temporal",
        "截至2025-03-05，返回每位患者最近一次费用至少1000元的"
        "就诊日期、科室和费用。",
        "SELECT v.patient_id, v.visit_date, v.department, v.cost "
        "FROM visits v WHERE v.visit_date<='2025-03-05' AND v.cost>=1000 "
        "AND v.visit_date=(SELECT MAX(v2.visit_date) AS max_visit_date FROM visits v2 "
        "WHERE v2.patient_id=v.patient_id AND v2.visit_date<='2025-03-05' "
        "AND v2.cost>=1000) ORDER BY v.patient_id, v.visit_id",
    ),
    (
        "knowledge_graph",
        "返回头实体类型为Disease、中文关系为临床表现且置信度至少0.5的"
        "头实体、尾实体和观察次数。",
        "SELECT head, tail, count FROM kg_triples "
        "WHERE relation_zh = '临床表现' AND head_type = 'Disease' "
        "AND confidence >= 0.5 "
        "ORDER BY count DESC, confidence DESC, head ASC, tail ASC",
    ),
    (
        "having",
        "对年龄至少50岁的患者按地区分组，"
        "找出至少有3位不同患者的地区。",
        "SELECT p.region FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE p.age>=50 GROUP BY p.region "
        "HAVING COUNT(DISTINCT p.patient_id)>=3 "
        "ORDER BY COUNT(DISTINCT p.patient_id) DESC, p.region",
    ),
    (
        "nested_correlated",
        "返回每个科室中住院天数至少0天、费用至少3000元且费用等于"
        "该科室在相同住院天数条件下最高费用的就诊编号、科室和费用。",
        "SELECT v.visit_id, v.department, v.cost FROM visits v "
        "WHERE v.los_days>=0 AND v.cost>=3000 "
        "AND v.cost=(SELECT MAX(v2.cost) AS max_cost FROM visits v2 "
        "WHERE v2.department=v.department AND v2.los_days>=0) "
        "ORDER BY v.department, v.visit_id",
    ),
    (
        "cte",
        "先汇总科室内分泌科中每位患者的累计费用，"
        "再返回累计费用至少3000元的前2位患者。",
        "WITH patient_cost AS (SELECT patient_id, SUM(cost) AS total_cost "
        "FROM visits WHERE department='内分泌科' GROUP BY patient_id) "
        "SELECT patient_id, total_cost FROM patient_cost "
        "WHERE total_cost>=3000 ORDER BY total_cost DESC, patient_id LIMIT 2",
    ),
    (
        "set_operation",
        "返回既在科室神经内科就诊过、又在科室内分泌科就诊过的患者编号。",
        "SELECT p.patient_id FROM patients p WHERE EXISTS ("
        "SELECT v1.patient_id AS patient_id FROM visits v1 "
        "WHERE v1.patient_id=p.patient_id AND v1.department='神经内科'"
        ") AND EXISTS (SELECT v2.patient_id AS patient_id FROM visits v2 "
        "WHERE v2.patient_id=p.patient_id AND v2.department='内分泌科'"
        ") ORDER BY p.patient_id",
    ),
    (
        "window",
        "对科室内分泌科中费用至少1000元的就诊按费用降序编号，"
        "返回前2名及其名次。",
        "SELECT visit_id, patient_id, cost, ROW_NUMBER() OVER "
        "(PARTITION BY department ORDER BY cost DESC, visit_id) AS visit_rank "
        "FROM visits WHERE department='内分泌科' AND cost>=1000 "
        "ORDER BY visit_rank LIMIT 2",
    ),
    (
        "filtering_canonical_literal",
        "查询诊断为腰椎间盘突出症且住院天数至少3天的就诊，"
        "返回就诊编号、科室和住院天数。",
        "SELECT visit_id, department, los_days FROM visits "
        "WHERE disease='腰椎间盘突出症' AND los_days>=3 "
        "ORDER BY los_days DESC, visit_id",
    ),
    (
        "join_canonical_disease_literal",
        "对诊断为骨质疏松症且费用至少1000元的就诊，"
        "按患者性别返回不同患者数和平均费用。",
        "SELECT p.gender, COUNT(DISTINCT p.patient_id) AS patient_count, "
        "AVG(v.cost) AS avg_cost FROM patients p "
        "JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE v.disease='骨质疏松症' AND v.cost>=1000 "
        "GROUP BY p.gender ORDER BY p.gender",
    ),
    (
        "join_canonical_exam_literal",
        "查询年龄至少30岁且检查为肝功能检查的患者姓名、年龄、诊断和就诊日期。",
        "SELECT p.name, p.age, v.disease, v.visit_date FROM patients p "
        "JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE p.age>=30 AND v.exam='肝功能检查' "
        "ORDER BY v.visit_date, p.patient_id",
    ),
    (
        "temporal_canonical_disease_literal",
        "返回2025-01-19及以后每位患者诊断为糖尿病肾病的首次就诊日期。",
        "SELECT patient_id, MIN(visit_date) AS first_visit_date FROM visits "
        "WHERE visit_date>='2025-01-19' AND disease='糖尿病肾病' "
        "GROUP BY patient_id ORDER BY patient_id",
    ),
]


@pytest.mark.parametrize("_category,question,expected", CASES)
def test_public_projection_development_regression(
    projection_db,
    _category,
    question,
    expected,
):
    assert NL2SQL(db_path=projection_db).rule_sql(question) == expected


def test_closed_kg_entity_type_and_inclusive_range_compiler(projection_db):
    nl = NL2SQL(db_path=projection_db)

    assert nl.rule_sql(
        "返回类型为Disease、提及次数在2至8之间的实体名称和提及次数。"
    ) == (
        "SELECT name, mention_count FROM kg_entities "
        "WHERE type='Disease' AND mention_count BETWEEN 2 AND 8 "
        "ORDER BY mention_count DESC, name"
    )


@pytest.mark.parametrize(
    "question",
    [
        "返回类型为Unknown、提及次数在2至8之间的实体名称和提及次数。",
        "返回类型为Disease、提及次数在8至2之间的实体名称和提及次数。",
        (
            "筛选年龄至少40岁的患者后，"
            "返回类型为Disease、提及次数在2至8之间的实体名称和提及次数。"
        ),
    ],
)
def test_closed_kg_entity_range_rejects_unknown_reversed_or_hidden_slots(
    projection_db,
    question,
):
    nl = NL2SQL(db_path=projection_db)
    assert nl._compile_kg_entity_range_sql(nl._normalize(question)) is None


@pytest.mark.parametrize(
    "question,expected",
    [
        (
            "围绕质量指标复核，在制作内部报表时，"
            "将提及次数按小于4、4到9、至少10分档，统计实体数；端点为4、9、10。",
            "SELECT CASE WHEN mention_count < 4 THEN '少于4' "
            "WHEN mention_count BETWEEN 4 AND 9 THEN '4到9' "
            "ELSE '至少10' END AS mention_band, COUNT(*) AS entity_count "
            "FROM kg_entities GROUP BY mention_band ORDER BY mention_band",
        ),
        (
            "围绕边质量检查，在生成审计视图时，"
            "先汇总每种关系的平均置信度，再连接原边并返回"
            "置信度高于本关系平均值的头实体、关系编码和尾实体。",
            "WITH relation_avg AS (SELECT relation, "
            "AVG(confidence) AS avg_confidence FROM kg_triples "
            "GROUP BY relation) SELECT t.head, t.relation, t.tail, "
            "t.confidence, ROUND(r.avg_confidence, 4) AS relation_avg "
            "FROM kg_triples t JOIN relation_avg r ON r.relation = t.relation "
            "WHERE t.confidence > r.avg_confidence "
            "ORDER BY t.relation, t.confidence DESC, t.head, t.tail",
        ),
        (
            "围绕图谱归属梳理，在输出汇总清单时，"
            "连接类型为Department的实体及其BELONGS_TO_DEPT入边，"
            "按科室统计不同疾病数和观察次数总和。",
            "SELECT e.name AS department, "
            "COUNT(DISTINCT t.head) AS disease_count, "
            "SUM(t.count) AS observations FROM kg_entities e "
            "JOIN kg_triples t ON t.tail = e.name "
            "WHERE e.type = 'Department' AND t.relation = 'BELONGS_TO_DEPT' "
            "GROUP BY e.name ORDER BY disease_count DESC, department",
        ),
        (
            "围绕科室成本测算，在形成成本清单时，"
            "在记录数至少2次的科室中按平均费用取前3名，阈值2、数量3。",
            "SELECT department, ROUND(AVG(cost), 2) AS avg_cost, "
            "COUNT(*) AS visit_count FROM visits GROUP BY department "
            "HAVING COUNT(*) >= 2 ORDER BY avg_cost DESC, department LIMIT 3",
        ),
    ],
)
def test_output_complete_structural_compilers(
    projection_db,
    question,
    expected,
):
    nl = NL2SQL(db_path=projection_db)
    nl._domains.setdefault("kg_entity_type", []).append("Department")
    nl._domains.setdefault("kg_relation", []).append("BELONGS_TO_DEPT")
    assert nl.rule_sql(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        (
            "将提及次数按小于4、5到9、至少10分档，"
            "统计实体数；端点为4、9、10。"
        ),
        (
            "在记录数至少2次的科室中按平均费用取前3名，"
            "阈值4、数量3。"
        ),
        (
            "围绕50岁患者复核，在制作内部报表时，"
            "在记录数至少2次的科室中按平均费用取前3名，阈值2、数量3。"
        ),
    ],
)
def test_output_complete_structural_compilers_reject_inconsistent_or_hidden_slots(
    projection_db,
    question,
):
    nl = NL2SQL(db_path=projection_db)
    assert nl._compile_output_complete_structural_sql(
        nl._normalize(question)
    ) is None


@pytest.mark.parametrize(
    "question,expected",
    [
        (
            "围绕图谱质量抽查，在生成核验清单时，"
            "返回头实体为疾病甲且关系中文名属于临床表现或诊断检查的"
            "尾实体、尾类型和置信度。",
            "SELECT tail, tail_type, confidence FROM kg_triples "
            "WHERE head = '疾病甲' AND relation_zh IN ('临床表现', '诊断检查') "
            "ORDER BY tail_type, tail",
        ),
        (
            "在HAS_SYMPTOM关系中，返回置信度等于同一头实体最高置信度的边，"
            "并列保留。",
            "SELECT t.head, t.tail, t.confidence FROM kg_triples t "
            "WHERE t.relation = 'HAS_SYMPTOM' "
            "AND t.confidence = (SELECT MAX(x.confidence) FROM kg_triples x "
            "WHERE x.head = t.head AND x.relation = 'HAS_SYMPTOM') "
            "ORDER BY t.head, t.tail",
        ),
        (
            "合并HAS_SYMPTOM的头实体与DIAGNOSED_BY的尾实体并保留来源标签"
            "症状疾病或诊断检查。",
            "SELECT head AS entity_name, '症状疾病' AS source_name "
            "FROM kg_triples WHERE relation = 'HAS_SYMPTOM' UNION ALL "
            "SELECT tail AS entity_name, '诊断检查' AS source_name "
            "FROM kg_triples WHERE relation = 'DIAGNOSED_BY' "
            "ORDER BY source_name, entity_name",
        ),
        (
            "围绕月份指标解释，在生成时间清单时，"
            "筛选所有年份中月份字符串为09的就诊，返回日期、患者、疾病和费用；"
            "阈值09。",
            "SELECT visit_date, patient_id, disease, cost FROM visits "
            "WHERE strftime('%m', visit_date) = '09' "
            "ORDER BY visit_date, visit_id",
        ),
        (
            "筛出实体类型为Disease、提及次数大于10的名称和次数，按次数降序。",
            "SELECT name, mention_count FROM kg_entities "
            "WHERE type = 'Disease' AND mention_count > 10 "
            "ORDER BY mention_count DESC, name",
        ),
        (
            "按科室汇总，仅列出总费用至少5000元且不同疾病不少于2种的科室。",
            "SELECT department, ROUND(SUM(cost), 2) AS total_cost, "
            "COUNT(DISTINCT disease) AS disease_count FROM visits "
            "GROUP BY department HAVING SUM(cost) >= 5000 "
            "AND COUNT(DISTINCT disease) >= 2 "
            "ORDER BY total_cost DESC, department",
        ),
        (
            "列出尾类型属于Symptom或Exam且置信度低于0.95的关系。",
            "SELECT head, relation, tail, tail_type, confidence FROM kg_triples "
            "WHERE tail_type IN ('Symptom', 'Exam') AND confidence < 0.95 "
            "ORDER BY confidence ASC, head, relation, tail",
        ),
        (
            "找出作为HAS_SYMPTOM尾实体被至少1个不同疾病连接的Symptom实体。",
            "SELECT e.name, e.mention_count FROM kg_entities e "
            "WHERE e.type = 'Symptom' AND (SELECT COUNT(DISTINCT t.head) "
            "FROM kg_triples t WHERE t.tail = e.name "
            "AND t.relation = 'HAS_SYMPTOM') >= 1 "
            "ORDER BY e.mention_count DESC, e.name",
        ),
        (
            "使用CTE计算各疾病平均住院天数，"
            "再返回平均住院天数高于全库平均值的疾病。",
            "WITH disease_los AS (SELECT disease, AVG(los_days) AS avg_los "
            "FROM visits GROUP BY disease), overall AS "
            "(SELECT AVG(los_days) AS overall_los FROM visits) "
            "SELECT d.disease, ROUND(d.avg_los, 2) AS avg_los "
            "FROM disease_los d CROSS JOIN overall o "
            "WHERE d.avg_los > o.overall_los "
            "ORDER BY d.avg_los DESC, d.disease",
        ),
    ],
)
def test_closed_compositional_compilers(projection_db, question, expected):
    assert NL2SQL(db_path=projection_db).rule_sql(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        (
            "返回头实体为未知疾病且关系中文名属于临床表现或诊断检查的"
            "尾实体、尾类型和置信度。"
        ),
        (
            "在UNKNOWN_RELATION关系中，返回置信度等于同一头实体最高置信度"
            "的边，并列保留。"
        ),
        (
            "合并HAS_SYMPTOM的头实体与DIAGNOSED_BY的尾实体并保留来源标签"
            "相同或相同。"
        ),
        (
            "筛选所有年份中月份字符串为13的就诊，"
            "返回日期、患者、疾病和费用。"
        ),
        (
            "围绕50岁患者抽查，在生成时间清单时，"
            "筛选所有年份中月份字符串为09的就诊，"
            "返回日期、患者、疾病和费用。"
        ),
        "筛出实体类型为Unknown、提及次数大于10的名称和次数，按次数降序。",
        "列出尾类型属于Symptom或Unknown且置信度低于0.95的关系。",
        "列出尾类型属于Symptom或Exam且置信度低于1.5的关系。",
        (
            "找出作为UNKNOWN_RELATION尾实体被至少1个不同疾病连接的"
            "Symptom实体。"
        ),
        (
            "围绕60岁患者检查，在制作清单时，按科室汇总，"
            "仅列出总费用至少5000元且不同疾病不少于2种的科室。"
        ),
    ],
)
def test_closed_compositional_compilers_reject_ungrounded_or_hidden_slots(
    projection_db,
    question,
):
    nl = NL2SQL(db_path=projection_db)
    assert nl._compile_closed_compositional_sql(nl._normalize(question)) is None


@pytest.mark.parametrize(
    "question",
    [
        "返回类型为Disease、提及次数在2至8之间的实体名称和提及次数。",
        (
            "围绕质量指标复核，在制作内部报表时，"
            "将提及次数按小于4、4到9、至少10分档，"
            "统计实体数；端点为4、9、10。"
        ),
        (
            "围绕边质量检查，在生成审计视图时，"
            "先汇总每种关系的平均置信度，再连接原边并返回"
            "置信度高于本关系平均值的头实体、关系编码和尾实体。"
        ),
        (
            "围绕图谱归属梳理，在输出汇总清单时，"
            "连接类型为Department的实体及其BELONGS_TO_DEPT入边，"
            "按科室统计不同疾病数和观察次数总和。"
        ),
        (
            "围绕科室成本测算，在形成成本清单时，"
            "在记录数至少2次的科室中按平均费用取前3名，"
            "阈值2、数量3。"
        ),
        (
            "围绕图谱质量抽查，在生成核验清单时，"
            "返回头实体为疾病甲且关系中文名属于临床表现或诊断检查的"
            "尾实体、尾类型和置信度。"
        ),
        (
            "在HAS_SYMPTOM关系中，返回置信度等于同一头实体最高置信度的边，"
            "并列保留。"
        ),
        (
            "合并HAS_SYMPTOM的头实体与DIAGNOSED_BY的尾实体并保留来源标签"
            "症状疾病或诊断检查。"
        ),
        (
            "围绕月份指标解释，在生成时间清单时，"
            "筛选所有年份中月份字符串为09的就诊，"
            "返回日期、患者、疾病和费用；阈值09。"
        ),
        "筛出实体类型为Disease、提及次数大于10的名称和次数，按次数降序。",
        "按科室汇总，仅列出总费用至少5000元且不同疾病不少于2种的科室。",
        "列出尾类型属于Symptom或Exam且置信度低于0.95的关系。",
        "找出作为HAS_SYMPTOM尾实体被至少1个不同疾病连接的Symptom实体。",
        (
            "使用CTE计算各疾病平均住院天数，"
            "再返回平均住院天数高于全库平均值的疾病。"
        ),
    ],
)
def test_model_first_uses_only_closed_authoritative_rules_before_llm(
    projection_db,
    question,
    monkeypatch,
):
    nl = NL2SQL(db_path=projection_db, agent_rule_policy="model_first")
    nl._domains.setdefault("kg_entity_type", []).append("Department")
    nl._domains.setdefault("kg_relation", []).append("BELONGS_TO_DEPT")
    monkeypatch.setattr(
        nl,
        "_llm_pipeline",
        lambda _question: pytest.fail("closed compiler must run before the model"),
    )

    result = nl.to_sql(question)

    assert result["engine"] == "rule"
    assert result["plan"]["route"] == "bounded_rule_before_model"
