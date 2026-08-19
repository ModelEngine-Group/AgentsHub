"""Public-development regressions for natural compositional Chinese questions.

The language frames are independently re-authored from public CSpider
``hospital_1`` training examples and projected onto this project's sealed
four-table medical schema.  They are development-selection fixtures, not
prospective blind evidence.
"""

import sqlite3

import pytest

from dki.analysis.nl2sql import NL2SQL
from dki.analysis.nl2sql_compositional import (
    _compile_anti_join,
    _compile_distinct_list,
    _compile_patient_cost_top,
    _compile_patient_having,
    _compile_per_patient_temporal,
    _compile_set_operation,
    _compile_top_rows,
    compile_compositional_medical_sql,
)
from dki.analysis.nl2sql_protocol import (
    execute_complete_readonly,
    validate_gold_sql,
)


@pytest.fixture()
def cspider_compositional_db(tmp_path):
    path = tmp_path / "cspider_compositional.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
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
    connection.executemany(
        "INSERT INTO patients VALUES (?,?,?,?,?)",
        [
            ("P1", "张伟", "男", 48, "北京"),
            ("P2", "李敏", "女", 35, "上海"),
            ("P3", "王芳", "女", 62, "广州"),
            ("P4", "赵强", "男", 28, "北京"),
            ("P5", "孙静", "女", 50, "成都"),
        ],
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (
                "V1",
                "P1",
                "2025-01-05",
                "心血管内科",
                "冠心病",
                "阿司匹林",
                "心电图",
                900.0,
                5,
            ),
            (
                "V2",
                "P1",
                "2025-02-10",
                "心血管内科",
                "冠心病",
                "阿司匹林",
                "心电图",
                1200.0,
                3,
            ),
            (
                "V3",
                "P2",
                "2025-01-20",
                "内分泌科",
                "糖尿病",
                "二甲双胍",
                "血糖检查",
                1800.0,
                4,
            ),
            (
                "V4",
                "P2",
                "2025-03-01",
                "神经内科",
                "偏头痛",
                "布洛芬",
                "头颅MRI",
                600.0,
                0,
            ),
            (
                "V5",
                "P3",
                "2025-01-15",
                "神经内科",
                "脑梗死",
                "阿司匹林",
                "头颅MRI",
                3000.0,
                10,
            ),
            (
                "V6",
                "P3",
                "2025-02-20",
                "康复科",
                "脑梗死",
                "",
                "康复评估",
                1500.0,
                7,
            ),
            (
                "V7",
                "P4",
                "2025-02-25",
                "内分泌科",
                "糖尿病",
                "二甲双胍",
                "血糖检查",
                700.0,
                1,
            ),
            (
                "V8",
                "P4",
                "2025-03-05",
                "心血管内科",
                "高血压",
                "缬沙坦",
                "心电图",
                2500.0,
                2,
            ),
        ],
    )
    connection.commit()
    connection.close()
    return path


CASES = [
    (
        "group_argmax",
        "哪个科室接诊的次数最多？",
        "SELECT department, COUNT(*) AS visit_count FROM visits "
        "GROUP BY department ORDER BY visit_count DESC, department ASC LIMIT 1",
    ),
    (
        "group_argmin",
        "找出就诊数量最少的科室。",
        "SELECT department, COUNT(*) AS visit_count FROM visits "
        "GROUP BY department ORDER BY visit_count ASC, department ASC LIMIT 1",
    ),
    (
        "joined_distinct_list",
        "列出曾经来院就诊的患者姓名。",
        "SELECT DISTINCT p.name FROM patients AS p JOIN visits AS v "
        "ON v.patient_id = p.patient_id ORDER BY p.name ASC",
    ),
    (
        "group_having_join",
        "哪些患者有超过一次就诊？请给出姓名和地区。",
        "SELECT p.name, p.region FROM patients AS p WHERE "
        "(SELECT COUNT(*) AS visit_count FROM visits AS v "
        "WHERE v.patient_id = p.patient_id) > 1 ORDER BY p.name ASC",
    ),
    (
        "global_latest_join",
        "最近一次就诊的患者叫什么名字？",
        "SELECT p.name FROM patients AS p JOIN visits AS v "
        "ON v.patient_id = p.patient_id "
        "ORDER BY v.visit_date DESC, v.visit_id DESC LIMIT 1",
    ),
    (
        "anti_join",
        "列出从来没有任何就诊记录的患者姓名。",
        "SELECT p.name FROM patients AS p WHERE NOT EXISTS "
        "(SELECT v.patient_id AS patient_id FROM visits AS v "
        "WHERE v.patient_id = p.patient_id) "
        "ORDER BY p.name ASC",
    ),
    (
        "group_count",
        "每个科室分别有多少次就诊？",
        "SELECT department, COUNT(*) AS visit_count FROM visits "
        "GROUP BY department ORDER BY department ASC",
    ),
    (
        "count_distinct",
        "总共有多少种不同的疾病？",
        "SELECT COUNT(DISTINCT disease) AS disease_count FROM visits",
    ),
    (
        "distinct_ordered_list",
        "按名称顺序列出所有不同的处方药物。",
        "SELECT DISTINCT drug_prescribed FROM visits "
        "WHERE drug_prescribed != '' ORDER BY drug_prescribed ASC",
    ),
    (
        "row_argmax_join",
        "费用最高的那次就诊对应的患者叫什么名字？",
        "SELECT p.name FROM patients AS p JOIN visits AS v "
        "ON v.patient_id = p.patient_id "
        "ORDER BY v.cost DESC, v.visit_id ASC LIMIT 1",
    ),
    (
        "row_argmin_join",
        "费用最低的就诊患者姓名是什么？",
        "SELECT p.name FROM patients AS p JOIN visits AS v "
        "ON v.patient_id = p.patient_id "
        "ORDER BY v.cost ASC, v.visit_id ASC LIMIT 1",
    ),
    (
        "scalar_average",
        "全部就诊记录的平均费用是多少？",
        "SELECT AVG(cost) AS avg_cost FROM visits",
    ),
    (
        "scalar_maximum",
        "就诊费用的最高值是多少？",
        "SELECT MAX(cost) AS max_cost FROM visits",
    ),
    (
        "scalar_minimum",
        "最低的一笔就诊费用是多少？",
        "SELECT MIN(cost) AS min_cost FROM visits",
    ),
    (
        "top_k_rows",
        "列出费用最高的3次就诊的编号和费用。",
        "SELECT visit_id, cost FROM visits "
        "ORDER BY cost DESC, visit_id ASC LIMIT 3",
    ),
    (
        "filtered_joined_list",
        "哪些患者去过神经内科？列出姓名。",
        "SELECT DISTINCT p.name FROM patients AS p JOIN visits AS v "
        "ON v.patient_id = p.patient_id "
        "WHERE v.department = '神经内科' ORDER BY p.name ASC",
    ),
    (
        "anti_correlated_value",
        "哪些患者从未被开过阿司匹林？列出姓名。",
        "SELECT p.name FROM patients AS p WHERE NOT EXISTS "
        "(SELECT v.patient_id AS patient_id FROM visits AS v "
        "WHERE v.patient_id = p.patient_id "
        "AND v.drug_prescribed = '阿司匹林') ORDER BY p.name ASC",
    ),
    (
        "set_union",
        "列出曾在心血管内科或内分泌科就诊过的患者编号。",
        "SELECT patient_id FROM visits WHERE department = '心血管内科' "
        "UNION SELECT patient_id FROM visits WHERE department = '内分泌科' "
        "ORDER BY patient_id ASC",
    ),
    (
        "set_except",
        "列出在心血管内科就诊过但没有在内分泌科就诊过的患者编号。",
        "SELECT patient_id FROM visits WHERE department = '心血管内科' "
        "EXCEPT SELECT patient_id FROM visits WHERE department = '内分泌科' "
        "ORDER BY patient_id ASC",
    ),
    (
        "set_intersect",
        "哪些患者既去过心血管内科又去过内分泌科？列出患者编号。",
        "SELECT patient_id FROM visits WHERE department = '心血管内科' "
        "INTERSECT SELECT patient_id FROM visits WHERE department = '内分泌科' "
        "ORDER BY patient_id ASC",
    ),
    (
        "latest_per_group",
        "列出每位患者最近一次就诊的患者编号、日期和科室。",
        "SELECT v.patient_id, v.visit_date, v.department FROM visits AS v "
        "WHERE v.visit_date = (SELECT MAX(v2.visit_date) FROM visits AS v2 "
        "WHERE v2.patient_id = v.patient_id) "
        "ORDER BY v.patient_id ASC, v.visit_id ASC",
    ),
    (
        "earliest_per_group",
        "每位患者第一次就诊的编号、患者编号和日期分别是什么？",
        "SELECT v.visit_id, v.patient_id, v.visit_date FROM visits AS v "
        "WHERE v.visit_date = (SELECT MIN(v2.visit_date) FROM visits AS v2 "
        "WHERE v2.patient_id = v.patient_id) "
        "ORDER BY v.patient_id ASC, v.visit_id ASC",
    ),
    (
        "group_disease_count",
        "每种疾病分别有多少次就诊？",
        "SELECT disease, COUNT(*) AS visit_count FROM visits "
        "GROUP BY disease ORDER BY disease ASC",
    ),
    (
        "group_patient_argmax",
        "哪种疾病涉及的不同患者人数最多？",
        "SELECT disease, COUNT(DISTINCT patient_id) AS patient_count "
        "FROM visits GROUP BY disease "
        "ORDER BY patient_count DESC, disease ASC LIMIT 1",
    ),
    (
        "group_drug_argmax",
        "哪种处方药物被开出的次数最多？",
        "SELECT drug_prescribed, COUNT(*) AS prescription_count FROM visits "
        "WHERE drug_prescribed != '' GROUP BY drug_prescribed "
        "ORDER BY prescription_count DESC, drug_prescribed ASC LIMIT 1",
    ),
    (
        "joined_group_top_k",
        "按患者汇总费用，列出累计费用最高的前3位患者姓名和累计费用。",
        "SELECT p.name, SUM(v.cost) AS total_cost FROM patients AS p "
        "JOIN visits AS v ON v.patient_id = p.patient_id "
        "GROUP BY p.patient_id, p.name "
        "ORDER BY total_cost DESC, p.name ASC LIMIT 3",
    ),
    (
        "ordered_projection",
        "按费用从高到低列出就诊编号、疾病和费用。",
        "SELECT visit_id, disease, cost FROM visits "
        "ORDER BY cost DESC, visit_id ASC",
    ),
    (
        "filtered_aggregate",
        "神经内科全部就诊的平均费用是多少？",
        "SELECT AVG(cost) AS avg_cost FROM visits "
        "WHERE department = '神经内科'",
    ),
    (
        "filtered_distinct_count",
        "心血管内科一共有多少位不同的就诊患者？",
        "SELECT COUNT(DISTINCT patient_id) AS patient_count FROM visits "
        "WHERE department = '心血管内科'",
    ),
    (
        "filtered_latest",
        "内分泌科最近一次就诊的编号和日期是什么？",
        "SELECT visit_id, visit_date FROM visits "
        "WHERE department = '内分泌科' "
        "ORDER BY visit_date DESC, visit_id DESC LIMIT 1",
    ),
]

EXPECTED_BY_CATEGORY = {
    category: expected for category, _question, expected in CASES
}

PARAPHRASES = [
    ("group_argmax", "就诊次数最多的是哪个科室？"),
    ("group_argmax", "请问哪个科室的就诊量最大？"),
    ("group_argmin", "接诊次数最少的是哪一个科室？"),
    ("joined_distinct_list", "哪些病人曾有过就诊记录？请列出名字。"),
    ("group_having_join", "就诊超过一回的病人有哪些？返回姓名和地区。"),
    ("global_latest_join", "最后一条就诊记录属于哪位患者？请给出姓名。"),
    ("anti_join", "找出从未到院看过病的患者姓名。"),
    ("group_count", "各科室的就诊量分别是多少？"),
    ("count_distinct", "不同疾病一共有几种？"),
    ("distinct_ordered_list", "把所有处方药去重后按名称列出来。"),
    ("scalar_average", "就诊费用平均是多少？"),
    ("scalar_maximum", "所有就诊中最大的费用是多少？"),
    ("top_k_rows", "请给出最贵的三次就诊记录编号及费用。"),
    ("filtered_joined_list", "请列出曾到神经内科看病的患者名字。"),
    ("anti_correlated_value", "找出没有使用过阿司匹林的病人姓名。"),
    (
        "set_union",
        "心血管内科和内分泌科任意一个科室去过的患者编号有哪些？",
    ),
    (
        "set_except",
        "哪些患者去过心血管内科却没去过内分泌科？返回患者编号。",
    ),
    (
        "set_intersect",
        "请给出同时在心血管内科与内分泌科看过病的患者编号。",
    ),
    (
        "latest_per_group",
        "请返回各患者末次就诊的患者编号、日期和科室。",
    ),
    (
        "earliest_per_group",
        "各个患者最早的就诊记录编号、患者编号和日期有哪些？",
    ),
    ("group_disease_count", "各类疾病的就诊量是多少？"),
    ("group_patient_argmax", "哪一种疾病的患者数最大？"),
    ("group_drug_argmax", "开药次数最多的是哪种药？"),
    (
        "joined_group_top_k",
        "总花费最高的前三位患者分别叫什么名字？",
    ),
    (
        "ordered_projection",
        "请按费用降序给出就诊编号、诊断及金额。",
    ),
    ("filtered_aggregate", "神经内科就诊一次平均要花多少钱？"),
    (
        "filtered_distinct_count",
        "心血管内科有几名不同患者来就诊？",
    ),
    (
        "filtered_latest",
        "内分泌科末次就诊发生在什么时候？请给出记录编号和日期。",
    ),
]


@pytest.mark.parametrize("_category,question,expected", CASES)
def test_cspider_public_compositional_rule(
    cspider_compositional_db,
    _category,
    question,
    expected,
):
    engine = NL2SQL(db_path=cspider_compositional_db)
    predicted = engine.rule_sql(question)
    assert predicted == expected
    validate_gold_sql(predicted)
    assert execute_complete_readonly(
        predicted,
        cspider_compositional_db,
    ) == execute_complete_readonly(expected, cspider_compositional_db)


@pytest.mark.parametrize("category,question", PARAPHRASES)
def test_cspider_public_compositional_paraphrase(
    cspider_compositional_db,
    category,
    question,
):
    expected = EXPECTED_BY_CATEGORY[category]
    if category == "joined_group_top_k" and "累计费用" not in question:
        expected = (
            "SELECT p.name FROM patients AS p JOIN visits AS v "
            "ON v.patient_id = p.patient_id "
            "GROUP BY p.patient_id, p.name "
            "ORDER BY SUM(v.cost) DESC, p.name ASC LIMIT 3"
        )
    predicted = NL2SQL(db_path=cspider_compositional_db).rule_sql(question)
    assert predicted == expected
    validate_gold_sql(predicted)
    assert execute_complete_readonly(
        predicted,
        cspider_compositional_db,
    ) == execute_complete_readonly(expected, cspider_compositional_db)


def test_top_rows_preserve_grounded_domain_filters(cspider_compositional_db):
    question = "列出神经内科费用最高的2次就诊的编号和费用。"
    domains = {"department": ["神经内科"]}
    sql = _compile_top_rows(question, domains)

    assert sql == (
        "SELECT visit_id, cost FROM visits WHERE department = '神经内科' "
        "ORDER BY cost DESC, visit_id ASC LIMIT 2"
    )
    assert compile_compositional_medical_sql(question, domains) == sql
    connection = sqlite3.connect(cspider_compositional_db)
    try:
        assert connection.execute(sql).fetchall() == [
            ("V5", 3000.0),
            ("V4", 600.0),
        ]
    finally:
        connection.close()


def test_patient_cost_top_dispatches_before_visit_row_topk():
    question = "按患者汇总就诊费用，列出累计费用最高的前2位患者姓名和累计费用。"

    sql = compile_compositional_medical_sql(question, {})

    assert sql is not None
    assert "SUM(v.cost) AS total_cost" in sql
    assert "GROUP BY p.patient_id, p.name" in sql
    assert "ORDER BY total_cost DESC, p.name ASC LIMIT 2" in sql
    assert "ORDER BY cost" not in sql


def test_patient_cost_top_applies_domains_before_grouped_ranking(
    cspider_compositional_db,
):
    question = "按患者汇总神经内科总费用，列出前2位患者姓名和累计费用。"
    domains = {"department": ["神经内科"]}
    sql = _compile_patient_cost_top(question, domains)

    assert sql is not None
    assert "v.department = '神经内科'" in sql
    assert compile_compositional_medical_sql(question, domains) == sql
    connection = sqlite3.connect(cspider_compositional_db)
    try:
        assert connection.execute(sql).fetchall() == [
            ("王芳", 3000.0),
            ("李敏", 600.0),
        ]
    finally:
        connection.close()


def test_patient_having_applies_domains_before_counting(cspider_compositional_db):
    question = "哪些患者就诊至少2次，限定神经内科，返回患者编号。"
    domains = {"department": ["神经内科"]}
    sql = _compile_patient_having(question, domains)

    assert sql is not None
    assert "v.department = '神经内科'" in sql
    assert compile_compositional_medical_sql(question, domains) == sql
    connection = sqlite3.connect(cspider_compositional_db)
    try:
        assert connection.execute(sql).fetchall() == []
    finally:
        connection.close()


def test_per_patient_temporal_applies_domains_to_outer_and_inner_visits(
    cspider_compositional_db,
):
    question = "列出每位患者最近一次神经内科就诊的患者编号、就诊日期和科室。"
    domains = {"department": ["神经内科"]}
    sql = _compile_per_patient_temporal(question, domains)

    assert sql is not None
    assert "v2.department = '神经内科'" in sql
    assert "v.department = '神经内科'" in sql
    assert compile_compositional_medical_sql(question, domains) == sql
    connection = sqlite3.connect(cspider_compositional_db)
    try:
        assert connection.execute(sql).fetchall() == [
            ("P2", "2025-03-01", "神经内科"),
            ("P3", "2025-01-15", "神经内科"),
        ]
    finally:
        connection.close()


def test_distinct_list_applies_grounded_department_filter(cspider_compositional_db):
    question = "列出神经内科不同的处方药物。"
    domains = {"department": ["神经内科"]}

    sql = _compile_distinct_list(question, domains)

    assert sql is not None
    assert "WHERE department = '神经内科' AND drug_prescribed != ''" in sql
    assert compile_compositional_medical_sql(question, domains) == sql
    connection = sqlite3.connect(cspider_compositional_db)
    try:
        assert {row[0] for row in connection.execute(sql).fetchall()} == {"布洛芬", "阿司匹林"}
    finally:
        connection.close()


def test_anti_join_keeps_all_cross_domain_visit_qualifiers(cspider_compositional_db):
    question = "列出从未在神经内科使用阿司匹林的患者编号。"
    domains = {
        "department": ["神经内科"],
        "drug": ["阿司匹林"],
    }

    sql = _compile_anti_join(question, domains)

    assert sql is not None
    assert "v.department = '神经内科' AND v.drug_prescribed = '阿司匹林'" in sql
    assert compile_compositional_medical_sql(question, domains) == sql
    connection = sqlite3.connect(cspider_compositional_db)
    try:
        assert connection.execute(sql).fetchall() == [("P1",), ("P2",), ("P4",), ("P5",)]
    finally:
        connection.close()


def test_set_operation_repeats_shared_cross_domain_filter_in_each_branch(cspider_compositional_db):
    question = "列出患糖尿病且曾在心血管内科或内分泌科就诊过的患者编号。"
    domains = {
        "department": ["心血管内科", "内分泌科"],
        "disease": ["糖尿病"],
    }

    sql = _compile_set_operation(question, domains)

    assert sql is not None
    assert sql.count("disease = '糖尿病'") == 2
    assert compile_compositional_medical_sql(question, domains) == sql
    connection = sqlite3.connect(cspider_compositional_db)
    try:
        assert connection.execute(sql).fetchall() == [("P2",), ("P4",)]
    finally:
        connection.close()


def test_set_operation_keeps_explicit_or_over_suffix_simultaneous_filter():
    question = "列出曾在心血管内科或内分泌科就诊同时患糖尿病的患者编号。"
    domains = {
        "department": ["心血管内科", "内分泌科"],
        "disease": ["糖尿病"],
    }

    sql = _compile_set_operation(question, domains)

    assert sql is not None
    assert " UNION " in sql
    assert " INTERSECT " not in sql
    assert sql.count("disease = '糖尿病'") == 2
    assert compile_compositional_medical_sql(question, domains) == sql


def test_set_operation_keeps_postfix_or_before_suffix_global_filter():
    question = "列出曾在心血管内科和内分泌科二者之一就诊且患糖尿病的患者编号。"
    domains = {
        "department": ["心血管内科", "内分泌科"],
        "disease": ["糖尿病"],
    }

    sql = _compile_set_operation(question, domains)

    assert sql is not None
    assert " UNION " in sql
    assert sql.count("disease = '糖尿病'") == 2
    assert compile_compositional_medical_sql(question, domains) == sql


def test_set_operation_rejects_unrepresented_third_operand_before_patient_list_fallback():
    question = "列出曾在心血管内科、内分泌科或神经内科就诊过的患者编号。"
    domains = {"department": ["心血管内科", "内分泌科", "神经内科"]}

    assert _compile_set_operation(question, domains) is None
    assert compile_compositional_medical_sql(question, domains) is None


def test_cross_dimension_or_is_terminal_not_an_and_patient_list_query():
    question = "列出曾在神经内科或使用阿司匹林的患者姓名。"
    domains = {
        "department": ["神经内科"],
        "drug": ["阿司匹林"],
    }

    assert compile_compositional_medical_sql(question, domains) is None


def test_set_operation_rejects_interstitial_branch_local_qualifier():
    question = "列出既在心血管内科患冠心病又在内分泌科就诊过的患者编号。"
    domains = {
        "department": ["心血管内科", "内分泌科"],
        "disease": ["冠心病"],
    }

    assert _compile_set_operation(question, domains) is None
    assert compile_compositional_medical_sql(question, domains) is None


@pytest.mark.parametrize(
    ("question", "domains"),
    [
        (
            "列出曾在心血管内科、内分泌科或肿瘤科就诊过的患者编号。",
            {"department": ["心血管内科", "内分泌科"]},
        ),
        (
            "列出曾在心血管内科或内分泌科患冠心病的患者编号。",
            {
                "department": ["心血管内科", "内分泌科"],
                "disease": ["糖尿病"],
            },
        ),
        (
            "列出曾在心血管内科 OR 肿瘤科就诊过的患者编号。",
            {"department": ["心血管内科"]},
        ),
        (
            "列出在心血管内科要么在肿瘤科就诊的患者编号。",
            {"department": ["心血管内科"]},
        ),
        (
            "列出患糖尿病或冠心病的患者编号。",
            {"disease": ["糖尿病"]},
        ),
    ],
)
def test_set_operation_rejects_ungrounded_explicit_value_slots(question, domains):
    assert _compile_set_operation(question, domains) is None
    assert compile_compositional_medical_sql(question, domains) is None


def test_set_intent_without_patient_id_projection_is_terminal():
    question = "列出曾在心血管内科或内分泌科就诊过的患者姓名。"
    domains = {"department": ["心血管内科", "内分泌科"]}

    assert compile_compositional_medical_sql(question, domains) is None


def test_independently_negated_anti_join_clauses_are_terminal():
    question = "列出从未在神经内科就诊且从未使用阿司匹林的患者编号。"
    domains = {
        "department": ["神经内科"],
        "drug": ["阿司匹林"],
    }

    assert _compile_anti_join(question, domains) is None
    assert compile_compositional_medical_sql(question, domains) is None


def test_distinct_list_rejects_unmatched_qualified_department():
    question = "列出在肿瘤科就诊的不同处方药物。"
    domains = {"department": ["神经内科"]}

    assert _compile_distinct_list(question, domains) is None
    assert compile_compositional_medical_sql(question, domains) is None


@pytest.mark.parametrize(
    "question",
    [
        "列出要么在心血管内科就诊、要么在内分泌科就诊过的患者编号。",
        "列出曾在心血管内科 or 内分泌科就诊过的患者编号。",
        "列出曾在心血管内科和内分泌科二者之一就诊过的患者编号。",
    ],
)
def test_set_operation_supports_controlled_or_variants(question):
    domains = {"department": ["心血管内科", "内分泌科"]}
    expected = (
        "SELECT patient_id FROM visits WHERE department = '心血管内科' "
        "UNION SELECT patient_id FROM visits WHERE department = '内分泌科' "
        "ORDER BY patient_id ASC"
    )

    assert _compile_set_operation(question, domains) == expected
    assert compile_compositional_medical_sql(question, domains) == expected


def test_cross_dimension_or_variant_is_terminal_for_anti_join():
    question = "列出从未在神经内科或是使用阿司匹林的患者姓名。"
    postfix_question = "列出从未在神经内科和使用阿司匹林二者之一的患者姓名。"
    domains = {
        "department": ["神经内科"],
        "drug": ["阿司匹林"],
    }

    assert _compile_anti_join(question, domains) is None
    assert compile_compositional_medical_sql(question, domains) is None
    assert _compile_anti_join(postfix_question, domains) is None
    assert compile_compositional_medical_sql(postfix_question, domains) is None
