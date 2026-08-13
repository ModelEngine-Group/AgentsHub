import sqlite3

import pytest

from dki.analysis.nl2sql import NL2SQL
from dki.analysis.nl2sql_protocol import execute_complete_readonly, validate_gold_sql


@pytest.fixture()
def css_rule_db(tmp_path):
    path = tmp_path / "css_rule.sqlite"
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
            ("P00001", "张伟", "男", 48, "北京"),
            ("P00002", "李敏", "女", 35, "上海"),
        ],
    )
    connection.executemany(
        "INSERT INTO visits VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (
                "emr_0001",
                "P00001",
                "2025-01-05",
                "心血管内科",
                "冠心病",
                "阿司匹林",
                "心电图",
                900.0,
                5,
            ),
            (
                "emr_0002",
                "P00001",
                "2025-01-20",
                "心血管内科",
                "冠心病",
                "阿司匹林",
                "心电图",
                1200.0,
                3,
            ),
            (
                "emr_0003",
                "P00002",
                "2025-02-01",
                "内分泌科",
                "2型糖尿病",
                "二甲双胍",
                "糖化血红蛋白",
                1800.0,
                4,
            ),
        ],
    )
    connection.commit()
    connection.close()
    return path


CASES = [
    (
        "参保人张伟1000元以内小金额结算次数有多少，时间是在2025-01-01到2025-01-31内？",
        "SELECT COUNT(*) FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE p.name='张伟' AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31' "
        "AND v.cost<=1000",
    ),
    (
        "从2025-01-01到2025-01-31参保人的编码为P00001的1000元以内小金额一共结算多少次呢？",
        "SELECT COUNT(*) FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE p.patient_id='P00001' AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31' "
        "AND v.cost<=1000",
    ),
    (
        "被开出阿司匹林的患者张伟在2025-01-01到2025-01-31期间开了多少次？",
        "SELECT COUNT(*) FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE p.name='张伟' AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31' "
        "AND v.drug_prescribed='阿司匹林'",
    ),
    (
        "患者P00001在2025-01-01到2025-01-31期间被开出阿司匹林共有多少次啊？",
        "SELECT COUNT(*) FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE p.patient_id='P00001' AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31' "
        "AND v.drug_prescribed='阿司匹林'",
    ),
    (
        "从2025-01-01到2025-01-31心血管内科开的阿司匹林的开药量是多少？",
        "SELECT COUNT(*) FROM visits v WHERE v.department='心血管内科' "
        "AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31' "
        "AND v.drug_prescribed='阿司匹林'",
    ),
    (
        "有多少患者是患有冠心病的？",
        "SELECT COUNT(DISTINCT v.patient_id) FROM visits v WHERE v.disease='冠心病'",
    ),
    (
        "从2025-01-01到2025-01-31，病患张伟医保的医疗费总额是多少？",
        "SELECT SUM(v.cost) FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE p.name='张伟' AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31'",
    ),
    (
        "查询医疗就诊emr_0001所需要的住院时长",
        "SELECT v.los_days FROM visits v WHERE v.visit_id='emr_0001'",
    ),
    (
        "门诊就诊emr_0001是多少号来就诊的？",
        "SELECT v.visit_date FROM visits v WHERE v.visit_id='emr_0001'",
    ),
    (
        "从2025-01-01到2025-01-31，患者张伟看病时被开出阿司匹林的次数总共有多少？",
        "SELECT COUNT(*) FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE p.name='张伟' AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31' "
        "AND v.drug_prescribed='阿司匹林'",
    ),
    (
        "在2025-01-01到2025-01-31这段时间，心血管内科有多少阿司匹林被开出？",
        "SELECT COUNT(*) FROM visits v WHERE v.department='心血管内科' "
        "AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31' "
        "AND v.drug_prescribed='阿司匹林'",
    ),
    (
        "有多少患过冠心病的人？",
        "SELECT COUNT(DISTINCT v.patient_id) FROM visits v WHERE v.disease='冠心病'",
    ),
    (
        "这位张伟的患者在2025-01-01到2025-01-31期间医保的医疗总消费金额有多少？",
        "SELECT SUM(v.cost) FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE p.name='张伟' AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31'",
    ),
    (
        "在2025-01-01到2025-01-31这段时间里病号张伟内医保的医疗费有多少钱？",
        "SELECT SUM(v.cost) FROM patients p JOIN visits v ON p.patient_id=v.patient_id "
        "WHERE p.name='张伟' AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31'",
    ),
    (
        "编号emr_0001的医疗就诊需要住院多久？",
        "SELECT v.los_days FROM visits v WHERE v.visit_id='emr_0001'",
    ),
    (
        "在emr_0001这次医疗就诊中，需要住院的时间如何？",
        "SELECT v.los_days FROM visits v WHERE v.visit_id='emr_0001'",
    ),
    (
        "门诊就诊emr_0001的时间是啥时候？",
        "SELECT v.visit_date FROM visits v WHERE v.visit_id='emr_0001'",
    ),
    (
        "冠心病患者的平均年龄是多少？",
        "SELECT ROUND(AVG(p.age), 2) AS avg_value FROM patients p "
        "WHERE EXISTS (SELECT v.patient_id AS patient_id FROM visits v "
        "WHERE v.patient_id=p.patient_id AND v.disease='冠心病')",
    ),
]


@pytest.mark.parametrize("question,expected", CASES)
def test_css_public_development_rule(css_rule_db, question, expected):
    engine = NL2SQL(db_path=css_rule_db)
    predicted = engine.rule_sql(question)
    assert predicted == expected
    validate_gold_sql(predicted)
    assert execute_complete_readonly(predicted, css_rule_db) == execute_complete_readonly(
        expected,
        css_rule_db,
    )
