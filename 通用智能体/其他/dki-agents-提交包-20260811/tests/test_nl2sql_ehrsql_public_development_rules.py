import sqlite3

import pytest

from dki.analysis.nl2sql import NL2SQL
from dki.analysis.nl2sql_protocol import (
    execute_complete_readonly,
    validate_gold_sql,
)


@pytest.fixture()
def official_v2_rule_db(tmp_path):
    path = tmp_path / "official_v2_rule.sqlite"
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
        INSERT INTO patients VALUES
            ('P00001', '张伟', '男', 48, '北京'),
            ('P00002', '李敏', '女', 35, '上海');
        INSERT INTO visits VALUES
            ('emr_0001', 'P00001', '2025-01-05', '消化内科',
             '胃溃疡', '美托洛尔', '腹部CT', 900.0, 5),
            ('emr_0002', 'P00001', '2025-01-20', '消化内科',
             '胃溃疡', '美托洛尔', '腹部CT', 1200.0, 3),
            ('emr_0003', 'P00002', '2025-02-01', '内分泌科',
             '糖尿病', '二甲双胍', '血糖检查', 1800.0, 4);
        """
    )
    connection.commit()
    connection.close()
    return path


CASES = [
    (
        "How much does the 腹部CT cost?",
        "SELECT DISTINCT v.cost FROM visits v "
        "WHERE v.exam='腹部CT' ORDER BY v.cost",
    ),
    (
        "What is the diagnosis cost for 胃溃疡?",
        "SELECT DISTINCT v.cost FROM visits v "
        "WHERE v.disease='胃溃疡' ORDER BY v.cost",
    ),
    (
        "What is the price of the drug 美托洛尔?",
        "SELECT DISTINCT v.cost FROM visits v "
        "WHERE v.drug_prescribed='美托洛尔' ORDER BY v.cost",
    ),
    (
        "What do 腹部CT costs?",
        "SELECT DISTINCT v.cost FROM visits v "
        "WHERE v.exam='腹部CT' ORDER BY v.cost",
    ),
    (
        "What is the charge for a 腹部CT?",
        "SELECT DISTINCT v.cost FROM visits v "
        "WHERE v.exam='腹部CT' ORDER BY v.cost",
    ),
    (
        "How old is patient P00001 during this hospital encounter?",
        "SELECT p.age FROM patients p WHERE p.patient_id='P00001'",
    ),
    (
        "Count the number of times patient P00001 has received 腹部CT.",
        "SELECT COUNT(*) FROM visits v "
        "WHERE v.patient_id='P00001' AND v.exam='腹部CT'",
    ),
    (
        "How many times has the 腹部CT procedure been performed?",
        "SELECT COUNT(*) FROM visits v WHERE v.exam='腹部CT'",
    ),
    (
        "Calculate the number of patients who had a 腹部CT lab test.",
        "SELECT COUNT(DISTINCT v.patient_id) FROM visits v "
        "WHERE v.exam='腹部CT'",
    ),
    (
        "How many patients have been diagnosed with 胃溃疡?",
        "SELECT COUNT(DISTINCT v.patient_id) FROM visits v "
        "WHERE v.disease='胃溃疡'",
    ),
    (
        "Tell me the maximum hospital cost when there is 胃溃疡.",
        "SELECT MAX(v.cost) FROM visits v WHERE v.disease='胃溃疡'",
    ),
    (
        "What is the maximum hospital cost for a drug named 美托洛尔?",
        "SELECT MAX(v.cost) FROM visits v "
        "WHERE v.drug_prescribed='美托洛尔'",
    ),
    (
        "从2025-01-01到2025-01-31消化内科的住院量是多少?",
        "SELECT COUNT(*) FROM visits v WHERE v.department='消化内科' "
        "AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31'",
    ),
    (
        "内分泌科在2025-01-01至2025-01-31这段时间的住院总量有多少",
        "SELECT COUNT(*) FROM visits v WHERE v.department='内分泌科' "
        "AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31'",
    ),
    (
        "从2025-01-01到2025-01-31消化内科的门诊量如何啊?",
        "SELECT COUNT(*) FROM visits v WHERE v.department='消化内科' "
        "AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31'",
    ),
    (
        "列出患者P00001就诊结算日期在2025-01-01到2025-01-31内的所有医疗就诊记录编号",
        "SELECT v.visit_id FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.patient_id='P00001' "
        "AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31'",
    ),
    (
        "就诊结算日期从2025-01-01到2025-01-31,患者P00001总共有多少医疗就诊的记录编号,给全部列出来",
        "SELECT v.visit_id FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.patient_id='P00001' "
        "AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31'",
    ),
    (
        "就诊结算日期在2025-01-01到2025-01-31内的患者P00001所有医疗就诊记录编号分别是有哪些?",
        "SELECT v.visit_id FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.patient_id='P00001' "
        "AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31'",
    ),
    (
        "从2025-01-01到2025-01-31,病人P00001所有的医疗就诊记录的编号列一下",
        "SELECT v.visit_id FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.patient_id='P00001' "
        "AND v.visit_date BETWEEN '2025-01-01' AND '2025-01-31'",
    ),
]


@pytest.mark.parametrize("question,expected", CASES)
def test_official_v2_public_development_rule(
    official_v2_rule_db,
    question,
    expected,
):
    engine = NL2SQL(db_path=official_v2_rule_db)
    predicted = engine.rule_sql(question)
    assert predicted == expected
    validate_gold_sql(predicted)
    assert execute_complete_readonly(
        predicted,
        official_v2_rule_db,
    ) == execute_complete_readonly(expected, official_v2_rule_db)
