from __future__ import annotations

import sqlite3

import pytest

from dki.analysis.nl2sql import NL2SQL
from dki.analysis.nl2sql_protocol import (
    execute_complete_readonly,
    validate_gold_sql,
)


@pytest.fixture()
def mimicsql_rule_db(tmp_path):
    path = tmp_path / "mimicsql_rule.sqlite"
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
            name TEXT, type TEXT, mention_count INTEGER
        );
        CREATE TABLE kg_triples (
            head TEXT, head_type TEXT, relation TEXT, relation_zh TEXT,
            tail TEXT, tail_type TEXT, confidence REAL, count INTEGER
        );
        INSERT INTO patients VALUES
            ('P00001', '刘伟', '女', 49, '深圳'),
            ('P00002', '张强', '男', 72, '北京'),
            ('P00003', '王芳', '女', 35, '上海');
        INSERT INTO visits VALUES
            ('V1', 'P00001', '2025-01-01', '内分泌科', '2型糖尿病',
             '二甲双胍', '糖化血红蛋白', 100.0, 5),
            ('V2', 'P00002', '2025-02-01', '心内科', '高血压',
             '缬沙坦', '动态血压监测', 200.0, 12),
            ('V3', 'P00003', '2025-03-01', '神经内科', '脑卒中',
             '阿司匹林', '头颅MRI', 300.0, 8),
            ('V4', 'P00001', '2025-04-01', '内分泌科', '高血压',
             '缬沙坦', '动态血压监测', 150.0, 0);
        """
    )
    connection.commit()
    connection.close()
    return path


CASES = [
    (
        "how many patients aged below 76 years had 高血压 as the primary disease?",
        "SELECT COUNT(DISTINCT p.patient_id) FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.age<76 AND v.disease='高血压'",
    ),
    (
        "count the number of patients who were given 缬沙坦.",
        "SELECT COUNT(DISTINCT v.patient_id) FROM visits v "
        "WHERE v.drug_prescribed='缬沙坦'",
    ),
    (
        "Count the number of patients who have 高血压 as their primary disease "
        "and were admitted for more than 7 days in hospital.",
        "SELECT COUNT(DISTINCT v.patient_id) FROM visits v "
        "WHERE v.disease='高血压' AND v.los_days>7",
    ),
    (
        "Tell me the number of patients less than 75 years of age with a "
        "prescription for the 缬沙坦 drug.",
        "SELECT COUNT(DISTINCT p.patient_id) FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.age<75 "
        "AND v.drug_prescribed='缬沙坦'",
    ),
    (
        "how many patients had the primary disease as 高血压 and underwent "
        "the procedure named 动态血压监测?",
        "SELECT COUNT(DISTINCT v.patient_id) FROM visits v "
        "WHERE v.disease='高血压' AND v.exam='动态血压监测'",
    ),
    (
        "how many patients are below 60 years of age?",
        "SELECT COUNT(DISTINCT p.patient_id) FROM patients p WHERE p.age<60",
    ),
    (
        "find out the primary disease and number of days for which patient "
        "with patient id P00001 was hospitalized in hospital.",
        "SELECT v.los_days, v.disease FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.patient_id='P00001'",
    ),
    (
        "calculate the maximum age of patients who had 高血压 primary disease "
        "and were hospitalized for 0 days.",
        "SELECT MAX(p.age) FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE v.disease='高血压' "
        "AND v.los_days=0",
    ),
    (
        "what is the minimum days of hospital stay for patients who are "
        "primarily diagnosed with 高血压?",
        "SELECT MIN(v.los_days) FROM visits v WHERE v.disease='高血压'",
    ),
    (
        "what is the primary disease and procedure long title of subject id "
        "P00001?",
        "SELECT v.disease, v.exam FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.patient_id='P00001'",
    ),
    (
        "what is the average age of 女 patients who are 40 years or above?",
        "SELECT AVG(p.age) FROM patients p WHERE p.gender='女' AND p.age>=40",
    ),
    (
        "what is the maximum age of 男 patients whose primary disease is 高血压?",
        "SELECT MAX(p.age) FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.gender='男' "
        "AND v.disease='高血压'",
    ),
    (
        "calculate the minimum age of 女 patients who have 2型糖尿病 primary disease.",
        "SELECT MIN(p.age) FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.gender='女' "
        "AND v.disease='2型糖尿病'",
    ),
    (
        "what is the age of subject id P00001 and his/her duration of stay "
        "in hospital?",
        "SELECT p.age, v.los_days FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.patient_id='P00001'",
    ),
    (
        "what is the gender and procedure short title of subject id P00002?",
        "SELECT p.gender, v.exam FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.patient_id='P00002'",
    ),
    (
        "what is the primary disease and procedure for the patient name 刘伟?",
        "SELECT v.disease, v.exam FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.name='刘伟'",
    ),
    (
        "what is the drug name of the patient 刘伟?",
        "SELECT v.drug_prescribed FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.name='刘伟'",
    ),
    (
        "how many patients are 男?",
        "SELECT COUNT(DISTINCT p.patient_id) FROM patients p "
        "WHERE p.gender='男'",
    ),
    (
        "what is the age of subject id P00002?",
        "SELECT p.age FROM patients p WHERE p.patient_id='P00002'",
    ),
    (
        "Specify the age and primary disease of patient id P00001",
        "SELECT p.age, v.disease FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.patient_id='P00001'",
    ),
    (
        "specify the primary disease and gender of patient id P00002",
        "SELECT p.gender, v.disease FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.patient_id='P00002'",
    ),
    (
        "what is the subject name of subject id P00003?",
        "SELECT p.name FROM patients p WHERE p.patient_id='P00003'",
    ),
    (
        "tell me the name and days for which patient with patient id P00001 "
        "was admitted in hospital.",
        "SELECT p.name, v.los_days FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.patient_id='P00001'",
    ),
    (
        "what is the diagnosis for patient name 张强?",
        "SELECT v.disease FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.name='张强'",
    ),
    (
        "what is the primary disease and drug name of the patient 刘伟?",
        "SELECT v.disease, v.drug_prescribed FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.name='刘伟'",
    ),
    (
        "report the primary disease and number of days for which patient 刘伟 "
        "was hospitalized.",
        "SELECT v.los_days, v.disease FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.name='刘伟'",
    ),
    (
        "give me the number of 高血压 primary disease patients who have "
        "stayed more than 7 days in hospital.",
        "SELECT COUNT(DISTINCT v.patient_id) FROM visits v "
        "WHERE v.disease='高血压' AND v.los_days>7",
    ),
    (
        "What is the number of patients younger than 60 years who have "
        "高血压 as their primary disease?",
        "SELECT COUNT(DISTINCT p.patient_id) FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.age<60 "
        "AND v.disease='高血压'",
    ),
    (
        "Which lab tests were conducted for patient id P00001? Also specify "
        "the name of the primary disease he was suffering with",
        "SELECT v.disease, v.exam FROM patients p JOIN visits v "
        "ON p.patient_id=v.patient_id WHERE p.patient_id='P00001'",
    ),
]


@pytest.mark.parametrize("question,expected", CASES)
def test_mimicsql_public_development_rule(
    mimicsql_rule_db,
    question,
    expected,
):
    engine = NL2SQL(db_path=mimicsql_rule_db)
    predicted = engine.rule_sql(question)
    assert predicted == expected
    validate_gold_sql(predicted)
    assert execute_complete_readonly(
        predicted,
        mimicsql_rule_db,
    ) == execute_complete_readonly(expected, mimicsql_rule_db)


@pytest.mark.parametrize(
    "question",
    [
        "how many patients had high blood pressure?",
        "what is the age and unknown insurance of patient id P00001?",
        "count the number of times patient P00001 received a 动态血压监测 "
        "procedure.",
        "what is the maximum age of patients?",
    ],
)
def test_mimicsql_rule_abstains_when_slots_are_not_fully_grounded(
    mimicsql_rule_db,
    question,
):
    engine = NL2SQL(db_path=mimicsql_rule_db)
    # The public MIMICSQL parser itself must not manufacture ungrounded
    # literals or answer event-count questions as distinct-patient counts.
    assert engine._rule_mimicsql_public_language(question) is None
