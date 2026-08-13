import sqlite3
from pathlib import Path

import pytest

from dki.analysis.nl2sql_css_projection import skeleton_role_v2
from dki.analysis.nl2sql_ehrsql_projection import (
    EHRProjectionRejected,
    deduplicate_ehrsql_items,
    load_ehr_target_domains,
    project_ehrsql_row,
)


@pytest.fixture()
def ehr_projection_db(tmp_path: Path) -> Path:
    path = tmp_path / "four_table.sqlite"
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
            ('P0001', '张伟', '男', 48, '北京'),
            ('P0002', '李敏', '女', 35, '上海');
        INSERT INTO visits VALUES
            ('V0001', 'P0001', '2025-01-10', '心内科', '高血压',
             '阿司匹林', '心电图', 3200.0, 2),
            ('V0002', 'P0002', '2025-02-11', '内分泌科', '糖尿病',
             '二甲双胍', '血糖检查', 1800.0, 1);
        """
    )
    connection.commit()
    connection.close()
    return path


def _row(tag: str, question: str, query: str, value: dict) -> dict:
    return {
        "db_id": "eicu",
        "id": "source-1",
        "q_tag": tag,
        "question": question,
        "query": query,
        "value": value,
        "is_impossible": False,
    }


def test_ehrsql_projection_is_deterministic_and_replaces_typed_values(
    ehr_projection_db,
):
    row = _row(
        "what is the cost of diagnosing {diagnosis_name}?",
        "What is the diagnosis cost for source disease?",
        "select distinct cost.cost from cost where diagnosisname='source disease'",
        {"diagnosis_name": "source disease"},
    )
    domains = load_ehr_target_domains(ehr_projection_db)
    first = project_ehrsql_row(
        row,
        domains,
        ehr_projection_db,
        source_dataset="eicu",
        source_split="train",
    )
    second = project_ehrsql_row(
        row,
        domains,
        ehr_projection_db,
        source_dataset="eicu",
        source_split="train",
    )
    assert first == second
    assert "source disease" not in first.item["question"].lower()
    assert "SELECT DISTINCT v.cost" in first.item["gold_sql"]
    assert first.item["provenance"]["partition_version"] == "v2"


def test_ehrsql_projection_rejects_temporal_semantics(ehr_projection_db):
    row = _row(
        "count the number of patients who were diagnosed with {diagnosis_name} [time_filter_global1].",
        "How many patients had source disease last year?",
        (
            "select count(*) from diagnosis where diagnosisname='source disease' "
            "and datetime(diagnosistime,'start of year')=datetime(current_time)"
        ),
        {"diagnosis_name": "source disease"},
    )
    with pytest.raises(
        EHRProjectionRejected,
        match="temporal_semantics_not_supported",
    ):
        project_ehrsql_row(
            row,
            load_ehr_target_domains(ehr_projection_db),
            ehr_projection_db,
            source_dataset="eicu",
            source_split="valid",
        )


def test_ehrsql_projection_rejects_question_without_cost_intent(
    ehr_projection_db,
):
    row = _row(
        "what is the cost of diagnosing {diagnosis_name}?",
        "What is the diagnostic source disease pneumothorax?",
        (
            "select distinct cost.cost from cost "
            "where diagnosisname='source disease'"
        ),
        {"diagnosis_name": "source disease"},
    )
    with pytest.raises(
        EHRProjectionRejected,
        match="question_intent_not_grounded",
    ):
        project_ehrsql_row(
            row,
            load_ehr_target_domains(ehr_projection_db),
            ehr_projection_db,
            source_dataset="mimic_iii",
            source_split="train",
        )


@pytest.mark.parametrize(
    "question,query",
    [
        (
            "What is patient source-patient's age during their first hospital visit?",
            (
                "select patient.age from patient where "
                "patient.uniquepid='source-patient' and "
                "patient.hospitaldischargetime is not null "
                "order by patient.hospitaladmittime asc limit 1"
            ),
        ),
        (
            "What is the diagnosis cost for source disease within the last 7 days?",
            (
                "select distinct cost.cost from cost "
                "where diagnosisname='source disease'"
            ),
        ),
    ],
)
def test_ehrsql_projection_rejects_unrepresentable_time_qualifiers(
    ehr_projection_db,
    question,
    query,
):
    row = _row(
        (
            "what is_verb the age of patient {patient_id} "
            "[time_filter_global1]?"
            if "patient" in question.lower()
            else "what is the cost of diagnosing {diagnosis_name}?"
        ),
        question,
        query,
        (
            {"patient_id": "source-patient"}
            if "patient" in question.lower()
            else {"diagnosis_name": "source disease"}
        ),
    )
    with pytest.raises(
        EHRProjectionRejected,
        match="temporal_(semantics|question)_not_supported",
    ):
        project_ehrsql_row(
            row,
            load_ehr_target_domains(ehr_projection_db),
            ehr_projection_db,
            source_dataset="eicu",
            source_split="train",
        )


def test_ehrsql_topk_grounding_and_age_join(ehr_projection_db):
    row = _row(
        "what are_verb the top [n_rank] frequent diagnoses of patients aged [age_group] [time_filter_global1]?",
        "What are the two most frequent diagnoses for patients in their 30s?",
        (
            "select disease from visits join patient on 1=1 "
            "where patient.age between 30 and 39 group by disease "
            "having count(*)>0 limit 2"
        ),
        {},
    )
    projected = project_ehrsql_row(
        row,
        load_ehr_target_domains(ehr_projection_db),
        ehr_projection_db,
        source_dataset="mimic_iii",
        source_split="train",
    )
    assert "top 2" in projected.item["question"].lower()
    assert "age between 30 and 39" in projected.item["question"].lower()
    assert "JOIN visits" in projected.item["gold_sql"]
    assert projected.item["difficulty"] == "extra"


def test_ehrsql_projection_deduplicates_and_uses_shared_v2_partition(
    ehr_projection_db,
):
    row = _row(
        "what is the gender of patient {patient_id}?",
        "What is the sex of patient source-patient?",
        "select distinct gender from patient where id='source-patient'",
        {"patient_id": "source-patient"},
    )
    projected = project_ehrsql_row(
        row,
        load_ehr_target_domains(ehr_projection_db),
        ehr_projection_db,
        source_dataset="eicu",
        source_split="train",
    )
    assert deduplicate_ehrsql_items([projected, projected]) == [projected]
    assert projected.skeleton_role in {"development", "holdout"}
    assert skeleton_role_v2(
        "SELECT GENDER FROM PATIENTS WHERE PATIENT_ID = '__LITERAL__'"
    ) == skeleton_role_v2(
        "SELECT GENDER FROM PATIENTS WHERE PATIENT_ID = '__LITERAL__'"
    )


def test_ehrsql_projection_source_does_not_import_rule_engine():
    source = (
        Path(__file__).resolve().parents[1]
        / "dki"
        / "analysis"
        / "nl2sql_ehrsql_projection.py"
    ).read_text(encoding="utf-8")
    assert "from dki.analysis.nl2sql import" not in source
    assert "import dki.analysis.nl2sql" not in source
