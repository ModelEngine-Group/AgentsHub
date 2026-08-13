import sqlite3
from pathlib import Path

import pytest

from dki.analysis.nl2sql_css_projection import (
    CSSSchema,
    ProjectionRejected,
    deduplicate_projected_items,
    load_target_domains,
    project_css_row,
    skeleton_role,
)


@pytest.fixture()
def css_projection_db(tmp_path: Path) -> Path:
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
                "V0001",
                "P00001",
                "2024-07-20",
                "心血管内科",
                "冠心病",
                "阿司匹林",
                "心电图",
                3200.0,
                5,
            ),
            (
                "V0002",
                "P00002",
                "2024-08-21",
                "内分泌科",
                "2型糖尿病",
                "二甲双胍",
                "糖化血红蛋白",
                1800.0,
                3,
            ),
        ],
    )
    connection.commit()
    connection.close()
    return path


def _schema() -> CSSSchema:
    return CSSSchema(
        db_id="medical_dev",
        columns=(
            (-1, "*"),
            (0, "PERSON_ID"),
            (0, "PERSON_NM"),
            (0, "PERSON_AGE"),
            (0, "MED_CLINIC_ID"),
            (0, "IN_HOSP_DATE"),
            (0, "MED_ORG_DEPT_NM"),
            (0, "IN_DIAG_DIS_NM"),
            (0, "MED_AMOUT"),
        ),
        table_names=("encounter",),
    )


def _filter_row():
    return {
        "db_id": "medical_dev",
        "question_id": "q1",
        "question": "患者90001在2020年1月1日以后有哪些诊断？",
        "query": (
            "SELECT IN_DIAG_DIS_NM FROM encounter "
            "WHERE PERSON_ID='90001' AND IN_HOSP_DATE>='2020-01-01'"
        ),
        "sql": {
            "select": [[0, [0, [0, 7, False], None]]],
            "from": {"table_units": [["table_unit", 0]], "conds": []},
            "where": [
                [False, 2, [0, [0, 1, False], None], "90001", None],
                "and",
                [False, 5, [0, [0, 5, False], None], "2020-01-01", None],
            ],
            "groupBy": [],
            "having": [],
            "orderBy": [],
            "limit": None,
            "intersect": None,
            "union": None,
            "except": None,
        },
    }


def test_css_projection_is_deterministic_and_grounded(css_projection_db):
    domains = load_target_domains(css_projection_db)
    first = project_css_row(
        _filter_row(),
        _schema(),
        domains,
        css_projection_db,
        source_split="train",
    )
    second = project_css_row(
        _filter_row(),
        _schema(),
        domains,
        css_projection_db,
        source_split="train",
    )
    v2 = project_css_row(
        _filter_row(),
        _schema(),
        domains,
        css_projection_db,
        source_split="train",
        partition_version="v2",
    )
    assert first == second
    assert v2.item["id"] != first.item["id"]
    assert v2.item["provenance"]["partition_version"] == "v2"
    assert "90001" not in first.item["question"]
    assert "2020" not in first.item["question"]
    assert "patients p JOIN visits v" in first.item["gold_sql"]
    assert first.item["provenance"]["source_split"] == "train"
    assert first.skeleton_role in {"development", "holdout"}


@pytest.mark.parametrize(
    "date_text",
    [
        "二零二零年一月一日",
        "二零年一月一日",
        "二0二零年一月一日",
        "2零二0年一月一日",
        "20-01-01",
        "2020.1.1",
    ],
)
def test_css_projection_accepts_public_chinese_date_forms(
    css_projection_db,
    date_text,
):
    row = _filter_row()
    row["question"] = f"患者90001在{date_text}以后有哪些诊断？"
    projected = project_css_row(
        row,
        _schema(),
        load_target_domains(css_projection_db),
        css_projection_db,
        source_split="train",
    )
    assert date_text not in projected.item["question"]
    assert "2020-01-01" not in projected.item["gold_sql"]


def test_css_projection_normalises_patient_count_to_distinct(
    css_projection_db,
):
    row = _filter_row()
    row["question"] = "患有源疾病的患者一共有多少人？"
    row["sql"]["select"] = [[3, [0, [0, 1, False], None]]]
    row["sql"]["where"] = [
        [False, 2, [0, [0, 7, False], None], "源疾病", None],
    ]
    projected = project_css_row(
        row,
        _schema(),
        load_target_domains(css_projection_db),
        css_projection_db,
        source_split="train",
    )
    assert "COUNT(DISTINCT p.patient_id)" in projected.item["gold_sql"]

    row["question"] = "有多少患过源疾病的人？"
    projected = project_css_row(
        row,
        _schema(),
        load_target_domains(css_projection_db),
        css_projection_db,
        source_split="train",
    )
    assert "COUNT(DISTINCT p.patient_id)" in projected.item["gold_sql"]


def test_css_projection_rejects_unmapped_columns(css_projection_db):
    row = _filter_row()
    row["sql"]["select"] = [[0, [0, [0, 8, False], None]]]
    schema = CSSSchema(
        db_id="medical_dev",
        columns=_schema().columns[:-1] + ((0, "HOSPITAL_SECRET_CODE"),),
        table_names=("encounter",),
    )
    with pytest.raises(ProjectionRejected, match="unmapped_source_column"):
        project_css_row(
            row,
            schema,
            load_target_domains(css_projection_db),
            css_projection_db,
            source_split="dev",
        )


def test_css_projection_rejects_null_from_annotation(css_projection_db):
    row = _filter_row()
    row["sql"]["from"] = None
    with pytest.raises(ProjectionRejected, match="missing_from_annotation"):
        project_css_row(
            row,
            _schema(),
            load_target_domains(css_projection_db),
            css_projection_db,
            source_split="dev",
        )


@pytest.mark.parametrize(
    "select_payload,question,reason",
    [
        ([[0, [0, [0, 0, False], None]]], "查询全部记录", "unbounded_star_projection"),
        (
            [[4, [0, [0, 4, False], None]]],
            "门诊医疗费总额是多少？",
            "non_numeric_aggregation",
        ),
        (
            [[3, [0, [0, 0, False], None]]],
            "编号为V0001的检验报告单有几份？",
            "unsupported_question_semantics",
        ),
    ],
)
def test_css_projection_rejects_semantic_schema_mismatch(
    css_projection_db,
    select_payload,
    question,
    reason,
):
    row = _filter_row()
    row["question"] = question
    row["sql"]["select"] = select_payload
    row["sql"]["where"] = []
    with pytest.raises(ProjectionRejected, match=reason):
        project_css_row(
            row,
            _schema(),
            load_target_domains(css_projection_db),
            css_projection_db,
            source_split="dev",
        )


def test_css_skeleton_partition_is_stable_and_family_wide():
    skeleton = "SELECT DISEASE FROM VISITS WHERE COST >= '__LITERAL__'"
    assert skeleton_role(skeleton) == skeleton_role(skeleton)
    assert skeleton_role(skeleton) in {"development", "holdout"}


def test_css_projection_deduplicates_normalised_questions(css_projection_db):
    domains = load_target_domains(css_projection_db)
    first = project_css_row(
        _filter_row(),
        _schema(),
        domains,
        css_projection_db,
        source_split="train",
    )
    assert deduplicate_projected_items([first, first]) == [first]


def test_css_projection_source_does_not_import_rule_engine():
    source = (
        Path(__file__).resolve().parents[1]
        / "dki"
        / "analysis"
        / "nl2sql_css_projection.py"
    ).read_text(encoding="utf-8")
    assert "from dki.analysis.nl2sql import" not in source
    assert "import dki.analysis.nl2sql" not in source
