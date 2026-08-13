from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path

import pytest

from dki.analysis.nl2sql_css_projection import (
    ProjectedItem,
    load_target_domains,
)
from dki.analysis.nl2sql_mimicsql_projection import (
    MIMICSQLProjectionRejected,
    project_mimicsql_row,
)
from dki.analysis.nl2sql_protocol import (
    execute_complete_readonly,
    validate_gold_sql,
)
from scripts import build_nl2sql_mimicsql_development as builder


@pytest.fixture()
def mimicsql_target_db(tmp_path: Path) -> Path:
    path = tmp_path / "insight.db"
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
             '阿司匹林', '头颅MRI', 300.0, 8);
        """
    )
    connection.commit()
    connection.close()
    return path


def _row(
    tag: int,
    *,
    question: str,
    tables: list[int],
    conditions: list[list[object]],
    select: list[list[int]] | None = None,
    aggregation: int = 1,
) -> dict:
    return {
        "key": f"{tag:032x}",
        "format": {
            "table": tables,
            "cond": conditions,
            "agg_col": select or [[0, 0]],
            "sel": aggregation,
        },
        "question_refine": question,
        "sql": "source SQL is provenance only",
    }


@pytest.mark.parametrize(
    ("row", "expected_column"),
    [
        (
            _row(
                1,
                question="how many patients were diagnosed with Source disease?",
                tables=[0, 1],
                conditions=[[1, 3, 0, "Source disease"]],
            ),
            "v.disease",
        ),
        (
            _row(
                2,
                question="how many patients underwent Source procedure?",
                tables=[0, 2],
                conditions=[[2, 4, 0, "Source procedure"]],
            ),
            "v.exam",
        ),
        (
            _row(
                3,
                question="how many patients were treated with Source drug?",
                tables=[0, 3],
                conditions=[[3, 4, 0, "Source drug"]],
            ),
            "v.drug_prescribed",
        ),
        (
            _row(
                4,
                question="how many patients had a Source lab test?",
                tables=[0, 4],
                conditions=[[4, 6, 0, "Source lab"]],
            ),
            "v.exam",
        ),
    ],
)
def test_mimicsql_projects_medical_join_concepts_deterministically(
    mimicsql_target_db: Path,
    row: dict,
    expected_column: str,
):
    domains = load_target_domains(mimicsql_target_db)
    first = project_mimicsql_row(
        row,
        domains,
        mimicsql_target_db,
        source_split="train",
    )
    second = project_mimicsql_row(
        row,
        domains,
        mimicsql_target_db,
        source_split="train",
    )
    assert first == second
    assert expected_column in first.item["gold_sql"]
    assert "patients p JOIN visits v" in first.item["gold_sql"]
    assert first.item["category"] == "multi_table_join"
    assert "Source " not in first.item["question"]
    columns, rows = execute_complete_readonly(
        first.item["gold_sql"],
        mimicsql_target_db,
    )
    assert columns
    assert rows and rows[0][0] > 0


def test_mimicsql_projects_gender_age_and_patient_attributes(
    mimicsql_target_db: Path,
):
    domains = load_target_domains(mimicsql_target_db)
    count_row = _row(
        10,
        question=(
            "how many female patients below 80 years of age were "
            "diagnosed with Source disease?"
        ),
        tables=[0, 1],
        conditions=[
            [0, 6, 0, "F"],
            [0, 4, 2, 80],
            [1, 4, 0, "Source disease"],
        ],
    )
    projected = project_mimicsql_row(
        count_row,
        domains,
        mimicsql_target_db,
        source_split="dev",
    )
    assert "p.gender =" in projected.item["gold_sql"]
    assert "p.age <" in projected.item["gold_sql"]
    gender_literal = (
        "女" if "'女'" in projected.item["gold_sql"] else "男"
    )
    assert gender_literal in projected.item["question"]
    validate_gold_sql(projected.item["gold_sql"])

    attributes_row = _row(
        11,
        question=(
            "what is the name and age of the patient with subject id 12345?"
        ),
        tables=[0],
        conditions=[[0, 0, 0, "12345"]],
        select=[[0, 2], [0, 4]],
        aggregation=0,
    )
    attributes = project_mimicsql_row(
        attributes_row,
        domains,
        mimicsql_target_db,
        source_split="dev",
    )
    assert attributes.item["gold_sql"].startswith(
        "SELECT p.name, p.age FROM patients p"
    )
    assert "12345" not in attributes.item["question"]
    _, result_rows = execute_complete_readonly(
        attributes.item["gold_sql"],
        mimicsql_target_db,
    )
    assert len(result_rows) == 1


def test_mimicsql_projects_only_explicit_numeric_aggregations(
    mimicsql_target_db: Path,
):
    domains = load_target_domains(mimicsql_target_db)
    row = _row(
        20,
        question=(
            "what is the maximum age of patients diagnosed with "
            "Source disease?"
        ),
        tables=[0, 1],
        conditions=[[1, 3, 0, "Source disease"]],
        select=[[0, 4]],
        aggregation=2,
    )
    projected = project_mimicsql_row(
        row,
        domains,
        mimicsql_target_db,
        source_split="train",
    )
    assert projected.item["gold_sql"].startswith(
        "SELECT MAX(p.age) FROM patients p JOIN visits v"
    )

    unsupported = _row(
        21,
        question="what is the maximum diagnosis for Source disease?",
        tables=[1],
        conditions=[[1, 3, 0, "Source disease"]],
        select=[[1, 4]],
        aggregation=2,
    )
    with pytest.raises(
        MIMICSQLProjectionRejected,
        match="unsupported_numeric_aggregation",
    ):
        project_mimicsql_row(
            unsupported,
            domains,
            mimicsql_target_db,
            source_split="train",
        )


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (
            _row(
                30,
                question="how many patients prefer English?",
                tables=[0],
                conditions=[[0, 7, 0, "English"]],
            ),
            "unsupported_source_column",
        ),
        (
            _row(
                31,
                question="how many patients had a diagnosis?",
                tables=[0, 1],
                conditions=[[1, 3, 0, "Source disease"]],
            ),
            "source_literal_not_found_in_question",
        ),
        (
            _row(
                32,
                question="which patients were diagnosed with Source disease?",
                tables=[0, 1],
                conditions=[[1, 3, 0, "Source disease"]],
            ),
            "count_intent_not_grounded",
        ),
        (
            _row(
                33,
                question=(
                    "how many patients had Source primary disease and "
                    "Source diagnosis?"
                ),
                tables=[0, 1],
                conditions=[
                    [0, 16, 0, "Source primary disease"],
                    [1, 3, 0, "Source diagnosis"],
                ],
            ),
            "duplicate_target_condition_concept",
        ),
    ],
)
def test_mimicsql_projection_fails_closed(
    mimicsql_target_db: Path,
    row: dict,
    reason: str,
):
    domains = load_target_domains(mimicsql_target_db)
    with pytest.raises(MIMICSQLProjectionRejected) as captured:
        project_mimicsql_row(
            row,
            domains,
            mimicsql_target_db,
            source_split="train",
        )
    assert captured.value.reason == reason


def test_mimicsql_required_ast_role_is_enforced(
    mimicsql_target_db: Path,
):
    domains = load_target_domains(mimicsql_target_db)
    row = _row(
        40,
        question="how many patients were diagnosed with Source disease?",
        tables=[0, 1],
        conditions=[[1, 3, 0, "Source disease"]],
    )
    projected = project_mimicsql_row(
        row,
        domains,
        mimicsql_target_db,
        source_split="train",
    )
    opposite = (
        "holdout"
        if projected.skeleton_role == "development"
        else "development"
    )
    with pytest.raises(MIMICSQLProjectionRejected) as captured:
        project_mimicsql_row(
            row,
            domains,
            mimicsql_target_db,
            source_split="train",
            required_role=opposite,
        )
    assert captured.value.reason == "skeleton_reserved_for_other_role"


def test_mimicsql_projector_does_not_import_rule_engine():
    source = inspect.getsource(
        __import__(
            "dki.analysis.nl2sql_mimicsql_projection",
            fromlist=["unused"],
        )
    )
    assert "from dki.analysis.nl2sql import" not in source
    assert "import dki.analysis.nl2sql" not in source


def test_development_builder_never_opens_adjacent_official_test(
    tmp_path: Path,
    mimicsql_target_db: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source_dir = tmp_path / "benchmark" / "mimicsql_natural_v2"
    source_dir.mkdir(parents=True)
    train_path = source_dir / "train.json"
    dev_path = source_dir / "dev.json"
    official_test = source_dir / "test.json"
    train_rows = [
        {"key": f"{index:032x}", "question_refine": f"train {index}"}
        for index in range(100)
    ]
    dev_rows = [
        {
            "key": f"{index + 100:032x}",
            "question_refine": f"dev {index}",
        }
        for index in range(50)
    ]
    train_path.write_text(
        "".join(json.dumps(row) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    dev_path.write_text(
        "".join(json.dumps(row) + "\n" for row in dev_rows),
        encoding="utf-8",
    )
    official_test.write_text(
        '{"secret":"must never be opened"}\n',
        encoding="utf-8",
    )

    def fake_project(
        row,
        _domains,
        _db_path,
        *,
        source_split,
        source_revision,
    ):
        identity = str(row["key"])
        skeleton_sha = "a" * 64
        return ProjectedItem(
            item={
                "id": f"fake-{identity}",
                "question": f"development question {identity}",
                "gold_sql": (
                    "SELECT patient_id FROM patients WHERE age >= 1"
                ),
                "difficulty": "hard",
                "category": "filtering",
                "provenance": {
                    "benchmark": "MIMICSQL-natural-v2",
                    "source_split": source_split,
                    "source_revision": source_revision,
                },
            },
            skeleton_role="development",
            skeleton_sha256=skeleton_sha,
        )

    monkeypatch.setattr(builder, "project_mimicsql_row", fake_project)
    original_open = Path.open
    opened: list[Path] = []

    def guarded_open(path: Path, *args, **kwargs):
        resolved = path.resolve()
        opened.append(resolved)
        if resolved == official_test.resolve():
            raise AssertionError("development builder opened official test")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    private_output = (
        tmp_path / "outputs" / "private" / "rid" / "development.jsonl"
    )
    public_output = (
        tmp_path / "outputs" / "remote_runs" / "rid" / "receipt.json"
    )
    receipt = builder.build_mimicsql_development(
        train_path=train_path,
        dev_path=dev_path,
        db_path=mimicsql_target_db,
        private_output=private_output,
        public_output=public_output,
        source_tree_sha256="b" * 64,
        source_revision=builder.MIMICSQL_SOURCE_REVISION,
        official_test_git_blob="c" * 40,
        created_at="2026-07-26T13:30:00Z",
        minimum_items=150,
        maximum_items=150,
    )
    assert receipt["development_selected"]["n"] == 150
    assert (
        receipt["official_test"]["content_opened_or_hashed_by_this_builder"]
        is False
    )
    assert official_test.resolve() not in opened
    assert private_output.is_file()
    public_text = public_output.read_text(encoding="utf-8")
    assert "development question" not in public_text
    assert "SELECT patient_id" not in public_text

    with pytest.raises(FileExistsError):
        builder.build_mimicsql_development(
            train_path=train_path,
            dev_path=dev_path,
            db_path=mimicsql_target_db,
            private_output=private_output,
            public_output=public_output,
            source_tree_sha256="b" * 64,
            source_revision=builder.MIMICSQL_SOURCE_REVISION,
            official_test_git_blob="c" * 40,
            created_at="2026-07-26T13:30:00Z",
            minimum_items=150,
            maximum_items=150,
        )


def test_development_builder_has_no_official_test_path_parameter():
    parameters = inspect.signature(
        builder.build_mimicsql_development
    ).parameters
    assert "test_path" not in parameters
    assert "holdout_path" not in parameters
