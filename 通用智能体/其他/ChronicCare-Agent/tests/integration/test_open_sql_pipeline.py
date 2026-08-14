from __future__ import annotations

import sqlite3

from analysis.open_sql import ast_guard, sql_executor
from analysis.open_sql.nl_security import classify_nl_security
from analysis.open_sql.question_rewriter import rewrite_question
from analysis.open_sql.schema_catalog import allowed_fields, allowed_tables
from analysis.open_sql.schema_linker import build_schema_links
from analysis.open_sql.sql_template_builder import build_template_sql


def _patch_catalog(monkeypatch, catalog: dict, db_path) -> None:
    monkeypatch.setattr(ast_guard, "get_schema_catalog", lambda: catalog)
    monkeypatch.setattr(ast_guard, "allowed_tables", allowed_tables)
    monkeypatch.setattr(ast_guard, "allowed_fields", allowed_fields)
    monkeypatch.setattr(sql_executor, "get_schema_catalog", lambda: catalog)
    monkeypatch.setattr(sql_executor, "allowed_tables", allowed_tables)
    monkeypatch.setattr(sql_executor, "allowed_fields", allowed_fields)
    monkeypatch.setattr(sql_executor, "db_path", lambda: db_path)


def test_open_sql_question_to_guarded_execution(
    monkeypatch,
    open_sql_cases,
    sample_schema_catalog,
    sample_sqlite_db,
) -> None:
    _patch_catalog(monkeypatch, sample_schema_catalog, sample_sqlite_db)
    case = next(item for item in open_sql_cases if item["id"] == "count-hypertension")
    assert classify_nl_security(case["question"])["safe"] is True
    spec = rewrite_question(case["question"])
    assert spec["intent"] == case["intent"]
    links = build_schema_links(spec, sample_schema_catalog)
    assert links["status"] == "success"
    template = build_template_sql(spec, links)
    guard = ast_guard.validate_sql(template["sql"])
    assert guard["safe"] is True
    assert guard["errors"] == []
    assert guard["normalized_sql"] is not None
    assert guard["warnings"]
    execution = sql_executor.execute_sql(guard["normalized_sql"])
    assert execution["status"] == "success"
    assert execution["rows"][0]["patient_count"] == case["expected_patient_count"]


def test_open_sql_followup_window_uses_fixture_database(
    monkeypatch,
    open_sql_cases,
    sample_schema_catalog,
    sample_sqlite_db,
) -> None:
    _patch_catalog(monkeypatch, sample_schema_catalog, sample_sqlite_db)
    case = next(item for item in open_sql_cases if item["id"] == "followup-high-risk-7d")
    spec = rewrite_question(case["question"])
    spec["analysis_context"] = {"as_of_date": case["as_of_date"]}
    links = build_schema_links(spec, sample_schema_catalog)
    template = build_template_sql(spec, links)
    guard = ast_guard.validate_sql(template["sql"])
    assert guard["safe"] is True
    assert guard["errors"] == []
    execution = sql_executor.execute_sql(guard["normalized_sql"])
    assert execution["status"] == "success"
    assert execution["rows"][0]["patient_count"] == case["expected_patient_count"]
    assert execution["rows"][0]["plan_count"] == 2


def test_natural_language_security_stops_pipeline_before_sql(open_sql_cases) -> None:
    case = next(item for item in open_sql_cases if item["id"] == "unsafe-delete")
    security = classify_nl_security(case["question"])
    assert security["safe"] is False
    assert security["code"] == "NL_SECURITY_POLICY_REJECTED"


def test_sample_database_contains_expected_fixture_rows(sample_sqlite_db) -> None:
    with sqlite3.connect(sample_sqlite_db) as connection:
        patients = connection.execute("SELECT COUNT(*) FROM patient_profile").fetchone()[0]
        followups = connection.execute("SELECT COUNT(*) FROM followup_plan").fetchone()[0]
        labs = connection.execute("SELECT COUNT(*) FROM lab_result").fetchone()[0]
    assert (patients, followups, labs) == (4, 4, 4)
