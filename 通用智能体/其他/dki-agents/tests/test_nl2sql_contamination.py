"""Contamination audit: fingerprints, disjointness, closure scan, CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dki.analysis.nl2sql_contamination import (
    CONTAMINATION_AUDIT_VERSION,
    audit_contamination,
    canonical_report_sha256,
    fingerprint_item,
    questions_are_near_duplicates,
    scan_runtime_closure,
    sql_ast_skeleton,
)

ROOT = Path(__file__).resolve().parent.parent

_SQL_A = "SELECT COUNT(*) FROM visits WHERE department = '心血管内科'"
_SQL_B = "SELECT COUNT(*) FROM visits WHERE department = '内分泌科'"
_SQL_C = "SELECT AVG(age) FROM patients WHERE region = '北京'"


def _item(item_id: str, question: str, sql: str) -> dict:
    return {"id": item_id, "question": question, "gold_sql": sql}


def test_fingerprint_normalizes_nfkc_width_variants():
    left = fingerprint_item(_item("a", "统计心血管内科的就诊数量？", _SQL_A))
    right = fingerprint_item(_item("b", "统计心血管内科的就诊数量?", _SQL_A))
    assert left.question_sha256 == right.question_sha256


def test_ast_skeleton_ignores_literals_only():
    assert sql_ast_skeleton(_SQL_A) == sql_ast_skeleton(_SQL_B)
    assert sql_ast_skeleton(_SQL_A) != sql_ast_skeleton(_SQL_C)


def test_near_duplicate_contract_small_edit():
    assert questions_are_near_duplicates(
        "统计每个科室的就诊次数并按次数排序",
        "统计每个科室的就诊次数并按照次数排序",
    )
    assert not questions_are_near_duplicates("统计科室就诊量", "查询北京患者平均年龄")


def test_exact_duplicate_is_disallowed():
    report = audit_contamination(
        [_item("c1", "统计心血管内科的就诊数量", _SQL_A)],
        [_item("h1", "统计心血管内科的就诊数量", _SQL_A)],
    )
    assert not report.clean
    assert report.hit_counts["exact_question"] == 1
    assert ("c1", "h1") in report.private_hits["exact_question"]


def test_ast_skeleton_overlap_is_disallowed_even_with_new_question():
    report = audit_contamination(
        [_item("c1", "内分泌科一共有多少次就诊", _SQL_B)],
        [_item("h1", "统计心血管内科的就诊数量", _SQL_A)],
    )
    assert not report.clean
    assert report.hit_counts["ast_skeleton"] == 1
    # Same-skeleton items also collapse into one transitive semantic family.
    assert report.hit_counts["semantic_family"] == 1


def test_disjoint_sets_are_clean():
    report = audit_contamination(
        [_item("c1", "查询北京患者的平均年龄", _SQL_C)],
        [_item("h1", "统计心血管内科的就诊数量", _SQL_A)],
    )
    assert report.clean
    assert all(
        report.hit_counts[category] == 0
        for category in (
            "exact_question",
            "near_question",
            "sql_exact",
            "ast_skeleton",
            "semantic_family",
        )
    )


def test_join_graph_overlap_is_advisory_not_blocking():
    report = audit_contamination(
        [_item("c1", "查询费用大于1000的就诊数量", "SELECT COUNT(*) FROM visits WHERE cost > 1000")],
        [_item("h1", "统计各科室平均费用", "SELECT department, AVG(cost) FROM visits GROUP BY department")],
    )
    assert report.clean
    assert report.hit_counts["join_graph"] == 1


def test_empty_inputs_fail_closed():
    with pytest.raises(ValueError, match="candidate_items"):
        audit_contamination([], [_item("h1", "问题", _SQL_A)])
    with pytest.raises(ValueError, match="historical_items"):
        audit_contamination([_item("c1", "问题", _SQL_A)], [])


def test_malformed_records_raise():
    with pytest.raises(ValueError, match="missing fields"):
        audit_contamination(
            [{"id": "c1", "question": "问题"}],
            [_item("h1", "历史问题", _SQL_A)],
        )
    with pytest.raises(ValueError, match="non-empty"):
        fingerprint_item(_item("c1", "  ", _SQL_A))
    with pytest.raises(ValueError):
        fingerprint_item(_item("c1", "问题", "DROP TABLE visits"))


def test_duplicate_ids_raise():
    with pytest.raises(ValueError, match="duplicate candidate"):
        audit_contamination(
            [
                _item("c1", "查询北京患者的平均年龄", _SQL_C),
                _item("c1", "统计各地区患者人数", "SELECT region, COUNT(*) FROM patients GROUP BY region"),
            ],
            [_item("h1", "统计心血管内科的就诊数量", _SQL_A)],
        )


def test_public_attestation_contains_no_text():
    question = "统计心血管内科的就诊数量"
    report = audit_contamination(
        [_item("c1", question, _SQL_A)],
        [_item("h1", question, _SQL_A)],
    )
    rendered = json.dumps(report.public_attestation(), ensure_ascii=False)
    assert question not in rendered
    assert "SELECT" not in rendered
    assert "c1" not in rendered and "h1" not in rendered
    assert report.public_attestation()["clean"] is False


def test_report_hash_is_deterministic():
    report = audit_contamination(
        [_item("c1", "查询北京患者的平均年龄", _SQL_C)],
        [_item("h1", "统计心血管内科的就诊数量", _SQL_A)],
    )
    first = report.public_attestation()
    second = report.public_attestation()
    assert first["report_sha256"] == second["report_sha256"]
    body = {key: value for key, value in first.items() if key != "report_sha256"}
    assert first["report_sha256"] == canonical_report_sha256(body)


def test_closure_scan_flags_forbidden_literal(tmp_path):
    package = tmp_path / "dki" / "analysis"
    package.mkdir(parents=True)
    (tmp_path / "dki" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "nl2sql.py").write_text(
        'from dki.analysis import helper\n', encoding="utf-8"
    )
    (package / "helper.py").write_text(
        'PATH = "_v28_local/history.json"\n', encoding="utf-8"
    )
    report = scan_runtime_closure(tmp_path)
    assert not report.clean
    assert report.n_modules >= 2
    assert any(finding.fragment == "_v28_local" for finding in report.findings)
    summary = report.public_summary()
    assert summary["n_findings"] == len(report.findings)
    assert "history.json" not in json.dumps(summary)


def test_closure_scan_prose_docstring_does_not_fire(tmp_path):
    package = tmp_path / "dki" / "analysis"
    package.mkdir(parents=True)
    (tmp_path / "dki" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    # Prose in docstrings mentions "plans"/"holdout" as words, not as path
    # segments; only path-shaped literals may fire.
    (package / "nl2sql.py").write_text(
        '"""Hard-gate model plans and holdout hygiene notes in prose."""\n'
        'LABEL = "plansettings"\n'
        "VALUE = 1\n",
        encoding="utf-8",
    )
    report = scan_runtime_closure(tmp_path)
    assert report.clean


def test_production_runtime_closure_is_clean():
    report = scan_runtime_closure(ROOT)
    offending = [
        finding
        for finding in report.findings
    ]
    assert report.clean, offending


def test_cli_end_to_end_clean_and_contaminated(tmp_path):
    candidate = tmp_path / "candidate.jsonl"
    historical = tmp_path / "historical.jsonl"
    private_out = tmp_path / "private.json"
    public_out = tmp_path / "public.json"
    candidate.write_text(
        json.dumps(_item("c1", "查询北京患者的平均年龄", _SQL_C), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    historical.write_text(
        json.dumps(_item("h1", "统计心血管内科的就诊数量", _SQL_A), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "audit_nl2sql_contamination.py"),
        "--candidate-jsonl",
        str(candidate),
        "--historical-jsonl",
        str(historical),
        "--repo-root",
        str(ROOT),
        "--private-report-out",
        str(private_out),
        "--public-attestation-out",
        str(public_out),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    public = json.loads(public_out.read_text(encoding="utf-8"))
    assert public["audit"]["clean"] is True
    assert public["audit"]["version"] == CONTAMINATION_AUDIT_VERSION

    # Contaminated: same candidate as history → nonzero exit.
    candidate.write_text(
        json.dumps(_item("c1", "统计心血管内科的就诊数量", _SQL_A), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 1, completed.stdout
    private = json.loads(private_out.read_text(encoding="utf-8"))
    assert private["audit"]["hits"]["exact_question"] == [["c1", "h1"]]
