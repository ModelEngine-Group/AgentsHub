import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_ehrsql_public_repository.py"
SPEC = importlib.util.spec_from_file_location("audit_ehrsql_public_repository", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _checkout(tmp_path):
    root = tmp_path / "ehrsql"
    (root / "dataset" / "ehrsql" / "mimic_iii").mkdir(parents=True)
    (root / "dataset" / "ehrsql" / "eicu").mkdir(parents=True)
    (root / "README.md").write_text("EHRSQL\n", encoding="utf-8")
    (root / "LICENSE").write_text("CC-BY-4.0\n", encoding="utf-8")
    (root / "dataset" / "ehrsql" / "CHANGELOG.md").write_text(
        "v1.5.1\n",
        encoding="utf-8",
    )
    (root / "dataset" / "ehrsql" / "tables.json").write_text(
        "[]\n",
        encoding="utf-8",
    )
    counter = 0
    for database in MODULE.DATASETS:
        for split in MODULE.SPLITS:
            counter += 1
            rows = [
                {
                    "id": f"id-{counter}",
                    "db_id": database,
                    "question": f"How many rows are in patients for {counter}?",
                    "query": "SELECT COUNT(*) FROM patients",
                    "is_impossible": False,
                    "split": split,
                },
                {
                    "id": f"id-{counter}-impossible",
                    "db_id": database,
                    "question": f"Unsupported question {counter}?",
                    "query": "",
                    "is_impossible": True,
                    "split": split,
                },
            ]
            (root / "dataset" / "ehrsql" / database / f"{split}.json").write_text(
                json.dumps(rows),
                encoding="utf-8",
            )
    return root


def test_public_repository_audit_is_aggregate_only(
    tmp_path,
    monkeypatch,
):
    checkout = _checkout(tmp_path)
    monkeypatch.setattr(MODULE, "_git_head", lambda path: "a" * 40)
    report = MODULE.audit(
        checkout=checkout,
        source_revision="a" * 40,
        source_tree_sha256="b" * 64,
    )
    assert report["benchmark"] == "EHRSQL-v1.5.1"
    assert report["totals"] == {
        "rows": 12,
        "answerable": 6,
        "unanswerable": 6,
        "unique_ids": 12,
        "parse_failures": 0,
        "unsafe_statements": 0,
    }
    assert report["contains_questions_or_sql"] is False
    assert report["four_table_blind_gate"] is False
    serialized = json.dumps(report)
    assert "How many rows" not in serialized
    assert "SELECT COUNT" not in serialized


def test_public_repository_audit_rejects_revision_mismatch(
    tmp_path,
    monkeypatch,
):
    checkout = _checkout(tmp_path)
    monkeypatch.setattr(MODULE, "_git_head", lambda path: "c" * 40)
    with pytest.raises(ValueError, match="revision"):
        MODULE.audit(
            checkout=checkout,
            source_revision="a" * 40,
            source_tree_sha256="b" * 64,
        )
