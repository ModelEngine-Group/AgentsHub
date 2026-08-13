import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FREEZE_SCRIPT = ROOT / "scripts" / "freeze_ehrsql_public_source.py"
FREEZE_SPEC = importlib.util.spec_from_file_location(
    "freeze_ehrsql_public_source",
    FREEZE_SCRIPT,
)
FREEZE = importlib.util.module_from_spec(FREEZE_SPEC)
assert FREEZE_SPEC.loader is not None
FREEZE_SPEC.loader.exec_module(FREEZE)

AUDIT_SCRIPT = ROOT / "scripts" / "audit_ehrsql_public_source.py"
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_ehrsql_public_source",
    AUDIT_SCRIPT,
)
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
assert AUDIT_SPEC.loader is not None
AUDIT_SPEC.loader.exec_module(AUDIT)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _checkout(path: Path) -> None:
    database = path / "data" / "mimic_iv" / "mimic_iv.sqlite"
    database.parent.mkdir(parents=True)
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE visits (
                patient_id INTEGER,
                department TEXT,
                cost REAL
            );
            INSERT INTO visits VALUES (1, 'cardiology', 100.0);
            INSERT INTO visits VALUES (2, 'neurology', 200.0);
            """
        )
        connection.commit()
    finally:
        connection.close()
    _write_json(
        path / "data" / "mimic_iv" / "tables.json",
        [
            {
                "db_id": "mimic_iv",
                "table_names_original": ["visits"],
                "column_names_original": [
                    [-1, "*"],
                    [0, "patient_id"],
                    [0, "department"],
                    [0, "cost"],
                ],
            }
        ],
    )
    (path / "LICENSE").write_text(
        "Creative Commons Attribution 4.0 International\n",
        encoding="utf-8",
    )
    (path / "README.md").write_text("EHRSQL 2024 fixture\n", encoding="utf-8")
    for split in ("train", "valid", "test"):
        answer_id = f"{split}-answer"
        abstain_id = f"{split}-abstain"
        question = f"{split} average cost by department"
        sql = (
            "SELECT department, AVG(cost) "
            "FROM visits GROUP BY department ORDER BY AVG(cost) DESC"
        )
        base = path / "data" / "mimic_iv" / split
        _write_json(
            base / "data.json",
            {
                "version": "fixture",
                "data": [
                    {"id": answer_id, "question": question},
                    {
                        "id": abstain_id,
                        "question": f"{split} unsupported clinical advice",
                    },
                ],
            },
        )
        _write_json(
            base / "label.json",
            {answer_id: sql, abstain_id: None},
        )
        _write_json(
            base / "annotated.json",
            [
                {
                    "id": answer_id,
                    "question": question,
                    "query": sql,
                    "template": "average cost by department",
                    "val_dict": {},
                },
                {
                    "id": abstain_id,
                    "question": f"{split} unsupported clinical advice",
                    "query": None,
                    "template": None,
                    "val_dict": {},
                },
            ],
        )


def test_ehrsql_freeze_and_aggregate_audit_are_create_once(
    tmp_path: Path,
    monkeypatch,
):
    checkout = tmp_path / "checkout"
    _checkout(checkout)
    revision = "a" * 40
    monkeypatch.setattr(FREEZE, "_git_head", lambda _: revision)
    output = tmp_path / "outputs" / "private" / "rid" / "ehrsql"
    manifest = FREEZE.freeze(
        checkout,
        output,
        source_revision=revision,
        source_url="https://github.com/glee4810/ehrsql-2024",
    )
    assert manifest["benchmark"] == "EHRSQL-2024-MIMIC-IV-Demo"
    assert manifest["dataset_role"] == "public_authoritative_curation_source"
    assert manifest["counts"]["test"] == {
        "rows": 2,
        "answerable": 1,
        "unanswerable": 1,
    }
    manifest_text = (output / "freeze_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "average cost" not in manifest_text
    assert "SELECT" not in manifest_text
    with pytest.raises(FileExistsError):
        FREEZE.freeze(
            checkout,
            output,
            source_revision=revision,
            source_url="https://github.com/glee4810/ehrsql-2024",
        )

    public = tmp_path / "outputs" / "remote_runs" / "rid" / "ehrsql.json"
    report = AUDIT.audit(
        output / "freeze_manifest.json",
        public,
        source_tree_sha256="b" * 64,
    )
    assert report["four_table_blind_gate"] is False
    assert report["upstream_test_is_public"] is True
    assert report["splits"]["test"]["rows"] == 2
    assert report["splits"]["test"]["answerable"] == 1
    assert report["splits"]["test"]["parse_failures"] == 0
    assert report["splits"]["test"]["explain_failures"] == 0
    assert report["splits"]["test"]["derived_difficulty"]["hard"] == 1
    public_text = public.read_text(encoding="utf-8")
    assert "average cost" not in public_text
    assert "SELECT" not in public_text
    with pytest.raises(FileExistsError):
        AUDIT.audit(
            output / "freeze_manifest.json",
            public,
            source_tree_sha256="b" * 64,
        )


def test_ehrsql_audit_rejects_tampered_frozen_file(
    tmp_path: Path,
    monkeypatch,
):
    checkout = tmp_path / "checkout"
    _checkout(checkout)
    revision = "c" * 40
    monkeypatch.setattr(FREEZE, "_git_head", lambda _: revision)
    output = tmp_path / "outputs" / "private" / "rid" / "ehrsql"
    FREEZE.freeze(
        checkout,
        output,
        source_revision=revision,
        source_url="https://github.com/glee4810/ehrsql-2024",
    )
    label = output / "source" / "data" / "mimic_iv" / "test" / "label.json"
    label.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="binding failed"):
        AUDIT.audit(
            output / "freeze_manifest.json",
            tmp_path / "public.json",
            source_tree_sha256="d" * 64,
        )


def test_ehrsql_freeze_rejects_mismatched_source_revision(
    tmp_path: Path,
    monkeypatch,
):
    checkout = tmp_path / "checkout"
    _checkout(checkout)
    monkeypatch.setattr(FREEZE, "_git_head", lambda _: "e" * 40)
    with pytest.raises(ValueError, match="revision"):
        FREEZE.freeze(
            checkout,
            tmp_path / "out",
            source_revision="f" * 40,
            source_url="https://github.com/glee4810/ehrsql-2024",
        )
