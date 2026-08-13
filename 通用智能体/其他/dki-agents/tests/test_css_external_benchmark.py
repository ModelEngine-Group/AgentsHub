import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_css_external_benchmark.py"
SPEC = importlib.util.spec_from_file_location(
    "freeze_css_external_benchmark", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

ATTEST_SCRIPT = ROOT / "scripts" / "attest_css_external_benchmark.py"
ATTEST_SPEC = importlib.util.spec_from_file_location(
    "attest_css_external_benchmark", ATTEST_SCRIPT
)
ATTEST_MODULE = importlib.util.module_from_spec(ATTEST_SPEC)
assert ATTEST_SPEC.loader is not None
ATTEST_SPEC.loader.exec_module(ATTEST_MODULE)

AUDIT_SCRIPT = ROOT / "scripts" / "audit_css_development.py"
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_css_development", AUDIT_SCRIPT
)
AUDIT_MODULE = importlib.util.module_from_spec(AUDIT_SPEC)
assert AUDIT_SPEC.loader is not None
AUDIT_SPEC.loader.exec_module(AUDIT_MODULE)


def _row(split: str, db_id: str) -> dict:
    return {
        "query": f"SELECT '{split}'",
        "db_id": db_id,
        "question": f"{split} question",
        "question_id": f"{split}-1",
    }


def _archive(path: Path, *, overlap: bool = False) -> str:
    ids = {
        "train": "db-train",
        "dev": "db-dev",
        "test": "db-train" if overlap else "db-test",
    }
    schemas = [
        {"db_id": db_id, "table_names_original": [], "column_names_original": []}
        for db_id in sorted(set(ids.values()))
    ]
    content = [
        {"db_id": db_id, "tables": {}}
        for db_id in sorted(set(ids.values()))
    ]
    members = {
        f"css/schema/{split}.json": json.dumps(
            [_row(split, db_id)], ensure_ascii=False
        )
        for split, db_id in ids.items()
    }
    members["css/tables.json"] = json.dumps(schemas, ensure_ascii=False)
    members["css/db_content.json"] = json.dumps(content, ensure_ascii=False)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_css_freeze_keeps_official_test_private_and_create_once(tmp_path):
    archive = tmp_path / "css.zip"
    digest = _archive(archive)
    output = tmp_path / "outputs" / "private" / "rid" / "css"
    payload = MODULE.freeze(
        archive,
        output,
        source_revision="a" * 40,
        source_url="https://example.invalid/css",
        expected_archive_sha256=digest,
    )
    assert payload["counts"] == {
        "train": 1,
        "dev": 1,
        "test": 1,
        "schemas": 3,
        "train_schemas": 1,
        "dev_schemas": 1,
        "test_schemas": 1,
        "duplicate_question_rows": {
            "train": 0,
            "dev": 0,
            "test": 0,
        },
    }
    assert payload["database_id_intersections"] == {
        "train_dev": [],
        "train_test": [],
        "dev_test": [],
    }
    assert (output / "holdout" / "test.json").is_file()
    manifest = (output / "freeze_manifest.json").read_text(encoding="utf-8")
    assert "test question" not in manifest
    assert "SELECT" not in manifest
    public = tmp_path / "public" / "css.json"
    attestation = ATTEST_MODULE.attest(
        output / "freeze_manifest.json",
        public,
        source_tree_sha256="d" * 64,
    )
    assert attestation["four_table_blind_gate"] is False
    assert attestation["holdout_consumed"] is False
    public_text = public.read_text(encoding="utf-8")
    assert "test question" not in public_text
    assert "SELECT" not in public_text
    with pytest.raises(FileExistsError):
        ATTEST_MODULE.attest(
            output / "freeze_manifest.json",
            public,
            source_tree_sha256="d" * 64,
        )
    audit_output = tmp_path / "public" / "css-development.json"
    audit = AUDIT_MODULE.audit(
        output / "freeze_manifest.json",
        audit_output,
        source_tree_sha256="e" * 64,
    )
    assert audit["holdout"] == {
        "content_parsed": False,
        "hash_verified": True,
        "consumed": False,
        "sha256": payload["files"]["holdout/test.json"]["sha256"],
    }
    assert audit["train"]["rows"] == 1
    assert audit["dev"]["rows"] == 1
    with pytest.raises(FileExistsError):
        MODULE.freeze(
            archive,
            output,
            source_revision="a" * 40,
            source_url="https://example.invalid/css",
            expected_archive_sha256=digest,
        )


def test_css_freeze_rejects_schema_overlap(tmp_path):
    archive = tmp_path / "css.zip"
    digest = _archive(archive, overlap=True)
    with pytest.raises(ValueError, match="leaks database ids"):
        MODULE.freeze(
            archive,
            tmp_path / "out",
            source_revision="a" * 40,
            source_url="https://example.invalid/css",
            expected_archive_sha256=digest,
        )


def test_css_archive_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.json", "{}")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="unsafe"):
            MODULE._safe_members(archive)
