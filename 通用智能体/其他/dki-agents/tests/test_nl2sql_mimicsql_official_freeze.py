from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dki.analysis.nl2sql_css_projection import ProjectedItem
from dki.analysis.nl2sql_mimicsql_projection import (
    MIMICSQL_SOURCE_REVISION,
    MIMICSQL_TEST_GIT_BLOB,
)
from dki.analysis.nl2sql_protocol import validate_gold_sql
from scripts import freeze_nl2sql_mimicsql_official_v3 as freezer


RID = "20260726T140000Z-g1234567-t" + "a" * 12


def _touch(path: Path, content: bytes = b"bound\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _context(tmp_path: Path) -> dict:
    private_root = (
        tmp_path
        / "outputs"
        / "private"
        / RID
        / "nl2sql_mimicsql_official_v3"
    )
    remote_root = tmp_path / "outputs" / "remote_runs" / RID
    development = _touch(
        private_root.parent
        / "nl2sql_mimicsql_v3"
        / "development.jsonl",
        b"".join(b'{"synthetic":true}\n' for _ in range(150)),
    )
    development_receipt = _touch(
        remote_root / "nl2sql" / "mimicsql_development_receipt.json"
    )
    source_tree = _touch(
        remote_root / "source" / "included_tree.json"
    )
    task1 = _touch(tmp_path / "lineage" / "manifest.json")
    kg = _touch(tmp_path / "lineage" / "medical_kg.json")
    db = _touch(tmp_path / "lineage" / "four_table.sqlite")
    official_test = _touch(
        tmp_path
        / "outputs"
        / "private"
        / "source"
        / freezer.TEST_RELATIVE_PATH,
        b'{"key":"synthetic"}\n',
    )
    output_manifest = private_root / "frozen" / "freeze_manifest.json"
    public_output = (
        remote_root
        / "nl2sql"
        / "mimicsql_official_freeze_attestation.json"
    )
    return {
        "rid": RID,
        "created_at": "2026-07-26T14:00:00Z",
        "control_root": private_root,
        "final_dir": output_manifest.parent,
        "staging_dir": (
            output_manifest.parent.parent
            / ".frozen.freeze-staging"
        ),
        "started_marker": private_root / "projection-started.json",
        "completed_marker": private_root / "projection-completed.json",
        "invalid_marker": private_root / "projection-invalid.json",
        "output_manifest": output_manifest,
        "public_output": public_output,
        "source_tree_manifest": source_tree,
        "source_tree_sha256": "a" * 64,
        "task1_manifest": task1,
        "kg_path": kg,
        "db_path": db,
        "development_path": development,
        "development_receipt": development_receipt,
        "development_rows": [{} for _ in range(150)],
        "development_questions": frozenset(),
        "development_skeletons": frozenset(),
        "official_test": {
            "path": official_test,
            "size_before_marker": official_test.stat().st_size,
            "git_blob_sha1": MIMICSQL_TEST_GIT_BLOB,
        },
        "source_revision": MIMICSQL_SOURCE_REVISION,
        "official_test_git_blob": MIMICSQL_TEST_GIT_BLOB,
    }


def _kwargs(context: dict) -> dict:
    return {
        "source_repo": context["official_test"]["path"].parents[3],
        "db_path": context["db_path"],
        "task1_manifest": context["task1_manifest"],
        "kg_path": context["kg_path"],
        "development_path": context["development_path"],
        "development_receipt": context["development_receipt"],
        "source_tree_manifest": context["source_tree_manifest"],
        "output_manifest": context["output_manifest"],
        "public_output": context["public_output"],
        "created_at": context["created_at"],
        "source_revision": context["source_revision"],
        "official_test_git_blob": context["official_test_git_blob"],
    }


def _holdout_items() -> list[ProjectedItem]:
    items: list[ProjectedItem] = []
    for index in range(120):
        sql = (
            "SELECT patient_id FROM patients "
            f"WHERE age >= {index + 1}"
        )
        skeleton = validate_gold_sql(sql)
        skeleton_sha256 = hashlib.sha256(
            skeleton.encode("utf-8")
        ).hexdigest()
        items.append(
            ProjectedItem(
                item={
                    "id": f"mimicsql-test-{index:04d}",
                    "question": f"private official question {index}",
                    "gold_sql": sql,
                    "difficulty": "hard" if index < 40 else "medium",
                    "category": "filtering",
                    "provenance": {
                        "benchmark": "MIMICSQL-natural-v2",
                        "source_split": "test",
                        "skeleton_role": "holdout",
                        "skeleton_sha256": skeleton_sha256,
                    },
                },
                skeleton_role="holdout",
                skeleton_sha256=skeleton_sha256,
            )
        )
    return items


def _patch_success(monkeypatch, context: dict) -> None:
    monkeypatch.setattr(freezer, "_preflight_freeze", lambda **_: context)
    monkeypatch.setattr(
        freezer,
        "_verify_nonholdout_stability",
        lambda _context: None,
    )

    def read_after_marker(*_args, **_kwargs):
        assert context["started_marker"].is_file()
        return (
            [{"synthetic": True}],
            {
                "git_blob_sha1": MIMICSQL_TEST_GIT_BLOB,
                "sha256": "1" * 64,
                "size": context["official_test"]["size_before_marker"],
                "n_source_rows": 1,
            },
        )

    monkeypatch.setattr(
        freezer,
        "_read_official_test_after_marker",
        read_after_marker,
    )
    items = _holdout_items()
    monkeypatch.setattr(
        freezer,
        "_project_holdout",
        lambda *_args, **_kwargs: (
            items,
            {
                "source_attempts": 1,
                "rejection_counts": {},
                "n_before_deduplication": 120,
                "n_after_deduplication": 120,
                "deduplication": {
                    "exact_duplicate_rows_removed": 0,
                    "ambiguous_questions_removed": 0,
                    "ambiguous_rows_removed": 0,
                },
            },
        ),
    )


def test_official_reader_refuses_before_irreversible_marker(
    tmp_path: Path,
) -> None:
    source = _touch(tmp_path / "test.json", b'{"key":"one"}\n')
    marker = tmp_path / "projection-started.json"
    data = source.read_bytes()
    with pytest.raises(RuntimeError, match="before projection marker"):
        freezer._read_official_test_after_marker(
            source,
            marker=marker,
            expected_git_blob=freezer._git_blob_sha1(data),
            expected_size=len(data),
        )
    _touch(marker)
    rows, binding = freezer._read_official_test_after_marker(
        source,
        marker=marker,
        expected_git_blob=freezer._git_blob_sha1(data),
        expected_size=len(data),
    )
    assert rows == [{"key": "one"}]
    assert binding["sha256"] == hashlib.sha256(data).hexdigest()


def test_git_metadata_preflight_does_not_open_official_test(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "outputs" / "private" / "source"
    test_path = _touch(
        repo / freezer.TEST_RELATIVE_PATH,
        b"official bytes must stay closed",
    )

    def fake_git(_repo: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return MIMICSQL_SOURCE_REVISION
        return (
            "100644 blob "
            f"{MIMICSQL_TEST_GIT_BLOB}\t"
            f"{freezer.TEST_RELATIVE_PATH.as_posix()}"
        )

    monkeypatch.setattr(freezer, "_run_git", fake_git)
    original_read_bytes = Path.read_bytes

    def refuse_test_read(path: Path) -> bytes:
        if path.resolve() == test_path.resolve():
            raise AssertionError("preflight opened official test")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", refuse_test_read)
    metadata = freezer._official_test_metadata(
        repo,
        source_revision=MIMICSQL_SOURCE_REVISION,
        official_test_git_blob=MIMICSQL_TEST_GIT_BLOB,
    )
    assert metadata["size_before_marker"] == test_path.stat().st_size


def test_preflight_failure_creates_no_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(
        freezer,
        "_preflight_freeze",
        lambda **_: (_ for _ in ()).throw(ValueError("preflight failed")),
    )
    with pytest.raises(ValueError, match="preflight failed"):
        freezer.freeze_mimicsql_official_v3(**_kwargs(context))
    assert not context["started_marker"].exists()
    assert not context["invalid_marker"].exists()


def test_success_is_create_once_and_public_receipt_is_aggregate_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    _patch_success(monkeypatch, context)
    receipt = freezer.freeze_mimicsql_official_v3(**_kwargs(context))
    manifest = json.loads(
        context["output_manifest"].read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 4
    assert manifest["kind"] == "mimicsql_v3_four_table_blind_freeze"
    assert manifest["n_holdout"] == 120
    assert manifest["n_hard"] == 40
    assert manifest["selection_used"] is False
    assert receipt["counts"]["n_holdout"] == 120
    assert context["completed_marker"].is_file()
    assert not context["invalid_marker"].exists()
    public_text = context["public_output"].read_text(encoding="utf-8")
    assert "private official question" not in public_text
    assert '"gold_sql"' not in public_text
    with pytest.raises(FileExistsError):
        freezer.freeze_mimicsql_official_v3(**_kwargs(context))


def test_post_marker_failure_permanently_invalidates_rid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    monkeypatch.setattr(freezer, "_preflight_freeze", lambda **_: context)
    monkeypatch.setattr(
        freezer,
        "_after_started_marker",
        lambda _context: (_ for _ in ()).throw(
            RuntimeError("injected post-marker failure")
        ),
    )
    with pytest.raises(RuntimeError, match="injected post-marker"):
        freezer.freeze_mimicsql_official_v3(**_kwargs(context))
    invalid = json.loads(
        context["invalid_marker"].read_text(encoding="utf-8")
    )
    assert invalid["status"] == "projection-invalid"
    assert invalid["failure_class"] == "RuntimeError"
    assert "injected post-marker" not in context[
        "invalid_marker"
    ].read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        freezer.freeze_mimicsql_official_v3(**_kwargs(context))
