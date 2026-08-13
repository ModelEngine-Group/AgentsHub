import importlib.util
import json
from pathlib import Path

import pytest

from dki.analysis.nl2sql_css_projection import ProjectedItem


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "project_nl2sql_official_v2_development.py"
SPEC = importlib.util.spec_from_file_location(
    "project_nl2sql_official_v2_development",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "sha256": MODULE._sha256(path),
        "size": path.stat().st_size,
    }


def test_source_loaders_never_touch_official_test_files(tmp_path):
    css_root = tmp_path / "css"
    css_files = {
        "development/train.json": _write_json(
            css_root / "development" / "train.json",
            [{"db_id": "css-db"}],
        ),
        "development/dev.json": _write_json(
            css_root / "development" / "dev.json",
            [{"db_id": "css-db"}],
        ),
        "metadata/tables.json": _write_json(
            css_root / "metadata" / "tables.json",
            [],
        ),
        # The file deliberately does not exist.  Its previously frozen hash is
        # sufficient until the one-shot holdout freezer is invoked.
        "holdout/test.json": {"sha256": "1" * 64, "size": 123},
    }
    css_manifest = css_root / "freeze_manifest.json"
    css_manifest.write_text(
        json.dumps(
            {
                "benchmark": "CSS",
                "official_split": "schema",
                "holdout_consumed": False,
                "files": css_files,
            }
        ),
        encoding="utf-8",
    )
    _, _, css_rows = MODULE._load_css_development(css_manifest)
    assert [split for split, _ in css_rows] == ["train", "dev"]

    ehr_root = tmp_path / "ehr"
    ehr_files = {}
    for database in MODULE.EHR_DATABASES:
        for split in MODULE.EHR_DEVELOPMENT_SPLITS:
            relative = f"dataset/ehrsql/{database}/{split}.json"
            ehr_files[relative] = _write_json(
                ehr_root / relative,
                [
                    {
                        "db_id": database,
                        "split": split,
                        "id": f"{database}-{split}",
                    }
                ],
            )
        ehr_files[f"dataset/ehrsql/{database}/test.json"] = {
            "sha256": "2" * 64,
            "size": 456,
        }
    for relative, payload in (
        ("dataset/ehrsql/tables.json", []),
        ("LICENSE", "license"),
        ("README.md", "readme"),
    ):
        path = ehr_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, list):
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path.write_text(payload, encoding="utf-8")
        ehr_files[relative] = {
            "sha256": MODULE._sha256(path),
            "size": path.stat().st_size,
        }
    audit = tmp_path / "ehr-audit.json"
    audit.write_text(
        json.dumps(
            {
                "benchmark": "EHRSQL-v1.5.1",
                "kind": "ehrsql_public_repository_structure_audit",
                "contains_questions_or_sql": False,
                "source_revision": "3" * 40,
                "files": ehr_files,
            }
        ),
        encoding="utf-8",
    )
    _, ehr_rows = MODULE._load_ehrsql_development(ehr_root, audit)
    assert len(ehr_rows) == 4


def test_v2_builder_is_create_once_and_public_receipt_is_aggregate_only(
    tmp_path,
    monkeypatch,
):
    css_manifest_path = tmp_path / "css-manifest.json"
    ehr_audit_path = tmp_path / "ehr-audit.json"
    db_path = tmp_path / "four-table.sqlite"
    for path in (css_manifest_path, ehr_audit_path, db_path):
        path.write_bytes(b"bound")
    source_entry = {"sha256": "4" * 64, "size": 1}
    css_manifest = {
        "files": {
            "development/train.json": source_entry,
            "development/dev.json": source_entry,
            "holdout/test.json": source_entry,
        }
    }
    ehr_files = {
        f"dataset/ehrsql/{database}/{split}.json": source_entry
        for database in MODULE.EHR_DATABASES
        for split in ("train", "valid", "test")
    }
    ehr_audit = {"files": ehr_files}
    css_rows = [
        ("train", {"db_id": "css-db", "index": index})
        for index in range(200)
    ]
    monkeypatch.setattr(
        MODULE,
        "_load_css_development",
        lambda _: (css_manifest, tmp_path, css_rows),
    )
    monkeypatch.setattr(
        MODULE,
        "_load_ehrsql_development",
        lambda *_: (ehr_audit, []),
    )
    monkeypatch.setattr(
        MODULE,
        "load_css_schemas",
        lambda *_args, **_kwargs: {"css-db": object()},
    )
    monkeypatch.setattr(MODULE, "load_target_domains", lambda _: object())
    monkeypatch.setattr(MODULE, "load_ehr_target_domains", lambda _: object())

    def fake_projection(row, *_args, **_kwargs):
        index = row["index"]
        role = "development" if index < 160 else "holdout"
        skeleton_index = index % 20 if role == "development" else 100 + index % 20
        skeleton = f"{skeleton_index:064x}"
        return ProjectedItem(
            item={
                "id": f"css-train-{index:04d}",
                "question": f"private marker question {index}",
                "gold_sql": f"SELECT {index}",
                "difficulty": "hard",
                "category": "filtering",
                "provenance": {
                    "benchmark": "CSS",
                    "skeleton_role": role,
                },
            },
            skeleton_role=role,
            skeleton_sha256=skeleton,
        )

    monkeypatch.setattr(MODULE, "project_css_row", fake_projection)
    private_output = (
        tmp_path / "outputs" / "private" / "rid" / "development.jsonl"
    )
    public_output = (
        tmp_path / "outputs" / "remote_runs" / "rid" / "receipt.json"
    )
    receipt = MODULE.build_v2_development(
        css_freeze_manifest=css_manifest_path,
        ehrsql_root=tmp_path,
        ehrsql_audit=ehr_audit_path,
        db_path=db_path,
        private_output=private_output,
        public_output=public_output,
        source_tree_sha256="5" * 64,
        created_at="2026-07-26T12:00:00Z",
        maximum_items=150,
    )
    assert receipt["development_selected"]["n"] == 150
    assert receipt["train_valid_holdout_role_capacity_proxy"]["n"] == 40
    assert "private marker question" in private_output.read_text(encoding="utf-8")
    public_text = public_output.read_text(encoding="utf-8")
    assert "private marker question" not in public_text
    assert '"gold_sql"' not in public_text
    with pytest.raises(FileExistsError):
        MODULE.build_v2_development(
            css_freeze_manifest=css_manifest_path,
            ehrsql_root=tmp_path,
            ehrsql_audit=ehr_audit_path,
            db_path=db_path,
            private_output=private_output,
            public_output=public_output,
            source_tree_sha256="5" * 64,
            created_at="2026-07-26T12:00:00Z",
            maximum_items=150,
        )
