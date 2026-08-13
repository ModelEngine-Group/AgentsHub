import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "index_nl2sql_public_donors.py"
SPEC = importlib.util.spec_from_file_location(
    "index_nl2sql_public_donors",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _binding(path: Path) -> dict:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )


def _sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    css = tmp_path / "css"
    css_rows = [
        {
            "question_id": "css-1",
            "db_id": "medical",
            "question": "统计每个科室的就诊次数",
            "query": (
                "SELECT department, COUNT(*) FROM visits "
                "GROUP BY department"
            ),
        }
    ]
    for split in ("train", "dev"):
        _write_json(css / "development" / f"{split}.json", css_rows)
    # Deliberately not JSON: indexer may hash but must never parse test content.
    holdout = css / "holdout" / "test.json"
    holdout.parent.mkdir(parents=True, exist_ok=True)
    holdout.write_text("NOT JSON AND MUST NOT BE PARSED", encoding="utf-8")
    css_files = {
        relative: _binding(css / Path(relative))
        for relative in (
            "development/train.json",
            "development/dev.json",
            "holdout/test.json",
        )
    }
    css_manifest = css / "freeze_manifest.json"
    _write_json(
        css_manifest,
        {
            "kind": "external_authoritative_nl2sql_benchmark_freeze",
            "benchmark": "CSS",
            "official_split": "schema",
            "source": {"revision": "a" * 40},
            "files": css_files,
        },
    )

    ehr = tmp_path / "ehr"
    ehr_row = [
        {
            "id": "ehr-1",
            "question": "How many visits were recorded?",
            "query": "SELECT COUNT(*) FROM admissions",
            "is_impossible": False,
        }
    ]
    ehr_files = {}
    for relative in (
        *MODULE.EHRSQL_DEVELOPMENT_FILES,
        *MODULE.EHRSQL_HOLDOUT_FILES,
    ):
        path = ehr / Path(relative)
        if relative in MODULE.EHRSQL_HOLDOUT_FILES:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("NOT JSON AND MUST NOT BE PARSED", encoding="utf-8")
        else:
            _write_json(path, ehr_row)
        ehr_files[relative] = _binding(path)
    audit = tmp_path / "ehr-audit.json"
    _write_json(
        audit,
        {
            "kind": "ehrsql_public_repository_structure_audit",
            "benchmark": "EHRSQL-v1.5.1",
            "source_revision": "b" * 40,
            "files": ehr_files,
        },
    )
    return css_manifest, audit, ehr


def test_public_donor_index_is_hash_only_and_never_parses_test(tmp_path):
    css_manifest, ehr_audit, ehr_root = _sources(tmp_path)
    private = tmp_path / "outputs" / "private" / "rid" / "donors.json"
    public = tmp_path / "outputs" / "remote_runs" / "rid" / "donors.json"
    private_payload, public_payload = MODULE.build_index(
        css_manifest=css_manifest,
        ehrsql_audit=ehr_audit,
        ehrsql_root=ehr_root,
        private_output=private,
        public_output=public,
        source_tree_sha256="c" * 64,
    )
    assert private_payload["official_source_test_content_parsed"] is False
    assert public_payload["official_source_test_content_parsed"] is False
    assert private_payload["aggregate"]["rows"] == 6
    assert private_payload["aggregate"]["languages"] == {
        "en": 4,
        "zh-CN": 2,
    }
    private_text = private.read_text(encoding="utf-8")
    public_text = public.read_text(encoding="utf-8")
    for secret in (
        "统计每个科室的就诊次数",
        "SELECT department",
        "How many visits were recorded?",
        "SELECT COUNT",
        "NOT JSON",
    ):
        assert secret not in private_text
        assert secret not in public_text
    assert public_payload["four_table_blind_gate"] is False
    with pytest.raises(FileExistsError):
        MODULE.build_index(
            css_manifest=css_manifest,
            ehrsql_audit=ehr_audit,
            ehrsql_root=ehr_root,
            private_output=private,
            public_output=public,
            source_tree_sha256="c" * 64,
        )


def test_public_donor_index_rejects_source_tampering(tmp_path):
    css_manifest, ehr_audit, ehr_root = _sources(tmp_path)
    target = (
        css_manifest.parent
        / "development"
        / "train.json"
    )
    target.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="binding failed"):
        MODULE.build_index(
            css_manifest=css_manifest,
            ehrsql_audit=ehr_audit,
            ehrsql_root=ehr_root,
            private_output=tmp_path / "private.json",
            public_output=tmp_path / "public.json",
            source_tree_sha256="d" * 64,
        )
