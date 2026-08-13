from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dki.analysis.nl2sql_protocol import execute_complete_readonly
from dki.analysis.nl2sql_semantic_corpus import (
    REQUIRED_COVERAGE,
    build_semantic_corpus,
    validate_family_separation,
    write_semantic_corpus,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_semantic_corpus_is_deterministic_and_family_disjoint(
    semantic_four_table_db: Path,
) -> None:
    first = build_semantic_corpus(semantic_four_table_db, dev_fraction=0.30)
    second = build_semantic_corpus(semantic_four_table_db, dev_fraction=0.30)

    assert first["train_bytes"] == second["train_bytes"]
    assert first["ood_dev_bytes"] == second["ood_dev_bytes"]
    assert first["manifest"] == second["manifest"]
    assert first["train"]
    assert first["ood_dev"]
    assert set(first["manifest"]["observed_coverage"]) == REQUIRED_COVERAGE
    assert validate_family_separation(first["train"], first["ood_dev"]) == {
        "semantic_family": True,
        "join_path": True,
        "ast_skeleton": True,
        "near_duplicate_family": True,
    }

    for row in [*first["train"], *first["ood_dev"]]:
        assert row["split"] in {"train", "ood_dev"}
        assert row["semantic_family"]
        assert row["ast"]["skeleton"]
        assert row["ast"]["family"]
        assert row["family_metadata"]["near_duplicate_family"]
        assert row["execution"]["complete"] is True
        assert row["execution"]["readonly"] is True


def test_every_gold_is_completely_executable_and_writes_only_to_temp_output(
    semantic_four_table_db: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "generated"
    result = write_semantic_corpus(
        semantic_four_table_db,
        output,
        dev_fraction=0.30,
    )

    assert set(path.name for path in output.iterdir()) == {
        "train.jsonl",
        "ood_dev.jsonl",
        "manifest.json",
    }
    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
    assert _file_sha256(result["train_path"]) == manifest["files"]["train.jsonl"]["sha256"]
    assert _file_sha256(result["ood_dev_path"]) == manifest["files"]["ood_dev.jsonl"]["sha256"]

    for split_path in (result["train_path"], result["ood_dev_path"]):
        for line in split_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            columns, rows = execute_complete_readonly(row["gold_sql"], semantic_four_table_db)
            assert columns == row["execution"]["columns"]
            assert len(rows) == row["execution"]["row_count"]
