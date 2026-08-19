from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dki.analysis.nl2sql_hard_negatives import (
    build_hard_negatives,
    mutation_labels,
)
from dki.analysis.nl2sql_semantic_corpus import (
    build_semantic_corpus,
    write_semantic_corpus,
)


ROOT = Path(__file__).resolve().parents[1]


def test_hard_negatives_are_single_labelled_executable_mutations(
    semantic_four_table_db: Path,
) -> None:
    corpus = build_semantic_corpus(semantic_four_table_db, dev_fraction=0.30)
    rows = [*corpus["train"], *corpus["ood_dev"]]
    first = build_hard_negatives(semantic_four_table_db, rows)
    second = build_hard_negatives(semantic_four_table_db, rows)

    assert first["output_bytes"] == second["output_bytes"]
    assert first["manifest"] == second["manifest"]
    assert first["negatives"]
    assert set(first["manifest"]["mutation_labels"]) <= set(mutation_labels())
    assert "not_exists_to_not_in" in first["manifest"]["mutation_labels"]
    assert "rank_to_row_number" in first["manifest"]["mutation_labels"]
    assert "union_all_to_union" in first["manifest"]["mutation_labels"]
    assert first["manifest"]["controlled_temporary_sqlite_witnesses"] >= 1

    for row in first["negatives"]:
        assert row["negative_sql"] != row["gold_sql"]
        assert row["mutation"]["single_semantic_mutation"] is True
        assert row["mutation"]["label"] in mutation_labels()
        assert row["readonly_executable"] is True
        witness = row["distinguishability"]
        assert witness["source"] in {
            "base_database",
            "controlled_temporary_sqlite",
        }
        assert (
            witness["gold_execution"]["result_sha256"]
            != witness["negative_execution"]["result_sha256"]
        )


def test_track_c_cli_tools_write_only_requested_temp_paths(
    semantic_four_table_db: Path,
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_command = [
        sys.executable,
        str(ROOT / "scripts" / "build_nl2sql_semantic_corpus.py"),
        "--db",
        str(semantic_four_table_db),
        "--output-dir",
        str(corpus_dir),
        "--dev-fraction",
        "0.30",
    ]
    corpus_run = subprocess.run(
        corpus_command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    corpus_summary = json.loads(corpus_run.stdout)
    assert Path(corpus_summary["train"]).is_file()
    assert Path(corpus_summary["ood_dev"]).is_file()

    negative_path = tmp_path / "hard_negatives.jsonl"
    negative_command = [
        sys.executable,
        str(ROOT / "scripts" / "build_nl2sql_hard_negatives.py"),
        "--db",
        str(semantic_four_table_db),
        "--corpus",
        str(corpus_dir),
        "--output",
        str(negative_path),
    ]
    negative_run = subprocess.run(
        negative_command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    negative_summary = json.loads(negative_run.stdout)
    assert negative_path.is_file()
    assert Path(negative_summary["manifest"]).is_file()
    assert negative_summary["rows"] > 0

    loaded = write_semantic_corpus(
        semantic_four_table_db,
        tmp_path / "library_corpus",
        dev_fraction=0.30,
    )
    assert loaded["train_path"].is_file()
