from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.validate_nl2sql_independent_delivery import main, validate_delivery


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _handoff(
    tmp_path: Path,
    *,
    source_version: str = "independent-curator-v11",
) -> dict[str, Path]:
    root = tmp_path / "outputs" / "private" / "20260802T000000Z-gabcdef0-t123456789abc"
    root.mkdir(parents=True)
    corpus = root / "corpus_v11.jsonl"
    corpus_rows = [
        {
            "id": "row-1",
            "question": "共有多少患者？",
            "gold_sql": "SELECT COUNT(*) AS patient_count FROM patients",
            "difficulty": "medium",
            "category": "aggregation_grouping",
        },
        {
            "id": "row-2",
            "question": "共有多少次就诊？",
            "gold_sql": "SELECT COUNT(*) AS visit_count FROM visits",
            "difficulty": "hard",
            "category": "multi_table_join",
        },
        {
            "id": "row-3",
            "question": "共有多少个图谱实体？",
            "gold_sql": "SELECT COUNT(*) AS entity_count FROM kg_entities",
            "difficulty": "extra",
            "category": "knowledge_graph",
        },
    ]
    corpus.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in corpus_rows),
        encoding="utf-8",
    )
    execution = root / "execution_audit_v11.jsonl"
    execution.write_bytes(b"{}\n{}\n{}\n")
    database = tmp_path / "sealed.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE patients (patient_id TEXT PRIMARY KEY);
            CREATE TABLE visits (visit_id TEXT PRIMARY KEY);
            CREATE TABLE kg_entities (entity_id TEXT PRIMARY KEY);
            CREATE TABLE kg_triples (subject TEXT, relation TEXT, object TEXT);
            """
        )
    attestation = root / "curator_attestation_v11.json"
    attestation.write_text(
        json.dumps(
            {
                "source_version": source_version,
                "corpus": {"record_count": 3, "sha256": _sha(corpus)},
                "database": {"sha256": _sha(database)},
                "independence_declarations": {
                    "implementation_or_source_code_accessed": False,
                    "nl2sql_implementation_accessed": False,
                    "tests_accessed": False,
                    "existing_nl2sql_eval_accessed": False,
                    "outputs_private_accessed": False,
                    "prior_curator_files_accessed": False,
                    "prior_holdout_accessed": False,
                    "failure_details_accessed": False,
                    "conversation_records_accessed": False,
                    "existing_questions_copied_or_paraphrased": False,
                    "database_schema_and_self_generated_query_results_only": True,
                },
                "execution_attestation": {
                    "all_queries_executed_successfully": True,
                    "all_results_fully_materialized": True,
                    "single_read_only_sqlite_query_per_record": True,
                    "allowed_tables_only": True,
                },
            }
        ),
        encoding="utf-8",
    )
    aggregate = root / "aggregate_audit_v11.json"
    aggregate.write_text(
        json.dumps(
            {
                "record_count": 3,
                "unique_id_count": 3,
                "unique_normalized_question_count": 3,
                "unique_gold_sql_count": 3,
                "execution": {"success_count": 3, "failure_count": 0},
            }
        ),
        encoding="utf-8",
    )
    sums = root / "SHA256SUMS.txt"
    sums.write_text(
        "\n".join(
            f"{_sha(path)}  {path.name}"
            for path in (corpus, attestation, aggregate, execution)
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "corpus": corpus,
        "attestation": attestation,
        "aggregate": aggregate,
        "execution": execution,
        "sums": sums,
        "database": database,
    }


def test_validates_five_member_handoff_without_exposing_corpus_rows(tmp_path: Path) -> None:
    paths = _handoff(tmp_path)
    result = validate_delivery(
        corpus=paths["corpus"],
        attestation=paths["attestation"],
        aggregate_audit=paths["aggregate"],
        execution_audit=paths["execution"],
        sha256s=paths["sums"],
        database=paths["database"],
        expected_count=3,
    )
    assert result["source_version"] == "independent-curator-v11"
    assert result["record_count"] == 3
    assert result["semantic_validation_count"] == 3
    assert result["bounded_execution_success_count"] == 3


@pytest.mark.parametrize(
    "source_version",
    [
        "independent-curator-v12",
        "independent-curator-v13",
        "independent-curator-v14",
        "independent-curator-v15",
        "independent-curator-v16",
        "independent-curator-v17",
        "independent-curator-v18",
        "independent-curator-v19",
        "independent-curator-v20",
        "independent-curator-v21",
        "independent-curator-v22",
        "independent-curator-v23",
        "independent-curator-v24",
        "independent-curator-v25",
        "independent-curator-v26",
        "independent-curator-v27",
    ],
)
def test_validates_new_engine_isolated_versions_under_the_same_five_file_contract(
    tmp_path: Path, source_version: str
) -> None:
    paths = _handoff(tmp_path, source_version=source_version)
    result = validate_delivery(
        corpus=paths["corpus"],
        attestation=paths["attestation"],
        aggregate_audit=paths["aggregate"],
        execution_audit=paths["execution"],
        sha256s=paths["sums"],
        database=paths["database"],
        expected_count=3,
    )
    assert result["source_version"] == source_version


def test_rejects_a_handoff_that_misses_the_exact_curation_contract(tmp_path: Path) -> None:
    paths = _handoff(tmp_path)
    with pytest.raises(ValueError, match="expected 300, received 3"):
        validate_delivery(
            corpus=paths["corpus"],
            attestation=paths["attestation"],
            aggregate_audit=paths["aggregate"],
            execution_audit=paths["execution"],
            sha256s=paths["sums"],
            database=paths["database"],
            expected_count=300,
        )


def test_rejection_receipt_is_aggregate_only(tmp_path: Path) -> None:
    paths = _handoff(tmp_path)
    output = tmp_path / "outputs" / "remote_runs" / "rid" / "nl2sql" / "v11.json"
    result = main(
        [
            "--corpus", str(paths["corpus"]),
            "--attestation", str(paths["attestation"]),
            "--aggregate-audit", str(paths["aggregate"]),
            "--execution-audit", str(paths["execution"]),
            "--sha256s", str(paths["sums"]),
            "--database", str(paths["database"]),
            "--expected-count", "300",
            "--source-tree-sha256", "a" * 64,
            "--output", str(output),
        ]
    )
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert result == 2
    assert receipt["status"] == "REJECTED_PRE_FREEZE"
    assert receipt["failure_code"] == "record_count_failed"
    assert receipt["observed_record_count"] == 3
    assert "question" not in receipt
    assert "gold_sql" not in receipt
    assert "prediction" not in receipt


def test_bounded_execution_failure_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _handoff(tmp_path)

    def interrupted(*_args, **_kwargs):
        raise sqlite3.OperationalError("interrupted")

    monkeypatch.setattr(
        "scripts.validate_nl2sql_independent_delivery.execute_complete_readonly",
        interrupted,
    )
    with pytest.raises(ValueError, match="opaque corpus bounded execution failed") as caught:
        validate_delivery(
            corpus=paths["corpus"],
            attestation=paths["attestation"],
            aggregate_audit=paths["aggregate"],
            execution_audit=paths["execution"],
            sha256s=paths["sums"],
            database=paths["database"],
            expected_count=3,
        )
    message = str(caught.value)
    assert "row-1" not in message
    assert "SELECT" not in message
    assert "患者" not in message
