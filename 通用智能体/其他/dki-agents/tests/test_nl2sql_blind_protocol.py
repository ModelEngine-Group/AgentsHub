import hashlib
import sqlite3
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlglot")

from dki.analysis.nl2sql_protocol import (
    execute_complete_readonly,
    validate_gold_sql,
    validate_question_sql_grounding,
)
from dki.analysis.nl2sql_eval import _score_item

FREEZE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "freeze_nl2sql_blind.py"
FREEZE_SPEC = importlib.util.spec_from_file_location("freeze_nl2sql_blind", FREEZE_SCRIPT)
FREEZE_MODULE = importlib.util.module_from_spec(FREEZE_SPEC)
FREEZE_SPEC.loader.exec_module(FREEZE_MODULE)
BLIND_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_nl2sql_blind.py"
BLIND_SPEC = importlib.util.spec_from_file_location("eval_nl2sql_blind", BLIND_SCRIPT)
BLIND_MODULE = importlib.util.module_from_spec(BLIND_SPEC)
BLIND_SPEC.loader.exec_module(BLIND_MODULE)


def test_freeze_protocol_accepts_current_five_file_handoffs():
    assert "independent-v13" in FREEZE_MODULE.CURATOR_FORMAT_CHOICES
    assert "independent-v14" in FREEZE_MODULE.CURATOR_FORMAT_CHOICES
    assert "independent-v13" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert "independent-v14" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert "independent-v15" in FREEZE_MODULE.CURATOR_FORMAT_CHOICES
    assert "independent-v15" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert "independent-v16" in FREEZE_MODULE.CURATOR_FORMAT_CHOICES
    assert "independent-v16" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert "independent-v17" in FREEZE_MODULE.CURATOR_FORMAT_CHOICES
    assert "independent-v17" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert "independent-v18" in FREEZE_MODULE.CURATOR_FORMAT_CHOICES
    assert "independent-v18" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert "independent-v19" in FREEZE_MODULE.CURATOR_FORMAT_CHOICES
    assert "independent-v19" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert "independent-v20" in FREEZE_MODULE.CURATOR_FORMAT_CHOICES
    assert "independent-v20" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert "independent-v21" in FREEZE_MODULE.CURATOR_FORMAT_CHOICES
    assert "independent-v21" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert "independent-v22" in FREEZE_MODULE.CURATOR_FORMAT_CHOICES
    assert "independent-v22" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert "independent-v23" in FREEZE_MODULE.CURATOR_FORMAT_CHOICES
    assert "independent-v23" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert "independent-v24" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert "independent-v25" in FREEZE_MODULE.CURATOR_FORMAT_CHOICES
    assert "independent-v25" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert "independent-v26" in FREEZE_MODULE.CURATOR_FORMAT_CHOICES
    assert "independent-v26" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert "independent-v27" in FREEZE_MODULE.CURATOR_FORMAT_CHOICES
    assert "independent-v27" in FREEZE_MODULE.INDEPENDENT_FIVE_FILE_FORMATS
    assert (
        "independent-curator-v13"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )
    assert (
        "independent-curator-v15"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )
    assert (
        "independent-curator-v16"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )
    assert (
        "independent-curator-v17"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )
    assert (
        "independent-curator-v18"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )
    assert (
        "independent-curator-v19"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )
    assert (
        "independent-curator-v20"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )
    assert (
        "independent-curator-v21"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )
    assert (
        "independent-curator-v22"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )
    assert (
        "independent-curator-v23"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )
    assert (
        "independent-curator-v24"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )
    assert (
        "independent-curator-v25"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )
    assert (
        "independent-curator-v26"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )
    assert (
        "independent-curator-v27"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )
    assert (
        "independent-curator-v14"
        in FREEZE_MODULE.SUPPORTED_INDEPENDENT_CURATOR_VERSIONS
    )


def _db(tmp_path):
    path = tmp_path / "sealed.db"
    conn = sqlite3.connect(path)
    conn.execute("create table patients (patient_id text)")
    conn.executemany("insert into patients values (?)", [(str(i),) for i in range(205)])
    conn.commit()
    conn.close()
    return path


def test_blind_outputs_separate_freeze_and_candidate_rids(tmp_path):
    freeze_rid = "freeze-rid"
    candidate_rid = "candidate-rid"
    manifest = (
        tmp_path
        / "outputs"
        / "private"
        / freeze_rid
        / "nl2sql_blind_v3"
        / "frozen"
        / "freeze_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    private_output = manifest.parent.parent / "blind_evaluation_details.json"
    public_output = (
        tmp_path
        / "outputs"
        / "remote_runs"
        / candidate_rid
        / "nl2sql"
        / "blind_attestation.json"
    )

    result = BLIND_MODULE._require_once_outputs(
        manifest,
        private_output,
        public_output,
    )

    assert result[3:] == (freeze_rid, candidate_rid)


def test_blind_private_output_must_remain_with_freeze_rid(tmp_path):
    manifest = (
        tmp_path
        / "outputs"
        / "private"
        / "freeze-rid"
        / "frozen"
        / "freeze_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    private_output = (
        tmp_path / "outputs" / "private" / "candidate-rid" / "details.json"
    )
    public_output = (
        tmp_path
        / "outputs"
        / "remote_runs"
        / "candidate-rid"
        / "nl2sql"
        / "blind_attestation.json"
    )

    with pytest.raises(SystemExit, match="freeze RID"):
        BLIND_MODULE._require_once_outputs(
            manifest,
            private_output,
            public_output,
        )


def test_gold_sql_rejects_writes_and_unapproved_tables():
    with pytest.raises(ValueError):
        validate_gold_sql("DELETE FROM patients")
    with pytest.raises(ValueError):
        validate_gold_sql("SELECT * FROM sqlite_master")
    with pytest.raises(ValueError):
        validate_gold_sql("SELECT * FROM patients; SELECT * FROM patients")


def test_gold_sql_allows_local_cte_names_but_not_forbidden_cte_sources():
    validate_gold_sql(
        "WITH patient_ids AS (SELECT patient_id FROM patients) "
        "SELECT patient_id FROM patient_ids"
    )
    with pytest.raises(ValueError, match="forbidden tables"):
        validate_gold_sql(
            "WITH leaked AS (SELECT name FROM sqlite_master) "
            "SELECT name FROM leaked"
        )


def test_question_sql_grounding_requires_explicit_predicate_literals():
    validate_question_sql_grounding(
        "查询地区为上海且年龄不低于18岁的患者，返回前10名。",
        "SELECT patient_id FROM patients WHERE region='上海' AND age>=18 LIMIT 10",
    )
    with pytest.raises(ValueError, match="not grounded"):
        validate_question_sql_grounding(
            "查询年龄不低于18岁的患者。",
            "SELECT patient_id FROM patients WHERE region='上海' AND age>=18",
        )
    with pytest.raises(ValueError, match="empty predicate literal"):
        validate_question_sql_grounding(
            "找出开具药物为且开具次数不低于1次的科室。",
            "SELECT department FROM visits WHERE drug_prescribed='' "
            "GROUP BY department HAVING COUNT(*)>=1",
        )


def test_complete_executor_does_not_apply_ui_row_limit(tmp_path):
    columns, rows = execute_complete_readonly("SELECT patient_id FROM patients", _db(tmp_path))
    assert columns == ["patient_id"]
    assert len(rows) == 205


class _StaticRuleNL2SQL:
    def __init__(self, sql):
        self.sql = sql

    def rule_sql(self, _question):
        return self.sql


def test_strict_scoring_rejects_extra_prediction_columns_for_empty_results(tmp_path):
    database = _db(tmp_path)
    item = {
        "id": "empty-result-column-contract",
        "question": "return no matching patient",
        "gold_sql": (
            "SELECT patient_id FROM patients "
            "WHERE patient_id = 'does-not-exist'"
        ),
        "difficulty": "hard",
        "category": "filter",
    }
    result = _score_item(
        _StaticRuleNL2SQL(
            "SELECT patient_id, patient_id FROM patients "
            "WHERE patient_id = 'does-not-exist'"
        ),
        item,
        "rule",
        database,
        strict=True,
    )
    assert result["correct"] is False
    assert result["reason"] == "strict_result_column_count_mismatch"
    assert result["gold_column_count"] == 1
    assert result["pred_column_count"] == 2
    assert result["column_count_matches"] is False


def test_strict_scoring_accepts_same_column_count_for_empty_results(tmp_path):
    database = _db(tmp_path)
    item = {
        "id": "empty-result-column-contract-control",
        "question": "return no matching patient",
        "gold_sql": (
            "SELECT patient_id FROM patients "
            "WHERE patient_id = 'does-not-exist'"
        ),
        "difficulty": "hard",
        "category": "filter",
    }
    result = _score_item(
        _StaticRuleNL2SQL(
            "SELECT patient_id FROM patients "
            "WHERE patient_id = 'does-not-exist'"
        ),
        item,
        "rule",
        database,
        strict=True,
    )
    assert result["correct"] is True
    assert result["gold_column_count"] == 1
    assert result["pred_column_count"] == 1
    assert result["column_count_matches"] is True


def _strict_order_score(tmp_path, question, gold_sql, pred_sql, **item_overrides):
    item = {
        "id": "result-order-contract",
        "question": question,
        "gold_sql": gold_sql,
        "difficulty": "hard",
        "category": "ordering",
        **item_overrides,
    }
    return _score_item(
        _StaticRuleNL2SQL(pred_sql),
        item,
        "rule",
        _db(tmp_path),
        strict=True,
    )


def test_incidental_gold_order_by_does_not_make_result_order_semantic(tmp_path):
    result = _strict_order_score(
        tmp_path,
        "返回全部患者编号",
        "SELECT patient_id FROM patients ORDER BY patient_id ASC",
        "SELECT patient_id FROM patients ORDER BY patient_id DESC",
    )

    assert result["correct"] is True
    assert result["result_order_sensitive"] is False
    assert result["result_order_policy"] == "question_explicit_final_order_v1"


def test_explicit_final_result_order_is_enforced(tmp_path):
    result = _strict_order_score(
        tmp_path,
        "返回全部患者编号，最终结果按患者编号升序排列",
        "SELECT patient_id FROM patients ORDER BY patient_id ASC",
        "SELECT patient_id FROM patients ORDER BY patient_id DESC",
    )

    assert result["correct"] is False
    assert result["result_order_sensitive"] is True


def test_window_order_does_not_imply_outer_result_order(tmp_path):
    window_sql = (
        "SELECT patient_id, "
        "ROW_NUMBER() OVER (ORDER BY CAST(patient_id AS INTEGER)) AS row_no "
        "FROM patients"
    )
    result = _strict_order_score(
        tmp_path,
        "为每位患者计算按编号得到的行号，并返回患者编号和行号",
        window_sql,
        window_sql + " ORDER BY CAST(patient_id AS INTEGER) DESC",
    )

    assert result["correct"] is True
    assert result["result_order_sensitive"] is False


def test_top_k_selection_accepts_same_selected_set_in_another_order(tmp_path):
    gold = (
        "SELECT patient_id FROM patients "
        "ORDER BY CAST(patient_id AS INTEGER) DESC LIMIT 3"
    )
    pred = (
        "SELECT patient_id FROM ("
        "SELECT patient_id FROM patients "
        "ORDER BY CAST(patient_id AS INTEGER) DESC LIMIT 3"
        ") AS selected ORDER BY CAST(patient_id AS INTEGER) ASC"
    )
    result = _strict_order_score(
        tmp_path,
        "返回患者编号最大的前3名",
        gold,
        pred,
    )

    assert result["correct"] is True
    assert result["result_order_sensitive"] is False


def test_result_order_override_is_boolean_and_authoritative(tmp_path):
    result = _strict_order_score(
        tmp_path,
        "返回全部患者编号",
        "SELECT patient_id FROM patients ORDER BY patient_id ASC",
        "SELECT patient_id FROM patients ORDER BY patient_id DESC",
        result_order_sensitive=True,
    )
    assert result["correct"] is False
    assert result["result_order_sensitive"] is True

    invalid_root = tmp_path / "invalid-override"
    invalid_root.mkdir()
    invalid = _strict_order_score(
        invalid_root,
        "返回全部患者编号",
        "SELECT patient_id FROM patients ORDER BY patient_id ASC",
        "SELECT patient_id FROM patients ORDER BY patient_id DESC",
        result_order_sensitive="yes",
    )
    assert invalid["correct"] is False
    assert "must be boolean" in invalid["reason"]


def test_ast_skeleton_groups_literal_variants_together():
    first = FREEZE_MODULE._sql_ast_skeleton("SELECT patient_id FROM patients WHERE patient_id = '1'")
    second = FREEZE_MODULE._sql_ast_skeleton("SELECT patient_id FROM patients WHERE patient_id = '2'")
    assert first == second


def test_freeze_entrypoint_help_resolves_repo_imports():
    completed = subprocess.run(
        [sys.executable, str(FREEZE_SCRIPT), "--help"],
        cwd=FREEZE_SCRIPT.parents[1],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--curator-attestation" in completed.stdout
    assert "--curator-pack-manifest" in completed.stdout
    assert "--curator-sha256s" in completed.stdout


def test_freeze_sync_writes_exact_bytes(tmp_path):
    path = tmp_path / "artifact.json"
    FREEZE_MODULE._write_bytes_synced(path, b"immutable")
    assert path.read_bytes() == b"immutable"


def test_curator_boundary_is_hash_bound_and_rejects_tampering(tmp_path):
    pack_root = tmp_path / "outputs" / "private" / "rid" / "curator_pack"
    pack_root.mkdir(parents=True)
    schema = pack_root / "schema.sql"
    schema.write_text("CREATE TABLE patients(patient_id TEXT);\n", encoding="utf-8")
    corpus = tmp_path / "outputs" / "private" / "rid" / "corpus.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    pack_manifest = pack_root / "pack_manifest.json"
    pack_manifest.write_text(
        json.dumps(
            {
                "kind": "engine_blind_nl2sql_curator_pack",
                "dataset_role": "independent_curation_input_only",
                "created_once": True,
                "contains_engine_implementation": False,
                "contains_questions_or_gold_sql": False,
                "files": {
                    "schema.sql": {
                        "sha256": hashlib.sha256(schema.read_bytes()).hexdigest(),
                        "size": schema.stat().st_size,
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    attestation = tmp_path / "outputs" / "private" / "rid" / "attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "kind": "independent_four_table_nl2sql_curator_attestation",
                "independent_curation": True,
                "engine_context_provided": False,
                "local_evaluation_context_provided": False,
                "curator_identity": "curator-a",
                "semantic_reviewer_identity": "reviewer-b",
                "created_at": "2026-07-26T08:00:00+00:00",
                "curator_pack_manifest_sha256": hashlib.sha256(
                    pack_manifest.read_bytes()
                ).hexdigest(),
                "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "n_items": 1,
                "all_gold_sql_executed": True,
                "all_pairs_semantically_reviewed": True,
                "project_local_question_reuse": False,
                "project_local_question_paraphrase": False,
                "development_or_holdout_preassigned": False,
            }
        ),
        encoding="utf-8",
    )

    FREEZE_MODULE._validate_pack_and_attestation(
        pack_manifest_path=pack_manifest,
        curator_attestation_path=attestation,
        corpus_path=corpus,
        n_rows=1,
    )
    schema.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="binding failed"):
        FREEZE_MODULE._validate_pack_and_attestation(
            pack_manifest_path=pack_manifest,
            curator_attestation_path=attestation,
            corpus_path=corpus,
            n_rows=1,
        )


def test_five_file_pack_and_sha256s_are_distinct_evidence(tmp_path):
    pack_root = tmp_path / "outputs" / "private" / "rid" / "curator_pack"
    pack_root.mkdir(parents=True)
    database = pack_root / "four_table.sqlite"
    database.write_bytes(b"sealed-db")
    pack_manifest = pack_root / "pack_manifest.json"
    pack_manifest.write_text(
        json.dumps(
            {
                "kind": "engine_blind_nl2sql_curator_pack",
                "dataset_role": "independent_curation_input_only",
                "created_once": True,
                "contains_engine_implementation": False,
                "contains_questions_or_gold_sql": False,
                "files": {
                    database.name: {
                        "sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
                        "size": database.stat().st_size,
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus_v23.jsonl"
    corpus.write_text('{"id":"one"}\n', encoding="utf-8")
    attestation = tmp_path / "curator_attestation_v23.json"
    aggregate = tmp_path / "aggregate_audit_v23.json"
    execution = tmp_path / "execution_audit_v23.jsonl"
    independence = {
        key: False
        for key in (
            "implementation_or_source_code_accessed",
            "nl2sql_implementation_accessed",
            "tests_accessed",
            "existing_nl2sql_eval_accessed",
            "outputs_private_accessed",
            "prior_curator_files_accessed",
            "prior_holdout_accessed",
            "failure_details_accessed",
            "conversation_records_accessed",
            "existing_questions_copied_or_paraphrased",
        )
    }
    independence["database_schema_and_self_generated_query_results_only"] = True
    attestation.write_text(
        json.dumps(
            {
                "source_version": "independent-curator-v23",
                "database": {
                    "sha256": hashlib.sha256(database.read_bytes()).hexdigest()
                },
                "corpus": {
                    "record_count": 1,
                    "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                },
                "independence_declarations": independence,
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
    aggregate.write_text(
        json.dumps(
            {
                "record_count": 1,
                "unique_id_count": 1,
                "unique_normalized_question_count": 1,
                "unique_gold_sql_count": 1,
                "execution": {"success_count": 1, "failure_count": 0},
            }
        ),
        encoding="utf-8",
    )
    execution.write_text("{}\n", encoding="utf-8")
    sha256s = tmp_path / "SHA256SUMS.txt"
    handoff_files = (corpus, attestation, aggregate, execution)
    sha256s.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in handoff_files
        ),
        encoding="utf-8",
    )

    pack = FREEZE_MODULE._validate_engine_blind_pack(pack_manifest)
    assert pack["files"][database.name]["sha256"] == hashlib.sha256(
        database.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="SHA256SUMS line"):
        FREEZE_MODULE._validate_independent_attestation(
            corpus_path=corpus,
            database_path=database,
            attestation_path=attestation,
            aggregate_audit_path=aggregate,
            execution_audit_path=execution,
            sha256s_path=pack_manifest,
            n_rows=1,
        )


def test_public_projection_attestation_is_database_and_corpus_bound(tmp_path):
    pack_root = tmp_path / "outputs" / "private" / "rid" / "curator_pack"
    pack_root.mkdir(parents=True)
    database = pack_root / "four_table.sqlite"
    database.write_bytes(b"sealed-db")
    corpus = tmp_path / "outputs" / "private" / "rid" / "corpus.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    pack_manifest = pack_root / "pack_manifest.json"
    pack_manifest.write_text(
        json.dumps(
            {
                "kind": "engine_blind_nl2sql_curator_pack",
                "dataset_role": "independent_curation_input_only",
                "created_once": True,
                "contains_engine_implementation": False,
                "contains_questions_or_gold_sql": False,
                "files": {
                    "four_table.sqlite": {
                        "sha256": hashlib.sha256(
                            database.read_bytes()
                        ).hexdigest(),
                        "size": database.stat().st_size,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    attestation = (
        tmp_path / "outputs" / "private" / "rid" / "projection.json"
    )
    payload = {
        "kind": "public_authoritative_four_table_projection_attestation",
        "mechanical_projection": True,
        "all_gold_sql_executed": True,
        "all_pairs_semantically_validated_by_construction": True,
        "engine_context_provided": False,
        "local_evaluation_context_provided": False,
        "project_local_question_reuse": False,
        "project_local_question_paraphrase": False,
        "development_or_holdout_preassigned": False,
        "curator_identity": "deterministic-projector",
        "semantic_reviewer_identity": "ast-execution-validator",
        "created_at": "2026-07-26T08:00:00Z",
        "n_items": 1,
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "independence_basis": {
            "official_source_test_content_parsed": False,
            "project_nl2sql_engine_imported": False,
            "project_local_evaluation_questions_used": False,
            "development_or_holdout_preassigned": False,
        },
        "bindings": {
            "database": {
                "sha256": hashlib.sha256(database.read_bytes()).hexdigest()
            }
        },
        "aggregate": {
            "n_items": 1,
            "all_gold_sql_validated": True,
            "all_gold_sql_executed_readonly_without_truncation": True,
            "all_result_sets_nonempty": True,
            "all_predicate_literals_grounded_in_question": True,
        },
    }
    attestation.write_text(json.dumps(payload), encoding="utf-8")

    FREEZE_MODULE._validate_pack_and_attestation(
        pack_manifest_path=pack_manifest,
        curator_attestation_path=attestation,
        corpus_path=corpus,
        n_rows=1,
    )
    payload["bindings"]["database"]["sha256"] = "0" * 64
    attestation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="database differs"):
        FREEZE_MODULE._validate_pack_and_attestation(
            pack_manifest_path=pack_manifest,
            curator_attestation_path=attestation,
            corpus_path=corpus,
            n_rows=1,
        )


def test_ast_group_holdout_selection_is_deterministic_and_bounded():
    rows = []
    groups = {}
    for group_index in range(150):
        indices = []
        for offset in range(2):
            index = len(rows)
            indices.append(index)
            rows.append(
                {
                    "difficulty": (
                        "medium" if group_index % 5 == 0 else "hard"
                    )
                }
            )
        groups[f"group-{group_index:03d}"] = indices

    first = FREEZE_MODULE._choose_holdout_indices(
        rows=rows,
        groups=groups,
        seed="fixed",
        minimum_holdout=120,
        maximum_holdout=150,
        minimum_hard=40,
    )
    second = FREEZE_MODULE._choose_holdout_indices(
        rows=rows,
        groups=groups,
        seed="fixed",
        minimum_holdout=120,
        maximum_holdout=150,
        minimum_hard=40,
    )
    assert first == second
    assert len(first) == 120
    assert sum(rows[index]["difficulty"] == "hard" for index in first) >= 40
    chosen = set(first)
    assert all(
        set(indices) <= chosen or set(indices).isdisjoint(chosen)
        for indices in groups.values()
    )


def test_independent_v11_handoff_is_hash_and_isolation_bound(tmp_path):
    corpus = tmp_path / "corpus_v11.jsonl"
    corpus.write_text('{"id":"one"}\n', encoding="utf-8")
    database = tmp_path / "insight.db"
    database.write_bytes(b"sealed-database")
    execution = tmp_path / "execution_audit_v11.jsonl"
    execution.write_text('{"id":"one","executed":true}\n', encoding="utf-8")
    aggregate = tmp_path / "aggregate_audit_v11.json"
    aggregate.write_text(
        json.dumps(
            {
                "record_count": 1,
                "unique_id_count": 1,
                "unique_normalized_question_count": 1,
                "unique_gold_sql_count": 1,
                "execution": {"success_count": 1, "failure_count": 0},
            }
        ),
        encoding="utf-8",
    )
    required_false = {
        key: False
        for key in (
            "implementation_or_source_code_accessed",
            "nl2sql_implementation_accessed",
            "tests_accessed",
            "existing_nl2sql_eval_accessed",
            "outputs_private_accessed",
            "prior_curator_files_accessed",
            "prior_holdout_accessed",
            "failure_details_accessed",
            "conversation_records_accessed",
            "existing_questions_copied_or_paraphrased",
        )
    }
    required_false["database_schema_and_self_generated_query_results_only"] = True
    attestation = tmp_path / "curator_attestation_v11.json"
    attestation.write_text(
        json.dumps(
            {
                "source_version": "independent-curator-v11",
                "database": {
                    "sha256": hashlib.sha256(database.read_bytes()).hexdigest()
                },
                "corpus": {
                    "record_count": 1,
                    "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                },
                "independence_declarations": required_false,
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
    files = (corpus, attestation, aggregate, execution)
    sha256s = tmp_path / "SHA256SUMS.txt"
    sha256s.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in files
        ),
        encoding="utf-8",
    )

    pack, declaration = FREEZE_MODULE._validate_independent_v11_attestation(
        corpus_path=corpus,
        database_path=database,
        attestation_path=attestation,
        aggregate_audit_path=aggregate,
        execution_audit_path=execution,
        sha256s_path=sha256s,
        n_rows=1,
    )
    assert pack["dataset_role"] == "independent_curation_input_only"
    assert declaration["source_version"] == "independent-curator-v11"

    execution.write_text('{"id":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256SUMS mismatch"):
        FREEZE_MODULE._validate_independent_v11_attestation(
            corpus_path=corpus,
            database_path=database,
            attestation_path=attestation,
            aggregate_audit_path=aggregate,
            execution_audit_path=execution,
            sha256s_path=sha256s,
            n_rows=1,
        )


def _gate_result(n, correct):
    return {"n": n, "n_correct": correct}


def _hard_details(n, correct):
    return [
        {"correct": index < correct, "engine": "rule"}
        for index in range(n)
    ]


def test_blind_gate_overall_uses_strict_integer_boundary():
    at_85_percent = BLIND_MODULE._evaluate_gates(
        _gate_result(120, 102), _hard_details(40, 40)
    )
    above_85_percent = BLIND_MODULE._evaluate_gates(
        _gate_result(120, 103), _hard_details(40, 40)
    )

    assert at_85_percent["overall_passed"] is False
    assert at_85_percent["passed"] is False
    assert at_85_percent["gate_reason"] == "overall_failed"
    assert above_85_percent["overall_passed"] is True
    assert above_85_percent["passed"] is True


def test_blind_gate_hard_boundary_and_reason_are_exact():
    below_hard_gate = BLIND_MODULE._evaluate_gates(
        _gate_result(120, 103), _hard_details(40, 27)
    )
    at_hard_gate = BLIND_MODULE._evaluate_gates(
        _gate_result(120, 103), _hard_details(40, 28)
    )
    too_small = BLIND_MODULE._evaluate_gates(
        _gate_result(120, 103), _hard_details(39, 39)
    )

    assert below_hard_gate["hard_passed"] is False
    assert below_hard_gate["gate_reason"] == "hard_failed"
    assert at_hard_gate["hard_passed"] is True
    assert at_hard_gate["passed"] is True
    assert too_small["hard_subset_present"] is False
    assert too_small["gate_reason"] == "hard_subset_insufficient"


def test_near_duplicate_question_family_cannot_be_split():
    rows = [
        {
            "id": "first",
            "question": "list patients from shanghai aged at least eighteen",
            "gold_sql": "SELECT patient_id FROM patients WHERE age >= 18",
            "difficulty": "hard",
            "category": "filtering",
        },
        {
            "id": "second",
            "question": "list patients from shanghai aged at least eighteen years",
            "gold_sql": "SELECT COUNT(*) FROM patients WHERE age >= 18",
            "difficulty": "hard",
            "category": "aggregation_grouping",
        },
    ]

    groups, stats = FREEZE_MODULE._partition_groups(rows)

    assert any(indices == [0, 1] for indices in groups.values())
    assert stats["n_near_duplicate_links"] == 1
    assert stats["n_partition_families"] == 1
