import json
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("sqlglot")

from scripts import project_nl2sql_public_donors as projection


def _donor(constructs):
    return {
        "dataset": "fixture",
        "source_revision": "a" * 40,
        "source_file": "development/train.json",
        "source_split": "train",
        "source_row_index": 1,
        "source_id_sha256": "1" * 64,
        "source_db_id_sha256": "2" * 64,
        "question_sha256": "3" * 64,
        "sql_sha256": "4" * 64,
        "ast_skeleton_sha256": "5" * 64,
        "constructs": list(constructs),
    }


def test_projection_declares_exact_300_item_quotas():
    assert sum(projection.CATEGORY_QUOTAS.values()) == 300
    difficulties = Counter()
    for category, quota in projection.CATEGORY_QUOTAS.items():
        sequence = projection._difficulty_sequence(category)
        assert len(sequence) == quota
        difficulties.update(sequence)
    assert dict(difficulties) == projection.EXPECTED_DIFFICULTY_QUOTAS


def test_projection_category_validator_accepts_cte_and_window():
    projection._validate_category_sql(
        "cte",
        "WITH x AS (SELECT patient_id FROM patients) "
        "SELECT patient_id FROM x",
    )
    projection._validate_category_sql(
        "window",
        "SELECT patient_id, ROW_NUMBER() OVER "
        "(ORDER BY patient_id) AS n FROM patients",
    )
    with pytest.raises(ValueError, match="lacks its declared construct"):
        projection._validate_category_sql(
            "window",
            "SELECT patient_id FROM patients",
        )


def test_donor_selector_requires_hash_bound_isolated_index():
    all_constructs = sorted(set(projection.DONOR_CONSTRUCT.values()))
    private = {
        "kind": "private_authoritative_nl2sql_donor_index",
        "contains_source_questions_or_sql": False,
        "official_source_test_content_parsed": False,
        "aggregate": {"rows": 1},
        "donors": [_donor(all_constructs)],
    }
    public = {
        "kind": "public_authoritative_nl2sql_donor_index_attestation",
        "private_index_sha256": "a" * 64,
        "official_source_test_content_parsed": False,
        "aggregate": {"rows": 1},
    }
    selector = projection.DonorSelector(private, public, "a" * 64)
    selected = selector.take("filtering")
    assert "question" not in selected
    assert "sql" not in selected
    public["private_index_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="binding failed"):
        projection.DonorSelector(private, public, "a" * 64)


def test_projection_public_receipt_redacts_private_rows_and_is_create_once(
    tmp_path,
):
    rows = [
        {
            "id": "one",
            "question": "敏感问题",
            "gold_sql": "SELECT patient_id FROM patients",
            "difficulty": "medium",
            "category": "filtering",
            "projection_provenance": {"private": True},
        }
    ]
    evidence = {
        "aggregate": {
            "n_items": 1,
            "all_gold_sql_validated": True,
            "all_gold_sql_executed_readonly_without_truncation": True,
            "all_result_sets_nonempty": True,
            "all_predicate_literals_grounded_in_question": True,
        },
        "bindings": {
            "database": {"sha256": "a" * 64},
            "source_tree_sha256": "b" * 64,
        },
    }
    private = tmp_path / "outputs" / "private" / "rid" / "projection"
    public = (
        tmp_path
        / "outputs"
        / "remote_runs"
        / "rid"
        / "projection.json"
    )
    projection.publish_projection(
        rows=rows,
        evidence=evidence,
        private_output_dir=private,
        public_output=public,
        created_at="2026-07-26T08:00:00Z",
    )
    public_text = public.read_text(encoding="utf-8")
    assert "敏感问题" not in public_text
    assert "SELECT patient_id" not in public_text
    assert json.loads(public_text)["contains_questions_or_gold_sql"] is False
    with pytest.raises(FileExistsError):
        projection.publish_projection(
            rows=rows,
            evidence=evidence,
            private_output_dir=private,
            public_output=public,
            created_at="2026-07-26T08:00:00Z",
        )


def test_projector_source_does_not_import_rule_engine():
    source = Path(projection.__file__).read_text(encoding="utf-8")
    assert "from dki.analysis.nl2sql import" not in source
    assert "data/nl2sql_eval" not in source
