import pytest

from dki import cli
from dki.paths import OUTPUT_DIR


def test_runtime_evaluations_cannot_overwrite_promoted_claim_receipts():
    kg_path = cli.runtime_evaluation_path("kg_extraction")
    nl2sql_path = cli.runtime_evaluation_path("nl2sql")

    assert kg_path == OUTPUT_DIR / "runtime_eval" / "kg_extraction_eval.json"
    assert nl2sql_path == OUTPUT_DIR / "runtime_eval" / "nl2sql_eval.json"
    assert kg_path != OUTPUT_DIR / "eval" / "kg_extraction_eval.json"
    assert nl2sql_path != OUTPUT_DIR / "eval" / "nl2sql_eval.json"
    assert all("runtime_eval" in str(path) for path in (kg_path, nl2sql_path))


def test_runtime_evaluation_path_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown runtime evaluation kind"):
        cli.runtime_evaluation_path("unknown")
