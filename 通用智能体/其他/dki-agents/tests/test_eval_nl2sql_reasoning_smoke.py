from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "eval_nl2sql_reasoning_smoke.py"
SPEC = importlib.util.spec_from_file_location("eval_nl2sql_reasoning_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _rows():
    return [
        {"id": "z", "question": "问题甲", "gold_sql": "SELECT 1", "category": "b"},
        {"id": "a", "question": "问题乙", "gold_sql": "SELECT 2", "category": "a"},
        {"id": "b", "question": "问题丙", "gold_sql": "SELECT 3", "category": "a"},
        {"id": "c", "question": "问题丁", "gold_sql": "SELECT 4", "category": "b"},
    ]


def test_selector_is_deterministic_and_does_not_mutate_rows():
    rows = _rows()
    original = copy.deepcopy(rows)
    first = MODULE._select_development_rows(rows, per_category=1)
    second = MODULE._select_development_rows(list(reversed(rows)), per_category=1)
    assert [(row["category"], row["question"]) for row in first] == [
        (row["category"], row["question"]) for row in second
    ]
    assert rows == original
    assert len(first) == 2


def test_selector_is_independent_of_ids_and_gold_sql():
    rows = _rows()
    selected = MODULE._select_development_rows(rows, per_category=1)
    changed = copy.deepcopy(rows)
    for index, row in enumerate(changed):
        row["id"] = f"changed-{index}"
        row["gold_sql"] = f"SELECT {100 + index}"
    selected_changed = MODULE._select_development_rows(changed, per_category=1)
    assert [row["question"] for row in selected] == [
        row["question"] for row in selected_changed
    ]


@pytest.mark.parametrize("value", [0, 11, True, "1"])
def test_selector_rejects_invalid_per_category(value):
    with pytest.raises(ValueError):
        MODULE._select_development_rows(_rows(), per_category=value)


def test_selector_requires_category():
    with pytest.raises(ValueError, match="category"):
        MODULE._select_development_rows(
            [{"question": "问题", "gold_sql": "SELECT 1"}],
            per_category=1,
        )


def test_public_aggregate_contains_only_counts():
    result = {
        "n": 2,
        "n_correct": 1,
        "accuracy": 0.5,
        "n_answered": 1,
        "n_abstained": 1,
        "coverage": 0.5,
        "execution_accuracy_answered": 1.0,
        "strict_result_contract": True,
        "by_difficulty": {"hard": {"n": 2, "correct": 1, "accuracy": 0.5}},
        "details": [
            {
                "id": "private-a",
                "question": "private question",
                "gold_sql": "SELECT secret",
                "pred_sql": "SELECT prediction",
                "category": "join",
                "difficulty": "hard",
                "engine": "llm",
                "correct": True,
            },
            {
                "id": "private-b",
                "question": "another private question",
                "gold_sql": "SELECT hidden",
                "pred_sql": "",
                "category": "join",
                "difficulty": "hard",
                "engine": "none",
                "correct": False,
            },
        ],
    }
    aggregate = MODULE._aggregate_result(result)
    encoded = json.dumps(aggregate, ensure_ascii=False)
    assert aggregate["by_category"]["join"] == {
        "n": 2,
        "correct": 1,
        "accuracy": 0.5,
        "answered": 1,
        "coverage": 0.5,
    }
    for secret in ("private-a", "private question", "SELECT secret", "prediction"):
        assert secret not in encoded


def test_script_has_no_holdout_content_reader():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "holdout.jsonl" not in source
    assert "holdout_sha256" not in source
    assert "read_holdout" not in source


def test_smoke_keeps_independent_v3_source_identity_guards():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "independent_curated_v3_four_table_blind_freeze" in source
    assert 'freeze.get("source_version", "")' in source
    assert "len(freeze_source_tree_sha256) != 64" in source
