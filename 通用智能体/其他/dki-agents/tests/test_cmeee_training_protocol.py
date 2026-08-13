import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_ner_cmeee.py"
SPEC = importlib.util.spec_from_file_location("train_ner_cmeee", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(text, index=0):
    return {"text": text, "entities": [], "index": index}


def test_internal_split_is_deterministic_and_text_group_disjoint():
    rows = [_row(f"病历{i}", i) for i in range(20)] + [_row("重复病历", 20), _row("重复病历", 21)]
    fit_a, val_a = MODULE._split_train_internal(rows, 0.2, 42)
    fit_b, val_b = MODULE._split_train_internal(rows, 0.2, 42)
    assert [row["index"] for row in fit_a] == [row["index"] for row in fit_b]
    assert [row["index"] for row in val_a] == [row["index"] for row in val_b]
    assert {row["text"] for row in fit_a}.isdisjoint({row["text"] for row in val_a})
    duplicate_locations = [
        any(row["text"] == "重复病历" for row in partition)
        for partition in (fit_a, val_a)
    ]
    assert sum(duplicate_locations) == 1


def test_internal_split_changes_with_seed_without_losing_rows():
    rows = [_row(f"病历{i}", i) for i in range(30)]
    fit_a, val_a = MODULE._split_train_internal(rows, 0.2, 1)
    fit_b, val_b = MODULE._split_train_internal(rows, 0.2, 2)
    assert {row["index"] for row in fit_a + val_a} == set(range(30))
    assert {row["index"] for row in fit_b + val_b} == set(range(30))
    assert {row["index"] for row in val_a} != {row["index"] for row in val_b}


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.1])
def test_internal_split_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError):
        MODULE._split_train_internal([_row("甲"), _row("乙")], ratio, 42)


def test_threshold_selection_tie_prefers_fixed_proximity_then_precision():
    records = [
        {"threshold": -1.0, "micro_f1": 0.8, "precision": 0.9},
        {"threshold": 0.0, "micro_f1": 0.8, "precision": 0.7},
        {"threshold": 1.0, "micro_f1": 0.8, "precision": 0.95},
    ]
    assert MODULE._select_threshold(records, 0.0)["threshold"] == 0.0
    assert MODULE._select_threshold([records[0], records[2]], 0.0)["threshold"] == 1.0


def test_metric_record_uses_exact_counts():
    metric = MODULE._metric_record(8, 2, 4, -0.5)
    assert metric["precision"] == pytest.approx(0.8)
    assert metric["recall"] == pytest.approx(2 / 3)
    assert metric["micro_f1"] == pytest.approx(8 / 11)


def test_metric_record_reports_macro_and_worst_type():
    metric = MODULE._metric_record(
        3, 2, 1, 0.0,
        per_type_counts={"dis": [3, 1, 0], "sym": [0, 1, 1]},
    )
    assert metric["per_type"]["dis"]["micro_f1"] == pytest.approx(6 / 7)
    assert metric["per_type"]["sym"]["micro_f1"] == 0
    assert metric["macro_f1"] == pytest.approx(3 / 7)
    assert metric["worst_type"] == {"type": "sym", "f1": 0, "support": 1}


def test_atomic_json_write_replaces_complete_document(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text('{"stale": true}', encoding="utf-8")
    MODULE._write_json_atomic(path, {"fresh": True})
    assert path.read_text(encoding="utf-8") == '{\n  "fresh": true\n}'
    assert not list(tmp_path.glob("*.tmp"))


def test_model_promotion_restores_old_directory_on_failed_replace(tmp_path, monkeypatch):
    output = tmp_path / "model"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    real_replace = MODULE.os.replace

    def fail_staging(source, destination):
        if ".staging-" in str(source) and Path(destination) == output:
            raise OSError("simulated promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(MODULE.os, "replace", fail_staging)
    with pytest.raises(OSError, match="simulated"):
        MODULE._promote_model_directory(
            output,
            lambda staging: (staging / "new.txt").write_text("new", encoding="utf-8"),
        )
    assert (output / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (output / "new.txt").exists()
