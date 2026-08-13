from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dki.nlp.cmeie_overfit_protocol import (
    canonical_overfit_json_bytes,
    load_canonical_overfit_preflight,
    overfit_gate_passed,
    recompute_overfit_metric,
)


def _metric(tp: int = 98, fp: int = 2, fn: int = 2) -> dict:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        0.0
        if 2 * tp + fp + fn == 0
        else 2 * tp / (2 * tp + fp + fn)
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def test_count_derived_metric_and_gate_use_exact_counts():
    metric = _metric(19, 1, 1)
    assert recompute_overfit_metric(metric) == metric
    assert overfit_gate_passed(
        metric=metric,
        schema_on=True,
        minimum_train_f1=0.95,
    )
    assert not overfit_gate_passed(
        metric=metric,
        schema_on=False,
        minimum_train_f1=0.95,
    )


def test_self_reported_high_f1_cannot_override_zero_true_positives():
    forged = {
        "tp": 0,
        "fp": 100,
        "fn": 100,
        "precision": 0.98,
        "recall": 0.98,
        "f1": 0.98,
    }
    with pytest.raises(ValueError, match="disagrees with tp/fp/fn"):
        recompute_overfit_metric(forged)
    with pytest.raises(ValueError, match="disagrees with tp/fp/fn"):
        overfit_gate_passed(
            metric=forged,
            schema_on=True,
            minimum_train_f1=0.95,
        )


@pytest.mark.parametrize("field", ["tp", "fp", "fn"])
def test_counts_are_strict_nonnegative_integers(field: str):
    for invalid in (True, -1, 1.0):
        metric = _metric()
        metric[field] = invalid
        with pytest.raises(ValueError, match="non-negative integer"):
            recompute_overfit_metric(metric)


def test_metric_field_set_is_exact():
    metric = _metric()
    metric["score"] = metric["f1"]
    with pytest.raises(ValueError, match="fields are invalid"):
        recompute_overfit_metric(metric)


def test_canonical_loader_binds_exact_bytes(tmp_path: Path):
    path = tmp_path / "overfit.json"
    payload = {"schema_version": 1, "metric": _metric()}
    raw = canonical_overfit_json_bytes(payload)
    path.write_bytes(raw)

    loaded, binding = load_canonical_overfit_preflight(path)
    assert loaded == payload
    assert binding["canonical_path"] == str(path)
    assert binding["size"] == len(raw)


def test_canonical_loader_rejects_missing_hardlink_and_noncanonical(
    tmp_path: Path,
):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_canonical_overfit_preflight(missing)

    path = tmp_path / "overfit.json"
    payload = {"schema_version": 1, "metric": _metric()}
    path.write_bytes(canonical_overfit_json_bytes(payload))
    os.link(path, tmp_path / "alias.json")
    with pytest.raises(ValueError, match="multiple hard links"):
        load_canonical_overfit_preflight(path)

    path.unlink()
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not canonical JSON"):
        load_canonical_overfit_preflight(path)


def test_canonical_loader_rejects_duplicate_keys(tmp_path: Path):
    path = tmp_path / "overfit.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        load_canonical_overfit_preflight(path)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_canonical_loader_rejects_manually_injected_nonfinite_value(
    tmp_path: Path,
    literal: str,
):
    path = tmp_path / "overfit.json"
    path.write_text(
        f'{{"diagnostic":{literal}}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-JSON or non-finite"):
        load_canonical_overfit_preflight(path)
