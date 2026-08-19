from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.materialize_cmeie_oof_folds import materialize


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    rows = [
        {"text": f"family-{index}@body", "spo_list": [{"predicate": "P"}]}
        for index in range(10)
    ]
    source = tmp_path / "train.jsonl"
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    base = {
        "source_train_sha256": _sha(source),
        "splits": {"fit": {"indices": list(range(10))}},
    }
    base_path = tmp_path / "split.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    partitions = []
    for fold in range(5):
        validation = [2 * fold, 2 * fold + 1]
        partitions.append(
            {
                "fold": fold,
                "train_indices": [i for i in range(10) if i not in validation],
                "validation_indices": validation,
                "family_intersection": [],
            }
        )
    oof = {
        "source_train_sha256": _sha(source),
        "base_split_manifest_sha256": _sha(base_path),
        "partitions": partitions,
    }
    oof_path = tmp_path / "oof.json"
    oof_path.write_text(json.dumps(oof), encoding="utf-8")
    return source, base_path, oof_path


def test_materialization_covers_fit_once_and_is_create_once(tmp_path: Path) -> None:
    source, base, oof = _fixture(tmp_path)
    output = tmp_path / "folds"
    receipt = materialize(
        source_train=source,
        base_split_path=base,
        oof_manifest_path=oof,
        output=output,
    )
    assert receipt["n_fit"] == 10
    assert receipt["validation_coverage_exactly_once"] is True
    validation_texts = []
    for fold in range(5):
        fold_dir = output / f"fold-{fold:02d}"
        validation_texts.extend(
            json.loads(line)["text"]
            for line in (fold_dir / "validation.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        )
        assert json.loads((fold_dir / "receipt.json").read_text())["family_intersection"] == []
    assert sorted(validation_texts) == [f"family-{i}@body" for i in range(10)]
    with pytest.raises(FileExistsError):
        materialize(
            source_train=source,
            base_split_path=base,
            oof_manifest_path=oof,
            output=output,
        )


def test_materialization_rejects_duplicate_validation_members(tmp_path: Path) -> None:
    source, base, oof_path = _fixture(tmp_path)
    oof = json.loads(oof_path.read_text())
    oof["partitions"][1]["validation_indices"] = oof["partitions"][0][
        "validation_indices"
    ]
    oof["partitions"][1]["train_indices"] = oof["partitions"][0][
        "train_indices"
    ]
    oof_path.write_text(json.dumps(oof), encoding="utf-8")
    with pytest.raises(ValueError, match="partitioning"):
        materialize(
            source_train=source,
            base_split_path=base,
            oof_manifest_path=oof_path,
            output=tmp_path / "bad",
        )
