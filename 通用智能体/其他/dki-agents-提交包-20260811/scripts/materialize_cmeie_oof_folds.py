"""Materialize the five fit-internal CMeIE OOF folds create-once.

The OOF manifest owns only source row indices.  This script turns those
indices into auditable train/validation JSONL shards without reading
calibration, outer, or official-dev data.  Every fit row is validation data in
exactly one fold and training data in the other four folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_cmeie_split_manifest import _publish_directory_create_once


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_row(row: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _shard_bytes(rows: Sequence[dict[str, Any]], indices: Sequence[int]) -> bytes:
    return b"".join(_canonical_row(rows[index]) for index in indices)


def validate_oof_contract(
    *,
    rows: Sequence[dict[str, Any]],
    source_train: Path,
    base_split: dict[str, Any],
    base_split_path: Path,
    oof_manifest: dict[str, Any],
) -> tuple[set[int], list[dict[str, Any]]]:
    """Validate source/split/fold bindings without accessing another role."""

    if _sha256(source_train) != base_split.get("source_train_sha256"):
        raise ValueError("base split is bound to different CMeIE source bytes")
    if oof_manifest.get("source_train_sha256") != _sha256(source_train):
        raise ValueError("OOF manifest is bound to different CMeIE source bytes")
    if oof_manifest.get("base_split_manifest_sha256") != _sha256(base_split_path):
        raise ValueError("OOF manifest is bound to a different base split")
    fit_values = base_split.get("splits", {}).get("fit", {}).get("indices")
    if not isinstance(fit_values, list) or not fit_values:
        raise ValueError("base split lacks fit indices")
    fit = {int(value) for value in fit_values}
    if len(fit) != len(fit_values) or any(
        index < 0 or index >= len(rows) for index in fit
    ):
        raise ValueError("base fit indices are invalid")
    partitions = oof_manifest.get("partitions")
    if not isinstance(partitions, list) or len(partitions) != 5:
        raise ValueError("OOF manifest must contain exactly five folds")
    validation_seen: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for expected_fold, raw in enumerate(partitions):
        if not isinstance(raw, dict) or raw.get("fold") != expected_fold:
            raise ValueError("OOF folds must be ordered 0..4")
        train = [int(value) for value in raw.get("train_indices", [])]
        validation = [int(value) for value in raw.get("validation_indices", [])]
        train_set, validation_set = set(train), set(validation)
        if (
            len(train_set) != len(train)
            or len(validation_set) != len(validation)
            or not train_set
            or not validation_set
            or train_set & validation_set
            or train_set | validation_set != fit
            or validation_seen & validation_set
            or raw.get("family_intersection") not in ([], ())
        ):
            raise ValueError("OOF fold violates fit-only disjoint partitioning")
        validation_seen.update(validation_set)
        normalized.append(
            {
                **raw,
                "train_indices": sorted(train_set),
                "validation_indices": sorted(validation_set),
            }
        )
    if validation_seen != fit:
        raise ValueError("OOF validation folds do not cover fit exactly once")
    return fit, normalized


def materialize(
    *,
    source_train: Path,
    base_split_path: Path,
    oof_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    source_train = Path(source_train).resolve(strict=True)
    base_split_path = Path(base_split_path).resolve(strict=True)
    oof_manifest_path = Path(oof_manifest_path).resolve(strict=True)
    output = Path(output).resolve()
    if os.path.lexists(output):
        raise FileExistsError(f"OOF fold directory already exists: {output}")
    rows = [
        json.loads(line)
        for line in source_train.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    base = json.loads(base_split_path.read_text(encoding="utf-8"))
    oof = json.loads(oof_manifest_path.read_text(encoding="utf-8"))
    fit, partitions = validate_oof_contract(
        rows=rows,
        source_train=source_train,
        base_split=base,
        base_split_path=base_split_path,
        oof_manifest=oof,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".staging", dir=output.parent
        )
    )
    try:
        fold_receipts: list[dict[str, Any]] = []
        for partition in partitions:
            fold = int(partition["fold"])
            fold_dir = staging / f"fold-{fold:02d}"
            fold_dir.mkdir()
            train_raw = _shard_bytes(rows, partition["train_indices"])
            validation_raw = _shard_bytes(rows, partition["validation_indices"])
            train_path = fold_dir / "train.jsonl"
            validation_path = fold_dir / "validation.jsonl"
            train_path.write_bytes(train_raw)
            validation_path.write_bytes(validation_raw)
            receipt = {
                "schema_version": 1,
                "fold": fold,
                "dataset_role": "fit_internal_oof",
                "n_train": len(partition["train_indices"]),
                "n_validation": len(partition["validation_indices"]),
                "train_indices_sha256": hashlib.sha256(
                    "\n".join(map(str, partition["train_indices"])).encode("utf-8")
                ).hexdigest(),
                "validation_indices_sha256": hashlib.sha256(
                    "\n".join(map(str, partition["validation_indices"])).encode("utf-8")
                ).hexdigest(),
                "train_sha256": hashlib.sha256(train_raw).hexdigest(),
                "validation_sha256": hashlib.sha256(validation_raw).hexdigest(),
                "family_intersection": [],
                "calibration_accessed": False,
                "outer_accessed": False,
                "official_dev_accessed": False,
            }
            (fold_dir / "receipt.json").write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            fold_receipts.append(receipt)
        root_receipt = {
            "schema_version": 1,
            "kind": "cmeie_fit_internal_oof_materialization",
            "source_train_sha256": _sha256(source_train),
            "base_split_manifest_sha256": _sha256(base_split_path),
            "oof_manifest_sha256": _sha256(oof_manifest_path),
            "n_fit": len(fit),
            "n_folds": 5,
            "validation_coverage_exactly_once": True,
            "folds": fold_receipts,
            "calibration_accessed": False,
            "outer_accessed": False,
            "official_dev_accessed": False,
        }
        (staging / "receipt.json").write_text(
            json.dumps(root_receipt, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        _publish_directory_create_once(staging, output)
        staging = None
        return root_receipt
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--oof-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize(
        source_train=args.source_train,
        base_split_path=args.base_split,
        oof_manifest_path=args.oof_manifest,
        output=args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
