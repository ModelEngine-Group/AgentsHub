"""Apply the fixed two-seed PRGC calibration gate without opening outer data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"PRGC gate evidence exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    args = parser.parse_args(argv)
    selections = [path.resolve() for path in args.selection]
    if len(selections) != 2 or len(set(selections)) != 2:
        raise SystemExit("PRGC early gate requires exactly two distinct seed selections")
    if args.output.exists():
        raise SystemExit(f"PRGC gate output exists: {args.output}")
    loaded: list[dict[str, Any]] = []
    for path in selections:
        if not path.is_file():
            raise SystemExit(f"PRGC selection is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("kind") != "cmeie_prgc_calibration_selection"
            or payload.get("model_family") != "prgc"
            or payload.get("outer_accessed") is not False
            or payload.get("official_dev_accessed") is not False
        ):
            raise SystemExit(f"invalid PRGC selection: {path}")
        loaded.append(payload)
    seeds = [item.get("seed") for item in loaded]
    if set(seeds) != {42, 43}:
        raise SystemExit("PRGC early gate requires seed 42 and seed 43")
    fields = ("fit_sha256", "calibration_sha256", "split_manifest_sha256", "predicates")
    if any(loaded[0].get(field) != loaded[1].get(field) for field in fields):
        raise SystemExit("PRGC seed selections differ in a data or label binding")
    metrics = [
        {
            "seed": item["seed"],
            "f1": float(item["calibration_metric"]["f1"]),
            "precision": float(item["calibration_metric"]["precision"]),
            "recall": float(item["calibration_metric"]["recall"]),
            "selection": str(path),
            "selection_sha256": _sha(path),
        }
        for item, path in zip(loaded, selections)
    ]
    both_below = all(item["f1"] < 0.45 for item in metrics)
    payload = {
        "schema_version": 1,
        "kind": "cmeie_prgc_seed_early_gate",
        "source_tree_sha256": args.source_tree_sha256,
        "dataset_role": "calibration_seed_gate",
        "selection_used": True,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "minimum_calibration_f1": 0.45,
        "metrics": sorted(metrics, key=lambda item: item["seed"]),
        "decision": "BLOCKED_MODEL_QUALITY" if both_below else "ADVANCE_TO_SEED_44",
        "advance_to_seed44": not both_below,
    }
    _write_once(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 2 if both_below else 0


if __name__ == "__main__":
    raise SystemExit(main())
