"""Build create-once five-fold family-disjoint OOF partitions inside CMeIE fit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_module():
    path = ROOT / "scripts" / "build_cmeie_split_manifest.py"
    spec = importlib.util.spec_from_file_location("_cmeie_split", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def make_manifest(rows: list[dict], source: Path, base: dict, base_path: Path, seed: str, folds: int = 5) -> dict:
    if sha(source) != base.get("source_train_sha256"):
        raise ValueError("base split is bound to different training data")
    fit = set(base["splits"]["fit"]["indices"])
    assignments = _split_module().family_assignments(rows)
    groups: dict[str, list[int]] = defaultdict(list)
    for index in sorted(fit):
        groups[assignments[index]].append(index)
    total_predicates = Counter(
        str(spo.get("predicate"))
        for indices in groups.values() for index in indices
        for spo in rows[index].get("spo_list", []) if spo.get("predicate")
    )
    bins = [[] for _ in range(folds)]
    sizes = [0] * folds
    predicates = [Counter() for _ in range(folds)]
    for family, indices in sorted(
        groups.items(), key=lambda item: (-len(item[1]), hashlib.sha256(f"{seed}\0{item[0]}".encode()).hexdigest())
    ):
        current = Counter(
            str(spo.get("predicate"))
            for index in indices for spo in rows[index].get("spo_list", []) if spo.get("predicate")
        )
        target = len(fit) / folds
        # Longest-family-first load balancing.  The old absolute-distance-to-
        # target objective filled one fold all the way to the target before it
        # touched the next fold, producing 3 non-empty members in a requested
        # 5-fold split.  Score the *post-add load* instead, including all
        # predicate loads, so an empty/light fold is preferred immediately.
        def load_score(fold: int) -> tuple[float, float, float, int]:
            size_load = (sizes[fold] + len(indices)) / max(1.0, target)
            predicate_load = sum(
                (predicates[fold][predicate] + current[predicate])
                / max(1.0, total / folds)
                for predicate, total in total_predicates.items()
            ) / max(1, len(total_predicates))
            return (
                size_load + predicate_load,
                size_load,
                predicate_load,
                fold,
            )

        choice = min(range(folds), key=load_score)
        bins[choice].extend(indices); sizes[choice] += len(indices); predicates[choice].update(current)
    payload = {"schema_version": 1, "seed": str(seed), "folds": folds, "source_train_sha256": sha(source), "base_split_manifest_sha256": sha(base_path), "fit_sha256": base["splits"]["fit"]["row_sha256"], "partitions": []}
    all_validation = set()
    for fold, validation in enumerate(bins):
        validation = sorted(validation); validation_set = set(validation); all_validation |= validation_set
        if not validation:
            raise RuntimeError("OOF partitioner produced an empty validation fold")
        train = sorted(fit - validation_set)
        validation_families = {assignments[index] for index in validation}
        train_families = {assignments[index] for index in train}
        if validation_families & train_families:
            raise RuntimeError("family leakage inside OOF fold")
        payload["partitions"].append({"fold": fold, "train_indices": train, "validation_indices": validation, "n_train": len(train), "n_validation": len(validation), "validation_size_error": abs(len(validation) - len(fit) / folds), "train_family_sha256": hashlib.sha256("\n".join(sorted(train_families)).encode()).hexdigest(), "validation_family_sha256": hashlib.sha256("\n".join(sorted(validation_families)).encode()).hexdigest(), "family_intersection": []})
    if all_validation != fit:
        raise RuntimeError("OOF validation partitions do not cover fit exactly once")
    payload["generator_sha256"] = sha(Path(__file__))
    return payload


def publish_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"OOF manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush(); os.fsync(handle.fileno()); temporary = Path(handle.name)
    try: os.link(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default="20260725")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.train.read_text(encoding="utf-8").splitlines() if line.strip()]
    base = json.loads(args.base_split.read_text(encoding="utf-8"))
    publish_once(args.output, make_manifest(rows, args.train, base, args.base_split, args.seed))
    print(json.dumps({"output": str(args.output), "folds": 5}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
