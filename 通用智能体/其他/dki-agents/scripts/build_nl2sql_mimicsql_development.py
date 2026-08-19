"""Build a create-once MIMICSQL train/dev projection without opening test.

The CLI deliberately accepts only explicit ``train.json`` and ``dev.json``
paths.  Official-test provenance is represented by the Git blob identity from
the pinned public repository; there is no test path argument and no directory
walk.  Detailed questions and gold SQL remain under ``outputs/private``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql_css_projection import (  # noqa: E402
    ProjectedItem,
    load_target_domains,
)
from dki.analysis.nl2sql_mimicsql_projection import (  # noqa: E402
    MIMICSQL_BENCHMARK,
    MIMICSQL_PROJECTION_SEED,
    MIMICSQL_SOURCE_REVISION,
    NL2SQL_V3_PARTITION_SEED,
    MIMICSQLProjectionRejected,
    deduplicate_mimicsql_items,
    project_mimicsql_row,
)


HARD_DIFFICULTIES = {"hard", "extra", "extra-hard"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _stage(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def _publish_pair_once(
    private_output: Path,
    private_bytes: bytes,
    public_output: Path,
    public_bytes: bytes,
) -> None:
    private_output = private_output.resolve()
    public_output = public_output.resolve()
    for path in (private_output, public_output):
        if path.exists():
            raise FileExistsError(
                f"MIMICSQL development artifact already exists: {path}"
            )
    private_temp = _stage(private_output, private_bytes)
    public_temp = _stage(public_output, public_bytes)
    private_published = False
    try:
        os.link(private_temp, private_output)
        private_published = True
        os.link(public_temp, public_output)
    except BaseException:
        if private_published:
            try:
                if (
                    private_output.is_file()
                    and _sha256(private_output)
                    == hashlib.sha256(private_bytes).hexdigest()
                ):
                    private_output.unlink()
            except OSError:
                pass
        raise
    finally:
        private_temp.unlink(missing_ok=True)
        public_temp.unlink(missing_ok=True)


def _validate_source_paths(train_path: Path, dev_path: Path) -> None:
    train_path = train_path.resolve()
    dev_path = dev_path.resolve()
    if train_path.name != "train.json" or dev_path.name != "dev.json":
        raise ValueError(
            "only explicit MIMICSQL train.json and dev.json are accepted"
        )
    if (
        train_path.parent != dev_path.parent
        or train_path.parent.name != "mimicsql_natural_v2"
    ):
        raise ValueError(
            "train/dev must share the mimicsql_natural_v2 directory"
        )
    if not train_path.is_file() or not dev_path.is_file():
        raise ValueError("MIMICSQL train/dev source file is missing")


def _load_jsonl(
    path: Path,
    *,
    split: str,
    seen_keys: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid MIMICSQL {split} JSON on line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"MIMICSQL {split} line {line_number} is not an object"
                )
            key = str(row.get("key", "")).strip()
            if not key:
                raise ValueError(
                    f"MIMICSQL {split} line {line_number} has no key"
                )
            if key in seen_keys:
                raise ValueError(f"duplicate MIMICSQL source key: {key}")
            seen_keys.add(key)
            rows.append(row)
    if not rows:
        raise ValueError(f"MIMICSQL {split} source is empty")
    return rows


def _balanced_select(
    items: list[ProjectedItem],
    maximum_items: int,
) -> list[ProjectedItem]:
    buckets: dict[tuple[str, str, str, str], list[ProjectedItem]] = (
        defaultdict(list)
    )
    for item in items:
        provenance = item.item["provenance"]
        key = (
            str(provenance["source_split"]),
            str(item.item["difficulty"]),
            str(item.item["category"]),
            item.skeleton_sha256,
        )
        buckets[key].append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item.item["id"])
    selected: list[ProjectedItem] = []
    keys = sorted(buckets)
    while len(selected) < maximum_items:
        advanced = False
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                advanced = True
                if len(selected) == maximum_items:
                    break
        if not advanced:
            break
    return selected


def _aggregate(items: Iterable[ProjectedItem]) -> dict[str, Any]:
    rows = list(items)
    difficulties = Counter(item.item["difficulty"] for item in rows)
    categories = Counter(item.item["category"] for item in rows)
    splits = Counter(
        item.item["provenance"]["source_split"] for item in rows
    )
    return {
        "n": len(rows),
        "n_hard": sum(
            count
            for difficulty, count in difficulties.items()
            if difficulty in HARD_DIFFICULTIES
        ),
        "source_split_counts": dict(sorted(splits.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "category_counts": dict(sorted(categories.items())),
        "unique_target_ast_skeletons": len(
            {item.skeleton_sha256 for item in rows}
        ),
    }


def build_mimicsql_development(
    *,
    train_path: Path,
    dev_path: Path,
    db_path: Path,
    private_output: Path,
    public_output: Path,
    source_tree_sha256: str,
    source_revision: str,
    official_test_git_blob: str,
    created_at: str,
    minimum_items: int,
    maximum_items: int,
) -> dict[str, Any]:
    """Project train/dev only and publish private/public artifacts atomically."""
    if not re.fullmatch(r"[0-9a-f]{64}", source_tree_sha256):
        raise ValueError(
            "source_tree_sha256 must be a full lowercase SHA-256"
        )
    if source_revision != MIMICSQL_SOURCE_REVISION:
        raise ValueError("MIMICSQL source revision is not the pinned commit")
    if not re.fullmatch(r"[0-9a-f]{40}", official_test_git_blob):
        raise ValueError(
            "official_test_git_blob must be a Git SHA-1 from the pinned tree"
        )
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
        created_at,
    ):
        raise ValueError("created_at must be UTC YYYY-MM-DDTHH:MM:SSZ")
    if minimum_items < 150:
        raise ValueError("minimum_items must be at least 150")
    if maximum_items < minimum_items:
        raise ValueError("maximum_items must be at least minimum_items")
    if "outputs/private/" not in private_output.resolve().as_posix():
        raise ValueError(
            "detailed MIMICSQL development data must remain private"
        )
    if "outputs/private/" in public_output.resolve().as_posix():
        raise ValueError("public receipt cannot be under outputs/private")
    for path in (private_output.resolve(), public_output.resolve()):
        if path.exists():
            raise FileExistsError(
                f"MIMICSQL development artifact already exists: {path}"
            )
    if not db_path.is_file():
        raise ValueError(f"target database not found: {db_path}")
    _validate_source_paths(train_path, dev_path)

    seen_keys: set[str] = set()
    source_rows = {
        "train": _load_jsonl(
            train_path,
            split="train",
            seen_keys=seen_keys,
        ),
        "dev": _load_jsonl(
            dev_path,
            split="dev",
            seen_keys=seen_keys,
        ),
    }
    domains = load_target_domains(db_path)
    projected: list[ProjectedItem] = []
    rejection_counts: Counter[str] = Counter()
    for split in ("train", "dev"):
        for row in source_rows[split]:
            try:
                projected.append(
                    project_mimicsql_row(
                        row,
                        domains,
                        db_path,
                        source_split=split,
                        source_revision=source_revision,
                    )
                )
            except MIMICSQLProjectionRejected as exc:
                rejection_counts[f"{split}/{exc.reason}"] += 1

    deduplicated, deduplication = deduplicate_mimicsql_items(projected)
    development = [
        item
        for item in deduplicated
        if item.skeleton_role == "development"
    ]
    holdout_capacity_proxy = [
        item for item in deduplicated if item.skeleton_role == "holdout"
    ]
    development_skeletons = {
        item.skeleton_sha256 for item in development
    }
    holdout_skeletons = {
        item.skeleton_sha256 for item in holdout_capacity_proxy
    }
    if development_skeletons & holdout_skeletons:
        raise ValueError("MIMICSQL target AST roles overlap")
    selected = _balanced_select(development, maximum_items)
    if len(selected) < minimum_items:
        raise ValueError(
            "MIMICSQL development projection is below the required floor: "
            f"{len(selected)} < {minimum_items}"
        )

    private_bytes = _jsonl_bytes(item.item for item in selected)
    receipt = {
        "schema_version": 1,
        "kind": "mimicsql_v3_four_table_development_receipt",
        "dataset_role": "development_selection_used",
        "created_at": created_at,
        "source_tree_sha256": source_tree_sha256,
        "benchmark": {
            "name": MIMICSQL_BENCHMARK,
            "source_revision": source_revision,
            "license": "MIT",
        },
        "partition": {
            "version": "v3",
            "seed": NL2SQL_V3_PARTITION_SEED,
            "projection_seed": MIMICSQL_PROJECTION_SEED,
            "development_holdout_skeleton_intersection": 0,
        },
        "hashes": {
            "builder_sha256": _sha256(Path(__file__)),
            "projection_module_sha256": _sha256(
                ROOT
                / "dki"
                / "analysis"
                / "nl2sql_mimicsql_projection.py"
            ),
            "protocol_sha256": _sha256(
                ROOT / "dki" / "analysis" / "nl2sql_protocol.py"
            ),
            "target_db_sha256": _sha256(db_path),
            "train_sha256": _sha256(train_path),
            "dev_sha256": _sha256(dev_path),
            "private_development_sha256": hashlib.sha256(
                private_bytes
            ).hexdigest(),
        },
        "source_rows": {
            split: len(rows) for split, rows in source_rows.items()
        },
        "projection": {
            "n_before_deduplication": len(projected),
            "n_after_deduplication": len(deduplicated),
            "deduplication": deduplication,
            "rejection_counts": dict(sorted(rejection_counts.items())),
        },
        "development_candidates": _aggregate(development),
        "development_selected": _aggregate(selected),
        "train_dev_holdout_role_capacity_proxy": _aggregate(
            holdout_capacity_proxy
        ),
        "official_test": {
            "source_split": "test",
            "git_blob_sha1": official_test_git_blob,
            "path_argument_accepted_by_this_builder": False,
            "content_opened_or_hashed_by_this_builder": False,
            "consumed": False,
        },
        "privacy": {
            "public_receipt_contains_questions": False,
            "public_receipt_contains_gold_sql": False,
            "public_receipt_contains_error_samples": False,
            "private_projection_allowed_in_submission": False,
        },
    }
    _publish_pair_once(
        private_output,
        private_bytes,
        public_output,
        _json_bytes(receipt),
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument(
        "--source-revision",
        default=MIMICSQL_SOURCE_REVISION,
    )
    parser.add_argument("--official-test-git-blob", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--minimum-items", type=int, default=150)
    parser.add_argument("--maximum-items", type=int, default=2000)
    args = parser.parse_args()
    try:
        receipt = build_mimicsql_development(
            train_path=args.train,
            dev_path=args.dev,
            db_path=args.db,
            private_output=args.private_output,
            public_output=args.public_output,
            source_tree_sha256=args.source_tree_sha256,
            source_revision=args.source_revision,
            official_test_git_blob=args.official_test_git_blob,
            created_at=args.created_at,
            minimum_items=args.minimum_items,
            maximum_items=args.maximum_items,
        )
    except (
        FileExistsError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
