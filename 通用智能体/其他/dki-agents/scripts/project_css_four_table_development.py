"""Create-once CSS train/dev projection for four-table NL2SQL development.

This entrypoint verifies the private CSS freeze but never opens the official
test JSON.  It emits detailed projected rows only under ``outputs/private`` and
an aggregate, hash-bound public receipt under ``outputs/remote_runs``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql_css_projection import (
    ProjectionRejected,
    deduplicate_projected_items,
    load_css_schemas,
    load_target_domains,
    project_css_row,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _publish_once(path: Path, data: bytes) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"projection artifact already exists: {path}")
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
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bound_path(base: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe CSS manifest path: {relative}")
    return base.joinpath(*rel.parts)


def _verify_css_freeze(manifest_path: Path) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (
        manifest.get("benchmark") != "CSS"
        or manifest.get("official_split") != "schema"
        or manifest.get("holdout_consumed") is not False
    ):
        raise ValueError("not an unconsumed official CSS schema freeze")
    base = Path(manifest_path).resolve().parent
    required = (
        "development/train.json",
        "development/dev.json",
        "holdout/test.json",
        "metadata/tables.json",
    )
    for relative in required:
        path = _bound_path(base, relative)
        expected = manifest["files"][relative]
        if (
            not path.is_file()
            or path.stat().st_size != int(expected["size"])
            or _sha256(path) != str(expected["sha256"])
        ):
            raise ValueError(f"CSS freeze binding failed: {relative}")
    return manifest, base


def build_development_projection(
    *,
    freeze_manifest: Path,
    db_path: Path,
    private_output: Path,
    public_output: Path,
    source_tree_sha256: str,
    created_at: str,
    maximum_items: int,
) -> dict[str, Any]:
    if len(source_tree_sha256) != 64:
        raise ValueError("source tree identity must be a full SHA-256")
    if maximum_items < 150:
        raise ValueError("development projection maximum must be at least 150")
    if "outputs/private/" not in private_output.resolve().as_posix():
        raise ValueError("detailed CSS projection must remain private")
    if "outputs/private/" in public_output.resolve().as_posix():
        raise ValueError("aggregate CSS projection receipt must be public")
    manifest, base = _verify_css_freeze(freeze_manifest)
    development_rows: list[tuple[str, dict[str, Any]]] = []
    development_db_ids: set[str] = set()
    source_files: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        relative = f"development/{split}.json"
        path = _bound_path(base, relative)
        rows = json.loads(path.read_text(encoding="utf-8"))
        source_files[split] = {
            "relative_path": relative,
            "sha256": _sha256(path),
            "rows": len(rows),
        }
        for row in rows:
            development_db_ids.add(str(row["db_id"]))
            development_rows.append((split, row))

    schemas = load_css_schemas(
        _bound_path(base, "metadata/tables.json"),
        allowed_db_ids=development_db_ids,
    )
    domains = load_target_domains(db_path)
    projected = []
    rejected: Counter[str] = Counter()
    for split, row in development_rows:
        try:
            projected.append(
                project_css_row(
                    row,
                    schemas[str(row["db_id"])],
                    domains,
                    db_path,
                    source_split=split,
                    required_role="development",
                )
            )
        except ProjectionRejected as exc:
            rejected[exc.reason] += 1
    candidates_before_deduplication = len(projected)
    projected = deduplicate_projected_items(projected)
    if len(projected) < 150:
        raise ValueError(
            f"CSS four-table development yield is below 150: {len(projected)}"
        )

    # Deterministic category/difficulty round-robin prevents the output from
    # being dominated by the largest source template family.
    buckets: dict[tuple[str, str], list[Any]] = {}
    for item in projected:
        key = (item.item["difficulty"], item.item["category"])
        buckets.setdefault(key, []).append(item)
    selected = []
    keys = sorted(buckets)
    while len(selected) < maximum_items:
        advanced = False
        for key in keys:
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop(0))
                advanced = True
                if len(selected) >= maximum_items:
                    break
        if not advanced:
            break
    rows = [item.item for item in selected]
    category_counts = Counter(row["category"] for row in rows)
    difficulty_counts = Counter(row["difficulty"] for row in rows)
    skeleton_counts = Counter(item.skeleton_sha256 for item in selected)
    # JSONL is the native evaluator input.  Each private row already carries
    # hash-only source provenance; the public receipt below binds the dataset
    # and contains aggregate diagnostics without exposing questions or SQL.
    private_bytes = _jsonl_bytes(rows)
    public_payload = {
        "schema_version": 1,
        "kind": "css_four_table_development_projection_receipt",
        "dataset_role": "development_selection_used",
        "created_at": created_at,
        "source_tree_sha256": source_tree_sha256,
        "projector_sha256": _sha256(Path(__file__)),
        "projection_module_sha256": _sha256(
            Path(__file__).resolve().parent.parent
            / "dki"
            / "analysis"
            / "nl2sql_css_projection.py"
        ),
        "freeze_manifest_sha256": _sha256(freeze_manifest),
        "db_sha256": _sha256(db_path),
        "source_files": source_files,
        "n_candidates_before_deduplication": candidates_before_deduplication,
        "n_candidates_after_deduplication": len(projected),
        "n_selected": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "unique_target_ast_skeletons": len(skeleton_counts),
        "rejection_counts": dict(sorted(rejected.items())),
        "private_output_sha256": hashlib.sha256(private_bytes).hexdigest(),
        "privacy": {
            "contains_questions": False,
            "contains_gold_sql": False,
            "contains_error_samples": False,
            "private_projection_allowed_in_submission": False,
            "private_output_format": "jsonl",
        },
        "official_test": {
            "content_parsed": False,
            "consumed": False,
            "sha256": manifest["files"]["holdout/test.json"]["sha256"],
        },
    }
    _publish_once(private_output, private_bytes)
    _publish_once(public_output, _json_bytes(public_payload))
    return public_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--maximum-items", type=int, default=1200)
    args = parser.parse_args()
    try:
        payload = build_development_projection(
            freeze_manifest=args.freeze_manifest,
            db_path=args.db,
            private_output=args.private_output,
            public_output=args.public_output,
            source_tree_sha256=args.source_tree_sha256,
            created_at=args.created_at,
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
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
