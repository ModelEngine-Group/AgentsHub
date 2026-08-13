"""Build the CSS + EHRSQL v2 development projection without opening test data.

Only the official CSS train/dev and EHRSQL train/valid files are parsed.  The
shared v2 target-AST partition is applied before selection, and rows assigned
to the holdout side are used only for an aggregate capacity proxy.  Detailed
development rows remain under ``outputs/private``; the public receipt contains
counts and hashes only.
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
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql_css_projection import (
    PROJECTION_SEED_V2,
    ProjectedItem,
    ProjectionRejected,
    load_css_schemas,
    load_target_domains,
    project_css_row,
)
from dki.analysis.nl2sql_ehrsql_projection import (
    EHRProjectionRejected,
    load_ehr_target_domains,
    project_ehrsql_row,
)
from dki.analysis.nl2sql_protocol import normalise_question, validate_gold_sql
from dki.analysis.nl2sql_sqlite import require_sqlite_sidecars_absent


EHR_DATABASES = ("eicu", "mimic_iii")
EHR_DEVELOPMENT_SPLITS = ("train", "valid")
CSS_DEVELOPMENT_SPLITS = ("train", "dev")
HARD_DIFFICULTIES = {"hard", "extra", "extra-hard"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _bound_path(base: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe source-manifest path: {relative}")
    return base.joinpath(*rel.parts)


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
            raise FileExistsError(f"v2 development artifact already exists: {path}")
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


def _verified_file(
    base: Path,
    relative: str,
    expected: dict[str, Any],
) -> Path:
    path = _bound_path(base, relative)
    if (
        not path.is_file()
        or path.stat().st_size != int(expected["size"])
        or _sha256(path) != str(expected["sha256"])
    ):
        raise ValueError(f"source binding failed: {relative}")
    return path


def _load_css_development(
    manifest_path: Path,
) -> tuple[dict[str, Any], Path, list[tuple[str, dict[str, Any]]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("benchmark") != "CSS"
        or manifest.get("official_split") != "schema"
        or manifest.get("holdout_consumed") is not False
    ):
        raise ValueError("not an unconsumed official CSS schema freeze")
    files = manifest.get("files", {})
    required = [
        *(f"development/{split}.json" for split in CSS_DEVELOPMENT_SPLITS),
        "metadata/tables.json",
        "holdout/test.json",
    ]
    if any(relative not in files for relative in required):
        raise ValueError("CSS freeze manifest is incomplete")
    base = manifest_path.resolve().parent
    rows: list[tuple[str, dict[str, Any]]] = []
    # Deliberately do not resolve, hash, or parse holdout/test.json here.
    for split in CSS_DEVELOPMENT_SPLITS:
        relative = f"development/{split}.json"
        path = _verified_file(base, relative, files[relative])
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"CSS {split} must be a JSON list")
        rows.extend((split, row) for row in payload)
    _verified_file(base, "metadata/tables.json", files["metadata/tables.json"])
    return manifest, base, rows


def _load_ehrsql_development(
    root: Path,
    audit_path: Path,
) -> tuple[dict[str, Any], list[tuple[str, str, dict[str, Any]]]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("benchmark") != "EHRSQL-v1.5.1"
        or audit.get("kind") != "ehrsql_public_repository_structure_audit"
        or audit.get("contains_questions_or_sql") is not False
        or not re.fullmatch(r"[0-9a-f]{40}", str(audit.get("source_revision", "")))
    ):
        raise ValueError("not a pinned aggregate-only EHRSQL-v1.5.1 audit")
    files = audit.get("files", {})
    required = [
        *(
            f"dataset/ehrsql/{database}/{split}.json"
            for database in EHR_DATABASES
            for split in EHR_DEVELOPMENT_SPLITS
        ),
        *(f"dataset/ehrsql/{database}/test.json" for database in EHR_DATABASES),
        "dataset/ehrsql/tables.json",
        "LICENSE",
        "README.md",
    ]
    if any(relative not in files for relative in required):
        raise ValueError("EHRSQL audit is incomplete")
    root = root.resolve()
    rows: list[tuple[str, str, dict[str, Any]]] = []
    # Test-file hashes are inherited from the prior aggregate-only audit.  This
    # builder never opens, hashes, stats, or parses either official test file.
    for database in EHR_DATABASES:
        for split in EHR_DEVELOPMENT_SPLITS:
            relative = f"dataset/ehrsql/{database}/{split}.json"
            path = _verified_file(root, relative, files[relative])
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"EHRSQL {database}/{split} must be a JSON list")
            for row in payload:
                if (
                    not isinstance(row, dict)
                    or row.get("db_id") != database
                    or row.get("split") != split
                ):
                    raise ValueError(
                        f"EHRSQL {database}/{split} row metadata is inconsistent"
                    )
                rows.append((database, split, row))
    for relative in ("dataset/ehrsql/tables.json", "LICENSE", "README.md"):
        _verified_file(root, relative, files[relative])
    return audit, rows


def _deduplicate_global(
    items: Iterable[ProjectedItem],
) -> tuple[list[ProjectedItem], dict[str, int]]:
    by_id: dict[str, ProjectedItem] = {}
    by_question: dict[str, list[ProjectedItem]] = defaultdict(list)
    for projected in items:
        identity = str(projected.item["id"])
        if identity in by_id and by_id[identity] != projected:
            raise ValueError(f"conflicting projected id: {identity}")
        by_id[identity] = projected
        by_question[
            normalise_question(str(projected.item["question"]))
        ].append(projected)
    selected: list[ProjectedItem] = []
    exact_duplicates = 0
    ambiguous_rows = 0
    ambiguous_questions = 0
    for group in by_question.values():
        canonical = {
            validate_gold_sql(str(projected.item["gold_sql"]))
            for projected in group
        }
        if len(canonical) != 1:
            ambiguous_questions += 1
            ambiguous_rows += len(group)
            continue
        group = sorted(group, key=lambda projected: projected.item["id"])
        selected.append(group[0])
        exact_duplicates += len(group) - 1
    return (
        sorted(selected, key=lambda projected: projected.item["id"]),
        {
            "exact_duplicate_rows_removed": exact_duplicates,
            "ambiguous_questions_removed": ambiguous_questions,
            "ambiguous_rows_removed": ambiguous_rows,
        },
    )


def _balanced_select(
    items: list[ProjectedItem],
    maximum_items: int,
) -> list[ProjectedItem]:
    buckets: dict[tuple[str, str, str, str], list[ProjectedItem]] = defaultdict(
        list
    )
    for item in items:
        provenance = item.item["provenance"]
        key = (
            str(provenance["benchmark"]),
            str(item.item["difficulty"]),
            str(item.item["category"]),
            item.skeleton_sha256,
        )
        buckets[key].append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda projected: projected.item["id"])
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


def _aggregate(items: list[ProjectedItem]) -> dict[str, Any]:
    difficulties = Counter(item.item["difficulty"] for item in items)
    categories = Counter(item.item["category"] for item in items)
    benchmarks = Counter(
        item.item["provenance"]["benchmark"] for item in items
    )
    return {
        "n": len(items),
        "n_hard": sum(
            count
            for difficulty, count in difficulties.items()
            if difficulty in HARD_DIFFICULTIES
        ),
        "benchmark_counts": dict(sorted(benchmarks.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "category_counts": dict(sorted(categories.items())),
        "unique_target_ast_skeletons": len(
            {item.skeleton_sha256 for item in items}
        ),
    }


def build_v2_development(
    *,
    css_freeze_manifest: Path,
    ehrsql_root: Path,
    ehrsql_audit: Path,
    db_path: Path,
    private_output: Path,
    public_output: Path,
    source_tree_sha256: str,
    created_at: str,
    maximum_items: int,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", source_tree_sha256):
        raise ValueError("source-tree-sha256 must be a full lowercase SHA-256")
    if maximum_items < 150:
        raise ValueError("maximum-items must be at least 150")
    if "outputs/private/" not in private_output.resolve().as_posix():
        raise ValueError("detailed v2 development projection must remain private")
    if "outputs/private/" in public_output.resolve().as_posix():
        raise ValueError("public receipt cannot be written under outputs/private")
    for path in (private_output.resolve(), public_output.resolve()):
        if path.exists():
            raise FileExistsError(
                f"v2 development artifact already exists: {path}"
            )
    if not db_path.is_file():
        raise ValueError(f"sealed target database not found: {db_path}")
    require_sqlite_sidecars_absent(db_path)

    css_manifest, css_base, css_rows = _load_css_development(
        css_freeze_manifest
    )
    ehr_audit, ehr_rows = _load_ehrsql_development(
        ehrsql_root,
        ehrsql_audit,
    )
    css_db_ids = {str(row["db_id"]) for _, row in css_rows}
    css_schemas = load_css_schemas(
        _bound_path(css_base, "metadata/tables.json"),
        allowed_db_ids=css_db_ids,
    )
    css_domains = load_target_domains(db_path)
    ehr_domains = load_ehr_target_domains(db_path)

    projected: list[ProjectedItem] = []
    rejection_counts: Counter[str] = Counter()
    source_attempts: Counter[str] = Counter()
    for split, row in css_rows:
        source_attempts[f"CSS/{split}"] += 1
        try:
            projected.append(
                project_css_row(
                    row,
                    css_schemas[str(row["db_id"])],
                    css_domains,
                    db_path,
                    source_split=split,
                    partition_version="v2",
                )
            )
        except ProjectionRejected as exc:
            rejection_counts[f"CSS/{split}/{exc.reason}"] += 1
    for database, split, row in ehr_rows:
        source_attempts[f"EHRSQL/{database}/{split}"] += 1
        try:
            projected.append(
                project_ehrsql_row(
                    row,
                    ehr_domains,
                    db_path,
                    source_dataset=database,
                    source_split=split,
                )
            )
        except EHRProjectionRejected as exc:
            rejection_counts[
                f"EHRSQL/{database}/{split}/{exc.reason}"
            ] += 1

    deduplicated, deduplication = _deduplicate_global(projected)
    development = [
        item for item in deduplicated if item.skeleton_role == "development"
    ]
    holdout_proxy = [
        item for item in deduplicated if item.skeleton_role == "holdout"
    ]
    development_skeletons = {item.skeleton_sha256 for item in development}
    holdout_skeletons = {item.skeleton_sha256 for item in holdout_proxy}
    if development_skeletons & holdout_skeletons:
        raise ValueError("shared v2 AST partition leaked across roles")
    selected = _balanced_select(development, maximum_items)
    if len(selected) < 150:
        raise ValueError(
            f"combined v2 development yield is below 150: {len(selected)}"
        )

    private_bytes = _jsonl_bytes(item.item for item in selected)
    source_files = {
        "CSS/train": css_manifest["files"]["development/train.json"],
        "CSS/dev": css_manifest["files"]["development/dev.json"],
        "EHRSQL/eicu/train": ehr_audit["files"][
            "dataset/ehrsql/eicu/train.json"
        ],
        "EHRSQL/eicu/valid": ehr_audit["files"][
            "dataset/ehrsql/eicu/valid.json"
        ],
        "EHRSQL/mimic_iii/train": ehr_audit["files"][
            "dataset/ehrsql/mimic_iii/train.json"
        ],
        "EHRSQL/mimic_iii/valid": ehr_audit["files"][
            "dataset/ehrsql/mimic_iii/valid.json"
        ],
    }
    receipt = {
        "schema_version": 1,
        "kind": "official_public_v2_four_table_development_receipt",
        "dataset_role": "development_selection_used",
        "created_at": created_at,
        "source_tree_sha256": source_tree_sha256,
        "partition": {
            "version": "v2",
            "seed": PROJECTION_SEED_V2,
            "development_holdout_skeleton_intersection": 0,
        },
        "hashes": {
            "builder_sha256": _sha256(Path(__file__)),
            "css_projection_module_sha256": _sha256(
                ROOT / "dki" / "analysis" / "nl2sql_css_projection.py"
            ),
            "ehrsql_projection_module_sha256": _sha256(
                ROOT / "dki" / "analysis" / "nl2sql_ehrsql_projection.py"
            ),
            "protocol_sha256": _sha256(
                ROOT / "dki" / "analysis" / "nl2sql_protocol.py"
            ),
            "css_freeze_manifest_sha256": _sha256(css_freeze_manifest),
            "ehrsql_audit_sha256": _sha256(ehrsql_audit),
            "target_db_sha256": _sha256(db_path),
            "private_development_sha256": hashlib.sha256(
                private_bytes
            ).hexdigest(),
        },
        "source_attempts": dict(sorted(source_attempts.items())),
        "source_files": source_files,
        "projection": {
            "n_before_global_deduplication": len(projected),
            "n_after_global_deduplication": len(deduplicated),
            "deduplication": deduplication,
            "rejection_counts": dict(sorted(rejection_counts.items())),
        },
        "development_candidates": _aggregate(development),
        "development_selected": _aggregate(selected),
        "train_valid_holdout_role_capacity_proxy": _aggregate(holdout_proxy),
        "official_tests": {
            "content_parsed_by_this_builder": False,
            "consumed": False,
            "files": {
                "CSS/test": css_manifest["files"]["holdout/test.json"],
                "EHRSQL/eicu/test": ehr_audit["files"][
                    "dataset/ehrsql/eicu/test.json"
                ],
                "EHRSQL/mimic_iii/test": ehr_audit["files"][
                    "dataset/ehrsql/mimic_iii/test.json"
                ],
            },
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
    parser.add_argument("--css-freeze-manifest", type=Path, required=True)
    parser.add_argument("--ehrsql-root", type=Path, required=True)
    parser.add_argument("--ehrsql-audit", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--maximum-items", type=int, default=3000)
    args = parser.parse_args()
    try:
        receipt = build_v2_development(
            css_freeze_manifest=args.css_freeze_manifest,
            ehrsql_root=args.ehrsql_root,
            ehrsql_audit=args.ehrsql_audit,
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
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
