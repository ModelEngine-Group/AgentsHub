"""Freeze the official CSS schema split as a private external benchmark.

CSS is an authoritative Chinese medical text-to-SQL benchmark, but it uses its
own 21 schemas.  This freezer therefore records it as *external schema
generalisation evidence*, never as the project's four-table prospective blind
gate.  Test questions and SQL remain private and are not printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


REQUIRED_ENTRIES = {
    "css/schema/train.json",
    "css/schema/dev.json",
    "css/schema/test.json",
    "css/tables.json",
    "css/db_content.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalise_question(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    result = {}
    for member in archive.infolist():
        name = member.filename.replace("\\", "/")
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\x00" in name
            or member.is_dir()
        ):
            if member.is_dir() and not (
                path.is_absolute() or ".." in path.parts or "\x00" in name
            ):
                continue
            raise ValueError(f"unsafe CSS archive member: {name!r}")
        if name in result:
            raise ValueError(f"duplicate CSS archive member: {name}")
        result[name] = member
    missing = REQUIRED_ENTRIES - set(result)
    if missing:
        raise ValueError(f"CSS archive lacks required entries: {sorted(missing)}")
    return result


def _load_rows(data: bytes, split: str) -> list[dict[str, Any]]:
    rows = json.loads(data.decode("utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"CSS {split} must be a JSON list")
    required = {"query", "db_id", "question", "question_id"}
    seen_ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"CSS {split} row {index} is not an object")
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"CSS {split} row {index} lacks fields: {sorted(missing)}"
            )
        if any(not str(row[key]).strip() for key in required):
            raise ValueError(f"CSS {split} row {index} has an empty field")
        question_id = str(row["question_id"])
        if question_id in seen_ids:
            raise ValueError(f"CSS {split} repeats question_id")
        seen_ids.add(question_id)
    return rows


def _write_synced(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def freeze(
    archive_path: Path,
    output_dir: Path,
    *,
    source_revision: str,
    source_url: str,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"CSS freeze destination already exists: {output_dir}"
        )
    if len(source_revision) != 40:
        raise ValueError("CSS source revision must be a full Git commit")
    archive_sha256 = _sha256(archive_path)
    if archive_sha256 != expected_archive_sha256:
        raise ValueError(
            "CSS archive hash mismatch: "
            f"expected {expected_archive_sha256}, received {archive_sha256}"
        )

    with zipfile.ZipFile(archive_path) as archive:
        members = _safe_members(archive)
        raw = {
            name: archive.read(members[name])
            for name in sorted(REQUIRED_ENTRIES)
        }

    train = _load_rows(raw["css/schema/train.json"], "schema.train")
    dev = _load_rows(raw["css/schema/dev.json"], "schema.dev")
    test = _load_rows(raw["css/schema/test.json"], "schema.test")
    tables = json.loads(raw["css/tables.json"].decode("utf-8"))
    db_content = json.loads(raw["css/db_content.json"].decode("utf-8"))
    if not isinstance(tables, list) or not isinstance(db_content, list):
        raise ValueError("CSS schema/content metadata must be JSON lists")

    split_db_ids = {
        "train": {str(row["db_id"]) for row in train},
        "dev": {str(row["db_id"]) for row in dev},
        "test": {str(row["db_id"]) for row in test},
    }
    duplicate_question_counts = {}
    for split, rows in (("train", train), ("dev", dev), ("test", test)):
        keys = [
            (
                str(row["db_id"]),
                _normalise_question(str(row["question"])),
            )
            for row in rows
        ]
        duplicate_question_counts[split] = len(keys) - len(set(keys))
    intersections = {
        "train_dev": sorted(split_db_ids["train"] & split_db_ids["dev"]),
        "train_test": sorted(split_db_ids["train"] & split_db_ids["test"]),
        "dev_test": sorted(split_db_ids["dev"] & split_db_ids["test"]),
    }
    if any(intersections.values()):
        raise ValueError(
            "CSS schema split leaks database ids across train/dev/test"
        )
    metadata_db_ids = {
        str(item.get("db_id", ""))
        for item in tables
        if isinstance(item, dict)
    }
    content_db_ids = {
        str(item.get("db_id", ""))
        for item in db_content
        if isinstance(item, dict)
    }
    required_db_ids = set().union(*split_db_ids.values())
    if not required_db_ids <= metadata_db_ids:
        raise ValueError("CSS tables.json lacks a split database id")
    if not required_db_ids <= content_db_ids:
        raise ValueError("CSS db_content.json lacks a split database id")

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.freeze-",
            dir=output_dir.parent,
        )
    )
    try:
        files = {
            "development/train.json": raw["css/schema/train.json"],
            "development/dev.json": raw["css/schema/dev.json"],
            "holdout/test.json": raw["css/schema/test.json"],
            "metadata/tables.json": raw["css/tables.json"],
            "metadata/db_content.json": raw["css/db_content.json"],
        }
        file_manifest = {}
        for relative, data in files.items():
            target = staging / relative
            _write_synced(target, data)
            file_manifest[relative] = {
                "sha256": _sha256_bytes(data),
                "size": len(data),
            }
        payload = {
            "schema_version": 1,
            "kind": "external_authoritative_nl2sql_benchmark_freeze",
            "benchmark": "CSS",
            "dataset_role": "external_schema_generalisation",
            "four_table_blind_gate": False,
            "selection_used": False,
            "holdout_consumed": False,
            "source": {
                "url": source_url,
                "revision": source_revision,
                "archive_sha256": archive_sha256,
            },
            "license_evidence": {
                "huggingface_card": "CC BY 4.0",
                "dataset_loader": "CC BY-SA 4.0",
                "code_repository": "Apache-2.0",
                "handling_policy": (
                    "private evaluation only; do not redistribute dataset "
                    "artifacts because official metadata is inconsistent"
                ),
            },
            "official_split": "schema",
            "counts": {
                "train": len(train),
                "dev": len(dev),
                "test": len(test),
                "schemas": len(required_db_ids),
                "train_schemas": len(split_db_ids["train"]),
                "dev_schemas": len(split_db_ids["dev"]),
                "test_schemas": len(split_db_ids["test"]),
                "duplicate_question_rows": duplicate_question_counts,
            },
            "database_id_intersections": intersections,
            "files": file_manifest,
            "privacy": {
                "holdout_questions_or_sql_in_public_receipt": False,
                "submission_package_allowed": False,
            },
            "freezer_sha256": _sha256(Path(__file__)),
        }
        _write_synced(
            staging / "freeze_manifest.json",
            (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
        os.rename(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--source-url",
        default="https://huggingface.co/datasets/zhanghanchong/css",
    )
    parser.add_argument("--archive-sha256", required=True)
    args = parser.parse_args()
    if "outputs/private/" not in args.output_dir.resolve().as_posix():
        raise SystemExit(
            "CSS benchmark freeze must stay below outputs/private/<RID>"
        )
    try:
        payload = freeze(
            args.archive,
            args.output_dir,
            source_revision=args.source_revision,
            source_url=args.source_url,
            expected_archive_sha256=args.archive_sha256,
        )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "benchmark": payload["benchmark"],
                "dataset_role": payload["dataset_role"],
                "counts": payload["counts"],
                "four_table_blind_gate": False,
                "holdout_consumed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
