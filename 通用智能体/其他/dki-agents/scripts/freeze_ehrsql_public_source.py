"""Freeze a pinned EHRSQL 2024 checkout as a private curation source.

EHRSQL's upstream test labels are public.  This artifact is therefore useful
for authoritative medical intents, SQL structures, and abstention examples,
but it is never a prospective blind result for the project's four-table
database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any


BENCHMARK = "EHRSQL-2024-MIMIC-IV-Demo"
SPLITS = ("train", "valid", "test")
REQUIRED_FILES = (
    "LICENSE",
    "README.md",
    "data/mimic_iv/mimic_iv.sqlite",
    "data/mimic_iv/tables.json",
    *(
        f"data/mimic_iv/{split}/{name}"
        for split in SPLITS
        for name in ("data.json", "label.json", "annotated.json")
    ),
)
OPTIONAL_FILES = tuple(
    f"data/mimic_iv/{split}/answer.json" for split in SPLITS
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalise_question(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", value).strip(),
    ).casefold()


def _bound_path(base: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe EHRSQL source path: {relative}")
    return base.joinpath(*rel.parts)


def _git_head(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


def _load_split(checkout: Path, split: str) -> dict[str, Any]:
    base = checkout / "data" / "mimic_iv" / split
    data_payload = json.loads((base / "data.json").read_text(encoding="utf-8"))
    labels = json.loads((base / "label.json").read_text(encoding="utf-8"))
    annotated = json.loads(
        (base / "annotated.json").read_text(encoding="utf-8")
    )
    rows = data_payload.get("data")
    if not isinstance(rows, list) or not isinstance(labels, dict):
        raise ValueError(f"EHRSQL {split} data/labels have an invalid shape")
    if not isinstance(annotated, list):
        raise ValueError(f"EHRSQL {split} annotated data has an invalid shape")

    ids = [str(row.get("id", "")) for row in rows]
    questions = [str(row.get("question", "")) for row in rows]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"EHRSQL {split} ids must be present and unique")
    if any(not question.strip() for question in questions):
        raise ValueError(f"EHRSQL {split} contains an empty question")
    if set(ids) != {str(key) for key in labels}:
        raise ValueError(f"EHRSQL {split} data/label ids do not match")
    annotated_ids = [str(row.get("id", "")) for row in annotated]
    if len(set(annotated_ids)) != len(annotated_ids) or set(ids) != set(
        annotated_ids
    ):
        raise ValueError(f"EHRSQL {split} data/annotated ids do not match")

    answerable = sum(
        value is not None
        and str(value).strip()
        and str(value).strip().casefold() not in {"null", "nan"}
        for value in labels.values()
    )
    return {
        "rows": len(rows),
        "answerable": int(answerable),
        "unanswerable": len(rows) - int(answerable),
        "normalised_question_hashes": {
            hashlib.sha256(_normalise_question(question).encode("utf-8")).hexdigest()
            for question in questions
        },
    }


def _write_json_synced(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def freeze(
    checkout: Path,
    output: Path,
    *,
    source_revision: str,
    source_url: str,
) -> dict[str, Any]:
    checkout = checkout.resolve()
    output = output.resolve()
    revision = source_revision.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("EHRSQL source revision must be a full Git SHA-1")
    if _git_head(checkout) != revision:
        raise ValueError("EHRSQL checkout revision does not match the pin")
    if not source_url.startswith("https://"):
        raise ValueError("EHRSQL source URL must be HTTPS")
    if output.exists():
        raise FileExistsError(f"EHRSQL freeze already exists: {output}")
    for relative in REQUIRED_FILES:
        if not _bound_path(checkout, relative).is_file():
            raise ValueError(f"EHRSQL checkout is missing {relative}")

    split_reports = {
        split: _load_split(checkout, split) for split in SPLITS
    }
    question_intersections = {
        "train_valid": len(
            split_reports["train"]["normalised_question_hashes"]
            & split_reports["valid"]["normalised_question_hashes"]
        ),
        "train_test": len(
            split_reports["train"]["normalised_question_hashes"]
            & split_reports["test"]["normalised_question_hashes"]
        ),
        "valid_test": len(
            split_reports["valid"]["normalised_question_hashes"]
            & split_reports["test"]["normalised_question_hashes"]
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.freeze-staging.",
            dir=output.parent,
        )
    )
    try:
        copied: dict[str, dict[str, int | str]] = {}
        selected = list(REQUIRED_FILES)
        selected.extend(
            relative
            for relative in OPTIONAL_FILES
            if _bound_path(checkout, relative).is_file()
        )
        for relative in selected:
            source = _bound_path(checkout, relative)
            destination = _bound_path(staging / "source", relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied[f"source/{relative}"] = {
                "size": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        manifest = {
            "schema_version": 1,
            "created_once": True,
            "benchmark": BENCHMARK,
            "dataset_role": "public_authoritative_curation_source",
            "source": {
                "url": source_url,
                "revision": revision,
                "revision_verified_from_checkout": True,
                "paper_url": "https://aclanthology.org/2024.clinicalnlp-1.23/",
            },
            "license": {
                "identifier": "CC-BY-4.0",
                "file": "source/LICENSE",
                "sha256": copied["source/LICENSE"]["sha256"],
            },
            "database_license": {
                "identifier": "ODbL-1.0",
                "upstream": "MIMIC-IV-Demo",
                "notice": (
                    "Preserve upstream attribution and license; the frozen "
                    "database remains private and is excluded from submission."
                ),
            },
            "counts": {
                split: {
                    "rows": split_reports[split]["rows"],
                    "answerable": split_reports[split]["answerable"],
                    "unanswerable": split_reports[split]["unanswerable"],
                }
                for split in SPLITS
            },
            "normalised_question_hash_intersections": question_intersections,
            "files": dict(sorted(copied.items())),
            "upstream_test_is_public": True,
            "selection_used": False,
            "four_table_blind_gate": False,
            "curation_policy": {
                "permitted": (
                    "derive medical intent quotas and anonymized SQL AST "
                    "structures for an independent curator"
                ),
                "forbidden": (
                    "claim upstream test as prospective four-table evaluation "
                    "or copy raw questions/SQL into a public submission"
                ),
            },
        }
        _write_json_synced(staging / "freeze_manifest.json", manifest)
        os.rename(staging, output)
        return manifest
    except Exception:
        # Preserve failed staging for inspection; never silently reuse it.
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args()
    if "outputs/private/" not in args.output.resolve().as_posix():
        raise SystemExit("EHRSQL raw source freeze must remain under outputs/private/<RID>")
    try:
        payload = freeze(
            args.checkout,
            args.output,
            source_revision=args.source_revision,
            source_url=args.source_url,
        )
    except (
        FileExistsError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "benchmark": payload["benchmark"],
                "source_revision": payload["source"]["revision"],
                "counts": payload["counts"],
                "four_table_blind_gate": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
