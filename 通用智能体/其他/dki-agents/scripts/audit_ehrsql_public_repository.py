"""Aggregate-only audit of the pinned public EHRSQL repository dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


DATASETS = ("mimic_iii", "eicu")
SPLITS = ("train", "valid", "test")
REQUIRED_METADATA = (
    "README.md",
    "LICENSE",
    "dataset/ehrsql/CHANGELOG.md",
    "dataset/ehrsql/tables.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_question(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _git_head(checkout: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip().lower()


def _publish_once(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"EHRSQL repository audit already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _structure(tree, exp) -> tuple[str, dict[str, bool]]:
    def anonymise(node):
        if isinstance(node, exp.Literal):
            return exp.Literal.string("__LITERAL__")
        return node

    skeleton = re.sub(
        r"\s+",
        " ",
        tree.transform(anonymise).sql(dialect="sqlite"),
    ).strip().upper()
    aggregate_types = tuple(
        kind
        for name in ("AggFunc", "Count", "Sum", "Avg", "Min", "Max")
        if (kind := getattr(exp, name, None)) is not None
    )
    return skeleton, {
        "aggregates": any(isinstance(node, aggregate_types) for node in tree.walk()),
        "groups": tree.find(exp.Group) is not None,
        "havings": tree.find(exp.Having) is not None,
        "joins": tree.find(exp.Join) is not None,
        "limits": tree.find(exp.Limit) is not None,
        "orders": tree.find(exp.Order) is not None,
        "set_operations": any(
            tree.find(kind) is not None for kind in (exp.Union, exp.Intersect, exp.Except)
        ),
        "subqueries": tree.find(exp.Subquery) is not None,
        "windows": tree.find(exp.Window) is not None,
    }


def _audit_file(path: Path, *, expected_db: str, expected_split: str, exp, parse):
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"EHRSQL data file is unreadable: {path}") from exc
    if not isinstance(rows, list):
        raise ValueError(f"EHRSQL data file must be a JSON list: {path}")
    required = {"id", "db_id", "question", "query", "is_impossible", "split"}
    ids = set()
    questions = set()
    skeletons = set()
    constructs = Counter()
    table_counts = Counter()
    answerable = 0
    impossible = 0
    parse_failures = 0
    non_readonly = 0
    multi_statement = 0
    forbidden = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Alter,
        exp.Attach,
        exp.Command,
    )
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or required - set(row):
            raise ValueError(f"EHRSQL row {index} lacks required fields")
        if row["db_id"] != expected_db or row["split"] != expected_split:
            raise ValueError(f"EHRSQL row {index} has inconsistent split metadata")
        identity = str(row["id"])
        if not identity or identity in ids:
            raise ValueError(f"EHRSQL {expected_db}/{expected_split} repeats an id")
        ids.add(identity)
        question = str(row["question"]).strip()
        if not question:
            raise ValueError(f"EHRSQL row {index} has an empty question")
        questions.add(_normalise_question(question))
        if bool(row["is_impossible"]):
            impossible += 1
            continue
        sql = str(row["query"]).strip()
        if not sql:
            raise ValueError(f"answerable EHRSQL row {index} has empty SQL")
        answerable += 1
        try:
            statements = parse(sql, read="sqlite")
        except Exception:
            parse_failures += 1
            continue
        if len(statements) != 1:
            multi_statement += 1
            continue
        tree = statements[0]
        if any(tree.find(kind) is not None for kind in forbidden):
            non_readonly += 1
            continue
        skeleton, flags = _structure(tree, exp)
        skeletons.add(skeleton)
        constructs.update(name for name, present in flags.items() if present)
        table_counts.update(
            table.name.casefold() for table in tree.find_all(exp.Table)
        )
    return {
        "rows": len(rows),
        "answerable": answerable,
        "unanswerable": impossible,
        "unique_ids": len(ids),
        "normalised_unique_questions": len(questions),
        "unique_anonymized_ast_skeletons": len(skeletons),
        "parse_failures": parse_failures,
        "multi_statement": multi_statement,
        "non_readonly": non_readonly,
        "unique_tables_referenced": len(table_counts),
        "rows_with_construct": dict(sorted(constructs.items())),
    }, ids, questions, skeletons


def audit(
    *,
    checkout: Path,
    source_revision: str,
    source_tree_sha256: str,
) -> dict[str, Any]:
    checkout = checkout.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("EHRSQL source revision must be a full Git SHA-1")
    if _git_head(checkout) != source_revision:
        raise ValueError("EHRSQL checkout revision differs from the requested pin")
    if not re.fullmatch(r"[0-9a-f]{64}", source_tree_sha256):
        raise ValueError("source-tree-sha256 must be a SHA-256")
    required_files = [checkout / relative for relative in REQUIRED_METADATA]
    required_files.extend(
        checkout / "dataset" / "ehrsql" / database / f"{split}.json"
        for database in DATASETS
        for split in SPLITS
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise ValueError(f"EHRSQL checkout is incomplete: {missing}")
    try:
        from sqlglot import exp, parse
    except ImportError as exc:
        raise RuntimeError("sqlglot is required for the EHRSQL audit") from exc

    split_stats = {}
    all_ids = set()
    question_sets = {}
    skeleton_sets = {}
    files = {}
    for path in required_files:
        relative = path.relative_to(checkout).as_posix()
        files[relative] = {"sha256": _sha256(path), "size": path.stat().st_size}
    for database in DATASETS:
        split_stats[database] = {}
        for split in SPLITS:
            path = checkout / "dataset" / "ehrsql" / database / f"{split}.json"
            stats, ids, questions, skeletons = _audit_file(
                path,
                expected_db=database,
                expected_split=split,
                exp=exp,
                parse=parse,
            )
            if all_ids & ids:
                raise ValueError("EHRSQL ids overlap across database/split files")
            all_ids |= ids
            split_stats[database][split] = stats
            question_sets[f"{database}/{split}"] = questions
            skeleton_sets[f"{database}/{split}"] = skeletons
    question_intersections = {}
    skeleton_intersections = {}
    labels = sorted(question_sets)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            key = f"{left}__{right}"
            question_intersections[key] = len(
                question_sets[left] & question_sets[right]
            )
            skeleton_intersections[key] = len(
                skeleton_sets[left] & skeleton_sets[right]
            )
    total_rows = sum(
        split_stats[database][split]["rows"]
        for database in DATASETS
        for split in SPLITS
    )
    total_answerable = sum(
        split_stats[database][split]["answerable"]
        for database in DATASETS
        for split in SPLITS
    )
    parse_failures = sum(
        split_stats[database][split]["parse_failures"]
        for database in DATASETS
        for split in SPLITS
    )
    unsafe = sum(
        split_stats[database][split]["multi_statement"]
        + split_stats[database][split]["non_readonly"]
        for database in DATASETS
        for split in SPLITS
    )
    if parse_failures or unsafe:
        raise ValueError(
            f"EHRSQL public source has parse/safety failures: {parse_failures}/{unsafe}"
        )
    return {
        "schema_version": 1,
        "kind": "ehrsql_public_repository_structure_audit",
        "benchmark": "EHRSQL-v1.5.1",
        "dataset_role": "public_authoritative_curation_source",
        "selection_used": True,
        "prospective_blind": False,
        "upstream_test_is_public": True,
        "four_table_blind_gate": False,
        "contains_questions_or_sql": False,
        "source_url": "https://github.com/glee4810/EHRSQL",
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "license": "CC-BY-4.0",
        "release_evidence": {
            "version": "1.5.1",
            "changelog_path": "dataset/ehrsql/CHANGELOG.md",
            "upstream_execution_claim": (
                "the pinned upstream changelog states that all 22,505 SQL "
                "queries execute successfully"
            ),
            "upstream_execution_claim_independently_reproduced": False,
        },
        "totals": {
            "rows": total_rows,
            "answerable": total_answerable,
            "unanswerable": total_rows - total_answerable,
            "unique_ids": len(all_ids),
            "parse_failures": parse_failures,
            "unsafe_statements": unsafe,
        },
        "splits": split_stats,
        "normalised_question_hash_intersections": question_intersections,
        "anonymized_ast_skeleton_intersections": skeleton_intersections,
        "files": files,
        "curation_recommendation": {
            "role": (
                "hospital-staff intent, time-expression, abstention, and "
                "anonymized SQL-structure source"
            ),
            "required_isolation": (
                "independent curator receives aggregate catalog and the "
                "four-table curator pack, never the project engine or failures"
            ),
            "four_table_reauthoring_required": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "outputs/private/" in args.output.resolve().as_posix():
        raise SystemExit("aggregate EHRSQL audit must be stored as public evidence")
    try:
        payload = audit(
            checkout=args.checkout,
            source_revision=args.source_revision.lower(),
            source_tree_sha256=args.source_tree_sha256.lower(),
        )
        _publish_once(args.output, payload)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "benchmark": payload["benchmark"],
                "totals": payload["totals"],
                "four_table_blind_gate": False,
                "contains_questions_or_sql": False,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
