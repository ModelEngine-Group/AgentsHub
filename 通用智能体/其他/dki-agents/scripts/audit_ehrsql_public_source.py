"""Aggregate-only structural audit of a frozen EHRSQL 2024 source.

The report intentionally contains no question text or SQL.  EHRSQL supplies
authoritative medical intent and query-structure priors for an independent
curator; its public upstream test set is not a prospective project blind set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.nl2sql_sqlite import connect_readonly


BENCHMARK = "EHRSQL-2024-MIMIC-IV-Demo"
SPLITS = ("train", "valid", "test")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(base: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe EHRSQL manifest path: {relative}")
    return base.joinpath(*rel.parts)


def _publish_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"EHRSQL source audit already exists: {path}")
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


def _normalise_question(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", value).strip(),
    ).casefold()


def _skeleton(tree, exp) -> str:
    def anonymise(node):
        if isinstance(node, exp.Literal):
            return exp.Literal.string("__LITERAL__")
        if isinstance(node, exp.Identifier):
            replacement = node.copy()
            replacement.set("this", "__IDENTIFIER__")
            return replacement
        return node

    return re.sub(
        r"\s+",
        " ",
        tree.transform(anonymise).sql(dialect="sqlite"),
    ).strip().upper()


def _structure(tree, exp) -> tuple[dict[str, int], str]:
    joins = sum(1 for _ in tree.find_all(exp.Join))
    subqueries = sum(1 for _ in tree.find_all(exp.Subquery))
    set_operations = sum(
        1
        for kind in (exp.Union, exp.Intersect, exp.Except)
        for _ in tree.find_all(kind)
    )
    windows = sum(1 for _ in tree.find_all(exp.Window))
    ctes = sum(1 for _ in tree.find_all(exp.CTE))
    groups = sum(1 for _ in tree.find_all(exp.Group))
    havings = sum(1 for _ in tree.find_all(exp.Having))
    orders = sum(1 for _ in tree.find_all(exp.Order))
    limits = sum(1 for _ in tree.find_all(exp.Limit))
    aggregates = sum(1 for _ in tree.find_all(exp.AggFunc))
    score = (
        joins
        + 2 * subqueries
        + 2 * set_operations
        + 2 * windows
        + ctes
        + groups
        + havings
    )
    if score >= 5:
        difficulty = "extra-hard"
    elif score >= 3:
        difficulty = "extra"
    elif score >= 1:
        difficulty = "hard"
    else:
        difficulty = "medium"
    return {
        "joins": joins,
        "subqueries": subqueries,
        "set_operations": set_operations,
        "windows": windows,
        "ctes": ctes,
        "groups": groups,
        "havings": havings,
        "orders": orders,
        "limits": limits,
        "aggregates": aggregates,
    }, difficulty


def _is_answerable(value: Any) -> bool:
    return (
        value is not None
        and bool(str(value).strip())
        and str(value).strip().casefold() not in {"null", "nan"}
    )


def _audit_split(
    source: Path,
    split: str,
    connection: sqlite3.Connection,
    *,
    exp,
    parse,
) -> tuple[dict[str, Any], set[str], set[str]]:
    base = source / "data" / "mimic_iv" / split
    data_payload = json.loads((base / "data.json").read_text(encoding="utf-8"))
    labels = json.loads((base / "label.json").read_text(encoding="utf-8"))
    rows = data_payload["data"]
    question_hashes = {
        hashlib.sha256(
            _normalise_question(str(row["question"])).encode("utf-8")
        ).hexdigest()
        for row in rows
    }
    skeletons: set[str] = set()
    clauses: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    parse_failures = 0
    multi_statement = 0
    non_readonly = 0
    explain_failures = 0
    answerable = 0
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
    for value in labels.values():
        if not _is_answerable(value):
            continue
        answerable += 1
        query = str(value)
        try:
            statements = parse(query, read="sqlite")
        except Exception:
            parse_failures += 1
            continue
        if len(statements) != 1:
            multi_statement += 1
            continue
        tree = statements[0]
        if any(tree.find(kind) for kind in forbidden):
            non_readonly += 1
            continue
        structure, difficulty = _structure(tree, exp)
        difficulties[difficulty] += 1
        for name, value_count in structure.items():
            if value_count:
                clauses[name] += 1
        skeletons.add(_skeleton(tree, exp))
        try:
            connection.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
        except sqlite3.Error:
            explain_failures += 1
    return (
        {
            "rows": len(rows),
            "answerable": answerable,
            "unanswerable": len(rows) - answerable,
            "normalised_unique_questions": len(question_hashes),
            "parse_failures": parse_failures,
            "multi_statement": multi_statement,
            "non_readonly": non_readonly,
            "explain_failures": explain_failures,
            "derived_difficulty": dict(sorted(difficulties.items())),
            "rows_with_construct": dict(sorted(clauses.items())),
            "unique_anonymized_ast_skeletons": len(skeletons),
        },
        skeletons,
        question_hashes,
    )


def audit(
    manifest_path: Path,
    output: Path,
    *,
    source_tree_sha256: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", source_tree_sha256):
        raise ValueError("source tree identity must be a full SHA-256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("benchmark") != BENCHMARK
        or manifest.get("dataset_role")
        != "public_authoritative_curation_source"
        or manifest.get("upstream_test_is_public") is not True
        or manifest.get("four_table_blind_gate") is not False
    ):
        raise ValueError("not a supported frozen EHRSQL public source")
    root = manifest_path.resolve().parent
    for relative, expected in manifest["files"].items():
        path = _bound_path(root, relative)
        if (
            not path.is_file()
            or path.stat().st_size != expected["size"]
            or _sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"EHRSQL freeze binding failed: {relative}")

    try:
        from sqlglot import exp, parse
    except ImportError as exc:
        raise RuntimeError("sqlglot is required for the EHRSQL audit") from exc

    source = root / "source"
    database = source / "data" / "mimic_iv" / "mimic_iv.sqlite"
    connection = connect_readonly(database)
    try:
        reports: dict[str, dict[str, Any]] = {}
        skeletons: dict[str, set[str]] = {}
        questions: dict[str, set[str]] = {}
        for split in SPLITS:
            report, split_skeletons, split_questions = _audit_split(
                source,
                split,
                connection,
                exp=exp,
                parse=parse,
            )
            reports[split] = report
            skeletons[split] = split_skeletons
            questions[split] = split_questions
    finally:
        connection.close()

    payload = {
        "schema_version": 1,
        "kind": "ehrsql_public_source_structure_audit",
        "benchmark": BENCHMARK,
        "dataset_role": "public_curation_source_selection_used",
        "selection_used": True,
        "source_tree_sha256": source_tree_sha256.lower(),
        "freeze_manifest_sha256": _sha256(manifest_path),
        "source_revision": manifest["source"]["revision"],
        "source_url": manifest["source"]["url"],
        "license": manifest["license"]["identifier"],
        "database_sha256": manifest["files"][
            "source/data/mimic_iv/mimic_iv.sqlite"
        ]["sha256"],
        "splits": reports,
        "anonymized_ast_skeleton_intersections": {
            "train_valid": len(skeletons["train"] & skeletons["valid"]),
            "train_test": len(skeletons["train"] & skeletons["test"]),
            "valid_test": len(skeletons["valid"] & skeletons["test"]),
        },
        "normalised_question_hash_intersections": {
            "train_valid": len(questions["train"] & questions["valid"]),
            "train_test": len(questions["train"] & questions["test"]),
            "valid_test": len(questions["valid"] & questions["test"]),
        },
        "curation_recommendation": {
            "role": (
                "medical intent, abstention, temporal-query, and anonymized "
                "SQL-structure source for an independent four-table curator"
            ),
            "required_isolation": (
                "curator receives schema/read-only DB/source catalog but no "
                "dki NL2SQL implementation, predictions, or failure details"
            ),
            "raw_question_or_sql_publication": False,
        },
        "upstream_test_is_public": True,
        "prospective_blind": False,
        "four_table_blind_gate": False,
        "contains_questions_or_sql": False,
        "auditor_sha256": _sha256(Path(__file__)),
    }
    _publish_once(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    args = parser.parse_args()
    if "outputs/private/" not in args.manifest.resolve().as_posix():
        raise SystemExit("EHRSQL source manifest must remain private")
    if "outputs/private/" in args.output.resolve().as_posix():
        raise SystemExit("EHRSQL aggregate audit must be public")
    try:
        payload = audit(
            args.manifest,
            args.output,
            source_tree_sha256=args.source_tree_sha256,
        )
    except (
        FileExistsError,
        OSError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "source_revision": payload["source_revision"],
                "splits": payload["splits"],
                "four_table_blind_gate": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
