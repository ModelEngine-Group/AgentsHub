"""Aggregate-only SQL-structure audit of CSS train/dev.

The official schema-test split is deliberately never parsed here.  Its file is
only hash-checked against the private freeze manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"CSS development audit already exists: {path}")
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


def _bound_path(base: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe CSS manifest path: {relative}")
    return base.joinpath(*rel.parts)


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


def _structure(tree, exp) -> tuple[dict[str, int | bool], str]:
    joins = sum(1 for _ in tree.find_all(exp.Join))
    subqueries = sum(1 for _ in tree.find_all(exp.Subquery))
    unions = sum(1 for _ in tree.find_all(exp.Union))
    intersects = sum(1 for _ in tree.find_all(exp.Intersect))
    excepts = sum(1 for _ in tree.find_all(exp.Except))
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
        + 2 * (unions + intersects + excepts)
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
        "set_operations": unions + intersects + excepts,
        "windows": windows,
        "ctes": ctes,
        "groups": groups,
        "havings": havings,
        "orders": orders,
        "limits": limits,
        "aggregates": aggregates,
    }, difficulty


def _audit_rows(
    rows: list[dict[str, Any]],
    exp,
    parse,
) -> tuple[dict[str, Any], set[str]]:
    clause_rows: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    schemas: Counter[str] = Counter()
    parse_failures = 0
    non_readonly = 0
    multi_statement = 0
    skeletons = set()
    for row in rows:
        schemas[str(row["db_id"])] += 1
        query = str(row["query"])
        try:
            statements = parse(query, read="sqlite")
        except Exception:
            parse_failures += 1
            continue
        if len(statements) != 1:
            multi_statement += 1
            continue
        tree = statements[0]
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
        if any(tree.find(kind) for kind in forbidden):
            non_readonly += 1
            continue
        structure, difficulty = _structure(tree, exp)
        difficulties[difficulty] += 1
        for name, value in structure.items():
            if value:
                clause_rows[name] += 1
        skeletons.add(_skeleton(tree, exp))
    return {
        "rows": len(rows),
        "schemas": len(schemas),
        "parse_failures": parse_failures,
        "multi_statement": multi_statement,
        "non_readonly": non_readonly,
        "derived_difficulty": dict(sorted(difficulties.items())),
        "rows_with_construct": dict(sorted(clause_rows.items())),
        "unique_anonymized_ast_skeletons": len(skeletons),
    }, skeletons


def audit(
    manifest_path: Path,
    output: Path,
    *,
    source_tree_sha256: str,
) -> dict[str, Any]:
    if len(source_tree_sha256) != 64:
        raise ValueError("source tree identity must be a full SHA-256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("benchmark") != "CSS"
        or manifest.get("official_split") != "schema"
        or manifest.get("holdout_consumed") is not False
    ):
        raise ValueError("not an unconsumed official CSS schema freeze")
    base = manifest_path.resolve().parent
    files = manifest["files"]
    for relative in (
        "development/train.json",
        "development/dev.json",
        "holdout/test.json",
        "metadata/tables.json",
    ):
        path = _bound_path(base, relative)
        expected = files[relative]
        if (
            not path.is_file()
            or path.stat().st_size != expected["size"]
            or _sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"CSS freeze binding failed: {relative}")

    train = json.loads(
        _bound_path(base, "development/train.json").read_text(encoding="utf-8")
    )
    dev = json.loads(
        _bound_path(base, "development/dev.json").read_text(encoding="utf-8")
    )
    try:
        from sqlglot import exp, parse
    except ImportError as exc:
        raise RuntimeError("sqlglot is required for the CSS audit") from exc

    train_report, train_skeletons = _audit_rows(train, exp, parse)
    dev_report, dev_skeletons = _audit_rows(dev, exp, parse)
    payload = {
        "schema_version": 1,
        "kind": "css_development_structure_audit",
        "benchmark": "CSS",
        "dataset_role": "development_selection_used",
        "source_tree_sha256": source_tree_sha256,
        "freeze_manifest_sha256": _sha256(manifest_path),
        "holdout": {
            "content_parsed": False,
            "hash_verified": True,
            "consumed": False,
            "sha256": files["holdout/test.json"]["sha256"],
        },
        "train": train_report,
        "dev": dev_report,
        "train_dev_anonymized_ast_skeleton_intersection": len(
            train_skeletons & dev_skeletons
        ),
        "difficulty_note": (
            "Derived structural labels are diagnostic only and are not "
            "official CSS difficulty labels."
        ),
        "four_table_blind_gate": False,
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
        raise SystemExit("CSS freeze manifest must remain private")
    if "outputs/private/" in args.output.resolve().as_posix():
        raise SystemExit("CSS aggregate development audit must be public")
    try:
        payload = audit(
            args.manifest,
            args.output,
            source_tree_sha256=args.source_tree_sha256,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "output": str(args.output),
                "train": payload["train"],
                "dev": payload["dev"],
                "holdout_consumed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
