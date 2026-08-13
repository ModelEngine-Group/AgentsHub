"""Compare NL2SQL source/database lineages in fully isolated processes.

The parent process never imports ``dki``.  Every matrix cell starts a fresh
Python interpreter whose ``PYTHONPATH`` contains exactly one requested engine
root.  A small, source-bound strict execution comparator is kept in this file
so changing the engine root does not silently change the evaluator as well.

This script is for development-lineage diagnosis only.  It refuses inputs
whose filename is not explicitly development-like and publishes one
create-once JSON receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_rows(rows: Iterable[sqlite3.Row], *, ordered: bool) -> list[tuple[Any, ...]]:
    result = [
        tuple(round(value, 4) if isinstance(value, float) else value for value in row)
        for row in rows
    ]
    return result if ordered else sorted(result, key=repr)


def _execute_complete_readonly(sql: str, database: Path) -> tuple[list[str], list[tuple[Any, ...]]]:
    stripped = sql.strip()
    if not stripped or stripped.count(";") > int(stripped.endswith(";")):
        raise ValueError("exactly one SQL statement is required")
    if not re.match(r"^(?:WITH\b|SELECT\b)", stripped, re.IGNORECASE):
        raise ValueError("only read-only SELECT/CTE statements are allowed")
    forbidden = re.compile(
        r"\b(?:ATTACH|DETACH|PRAGMA|INSERT|UPDATE|DELETE|REPLACE|CREATE|DROP|ALTER|VACUUM)\b",
        re.IGNORECASE,
    )
    if forbidden.search(stripped):
        raise ValueError("write or administration SQL is forbidden")
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        cursor = connection.execute(stripped)
        columns = [str(item[0]) for item in (cursor.description or [])]
        return columns, cursor.fetchall()
    finally:
        connection.close()


def _load_development(path: Path) -> list[dict[str, Any]]:
    if "development" not in path.name.casefold():
        raise ValueError("lineage matrix only accepts an explicitly named development input")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {"id", "question", "gold_sql", "difficulty", "category"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("development rows do not satisfy the strict input schema")
    ids = [str(row["id"]) for row in rows]
    questions = ["".join(str(row["question"]).split()).casefold() for row in rows]
    if len(ids) != len(set(ids)) or len(questions) != len(set(questions)):
        raise ValueError("development ids and normalised questions must be unique")
    return rows


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _cell(engine_root: Path, database: Path, development: Path) -> dict[str, Any]:
    engine_root = engine_root.resolve()
    if sys.path[0] != str(engine_root):
        sys.path.insert(0, str(engine_root))
    from dki.analysis.nl2sql import NL2SQL  # imported only inside the isolated child
    import dki.analysis.nl2sql as engine_module

    module_path = Path(engine_module.__file__ or "").resolve()
    if not _under(module_path, engine_root):
        raise RuntimeError(
            f"engine import escaped requested root: {module_path} not under {engine_root}"
        )
    rows = _load_development(development)
    engine = NL2SQL(db_path=database)
    correct = answered = 0
    by_difficulty: dict[str, dict[str, int]] = {}
    for row in rows:
        predicted = engine.rule_sql(str(row["question"]))
        difficulty = str(row["difficulty"])
        layer = by_difficulty.setdefault(difficulty, {"n": 0, "correct": 0})
        layer["n"] += 1
        if not predicted:
            continue
        answered += 1
        gold_columns, gold_rows = _execute_complete_readonly(
            str(row["gold_sql"]), database
        )
        pred_columns, pred_rows = _execute_complete_readonly(str(predicted), database)
        ordered = bool(re.search(r"\bORDER\s+BY\b", str(row["gold_sql"]), re.I))
        is_correct = len(gold_columns) == len(pred_columns) and _normalise_rows(
            gold_rows, ordered=ordered
        ) == _normalise_rows(pred_rows, ordered=ordered)
        if is_correct:
            correct += 1
            layer["correct"] += 1
    for values in by_difficulty.values():
        values["ratio"] = values["correct"] / max(1, values["n"])
    return {
        "n": len(rows),
        "correct": correct,
        "ratio": correct / max(1, len(rows)),
        "answered": answered,
        "abstained": len(rows) - answered,
        "coverage": answered / max(1, len(rows)),
        "execution_accuracy_answered": correct / max(1, answered),
        "by_difficulty": by_difficulty,
        "engine_root": str(engine_root),
        "engine_module": str(module_path),
        "engine_sha256": _sha256(module_path),
        "database": str(database.resolve()),
        "database_sha256": _sha256(database),
        "development": str(development.resolve()),
        "development_sha256": _sha256(development),
    }


def _publish_json_once(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite existing receipt: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise argparse.ArgumentTypeError("dataset name must be lowercase ASCII")
    return name, Path(raw_path)


def _run_isolated(
    *, script: Path, engine_root: Path, database: Path, development: Path
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = str(engine_root.resolve())
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "_cell",
            "--engine-root",
            str(engine_root),
            "--database",
            str(database),
            "--development",
            str(development),
        ],
        cwd=engine_root,
        env=environment,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated lineage cell failed "
            f"(rc={completed.returncode}): {completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def _parent(args: argparse.Namespace) -> int:
    datasets = dict(args.dataset)
    if len(datasets) != len(args.dataset):
        raise ValueError("dataset names must be unique")
    engine_roots = {
        "historical_engine": args.historical_engine_root.resolve(),
        "current_engine": args.current_engine_root.resolve(),
    }
    databases = {
        "historical_db": args.historical_db.resolve(),
        "current_db": args.current_db.resolve(),
    }
    for path in [*engine_roots.values(), *databases.values(), *datasets.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    script = Path(__file__).resolve()
    matrix: dict[str, Any] = {}
    for dataset_name, development in datasets.items():
        cells: dict[str, Any] = {}
        for engine_name, engine_root in engine_roots.items():
            for database_name, database in databases.items():
                label = f"{engine_name}__{database_name}"
                cells[label] = _run_isolated(
                    script=script,
                    engine_root=engine_root,
                    database=database,
                    development=development.resolve(),
                )
        matrix[dataset_name] = cells
    payload = {
        "schema_version": 1,
        "kind": "nl2sql_development_lineage_isolation_matrix",
        "process_isolation": "one_fresh_python_interpreter_per_cell",
        "fixed_comparator_sha256": _sha256(script),
        "python": sys.version,
        "matrix": matrix,
    }
    _publish_json_once(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")
    cell = subparsers.add_parser("_cell", help=argparse.SUPPRESS)
    cell.add_argument("--engine-root", type=Path, required=True)
    cell.add_argument("--database", type=Path, required=True)
    cell.add_argument("--development", type=Path, required=True)
    parser.add_argument("--historical-engine-root", type=Path)
    parser.add_argument("--current-engine-root", type=Path)
    parser.add_argument("--historical-db", type=Path)
    parser.add_argument("--current-db", type=Path)
    parser.add_argument("--dataset", type=_parse_named_path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.mode == "_cell":
        result = _cell(args.engine_root, args.database, args.development)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    required = {
        "historical_engine_root": args.historical_engine_root,
        "current_engine_root": args.current_engine_root,
        "historical_db": args.historical_db,
        "current_db": args.current_db,
        "dataset": args.dataset,
        "output": args.output,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("missing arguments: " + ", ".join(missing))
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
