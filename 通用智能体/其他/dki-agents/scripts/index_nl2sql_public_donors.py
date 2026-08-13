"""Index authoritative public NL2SQL donors without exposing source content.

Only selection-used train/validation rows are parsed.  Public test files are
hash-verified against their frozen source manifests but are never loaded as
JSON and never used for donor selection.  The private index contains source
row pointers and hashes, not questions or SQL; the public receipt is aggregate
only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


CSS_DEVELOPMENT_FILES = (
    "development/train.json",
    "development/dev.json",
)
CSS_HOLDOUT_FILE = "holdout/test.json"
EHRSQL_DEVELOPMENT_FILES = (
    "dataset/ehrsql/eicu/train.json",
    "dataset/ehrsql/eicu/valid.json",
    "dataset/ehrsql/mimic_iii/train.json",
    "dataset/ehrsql/mimic_iii/valid.json",
)
EHRSQL_HOLDOUT_FILES = (
    "dataset/ehrsql/eicu/test.json",
    "dataset/ehrsql/mimic_iii/test.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bound(base: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe source path: {relative}")
    candidate = base.joinpath(*rel.parts).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"source path escapes root: {relative}") from exc
    return candidate


def _verify_binding(
    base: Path,
    relative: str,
    binding: dict[str, Any],
) -> Path:
    path = _bound(base, relative)
    if (
        not path.is_file()
        or path.stat().st_size != binding.get("size")
        or _sha256(path) != binding.get("sha256")
    ):
        raise ValueError(f"public source binding failed: {relative}")
    return path


def _normalise_ast(sql: str) -> tuple[str, list[str]]:
    try:
        from sqlglot import exp, parse
        from sqlglot.errors import ErrorLevel
    except ImportError as exc:
        raise RuntimeError(
            "sqlglot is required for public donor indexing"
        ) from exc
    statements = parse(sql, read="sqlite")
    if len(statements) != 1:
        raise ValueError("donor SQL must contain exactly one statement")
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
        raise ValueError("donor SQL is not read-only")

    constructs = set()
    if tree.find(exp.CTE):
        constructs.add("cte")
    if tree.find(exp.Window):
        constructs.add("window")
    if any(tree.find(kind) for kind in (exp.Union, exp.Intersect, exp.Except)):
        constructs.add("set_operation")
    if tree.find(exp.Having):
        constructs.add("having")
    if tree.find(exp.Subquery):
        constructs.add("nested_subquery")
    if tree.find(exp.Join):
        constructs.add("multi_table_join")
    if tree.find(exp.Group) or tree.find(exp.AggFunc):
        constructs.add("aggregation_grouping")
    if tree.find(exp.Order) and tree.find(exp.Limit):
        constructs.add("top_k")
    if tree.find(exp.Where):
        constructs.add("filtering")

    temporal_columns = {
        str(column.name).casefold()
        for column in tree.find_all(exp.Column)
        if any(
            marker in str(column.name).casefold()
            for marker in (
                "date",
                "time",
                "year",
                "month",
                "day",
                "rq",
                "sj",
            )
        )
    }
    if temporal_columns:
        constructs.add("temporal")

    def anonymise(node):
        if isinstance(node, exp.Literal):
            return exp.Literal.string("__LITERAL__")
        if isinstance(node, exp.Identifier):
            replacement = node.copy()
            replacement.set("this", "__IDENTIFIER__")
            return replacement
        return node

    skeleton = re.sub(
        r"\s+",
        " ",
        tree.transform(anonymise).sql(
            dialect="sqlite",
            unsupported_level=ErrorLevel.IGNORE,
        ),
    ).strip().upper()
    return skeleton, sorted(constructs)


def _iter_css(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("kind")
        != "external_authoritative_nl2sql_benchmark_freeze"
        or manifest.get("benchmark") != "CSS"
        or manifest.get("official_split") != "schema"
    ):
        raise ValueError("not a supported CSS source freeze")
    root = manifest_path.resolve().parent
    files = manifest.get("files", {})
    for relative in (*CSS_DEVELOPMENT_FILES, CSS_HOLDOUT_FILE):
        if relative not in files:
            raise ValueError(f"CSS source lacks binding: {relative}")
        _verify_binding(root, relative, files[relative])

    donors = []
    failures = Counter()
    for relative in CSS_DEVELOPMENT_FILES:
        split = PurePosixPath(relative).stem
        rows = json.loads(_bound(root, relative).read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"CSS {split} is not a JSON list")
        for row_index, row in enumerate(rows):
            try:
                question = str(row["question"])
                sql = str(row["query"])
                skeleton, constructs = _normalise_ast(sql)
            except Exception as exc:
                failures[type(exc).__name__] += 1
                continue
            source_id = str(row.get("question_id", row_index))
            donors.append(
                {
                    "dataset": "CSS",
                    "source_revision": manifest["source"]["revision"],
                    "source_file": relative,
                    "source_split": split,
                    "source_row_index": row_index,
                    "source_id_sha256": _sha256_text(source_id),
                    "source_db_id_sha256": _sha256_text(
                        str(row.get("db_id", ""))
                    ),
                    "question_sha256": _sha256_text(question),
                    "sql_sha256": _sha256_text(sql),
                    "ast_skeleton_sha256": _sha256_text(skeleton),
                    "constructs": constructs,
                    "language": "zh-CN",
                }
            )
    return donors, {
        "source_manifest_sha256": _sha256(manifest_path),
        "source_revision": manifest["source"]["revision"],
        "development_files_parsed": list(CSS_DEVELOPMENT_FILES),
        "holdout_files_hash_verified": [CSS_HOLDOUT_FILE],
        "holdout_content_parsed": False,
        "parse_failures": dict(sorted(failures.items())),
    }


def _iter_ehrsql(
    audit_path: Path,
    source_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("kind") != "ehrsql_public_repository_structure_audit"
        or audit.get("benchmark") != "EHRSQL-v1.5.1"
    ):
        raise ValueError("not a supported EHRSQL public source audit")
    source_root = source_root.resolve()
    files = audit.get("files", {})
    for relative in (*EHRSQL_DEVELOPMENT_FILES, *EHRSQL_HOLDOUT_FILES):
        if relative not in files:
            raise ValueError(f"EHRSQL audit lacks binding: {relative}")
        _verify_binding(source_root, relative, files[relative])

    donors = []
    failures = Counter()
    unanswerable = 0
    for relative in EHRSQL_DEVELOPMENT_FILES:
        parts = PurePosixPath(relative).parts
        database = parts[-2]
        split = PurePosixPath(relative).stem
        rows = json.loads(
            _bound(source_root, relative).read_text(encoding="utf-8")
        )
        if not isinstance(rows, list):
            raise ValueError(f"EHRSQL {database}/{split} is not a JSON list")
        for row_index, row in enumerate(rows):
            if bool(row.get("is_impossible")):
                unanswerable += 1
                continue
            try:
                question = str(row["question"])
                sql = str(row["query"])
                skeleton, constructs = _normalise_ast(sql)
            except Exception as exc:
                failures[type(exc).__name__] += 1
                continue
            source_id = str(row.get("id", row_index))
            donors.append(
                {
                    "dataset": "EHRSQL-v1.5.1",
                    "source_revision": audit["source_revision"],
                    "source_file": relative,
                    "source_split": f"{database}/{split}",
                    "source_row_index": row_index,
                    "source_id_sha256": _sha256_text(source_id),
                    "source_db_id_sha256": _sha256_text(database),
                    "question_sha256": _sha256_text(question),
                    "sql_sha256": _sha256_text(sql),
                    "ast_skeleton_sha256": _sha256_text(skeleton),
                    "constructs": constructs,
                    "language": "en",
                }
            )
    return donors, {
        "source_audit_sha256": _sha256(audit_path),
        "source_revision": audit["source_revision"],
        "development_files_parsed": list(EHRSQL_DEVELOPMENT_FILES),
        "holdout_files_hash_verified": list(EHRSQL_HOLDOUT_FILES),
        "holdout_content_parsed": False,
        "unanswerable_rows_excluded": unanswerable,
        "parse_failures": dict(sorted(failures.items())),
    }


def _aggregate(
    donors: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    languages = Counter()
    skeletons = set()
    rows = 0
    for donor in donors:
        rows += 1
        dataset = str(donor["dataset"])
        split = f"{dataset}:{donor['source_split']}"
        languages[str(donor["language"])] += 1
        skeletons.add(str(donor["ast_skeleton_sha256"]))
        by_dataset[dataset]["rows"] += 1
        by_split[split]["rows"] += 1
        for construct in donor["constructs"]:
            by_dataset[dataset][construct] += 1
            by_split[split][construct] += 1
    return {
        "rows": rows,
        "unique_anonymized_ast_skeletons": len(skeletons),
        "languages": dict(sorted(languages.items())),
        "by_dataset": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_dataset.items())
        },
        "by_split": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_split.items())
        },
    }


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _publish_pair_once(
    private_output: Path,
    private_payload: dict[str, Any],
    public_output: Path,
    public_payload: dict[str, Any],
) -> None:
    if private_output.exists() or public_output.exists():
        raise FileExistsError(
            "public donor index output exists; choose a new RID"
        )
    private_output.parent.mkdir(parents=True, exist_ok=True)
    public_output.parent.mkdir(parents=True, exist_ok=True)
    temporaries: list[tuple[Path, Path]] = []
    try:
        for destination, payload in (
            (private_output, private_payload),
            (public_output, public_payload),
        ):
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(_json_bytes(payload))
                handle.flush()
                os.fsync(handle.fileno())
                temporaries.append((Path(handle.name), destination))
        for temporary, destination in temporaries:
            os.link(temporary, destination)
    finally:
        for temporary, _ in temporaries:
            temporary.unlink(missing_ok=True)


def build_index(
    *,
    css_manifest: Path,
    ehrsql_audit: Path,
    ehrsql_root: Path,
    private_output: Path,
    public_output: Path,
    source_tree_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", source_tree_sha256):
        raise ValueError("source tree identity must be a full SHA-256")
    css_donors, css_receipt = _iter_css(css_manifest)
    ehrsql_donors, ehrsql_receipt = _iter_ehrsql(
        ehrsql_audit,
        ehrsql_root,
    )
    donors = sorted(
        [*css_donors, *ehrsql_donors],
        key=lambda row: (
            row["dataset"],
            row["source_file"],
            row["source_row_index"],
        ),
    )
    keys = {
        (
            row["dataset"],
            row["source_file"],
            row["source_row_index"],
        )
        for row in donors
    }
    if len(keys) != len(donors):
        raise ValueError("public donor index contains duplicate source rows")
    aggregate = _aggregate(donors)
    private_payload = {
        "schema_version": 1,
        "kind": "private_authoritative_nl2sql_donor_index",
        "dataset_role": "public_source_projection_input_selection_used",
        "selection_used": True,
        "source_tree_sha256": source_tree_sha256.lower(),
        "sources": {
            "CSS": css_receipt,
            "EHRSQL-v1.5.1": ehrsql_receipt,
        },
        "aggregate": aggregate,
        "donors": donors,
        "contains_source_questions_or_sql": False,
        "official_source_test_content_parsed": False,
        "submission_package_allowed": False,
        "indexer_sha256": _sha256(Path(__file__)),
    }
    private_bytes = _json_bytes(private_payload)
    public_payload = {
        "schema_version": 1,
        "kind": "public_authoritative_nl2sql_donor_index_attestation",
        "dataset_role": private_payload["dataset_role"],
        "selection_used": True,
        "source_tree_sha256": source_tree_sha256.lower(),
        "private_index_sha256": hashlib.sha256(private_bytes).hexdigest(),
        "sources": private_payload["sources"],
        "aggregate": aggregate,
        "contains_source_questions_or_sql": False,
        "contains_source_row_pointers": False,
        "official_source_test_content_parsed": False,
        "four_table_blind_gate": False,
        "projection_required": True,
        "indexer_sha256": private_payload["indexer_sha256"],
    }
    _publish_pair_once(
        private_output,
        private_payload,
        public_output,
        public_payload,
    )
    return private_payload, public_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--css-manifest", type=Path, required=True)
    parser.add_argument("--ehrsql-audit", type=Path, required=True)
    parser.add_argument("--ehrsql-root", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    args = parser.parse_args()
    private_marker = "outputs/private/"
    if private_marker not in args.private_output.resolve().as_posix():
        raise SystemExit("detailed donor index must remain under outputs/private/<RID>")
    if private_marker in args.public_output.resolve().as_posix():
        raise SystemExit("aggregate donor attestation must not be private")
    try:
        private_payload, public_payload = build_index(
            css_manifest=args.css_manifest,
            ehrsql_audit=args.ehrsql_audit,
            ehrsql_root=args.ehrsql_root,
            private_output=args.private_output,
            public_output=args.public_output,
            source_tree_sha256=args.source_tree_sha256,
        )
    except (
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "private_output": str(args.private_output),
                "public_output": str(args.public_output),
                "rows": public_payload["aggregate"]["rows"],
                "unique_anonymized_ast_skeletons": public_payload[
                    "aggregate"
                ]["unique_anonymized_ast_skeletons"],
                "official_source_test_content_parsed": False,
                "four_table_blind_gate": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
