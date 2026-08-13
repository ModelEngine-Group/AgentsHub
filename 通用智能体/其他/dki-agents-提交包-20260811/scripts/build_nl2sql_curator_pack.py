"""Build a create-once, engine-blind NL2SQL curation input pack.

The pack is deliberately one-way: it contains the four-table schema, a
read-only SQLite snapshot, aggregate-only catalogs for pinned public
benchmarks, and the curation contract.  It never contains the project's
NL2SQL implementation, local evaluation questions, predictions, or failure
details.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.analysis.database import database_matches_manifest
from dki.analysis.nl2sql_protocol import ALLOWED_CATEGORIES, ALLOWED_TABLES
from dki.analysis.nl2sql_sqlite import connect_readonly
from dki.pipeline.artifacts import validate_task2_artifact


TOTAL_ITEMS = 300
DIFFICULTY_QUOTAS = {
    "medium": 90,
    "hard": 150,
    "extra": 30,
    "extra-hard": 30,
}
CATEGORY_QUOTAS = {
    "filtering": 30,
    "aggregation_grouping": 45,
    "top_k": 24,
    "multi_table_join": 45,
    "temporal": 30,
    "knowledge_graph": 36,
    "having": 15,
    "nested_correlated": 30,
    "cte": 20,
    "set_operation": 10,
    "window": 15,
}
FORBIDDEN_CONTEXT = (
    "dki/analysis/nl2sql.py",
    "data/nl2sql_eval*.jsonl",
    "tests/test_nl2sql*",
    "development predictions",
    "holdout predictions",
    "failure details",
)
INDEPENDENCE_FALSE_DECLARATIONS = (
    "implementation_or_source_code_accessed",
    "nl2sql_implementation_accessed",
    "tests_accessed",
    "existing_nl2sql_eval_accessed",
    "outputs_private_accessed",
    "prior_curator_files_accessed",
    "prior_holdout_accessed",
    "failure_details_accessed",
    "conversation_records_accessed",
    "existing_questions_copied_or_paraphrased",
)
EXECUTION_TRUE_DECLARATIONS = (
    "all_queries_executed_successfully",
    "all_results_fully_materialized",
    "single_read_only_sqlite_query_per_record",
    "allowed_tables_only",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_synced(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


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


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug[:48] or "source"


def _validate_source_audit(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"public-source audit does not exist: {path}")
    if path.stat().st_size > 1024 * 1024:
        raise ValueError(
            f"public-source audit is unexpectedly large and may contain rows: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"public-source audit is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"public-source audit must be a JSON object: {path}")
    if not str(payload.get("benchmark", "")).strip():
        raise ValueError(f"public-source audit lacks benchmark identity: {path}")
    if payload.get("four_table_blind_gate") is not False:
        raise ValueError(
            "an external source must be labeled four_table_blind_gate=false"
        )
    if payload.get("contains_questions_or_sql") not in (None, False):
        raise ValueError("source audit declares question or SQL content")
    privacy = payload.get("privacy", {})
    if isinstance(privacy, dict):
        leaking = [
            key
            for key in (
                "contains_questions",
                "contains_gold_sql",
                "contains_predictions",
                "contains_error_samples",
            )
            if privacy.get(key) is True
        ]
        if leaking:
            raise ValueError(
                f"source audit is not aggregate-only: {', '.join(leaking)}"
            )
    if any(
        key in payload
        for key in ("questions", "gold_sql", "predictions", "details", "error_samples")
    ):
        raise ValueError("source audit contains row-level fields")
    return payload


def _schema_contract(db_path: Path) -> tuple[str, dict[str, Any]]:
    connection = connect_readonly(db_path)
    try:
        table_rows = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        actual_tables = {str(row[0]) for row in table_rows}
        if actual_tables != set(ALLOWED_TABLES):
            raise ValueError(
                "curator database table set differs from the four-table contract: "
                f"{sorted(actual_tables)}"
            )
        statements = []
        profile: dict[str, Any] = {}
        for table, statement in table_rows:
            if not statement:
                raise ValueError(f"table {table} lacks CREATE SQL")
            statements.append(str(statement).rstrip(";") + ";")
            columns = connection.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchall()
            profile[str(table)] = {
                "columns": [
                    {
                        "name": str(column[1]),
                        "type": str(column[2]),
                        "not_null": bool(column[3]),
                        "primary_key_position": int(column[5]),
                    }
                    for column in columns
                ],
                "row_count": int(
                    connection.execute(
                        f'SELECT COUNT(*) FROM "{table}"'
                    ).fetchone()[0]
                ),
            }
        return "\n\n".join(statements) + "\n", profile
    finally:
        connection.close()


def _handoff_names(curator_source_version: str) -> dict[str, str]:
    match = re.fullmatch(r"independent-curator-(v[0-9]+[a-z]?)", curator_source_version)
    if not match:
        raise ValueError(
            "curator-source-version must follow independent-curator-vNN"
        )
    suffix = match.group(1)
    return {
        "corpus": f"corpus_{suffix}.jsonl",
        "attestation": f"curator_attestation_{suffix}.json",
        "aggregate_audit": f"aggregate_audit_{suffix}.json",
        "execution_audit": f"execution_audit_{suffix}.jsonl",
        "sha256s": "SHA256SUMS.txt",
    }


def _five_file_contract(
    *, curator_source_version: str, database_sha256: str
) -> dict[str, Any]:
    names = _handoff_names(curator_source_version)
    return {
        "schema_version": 1,
        "kind": "independent_five_file_handoff_contract",
        "source_version": curator_source_version,
        "filenames": names,
        "attestation_required_shape": {
            "source_version": curator_source_version,
            "corpus": {
                "record_count": TOTAL_ITEMS,
                "sha256": "lowercase SHA-256 of the finalized corpus bytes",
            },
            "database": {"sha256": database_sha256},
            "independence_declarations": {
                **{key: False for key in INDEPENDENCE_FALSE_DECLARATIONS},
                "database_schema_and_self_generated_query_results_only": True,
            },
            "execution_attestation": {
                key: True for key in EXECUTION_TRUE_DECLARATIONS
            },
        },
        "aggregate_audit_required_shape": {
            "record_count": TOTAL_ITEMS,
            "unique_id_count": TOTAL_ITEMS,
            "unique_normalized_question_count": TOTAL_ITEMS,
            "unique_gold_sql_count": TOTAL_ITEMS,
            "execution": {"success_count": TOTAL_ITEMS, "failure_count": 0},
        },
        "execution_audit_contract": {
            "format": "UTF-8 JSONL",
            "nonempty_record_count": TOTAL_ITEMS,
            "row_content_is_private": True,
        },
        "sha256s_contract": {
            "format": "<lowercase sha256><two spaces><basename>",
            "exact_file_set": [
                names["corpus"],
                names["attestation"],
                names["aggregate_audit"],
                names["execution_audit"],
            ],
            "self_entry_for_sha256s_forbidden": True,
        },
        "publication_order": [
            "finalize corpus and execution audit",
            "compute their hashes",
            "write attestation and aggregate audit from finalized facts",
            "compute all four deliverable hashes and write SHA256SUMS",
            "atomically rename a staging directory to the final output",
        ],
    }


def _curation_spec(
    source_entries: list[dict[str, Any]], *, curator_source_version: str
) -> dict[str, Any]:
    if sum(DIFFICULTY_QUOTAS.values()) != TOTAL_ITEMS:
        raise AssertionError("difficulty quotas do not total 300")
    if sum(CATEGORY_QUOTAS.values()) != TOTAL_ITEMS:
        raise AssertionError("category quotas do not total 300")
    if set(CATEGORY_QUOTAS) != ALLOWED_CATEGORIES:
        raise AssertionError("category quotas differ from the blind protocol")
    return {
        "schema_version": 1,
        "kind": "independent_four_table_nl2sql_curation_contract",
        "curator_source_version": curator_source_version,
        "language": "zh-CN",
        "n_items": TOTAL_ITEMS,
        "difficulty_quotas": DIFFICULTY_QUOTAS,
        "category_quotas": CATEGORY_QUOTAS,
        "output_contract": {
            "format": "UTF-8 JSONL, one object per line",
            "required_fields": [
                "id",
                "question",
                "gold_sql",
                "difficulty",
                "category",
            ],
            "unique_fields": ["id", "normalized_question"],
            "allowed_difficulties": sorted(DIFFICULTY_QUOTAS),
            "allowed_categories": sorted(CATEGORY_QUOTAS),
            "exactly_one_sqlite_statement": True,
            "readonly": True,
            "allowed_tables": sorted(ALLOWED_TABLES),
        },
        "construction_policy": {
            "use_public_sources_for": [
                "medical intent diversity",
                "anonymized AST structures",
                "temporal and abstention patterns",
                "join, aggregation, subquery, CTE, set, and window coverage",
            ],
            "independently_reauthor_for_four_table_schema": True,
            "copy_project_local_questions": False,
            "paraphrase_project_local_questions": False,
            "execute_every_gold_sql_against_supplied_database": True,
            "semantic_review_every_question_sql_pair": True,
            "keep_ast_skeleton_families_together_at_freeze": True,
            "do_not_preassign_development_or_holdout": True,
        },
        "review_policy": {
            "first_pass": "author executes and validates the gold query",
            "second_pass": (
                "an independent semantic reviewer checks that the Chinese "
                "question entails every predicate, literal, grouping, ordering, "
                "limit, and projection in the SQL"
            ),
            "near_duplicate_check": (
                "reject verbatim and near-duplicate questions, including any "
                "question derived from the project's local development files"
            ),
            "machine_grounding_gate": {
                "ast_nodes": [
                    "EQ",
                    "NEQ",
                    "GT",
                    "GTE",
                    "LT",
                    "LTE",
                    "LIKE",
                    "ILIKE",
                    "BETWEEN",
                    "IN",
                    "LIMIT",
                    "OFFSET",
                ],
                "question_normalization": "Unicode NFKC plus casefold",
                "numeric_rule": (
                    "every numeric predicate/LIMIT/OFFSET literal must occur "
                    "as an explicit standalone numeric token with Decimal-equivalent value"
                ),
                "string_rule": (
                    "every string predicate literal must occur explicitly after "
                    "NFKC/casefold/whitespace compaction; LIKE percent and underscore "
                    "wildcards may be removed before matching"
                ),
                "implicit_paraphrase_is_insufficient": True,
                "empty_string_predicate_forbidden": True,
                "authoritative_preflight": "grounding_preflight.py",
                "example": (
                    "if SQL contains age >= 18, the question must explicitly contain "
                    "the numeric value 18; wording such as adult alone is insufficient"
                ),
                "required_before_publication": True,
            },
        },
        "public_source_catalog": [
            {
                "benchmark": entry["benchmark"],
                "catalog_file": entry["catalog_file"],
                "catalog_sha256": entry["catalog_sha256"],
                "dataset_role": entry.get("dataset_role"),
                "source_url": entry.get("source_url"),
                "source_revision": entry.get("source_revision"),
            }
            for entry in source_entries
        ],
        "isolation": {
            "engine_context_provided": False,
            "local_evaluation_context_provided": False,
            "forbidden_context": list(FORBIDDEN_CONTEXT),
            "curator_must_not_request_forbidden_context": True,
        },
        "five_file_handoff_contract": "five_file_handoff_contract.json",
    }


def _readme(curator_source_version: str) -> str:
    names = _handoff_names(curator_source_version)
    return f"""# Independent four-table NL2SQL curator pack

This directory is the complete input boundary for independent curation.

Use `four_table.sqlite` in read-only mode and `schema.sql` as the authoritative
schema. Produce exactly 300 Chinese question/SQL pairs that satisfy
`curation_spec.json`. Public benchmark catalogs contain aggregate structural
guidance only; re-author every item for this database.

Do not request or inspect the project's NL2SQL engine, local evaluation sets,
predictions, or failure analyses. Do not pre-label items as development or
holdout. Return exactly these five files: `{names['corpus']}`,
`{names['attestation']}`, `{names['aggregate_audit']}`,
`{names['execution_audit']}`, and `{names['sha256s']}`. The separate freezer
assigns AST-disjoint splits.

Before writing any deliverable, read `five_file_handoff_contract.json`. Its
field names and nesting are literal, not examples. In particular, use
`corpus.record_count`, `independence_declarations`, `execution_attestation`,
and the exact aggregate audit keys stated there.

Every SQL must be a single read-only SQLite query over only `patients`,
`visits`, `kg_entities`, and `kg_triples`, must execute completely, and must be
semantically entailed by its question.

Before publication, enforce the literal machine gate in
`curation_spec.json.review_policy.machine_grounding_gate` across all 300 rows.
Every direct comparison, BETWEEN, IN, LIKE, LIMIT, or OFFSET literal must be
explicit in the Chinese question. Numeric implication is not enough: if SQL
contains `age >= 18`, the question must contain the standalone number `18`;
writing only "adult" or "成年人" fails the contract.
An empty string used as a direct SQL predicate is always invalid. Run
`python grounding_preflight.py --corpus {names['corpus']}` and require its
aggregate result to report `n_rows=300`, `n_failed=0`, and `gate_passed=true`
before computing the final five-file hashes.
"""


def build_pack(
    *,
    output_dir: Path,
    db_path: Path,
    task1_manifest: Path,
    kg_path: Path,
    source_audits: Iterable[Path],
    source_tree_sha256: str,
    curator_source_version: str = "independent-curator-v21",
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if "outputs/private/" not in output_dir.as_posix():
        raise ValueError("curator pack must remain under outputs/private/<RID>")
    if output_dir.exists():
        raise FileExistsError(f"curator pack already exists: {output_dir}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", source_tree_sha256):
        raise ValueError("source-tree-sha256 must be a 64-character SHA-256")
    handoff_names = _handoff_names(curator_source_version)

    source_paths = [Path(path).resolve() for path in source_audits]
    if not source_paths:
        raise ValueError("at least one authoritative public-source audit is required")
    source_payloads = [
        (path, _validate_source_audit(path)) for path in source_paths
    ]

    validate_task2_artifact(kg_path, task1_path=task1_manifest)
    if not database_matches_manifest(db_path, task1_manifest, kg_path):
        raise ValueError(
            "curator database is not bound to the supplied Task 1/KG lineage"
        )
    db_provenance = Path(db_path).with_suffix(".provenance.json")
    if not db_provenance.is_file():
        raise FileNotFoundError(f"database provenance is missing: {db_provenance}")
    schema_sql, schema_profile = _schema_contract(db_path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.curator-pack-",
            dir=output_dir.parent,
        )
    )
    try:
        db_target = staging / "four_table.sqlite"
        shutil.copyfile(db_path, db_target)
        os.chmod(db_target, 0o444)
        provenance_target = staging / "four_table.provenance.json"
        _write_synced(provenance_target, db_provenance.read_bytes())
        _write_synced(staging / "schema.sql", schema_sql.encode("utf-8"))
        _write_synced(
            staging / "schema_profile.json",
            _json_bytes(
                {
                    "schema_version": 1,
                    "allowed_tables": sorted(ALLOWED_TABLES),
                    "tables": schema_profile,
                    "contains_row_values": False,
                }
            ),
        )

        source_entries: list[dict[str, Any]] = []
        for index, (source_path, payload) in enumerate(source_payloads, start=1):
            audit_hash = _sha256(source_path)
            benchmark = str(payload["benchmark"])
            filename = (
                f"{index:02d}-{_safe_slug(benchmark)}-{audit_hash[:12]}.json"
            )
            target = staging / "public_source_catalog" / filename
            _write_synced(target, source_path.read_bytes())
            nested_source = payload.get("source")
            if not isinstance(nested_source, dict):
                nested_source = {}
            source_entries.append(
                {
                    "benchmark": benchmark,
                    "catalog_file": target.relative_to(staging).as_posix(),
                    "catalog_sha256": audit_hash,
                    "dataset_role": payload.get("dataset_role"),
                    "source_url": payload.get("source_url")
                    or nested_source.get("url"),
                    "source_revision": payload.get("source_revision")
                    or nested_source.get("revision"),
                    "input_path": str(source_path),
                }
            )

        five_file_contract = _five_file_contract(
            curator_source_version=curator_source_version,
            database_sha256=_sha256(db_target),
        )
        _write_synced(
            staging / "five_file_handoff_contract.json",
            _json_bytes(five_file_contract),
        )
        spec = _curation_spec(
            source_entries, curator_source_version=curator_source_version
        )
        _write_synced(staging / "curation_spec.json", _json_bytes(spec))
        grounding_preflight = ROOT / "scripts" / "nl2sql_curator_grounding_check.py"
        _write_synced(
            staging / "grounding_preflight.py",
            grounding_preflight.read_bytes(),
        )
        attestation_template = {
            "schema_version": 1,
            "kind": "isolated_nl2sql_curator_attestation",
            **five_file_contract["attestation_required_shape"],
        }
        _write_synced(
            staging / "curator_attestation.template.json",
            _json_bytes(attestation_template),
        )
        _write_synced(
            staging / "aggregate_audit.template.json",
            _json_bytes(five_file_contract["aggregate_audit_required_shape"]),
        )
        _write_synced(
            staging / "README.md",
            _readme(curator_source_version).encode("utf-8"),
        )

        file_bindings = {}
        for path in sorted(
            candidate for candidate in staging.rglob("*") if candidate.is_file()
        ):
            relative = path.relative_to(staging).as_posix()
            file_bindings[relative] = {
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
        manifest = {
            "schema_version": 1,
            "kind": "engine_blind_nl2sql_curator_pack",
            "created_once": True,
            "dataset_role": "independent_curation_input_only",
            "four_table_blind_gate": False,
            "contains_questions_or_gold_sql": False,
            "contains_engine_implementation": False,
            "source_tree_sha256": source_tree_sha256.lower(),
            "builder_sha256": _sha256(Path(__file__)),
            "lineage": {
                "task1_manifest_path": str(Path(task1_manifest).resolve()),
                "task1_manifest_sha256": _sha256(Path(task1_manifest)),
                "kg_path": str(Path(kg_path).resolve()),
                "kg_sha256": _sha256(Path(kg_path)),
                "database_path": str(Path(db_path).resolve()),
                "database_sha256": _sha256(Path(db_path)),
                "database_provenance_sha256": _sha256(db_provenance),
            },
            "public_sources": source_entries,
            "contract": {
                "n_items": TOTAL_ITEMS,
                "difficulty_quotas": DIFFICULTY_QUOTAS,
                "category_quotas": CATEGORY_QUOTAS,
                "curator_source_version": curator_source_version,
                "handoff_filenames": handoff_names,
            },
            "files": file_bindings,
        }
        _write_synced(staging / "pack_manifest.json", _json_bytes(manifest))
        os.rename(staging, output_dir)
    except Exception:
        try:
            os.chmod(staging / "four_table.sqlite", 0o666)
        except OSError:
            pass
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--task1-manifest", type=Path, required=True)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument(
        "--source-audit",
        type=Path,
        action="append",
        required=True,
        help="aggregate-only audit for a pinned authoritative public benchmark",
    )
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--curator-source-version", required=True)
    args = parser.parse_args()
    try:
        manifest = build_pack(
            output_dir=args.output_dir,
            db_path=args.db,
            task1_manifest=args.task1_manifest,
            kg_path=args.kg,
            source_audits=args.source_audit,
            source_tree_sha256=args.source_tree_sha256,
            curator_source_version=args.curator_source_version,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "kind": manifest["kind"],
                "dataset_role": manifest["dataset_role"],
                "n_items": manifest["contract"]["n_items"],
                "n_public_sources": len(manifest["public_sources"]),
                "contains_engine_implementation": False,
                "contains_questions_or_gold_sql": False,
                "output": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
