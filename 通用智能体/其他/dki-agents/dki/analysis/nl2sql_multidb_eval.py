"""对预生成 NL2SQL 预测执行严格、多数据库、快照绑定的评测。"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

from dki.analysis.nl2sql_statistics import (
    DEFAULT_FAMILY_KEY,
    MAX_BOOTSTRAP_ITERATIONS,
    canonical_family_id,
    family_cluster_bootstrap,
    wilson_interval,
)

MULTIDB_EVAL_POLICY = "precomputed_strict_schema_identical_multidb_v1"
RESULT_ORDER_POLICY = "question_explicit_final_order_v1"
GOLD_SQL_POLICY = "exact_per_database_map_or_single_global_v1"
SCHEMA_IDENTITY_POLICY = "sqlite_master_exact_schema_identity_v1"
SNAPSHOT_POLICY = "verified_private_database_snapshots_v1"

_DETAIL_KEYS = frozenset(
    {
        "id",
        "answered",
        "correct",
        "robust_correct",
        "robust_failure_reasons",
        "prediction_sha256",
        "prediction_provenance_hash",
        "difficulty",
        "category",
        "capabilities",
        "family_id",
        "gold_sql_policy",
        "database_results",
    }
)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _multidb_policy_hash(result_order_policy: str) -> str:
    return _canonical_hash(
        {
            "policy": MULTIDB_EVAL_POLICY,
            "prediction_phase": "precomputed_only_exactly_one_sql_per_item",
            "comparison": "canonical_strict_complete_result_equivalence",
            "result_order_policy": result_order_policy,
            "robust_correct": "correct_on_every_bound_database",
            "gold_sql_policy": GOLD_SQL_POLICY,
            "schema_policy": SCHEMA_IDENTITY_POLICY,
            "snapshot_policy": SNAPSHOT_POLICY,
        }
    )


MULTIDB_EVAL_POLICY_HASH = _multidb_policy_hash(RESULT_ORDER_POLICY)


class MultiDBEvaluationError(ValueError):
    """评测输入、依赖或全局合同不满足时的基类。"""


class PredictionInputError(MultiDBEvaluationError):
    """预生成预测缺失、重复或存在歧义。"""


class DatabaseBindingError(MultiDBEvaluationError):
    """数据库绑定、哈希、快照或 schema 合同无效。"""


class ItemInputError(MultiDBEvaluationError):
    """评测题目 ID、gold SQL 或元数据无效。"""


class FailureReason(str, Enum):
    PREDICTION_ABSTAINED = "prediction_abstained"
    GOLD_SQL_INVALID = "gold_sql_invalid"
    PREDICTION_SQL_INVALID = "prediction_sql_invalid"
    EXECUTION_ERROR = "execution_error"
    STRICT_COLUMN_COUNT_MISMATCH = "strict_column_count_mismatch"
    RESULT_MISMATCH = "result_mismatch"


class DatabaseBinding(TypedDict, total=False):
    id: str
    path: str | Path
    db_path: str | Path
    role: str
    is_base: bool
    db_sha256: str
    sha256: str
    schema_sha256: str


class _StrictBackend(TypedDict):
    name: str
    canonical: bool
    result_order_policy: str
    validate: Any
    execute: Any
    comparator: Any
    connect_readonly: Any
    require_sidecars_absent: Any
    normalise_capabilities: Any
    classify_capabilities: Any
    require_hash: Any
    sha256_file: Any
    snapshot_database: Any


class _ResolvedBinding(TypedDict):
    id: str
    path: Path
    role: str
    db_sha256: str
    schema_sha256: str


def _canonical_database_id(value: Any) -> str:
    if not isinstance(value, str):
        raise DatabaseBindingError("database ID must be a string")
    identifier = unicodedata.normalize("NFKC", value).strip()
    if not identifier:
        raise DatabaseBindingError("database ID must not be empty")
    if any(unicodedata.category(char).startswith("C") for char in identifier):
        raise DatabaseBindingError("database ID must not contain control characters")
    return identifier


def _canonical_item_id(value: Any) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ItemInputError("item ID must be a string or integer")
    if isinstance(value, str):
        identifier = unicodedata.normalize("NFKC", value).strip()
        if not identifier:
            raise ItemInputError("item ID must not be empty")
        if any(unicodedata.category(char).startswith("C") for char in identifier):
            raise ItemInputError("item ID must not contain control characters")
        return identifier
    return value


def _resolve_strict_backend() -> _StrictBackend:
    """一次性解析共享严格依赖；任何缺失都使整次评测失败。"""

    try:
        protocol_module = importlib.import_module("dki.analysis.nl2sql_protocol")
        evaluator_module = importlib.import_module("dki.analysis.nl2sql_eval")
        sqlite_module = importlib.import_module("dki.analysis.nl2sql_sqlite")
        capability_module = importlib.import_module(
            "dki.analysis.nl2sql_capabilities"
        )
        evidence_module = importlib.import_module(
            "dki.analysis.nl2sql_development_evidence"
        )
        importlib.import_module("sqlglot")
    except Exception as exc:
        raise MultiDBEvaluationError(
            "canonical Wave NL2SQL evaluation dependencies are unavailable"
        ) from exc

    dependencies = {
        "validate": getattr(protocol_module, "validate_gold_sql", None),
        "execute": getattr(protocol_module, "execute_complete_readonly", None),
        "comparator": getattr(evaluator_module, "compare_sql_results", None),
        "connect_readonly": getattr(sqlite_module, "connect_readonly", None),
        "require_sidecars_absent": getattr(
            sqlite_module, "require_sqlite_sidecars_absent", None
        ),
        "normalise_capabilities": getattr(
            capability_module, "normalise_capabilities", None
        ),
        "classify_capabilities": getattr(
            capability_module, "classify_nl2sql_capabilities", None
        ),
        "require_hash": getattr(
            evidence_module, "require_lowercase_sha256", None
        ),
        "sha256_file": getattr(evidence_module, "sha256_file", None),
        "snapshot_database": getattr(
            evidence_module, "_copy_database_snapshot", None
        ),
    }
    missing = sorted(
        name for name, dependency in dependencies.items() if not callable(dependency)
    )
    if missing:
        raise MultiDBEvaluationError(
            "canonical Wave NL2SQL helper(s) are not callable: "
            + ", ".join(missing)
        )

    comparator = dependencies["comparator"]
    try:
        inspect.signature(comparator).bind(
            "SELECT 1",
            "SELECT 1",
            Path("placeholder.sqlite"),
            question="",
            result_order_sensitive=None,
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise MultiDBEvaluationError(
            "canonical compare_sql_results does not support the strict API"
        ) from exc

    result_order_policy = getattr(evaluator_module, "RESULT_ORDER_POLICY", None)
    if not isinstance(result_order_policy, str) or not result_order_policy:
        raise MultiDBEvaluationError(
            "canonical evaluator does not expose RESULT_ORDER_POLICY"
        )
    return {
        "name": "canonical_nl2sql_eval",
        "canonical": True,
        "result_order_policy": result_order_policy,
        **dependencies,
    }


def _binding_rows(database_bindings: Any) -> list[dict[str, Any]]:
    if isinstance(database_bindings, Mapping):
        rows: list[dict[str, Any]] = []
        for raw_database_id, raw_binding in database_bindings.items():
            database_id = _canonical_database_id(raw_database_id)
            if isinstance(raw_binding, Mapping):
                row = dict(raw_binding)
                supplied_id = row.get("id")
                if supplied_id is not None:
                    canonical_supplied_id = _canonical_database_id(supplied_id)
                    if canonical_supplied_id != database_id:
                        raise DatabaseBindingError(
                            "database binding ID disagrees with its mapping key"
                        )
                row["id"] = database_id
            else:
                row = {"id": database_id, "path": raw_binding}
            rows.append(row)
        return rows
    if isinstance(database_bindings, (str, bytes)) or not isinstance(
        database_bindings, Sequence
    ):
        raise DatabaseBindingError(
            "database_bindings must be a mapping or concrete sequence"
        )
    rows = []
    for index, binding in enumerate(database_bindings):
        if not isinstance(binding, Mapping):
            raise DatabaseBindingError(
                f"database_bindings[{index}] must be a mapping"
            )
        rows.append(dict(binding))
    return rows


def _schema_payload(path: Path, backend: _StrictBackend) -> list[dict[str, Any]]:
    connection = backend["connect_readonly"](path)
    try:
        rows = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' "
            "AND type IN ('table', 'view', 'index', 'trigger') "
            "ORDER BY type, name, tbl_name"
        ).fetchall()
    finally:
        connection.close()
    return [
        {"type": row[0], "name": row[1], "table": row[2], "sql": row[3] or ""}
        for row in rows
    ]


def _resolve_bindings(
    database_bindings: Any,
    *,
    base_database_id: str,
    backend: _StrictBackend,
    snapshot_root: Path,
) -> list[_ResolvedBinding]:
    rows = _binding_rows(database_bindings)
    if len(rows) < 2:
        raise DatabaseBindingError(
            "multi-database evaluation requires a base and at least one variant"
        )

    explicit_base_ids: list[str] = []
    resolved: list[_ResolvedBinding] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for index, row in enumerate(rows):
        database_id = _canonical_database_id(row.get("id"))
        if database_id in seen_ids:
            raise DatabaseBindingError(f"duplicate database ID: {database_id!r}")
        seen_ids.add(database_id)

        path_aliases = {
            key: row[key]
            for key in ("path", "db_path")
            if key in row and row[key] is not None
        }
        if not path_aliases:
            raise DatabaseBindingError(
                f"database binding {database_id!r} requires path or db_path"
            )
        canonical_paths: dict[str, Path] = {}
        for alias, raw_path in path_aliases.items():
            if not isinstance(raw_path, (str, Path)):
                raise DatabaseBindingError(
                    f"database binding {database_id!r} {alias} must be a path"
                )
            unresolved = Path(raw_path).expanduser()
            if unresolved.is_symlink():
                raise DatabaseBindingError(
                    f"database binding {database_id!r} must not be a symlink"
                )
            canonical_paths[alias] = unresolved.resolve()
        if len(set(canonical_paths.values())) != 1:
            raise DatabaseBindingError(
                f"database binding {database_id!r} path and db_path disagree"
            )
        source_path = next(iter(canonical_paths.values()))
        if not source_path.is_file():
            raise DatabaseBindingError(
                f"database binding {database_id!r} is not a regular file"
            )
        if source_path in seen_paths or any(
            source_path.samefile(seen) for seen in seen_paths
        ):
            raise DatabaseBindingError("database bindings must use distinct files")
        seen_paths.add(source_path)
        backend["require_sidecars_absent"](source_path)

        is_base = row.get("is_base")
        if is_base is not None and not isinstance(is_base, bool):
            raise DatabaseBindingError("is_base must be boolean when supplied")
        raw_role = row.get("role")
        if raw_role is None:
            role = "base" if is_base is True else "variant"
        else:
            role = str(raw_role).strip().casefold()
            if role not in {"base", "variant"}:
                raise DatabaseBindingError(
                    "database role must be 'base' or 'variant'"
                )
            if is_base is not None and (role == "base") != is_base:
                raise DatabaseBindingError("database role and is_base disagree")
        if role == "base":
            explicit_base_ids.append(database_id)

        db_sha256 = backend["sha256_file"](source_path)
        expected_hashes = {
            alias: backend["require_hash"](
                value, f"database {database_id!r} {alias}"
            )
            for alias, value in (
                ("db_sha256", row.get("db_sha256")),
                ("sha256", row.get("sha256")),
            )
            if value is not None
        }
        if len(set(expected_hashes.values())) > 1:
            raise DatabaseBindingError(
                f"database {database_id!r} db_sha256 and sha256 disagree"
            )
        if expected_hashes and next(iter(expected_hashes.values())) != db_sha256:
            raise DatabaseBindingError(
                f"database {database_id!r} content hash does not match binding"
            )

        snapshot_directory = snapshot_root / f"database-{index:04d}"
        snapshot_directory.mkdir()
        snapshot_path = backend["snapshot_database"](
            source_path,
            expected_sha256=db_sha256,
            temporary_directory=snapshot_directory,
        )
        if backend["sha256_file"](snapshot_path) != db_sha256:
            raise DatabaseBindingError(
                f"database {database_id!r} snapshot hash mismatch"
            )
        schema_sha256 = _canonical_hash(
            {
                "policy": SCHEMA_IDENTITY_POLICY,
                "objects": _schema_payload(snapshot_path, backend),
            }
        )
        expected_schema_hash = row.get("schema_sha256")
        if expected_schema_hash is not None:
            expected = backend["require_hash"](
                expected_schema_hash,
                f"database {database_id!r} schema hash",
            )
            if expected != schema_sha256:
                raise DatabaseBindingError(
                    f"database {database_id!r} schema hash does not match binding"
                )
        resolved.append(
            {
                "id": database_id,
                "path": snapshot_path,
                "role": "variant",
                "db_sha256": db_sha256,
                "schema_sha256": schema_sha256,
            }
        )

    if len(set(explicit_base_ids)) > 1:
        raise DatabaseBindingError("exactly one database may be marked as base")
    requested_base = _canonical_database_id(base_database_id)
    if explicit_base_ids:
        explicit_base = explicit_base_ids[0]
        if requested_base in seen_ids and explicit_base != requested_base:
            raise DatabaseBindingError(
                "base_database_id disagrees with the binding marked as base"
            )
        base_id = explicit_base
    else:
        base_id = requested_base
    if base_id not in seen_ids:
        raise DatabaseBindingError(
            f"base database ID {base_id!r} is not present in bindings"
        )

    schema_hashes = {binding["schema_sha256"] for binding in resolved}
    if len(schema_hashes) != 1:
        raise DatabaseBindingError("database variants are not schema-identical")
    for binding in resolved:
        binding["role"] = "base" if binding["id"] == base_id else "variant"
    return sorted(
        resolved,
        key=lambda binding: (binding["role"] != "base", binding["id"]),
    )


def _resolve_items(items: Any) -> list[dict[str, Any]]:
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise ItemInputError("items must be a concrete sequence of mappings")
    if not items:
        raise ItemInputError("items must not be empty")
    resolved: list[dict[str, Any]] = []
    seen: set[tuple[type[Any], Any]] = set()
    for index, raw_item in enumerate(items):
        if not isinstance(raw_item, Mapping):
            raise ItemInputError(f"items[{index}] must be a mapping")
        item = dict(raw_item)
        item_id = _canonical_item_id(item.get("id"))
        identity = (type(item_id), item_id)
        if identity in seen:
            raise ItemInputError(f"duplicate item ID: {item_id!r}")
        seen.add(identity)
        item["id"] = item_id
        if "gold_sql" not in item and "gold_sql_by_database" not in item:
            raise ItemInputError(
                f"item {item_id!r} requires gold_sql or gold_sql_by_database"
            )
        resolved.append(item)
    return resolved


def _bind_gold_sql(
    items: Sequence[Mapping[str, Any]],
    database_ids: Sequence[str],
    *,
    base_database_id: str,
) -> list[dict[str, Any]]:
    expected_ids = set(database_ids)
    bound: list[dict[str, Any]] = []
    for raw_item in items:
        item = dict(raw_item)
        by_database = item.get("gold_sql_by_database")
        if by_database is not None:
            if not isinstance(by_database, Mapping):
                raise ItemInputError("gold_sql_by_database must be a mapping")
            canonical_map: dict[str, str] = {}
            for raw_id, raw_sql in by_database.items():
                database_id = _canonical_database_id(raw_id)
                if database_id in canonical_map:
                    raise ItemInputError(
                        "gold_sql_by_database contains duplicate canonical IDs"
                    )
                if not isinstance(raw_sql, str) or not raw_sql.strip():
                    raise ItemInputError(
                        f"item {item['id']!r} has invalid gold SQL for {database_id!r}"
                    )
                canonical_map[database_id] = raw_sql.strip()
            if set(canonical_map) != expected_ids:
                unknown = sorted(set(canonical_map) - expected_ids)
                missing = sorted(expected_ids - set(canonical_map))
                raise ItemInputError(
                    "gold_sql_by_database must exactly match bound database IDs "
                    f"(unknown={unknown}, missing={missing})"
                )
            global_gold = item.get("gold_sql")
            if global_gold is not None:
                if not isinstance(global_gold, str) or not global_gold.strip():
                    raise ItemInputError("gold_sql must be a non-empty string when supplied")
                if global_gold.strip() != canonical_map[base_database_id]:
                    raise ItemInputError(
                        "gold_sql disagrees with the base gold_sql_by_database entry"
                    )
            policy = "per_database_exact_no_fallback"
        else:
            gold_sql = item.get("gold_sql")
            if not isinstance(gold_sql, str) or not gold_sql.strip():
                raise ItemInputError(f"item {item['id']!r} has invalid gold_sql")
            canonical_map = {
                database_id: gold_sql.strip() for database_id in database_ids
            }
            policy = "single_global_reused_for_all_databases"
        item["_gold_sql_by_database"] = canonical_map
        item["_gold_sql_policy"] = policy
        item["_base_gold_sql"] = canonical_map[base_database_id]
        bound.append(item)
    return bound


def _strict_json_value(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PredictionInputError(f"{label} must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise PredictionInputError(f"{label} keys must be strings")
            if key in result:
                raise PredictionInputError(f"{label} contains duplicate keys")
            result[key] = _strict_json_value(nested, label=f"{label}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _strict_json_value(nested, label=f"{label}[{index}]")
            for index, nested in enumerate(value)
        ]
    raise PredictionInputError(f"{label} must be strict JSON data")


def _prediction_rows(predictions: Any) -> list[tuple[Any, Any]]:
    if callable(predictions):
        raise PredictionInputError("prediction generators or callables are prohibited")
    if isinstance(predictions, Mapping):
        rows: list[tuple[Any, Any]] = []
        for raw_id, raw_prediction in predictions.items():
            outer_id = _canonical_item_id(raw_id)
            if isinstance(raw_prediction, Mapping):
                snapshot = dict(raw_prediction)
                for alias in ("id", "item_id"):
                    if alias in snapshot and snapshot[alias] is not None:
                        if _canonical_item_id(snapshot[alias]) != outer_id:
                            raise PredictionInputError(
                                f"prediction {alias} disagrees with its mapping key"
                            )
                rows.append((outer_id, snapshot))
            else:
                rows.append((outer_id, raw_prediction))
        return rows
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise PredictionInputError(
            "predictions must be a mapping or concrete sequence of rows; generators are prohibited"
        )
    rows = []
    for index, raw_row in enumerate(predictions):
        if not isinstance(raw_row, Mapping):
            raise PredictionInputError(f"predictions[{index}] must be a mapping")
        row = dict(raw_row)
        ids = [
            _canonical_item_id(row[key])
            for key in ("id", "item_id")
            if key in row and row[key] is not None
        ]
        if not ids:
            raise PredictionInputError(f"predictions[{index}] requires id or item_id")
        if len({(type(value), value) for value in ids}) != 1:
            raise PredictionInputError(f"predictions[{index}] id aliases disagree")
        rows.append((ids[0], row))
    return rows


def _prediction_sql_and_provenance(
    raw_prediction: Any,
    *,
    item_id: str | int,
) -> tuple[str, str]:
    identity = {"id_type": type(item_id).__name__, "id": item_id}
    if raw_prediction is None:
        return "", _canonical_hash(
            {"identity": identity, "sql_aliases": {}, "metadata": None}
        )
    if isinstance(raw_prediction, str):
        sql = raw_prediction.strip()
        return sql, _canonical_hash(
            {
                "identity": identity,
                "sql_aliases": {"mapping_value": sql},
                "metadata": {},
            }
        )
    if not isinstance(raw_prediction, Mapping):
        raise PredictionInputError(
            "each prediction must be SQL text, null abstention, or a mapping"
        )
    prediction = dict(raw_prediction)
    sql_aliases: dict[str, str] = {}
    for alias in ("pred_sql", "prediction_sql", "sql"):
        if alias not in prediction:
            continue
        value = prediction[alias]
        if value is not None and not isinstance(value, str):
            raise PredictionInputError(
                f"prediction {alias} must be a string or null abstention"
            )
        sql_aliases[alias] = str(value or "").strip()
    if not sql_aliases:
        raise PredictionInputError(
            "prediction rows require pred_sql, prediction_sql, or sql"
        )
    if len(set(sql_aliases.values())) != 1:
        raise PredictionInputError("prediction SQL aliases disagree")
    metadata = {
        key: _strict_json_value(value, label=f"prediction metadata {key!r}")
        for key, value in prediction.items()
        if key not in {"id", "item_id", "pred_sql", "prediction_sql", "sql"}
    }
    sql = next(iter(sql_aliases.values()))
    return sql, _canonical_hash(
        {"identity": identity, "sql_aliases": sql_aliases, "metadata": metadata}
    )


def _resolve_predictions(
    predictions: Any,
    items: Sequence[Mapping[str, Any]],
) -> dict[tuple[type[Any], Any], dict[str, Any]]:
    item_keys = {(type(item["id"]), item["id"]) for item in items}
    resolved: dict[tuple[type[Any], Any], dict[str, Any]] = {}
    for raw_id, raw_prediction in _prediction_rows(predictions):
        item_id = _canonical_item_id(raw_id)
        identity = (type(item_id), item_id)
        if identity in resolved:
            raise PredictionInputError(f"duplicate prediction ID: {item_id!r}")
        sql, provenance_hash = _prediction_sql_and_provenance(
            raw_prediction, item_id=item_id
        )
        resolved[identity] = {
            "sql": sql,
            "prediction_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            "prediction_provenance_hash": provenance_hash,
        }
    missing = item_keys - set(resolved)
    extra = set(resolved) - item_keys
    if missing:
        raise PredictionInputError(
            f"missing predictions for {len(missing)} evaluation item(s)"
        )
    if extra:
        raise PredictionInputError(
            f"predictions contain {len(extra)} unknown evaluation item(s)"
        )
    return resolved


def _typed_failure(
    reason: FailureReason,
    *,
    exception: Exception | None = None,
    comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verdict: dict[str, Any] = {"correct": False, "failure_reason": reason.value}
    if exception is not None:
        verdict["exception_type"] = type(exception).__name__
    if comparison is not None:
        for key in (
            "gold_column_count",
            "pred_column_count",
            "column_count_matches",
            "result_order_sensitive",
            "result_order_policy",
        ):
            value = comparison.get(key)
            if isinstance(value, (bool, int, str)):
                verdict[key] = value
    return verdict


def _score_on_database(
    item: Mapping[str, Any],
    prediction_sql: str,
    binding: _ResolvedBinding,
    backend: _StrictBackend,
) -> dict[str, Any]:
    gold_sql = item["_gold_sql_by_database"][binding["id"]]
    try:
        backend["validate"](gold_sql)
    except Exception as exc:
        return _typed_failure(FailureReason.GOLD_SQL_INVALID, exception=exc)
    if not prediction_sql:
        return _typed_failure(FailureReason.PREDICTION_ABSTAINED)
    try:
        backend["validate"](prediction_sql)
    except Exception as exc:
        return _typed_failure(FailureReason.PREDICTION_SQL_INVALID, exception=exc)
    override = item.get("result_order_sensitive")
    if override is not None and not isinstance(override, bool):
        raise ItemInputError("result_order_sensitive must be boolean when supplied")
    try:
        comparison = backend["comparator"](
            gold_sql,
            prediction_sql,
            binding["path"],
            question=str(item.get("question") or ""),
            result_order_sensitive=override,
            strict=True,
        )
    except Exception as exc:
        return _typed_failure(FailureReason.EXECUTION_ERROR, exception=exc)
    if not isinstance(comparison, Mapping) or not isinstance(
        comparison.get("correct"), bool
    ):
        raise MultiDBEvaluationError(
            "canonical strict comparator returned an invalid result"
        )
    if comparison["correct"]:
        verdict: dict[str, Any] = {"correct": True, "failure_reason": None}
        for key in (
            "gold_column_count",
            "pred_column_count",
            "column_count_matches",
            "result_order_sensitive",
            "result_order_policy",
        ):
            value = comparison.get(key)
            if isinstance(value, (bool, int, str)):
                verdict[key] = value
        return verdict
    reason = (
        FailureReason.STRICT_COLUMN_COUNT_MISMATCH
        if comparison.get("column_count_matches") is False
        else FailureReason.RESULT_MISMATCH
    )
    return _typed_failure(reason, comparison=comparison)


def _metadata_label(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ItemInputError(f"{label} metadata must be a string")
    canonical = unicodedata.normalize("NFKC", value).strip()
    if not canonical:
        return None
    if any(unicodedata.category(char).startswith("C") for char in canonical):
        raise ItemInputError(f"{label} metadata must not contain control characters")
    return canonical


def _capabilities(
    item: Mapping[str, Any], backend: _StrictBackend
) -> tuple[str, ...]:
    explicit = item.get("capabilities", item.get("capability"))
    if explicit is not None:
        return tuple(backend["normalise_capabilities"](explicit))
    return tuple(
        backend["classify_capabilities"](
            str(item.get("question") or ""), item["_base_gold_sql"]
        )
    )


def _family(item: Mapping[str, Any], *, family_key: str) -> str | None:
    if family_key in item and item[family_key] is not None:
        return canonical_family_id(item[family_key])
    if (
        family_key == DEFAULT_FAMILY_KEY
        and "family_id" in item
        and item["family_id"] is not None
    ):
        return canonical_family_id(item["family_id"])
    metadata = item.get("family_metadata")
    if family_key == DEFAULT_FAMILY_KEY and isinstance(metadata, Mapping):
        if "partition_component" in metadata:
            return canonical_family_id(metadata["partition_component"])
    return None


def _summary_bucket() -> dict[str, int]:
    return {"n": 0, "correct": 0, "robust_correct": 0, "answered": 0}


def _aggregate_dimension(
    details: Sequence[Mapping[str, Any]],
    metadata_key: str,
    *,
    multi_value: bool = False,
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, int]] = {}
    for detail in details:
        raw_value = detail.get(metadata_key)
        if multi_value:
            values = tuple(raw_value or ())
        elif raw_value is None:
            values = ()
        else:
            values = (raw_value,)
        for value in values:
            bucket = buckets.setdefault(str(value), _summary_bucket())
            bucket["n"] += 1
            bucket["correct"] += int(bool(detail.get("correct")))
            bucket["robust_correct"] += int(bool(detail.get("robust_correct")))
            bucket["answered"] += int(bool(detail.get("answered")))
    result: dict[str, dict[str, Any]] = {}
    for value in sorted(buckets):
        bucket = buckets[value]
        total = bucket["n"]
        result[value] = {
            "n": total,
            "correct": bucket["correct"],
            "n_correct": bucket["correct"],
            "accuracy": bucket["correct"] / total,
            "robust_correct": bucket["robust_correct"],
            "n_robust_correct": bucket["robust_correct"],
            "robust_accuracy": bucket["robust_correct"] / total,
            "answered": bucket["answered"],
            "coverage": bucket["answered"] / total,
        }
    return result


def _preflight_uncertainty(
    items: Sequence[Mapping[str, Any]],
    *,
    family_key: str,
    confidence_level: float,
    bootstrap_seed: int | None,
    bootstrap_iterations: int,
) -> None:
    wilson_interval(0, 1, confidence_level=confidence_level)
    if bootstrap_seed is None:
        return
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise TypeError("bootstrap_seed must be an integer")
    if isinstance(bootstrap_iterations, bool) or not isinstance(
        bootstrap_iterations, int
    ):
        raise TypeError("bootstrap_iterations must be an integer")
    if not 1 < bootstrap_iterations <= MAX_BOOTSTRAP_ITERATIONS:
        raise ValueError(
            "bootstrap_iterations must be between 2 and "
            f"{MAX_BOOTSTRAP_ITERATIONS}"
        )
    families = [_family(item, family_key=family_key) for item in items]
    if any(family is None for family in families):
        raise ValueError("bootstrap requires a valid family ID for every item")
    if len(set(families)) < 2:
        raise ValueError("bootstrap requires at least two distinct families")


def evaluate_predictions_across_databases(
    items: Sequence[Mapping[str, Any]],
    predictions: Mapping[Any, Any] | Sequence[Mapping[str, Any]],
    database_bindings: Mapping[str, Any] | Sequence[DatabaseBinding],
    *,
    base_database_id: str = "base",
    family_key: str = DEFAULT_FAMILY_KEY,
    strict: bool = True,
    confidence_level: float = 0.95,
    bootstrap_seed: int | None = None,
    bootstrap_iterations: int = 10_000,
) -> dict[str, Any]:
    """在不可变数据库快照上复用每题唯一的预生成 SQL 并严格评分。"""

    if strict is not True:
        raise MultiDBEvaluationError("multi-database evaluation requires strict=True")
    if not isinstance(family_key, str) or not family_key.strip():
        raise MultiDBEvaluationError("family_key must be a non-empty string")
    if family_key in _DETAIL_KEYS:
        raise MultiDBEvaluationError(
            f"family_key {family_key!r} is reserved by evaluator details"
        )

    resolved_items = _resolve_items(items)
    resolved_predictions = _resolve_predictions(predictions, resolved_items)
    backend = _resolve_strict_backend()
    with tempfile.TemporaryDirectory(prefix="nl2sql-multidb-") as temporary:
        bindings = _resolve_bindings(
            database_bindings,
            base_database_id=base_database_id,
            backend=backend,
            snapshot_root=Path(temporary),
        )
        base_id = bindings[0]["id"]
        bound_items = _bind_gold_sql(
            resolved_items,
            [binding["id"] for binding in bindings],
            base_database_id=base_id,
        )
        _preflight_uncertainty(
            bound_items,
            family_key=family_key,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed,
            bootstrap_iterations=bootstrap_iterations,
        )

        per_database: dict[str, dict[str, Any]] = {
            binding["id"]: {
                "n": len(bound_items),
                "correct": 0,
                "n_correct": 0,
                "accuracy": 0.0,
                "db_sha256": binding["db_sha256"],
                "schema_sha256": binding["schema_sha256"],
                "role": binding["role"],
                "error_taxonomy": Counter(),
            }
            for binding in bindings
        }
        error_taxonomy: Counter[str] = Counter()
        robust_item_failures: Counter[str] = Counter()
        details: list[dict[str, Any]] = []

        for item in bound_items:
            identity = (type(item["id"]), item["id"])
            prediction = resolved_predictions[identity]
            prediction_sql = prediction["sql"]
            database_results: dict[str, dict[str, Any]] = {}
            failure_reasons: set[str] = set()
            for binding in bindings:
                verdict = _score_on_database(
                    item, prediction_sql, binding, backend
                )
                database_results[binding["id"]] = verdict
                if verdict["correct"]:
                    per_database[binding["id"]]["correct"] += 1
                    per_database[binding["id"]]["n_correct"] += 1
                else:
                    reason = str(verdict["failure_reason"])
                    failure_reasons.add(reason)
                    error_taxonomy[reason] += 1
                    per_database[binding["id"]]["error_taxonomy"][reason] += 1

            base_correct = bool(database_results[base_id]["correct"])
            robust_correct = all(
                bool(database_results[binding["id"]]["correct"])
                for binding in bindings
            )
            for reason in failure_reasons:
                robust_item_failures[reason] += 1
            detail: dict[str, Any] = {
                "id": item["id"],
                "answered": bool(prediction_sql),
                "correct": base_correct,
                "robust_correct": robust_correct,
                "robust_failure_reasons": tuple(sorted(failure_reasons)),
                "prediction_sha256": prediction["prediction_sha256"],
                "prediction_provenance_hash": prediction[
                    "prediction_provenance_hash"
                ],
                "difficulty": _metadata_label(
                    item.get("difficulty"), label="difficulty"
                ),
                "category": _metadata_label(
                    item.get("category"), label="category"
                ),
                "capabilities": _capabilities(item, backend),
                "family_id": _family(item, family_key=family_key),
                "gold_sql_policy": item["_gold_sql_policy"],
                "database_results": database_results,
            }
            details.append(detail)

        for binding in bindings:
            if backend["sha256_file"](binding["path"]) != binding["db_sha256"]:
                raise DatabaseBindingError(
                    f"database {binding['id']!r} snapshot changed during evaluation"
                )
        for summary in per_database.values():
            summary["accuracy"] = summary["correct"] / summary["n"]
            summary["error_taxonomy"] = dict(
                sorted(summary["error_taxonomy"].items())
            )

        total = len(details)
        n_correct = sum(int(detail["correct"]) for detail in details)
        n_robust_correct = sum(
            int(detail["robust_correct"]) for detail in details
        )
        n_answered = sum(int(detail["answered"]) for detail in details)
        prediction_set_hash = _canonical_hash(
            sorted(
                (
                    {
                        "id_type": type(item["id"]).__name__,
                        "id": item["id"],
                        "prediction_sha256": resolved_predictions[
                            (type(item["id"]), item["id"])
                        ]["prediction_sha256"],
                        "prediction_provenance_hash": resolved_predictions[
                            (type(item["id"]), item["id"])
                        ]["prediction_provenance_hash"],
                    }
                    for item in bound_items
                ),
                key=lambda row: (row["id_type"], repr(row["id"])),
            )
        )
        binding_hash = _canonical_hash(
            [
                {
                    "id": binding["id"],
                    "role": binding["role"],
                    "db_sha256": binding["db_sha256"],
                    "schema_sha256": binding["schema_sha256"],
                }
                for binding in bindings
            ]
        )

        uncertainty: dict[str, Any] = {
            "base_wilson": wilson_interval(
                n_correct, total, confidence_level=confidence_level
            ),
            "robust_wilson": wilson_interval(
                n_robust_correct, total, confidence_level=confidence_level
            ),
        }
        if bootstrap_seed is not None:
            base_records = [
                {family_key: detail["family_id"], "correct": detail["correct"]}
                for detail in details
            ]
            robust_records = [
                {
                    family_key: detail["family_id"],
                    "correct": detail["robust_correct"],
                }
                for detail in details
            ]
            uncertainty["base_family_cluster_bootstrap"] = (
                family_cluster_bootstrap(
                    base_records,
                    family_key=family_key,
                    confidence_level=confidence_level,
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed,
                )
            )
            uncertainty["robust_family_cluster_bootstrap"] = (
                family_cluster_bootstrap(
                    robust_records,
                    family_key=family_key,
                    confidence_level=confidence_level,
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed,
                )
            )

        return {
            "policy": MULTIDB_EVAL_POLICY,
            "policy_hash": _multidb_policy_hash(
                backend["result_order_policy"]
            ),
            "strict_result_contract": True,
            "strict_backend": backend["name"],
            "canonical_strict_backend": True,
            "result_order_policy": backend["result_order_policy"],
            "gold_sql_policy": GOLD_SQL_POLICY,
            "schema_identity_policy": SCHEMA_IDENTITY_POLICY,
            "snapshot_policy": SNAPSHOT_POLICY,
            "base_database_id": base_id,
            "n_databases": len(bindings),
            "binding_hash": binding_hash,
            "prediction_set_hash": prediction_set_hash,
            "n": total,
            "n_correct": n_correct,
            "accuracy": n_correct / total,
            "n_answered": n_answered,
            "n_abstained": total - n_answered,
            "coverage": n_answered / total,
            "n_robust_correct": n_robust_correct,
            "robust_correct_n": n_robust_correct,
            "robust_accuracy_den": total,
            "robust_accuracy": n_robust_correct / total,
            "by_database": per_database,
            "by_difficulty": _aggregate_dimension(details, "difficulty"),
            "by_category": _aggregate_dimension(details, "category"),
            "by_capability": _aggregate_dimension(
                details, "capabilities", multi_value=True
            ),
            "by_family": _aggregate_dimension(details, "family_id"),
            "error_taxonomy": dict(sorted(error_taxonomy.items())),
            "error_taxonomy_unit": "item_database_verdict",
            "robust_item_failure_taxonomy": dict(
                sorted(robust_item_failures.items())
            ),
            "robust_item_failure_taxonomy_unit": "failed_item_reason_membership",
            "uncertainty": uncertainty,
            "details": details,
        }


__all__ = [
    "DatabaseBinding",
    "DatabaseBindingError",
    "FailureReason",
    "GOLD_SQL_POLICY",
    "ItemInputError",
    "MULTIDB_EVAL_POLICY",
    "MULTIDB_EVAL_POLICY_HASH",
    "MultiDBEvaluationError",
    "PredictionInputError",
    "SCHEMA_IDENTITY_POLICY",
    "SNAPSHOT_POLICY",
    "evaluate_predictions_across_databases",
]
