"""Wave 1C 预生成预测的严格多数据库评测测试。"""

import hashlib
import re
import shutil
import sqlite3
from collections.abc import Mapping
from pathlib import Path

import pytest

import dki.analysis.nl2sql_multidb_eval as multidb
from dki.analysis.nl2sql_multidb_eval import (
    DatabaseBindingError,
    FailureReason,
    ItemInputError,
    MultiDBEvaluationError,
    PredictionInputError,
    evaluate_predictions_across_databases,
)


def _make_db(path, values, *, extra_column=False):
    connection = sqlite3.connect(path)
    try:
        if extra_column:
            connection.execute(
                "CREATE TABLE visits (id INTEGER PRIMARY KEY, value INTEGER, note TEXT)"
            )
            connection.executemany(
                "INSERT INTO visits (id, value, note) VALUES (?, ?, ?)",
                [(index, value, None) for index, value in enumerate(values, 1)],
            )
        else:
            connection.execute(
                "CREATE TABLE visits (id INTEGER PRIMARY KEY, value INTEGER)"
            )
            connection.executemany(
                "INSERT INTO visits (id, value) VALUES (?, ?)",
                [(index, value) for index, value in enumerate(values, 1)],
            )
        connection.commit()
    finally:
        connection.close()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _connect_readonly(path):
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)


def _execute(sql, path):
    connection = _connect_readonly(path)
    try:
        cursor = connection.execute(sql.strip().rstrip(";"))
        return [column[0] for column in cursor.description], [
            tuple(row) for row in cursor.fetchall()
        ]
    finally:
        connection.close()


def _normalise(rows, ordered):
    normalized = [
        tuple(round(value, 4) if isinstance(value, float) else value for value in row)
        for row in rows
    ]
    return normalized if ordered else sorted(normalized, key=repr)


def _fake_backend():
    aliases = {"set": "set_operation", "nulls": "null", "dates": "date"}

    def validate(sql):
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("invalid SQL")
        statement = sql.strip().rstrip(";").strip()
        if ";" in statement or not statement.casefold().startswith(("select", "with")):
            raise ValueError("invalid read-only SQL")

    def comparator(
        gold_sql,
        prediction_sql,
        db_path,
        *,
        question="",
        result_order_sensitive=None,
        strict=False,
    ):
        del question
        assert strict is True
        gold_columns, gold_rows = _execute(gold_sql, db_path)
        pred_columns, pred_rows = _execute(prediction_sql, db_path)
        ordered = bool(result_order_sensitive)
        column_count_matches = len(gold_columns) == len(pred_columns)
        return {
            "correct": column_count_matches
            and _normalise(gold_rows, ordered) == _normalise(pred_rows, ordered),
            "gold_column_count": len(gold_columns),
            "pred_column_count": len(pred_columns),
            "column_count_matches": column_count_matches,
            "result_order_sensitive": ordered,
            "result_order_policy": "test_question_order_v1",
        }

    def require_sidecars_absent(path):
        for suffix in ("-wal", "-shm", "-journal"):
            if Path(f"{path}{suffix}").exists():
                raise ValueError("sidecar")

    def require_hash(value, label):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{label} must be lowercase SHA-256")
        return value

    def snapshot_database(source_path, *, expected_sha256, temporary_directory):
        require_sidecars_absent(source_path)
        destination = Path(temporary_directory) / "development.sqlite"
        shutil.copyfile(source_path, destination)
        if _sha256_file(destination) != expected_sha256:
            raise ValueError("snapshot hash mismatch")
        return destination

    def normalise_capabilities(value):
        if isinstance(value, Mapping):
            values = [key for key, enabled in value.items() if bool(enabled)]
        elif isinstance(value, str):
            values = [token for token in re.split(r"[,;|]+", value) if token.strip()]
        else:
            values = list(value)
        normalized = {
            aliases.get(
                re.sub(r"[\s\-/]+", "_", str(item).strip().casefold()),
                re.sub(r"[\s\-/]+", "_", str(item).strip().casefold()),
            )
            for item in values
        }
        order = ("aggregation_stage", "set_operation", "date", "null", "output_contract")
        return tuple([name for name in order if name in normalized] + sorted(normalized - set(order)))

    def classify_capabilities(question, sql):
        text = f"{question} {sql}".casefold()
        labels = []
        if any(token in text for token in ("count", "max(", "min(", "avg(")):
            labels.append("aggregation_stage")
        labels.append("output_contract")
        return tuple(labels)

    return {
        "name": "canonical_test_backend",
        "canonical": True,
        "result_order_policy": "test_question_order_v1",
        "validate": validate,
        "execute": _execute,
        "comparator": comparator,
        "connect_readonly": _connect_readonly,
        "require_sidecars_absent": require_sidecars_absent,
        "normalise_capabilities": normalise_capabilities,
        "classify_capabilities": classify_capabilities,
        "require_hash": require_hash,
        "sha256_file": _sha256_file,
        "snapshot_database": snapshot_database,
    }


@pytest.fixture(autouse=True)
def canonical_backend(monkeypatch):
    backend = _fake_backend()
    monkeypatch.setattr(multidb, "_resolve_strict_backend", lambda: backend)
    return backend


@pytest.fixture
def databases(tmp_path):
    base = tmp_path / "base.sqlite"
    variant = tmp_path / "variant.sqlite"
    _make_db(base, [1, 2])
    _make_db(variant, [1, 2, 3])
    return base, variant


def _bindings(base, variant):
    return [
        {"id": "base", "path": base, "role": "base"},
        {"id": "duplicate_variant", "path": variant, "role": "variant"},
    ]


def _items():
    return [
        {
            "id": "stable",
            "question": "How many visits are there?",
            "gold_sql": "SELECT COUNT(*) FROM visits",
            "difficulty": "easy",
            "category": "aggregation",
            "capabilities": ["aggregation stage", "output-contract"],
            "partition_family_id": "family-count",
        },
        {
            "id": "fragile",
            "question": "Return the maximum visit id",
            "gold_sql": "SELECT MAX(id) FROM visits",
            "difficulty": "hard",
            "category": "aggregation",
            "capability": "aggregation_stage",
            "partition_family_id": "family-max",
        },
        {
            "id": "abstain",
            "question": "Return the minimum visit id",
            "gold_sql": "SELECT MIN(id) FROM visits",
            "difficulty": "hard",
            "category": "extreme",
            "capability": "output contract",
            "partition_family_id": "family-max",
        },
    ]


def test_prediction_rows_are_snapshotted_once_and_never_generated(databases):
    base, variant = databases

    class OneReadPrediction(Mapping):
        def __init__(self):
            self.data = {"id": "stable", "sql": "SELECT COUNT(*) FROM visits"}
            self.sql_reads = 0

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, key):
            if key == "sql":
                self.sql_reads += 1
                if self.sql_reads > 1:
                    raise AssertionError("prediction SQL was resolved more than once")
            return self.data[key]

    row = OneReadPrediction()
    result = evaluate_predictions_across_databases(
        [_items()[0]], [row], _bindings(base, variant)
    )
    assert row.sql_reads == 1
    assert result["n_robust_correct"] == 1
    assert result["snapshot_policy"] == multidb.SNAPSHOT_POLICY
    with pytest.raises(PredictionInputError, match="generator"):
        evaluate_predictions_across_databases(
            [_items()[0]],
            (row for row in [{"id": "stable", "sql": "SELECT 1"}]),
            _bindings(base, variant),
        )


def test_robust_correct_requires_every_database_and_reports_aggregates(databases):
    base, variant = databases
    result = evaluate_predictions_across_databases(
        _items(),
        {
            "stable": "SELECT COUNT(*) FROM visits",
            "fragile": "SELECT 2",
            "abstain": "",
        },
        _bindings(base, variant),
        bootstrap_seed=23,
        bootstrap_iterations=500,
    )
    assert result["n_correct"] == 2
    assert result["n_robust_correct"] == 1
    assert result["robust_accuracy"] == pytest.approx(1 / 3)
    details = {detail["id"]: detail for detail in result["details"]}
    assert details["fragile"]["correct"] is True
    assert details["fragile"]["robust_correct"] is False
    assert details["fragile"]["robust_failure_reasons"] == ("result_mismatch",)
    assert details["abstain"]["answered"] is False
    assert result["by_category"]["aggregation"]["robust_correct"] == 1
    assert result["by_family"]["family-max"]["n"] == 2
    assert result["by_capability"]["aggregation_stage"]["n"] == 2
    assert result["uncertainty"]["robust_family_cluster_bootstrap"]["seed"] == 23


def test_typed_strict_column_failure(databases):
    base, variant = databases
    result = evaluate_predictions_across_databases(
        [{"id": "wide", "gold_sql": "SELECT id FROM visits"}],
        {"wide": "SELECT id, value FROM visits"},
        _bindings(base, variant),
    )
    verdict = result["details"][0]["database_results"]["base"]
    assert verdict["failure_reason"] == FailureReason.STRICT_COLUMN_COUNT_MISMATCH.value
    assert result["error_taxonomy"] == {"strict_column_count_mismatch": 2}


def test_immutable_snapshots_are_scored_not_mutated_sources(tmp_path, canonical_backend):
    base = tmp_path / "base.sqlite"
    variant = tmp_path / "variant.sqlite"
    _make_db(base, [1, 2])
    _make_db(variant, [1, 2])
    original_snapshot = canonical_backend["snapshot_database"]

    def snapshot_then_mutate(source_path, **kwargs):
        snapshot = original_snapshot(source_path, **kwargs)
        connection = sqlite3.connect(source_path)
        try:
            connection.execute("INSERT INTO visits VALUES (99, 99)")
            connection.commit()
        finally:
            connection.close()
        return snapshot

    canonical_backend["snapshot_database"] = snapshot_then_mutate
    result = evaluate_predictions_across_databases(
        [{"id": "snapshot", "gold_sql": "SELECT COUNT(*) FROM visits"}],
        {"snapshot": "SELECT 2"},
        _bindings(base, variant),
    )
    assert result["n_robust_correct"] == 1
    assert sqlite3.connect(base).execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 3


def test_exact_gold_map_keys_fail_closed(databases):
    base, variant = databases
    item = {
        "id": "gold-map",
        "gold_sql_by_database": {
            "base": "SELECT COUNT(*) FROM visits",
            "duplicate_varaint": "SELECT SUM(value) FROM visits",
        },
    }
    with pytest.raises(ItemInputError, match="exactly match"):
        evaluate_predictions_across_databases(
            [item], {"gold-map": "SELECT COUNT(*) FROM visits"}, _bindings(base, variant)
        )


def test_prediction_aliases_and_ids_must_agree(databases):
    base, variant = databases
    item = _items()[0]
    with pytest.raises(PredictionInputError, match="SQL aliases disagree"):
        evaluate_predictions_across_databases(
            [item],
            {"stable": {"pred_sql": "SELECT 1", "sql": "SELECT 2"}},
            _bindings(base, variant),
        )
    with pytest.raises(PredictionInputError, match="disagrees with its mapping key"):
        evaluate_predictions_across_databases(
            [item],
            {"stable": {"id": "other", "sql": "SELECT 1"}},
            _bindings(base, variant),
        )


def test_binding_aliases_and_canonical_ids_fail_closed(tmp_path):
    base = tmp_path / "base.sqlite"
    variant = tmp_path / "variant.sqlite"
    other = tmp_path / "other.sqlite"
    for path in (base, variant, other):
        _make_db(path, [1])
    item = {"id": "q", "gold_sql": "SELECT COUNT(*) FROM visits"}
    prediction = {"q": "SELECT COUNT(*) FROM visits"}
    result = evaluate_predictions_across_databases(
        [item],
        prediction,
        {
            " base ": {"id": "base", "path": base, "role": "base"},
            "variant": {"path": variant},
        },
    )
    assert result["base_database_id"] == "base"
    with pytest.raises(DatabaseBindingError, match="path and db_path disagree"):
        evaluate_predictions_across_databases(
            [item], prediction,
            [
                {"id": "base", "path": base, "db_path": other, "role": "base"},
                {"id": "variant", "path": variant},
            ],
        )
    with pytest.raises(DatabaseBindingError, match="db_sha256 and sha256 disagree"):
        evaluate_predictions_across_databases(
            [item], prediction,
            [
                {"id": "base", "path": base, "role": "base"},
                {
                    "id": "variant",
                    "path": variant,
                    "db_sha256": _sha256_file(variant),
                    "sha256": "0" * 64,
                },
            ],
        )
    with pytest.raises(DatabaseBindingError, match="role and is_base disagree"):
        evaluate_predictions_across_databases(
            [item], prediction,
            [
                {"id": "base", "path": base, "role": "base"},
                {"id": "variant", "path": variant, "role": "variant", "is_base": True},
            ],
        )


def test_capabilities_use_canonical_normalisation(databases):
    base, variant = databases
    result = evaluate_predictions_across_databases(
        [
            {
                "id": "caps",
                "gold_sql": "SELECT id FROM visits",
                "capabilities": ["set", "nulls", "dates"],
            }
        ],
        {"caps": "SELECT id FROM visits"},
        _bindings(base, variant),
    )
    assert result["details"][0]["capabilities"] == ("set_operation", "date", "null")
    assert set(result["by_capability"]) == {"set_operation", "date", "null"}


def test_failure_taxonomy_preserves_all_reasons_order_independently(
    tmp_path, canonical_backend
):
    paths = []
    for name in ("base", "mismatch", "error"):
        path = tmp_path / f"{name}.sqlite"
        _make_db(path, [1])
        paths.append(path)
    original = canonical_backend["comparator"]

    def comparator(*args, **kwargs):
        parent = Path(args[2]).parent.name
        if parent == "database-0001":
            return {
                "correct": False,
                "column_count_matches": True,
                "gold_column_count": 1,
                "pred_column_count": 1,
                "result_order_sensitive": False,
                "result_order_policy": "test_question_order_v1",
            }
        if parent == "database-0002":
            raise sqlite3.OperationalError("forced")
        return original(*args, **kwargs)

    canonical_backend["comparator"] = comparator
    result = evaluate_predictions_across_databases(
        [{"id": "reasons", "gold_sql": "SELECT id FROM visits"}],
        {"reasons": "SELECT id FROM visits"},
        [
            {"id": "base", "path": paths[0], "role": "base"},
            {"id": "z_mismatch", "path": paths[1]},
            {"id": "a_error", "path": paths[2]},
        ],
    )
    assert result["details"][0]["robust_failure_reasons"] == (
        "execution_error",
        "result_mismatch",
    )
    assert result["robust_item_failure_taxonomy"] == {
        "execution_error": 1,
        "result_mismatch": 1,
    }


def test_reserved_family_key_and_missing_family_fail_closed(databases):
    base, variant = databases
    item = {"id": "family", "gold_sql": "SELECT COUNT(*) FROM visits"}
    with pytest.raises(MultiDBEvaluationError, match="reserved"):
        evaluate_predictions_across_databases(
            [item], {"family": "SELECT 2"}, _bindings(base, variant), family_key="answered"
        )
    with pytest.raises(ValueError, match="valid family ID"):
        evaluate_predictions_across_databases(
            [item],
            {"family": "SELECT 2"},
            _bindings(base, variant),
            bootstrap_seed=1,
            bootstrap_iterations=10,
        )


def test_schema_identity_hash_and_dependency_fail_closed(
    tmp_path, monkeypatch
):
    base = tmp_path / "base.sqlite"
    incompatible = tmp_path / "incompatible.sqlite"
    compatible = tmp_path / "compatible.sqlite"
    _make_db(base, [1, 2])
    _make_db(incompatible, [1, 2], extra_column=True)
    _make_db(compatible, [1, 2, 3])
    item = _items()[0]
    with pytest.raises(DatabaseBindingError, match="schema-identical"):
        evaluate_predictions_across_databases(
            [item], {"stable": "SELECT COUNT(*) FROM visits"}, _bindings(base, incompatible)
        )
    with pytest.raises(DatabaseBindingError, match="content hash"):
        evaluate_predictions_across_databases(
            [item],
            {"stable": "SELECT COUNT(*) FROM visits"},
            [
                {"id": "base", "path": base, "role": "base"},
                {"id": "variant", "path": compatible, "db_sha256": "0" * 64},
            ],
        )
    monkeypatch.setattr(
        multidb,
        "_resolve_strict_backend",
        lambda: (_ for _ in ()).throw(
            MultiDBEvaluationError("canonical dependencies unavailable")
        ),
    )
    with pytest.raises(MultiDBEvaluationError, match="canonical dependencies"):
        evaluate_predictions_across_databases(
            [item], {"stable": "SELECT COUNT(*) FROM visits"}, _bindings(base, compatible)
        )
