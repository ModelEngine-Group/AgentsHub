import hashlib
import json
from pathlib import Path

import pytest

import dki.analysis.nl2sql_seal as seal_module
from dki.analysis.nl2sql_official_v2_development_anchor import (
    OFFICIAL_V2_DEVELOPMENT_ANCHORS,
    trusted_evidence_binding,
)


ANCHOR = OFFICIAL_V2_DEVELOPMENT_ANCHORS[0]
SOURCE_TREE_SHA256 = "c" * 64


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _gate_receipt_payload(*, source_tree_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "rid": ANCHOR.rid,
        "dataset_role": "development_selection_used",
        "selection_used": True,
        "created_once": True,
        "engine": "rule",
        "input_sha256": ANCHOR.private_development_sha256,
        "database_sha256": ANCHOR.target_db_sha256,
        "development_evidence_sha256": ANCHOR.receipt_sha256,
        "development_evidence_kind": (
            "official_public_v2_four_table_development_receipt"
        ),
        "trusted_evidence": trusted_evidence_binding(ANCHOR),
        "engine_source_sha256": seal_module.sha256_file(
            seal_module.ROOT / "dki" / "analysis" / "nl2sql.py"
        ),
        "source_tree_sha256": source_tree_sha256,
        "overall": {
            "n": 180,
            "n_correct": 166,
            "n_answered": 175,
            "strict_result_contract": True,
            "by_difficulty": {
                "medium": {"n": 60, "correct": 60},
                "hard": {"n": 60, "correct": 56},
                "extra": {"n": 30, "correct": 28},
                "extra-hard": {"n": 30, "correct": 22},
            },
        },
    }


def _gate_fixture(tmp_path: Path, monkeypatch):
    receipt = tmp_path / "synthetic-development-receipt.json"
    database = tmp_path / "synthetic-development.sqlite"
    database.write_bytes(b"synthetic database bytes; never opened as SQLite")
    freeze = {
        "development": {"sha256": ANCHOR.private_development_sha256},
        "database": {"sha256": ANCHOR.target_db_sha256},
    }
    payload = _gate_receipt_payload(source_tree_sha256=SOURCE_TREE_SHA256)
    _write_json(receipt, payload)

    original_sha256_file = seal_module.sha256_file

    def anchored_sha256(path: Path) -> str:
        if Path(path).resolve() == database.resolve():
            return ANCHOR.target_db_sha256
        return original_sha256_file(path)

    monkeypatch.setattr(seal_module, "sha256_file", anchored_sha256)
    return {
        "database": database,
        "freeze": freeze,
        "payload": payload,
        "receipt": receipt,
    }


def _validate_gate(fixture: dict[str, object]) -> dict[str, object]:
    return seal_module._validate_development_gate(
        fixture["receipt"],
        freeze=fixture["freeze"],
        db_path=fixture["database"],
        source_tree_sha256=SOURCE_TREE_SHA256,
    )


def test_development_gate_requires_exact_source_controlled_binding(
    tmp_path,
    monkeypatch,
):
    fixture = _gate_fixture(tmp_path, monkeypatch)

    result = _validate_gate(fixture)

    assert result["trusted_evidence"] == trusted_evidence_binding(ANCHOR)
    assert result["hard_n"] == 120
    assert result["hard_correct"] == 106
    assert result["extra_hard_n"] == 30
    assert result["extra_hard_correct"] == 22


@pytest.mark.parametrize("mutation", ["missing", "altered"])
def test_development_gate_rejects_missing_or_altered_trusted_binding(
    tmp_path,
    monkeypatch,
    mutation,
):
    fixture = _gate_fixture(tmp_path, monkeypatch)
    payload = fixture["payload"]
    if mutation == "missing":
        payload.pop("trusted_evidence")
    else:
        binding = dict(payload["trusted_evidence"])
        binding["rid"] = "20260726T122144Z-g30fce24-t000000000000"
        payload["trusted_evidence"] = binding
    _write_json(fixture["receipt"], payload)

    with pytest.raises(ValueError, match="trusted evidence binding"):
        _validate_gate(fixture)


def test_development_gate_rejects_self_consistent_non_anchor_hashes(
    tmp_path,
    monkeypatch,
):
    fixture = _gate_fixture(tmp_path, monkeypatch)
    replacement_input = "d" * 64
    replacement_database = "e" * 64
    replacement_evidence = "f" * 64
    fixture["freeze"]["development"]["sha256"] = replacement_input
    fixture["freeze"]["database"]["sha256"] = replacement_database
    fixture["payload"].update(
        {
            "input_sha256": replacement_input,
            "database_sha256": replacement_database,
            "development_evidence_sha256": replacement_evidence,
        }
    )
    _write_json(fixture["receipt"], fixture["payload"])

    original_sha256_file = seal_module.sha256_file

    def self_consistent_database_hash(path: Path) -> str:
        if Path(path).resolve() == fixture["database"].resolve():
            return replacement_database
        return original_sha256_file(path)

    monkeypatch.setattr(seal_module, "sha256_file", self_consistent_database_hash)
    with pytest.raises(ValueError, match="trusted anchor or sealed candidate"):
        _validate_gate(fixture)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", "180"),
        ("n_correct", True),
        ("n_answered", 175.0),
    ],
)
def test_development_gate_rejects_coerced_overall_counts(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    fixture = _gate_fixture(tmp_path, monkeypatch)
    fixture["payload"]["overall"][field] = value
    _write_json(fixture["receipt"], fixture["payload"])

    with pytest.raises(ValueError, match="non-negative integer"):
        _validate_gate(fixture)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", "50"),
        ("correct", False),
        ("correct", 39.0),
    ],
)
def test_development_gate_rejects_coerced_hard_counts(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    fixture = _gate_fixture(tmp_path, monkeypatch)
    fixture["payload"]["overall"]["by_difficulty"]["hard"][field] = value
    _write_json(fixture["receipt"], fixture["payload"])

    with pytest.raises(ValueError, match="non-negative integer"):
        _validate_gate(fixture)


def _quality_overall(
    *,
    medium: tuple[int, int],
    hard: tuple[int, int],
    extra: tuple[int, int],
    extra_hard: tuple[int, int],
    answered: int | None = None,
) -> dict[str, object]:
    layers = {
        "medium": {"n": medium[0], "correct": medium[1]},
        "hard": {"n": hard[0], "correct": hard[1]},
        "extra": {"n": extra[0], "correct": extra[1]},
        "extra-hard": {"n": extra_hard[0], "correct": extra_hard[1]},
    }
    n = sum(layer["n"] for layer in layers.values())
    correct = sum(layer["correct"] for layer in layers.values())
    return {
        "n": n,
        "n_correct": correct,
        "n_answered": n if answered is None else answered,
        "by_difficulty": layers,
    }


def test_development_quality_gate_enforces_original_release_thresholds():
    passing = _quality_overall(
        medium=(54, 54),
        hard=(85, 78),
        extra=(22, 20),
        extra_hard=(19, 14),
        answered=176,
    )

    gate = seal_module._development_quality_gate(passing)

    assert gate["gate_thresholds"] == {
        "minimum_n": 180,
        "overall_minimum_percent": 92,
        "hard_plus_minimum_percent": 85,
        "extra_hard_minimum_percent": 70,
        "coverage_minimum_percent": 97,
    }
    assert gate["original_gate_thresholds"] == gate["gate_thresholds"]
    assert gate["original_gate_passed"] is True
    assert gate["release_eligible"] is True
    assert gate["gate_override"] is None
    assert gate["n_correct"] == 166
    assert gate["n_answered"] == 176
    assert gate["hard_n"] == 126
    assert gate["hard_correct"] == 112


def test_development_quality_gate_rejects_historical_probe_metrics():
    historical_probe = _quality_overall(
        medium=(54, 51),
        hard=(85, 68),
        extra=(22, 16),
        extra_hard=(19, 13),
        answered=174,
    )

    with pytest.raises(ValueError, match="development_overall_below_0_92"):
        seal_module._development_quality_gate(historical_probe)


def test_development_quality_gate_rejects_low_coverage():
    low_coverage = _quality_overall(
        medium=(54, 54),
        hard=(85, 78),
        extra=(22, 20),
        extra_hard=(19, 14),
        answered=174,
    )

    with pytest.raises(ValueError, match="development_coverage_below_0_97"):
        seal_module._development_quality_gate(low_coverage)


def test_development_quality_gate_n_180_boundary_is_exact():
    passing = _quality_overall(
        medium=(54, 54),
        hard=(85, 78),
        extra=(22, 20),
        extra_hard=(19, 14),
        answered=176,
    )
    failing = _quality_overall(
        medium=(53, 50),
        hard=(85, 74),
        extra=(22, 19),
        extra_hard=(19, 14),
        answered=174,
    )

    assert seal_module._development_quality_gate(passing)["n"] == 180
    with pytest.raises(ValueError, match="development_n_below_180"):
        seal_module._development_quality_gate(failing)


def test_development_quality_gate_reports_missing_extra_hard_separately():
    overall = _quality_overall(
        medium=(100, 100),
        hard=(60, 60),
        extra=(20, 20),
        extra_hard=(0, 0),
    )

    with pytest.raises(ValueError, match="development_extra_hard_missing"):
        seal_module._development_quality_gate(overall)


def _regression_payload(*, correct: int = 112) -> dict[str, object]:
    return {
        "n": 120,
        "n_correct": correct,
        "n_answered": 120,
        "by_difficulty": {
            "easy": {"n": 44, "correct": 44},
            "hard": {"n": 76, "correct": correct - 44},
        },
    }


def test_regression_quality_gate_keeps_known_120_exact_boundary():
    assert seal_module._regression_quality_gate(_regression_payload())["n_correct"] == 112
    with pytest.raises(ValueError, match="known_120_correct_below_112"):
        seal_module._regression_quality_gate(_regression_payload(correct=111))


def test_regression_quality_gate_keeps_easy_44_exact_boundary():
    payload = _regression_payload()
    payload["by_difficulty"]["easy"]["correct"] = 43

    with pytest.raises(ValueError, match="easy_44_not_44_of_44"):
        seal_module._regression_quality_gate(payload)


def _independent_gate_fixture(tmp_path: Path, monkeypatch):
    rid = "20260730T120000Z-g30fce24-t" + SOURCE_TREE_SHA256[:12]
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    development = frozen / "development.jsonl"
    development.write_text(
        json.dumps(
            {
                "id": "independent-dev-0001",
                "question": "统计患者总数",
                "gold_sql": "SELECT COUNT(*) FROM patients",
                "difficulty": "hard",
                "category": "aggregation_grouping",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    freeze_manifest = frozen / "freeze_manifest.json"
    freeze_manifest.write_text("{}\n", encoding="utf-8")
    database = tmp_path / "database.sqlite"
    database.write_bytes(b"synthetic database bytes")
    result = {
        "n": 180,
        "n_correct": 170,
        "accuracy": 170 / 180,
        "n_answered": 176,
        "n_abstained": 4,
        "coverage": 176 / 180,
        "execution_accuracy_answered": 170 / 176,
        "strict_result_contract": True,
        "by_difficulty": {
            "medium": {"n": 60, "correct": 58},
            "hard": {"n": 60, "correct": 57},
            "extra": {"n": 30, "correct": 28},
            "extra-hard": {"n": 30, "correct": 27},
        },
        "by_category": {},
        "details": [{"id": "private-detail"}],
    }
    calls = []

    def recompute(rows, engine, db_path, *, strict):
        calls.append((rows, engine, Path(db_path), strict))
        return result

    monkeypatch.setattr(seal_module, "evaluate_set", recompute)
    freeze = {
        "kind": "independent_curated_v3_four_table_blind_freeze",
        "rid": rid,
        "development": {
            "path": development.name,
            "size": development.stat().st_size,
            "sha256": seal_module.sha256_file(development),
        },
        "database": {"sha256": seal_module.sha256_file(database)},
        "source_version": SOURCE_TREE_SHA256,
        "independent_curation": {
            "curator_pack_manifest_sha256": "a" * 64,
            "attestation_sha256": "b" * 64,
        },
    }
    overall = {key: value for key, value in result.items() if key != "details"}
    receipt = tmp_path / "independent-development-receipt.json"
    payload = {
        "schema_version": 1,
        "rid": rid,
        "dataset_role": "development_selection_used",
        "selection_used": True,
        "created_once": True,
        "engine": "rule",
        "development_evidence_kind": (
            "independent_curated_v3_four_table_development_receipt"
        ),
        "freeze_manifest_sha256": seal_module.sha256_file(freeze_manifest),
        "input_sha256": freeze["development"]["sha256"],
        "database_sha256": freeze["database"]["sha256"],
        "engine_source_sha256": seal_module.sha256_file(
            seal_module.ROOT / "dki" / "analysis" / "nl2sql.py"
        ),
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "freeze_source_tree_sha256": SOURCE_TREE_SHA256,
        "candidate_source_differs_from_freeze": False,
        "evaluator_sha256": seal_module.sha256_file(
            seal_module.ROOT
            / "scripts"
            / "eval_nl2sql_frozen_development.py"
        ),
        "protocol_sha256": seal_module.sha256_file(
            seal_module.ROOT / "dki" / "analysis" / "nl2sql_protocol.py"
        ),
        "overall": overall,
    }
    _write_json(receipt, payload)
    return {
        "calls": calls,
        "database": database,
        "freeze": freeze,
        "freeze_manifest": freeze_manifest,
        "payload": payload,
        "receipt": receipt,
    }


def test_independent_development_gate_recomputes_frozen_development(
    tmp_path,
    monkeypatch,
):
    fixture = _independent_gate_fixture(tmp_path, monkeypatch)
    result = seal_module._validate_development_gate(
        fixture["receipt"],
        freeze=fixture["freeze"],
        freeze_manifest=fixture["freeze_manifest"],
        db_path=fixture["database"],
        source_tree_sha256=SOURCE_TREE_SHA256,
    )

    assert result["n"] == 180
    assert result["hard_n"] == 120
    assert result["trusted_evidence"]["authority"] == (
        "independent_curated_v3_freeze_recomputation"
    )
    assert len(fixture["calls"]) == 1
    assert fixture["calls"][0][1:] == (
        "rule",
        fixture["database"],
        True,
    )


def test_independent_development_gate_accepts_distinct_candidate_rid(
    tmp_path,
    monkeypatch,
):
    fixture = _independent_gate_fixture(tmp_path, monkeypatch)
    candidate_rid = "20260730T120000Z-g1234567-tdddddddddddd"
    candidate_tree = "d" * 64
    fixture["payload"].update(
        {
            "schema_version": 2,
            "rid": candidate_rid,
            "candidate_rid": candidate_rid,
            "freeze_rid": fixture["freeze"]["rid"],
            "source_tree_sha256": candidate_tree,
            "candidate_source_differs_from_freeze": True,
        }
    )
    _write_json(fixture["receipt"], fixture["payload"])

    result = seal_module._validate_development_gate(
        fixture["receipt"],
        freeze=fixture["freeze"],
        freeze_manifest=fixture["freeze_manifest"],
        db_path=fixture["database"],
        source_tree_sha256=candidate_tree,
        candidate_rid=candidate_rid,
    )

    assert result["n"] == 180
    assert result["hard_correct"] == 112


def test_independent_development_gate_rejects_legacy_cross_rid_receipt(
    tmp_path,
    monkeypatch,
):
    fixture = _independent_gate_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="strict engine gate"):
        seal_module._validate_development_gate(
            fixture["receipt"],
            freeze=fixture["freeze"],
            freeze_manifest=fixture["freeze_manifest"],
            db_path=fixture["database"],
            source_tree_sha256=SOURCE_TREE_SHA256,
            candidate_rid="20260730T120000Z-g1234567-tcccccccccccc",
        )


def test_independent_development_gate_rejects_forged_metrics(
    tmp_path,
    monkeypatch,
):
    fixture = _independent_gate_fixture(tmp_path, monkeypatch)
    fixture["payload"]["overall"]["n_correct"] = 180
    _write_json(fixture["receipt"], fixture["payload"])

    with pytest.raises(ValueError, match="differ from recomputation"):
        seal_module._validate_development_gate(
            fixture["receipt"],
            freeze=fixture["freeze"],
            freeze_manifest=fixture["freeze_manifest"],
            db_path=fixture["database"],
            source_tree_sha256=SOURCE_TREE_SHA256,
        )


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_verify_candidate_seal_rechecks_recorded_trusted_gate_binding(
    tmp_path,
    monkeypatch,
):
    source_tree_sha256 = "b" * 64
    freeze_manifest = tmp_path / "freeze.json"
    task1_manifest = tmp_path / "task1.json"
    kg_path = tmp_path / "kg.json"
    database_path = tmp_path / "database.sqlite"
    source_manifest = tmp_path / "source-tree.json"
    execution_bundle = tmp_path / "execution-bundle.json"
    dependency_lock = tmp_path / "environment.lock.json"
    development_receipt = tmp_path / "development.json"
    regression_receipt = tmp_path / "regression.json"
    for path in (
        freeze_manifest,
        task1_manifest,
        kg_path,
        database_path,
        source_manifest,
        execution_bundle,
        dependency_lock,
        development_receipt,
        regression_receipt,
    ):
        path.write_text("synthetic source-only test artifact", encoding="utf-8")

    development_gate = {
        "n": 150,
        "n_correct": 135,
        "n_answered": 143,
        "hard_n": 50,
        "hard_correct": 39,
        "trusted_evidence": trusted_evidence_binding(ANCHOR),
    }
    regression_gate = {
        "n": 120,
        "n_correct": 112,
        "n_answered": 113,
        "easy_n": 44,
        "easy_correct": 44,
    }
    artifacts = {
        "freeze_manifest": _binding(freeze_manifest),
        "task1_manifest": _binding(task1_manifest),
        "task2_kg": _binding(kg_path),
        "database": _binding(database_path),
        "source_tree_manifest": _binding(source_manifest),
        "execution_bundle": _binding(execution_bundle),
        "dependency_lock": _binding(dependency_lock),
        "development_gate": _binding(development_receipt),
        "known_120_regression_gate": _binding(regression_receipt),
        "blind_evaluator": _binding(
            seal_module.ROOT / "scripts" / "eval_nl2sql_blind.py"
        ),
    }
    execution_bundle_closure = {
        "closure_sha256": "d" * 64,
        "entry_modules": ["dki.analysis.nl2sql"],
        "n_modules": 1,
        "n_files": 1,
    }
    seal = {
        "schema_version": 1,
        "kind": "sealed_nl2sql_blind_candidate",
        "status": "sealed",
        "engine": "rule",
        "holdout_opened_by_sealer": False,
        "holdout_sha256_declared": "a" * 64,
        "source_tree_sha256": source_tree_sha256,
        "freeze_source_tree_sha256": None,
        "candidate_source_differs_from_freeze": False,
        "environment": {},
        "artifacts": artifacts,
        "preconsumption_gates": {
            "development": {
                **development_gate,
                "trusted_evidence": {"substituted": "binding"},
            },
            "known_120_regression": regression_gate,
        },
        "execution_bundle_closure": execution_bundle_closure,
    }
    seal_path = tmp_path / "candidate-seal.json"
    _write_json(seal_path, seal)
    freeze = {
        "source_version": None,
        "holdout": {"sha256": "a" * 64},
    }

    monkeypatch.setattr(seal_module, "_assert_freeze_unconsumed", lambda _: None)
    monkeypatch.setattr(
        seal_module,
        "_verify_source_tree_manifest",
        lambda _: {"included_tree_sha256": source_tree_sha256},
    )
    monkeypatch.setattr(
        seal_module,
        "_verify_execution_bundle_closure",
        lambda *_args, **_kwargs: {
            "archive": _binding(execution_bundle),
            **execution_bundle_closure,
        },
    )
    monkeypatch.setattr(seal_module, "_environment", lambda: {})
    monkeypatch.setattr(
        seal_module,
        "_validate_dependency_lock",
        lambda _: {"source_tree_sha256": source_tree_sha256},
    )
    monkeypatch.setattr(seal_module, "validate_task2_artifact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(seal_module, "database_matches_manifest", lambda *_args: True)
    monkeypatch.setattr(seal_module, "_freeze_contract", lambda **_kwargs: freeze)
    monkeypatch.setattr(
        seal_module,
        "_validate_development_gate",
        lambda *_args, **_kwargs: development_gate,
    )
    monkeypatch.setattr(
        seal_module,
        "_validate_regression_gate",
        lambda *_args, **_kwargs: regression_gate,
    )

    with pytest.raises(ValueError, match="preconsumption gates no longer validate"):
        seal_module.verify_candidate_seal(
            seal_path,
            freeze_manifest=freeze_manifest,
            task1_manifest=task1_manifest,
            kg_path=kg_path,
            db_path=database_path,
            engine="rule",
        )


def test_verify_candidate_seal_rejects_missing_execution_bundle_binding(
    tmp_path,
    monkeypatch,
):
    seal_path = tmp_path / "candidate-seal.json"
    _write_json(
        seal_path,
        {
            "schema_version": 1,
            "kind": "sealed_nl2sql_blind_candidate",
            "status": "sealed",
            "engine": "rule",
            "holdout_opened_by_sealer": False,
            "artifacts": {
                key: {}
                for key in seal_module.REQUIRED_SEAL_ARTIFACT_BINDINGS
                if key != "execution_bundle"
            },
        },
    )
    monkeypatch.setattr(seal_module, "_assert_freeze_unconsumed", lambda _: None)

    with pytest.raises(
        ValueError,
        match="required artifact binding: execution_bundle",
    ):
        seal_module.verify_candidate_seal(
            seal_path,
            freeze_manifest=tmp_path / "freeze.json",
            task1_manifest=tmp_path / "task1.json",
            kg_path=tmp_path / "kg.json",
            db_path=tmp_path / "database.sqlite",
            engine="rule",
        )
