import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import pytest

from dki.analysis.database import build_database
import dki.analysis.nl2sql_seal as seal_module
from dki.analysis.nl2sql_seal import (
    INDEPENDENT_FIVE_FILE_CURATOR_FORMATS,
    INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS,
    REQUIRED_SOURCE_PATHS,
    seal_candidate,
    verify_candidate_seal,
)
from dki.analysis.nl2sql_source_closure import (
    DEFAULT_ENTRY_MODULES,
    build_source_closure_manifest,
    discover_source_closure,
)
from dki.pipeline.artifacts import write_task1_manifest, write_task2_receipt


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_seal_accepts_current_five_file_protocol_labels():
    assert "independent-v14" in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS
    assert "independent-curator-v14" in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS
    assert "independent-v15" in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS
    assert "independent-curator-v15" in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS
    assert "independent-v16" in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS
    assert "independent-curator-v16" in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS
    assert "independent-v17" in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS
    assert "independent-curator-v17" in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS
    assert "independent-v18" in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS
    assert "independent-curator-v18" in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS
    assert "independent-v19" in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS
    assert "independent-curator-v19" in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS
    assert "independent-v20" in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS
    assert "independent-curator-v20" in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS
    assert "independent-v21" in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS
    assert "independent-curator-v21" in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS
    assert "independent-v22" in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS
    assert "independent-curator-v22" in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS
    assert "independent-v23" in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS
    assert "independent-curator-v23" in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS
    assert "independent-v24" in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS
    assert "independent-curator-v24" in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS
    assert "independent-v25" in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS
    assert "independent-curator-v25" in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS
    assert "independent-v26" in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS
    assert "independent-curator-v26" in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS
    assert "independent-v27" in INDEPENDENT_FIVE_FILE_CURATOR_FORMATS
    assert "independent-curator-v27" in INDEPENDENT_FIVE_FILE_CURATOR_SOURCE_VERSIONS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lineage(tmp_path, monkeypatch):
    import dki.pipeline.artifacts as artifacts

    root = tmp_path / "project"
    source = root / "data" / "raw"
    structured = source / "structured"
    clean = root / "outputs" / "processed" / "runs" / "task1" / "clean"
    structured.mkdir(parents=True)
    clean.mkdir(parents=True)
    (clean / "doc.txt").write_text("medical text", encoding="utf-8")
    (structured / "patients.csv").write_text(
        "patient_id,name,gender,age,region\np1,A,F,30,Shanghai\n",
        encoding="utf-8",
    )
    (structured / "visits.csv").write_text(
        "visit_id,patient_id,visit_date,department,disease,"
        "drug_prescribed,exam,cost,los_days\n"
        "v1,p1,2026-01-01,Internal,Cold,DrugA,ExamA,10,0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(artifacts, "PROJECT_ROOT", root)
    monkeypatch.setattr(
        artifacts,
        "PROCESSED_DIR",
        root / "outputs" / "processed",
    )
    monkeypatch.setattr(artifacts, "TASK_DIR", root / "outputs" / "tasks")
    task1 = write_task1_manifest(
        task_id="task1",
        clean_dir=clean,
        source_text_dir=source,
        structured_source_dir=structured,
    )
    kg = root / "outputs" / "kg" / "medical_kg.json"
    kg.parent.mkdir(parents=True)
    kg.write_text('{"entities": {}, "triples": []}', encoding="utf-8")
    write_task2_receipt(kg, task1_path=task1, task_id="task2")
    db = root / "outputs" / "insight.db"
    build_database(db, manifest_path=task1, kg_path=kg)
    return task1, kg, db


def _source_tree_manifest(path: Path, *, extra_paths=()) -> Path:
    entries = []
    digest = hashlib.sha256()
    for relative in sorted(set(REQUIRED_SOURCE_PATHS) | set(extra_paths)):
        source = ROOT / relative
        entry = {
            "path": relative,
            "size": source.stat().st_size,
            "sha256": _sha256(source),
        }
        entries.append(entry)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "included_tree_sha256": digest.hexdigest(),
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return path


def _execution_closure_paths():
    return {
        entry["path"]
        for entry in discover_source_closure(ROOT, DEFAULT_ENTRY_MODULES)
    }


def _execution_bundle(path: Path, dependency_lock: Path) -> Path:
    del dependency_lock
    build_source_closure_manifest(
        path,
        source_root=ROOT,
        entry_modules=DEFAULT_ENTRY_MODULES,
    )
    return path


def _dependency_lock(path: Path, source_tree_sha256: str) -> Path:
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name", "")).strip()
        version = str(distribution.version).strip()
        if name and version:
            packages[name.casefold()] = {"name": name, "version": version}
    ordered = [packages[name] for name in sorted(packages)]
    canonical = "".join(
        f"{item['name'].casefold()}=={item['version']}\n" for item in ordered
    ).encode("utf-8")
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "python_environment_lock",
                "created_once": True,
                "source_tree_sha256": source_tree_sha256,
                "python": {
                    "version": platform.python_version(),
                    "implementation": platform.python_implementation(),
                },
                "platform": platform.platform(),
                "packages": ordered,
                "canonical_packages_sha256": hashlib.sha256(
                    canonical
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_candidate_seal_preflights_every_non_holdout_artifact(
    tmp_path,
    monkeypatch,
):
    task1, kg, db = _lineage(tmp_path, monkeypatch)
    # This integration fixture uses deliberately synthetic lineage hashes, which
    # cannot satisfy the reviewed static official-v2 development anchor.  The
    # anchor gate itself is covered by test_nl2sql_candidate_seal_anchor.py.
    synthetic_development_gate = {
        "n": 180,
        "n_correct": 180,
        "n_answered": 180,
        "hard_n": 120,
        "hard_correct": 120,
        "trusted_evidence": {"synthetic_test_stub": True},
    }
    monkeypatch.setattr(
        seal_module,
        "_validate_development_gate",
        lambda *_args, **_kwargs: synthetic_development_gate,
    )
    freeze = tmp_path / "outputs" / "private" / "rid" / "freeze_manifest.json"
    freeze.parent.mkdir(parents=True)
    freeze.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "created_once": True,
                "selection_used": False,
                "n_development": 180,
                "n_holdout": 120,
                "n_hard": 80,
                "task1_manifest": {"sha256": _sha256(task1)},
                "kg": {"sha256": _sha256(kg)},
                "database": {"sha256": _sha256(db)},
                "source_version": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    tree = _source_tree_manifest(
        tmp_path / "included_tree.json",
        extra_paths=_execution_closure_paths(),
    )
    tree_hash = json.loads(tree.read_text())["included_tree_sha256"]
    engine_sha256 = _sha256(ROOT / "dki" / "analysis" / "nl2sql.py")
    development_receipt = tmp_path / "development_receipt.json"
    development_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_role": "development_selection_used",
                "selection_used": True,
                "engine": "rule",
                "input_sha256": "d" * 64,
                "database_sha256": _sha256(db),
                "engine_source_sha256": engine_sha256,
                "source_tree_sha256": tree_hash,
                "overall": {
                    "n": 180,
                    "n_correct": 180,
                    "n_answered": 180,
                    "strict_result_contract": True,
                    "by_difficulty": {
                        "medium": {"n": 60, "correct": 60},
                        "hard": {"n": 120, "correct": 120},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    freeze_payload = json.loads(freeze.read_text(encoding="utf-8"))
    freeze_payload["development"] = {"sha256": "d" * 64}
    freeze.write_text(json.dumps(freeze_payload), encoding="utf-8")
    frozen_manifest_bytes = freeze.read_bytes()
    base = tmp_path / "base.jsonl"
    hard = tmp_path / "hard.jsonl"
    base.write_text("{}\n", encoding="utf-8")
    hard.write_text("{}\n", encoding="utf-8")
    regression_receipt = tmp_path / "known_120.json"
    regression_receipt.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "dataset_role": "development_selection_used",
                "selection_used": True,
                "engine": "rule",
                "strict_result_contract": True,
                "database_sha256": _sha256(db),
                "engine_source_sha256": engine_sha256,
                "source_tree_sha256": tree_hash,
                "input_files": {
                    "base": {
                        "path": str(base.resolve()),
                        "sha256": _sha256(base),
                    },
                    "hard": {
                        "path": str(hard.resolve()),
                        "sha256": _sha256(hard),
                    },
                },
                "n": 120,
                "n_correct": 112,
                "n_answered": 113,
                "by_difficulty": {
                    "easy": {"n": 44, "correct": 44},
                    "medium": {"n": 8, "correct": 8},
                    "hard": {"n": 68, "correct": 60},
                },
            }
        ),
        encoding="utf-8",
    )
    dependency_lock = _dependency_lock(
        tmp_path / "environment.lock.json",
        tree_hash,
    )
    execution_bundle = _execution_bundle(
        tmp_path / "nl2sql_execution_bundle.zip",
        dependency_lock,
    )
    seal = tmp_path / "outputs" / "remote_runs" / "rid" / "candidate_seal.json"

    payload = seal_candidate(
        output=seal,
        freeze_manifest=freeze,
        development_receipt=development_receipt,
        regression_receipt=regression_receipt,
        task1_manifest=task1,
        kg_path=kg,
        db_path=db,
        source_tree_manifest=tree,
        execution_bundle=execution_bundle,
        dependency_spec=ROOT / "pyproject.toml",
        dependency_lock=dependency_lock,
    )
    assert payload["status"] == "sealed"
    assert payload["engine"] == "rule"
    assert payload["holdout_opened_by_sealer"] is False
    assert payload["freeze_source_tree_sha256"] == "a" * 64
    assert payload["candidate_source_differs_from_freeze"] is True
    verify_candidate_seal(
        seal,
        freeze_manifest=freeze,
        task1_manifest=task1,
        kg_path=kg,
        db_path=db,
        engine="rule",
    )
    with pytest.raises(FileExistsError):
        seal_candidate(
            output=seal,
            freeze_manifest=freeze,
            development_receipt=development_receipt,
            regression_receipt=regression_receipt,
            task1_manifest=task1,
            kg_path=kg,
            db_path=db,
            source_tree_manifest=tree,
            execution_bundle=execution_bundle,
            dependency_spec=ROOT / "pyproject.toml",
            dependency_lock=dependency_lock,
        )

    import dki.paths as paths

    monkeypatch.setattr(paths, "LLM_API_KEY", "test-secret")
    monkeypatch.setattr(paths, "LLM_API_URL", "https://provider.invalid/v1")
    monkeypatch.setattr(paths, "LLM_MODEL_NAME", "sealed-model")
    agent_seal = seal.with_name("agent_candidate_seal.json")
    agent_payload = seal_candidate(
        output=agent_seal,
        freeze_manifest=freeze,
        development_receipt=development_receipt,
        regression_receipt=regression_receipt,
        task1_manifest=task1,
        kg_path=kg,
        db_path=db,
        source_tree_manifest=tree,
        execution_bundle=execution_bundle,
        dependency_spec=ROOT / "pyproject.toml",
        dependency_lock=dependency_lock,
        engine="agent",
        llm_strategy="legacy_sql",
        agent_rule_policy="model_first",
        llm_reasoning_mode="adaptive",
        llm_completion_tokens=4096,
    )
    assert agent_payload["engine"] == "agent"
    assert agent_payload["agent_profile"]["contains_api_key"] is False
    assert agent_payload["agent_profile"]["agent_rule_policy"] == "model_first"
    assert agent_payload["agent_profile"]["llm_reasoning_mode"] == "adaptive"
    assert agent_payload["agent_profile"]["llm_completion_tokens"] == 4096
    verify_candidate_seal(
        agent_seal,
        freeze_manifest=freeze,
        task1_manifest=task1,
        kg_path=kg,
        db_path=db,
        engine="agent",
        llm_strategy="legacy_sql",
        agent_rule_policy="model_first",
        llm_reasoning_mode="adaptive",
        llm_completion_tokens=4096,
    )
    monkeypatch.setattr(paths, "LLM_API_URL", "https://changed.invalid/v1")
    with pytest.raises(ValueError, match="runtime profile differs"):
        verify_candidate_seal(
            agent_seal,
            freeze_manifest=freeze,
            task1_manifest=task1,
            kg_path=kg,
            db_path=db,
            engine="agent",
            llm_strategy="legacy_sql",
            agent_rule_policy="model_first",
            llm_reasoning_mode="adaptive",
            llm_completion_tokens=4096,
        )
    monkeypatch.setattr(paths, "LLM_API_URL", "https://provider.invalid/v1")

    freeze.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact binding failed"):
        verify_candidate_seal(
            seal,
            freeze_manifest=freeze,
            task1_manifest=task1,
            kg_path=kg,
            db_path=db,
            engine="rule",
        )

    freeze.write_bytes(frozen_manifest_bytes)
    (freeze.parent / "consumed-started.json").write_text(
        '{"status":"consumed-started"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="already been consumed"):
        verify_candidate_seal(
            seal,
            freeze_manifest=freeze,
            task1_manifest=task1,
            kg_path=kg,
            db_path=db,
            engine="rule",
        )
    with pytest.raises(ValueError, match="already been consumed"):
        seal_candidate(
            output=seal.with_name("second_candidate_seal.json"),
            freeze_manifest=freeze,
            development_receipt=development_receipt,
            regression_receipt=regression_receipt,
            task1_manifest=task1,
            kg_path=kg,
            db_path=db,
            source_tree_manifest=tree,
            execution_bundle=execution_bundle,
            dependency_spec=ROOT / "pyproject.toml",
            dependency_lock=dependency_lock,
        )


def test_official_v2_freeze_contract_binds_completion_without_opening_holdout(
    tmp_path,
    monkeypatch,
):
    import dki.analysis.nl2sql_seal as seal_module

    task1, kg, db = _lineage(tmp_path, monkeypatch)
    rid = "20260726T130000Z-g1234567-t" + "a" * 12
    control = (
        tmp_path / "outputs" / "private" / rid / "nl2sql_blind_v2"
    )
    frozen = control / "frozen_set"
    frozen.mkdir(parents=True)
    development = frozen / "development.jsonl"
    development.write_text("{}\n", encoding="utf-8")
    holdout = frozen / "holdout.jsonl"
    holdout.write_bytes(b"sealed holdout bytes")
    started = control / "projection-started.json"
    started.write_text(
        json.dumps(
            {
                "status": "projection-started",
                "rid": rid,
                "official_test_content_opened_before_this_marker": False,
                "reuse_permitted_after_this_marker": False,
            }
        ),
        encoding="utf-8",
    )
    freeze = frozen / "freeze_manifest.json"
    freeze.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "kind": "official_public_v2_four_table_blind_freeze",
                "created_once": True,
                "selection_used": False,
                "n_development": 180,
                "n_holdout": 120,
                "n_hard": 80,
                "task1_manifest": {"sha256": _sha256(task1)},
                "kg": {"sha256": _sha256(kg)},
                "database": {"sha256": _sha256(db)},
                "partition": {
                    "version": "v2",
                    "development_holdout_normalised_question_intersection": 0,
                    "development_holdout_skeleton_intersection": 0,
                },
                "development": {
                    "path": development.name,
                    "size": development.stat().st_size,
                    "sha256": _sha256(development),
                },
                "holdout": {
                    "path": holdout.name,
                    "size": holdout.stat().st_size,
                    "sha256": _sha256(holdout),
                },
                "evidence": {
                    "projection_started": {
                        "path": str(started.resolve()),
                        "size": started.stat().st_size,
                        "sha256": _sha256(started),
                    }
                },
                "independence": {
                    "official_test_content_opened_before_marker": False,
                    "official_test_content_opened_after_marker": True,
                    "project_nl2sql_engine_imported_by_projector": False,
                    "holdout_used_for_selection": False,
                    "holdout_reusable_after_failure": False,
                },
                "privacy": {
                    "manifest_contains_questions": False,
                    "manifest_contains_gold_sql": False,
                    "manifest_contains_error_samples": False,
                    "official_source_or_projection_allowed_in_submission": False,
                },
            }
        ),
        encoding="utf-8",
    )
    public = (
        tmp_path
        / "outputs"
        / "remote_runs"
        / rid
        / "nl2sql"
        / "official_v2_freeze_attestation.json"
    )
    public.parent.mkdir(parents=True)
    public.write_text(
        json.dumps(
            {
                "kind": "official_public_v2_blind_freeze_attestation",
                "status": "frozen-unconsumed",
                "rid": rid,
                "freeze_manifest_sha256": _sha256(freeze),
                "selection_used": False,
                "official_test_content_opened_after_marker": True,
                "contains_questions": False,
                "contains_gold_sql": False,
                "contains_predictions": False,
                "contains_error_samples": False,
                "counts": {
                    "n_development": 180,
                    "n_holdout": 120,
                    "n_hard": 80,
                },
            }
        ),
        encoding="utf-8",
    )
    completed = control / "projection-completed.json"
    completed.write_text(
        json.dumps(
            {
                "status": "projection-completed",
                "rid": rid,
                "projection_started_sha256": _sha256(started),
                "freeze_manifest_sha256": _sha256(freeze),
                "public_attestation_sha256": _sha256(public),
                "n_holdout": 120,
                "n_hard": 80,
                "selection_used": False,
            }
        ),
        encoding="utf-8",
    )

    original_sha256_file = seal_module.sha256_file

    def refuse_holdout_hash(path):
        if Path(path).resolve() == holdout.resolve():
            raise AssertionError("sealer opened the holdout")
        return original_sha256_file(path)

    monkeypatch.setattr(seal_module, "sha256_file", refuse_holdout_hash)
    payload = seal_module._freeze_contract(
        freeze_manifest=freeze,
        task1_manifest=task1,
        kg_path=kg,
        db_path=db,
    )
    assert payload["n_holdout"] == 120
    completed.unlink()
    with pytest.raises(ValueError, match="projection control evidence"):
        seal_module._freeze_contract(
            freeze_manifest=freeze,
            task1_manifest=task1,
            kg_path=kg,
            db_path=db,
        )


def test_mimicsql_v3_freeze_contract_never_hashes_holdout(
    tmp_path,
    monkeypatch,
):
    import dki.analysis.nl2sql_seal as seal_module

    task1, kg, db = _lineage(tmp_path, monkeypatch)
    rid = "20260726T140000Z-g1234567-t" + "b" * 12
    control = (
        tmp_path
        / "outputs"
        / "private"
        / rid
        / "nl2sql_mimicsql_official_v3"
    )
    frozen = control / "frozen"
    frozen.mkdir(parents=True)
    development = frozen / "development.jsonl"
    development.write_text("{}\n", encoding="utf-8")
    holdout = frozen / "holdout.jsonl"
    holdout.write_bytes(b"sealed MIMICSQL holdout bytes")
    started = control / "projection-started.json"
    started.write_text(
        json.dumps(
            {
                "status": "projection-started",
                "rid": rid,
                "official_test_content_opened_before_this_marker": False,
                "reuse_permitted_after_this_marker": False,
            }
        ),
        encoding="utf-8",
    )
    freeze = frozen / "freeze_manifest.json"
    freeze.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "kind": "mimicsql_v3_four_table_blind_freeze",
                "created_once": True,
                "selection_used": False,
                "n_development": 180,
                "n_holdout": 120,
                "n_hard": 100,
                "task1_manifest": {"sha256": _sha256(task1)},
                "kg": {"sha256": _sha256(kg)},
                "database": {"sha256": _sha256(db)},
                "benchmark": {
                    "name": "MIMICSQL-natural-v2",
                    "source_revision": "1" * 40,
                },
                "partition": {
                    "version": "v3",
                    "development_holdout_normalised_question_intersection": 0,
                    "development_holdout_skeleton_intersection": 0,
                },
                "development": {
                    "path": development.name,
                    "size": development.stat().st_size,
                    "sha256": _sha256(development),
                },
                "holdout": {
                    "path": holdout.name,
                    "size": holdout.stat().st_size,
                    "sha256": _sha256(holdout),
                },
                "evidence": {
                    "projection_started": {
                        "path": str(started.resolve()),
                        "size": started.stat().st_size,
                        "sha256": _sha256(started),
                    }
                },
                "independence": {
                    "official_test_content_opened_before_marker": False,
                    "official_test_content_opened_after_marker": True,
                    "project_nl2sql_engine_imported_by_projector": False,
                    "holdout_used_for_selection": False,
                    "holdout_reusable_after_failure": False,
                },
                "privacy": {
                    "manifest_contains_questions": False,
                    "manifest_contains_gold_sql": False,
                    "manifest_contains_error_samples": False,
                    "official_source_or_projection_allowed_in_submission": False,
                },
            }
        ),
        encoding="utf-8",
    )
    public = (
        tmp_path
        / "outputs"
        / "remote_runs"
        / rid
        / "nl2sql"
        / "mimicsql_official_freeze_attestation.json"
    )
    public.parent.mkdir(parents=True)
    public.write_text(
        json.dumps(
            {
                "kind": "mimicsql_v3_official_blind_freeze_attestation",
                "status": "frozen-unconsumed",
                "rid": rid,
                "freeze_manifest_sha256": _sha256(freeze),
                "selection_used": False,
                "official_test_content_opened_after_marker": True,
                "contains_questions": False,
                "contains_gold_sql": False,
                "contains_predictions": False,
                "contains_error_samples": False,
                "counts": {
                    "n_development": 180,
                    "n_holdout": 120,
                    "n_hard": 100,
                },
            }
        ),
        encoding="utf-8",
    )
    completed = control / "projection-completed.json"
    completed.write_text(
        json.dumps(
            {
                "status": "projection-completed",
                "rid": rid,
                "projection_started_sha256": _sha256(started),
                "freeze_manifest_sha256": _sha256(freeze),
                "public_attestation_sha256": _sha256(public),
                "n_holdout": 120,
                "n_hard": 100,
                "selection_used": False,
            }
        ),
        encoding="utf-8",
    )

    original_sha256_file = seal_module.sha256_file

    def refuse_holdout_hash(path):
        if Path(path).resolve() == holdout.resolve():
            raise AssertionError("sealer opened the MIMICSQL holdout")
        return original_sha256_file(path)

    monkeypatch.setattr(seal_module, "sha256_file", refuse_holdout_hash)
    payload = seal_module._freeze_contract(
        freeze_manifest=freeze,
        task1_manifest=task1,
        kg_path=kg,
        db_path=db,
    )
    assert payload["n_holdout"] == 120
    public.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="public freeze attestation"):
        seal_module._freeze_contract(
            freeze_manifest=freeze,
            task1_manifest=task1,
            kg_path=kg,
            db_path=db,
        )


def test_independent_v3_freeze_contract_never_hashes_holdout(
    tmp_path,
    monkeypatch,
):
    import dki.analysis.nl2sql_seal as seal_module

    task1, kg, db = _lineage(tmp_path, monkeypatch)
    rid = "20260726T150000Z-g1234567-t" + "c" * 12
    control = (
        tmp_path / "outputs" / "private" / rid / "nl2sql_blind_v3"
    )
    frozen = control / "frozen"
    frozen.mkdir(parents=True)
    development = frozen / "development.jsonl"
    development.write_text("{}\n", encoding="utf-8")
    holdout = frozen / "holdout.jsonl"
    holdout.write_bytes(b"sealed independent holdout bytes")

    pack = control / "evidence" / "pack_manifest.json"
    pack.parent.mkdir(parents=True)
    pack.write_text(
        json.dumps(
            {
                "kind": "engine_blind_nl2sql_curator_pack",
                "dataset_role": "independent_curation_input_only",
                "created_once": True,
                "contains_engine_implementation": False,
                "contains_questions_or_gold_sql": False,
            }
        ),
        encoding="utf-8",
    )
    declaration = {
        "kind": "independent_four_table_nl2sql_curator_attestation",
        "independent_curation": True,
        "all_gold_sql_executed": True,
        "all_pairs_semantically_reviewed": True,
        "engine_context_provided": False,
        "local_evaluation_context_provided": False,
        "project_local_question_reuse": False,
        "project_local_question_paraphrase": False,
        "development_or_holdout_preassigned": False,
        "curator_identity": "independent-curator",
        "semantic_reviewer_identity": "independent-reviewer",
        "curator_pack_manifest_sha256": _sha256(pack),
        "corpus_sha256": "d" * 64,
    }
    attestation = control / "evidence" / "curator_attestation.json"
    attestation.write_text(json.dumps(declaration), encoding="utf-8")

    freeze = frozen / "freeze_manifest.json"
    freeze_payload = {
        "schema_version": 4,
        "kind": "independent_curated_v3_four_table_blind_freeze",
        "rid": rid,
        "created_once": True,
        "selection_used": False,
        "source_version": "a" * 64,
        "freeze_tool_sha256": _sha256(
            ROOT / "scripts" / "freeze_nl2sql_blind.py"
        ),
        "protocol_sha256": _sha256(
            ROOT / "dki" / "analysis" / "nl2sql_protocol.py"
        ),
        "n_total": 300,
        "n_development": 180,
        "n_holdout": 120,
        "n_hard": 90,
        "n_sql_ast_groups": 30,
        "input_sha256": declaration["corpus_sha256"],
        "task1_manifest": {"sha256": _sha256(task1)},
        "kg": {"sha256": _sha256(kg)},
        "database": {"sha256": _sha256(db)},
        "independent_curation": {
            "curator_pack_manifest_path": str(pack.resolve()),
            "curator_pack_manifest_sha256": _sha256(pack),
            "attestation_path": str(attestation.resolve()),
            "attestation_sha256": _sha256(attestation),
            "declaration": declaration,
        },
        "partition": {
            "version": "v3",
            "normalised_question_intersection": 0,
            "sql_ast_skeleton_intersection": 0,
            "whole_ast_groups_only": True,
        },
        "privacy": {
            "manifest_contains_questions": False,
            "manifest_contains_gold_sql": False,
            "manifest_contains_predictions": False,
            "manifest_contains_error_samples": False,
            "raw_corpus_or_frozen_rows_allowed_in_submission": False,
        },
        "development": {
            "path": development.name,
            "size": development.stat().st_size,
            "sha256": _sha256(development),
        },
        "holdout": {
            "path": holdout.name,
            "size": holdout.stat().st_size,
            "sha256": _sha256(holdout),
        },
    }
    freeze.write_text(json.dumps(freeze_payload), encoding="utf-8")

    public = (
        tmp_path
        / "outputs"
        / "remote_runs"
        / rid
        / "nl2sql"
        / "independent_v3_freeze_attestation.json"
    )
    public.parent.mkdir(parents=True)
    public.write_text(
        json.dumps(
            {
                "kind": "independent_curated_v3_blind_freeze_attestation",
                "status": "frozen-unconsumed",
                "rid": rid,
                "source_tree_sha256": freeze_payload[
                    "source_version"
                ],
                "freeze_manifest_sha256": _sha256(freeze),
                "selection_used": False,
                "contains_questions": False,
                "contains_gold_sql": False,
                "contains_predictions": False,
                "contains_error_samples": False,
                "counts": {
                    "n_total": 300,
                    "n_development": 180,
                    "n_holdout": 120,
                    "n_hard": 90,
                    "n_sql_ast_groups": 30,
                },
                "bindings": {
                    "curator_pack_manifest_sha256": _sha256(pack),
                    "curator_attestation_sha256": _sha256(attestation),
                    "corpus_sha256": declaration["corpus_sha256"],
                    "task1_manifest_sha256": _sha256(task1),
                    "kg_sha256": _sha256(kg),
                    "database_sha256": _sha256(db),
                },
            }
        ),
        encoding="utf-8",
    )

    original_sha256_file = seal_module.sha256_file

    def refuse_holdout_hash(path):
        if Path(path).resolve() == holdout.resolve():
            raise AssertionError("sealer opened independent holdout")
        return original_sha256_file(path)

    monkeypatch.setattr(seal_module, "sha256_file", refuse_holdout_hash)
    payload = seal_module._freeze_contract(
        freeze_manifest=freeze,
        task1_manifest=task1,
        kg_path=kg,
        db_path=db,
    )
    assert payload["n_holdout"] == 120
    declaration["all_pairs_semantically_reviewed"] = False
    freeze_payload["independent_curation"]["declaration"] = declaration
    freeze.write_text(json.dumps(freeze_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="curation/isolation contract"):
        seal_module._freeze_contract(
            freeze_manifest=freeze,
            task1_manifest=task1,
            kg_path=kg,
            db_path=db,
        )
