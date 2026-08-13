import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from dki.analysis.database import build_database
from dki.pipeline.artifacts import write_task1_manifest, write_task2_receipt


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_nl2sql_curator_pack.py"
SPEC = importlib.util.spec_from_file_location("build_nl2sql_curator_pack", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


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


def _audit(path: Path, *, benchmark: str = "EHRSQL") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "public_source_structure_audit",
                "benchmark": benchmark,
                "dataset_role": "public_curation_source_selection_used",
                "four_table_blind_gate": False,
                "contains_questions_or_sql": False,
                "source_url": "https://example.invalid/benchmark",
                "source_revision": "1" * 40,
                "splits": {"test": {"rows": 200}},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_curator_pack_is_create_once_engine_blind_and_hash_bound(
    tmp_path,
    monkeypatch,
):
    task1, kg, db = _lineage(tmp_path, monkeypatch)
    first = _audit(tmp_path / "ehrsql.json", benchmark="EHRSQL")
    second = _audit(tmp_path / "css.json", benchmark="CSS")
    output = (
        tmp_path
        / "outputs"
        / "private"
        / "rid"
        / "nl2sql_blind_v4"
        / "curator_pack"
    )

    manifest = MODULE.build_pack(
        output_dir=output,
        db_path=db,
        task1_manifest=task1,
        kg_path=kg,
        source_audits=[first, second],
        source_tree_sha256="a" * 64,
    )

    assert manifest["contains_engine_implementation"] is False
    assert manifest["contains_questions_or_gold_sql"] is False
    assert manifest["contract"]["n_items"] == 300
    assert manifest["contract"]["curator_source_version"] == "independent-curator-v21"
    assert sum(manifest["contract"]["difficulty_quotas"].values()) == 300
    assert sum(manifest["contract"]["category_quotas"].values()) == 300
    assert manifest["lineage"]["database_sha256"] == _sha256(db)
    assert (output / "four_table.sqlite").read_bytes() == db.read_bytes()
    assert set(json.loads((output / "schema_profile.json").read_text())["tables"]) == {
        "patients",
        "visits",
        "kg_entities",
        "kg_triples",
    }
    spec = json.loads((output / "curation_spec.json").read_text())
    assert spec["isolation"]["engine_context_provided"] is False
    assert spec["construction_policy"]["do_not_preassign_development_or_holdout"]
    grounding = spec["review_policy"]["machine_grounding_gate"]
    assert grounding["required_before_publication"] is True
    assert grounding["implicit_paraphrase_is_insufficient"] is True
    assert grounding["empty_string_predicate_forbidden"] is True
    assert grounding["authoritative_preflight"] == "grounding_preflight.py"
    assert "18" in grounding["example"]
    assert (output / "grounding_preflight.py").is_file()
    assert len(list((output / "public_source_catalog").glob("*.json"))) == 2
    handoff = json.loads((output / "five_file_handoff_contract.json").read_text())
    assert handoff["attestation_required_shape"]["corpus"] == {
        "record_count": 300,
        "sha256": "lowercase SHA-256 of the finalized corpus bytes",
    }
    assert handoff["attestation_required_shape"]["database"]["sha256"] == _sha256(db)
    assert all(
        value is False
        for key, value in handoff["attestation_required_shape"][
            "independence_declarations"
        ].items()
        if key != "database_schema_and_self_generated_query_results_only"
    )
    assert handoff["sha256s_contract"]["exact_file_set"] == [
        "corpus_v21.jsonl",
        "curator_attestation_v21.json",
        "aggregate_audit_v21.json",
        "execution_audit_v21.jsonl",
    ]
    assert not any(path.name == "nl2sql.py" for path in output.rglob("*"))
    with pytest.raises(FileExistsError):
        MODULE.build_pack(
            output_dir=output,
            db_path=db,
            task1_manifest=task1,
            kg_path=kg,
            source_audits=[first],
            source_tree_sha256="a" * 64,
        )


def test_curator_pack_rejects_row_level_or_gate_mislabeled_source(
    tmp_path,
):
    leaking = _audit(tmp_path / "leaking.json")
    payload = json.loads(leaking.read_text())
    payload["questions"] = ["hidden"]
    leaking.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="row-level"):
        MODULE._validate_source_audit(leaking)

    mislabeled = _audit(tmp_path / "mislabeled.json")
    payload = json.loads(mislabeled.read_text())
    payload["four_table_blind_gate"] = True
    mislabeled.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="four_table_blind_gate=false"):
        MODULE._validate_source_audit(mislabeled)


def test_curator_pack_requires_private_create_once_destination(
    tmp_path,
    monkeypatch,
):
    task1, kg, db = _lineage(tmp_path, monkeypatch)
    audit = _audit(tmp_path / "source.json")
    with pytest.raises(ValueError, match="outputs/private"):
        MODULE.build_pack(
            output_dir=tmp_path / "public" / "curator_pack",
            db_path=db,
            task1_manifest=task1,
            kg_path=kg,
            source_audits=[audit],
            source_tree_sha256="b" * 64,
        )
