import json

import pytest

from dki.analysis.database import build_database, database_matches_manifest
from dki.pipeline.artifacts import (
    load_task1_manifest,
    task1_clean_dir,
    task1_table_paths,
    validate_task2_artifact,
    write_task1_manifest,
    write_task2_receipt,
)


def test_manifest_snapshot_owns_downstream_inputs(tmp_path, monkeypatch):
    import dki.pipeline.artifacts as artifacts

    root = tmp_path / "project"
    source = root / "data" / "raw"
    structured = source / "structured"
    clean = root / "outputs" / "processed" / "runs" / "task1" / "clean"
    structured.mkdir(parents=True)
    clean.mkdir(parents=True)
    (clean / "doc.txt").write_text("脱敏后的文本", encoding="utf-8")
    (structured / "patients.csv").write_text("patient_id,name,gender,age,region\np1,a,男,30,北京\n", encoding="utf-8")
    (structured / "visits.csv").write_text("visit_id,patient_id,visit_date,department,disease,drug_prescribed,exam,cost,los_days\nv1,p1,2026-01-01,内科,感冒,,,1,0\n", encoding="utf-8")
    monkeypatch.setattr(artifacts, "PROJECT_ROOT", root)
    monkeypatch.setattr(artifacts, "PROCESSED_DIR", root / "outputs" / "processed")
    monkeypatch.setattr(artifacts, "TASK_DIR", root / "outputs" / "tasks")

    path = write_task1_manifest(task_id="task1", clean_dir=clean, source_text_dir=source, structured_source_dir=structured)
    manifest = load_task1_manifest(path)
    assert task1_clean_dir(manifest) == clean
    assert set(task1_table_paths(manifest)) == {"patients", "visits"}
    assert json.loads(path.read_text(encoding="utf-8"))["task_id"] == "task1"

    kg = root / "outputs" / "kg" / "medical_kg.json"
    kg.parent.mkdir(parents=True)
    kg.write_text('{"entities": {}, "triples": []}', encoding="utf-8")
    write_task2_receipt(kg, task1_path=path, task_id="task2")
    validate_task2_artifact(kg, task1_path=path)

    db = root / "outputs" / "insight.db"
    build_database(db, manifest_path=path, kg_path=kg)
    assert database_matches_manifest(db, path, kg)
    with db.open("ab") as handle:
        handle.write(b"tampered")
    assert not database_matches_manifest(db, path, kg)

    kg.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        validate_task2_artifact(kg, task1_path=path)


def test_manifest_refuses_incomplete_structured_snapshot(tmp_path, monkeypatch):
    import dki.pipeline.artifacts as artifacts

    root = tmp_path / "project"
    source = root / "data" / "raw"
    structured = source / "structured"
    clean = root / "outputs" / "processed" / "runs" / "task1" / "clean"
    structured.mkdir(parents=True)
    clean.mkdir(parents=True)
    (clean / "doc.txt").write_text("text", encoding="utf-8")
    (structured / "patients.csv").write_text("patient_id\np1\n", encoding="utf-8")
    monkeypatch.setattr(artifacts, "PROJECT_ROOT", root)
    monkeypatch.setattr(artifacts, "PROCESSED_DIR", root / "outputs" / "processed")
    monkeypatch.setattr(artifacts, "TASK_DIR", root / "outputs" / "tasks")

    with pytest.raises(FileNotFoundError, match="visits.csv"):
        write_task1_manifest(
            task_id="task1",
            clean_dir=clean,
            source_text_dir=source,
            structured_source_dir=structured,
        )
