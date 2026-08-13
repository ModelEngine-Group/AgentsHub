import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path

import pytest

from dki.analysis.nl2sql_css_projection import ProjectedItem
from dki.analysis.nl2sql_protocol import validate_gold_sql


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_nl2sql_official_v2.py"
SPEC = importlib.util.spec_from_file_location(
    "freeze_nl2sql_official_v2",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

RID = "20260726T130000Z-g1234567-t" + "a" * 12

# Wall-clock budget for a spawned child to boot, import this module chain and
# report its race outcome.  Measured at 8-13s for two children on an otherwise
# idle aarch64 host, and the suite runs there at load average 20-40, so the
# previous 20s left under 2x headroom and expired intermittently.  This bounds
# the harness wait only; the race assertions themselves are unchanged, and a
# genuine publish deadlock still fails the test rather than hanging forever.
_SPAWN_REPORT_TIMEOUT_S = 120


def _race_publish(path: str, start, queue) -> None:
    from scripts.freeze_nl2sql_official_v2 import _publish_json_once

    start.wait(timeout=10)
    try:
        _publish_json_once(Path(path), {"status": "winner"})
    except FileExistsError:
        queue.put("occupied")
    else:
        queue.put("published")


def _touch(path: Path, content: bytes = b"bound\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _synthetic_context(tmp_path: Path) -> dict:
    private_root = (
        tmp_path / "outputs" / "private" / RID / "nl2sql_blind_v2"
    )
    remote_root = tmp_path / "outputs" / "remote_runs" / RID
    development = _touch(
        private_root / "development.jsonl",
        b"".join(b'{"synthetic":true}\n' for _ in range(150)),
    )
    task1 = _touch(tmp_path / "lineage" / "manifest.json")
    kg = _touch(tmp_path / "lineage" / "medical_kg.json")
    db = _touch(tmp_path / "lineage" / "four_table.sqlite")
    source_tree = _touch(remote_root / "source" / "included_tree.json")
    css_manifest = _touch(tmp_path / "external" / "css" / "freeze.json")
    ehr_audit = _touch(tmp_path / "external" / "ehr-audit.json")
    projection_receipt = _touch(
        remote_root / "nl2sql" / "development_projection_receipt.json"
    )
    development_evaluation = _touch(
        private_root / "development_evaluation.json"
    )
    regression = _touch(private_root / "known_120.json")
    output_manifest = private_root / "frozen_set" / "freeze_manifest.json"
    public_output = (
        remote_root / "nl2sql" / "official_v2_freeze_attestation.json"
    )
    return {
        "rid": RID,
        "control_root": private_root,
        "started_marker": private_root / "projection-started.json",
        "invalid_marker": private_root / "projection-invalid.json",
        "completed_marker": private_root / "projection-completed.json",
        "output_manifest": output_manifest,
        "public_output": public_output,
        "created_at": "2026-07-26T13:00:00Z",
        "source_tree_manifest": source_tree,
        "source_tree_sha256": "a" * 64,
        "task1_manifest": task1,
        "kg_path": kg,
        "db_path": db,
        "css_freeze_manifest": css_manifest,
        "ehrsql_audit": ehr_audit,
        "development_path": development,
        "development_rows": [{} for _ in range(150)],
        "development_questions": frozenset(),
        "development_skeletons": frozenset(),
        "development_projection_receipt": projection_receipt,
        "development_evaluation_receipt": development_evaluation,
        "regression_receipt": regression,
        "development_gate": {
            "n": 150,
            "n_correct": 150,
            "n_answered": 150,
            "hard_n": 40,
            "hard_correct": 40,
        },
        "regression_gate": {
            "n": 120,
            "n_correct": 112,
            "n_answered": 113,
            "easy_n": 44,
            "easy_correct": 44,
        },
    }


def _holdout_items() -> list[ProjectedItem]:
    items = []
    for index in range(120):
        sql = (
            "SELECT patient_id FROM patients "
            f"WHERE age >= {index + 1}"
        )
        skeleton = validate_gold_sql(sql)
        skeleton_sha256 = hashlib.sha256(
            skeleton.encode("utf-8")
        ).hexdigest()
        items.append(
            ProjectedItem(
                item={
                    "id": f"official-{index:04d}",
                    "question": f"synthetic private question {index}",
                    "gold_sql": sql,
                    "difficulty": "hard" if index < 40 else "medium",
                    "category": "filtering",
                    "provenance": {
                        "benchmark": "CSS",
                        "partition_version": "v2",
                        "skeleton_role": "holdout",
                        "skeleton_sha256": skeleton_sha256,
                    },
                },
                skeleton_role="holdout",
                skeleton_sha256=skeleton_sha256,
            )
        )
    return items


def _freeze_kwargs(context: dict) -> dict:
    return {
        "css_freeze_manifest": context["css_freeze_manifest"],
        "ehrsql_root": context["ehrsql_audit"].parent,
        "ehrsql_audit": context["ehrsql_audit"],
        "db_path": context["db_path"],
        "task1_manifest": context["task1_manifest"],
        "kg_path": context["kg_path"],
        "development_path": context["development_path"],
        "development_projection_receipt": context[
            "development_projection_receipt"
        ],
        "development_evaluation_receipt": context[
            "development_evaluation_receipt"
        ],
        "regression_receipt": context["regression_receipt"],
        "source_tree_manifest": context["source_tree_manifest"],
        "output_manifest": context["output_manifest"],
        "public_output": context["public_output"],
        "created_at": context["created_at"],
    }


def _patch_success(monkeypatch, context):
    monkeypatch.setattr(MODULE, "_preflight_freeze", lambda **_: context)

    def load_after_marker(current):
        assert current["started_marker"].is_file()
        return (
            [],
            {},
            {
                "CSS/test": {
                    "sha256": "1" * 64,
                    "size": 123,
                }
            },
        )

    monkeypatch.setattr(MODULE, "_load_official_test_rows", load_after_marker)
    items = _holdout_items()
    monkeypatch.setattr(
        MODULE,
        "_project_official_holdout",
        lambda *_: (
            items,
            {
                "source_attempts": {"CSS/test": 120},
                "rejection_counts": {},
                "n_before_global_deduplication": 120,
                "n_after_global_deduplication": 120,
                "deduplication": {
                    "exact_duplicate_rows_removed": 0,
                    "ambiguous_questions_removed": 0,
                    "ambiguous_rows_removed": 0,
                },
            },
        ),
    )


def test_preflight_failure_does_not_create_irreversible_marker(
    tmp_path,
    monkeypatch,
):
    context = _synthetic_context(tmp_path)
    monkeypatch.setattr(
        MODULE,
        "_preflight_freeze",
        lambda **_: (_ for _ in ()).throw(ValueError("preflight failed")),
    )
    with pytest.raises(ValueError, match="preflight failed"):
        MODULE.freeze_official_v2(**_freeze_kwargs(context))
    assert not context["started_marker"].exists()
    assert not context["invalid_marker"].exists()


def test_official_content_reader_requires_marker(tmp_path):
    source = _touch(tmp_path / "official-test.json", b"[]")
    binding = {
        "sha256": MODULE._sha256(source),
        "size": source.stat().st_size,
    }
    marker = tmp_path / "projection-started.json"
    with pytest.raises(RuntimeError, match="before projection marker"):
        MODULE._read_official_json_after_marker(
            source,
            binding,
            marker,
            "synthetic official test",
        )
    _touch(marker, b"started")
    assert (
        MODULE._read_official_json_after_marker(
            source,
            binding,
            marker,
            "synthetic official test",
        )
        == []
    )


def test_success_is_atomic_create_once_and_public_receipt_is_aggregate_only(
    tmp_path,
    monkeypatch,
):
    context = _synthetic_context(tmp_path)
    _patch_success(monkeypatch, context)
    preflight = MODULE.preflight_official_v2(**_freeze_kwargs(context))
    assert preflight["status"] == "preflight-passed"
    assert preflight["official_test_content_opened"] is False
    assert preflight["projection_marker_created"] is False
    assert not context["started_marker"].exists()
    receipt = MODULE.freeze_official_v2(**_freeze_kwargs(context))
    manifest = json.loads(
        context["output_manifest"].read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 4
    assert manifest["created_once"] is True
    assert manifest["selection_used"] is False
    assert manifest["n_holdout"] == 120
    assert manifest["n_hard"] == 40
    assert (
        manifest["partition"][
            "development_holdout_skeleton_intersection"
        ]
        == 0
    )
    assert receipt["counts"]["n_holdout"] == 120
    assert context["completed_marker"].is_file()
    public_text = context["public_output"].read_text(encoding="utf-8")
    assert "synthetic private question" not in public_text
    assert '"gold_sql"' not in public_text
    with pytest.raises(FileExistsError):
        MODULE.freeze_official_v2(**_freeze_kwargs(context))
    assert not context["invalid_marker"].exists()


def test_post_marker_crash_invalidates_rid_and_cannot_be_retried(
    tmp_path,
    monkeypatch,
):
    context = _synthetic_context(tmp_path)
    monkeypatch.setattr(MODULE, "_preflight_freeze", lambda **_: context)
    monkeypatch.setattr(
        MODULE,
        "_after_started_marker",
        lambda _: (_ for _ in ()).throw(RuntimeError("injected crash")),
    )
    with pytest.raises(RuntimeError, match="injected crash"):
        MODULE.freeze_official_v2(**_freeze_kwargs(context))
    invalid = json.loads(
        context["invalid_marker"].read_text(encoding="utf-8")
    )
    assert invalid["status"] == "projection-invalid"
    assert invalid["failure_class"] == "RuntimeError"
    assert "injected crash" not in context["invalid_marker"].read_text(
        encoding="utf-8"
    )
    with pytest.raises(FileExistsError):
        MODULE.freeze_official_v2(**_freeze_kwargs(context))


def test_two_processes_cannot_publish_the_same_marker(tmp_path):
    marker = tmp_path / "projection-started.json"
    process_context = multiprocessing.get_context("spawn")
    start = process_context.Event()
    queue = process_context.Queue()
    processes = [
        process_context.Process(
            target=_race_publish,
            args=(str(marker), start, queue),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    results = [queue.get(timeout=_SPAWN_REPORT_TIMEOUT_S) for _ in processes]
    for process in processes:
        process.join(timeout=_SPAWN_REPORT_TIMEOUT_S)
        assert process.exitcode == 0
    assert sorted(results) == ["occupied", "published"]
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "status": "winner"
    }
