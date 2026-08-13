from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.eval_nl2sql_consumed_development as reaudit


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    }


def _consumed_freeze(tmp_path: Path) -> dict[str, Path]:
    rid = "freeze-v28"
    frozen = (
        tmp_path
        / "outputs"
        / "private"
        / rid
        / "nl2sql_blind_v3"
        / "frozen"
    )
    development = _write(frozen / "development.jsonl", b"{}\n")
    holdout = _write(frozen / "holdout.jsonl", b"holdout-bytes-must-not-be-read")
    task1 = _write(tmp_path / "lineage" / "task1.json", b"task1")
    kg = _write(tmp_path / "lineage" / "kg.json", b"kg")
    db = _write(tmp_path / "lineage" / "task3.sqlite", b"sqlite")
    source_manifest = _write(
        tmp_path / "lineage" / "included_tree_manifest.json",
        b"{}\n",
    )
    manifest = frozen / "freeze_manifest.json"
    manifest_payload = {
        "schema_version": 4,
        "kind": reaudit.FREEZE_KIND,
        "created_once": True,
        "selection_used": False,
        "rid": rid,
        "n_development": 180,
        "n_holdout": 120,
        "n_hard": 90,
        "partition": {
            "version": "v4",
            "normalised_question_intersection": 0,
            "near_duplicate_question_intersection": 0,
            "sql_ast_skeleton_intersection": 0,
        },
        "privacy": {
            "manifest_contains_questions": False,
            "manifest_contains_gold_sql": False,
            "manifest_contains_predictions": False,
            "manifest_contains_error_samples": False,
        },
        "task1_manifest": _binding(task1),
        "kg": _binding(kg),
        "database": _binding(db),
        "development": {
            "path": development.name,
            "sha256": hashlib.sha256(development.read_bytes()).hexdigest(),
            "size": development.stat().st_size,
        },
        "holdout": {
            "path": holdout.name,
            "sha256": "a" * 64,
            "size": holdout.stat().st_size,
        },
    }
    _write(
        manifest,
        (json.dumps(manifest_payload, sort_keys=True) + "\n").encode("utf-8"),
    )
    _write(frozen / reaudit.STARTED_MARKER_NAME, b"{}\n")
    consumption = {
        "schema_version": 1,
        "status": "consumed",
        "freeze_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "candidate_seal_sha256": "b" * 64,
        "gate_passed": False,
    }
    _write(
        frozen / reaudit.RECEIPT_NAME,
        (json.dumps(consumption, sort_keys=True) + "\n").encode("utf-8"),
    )
    return {
        "manifest": manifest,
        "development": development,
        "holdout": holdout,
        "task1": task1,
        "kg": kg,
        "db": db,
        "source_manifest": source_manifest,
    }


def test_post_consumption_contract_never_reads_or_hashes_holdout(
    monkeypatch,
    tmp_path,
):
    paths = _consumed_freeze(tmp_path)
    holdout = paths["holdout"].resolve()
    original_open = Path.open
    original_sha256 = reaudit._sha256

    def guarded_open(self, *args, **kwargs):
        if self.resolve() == holdout:
            raise AssertionError("post-consumption re-audit opened holdout bytes")
        return original_open(self, *args, **kwargs)

    def guarded_sha256(path):
        if Path(path).resolve() == holdout:
            raise AssertionError("post-consumption re-audit hashed holdout bytes")
        return original_sha256(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(reaudit, "_sha256", guarded_sha256)

    contract = reaudit._load_post_consumption_contract(
        manifest=paths["manifest"],
        task1_manifest=paths["task1"],
        kg_path=paths["kg"],
        db_path=paths["db"],
        source_tree_manifest=paths["source_manifest"],
        root=tmp_path / "different-current-candidate-source",
    )

    assert contract["development_path"] == paths["development"]
    assert contract["holdout_stat_checked"] is True
    assert contract["holdout_opened"] is False
    assert contract["holdout_hashed"] is False


def test_post_consumption_contract_rejects_invalidated_freeze(tmp_path):
    paths = _consumed_freeze(tmp_path)
    invalid = paths["manifest"].parent / reaudit.INVALID_MARKER_NAME
    invalid.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="consumed-invalid"):
        reaudit._load_post_consumption_contract(
            manifest=paths["manifest"],
            task1_manifest=paths["task1"],
            kg_path=paths["kg"],
            db_path=paths["db"],
            source_tree_manifest=paths["source_manifest"],
            root=tmp_path,
        )


def test_holdout_alias_is_rejected_before_any_byte_read(monkeypatch, tmp_path):
    paths = _consumed_freeze(tmp_path)
    holdout = paths["holdout"].resolve()
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        if self.resolve() == holdout:
            raise AssertionError("holdout bytes were opened before alias rejection")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="aliases the frozen holdout"):
        reaudit._load_post_consumption_contract(
            manifest=paths["manifest"],
            task1_manifest=paths["task1"],
            kg_path=paths["kg"],
            db_path=paths["db"],
            source_tree_manifest=paths["holdout"],
            root=tmp_path,
        )


def test_source_tree_private_entry_is_rejected_before_verifier_hashes_holdout(
    monkeypatch,
    tmp_path,
):
    paths = _consumed_freeze(tmp_path)
    holdout = paths["holdout"].resolve()
    source_manifest = paths["source_manifest"]
    source_manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {"path": "outputs/private/freeze/holdout.jsonl"}
                ]
            }
        ),
        encoding="utf-8",
    )
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        if self.resolve() == holdout:
            raise AssertionError("source preflight opened holdout bytes")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="must not contain outputs/private"):
        reaudit._preflight_source_tree_without_private(
            source_manifest,
            holdout_path=paths["holdout"],
            root=tmp_path,
        )


def test_public_reaudit_receipt_is_aggregate_only_and_not_blind_eligible():
    sentinel_question = "SENTINEL_PRIVATE_QUESTION"
    sentinel_gold = "SELECT secret_gold FROM forbidden"
    sentinel_prediction = "SELECT secret_prediction FROM forbidden"
    result = {
        "engine": "agent",
        "n": 180,
        "n_correct": 166,
        "accuracy": 0.9222,
        "n_answered": 180,
        "n_abstained": 0,
        "coverage": 1.0,
        "execution_accuracy_answered": 0.9222,
        "strict_result_contract": True,
        "result_order_policy": "question_explicit_final_order_v1",
        "by_difficulty": {
            "medium": {"n": 52, "correct": 48, "accuracy": 0.9231},
            "hard": {"n": 80, "correct": 74, "accuracy": 0.925},
            "extra": {"n": 30, "correct": 27, "accuracy": 0.9},
            "extra-hard": {"n": 18, "correct": 17, "accuracy": 0.9444},
        },
        "details": [
            {
                "question": sentinel_question,
                "gold_sql": sentinel_gold,
                "pred_sql": sentinel_prediction,
                "correct": False,
            }
        ],
    }
    common = {
        "schema_version": 1,
        "rid": "candidate-rid",
        "dataset_role": reaudit.DATASET_ROLE,
        "selection_used": True,
        "blind_claim": False,
        "candidate_seal_eligible": False,
    }

    private, public = reaudit._build_receipts(common=common, result=result)
    public_text = json.dumps(public, ensure_ascii=False, sort_keys=True)

    assert private["details"] == result["details"]
    assert "details" not in public
    assert public["dataset_role"] == reaudit.DATASET_ROLE
    assert public["blind_claim"] is False
    assert public["candidate_seal_eligible"] is False
    assert public["development_gate"]["passed"] is True
    assert public["contains_questions"] is False
    assert public["contains_gold_sql"] is False
    assert public["contains_predictions"] is False
    assert sentinel_question not in public_text
    assert sentinel_gold not in public_text
    assert sentinel_prediction not in public_text


def test_reaudit_pair_is_create_once_and_hash_binds_private_details(tmp_path):
    private_path = tmp_path / "private" / "result.json"
    public_path = tmp_path / "public" / "attestation.json"
    private = {"rid": "candidate", "details": [{"private": "row"}]}
    public = {
        "rid": "candidate",
        "blind_claim": False,
        "candidate_seal_eligible": False,
    }

    reaudit._publish_pair_once(
        private_path=private_path,
        private_payload=private,
        public_path=public_path,
        public_payload=public,
    )
    public_document = json.loads(public_path.read_text(encoding="utf-8"))

    assert public_document["pairing"]["private_sha256"] == hashlib.sha256(
        private_path.read_bytes()
    ).hexdigest()
    with pytest.raises(FileExistsError, match="already exists"):
        reaudit._publish_pair_once(
            private_path=private_path,
            private_payload=private,
            public_path=public_path,
            public_payload=public,
        )


def test_reaudit_pair_rolls_back_its_private_link_when_public_publish_fails(
    monkeypatch,
    tmp_path,
):
    private_path = tmp_path / "private" / "result.json"
    public_path = tmp_path / "public" / "attestation.json"
    original_link = reaudit.os.link
    calls = 0

    def fail_second_link(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated public create-once failure")
        return original_link(source, destination)

    monkeypatch.setattr(reaudit.os, "link", fail_second_link)
    with pytest.raises(OSError, match="simulated public"):
        reaudit._publish_pair_once(
            private_path=private_path,
            private_payload={"rid": "candidate", "details": []},
            public_path=public_path,
            public_payload={"rid": "candidate", "blind_claim": False},
        )

    assert not private_path.exists()
    assert not public_path.exists()


def test_current_source_bindings_require_post_consumption_runner(tmp_path):
    required = reaudit._required_source_paths("agent", "query_plan_v2")
    entries = []
    for index, relative in enumerate(required):
        path = _write(tmp_path / relative, f"source-{index}".encode("utf-8"))
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    source_tree = {"entries": entries}

    bindings = reaudit._current_source_bindings(
        source_tree,
        root=tmp_path,
        engine="agent",
        llm_strategy="query_plan_v2",
    )
    assert set(bindings) == set(required)

    source_tree["entries"] = [
        entry
        for entry in entries
        if entry["path"] != "scripts/eval_nl2sql_consumed_development.py"
    ]
    with pytest.raises(ValueError, match="does not bind current source"):
        reaudit._current_source_bindings(
            source_tree,
            root=tmp_path,
            engine="agent",
            llm_strategy="query_plan_v2",
        )
