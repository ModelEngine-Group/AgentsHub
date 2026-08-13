import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from dki.nlp import cmeie_joint_evaluation as joint
from dki.nlp import cmeie_run_protocol as run_protocol


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _make_run(parent: Path) -> tuple[Path, run_protocol.RunBinding]:
    source = b"source\n"
    entries = [
        {
            "path": "dki/example.py",
            "size": len(source),
            "sha256": _sha(source),
        }
    ]
    tree = run_protocol.included_tree_sha256(entries)
    head = "30fce24" + "a" * 33
    rid = f"20260729T120000Z-g{head[:7]}-t{tree[:12]}"
    root = parent / f"dki-agents-{rid}"
    source_root = root / "source" / "dki-agents"
    (source_root / "dki").mkdir(parents=True)
    (source_root / "dki" / "example.py").write_bytes(source)
    manifest_dir = root / "evidence" / "source"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "included_tree.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rid": rid,
                "git_head": head,
                "algorithm": run_protocol.SOURCE_MANIFEST_ALGORITHM,
                "included_tree_sha256": tree,
                "entries": entries,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "evidence" / "cmeie").mkdir()
    return root, run_protocol.load_run_binding(root)


@dataclass
class _Predictor:
    seed: int
    calls: list[tuple[int, str]]

    def predict(self, row):
        self.calls.append((self.seed, row["text"]))
        return set(row.get(f"prediction_{self.seed}", ()))


def test_three_seeds_evaluate_in_one_row_major_pass():
    calls: list[tuple[int, str]] = []
    rows = [
        {
            "text": "a",
            "gold": {("a", "p", "b")},
            "prediction_42": {("a", "p", "b")},
            "prediction_43": set(),
            "prediction_44": {("x", "p", "y")},
        },
        {
            "text": "b",
            "gold": {("c", "p", "d")},
            "prediction_42": set(),
            "prediction_43": {("c", "p", "d")},
            "prediction_44": {("c", "p", "d")},
        },
    ]
    metrics = joint.evaluate_rows_once(
        rows,
        [_Predictor(seed, calls) for seed in joint.ALL_SEEDS],
        gold_fn=lambda row: row["gold"],
    )

    assert calls == [
        (42, "a"),
        (43, "a"),
        (44, "a"),
        (42, "b"),
        (43, "b"),
        (44, "b"),
    ]
    assert metrics[42]["counts"] == {"tp": 1, "fp": 0, "fn": 1}
    assert metrics[43]["counts"] == {"tp": 1, "fp": 0, "fn": 1}
    assert metrics[44]["counts"] == {"tp": 1, "fp": 1, "fn": 1}


def test_outer_integer_boundaries_are_exact():
    passing = {
        42: joint.metric_from_counts(48, 52, 52),
        43: joint.metric_from_counts(50, 50, 50),
        44: joint.metric_from_counts(60, 40, 40),
    }
    assert joint.outer_metrics_pass(passing)
    assert joint.exact_outer_gate_observation(passing) == {
        "median_f1": {"numerator": 1, "denominator": 2},
        "worst_f1": {"numerator": 12, "denominator": 25},
        "passed": True,
    }

    below_worst = dict(passing)
    below_worst[42] = joint.metric_from_counts(47, 53, 53)
    assert not joint.outer_metrics_pass(below_worst)

    exact_precision = dict(passing)
    exact_precision[44] = joint.metric_from_counts(90, 110, 60)
    assert joint.outer_metrics_pass(exact_precision)
    below_precision = dict(passing)
    below_precision[44] = joint.metric_from_counts(449, 551, 251)
    assert not joint.outer_metrics_pass(below_precision)


def test_official_f1_exact_half_boundary():
    assert joint.official_metric_pass(joint.metric_from_counts(1, 1, 1))
    assert not joint.official_metric_pass(
        joint.metric_from_counts(49, 51, 51)
    )


def test_metric_rejects_display_ratio_inconsistent_with_counts():
    metric = joint.metric_from_counts(1, 1, 1)
    metric["f1"] = {"numerator": 5001, "denominator": 10000}
    with pytest.raises(joint.JointEvaluationError, match="integer counts"):
        joint.validate_metric(metric, label="metric")


def _selection(seed: int, *, tree: str = "a" * 64) -> dict:
    common = {
        field: tree
        for field in (
            "source_manifest_sha256",
            "source_tree_sha256",
            "source_tree_manifest_sha256",
            "train_sha256",
            "source_train_sha256",
            "fit_shard_sha256",
            "calibration_shard_sha256",
            "split_manifest_sha256",
            "base_model_tree_sha256",
            "tokenizer_vocabulary_sha256",
            "label_table_sha256",
            "schema_constraints_sha256",
            "training_script_sha256",
            "evaluator_sha256",
            "model_module_sha256",
            "alignment_module_sha256",
            "globalpointer_module_sha256",
            "artifact_integrity_module_sha256",
            "extraction_contract_module_sha256",
            "selector_sha256",
            "selection_protocol_sha256",
            "run_protocol_sha256",
        )
    }
    return {
        "schema_version": 3,
        "kind": "cmeie_gplinker_checkpoint_selection",
        "dataset_role": "calibration_selection",
        "selection_used": True,
        "outer_accessed": False,
        "official_dev_accessed": False,
        "seed": seed,
        "thresholds": [-0.5, 0.0, 0.5],
        "schema_constraint": "on",
        "grid_size": 54,
        "rid": "rid",
        "head": "b" * 40,
        "tree": tree,
        "run_root": "/run",
        "source_manifest_path": "/run/manifest.json",
        "source_manifest_size": 1,
        "source_root": "/run/source",
        "base_model": "/model",
        "max_length": 320,
        "parameter_count": 1,
        **common,
    }


def test_selection_set_requires_exact_seeds_and_common_bindings():
    selections, common = joint.validate_selection_set(
        [_selection(seed) for seed in joint.ALL_SEEDS]
    )
    assert tuple(sorted(selections)) == joint.ALL_SEEDS
    assert common["max_length"] == 320

    changed = [_selection(seed) for seed in joint.ALL_SEEDS]
    changed[-1]["split_manifest_sha256"] = "c" * 64
    with pytest.raises(joint.JointEvaluationError, match="common binding"):
        joint.validate_selection_set(changed)

    with pytest.raises(joint.JointEvaluationError, match="exactly seeds"):
        joint.validate_selection_set([_selection(42), _selection(43)])


def test_completion_publish_is_create_once_and_canonical(tmp_path: Path):
    path = tmp_path / "receipt.json"
    binding = joint.publish_json_once(
        path,
        {"schema_version": 1, "value": 2},
        label="test receipt",
    )
    assert binding["sha256"] == _sha(
        b'{"schema_version":1,"value":2}\n'
    )
    with pytest.raises(FileExistsError, match="already exists"):
        joint.publish_json_once(
            path,
            {"schema_version": 1, "value": 3},
            label="test receipt",
        )


def test_marker_validator_rejects_fabricated_marker(tmp_path: Path):
    _, binding = _make_run(tmp_path)
    paths = run_protocol.canonical_run_paths(binding)
    paths.outer_started.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "cmeie_dataset_access_started",
                "started": True,
                "completion_recorded": False,
                "rid": binding.rid,
                "head": binding.git_head,
                "tree": binding.included_tree_sha256,
                "run_root": str(binding.run_root),
                "source_manifest_path": str(binding.source_manifest),
                "source_manifest_sha256": binding.source_manifest_sha256,
                "dataset_role": "outer",
                "dataset_sha256": "a" * 64,
                "split_sha256": "b" * 64,
                "dataset_key_sha256": "0" * 64,
                "input_model_evidence": {},
                "input_model_evidence_sha256": _sha(b"{}\n"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(joint.JointEvaluationError, match="binding is invalid"):
        joint.load_and_validate_started_marker(
            binding,
            role="outer",
            dataset_sha256="a" * 64,
            split_sha256="b" * 64,
        )


def test_completion_loader_rejects_fabricated_canonical_receipt(
    tmp_path: Path,
):
    _, binding = _make_run(tmp_path)
    path = run_protocol.canonical_run_paths(binding).outer_completion
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "fabricated",
                "dataset_role": "outer",
                "rid": binding.rid,
                "head": binding.git_head,
                "tree": binding.included_tree_sha256,
                "run_root": str(binding.run_root),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(joint.JointEvaluationError, match="binding is invalid"):
        joint.load_canonical_completion(
            binding,
            role="outer",
            expected_kind=joint.OUTER_BATCH_KIND,
            expected_schema_version=joint.OUTER_BATCH_SCHEMA_VERSION,
        )


def test_model_loader_keeps_secure_load_and_reverification_guards():
    source = inspect.getsource(joint.load_predictor)
    assert "weights_only=True" in source
    assert "reject_multiple_links=True" in source
    assert source.count("_verify_bundle(selection)") >= 2
    assert source.count("load_and_validate_extraction_contract(") >= 3
