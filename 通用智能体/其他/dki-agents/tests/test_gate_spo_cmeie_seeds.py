import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import pytest

from dki.nlp import cmeie_joint_evaluation as joint
from dki.nlp import cmeie_run_protocol as run_protocol


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gate_spo_cmeie_seeds.py"
SPEC = importlib.util.spec_from_file_location(
    "gate_spo_cmeie_seeds",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


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
    rid = f"20260729T140000Z-g{head[:7]}-t{tree[:12]}"
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


def _calibration(tp: int, fp: int, fn: int):
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def _selection(seed: int, calibration) -> dict:
    digest = f"{seed:02d}" * 32
    return {
        "seed": seed,
        "calibration_metric": calibration,
        "training_receipt_sha256": digest,
        "checkpoint_sha256": digest,
        "checkpoint_bundle_manifest_sha256": digest,
        "checkpoint_bundle_tree_sha256": digest,
        "extraction_contract_sha256": digest,
        "semantic_contract_sha256": digest,
    }


def test_calibration_choice_never_uses_outer_ranking():
    calibration = {
        42: MODULE._calibration_metric(
            _calibration(80, 20, 20),
            label="42",
        ),
        43: MODULE._calibration_metric(
            _calibration(70, 30, 30),
            label="43",
        ),
        44: MODULE._calibration_metric(
            _calibration(60, 40, 40),
            label="44",
        ),
    }
    assert MODULE._calibration_choice(calibration) == 42
    outer = {
        42: joint.metric_from_counts(48, 52, 52),
        43: joint.metric_from_counts(50, 50, 50),
        44: joint.metric_from_counts(99, 1, 1),
    }
    assert joint.outer_metrics_pass(outer)
    assert MODULE._calibration_choice(calibration) == 42


def test_outer_batch_metric_parser_uses_exact_integer_ratios():
    batch = {
        "models": [
            {"seed": 42, "metric": joint.metric_from_counts(48, 52, 52)},
            {"seed": 43, "metric": joint.metric_from_counts(50, 50, 50)},
            {"seed": 44, "metric": joint.metric_from_counts(60, 40, 40)},
        ]
    }
    metrics = MODULE._batch_outer_metrics(batch)
    assert joint.outer_metrics_pass(metrics)

    batch["models"][0]["metric"]["f1"] = {
        "numerator": 49,
        "denominator": 100,
    }
    with pytest.raises(MODULE.GateInputError, match="integer counts"):
        MODULE._batch_outer_metrics(batch)


def test_outer_gate_exact_thresholds_include_boundaries():
    passing = {
        42: joint.metric_from_counts(48, 52, 52),
        43: joint.metric_from_counts(50, 50, 50),
        44: joint.metric_from_counts(90, 110, 60),
    }
    assert joint.outer_metrics_pass(passing)
    assert joint.metric_ratio(passing[42], "f1") == Fraction(12, 25)
    assert joint.metric_ratio(passing[43], "f1") == Fraction(1, 2)
    assert joint.metric_ratio(passing[44], "precision") == Fraction(9, 20)


def test_fabricated_batch_receipt_is_rejected_before_gate(
    tmp_path: Path,
    monkeypatch,
):
    _, binding = _make_run(tmp_path)
    selections = {
        seed: _selection(seed, _calibration(60, 40, 40))
        for seed in joint.ALL_SEEDS
    }
    selection_paths = {
        seed: binding.run_root / f"selection-{seed}.json"
        for seed in joint.ALL_SEEDS
    }
    selection_bindings = {
        seed: {"sha256": f"{seed:02d}" * 32, "size": 1}
        for seed in joint.ALL_SEEDS
    }
    monkeypatch.setattr(
        MODULE,
        "load_canonical_completion",
        lambda *_args, **_kwargs: (
            {
                "schema_version": 1,
                "kind": joint.OUTER_BATCH_KIND,
                "dataset_role": "outer",
                "outer_accessed": True,
                "official_dev_accessed": False,
                "selection_used": False,
                "single_dataset_read": True,
                "same_row_loop_for_all_seeds": True,
                "completed": True,
                "common_bindings": {"wrong": True},
            },
            {"sha256": "a" * 64, "size": 1},
        ),
    )
    with pytest.raises(MODULE.GateInputError, match="common or selection"):
        MODULE._validate_outer_batch(
            binding,
            selections=selections,
            selection_paths=selection_paths,
            selection_bindings=selection_bindings,
            common={"split_manifest_sha256": "a" * 64},
        )


def test_outer_receipt_binds_preflight_dev_hash_and_batch_sha(
    tmp_path: Path,
    monkeypatch,
):
    _, binding = _make_run(tmp_path)
    digest = "a" * 64
    selections = {
        seed: _selection(seed, _calibration(60 + seed - 42, 40, 40))
        for seed in joint.ALL_SEEDS
    }
    paths = {
        seed: binding.run_root / f"selection-{seed}.json"
        for seed in joint.ALL_SEEDS
    }
    bindings = {
        seed: {"sha256": f"{seed:02d}" * 32, "size": 1}
        for seed in joint.ALL_SEEDS
    }
    calibration = {
        seed: MODULE._calibration_metric(
            selections[seed]["calibration_metric"],
            label=str(seed),
        )
        for seed in joint.ALL_SEEDS
    }
    common = {
        "train_sha256": digest,
        "base_model": "/model",
        "base_model_tree_sha256": digest,
        "split_manifest_sha256": digest,
    }
    monkeypatch.setattr(
        MODULE,
        "_selection_inputs",
        lambda *_args, **_kwargs: (
            selections,
            paths,
            bindings,
            calibration,
            common,
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "_validate_early_receipt",
        lambda *_args, **_kwargs: (
            {"schema_version": 2},
            {"sha256": "b" * 64},
        ),
    )
    preflight = {
        "preflight_receipt_path": str(
            run_protocol.canonical_run_paths(binding).preflight_receipt
        ),
        "preflight_receipt_sha256": "c" * 64,
        "cmeie_dev_sha256": "d" * 64,
        "cmeie_dev_rows": 3585,
    }
    monkeypatch.setattr(
        MODULE,
        "_preflight_binding",
        lambda *_args, **_kwargs: preflight,
    )
    outer_metrics = {
        42: joint.metric_from_counts(48, 52, 52),
        43: joint.metric_from_counts(50, 50, 50),
        44: joint.metric_from_counts(60, 40, 40),
    }
    monkeypatch.setattr(
        MODULE,
        "_validate_outer_batch",
        lambda *_args, **_kwargs: (
            {"schema_version": 1},
            {"sha256": "e" * 64},
            outer_metrics,
            {"dataset_key_sha256": "f" * 64},
            {"sha256": "1" * 64},
        ),
    )

    payload, passed = MODULE._build_outer_receipt(
        run_binding=binding,
        selection_paths=list(paths.values()),
    )
    assert passed
    assert payload["outer_batch_receipt"]["sha256"] == "e" * 64
    assert payload["preflight_receipt_sha256"] == "c" * 64
    assert payload["expected_official_dev_sha256"] == "d" * 64
    assert payload["expected_official_dev_rows"] == 3585
    assert payload["calibration_choice"]["seed"] == 44


def test_preflight_rejection_prevents_outer_gate_build(
    tmp_path: Path,
    monkeypatch,
):
    _, binding = _make_run(tmp_path)
    monkeypatch.setattr(
        MODULE,
        "_selection_inputs",
        lambda *_args, **_kwargs: ({}, {}, {}, {}, {}),
    )
    monkeypatch.setattr(
        MODULE,
        "_validate_early_receipt",
        lambda *_args, **_kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        MODULE,
        "_preflight_binding",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MODULE.GateInputError("preflight not ready")
        ),
    )
    with pytest.raises(MODULE.GateInputError, match="preflight not ready"):
        MODULE._build_outer_receipt(
            run_binding=binding,
            selection_paths=[],
        )


def test_outer_cli_accepts_one_batch_receipt_not_three_evaluations():
    parser = MODULE._parser()
    outer = next(
        action
        for action in parser._actions
        if action.dest == "command"
    ).choices["outer"]
    options = {
        option
        for action in outer._actions
        for option in action.option_strings
    }
    assert "--outer-batch-receipt" in options
    assert "--preflight-receipt" in options
    assert "--outer-eval" not in options
    assert "--run-root" in options
