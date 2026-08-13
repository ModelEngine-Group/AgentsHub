import copy
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from dki.evidence_io import load_bound_json
from dki.nlp import cmeie_joint_evaluation as joint
from dki.nlp import cmeie_run_protocol as run_protocol


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_spo_cmeie_batch.py"
SPEC = importlib.util.spec_from_file_location("eval_spo_cmeie_batch", SCRIPT)
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
    rid = f"20260729T130000Z-g{head[:7]}-t{tree[:12]}"
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
    (root / "evidence" / "cmeie" / "split").mkdir(parents=True)
    return root, run_protocol.load_run_binding(root)


def _selection(seed: int, binding, checkpoint: Path) -> dict:
    digest = "a" * 64
    return {
        "seed": seed,
        "selected_epoch": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_bundle_manifest_sha256": digest,
        "checkpoint_bundle_tree_sha256": digest,
        "extraction_contract_sha256": digest,
        "semantic_contract_sha256": digest,
        "training_receipt_sha256": digest,
        "thresholds": [-0.5, 0.0, 0.5],
        "schema_constraint": "on",
        "bundle_binding": {"bundle": seed},
        "contract_binding": {"contract": seed},
        "split_manifest_sha256": "",
        "rid": binding.rid,
        "head": binding.git_head,
        "tree": binding.included_tree_sha256,
    }


@dataclass
class _Predictor:
    seed: int
    calls: list[tuple[int, str]]
    bundle_binding: dict
    contract_binding: dict
    checkpoint_binding: dict

    def predict(self, row):
        self.calls.append((self.seed, row["text"]))
        return set(row["gold"])


def _outer_fixture(tmp_path: Path, monkeypatch):
    root, binding = _make_run(tmp_path)
    paths = run_protocol.canonical_run_paths(binding)
    shard = paths.split_dir / "outer.jsonl"
    shard.write_text('{"text":"x","spo_list":[]}\n', encoding="utf-8")
    shard_sha = _sha(shard.read_bytes())
    manifest = paths.split_dir / "split_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "splits": {
                    "outer": {"shard": {"sha256": shard_sha}}
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_payload, manifest_binding = load_bound_json(
        manifest,
        "test manifest",
        require_object=True,
    )
    selections = {}
    selection_paths = {}
    selection_bindings = {}
    for seed in MODULE.ALL_SEEDS:
        candidate = paths.seed_model(seed)
        candidate.mkdir(parents=True)
        selection_path = candidate / "selection.json"
        selection_path.write_text(
            json.dumps({"seed": seed}) + "\n",
            encoding="utf-8",
        )
        checkpoint = candidate / "epoch-01" / "gplinker.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(str(seed).encode())
        selections[seed] = _selection(seed, binding, checkpoint)
        selections[seed]["split_manifest_sha256"] = manifest_binding[
            "sha256"
        ]
        selection_paths[seed] = selection_path
        _, selection_bindings[seed] = load_bound_json(
            selection_path,
            f"selection {seed}",
            require_object=True,
        )
    common = {"split_manifest_sha256": manifest_binding["sha256"]}
    monkeypatch.setattr(
        MODULE,
        "_selection_inputs",
        lambda *_args, **_kwargs: (
            selections,
            selection_paths,
            selection_bindings,
            common,
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "_expected_outer_manifest",
        lambda *_args, **_kwargs: (
            manifest_payload,
            manifest_binding,
            {},
        ),
    )
    monkeypatch.setattr(MODULE, "_source_bindings", lambda _binding: {})
    return (
        root,
        binding,
        paths,
        manifest,
        shard,
        shard_sha,
        manifest_binding,
        selections,
    )


def _calibration(tp: int, fp: int, fn: int) -> dict:
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


def _official_selection(
    seed: int,
    binding: run_protocol.RunBinding,
    calibration: dict,
) -> dict:
    digest = hashlib.sha256(f"seed-{seed}".encode()).hexdigest()
    candidate = run_protocol.canonical_run_paths(binding).seed_model(seed)
    checkpoint = candidate / "epoch-01" / "gplinker.pt"
    return {
        "seed": seed,
        "selected_epoch": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_bundle_manifest_sha256": digest,
        "checkpoint_bundle_tree_sha256": digest,
        "extraction_contract_sha256": digest,
        "semantic_contract_sha256": digest,
        "training_receipt_sha256": digest,
        "thresholds": [-0.5, 0.0, 0.5],
        "schema_constraint": "on",
        "bundle_binding": {"seed": seed, "kind": "bundle"},
        "contract_binding": {"seed": seed, "kind": "contract"},
        "calibration_metric": calibration,
        "split_manifest_sha256": "a" * 64,
        "rid": binding.rid,
        "head": binding.git_head,
        "tree": binding.included_tree_sha256,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _official_gate_fixture(
    tmp_path: Path,
    monkeypatch,
) -> dict:
    root, binding = _make_run(tmp_path)
    paths = run_protocol.canonical_run_paths(binding)
    selections = {
        42: _official_selection(
            42,
            binding,
            _calibration(80, 20, 20),
        ),
        43: _official_selection(
            43,
            binding,
            _calibration(70, 30, 30),
        ),
        44: _official_selection(
            44,
            binding,
            _calibration(60, 40, 40),
        ),
    }
    selection_paths = {}
    selection_bindings = {}
    for seed in MODULE.ALL_SEEDS:
        path = paths.seed_model(seed) / "selection.json"
        _write_json(path, selections[seed])
        selection_paths[seed] = path
        _, selection_bindings[seed] = load_bound_json(
            path,
            f"official selection {seed}",
            reject_multiple_links=True,
            require_object=True,
        )

    common = {
        "split_manifest_sha256": "a" * 64,
        "train_sha256": "b" * 64,
        "base_model": str(paths.base_model),
        "base_model_tree_sha256": "c" * 64,
    }
    preflight = {
        "preflight_receipt_path": str(paths.preflight_receipt),
        "preflight_receipt_sha256": "d" * 64,
        "cmeie_dev_sha256": "e" * 64,
        "cmeie_dev_rows": MODULE.OFFICIAL_DEV_EXPECTED_ROWS,
    }
    early = {
        "schema_version": 2,
        "kind": "cmeie_gplinker_seed_early_gate",
    }
    early_path = paths.evidence_dir / "early_gate.json"
    _write_json(early_path, early)
    _, early_binding = load_bound_json(
        early_path,
        "official early gate",
        reject_multiple_links=True,
        require_object=True,
    )
    batch = {"schema_version": MODULE.OUTER_BATCH_SCHEMA_VERSION}
    batch_binding = {"sha256": "f" * 64, "size": 1}
    marker = {"dataset_key_sha256": "1" * 64}
    marker_binding = {"sha256": "2" * 64, "size": 1}
    state = {
        "outer_metrics": {
            42: joint.metric_from_counts(48, 52, 52),
            43: joint.metric_from_counts(50, 50, 50),
            44: joint.metric_from_counts(60, 40, 40),
        }
    }

    calibration_metrics = MODULE._calibration_metrics(selections)
    chosen_seed = MODULE._calibration_choice(calibration_metrics)
    chosen_selection = selections[chosen_seed]
    choice = {
        "seed": chosen_seed,
        "tie_break": [
            "calibration_f1_desc",
            "calibration_precision_desc",
            "seed_asc",
        ],
        "selection_receipt_sha256": selection_bindings[chosen_seed][
            "sha256"
        ],
        "checkpoint_sha256": chosen_selection["checkpoint_sha256"],
        "checkpoint_bundle_manifest_sha256": chosen_selection[
            "checkpoint_bundle_manifest_sha256"
        ],
        "checkpoint_bundle_tree_sha256": chosen_selection[
            "checkpoint_bundle_tree_sha256"
        ],
    }
    observation = joint.exact_outer_gate_observation(
        state["outer_metrics"]
    )
    gate_script_sha = _sha(
        (ROOT / "scripts" / "gate_spo_cmeie_seeds.py").read_bytes()
    )
    gate = {
        "schema_version": MODULE.OUTER_GATE_SCHEMA_VERSION,
        "kind": MODULE.OUTER_GATE_KIND,
        "dataset_role": "outer_gate",
        "selection_used": False,
        "outer_accessed": True,
        "official_dev_accessed": False,
        "rid": binding.rid,
        "head": binding.git_head,
        "tree": binding.included_tree_sha256,
        "run_root": str(root),
        "expected_seeds": list(MODULE.ALL_SEEDS),
        "early_gate_receipt": {
            "path": str(early_path),
            "sha256": early_binding["sha256"],
            "schema_version": early["schema_version"],
        },
        "selection_receipts": joint.selection_evidence(
            selections,
            selection_paths,
            selection_bindings,
        ),
        "outer_batch_receipt": {
            "path": str(paths.outer_completion),
            "sha256": batch_binding["sha256"],
            "schema_version": batch["schema_version"],
        },
        "outer_access_marker": {
            "path": str(paths.outer_started),
            "sha256": marker_binding["sha256"],
            "dataset_key_sha256": marker["dataset_key_sha256"],
        },
        "preflight_receipt_path": preflight["preflight_receipt_path"],
        "preflight_receipt_sha256": preflight[
            "preflight_receipt_sha256"
        ],
        "preflight_binding": preflight,
        "expected_official_dev_sha256": preflight["cmeie_dev_sha256"],
        "expected_official_dev_rows": preflight["cmeie_dev_rows"],
        "common_bindings": common,
        "calibration_metrics": MODULE._calibration_metric_evidence(
            calibration_metrics
        ),
        "outer_metrics": [
            {"seed": seed, **state["outer_metrics"][seed]}
            for seed in MODULE.ALL_SEEDS
        ],
        "thresholds": {
            "median_f1_minimum": joint.ratio_payload(
                joint.OUTER_MEDIAN_F1_FLOOR
            ),
            "worst_f1_minimum": joint.ratio_payload(
                joint.OUTER_WORST_F1_FLOOR
            ),
            "per_seed_precision_minimum": joint.ratio_payload(
                joint.OUTER_PRECISION_RECALL_FLOOR
            ),
            "per_seed_recall_minimum": joint.ratio_payload(
                joint.OUTER_PRECISION_RECALL_FLOOR
            ),
        },
        "observed": {
            "median_f1": observation["median_f1"],
            "worst_f1": observation["worst_f1"],
        },
        "calibration_choice": choice,
        "passed": True,
        "decision": "official_dev_authorized",
        "official_dev_authorized": True,
        "authorized_candidate": copy.deepcopy(choice),
        "gate_script_sha256": gate_script_sha,
    }
    _write_json(paths.outer_gate, gate)

    def validated_batch(*_args, **_kwargs):
        return (
            batch,
            batch_binding,
            state["outer_metrics"],
            marker,
            marker_binding,
        )

    monkeypatch.setattr(
        MODULE,
        "_validate_outer_batch_for_official",
        validated_batch,
    )
    monkeypatch.setattr(
        MODULE,
        "source_member_binding",
        lambda _binding, relative: {
            "relative_path": relative,
            "sha256": gate_script_sha,
        },
    )
    return {
        "root": root,
        "binding": binding,
        "paths": paths,
        "selections": selections,
        "selection_paths": selection_paths,
        "selection_bindings": selection_bindings,
        "common": common,
        "preflight": preflight,
        "state": state,
        "gate": gate,
        "gate_script_sha": gate_script_sha,
    }


def _load_gate_from_fixture(fixture: dict):
    return MODULE._load_outer_gate(
        fixture["binding"],
        selections=fixture["selections"],
        selection_paths=fixture["selection_paths"],
        selection_bindings=fixture["selection_bindings"],
        common=fixture["common"],
        requested_seed=42,
        preflight=fixture["preflight"],
    )


def test_outer_opens_shard_once_and_runs_three_seeds_same_batch(
    tmp_path: Path,
    monkeypatch,
):
    (
        root,
        _,
        paths,
        manifest,
        shard,
        shard_sha,
        manifest_binding,
        _,
    ) = _outer_fixture(tmp_path, monkeypatch)
    calls: list[tuple[int, str]] = []
    reads = {"count": 0}

    def predictor_loader(selection, *, device_name):
        assert device_name == "cpu"
        seed = selection["seed"]
        return _Predictor(
            seed,
            calls,
            selection["bundle_binding"],
            selection["contract_binding"],
            {
                "canonical_path": selection["checkpoint"],
                "sha256": selection["checkpoint_sha256"],
            },
        )

    def role_loader(*_args, **_kwargs):
        reads["count"] += 1
        shard.read_bytes()
        return (
            {},
            [{"text": "row", "gold": {("s", "p", "o")}}],
            {
                "role": "outer",
                "outer_accessed": True,
                "shard_sha256": shard_sha,
                "split_manifest_sha256": manifest_binding["sha256"],
            },
        )

    payload, _ = MODULE.run_outer(
        run_root=root,
        selection_paths=[
            paths.seed_model(seed) / "selection.json"
            for seed in MODULE.ALL_SEEDS
        ],
        split_manifest_path=manifest,
        outer_shard_path=shard,
        device="cpu",
        predictor_loader=predictor_loader,
        role_loader=role_loader,
    )

    assert reads["count"] == 1
    assert calls == [(42, "row"), (43, "row"), (44, "row")]
    assert [item["seed"] for item in payload["models"]] == [42, 43, 44]
    assert payload["single_dataset_read"] is True
    assert paths.outer_started.is_file()
    assert paths.outer_completion.is_file()


def test_outer_crash_after_marker_blocks_retry_even_for_changed_candidate(
    tmp_path: Path,
    monkeypatch,
):
    root, _, paths, manifest, shard, *_ = _outer_fixture(
        tmp_path,
        monkeypatch,
    )
    role_calls = {"count": 0}

    def role_loader(*_args, **_kwargs):
        role_calls["count"] += 1
        raise RuntimeError("simulated post-marker crash")

    def predictor_loader(selection, *, device_name):
        return _Predictor(
            selection["seed"],
            [],
            selection["bundle_binding"],
            selection["contract_binding"],
            {},
        )

    kwargs = {
        "run_root": root,
        "selection_paths": [
            paths.seed_model(seed) / "selection.json"
            for seed in MODULE.ALL_SEEDS
        ],
        "split_manifest_path": manifest,
        "outer_shard_path": shard,
        "device": "cpu",
        "predictor_loader": predictor_loader,
        "role_loader": role_loader,
    }
    with pytest.raises(RuntimeError, match="post-marker crash"):
        MODULE.run_outer(**kwargs)
    assert paths.outer_started.is_file()
    assert not paths.outer_completion.exists()

    # A different candidate/model evidence cannot mint a new dataset key.
    with pytest.raises(FileExistsError, match="already exists"):
        MODULE.run_outer(**kwargs)
    assert role_calls["count"] == 1


def test_official_3585_rows_with_wrong_sha_is_rejected(tmp_path: Path):
    path = tmp_path / "dev.jsonl"
    row = '{"text":"x","spo_list":[]}'
    path.write_text(
        "\n".join([row] * MODULE.OFFICIAL_DEV_EXPECTED_ROWS) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        MODULE.JointEvaluationError,
        match="differs from preflight",
    ):
        MODULE._parse_official_dev_once(
            path,
            expected_sha256="0" * 64,
        )


def test_official_gate_is_recomputed_from_canonical_evidence(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _official_gate_fixture(tmp_path, monkeypatch)
    gate, binding = _load_gate_from_fixture(fixture)
    assert gate["calibration_choice"]["seed"] == 42
    assert binding["sha256"] == _sha(
        fixture["paths"].outer_gate.read_bytes()
    )


def test_forged_passing_gate_metrics_are_rejected(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _official_gate_fixture(tmp_path, monkeypatch)
    forged = copy.deepcopy(fixture["gate"])
    forged["outer_metrics"][0] = {
        "seed": 42,
        **joint.metric_from_counts(99, 1, 1),
    }
    _write_json(fixture["paths"].outer_gate, forged)

    with pytest.raises(
        MODULE.JointEvaluationError,
        match="outer_metrics",
    ):
        _load_gate_from_fixture(fixture)


def test_gate_cannot_switch_the_calibration_chosen_seed(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _official_gate_fixture(tmp_path, monkeypatch)
    forged = copy.deepcopy(fixture["gate"])
    seed = 43
    selection = fixture["selections"][seed]
    wrong_choice = {
        "seed": seed,
        "tie_break": [
            "calibration_f1_desc",
            "calibration_precision_desc",
            "seed_asc",
        ],
        "selection_receipt_sha256": fixture["selection_bindings"][seed][
            "sha256"
        ],
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "checkpoint_bundle_manifest_sha256": selection[
            "checkpoint_bundle_manifest_sha256"
        ],
        "checkpoint_bundle_tree_sha256": selection[
            "checkpoint_bundle_tree_sha256"
        ],
    }
    forged["calibration_choice"] = wrong_choice
    forged["authorized_candidate"] = copy.deepcopy(wrong_choice)
    _write_json(fixture["paths"].outer_gate, forged)

    with pytest.raises(
        MODULE.JointEvaluationError,
        match="calibration_choice",
    ):
        _load_gate_from_fixture(fixture)


def test_real_canonical_batch_failure_cannot_be_overridden_by_gate(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _official_gate_fixture(tmp_path, monkeypatch)
    fixture["state"]["outer_metrics"] = {
        42: joint.metric_from_counts(47, 53, 53),
        43: joint.metric_from_counts(50, 50, 50),
        44: joint.metric_from_counts(60, 40, 40),
    }

    with pytest.raises(
        MODULE.JointEvaluationError,
        match="canonical outer batch does not pass",
    ):
        _load_gate_from_fixture(fixture)


def test_gate_script_hash_is_checked_against_current_source(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _official_gate_fixture(tmp_path, monkeypatch)
    forged = copy.deepcopy(fixture["gate"])
    forged["gate_script_sha256"] = "0" * 64
    _write_json(fixture["paths"].outer_gate, forged)

    with pytest.raises(
        MODULE.JointEvaluationError,
        match="gate_script_sha256",
    ):
        _load_gate_from_fixture(fixture)


def test_selection_tamper_during_gate_validation_is_rejected(
    tmp_path: Path,
    monkeypatch,
):
    fixture = _official_gate_fixture(tmp_path, monkeypatch)
    tampered = copy.deepcopy(fixture["selections"][43])
    tampered["selected_epoch"] = 9
    _write_json(fixture["selection_paths"][43], tampered)

    with pytest.raises(
        MODULE.JointEvaluationError,
        match="selection changed",
    ):
        _load_gate_from_fixture(fixture)


def test_official_batch_metric_parser_recomputes_integer_ratios():
    batch = {
        "models": [
            {
                "seed": seed,
                "metric": joint.metric_from_counts(50, 50, 50),
            }
            for seed in MODULE.ALL_SEEDS
        ]
    }
    metrics = MODULE._batch_outer_metrics(batch)
    assert joint.outer_metrics_pass(metrics)

    batch["models"][0]["metric"]["precision"] = {
        "numerator": 99,
        "denominator": 100,
    }
    with pytest.raises(
        MODULE.JointEvaluationError,
        match="integer counts",
    ):
        MODULE._batch_outer_metrics(batch)


def test_official_batch_must_match_all_three_selection_receipts(
    tmp_path: Path,
    monkeypatch,
):
    real_validator = MODULE._validate_outer_batch_for_official
    fixture = _official_gate_fixture(tmp_path, monkeypatch)
    invalid_batch = {
        "schema_version": MODULE.OUTER_BATCH_SCHEMA_VERSION,
        "kind": MODULE.OUTER_BATCH_KIND,
        "dataset_role": run_protocol.OUTER_DATASET_ROLE,
        "outer_accessed": True,
        "official_dev_accessed": False,
        "selection_used": False,
        "single_dataset_read": True,
        "same_row_loop_for_all_seeds": True,
        "completed": True,
        "common_bindings": fixture["common"],
        "selection_receipts": [],
    }
    monkeypatch.setattr(
        MODULE,
        "load_canonical_completion",
        lambda *_args, **_kwargs: (
            invalid_batch,
            {"sha256": "f" * 64, "size": 1},
        ),
    )
    with pytest.raises(
        MODULE.JointEvaluationError,
        match="common or selection",
    ):
        real_validator(
            fixture["binding"],
            selections=fixture["selections"],
            selection_paths=fixture["selection_paths"],
            selection_bindings=fixture["selection_bindings"],
            common=fixture["common"],
        )


def test_preflight_failure_happens_before_official_marker(
    tmp_path: Path,
    monkeypatch,
):
    root, binding = _make_run(tmp_path)
    paths = run_protocol.canonical_run_paths(binding)
    candidate = paths.seed_model(42)
    candidate.mkdir(parents=True)
    selection_path = candidate / "selection.json"
    selection = {"seed": 42, "split_manifest_sha256": "a" * 64}
    selection_path.write_text(json.dumps(selection) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        MODULE,
        "_selection_inputs",
        lambda *_args, **_kwargs: (
            {
                42: selection,
                43: {"seed": 43},
                44: {"seed": 44},
            },
            {
                seed: paths.seed_model(seed) / "selection.json"
                for seed in MODULE.ALL_SEEDS
            },
            {
                seed: {"sha256": f"{seed:02d}" * 32, "size": 1}
                for seed in MODULE.ALL_SEEDS
            },
            {
                "train_sha256": "a" * 64,
                "base_model": str(paths.base_model),
                "base_model_tree_sha256": "b" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "_preflight_binding",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MODULE.JointEvaluationError("preflight failed")
        ),
    )

    with pytest.raises(MODULE.JointEvaluationError, match="preflight failed"):
        MODULE.run_official_dev_confirmation(
            run_root=root,
            selection_path=selection_path,
            preflight_receipt_path=paths.preflight_receipt,
            outer_gate_receipt_path=paths.outer_gate,
            official_dev_path=tmp_path / "dev.jsonl",
            device="cpu",
        )
    assert not paths.official_dev_marker.exists()
    assert not paths.official_dev_completion.exists()


def test_both_batch_subcommands_require_run_root():
    parser = MODULE._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "outer",
                "--selection",
                "a",
                "--split-manifest",
                "b",
                "--outer-shard",
                "c",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "official-dev-confirmation",
                "--selection",
                "a",
                "--preflight-receipt",
                "b",
                "--outer-gate-receipt",
                "c",
                "--official-dev",
                "d",
            ]
        )
