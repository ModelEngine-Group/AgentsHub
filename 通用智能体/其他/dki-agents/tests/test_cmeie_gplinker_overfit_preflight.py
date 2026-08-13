import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_cmeie_gplinker_overfit.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_cmeie_gplinker_overfit",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
BUILDER_SCRIPT = ROOT / "scripts" / "build_cmeie_split_manifest.py"
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_cmeie_split_manifest_for_smoke_tests",
    BUILDER_SCRIPT,
)
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
assert BUILDER_SPEC.loader is not None
BUILDER_SPEC.loader.exec_module(BUILDER)

from dki.nlp import cmeie_split_protocol as split_protocol


def _row(*predicates: str) -> dict:
    return {
        "text": "fixture",
        "spo_list": [
            {"predicate": predicate}
            for predicate in predicates
        ],
    }


def test_fit_smoke_selection_greedily_covers_predicates():
    rows = [
        _row("a"),
        _row("a", "b"),
        _row("c"),
        _row(),
        _row("d", "e", "f"),
    ]
    selected = MODULE.select_fit_smoke_indices(
        rows,
        [0, 1, 2, 3, 4],
        3,
    )
    assert selected == [4, 1, 2]


def test_fit_smoke_selection_rejects_unlabelled_shortfall():
    with pytest.raises(ValueError, match="fewer labelled rows"):
        MODULE.select_fit_smoke_indices(
            [_row(), _row("a")],
            [0, 1],
            2,
        )


def test_preflight_adamw_policy_matches_formal_training():
    assert MODULE._uses_weight_decay("relation_head.q.weight")
    assert not MODULE._uses_weight_decay("relation_head.q.bias")
    assert not MODULE._uses_weight_decay(
        "encoder.layer.0.LayerNorm.weight"
    )


def test_base_model_tree_report_lists_sorted_recomputable_entries(
    tmp_path: Path,
):
    model = tmp_path / "model"
    nested = model / "tokenizer"
    nested.mkdir(parents=True)
    (model / "config.json").write_bytes(b"config")
    (nested / "vocab.json").write_bytes(b"vocabulary")

    report = MODULE._tree_report(model)

    assert report["root"] == str(model.resolve())
    assert report["algorithm"] == MODULE.TREE_HASH_ALGORITHM
    assert [entry["path"] for entry in report["entries"]] == [
        "config.json",
        "tokenizer/vocab.json",
    ]
    digest = hashlib.sha256()
    for entry in report["entries"]:
        digest.update(entry["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    assert report["sha256"] == digest.hexdigest()
    assert report["n_files"] == 2
    assert report["total_bytes"] == len(b"configvocabulary")


def test_artifact_binding_hashes_are_canonical_and_input_sensitive():
    first = MODULE._artifact_binding_hashes(
        tokenizer_vocabulary={"z": 9, "a": 1},
        entity_labels=["subject", "object"],
        predicates=["treats"],
        schema_constraints={"treats": [["subject", "object"]]},
    )
    reordered = MODULE._artifact_binding_hashes(
        tokenizer_vocabulary={"a": 1, "z": 9},
        entity_labels=["subject", "object"],
        predicates=["treats"],
        schema_constraints={"treats": [["subject", "object"]]},
    )
    changed = MODULE._artifact_binding_hashes(
        tokenizer_vocabulary={"a": 1, "z": 10},
        entity_labels=["subject", "object"],
        predicates=["treats"],
        schema_constraints={"treats": [["subject", "object"]]},
    )

    assert first == reordered
    assert all(len(value) == 64 for value in first.values())
    assert (
        first["tokenizer_vocabulary_sha256"]
        != changed["tokenizer_vocabulary_sha256"]
    )


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_nonfinite_training_diagnostics_are_published_as_json_null(
    tmp_path: Path,
    value: float,
):
    path = tmp_path / "overfit.json"
    diagnostic = MODULE._json_safe_diagnostic_float(value)
    payload = {
        "snapshots": [
            {
                "previous_pre_clip_grad_norm": diagnostic,
            }
        ]
    }

    MODULE._publish_once(path, payload)

    encoded = path.read_bytes()
    assert encoded == MODULE.canonical_overfit_json_bytes(payload)
    assert json.loads(encoded)["snapshots"][0][
        "previous_pre_clip_grad_norm"
    ] is None
    assert b"NaN" not in encoded
    assert b"Infinity" not in encoded


def test_nonfinite_training_summary_uses_json_safe_nulls():
    summary = MODULE._summary([1.0, float("inf")])

    assert summary == {
        "count": 2,
        "minimum": None,
        "median": None,
        "maximum": None,
    }


def test_overfit_publisher_rejects_unsanitized_nonfinite_value(
    tmp_path: Path,
):
    path = tmp_path / "overfit.json"

    with pytest.raises(ValueError, match="non-JSON or non-finite"):
        MODULE._publish_once(path, {"diagnostic": float("inf")})

    assert not path.exists()


def test_copied_smoke_artifact_path_is_not_canonical(tmp_path):
    expected = tmp_path / "run" / "evidence" / "cmeie" / "overfit.json"
    copied = tmp_path / "copy" / "overfit.json"

    with pytest.raises(ValueError, match="canonical run path"):
        MODULE._require_canonical_path(
            copied,
            expected,
            label="overfit output",
        )


def test_strict_preflight_passes_every_authoritative_binding(
    tmp_path,
    monkeypatch,
):
    receipt = tmp_path / "run" / "evidence" / "cmeie" / "preflight.json"
    model = tmp_path / "run" / "models" / "cmeie_base_model"
    binding = SimpleNamespace(
        rid="20260729T120000Z-gbbbbbbb-taaaaaaaaaaaa",
        git_head="b" * 40,
        included_tree_sha256="a" * 64,
        source_manifest_sha256="c" * 64,
    )
    paths = SimpleNamespace(
        preflight_receipt=receipt,
        base_model=model,
    )
    calls = []

    def validate(path, **kwargs):
        calls.append((path, kwargs))
        return {
            "preflight_receipt_path": str(receipt),
            "base_model_canonical_path": str(model),
            "preflight_producer_script_sha256": "d" * 64,
        }

    monkeypatch.setattr(
        MODULE,
        "validate_cmeie_npu_preflight_receipt",
        validate,
    )
    common = MODULE.validate_strict_npu_preflight(
        run_binding=binding,
        paths=paths,
        source_train_sha256="e" * 64,
        base_model_tree={"sha256": "f" * 64},
        smoke_script_sha256="1" * 64,
    )

    assert common["preflight_receipt_path"] == str(receipt)
    assert calls == [
        (
            str(receipt),
            {
                "expected_source_tree_sha256": "a" * 64,
                "expected_source_manifest_sha256": "c" * 64,
                "expected_cmeie_train_sha256": "e" * 64,
                "expected_base_model_canonical_path": str(model),
                "expected_base_model_tree_sha256": "f" * 64,
                "expected_run_binding": {
                    "rid": binding.rid,
                    "git_head": "b" * 40,
                    "source_tree_sha256": "a" * 64,
                    "source_manifest_sha256": "c" * 64,
                },
            },
        )
    ]


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (FileNotFoundError("strict receipt missing"), "missing"),
        (ValueError("ready flag is false"), "ready flag"),
        (ValueError("base model tree mismatch"), "model tree"),
        (ValueError("source tree mismatch"), "source tree"),
    ],
)
def test_strict_preflight_failure_blocks_smoke(
    tmp_path,
    monkeypatch,
    failure,
    message,
):
    binding = SimpleNamespace(
        rid="20260729T120000Z-gbbbbbbb-taaaaaaaaaaaa",
        git_head="b" * 40,
        included_tree_sha256="a" * 64,
        source_manifest_sha256="c" * 64,
    )
    paths = SimpleNamespace(
        preflight_receipt=tmp_path / "preflight.json",
        base_model=tmp_path / "model",
    )

    def reject(*args, **kwargs):
        raise failure

    monkeypatch.setattr(
        MODULE,
        "validate_cmeie_npu_preflight_receipt",
        reject,
    )
    with pytest.raises(type(failure), match=message):
        MODULE.validate_strict_npu_preflight(
            run_binding=binding,
            paths=paths,
            source_train_sha256="e" * 64,
            base_model_tree={"sha256": "f" * 64},
            smoke_script_sha256="1" * 64,
        )


def test_strict_preflight_rejects_smoke_as_remote_producer(
    tmp_path,
    monkeypatch,
):
    receipt = tmp_path / "preflight.json"
    model = tmp_path / "model"
    binding = SimpleNamespace(
        rid="20260729T120000Z-gbbbbbbb-taaaaaaaaaaaa",
        git_head="b" * 40,
        included_tree_sha256="a" * 64,
        source_manifest_sha256="c" * 64,
    )
    paths = SimpleNamespace(
        preflight_receipt=receipt,
        base_model=model,
    )
    smoke_sha256 = "1" * 64
    monkeypatch.setattr(
        MODULE,
        "validate_cmeie_npu_preflight_receipt",
        lambda *args, **kwargs: {
            "preflight_receipt_path": str(receipt),
            "base_model_canonical_path": str(model),
            "preflight_producer_script_sha256": smoke_sha256,
        },
    )

    with pytest.raises(ValueError, match="must differ"):
        MODULE.validate_strict_npu_preflight(
            run_binding=binding,
            paths=paths,
            source_train_sha256="e" * 64,
            base_model_tree={"sha256": "f" * 64},
            smoke_script_sha256=smoke_sha256,
        )


def test_split_source_tree_binding_requires_v6_and_both_hashes():
    expected_tree = "a" * 64
    expected_manifest = "b" * 64
    assert MODULE.validate_split_source_tree_binding(
        {
            "schema_version": 6,
            "source_tree_sha256": "A" * 64,
            "source_tree_manifest_sha256": "B" * 64,
        },
        expected_tree,
        expected_manifest,
    ) == {
        "source_tree_sha256": expected_tree,
        "source_tree_manifest_sha256": expected_manifest,
    }
    with pytest.raises(ValueError, match="schema is not v6"):
        MODULE.validate_split_source_tree_binding(
            {
                "schema_version": 5,
                "source_tree_sha256": expected_tree,
                "source_tree_manifest_sha256": expected_manifest,
            },
            expected_tree,
            expected_manifest,
        )
    with pytest.raises(ValueError, match="manifest SHA-256 is invalid"):
        MODULE.validate_split_source_tree_binding(
            {
                "schema_version": 6,
                "source_tree_sha256": expected_tree,
            },
            expected_tree,
            expected_manifest,
        )
    with pytest.raises(ValueError, match="differs from the candidate"):
        MODULE.validate_split_source_tree_binding(
            {
                "schema_version": 6,
                "source_tree_sha256": "c" * 64,
                "source_tree_manifest_sha256": expected_manifest,
            },
            expected_tree,
            expected_manifest,
        )
    with pytest.raises(ValueError, match="differs from the current file"):
        MODULE.validate_split_source_tree_binding(
            {
                "schema_version": 6,
                "source_tree_sha256": expected_tree,
                "source_tree_manifest_sha256": "c" * 64,
            },
            expected_tree,
            expected_manifest,
        )


def test_overfit_gate_requires_schema_constraint_on():
    passing = {
        "tp": 95,
        "fp": 5,
        "fn": 5,
        "precision": 0.95,
        "recall": 0.95,
        "f1": 0.95,
    }
    assert MODULE.overfit_gate_passed(
        metric=passing,
        schema_on=True,
        minimum_train_f1=0.95,
    )
    assert not MODULE.overfit_gate_passed(
        metric={
            "tp": 100,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
        schema_on=False,
        minimum_train_f1=0.95,
    )
    assert not MODULE.overfit_gate_passed(
        metric={
            "tp": 94,
            "fp": 6,
            "fn": 6,
            "precision": 0.94,
            "recall": 0.94,
            "f1": 0.94,
        },
        schema_on=True,
        minimum_train_f1=0.95,
    )


def test_overfit_gate_rejects_self_reported_f1_that_disagrees_with_counts():
    with pytest.raises(ValueError, match="disagrees with tp/fp/fn"):
        MODULE.overfit_gate_passed(
            metric={
                "tp": 0,
                "fp": 100,
                "fn": 100,
                "precision": 0.98,
                "recall": 0.98,
                "f1": 0.98,
            },
            schema_on=True,
            minimum_train_f1=0.95,
        )


def test_fixed_overfit_protocol_rejects_flags_seed_training_would_reject():
    args = SimpleNamespace(
        device="npu",
        sample_size=8,
        steps=400,
        max_length=320,
        fp16=True,
        minimum_train_f1=0.95,
        pointer_variant="efficient",
        head_reduction="balanced",
        positive_weight=1.0,
        positive_margin=0.0,
        encoder_learning_rate=2e-5,
        head_learning_rate=1e-3,
        freeze_encoder=False,
    )
    MODULE.validate_fixed_overfit_protocol(
        args,
        threshold_values=(-0.5, 0.0, 0.5),
    )

    args.fp16 = False
    args.freeze_encoder = True
    with pytest.raises(
        ValueError,
        match="fp16=False.*freeze_encoder=True",
    ):
        MODULE.validate_fixed_overfit_protocol(
            args,
            threshold_values=(-0.5, 0.0, 0.5),
        )


def test_smoke_calls_fit_shard_validator_before_model_work(
    tmp_path,
    monkeypatch,
):
    fit_data = tmp_path / "fit.jsonl"
    fit_data.write_text("{}\n", encoding="utf-8")
    source_manifest = tmp_path / "included_tree.json"
    source_manifest.write_text(
        json.dumps({"included_tree_sha256": "a" * 64}),
        encoding="utf-8",
    )
    split_manifest = tmp_path / "split_manifest.json"
    split_manifest.write_text("{}", encoding="utf-8")
    output = tmp_path / "receipt.json"
    calls = []

    def reject(*args, **kwargs):
        calls.append((args, kwargs))
        raise ValueError("full validator sentinel")

    monkeypatch.setattr(
        MODULE,
        "load_and_validate_role_shard",
        reject,
    )
    monkeypatch.setattr(
        MODULE,
        "validate_canonical_smoke_paths",
        lambda args: (
            SimpleNamespace(
                included_tree_sha256="a" * 64,
                source_root=tmp_path,
            ),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "validate_source_tree_manifest",
        lambda *args, **kwargs: {
            "included_tree_sha256": "a" * 64,
            "git_head": "b" * 40,
            "n_files": 1,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--run-root",
            str(tmp_path / "run"),
            "--fit-data",
            str(fit_data),
            "--split-manifest",
            str(split_manifest),
            "--base-model",
            str(tmp_path / "model"),
            "--output",
            str(output),
            "--source-tree-sha256",
            "a" * 64,
                "--source-tree-manifest",
                str(source_manifest),
                "--fp16",
            ],
        )

    with pytest.raises(SystemExit, match="full validator sentinel"):
        MODULE.main()
    assert len(calls) == 1


def test_smoke_fit_loader_cannot_open_source_or_outer(
    tmp_path,
    monkeypatch,
):
    rows = [
        {
            "text": f"唯一标题{i}@正文{i}",
            "spo_list": [{"predicate": "P"}],
        }
        for i in range(30)
    ]
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in rows
        ),
        encoding="utf-8",
    )
    split_dir = tmp_path / "split"
    BUILDER.build_split_directory(
        source=source,
        output_dir=split_dir,
        seed="20260725",
        source_tree_sha256="a" * 64,
        source_tree_manifest_sha256="b" * 64,
    )
    forbidden = {
        source.absolute(),
        (
            split_dir / split_protocol.SHARD_FILENAMES["outer"]
        ).absolute(),
    }
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.absolute() in forbidden:
            raise AssertionError(f"forbidden data access: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    fit, binding = MODULE.load_isolated_fit_data(
        fit_data=split_dir / split_protocol.SHARD_FILENAMES["fit"],
        split_manifest=split_dir / split_protocol.MANIFEST_FILENAME,
        source_tree_sha256="a" * 64,
        source_tree_manifest_sha256="b" * 64,
    )

    assert fit
    assert binding["role"] == "fit"
    assert binding["source_train_accessed"] is False
    assert binding["outer_accessed"] is False
