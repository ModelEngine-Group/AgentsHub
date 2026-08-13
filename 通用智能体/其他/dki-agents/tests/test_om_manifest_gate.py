"""Release A hard-gate tests: OmManifest must be enforced (fail closed) before
any ACL/OM load, both as a standalone preload check and wired into the runtime
boundary. These run on CPU without ACL/transformers."""

import json

import pytest

from dki.bench.npu.om_manifest import (
    OmCompatibilityError,
    OmManifest,
    TensorSpec,
    enforce_preload_manifest,
    sha256_tree,
)
from dki.bench.npu.om_runtime import FailureKind, OmRuntime, OmRuntimeError
from dki.evidence import sha256_file


def _make_artifacts(tmp_path, soc="Ascend910B2C", cann="8.5.1"):
    om = tmp_path / "model.om"
    tok = tmp_path / "tokenizer.json"
    lab = tmp_path / "labels.json"
    om.write_bytes(b"om-bytes")
    tok.write_text("tokenizer", encoding="utf-8")
    lab.write_text("labels", encoding="utf-8")
    inputs = (
        TensorSpec("input_ids", "int64", (64, 256)),
        TensorSpec("attention_mask", "int64", (64, 256)),
        TensorSpec("token_type_ids", "int64", (64, 256)),
    )
    outputs = (TensorSpec("logits", "float32", (64, 256, 13)),)
    manifest = OmManifest(
        1, "medical_ner_bert", sha256_file(om), sha256_file(tok), sha256_file(lab),
        soc, cann, inputs, outputs,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "model_id": manifest.model_id,
        "om_sha256": manifest.om_sha256,
        "tokenizer_sha256": manifest.tokenizer_sha256,
        "labels_sha256": manifest.labels_sha256,
        "soc_version": soc,
        "cann_version": cann,
        "inputs": [{"name": s.name, "dtype": s.dtype, "shape": list(s.shape)} for s in inputs],
        "outputs": [{"name": s.name, "dtype": s.dtype, "shape": list(s.shape)} for s in outputs],
    }, ensure_ascii=False), encoding="utf-8")
    return path, om, tok, lab


def _config(path, om, tok, lab, **extra):
    cfg = {
        "manifestPath": str(path), "omPath": str(om),
        "tokenizerPath": str(tok), "labelsPath": str(lab),
    }
    cfg.update(extra)
    return cfg


def test_no_manifest_is_noop():
    assert enforce_preload_manifest({}) is None
    assert enforce_preload_manifest({"omPath": "x"}) is None


def test_gate_passes_with_matching_artifacts(tmp_path):
    path, om, tok, lab = _make_artifacts(tmp_path)
    manifest = enforce_preload_manifest(_config(path, om, tok, lab, socVersion="Ascend910B2C", cannVersion="8.5.1"))
    assert manifest is not None and manifest.model_id == "medical_ner_bert"


def test_gate_requires_all_artifact_paths(tmp_path):
    path, om, tok, lab = _make_artifacts(tmp_path)
    with pytest.raises(OmCompatibilityError, match="artifact path"):
        enforce_preload_manifest({"manifestPath": str(path), "omPath": str(om)})


def test_gate_fails_closed_on_hash_mismatch(tmp_path):
    path, om, tok, lab = _make_artifacts(tmp_path)
    tok.write_text("tampered", encoding="utf-8")
    with pytest.raises(OmCompatibilityError, match="tokenizer hash mismatch"):
        enforce_preload_manifest(_config(path, om, tok, lab))


def test_gate_fails_closed_on_soc_mismatch(tmp_path):
    path, om, tok, lab = _make_artifacts(tmp_path)
    with pytest.raises(OmCompatibilityError, match="SoC mismatch"):
        enforce_preload_manifest(_config(path, om, tok, lab, socVersion="Ascend910B4"))


def test_verify_io_split_matches_and_rejects(tmp_path):
    path, om, tok, lab = _make_artifacts(tmp_path)
    manifest = OmManifest.load(path)
    manifest.verify_io(inputs=manifest.inputs, outputs=manifest.outputs)
    with pytest.raises(OmCompatibilityError, match="output signature"):
        manifest.verify_io(inputs=manifest.inputs, outputs=(TensorSpec("logits", "float32", (64, 256, 7)),))


def test_runtime_gate_fails_closed_before_factory_runs(tmp_path):
    path, om, tok, lab = _make_artifacts(tmp_path)
    om.write_bytes(b"tampered-om")  # break OM hash
    factory_calls = []

    def factory():
        factory_calls.append(1)
        raise AssertionError("factory must not run when the manifest gate fails")

    runtime = OmRuntime(factory, preload_config=_config(path, om, tok, lab))
    with pytest.raises(OmRuntimeError) as raised:
        runtime.execute(1)
    assert raised.value.kind in {FailureKind.COMPATIBILITY, FailureKind.ACL_FATAL}
    assert factory_calls == []


def test_runtime_gate_allows_open_when_manifest_matches(tmp_path):
    path, om, tok, lab = _make_artifacts(tmp_path)
    events = []

    class FakeAdapter:
        def open(self):
            events.append("open")

        def execute(self, value):
            return value + 1

        def close(self):
            events.append("close")

    runtime = OmRuntime(lambda: FakeAdapter(), preload_config=_config(path, om, tok, lab))
    assert runtime.execute(41) == 42
    assert events == ["open"]
    runtime.close()


def test_schema_v3_binds_checkpoint_tree_export_receipt_and_atc_log(tmp_path):
    path, om, tok, lab = _make_artifacts(tmp_path)
    base = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "weights.bin").write_bytes(b"weights")
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    atc_log = tmp_path / "atc.log"
    receipt = tmp_path / "onnx.export.json"
    receipt.write_text(json.dumps({
        "schema_version": 1,
        "kind": "cmeee_globalpointer_onnx_export",
        "model_family": "cmeee_globalpointer",
        "model_id": base["model_id"],
        "checkpoint_kind": "directory_tree_v1",
        "checkpoint_sha256": sha256_tree(checkpoint),
        "tokenizer_sha256": sha256_file(tok),
        "labels_sha256": sha256_file(lab),
        "onnx_sha256": sha256_file(onnx),
        "inputs": base["inputs"],
        "outputs": base["outputs"],
    }), encoding="utf-8")
    atc_log.write_text(json.dumps({
        "schema_version": 1,
        "kind": "atc_compile_log",
        "returncode": 0,
        "onnx_sha256": sha256_file(onnx),
        "om_sha256": sha256_file(om),
        "export_receipt_sha256": sha256_file(receipt),
        "soc_version": base["soc_version"],
        "output_path": str(om.resolve()),
    }), encoding="utf-8")
    base.update({
        "schema_version": 3,
        "model_family": "cmeee_globalpointer",
        "checkpoint_kind": "directory_tree_v1",
        "checkpoint_sha256": sha256_tree(checkpoint),
        "export_receipt_sha256": sha256_file(receipt),
        "onnx_sha256": sha256_file(onnx),
        "atc_log_sha256": sha256_file(atc_log),
    })
    path.write_text(json.dumps(base), encoding="utf-8")

    manifest = enforce_preload_manifest(_config(
        path,
        om,
        tok,
        lab,
        checkpointPath=str(checkpoint),
        exportReceiptPath=str(receipt),
        atcLogPath=str(atc_log),
    ))
    assert manifest is not None and manifest.schema_version == 3

    (checkpoint / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(OmCompatibilityError, match="checkpoint hash mismatch"):
        manifest.verify_artifacts(om, tok, lab, checkpoint, receipt, atc_log)
