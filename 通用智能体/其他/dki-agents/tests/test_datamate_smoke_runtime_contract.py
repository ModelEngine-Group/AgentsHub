"""Regression tests for the DataMate OM-shadow smoke runtime contract."""

import json

import pytest

from dki.operators.medical_entity_extractor_om_shadow.process import (
    MedicalEntityExtractorOmShadow,
)
from scripts.datamate_smoke import _FailingRuntime, _StubRuntime


def _sample(text: str = "患者高血压") -> dict:
    return {
        "fileId": "smoke-id",
        "fileName": "smoke.txt",
        "text": text,
        "data": b"",
        "ext_params": {},
    }


def test_stub_runtime_implements_batch_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("DKI_SHADOW_HASH_KEY", "smoke-contract-test")
    runtime = _StubRuntime()

    predictions = runtime.execute_batch(
        ["高血压", "糖尿病"],
        timeout_s=0.5,
    )

    assert len(predictions) == 2
    assert all(len(batch) == 1 for batch in predictions)
    assert [batch[0]["name"] for batch in predictions] == ["高血压", "高血压"]

    sidecar = tmp_path / "shadow.jsonl"
    operator = MedicalEntityExtractorOmShadow(
        shadow_runtime=runtime,
        sidecarPath=str(sidecar),
    )
    try:
        output = operator.execute(_sample())
        assert json.loads(output["text"])["entities"]
        assert operator.flush(timeout_s=3.0)
    finally:
        operator.close()
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["error"] is None
    assert record["shadow_count"] == 1


def test_failing_runtime_implements_batch_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("DKI_SHADOW_HASH_KEY", "smoke-contract-test")
    runtime = _FailingRuntime()

    with pytest.raises(RuntimeError, match="synthetic shadow failure"):
        runtime.execute_batch(["高血压"], timeout_s=0.5)

    sidecar = tmp_path / "shadow.jsonl"
    operator = MedicalEntityExtractorOmShadow(
        shadow_runtime=runtime,
        sidecarPath=str(sidecar),
    )
    try:
        output = operator.execute(_sample())
        assert json.loads(output["text"])["entities"]
        assert operator.flush(timeout_s=3.0)
    finally:
        operator.close()
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["error"] == "RuntimeError"
    assert record["shadow_count"] is None


def test_legacy_single_item_runtime_is_rejected(monkeypatch):
    monkeypatch.setenv("DKI_SHADOW_HASH_KEY", "smoke-contract-test")

    class LegacyRuntime:
        def execute(self, text, timeout_s=1.0):
            del text, timeout_s
            return []

    with pytest.raises(ValueError, match="execute_batch"):
        MedicalEntityExtractorOmShadow(shadow_runtime=LegacyRuntime())
