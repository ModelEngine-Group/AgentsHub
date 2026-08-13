import copy
import json
import os
import time

import pytest

from dki.bench.npu.om_runtime import OmRuntime
from dki.operators.medical_entity_extractor_om_shadow.process import MedicalEntityExtractorOmShadow


@pytest.fixture(autouse=True)
def _shadow_hash_secret(monkeypatch):
    monkeypatch.setenv("DKI_SHADOW_HASH_KEY", "unit-test-only-secret")


class FakeRuntime:
    def __init__(self, failure=None, delay=0):
        self.failure = failure
        self.delay = delay
        self.closed = False
        self.batches = []

    def execute_batch(self, texts, timeout_s=1):
        self.batches.append(list(texts))
        if self.delay:
            time.sleep(self.delay)
        if self.failure:
            raise self.failure
        return [[{"span": [0, 1], "type": "Disease"}] for _ in texts]

    def close(self):
        self.closed = True


class OmAdapter(FakeRuntime):
    def open(self):
        return None


class OmAdapterFactory:
    def __call__(self):
        return OmAdapter()


def _sample(text="患者高血压"):
    return {
        "fileId": "patient-raw-id",
        "fileName": "x.txt",
        "text": text,
        "data": b"",
        "ext_params": {},
    }


def _public_projection(sample):
    payload = json.loads(sample["text"])
    return payload, sample["ext_params"]


def test_enabled_shadow_without_runtime_fails_closed():
    with pytest.raises(ValueError, match="requires shadow_runtime"):
        MedicalEntityExtractorOmShadow(shadowEnabled=True)


def test_disabled_shadow_ignores_unavailable_factory():
    op = MedicalEntityExtractorOmShadow(
        shadowEnabled=False, shadowRuntimeFactory="no_such_module.factory"
    )
    assert json.loads(op.execute(_sample())["text"])["entities"]


def test_enabled_shadow_requires_pseudonymization_secret(monkeypatch):
    monkeypatch.delenv("DKI_SHADOW_HASH_KEY", raising=False)
    with pytest.raises(ValueError, match="pseudonymization secret"):
        MedicalEntityExtractorOmShadow(shadow_runtime=FakeRuntime())


def test_custom_datamate_field_keys_are_forwarded():
    op = MedicalEntityExtractorOmShadow(shadowEnabled=False, text_key="content")
    sample = _sample()
    sample["content"] = sample.pop("text")
    output = op.execute(sample)
    assert json.loads(output["content"])["entities"]


def test_factory_result_contract_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "dki.operators.medical_entity_extractor_om_shadow.process._load_factory",
        lambda _: lambda **kwargs: {},
    )
    with pytest.raises(ValueError, match="initialization failed"):
        MedicalEntityExtractorOmShadow(shadowRuntimeFactory="fake.factory")


def test_shadow_batches_queued_requests_in_one_runtime_call(tmp_path):
    runtime = FakeRuntime()
    op = MedicalEntityExtractorOmShadow(
        shadow_runtime=runtime,
        queueDepth=4,
        maxBatchSize=4,
        maxWaitMs=100,
        sidecarPath=str(tmp_path / "shadow.jsonl"),
    )
    for text in ("高血压", "糖尿病", "肺结核"):
        op.execute(_sample(text))
    assert op.flush()
    assert runtime.batches == [["高血压", "糖尿病", "肺结核"]]
    records = [
        json.loads(line)
        for line in (tmp_path / "shadow.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert all(record["error"] is None for record in records)
    op.close()


def test_shadow_does_not_change_primary_output(tmp_path):
    baseline = MedicalEntityExtractorOmShadow(shadowEnabled=False)
    shadow = MedicalEntityExtractorOmShadow(
        shadow_runtime=FakeRuntime(), sidecarPath=str(tmp_path / "shadow.jsonl")
    )
    left = baseline.execute(copy.deepcopy(_sample()))
    right = shadow.execute(copy.deepcopy(_sample()))
    assert _public_projection(left) == _public_projection(right)
    assert shadow.flush()
    record = json.loads((tmp_path / "shadow.jsonl").read_text(encoding="utf-8"))
    assert "text_hmac_sha256" in record
    assert "text_sha256" not in record
    assert "patient-raw-id" not in json.dumps(record)
    assert "患者高血压" not in json.dumps(record, ensure_ascii=False)
    shadow.close()


def test_real_om_runtime_claims_shadow_worker_thread(tmp_path):
    runtime = OmRuntime(OmAdapterFactory())
    op = MedicalEntityExtractorOmShadow(
        shadow_runtime=runtime, sidecarPath=str(tmp_path / "shadow.jsonl")
    )
    op.execute(_sample())
    assert op.flush()
    op.close()
    record = json.loads((tmp_path / "shadow.jsonl").read_text(encoding="utf-8"))
    assert record["error"] is None
    assert runtime.health.state == "closed"


def test_shadow_failure_is_isolated(tmp_path):
    op = MedicalEntityExtractorOmShadow(
        shadow_runtime=FakeRuntime(RuntimeError("NPU unavailable")),
        sidecarPath=str(tmp_path / "shadow.jsonl"),
    )
    output = op.execute(_sample())
    assert json.loads(output["text"])["entities"]
    assert op.flush()
    assert json.loads((tmp_path / "shadow.jsonl").read_text())["error"] == "RuntimeError"
    op.close()


def test_shadow_wrapper_loads_file_when_first_operator(tmp_path):
    source = tmp_path / "record.txt"
    source.write_text("患者高血压", encoding="utf-8")
    sample = _sample("")
    sample.update({"filePath": str(source), "fileType": "txt"})
    op = MedicalEntityExtractorOmShadow(shadowEnabled=False, is_first_op=True)
    output = op.execute(sample)
    assert json.loads(output["text"])["entities"]


def test_shadow_preserves_section_primary_semantics():
    payload = json.dumps({"sections": {"诊断": "高血压", "治疗方案": "给予阿司匹林"}}, ensure_ascii=False)
    sample = _sample(payload)
    baseline = MedicalEntityExtractorOmShadow(shadowEnabled=False)
    output = baseline.execute(sample)
    parsed = json.loads(output["text"])
    assert parsed["sections"] == {"诊断": "高血压", "治疗方案": "给予阿司匹林"}
    assert all(entity["span_base"] == "section" for entity in parsed["entities"])
    assert {entity["section"] for entity in parsed["entities"]} == {"诊断", "治疗方案"}


def test_close_drains_pending_shadow_work(tmp_path):
    op = MedicalEntityExtractorOmShadow(
        shadow_runtime=FakeRuntime(delay=0.03),
        queueDepth=8,
        shadowTimeoutMs=1000,
        sidecarPath=str(tmp_path / "shadow.jsonl"),
    )
    for index in range(4):
        op.execute(_sample(f"高血压{index}"))
    op.close()
    records = [json.loads(line) for line in (tmp_path / "shadow.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4


def test_queue_pressure_drops_shadow_without_blocking():
    op = MedicalEntityExtractorOmShadow(
        shadow_runtime=FakeRuntime(delay=0.2), queueDepth=1, shadowTimeoutMs=1000
    )
    started = time.perf_counter()
    outputs = [op.execute(_sample(str(index))) for index in range(10)]
    elapsed = time.perf_counter() - started
    assert all(json.loads(item["text"])["entities"] == [] for item in outputs)
    assert op.dropped > 0
    assert elapsed < 0.15
    op.close()
