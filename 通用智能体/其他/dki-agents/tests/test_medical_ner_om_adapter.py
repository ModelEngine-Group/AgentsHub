"""CPU unit tests for the production medical-NER OM adapter's text->entities
logic, using fake tokenizer/runner (no ACL, no transformers). Locks the
tokenize -> OM argmax -> canonical BIO decode -> global span -> normalize path
so it is covered without the NPU host."""

import numpy as np
import pytest

from dki.bench.npu.medical_ner_om_adapter import MedicalNerOmAdapter
from dki.nlp.bio_decode import BioDecoder


class _FakeEncoding(dict):
    pass


class _FakeTokenizer:
    """Char-level fake: each Chinese char is one content token; wraps CLS/SEP
    with (0,0) offsets like an HF fast tokenizer."""

    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __call__(self, text, truncation=False, return_offsets_mapping=True,
                 return_tensors="np", add_special_tokens=True):
        ids = [self.cls_token_id]
        offsets = [[0, 0]]
        for i, _ch in enumerate(text):
            ids.append(1000 + i)
            offsets.append([i, i + 1])
        ids.append(self.sep_token_id)
        offsets.append([0, 0])
        return _FakeEncoding(
            input_ids=np.asarray([ids], dtype=np.int64),
            offset_mapping=np.asarray([offsets], dtype=np.int64),
        )


class _FakeRunner:
    """Returns crafted logits whose argmax marks the first three content tokens
    as B/I/I-Disease (offset by the leading CLS column)."""

    def __init__(self, batch, seq_len, num_labels):
        self.output_shape = (batch, seq_len, num_labels)
        self.input_ptrs = {"input_ids": 0, "attention_mask": 0, "token_type_ids": 0}
        self._batch = batch
        self._seq_len = seq_len
        self._num_labels = num_labels
        self.closed = False
        self.calls = 0

    def execute(self, arrays):
        self.calls += 1
        logits = np.zeros((self._batch, self._seq_len, self._num_labels), dtype=np.float32)
        logits[..., 0] = 1.0  # default O
        # 每个非 padding 行的前三个内容 token -> B/I/I-Disease。
        for row in range(self._batch):
            if not arrays["attention_mask"][row].any():
                continue
            for col, label in ((1, 1), (2, 2), (3, 2)):
                logits[row, col, :] = 0.0
                logits[row, col, label] = 5.0
        return [logits.reshape(-1)]

    def close(self):
        self.closed = True


def _prepared_adapter(batch=1, seq_len=8, overlap=0):
    adapter = MedicalNerOmAdapter(
        omPath="fake.om", modelName="fake-model", seqLen=seq_len, overlap=overlap,
    )
    runner = _FakeRunner(batch=batch, seq_len=seq_len, num_labels=3)
    adapter._runner = runner
    adapter._tokenizer = _FakeTokenizer()
    adapter._decoder = BioDecoder({0: "O", 1: "B-Disease", 2: "I-Disease"})
    adapter._vocab = {"高血压": ("高血压", "Disease")}
    adapter._batch = batch
    adapter._num_labels = 3
    adapter._cls_id = 101
    adapter._sep_id = 102
    adapter._pad_id = 0
    adapter._opened = True
    return adapter, runner


def test_adapter_extracts_typed_entity_with_global_span():
    adapter, _ = _prepared_adapter()
    entities = adapter.execute("高血压患者")
    assert len(entities) == 1
    ent = entities[0]
    assert ent["type"] == "Disease"
    assert ent["span"] == [0, 3]
    assert ent["mention"] == "高血压"
    assert ent["name"] == "高血压"  # canonical via vocab
    assert ent["source"] == "om"


def test_adapter_empty_text_returns_no_entities():
    adapter, _ = _prepared_adapter()
    assert adapter.execute("   ") == []
    assert adapter.execute({"text": ""}) == []


def test_adapter_accepts_dict_payload():
    adapter, _ = _prepared_adapter()
    entities = adapter.execute({"text": "高血压患者"})
    assert [e["name"] for e in entities] == ["高血压"]


def test_execute_batch_fills_static_rows_across_documents():
    adapter, runner = _prepared_adapter(batch=4)
    payloads = ["高血压患者", {"text": "糖尿病患者"}, "   ", "肺结核患者"]
    results = adapter.execute_batch(payloads)
    assert len(results) == len(payloads)
    assert [[e["mention"] for e in row] for row in results] == [
        ["高血压"], ["糖尿病"], [], ["肺结核"],
    ]
    assert runner.calls == 1, "三个文档窗口应共同填入一次静态 batch"


def test_execute_batch_flattens_long_document_windows_with_other_documents():
    adapter, runner = _prepared_adapter(batch=3, seq_len=8, overlap=0)
    # capacity=6：第一篇产生两个窗口，第二篇一个窗口，恰好一次静态 batch。
    results = adapter.execute_batch(["甲乙丙丁戊己庚辛", "高血压患者"])
    assert len(results) == 2
    assert runner.calls == 1
    assert results[0][0]["span"] == [0, 3]
    assert results[1][0]["mention"] == "高血压"


def test_execute_batch_only_tail_pads_final_om_call():
    adapter, runner = _prepared_adapter(batch=2)
    results = adapter.execute_batch(["高血压患者", "糖尿病患者", "肺结核患者"])
    assert len(results) == 3
    assert runner.calls == 2


def test_execute_batch_rejects_scalar_container():
    adapter, _ = _prepared_adapter()
    for bad in ("高血压", {"text": "高血压"}, None):
        try:
            adapter.execute_batch(bad)
        except TypeError:
            pass
        else:
            raise AssertionError(f"expected TypeError for {bad!r}")


@pytest.mark.parametrize("bad", [None, 7, [], b"text", {"text": None}, {}])
def test_execute_batch_rejects_invalid_elements_without_running_om(bad):
    adapter, runner = _prepared_adapter(batch=2)
    with pytest.raises(TypeError, match=r"payloads\[1\]"):
        adapter.execute_batch(["高血压", bad])
    assert runner.calls == 0


def test_execute_batch_rejects_batches_beyond_configured_limits():
    adapter, runner = _prepared_adapter(batch=2)
    adapter.max_documents = 1
    with pytest.raises(ValueError, match="documents"):
        adapter.execute_batch(["高血压", "糖尿病"])
    adapter.max_documents = 2
    adapter.max_total_characters = 3
    with pytest.raises(ValueError, match="characters"):
        adapter.execute_batch(["高血压患者"])
    adapter.max_total_characters = 100
    adapter.max_total_windows = 1
    with pytest.raises(ValueError, match="token windows"):
        adapter.execute_batch(["甲乙丙丁戊己庚辛"])
    assert runner.calls == 0


def test_adapter_requires_open():
    adapter = MedicalNerOmAdapter(omPath="fake.om", modelName="fake-model", seqLen=8, overlap=0)
    try:
        adapter.execute("高血压")
    except RuntimeError as exc:
        assert "not open" in str(exc)
    else:
        raise AssertionError("execute before open must raise")


def test_adapter_rejects_bad_geometry():
    for kwargs in ({"seqLen": 2}, {"seqLen": 8, "overlap": 6}):
        try:
            MedicalNerOmAdapter(omPath="fake.om", modelName="fake-model", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_adapter_close_releases_runner():
    adapter, runner = _prepared_adapter()
    adapter.close()
    assert runner.closed is True
    assert adapter._runner is None
