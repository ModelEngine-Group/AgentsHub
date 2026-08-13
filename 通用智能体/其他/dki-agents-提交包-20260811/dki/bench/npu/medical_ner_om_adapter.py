"""生产级医疗 NER 的 ACL/OM 适配器（text -> 实体列表）。

这是 Release A 的产品化缺口补齐：此前目标机只验证过“数值验收 adapter”
（固定 (64,256) 占位 tensor，返回 output sum/shape），只证明 process-owned ACL
生命周期正确，并未证明**完整医疗文本能经该 Runtime 得到正确实体**。

本模块把 `scripts/eval_om_ner.py` / `scripts/acl_om_pipeline_bench.py` 已验证的
    text -> tokenize -> OM execute -> argmax -> canonical BIO decode_typed
      -> 全局 span 还原 -> 词典归一 -> entities
收敛为一个实现 `open()/execute(payload)/close()` 的适配器，可直接作为
`ProcessOmRuntime` 的 `adapterFactory`，再由 `MedicalEntityExtractorOmShadow`
的 `shadowRuntimeFactory` 调用（`runtime.execute(text, timeout_s=...)`）。

关键约束：
- 所有重依赖（`acl`、`transformers`）在 `open()`/`execute()` 内惰性导入，使本模块
  在无 NPU 的 CPU 机上仍可被导入与打包（ZIP 依赖闭包友好）。
- OmManifest 在**加载 OM 之前**由 `enforce_preload_manifest` 强制消费（hash + SoC/CANN），
  加载之后再核对 IO 签名；任一失败直接 fail closed，绝不静默降级。
- 超过单窗口容量的长文本走 `dki.nlp.windowing` 的全局坐标窗口 + core ownership 合并，
  不静默截断。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from dki.bench.npu.om_manifest import OmCompatibilityError, enforce_preload_manifest
from dki.nlp.bio_decode import BioDecoder
from dki.nlp.windowing import merge_window_entities, plan_token_windows


class MedicalNerOmAdapter:
    """单文本医疗 NER 的 OM 适配器。

    payload 允许是 str（病历文本）或 {"text": str}；返回实体 dict 列表：
    ``[{"name","type","mention","span":[s,e],"source":"om","confidence"}]``。
    """

    def __init__(
        self,
        *,
        omPath: str,
        modelName: str,
        seqLen: int = 256,
        device: int = 0,
        overlap: int = 32,
        manifestPath: Optional[str] = None,
        tokenizerPath: Optional[str] = None,
        labelsPath: Optional[str] = None,
        socVersion: Optional[str] = None,
        cannVersion: Optional[str] = None,
        dictPath: Optional[str] = None,
        maxDocuments: int = 256,
        maxTotalCharacters: int = 2_000_000,
        maxTotalWindows: int = 8192,
        confidence: float = 0.8,
    ):
        if not omPath:
            raise ValueError("omPath is required")
        if seqLen <= 2:
            raise ValueError("seqLen must exceed the two special tokens")
        if overlap < 0 or overlap >= seqLen - 2:
            raise ValueError("overlap must satisfy 0 <= overlap < seqLen-2")
        self._config: Dict[str, Any] = {
            "manifestPath": manifestPath,
            "omPath": omPath,
            "tokenizerPath": tokenizerPath,
            "labelsPath": labelsPath,
            "socVersion": socVersion,
            "cannVersion": cannVersion,
        }
        self.om_path = Path(omPath)
        self.model_name = modelName
        self.seq_len = int(seqLen)
        self.capacity = self.seq_len - 2  # 预留 [CLS]/[SEP]
        self.device = int(device)
        self.overlap = int(overlap)
        self.dict_path = dictPath
        self.max_documents = int(maxDocuments)
        self.max_total_characters = int(maxTotalCharacters)
        self.max_total_windows = int(maxTotalWindows)
        if min(self.max_documents, self.max_total_characters, self.max_total_windows) <= 0:
            raise ValueError("batch limits must be positive")
        self.confidence = float(confidence)

        self._runner = None
        self._tokenizer = None
        self._decoder: Optional[BioDecoder] = None
        self._vocab: Dict[str, Any] = {}
        self._batch = 1
        self._num_labels = 0
        self._cls_id = 0
        self._sep_id = 0
        self._pad_id = 0
        self._opened = False

    # ---------------------------------------------------------------- open
    def open(self) -> None:
        if self._opened:
            return
        # 1) 加载前强制门：hash + SoC/CANN（无 manifestPath 时返回 None，不阻塞）。
        manifest = enforce_preload_manifest(self._config)

        # 2) 惰性导入重依赖（仅 NPU 机需要）。
        from transformers import AutoConfig, AutoTokenizer

        from dki.bench.npu.static_om_runner import StaticOmRunner

        tokenizer_source = str(Path(tokenizerPath).parent) if (tokenizerPath := self._config.get("tokenizerPath")) else self.model_name
        labels_source = str(Path(labelsPath).parent) if (labelsPath := self._config.get("labelsPath")) else self.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        model_config = AutoConfig.from_pretrained(labels_source)
        self._decoder = BioDecoder(model_config.id2label)
        self._cls_id = int(getattr(self._tokenizer, "cls_token_id", 0) or 0)
        self._sep_id = int(getattr(self._tokenizer, "sep_token_id", 0) or 0)
        self._pad_id = int(getattr(self._tokenizer, "pad_token_id", 0) or 0)

        from dki.operators.medical_entity_extractor.process import load_vocab

        self._vocab = load_vocab(self.dict_path)

        # 3) 加载 OM，读取真实静态 batch / num_labels。
        self._runner = StaticOmRunner(self.om_path, self.device)
        try:
            out_shape = tuple(int(x) for x in self._runner.output_shape)
            if len(out_shape) != 3:
                raise OmCompatibilityError(f"unexpected OM output rank: {out_shape}")
            self._batch, out_seq, self._num_labels = out_shape
            if out_seq != self.seq_len:
                raise OmCompatibilityError(
                    f"OM sequence dim {out_seq} != configured seqLen {self.seq_len}"
                )

            # 4) 加载后核对 IO 签名（补 enforce_preload_manifest 的 pre-load 检查）。
            if manifest is not None:
                self._verify_loaded_io(manifest)
        except BaseException:
            self._runner.close()
            self._runner = None
            raise
        self._opened = True

    def _verify_loaded_io(self, manifest) -> None:
        inputs = getattr(self._runner, "input_specs", None)
        outputs = getattr(self._runner, "output_specs", None)
        if inputs is None or outputs is None:
            raise OmCompatibilityError("ACL runner does not expose full loaded IO specs")
        manifest.verify_io(inputs=inputs, outputs=outputs)

    # ------------------------------------------------------------- execute
    def execute(self, payload: Any) -> List[Dict[str, Any]]:
        """向后兼容的单文档入口；实现统一委托给跨文档微批路径。"""
        return self.execute_batch([payload])[0]

    def execute_batch(self, payloads: List[Any]) -> List[List[Dict[str, Any]]]:
        """跨文档窗口微批推理，返回与 ``payloads`` 一一对应且同序的实体列表。

        所有非空文档先独立 tokenize/window，再将窗口按文档顺序扁平化并填入
        静态 OM batch。仅最后一次 execute 允许 tail padding；文档边界不触发填充。
        """
        if not self._opened or self._runner is None:
            raise RuntimeError("adapter is not open")
        if isinstance(payloads, (str, bytes, dict)) or payloads is None:
            raise TypeError("payloads must be a sequence of text payloads")
        payload_list = list(payloads)
        if len(payload_list) > self.max_documents:
            raise ValueError(
                f"batch has {len(payload_list)} documents, limit is {self.max_documents}"
            )
        results: List[List[Dict[str, Any]]] = [[] for _ in payload_list]
        if not payload_list:
            return results

        documents = []
        flat_windows = []
        total_characters = 0
        for doc_index, payload in enumerate(payload_list):
            if isinstance(payload, str):
                text = payload
            elif isinstance(payload, dict) and isinstance(payload.get("text"), str):
                text = payload["text"]
            else:
                raise TypeError(
                    f"payloads[{doc_index}] must be a string or mapping with string text"
                )
            total_characters += len(text)
            if total_characters > self.max_total_characters:
                raise ValueError(
                    f"batch text has {total_characters} characters, limit is "
                    f"{self.max_total_characters}"
                )
            if not text.strip():
                documents.append((text, [], []))
                continue
            enc = self._tokenizer(
                text,
                truncation=False,
                return_offsets_mapping=True,
                return_tensors="np",
                add_special_tokens=True,
            )
            ids_row = enc["input_ids"][0]
            offsets_row = enc["offset_mapping"][0]
            content_ids: List[int] = []
            content_offsets: List[List[int]] = []
            for token_id, (start, end) in zip(ids_row, offsets_row):
                if int(end) > int(start):
                    content_ids.append(int(token_id))
                    content_offsets.append([int(start), int(end)])
            documents.append((text, content_ids, content_offsets))
            if content_ids:
                windows = plan_token_windows(
                    content_offsets, capacity=self.capacity, overlap=self.overlap,
                )
                if len(flat_windows) + len(windows) > self.max_total_windows:
                    raise ValueError(
                        f"batch expands beyond {self.max_total_windows} token windows"
                    )
                flat_windows.extend((doc_index, window) for window in windows)

        per_document_windows = [[] for _ in payload_list]
        for chunk_start in range(0, len(flat_windows), self._batch):
            chunk = flat_windows[chunk_start:chunk_start + self._batch]
            arrays, row_meta = self._build_flat_batch(chunk, documents)
            logits = self._runner.execute(arrays)[0]
            n_elems = self._batch * self.seq_len * self._num_labels
            pred = logits[:n_elems].reshape(
                (self._batch, self.seq_len, self._num_labels)
            ).argmax(axis=-1)
            for row, (doc_index, window, span_offsets) in enumerate(row_meta):
                win_len = len(span_offsets)
                if win_len == 0:
                    continue
                text = documents[doc_index][0]
                row_pred = pred[row, 1:1 + win_len][np.newaxis, :]
                row_off = np.asarray(span_offsets, dtype=np.int64)[np.newaxis, :, :]
                typed = self._decoder.decode_typed(row_pred, row_off)[0]
                entities = []
                for ent in typed:
                    anchor = window.token_start + int(ent["b_col"])
                    entities.append({
                        "type": ent["type"],
                        "span": [int(ent["char_start"]), int(ent["char_end"])],
                        "anchor_token": anchor,
                        "mention": text[int(ent["char_start"]):int(ent["char_end"])],
                    })
                per_document_windows[doc_index].append((window, entities))

        for doc_index, window_results in enumerate(per_document_windows):
            text = documents[doc_index][0]
            merged = merge_window_entities(window_results)
            for item in merged:
                mention = item.get("mention") or text[item["span"][0]:item["span"][1]]
                canonical = self._vocab.get(mention, (mention, item["type"]))[0]
                results[doc_index].append({
                    "name": canonical,
                    "type": item["type"],
                    "mention": mention,
                    "span": [int(item["span"][0]), int(item["span"][1])],
                    "source": "om",
                    "confidence": self.confidence,
                })
        return results

    def _build_flat_batch(self, chunk, documents):
        arrays = {
            "input_ids": np.full((self._batch, self.seq_len), self._pad_id, dtype=np.int64),
            "attention_mask": np.zeros((self._batch, self.seq_len), dtype=np.int64),
            "token_type_ids": np.zeros((self._batch, self.seq_len), dtype=np.int64),
        }
        row_meta = []
        for row, (doc_index, window) in enumerate(chunk):
            _, content_ids, content_offsets = documents[doc_index]
            span_ids = content_ids[window.token_start:window.token_end]
            span_offsets = content_offsets[window.token_start:window.token_end]
            win_len = len(span_ids)
            arrays["input_ids"][row, 0] = self._cls_id
            arrays["input_ids"][row, 1:1 + win_len] = span_ids
            arrays["input_ids"][row, 1 + win_len] = self._sep_id
            arrays["attention_mask"][row, 0:2 + win_len] = 1
            row_meta.append((doc_index, window, span_offsets))
        return arrays, row_meta

    # --------------------------------------------------------------- close
    def close(self) -> None:
        self._opened = False
        runner, self._runner = self._runner, None
        if runner is not None:
            runner.close()


def create_adapter(**config) -> MedicalNerOmAdapter:
    """Dotted `adapterFactory`（供 ProcessOmRuntime 在 spawn 子进程构造）。"""
    return MedicalNerOmAdapter(**config)


def create_process_runtime(**config):
    """Dotted `shadowRuntimeFactory`：包一层进程独占 Runtime。

    `MedicalEntityExtractorOmShadow` 以 `runtime.execute(text, timeout_s=...)`
    调用返回对象；`ProcessOmRuntime.execute(payload, timeout_s=...)` 恰好匹配，
    并提供硬超时、熔断、重启≤1 与错误脱敏 IPC。
    """
    from dki.bench.npu.om_runtime import ProcessOmRuntime

    runtime_keys = {
        "queueDepth", "maxRestarts", "startTimeoutS", "mpStartMethod",
        "maxBatchItems", "maxPayloadBytes",
    }
    runtime_kwargs = {key: config.pop(key) for key in list(config) if key in runtime_keys}
    return ProcessOmRuntime(
        adapterFactory="dki.bench.npu.medical_ner_om_adapter.create_adapter",
        adapterConfig=config,
        **runtime_kwargs,
    )
