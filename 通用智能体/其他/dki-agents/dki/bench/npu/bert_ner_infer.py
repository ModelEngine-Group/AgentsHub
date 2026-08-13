"""BertNerBackend —— Transformer NER 推理后端（MedicalEntityExtractor 的 bert 后端）。

昇腾 NPU 工程适配要点（对应 docs/npu_optimization.md 的优化项）：
1. torch_npu 延迟导入 + transfer_to_npu 自动迁移；
2. 静态 shape 分桶 padding：把变长文本 pad 到固定桶（64/128/256），
   避免动态 shape 触发反复图编译（昇腾图编译成本高，静态化后编译缓存可复用）；
3. 二进制算子模式：torch.npu.set_compile_mode(jit_compile=False)，
   使用预编译二进制算子库，消除首次编译开销；
4. 批处理：batch 维度同样固定，warmup 阶段对全部 (bucket, batch) 组合预热；
   分桶模式下尾批会补齐到固定 batch，再切回真实样本结果；
5. 推理上下文瘦身：缓存 torch device，并使用 torch.inference_mode()；
6. CPU 侧 fast tokenizer 并行：NPU/GPU 推理时默认开启 TOKENIZERS_PARALLELISM，
   避免评测环境继承关闭并行的变量后让 tokenization 成为瓶颈；
7. 混合精度：half=True 时 model.half()，910B 上 FP16 吞吐显著高于 FP32。

与 DataMate 的衔接：算子 metadata.yml 声明 runtime.npu 资源 → 平台
RayDataset._run_single_op 以 resources={"npu": n} 调度 → base_op.use_npu()
判定 accelerator=="npu" 且 torch_npu 可用 → 本后端 device="npu"。
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List

SEQ_BUCKETS = [64, 128, 256]

# 微调医疗 NER 模型的默认位置（scripts/train_ner.py 的产物）。存在则优先加载，
# 其 config.id2label 为真实 B-Disease/I-Symptom… 13 标签；不存在则回退 bert-base-chinese。
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FINETUNED_DIR = _PROJECT_ROOT / "models" / "medical_ner_bert"

# 本项目 6 类医疗实体（与 dki.nlp.labels.ENTITY_ORDER 一致）。微调模型输出的
# tag 已是这些真实类型，直接采用；未微调占位模型输出通用 NER tag，走下方映射兜底。
_KNOWN_TYPES = {"Disease", "Symptom", "Drug", "Exam", "Treatment", "Department"}

# 占位类型映射：仅未微调 bert-base-chinese 基线（通用 NER 标签，医疗类型无意义，
# 只用于吞吐基线/链路验证）时兜底。微调后此映射不再生效。
_LABEL_TYPE_MAP = {"ORG": "Department", "PER": "Disease", "LOC": "Symptom", "MISC": "Drug"}


def _resolve_entity_type(tag: str) -> str:
    """真实医疗类型优先，占位通用 NER 标签兜底（向后兼容未微调基线）。"""
    if tag in _KNOWN_TYPES:
        return tag
    return _LABEL_TYPE_MAP.get(tag, "Disease")

_INSTALL_HINT = (
    "bert 后端需要 torch 与 transformers：\n"
    "  CPU:  pip install torch transformers\n"
    "  NPU:  按 CANN 版本安装 torch + torch_npu（如 torch==2.7.1 + torch_npu==2.7.1.post2），"
    "并 source /usr/local/Ascend/.../set_env.sh"
)


class BertNerBackend:
    def __init__(self, model_name: str = None, device: str = "cpu",
                 half: bool = False, batch_size: int = 8, bucketing: bool = True,
                 quality_recall: bool = False):
        """bucketing=False 为消融模式：按批内最长动态 padding（动态 shape），
        用于量化"静态分桶"优化项本身的收益（NPU 上动态 shape 会反复触发图编译）。

        model_name=None 时自动探测：优先用微调医疗 NER（DEFAULT_FINETUNED_DIR，
        真实 13 标签），否则回退 bert-base-chinese（未微调占位，仅吞吐基线）。
        显式传入 model_name 则以其为准。"""
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"{_INSTALL_HINT}\n原始错误: {e}")

        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        if model_name is None:
            if DEFAULT_FINETUNED_DIR.exists() and (DEFAULT_FINETUNED_DIR / "config.json").exists():
                model_name = str(DEFAULT_FINETUNED_DIR)
                self.finetuned = True
            else:
                model_name = "bert-base-chinese"
                self.finetuned = False
        else:
            self.finetuned = Path(model_name).exists()
        self.model_name = model_name
        self.quality_mode = Path(model_name).name.startswith("medical_ner_quality")
        self.quality_recall = self.quality_mode and quality_recall
        self._baseline_backend = None

        self.device = device
        self.batch_size = batch_size
        self.bucketing = bucketing
        if device != "cpu":
            # 保留用户显式配置；无配置时避免 CPU 侧 tokenizer 线程并行被关闭。
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        if device == "npu":
            try:
                import torch_npu  # noqa: F401
                from torch_npu.contrib import transfer_to_npu  # noqa: F401
                # 二进制算子模式：避免在线编译（昇腾优化点 3）
                torch.npu.set_compile_mode(jit_compile=False)
            except ImportError as e:
                raise RuntimeError(f"NPU 模式需要 torch_npu（CANN 环境）。{_INSTALL_HINT}\n原始错误: {e}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.eval()
        if half and device != "cpu":
            self.model = self.model.half()
        self._torch_device_name = self._resolve_torch_device()
        self.model = self.model.to(self._torch_device_name)
        self._torch = torch
        # 微调模型：可选把 surface 提及归一到词典规范名（与 dict/gold 口径一致，
        # 便于实体级评测）。词典缺失或加载失败则跳过，直接用 surface。
        self._canon_vocab = None
        if self.finetuned:
            try:
                from dki.operators.medical_entity_extractor.process import load_vocab
                self._canon_vocab = load_vocab()
            except Exception:
                self._canon_vocab = None

    def _resolve_torch_device(self):
        import torch

        if self.device == "npu":
            return "npu:0"
        if self.device == "cuda" and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    @staticmethod
    def _bucket_len(n: int) -> int:
        """静态 shape 分桶（昇腾优化点 2）。"""
        for b in SEQ_BUCKETS:
            if n <= b:
                return b
        return SEQ_BUCKETS[-1]

    def warmup(self, iters: int = 3, sample_texts: List[str] = None):
        """预热触发编译缓存（昇腾优化点 4）。

        - 分桶模式：对全部 seq bucket 预热，此后计时期零编译；
        - 动态模式（消融）：仅做一次模型加载级预热，计时期的形状变化
          会持续触发编译——这正是消融要量化的开销。
        """
        if self.bucketing:
            for bucket in SEQ_BUCKETS:
                dummy = ["测" * (bucket - 2)] * self.batch_size
                for _ in range(iters):
                    self._forward(dummy, bucket, return_offsets=False)
        else:
            dummy = (sample_texts or ["测" * 62])[: self.batch_size]
            self._forward(dummy, None, return_offsets=False)

    def _encode(self, texts: List[str], bucket, return_offsets: bool):
        if self.bucketing and bucket is not None:
            kwargs = dict(
                padding="max_length",   # pad 到桶上限：静态 shape
                max_length=bucket,
            )
        else:
            kwargs = dict(
                padding="longest",      # 消融：按批内最长动态 padding
                max_length=SEQ_BUCKETS[-1],
            )

        try:
            enc = self.tokenizer(
                texts,
                truncation=True,
                return_offsets_mapping=return_offsets,
                return_tensors="pt",
                **kwargs,
            )
        except NotImplementedError:
            # 慢 tokenizer 不支持 offset mapping。保留兼容路径，decode 再按单条文本回退。
            enc = self.tokenizer(
                texts,
                truncation=True,
                return_tensors="pt",
                **kwargs,
            )
        offsets = enc.pop("offset_mapping", None)
        return enc, offsets

    def _forward(self, texts: List[str], bucket, return_offsets: bool = True):
        enc, offsets = self._encode(texts, bucket, return_offsets)
        enc = {k: v.to(self._torch_device_name) for k, v in enc.items()}
        with self._torch.inference_mode():
            out = self.model(**enc)
        return out.logits, offsets

    def extract(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """批量 NER：返回每条文本的实体列表 [{name,type,span,source,confidence}]。"""
        if not texts:
            return []

        if self.quality_recall:
            # 可选高召回两阶段模式。先跑 quality 主模型，再用冻结原 BERT 补互补实体；
            # 独立 backend 避免递归启用 quality 规则，不影响任何默认性能路径。
            self.quality_recall = False
            try:
                primary = self.extract(texts)
            finally:
                self.quality_recall = True
            if self._baseline_backend is None:
                self._baseline_backend = BertNerBackend(
                    model_name=str(DEFAULT_FINETUNED_DIR),
                    device=self.device,
                    half=self.model.dtype == self._torch.float16,
                    batch_size=self.batch_size,
                    bucketing=self.bucketing,
                )
            baseline = self._baseline_backend.extract(texts)
            from dki.nlp.quality_rules import merge_quality_recall

            return [
                merge_quality_recall(text, q_entities, b_entities)
                for text, q_entities, b_entities in zip(texts, primary, baseline)
            ]

        results: List[List[Dict[str, Any]]] = [[] for _ in texts]
        if self.bucketing:
            indexed_batches = []
            by_bucket: Dict[int, List[tuple[int, str]]] = {b: [] for b in SEQ_BUCKETS}
            for idx, text in enumerate(texts):
                by_bucket[self._bucket_len(len(text) + 2)].append((idx, text))
            for bucket in SEQ_BUCKETS:
                bucket_items = by_bucket[bucket]
                for i in range(0, len(bucket_items), self.batch_size):
                    indexed_batches.append((bucket, bucket_items[i: i + self.batch_size]))
        else:
            indexed_batches = [
                (None, list(enumerate(texts))[i: i + self.batch_size])
                for i in range(0, len(texts), self.batch_size)
            ]

        for bucket, indexed_batch in indexed_batches:
            indices = [idx for idx, _ in indexed_batch]
            batch = [text for _, text in indexed_batch]
            real_batch_size = len(batch)
            if self.bucketing and real_batch_size < self.batch_size:
                batch = batch + [batch[-1]] * (self.batch_size - real_batch_size)

            logits, offsets = self._forward(batch, bucket, return_offsets=False)
            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            offsets_by_text = offsets.tolist() if offsets is not None else None
            for j, idx in enumerate(indices):
                text = texts[idx]
                one_offsets = offsets_by_text[j] if offsets_by_text is not None else None
                decoded = self._decode(text, one_offsets, pred_ids[j])
                if self.quality_mode:
                    from dki.nlp.quality_rules import calibrate_quality_entities

                    decoded = calibrate_quality_entities(text, decoded)
                results[idx] = decoded
        return results

    def _decode(self, text: str, offsets, pred_ids: List[int]) -> List[Dict[str, Any]]:
        """BIO 解码（配合 offset 映射还原 span）。"""
        id2label = self.model.config.id2label
        if offsets is None:
            try:
                offsets = self.tokenizer(
                    text, truncation=True, max_length=len(pred_ids),
                    return_offsets_mapping=True,
                )["offset_mapping"]
            except Exception:
                offsets = None
        ents = []
        cur = None
        for k, pid in enumerate(pred_ids):
            label = str(id2label.get(pid, "O"))
            if offsets is None or k >= len(offsets):
                break
            start, end = offsets[k]
            if start == end:  # special token
                continue
            if label.startswith("B-"):
                if cur:
                    ents.append(cur)
                cur = {"s": start, "e": end, "tag": label[2:]}
            elif label.startswith("I-") and cur and cur["tag"] == label[2:]:
                cur["e"] = end
            else:
                if cur:
                    ents.append(cur)
                cur = None
        if cur:
            ents.append(cur)
        return [
            self._make_entity(text, e)
            for e in ents
        ]

    def _make_entity(self, text: str, e: Dict[str, Any]) -> Dict[str, Any]:
        surface = text[e["s"]:e["e"]]
        etype = _resolve_entity_type(e["tag"])
        # 微调模型：把 surface 归一到词典规范名（与 dict/gold 口径一致）
        name = surface
        if self._canon_vocab is not None:
            hit = self._canon_vocab.get(surface)
            if hit is not None:
                name = hit[0]
        return {
            "name": name,
            "mention": surface,
            "type": etype,
            "span": [e["s"], e["e"]],
            "source": "bert",
            "confidence": 0.8,
        }

    # ------------------------------------------------------ 基准接口
    def benchmark(self, texts: List[str], iters: int = 20, warmup_iters: int = 3) -> Dict[str, Any]:
        """吞吐/延迟基准：warmup 排除（分桶模式的）编译时间后计时。"""
        t0 = time.perf_counter()
        self.warmup(iters=warmup_iters, sample_texts=texts)
        warmup_s = time.perf_counter() - t0

        lat_ms = []
        n_samples = 0
        t_start = time.perf_counter()
        for _ in range(iters):
            t1 = time.perf_counter()
            self.extract(texts)
            lat_ms.append((time.perf_counter() - t1) * 1000)
            n_samples += len(texts)
        total_s = time.perf_counter() - t_start
        lat_sorted = sorted(lat_ms)
        return {
            "device": self.device,
            "bucketing": self.bucketing,
            "batch_size": self.batch_size,
            "n_texts_per_iter": len(texts),
            "iters": iters,
            "warmup_s": round(warmup_s, 3),
            "throughput_samples_s": round(n_samples / total_s, 2),
            "batch_latency_p50_ms": round(lat_sorted[len(lat_sorted) // 2], 2),
            "batch_latency_p95_ms": round(lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))], 2),
        }
