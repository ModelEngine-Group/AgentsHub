"""实体识别算子的 NPU（昇腾 Ascend）优化实现。

优化思路：把「多模式字符串匹配」重写为可在 NPU 上高效执行的
稠密张量运算（batch 化 + 无分支 + 大规模并行）：

1. 文本批量编码为 Unicode 码点矩阵  T ∈ int32[B, L]（padding=0）
2. 词典编码为码点矩阵                P ∈ int32[N, K]（padding=0）
3. 对每个偏移 k∈[0,K)：比较 T[:, i+k] == P[:, k]（广播成 [B, N, L]），
   词条命中即所有非 padding 位置全部相等 —— 一次张量运算完成
   全部 文档×词条×位置 的匹配判定，天然适配 NPU 的向量核（Vector Core）。

设备选择：
- 存在 torch_npu（CANN 环境）时走 NPU 设备执行；
- 存在 torch 时走 torch CPU（同一套张量代码，便于对比）；
- 否则用 NumPy 模拟同样的并行算法（本仓库 CI/无 NPU 环境下的可复现基线）。

与 CPU Aho-Corasick 的取舍分析见 docs/npu-optimization.md。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import Operator, OperatorMetadata
from ..entity_extract import load_dictionary, _resolve_overlaps
from ...kg.normalize import EntityNormalizer
from ...kg.schema import Entity

try:  # CANN / 昇腾环境
    import torch

    try:
        import torch_npu  # noqa: F401

        _DEVICE = "npu"
    except ImportError:
        _DEVICE = "cpu"
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    _DEVICE = "numpy"


def _encode(texts: list[str], max_len: int) -> np.ndarray:
    mat = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        codes = [ord(c) for c in t[:max_len]]
        mat[i, : len(codes)] = codes
    return mat


class EntityExtractNPUOperator(Operator):
    """张量化多模式匹配 NER 算子（NPU/torch/NumPy 三级后端）。"""

    metadata = OperatorMetadata(
        name="entity_extract_npu",
        description="医疗实体识别（NPU 优化版）：张量化多模式匹配，批量并行",
        op_type="mapper",
        inputs={"batch": "list[str] 文档列表"},
        output_type="list[list[Entity]]",
    )

    def __init__(self, dictionary: dict[str, str] | None = None,
                 normalize: bool = True) -> None:
        self.term2type = dictionary or load_dictionary()
        if normalize:
            # 与 CPU 算子对齐：同义词别名也加入匹配词典，保证两后端结果一致
            canon2type = dict(self.term2type)
            for alias, canon in EntityNormalizer().canonical.items():
                etype = canon2type.get(canon)
                if etype and alias not in self.term2type:
                    self.term2type[alias] = etype
        self.terms = list(self.term2type)
        self.max_term_len = max(len(t) for t in self.terms)
        self.pattern_mat = _encode(self.terms, self.max_term_len)  # [N, K]
        self.term_lens = np.array([len(t) for t in self.terms], dtype=np.int32)
        self.backend = _DEVICE

    def process(self, batch: list[str], **kwargs: Any) -> list[list[Entity]]:
        max_len = max((len(t) for t in batch), default=1)
        text_mat = _encode(batch, max_len)  # [B, L]
        if _HAS_TORCH:
            hits = self._match_torch(text_mat)
        else:
            hits = self._match_numpy(text_mat)
        return self._decode(hits, batch)  # hits: [M, 3] 稀疏命中三元组 (b, n, l)

    # ---- 后端实现 -------------------------------------------------------

    def _match_numpy(self, text_mat: np.ndarray) -> np.ndarray:
        """返回稀疏命中索引 [M, 3]（词条 n 在文档 b 的位置 l 起始命中）。"""
        B, L = text_mat.shape
        N, K = self.pattern_mat.shape
        ok = np.ones((B, N, L), dtype=bool)
        for k in range(K):
            # 词条第 k 个字符与文本第 l+k 个字符比较
            shifted = np.zeros((B, L), dtype=np.int32)
            if L > k:
                shifted[:, : L - k] = text_mat[:, k:]
            pk = self.pattern_mat[:, k]  # [N]
            eq = shifted[:, None, :] == pk[None, :, None]  # [B, N, L]
            active = (self.term_lens > k)[None, :, None]  # padding 位置视为命中
            ok &= eq | ~active
        return np.argwhere(ok)

    # 单块中间张量预算上限（bool 元素数），超出时沿批维分块以控制峰值显存
    _CHUNK_ELEMS = 256 * 1024 * 1024

    def _match_torch(self, text_mat: np.ndarray) -> np.ndarray:
        device = _DEVICE if _DEVICE == "npu" else "cpu"
        P = torch.from_numpy(self.pattern_mat).to(device)
        lens = torch.from_numpy(self.term_lens).to(device)
        B, L = text_mat.shape
        N, K = P.shape
        chunk = max(1, self._CHUNK_ELEMS // (N * L))
        parts: list[np.ndarray] = []
        for off in range(0, B, chunk):
            T = torch.from_numpy(text_mat[off:off + chunk]).to(device)
            Bc = T.shape[0]
            ok = torch.ones((Bc, N, L), dtype=torch.bool, device=device)
            for k in range(K):
                shifted = torch.zeros((Bc, L), dtype=T.dtype, device=device)
                if L > k:
                    shifted[:, : L - k] = T[:, k:]
                eq = shifted.unsqueeze(1) == P[:, k].view(1, N, 1)
                active = (lens > k).view(1, N, 1)
                ok &= eq | ~active
            # 设备端稀疏化：只回传命中索引 [M, 3]，避免搬运整块布尔张量
            idx = torch.nonzero(ok).cpu().numpy()
            idx[:, 0] += off
            parts.append(idx)
        return np.concatenate(parts) if parts else np.empty((0, 3), dtype=np.int64)

    def _decode(self, hits: np.ndarray, batch: list[str]) -> list[list[Entity]]:
        if len(hits) == 0:
            return [[] for _ in batch]
        # 向量化过滤 + 按文档分组，避免逐命中 Python 循环
        b, n, start = hits[:, 0], hits[:, 1], hits[:, 2]
        end = start + self.term_lens[n]
        doc_lens = np.fromiter((len(t) for t in batch), dtype=np.int64, count=len(batch))
        keep = end <= doc_lens[b]
        b, n, start, end = b[keep], n[keep], start[keep], end[keep]
        bounds = np.searchsorted(b, np.arange(len(batch) + 1))
        results: list[list[Entity]] = []
        for d in range(len(batch)):
            lo, hi = bounds[d], bounds[d + 1]
            doc_hits = []
            for i in range(lo, hi):
                term = self.terms[n[i]]
                doc_hits.append((int(start[i]), int(end[i]), term, self.term2type[term]))
            kept = _resolve_overlaps(doc_hits)
            results.append([Entity(text=w, etype=t, start=s, end=e) for s, e, w, t in kept])
        return results
