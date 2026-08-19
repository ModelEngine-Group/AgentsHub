"""GlobalPointer / Efficient GlobalPointer 嵌套 NER 头 + 多标签分类交叉熵 + span 解码。

用于 CMeEE-V2 这类含**嵌套实体**的中文医疗 NER：把 NER 视为对 (start, end) token
对的打分（带 RoPE），天然支持嵌套/重叠。Efficient 版共享抽取、分离类型分类，
在多类型难任务（含 CMeEE）上通常优于原版且参数更少。

纯 torch 实现；torch 在模块内使用（调用方在 NPU/GPU 机上 import）。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def _apply_span_masks(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pad / lower-triangular / CLS / SEP masking on (b, heads, seq, seq) logits."""
    _, _, seq, _ = logits.shape
    positions = torch.arange(seq, device=logits.device)
    lengths = mask.long().sum(dim=-1).clamp_min(1)
    special = positions[None, :] == 0
    special = special | (positions[None, :] == (lengths - 1)[:, None])
    valid_token = mask.bool() & ~special
    valid_span = (
        valid_token[:, None, :, None]
        & valid_token[:, None, None, :]
        & (positions[:, None] <= positions[None, :])[None, None, :, :]
    )
    return logits.float().masked_fill(~valid_span, -1e12)


def _rope_rotate(qw: torch.Tensor, kw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE for classic GP Q/K with shape (batch, seq, heads, dim).

    Cos/sin must broadcast as (1, seq, 1, dim) — NOT (1, 1, seq, dim).
    A prior refactor used the latter and silently destroyed classic checkpoints
    during ensemble/rescore (~0.26 F1). Efficient GP uses a separate path.
    """
    if qw.dim() != 4:
        raise ValueError(f"classic RoPE expects 4D (b,seq,heads,dim), got {tuple(qw.shape)}")
    _, seq, _, dim = qw.shape
    device, dtype = qw.device, qw.dtype
    idx = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    theta = 1.0 / (10000 ** (idx / dim))
    pos = torch.arange(seq, dtype=torch.float32, device=device)
    freqs = torch.einsum("n,d->nd", pos, theta)
    emb = torch.stack([freqs, freqs], dim=-1).reshape(seq, dim).to(dtype)
    cos = emb.cos()[None, :, None, :]
    sin = emb.sin()[None, :, None, :]

    def rot(t: torch.Tensor) -> torch.Tensor:
        t2 = torch.stack([-t[..., 1::2], t[..., ::2]], dim=-1).reshape_as(t)
        return t * cos + t2 * sin

    return rot(qw), rot(kw)


def _rope_rotate_efficient(qw: torch.Tensor, kw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE matching community Efficient-GlobalPointer-torch (sin/cos interleave)."""
    seq, dim = qw.shape[-2], qw.shape[-1]
    device, dtype = qw.device, qw.dtype
    position_ids = torch.arange(seq, dtype=torch.float32, device=device)[None]
    indices = torch.arange(dim // 2, dtype=torch.float32, device=device)
    indices = torch.pow(10000.0, -2 * indices / dim)
    embeddings = torch.einsum("bn,d->bnd", position_ids, indices)
    embeddings = torch.stack([embeddings.sin(), embeddings.cos()], dim=-1).reshape(1, seq, dim)
    embeddings = embeddings.to(dtype)
    # Broadcast to (b, seq, dim)
    while embeddings.dim() < qw.dim():
        embeddings = embeddings.unsqueeze(0)
    embeddings = embeddings.expand_as(qw)
    cos_pos = embeddings[..., 1::2].repeat_interleave(2, dim=-1)
    sin_pos = embeddings[..., ::2].repeat_interleave(2, dim=-1)

    def rot(t: torch.Tensor) -> torch.Tensor:
        t2 = torch.stack([-t[..., 1::2], t[..., ::2]], dim=-1).reshape_as(t)
        return t * cos_pos + t2 * sin_pos

    return rot(qw), rot(kw)


class GlobalPointer(nn.Module):
    """原版：每类型独立 Q/K -> (batch, heads, seq, seq)。"""

    def __init__(self, hidden_size: int, heads: int, inner_dim: int = 64, rope: bool = True):
        super().__init__()
        self.heads = heads
        self.inner_dim = inner_dim
        self.rope = rope
        self.dense = nn.Linear(hidden_size, heads * inner_dim * 2)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, seq, _ = hidden.shape
        x = self.dense(hidden).view(b, seq, self.heads, self.inner_dim * 2)
        qw, kw = x[..., : self.inner_dim], x[..., self.inner_dim :]
        if self.rope:
            qw, kw = _rope_rotate(qw, kw)
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw) / (self.inner_dim ** 0.5)
        return _apply_span_masks(logits, mask)


class EfficientGlobalPointer(nn.Module):
    """Efficient GlobalPointer：共享抽取 + 类型偏置（苏剑林 / archives/8877）。"""

    def __init__(self, hidden_size: int, heads: int, inner_dim: int = 64, rope: bool = True):
        super().__init__()
        self.heads = heads
        self.inner_dim = inner_dim
        self.rope = rope
        self.dense_1 = nn.Linear(hidden_size, inner_dim * 2)
        self.dense_2 = nn.Linear(hidden_size, heads * 2)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs = self.dense_1(hidden)
        qw, kw = outputs[..., ::2], outputs[..., 1::2]
        if self.rope:
            qw, kw = _rope_rotate_efficient(qw, kw)
        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / (self.inner_dim ** 0.5)
        bias = torch.einsum("bnh->bhn", self.dense_2(hidden)) / 2.0
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        return _apply_span_masks(logits, mask)


def multilabel_categorical_crossentropy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """苏剑林多标签分类交叉熵。y_*: (n, seq*seq)；可选 per-row 权重 (n,)。"""
    y_true = y_true.float()
    y_pred = y_pred.float()
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg = torch.logsumexp(y_pred_neg, dim=-1)
    pos = torch.logsumexp(y_pred_pos, dim=-1)
    loss = neg + pos
    if sample_weight is None:
        return loss.mean()
    weight = sample_weight.float().reshape_as(loss)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def gp_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    type_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """logits/labels: (b, heads, seq, seq). type_weights: (heads,) optional."""
    b, h, _, _ = logits.shape
    flat_true = labels.reshape(b * h, -1)
    flat_pred = logits.reshape(b * h, -1)
    if type_weights is None:
        return multilabel_categorical_crossentropy(flat_true, flat_pred)
    weights = type_weights.to(device=logits.device, dtype=torch.float32)
    row_weight = weights[None, :].expand(b, h).reshape(b * h)
    return multilabel_categorical_crossentropy(flat_true, flat_pred, row_weight)


def decode_spans(logits: torch.Tensor, threshold: float = 0.0) -> List[List[Tuple[int, int, int]]]:
    """(b, heads, seq, seq) -> 每样本 [(type_id, start_tok, end_tok)]。"""
    out: List[List[Tuple[int, int, int]]] = []
    for i in range(logits.shape[0]):
        spans = []
        idx = (logits[i] > threshold).nonzero(as_tuple=False)
        for h, s, e in idx.tolist():
            spans.append((int(h), int(s), int(e)))
        out.append(spans)
    return out


def build_pointer(kind: str, hidden_size: int, num_types: int, inner_dim: int = 64) -> nn.Module:
    kind = (kind or "classic").lower()
    if kind in {"classic", "globalpointer", "gp"}:
        return GlobalPointer(hidden_size, num_types, inner_dim)
    if kind in {"efficient", "egp", "efficient_globalpointer"}:
        return EfficientGlobalPointer(hidden_size, num_types, inner_dim)
    raise ValueError(f"unknown pointer kind: {kind}")


class EncoderGlobalPointer(nn.Module):
    """RoBERTa/BERT 编码器 + 可选 BiLSTM + SoftLexicon + GlobalPointer / EGP（<1B）。"""

    def __init__(
        self,
        encoder,
        hidden_size: int,
        num_types: int,
        inner_dim: int = 64,
        pointer: str = "classic",
        softlex_dim: int = 0,
        softlex_mode: str = "gated",
        use_bilstm: bool = False,
        bilstm_layers: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.pointer_kind = pointer
        self.softlex_dim = int(softlex_dim)
        self.softlex_mode = (softlex_mode or "gated").lower()
        self.use_bilstm = bool(use_bilstm)
        self.bilstm = None
        gp_in = hidden_size
        if self.use_bilstm:
            self.bilstm = nn.LSTM(
                hidden_size,
                hidden_size // 2,
                num_layers=max(1, int(bilstm_layers)),
                batch_first=True,
                bidirectional=True,
            )
            gp_in = hidden_size  # biLSTM hidden*2 with hidden//2 each side == hidden_size
        self.gp = build_pointer(pointer, gp_in, num_types, inner_dim)
        self.softlex_proj = None
        self.softlex_gate = None
        if self.softlex_dim > 0:
            self.softlex_proj = nn.Linear(self.softlex_dim, gp_in)
            # Zero-init so SoftLex starts as a no-op and cannot drown encoder features.
            nn.init.zeros_(self.softlex_proj.weight)
            nn.init.zeros_(self.softlex_proj.bias)
            if self.softlex_mode == "gated":
                self.softlex_gate = nn.Parameter(torch.zeros(1))
            elif self.softlex_mode != "add":
                raise ValueError(f"unknown softlex_mode: {softlex_mode}")

    def forward(self, input_ids, attention_mask, token_type_ids=None, softlex=None):
        kwargs = {"attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(input_ids, **kwargs)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if self.bilstm is not None:
            hidden, _ = self.bilstm(hidden)
            hidden = hidden * attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        if self.softlex_proj is not None:
            if softlex is None:
                softlex = hidden.new_zeros(hidden.size(0), hidden.size(1), self.softlex_dim)
            delta = self.softlex_proj(softlex.to(hidden.dtype))
            if self.softlex_gate is not None:
                delta = self.softlex_gate * delta
            hidden = hidden + delta
        return self.gp(hidden, attention_mask)


class FGM:
    """Fast Gradient Method on word embeddings."""

    def __init__(self, model: nn.Module, epsilon: float = 0.5):
        self.model = model
        self.epsilon = float(epsilon)
        self._backup = {}

    def attack(self) -> bool:
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if "word_embeddings" not in name:
                continue
            norm = param.grad.norm()
            if not torch.isfinite(norm) or float(norm) < 1e-12:
                continue
            self._backup[name] = param.data.clone()
            param.data.add_(self.epsilon * param.grad / norm)
            return True
        return False

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()
