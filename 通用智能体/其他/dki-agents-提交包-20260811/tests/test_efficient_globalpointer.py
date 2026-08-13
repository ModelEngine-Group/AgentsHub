"""CPU unit tests for Efficient GlobalPointer shapes and finite loss."""
from __future__ import annotations

import torch

from dki.nlp.globalpointer import (
    EfficientGlobalPointer,
    EncoderGlobalPointer,
    GlobalPointer,
    gp_loss,
)


def test_efficient_gp_shape_and_finite_loss():
    torch.manual_seed(0)
    b, seq, h, heads, inner = 2, 16, 32, 9, 8
    hidden = torch.randn(b, seq, h)
    mask = torch.ones(b, seq)
    mask[:, -2:] = 0
    gp = EfficientGlobalPointer(h, heads, inner)
    logits = gp(hidden, mask)
    assert logits.shape == (b, heads, seq, seq)
    labels = torch.zeros_like(logits)
    labels[0, 0, 1, 3] = 1.0
    labels[1, 2, 2, 5] = 1.0
    loss = gp_loss(logits, labels)
    assert torch.isfinite(loss)
    weights = torch.linspace(0.5, 1.5, heads)
    loss_w = gp_loss(logits, labels, weights)
    assert torch.isfinite(loss_w)


def test_classic_gp_still_works():
    torch.manual_seed(0)
    gp = GlobalPointer(24, 3, 8)
    hidden = torch.randn(2, 10, 24)
    mask = torch.ones(2, 10)
    logits = gp(hidden, mask)
    assert logits.shape == (2, 3, 10, 10)
    # RoPE broadcast must keep head dim distinct from seq dim.
    assert torch.isfinite(logits).all()


def test_classic_rope_broadcast_shape():
    """Regression: cos/sin must be (1,seq,1,dim) for (b,seq,heads,dim)."""
    torch.manual_seed(0)
    from dki.nlp.globalpointer import _rope_rotate

    qw = torch.randn(2, 12, 9, 8)
    kw = torch.randn(2, 12, 9, 8)
    out_q, out_k = _rope_rotate(qw, kw)
    assert out_q.shape == qw.shape
    assert out_k.shape == kw.shape
    # Different heads must not receive identical rotated vectors solely due to bad broadcast.
    assert not torch.allclose(out_q[:, :, 0, :], out_q[:, :, 1, :])


def test_encoder_wrapper_pointer_switch():
    class _Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 16)
            self.config = type("C", (), {"hidden_size": 16})()

        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            x = self.linear(input_ids.float().unsqueeze(-1).expand(-1, -1, 8))
            return type("O", (), {"last_hidden_state": x})()

    model = EncoderGlobalPointer(_Enc(), 16, 4, 8, pointer="efficient")
    ids = torch.randint(0, 5, (2, 6))
    mask = torch.ones(2, 6)
    out = model(ids, mask)
    assert out.shape == (2, 4, 6, 6)
