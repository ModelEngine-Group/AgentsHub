"""探针：确认 transformers 5.14.1 的 AutoModel 在 NPU 上是否给出
（a）随位置变化、（b）随输入变化 的隐状态，以及 GlobalPointer 头在
真实编码器上能否把极少量样本过拟合（positive logit 越过 0）。

在目标机跑：python3 scripts/_encoder_probe.py --device npu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu", choices=["npu", "cuda", "cpu"])
    ap.add_argument("--base-model", default="hfl/chinese-roberta-wwm-ext")
    args = ap.parse_args()

    import torch
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    from transformers import AutoModel, AutoTokenizer
    dev = torch.device(args.device if args.device != "npu" else "npu")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    enc = AutoModel.from_pretrained(args.base_model).to(dev).eval()

    texts = ["房室结消融和起搏器植入作为反复发作的替代疗法。",
             "患者出现发热、咳嗽、乏力等症状，考虑肺炎。"]
    b = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    b = {k: v.to(dev) for k, v in b.items()}
    with torch.no_grad():
        out = enc(**{k: v for k, v in b.items() if k in ("input_ids", "attention_mask", "token_type_ids")})
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    h = h.float().cpu()
    print("hidden shape", tuple(h.shape))
    # 位置间方差（同一句不同 token 是否不同）
    pos_std = h[0].std(dim=0).mean().item()
    # 句间差异（同一位置不同句）
    L = min(h.shape[1], 8)
    cross = (h[0, :L] - h[1, :L]).abs().mean().item()
    print(f"per-position std (mean over dims) = {pos_std:.4f}")
    print(f"cross-sentence abs diff (first {L} tokens) = {cross:.4f}")
    print("token0 vec[:6]", [round(v, 3) for v in h[0, 0, :6].tolist()])
    print("token1 vec[:6]", [round(v, 3) for v in h[0, 1, :6].tolist()])

    # 真实编码器 + GP 头，过拟合 2 句里的少量 span
    from dki.nlp.globalpointer import GlobalPointer, decode_spans, gp_loss
    heads, inner = 9, 64
    seq = h.shape[1]
    hidden_dim = h.shape[2]
    gp = GlobalPointer(hidden_dim, heads, inner).to(dev)
    # 造几个 span：句0 head0 (1,5),(7,9)；句1 head1 (2,4)
    import torch as T
    labels = T.zeros(2, heads, seq, seq)
    for (bi, hh, s, e) in [(0, 0, 1, 5), (0, 0, 7, 9), (1, 1, 2, 4)]:
        if s < seq and e < seq:
            labels[bi, hh, s, e] = 1.0
    labels = labels.to(dev)
    hid = h.to(dev)
    mask = b["attention_mask"].float()
    opt = T.optim.AdamW(gp.parameters(), lr=1e-3)
    for step in range(300):
        logits = gp(hid, mask)
        loss = gp_loss(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        lg = gp(hid, mask).float().cpu()
    pos = [float(lg[bi, hh, s, e]) for (bi, hh, s, e) in [(0, 0, 1, 5), (0, 0, 7, 9), (1, 1, 2, 4)]]
    print(f"[frozen-encoder + GP overfit] final loss={float(loss):.4f} pos_logits={[round(v,2) for v in pos]} global_max={float(lg.max()):.2f}")
    print("ENCODER_PROBE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
