"""GlobalPointer 单元诊断：脱离真实数据，构造可控张量，验证
损失/解码/RoPE 是否自洽、能否把少量 span 过拟合到解码可恢复。

在目标机跑：python3 scripts/_gp_unit_diag.py --device npu
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
    args = ap.parse_args()

    import torch
    if args.device == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    dev = torch.device(args.device if args.device != "npu" else "npu")

    from dki.nlp.globalpointer import GlobalPointer, decode_spans, gp_loss

    torch.manual_seed(0)
    b, seq, hidden, heads, inner = 4, 24, 64, 3, 32
    # 直接学一个从固定 hidden 到固定 span 的映射（过拟合能力测试）
    hidden_state = torch.randn(b, seq, hidden, device=dev)
    mask = torch.ones(b, seq, device=dev)
    # 每个样本放 2 个 span，类型/位置固定
    gold = [[(0, 2, 4), (1, 6, 9)],
            [(2, 1, 3), (0, 10, 14)],
            [(1, 0, 0), (2, 5, 5)],
            [(0, 3, 7), (1, 8, 8)]]
    labels = torch.zeros(b, heads, seq, seq, device=dev)
    for bi, spans in enumerate(gold):
        for (h, s, e) in spans:
            labels[bi, h, s, e] = 1.0

    gp = GlobalPointer(hidden, heads, inner).to(dev)
    opt = torch.optim.AdamW(gp.parameters(), lr=1e-3)
    for step in range(400):
        logits = gp(hidden_state, mask)
        loss = gp_loss(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == 399:
            with torch.no_grad():
                pred = decode_spans(gp(hidden_state, mask).float().cpu(), threshold=0.0)
                got = sum(len(set(p) & set(gold[i])) for i, p in enumerate(pred))
                total_pred = sum(len(p) for p in pred)
                total_gold = sum(len(g) for g in gold)
            print(f"step {step:3d} loss={float(loss):.4f} pred={total_pred} "
                  f"correct={got}/{total_gold}", flush=True)

    ok = got == total_gold and total_pred == total_gold
    print("UNIT_OVERFIT", "OK" if ok else "FAIL",
          f"(correct={got}/{total_gold}, pred={total_pred})")
    # 额外：打印最终正/负位置的 logit 分布，看阈值 0 是否分得开
    with torch.no_grad():
        lg = gp(hidden_state, mask)
        pos_vals, neg_vals = [], []
        for bi, spans in enumerate(gold):
            for (h, s, e) in spans:
                pos_vals.append(float(lg[bi, h, s, e]))
        neg_vals.append(float(lg[0, 0, 0, 1]))
        neg_vals.append(float(lg[1, 1, 2, 3]))
    print("pos_logits", [round(v, 2) for v in pos_vals])
    print("neg_logits_sample", [round(v, 2) for v in neg_vals])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
