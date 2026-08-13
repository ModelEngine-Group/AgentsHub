"""微调 bert-base-chinese 成真实医疗 NER（token-classification, 13 BIO 标签）。

数据来源：dki.nlp.weak_label 的远程监督弱标注（dict 后端 F1≈99.97% 打的 char span）。
标签对齐：fast tokenizer 的 offset_mapping 把 char span 映射到 token 级 BIO
         （每个 token 取其起始字符所属实体的 B-/I-；special/subword 续词 → -100 忽略）。

刻意用**手写 torch 训练循环**而非 transformers.Trainer：
- 规模极小（160 样本、bert-base、5 epoch），手写循环 <40 行、无样板；
- 免疫 transformers 版本间 TrainingArguments API 漂移（服务器 transformers 5.x）。

确定性：固定 seed（torch/np/random），CPU 训练（bert-base 此规模秒级/分钟级，
无需 NPU；且保证"任意机可复现"）。

产物：
- models/medical_ner_bert/            微调权重 + tokenizer + config（id2label 为真实 B-Disease…）
- outputs/ner_finetune/metrics.json   三后端实体级 F1 对比（微调bert / dict / 未微调bert）

用法（服务器）：
    python scripts/train_ner.py --epochs 5
    python scripts/train_ner.py --epochs 5 --skip-eval-baselines   # 只训练，不跑对比
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.nlp.labels import LABELS, NUM_LABELS, id2label, label2id  # noqa: E402
from dki.paths import OUTPUT_DIR, PROJECT_ROOT  # noqa: E402

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "medical_ner_bert"
NER_DATA_DIR = PROJECT_ROOT / "data" / "ner"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _char_labels(text_len: int, spans: List[List[object]]) -> List[int]:
    """字符级标签 id 序列（O=0，实体首字 B-，其余 I-）。"""
    labels = [0] * text_len
    for s, e, etype in spans:
        s, e = int(s), int(e)
        if not (0 <= s < e <= text_len):
            continue
        labels[s] = label2id[f"B-{etype}"]
        for i in range(s + 1, e):
            labels[i] = label2id[f"I-{etype}"]
    return labels


def _align_tokens(offsets: List[Tuple[int, int]], char_labels: List[int]) -> List[int]:
    """用 offset_mapping 把字符级标签对齐到 token 级（special/pad → -100）。

    每个 token 取其起始字符的标签；若起始字符是 I- 而 token 是实体内部续词，
    保持 I-（简化处理，对中文 1 字≈1 token 足够稳）。
    """
    token_labels = []
    for (start, end) in offsets:
        if start == end:  # special token / padding
            token_labels.append(-100)
            continue
        lab = char_labels[start] if start < len(char_labels) else 0
        token_labels.append(lab)
    return token_labels


def _encode_dataset(rows: List[dict], tokenizer, max_length: int):
    import torch

    all_ids, all_mask, all_type, all_labels = [], [], [], []
    for r in rows:
        text = r["text"]
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc["offset_mapping"][0].tolist()
        char_labels = _char_labels(len(text), r["spans"])
        token_labels = _align_tokens(offsets, char_labels)
        all_ids.append(enc["input_ids"][0])
        all_mask.append(enc["attention_mask"][0])
        all_type.append(enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))[0])
        all_labels.append(torch.tensor(token_labels, dtype=torch.long))
    return (
        torch.stack(all_ids),
        torch.stack(all_mask),
        torch.stack(all_type),
        torch.stack(all_labels),
    )


def train(args) -> Dict[str, object]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    _set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=NUM_LABELS,
        id2label={i: lab for i, lab in id2label.items()},
        label2id=dict(label2id),
    )
    device = torch.device(args.device)
    model.to(device)

    train_rows = _load_jsonl(NER_DATA_DIR / "train.jsonl")
    eval_rows = _load_jsonl(NER_DATA_DIR / "eval.jsonl")
    print(f"[data] train={len(train_rows)} eval={len(eval_rows)} labels={NUM_LABELS}")

    ids, mask, ttype, labels = _encode_dataset(train_rows, tokenizer, args.max_length)
    ds = TensorDataset(ids, mask, ttype, labels)
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=g)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(loader) * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda step: min(1.0, step / max(1, warmup))
        * max(0.0, (total_steps - step) / max(1, total_steps - warmup)) if step >= warmup
        else step / max(1, warmup),
    )

    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_loss = 0.0
        for batch in loader:
            b_ids, b_mask, b_type, b_lab = [x.to(device) for x in batch]
            optim.zero_grad()
            out = model(input_ids=b_ids, attention_mask=b_mask,
                        token_type_ids=b_type, labels=b_lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            ep_loss += out.loss.item()
        print(f"[epoch {epoch+1}/{args.epochs}] loss={ep_loss/len(loader):.4f} "
              f"elapsed={time.time()-t0:.1f}s")
    train_s = time.time() - t0

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "label_list.json").write_text(
        json.dumps({"labels": LABELS, "id2label": {str(k): v for k, v in id2label.items()}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[√] 微调模型已保存 → {out_dir}  (训练耗时 {train_s:.1f}s)")

    metrics: Dict[str, object] = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "device": args.device,
        "num_labels": NUM_LABELS,
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "train_seconds": round(train_s, 1),
    }
    if not args.skip_eval_baselines:
        metrics["eval_f1"] = _eval_three_backends(out_dir, eval_rows, device)

    metrics_dir = OUTPUT_DIR / "ner_finetune"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    # 指标文件名带模型标识，避免多模型对比时互相覆盖（bert / rbt3 / rbt6 …）
    tag = out_dir.name.replace("medical_ner_", "") or "model"
    metrics["model_tag"] = tag
    (metrics_dir / f"metrics_{tag}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 兼容旧路径：默认 bert 模型也写一份 metrics.json（Notebook/README 引用）
    if tag == "bert":
        (metrics_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"[√] 指标 → {metrics_dir / f'metrics_{tag}.json'}")
    print(json.dumps(metrics.get("eval_f1", {}), ensure_ascii=False, indent=2))
    return metrics


def _predict_finetuned(model_dir: Path, texts: List[str], device) -> List[List[dict]]:
    """用微调模型对 eval 文本抽实体（BIO 解码 + 规范名归一）。"""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from dki.operators.medical_entity_extractor.process import load_vocab

    vocab = load_vocab()  # 表面形式 -> (规范名, 类型)，用于把 surface 归一到 canonical
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(device).eval()
    id2lab = model.config.id2label

    results = []
    for text in texts:
        enc = tok(text, truncation=True, max_length=256,
                  return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.inference_mode():
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        pred = logits.argmax(-1)[0].tolist()
        ents = _decode_bio(text, offsets, pred, id2lab, vocab)
        results.append(ents)
    return results


def _decode_bio(text, offsets, pred_ids, id2lab, vocab) -> List[dict]:
    ents, cur = [], None
    for k, pid in enumerate(pred_ids):
        if k >= len(offsets):
            break
        s, e = offsets[k]
        if s == e:
            continue
        lab = str(id2lab.get(pid, id2lab.get(str(pid), "O")))
        if lab.startswith("B-"):
            if cur:
                ents.append(cur)
            cur = {"s": s, "e": e, "type": lab[2:]}
        elif lab.startswith("I-") and cur and cur["type"] == lab[2:]:
            cur["e"] = e
        else:
            if cur:
                ents.append(cur)
            cur = None
    if cur:
        ents.append(cur)
    out = []
    for en in ents:
        surface = text[en["s"]:en["e"]]
        canonical = vocab.get(surface, (surface, en["type"]))[0]  # 归一到规范名
        out.append({
            "name": canonical,
            "type": en["type"],
            "mention": surface,
            "span": [en["s"], en["e"]],
        })
    return out


def _eval_three_backends(model_dir: Path, eval_rows: List[dict], device) -> Dict[str, object]:
    """在 eval 集（训练不可见）上对比：微调bert / dict / 未微调bert 的实体级 F1。"""
    from dki.nlp.eval_ner import evaluate_entities
    from dki.operators.medical_entity_extractor.process import _resolve_spans, load_vocab

    eval_ids = [r["doc_id"] for r in eval_rows]
    texts = [r["text"] for r in eval_rows]
    vocab = load_vocab()

    # dict 后端
    dict_pred = {}
    for did, text in zip(eval_ids, texts):
        hits = [w for w in vocab if w in text]
        dict_pred[did] = [{"name": e["name"], "type": e["type"]}
                          for e in _resolve_spans(text, hits, vocab)]

    # 微调 bert
    ft = _predict_finetuned(model_dir, texts, device)
    ft_pred = {did: ents for did, ents in zip(eval_ids, ft)}

    def _sub(pred_map):
        return {d: pred_map.get(d, []) for d in eval_ids}

    res = {
        "eval_doc_ids": eval_ids,
        "finetuned_bert": evaluate_entities(_sub(ft_pred), doc_ids=eval_ids),
        "dict": evaluate_entities(_sub(dict_pred), doc_ids=eval_ids),
    }
    for k in ("finetuned_bert", "dict"):
        r = res[k]
        print(f"[eval {k}] P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="微调医疗 NER（bert-base-chinese, 13 BIO 标签）")
    ap.add_argument("--base-model", default="bert-base-chinese")
    ap.add_argument("--output", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "npu"])
    ap.add_argument("--skip-eval-baselines", action="store_true")
    args = ap.parse_args()

    if not (NER_DATA_DIR / "train.jsonl").exists():
        print(f"[!] 缺少训练数据 {NER_DATA_DIR}/train.jsonl；"
              f"先运行 python -c \"from dki.nlp.weak_label import build_ner_dataset; build_ner_dataset()\"")
        sys.exit(2)
    # 内容哈希级门：训练语料目录里绝不能混入 final3/blind（重命名也拒绝）。
    from dki.evidence.final3_guard import guard_dir
    guard_dir(NER_DATA_DIR, "training")
    train(args)


if __name__ == "__main__":
    main()
