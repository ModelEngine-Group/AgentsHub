"""质量增强 NER 微调：原始 ID 数据 + 隔离口语 OOD 数据混合训练。

保证现有性能不变的设计：
- 输入 checkpoint 是 models/medical_ner_bert（当前 0.9936 ID F1 的 BERT），
  输出到独立 models/medical_ner_quality，绝不覆盖 baseline/性能模型；
- 原 160 份 ID train 每个 epoch 全量保留，防止 OOD 微调造成灾难性遗忘；
- OOD quality train 仅来自 data/ood_train/train.jsonl，最终 OOD test
  data/ood/ood_eval.jsonl 从不读取；
- 用质量 dev 作训练中检查；最终 OOD test 只由 scripts/eval_ood.py 在训练结束后评测。

用法（服务器）：
  python3 scripts/train_ner_quality.py --epochs 5 --ood-repeat 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dki.nlp.labels import NUM_LABELS, id2label, label2id
from dki.paths import OUTPUT_DIR, PROJECT_ROOT

# 复用经过验证的 token-BIO 对齐和解码逻辑
from train_ner import _encode_dataset, _load_jsonl, _predict_finetuned, _set_seed

BASE_MODEL = PROJECT_ROOT / "models" / "medical_ner_bert"
OUTPUT_MODEL = PROJECT_ROOT / "models" / "medical_ner_quality"
ID_DATA = PROJECT_ROOT / "data" / "ner"
OOD_DATA = PROJECT_ROOT / "data" / "ood_train"


def _mentions_to_spans(row: dict) -> dict:
    """把 LLM 保证逐字出现的 mention 转为无重叠 char spans。

    同一 mention 多次出现时选第一个未占用位置；重叠时最长优先，保证 BIO 标签合法。
    """
    text = row["text"]
    candidates = []
    for ent in row.get("entities", []):
        mention = str(ent.get("mention", "")).strip()
        etype = ent.get("type")
        if not mention or etype not in {"Disease", "Symptom", "Drug", "Exam", "Treatment", "Department"}:
            continue
        start = 0
        while True:
            idx = text.find(mention, start)
            if idx < 0:
                break
            candidates.append((idx, idx + len(mention), etype))
            start = idx + 1
    candidates.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    chosen, occupied = [], []
    for s, e, typ in candidates:
        if any(not (e <= os or s >= oe) for os, oe in occupied):
            continue
        chosen.append([s, e, typ])
        occupied.append((s, e))
    chosen.sort(key=lambda x: x[0])
    return {"doc_id": row.get("doc_id", "quality"), "text": text, "spans": chosen}


def _load_quality(path: Path) -> List[dict]:
    return [_mentions_to_spans(json.loads(line)) for line in path.open(encoding="utf-8") if line.strip()]


def _span_prf(pred_rows: List[List[dict]], gold_rows: List[dict]) -> dict:
    """同类型 span-overlap 微平均（quality dev，避免规范名写法误差）。"""
    tp = fp = fn = 0
    by_type = {entity_type: {"tp": 0, "fp": 0, "fn": 0} for entity_type in (
        "Disease", "Symptom", "Drug", "Exam", "Treatment", "Department",
    )}
    for pred, gold in zip(pred_rows, gold_rows):
        matched = [False] * len(gold["spans"])
        for ent in pred:
            span = ent.get("span")
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                ps, pe = 0, 0
            else:
                ps, pe = int(span[0]), int(span[1])
            pt = ent.get("type", "")
            hit = False
            for i, (gs, ge, gt) in enumerate(gold["spans"]):
                if not matched[i] and pt == gt and not (pe <= gs or ps >= ge):
                    matched[i] = True
                    hit = True
                    break
            if hit:
                tp += 1
                if pt in by_type:
                    by_type[pt]["tp"] += 1
            else:
                fp += 1
                if pt in by_type:
                    by_type[pt]["fp"] += 1
        for is_matched, (_, _, gt) in zip(matched, gold["spans"]):
            if not is_matched:
                fn += 1
                by_type[gt]["fn"] += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    metrics = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
               "tp": tp, "fp": fp, "fn": fn}
    type_metrics = {}
    for entity_type, counts in by_type.items():
        if not sum(counts.values()):
            continue
        type_p = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        type_r = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        type_f1 = 2 * type_p * type_r / (type_p + type_r) if type_p + type_r else 0.0
        type_metrics[entity_type] = {
            "precision": round(type_p, 4), "recall": round(type_r, 4),
            "f1": round(type_f1, 4), **counts,
        }
    metrics["by_type"] = type_metrics
    return metrics


def _selection_score(metrics: dict, recall_weight: float, macro_weight: float) -> float:
    """兼顾微平均和稀有类别，避免症状多数类支配 checkpoint 选择。"""
    type_f1 = [value["f1"] for value in metrics.get("by_type", {}).values()]
    macro_f1 = sum(type_f1) / len(type_f1) if type_f1 else 0.0
    return metrics["f1"] + recall_weight * metrics["recall"] + macro_weight * macro_f1


def _eval_id(model_dir: Path, device: str) -> dict:
    """原始 ID 验证集：复用现有实体级 evaluator（规范名匹配）。"""
    from dki.nlp.eval_ner import evaluate_entities

    rows = _load_jsonl(ID_DATA / "eval.jsonl")
    backend_rows = _predict_finetuned(model_dir, [r["text"] for r in rows], device)
    predicted = {r["doc_id"]: ents for r, ents in zip(rows, backend_rows)}
    return evaluate_entities(predicted, doc_ids=[r["doc_id"] for r in rows])


def train(args) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    if args.device == "npu":
        try:
            import torch_npu  # noqa: F401

            torch.npu.set_compile_mode(jit_compile=False)
            torch.npu.manual_seed_all(args.seed)
        except ImportError as exc:
            raise RuntimeError("--device npu 需要已安装并匹配 CANN 的 torch_npu") from exc

    _set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    id_train = _load_jsonl(ID_DATA / "train.jsonl")
    quality_train = _load_quality(args.ood_data / "train.jsonl")
    quality_dev = _load_quality(args.ood_data / "dev.jsonl")
    if not quality_train or not quality_dev:
        raise RuntimeError("质量 OOD train/dev 为空；先运行 build_quality_data.py")

    # 每 epoch 保留所有 ID 样本；OOD 重复提高口语表达权重，但不让其淹没 ID 语料
    train_rows = id_train + quality_train * args.ood_repeat
    random.Random(args.seed).shuffle(train_rows)
    print(f"[data] id_train={len(id_train)} + quality_train={len(quality_train)}x{args.ood_repeat} "
          f"= {len(train_rows)}; quality_dev={len(quality_dev)}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model))
    model = AutoModelForTokenClassification.from_pretrained(
        str(args.base_model), num_labels=NUM_LABELS,
        id2label={i: lab for i, lab in id2label.items()}, label2id=dict(label2id),
    )
    device = torch.device(args.device)
    model.to(device)

    ids, mask, ttype, labels = _encode_dataset(train_rows, tokenizer, args.max_length)
    ds = TensorDataset(ids, mask, ttype, labels)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=generator)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"selection_score": -1.0, "epoch": 0}
    start = time.time()
    output = Path(args.output)
    tmp = output.parent / f".{output.name}_tmp"
    for stale in (output, tmp):
        if stale.exists():
            shutil.rmtree(stale)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            b_ids, b_mask, b_type, b_lab = [x.to(device) for x in batch]
            optim.zero_grad()
            out = model(input_ids=b_ids, attention_mask=b_mask, token_type_ids=b_type, labels=b_lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            if args.device == "npu":
                torch.npu.synchronize()
            total_loss += out.loss.item()

        # 保存临时 checkpoint → 用真实后端解码做 quality dev span-F1
        tmp.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)
        preds = _predict_finetuned(tmp, [r["text"] for r in quality_dev], device)
        dev = _span_prf(preds, quality_dev)
        selection_score = _selection_score(dev, args.recall_weight, args.macro_weight)
        print(f"[epoch {epoch}/{args.epochs}] loss={total_loss/len(loader):.4f} "
              f"quality-dev P={dev['precision']:.4f} R={dev['recall']:.4f} F1={dev['f1']:.4f} "
              f"score={selection_score:.4f}")
        if selection_score > best["selection_score"]:
            best = {
                "selection_score": round(selection_score, 6),
                "quality_dev_f1": dev["f1"],
                "quality_dev": dev,
                "epoch": epoch,
            }
            output.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
            (output / "label_list.json").write_text(
                json.dumps({"labels": [id2label[i] for i in range(NUM_LABELS)]}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # internal holdout 在训练与 checkpoint 选择中从未使用，只在最佳权重落盘后评测一次。
    quality_holdout = _load_quality(args.ood_data / "holdout.jsonl")
    holdout_pred = _predict_finetuned(output, [r["text"] for r in quality_holdout], args.device)
    holdout_eval = _span_prf(holdout_pred, quality_holdout)
    shutil.rmtree(tmp, ignore_errors=True)

    metrics = {
        "base_model": str(args.base_model), "output": str(output), "epochs": args.epochs,
        "ood_data": str(args.ood_data),
        "lr": args.lr, "batch_size": args.batch_size, "ood_repeat": args.ood_repeat,
        "recall_weight": args.recall_weight, "macro_weight": args.macro_weight,
        "id_train_count": len(id_train), "quality_train_count": len(quality_train),
        "quality_dev_count": len(quality_dev), "quality_holdout_count": len(quality_holdout),
        "train_seconds": round(time.time() - start, 1),
        "best": best,
        "quality_holdout": holdout_eval,
        "id_eval": _eval_id(output, args.device),
        "test_isolation": (
            "no fresh final2/future blind labels or text enter training/model selection; "
            "explicitly reclassified old development/error-analysis rows may be supplied by --ood-data; "
            "the data builder uses SHA-256 hashes to reject every protected exact overlap"
        ),
    }
    out_metrics = OUTPUT_DIR / "ner_finetune" / "metrics_quality.json"
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] metrics →", out_metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="OOD 质量增强 NER 微调（不替换性能模型）")
    ap.add_argument("--base-model", default=str(BASE_MODEL))
    ap.add_argument("--output", default=str(OUTPUT_MODEL))
    ap.add_argument("--ood-data", type=Path, default=OOD_DATA,
                    help="包含 train/dev/holdout.jsonl 的隔离质量数据目录")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--ood-repeat", type=int, default=3)
    ap.add_argument("--recall-weight", type=float, default=0.1,
                    help="checkpoint score 中 dev recall 的权重")
    ap.add_argument("--macro-weight", type=float, default=0.15,
                    help="checkpoint score 中逐实体类型 macro-F1 的权重")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--device", choices=["cpu", "cuda", "npu"], default="cpu")
    args = ap.parse_args()
    # 内容哈希级门：训练输入目录里绝不能混入 final3/blind（重命名也拒绝）。
    from dki.evidence.final3_guard import guard_dir
    guard_dir(args.ood_data, "training")
    train(args)


if __name__ == "__main__":
    main()
