"""Profile bounded NER hot-shape candidates without reading blind corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def select_lattice(token_lengths, sequence_lengths=(64, 128, 256), max_nodes=3):
    counts = Counter()
    overflow = 0
    for tokens in token_lengths:
        required = int(tokens) + 2
        chosen = next((value for value in sorted(sequence_lengths) if value >= required), None)
        if chosen is None:
            overflow += 1
        else:
            counts[chosen] += 1
    ranked = sorted(counts, key=lambda length: (-counts[length], length))[:max_nodes]
    ranked.sort()
    return {
        "sequence_nodes": ranked,
        "counts": {str(key): counts[key] for key in sorted(counts)},
        "overflow": overflow,
        "total": len(token_lengths),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="development/production-mix JSONL，禁止 blind/final3")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-lengths", default="64,128,256")
    parser.add_argument("--max-nodes", type=int, default=3)
    args = parser.parse_args()
    input_path = Path(args.input)
    # 内容哈希级门：重命名也拒绝（不再仅靠文件名判断）。
    from dki.evidence.final3_guard import guard_path
    guard_path(input_path, "profiling")
    # 文件名启发式作为二次防线，防止清单缺条目的新盲测集意外流入。
    lowered = input_path.name.casefold()
    if "blind" in lowered or "final3" in lowered:
        raise RuntimeError("blind/final3 data is forbidden for hot-shape profiling")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    texts = [json.loads(line)["text"] for line in input_path.open(encoding="utf-8") if line.strip()]
    lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
    report = select_lattice(lengths, tuple(int(v) for v in args.sequence_lengths.split(",")), args.max_nodes)
    report.update({
        "schema_version": 1,
        "dataset_role": "production_mix_development",
        "corpus_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "model_name": args.model_name,
    })
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
