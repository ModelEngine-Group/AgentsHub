"""Export the BERT NER benchmark model to a static-shape ONNX file.

This is an optional NPU upper-bound probe. The main project path remains the
torch_npu backend in dki.bench.npu.bert_ner_infer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification


def export_onnx(model_name: str, output: Path, batch_size: int, seq_len: int, half: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval()
    if half:
        model = model.half()

    input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (input_ids, attention_mask, token_type_ids),
            str(output),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )
    print(f"[OK] exported {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export static BERT NER ONNX for ATC probes")
    parser.add_argument("--model", default="bert-base-chinese")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()
    export_onnx(args.model, args.output, args.batch, args.seq_len, args.half)


if __name__ == "__main__":
    main()
