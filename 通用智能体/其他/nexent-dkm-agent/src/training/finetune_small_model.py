"""Fine-tune Qwen2.5-0.5B with QLoRA for task-1 data orchestration.

Hardware requirement: ~6 GB VRAM with 4-bit quantization.

Usage:
    python src/training/finetune_small_model.py
    python src/training/finetune_small_model.py --model-name Qwen/Qwen2.5-0.5B-Instruct
    python src/training/finetune_small_model.py --epochs 5 --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Share the exact system prompt with inference so the fine-tuned adapter is
# trained on, and queried with, the same prompt (a mismatch silently degrades
# the model back to base behaviour).
from src.agents.data_processing_agent.local_model_planner import _SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune small model for task-1 orchestration.")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model name or path.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[2] / "data" / "training"),
        help="Directory with train/val JSONL files.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def tokenize_fn(examples, tokenizer, max_length):
    """Build supervised causal-LM examples (prompt masked, response learned).

    The prompt (system + ``Task:/Input:`` user message, matching inference) and
    the response are concatenated into one sequence; prompt tokens and padding
    get label ``-100`` so loss is computed only over the response plus its EOS.
    """
    input_ids_list = []
    attention_list = []
    labels_list = []
    pad_id = tokenizer.pad_token_id
    eos = tokenizer.eos_token or ""

    for instr, inp, out in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {instr}\nInput: {inp}"},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(out + eos, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + response_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
        attention = [1] * len(input_ids)

        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [pad_id] * pad_len
            attention = attention + [0] * pad_len
            labels = labels + [-100] * pad_len

        input_ids_list.append(input_ids)
        attention_list.append(attention)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_list,
        "labels": labels_list,
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = args.output_dir or str(data_dir / "model_output")

    print(f"Loading data from {data_dir}")
    train_data = load_jsonl(data_dir / "task_orchestration_train.jsonl")
    val_data = load_jsonl(data_dir / "task_orchestration_val.jsonl")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=_get_bnb_config(),
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    train_dataset = train_dataset.map(
        lambda x: tokenize_fn(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_fn(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        warmup_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


def _get_bnb_config():
    try:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
    except ImportError:
        return None


if __name__ == "__main__":
    main()
