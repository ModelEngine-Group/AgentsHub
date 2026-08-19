"""Local-model NL2SQL inference for task 3.

Loads a QLoRA fine-tuned small model (e.g. Qwen2.5-0.5B trained by
``src.training.finetune_analysis_model``) and uses it to translate a
natural-language analysis question into a single read-only SQL statement
against the task-2 graph analytics schema (tables ``nodes`` / ``edges``).

Returns ``None`` whenever the model is unavailable, inference fails, or the
output is not a SELECT statement, so callers can fall back to the LLM or the
template translator. The model is never trusted blindly: the caller validates
and executes the returned SQL through the read-only guard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.common.device import get_device, model_load_kwargs, move_model_to_device
from src.operators.analysis_ops.analysis_prompts import NL2SQL_SYSTEM, build_nl2sql_user
from src.operators.analysis_ops.llm_nl2sql import _extract_sql

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, Any] = {}


def predict_sql(model_path: str | None, question: str) -> str | None:
    """Run local-model NL2SQL inference. Returns a SELECT string or None."""

    if not model_path or not question:
        return None

    try:
        model, tokenizer = _load_model(model_path)
    except Exception:
        logger.warning("Failed to load NL2SQL local model from %s", model_path, exc_info=True)
        return None

    messages = [
        {"role": "system", "content": NL2SQL_SYSTEM},
        {"role": "user", "content": build_nl2sql_user(question)},
    ]

    try:
        import torch

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return _extract_sql(response)
    except Exception:
        logger.warning("NL2SQL local model inference failed.", exc_info=True)
        return None


def _load_model(model_path: str):
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    load_kwargs = model_load_kwargs(device)

    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        base_path = _resolve_base_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            trust_remote_code=True,
            **load_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **load_kwargs,
        )

    model = move_model_to_device(model, device)
    model.eval()
    _MODEL_CACHE[model_path] = (model, tokenizer)
    return model, tokenizer


def _resolve_base_model(adapter_path: str) -> str:
    """Resolve the base model path from a LoRA adapter config or convention."""
    config_path = Path(adapter_path) / "adapter_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            base = config.get("base_model_name_or_path", "")
            if base and Path(base).exists():
                return base
        except (json.JSONDecodeError, OSError):
            pass
    resolved = Path(adapter_path).resolve()
    parents = resolved.parents
    if len(parents) > 3:
        default = parents[3] / "models" / "qwen2.5-0.5b-instruct"
        if default.exists():
            return str(default)
    return "Qwen/Qwen2.5-0.5B-Instruct"
