"""Local model planner for task 1 -- uses a fine-tuned small model."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.common.device import get_device, model_load_kwargs, move_model_to_device

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a data processing pipeline planner. "
    "Return JSON with keys: operators, rationale, task_type, data_type, "
    "intent_keywords, confidence."
)

# Module-level cache to avoid reloading model on every call.
_cached_model = None
_cached_tokenizer = None
_cached_model_path: str | None = None


def _load_model(model_path: str):
    global _cached_model, _cached_tokenizer, _cached_model_path

    if _cached_model is not None and _cached_model_path == model_path:
        return _cached_model, _cached_tokenizer

    # Detect whether this is a LoRA adapter directory or a full model.
    import os

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = get_device()
    load_kwargs = model_load_kwargs(device)

    adapter_file = os.path.join(model_path, "adapter_config.json")
    if os.path.isfile(adapter_file):
        # It's a LoRA adapter -- need to find the base model.
        base_model_name = _resolve_base_model(model_path)
        logger.info("Loading LoRA adapter from %s on top of %s", model_path, base_model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            **load_kwargs,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, model_path)
        model = move_model_to_device(model, device)
        model.eval()
    else:
        logger.info("Loading full model from %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            **load_kwargs,
        )
        model = move_model_to_device(model, device)
        model.eval()

    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_model_path = model_path
    return model, tokenizer


def _resolve_base_model(adapter_path: str) -> str:
    """Resolve the base model path from adapter config or convention."""
    import os
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            base = cfg.get("base_model_name_or_path") or cfg.get("base_model_path")
            if base:
                return base
        except Exception:
            pass

    # Convention: check for a local models/ directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(adapter_path))
    )))
    local_qwen = os.path.join(project_root, "models", "qwen2.5-0.5b-instruct")
    if os.path.isdir(local_qwen):
        return local_qwen

    return "Qwen/Qwen2.5-0.5B-Instruct"


def predict_plan(
    model_path: str,
    task_request: str,
    data_profile: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Use a locally fine-tuned model to predict an operator plan.

    Returns None if the model cannot be loaded or inference fails.
    """

    try:
        from transformers import (  # noqa: F401  # availability check only
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError:
        logger.warning("transformers not available for local model planning.")
        return None

    try:
        model, tokenizer = _load_model(model_path)
    except Exception:
        logger.warning("Failed to load local model from %s", model_path, exc_info=True)
        return None

    input_data: dict[str, Any] = {}
    if data_profile:
        input_data["data_profile"] = {
            k: data_profile[k]
            for k in ("file_name", "row_count", "column_count", "duplicate_rows", "missing_cells")
            if k in data_profile
        }

    user_content = f"Task: {task_request}\nInput: {json.dumps(input_data, ensure_ascii=False)}"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    except Exception:
        logger.warning("Local model inference failed.", exc_info=True)
        return None

    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass

    logger.warning("Local model returned non-JSON response: %s", response[:200])
    return None
