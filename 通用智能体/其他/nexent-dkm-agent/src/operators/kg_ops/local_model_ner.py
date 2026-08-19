"""Local model inference for task-2 KG entity extraction.

Loads a QLoRA fine-tuned model and uses it for NER when available.
Falls back to the rule-based dictionary approach otherwise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.common.device import get_device, model_load_kwargs, move_model_to_device
from src.operators.kg_ops.kg_prompts import KG_SYSTEM, build_kg_user

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, Any] = {}


def predict_kg_entities(
    model_path: str | None,
    text: str,
) -> dict[str, list[str]] | None:
    """Run local model NER inference. Returns entity dict or None on failure."""
    if not model_path:
        return None

    try:
        model, tokenizer = _load_model(model_path)
    except Exception:
        logger.warning("Failed to load KG local model from %s", model_path, exc_info=True)
        return None

    messages = [
        {"role": "system", "content": KG_SYSTEM},
        {"role": "user", "content": build_kg_user(text[:1000])},
    ]

    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        import torch
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _parse_entity_response(response)
    except Exception:
        logger.warning("KG local model inference failed.", exc_info=True)
        return None


def _load_model(model_path: str):
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    load_kwargs = model_load_kwargs(device)

    # Check if LoRA adapter
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
    """Try to find the base model for a LoRA adapter."""
    config_path = Path(adapter_path) / "adapter_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        base = config.get("base_model_name_or_path", "")
        if base and Path(base).exists():
            return base
    # Fall back to default local path: go up 3 levels from adapter_path
    resolved = Path(adapter_path).resolve()
    parents = resolved.parents
    if len(parents) > 3:
        default = parents[3] / "models" / "qwen2.5-0.5b-instruct"
        if default.exists():
            return str(default)
    return "Qwen/Qwen2.5-0.5B-Instruct"


def _parse_entity_response(response: str) -> dict[str, list[str]] | None:
    """Parse the model's JSON response into entity dict."""
    import re

    text = response.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
        else:
            return None

    entities = data.get("entities", {})
    valid_types = {"Disease", "Symptom", "Drug", "Examination", "Treatment"}
    result = {}
    for etype, names in entities.items():
        if etype in valid_types and isinstance(names, list):
            result[etype] = [str(n) for n in names if n]
    return result if result else None
