"""Local-model planning inference for task 3.

Loads a QLoRA fine-tuned small model (trained by
``src.training.finetune_analysis_model --task planning``) and uses it to turn a
natural-language analysis request into an operator-plan JSON. The prompt is
built with the shared :mod:`analysis_prompts` helpers so it matches the training
format exactly; a mismatch would make the adapter behave like the base model.

Returns ``None`` whenever the model is unavailable, inference fails, or the
output is not parseable JSON, so the hybrid planner can fall back to the LLM or
rule-based planner.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.operators.analysis_ops.analysis_prompts import (
    PLANNING_SYSTEM,
    build_planning_user,
)
from src.operators.analysis_ops.local_model_nl2sql import _load_model

logger = logging.getLogger(__name__)


def predict_plan(model_path: str | None, request: str) -> dict[str, Any] | None:
    """Run local-model planning inference. Returns a plan dict or None."""

    if not model_path or not request:
        return None

    try:
        model, tokenizer = _load_model(model_path)
    except Exception:
        logger.warning("Failed to load planning local model from %s", model_path, exc_info=True)
        return None

    messages = [
        {"role": "system", "content": PLANNING_SYSTEM},
        {"role": "user", "content": build_planning_user(request)},
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
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
    except Exception:
        logger.warning("Planning local model inference failed.", exc_info=True)
        return None

    return _parse_plan(response)


def _parse_plan(response: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response."""
    start = response.find("{")
    end = response.rfind("}") + 1
    if start < 0 or end <= start:
        logger.warning("Planning local model returned non-JSON response: %s", response[:200])
        return None
    try:
        return json.loads(response[start:end])
    except json.JSONDecodeError:
        logger.warning("Planning local model returned invalid JSON: %s", response[:200])
        return None
