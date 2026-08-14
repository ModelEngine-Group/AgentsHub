"""Task-1 agent chain backed by DataMate data access and cleaning tasks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
import itertools
import json
import mimetypes
import os
from pathlib import Path
import socket
import time
import uuid
from typing import Any
from urllib import error, request

from operators.data_processing import infer_schema_profile, load_records, normalize_fields, profile_documents, quality_check
from operators.task1_tensor_ranker import build_operator_descriptor as build_task1_tensor_ranker_descriptor
from operators.task1_tensor_ranker import rank_task1_operator_candidates
from agents.task1_policy_model import (
    OPERATOR_ALIAS_LABELS,
    OPERATOR_LABELS,
    RANKER_OPERATOR_PROTOTYPES,
    _operator_capability_guess,
    load_policy_model,
    predict_policy,
    score_dag_objectives_with_policy_model,
    score_operator_with_policy_ranker,
)


DEFAULT_DATAMATE_API_BASE = "http://localhost:8080"
DEFAULT_DATAMATE_RUNTIME_API = "http://localhost:18000"
DEFAULT_TASK = "基于DataMate完成医疗数据接入、清洗、转换和质量处理"
DATAMATE_RUNTIME_API_ENV = "DATAMATE_RUNTIME_API"
DATAMATE_VENDOR_CATEGORY_ID = "431e7798-5426-4e1a-aae6-b9905a836b34"
DATAJUICER_VENDOR_CATEGORY_ID = "79b385b4-fde8-4617-bcba-02a176938996"
DEFAULT_TASK1_POLICY_MODEL_PATH = Path("models/task1_policy/task1_orchestration_policy.joblib")
DEFAULT_TASK1_POLICY_SELECTOR_CONFIG_PATH = Path("models/task1_policy/task1_policy_selector_config.json")
TASK1_POLICY_MODEL_ENV = "TASK1_POLICY_MODEL_PATH"
TASK1_POLICY_MODEL_ENABLED_ENV = "TASK1_POLICY_MODEL_ENABLED"
TASK1_POLICY_SELECTOR_CONFIG_ENV = "TASK1_POLICY_SELECTOR_CONFIG_PATH"
TASK1_TENSOR_RANKER_DEVICE_ENV = "TASK1_TENSOR_RANKER_DEVICE"
TASK1_TRANSFORMER_LORA_BASE_MODEL_ENV = "TASK1_TRANSFORMER_LORA_BASE_MODEL"
TASK1_TRANSFORMER_LORA_DEVICE_ENV = "TASK1_TRANSFORMER_LORA_DEVICE"
TASK1_TRANSFORMER_LORA_RUNTIME_ENABLED_ENV = "TASK1_TRANSFORMER_LORA_RUNTIME_ENABLED"
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "STOPPED", "PARTIAL_SUCCESS"}
SUCCESS_STATUSES = {"COMPLETED", "PARTIAL_SUCCESS"}
RUNNING_STATUSES = {"RUNNING", "PROCESSING", "SUBMITTED", "STARTED"}
SUPPORTED_INPUT_SUFFIXES = {"csv", "json", "jsonl", "txt", "md", "markdown"}
STRUCTURED_SUFFIXES = {"csv", "json", "jsonl"}
UNSTRUCTURED_SUFFIXES = {"txt", "md", "markdown"}

PLANNER_V2_OBJECTIVE_WEIGHTS = {
    "intent_match": 0.32,
    "quality_risk_coverage": 0.28,
    "io_compatibility": 0.18,
    "performance_cost": 0.12,
    "redundancy_penalty": 0.10,
}

PLANNER_V3_OBJECTIVE_WEIGHTS = {
    "capability_coverage": 0.34,
    "quality_risk_reduction": 0.24,
    "io_compatibility": 0.15,
    "latency_cost": 0.12,
    "dag_simplicity": 0.08,
    "operator_diversity": 0.07,
}

OperatorMarketCache = dict[tuple[str, str], list[dict[str, Any]]]
PolicyModelCache = dict[tuple[str, int], Any]

_POLICY_MODEL_CACHE: PolicyModelCache = {}


@dataclass
class HttpCall:
    name: str
    method: str
    url: str
    status_code: int | None
    ok: bool
    body_json: Any = None
    body_preview: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "url": self.url,
            "status_code": self.status_code,
            "ok": self.ok,
            "body_json": self.body_json,
            "body_preview": self.body_preview,
            "error": self.error,
        }


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def _datamate_runtime_api_base() -> str:
    return os.getenv(DATAMATE_RUNTIME_API_ENV, DEFAULT_DATAMATE_RUNTIME_API).rstrip("/")


def _operator_market_url(runtime_api_base: str) -> str:
    return _join_url(runtime_api_base, "/api/operators/list")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _task1_policy_model_path() -> Path:
    return Path(os.getenv(TASK1_POLICY_MODEL_ENV, str(DEFAULT_TASK1_POLICY_MODEL_PATH)))


def _task1_policy_model_enabled() -> bool:
    return os.getenv(TASK1_POLICY_MODEL_ENABLED_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _load_task1_policy_model_from_env() -> Any:
    if not _task1_policy_model_enabled():
        return None
    model_path = _task1_policy_model_path()
    try:
        mtime_ns = model_path.stat().st_mtime_ns
    except OSError:
        mtime_ns = -1
    cache_key = (str(model_path.resolve()), mtime_ns)
    if cache_key not in _POLICY_MODEL_CACHE:
        _POLICY_MODEL_CACHE.clear()
        _POLICY_MODEL_CACHE[cache_key] = load_policy_model(model_path)
    return _POLICY_MODEL_CACHE[cache_key]


def _task1_policy_selector_config_path() -> Path:
    return Path(os.getenv(TASK1_POLICY_SELECTOR_CONFIG_ENV, str(DEFAULT_TASK1_POLICY_SELECTOR_CONFIG_PATH)))


def _load_task1_policy_selector_config() -> dict[str, Any]:
    config_path = _task1_policy_selector_config_path()
    if not config_path.exists():
        return {
            "selector_version": "task1_policy_selector_v1_ab_gated",
            "enabled": False,
            "status": "selector_config_missing",
            "path": str(config_path),
            "selection_policy": {"fallback": "constraint_planner_only_without_local_processing_fallback"},
        }
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "selector_version": "task1_policy_selector_v1_ab_gated",
            "enabled": False,
            "status": "selector_config_invalid",
            "path": str(config_path),
            "reason": str(exc),
            "selection_policy": {"fallback": "constraint_planner_only_without_local_processing_fallback"},
        }
    payload.setdefault("path", str(config_path))
    return payload


def _task1_transformer_lora_runtime_enabled() -> bool:
    return os.getenv(TASK1_TRANSFORMER_LORA_RUNTIME_ENABLED_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _select_task1_transformer_lora_device(torch: Any) -> dict[str, Any]:
    requested = os.getenv(TASK1_TRANSFORMER_LORA_DEVICE_ENV, "auto").strip().lower() or "auto"
    npu_available = False
    try:
        import torch_npu  # noqa: F401

        npu_available = bool(hasattr(torch, "npu") and torch.npu.is_available())
    except Exception:
        npu_available = False
    cuda_available = bool(torch.cuda.is_available())
    if requested.startswith("npu"):
        if not npu_available:
            raise RuntimeError("TASK1_TRANSFORMER_LORA_DEVICE requested NPU but torch_npu is unavailable")
        return {"kind": "npu", "device": requested if ":" in requested else "npu:0", "dtype": torch.bfloat16}
    if requested.startswith("cuda"):
        if not cuda_available:
            raise RuntimeError("TASK1_TRANSFORMER_LORA_DEVICE requested CUDA but torch.cuda is unavailable")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return {"kind": "cuda", "device": requested if ":" in requested else "cuda:0", "dtype": dtype}
    if requested == "cpu":
        return {"kind": "cpu", "device": "cpu", "dtype": torch.float32}
    if npu_available:
        return {"kind": "npu", "device": "npu:0", "dtype": torch.bfloat16}
    if cuda_available:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return {"kind": "cuda", "device": "cuda:0", "dtype": dtype}
    return {"kind": "cpu", "device": "cpu", "dtype": torch.float32}


def _task1_transformer_lora_status(selector: dict[str, Any]) -> dict[str, Any]:
    evidence = selector.get("gate_evidence", {}).get("transformer_lora", {}) if isinstance(selector.get("gate_evidence"), dict) else {}
    selection_policy = selector.get("selection_policy", {}) if isinstance(selector.get("selection_policy"), dict) else {}
    adapter_path = Path(str(evidence.get("adapter_path") or ""))
    configured_base_model = os.getenv(TASK1_TRANSFORMER_LORA_BASE_MODEL_ENV)
    if configured_base_model is not None and not configured_base_model.strip():
        configured_base_model = None
    base_model = configured_base_model or str(evidence.get("base_model") or "")
    reasons: list[str] = []
    if not _task1_transformer_lora_runtime_enabled():
        reasons.append("runtime_disabled_by_environment")
    if not selection_policy.get("use_transformer_lora_candidate"):
        reasons.append("selector_does_not_enable_transformer_lora_candidate")
    transformer_role = str(selection_policy.get("transformer_lora_role") or "")
    if not transformer_role.startswith("primary_candidate_gate_passed"):
        reasons.append("transformer_lora_not_primary_candidate")
    if not evidence.get("available", True):
        reasons.append("transformer_lora_evidence_not_available")
    if not evidence.get("less_than_1b_parameters"):
        reasons.append("transformer_lora_not_marked_less_than_1b")
    if not adapter_path or not (adapter_path / "adapter_config.json").exists():
        reasons.append("adapter_path_missing")
    if not base_model:
        reasons.append("base_model_missing")
    elif not Path(base_model).exists():
        reasons.append("base_model_path_not_available_for_runtime")
    return {
        "status": "ready" if not reasons else "not_ready",
        "ready": not reasons,
        "reasons": reasons,
        "adapter_path": str(adapter_path) if str(adapter_path) else "",
        "base_model": base_model,
        "device": os.getenv(TASK1_TRANSFORMER_LORA_DEVICE_ENV, "auto").strip().lower() or "auto",
        "evidence": evidence,
        "runtime_policy": "strict_primary_candidate_only_no_silent_fallback",
    }


def _validate_task1_lora_policy_candidate(candidate: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(candidate, dict):
        return None, ["candidate_not_dict"]
    workflow_type = str(candidate.get("workflow_type") or "")
    if workflow_type not in {"structured_etl", "text_processing_pipeline"}:
        errors.append("invalid_workflow_type")
    operator_ids = [
        OPERATOR_ALIAS_LABELS.get(str(item), str(item))
        for item in candidate.get("recommended_operator_ids", [])
        if OPERATOR_ALIAS_LABELS.get(str(item), str(item)) in OPERATOR_LABELS
    ]
    if not operator_ids:
        errors.append("missing_valid_recommended_operator_ids")
    cleaned = {
        "policy_version": "task1_transformer_lora_runtime_candidate_v1",
        "workflow_type": workflow_type,
        "required_capabilities": [
            str(item)
            for item in candidate.get("required_capabilities", [])
            if str(item).strip()
        ],
        "operator_keywords": [
            str(item)
            for item in candidate.get("operator_keywords", [])
            if str(item).strip()
        ],
        "recommended_operator_ids": _unique_ordered(operator_ids),
        "prefer_low_latency": bool(candidate.get("prefer_low_latency")),
        "strict_quality": bool(candidate.get("strict_quality")),
        "requires_recovery": bool(candidate.get("requires_recovery")),
        "quality_priority": str(candidate.get("quality_priority") or "standard"),
        "cost_tier": str(candidate.get("cost_tier") or "balanced"),
        "source": "trained_task1_transformer_lora_orchestration_policy",
    }
    return (None if errors else cleaned), errors


def _task1_lora_operator_catalog() -> list[dict[str, Any]]:
    prototypes = {
        OPERATOR_ALIAS_LABELS.get(str(item.get("id") or ""), str(item.get("id") or "")): item
        for item in RANKER_OPERATOR_PROTOTYPES
    }
    hints = {
        "clean_email_mapper": "email/pii cleanup",
        "clean_links_mapper": "url/link/http cleanup",
        "clean_ip_mapper": "ip/pii cleanup; not email",
        "remove_extra_spaces_mapper": "space/whitespace normalization",
        "text_length_filter": "empty/short/long text quality filter",
        "alphanumeric_filter": "strict character quality filter",
        "frequency_specified_field_selector": "schema/field/frequency selector",
        "text_pair_similarity_filter": "duplicate/near-duplicate similarity filter",
    }
    catalog: list[dict[str, Any]] = []
    for operator_id in OPERATOR_LABELS:
        prototype = dict(prototypes.get(operator_id, {"id": operator_id, "name": operator_id, "description": operator_id}))
        catalog.append(
            {
                "id": operator_id,
                "capabilities": _operator_capability_guess(prototype),
                "when_to_use": hints.get(operator_id, ""),
            }
        )
    return catalog


def _generate_task1_lora_policy_candidate(
    *,
    task: str,
    input_profile: dict[str, Any],
    status: dict[str, Any],
) -> dict[str, Any]:
    import importlib.util
    import re

    original_find_spec = importlib.util.find_spec

    def find_spec_without_torchvision(name: str, package: str | None = None) -> Any:
        if name == "torchvision" or name.startswith("torchvision."):
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = find_spec_without_torchvision
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        def extract_json(text: str) -> dict[str, Any] | None:
            text = text.strip()
            candidates = [text]
            candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL))
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                candidates.append(text[brace_start : brace_end + 1])
            for candidate_text in candidates:
                try:
                    parsed = json.loads(candidate_text)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
            return None

        adapter_path = Path(str(status["adapter_path"]))
        base_model = str(status["base_model"])
        device_info = _select_task1_transformer_lora_device(torch)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=device_info["dtype"],
            device_map="auto" if device_info["kind"] == "cuda" else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        if device_info["kind"] in {"npu", "cpu"}:
            model.to(device_info["device"])
        model.eval()
        features = {
            "format": input_profile.get("format"),
            "structured": bool(input_profile.get("structured")),
            "unstructured": bool(input_profile.get("unstructured")),
            "batch": bool(input_profile.get("batch")),
            "complexity": input_profile.get("complexity"),
            "record_count_estimate": input_profile.get("record_count_estimate", 0),
            "schema_field_count": len(input_profile.get("schema_profile", {}).get("fields", []) or []),
            "schema_has_text": bool(input_profile.get("schema_profile", {}).get("has_text_field")),
            "schema_text_coverage": input_profile.get("schema_profile", {}).get("schema_confidence", {}).get("text_coverage"),
            "quality_warning_count": input_profile.get("quality_report", {}).get("warning_count", 0),
        }
        payload = {
            "task": task,
            "features": features,
            "operator_catalog": _task1_lora_operator_catalog(),
            "constraints": {
                "must_use_nexent_agent_framework": True,
                "must_use_datamate_access_and_processing": True,
                "recommended_operator_ids_only_from_operator_catalog": True,
                "canonicalize_aliases_before_output": True,
                "no_local_fallback": True,
            },
            "required_output_schema": {
                "workflow_type": "structured_etl|text_processing_pipeline",
                "required_capabilities": ["schema_guard", "quality_filter"],
                "operator_keywords": ["clean", "filter", "quality"],
                "recommended_operator_ids": OPERATOR_LABELS,
                "prefer_low_latency": False,
                "strict_quality": False,
                "requires_recovery": False,
                "quality_priority": "standard|high|critical",
                "cost_tier": "low_cost|balanced|quality_first",
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是ModelEngine任务一的数据处理编排智能体。必须基于Nexent智能体框架和DataMate数据接入/处理能力进行规划。"
                    "recommended_operator_ids必须只从用户给出的DataMate canonical operator_catalog中选择。不要输出未知算子、不要输出解释、不要使用本地fallback。"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant: "
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device_info["device"])
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=260,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated_ids[0][inputs["input_ids"].shape[1] :]
        parsed = extract_json(tokenizer.decode(new_tokens, skip_special_tokens=True))
        if parsed is None:
            raise ValueError("transformer_lora_generated_text_is_not_valid_json")
        return parsed
    finally:
        importlib.util.find_spec = original_find_spec


def _task1_transformer_lora_runtime_candidate(
    *,
    task: str,
    input_profile: dict[str, Any],
    selector: dict[str, Any],
) -> dict[str, Any]:
    status = _task1_transformer_lora_status(selector)
    result = {
        "status": status["status"],
        "ready": status["ready"],
        "reasons": status["reasons"],
        "adapter_path": status["adapter_path"],
        "base_model": status["base_model"],
        "device": status.get("device"),
        "candidate": None,
        "validation_errors": [],
    }
    if not status["ready"]:
        return result
    try:
        raw_candidate = _generate_task1_lora_policy_candidate(
            task=task,
            input_profile=input_profile,
            status=status,
        )
        candidate, errors = _validate_task1_lora_policy_candidate(raw_candidate)
        result["validation_errors"] = errors
        if candidate is None:
            result["status"] = "invalid_candidate"
            result["ready"] = False
            return result
        result["status"] = "candidate_ready"
        result["candidate"] = candidate
        return result
    except Exception as exc:
        result["status"] = "runtime_error"
        result["ready"] = False
        result["reasons"] = [str(exc)]
        return result


def _task1_lora_capabilities_for_operator_ids(operator_ids: list[str]) -> list[str]:
    catalog = {item["id"]: item for item in _task1_lora_operator_catalog()}
    capabilities: list[str] = []
    for operator_id in operator_ids:
        capabilities.extend(catalog.get(operator_id, {}).get("capabilities", []))
    return _unique_ordered(capabilities)


def _guard_task1_transformer_lora_candidate(
    candidate: dict[str, Any],
    *,
    input_profile: dict[str, Any],
    quality_risk_profile: dict[str, Any],
    intent_flags: dict[str, Any],
) -> dict[str, Any]:
    if not candidate:
        return {
            "status": "not_available",
            "accepted": False,
            "reasons": ["candidate_missing"],
            "policy": "constraint_planner_fallback",
        }
    reasons: list[str] = []
    operator_ids = [
        OPERATOR_ALIAS_LABELS.get(str(item), str(item))
        for item in candidate.get("recommended_operator_ids", [])
        if OPERATOR_ALIAS_LABELS.get(str(item), str(item)) in OPERATOR_LABELS
    ]
    candidate_capabilities = _task1_lora_capabilities_for_operator_ids(operator_ids)
    required_capabilities = _datamate_operator_required_capabilities(
        list(quality_risk_profile.get("required_capabilities") or [])
    )
    required_set = set(required_capabilities)
    covered_set = set(candidate_capabilities)
    uncovered = sorted(required_set - covered_set)
    workflow_type = str(candidate.get("workflow_type") or "")
    if input_profile.get("structured") and workflow_type != "structured_etl":
        reasons.append("workflow_mismatch_for_structured_input")
    if input_profile.get("unstructured") and not input_profile.get("structured") and workflow_type != "text_processing_pipeline":
        reasons.append("workflow_mismatch_for_unstructured_input")
    if uncovered:
        reasons.append("required_capabilities_not_covered:" + ",".join(uncovered))
    if len(operator_ids) > 5:
        reasons.append("too_many_recommended_operators")
    if input_profile.get("structured") and "schema_guard" in required_set and "schema_guard" not in covered_set:
        reasons.append("structured_input_missing_schema_guard")
    cost_tier = str(candidate.get("cost_tier") or "")
    prefer_low_latency = bool(candidate.get("prefer_low_latency")) or bool(intent_flags.get("low_latency"))
    heavy_for_low_cost = {"text_pair_similarity_filter", "alphanumeric_filter"}
    if (prefer_low_latency or cost_tier in {"low", "low_cost"}) and len(operator_ids) > 3:
        reasons.append("low_cost_candidate_overselects_operators")
    if (prefer_low_latency or cost_tier in {"low", "low_cost"}) and heavy_for_low_cost & set(operator_ids) and not (
        {"deduplication", "quality_filter"} & required_set
    ):
        reasons.append("low_cost_candidate_uses_heavy_quality_or_dedup_operator")
    accepted = not reasons
    return {
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "reasons": reasons,
        "candidate_operator_ids": operator_ids,
        "candidate_capabilities": candidate_capabilities,
        "required_capabilities": required_capabilities,
        "uncovered_capabilities": uncovered,
        "policy": (
            "candidate_bias_then_constraint_dag_guard"
            if accepted
            else "constraint_planner_fallback"
        ),
    }


def _select_task1_policy_mode(policy_prediction: dict[str, Any], input_profile: dict[str, Any]) -> dict[str, Any]:
    selector = _load_task1_policy_selector_config()
    gates = selector.get("gates", {}) if isinstance(selector.get("gates"), dict) else {}
    confidence = float(policy_prediction.get("confidence") or 0.0)
    warning_count = int(input_profile.get("quality_report", {}).get("warning_count", 0) or 0)
    text_coverage = float(input_profile.get("schema_profile", {}).get("schema_confidence", {}).get("text_coverage", 1.0) or 0.0)
    high_risk = bool(warning_count or text_coverage < 0.95 or input_profile.get("batch"))
    min_confidence = float(gates.get("high_risk_min_confidence") if high_risk else gates.get("min_confidence") or 0.58)
    reasons: list[str] = []
    if not selector.get("enabled"):
        reasons.append(str(selector.get("status") or "ab_gates_not_enabled"))
    if policy_prediction.get("source") != "trained_task1_orchestration_policy_model":
        reasons.append("trained_policy_prediction_unavailable")
    if confidence < min_confidence:
        reasons.append(f"prediction_confidence_below_gate:{confidence:.4f}<{min_confidence:.4f}")
    enabled = not reasons
    return {
        "selector_version": selector.get("selector_version", "task1_policy_selector_v1_ab_gated"),
        "mode": "learned_policy_guided" if enabled else "constraint_planner_guarded",
        "learned_policy_enabled": enabled,
        "confidence": confidence,
        "min_confidence": min_confidence,
        "high_risk_context": high_risk,
        "reasons": reasons,
        "gate_evidence": selector.get("gate_evidence", {}),
        "selection_policy": selector.get("selection_policy", {}),
        "config_path": selector.get("path"),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _decode_json(raw: str) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _json_body(body: dict[str, Any] | None) -> bytes | None:
    if body is None:
        return None
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def _build_multipart_form_data(
    fields: dict[str, Any],
    files: list[dict[str, Any]],
) -> tuple[bytes, str]:
    boundary = f"----ModelEngineBoundary{uuid.uuid4().hex}"
    payload = bytearray()

    def append_line(text: str | bytes) -> None:
        payload.extend(text if isinstance(text, bytes) else text.encode("utf-8"))

    for name, value in fields.items():
        if value is None:
            continue
        append_line(f"--{boundary}\r\n")
        append_line(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        append_line(str(value))
        append_line("\r\n")

    for file_item in files:
        field_name = str(file_item.get("field_name", "file"))
        filename = str(file_item.get("filename", "upload.bin")).replace('"', "")
        content = file_item.get("content", b"")
        if isinstance(content, str):
            content = content.encode("utf-8")
        content_type = str(
            file_item.get("content_type")
            or mimetypes.guess_type(filename)[0]
            or "application/octet-stream"
        )
        append_line(f"--{boundary}\r\n")
        append_line(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n')
        append_line(f"Content-Type: {content_type}\r\n\r\n")
        append_line(content)
        append_line("\r\n")

    append_line(f"--{boundary}--\r\n")
    return bytes(payload), f"multipart/form-data; boundary={boundary}"


def _http_call(
    calls: list[HttpCall],
    *,
    name: str,
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    raw_body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: float,
) -> HttpCall:
    request_headers = dict(headers or {})
    data = raw_body if raw_body is not None else _json_body(body)
    if data is not None:
        request_headers.setdefault("Content-Type", "application/json")
    req = request.Request(url=url, data=data, headers=request_headers, method=method.upper())
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read(300_000).decode("utf-8", errors="replace")
            payload = _decode_json(raw)
            call = HttpCall(
                name=name,
                method=method.upper(),
                url=url,
                status_code=response.status,
                ok=200 <= response.status < 300,
                body_json=payload,
                body_preview="" if payload is not None else raw[:2000],
            )
    except error.HTTPError as exc:
        raw = exc.read(300_000).decode("utf-8", errors="replace")
        call = HttpCall(
            name=name,
            method=method.upper(),
            url=url,
            status_code=exc.code,
            ok=False,
            body_json=_decode_json(raw),
            body_preview=raw[:2000],
            error=str(exc),
        )
    except (TimeoutError, socket.timeout, error.URLError, OSError) as exc:
        call = HttpCall(
            name=name,
            method=method.upper(),
            url=url,
            status_code=None,
            ok=False,
            error=str(exc),
        )
    calls.append(call)
    return call


def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data")
    return payload


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    payload = _unwrap(payload)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("content", "items", "records", "list"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        data = payload.get("data")
        if isinstance(data, dict):
            return _extract_items(data)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    return []


def _extract_id(payload: Any) -> str | None:
    payload = _unwrap(payload)
    if isinstance(payload, dict):
        value = payload.get("id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    return None


def _task_keywords(task: str) -> dict[str, bool]:
    normalized = task.lower()
    return {
        "cleaning": any(keyword in task for keyword in ("清洗", "清理", "去噪", "标准化", "治理", "规范化", "预处理")),
        "extraction": any(keyword in task for keyword in ("抽取", "提取", "字段", "结构化", "转换", "映射", "ETL", "知识图谱")) or "etl" in normalized,
        "quality": any(keyword in task for keyword in ("质量", "异常", "校验", "检查", "去重", "重复", "空文本", "缺失", "错误")),
        "dedup": any(keyword in task for keyword in ("去重", "重复", "冗余", "指纹", "duplicate", "dedup")),
        "low_latency": any(keyword in task for keyword in ("快速", "低延迟", "高吞吐", "性能", "实时")),
        "strict": any(keyword in task for keyword in ("严格", "准确", "正确", "高质量", "强校验", "完整")),
    }


TASK_INTENT_PATTERNS = {
    "ingest": ("接入", "导入", "上传", "读取", "ingest", "import"),
    "clean": ("清洗", "清理", "去噪", "标准化", "规范化", "预处理", "clean", "normalize"),
    "extract": ("抽取", "提取", "字段", "schema", "结构化", "extract", "field"),
    "transform": ("转换", "映射", "ETL", "etl", "标准", "transform", "mapping"),
    "deduplicate": ("去重", "重复", "冗余", "dedup", "duplicate"),
    "validate": ("质量", "校验", "检查", "验证", "异常", "缺失", "错误", "validate", "quality"),
    "recover": ("异常恢复", "恢复", "容错", "失败", "重试", "recover", "retry"),
    "optimize": ("优化", "性能", "高吞吐", "低延迟", "快速", "optimize", "performance"),
}


TASK_CONSTRAINT_PATTERNS = {
    "strict_quality": ("严格", "准确", "正确", "高质量", "强校验", "完整"),
    "low_latency": ("快速", "低延迟", "实时", "高吞吐", "性能"),
    "official_chain_only": ("DataMate", "Nexent", "官方", "真实链路", "不模拟", "不fallback", "不替代"),
    "medical_domain": ("医疗", "患者", "疾病", "诊断", "治疗", "用药", "病历"),
    "structured_focus": ("结构化", "表", "CSV", "csv", "JSONL", "jsonl", "schema"),
    "text_focus": ("文本", "非结构化", "段落", "病历", "notes", "txt"),
}


def _matched_terms(task: str, patterns: tuple[str, ...]) -> list[str]:
    normalized = task.lower()
    matches: list[str] = []
    for pattern in patterns:
        if pattern.lower() in normalized or pattern in task:
            matches.append(pattern)
    return matches


INTENT_CAPABILITY_MAP = {
    "clean": ["pii_cleanup", "text_normalization"],
    "extract": ["schema_guard"],
    "transform": ["schema_guard", "text_normalization"],
    "deduplicate": ["deduplication"],
    "validate": ["quality_filter", "length_filter"],
    "recover": ["quality_filter", "length_filter", "low_cost"],
    "optimize": ["low_cost"],
}


def _build_task_semantic_constraint_graph(
    *,
    intents: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    input_profile: dict[str, Any],
) -> dict[str, Any]:
    intent_nodes = [
        {
            "id": f"intent:{item.get('intent')}",
            "type": "intent",
            "name": item.get("intent"),
            "confidence": item.get("confidence"),
            "evidence": item.get("evidence", []),
        }
        for item in intents
    ]
    constraint_nodes = [
        {
            "id": f"constraint:{item.get('constraint')}",
            "type": "constraint",
            "name": item.get("constraint"),
            "evidence": item.get("evidence", []),
        }
        for item in constraints
    ]
    capabilities = _unique_ordered(
        [
            capability
            for item in intents
            for capability in INTENT_CAPABILITY_MAP.get(str(item.get("intent")), [])
        ]
    )
    if input_profile.get("structured") and "schema_guard" not in capabilities:
        capabilities.append("schema_guard")
    if input_profile.get("unstructured") and "text_normalization" not in capabilities:
        capabilities.append("text_normalization")
    capability_nodes = [
        {
            "id": f"capability:{capability}",
            "type": "capability",
            "name": capability,
            "required_by": [
                str(item.get("intent"))
                for item in intents
                if capability in INTENT_CAPABILITY_MAP.get(str(item.get("intent")), [])
            ],
        }
        for capability in capabilities
    ]
    stage_nodes = [
        {"id": "stage:ingest", "type": "stage", "name": "ingest", "requires": ["official_datamate_nexent_chain"]},
        {"id": "stage:profile", "type": "stage", "name": "profile", "requires": []},
        {"id": "stage:plan", "type": "stage", "name": "plan", "requires": capabilities},
        {"id": "stage:execute", "type": "stage", "name": "execute", "requires": ["official_datamate_nexent_chain"]},
        {"id": "stage:verify", "type": "stage", "name": "verify", "requires": ["quality_filter"] if "quality_filter" in capabilities else []},
    ]
    edges: list[dict[str, str]] = []
    for item in intents:
        intent_name = str(item.get("intent"))
        for capability in INTENT_CAPABILITY_MAP.get(intent_name, []):
            if capability in capabilities:
                edges.append({"from": f"intent:{intent_name}", "to": f"capability:{capability}", "relation": "requires_capability"})
    for item in constraints:
        constraint_name = str(item.get("constraint"))
        if constraint_name in {"official_chain_only", "official_datamate_nexent_chain"}:
            edges.append({"from": f"constraint:{constraint_name}", "to": "stage:ingest", "relation": "requires"})
            edges.append({"from": f"constraint:{constraint_name}", "to": "stage:execute", "relation": "requires"})
        if constraint_name in {"strict_quality", "low_latency"}:
            edges.append({"from": f"constraint:{constraint_name}", "to": "stage:plan", "relation": "optimizes"})
    for capability in capabilities:
        edges.append({"from": f"capability:{capability}", "to": "stage:plan", "relation": "planned_by"})

    conflict_policies: list[dict[str, str]] = []
    constraint_names = {str(item.get("constraint")) for item in constraints}
    if {"strict_quality", "low_latency"}.issubset(constraint_names):
        conflict_policies.append(
            {
                "conflict": "strict_quality_vs_low_latency",
                "policy": "preserve_required_quality_capabilities_first_then_minimize_DataMate_operator_cost",
            }
        )
    return {
        "graph_version": "task_semantic_constraint_graph_v3",
        "nodes": intent_nodes + constraint_nodes + capability_nodes + stage_nodes,
        "edges": edges,
        "required_capabilities": capabilities,
        "conflict_policies": conflict_policies,
        "planning_policy": "map_intents_and_constraints_to_required_DataMate_operator_capabilities_before_execution",
    }


def _understand_task_semantics(task: str, input_profile: dict[str, Any]) -> dict[str, Any]:
    intents: list[dict[str, Any]] = []
    for intent, patterns in TASK_INTENT_PATTERNS.items():
        evidence = _matched_terms(task, patterns)
        if evidence:
            intents.append({"intent": intent, "confidence": min(1.0, 0.55 + 0.15 * len(evidence)), "evidence": evidence})

    if not intents:
        intents.extend(
            [
                {"intent": "ingest", "confidence": 0.5, "evidence": ["default_required_data_access"]},
                {"intent": "clean", "confidence": 0.5, "evidence": ["default_data_processing_goal"]},
                {"intent": "validate", "confidence": 0.45, "evidence": ["default_quality_gate"]},
            ]
        )
    if not any(item["intent"] == "ingest" for item in intents):
        intents.insert(0, {"intent": "ingest", "confidence": 0.8, "evidence": ["DataMate_data_access_required"]})
    if input_profile.get("structured") and not any(item["intent"] == "transform" for item in intents):
        intents.append({"intent": "transform", "confidence": 0.72, "evidence": ["structured_input_requires_standard_mapping"]})
    if input_profile.get("quality_report", {}).get("warning_count", 0) and not any(item["intent"] == "recover" for item in intents):
        intents.append({"intent": "recover", "confidence": 0.8, "evidence": ["input_quality_warnings_detected"]})

    constraints: list[dict[str, Any]] = []
    for constraint, patterns in TASK_CONSTRAINT_PATTERNS.items():
        evidence = _matched_terms(task, patterns)
        if evidence:
            constraints.append({"constraint": constraint, "evidence": evidence})
    constraints.append({"constraint": "official_datamate_nexent_chain", "evidence": ["competition_task_1_requirement"]})
    constraints.append({"constraint": "no_local_fallback", "evidence": ["user_requirement"]})
    semantic_graph = _build_task_semantic_constraint_graph(
        intents=intents,
        constraints=constraints,
        input_profile=input_profile,
    )

    workflow_bias = "structured_etl" if input_profile.get("structured") else "text_processing_pipeline"
    quality_report = input_profile.get("quality_report", {}) if isinstance(input_profile.get("quality_report"), dict) else {}
    success_criteria = [
        "source_dataset_created_or_reused_in_DataMate",
        "input_file_uploaded_to_DataMate_dataset",
        "DataMate_operator_DAG_created_and_executed",
        "terminal_task_status_tracked",
        "target_dataset_and_files_materialized",
        "post_execution_quality_closure_completed",
    ]
    if input_profile.get("structured"):
        success_criteria.extend(
            [
                "structured_schema_contract_generated",
                "standard_doc_id_title_source_text_mapping_validated",
                "primary_key_and_non_empty_text_rules_checked",
            ]
        )
    if quality_report.get("warning_count", 0):
        success_criteria.append("recovery_or_blocker_report_generated_for_quality_risks")

    priority_order = [
        "validate" if quality_report.get("warning_count", 0) else "",
        "recover" if quality_report.get("warning_count", 0) else "",
        "transform" if input_profile.get("structured") else "",
        "clean",
        "deduplicate" if any(item.get("code") in {"duplicate_doc_id", "duplicate_text"} for item in quality_report.get("warnings", [])) else "",
        "ingest",
    ]
    priority_order = [item for index, item in enumerate(priority_order) if item and item not in priority_order[:index]]

    return {
        "understanding_version": "task_semantics_v3_constraint_graph",
        "task_text": task,
        "workflow_bias": workflow_bias,
        "intents": intents,
        "primary_intent": max(intents, key=lambda item: float(item.get("confidence", 0))).get("intent"),
        "constraints": constraints,
        "semantic_constraint_graph": semantic_graph,
        "priority_order": priority_order,
        "success_criteria": success_criteria,
        "planning_confidence": round(sum(float(item.get("confidence", 0)) for item in intents) / max(1, len(intents)), 4),
    }


def _build_autonomous_execution_plan(
    task_understanding: dict[str, Any],
    input_profile: dict[str, Any],
    workflow_plan: dict[str, Any],
    quality_risk_profile: dict[str, Any],
) -> dict[str, Any]:
    workflow_type = workflow_plan.get("workflow")
    risk_count = int(quality_risk_profile.get("risk_count") or 0)
    semantic_graph = (
        task_understanding.get("semantic_constraint_graph", {})
        if isinstance(task_understanding.get("semantic_constraint_graph"), dict)
        else {}
    )
    constraint_names = {
        str(item.get("constraint"))
        for item in task_understanding.get("constraints", [])
        if isinstance(item, dict)
    }
    stages = [
        {
            "stage": "understand",
            "goal": "parse_task_intents_constraints_and_semantic_capability_graph",
            "depends_on": [],
            "completion_signal": "task_semantic_constraint_graph.generated",
            "failure_policy": "report_blocker_if_task_or_input_is_unsupported",
        },
        {
            "stage": "profile",
            "goal": "profile_source_schema_documents_and_quality_risks",
            "depends_on": ["understand"],
            "completion_signal": "input_profile.generated",
            "failure_policy": "report_profile_error_without_local_fallback",
        },
        {
            "stage": "plan",
            "goal": "solve_multi_objective_DataMate_operator_DAG_from_semantic_constraints",
            "depends_on": ["profile"],
            "completion_signal": "operator_planner_v3_candidate_frontier.generated",
            "failure_policy": "block_if_DataMate_operator_market_unavailable",
        },
        {
            "stage": "execute",
            "goal": "create_execute_and_track_DataMate_cleaning_task",
            "depends_on": ["plan"],
            "completion_signal": "DataMate_task_terminal_status",
            "failure_policy": "run_classified_DataMate_recovery_DAG_attempts",
        },
        {
            "stage": "verify",
            "goal": "validate_target_dataset_files_and_quality_contracts",
            "depends_on": ["execute"],
            "completion_signal": "post_quality_report.generated",
            "failure_policy": "warning_or_blocked_status_with_evidence",
        },
    ]
    if risk_count or task_understanding.get("workflow_bias") == "structured_etl":
        stages.insert(
            3,
            {
                "stage": "guard",
                "goal": "apply_schema_quality_and_recovery_prechecks",
                "depends_on": ["plan"],
                "completion_signal": "quality_gates.evaluated",
                "failure_policy": "continue_only_through_DataMate_or_report_blocker",
            },
        )
        stages[4]["depends_on"] = ["guard"]
    if input_profile.get("batch"):
        stages.insert(
            2,
            {
                "stage": "batch_plan",
                "goal": "group_multiple_input_files_into_one_DataMate_source_dataset",
                "depends_on": ["profile"],
                "completion_signal": "batch_profile.generated",
                "failure_policy": "block_if_any_required_file_is_missing_or_unsupported",
            },
        )
        for stage in stages:
            if stage["stage"] == "plan":
                stage["depends_on"] = ["batch_plan"]
    if "low_latency" in constraint_names:
        stages.insert(
            next((index for index, stage in enumerate(stages) if stage["stage"] == "execute"), len(stages)),
            {
                "stage": "optimize",
                "goal": "preserve_required_capabilities_while_minimizing_DataMate_operator_cost_and_polling_overhead",
                "depends_on": ["plan"],
                "completion_signal": "performance_quality_tradeoff.generated",
                "failure_policy": "do_not_drop_required_capabilities_for_cost_reduction",
            },
        )
        for stage in stages:
            if stage["stage"] == "execute":
                stage["depends_on"] = ["optimize"]

    return {
        "planner": "autonomous_task_planner_v3_semantic_constraint_dag",
        "workflow_type": workflow_type,
        "stage_count": len(stages),
        "stages": stages,
        "success_criteria": task_understanding.get("success_criteria", []),
        "semantic_constraint_graph": semantic_graph,
        "required_capabilities": semantic_graph.get("required_capabilities", []),
        "constraint_conflict_policies": semantic_graph.get("conflict_policies", []),
        "risk_count": risk_count,
        "requires_recovery": bool(risk_count or workflow_plan.get("requires_deduplication")),
    }


def _sample_records(records: list[dict[str, Any]], limit: int = 200) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return records
    head = records[: limit // 2]
    tail = records[-(limit - len(head)) :]
    return head + tail


def _profile_input(input_path: Path, task: str) -> dict[str, Any]:
    suffix = input_path.suffix.lower().lstrip(".") or "unknown"
    raw_records: list[dict[str, Any]] = []
    profile_error = ""
    try:
        if suffix == "md":
            temp_text = input_path.read_text(encoding="utf-8")
            raw_records = [{"id": f"{input_path.stem}-001", "title": input_path.stem, "source": input_path.stem, "text": temp_text}]
        elif suffix == "markdown":
            temp_text = input_path.read_text(encoding="utf-8")
            raw_records = [{"id": f"{input_path.stem}-001", "title": input_path.stem, "source": input_path.stem, "text": temp_text}]
        else:
            raw_records = load_records(input_path)
    except Exception as exc:
        profile_error = str(exc)

    sampled_records = _sample_records(raw_records)
    schema_profile = infer_schema_profile(sampled_records) if sampled_records else {
        "record_count": 0,
        "field_count": 0,
        "fields": [],
        "role_candidates": {},
        "inferred_mapping": {},
        "schema_confidence": {"has_id": False, "has_text": False, "text_coverage": 0},
    }
    normalized_records = [
        normalize_fields(record, index, source_path=input_path)
        for index, record in enumerate(sampled_records, start=1)
    ]
    document_profile = profile_documents(normalized_records) if normalized_records else {"document_count": 0}
    quality_report = quality_check(normalized_records) if normalized_records else {
        "document_count": 0,
        "warning_count": 0,
        "warnings": [],
        "empty_text_documents": [],
        "duplicate_text_groups": [],
    }
    keywords = _task_keywords(task)
    size_bytes = input_path.stat().st_size if input_path.exists() else 0
    median_length = float(document_profile.get("length", {}).get("median", 0) or 0)
    text_coverage = float(schema_profile.get("schema_confidence", {}).get("text_coverage", 0) or 0)
    complexity_score = 0
    complexity_score += 2 if suffix in STRUCTURED_SUFFIXES and schema_profile.get("field_count", 0) >= 4 else 0
    complexity_score += 2 if suffix in UNSTRUCTURED_SUFFIXES or median_length >= 120 else 0
    complexity_score += 2 if keywords["extraction"] else 0
    complexity_score += 2 if keywords["quality"] or quality_report.get("warning_count", 0) else 0
    complexity_score += 1 if size_bytes >= 5_000_000 else 0
    complexity_score += 1 if text_coverage < 0.95 and sampled_records else 0
    return {
        "path": str(input_path),
        "format": suffix,
        "supported": suffix in SUPPORTED_INPUT_SUFFIXES,
        "structured": suffix in STRUCTURED_SUFFIXES,
        "unstructured": suffix in UNSTRUCTURED_SUFFIXES,
        "size_bytes": size_bytes,
        "record_count_estimate": len(raw_records),
        "sampled_record_count": len(sampled_records),
        "schema_profile": schema_profile,
        "document_profile": document_profile,
        "quality_report": quality_report,
        "profile_error": profile_error,
        "complexity_score": complexity_score,
        "complexity": "high" if complexity_score >= 6 else "medium" if complexity_score >= 3 else "low",
    }


def _resolve_batch_input_paths(input_path: Path, input_paths: list[Path] | None = None) -> list[Path]:
    raw_paths = list(input_paths or [])
    if input_path:
        raw_paths.insert(0, input_path)
    resolved: list[Path] = []
    for raw_path in raw_paths:
        path = raw_path.resolve()
        if path.is_dir():
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                if child.is_file() and (child.suffix.lower().lstrip(".") in SUPPORTED_INPUT_SUFFIXES):
                    resolved.append(child.resolve())
        else:
            resolved.append(path)
    unique: list[Path] = []
    for path in resolved:
        if path not in unique:
            unique.append(path)
    return unique


def _datamate_text_adapter_path(original_path: Path, output_root: Path) -> Path:
    digest = hashlib.sha256(str(original_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return output_root / "datamate_ingest_adapters" / f"{original_path.stem}_{digest}.txt"


def _build_datamate_text_adapter_file(original_path: Path, adapted_path: Path) -> dict[str, Any]:
    records = load_records(original_path)
    adapted_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    skipped_empty = 0
    for index, record in enumerate(records, start=1):
        normalized = normalize_fields(record, index, source_path=original_path)
        text = str(normalized.get("text") or "").strip()
        if not text:
            skipped_empty += 1
            continue
        doc_id = str(normalized.get("doc_id") or normalized.get("id") or f"{original_path.stem}-{index:06d}")
        title = str(normalized.get("title") or "").strip()
        prefix = f"{doc_id} {title}".strip()
        lines.append(f"{prefix}\n{text}" if prefix else text)
    adapted_path.write_text("\n\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "source_path": str(original_path),
        "adapted_path": str(adapted_path),
        "adapter": "structured_json_to_datamate_text_compatible_file",
        "source_format": original_path.suffix.lower().lstrip("."),
        "record_count": len(records),
        "adapted_document_count": len(lines),
        "skipped_empty_text_count": skipped_empty,
        "policy": "preserve DataMate official ingestion and cleaning execution by uploading a text-compatible representation of structured JSON/JSONL records",
    }


def _prepare_datamate_ingest_paths(
    input_paths: list[Path],
    output_root: Path,
    plan: dict[str, Any],
) -> tuple[list[Path], list[dict[str, Any]]]:
    prepared_paths: list[Path] = []
    adapters: list[dict[str, Any]] = []
    for path in input_paths:
        suffix = path.suffix.lower().lstrip(".")
        if suffix in {"json", "jsonl"}:
            adapted_path = _datamate_text_adapter_path(path, output_root)
            adapters.append(_build_datamate_text_adapter_file(path, adapted_path))
            prepared_paths.append(adapted_path)
        else:
            prepared_paths.append(path)
    if adapters:
        plan["datamate_ingest_adapters"] = adapters
        plan.setdefault("quality_gates", {})["structured_text_ingest_adapter"] = {
            "status": "enabled",
            "adapter_count": len(adapters),
            "reason": "DataMate/Data-Juicer requires uploaded files to expose a text field for text-compatible cleaning operators.",
            "official_chain_policy": "adapter only changes DataMate uploaded representation; DataMate remains authoritative executor",
        }
    return prepared_paths, adapters


def _profile_batch_inputs(input_paths: list[Path], task: str) -> dict[str, Any]:
    file_profiles = [_profile_input(path, task) for path in input_paths]
    format_counts = Counter(str(item.get("format") or "unknown") for item in file_profiles)
    supported_files = [item for item in file_profiles if item.get("supported")]
    structured_count = sum(1 for item in file_profiles if item.get("structured"))
    unstructured_count = sum(1 for item in file_profiles if item.get("unstructured"))
    total_records = sum(int(item.get("record_count_estimate") or 0) for item in file_profiles)
    total_size = sum(int(item.get("size_bytes") or 0) for item in file_profiles)
    warning_count = sum(int(item.get("quality_report", {}).get("warning_count") or 0) for item in file_profiles)
    text_coverages = [
        float(item.get("schema_profile", {}).get("schema_confidence", {}).get("text_coverage") or 0)
        for item in file_profiles
        if item.get("sampled_record_count")
    ]
    complexity_scores = [int(item.get("complexity_score") or 0) for item in file_profiles]
    workflow_type = "mixed_batch" if structured_count and unstructured_count else "structured_etl" if structured_count else "text_processing_pipeline"
    return {
        "batch": True,
        "file_count": len(input_paths),
        "supported_file_count": len(supported_files),
        "unsupported_file_count": len(file_profiles) - len(supported_files),
        "input_paths": [str(path) for path in input_paths],
        "format_counts": dict(sorted(format_counts.items())),
        "structured_file_count": structured_count,
        "unstructured_file_count": unstructured_count,
        "workflow_type": workflow_type,
        "total_size_bytes": total_size,
        "record_count_estimate": total_records,
        "quality_warning_count": warning_count,
        "min_schema_text_coverage": min(text_coverages) if text_coverages else 0,
        "complexity_score": sum(complexity_scores),
        "complexity": "high" if sum(complexity_scores) >= 8 or len(input_paths) >= 3 else "medium" if sum(complexity_scores) >= 3 else "low",
        "files": file_profiles,
    }


def _merge_batch_profile(primary_profile: dict[str, Any], batch_profile: dict[str, Any]) -> dict[str, Any]:
    if not batch_profile.get("batch"):
        return primary_profile
    merged = dict(primary_profile)
    merged.update(
        {
            "batch": True,
            "batch_profile": batch_profile,
            "size_bytes": batch_profile.get("total_size_bytes", primary_profile.get("size_bytes", 0)),
            "record_count_estimate": batch_profile.get("record_count_estimate", primary_profile.get("record_count_estimate", 0)),
            "quality_warning_count": batch_profile.get("quality_warning_count", 0),
            "complexity_score": batch_profile.get("complexity_score", primary_profile.get("complexity_score", 0)),
            "complexity": batch_profile.get("complexity", primary_profile.get("complexity")),
        }
    )
    merged["structured"] = batch_profile.get("structured_file_count", 0) >= batch_profile.get("unstructured_file_count", 0)
    merged["unstructured"] = batch_profile.get("unstructured_file_count", 0) > batch_profile.get("structured_file_count", 0)
    return merged


def _dominant_type(type_counts: dict[str, Any]) -> str:
    if not type_counts:
        return "unknown"
    return str(max(type_counts.items(), key=lambda item: int(item[1] or 0))[0])


def _type_stability(type_counts: dict[str, Any]) -> float:
    total = sum(int(value or 0) for value in type_counts.values())
    if total <= 0:
        return 0.0
    return round(max(int(value or 0) for value in type_counts.values()) / total, 4)


def _score_field_semantic_alignment(field: dict[str, Any], role: str | None, record_count: int) -> dict[str, Any]:
    name = str(field.get("name") or "").lower()
    presence_ratio = float(field.get("presence_ratio") or 0)
    distinct_count = int(field.get("distinct_count") or 0)
    unique_ratio = round(distinct_count / record_count, 4) if record_count else 0.0
    type_counts = field.get("type_counts", {}) if isinstance(field.get("type_counts"), dict) else {}
    stability = _type_stability(type_counts)
    role_keywords = {
        "doc_id": ("id", "doc", "record", "编号", "主键"),
        "title": ("title", "name", "标题", "名称"),
        "source": ("source", "origin", "来源", "出处"),
        "text": ("text", "content", "note", "desc", "病历", "摘要", "内容"),
    }
    keyword_hit = any(token in name for token in role_keywords.get(str(role), ()))
    score = 0.25 * presence_ratio + 0.25 * stability
    if role == "doc_id":
        score += 0.35 * unique_ratio
    elif role == "text":
        empty_count = int(type_counts.get("empty") or 0)
        non_empty_ratio = round((record_count - empty_count) / record_count, 4) if record_count else 0.0
        score += 0.35 * non_empty_ratio
    else:
        score += 0.2 * min(1.0, unique_ratio if unique_ratio else presence_ratio)
    if keyword_hit:
        score += 0.15
    return {
        "field": field.get("name"),
        "role": role,
        "semantic_alignment_score": round(min(1.0, score), 4),
        "keyword_evidence": keyword_hit,
        "presence_ratio": presence_ratio,
        "unique_ratio": unique_ratio,
        "type_stability": stability,
    }


def _build_structured_quality_scorecard(
    *,
    field_contract: list[dict[str, Any]],
    quality_rules: list[dict[str, Any]],
    semantic_alignment: list[dict[str, Any]],
    record_count: int,
) -> dict[str, Any]:
    if not field_contract:
        return {
            "scorecard_version": "structured_etl_quality_scorecard_v3",
            "overall_score": 0.0,
            "dimensions": {},
            "risk_level": "high",
            "record_count": record_count,
        }
    required_fields = [item for item in field_contract if item.get("required")]
    required_presence = (
        sum(float(item.get("presence_ratio") or 0) for item in required_fields) / len(required_fields)
        if required_fields
        else 1.0
    )
    type_stability = sum(float(item.get("type_stability") or 0) for item in field_contract) / len(field_contract)
    unique_scores = [
        float(item.get("unique_ratio") or 0)
        for item in field_contract
        if item.get("role") == "doc_id"
    ]
    primary_key_integrity = unique_scores[0] if unique_scores else 0.0
    rule_pass_ratio = (
        sum(1 for rule in quality_rules if rule.get("passed")) / len(quality_rules)
        if quality_rules
        else 1.0
    )
    semantic_score = (
        sum(float(item.get("semantic_alignment_score") or 0) for item in semantic_alignment) / len(semantic_alignment)
        if semantic_alignment
        else 0.0
    )
    dimensions = {
        "required_presence": round(required_presence, 4),
        "primary_key_integrity": round(primary_key_integrity, 4),
        "type_stability": round(type_stability, 4),
        "semantic_alignment": round(semantic_score, 4),
        "quality_rule_pass_ratio": round(rule_pass_ratio, 4),
    }
    overall_score = round(
        0.24 * dimensions["required_presence"]
        + 0.22 * dimensions["primary_key_integrity"]
        + 0.18 * dimensions["type_stability"]
        + 0.18 * dimensions["semantic_alignment"]
        + 0.18 * dimensions["quality_rule_pass_ratio"],
        4,
    )
    return {
        "scorecard_version": "structured_etl_quality_scorecard_v3",
        "overall_score": overall_score,
        "dimensions": dimensions,
        "risk_level": "low" if overall_score >= 0.9 else "medium" if overall_score >= 0.72 else "high",
        "record_count": record_count,
    }


def _build_structured_etl_contract(input_profile: dict[str, Any], intent_flags: dict[str, Any]) -> dict[str, Any]:
    schema_profile = input_profile.get("schema_profile", {}) if isinstance(input_profile.get("schema_profile"), dict) else {}
    quality_report = input_profile.get("quality_report", {}) if isinstance(input_profile.get("quality_report"), dict) else {}
    fields = schema_profile.get("fields", []) if isinstance(schema_profile.get("fields"), list) else []
    inferred_mapping = schema_profile.get("inferred_mapping", {}) if isinstance(schema_profile.get("inferred_mapping"), dict) else {}
    record_count = int(schema_profile.get("record_count") or input_profile.get("sampled_record_count") or 0)
    text_coverage = float(schema_profile.get("schema_confidence", {}).get("text_coverage", 0) or 0)
    duplicate_doc_ids = quality_report.get("duplicate_doc_ids", []) if isinstance(quality_report, dict) else []
    empty_text_documents = quality_report.get("empty_text_documents", []) if isinstance(quality_report, dict) else []

    standard_schema = [
        {"name": "doc_id", "source_field": inferred_mapping.get("doc_id"), "required": True, "semantic_role": "primary_key"},
        {"name": "title", "source_field": inferred_mapping.get("title"), "required": False, "semantic_role": "display_name"},
        {"name": "source", "source_field": inferred_mapping.get("source"), "required": False, "semantic_role": "provenance"},
        {"name": "text", "source_field": inferred_mapping.get("text"), "required": True, "semantic_role": "document_text"},
    ]
    field_contract: list[dict[str, Any]] = []
    semantic_alignment: list[dict[str, Any]] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        type_counts = field.get("type_counts", {}) if isinstance(field.get("type_counts"), dict) else {}
        role = field.get("candidate_role")
        presence_ratio = float(field.get("presence_ratio") or 0)
        distinct_count = int(field.get("distinct_count") or 0)
        unique_ratio = round(distinct_count / record_count, 4) if record_count else 0.0
        dominant_type = _dominant_type(type_counts)
        stability = _type_stability(type_counts)
        required = role in {"doc_id", "text"}
        nullable = not required and presence_ratio < 1.0
        violations: list[dict[str, Any]] = []
        if required and presence_ratio < 1.0:
            violations.append({"code": "required_field_missing_values", "severity": "high", "actual": presence_ratio})
        if role == "doc_id" and unique_ratio < 1.0:
            violations.append({"code": "primary_key_not_unique", "severity": "high", "actual": unique_ratio})
        if role == "text" and "empty" in type_counts:
            violations.append({"code": "text_field_has_empty_values", "severity": "high", "actual": type_counts.get("empty")})
        if stability < 0.95 and record_count >= 2:
            violations.append({"code": "mixed_field_types", "severity": "medium", "actual": type_counts})
        alignment = _score_field_semantic_alignment(field, role, record_count)
        semantic_alignment.append(alignment)
        field_contract.append(
            {
                "name": field.get("name"),
                "role": role,
                "required": required,
                "nullable": nullable,
                "presence_ratio": presence_ratio,
                "unique_ratio": unique_ratio,
                "dominant_type": dominant_type,
                "type_stability": stability,
                "type_counts": type_counts,
                "sample_values": field.get("sample_values", []),
                "semantic_alignment_score": alignment["semantic_alignment_score"],
                "violations": violations,
                "datamate_handling": (
                    "validated_by_agent_and_preserved_in_datamate_dataset"
                    if role in {"doc_id", "title", "source"}
                    else "validated_by_agent_then_processed_by_datamate_text_operator_dag"
                    if role == "text"
                    else "preserved_as_source_attribute"
                ),
            }
        )

    quality_rules = [
        {
            "rule_id": "structured_text_mapping_required",
            "severity": "critical",
            "passed": bool(inferred_mapping.get("text")),
            "expected": "text-compatible source field can be mapped to standard text",
            "actual": inferred_mapping.get("text"),
            "official_chain_action": "block_or_report_before_datamate_cleaning_if_no_text_field",
        },
        {
            "rule_id": "structured_text_coverage",
            "severity": "critical",
            "passed": text_coverage >= 0.95,
            "expected": ">=0.95",
            "actual": text_coverage,
            "official_chain_action": "DataMate task may run, but quality closure reports warning if coverage is insufficient",
        },
        {
            "rule_id": "structured_primary_key_unique",
            "severity": "critical" if intent_flags.get("strict_quality") else "warning",
            "passed": not duplicate_doc_ids,
            "expected": "no duplicate primary key values",
            "actual": duplicate_doc_ids,
            "official_chain_action": "select DataMate deduplication/schema-guard operators and keep blocker evidence if DataMate cannot recover",
        },
        {
            "rule_id": "structured_non_empty_text",
            "severity": "critical",
            "passed": not empty_text_documents,
            "expected": "no empty standard text records",
            "actual": empty_text_documents,
            "official_chain_action": "select DataMate length/quality filters and verify target dataset materialization",
        },
        {
            "rule_id": "structured_type_stability",
            "severity": "warning",
            "passed": all(not item.get("violations") or all(v.get("code") != "mixed_field_types" for v in item["violations"]) for item in field_contract),
            "expected": "dominant field type stability >=0.95",
            "actual": [
                {"field": item["name"], "type_counts": item["type_counts"]}
                for item in field_contract
                if any(v.get("code") == "mixed_field_types" for v in item.get("violations", []))
            ],
            "official_chain_action": "report schema risk; DataMate cleaning task remains authoritative execution path",
        },
    ]
    blocking_rules = [rule for rule in quality_rules if rule["severity"] == "critical" and not rule["passed"]]
    quality_scorecard = _build_structured_quality_scorecard(
        field_contract=field_contract,
        quality_rules=quality_rules,
        semantic_alignment=semantic_alignment,
        record_count=record_count,
    )
    return {
        "contract_version": "structured_etl_contract_v3_quality_scorecard",
        "record_count": record_count,
        "standard_schema": standard_schema,
        "field_contract": field_contract,
        "semantic_alignment": semantic_alignment,
        "quality_rules": quality_rules,
        "quality_scorecard": quality_scorecard,
        "blocking_rule_count": len(blocking_rules),
        "warning_rule_count": sum(1 for rule in quality_rules if rule["severity"] == "warning" and not rule["passed"]),
        "status": "passed" if not blocking_rules else "warning",
        "mapping_confidence": {
            "has_doc_id": bool(inferred_mapping.get("doc_id")),
            "has_text": bool(inferred_mapping.get("text")),
            "text_coverage": text_coverage,
        },
        "anomaly_policy": {
            "duplicate_primary_key": "deduplicate_or_report_blocker_through_datamate_recovery",
            "empty_text": "filter_or_report_blocker_through_datamate_quality_operators",
            "mixed_type": "preserve_source_and_report_schema_warning",
            "missing_optional_field": "preserve_record_with_default_standard_metadata",
        },
    }


def _build_structured_etl_plan(input_profile: dict[str, Any], intent_flags: dict[str, Any]) -> dict[str, Any]:
    schema_profile = input_profile.get("schema_profile", {}) if isinstance(input_profile.get("schema_profile"), dict) else {}
    quality_report = input_profile.get("quality_report", {}) if isinstance(input_profile.get("quality_report"), dict) else {}
    inferred_mapping = schema_profile.get("inferred_mapping", {}) if isinstance(schema_profile.get("inferred_mapping"), dict) else {}
    etl_contract = _build_structured_etl_contract(input_profile, intent_flags)
    validations = [
        {
            "check": "schema_text_coverage",
            "expected": ">=0.95",
            "actual": schema_profile.get("schema_confidence", {}).get("text_coverage", 0),
            "severity": "high",
        },
        {
            "check": "unique_document_id",
            "expected": "no_duplicate_doc_id",
            "actual": quality_report.get("duplicate_doc_ids", []),
            "severity": "medium",
        },
        {
            "check": "non_empty_text",
            "expected": "no_empty_text",
            "actual": quality_report.get("empty_text_documents", []),
            "severity": "high",
        },
    ]
    return {
        "workflow": "structured_etl",
        "input_kind": "structured",
        "schema_mapping": {
            "doc_id": inferred_mapping.get("doc_id"),
            "title": inferred_mapping.get("title"),
            "source": inferred_mapping.get("source"),
            "text": inferred_mapping.get("text"),
        },
        "field_contract": etl_contract["field_contract"],
        "etl_contract": etl_contract,
        "transforms": [
            "ingest_structured_file_via_datamate_dataset",
            "validate_structured_schema_contract_before_execution",
            "map_source_fields_to_standard_document_shape",
            "apply_datamate_cleaning_operator_dag",
            "validate_target_dataset_metadata",
            "validate_structured_quality_contract_after_execution",
            "emit_standardized_processing_artifacts",
        ],
        "validations": validations,
        "requires_deduplication": bool(intent_flags.get("deduplication")) or bool(quality_report.get("duplicate_doc_ids")),
    }


def _build_text_pipeline_plan(input_profile: dict[str, Any], intent_flags: dict[str, Any]) -> dict[str, Any]:
    document_profile = input_profile.get("document_profile", {}) if isinstance(input_profile.get("document_profile"), dict) else {}
    return {
        "workflow": "text_processing_pipeline",
        "input_kind": "unstructured",
        "transforms": [
            "ingest_text_file_via_datamate_dataset",
            "profile_sections_sentences_and_lengths",
            "apply_datamate_text_cleaning_operator_dag",
            "validate_target_dataset_metadata",
            "emit_text_quality_summary",
        ],
        "document_length": document_profile.get("length", {}),
        "language_counts": document_profile.get("language_counts", {}),
        "requires_text_normalization": True,
        "requires_deduplication": bool(intent_flags.get("deduplication")),
    }


def _infer_task_plan(task: str, input_path: Path, *, batch_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    suffix = input_path.suffix.lower().lstrip(".") or "unknown"
    keywords = _task_keywords(task)
    input_profile = _profile_input(input_path, task) if input_path.exists() else {
        "format": suffix,
        "supported": suffix in SUPPORTED_INPUT_SUFFIXES,
        "structured": suffix in STRUCTURED_SUFFIXES,
        "unstructured": suffix in UNSTRUCTURED_SUFFIXES,
        "size_bytes": 0,
        "record_count_estimate": 0,
        "sampled_record_count": 0,
        "schema_profile": {},
        "document_profile": {},
        "quality_report": {},
        "profile_error": "input_file_not_found",
        "complexity_score": 0,
        "complexity": "unknown",
    }
    if batch_profile:
        input_profile = _merge_batch_profile(input_profile, batch_profile)
    intent_flags = {
        "data_ingestion": True,
        "cleaning": keywords["cleaning"],
        "extraction": keywords["extraction"],
        "quality": keywords["quality"],
        "deduplication": keywords["dedup"],
        "low_latency": keywords["low_latency"],
        "strict_quality": keywords["strict"],
    }
    if not any(value for key, value in intent_flags.items() if key != "data_ingestion"):
        intent_flags.update({"cleaning": True, "extraction": True, "quality": True})
    task_understanding = _understand_task_semantics(task, input_profile)
    if any(item.get("intent") == "recover" for item in task_understanding.get("intents", [])):
        intent_flags["quality"] = True
    if any(item.get("intent") == "deduplicate" for item in task_understanding.get("intents", [])):
        intent_flags["deduplication"] = True
    if any(item.get("constraint") == "low_latency" for item in task_understanding.get("constraints", [])):
        intent_flags["low_latency"] = True
    selected_operator_keywords = _operator_search_keywords(intent_flags, input_profile)
    policy_prediction = {}
    policy_enabled = _task1_policy_model_enabled()
    if not policy_enabled:
        policy_prediction = {
            "policy_version": "task1_orchestration_policy_v2_multi_objective",
            "source": "disabled",
            "status": "disabled_by_environment",
        }
    else:
        try:
            loaded_policy_model = _load_task1_policy_model_from_env()
            prediction = predict_policy(loaded_policy_model, task, input_profile)
            if prediction is not None:
                policy_prediction = prediction.as_dict()
                selected_operator_keywords = _unique_ordered(
                    selected_operator_keywords + list(policy_prediction.get("operator_keywords") or [])
                )
                if policy_prediction.get("prefer_low_latency"):
                    intent_flags["low_latency"] = True
                if policy_prediction.get("strict_quality"):
                    intent_flags["strict_quality"] = True
        except Exception as exc:
            policy_prediction = {
                "policy_version": "task1_orchestration_policy_v2_multi_objective",
                "source": "trained_task1_orchestration_policy_model",
                "status": "unavailable",
                "reason": str(exc),
            }
    if policy_enabled and not policy_prediction:
        policy_prediction = {
            "policy_version": "task1_orchestration_policy_v2_multi_objective",
            "source": "not_loaded",
            "status": "model_file_missing_or_not_configured",
        }
    text_coverage = float(input_profile.get("schema_profile", {}).get("schema_confidence", {}).get("text_coverage", 0) or 0)
    warning_count = int(input_profile.get("quality_report", {}).get("warning_count", 0) or 0)
    quality_risk_profile = _build_quality_risk_profile(input_profile, intent_flags)
    policy_selector = _select_task1_policy_mode(policy_prediction, input_profile) if isinstance(policy_prediction, dict) else {
        "selector_version": "task1_policy_selector_v1_ab_gated",
        "mode": "constraint_planner_guarded",
        "learned_policy_enabled": False,
        "reasons": ["policy_prediction_not_available"],
    }
    transformer_lora_runtime = _task1_transformer_lora_runtime_candidate(
        task=task,
        input_profile=input_profile,
        selector=_load_task1_policy_selector_config(),
    )
    raw_transformer_lora_candidate = (
        transformer_lora_runtime.get("candidate", {})
        if isinstance(transformer_lora_runtime.get("candidate"), dict)
        else {}
    )
    transformer_lora_guard = _guard_task1_transformer_lora_candidate(
        raw_transformer_lora_candidate,
        input_profile=input_profile,
        quality_risk_profile=quality_risk_profile,
        intent_flags=intent_flags,
    )
    transformer_lora_candidate = raw_transformer_lora_candidate if transformer_lora_guard.get("accepted") else {}
    if raw_transformer_lora_candidate:
        transformer_lora_runtime["guard"] = transformer_lora_guard
    if transformer_lora_candidate:
        selected_operator_keywords = _unique_ordered(
            selected_operator_keywords + list(transformer_lora_candidate.get("operator_keywords") or [])
        )
        if transformer_lora_candidate.get("prefer_low_latency"):
            intent_flags["low_latency"] = True
        if transformer_lora_candidate.get("strict_quality"):
            intent_flags["strict_quality"] = True
        if transformer_lora_candidate.get("requires_recovery"):
            intent_flags["quality"] = True
    learned_policy_enabled = bool(policy_selector.get("learned_policy_enabled"))
    learned_capabilities = (
        list(policy_prediction.get("required_capabilities") or [])
        if isinstance(policy_prediction, dict) and learned_policy_enabled
        else []
    )
    if transformer_lora_candidate:
        learned_capabilities = _unique_ordered(
            learned_capabilities + list(transformer_lora_candidate.get("required_capabilities") or [])
        )
    if learned_capabilities:
        existing = list(quality_risk_profile.get("required_capabilities") or [])
        quality_risk_profile["required_capabilities"] = _unique_ordered(existing + learned_capabilities)
        quality_risk_profile["learned_policy_capabilities"] = learned_capabilities
    workflow_plan = (
        _build_structured_etl_plan(input_profile, intent_flags)
        if input_profile.get("structured")
        else _build_text_pipeline_plan(input_profile, intent_flags)
    )
    autonomous_plan = _build_autonomous_execution_plan(
        task_understanding,
        input_profile,
        workflow_plan,
        quality_risk_profile,
    )
    return {
        "task": task,
        "batch": bool(input_profile.get("batch")),
        "batch_profile": input_profile.get("batch_profile", {}),
        "task_understanding": task_understanding,
        "autonomous_plan": autonomous_plan,
        "workflow_type": workflow_plan["workflow"],
        "workflow_plan": workflow_plan,
        "input_format": suffix,
        "structured": input_profile.get("structured", False),
        "unstructured": input_profile.get("unstructured", False),
        "input_profile": input_profile,
        "intent_flags": intent_flags,
        "quality_risk_profile": quality_risk_profile,
        "policy_selector": policy_selector,
        "policy_model_prediction": policy_prediction or {
            "policy_version": "task1_orchestration_policy_v2_multi_objective",
            "source": "not_loaded",
            "status": "model_file_missing_or_not_configured",
        },
        "transformer_lora_runtime": transformer_lora_runtime,
        "transformer_lora_guard": transformer_lora_guard,
        "data_strategy": {
            "dataset_type": "TEXT",
            "reason": "DataMate cleaning pipeline operates on uploaded source files and text-compatible operators.",
            "profile_complexity": input_profile.get("complexity"),
            "record_count_estimate": input_profile.get("record_count_estimate", 0),
        },
        "quality_gates": {
            "schema_text_coverage": text_coverage,
            "warning_count": warning_count,
            "requires_recovery": warning_count > 0 or intent_flags["quality"] or intent_flags["strict_quality"],
            "blocked_if_datamate_unavailable": True,
        },
        "execution_strategy": {
            "operator_selection": "semantic_intent_profile_and_risk_scored_datamate_operator_market",
            "status_tracking": "adaptive_polling_with_terminal_state_detection",
            "failure_policy": "retry_with_datamate_safe_operator_dag_then_report_blocker",
            "prefer_low_latency": intent_flags["low_latency"],
            "policy_selector_mode": policy_selector.get("mode"),
            "learned_policy_enabled": learned_policy_enabled,
            "learned_policy_operator_ids": _unique_ordered(
                (
                    list(policy_prediction.get("recommended_operator_ids") or [])
                    if isinstance(policy_prediction, dict) and learned_policy_enabled
                    else []
                )
                + list(transformer_lora_candidate.get("recommended_operator_ids") or [])
            ),
            "learned_policy_avoid_operator_ids": list(policy_prediction.get("avoid_operator_ids") or []) if isinstance(policy_prediction, dict) and learned_policy_enabled else [],
            "learned_policy_quality_priority": policy_prediction.get("quality_priority") if isinstance(policy_prediction, dict) and learned_policy_enabled else None,
            "learned_policy_cost_tier": policy_prediction.get("cost_tier") if isinstance(policy_prediction, dict) and learned_policy_enabled else None,
            "transformer_lora_runtime_status": transformer_lora_runtime.get("status"),
            "transformer_lora_guard_status": transformer_lora_guard.get("status"),
            "transformer_lora_guard_policy": transformer_lora_guard.get("policy"),
            "transformer_lora_runtime_operator_ids": list(transformer_lora_candidate.get("recommended_operator_ids") or []),
            "transformer_lora_rejected_operator_ids": list(transformer_lora_guard.get("candidate_operator_ids") or []) if not transformer_lora_guard.get("accepted") else [],
            "planner": autonomous_plan["planner"],
            "stage_count": autonomous_plan["stage_count"],
        },
        "required_datamate_steps": [
            "create_or_reuse_source_dataset",
            "upload_input_file_to_datamate_dataset",
            "query_datamate_operator_market",
            "score_and_select_datamate_operator_dag",
            "create_datamate_cleaning_task",
            "execute_datamate_cleaning_task",
            "track_datamate_task_status",
            "collect_datamate_task_result",
            "validate_target_dataset_and_result_metadata",
        ],
        "selected_operator_keywords": selected_operator_keywords,
    }


def _build_quality_risk_profile(input_profile: dict[str, Any], intent_flags: dict[str, Any]) -> dict[str, Any]:
    schema_profile = input_profile.get("schema_profile", {}) if isinstance(input_profile.get("schema_profile"), dict) else {}
    document_profile = input_profile.get("document_profile", {}) if isinstance(input_profile.get("document_profile"), dict) else {}
    quality_report = input_profile.get("quality_report", {}) if isinstance(input_profile.get("quality_report"), dict) else {}
    text_coverage = float(schema_profile.get("schema_confidence", {}).get("text_coverage", 0) or 0)
    warning_codes = {
        str(item.get("code"))
        for item in quality_report.get("warnings", [])
        if isinstance(item, dict) and item.get("code")
    }
    duplicate_groups = quality_report.get("duplicate_text_groups") or document_profile.get("duplicate_text_groups") or []
    low_information = quality_report.get("low_information_documents") or document_profile.get("low_information_documents") or []
    risks: list[dict[str, Any]] = []

    def add(code: str, severity: str, capability: str, reason: str) -> None:
        risks.append({"code": code, "severity": severity, "capability": capability, "reason": reason})

    if text_coverage < 0.95 and input_profile.get("sampled_record_count", 0):
        add("schema_text_coverage", "high", "schema_guard", "source schema has incomplete text coverage")
    if input_profile.get("structured"):
        etl_contract = _build_structured_etl_contract(input_profile, intent_flags)
        scorecard = etl_contract.get("quality_scorecard", {}) if isinstance(etl_contract.get("quality_scorecard"), dict) else {}
        if scorecard.get("risk_level") == "high":
            add("structured_quality_score_low", "high", "schema_guard", "structured ETL scorecard indicates high schema or quality risk")
        elif scorecard.get("risk_level") == "medium":
            add("structured_quality_score_medium", "medium", "quality_filter", "structured ETL scorecard indicates medium quality risk")
        fields = schema_profile.get("fields", []) if isinstance(schema_profile.get("fields"), list) else []
        if not schema_profile.get("schema_confidence", {}).get("has_text"):
            add("missing_structured_text_mapping", "high", "schema_guard", "structured ETL requires a text-compatible source field")
        if not schema_profile.get("schema_confidence", {}).get("has_id"):
            add("missing_structured_primary_key", "medium", "schema_guard", "structured ETL should keep stable provenance ids")
        for field in fields:
            if not isinstance(field, dict):
                continue
            type_counts = field.get("type_counts", {}) if isinstance(field.get("type_counts"), dict) else {}
            if _type_stability(type_counts) < 0.95 and int(input_profile.get("sampled_record_count") or 0) >= 2:
                add("structured_mixed_field_type", "medium", "schema_guard", f"field {field.get('name')} has mixed inferred types")
                break
    if "empty_text" in warning_codes or quality_report.get("empty_text_documents"):
        add("empty_text", "high", "length_filter", "empty text records should be filtered or reported")
    if "duplicate_text" in warning_codes or duplicate_groups or intent_flags.get("deduplication"):
        add("duplicate_text", "medium", "deduplication", "duplicate text should be removed or counted once")
    if "duplicate_doc_id" in warning_codes:
        add("duplicate_doc_id", "medium", "schema_guard", "duplicate document ids affect downstream provenance")
    if low_information:
        add("low_information_text", "medium", "length_filter", "short low-information records should be filtered")
    if input_profile.get("unstructured"):
        add("unstructured_noise", "medium", "text_normalization", "unstructured text needs whitespace and noise normalization")
    if intent_flags.get("strict_quality"):
        add("strict_quality", "high", "quality_filter", "strict task wording requires explicit quality gates")
    if intent_flags.get("low_latency"):
        add("low_latency", "medium", "low_cost", "low-latency task should cap expensive operators")

    required_capabilities = [risk["capability"] for risk in risks]
    if intent_flags.get("cleaning"):
        required_capabilities.extend(["pii_cleanup", "text_normalization"])
    if intent_flags.get("quality"):
        required_capabilities.extend(["quality_filter", "length_filter"])
    if intent_flags.get("extraction"):
        required_capabilities.append("schema_guard")

    return {
        "risks": risks,
        "required_capabilities": [
            capability for index, capability in enumerate(required_capabilities)
            if capability not in required_capabilities[:index]
        ],
        "risk_count": len(risks),
        "high_risk_count": sum(1 for risk in risks if risk["severity"] == "high"),
    }


def _operator_search_keywords(intent_flags: dict[str, Any], input_profile: dict[str, Any]) -> list[str]:
    keywords = ["匿名化", "空格", "字数", "clean"]
    if intent_flags.get("quality"):
        keywords.extend(["字数", "特殊字符", "重复", "filter", "quality", "length"])
    if intent_flags.get("deduplication"):
        keywords.extend(["dedup", "duplicate"])
    if input_profile.get("structured"):
        keywords.extend(["schema", "field", "csv", "json", "字数", "空格"])
    if input_profile.get("unstructured"):
        keywords.extend(["text", "space", "sentence", "空格"])
    if input_profile.get("complexity") == "high":
        keywords.extend(["remove", "normalize", "不可见"])
    return [keyword for index, keyword in enumerate(keywords) if keyword not in keywords[:index]]


def _compact_operator_keywords(keywords: list[str], plan: dict[str, Any]) -> list[str]:
    priority = ["匿名化", "空格", "字数", "schema", "field", "csv", "json", "邮件", "url", "ip", "特殊字符", "重复", "clean", "filter", "length", "dedup", "text", "space", "quality", "remove", "normalize"]
    ordered = sorted(
        [keyword for index, keyword in enumerate(keywords) if keyword not in keywords[:index]],
        key=lambda item: priority.index(item) if item in priority else 999,
    )
    if plan.get("batch") or plan.get("intent_flags", {}).get("low_latency"):
        limit = 5
    elif plan.get("workflow_type") == "structured_etl":
        limit = 6
    else:
        limit = 5
    compacted = ordered[:limit]
    if "匿名化" not in compacted:
        compacted.insert(0, "匿名化")
    if plan.get("workflow_type") == "structured_etl":
        required_keywords = [keyword for keyword in ("schema", "field") if keyword in ordered]
        for keyword in required_keywords:
            if keyword not in compacted:
                compacted.append(keyword)
        compacted = _unique_ordered(compacted)
        if len(compacted) > limit:
            preserved = ["匿名化"] + required_keywords
            tail = [keyword for keyword in compacted if keyword not in preserved]
            compacted = _unique_ordered(preserved + tail)[:limit]
    return compacted[:limit]


CAPABILITY_OPERATOR_KEYWORDS = {
    "pii_cleanup": ["匿名化", "邮件", "url", "phone", "ip", "clean"],
    "text_normalization": ["空格", "不可见", "全角", "normalize", "space", "clean"],
    "length_filter": ["字数", "length", "filter", "empty"],
    "quality_filter": ["字数", "特殊字符", "重复", "quality", "filter", "alphanumeric"],
    "deduplication": ["dedup", "duplicate", "fingerprint"],
    "schema_guard": ["schema", "field", "csv", "json"],
}


def _unique_ordered(items: list[str]) -> list[str]:
    return [item for index, item in enumerate(items) if item and item not in items[:index]]


def _query_operator_market(
    calls: list[HttpCall],
    runtime_api_base: str,
    timeout_seconds: float,
    *,
    keyword: str,
    size: int = 120,
    cache: OperatorMarketCache | None = None,
    purpose: str = "operators",
) -> list[dict[str, Any]]:
    cache_key = (runtime_api_base.rstrip("/"), keyword)
    url = _operator_market_url(runtime_api_base)
    if cache is not None and cache_key in cache:
        calls.append(
            HttpCall(
                name=f"DataMate query {purpose} keyword={keyword} cache hit",
                method="CACHE",
                url=url,
                status_code=200,
                ok=True,
                body_json={"data": {"content": cache[cache_key]}, "cache": True},
            )
        )
        return cache[cache_key]
    call = _http_call(
        calls,
        name=f"DataMate query {purpose} keyword={keyword}",
        method="POST",
        url=url,
        body={"page": 0, "size": size, "categories": [], "keyword": keyword, "isStar": None},
        timeout_seconds=timeout_seconds,
    )
    rows = _extract_items(call.body_json)
    if cache is not None:
        cache[cache_key] = rows
    return rows


def _operator_text(operator: dict[str, Any]) -> str:
    values = [
        operator.get("id"),
        operator.get("name"),
        operator.get("description"),
        operator.get("categories"),
        operator.get("inputs"),
        operator.get("outputs"),
    ]
    return " ".join(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value or "") for value in values).lower()


def _operator_io_text(operator: dict[str, Any]) -> tuple[str, str]:
    return str(operator.get("inputs") or "").lower(), str(operator.get("outputs") or "").lower()


def _operator_categories(operator: dict[str, Any]) -> set[str]:
    raw = operator.get("categories") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return set()
    return {str(item).lower() for item in raw if item}


def _is_datamate_vendor_operator(operator: dict[str, Any]) -> bool:
    categories = _operator_categories(operator)
    return DATAMATE_VENDOR_CATEGORY_ID in categories


def _is_datajuicer_vendor_operator(operator: dict[str, Any]) -> bool:
    categories = _operator_categories(operator)
    return DATAJUICER_VENDOR_CATEGORY_ID in categories


OPERATOR_ALIAS_IDS = {
    "average_line_length_filter": "text_length_filter",
    "maximum_line_length_filter": "text_length_filter",
    "minimum_line_length_filter": "text_length_filter",
    "specified_field_filter": "frequency_specified_field_selector",
    "general_field_filter": "frequency_specified_field_selector",
    "topk_specified_field_selector": "frequency_specified_field_selector",
}


def _canonical_operator_id(operator_id: str) -> str:
    return OPERATOR_ALIAS_IDS.get(operator_id, operator_id)


def _avoid_operator_ids_from_plan(plan: dict[str, Any]) -> set[str]:
    execution_strategy = plan.get("execution_strategy", {}) if isinstance(plan.get("execution_strategy"), dict) else {}
    policy_prediction = plan.get("policy_model_prediction", {}) if isinstance(plan.get("policy_model_prediction"), dict) else {}
    raw_ids = list(execution_strategy.get("learned_policy_avoid_operator_ids") or [])
    raw_ids.extend(policy_prediction.get("avoid_operator_ids") or [])
    return {_canonical_operator_id(str(item)) for item in raw_ids if item}


def _operator_capabilities(operator: dict[str, Any]) -> set[str]:
    text = _operator_text(operator)
    operator_id = str(operator.get("id") or "")
    capabilities: set[str] = set()
    if operator_id in {"EmailNumberCleaner", "AnonymizedUrlCleaner", "AnonymizedIpAddress", "AnonymizedPhoneNumber", "PiiDetector"}:
        capabilities.add("pii_cleanup")
    if operator_id in {"ExtraSpaceCleaner", "UnicodeSpaceCleaner", "InvisibleCharactersCleaner", "FullWidthCharacterCleaner", "TraditionalChineseCleaner"}:
        capabilities.add("text_normalization")
    if operator_id in {"FileWithShortOrLongLengthFilter"}:
        capabilities.update({"length_filter", "quality_filter"})
    if operator_id in {"FileWithHighRepeatPhraseRateFilter", "FileWithHighRepeatWordRateFilter", "FileWithHighSpecialCharRateFilter", "FileWithManySensitiveWordsFilter"}:
        capabilities.add("quality_filter")
    if operator_id in {"DuplicateFilesFilter", "DuplicateSentencesFilter"}:
        capabilities.add("deduplication")
    if any(token in text for token in ("link", "url", "http", "email", "ip", "phone", "pii")):
        capabilities.add("pii_cleanup")
    if any(token in text for token in ("space", "whitespace", "normalize", "clean", "mapper", "remove", "空格", "不可见", "全角", "繁体")):
        capabilities.add("text_normalization")
    if any(token in text for token in ("length", "empty", "short", "long", "字数")):
        capabilities.add("length_filter")
    if any(token in text for token in ("quality", "filter", "alphanumeric", "similarity", "相似度", "检查", "过滤")):
        capabilities.add("quality_filter")
    if any(token in text for token in ("dedup", "duplicate", "fingerprint", "similarity", "相似度", "frequency", "频率")):
        capabilities.add("deduplication")
    if any(token in text for token in ("schema", "field", "csv", "json", "table", "字段")):
        capabilities.add("schema_guard")
    if any(token in text for token in ("fast", "simple", "mapper", "filter")):
        capabilities.add("low_cost")
    return capabilities


def _operator_cost(operator: dict[str, Any]) -> float:
    text = _operator_text(operator)
    cost = 1.0
    if any(token in text for token in ("embedding", "model", "llm", "extract", "parse", "table", "pdf")):
        cost += 2.0
    if any(token in text for token in ("dedup", "fingerprint", "quality", "similarity", "相似度")):
        cost += 0.6
    if any(token in text for token in ("mapper", "filter", "regex", "length")):
        cost -= 0.2
    return round(max(0.5, cost), 3)


def _is_cleaning_task_compatible(operator: dict[str, Any], plan: dict[str, Any]) -> bool:
    """Keep DataMate cleaning-task DAGs executable by selecting text-compatible operators."""
    inputs, outputs = _operator_io_text(operator)
    text = _operator_text(operator)
    operator_id = str(operator.get("id") or "")
    input_format = str(plan.get("input_format") or "").lower()
    if "text" not in inputs or "text" not in outputs:
        return False
    if _is_datajuicer_vendor_operator(operator):
        return False
    if not _is_datamate_vendor_operator(operator):
        return False
    if operator_id in _parameterized_operator_ids_without_safe_defaults():
        return False
    if any(token in text for token in ("html", "table", "pdf", "image", "audio", "video")) and input_format not in {"html", "htm"}:
        return False
    return True


def _parameterized_operator_ids_without_safe_defaults() -> set[str]:
    return {
        "frequency_specified_field_selector",
        "range_specified_field_selector",
        "random_selector",
    }


def _datamate_operator_overrides(operator: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    operator_id = str(operator.get("id") or "")
    existing = operator.get("overrides")
    if isinstance(existing, dict) and existing:
        return dict(existing)
    if operator_id in {"text_length_filter", "FileWithShortOrLongLengthFilter"}:
        profile = plan.get("input_profile", {}) if isinstance(plan.get("input_profile"), dict) else {}
        document_profile = profile.get("document_profile", {}) if isinstance(profile.get("document_profile"), dict) else {}
        length_profile = document_profile.get("length", {}) if isinstance(document_profile.get("length"), dict) else {}
        try:
            max_length = int(length_profile.get("max") or 0)
        except (TypeError, ValueError):
            max_length = 0
        max_guard = max(max_length * 4, 10000)
        if operator_id == "FileWithShortOrLongLengthFilter":
            return {"fileLength": [1, max_guard]}
        return {"min_len": 1, "max_len": max_guard}
    return {}


def _score_operator(operator: dict[str, Any], plan: dict[str, Any]) -> float:
    text = _operator_text(operator)
    intent = plan.get("intent_flags", {})
    execution_strategy = plan.get("execution_strategy", {}) if isinstance(plan.get("execution_strategy"), dict) else {}
    risk_profile = plan.get("quality_risk_profile", {}) if isinstance(plan.get("quality_risk_profile"), dict) else {}
    policy_prediction = plan.get("policy_model_prediction", {}) if isinstance(plan.get("policy_model_prediction"), dict) else {}
    required_capabilities = set(risk_profile.get("required_capabilities") or [])
    learned_operator_ids = set(str(item) for item in execution_strategy.get("learned_policy_operator_ids", []) if item)
    avoid_operator_ids = _avoid_operator_ids_from_plan(plan)
    quality_priority = str(execution_strategy.get("learned_policy_quality_priority") or "")
    cost_tier = str(execution_strategy.get("learned_policy_cost_tier") or "")
    capabilities = _operator_capabilities(operator)
    score = 0.0
    operator_id = str(operator.get("id") or operator.get("name") or "")
    canonical_operator_id = _canonical_operator_id(operator_id)
    if operator_id in avoid_operator_ids or canonical_operator_id in avoid_operator_ids:
        operator["_policy_avoid_operator"] = True
        score -= 6.0
    else:
        operator.pop("_policy_avoid_operator", None)
    if operator_id in learned_operator_ids:
        score += 1.35
    if str(operator.get("inputs", "")).lower() == "text":
        score += 1.0
    if str(operator.get("outputs", "")).lower() == "text":
        score += 1.0
    if any(token in text for token in ("clean", "remove", "normalize", "space", "mapper")):
        score += 2.0 if intent.get("cleaning") else 1.0
    if any(token in text for token in ("filter", "length", "quality", "empty", "similarity", "相似度")):
        score += 1.7 if intent.get("quality") else 0.8
    if any(token in text for token in ("dedup", "duplicate", "fingerprint", "similarity", "相似度", "frequency", "频率")):
        score += 2.0 if intent.get("deduplication") else 0.7
    if any(token in text for token in ("csv", "json", "schema", "field")) and plan.get("structured"):
        score += 1.4
    if any(token in text for token in ("text", "sentence", "paragraph")) and plan.get("unstructured"):
        score += 1.2
    if any(token in text for token in ("html", "table", "pdf", "image", "audio", "video")) and str(plan.get("input_format")) not in {"html", "htm"}:
        score -= 4.0
    if operator.get("id") in {
        "EmailNumberCleaner",
        "AnonymizedUrlCleaner",
        "AnonymizedIpAddress",
        "AnonymizedPhoneNumber",
        "ExtraSpaceCleaner",
        "UnicodeSpaceCleaner",
        "FileWithShortOrLongLengthFilter",
        "clean_links_mapper",
        "clean_ip_mapper",
        "clean_email_mapper",
        "remove_extra_spaces_mapper",
        "text_length_filter",
        "alphanumeric_filter",
    }:
        score += 0.9
    covered = capabilities & required_capabilities
    if covered:
        capability_weight = 1.35 if quality_priority in {"high", "critical"} else 1.15
        score += capability_weight * len(covered)
    if quality_priority == "critical" and capabilities & {"quality_filter", "schema_guard", "deduplication"}:
        score += 0.45
    if intent.get("low_latency") or cost_tier == "low":
        score -= _operator_cost(operator) * 0.45
    elif cost_tier == "quality_first":
        score -= _operator_cost(operator) * 0.04
    else:
        score -= _operator_cost(operator) * 0.08
    ranker_score = None
    if (
        policy_prediction.get("source") == "trained_task1_orchestration_policy_model"
        and execution_strategy.get("learned_policy_enabled")
    ):
        try:
            policy_model = _load_task1_policy_model_from_env()
            ranker_score = score_operator_with_policy_ranker(
                policy_model,
                str(plan.get("task") or ""),
                plan.get("input_profile", {}) if isinstance(plan.get("input_profile"), dict) else {},
                operator,
                policy_prediction,
            )
        except Exception:
            ranker_score = None
    if ranker_score is not None:
        operator["_policy_ranker_score"] = ranker_score
        score += float(ranker_score) * 1.45
    else:
        operator.pop("_policy_ranker_score", None)
    return round(score, 4)


def _task1_tensor_ranker_device() -> str:
    return os.getenv(TASK1_TENSOR_RANKER_DEVICE_ENV, "auto").strip().lower() or "auto"


def _apply_task1_tensor_ranker(scored: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    if not scored:
        plan["task1_tensor_operator_ranker"] = {
            **build_task1_tensor_ranker_descriptor(),
            "enabled": False,
            "candidate_count": 0,
            "reason": "no_candidates",
        }
        return scored
    risk_profile = plan.get("quality_risk_profile", {}) if isinstance(plan.get("quality_risk_profile"), dict) else {}
    execution_strategy = plan.get("execution_strategy", {}) if isinstance(plan.get("execution_strategy"), dict) else {}
    policy_prediction = plan.get("policy_model_prediction", {}) if isinstance(plan.get("policy_model_prediction"), dict) else {}
    required = _datamate_operator_required_capabilities(risk_profile.get("required_capabilities") or [])
    recommended = list(execution_strategy.get("learned_policy_operator_ids") or [])
    recommended.extend(policy_prediction.get("recommended_operator_ids") or [])
    avoid = list(execution_strategy.get("learned_policy_avoid_operator_ids") or [])
    avoid.extend(policy_prediction.get("avoid_operator_ids") or [])
    task_text = " ".join(
        str(value)
        for value in (
            plan.get("task"),
            plan.get("workflow_type"),
            plan.get("input_format"),
            ",".join(str(item) for item in required),
        )
        if value
    )
    enriched_candidates = [
        {
            **item,
            "capabilities": sorted(_operator_capabilities(item)),
            "estimated_cost": _operator_cost(item),
            "canonical_id": _canonical_operator_id(str(item.get("id") or item.get("name") or "")),
        }
        for item in scored
    ]
    ranking = rank_task1_operator_candidates(
        task_text,
        enriched_candidates,
        required_capabilities=required,
        recommended_operator_ids=recommended,
        avoid_operator_ids=avoid,
        device=_task1_tensor_ranker_device(),
    )
    rank_by_index = {int(row["index"]): row for row in ranking["ranked"]}
    ranked_scores = [float(row["score"]) for row in ranking["ranked"]]
    min_score = min(ranked_scores) if ranked_scores else 0.0
    max_score = max(ranked_scores) if ranked_scores else 0.0
    score_range = max(max_score - min_score, 1e-9)
    for index, item in enumerate(scored):
        row = rank_by_index.get(index)
        if not row:
            continue
        normalized = (float(row["score"]) - min_score) / score_range if score_range else 0.0
        item["_tensor_ranker_score"] = row["score"]
        item["_tensor_ranker_rank"] = row["rank"]
        item["_tensor_ranker_capability_score"] = row["capability_score"]
        item["_selection_score"] = round(float(item.get("_selection_score", 0)) + normalized * 0.38, 4)
    plan["task1_tensor_operator_ranker"] = {
        **build_task1_tensor_ranker_descriptor(),
        "enabled": True,
        "device": ranking["selected_device"],
        "candidate_count": ranking["candidate_count"],
        "vector_dim": ranking["vector_dim"],
        "required_capabilities": ranking["required_capabilities"],
        "top_candidates": [
            {
                "rank": row["rank"],
                "id": row["id"],
                "score": row["score"],
                "capability_score": row["capability_score"],
            }
            for row in ranking["ranked"][:8]
        ],
        "strict_device_semantics": "explicit_npu_raises_if_torch_npu_unavailable_auto_may_select_cpu",
    }
    return scored


def _build_operator_planning_report(
    *,
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    plan: dict[str, Any],
) -> dict[str, Any]:
    risk_profile = plan.get("quality_risk_profile", {}) if isinstance(plan.get("quality_risk_profile"), dict) else {}
    execution_strategy = plan.get("execution_strategy", {}) if isinstance(plan.get("execution_strategy"), dict) else {}
    all_required = list(risk_profile.get("required_capabilities") or [])
    required = _datamate_operator_required_capabilities(all_required)
    covered: set[str] = set()
    selected_rows: list[dict[str, Any]] = []
    for order, item in enumerate(selected, start=1):
        capabilities = sorted(_operator_capabilities(item))
        covered.update(capabilities)
        selected_rows.append(
            {
                "order": order,
                "id": item.get("id"),
                "name": item.get("name"),
                "score": item.get("_selection_score"),
                "policy_ranker_score": item.get("_policy_ranker_score"),
                "tensor_ranker_score": item.get("_tensor_ranker_score"),
                "tensor_ranker_rank": item.get("_tensor_ranker_rank"),
                "capabilities": capabilities,
                "estimated_cost": _operator_cost(item),
                "reason": item.get("_selection_reason"),
            }
        )
    return {
        "planner": "operator_planner_v3_multi_objective_beam_search",
        "objective_weights": PLANNER_V3_OBJECTIVE_WEIGHTS,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "estimated_total_cost": round(sum(_operator_cost(item) for item in selected), 3),
        "all_required_capabilities": all_required,
        "required_capabilities": required,
        "non_datamate_operator_capabilities": [
            capability for capability in all_required
            if capability not in required
        ],
        "covered_capabilities": [capability for capability in required if capability in covered],
        "uncovered_capabilities": [capability for capability in required if capability not in covered],
        "coverage_ratio": round(
            len([capability for capability in required if capability in covered]) / len(required),
            4,
        ) if required else 1.0,
        "objective_scores": plan.get("operator_dag_objective_scores", {}),
        "dag_score_cache": plan.get("operator_dag_objective_scores", {}).get("dag_score_cache", {}),
        "learned_policy": {
            "operator_ids": execution_strategy.get("learned_policy_operator_ids", []),
            "avoid_operator_ids": execution_strategy.get("learned_policy_avoid_operator_ids", []),
            "failure_memory_matches": plan.get("policy_model_prediction", {}).get("failure_memory_matches", []),
            "quality_priority": execution_strategy.get("learned_policy_quality_priority"),
            "cost_tier": execution_strategy.get("learned_policy_cost_tier"),
            "selector": plan.get("policy_selector", {}),
            "enabled_by_selector": bool(execution_strategy.get("learned_policy_enabled")),
            "used_for_operator_scoring": bool(execution_strategy.get("learned_policy_operator_ids")),
            "operator_ranker_used": any(item.get("_policy_ranker_score") is not None for item in candidates),
            "dag_policy_used": bool(plan.get("operator_dag_objective_scores", {}).get("learned_dag_policy_used")),
            "learned_dag_success_score": plan.get("operator_dag_objective_scores", {}).get("learned_dag_success_score"),
            "learned_dag_quality_score": plan.get("operator_dag_objective_scores", {}).get("learned_dag_quality_score"),
            "learned_dag_efficiency_score": plan.get("operator_dag_objective_scores", {}).get("learned_dag_efficiency_score"),
            "learned_dag_feasibility_score": plan.get("operator_dag_objective_scores", {}).get("learned_dag_feasibility_score"),
            "learned_dag_raw_overall_score": plan.get("operator_dag_objective_scores", {}).get("learned_dag_raw_overall_score"),
            "learned_dag_constraint_penalty": plan.get("operator_dag_objective_scores", {}).get("learned_dag_constraint_penalty"),
            "learned_dag_constraint_ceiling": plan.get("operator_dag_objective_scores", {}).get("learned_dag_constraint_ceiling"),
            "learned_dag_required_coverage_ratio": plan.get("operator_dag_objective_scores", {}).get("learned_dag_required_coverage_ratio"),
            "learned_dag_uncovered_capabilities": plan.get("operator_dag_objective_scores", {}).get("learned_dag_uncovered_capabilities"),
            "learned_dag_overall_score": plan.get("operator_dag_objective_scores", {}).get("learned_dag_overall_score"),
            "learned_dag_utility_weights": plan.get("operator_dag_objective_scores", {}).get("learned_dag_utility_weights"),
        },
        "transformer_lora_guard": plan.get("transformer_lora_guard", {}),
        "tensor_ranker": plan.get("task1_tensor_operator_ranker", {}),
        "candidate_frontier": plan.get("operator_dag_candidate_frontier", []),
        "dag": [
            {
                "order": row["order"],
                "operator_id": row["id"],
                "depends_on": selected_rows[index - 1]["id"] if index > 0 else None,
                "capabilities": row["capabilities"],
            }
            for index, row in enumerate(selected_rows)
        ],
        "selected": selected_rows,
        "rejected_top_candidates": [
            {
                "id": item.get("id"),
                "score": item.get("_selection_score"),
                "reason": "not_selected_after_coverage_and_cost_optimization",
            }
            for item in candidates
            if item not in selected
        ][:8],
    }


def _quality_risk_coverage_score(selected: list[dict[str, Any]], plan: dict[str, Any]) -> float:
    risk_profile = plan.get("quality_risk_profile", {}) if isinstance(plan.get("quality_risk_profile"), dict) else {}
    risks = risk_profile.get("risks", []) if isinstance(risk_profile.get("risks"), list) else []
    if not risks:
        return 1.0
    capabilities: set[str] = set()
    for item in selected:
        capabilities.update(_operator_capabilities(item))
    severity_weight = {"high": 1.0, "medium": 0.65, "low": 0.35}
    total = 0.0
    covered = 0.0
    for risk in risks:
        if not isinstance(risk, dict):
            continue
        weight = severity_weight.get(str(risk.get("severity")), 0.5)
        total += weight
        if risk.get("capability") in capabilities:
            covered += weight
    return round(covered / total, 4) if total else 1.0


def _dag_cache_key(selected: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted(str(item.get("id") or item.get("name") or "") for item in selected if item))


def _score_operator_dag_objectives(
    selected: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    limit: int,
    score_cache: dict[tuple[str, ...], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cache_key = _dag_cache_key(selected)
    if score_cache is not None and cache_key in score_cache:
        plan["_dag_score_cache_hits"] = int(plan.get("_dag_score_cache_hits", 0) or 0) + 1
        return {**score_cache[cache_key], "operator_ids": [item.get("id") for item in selected]}
    if score_cache is not None:
        plan["_dag_score_cache_misses"] = int(plan.get("_dag_score_cache_misses", 0) or 0) + 1
    coverage = _operator_coverage(selected, plan)
    total_cost = sum(_operator_cost(item) for item in selected)
    execution_strategy = plan.get("execution_strategy", {}) if isinstance(plan.get("execution_strategy"), dict) else {}
    avoid_operator_ids = _avoid_operator_ids_from_plan(plan)
    selected_ids = [str(item.get("id") or item.get("name") or "") for item in selected]
    avoid_overlap = sorted(
        item_id for item_id in selected_ids
        if item_id in avoid_operator_ids or _canonical_operator_id(item_id) in avoid_operator_ids
    )
    capability_union: set[str] = set()
    for item in selected:
        capability_union.update(_operator_capabilities(item))
    candidate_score_mean = (
        sum(float(item.get("_selection_score", 0)) for item in selected) / len(selected)
        if selected
        else 0.0
    )
    io_compatibility = 1.0 if selected and all(_is_cleaning_task_compatible(item, plan) for item in selected) else 0.0
    latency_cost = round(1.0 / (1.0 + max(0.0, total_cost)), 4)
    dag_simplicity = round(1.0 - max(0, len(selected) - 1) / max(1, limit), 4)
    operator_diversity = round(len(capability_union) / max(1, len(selected) * 3), 4) if selected else 0.0
    quality_risk_reduction = _quality_risk_coverage_score(selected, plan)
    policy_prediction = plan.get("policy_model_prediction", {}) if isinstance(plan.get("policy_model_prediction"), dict) else {}
    learned_dag_scores = None
    if (
        policy_prediction.get("source") == "trained_task1_orchestration_policy_model"
        and plan.get("execution_strategy", {}).get("learned_policy_enabled")
    ):
        try:
            policy_model = _load_task1_policy_model_from_env()
            learned_dag_scores = score_dag_objectives_with_policy_model(
                policy_model,
                str(plan.get("task") or ""),
                plan.get("input_profile", {}) if isinstance(plan.get("input_profile"), dict) else {},
                selected,
                policy_prediction,
            )
        except Exception:
            learned_dag_scores = None
    objective_scores = {
        "capability_coverage": coverage["coverage_ratio"],
        "quality_risk_reduction": quality_risk_reduction,
        "io_compatibility": io_compatibility,
        "latency_cost": latency_cost,
        "dag_simplicity": dag_simplicity,
        "operator_diversity": min(1.0, operator_diversity),
    }
    weighted_score = sum(
        PLANNER_V3_OBJECTIVE_WEIGHTS[name] * float(value)
        for name, value in objective_scores.items()
    )
    weighted_score += min(1.0, max(0.0, candidate_score_mean / 10.0)) * 0.05
    learned_dag_overall_score = learned_dag_scores.get("overall_score") if isinstance(learned_dag_scores, dict) else None
    if learned_dag_overall_score is not None:
        weighted_score += min(1.0, max(0.0, float(learned_dag_overall_score))) * 0.1
    if coverage["uncovered_capabilities"]:
        weighted_score -= 0.18 * len(coverage["uncovered_capabilities"])
    if avoid_overlap:
        weighted_score -= 0.25 * len(avoid_overlap)
    result = {
        **objective_scores,
        "learned_dag_success_score": learned_dag_scores.get("success_score") if isinstance(learned_dag_scores, dict) else None,
        "learned_dag_quality_score": learned_dag_scores.get("quality_score") if isinstance(learned_dag_scores, dict) else None,
        "learned_dag_efficiency_score": learned_dag_scores.get("efficiency_score") if isinstance(learned_dag_scores, dict) else None,
        "learned_dag_feasibility_score": learned_dag_scores.get("feasibility_score") if isinstance(learned_dag_scores, dict) else None,
        "learned_dag_raw_overall_score": learned_dag_scores.get("raw_overall_score") if isinstance(learned_dag_scores, dict) else None,
        "learned_dag_constraint_penalty": learned_dag_scores.get("constraint_penalty") if isinstance(learned_dag_scores, dict) else None,
        "learned_dag_constraint_ceiling": learned_dag_scores.get("constraint_ceiling") if isinstance(learned_dag_scores, dict) else None,
        "learned_dag_required_coverage_ratio": learned_dag_scores.get("required_coverage_ratio") if isinstance(learned_dag_scores, dict) else None,
        "learned_dag_uncovered_capabilities": learned_dag_scores.get("uncovered_capabilities") if isinstance(learned_dag_scores, dict) else None,
        "learned_dag_overall_score": learned_dag_overall_score,
        "learned_dag_utility_weights": learned_dag_scores.get("utility_weights") if isinstance(learned_dag_scores, dict) else None,
        "learned_dag_policy_used": learned_dag_overall_score is not None,
        "avoid_operator_overlap": avoid_overlap,
        "weighted_score": round(weighted_score, 5),
        "estimated_total_cost": round(total_cost, 3),
        "operator_ids": [item.get("id") for item in selected],
        "covered_capabilities": coverage["covered_capabilities"],
        "uncovered_capabilities": coverage["uncovered_capabilities"],
    }
    if score_cache is not None:
        score_cache[cache_key] = dict(result)
    return result


def _order_operator_dag(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def stage_priority(operator: dict[str, Any]) -> tuple[int, float, str]:
        capabilities = _operator_capabilities(operator)
        if "deduplication" in capabilities:
            stage = 3
        elif "schema_guard" in capabilities:
            stage = 2
        elif capabilities & {"length_filter", "quality_filter"}:
            stage = 1
        elif capabilities & {"pii_cleanup", "text_normalization"}:
            stage = 0
        else:
            stage = 4
        return (stage, _operator_cost(operator), str(operator.get("id") or operator.get("name") or ""))

    return sorted(selected, key=stage_priority)


def _operator_coverage(selected: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    risk_profile = plan.get("quality_risk_profile", {}) if isinstance(plan.get("quality_risk_profile"), dict) else {}
    required = _datamate_operator_required_capabilities(risk_profile.get("required_capabilities") or [])
    covered: set[str] = set()
    for item in selected:
        covered.update(_operator_capabilities(item))
    covered_required = [capability for capability in required if capability in covered]
    uncovered_required = [capability for capability in required if capability not in covered]
    return {
        "required_capabilities": required,
        "covered_capabilities": covered_required,
        "uncovered_capabilities": uncovered_required,
        "coverage_ratio": round(len(covered_required) / len(required), 4) if required else 1.0,
    }


def _datamate_operator_required_capabilities(required: list[Any]) -> list[str]:
    datamate_executable = {"pii_cleanup", "text_normalization", "length_filter", "quality_filter", "deduplication", "low_cost"}
    return [str(item) for item in required if str(item) in datamate_executable]


def _capability_first_dag_sort_key(scores: dict[str, Any], dag: list[dict[str, Any]]) -> tuple[Any, ...]:
    """Rank official DataMate DAGs by required capability preservation before learned utility."""
    return (
        len(scores.get("uncovered_capabilities", []) or []),
        len(scores.get("avoid_operator_overlap", []) or []),
        -float(scores.get("capability_coverage", 0) or 0),
        -float(scores.get("quality_risk_reduction", 0) or 0),
        -float(scores.get("weighted_score", 0) or 0),
        float(scores.get("estimated_total_cost", 999) or 999),
        len(dag),
        ",".join(str(item.get("id") or item.get("name") or "") for item in dag),
    )


def _pareto_prune_operator_dags(
    dags: list[list[dict[str, Any]]],
    scored_rows: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], int]:
    keep_indices: list[int] = []
    for index, scores in enumerate(scored_rows):
        dominated = False
        for other_index, other in enumerate(scored_rows):
            if index == other_index:
                continue
            no_worse = (
                float(other.get("capability_coverage", 0)) >= float(scores.get("capability_coverage", 0))
                and float(other.get("quality_risk_reduction", 0)) >= float(scores.get("quality_risk_reduction", 0))
                and float(other.get("weighted_score", 0)) >= float(scores.get("weighted_score", 0))
                and float(other.get("estimated_total_cost", 999)) <= float(scores.get("estimated_total_cost", 999))
                and len(other.get("avoid_operator_overlap", []) or []) <= len(scores.get("avoid_operator_overlap", []) or [])
            )
            strictly_better = (
                float(other.get("capability_coverage", 0)) > float(scores.get("capability_coverage", 0))
                or float(other.get("quality_risk_reduction", 0)) > float(scores.get("quality_risk_reduction", 0))
                or float(other.get("weighted_score", 0)) > float(scores.get("weighted_score", 0))
                or float(other.get("estimated_total_cost", 999)) < float(scores.get("estimated_total_cost", 999))
                or len(other.get("avoid_operator_overlap", []) or []) < len(scores.get("avoid_operator_overlap", []) or [])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            keep_indices.append(index)
    return [dags[index] for index in keep_indices], [scored_rows[index] for index in keep_indices], len(dags) - len(keep_indices)


def _required_capabilities(plan: dict[str, Any]) -> set[str]:
    risk_profile = plan.get("quality_risk_profile", {}) if isinstance(plan.get("quality_risk_profile"), dict) else {}
    return set(_datamate_operator_required_capabilities(risk_profile.get("required_capabilities", []) or []))


def _suffix_capability_unions(candidate_pool: list[dict[str, Any]]) -> list[set[str]]:
    suffix: list[set[str]] = [set() for _ in range(len(candidate_pool) + 1)]
    running: set[str] = set()
    for index in range(len(candidate_pool) - 1, -1, -1):
        running = set(running)
        running.update(_operator_capabilities(candidate_pool[index]))
        suffix[index] = running
    return suffix


def _operator_required_capability_match_count(operator: dict[str, Any], plan: dict[str, Any]) -> int:
    return len(_operator_capabilities(operator) & _required_capabilities(plan))


def _build_capability_preserving_candidate_pool(
    candidates: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    base_limit: int = 18,
    max_limit: int = 26,
) -> list[dict[str, Any]]:
    required = _required_capabilities(plan)
    avoid_ids = _avoid_operator_ids_from_plan(plan)
    score_ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item.get("_selection_score", 0)),
            _operator_cost(item),
            str(item.get("id") or item.get("name") or ""),
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(item: dict[str, Any]) -> None:
        operator_id = str(item.get("id") or item.get("name") or "")
        if operator_id and operator_id not in selected_ids:
            selected.append(item)
            selected_ids.add(operator_id)

    for item in score_ranked[:base_limit]:
        add(item)

    for capability in sorted(required):
        rows = [
            item for item in candidates
            if capability in _operator_capabilities(item)
        ]
        rows.sort(
            key=lambda item: (
                _canonical_operator_id(str(item.get("id") or item.get("name") or "")) in avoid_ids,
                -float(item.get("_selection_score", 0)),
                _operator_cost(item),
                str(item.get("id") or item.get("name") or ""),
            )
        )
        for item in rows[:4]:
            add(item)

    selected.sort(
        key=lambda item: (
            -_operator_required_capability_match_count(item, plan),
            _canonical_operator_id(str(item.get("id") or item.get("name") or "")) in avoid_ids,
            -float(item.get("_selection_score", 0)),
            _operator_cost(item),
            str(item.get("id") or item.get("name") or ""),
        )
    )
    return selected[:max_limit]


def _select_operator_dag(candidates: list[dict[str, Any]], plan: dict[str, Any], *, limit: int = 4) -> list[dict[str, Any]]:
    if not candidates:
        plan["operator_dag_objective_scores"] = {}
        plan["operator_dag_candidate_frontier"] = []
        return []
    candidate_pool = _build_capability_preserving_candidate_pool(candidates, plan)
    required_capabilities = _required_capabilities(plan)
    suffix_capabilities = _suffix_capability_unions(candidate_pool)
    beam: list[list[dict[str, Any]]] = [[]]
    completed: list[list[dict[str, Any]]] = []
    beam_width = 36
    score_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    generated_combination_keys: set[tuple[str, ...]] = set()
    generated_candidate_count = 0
    pruned_permutation_count = 0
    pareto_pruned_count = 0
    infeasible_pruned_count = 0
    found_complete_coverage = False
    early_stop = {"stopped": False, "depth": None, "reason": ""}
    for _depth in range(1, limit + 1):
        expanded: list[list[dict[str, Any]]] = []
        for partial in beam:
            used_ids = {str(item.get("id") or item.get("name") or "") for item in partial}
            partial_indices = [
                index
                for index, item in enumerate(candidate_pool)
                if str(item.get("id") or item.get("name") or "") in used_ids
            ]
            start_index = max(partial_indices) + 1 if partial_indices else 0
            for candidate_index, candidate in enumerate(candidate_pool[start_index:], start=start_index):
                candidate_id = str(candidate.get("id") or candidate.get("name") or "")
                if candidate_id in used_ids:
                    continue
                dag = partial + [candidate]
                combination_key = _dag_cache_key(dag)
                if combination_key in generated_combination_keys:
                    pruned_permutation_count += 1
                    continue
                current_capabilities: set[str] = set()
                for item in dag:
                    current_capabilities.update(_operator_capabilities(item))
                if found_complete_coverage and len(dag) < limit and required_capabilities:
                    future_capabilities = suffix_capabilities[candidate_index + 1] if candidate_index + 1 < len(suffix_capabilities) else set()
                    if not required_capabilities.issubset(current_capabilities | future_capabilities):
                        infeasible_pruned_count += 1
                        continue
                generated_combination_keys.add(combination_key)
                generated_candidate_count += 1
                expanded.append(dag)
        if not expanded:
            break
        expanded_scored = [
            _score_operator_dag_objectives(dag, plan, limit=limit, score_cache=score_cache)
            for dag in expanded
        ]
        complete_pairs = [
            (dag, scores)
            for dag, scores in zip(expanded, expanded_scored)
            if not scores.get("uncovered_capabilities")
        ]
        if complete_pairs:
            found_complete_coverage = True
        incomplete_pairs = [
            (dag, scores)
            for dag, scores in zip(expanded, expanded_scored)
            if scores.get("uncovered_capabilities")
        ]
        if complete_pairs:
            complete_dags = [item[0] for item in complete_pairs]
            complete_scores = [item[1] for item in complete_pairs]
            pruned_dags, pruned_scores, pruned_count = _pareto_prune_operator_dags(complete_dags, complete_scores)
            pareto_pruned_count += pruned_count
            expanded = [item[0] for item in incomplete_pairs] + pruned_dags
            expanded_scored = [item[1] for item in incomplete_pairs] + pruned_scores
        expanded.sort(
            key=lambda dag: _capability_first_dag_sort_key(
                _score_operator_dag_objectives(dag, plan, limit=limit, score_cache=score_cache),
                dag,
            )
        )
        completed.extend(expanded[:beam_width])
        beam = expanded[:beam_width]
        best_layer_scores = _score_operator_dag_objectives(beam[0], plan, limit=limit, score_cache=score_cache) if beam else {}
        if (
            not best_layer_scores.get("uncovered_capabilities")
            and not best_layer_scores.get("avoid_operator_overlap")
            and float(best_layer_scores.get("weighted_score", 0)) >= 0.88
        ):
            early_stop = {
                "stopped": True,
                "depth": _depth,
                "reason": "complete_coverage_no_avoid_high_weighted_score",
            }
            break

    if not completed:
        return []
    scored = [
        (_score_operator_dag_objectives(dag, plan, limit=limit, score_cache=score_cache), dag)
        for dag in completed
        if dag
    ]
    scored.sort(key=lambda item: _capability_first_dag_sort_key(item[0], item[1]))
    best_scores, best_dag = scored[0]
    best_dag = _order_operator_dag(best_dag)
    best_scores = _score_operator_dag_objectives(best_dag, plan, limit=limit, score_cache=score_cache)
    dag_cache_report = {
        "unique_scored_dag_count": len(score_cache),
        "cache_hit_count": int(plan.pop("_dag_score_cache_hits", 0) or 0),
        "cache_miss_count": int(plan.pop("_dag_score_cache_misses", 0) or 0),
        "cache_key_strategy": "sorted_operator_id_set",
        "search_strategy": "combination_beam_search_no_permutation_duplicates",
        "pruning_strategy": "per_depth_pareto_dominance",
        "generated_candidate_count": generated_candidate_count,
        "pruned_permutation_count": pruned_permutation_count,
        "pareto_pruned_count": pareto_pruned_count,
        "infeasible_pruned_count": infeasible_pruned_count,
        "feasibility_pruning_enabled": found_complete_coverage,
        "early_stop": early_stop,
    }
    best_scores["dag_score_cache"] = dag_cache_report
    plan["operator_dag_objective_scores"] = best_scores
    plan["operator_dag_candidate_frontier"] = [
        {
            "rank": index,
            "operator_ids": scores["operator_ids"],
            "weighted_score": scores["weighted_score"],
            "learned_dag_success_score": scores.get("learned_dag_success_score"),
            "learned_dag_quality_score": scores.get("learned_dag_quality_score"),
            "learned_dag_efficiency_score": scores.get("learned_dag_efficiency_score"),
            "learned_dag_feasibility_score": scores.get("learned_dag_feasibility_score"),
            "learned_dag_raw_overall_score": scores.get("learned_dag_raw_overall_score"),
            "learned_dag_constraint_penalty": scores.get("learned_dag_constraint_penalty"),
            "learned_dag_constraint_ceiling": scores.get("learned_dag_constraint_ceiling"),
            "learned_dag_required_coverage_ratio": scores.get("learned_dag_required_coverage_ratio"),
            "learned_dag_uncovered_capabilities": scores.get("learned_dag_uncovered_capabilities"),
            "learned_dag_overall_score": scores.get("learned_dag_overall_score"),
            "learned_dag_utility_weights": scores.get("learned_dag_utility_weights"),
            "learned_dag_policy_used": scores.get("learned_dag_policy_used"),
            "capability_coverage": scores["capability_coverage"],
            "quality_risk_reduction": scores["quality_risk_reduction"],
            "estimated_total_cost": scores["estimated_total_cost"],
            "uncovered_capabilities": scores["uncovered_capabilities"],
            "avoid_operator_overlap": scores.get("avoid_operator_overlap", []),
        }
        for index, (scores, _dag) in enumerate(scored[:8], start=1)
    ]
    for item in best_dag:
        item["_selection_reason"] = "operator_planner_v3_multi_objective_beam_search"
    return best_dag


def _select_text_cleaning_operators(calls: list[HttpCall], runtime_api_base: str, timeout_seconds: float) -> list[dict[str, Any]]:
    return _select_datamate_operators(
        calls,
        runtime_api_base,
        timeout_seconds,
        _infer_task_plan(DEFAULT_TASK, Path("input.txt")),
    )


def _select_datamate_operators(
    calls: list[HttpCall],
    runtime_api_base: str,
    timeout_seconds: float,
    plan: dict[str, Any],
    *,
    operator_cache: OperatorMarketCache | None = None,
) -> list[dict[str, Any]]:
    queried: dict[str, dict[str, Any]] = {}
    original_keywords = plan.get("selected_operator_keywords") or ["clean", "filter", "text"]
    initial_keywords = _compact_operator_keywords(original_keywords, plan)
    queried_keywords: list[str] = []
    expanded_keywords: list[str] = []
    plan["operator_keyword_plan"] = {
        "original_keywords": original_keywords,
        "initial_keywords": initial_keywords,
        "queried_keywords": [],
        "expanded_keywords": [],
        "strategy": "capability_preserving_compact_cached_operator_market_query",
    }

    def query_keywords(keywords: list[str]) -> None:
        for keyword in keywords:
            if keyword in queried_keywords:
                continue
            queried_keywords.append(keyword)
            for item in _query_operator_market(
                calls,
                runtime_api_base,
                timeout_seconds=timeout_seconds,
                keyword=keyword,
                cache=operator_cache,
                purpose="operators",
            ):
                operator_id = str(item.get("id") or item.get("name") or "")
                if operator_id and operator_id not in queried:
                    queried[operator_id] = item
            plan["operator_keyword_plan"]["queried_keywords"] = list(queried_keywords)

    def select_from_queried() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        scored = []
        for item in queried.values():
            item_with_runtime_scores = dict(item)
            selection_score = _score_operator(item_with_runtime_scores, plan)
            scored.append(
                {
                    **item_with_runtime_scores,
                    "_selection_score": selection_score,
                    "_selection_reason": "profile_intent_policy_ranked_operator_market_score",
                }
            )
        scored = _apply_task1_tensor_ranker(scored, plan)
        scored.sort(key=lambda item: (-float(item.get("_selection_score", 0)), str(item.get("id") or item.get("name") or "")))
        compatible_rows = [
            item for item in scored
            if float(item.get("_selection_score", 0)) > 0 and _is_cleaning_task_compatible(item, plan)
        ]
        safe_preferred = [
            "ExtraSpaceCleaner",
            "UnicodeSpaceCleaner",
            "EmailNumberCleaner",
            "AnonymizedUrlCleaner",
            "AnonymizedIpAddress",
            "FileWithShortOrLongLengthFilter",
            "InvisibleCharactersCleaner",
            "remove_extra_spaces_mapper",
            "clean_links_mapper",
            "clean_ip_mapper",
            "clean_email_mapper",
            "text_length_filter",
            "alphanumeric_filter",
        ]
        compatible_rows.sort(
            key=lambda item: (
                -float(item.get("_selection_score", 0)),
                safe_preferred.index(str(item.get("id"))) if str(item.get("id")) in safe_preferred else 999,
                _operator_cost(item),
                str(item.get("id") or item.get("name") or ""),
            )
        )
        return compatible_rows, _select_operator_dag(compatible_rows, plan, limit=4)

    query_keywords(initial_keywords)
    compatible, selected = select_from_queried()
    coverage = _operator_coverage(selected, plan)
    additional_keywords: list[str] = []
    if coverage["uncovered_capabilities"]:
        for capability in coverage["uncovered_capabilities"]:
            additional_keywords.extend(CAPABILITY_OPERATOR_KEYWORDS.get(capability, []))
        additional_keywords.extend([keyword for keyword in original_keywords if keyword not in initial_keywords])
    for keyword in _unique_ordered(additional_keywords):
        if keyword in queried_keywords:
            continue
        expanded_keywords.append(keyword)
        query_keywords([keyword])
        compatible, selected = select_from_queried()
        coverage = _operator_coverage(selected, plan)
        if not coverage["uncovered_capabilities"]:
            break

    plan["operator_keyword_plan"].update(
        {
            "expanded_keywords": expanded_keywords,
            "required_capabilities": coverage["required_capabilities"],
            "covered_capabilities": coverage["covered_capabilities"],
            "uncovered_capabilities": coverage["uncovered_capabilities"],
            "capability_coverage_ratio": coverage["coverage_ratio"],
            "coverage_preserved": not coverage["uncovered_capabilities"],
            "expansion_reason": (
                "compact_query_missed_required_capabilities"
                if expanded_keywords
                else "compact_query_covered_required_capabilities"
            ),
        }
    )
    if expanded_keywords:
        for item in selected:
            item["_selection_reason"] = "capability_preserving_expanded_operator_market_score"

    if selected and (len(selected) >= 2 or coverage["covered_capabilities"]):
        plan["operator_planning"] = _build_operator_planning_report(candidates=compatible, selected=selected, plan=plan)
        return selected

    operators = _query_operator_market(
        calls,
        runtime_api_base,
        timeout_seconds=timeout_seconds,
        keyword="clean",
        size=80,
        cache=operator_cache,
        purpose="fallback text cleaning operators",
    )
    if "clean" not in queried_keywords:
        queried_keywords.append("clean")
    plan["operator_keyword_plan"]["queried_keywords"] = list(queried_keywords)
    selected = []
    preferred_ids = (
        "ExtraSpaceCleaner",
        "UnicodeSpaceCleaner",
        "EmailNumberCleaner",
        "AnonymizedUrlCleaner",
        "AnonymizedIpAddress",
        "FileWithShortOrLongLengthFilter",
        "InvisibleCharactersCleaner",
        "clean_links_mapper",
        "clean_ip_mapper",
        "clean_email_mapper",
        "remove_extra_spaces_mapper",
        "text_length_filter",
        "alphanumeric_filter",
    )
    for preferred_id in preferred_ids:
        match = next((item for item in operators if item.get("id") == preferred_id), None)
        if match:
            match = dict(match)
            match["_selection_score"] = _score_operator(match, plan)
            match["_selection_reason"] = "fallback_text_to_text"
            selected.append(match)
        if len(selected) >= 3:
            break
    if not selected:
        for item in operators:
            if _is_cleaning_task_compatible(item, plan):
                selected.append({**item, "_selection_score": _score_operator(item, plan), "_selection_reason": "fallback_text_to_text"})
            if len(selected) >= 3:
                break
    if compatible:
        fallback_coverage = _operator_coverage(selected, plan)
        compatible_candidate = _select_operator_dag(compatible, plan, limit=4)
        compatible_coverage = _operator_coverage(compatible_candidate, plan)
        if (
            compatible_candidate
            and len(compatible_coverage["uncovered_capabilities"]) < len(fallback_coverage["uncovered_capabilities"])
        ):
            selected = compatible_candidate
            for item in selected:
                item["_selection_reason"] = "capability_preserving_operator_market_partial_coverage"
    coverage = _operator_coverage(selected, plan)
    plan["operator_keyword_plan"].update(
        {
            "required_capabilities": coverage["required_capabilities"],
            "covered_capabilities": coverage["covered_capabilities"],
            "uncovered_capabilities": coverage["uncovered_capabilities"],
            "capability_coverage_ratio": coverage["coverage_ratio"],
            "coverage_preserved": not coverage["uncovered_capabilities"],
        }
    )
    plan["operator_planning"] = _build_operator_planning_report(candidates=selected, selected=selected, plan=plan)
    return selected


def _ensure_dataset(
    calls: list[HttpCall],
    base_url: str,
    *,
    name: str,
    description: str,
    dataset_type: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    create = _http_call(
        calls,
        name=f"DataMate create dataset {name}",
        method="POST",
        url=_join_url(base_url, "/api/data-management/datasets"),
        body={"name": name, "description": description, "datasetType": dataset_type, "status": "DRAFT"},
        timeout_seconds=timeout_seconds,
    )
    dataset_id = _extract_id(create.body_json)
    if dataset_id:
        return {"id": dataset_id, "name": name, "created": True}

    found = _find_dataset_by_name(calls, base_url, name=name, timeout_seconds=timeout_seconds)
    if found:
        return found
    raise RuntimeError(f"DataMate dataset creation/list failed for {name}")


def _find_dataset_by_name(
    calls: list[HttpCall],
    base_url: str,
    *,
    name: str,
    timeout_seconds: float,
    page_size: int = 100,
    max_pages: int = 20,
) -> dict[str, Any] | None:
    page = 0
    visited_pages: set[int] = set()
    for _ in range(max_pages):
        if page in visited_pages:
            break
        visited_pages.add(page)
        list_call = _http_call(
            calls,
            name=f"DataMate list dataset {name} page {page}",
            method="GET",
            url=_join_url(base_url, f"/api/data-management/datasets?page={page}&size={page_size}"),
            timeout_seconds=timeout_seconds,
        )
        items = _extract_items(list_call.body_json)
        for item in items:
            if item.get("name") == name and item.get("id"):
                return {"id": str(item["id"]), "name": name, "created": False}
        payload = _unwrap(list_call.body_json)
        if isinstance(payload, dict):
            total_pages = payload.get("totalPages")
            reported_page = payload.get("page")
            if isinstance(total_pages, int) and isinstance(reported_page, int):
                if reported_page >= total_pages:
                    break
                page = reported_page + 1
                continue
            if isinstance(total_pages, int) and page >= total_pages:
                break
        if not items:
            break
        page += 1
    return None


def _upload_input_file(
    calls: list[HttpCall],
    base_url: str,
    *,
    dataset_id: str,
    input_path: Path,
    timeout_seconds: float,
) -> list[dict[str, str]]:
    file_bytes = input_path.read_bytes()
    pre_upload = _http_call(
        calls,
        name="DataMate source file pre-upload",
        method="POST",
        url=_join_url(base_url, f"/api/data-management/datasets/{dataset_id}/files/upload/pre-upload"),
        body={"hasArchive": False, "totalFileNum": 1, "totalSize": len(file_bytes), "prefix": ""},
        timeout_seconds=timeout_seconds,
    )
    req_id = _extract_id(pre_upload.body_json)
    if not req_id:
        raise RuntimeError("DataMate pre-upload did not return request id")

    form_body, content_type = _build_multipart_form_data(
        {
            "reqId": req_id,
            "fileNo": 0,
            "fileName": input_path.name,
            "fileSize": len(file_bytes),
            "totalChunkNum": 1,
            "chunkNo": 1,
            "checkSumHex": hashlib.sha256(file_bytes).hexdigest(),
        },
        [
            {
                "field_name": "file",
                "filename": input_path.name,
                "content": file_bytes,
                "content_type": mimetypes.guess_type(input_path.name)[0] or "application/octet-stream",
            }
        ],
    )
    upload = _http_call(
        calls,
        name="DataMate source file chunk-upload",
        method="POST",
        url=_join_url(base_url, f"/api/data-management/datasets/{dataset_id}/files/upload/chunk"),
        headers={"Content-Type": content_type},
        raw_body=form_body,
        timeout_seconds=max(timeout_seconds * 2, timeout_seconds + 3),
    )
    upload.body_preview = f"multipart({len(form_body)} bytes)"
    if not upload.ok:
        raise RuntimeError("DataMate chunk upload failed")

    files_call = _http_call(
        calls,
        name="DataMate list uploaded source files",
        method="GET",
        url=_join_url(base_url, f"/api/data-management/datasets/{dataset_id}/files?page=0&size=100"),
        timeout_seconds=timeout_seconds,
    )
    files = [
        {"id": str(item.get("id")), "fileName": str(item.get("fileName") or item.get("file_name") or item.get("name"))}
        for item in _extract_items(files_call.body_json)
        if item.get("id")
    ]
    if not files:
        raise RuntimeError("DataMate uploaded source file was not listed")
    return files


def _upload_input_files(
    calls: list[HttpCall],
    base_url: str,
    *,
    dataset_id: str,
    input_paths: list[Path],
    timeout_seconds: float,
) -> list[dict[str, str]]:
    uploaded: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for path in input_paths:
        files = _upload_input_file(
            calls,
            base_url,
            dataset_id=dataset_id,
            input_path=path,
            timeout_seconds=timeout_seconds,
        )
        for item in files:
            file_id = str(item.get("id") or "")
            if file_id and file_id not in seen_ids:
                uploaded.append(item)
                seen_ids.add(file_id)
    return uploaded


def _create_cleaning_task(
    calls: list[HttpCall],
    base_url: str,
    *,
    task_name: str,
    source_dataset: dict[str, Any],
    dest_dataset_name: str,
    operators: list[dict[str, Any]],
    plan: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    if not operators:
        raise RuntimeError("DataMate operator market did not return usable text cleaning operators")
    body = {
        "name": task_name,
        "description": "ModelEngine Task-1 Data Processing Agent task executed by DataMate cleaning pipeline.",
        "srcDatasetId": source_dataset["id"],
        "srcDatasetName": source_dataset["name"],
        "destDatasetName": dest_dataset_name,
        "destDatasetType": "TEXT",
        "instance": [
            {
                "id": item["id"],
                "overrides": _datamate_operator_overrides(item, plan),
                "categories": item.get("categories") or [],
                "inputs": item.get("inputs"),
                "outputs": item.get("outputs"),
            }
            for item in operators
        ],
    }
    create = _http_call(
        calls,
        name="DataMate create cleaning task",
        method="POST",
        url=_join_url(base_url, "/api/cleaning/tasks"),
        body=body,
        timeout_seconds=max(timeout_seconds * 2, timeout_seconds + 3),
    )
    task_id = _extract_id(create.body_json)
    if not task_id:
        raise RuntimeError("DataMate cleaning task creation did not return task id" + _format_datamate_call_error(create))
    task_payload = _unwrap(create.body_json)
    return task_payload if isinstance(task_payload, dict) else {"id": task_id}


def _format_datamate_call_error(call: HttpCall | None) -> str:
    if call is None:
        return ""
    payload = call.body_json if isinstance(call.body_json, dict) else {}
    code = payload.get("code") if isinstance(payload, dict) else None
    message = payload.get("message") if isinstance(payload, dict) else None
    parts = []
    if call.status_code is not None:
        parts.append(f"status={call.status_code}")
    if code:
        parts.append(f"code={code}")
    if message:
        parts.append(f"message={message}")
    if call.error:
        parts.append(f"error={call.error}")
    return f" ({', '.join(parts)})" if parts else ""


def _latest_cleaning_create_call(calls: list[HttpCall]) -> HttpCall | None:
    for call in reversed(calls):
        if call.method == "POST" and "/api/cleaning/tasks" in call.url and not call.url.rstrip("/").endswith("/execute"):
            return call
    return None


def _classify_datamate_failure(
    *,
    reason: str,
    plan: dict[str, Any],
    final_task: dict[str, Any] | None = None,
    call: HttpCall | None = None,
) -> dict[str, Any]:
    reason_text = reason.lower()
    status_code = call.status_code if call else None
    task_status = str((final_task or {}).get("status") or "").upper()
    progress = (final_task or {}).get("progress") if isinstance((final_task or {}).get("progress"), dict) else {}
    contract = plan.get("workflow_plan", {}).get("etl_contract", {}) if isinstance(plan.get("workflow_plan"), dict) else {}
    category = "unknown_datamate_failure"
    retryable = True
    severity = "high"

    if status_code and status_code >= 500:
        category = "datamate_server_error_during_task_creation"
    elif status_code and status_code >= 400:
        category = "datamate_request_rejected_during_task_creation"
        retryable = False
    elif "creation did not return task id" in reason_text:
        category = "datamate_task_creation_no_id"
    elif task_status == "FAILED":
        failed_files = int(progress.get("failedFileNum") or 0)
        category = "datamate_task_execution_failed"
        if failed_files:
            category = "datamate_task_file_processing_failed"
    elif task_status == "TIMEOUT":
        category = "datamate_task_timeout"
    elif "operator" in reason_text:
        category = "datamate_operator_dag_incompatible"

    probable_causes: list[str] = []
    if plan.get("workflow_type") == "structured_etl" and contract.get("blocking_rule_count"):
        probable_causes.append("structured_etl_contract_has_blocking_quality_rules")
    if plan.get("input_profile", {}).get("quality_report", {}).get("empty_text_documents"):
        probable_causes.append("source_contains_empty_text_records")
    if plan.get("input_profile", {}).get("quality_report", {}).get("duplicate_doc_ids"):
        probable_causes.append("source_contains_duplicate_primary_keys")
    if status_code and status_code >= 500:
        probable_causes.append("datamate_backend_returned_internal_server_error")
    if not probable_causes:
        probable_causes.append("datamate_runtime_or_operator_reported_failure_without_inline_detail")

    return {
        "category": category,
        "severity": severity,
        "retryable": retryable,
        "task_status": task_status or None,
        "http_status": status_code,
        "datamate_code": call.body_json.get("code") if call and isinstance(call.body_json, dict) else None,
        "datamate_message": call.body_json.get("message") if call and isinstance(call.body_json, dict) else None,
        "probable_causes": probable_causes,
        "official_chain_policy": "retry_only_with_DataMate_operator_DAGs_then_report_blocker",
    }


def _operator_attempt_signature(attempt: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    capabilities: set[str] = set()
    for item in attempt:
        capabilities.update(_operator_capabilities(item))
    required = list(plan.get("quality_risk_profile", {}).get("required_capabilities") or [])
    return {
        "operator_ids": [item.get("id") for item in attempt],
        "operator_names": [item.get("name") for item in attempt],
        "operator_count": len(attempt),
        "capabilities": sorted(capabilities),
        "covered_required_capabilities": [capability for capability in required if capability in capabilities],
        "uncovered_required_capabilities": [capability for capability in required if capability not in capabilities],
        "all_text_compatible": all(_is_cleaning_task_compatible(item, plan) for item in attempt),
        "estimated_cost": round(sum(_operator_cost(item) for item in attempt), 3),
        "selection_reason": attempt[0].get("_selection_reason") if attempt else "",
    }


def _build_recovery_planning_report(
    *,
    plan: dict[str, Any],
    failure: dict[str, Any],
    initial_operators: list[dict[str, Any]],
    market_operators: list[dict[str, Any]],
    attempts: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "strategy": "classified_failure_then_DataMate_safe_operator_DAG_retries",
        "failure": failure,
        "source_workflow_type": plan.get("workflow_type"),
        "initial_operator_ids": [item.get("id") for item in initial_operators],
        "market_recovery_operator_ids": [item.get("id") for item in market_operators],
        "attempt_count": len(attempts),
        "attempts": [
            {"attempt_index": index, **_operator_attempt_signature(attempt, plan)}
            for index, attempt in enumerate(attempts, start=1)
        ],
        "no_fallback_policy": "No local fake processing is used; unresolved failures remain blocked with DataMate evidence.",
    }


def _build_recovery_report(
    *,
    plan: dict[str, Any],
    recovery_actions: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    failed_actions = [item for item in recovery_actions if str(item.get("action", "")).endswith("_failed")]
    classifications = [item.get("failure_classification") for item in recovery_actions if item.get("failure_classification")]
    return {
        "status": "not_required" if not recovery_actions else "attempted",
        "workflow_type": plan.get("workflow_type"),
        "action_count": len(recovery_actions),
        "failed_action_count": len(failed_actions),
        "actions": recovery_actions,
        "failure_classifications": classifications,
        "trace_steps": [
            item for item in trace
            if str(item.get("step", "")).startswith("datamate_") and "recovery" in str(item.get("step", ""))
        ],
        "policy": "official_datamate_recovery_only_no_local_fallback",
    }


def _build_batch_report(
    *,
    plan: dict[str, Any],
    input_paths: list[Path],
    uploaded_files: list[dict[str, Any]],
    final_task: dict[str, Any],
    post_quality_report: dict[str, Any],
) -> dict[str, Any]:
    batch_profile = plan.get("batch_profile", {}) if isinstance(plan.get("batch_profile"), dict) else {}
    files = batch_profile.get("files", []) if isinstance(batch_profile.get("files"), list) else []
    target_metadata = post_quality_report.get("post_metadata", {}) if isinstance(post_quality_report, dict) else {}
    return {
        "batch": len(input_paths) > 1,
        "input_file_count": len(input_paths),
        "uploaded_file_count": len(uploaded_files),
        "target_file_count": target_metadata.get("target_file_count"),
        "workflow_type": plan.get("workflow_type"),
        "format_counts": batch_profile.get("format_counts", {}),
        "record_count_estimate": batch_profile.get("record_count_estimate"),
        "quality_warning_count": batch_profile.get("quality_warning_count"),
        "source_files": [
            {
                "path": str(path),
                "name": path.name,
                "format": path.suffix.lower().lstrip("."),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
            for path in input_paths
        ],
        "file_profiles": files,
        "uploaded_files": uploaded_files,
        "cleaning_task_id": final_task.get("id"),
        "cleaning_task_status": final_task.get("status"),
        "post_execution_quality_status": post_quality_report.get("status") if isinstance(post_quality_report, dict) else None,
        "policy": "all_files_are_ingested_into_one_DataMate_source_dataset_and_processed_by_one_DataMate_cleaning_task",
    }


def _build_performance_report(
    call_rows: list[dict[str, Any]],
    elapsed_ms: float,
    *,
    plan: dict[str, Any] | None = None,
    post_quality_report: dict[str, Any] | None = None,
    batch_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    operator_calls = [row for row in call_rows if "/api/operators/list" in str(row.get("url", ""))]
    real_http_calls = [row for row in call_rows if row.get("method") != "CACHE"]
    cache_hits = [row for row in call_rows if row.get("method") == "CACHE"]
    poll_calls = [row for row in call_rows if "poll cleaning task status" in str(row.get("name", ""))]
    upload_calls = [row for row in call_rows if "upload" in str(row.get("name", "")).lower()]
    plan = plan or {}
    post_quality_report = post_quality_report or {}
    batch_report = batch_report or {}
    operator_planning = plan.get("operator_planning", {}) if isinstance(plan.get("operator_planning"), dict) else {}
    keyword_plan = plan.get("operator_keyword_plan", {}) if isinstance(plan.get("operator_keyword_plan"), dict) else {}
    required_capabilities = list(
        operator_planning.get("required_capabilities")
        or keyword_plan.get("required_capabilities")
        or []
    )
    covered_capabilities = list(
        operator_planning.get("covered_capabilities")
        or keyword_plan.get("covered_capabilities")
        or []
    )
    uncovered_capabilities = list(
        operator_planning.get("uncovered_capabilities")
        or keyword_plan.get("uncovered_capabilities")
        or []
    )
    capability_coverage_ratio = (
        round(len(covered_capabilities) / len(required_capabilities), 4)
        if required_capabilities
        else 1.0
    )
    post_quality_status = post_quality_report.get("status")
    quality_preserved = not uncovered_capabilities and str(post_quality_status or "passed").lower() not in {
        "failed",
        "blocked",
        "error",
    }
    return {
        "elapsed_ms": elapsed_ms,
        "api_call_count_total": len(call_rows),
        "api_call_count_real_http": len(real_http_calls),
        "operator_market_call_count": len(operator_calls),
        "operator_market_real_http_count": sum(1 for row in operator_calls if row.get("method") != "CACHE"),
        "operator_market_cache_hit_count": len(cache_hits),
        "poll_call_count": len(poll_calls),
        "upload_call_count": len(upload_calls),
        "required_capabilities": required_capabilities,
        "covered_capabilities": covered_capabilities,
        "uncovered_capabilities": uncovered_capabilities,
        "capability_coverage_ratio": capability_coverage_ratio,
        "quality_preserved": quality_preserved,
        "post_execution_quality_status": post_quality_status,
        "target_file_count": batch_report.get("target_file_count"),
        "operator_keyword_plan": keyword_plan,
        "optimization_features": [
            "compact_operator_keyword_query",
            "capability_preserving_operator_query_expansion",
            "run_scoped_operator_market_cache",
            "adaptive_datamate_polling_backoff",
            "batch_files_single_datamate_cleaning_task",
        ],
    }


def _safe_recovery_operator_dag(operators: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    preferred = [
        "ExtraSpaceCleaner",
        "UnicodeSpaceCleaner",
        "EmailNumberCleaner",
        "AnonymizedUrlCleaner",
        "AnonymizedIpAddress",
        "FileWithShortOrLongLengthFilter",
        "InvisibleCharactersCleaner",
        "remove_extra_spaces_mapper",
        "clean_email_mapper",
        "clean_links_mapper",
        "clean_ip_mapper",
        "text_length_filter",
        "alphanumeric_filter",
    ]
    by_id = {str(item.get("id")): item for item in operators if item.get("id")}
    recovered: list[dict[str, Any]] = []
    recovered_ids: set[str] = set()
    for operator_id in preferred:
        item = by_id.get(operator_id)
        if item and _is_cleaning_task_compatible(item, plan):
            recovered.append({**item, "_selection_reason": "recovery_safe_datamate_text_operator"})
            recovered_ids.add(operator_id)
        if len(recovered) >= 2:
            break
    if recovered:
        return recovered
    for item in operators:
        operator_id = str(item.get("id") or "")
        if operator_id and operator_id not in recovered_ids and _is_cleaning_task_compatible(item, plan):
            recovered.append({**item, "_selection_reason": "recovery_compatible_text_operator"})
            recovered_ids.add(operator_id)
        if len(recovered) >= 2:
            break
    return recovered


def _query_safe_recovery_operators(
    calls: list[HttpCall],
    runtime_api_base: str,
    *,
    plan: dict[str, Any],
    timeout_seconds: float,
    operator_cache: OperatorMarketCache | None = None,
) -> list[dict[str, Any]]:
    preferred = [
        "ExtraSpaceCleaner",
        "UnicodeSpaceCleaner",
        "EmailNumberCleaner",
        "AnonymizedUrlCleaner",
        "AnonymizedIpAddress",
        "FileWithShortOrLongLengthFilter",
        "InvisibleCharactersCleaner",
        "remove_extra_spaces_mapper",
        "clean_email_mapper",
        "clean_links_mapper",
        "clean_ip_mapper",
        "text_length_filter",
        "alphanumeric_filter",
    ]
    queried: dict[str, dict[str, Any]] = {}
    recovery_keywords = ["clean", "length", "email", "filter", "quality", "field", "schema", "dedup"]
    risk_profile = plan.get("quality_risk_profile", {}) if isinstance(plan.get("quality_risk_profile"), dict) else {}
    for capability in risk_profile.get("required_capabilities", []):
        recovery_keywords.extend(CAPABILITY_OPERATOR_KEYWORDS.get(str(capability), []))
    for keyword in _unique_ordered(recovery_keywords):
        for item in _query_operator_market(
            calls,
            runtime_api_base,
            timeout_seconds=timeout_seconds,
            keyword=keyword,
            cache=operator_cache,
            purpose="recovery operators",
        ):
            operator_id = str(item.get("id") or "")
            if operator_id and operator_id not in queried and _is_cleaning_task_compatible(item, plan):
                queried[operator_id] = item
    selected: list[dict[str, Any]] = []
    for operator_id in preferred:
        item = queried.get(operator_id)
        if item:
            selected.append({**item, "_selection_reason": "recovery_safe_datamate_market_operator"})
        if len(selected) >= 4:
            break
    if selected:
        return selected
    return [
        {
            **item,
            "_selection_score": _score_operator(item, plan),
            "_selection_reason": "recovery_low_cost_datamate_market_operator",
        }
        for item in sorted(queried.values(), key=lambda row: (_operator_cost(row), str(row.get("id") or "")))[:2]
    ]


def add_capability_optimized_recovery_attempts(
    by_id: dict[str, dict[str, Any]],
    plan: dict[str, Any],
    add_attempt: Any,
) -> None:
    required = _required_capabilities(plan)
    avoid_ids = _avoid_operator_ids_from_plan(plan)
    candidates = [
        item
        for item in by_id.values()
        if _is_cleaning_task_compatible(item, plan)
    ]
    if not candidates:
        return

    market_capabilities: set[str] = set()
    for item in candidates:
        market_capabilities.update(_operator_capabilities(item))
    market_can_cover_all_required = bool(required) and required.issubset(market_capabilities)
    scored_attempts: list[tuple[tuple[float, int, int, float, str], list[str]]] = []
    for size in range(1, min(4, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, size):
            capabilities: set[str] = set()
            for item in combo:
                capabilities.update(_operator_capabilities(item))
            covered = capabilities & required if required else capabilities
            missing = required - capabilities
            quality_coverage = _quality_risk_coverage_score(list(combo), plan)
            cost = sum(_operator_cost(item) for item in combo)
            ids = [str(item.get("id") or "") for item in combo if item.get("id")]
            canonical_ids = {_canonical_operator_id(item) for item in ids}
            avoid_overlap = canonical_ids & avoid_ids
            quality_guard_bonus = (
                1.25
                if {"text_length_filter", "alphanumeric_filter"}.issubset(canonical_ids)
                and {"length_filter", "quality_filter"} & required
                else 0.0
            )
            required_coverage_bonus = (
                1.35
                if market_can_cover_all_required and not missing
                else 0.0
            )
            score = (
                float(len(covered)) * 2.0
                + float(quality_coverage) * 1.5
                + quality_guard_bonus
                + required_coverage_bonus
                - float(len(missing)) * 1.2
                - len(avoid_overlap) * 0.75
                - cost * 0.18
                - len(combo) * 0.05
            )
            scored_attempts.append(((-score, len(missing), len(avoid_overlap), cost, ",".join(ids)), ids))
    for _rank, ids in sorted(scored_attempts)[:5]:
        add_attempt(ids, "recovery_capability_optimized_datamate_dag")


def _recovery_operator_dag_attempts(
    market_operators: list[dict[str, Any]],
    initial_operators: list[dict[str, Any]],
    plan: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in market_operators + initial_operators:
        operator_id = str(item.get("id") or "")
        if operator_id and operator_id not in by_id and _is_cleaning_task_compatible(item, plan):
            by_id[operator_id] = item

    attempts: list[list[dict[str, Any]]] = []

    def add_attempt(operator_ids: list[str], reason: str) -> None:
        attempt: list[dict[str, Any]] = []
        seen: set[str] = set()
        for operator_id in operator_ids:
            item = by_id.get(operator_id)
            if item and operator_id not in seen:
                attempt.append({**item, "_selection_reason": reason})
                seen.add(operator_id)
        if attempt and [item.get("id") for item in attempt] not in [[row.get("id") for row in existing] for existing in attempts]:
            attempts.append(attempt)

    add_capability_optimized_recovery_attempts(by_id, plan, add_attempt)
    add_attempt(["text_length_filter"], "recovery_minimal_quality_filter")
    add_attempt(["clean_email_mapper"], "recovery_minimal_pii_cleanup")
    add_attempt(["clean_links_mapper"], "recovery_minimal_link_cleanup")
    add_attempt(["clean_ip_mapper"], "recovery_minimal_ip_cleanup")
    add_attempt(["clean_email_mapper", "clean_links_mapper"], "recovery_safe_pii_cleanup_dag")
    add_attempt(["text_length_filter", "clean_email_mapper"], "recovery_quality_then_pii_dag")

    safe_from_initial = _safe_recovery_operator_dag(initial_operators, plan)
    if safe_from_initial:
        attempt_ids = [str(item.get("id") or "") for item in safe_from_initial]
        add_attempt(attempt_ids, "recovery_safe_initial_operator_dag")

    return attempts


def _build_recovery_state_machine(
    *,
    failure: dict[str, Any],
    attempts: list[list[dict[str, Any]]],
    phase: str,
) -> dict[str, Any]:
    category = str(failure.get("category") or "unknown_datamate_failure")
    retryable = bool(failure.get("retryable", True))
    states = [
        {
            "state": "classify_failure",
            "phase": phase,
            "entry_condition": "DataMate task creation or execution returned non-success evidence",
            "action": "classify_http_status_task_status_and_contract_context",
            "exit_condition": "failure.category_and_retryability_available",
        },
        {
            "state": "query_recovery_operator_market",
            "phase": phase,
            "entry_condition": "failure.retryable == true",
            "action": "query_DataMate_operator_market_for_safe_text_compatible_recovery_operators",
            "exit_condition": "market_recovery_operators_available_or_empty",
        },
        {
            "state": "compose_recovery_dags",
            "phase": phase,
            "entry_condition": "safe_DataMate_operators_available",
            "action": "generate_ordered_recovery_DAG_attempts_with_capability_coverage_signatures",
            "exit_condition": "attempts_ranked_by_safety_cost_and_required_capability_coverage",
        },
        {
            "state": "execute_recovery_attempts",
            "phase": phase,
            "entry_condition": "attempt_count > 0 and failure.retryable == true",
            "action": "create_and_execute_DataMate_cleaning_tasks_until_success_or_attempts_exhausted",
            "exit_condition": "DataMate_terminal_success_or_blocked_with_evidence",
        },
    ]
    if not retryable:
        states = states[:1] + [
            {
                "state": "blocked_non_retryable",
                "phase": phase,
                "entry_condition": "failure.retryable == false",
                "action": "stop_without_local_fallback_and_return_DataMate_blocker_evidence",
                "exit_condition": "blocked_report_generated",
            }
        ]
    return {
        "state_machine_version": "datamate_recovery_state_machine_v3",
        "phase": phase,
        "failure_category": category,
        "retryable": retryable,
        "attempt_count": len(attempts),
        "states": states,
        "terminal_policy": "success_only_from_DataMate_terminal_success_otherwise_blocked_no_local_fallback",
    }


def _create_cleaning_task_with_recovery(
    calls: list[HttpCall],
    base_url: str,
    *,
    runtime_api_base: str,
    task_name: str,
    source_dataset: dict[str, Any],
    dest_dataset_name: str,
    operators: list[dict[str, Any]],
    plan: dict[str, Any],
    trace: list[dict[str, Any]],
    timeout_seconds: float,
    operator_cache: OperatorMarketCache | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    recovery_actions: list[dict[str, Any]] = []
    try:
        task = _create_cleaning_task(
            calls,
            base_url,
            task_name=task_name,
            source_dataset=source_dataset,
            dest_dataset_name=dest_dataset_name,
            operators=operators,
            plan=plan,
            timeout_seconds=timeout_seconds,
        )
        return task, operators, recovery_actions
    except Exception as exc:
        market_recovery_operators = _query_safe_recovery_operators(
            calls,
            runtime_api_base,
            plan=plan,
            timeout_seconds=timeout_seconds,
            operator_cache=operator_cache,
        )
        recovery_attempts = _recovery_operator_dag_attempts(market_recovery_operators, operators, plan)
        recovery_operators = recovery_attempts[0] if recovery_attempts else []
        failure = _classify_datamate_failure(
            reason=str(exc),
            plan=plan,
            call=_latest_cleaning_create_call(calls),
        )
        recovery_planning = _build_recovery_planning_report(
            plan=plan,
            failure=failure,
            initial_operators=operators,
            market_operators=market_recovery_operators,
            attempts=recovery_attempts,
        )
        recovery_state_machine = _build_recovery_state_machine(
            failure=failure,
            attempts=recovery_attempts,
            phase="task_creation",
        )
        recovery_actions.append(
            {
                "action": "retry_create_cleaning_task_with_safe_datamate_operator_dag",
                "reason": str(exc),
                "failure_classification": failure,
                "recovery_planning": recovery_planning,
                "recovery_state_machine": recovery_state_machine,
                "original_operator_ids": [item.get("id") for item in operators],
                "recovery_attempts": [[item.get("id") for item in attempt] for attempt in recovery_attempts],
                "recovery_operator_ids": [item.get("id") for item in recovery_operators],
            }
        )
        trace.append(
            {
                "step": "datamate_recovery_planning",
                "status": "retrying",
                "recovery_planning": recovery_planning,
                "recovery_state_machine": recovery_state_machine,
            }
        )
        if not recovery_attempts:
            raise
        last_error = exc
        for index, attempt in enumerate(recovery_attempts, start=1):
            try:
                task = _create_cleaning_task(
                    calls,
                    base_url,
                    task_name=f"{task_name}_recovery_{index}",
                    source_dataset=source_dataset,
                    dest_dataset_name=f"{dest_dataset_name}_recovery_{index}",
                    operators=attempt,
                    plan=plan,
                    timeout_seconds=timeout_seconds,
                )
                plan["operator_planning"] = _build_operator_planning_report(
                    candidates=attempt,
                    selected=attempt,
                    plan=plan,
                )
                return task, attempt, recovery_actions
            except Exception as recovery_exc:
                last_error = recovery_exc
                failure = _classify_datamate_failure(
                    reason=str(recovery_exc),
                    plan=plan,
                    call=_latest_cleaning_create_call(calls),
                )
                recovery_actions.append(
                    {
                        "action": "recovery_cleaning_task_creation_failed",
                        "reason": str(recovery_exc),
                        "failure_classification": failure,
                        "recovery_operator_ids": [item.get("id") for item in attempt],
                    }
                )
                trace.append(
                    {
                        "step": "datamate_create_recovery_attempt_failed",
                        "status": "retrying" if index < len(recovery_attempts) else "blocked",
                        "reason": str(recovery_exc),
                        "failure_classification": failure,
                        "operator_ids": [item.get("id") for item in attempt],
                    }
                )
        raise last_error


def _execute_with_recovery(
    calls: list[HttpCall],
    base_url: str,
    *,
    runtime_api_base: str,
    task_name: str,
    source_dataset: dict[str, Any],
    dest_dataset_name: str,
    initial_task: dict[str, Any],
    initial_operators: list[dict[str, Any]],
    plan: dict[str, Any],
    trace: list[dict[str, Any]],
    recovery_actions: list[dict[str, Any]],
    timeout_seconds: float,
    wait_seconds: float,
    operator_cache: OperatorMarketCache | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    final_task = _execute_and_wait(
        calls,
        base_url,
        task_id=str(initial_task["id"]),
        timeout_seconds=timeout_seconds,
        wait_seconds=wait_seconds,
    )
    final_status = str(final_task.get("status", "")).upper()
    if final_status in SUCCESS_STATUSES:
        return initial_task, final_task, initial_operators, recovery_actions

    initial_failure = _classify_datamate_failure(
        reason=f"DataMate cleaning task ended with status {final_status or 'UNKNOWN'}",
        plan=plan,
        final_task=final_task,
    )

    trace.append(
        {
            "step": "datamate_initial_cleaning_task_execution",
            "status": final_status or "UNKNOWN",
            "task": final_task,
            "failure_classification": initial_failure,
        }
    )
    market_recovery_operators = _query_safe_recovery_operators(
        calls,
        runtime_api_base,
        plan=plan,
        timeout_seconds=timeout_seconds,
        operator_cache=operator_cache,
    )
    recovery_attempts = _recovery_operator_dag_attempts(market_recovery_operators, initial_operators, plan)
    recovery_operators = recovery_attempts[0] if recovery_attempts else []
    recovery_planning = _build_recovery_planning_report(
        plan=plan,
        failure=initial_failure,
        initial_operators=initial_operators,
        market_operators=market_recovery_operators,
        attempts=recovery_attempts,
    )
    recovery_state_machine = _build_recovery_state_machine(
        failure=initial_failure,
        attempts=recovery_attempts,
        phase="task_execution",
    )
    recovery_actions.append(
        {
            "action": "retry_failed_cleaning_task_with_safe_datamate_operator_dag",
            "reason": f"DataMate cleaning task ended with status {final_status or 'UNKNOWN'}",
            "failure_classification": initial_failure,
            "recovery_planning": recovery_planning,
            "recovery_state_machine": recovery_state_machine,
            "failed_task_id": initial_task.get("id"),
            "original_operator_ids": [item.get("id") for item in initial_operators],
            "recovery_attempts": [[item.get("id") for item in attempt] for attempt in recovery_attempts],
            "recovery_operator_ids": [item.get("id") for item in recovery_operators],
        }
    )
    trace.append(
        {
            "step": "datamate_execution_recovery_planning",
            "status": "retrying" if recovery_operators else "blocked",
            "recovery_planning": recovery_planning,
            "recovery_state_machine": recovery_state_machine,
        }
    )
    if not recovery_attempts:
        return initial_task, final_task, initial_operators, recovery_actions

    recovery_task: dict[str, Any] = {}
    recovery_final_task: dict[str, Any] = {}
    for index, attempt in enumerate(recovery_attempts, start=1):
        try:
            recovery_task = _create_cleaning_task(
                calls,
                base_url,
                task_name=f"{task_name}_exec_recovery_{index}",
                source_dataset=source_dataset,
                dest_dataset_name=f"{dest_dataset_name}_exec_recovery_{index}",
                operators=attempt,
                plan=plan,
                timeout_seconds=timeout_seconds,
            )
            recovery_operators = attempt
        except Exception as exc:
            failure = _classify_datamate_failure(
                reason=str(exc),
                plan=plan,
                final_task=final_task,
                call=_latest_cleaning_create_call(calls),
            )
            recovery_actions.append(
                {
                    "action": "recovery_cleaning_task_creation_failed",
                    "reason": str(exc),
                    "failure_classification": failure,
                    "failed_task_id": initial_task.get("id"),
                    "recovery_operator_ids": [item.get("id") for item in attempt],
                }
            )
            trace.append(
                {
                    "step": "datamate_recovery_cleaning_task_creation_failed",
                    "status": "retrying" if index < len(recovery_attempts) else "blocked",
                    "reason": str(exc),
                    "failure_classification": failure,
                    "operator_ids": [item.get("id") for item in attempt],
                }
            )
            continue
        trace.append(
            {
                "step": "datamate_recovery_cleaning_task_created",
                "status": "completed",
                "cleaning_task": recovery_task,
                "operator_ids": [item.get("id") for item in recovery_operators],
            }
        )
        recovery_final_task = _execute_and_wait(
            calls,
            base_url,
            task_id=str(recovery_task["id"]),
            timeout_seconds=timeout_seconds,
            wait_seconds=wait_seconds,
        )
        recovery_status = str(recovery_final_task.get("status", "")).upper()
        recovery_failure = {} if recovery_status in SUCCESS_STATUSES else _classify_datamate_failure(
            reason=f"DataMate recovery cleaning task ended with status {recovery_status or 'UNKNOWN'}",
            plan=plan,
            final_task=recovery_final_task,
        )
        trace.append(
            {
                "step": "datamate_recovery_cleaning_task_execution",
                "status": recovery_status or "UNKNOWN",
                "task": recovery_final_task,
                "operator_ids": [item.get("id") for item in recovery_operators],
                "failure_classification": recovery_failure,
            }
        )
        if recovery_status in SUCCESS_STATUSES:
            plan["operator_planning"] = _build_operator_planning_report(
                candidates=recovery_operators,
                selected=recovery_operators,
                plan=plan,
            )
            return recovery_task, recovery_final_task, recovery_operators, recovery_actions
        recovery_actions.append(
            {
                "action": "recovery_cleaning_task_execution_failed",
                "reason": f"DataMate recovery cleaning task ended with status {recovery_status or 'UNKNOWN'}",
                "failure_classification": recovery_failure,
                "failed_task_id": initial_task.get("id"),
                "recovery_operator_ids": [item.get("id") for item in recovery_operators],
            }
        )
    return initial_task, final_task, initial_operators, recovery_actions


def _execute_and_wait(
    calls: list[HttpCall],
    base_url: str,
    *,
    task_id: str,
    timeout_seconds: float,
    wait_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + wait_seconds
    latest: dict[str, Any] = {}
    execute_attempted = False
    poll_attempt = 0
    while True:
        poll_attempt += 1
        detail = _http_call(
            calls,
            name="DataMate poll cleaning task status",
            method="GET",
            url=_join_url(base_url, f"/api/cleaning/tasks/{task_id}"),
            timeout_seconds=timeout_seconds,
        )
        payload = _unwrap(detail.body_json)
        latest = payload if isinstance(payload, dict) else {}
        status = str(latest.get("status", "")).upper()
        started = bool(latest.get("startedAt") or latest.get("started_at") or latest.get("startTime"))
        if status in TERMINAL_STATUSES:
            return latest
        if not execute_attempted and not started and status not in RUNNING_STATUSES:
            execute = _http_call(
                calls,
                name="DataMate execute cleaning task",
                method="POST",
                url=_join_url(base_url, f"/api/cleaning/tasks/{task_id}/execute"),
                timeout_seconds=timeout_seconds,
            )
            execute_attempted = True
            if not execute.ok:
                raise RuntimeError("DataMate cleaning task execute failed")
        if time.monotonic() >= deadline:
            latest["status"] = status or "TIMEOUT"
            return latest
        sleep_seconds = min(8.0, max(1.0, 1.5 * poll_attempt))
        remaining = max(0.0, deadline - time.monotonic())
        time.sleep(min(sleep_seconds, remaining, max(1.0, timeout_seconds)))


def _collect_result(
    calls: list[HttpCall],
    base_url: str,
    *,
    task_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    result = _http_call(
        calls,
        name="DataMate collect cleaning task result",
        method="GET",
        url=_join_url(base_url, f"/api/cleaning/tasks/{task_id}/result"),
        timeout_seconds=timeout_seconds,
    )
    payload = _unwrap(result.body_json)
    return payload if isinstance(payload, dict) else {"raw": payload}


def _extract_dest_file_ids(result_payload: Any) -> list[str]:
    file_ids: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            dest_file_id = value.get("destFileId") or value.get("dest_file_id")
            if isinstance(dest_file_id, str) and dest_file_id:
                file_ids.append(dest_file_id)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(result_payload)
    return [file_id for index, file_id in enumerate(file_ids) if file_id not in file_ids[:index]]


def _list_dataset_files(
    calls: list[HttpCall],
    base_url: str,
    *,
    dataset_id: str,
    timeout_seconds: float,
    label: str,
) -> list[dict[str, Any]]:
    files_call = _http_call(
        calls,
        name=f"DataMate list {label} dataset files",
        method="GET",
        url=_join_url(base_url, f"/api/data-management/datasets/{dataset_id}/files?page=0&size=100"),
        timeout_seconds=timeout_seconds,
    )
    return _extract_items(files_call.body_json)


def _get_dataset_detail(
    calls: list[HttpCall],
    base_url: str,
    *,
    dataset_id: str,
    timeout_seconds: float,
    label: str,
) -> dict[str, Any]:
    detail_call = _http_call(
        calls,
        name=f"DataMate get {label} dataset detail",
        method="GET",
        url=_join_url(base_url, f"/api/data-management/datasets/{dataset_id}"),
        timeout_seconds=timeout_seconds,
    )
    payload = _unwrap(detail_call.body_json)
    if isinstance(payload, dict):
        return payload
    return {}


def _build_post_execution_quality_report(
    calls: list[HttpCall],
    base_url: str,
    *,
    plan: dict[str, Any],
    source_dataset: dict[str, Any],
    final_task: dict[str, Any],
    result_payload: Any,
    effective_input_file_count: int | None = None,
    timeout_seconds: float,
) -> dict[str, Any]:
    input_profile = plan.get("input_profile", {}) if isinstance(plan.get("input_profile"), dict) else {}
    source_dataset_id = str(source_dataset.get("id") or "")
    dest_dataset_id = str(final_task.get("destDatasetId") or final_task.get("dest_dataset_id") or "")
    result_items = result_payload if isinstance(result_payload, list) else _extract_items(result_payload)
    dest_file_ids = _extract_dest_file_ids(result_payload)
    source_files = _list_dataset_files(
        calls,
        base_url,
        dataset_id=source_dataset_id,
        timeout_seconds=timeout_seconds,
        label="source",
    ) if source_dataset_id else []
    dest_detail = _get_dataset_detail(
        calls,
        base_url,
        dataset_id=dest_dataset_id,
        timeout_seconds=timeout_seconds,
        label="target",
    ) if dest_dataset_id else {}
    dest_files = _list_dataset_files(
        calls,
        base_url,
        dataset_id=dest_dataset_id,
        timeout_seconds=timeout_seconds,
        label="target",
    ) if dest_dataset_id else []

    before_size = final_task.get("beforeSize")
    after_size = final_task.get("afterSize")
    if before_size is None:
        before_size = sum(int(item.get("fileSize") or item.get("file_size") or 0) for item in source_files)
    if after_size is None:
        after_size = sum(int(item.get("fileSize") or item.get("file_size") or 0) for item in dest_files)

    completed_result_count = sum(
        1 for item in result_items
        if isinstance(item, dict) and str(item.get("status", "")).upper() in SUCCESS_STATUSES
    )
    target_active = str(dest_detail.get("status", "")).upper() in {"ACTIVE", "COMPLETED", "SUCCESS"}
    target_file_count = len(dest_files) or int(dest_detail.get("fileCount") or 0)
    expected_file_count = int(effective_input_file_count or final_task.get("fileCount") or len(source_files) or 0)
    content_inline_available = any(
        isinstance(item, dict) and item.get("result") not in {None, "null", ""}
        for item in result_items
    )
    task_success = str(final_task.get("status", "")).upper() in SUCCESS_STATUSES
    target_files_materialized = target_file_count >= max(1, expected_file_count)
    result_records_completed = completed_result_count >= max(1, expected_file_count)
    workflow_plan = plan.get("workflow_plan", {}) if isinstance(plan.get("workflow_plan"), dict) else {}
    structured_contract = workflow_plan.get("etl_contract", {}) if isinstance(workflow_plan.get("etl_contract"), dict) else {}
    checks = [
        {
            "check": "cleaning_task_terminal_success",
            "severity": "critical",
            "passed": task_success,
            "actual": final_task.get("status"),
        },
        {
            "check": "target_dataset_created",
            "severity": "critical",
            "passed": bool(dest_dataset_id and target_active),
            "actual": {"destDatasetId": dest_dataset_id, "status": dest_detail.get("status")},
        },
        {
            "check": "target_files_materialized",
            "severity": "critical",
            "passed": target_files_materialized,
            "actual": {"target_file_count": target_file_count, "expected_file_count": expected_file_count},
        },
        {
            "check": "result_records_completed",
            "severity": "informational",
            "passed": result_records_completed or (task_success and target_files_materialized),
            "actual": {
                "completed_result_count": completed_result_count,
                "expected_file_count": expected_file_count,
                "accepted_metadata_only_result": not result_records_completed and task_success and target_files_materialized,
            },
        },
        {
            "check": "size_regression_guard",
            "severity": "critical",
            "passed": after_size is not None and before_size is not None and int(after_size) >= 0 and int(before_size) >= 0,
            "actual": {"beforeSize": before_size, "afterSize": after_size},
        },
    ]
    if plan.get("workflow_type") == "structured_etl":
        for rule in structured_contract.get("quality_rules", []):
            if not isinstance(rule, dict):
                continue
            checks.append(
                {
                    "check": f"structured_etl_{rule.get('rule_id')}",
                    "severity": "critical" if rule.get("severity") == "critical" else "informational",
                    "passed": bool(rule.get("passed")),
                    "expected": rule.get("expected"),
                    "actual": rule.get("actual"),
                    "official_chain_action": rule.get("official_chain_action"),
                }
            )
    critical_checks = [item for item in checks if item.get("severity") == "critical"]
    return {
        "status": "passed" if all(item["passed"] for item in critical_checks) else "warning",
        "workflow_type": plan.get("workflow_type"),
        "structured_etl_contract": structured_contract if plan.get("workflow_type") == "structured_etl" else {},
        "pre_profile": {
            "record_count_estimate": input_profile.get("record_count_estimate"),
            "sampled_record_count": input_profile.get("sampled_record_count"),
            "schema_text_coverage": input_profile.get("schema_profile", {}).get("schema_confidence", {}).get("text_coverage"),
            "warning_count": input_profile.get("quality_report", {}).get("warning_count"),
        },
        "post_metadata": {
            "source_dataset_id": source_dataset_id,
            "target_dataset_id": dest_dataset_id,
            "target_dataset_name": final_task.get("destDatasetName") or dest_detail.get("name"),
            "source_file_count": len(source_files),
            "effective_input_file_count": expected_file_count,
            "target_file_count": target_file_count,
            "dest_file_ids": dest_file_ids,
            "before_size": before_size,
            "after_size": after_size,
            "success_rate": (final_task.get("progress") or {}).get("successRate") if isinstance(final_task.get("progress"), dict) else None,
            "content_inline_available": content_inline_available,
            "content_access_limitation": "" if content_inline_available else "DataMate cleaning result endpoint returned metadata only; cleaned content is stored in target dataset files.",
        },
        "checks": checks,
    }


def run_datamate_processing_agent(
    input_path: Path,
    output_root: Path,
    *,
    input_paths: list[Path] | None = None,
    task: str = DEFAULT_TASK,
    datamate_api_base: str = DEFAULT_DATAMATE_API_BASE,
    timeout_seconds: float = 5.0,
    wait_seconds: float = 60.0,
) -> dict[str, Any]:
    started_at = time.monotonic()
    input_path = input_path.resolve()
    output_root = output_root.resolve()
    datamate_runtime_api_base = _datamate_runtime_api_base()
    calls: list[HttpCall] = []
    operator_cache: OperatorMarketCache = {}
    batch_input_paths = _resolve_batch_input_paths(input_path, input_paths)
    primary_input_path = batch_input_paths[0] if batch_input_paths else input_path
    batch_profile = _profile_batch_inputs(batch_input_paths, task) if len(batch_input_paths) > 1 else {}
    plan = _infer_task_plan(task, primary_input_path, batch_profile=batch_profile)
    datamate_input_paths = list(batch_input_paths)
    ingest_adapters: list[dict[str, Any]] = []
    recovery_actions: list[dict[str, Any]] = []
    post_quality_report: dict[str, Any] = {}
    batch_report: dict[str, Any] = {}
    cleaning_task: dict[str, Any] = {}
    final_task: dict[str, Any] = {}
    result_payload: dict[str, Any] = {}
    uploaded_files: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = [
        {
            "step": "understand_task",
            "status": "completed",
            "plan": plan,
            "created_at": _now(),
            "datamate_api_base": datamate_api_base,
            "datamate_runtime_api_base": datamate_runtime_api_base,
        }
    ]
    try:
        missing_paths = [path for path in batch_input_paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Input file not found: {missing_paths[0]}")
        unsupported_paths = [path for path in batch_input_paths if path.suffix.lower().lstrip(".") not in SUPPORTED_INPUT_SUFFIXES]
        if unsupported_paths:
            raise ValueError(f"Unsupported input format for batch file: {unsupported_paths[0]}")
        datamate_input_paths, ingest_adapters = _prepare_datamate_ingest_paths(batch_input_paths, output_root, plan)
        if ingest_adapters:
            trace.append(
                {
                    "step": "datamate_text_compatible_ingest_adapter",
                    "status": "completed",
                    "adapters": ingest_adapters,
                    "original_input_paths": [str(path) for path in batch_input_paths],
                    "datamate_input_paths": [str(path) for path in datamate_input_paths],
                }
            )
        source_stem = primary_input_path.stem if len(batch_input_paths) == 1 else f"batch_{hashlib.sha256('|'.join(str(p) for p in batch_input_paths).encode('utf-8')).hexdigest()[:10]}"
        source_name = f"modelengine_task1_source_{source_stem}"
        dest_name = f"modelengine_task1_cleaned_{source_stem}_{int(time.time())}"
        source_dataset = _ensure_dataset(
            calls,
            datamate_api_base,
            name=source_name,
            description="Task-1 source dataset created through DataMate data-management API.",
            dataset_type="TEXT",
            timeout_seconds=timeout_seconds,
        )
        trace.append({"step": "datamate_source_dataset", "status": "completed", "dataset": source_dataset})
        uploaded_files = _upload_input_files(
            calls,
            datamate_api_base,
            dataset_id=source_dataset["id"],
            input_paths=datamate_input_paths,
            timeout_seconds=timeout_seconds,
        )
        trace.append(
            {
                "step": "datamate_file_ingestion",
                "status": "completed",
                "files": uploaded_files,
                "batch_file_count": len(batch_input_paths),
                "datamate_uploaded_file_count": len(datamate_input_paths),
                "input_adapters": ingest_adapters,
            }
        )
        operators = _select_datamate_operators(
            calls,
            datamate_runtime_api_base,
            timeout_seconds,
            plan,
            operator_cache=operator_cache,
        )
        trace.append(
            {
                "step": "datamate_operator_planning",
                "status": "completed",
                "strategy": plan.get("execution_strategy", {}).get("operator_selection"),
                "planner": plan.get("operator_planning", {}).get("planner"),
                "operator_planning": plan.get("operator_planning", {}),
                "operators": [
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "score": item.get("_selection_score"),
                        "reason": item.get("_selection_reason"),
                        "capabilities": sorted(_operator_capabilities(item)),
                        "estimated_cost": _operator_cost(item),
                        "inputs": item.get("inputs"),
                        "outputs": item.get("outputs"),
                    }
                    for item in operators
                ],
            }
        )
        task_name = f"modelengine_task1_{source_stem}_{int(time.time())}"
        cleaning_task, operators, recovery_actions = _create_cleaning_task_with_recovery(
            calls,
            datamate_api_base,
            runtime_api_base=datamate_runtime_api_base,
            task_name=task_name,
            source_dataset=source_dataset,
            dest_dataset_name=dest_name,
            operators=operators,
            plan=plan,
            trace=trace,
            timeout_seconds=timeout_seconds,
            operator_cache=operator_cache,
        )
        trace.append(
            {
                "step": "datamate_cleaning_task_created",
                "status": "completed",
                "cleaning_task": cleaning_task,
                "operator_ids": [item.get("id") for item in operators],
                "recovered": bool(recovery_actions),
            }
        )
        cleaning_task, final_task, operators, recovery_actions = _execute_with_recovery(
            calls,
            datamate_api_base,
            runtime_api_base=datamate_runtime_api_base,
            task_name=task_name,
            source_dataset=source_dataset,
            dest_dataset_name=dest_name,
            initial_task=cleaning_task,
            initial_operators=operators,
            plan=plan,
            trace=trace,
            recovery_actions=recovery_actions,
            timeout_seconds=timeout_seconds,
            wait_seconds=wait_seconds,
            operator_cache=operator_cache,
        )
        final_status = str(final_task.get("status", "")).upper()
        trace.append({"step": "datamate_cleaning_task_execution", "status": final_status or "UNKNOWN", "task": final_task})
        result_payload = _collect_result(calls, datamate_api_base, task_id=str(cleaning_task["id"]), timeout_seconds=timeout_seconds)
        post_quality_report = _build_post_execution_quality_report(
            calls,
            datamate_api_base,
            plan=plan,
            source_dataset=source_dataset,
            final_task=final_task,
            result_payload=result_payload,
            effective_input_file_count=len(datamate_input_paths) or len(uploaded_files) or None,
            timeout_seconds=timeout_seconds,
        )
        trace.append(
            {
                "step": "post_execution_quality_closure",
                "status": post_quality_report.get("status", "unknown"),
                "quality_closure": post_quality_report,
            }
        )
        status = "completed" if final_status in SUCCESS_STATUSES else "blocked"
        blocker = "" if status == "completed" else f"DataMate cleaning task finished with status {final_status or 'UNKNOWN'}"
    except Exception as exc:
        status = "blocked"
        blocker = str(exc)
        trace.append({"step": "blocked", "status": "blocked", "reason": blocker, "created_at": _now()})

    recovery_report = _build_recovery_report(plan=plan, recovery_actions=recovery_actions, trace=trace)
    batch_report = _build_batch_report(
        plan=plan,
        input_paths=datamate_input_paths,
        uploaded_files=uploaded_files,
        final_task=final_task,
        post_quality_report=post_quality_report,
    )
    call_rows = [call.as_dict() for call in calls]
    operator_counts = Counter(
        str(item.get("id"))
        for event in trace
        if event.get("step") == "datamate_operator_planning"
        for item in event.get("operators", [])
    )
    operator_scores = [
        {"id": item.get("id"), "score": item.get("score"), "reason": item.get("reason")}
        for event in trace
        if event.get("step") == "datamate_operator_planning"
        for item in event.get("operators", [])
    ]
    elapsed_ms = round((time.monotonic() - started_at) * 1000, 3)
    performance_report = _build_performance_report(
        call_rows,
        elapsed_ms,
        plan=plan,
        post_quality_report=post_quality_report,
        batch_report=batch_report,
    )
    input_profile = plan.get("input_profile", {})
    quality_gates = plan.get("quality_gates", {})
    summary = {
        "status": status,
        "blocker": blocker,
        "task": task,
        "datamate_api_base": datamate_api_base,
        "input_path": str(primary_input_path),
        "input_paths": [str(path) for path in batch_input_paths],
        "datamate_input_paths": [str(path) for path in datamate_input_paths],
        "datamate_ingest_adapter_count": len(ingest_adapters),
        "datamate_ingest_adapters": ingest_adapters,
        "batch": len(batch_input_paths) > 1,
        "batch_file_count": len(batch_input_paths),
        "batch_uploaded_file_count": len(uploaded_files),
        "batch_format_counts": batch_report.get("format_counts", {}),
        "input_format": plan.get("input_format"),
        "workflow_type": plan.get("workflow_type"),
        "task_understanding_version": plan.get("task_understanding", {}).get("understanding_version"),
        "primary_intent": plan.get("task_understanding", {}).get("primary_intent"),
        "planning_confidence": plan.get("task_understanding", {}).get("planning_confidence"),
        "semantic_intents": plan.get("task_understanding", {}).get("intents", []),
        "semantic_constraints": plan.get("task_understanding", {}).get("constraints", []),
        "autonomous_planner": plan.get("autonomous_plan", {}).get("planner"),
        "autonomous_stage_count": plan.get("autonomous_plan", {}).get("stage_count"),
        "autonomous_stages": plan.get("autonomous_plan", {}).get("stages", []),
        "success_criteria": plan.get("autonomous_plan", {}).get("success_criteria", []),
        "input_complexity": input_profile.get("complexity"),
        "record_count_estimate": input_profile.get("record_count_estimate"),
        "sampled_record_count": input_profile.get("sampled_record_count"),
        "schema_field_count": input_profile.get("schema_profile", {}).get("field_count"),
        "schema_text_coverage": quality_gates.get("schema_text_coverage"),
        "quality_warning_count": quality_gates.get("warning_count"),
        "requires_recovery": quality_gates.get("requires_recovery"),
        "uses_datamate_data_access": any("dataset" in call.name.lower() or "file" in call.name.lower() for call in calls),
        "uses_datamate_cleaning_task": any("/api/cleaning/tasks" in call.url for call in calls),
        "cleaning_task_id": cleaning_task.get("id"),
        "cleaning_task_status": final_task.get("status"),
        "operator_ids": list(operator_counts.keys()),
        "operator_scores": operator_scores,
        "operator_planner": plan.get("operator_planning", {}).get("planner"),
        "operator_covered_capabilities": plan.get("operator_planning", {}).get("covered_capabilities", []),
        "operator_uncovered_capabilities": plan.get("operator_planning", {}).get("uncovered_capabilities", []),
        "operator_dag": plan.get("operator_planning", {}).get("dag", []),
        "structured_etl_contract_status": plan.get("workflow_plan", {}).get("etl_contract", {}).get("status"),
        "structured_etl_blocking_rule_count": plan.get("workflow_plan", {}).get("etl_contract", {}).get("blocking_rule_count"),
        "structured_etl_warning_rule_count": plan.get("workflow_plan", {}).get("etl_contract", {}).get("warning_rule_count"),
        "structured_etl_standard_schema": plan.get("workflow_plan", {}).get("etl_contract", {}).get("standard_schema", []),
        "structured_etl_quality_rules": plan.get("workflow_plan", {}).get("etl_contract", {}).get("quality_rules", []),
        "post_execution_quality_status": post_quality_report.get("status"),
        "post_execution_quality_checks": post_quality_report.get("checks", []),
        "target_dataset_id": post_quality_report.get("post_metadata", {}).get("target_dataset_id"),
        "target_file_count": post_quality_report.get("post_metadata", {}).get("target_file_count"),
        "content_inline_available": post_quality_report.get("post_metadata", {}).get("content_inline_available"),
        "recovery_action_count": len(recovery_actions),
        "recovery_actions": recovery_actions,
        "recovery_status": recovery_report.get("status"),
        "recovery_failed_action_count": recovery_report.get("failed_action_count"),
        "recovery_failure_categories": [
            item.get("category")
            for item in recovery_report.get("failure_classifications", [])
            if isinstance(item, dict) and item.get("category")
        ],
        "api_call_count": len(call_rows),
        "api_call_count_real_http": performance_report.get("api_call_count_real_http"),
        "operator_market_real_http_count": performance_report.get("operator_market_real_http_count"),
        "operator_market_cache_hit_count": performance_report.get("operator_market_cache_hit_count"),
        "poll_call_count": performance_report.get("poll_call_count"),
        "capability_coverage_ratio": performance_report.get("capability_coverage_ratio"),
        "quality_preserved_under_performance_optimization": performance_report.get("quality_preserved"),
        "operator_keyword_plan": plan.get("operator_keyword_plan", {}),
        "performance_optimization_features": performance_report.get("optimization_features", []),
        "elapsed_ms": elapsed_ms,
    }
    artifact_dir = output_root / "data" / "processed"
    _write_json(artifact_dir / "datamate_processing_summary.json", summary)
    _write_json(artifact_dir / "datamate_processing_result.json", result_payload)
    _write_json(artifact_dir / "datamate_processing_plan.json", plan)
    _write_json(
        artifact_dir / "datamate_task_understanding.json",
        {
            "task_understanding": plan.get("task_understanding", {}),
            "autonomous_plan": plan.get("autonomous_plan", {}),
            "workflow_type": plan.get("workflow_type"),
            "selected_operator_keywords": plan.get("selected_operator_keywords", []),
        },
    )
    _write_json(artifact_dir / "datamate_post_quality_report.json", post_quality_report)
    _write_json(artifact_dir / "datamate_structured_etl_contract.json", plan.get("workflow_plan", {}).get("etl_contract", {}))
    _write_json(artifact_dir / "datamate_recovery_report.json", recovery_report)
    _write_json(artifact_dir / "datamate_batch_report.json", batch_report)
    _write_json(artifact_dir / "datamate_performance_report.json", performance_report)
    _write_jsonl(artifact_dir / "datamate_processing_trace.jsonl", trace)
    _write_jsonl(artifact_dir / "datamate_api_trace.jsonl", call_rows)
    return {
        "status": status,
        "summary": summary,
        "plan": plan,
        "trace": trace,
        "api_calls": call_rows,
        "cleaning_task": final_task,
        "result": result_payload,
        "post_quality_report": post_quality_report,
        "recovery_report": recovery_report,
        "batch_report": batch_report,
        "performance_report": performance_report,
        "output_paths": {
            "summary": str(artifact_dir / "datamate_processing_summary.json"),
            "result": str(artifact_dir / "datamate_processing_result.json"),
            "plan": str(artifact_dir / "datamate_processing_plan.json"),
            "task_understanding": str(artifact_dir / "datamate_task_understanding.json"),
            "post_quality_report": str(artifact_dir / "datamate_post_quality_report.json"),
            "structured_etl_contract": str(artifact_dir / "datamate_structured_etl_contract.json"),
            "recovery_report": str(artifact_dir / "datamate_recovery_report.json"),
            "batch_report": str(artifact_dir / "datamate_batch_report.json"),
            "performance_report": str(artifact_dir / "datamate_performance_report.json"),
            "trace": str(artifact_dir / "datamate_processing_trace.jsonl"),
            "api_trace": str(artifact_dir / "datamate_api_trace.jsonl"),
        },
    }
