from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

SAFETY_NOTE = "本项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。"
DEFAULT_MODEL_SERVICE_URLS = [
    "http://127.0.0.1:18080/health",
    "http://127.0.0.1:18081/health",
]
RECOMMENDED_NPU_TARGETS = [
    "chronic_entity_extract",
    "chronic_relation_extract",
    "text_embedding",
    "open_nl2sql_model_inference",
]


@dataclass
class NPURuntimeStatus:
    npu_available: bool
    backend: str
    reason: str
    fallback_enabled: bool = True


def _try_import(module_name: str) -> tuple[bool, str | None]:
    try:
        __import__(module_name)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _run_probe(command: List[str], timeout: float = 3.0) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "available": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip()[:2000],
            "stderr": completed.stderr.strip()[:2000],
            "duration_sec": round(time.perf_counter() - started, 4),
        }
    except FileNotFoundError as exc:
        return {"available": False, "error": str(exc), "duration_sec": round(time.perf_counter() - started, 4)}
    except subprocess.TimeoutExpired as exc:
        return {"available": False, "error": f"timeout after {timeout}s: {exc}", "duration_sec": round(time.perf_counter() - started, 4)}


def _check_http_health(url: str, timeout: float = 1.5) -> Dict[str, Any]:
    started = time.perf_counter()
    request = urllib.request.Request(url, method="GET", headers={"Accept": "application/json,text/plain,*/*"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(500).decode("utf-8", errors="replace")
            return {
                "url": url,
                "available": 200 <= response.status < 500,
                "status_code": response.status,
                "body_preview": body,
                "duration_sec": round(time.perf_counter() - started, 4),
            }
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        return {"url": url, "available": False, "error": str(exc), "duration_sec": round(time.perf_counter() - started, 4)}


def detect_npu_runtime() -> Dict[str, Any]:
    torch_available, torch_error = _try_import("torch")
    torch_npu_available, torch_npu_error = _try_import("torch_npu")
    torch_npu_device_available = False
    torch_npu_device_error = None
    if torch_available:
        try:
            import torch  # type: ignore

            npu_obj = getattr(torch, "npu", None)
            if npu_obj is not None and hasattr(npu_obj, "is_available"):
                torch_npu_device_available = bool(npu_obj.is_available())
            else:
                torch_npu_device_error = "torch.npu is not exposed by this torch build"
        except Exception as exc:
            torch_npu_device_error = str(exc)

    npu_smi = _run_probe(["npu-smi", "info"])
    cann_env_keys = ["ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "LD_LIBRARY_PATH"]
    cann_env = {key: "configured" for key in cann_env_keys if os.environ.get(key)}
    model_services = [_check_http_health(url) for url in DEFAULT_MODEL_SERVICE_URLS]
    model_service_detected = any(item.get("available") for item in model_services)
    device_candidates = ["/dev/davinci0", "/dev/davinci_manager", "/dev/hisi_hdc"]
    visible_devices = [path for path in device_candidates if os.path.exists(path)]

    reasons: List[str] = []
    if not torch_available:
        reasons.append(f"torch import failed: {torch_error}")
    if not torch_npu_available:
        reasons.append(f"torch_npu import failed: {torch_npu_error}")
    if torch_npu_available and not torch_npu_device_available:
        reasons.append(torch_npu_device_error or "torch_npu imported but torch.npu is not available")
    if not npu_smi.get("available"):
        reasons.append("npu-smi info is not executable in the current runtime")
    if not visible_devices:
        reasons.append("no Ascend device nodes are visible in the current runtime")
    if not model_service_detected:
        reasons.append("no local NPU model service detected on 18080/18081")

    npu_available = bool(torch_npu_available and torch_npu_device_available) or model_service_detected
    backend = "torch_npu" if torch_npu_available and torch_npu_device_available else ("http_model_service" if model_service_detected else "cpu_fallback")
    return {
        "status": "success",
        "npu_available": npu_available,
        "backend": backend,
        "torch_available": torch_available,
        "torch_error": torch_error,
        "torch_npu_available": torch_npu_available,
        "torch_npu_error": torch_npu_error,
        "torch_npu_device_available": torch_npu_device_available,
        "torch_npu_device_error": torch_npu_device_error,
        "npu_smi_available": bool(npu_smi.get("available")),
        "npu_smi": npu_smi,
        "cann_env_detected": bool(cann_env),
        "cann_env": cann_env,
        "visible_npu_devices": visible_devices,
        "model_service_detected": model_service_detected,
        "model_services": model_services,
        "fallback_enabled": True,
        "fallback_required": not npu_available,
        "recommended_npu_targets": RECOMMENDED_NPU_TARGETS,
        "notes": reasons,
        "reason": "；".join(reasons) if reasons else "NPU runtime is available.",
        "safety_note": SAFETY_NOTE,
    }


def select_npu_backend(prefer_service: bool = True) -> Dict[str, Any]:
    status = detect_npu_runtime()
    if prefer_service and status.get("model_service_detected"):
        status["backend"] = "http_model_service"
    elif status.get("torch_npu_device_available"):
        status["backend"] = "torch_npu"
    else:
        status["backend"] = "cpu_fallback"
    return status


def run_with_fallback(
    npu_fn: Callable[[], Dict[str, Any]],
    cpu_fn: Callable[[], Dict[str, Any]],
    *,
    use_npu: bool,
    fallback: bool = True,
    prefer_service: bool = True,
) -> Dict[str, Any]:
    status = select_npu_backend(prefer_service=prefer_service)
    started = time.perf_counter()
    if use_npu and status.get("npu_available"):
        try:
            payload = npu_fn()
            payload.setdefault("backend", status.get("backend"))
            payload.setdefault("fallback_used", False)
            payload.setdefault("runtime_status", status)
            payload["duration_sec"] = payload.get("duration_sec", round(time.perf_counter() - started, 4))
            return payload
        except Exception as exc:
            if not fallback:
                raise
            fallback_payload = cpu_fn()
            fallback_payload.update(
                {
                    "backend": "cpu_fallback",
                    "fallback_used": True,
                    "fallback_reason": f"NPU execution failed: {exc}",
                    "runtime_status": status,
                    "duration_sec": fallback_payload.get("duration_sec", round(time.perf_counter() - started, 4)),
                }
            )
            return fallback_payload
    if use_npu and not fallback:
        raise RuntimeError(status.get("reason") or "NPU unavailable and fallback disabled.")
    payload = cpu_fn()
    payload.update(
        {
            "backend": "cpu_fallback",
            "fallback_used": bool(use_npu),
            "fallback_reason": status.get("reason") if use_npu else "use_npu=false; CPU stable mainline selected.",
            "runtime_status": status,
            "duration_sec": payload.get("duration_sec", round(time.perf_counter() - started, 4)),
        }
    )
    return payload


def to_markdown_report(title: str, payload: Dict[str, Any]) -> str:
    lines = [
        f"# {title}",
        "",
        f"- status: `{payload.get('status')}`",
        f"- npu_available: `{payload.get('npu_available')}`",
        f"- backend: `{payload.get('backend')}`",
        f"- fallback_enabled: `{payload.get('fallback_enabled', True)}`",
        "",
        "## Recommended NPU Targets",
    ]
    for item in payload.get("recommended_npu_targets", RECOMMENDED_NPU_TARGETS):
        lines.append(f"- {item}")
    if payload.get("notes"):
        lines.extend(["", "## Notes"])
        for item in payload.get("notes", []):
            lines.append(f"- {item}")
    lines.extend(["", f"安全声明：{payload.get('safety_note', SAFETY_NOTE)}", ""])
    return "\n".join(lines)


def dumps_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
