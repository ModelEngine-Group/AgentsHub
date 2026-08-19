"""Runtime device selection helpers for CPU, CUDA, and Ascend NPU paths."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any


@dataclass(frozen=True)
class DeviceSpec:
    """Small, import-safe description of the selected compute device."""

    kind: str
    device: str
    module: str | None = None
    reason: str = ""


def get_device(prefer: str = "auto") -> DeviceSpec:
    """Return the preferred available device without requiring torch at import time."""

    preference = (prefer or "auto").lower()
    if preference not in {"auto", "npu", "cuda", "cpu"}:
        raise ValueError("prefer must be one of: auto, npu, cuda, cpu")
    if preference == "cpu":
        return _cpu_device("cpu requested")

    errors: list[str] = []
    if preference in {"auto", "npu"}:
        npu = _detect_npu(errors)
        if npu:
            return npu
        if preference == "npu":
            return _cpu_device("; ".join(errors) or "npu unavailable")

    if preference in {"auto", "cuda"}:
        cuda = _detect_cuda(errors)
        if cuda:
            return cuda
        if preference == "cuda":
            return _cpu_device("; ".join(errors) or "cuda unavailable")

    return _cpu_device("; ".join(errors) or "accelerator unavailable")


def model_load_kwargs(device: DeviceSpec | None = None) -> dict[str, Any]:
    """Return Hugging Face model-loading kwargs for the selected device."""

    selected = device or get_device()
    if selected.kind == "npu":
        return {}
    if selected.kind == "cuda":
        return {"device_map": "auto"}
    return {"device_map": "cpu"}


def move_model_to_device(model: Any, device: DeviceSpec | None = None) -> Any:
    """Move a loaded model to devices that are not handled by device_map."""

    selected = device or get_device()
    if selected.kind != "npu":
        return model

    to_device = getattr(model, "to", None)
    if callable(to_device):
        moved = to_device(selected.device)
        return moved if moved is not None else model
    return model


def _detect_npu(errors: list[str]) -> DeviceSpec | None:
    try:
        importlib.import_module("torch_npu")
    except Exception as exc:
        errors.append(f"torch_npu: {str(exc) or type(exc).__name__}")
        return None

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        errors.append(f"torch: {str(exc) or type(exc).__name__}")
        return None

    npu = getattr(torch, "npu", None)
    is_available = getattr(npu, "is_available", None)
    if callable(is_available):
        try:
            if is_available():
                return DeviceSpec(
                    kind="npu",
                    device="npu:0",
                    module="torch_npu",
                    reason="torch_npu and torch.npu are available",
                )
        except Exception as exc:
            errors.append(f"torch.npu.is_available: {str(exc) or type(exc).__name__}")
            return None
    errors.append("torch.npu is unavailable")
    return None


def _detect_cuda(errors: list[str]) -> DeviceSpec | None:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        errors.append(f"torch: {str(exc) or type(exc).__name__}")
        return None

    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if callable(is_available):
        try:
            if is_available():
                return DeviceSpec(
                    kind="cuda",
                    device="cuda:0",
                    module="torch",
                    reason="torch.cuda is available",
                )
        except Exception as exc:
            errors.append(f"torch.cuda.is_available: {str(exc) or type(exc).__name__}")
            return None
    errors.append("torch.cuda is unavailable")
    return None


def _cpu_device(reason: str) -> DeviceSpec:
    return DeviceSpec(kind="cpu", device="cpu", module=None, reason=reason)
