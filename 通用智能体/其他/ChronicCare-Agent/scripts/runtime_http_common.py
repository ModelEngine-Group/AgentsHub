from __future__ import annotations

"""Shared HTTP helpers for runtime health and integration checks."""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple


def _build_request(url: str, method: str = "GET", payload: Dict[str, Any] | None = None) -> urllib.request.Request:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(url, data=body, headers=headers, method=method.upper())


def request_text(url: str, *, method: str = "GET", payload: Dict[str, Any] | None = None, timeout: int = 30) -> Tuple[str, int, float]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = _build_request(url, method=method, payload=payload)
    started = time.perf_counter()
    with opener.open(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return body, int(response.status), latency_ms


def request_json(url: str, *, method: str = "GET", payload: Dict[str, Any] | None = None, timeout: int = 30) -> Tuple[Dict[str, Any], int, float]:
    body, status_code, latency_ms = request_text(url, method=method, payload=payload, timeout=timeout)
    return json.loads(body), status_code, latency_ms


def safe_request_json(url: str, *, method: str = "GET", payload: Dict[str, Any] | None = None, timeout: int = 30) -> Dict[str, Any]:
    try:
        body, status_code, latency_ms = request_json(url, method=method, payload=payload, timeout=timeout)
        return {
            "ok": status_code < 400,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "json": body,
        }
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status_code": int(exc.code),
            "error": detail or str(exc),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
