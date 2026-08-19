"""Small OpenAI-compatible proxy for DataMate/Nexent model access."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
from urllib import error, parse, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIMENSION = 512
DEFAULT_PORT = 18089
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")
TRUE_VALUES = {"1", "true", "yes", "on"}


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _env(name: str, default: str = "") -> str:
    if os.getenv(name):
        return str(os.getenv(name))
    return _load_env_file(PROJECT_ROOT / ".env").get(name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name, "true" if default else "false").strip().lower()
    return value in TRUE_VALUES


def _json_response(status: int, payload: dict[str, Any]) -> tuple[int, str, bytes]:
    return status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length > 0 else b""
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    normalized = text.lower()
    for match in TOKEN_RE.finditer(normalized):
        value = match.group(0)
        tokens.append(value)
        if any("\u4e00" <= ch <= "\u9fff" for ch in value):
            tokens.extend(value[index : index + size] for size in (1, 2, 3, 4) for index in range(max(0, len(value) - size + 1)))
    return [token for token in tokens if token]


def embed_text(text: str, dimension: int = DEFAULT_DIMENSION) -> list[float]:
    vector = [0.0] * dimension
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + min(len(token), 8) * 0.08
        vector[bucket] += sign * weight

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 8) for value in vector]


def _usage_for_inputs(inputs: list[str]) -> dict[str, int]:
    prompt_tokens = sum(max(1, len(item) // 2) for item in inputs)
    return {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}


def embeddings_response(payload: dict[str, Any], *, dimension: int = DEFAULT_DIMENSION) -> dict[str, Any]:
    raw_input = payload.get("input", "")
    if isinstance(raw_input, list):
        inputs = [str(item) for item in raw_input]
    else:
        inputs = [str(raw_input)]
    model = str(payload.get("model") or "modelengine-local-embedding")
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": embed_text(text, dimension=dimension),
            }
            for index, text in enumerate(inputs)
        ],
        "model": model,
        "usage": _usage_for_inputs(inputs),
    }


def _openai_endpoint(base_url: str, api_path: str) -> str:
    base = base_url.rstrip("/")
    path = "/" + api_path.lstrip("/")
    if base.endswith("/v1") and path.startswith("/v1/"):
        path = path[len("/v1") :]
    return parse.urljoin(base + "/", path.lstrip("/"))


def forward_chat_completion(payload: dict[str, Any], *, timeout_seconds: float = 45.0) -> dict[str, Any]:
    upstream_base = _env("MODEL_PROXY_UPSTREAM_API_BASE", _env("OPENAI_API_BASE", "http://pixel.try-chatapi.com/"))
    upstream_key = _env("MODEL_PROXY_UPSTREAM_API_KEY", _env("OPENAI_API_KEY", ""))
    upstream_model = _env("MODEL_PROXY_UPSTREAM_LLM_MODEL", _env("LLM_MODEL", "gpt-5.4"))
    payload = dict(payload)
    payload.setdefault("model", upstream_model)
    payload.setdefault("stream", False)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    endpoint = _openai_endpoint(upstream_base, "v1/chat/completions")
    req = request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {upstream_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with request.urlopen(req, timeout=timeout_seconds) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fallback_chat_completion(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages") if isinstance(payload, dict) else []
    last_user = ""
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                last_user = str(message.get("content", ""))
                break
    return {
        "id": f"chatcmpl-local-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(payload.get("model") or "modelengine-local-chat"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"ok: {last_user[:120]}",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": max(1, len(last_user) // 2), "completion_tokens": 4, "total_tokens": max(5, len(last_user) // 2 + 4)},
    }


def upstream_error_response(exc: BaseException) -> dict[str, Any]:
    upstream_base = _env("MODEL_PROXY_UPSTREAM_API_BASE", _env("OPENAI_API_BASE", ""))
    return {
        "error": {
            "type": "upstream_unavailable",
            "message": (
                "Model proxy upstream request failed and local fallback is disabled. "
                "Start the remote NPU/OpenAI-compatible service or enable "
                "MODEL_PROXY_ALLOW_LOCAL_FALLBACK=true for development only."
            ),
            "upstream_base": upstream_base,
            "exception": repr(exc),
        }
    }


class ModelAPIProxyHandler(BaseHTTPRequestHandler):
    server_version = "ModelEngineModelProxy/1.0"

    def do_GET(self) -> None:  # noqa: N802
        route = parse.urlparse(self.path).path.rstrip("/")
        if route in ("", "/v1", "/health"):
            self._send(*_json_response(HTTPStatus.OK, {"status": "ok"}))
            return
        if route in ("/models", "/v1/models"):
            self._send(
                *_json_response(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [
                            {"id": _env("LLM_MODEL", "gpt-5.4"), "object": "model", "owned_by": "upstream"},
                            {"id": "modelengine-local-embedding", "object": "model", "owned_by": "modelengine"},
                            {"id": "text-embedding-3-small", "object": "model", "owned_by": "modelengine"},
                        ],
                    },
                )
            )
            return
        self._send(*_json_response(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": route}))

    def do_POST(self) -> None:  # noqa: N802
        route = parse.urlparse(self.path).path.rstrip("/")
        try:
            payload = _read_json_body(self)
        except json.JSONDecodeError as exc:
            self._send(*_json_response(HTTPStatus.BAD_REQUEST, {"error": f"invalid_json: {exc}"}))
            return

        if route in ("/embeddings", "/v1/embeddings"):
            self._send(*_json_response(HTTPStatus.OK, embeddings_response(payload)))
            return

        if route in ("/chat/completions", "/v1/chat/completions"):
            try:
                response_payload = forward_chat_completion(payload)
            except (OSError, error.URLError, error.HTTPError, json.JSONDecodeError) as exc:
                if _env_bool("MODEL_PROXY_ALLOW_LOCAL_FALLBACK", False):
                    response_payload = fallback_chat_completion(payload)
                    self._send(*_json_response(HTTPStatus.OK, response_payload))
                    return
                self._send(*_json_response(HTTPStatus.BAD_GATEWAY, upstream_error_response(exc)))
                return
            self._send(*_json_response(HTTPStatus.OK, response_payload))
            return

        self._send(*_json_response(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": route}))

    def log_message(self, format: str, *args: Any) -> None:
        if os.getenv("MODEL_PROXY_DEBUG") == "true":
            super().log_message(format, *args)

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def start_model_api_proxy(host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> ReusableThreadingHTTPServer:
    server = ReusableThreadingHTTPServer((host, port), ModelAPIProxyHandler)
    thread = threading.Thread(target=server.serve_forever, name="model-api-proxy", daemon=True)
    thread.start()
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small OpenAI-compatible model proxy.")
    parser.add_argument("--host", default=os.getenv("MODEL_PROXY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MODEL_PROXY_PORT", str(DEFAULT_PORT))))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    server = start_model_api_proxy(args.host, args.port)
    print(f"Model proxy listening on http://{args.host}:{args.port}/v1")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
