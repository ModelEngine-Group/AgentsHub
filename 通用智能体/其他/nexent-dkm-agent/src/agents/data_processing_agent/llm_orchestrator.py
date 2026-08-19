"""OpenAI-compatible LLM smoke orchestration for the task-1 agent."""

from __future__ import annotations

import http.client
import json
import socket
import ssl
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.agents.data_processing_agent.nexent_adapter import TOOL_NAME
from src.common.llm_config import (
    load_env_config_values,
    load_json_config_values,
    openai_extra_body,
)

SYSTEM_PROMPT = (
    "You are a Nexent-style data processing agent orchestrator. "
    "Return exactly one JSON object with keys 'tool' and 'arguments'. "
    f"The only allowed tool is '{TOOL_NAME}'. "
    "Use datamate_mode='dry_run' unless the user explicitly requests a real "
    "DataMate submission."
)

ALLOWED_ENV_KEYS = {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"}
ALLOWED_CONFIG_KEYS = {"api_key", "base_url", "model_name"}

_FAKE_IP_PREFIXES = ("198.16.", "198.17.", "198.18.", "198.19.")


def load_config_file(path: str | Path) -> dict[str, str]:
    """Load a local JSON config file for private LLM smoke tests."""

    config_path = Path(path)
    if not config_path.exists():
        return {}

    payload = load_json_config_values(config_path)
    values: dict[str, str] = {}
    for key in ALLOWED_CONFIG_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            values[key] = value.strip()
    return values


def load_env_file(path: str | Path) -> dict[str, str]:
    """Load a small local env file without expanding shell syntax."""

    env_path = Path(path)
    if not env_path.exists():
        return {}

    parsed = load_env_config_values(env_path)
    reverse_keys = {value: key for key, value in {
        "OPENAI_API_KEY": "api_key",
        "OPENAI_BASE_URL": "base_url",
        "OPENAI_MODEL": "model_name",
    }.items()}
    values = {
        reverse_keys[key]: str(value)
        for key, value in parsed.items()
        if key in reverse_keys and str(value).strip()
    }
    return values


def build_chat_completion_payload(
    agent_spec: dict[str, Any],
    model_name: str,
    task_request: str,
    input_path: str | None = None,
    datamate_mode: str = "dry_run",
) -> dict[str, Any]:
    """Build a secret-free OpenAI-compatible chat completion payload."""

    user_payload = {
        "agent_spec": agent_spec,
        "task_request": task_request,
        "input_path": input_path,
        "datamate_mode": datamate_mode,
        "required_response_shape": {
            "tool": TOOL_NAME,
            "arguments": {
                "task_request": task_request,
                "input_path": input_path,
                "datamate_mode": datamate_mode,
            },
        },
    }
    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
        "temperature": 0,
    }


def _is_fake_ip(ip: str) -> bool:
    return any(ip.startswith(prefix) for prefix in _FAKE_IP_PREFIXES)


def _resolve_real_ip_via_dns(hostname: str) -> str | None:
    """Resolve the real IP by querying public DNS over UDP directly."""
    import struct

    for dns_server in ["223.5.5.5", "114.114.114.114"]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(3)
            msg = b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
            for part in hostname.rstrip(".").split("."):
                encoded = part.encode()
                msg += bytes([len(encoded)]) + encoded
            msg += b"\x00\x00\x01\x00\x01"
            s.sendto(msg, (dns_server, 53))
            data, _ = s.recvfrom(1024)
            s.close()
            answer_count = struct.unpack("!H", data[6:8])[0]
            if answer_count == 0:
                continue
            idx = 12
            while idx < len(data) and data[idx] != 0:
                idx += data[idx] + 1
            idx += 5  # skip QTYPE + QCLASS
            for _ in range(answer_count):
                # skip NAME (may be compressed pointer)
                if idx < len(data) and (data[idx] & 0xC0) == 0xC0:
                    idx += 2
                else:
                    while idx < len(data) and data[idx] != 0:
                        idx += data[idx] + 1
                    idx += 1  # skip null terminator
                if idx + 10 > len(data):
                    break
                rtype = struct.unpack("!H", data[idx:idx + 2])[0]
                rdlen = struct.unpack("!H", data[idx + 8:idx + 10])[0]
                idx += 10
                rdata = data[idx:idx + rdlen]
                idx += rdlen
                if rtype == 1 and len(rdata) == 4:
                    ip = ".".join(str(b) for b in rdata)
                    if not _is_fake_ip(ip):
                        return ip
        except Exception:
            continue
    return None


def _direct_https_post(
    hostname: str,
    port: int,
    path: str,
    headers: dict[str, str],
    body: bytes,
    real_ip: str,
    timeout: float,
) -> dict[str, Any]:
    """Send an HTTPS POST via http.client connecting to *real_ip* directly."""

    conn = http.client.HTTPSConnection(
        real_ip, port, timeout=timeout,
        context=ssl.create_default_context(),
    )
    # Override the Host header so the server sees the original hostname.
    full_headers = {**headers, "Host": hostname}
    conn.request("POST", path, body=body, headers=full_headers)
    resp = conn.getresponse()
    resp_data = resp.read().decode("utf-8")
    conn.close()
    if resp.status >= 400:
        raise RuntimeError(
            f"LLM endpoint returned HTTP {resp.status}: {resp_data[:300]}"
        )
    return json.loads(resp_data)


def request_chat_completion(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat completions endpoint.

    When a Clash TUN with fake-ip is active, DNS returns virtual IPs in
    198.16-19.x.x which route traffic through proxy nodes.  For domestic
    API endpoints this causes SSL handshake timeouts.  This function
    detects the fake-ip scenario and falls back to connecting directly
    to the resolved real IP.
    """

    url = f"{base_url.rstrip('/')}/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Check if the hostname resolves to a Clash fake-ip.
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    try:
        resolved_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        resolved_ip = ""

    if _is_fake_ip(resolved_ip):
        real_ip = _resolve_real_ip_via_dns(hostname)
        if real_ip:
            return _direct_https_post(
                hostname=hostname,
                port=parsed.port or 443,
                path=parsed.path or "/",
                headers=headers,
                body=body,
                real_ip=real_ip,
                timeout=timeout,
            )

    request = Request(url=url, method="POST", data=body, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def completion_text(response: dict[str, Any]) -> str:
    """Extract the first assistant text from an OpenAI-compatible response."""

    choices = response.get("choices", [])
    if not choices:
        raise ValueError("LLM response did not include choices.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response did not include assistant content.")
    return content


def parse_tool_call_json(text: str) -> dict[str, Any]:
    """Parse the JSON tool call returned by the LLM."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM tool call must be a JSON object.")
    return payload


def validate_tool_call(
    tool_call: dict[str, Any],
    allow_submit: bool = False,
) -> dict[str, Any]:
    """Validate and normalize the LLM-selected tool call."""

    if tool_call.get("tool") != TOOL_NAME:
        raise ValueError(f"LLM selected unsupported tool: {tool_call.get('tool')}")
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("LLM tool call must include object arguments.")

    mode = arguments.get("datamate_mode") or "dry_run"
    if mode not in {"dry_run", "submit", "auto"}:
        raise ValueError("datamate_mode must be 'dry_run', 'submit', or 'auto'.")
    if mode == "submit" and not allow_submit:
        raise ValueError("LLM requested submit mode; rerun with allow_submit=True.")

    return {
        "tool": TOOL_NAME,
        "arguments": {
            "task_request": arguments.get("task_request"),
            "input_path": arguments.get("input_path"),
            "datamate_mode": mode,
        },
    }


# ---------------------------------------------------------------------------
# LLM-assisted planning
# ---------------------------------------------------------------------------

PLANNING_SYSTEM_PROMPT = (
    "You are a data processing pipeline planner. "
    "Given a user request, a data profile, and a list of available operators, "
    "return a JSON object with the optimal operator sequence. "
    "Keys: operators (list of strings), rationale (list of strings), "
    "task_type (string), data_type (string), intent_keywords (list of strings), "
    "confidence (float 0-1). Only use operators from the provided list."
)


def request_plan(
    base_url: str,
    api_key: str,
    model_name: str,
    task_request: str,
    data_profile: dict[str, Any] | None = None,
    available_operators: list[str] | None = None,
    timeout: float = 30.0,
    llm_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ask the LLM to produce an operator plan for a data processing task."""

    user_content: dict[str, Any] = {
        "task_request": task_request,
        "available_operators": available_operators or [],
    }
    if data_profile:
        user_content["data_profile"] = {
            k: data_profile[k]
            for k in ("file_name", "row_count", "column_count", "duplicate_rows", "missing_cells", "columns")
            if k in data_profile
        }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
        ],
        "temperature": 0,
    }
    payload.update(openai_extra_body(llm_config))

    response = request_chat_completion(
        base_url=base_url,
        api_key=api_key,
        payload=payload,
        timeout=timeout,
    )

    raw_text = completion_text(response)
    try:
        parsed = parse_tool_call_json(raw_text)
    except (ValueError, json.JSONDecodeError):
        parsed = {"operators": []}

    if "operators" not in parsed or not isinstance(parsed["operators"], list):
        parsed["operators"] = []

    return parsed
