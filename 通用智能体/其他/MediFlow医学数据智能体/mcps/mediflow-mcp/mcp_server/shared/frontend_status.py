"""
外部服务入口状态模块。

该模块统一返回 Nexent、DataMate 以及可选分析前端的公网入口。
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request


PUBLIC_DOMAIN = os.environ.get("CCF_PUBLIC_DOMAIN", "example.com").strip()


def _service_port(name: str, default: int) -> int:
    env_key = f"CCF_{name.upper()}_PORT"
    value = os.environ.get(env_key, "").strip()
    return int(value) if value else default


def _default_public_url(subdomain: str, path: str = "/") -> str:
    if not PUBLIC_DOMAIN:
        return ""
    return f"https://{subdomain}.{PUBLIC_DOMAIN}{path}"


SERVICE_DEFINITIONS = [
    {
        "name": "analytics_frontend",
        "label": "可选分析前端",
        "port": _service_port("analytics_frontend", 8765),
        "path": "/",
        "env_url": "CCF_ANALYTICS_FRONTEND_URL",
        "public_subdomain": "demo",
        "purpose": "可选的知识图谱、统计图表和查询结果展示入口；不属于 MCP 运行包",
    },
    {
        "name": "datamate_frontend",
        "label": "DataMate 前端",
        "port": _service_port("datamate_frontend", 30000),
        "path": "/",
        "env_url": "CCF_DATAMATE_FRONTEND_URL",
        "public_subdomain": "datamate",
        "start_command": "docker compose up datamate-frontend",
        "purpose": "数据集、算子和清洗任务状态查看",
    },
    {
        "name": "nexent_frontend",
        "label": "Nexent 前端",
        "port": _service_port("nexent_frontend", 3000),
        "path": "/",
        "env_url": "CCF_NEXENT_FRONTEND_URL",
        "public_subdomain": "nexent",
        "start_command": "docker compose up nexent",
        "purpose": "数据处理、知识图谱和数据分析智能体的对话入口",
    },
]


def _public_url(service: dict) -> str:
    configured = os.environ.get(service["env_url"], "").strip()
    legacy_fragments = ("127.0.0.1", "localhost")
    if configured and not any(fragment in configured for fragment in legacy_fragments):
        return configured
    return _default_public_url(service["public_subdomain"], service["path"])


def _probe_local(port: int, path: str = "/", timeout: float = 2.0) -> dict:
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return {"local_url": url, "reachable": True, "http_status": response.status, "error": ""}
    except urllib.error.HTTPError as exc:
        return {"local_url": url, "reachable": True, "http_status": exc.code, "error": ""}
    except Exception as exc:
        return {"local_url": url, "reachable": False, "http_status": None, "error": str(exc)}


def get_validation_frontend_status_payload() -> dict:
    result = {}
    for service in SERVICE_DEFINITIONS:
        url = _public_url(service)
        if url:
            result[service["label"]] = url
    return result
