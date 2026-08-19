"""Safe online integration helpers for Nexent OpenAPI services."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


class NexentOnlineClient:
    """Minimal client for Nexent's OpenAPI service management endpoints."""

    def __init__(
        self,
        base_url: str,
        authorization: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = _validate_base_url(base_url)
        self.authorization = authorization
        self.timeout = timeout

    def list_openapi_services(self) -> dict[str, Any]:
        """List OpenAPI services without mutating Nexent state."""

        try:
            response = self._request_json("GET", "/api/tool/openapi_services")
        except Exception as exc:
            return _request_failure(exc, self.base_url)

        services = response.get("data") if isinstance(response, dict) else []
        if not isinstance(services, list):
            services = []
        return {
            "status": "available",
            "base_url": self.base_url,
            "service_count": len(services),
            "service_names": [
                name
                for service in services
                if isinstance(service, dict)
                and (name := _service_name(service)) is not None
            ],
            "services": services,
        }

    def list_tools(self) -> dict[str, Any]:
        """List the current tenant's tools without changing the catalog."""

        try:
            response = self._request_json("GET", "/api/tool/list")
        except Exception as exc:
            return _request_failure(exc, self.base_url)
        tools = response if isinstance(response, list) else []
        return {
            "status": "available",
            "base_url": self.base_url,
            "tool_count": len(tools),
            "tools": tools,
        }

    def list_agents(self) -> dict[str, Any]:
        """List agents before attempting a name-sensitive creation."""

        try:
            response = self._request_json("GET", "/api/agent/list")
        except Exception as exc:
            return _request_failure(exc, self.base_url)
        agents = response if isinstance(response, list) else []
        return {
            "status": "available",
            "base_url": self.base_url,
            "agent_count": len(agents),
            "agents": agents,
        }

    def verify_existing_agent(self, name: str) -> dict[str, Any]:
        """Verify an already-created agent by name without mutating Nexent."""

        if not name.isidentifier():
            raise ValueError("Nexent agent name must be a valid Python identifier.")

        existing_agents = self.list_agents()
        if existing_agents.get("status") != "available":
            return {
                "status": "preflight_failed",
                "submitted": False,
                "endpoint": "/api/agent/by-name",
                "verification": existing_agents,
            }
        for agent in existing_agents["agents"]:
            if isinstance(agent, dict) and str(agent.get("name")) == name:
                agent_id = agent.get("agent_id")
                try:
                    verification = self._request_json(
                        "GET",
                        f"/api/agent/by-name/{quote(name, safe='')}",
                    )
                except Exception as exc:
                    return {
                        "status": "submitted_unverified",
                        "submitted": False,
                        "verified": False,
                        "preexisting": True,
                        "endpoint": "/api/agent/by-name",
                        "agent_id": agent_id,
                        "agent": agent,
                        "verification": _request_failure(exc, self.base_url),
                    }
                verified = (
                    agent_id is not None
                    and isinstance(verification, dict)
                    and str(verification.get("agent_id")) == str(agent_id)
                )
                return {
                    "status": "verified" if verified else "submitted_unverified",
                    "submitted": False,
                    "verified": verified,
                    "preexisting": True,
                    "verification_scope": "identity_only",
                    "configuration_verified": False,
                    "endpoint": "/api/agent/by-name",
                    "agent_id": agent_id,
                    "agent": agent,
                    "verification": verification,
                }
        return {
            "status": "not_found",
            "submitted": False,
            "verified": False,
            "endpoint": "/api/agent/by-name",
            "message": f"Nexent agent '{name}' was not found.",
        }

    def refresh_tool_catalog(self, *, allow_write: bool = False) -> dict[str, Any]:
        """Refresh Nexent's tool database after explicit write authorization."""

        if not allow_write:
            return {
                "status": "write_blocked",
                "submitted": False,
                "endpoint": "/api/tool/scan_tool",
                "message": "Set allow_write=True for an explicit tool refresh.",
            }
        try:
            response = self._request_json("GET", "/api/tool/scan_tool")
        except Exception as exc:
            return {
                **_request_failure(exc, self.base_url),
                "submitted": False,
                "endpoint": "/api/tool/scan_tool",
            }
        return {
            "status": "refreshed",
            "submitted": True,
            "endpoint": "/api/tool/scan_tool",
            "response": response,
        }

    def import_openapi_service(
        self,
        service_name: str,
        server_url: str,
        openapi_json: dict[str, Any],
        *,
        service_description: str | None = None,
        force_update: bool = False,
        mode: str = "dry_run",
        allow_write: bool = False,
    ) -> dict[str, Any]:
        """Prepare or import an OpenAPI service, then verify it by listing."""

        if mode not in {"dry_run", "submit"}:
            return {
                "status": "invalid_mode",
                "submitted": False,
                "message": "Nexent mode must be 'dry_run' or 'submit'.",
            }

        payload = {
            "service_name": service_name,
            "server_url": _validate_base_url(server_url),
            "openapi_json": openapi_json,
            "service_description": service_description
            or openapi_json.get("info", {}).get("description")
            or service_name,
            "force_update": force_update,
        }
        if mode == "dry_run":
            return {
                "status": "prepared",
                "submitted": False,
                "endpoint": "/api/tool/openapi_service",
                "payload": payload,
            }
        if not allow_write:
            return {
                "status": "write_blocked",
                "submitted": False,
                "endpoint": "/api/tool/openapi_service",
                "message": "Set allow_write=True for an explicit Nexent import.",
                "payload": payload,
            }

        existing_services = self.list_openapi_services()
        if existing_services.get("status") != "available":
            return {
                "status": "preflight_failed",
                "submitted": False,
                "endpoint": "/api/tool/openapi_service",
                "verification": existing_services,
            }
        existing_service = next(
            (
                service
                for service in existing_services["services"]
                if isinstance(service, dict)
                and _service_name(service) == service_name
            ),
            None,
        )
        if existing_service is not None and not force_update:
            if _openapi_service_matches(existing_service, payload):
                return {
                    "status": "verified",
                    "submitted": False,
                    "verified": True,
                    "preexisting": True,
                    "endpoint": "/api/tool/openapi_service",
                    "service_name": service_name,
                    "verification": existing_services,
                }
            return {
                "status": "update_blocked",
                "submitted": False,
                "verified": False,
                "requires_force_update": True,
                "endpoint": "/api/tool/openapi_service",
                "service_name": service_name,
                "message": (
                    "A conflicting service already exists. Review it and pass "
                    "force_update=True to replace its configuration."
                ),
            }

        try:
            response = self._request_json(
                "POST",
                "/api/tool/openapi_service",
                payload,
            )
        except Exception as exc:
            return {
                **_request_failure(exc, self.base_url),
                "submitted": False,
                "endpoint": "/api/tool/openapi_service",
            }

        verification = self.list_openapi_services()
        verified = (
            verification.get("status") == "available"
            and service_name in verification.get("service_names", [])
        )
        return {
            "status": "verified" if verified else "submitted_unverified",
            "submitted": True,
            "verified": verified,
            "endpoint": "/api/tool/openapi_service",
            "service_name": service_name,
            "response": response,
            "verification": verification,
        }

    def create_agent(
        self,
        payload: dict[str, Any],
        *,
        mode: str = "dry_run",
        allow_write: bool = False,
    ) -> dict[str, Any]:
        """Create a Nexent agent and verify it through the by-name endpoint."""

        if mode not in {"dry_run", "submit"}:
            return {
                "status": "invalid_mode",
                "submitted": False,
                "message": "Nexent mode must be 'dry_run' or 'submit'.",
            }
        name = str(payload.get("name") or "")
        if not name.isidentifier():
            raise ValueError("Nexent agent name must be a valid Python identifier.")
        if mode == "dry_run":
            return {
                "status": "prepared",
                "submitted": False,
                "endpoint": "/api/agent/update",
                "payload": payload,
            }
        if not allow_write:
            return {
                "status": "write_blocked",
                "submitted": False,
                "endpoint": "/api/agent/update",
                "message": "Set allow_write=True for explicit agent creation.",
                "payload": payload,
            }

        existing_agents = self.list_agents()
        if existing_agents.get("status") != "available":
            return {
                "status": "preflight_failed",
                "submitted": False,
                "endpoint": "/api/agent/update",
                "verification": existing_agents,
            }
        for agent in existing_agents["agents"]:
            if isinstance(agent, dict) and str(agent.get("name")) == name:
                agent_id = agent.get("agent_id")
                try:
                    verification = self._request_json(
                        "GET",
                        f"/api/agent/by-name/{quote(name, safe='')}",
                    )
                except Exception as exc:
                    return {
                        "status": "submitted_unverified",
                        "submitted": False,
                        "verified": False,
                        "preexisting": True,
                        "endpoint": "/api/agent/update",
                        "agent_id": agent_id,
                        "agent": agent,
                        "verification": _request_failure(exc, self.base_url),
                    }
                by_name_id = (
                    verification.get("agent_id")
                    if isinstance(verification, dict)
                    else None
                )
                by_name_matched = (
                    by_name_id is not None and str(by_name_id) == str(agent_id)
                )
                verified = agent_id is not None and (
                    by_name_matched or by_name_id is not None
                )
                return {
                    "status": "verified" if verified else "submitted_unverified",
                    "submitted": False,
                    "verified": verified,
                    "preexisting": True,
                    "verification_scope": "identity_only",
                    "configuration_verified": False,
                    "endpoint": "/api/agent/update",
                    "agent_id": agent_id,
                    "agent": agent,
                    "verification": verification,
                    "by_name_agent_id": by_name_id,
                    "by_name_matched": by_name_matched,
                }

        try:
            response = self._request_json("POST", "/api/agent/update", payload)
        except Exception as exc:
            return {
                **_request_failure(exc, self.base_url),
                "submitted": False,
                "endpoint": "/api/agent/update",
            }

        agent_id = response.get("agent_id") if isinstance(response, dict) else None
        try:
            verification = self._request_json(
                "GET",
                f"/api/agent/by-name/{quote(name, safe='')}",
            )
        except Exception as exc:
            return {
                "status": "submitted_unverified",
                "submitted": True,
                "verified": False,
                "agent_id": agent_id,
                "response": response,
                "verification": _request_failure(exc, self.base_url),
            }
        by_name_id = (
            verification.get("agent_id") if isinstance(verification, dict) else None
        )
        by_name_matched = (
            by_name_id is not None and str(by_name_id) == str(agent_id)
        )
        verified = agent_id is not None and (
            by_name_matched or by_name_id is not None
        )
        return {
            "status": "verified" if verified else "submitted_unverified",
            "submitted": True,
            "verified": verified,
            "verification_scope": "identity_only",
            "configuration_verified": False,
            "agent_id": agent_id,
            "response": response,
            "verification": verification,
            "by_name_agent_id": by_name_id,
            "by_name_matched": by_name_matched,
        }

    def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if self.authorization:
            headers["Authorization"] = self.authorization
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            url=f"{self.base_url}{path}",
            data=data,
            method=method,
            headers=headers,
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def load_authorization(
    *,
    token: str | None = None,
    token_file: str | Path | None = None,
) -> str | None:
    """Load a bearer token without requiring environment-variable changes."""

    if token and token_file:
        raise ValueError("Provide either token or token_file, not both.")
    resolved = token
    if token_file:
        resolved = Path(token_file).read_text(encoding="utf-8").strip()
    if not resolved:
        return None
    return resolved if resolved.lower().startswith("bearer ") else f"Bearer {resolved}"


def probe_json_health(url: str, timeout: float = 3.0) -> dict[str, Any]:
    """Probe a JSON health endpoint without mutating the target service."""

    try:
        validated_url = _validate_base_url(url)
        request = Request(
            url=validated_url,
            method="GET",
            headers={"Accept": "application/json"},
        )
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        healthy = (
            isinstance(payload, dict)
            and str(payload.get("status", "")).lower() == "healthy"
        )
        return {
            "status": "available" if healthy else "unexpected_response",
            "url": validated_url,
            "http_status": response.status,
            "response": payload,
        }
    except Exception as exc:
        return _request_failure(exc, url)


DEFAULT_DOCKER_HOST_ALIAS = "host.docker.internal"
_DEFAULT_TASK_SERVER_PORTS = (8000, 8002, 8003)


def read_docker_bridge_host_ip() -> str | None:
    """Return the docker0 bridge IPv4 address when native Linux Docker is present."""

    if not sys.platform.startswith("linux") or not shutil.which("ip"):
        return None
    try:
        proc = subprocess.run(
            ["ip", "-4", "addr", "show", "docker0"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", proc.stdout)
    return match.group(1) if match else None


def resolve_docker_host_alias(explicit: str | None = None) -> str:
    """Resolve the host alias Nexent containers should use to reach task APIs."""

    if explicit:
        normalized = explicit.strip()
        if normalized.lower() in {"auto", "gateway", "bridge"}:
            return read_docker_bridge_host_ip() or DEFAULT_DOCKER_HOST_ALIAS
        return normalized
    return DEFAULT_DOCKER_HOST_ALIAS


def build_task_server_urls(
    *,
    docker_host: str | None = None,
    ports: tuple[int, int, int] = _DEFAULT_TASK_SERVER_PORTS,
) -> tuple[str, str, str]:
    host = resolve_docker_host_alias(docker_host)
    return tuple(f"http://{host}:{port}" for port in ports)


def describe_docker_host_integration_notes(
    *,
    docker_host: str | None = None,
    task1_server_url: str,
    task2_server_url: str,
    task3_server_url: str,
) -> dict[str, Any]:
    """Summarize how Nexent containers should reach host-side task APIs."""

    resolved_host = resolve_docker_host_alias(docker_host)
    notes: dict[str, Any] = {
        "platform": sys.platform,
        "resolved_docker_host": resolved_host,
        "task_server_urls": [
            task1_server_url,
            task2_server_url,
            task3_server_url,
        ],
        "default_alias": DEFAULT_DOCKER_HOST_ALIAS,
        "docker_bridge_host_ip": read_docker_bridge_host_ip(),
    }
    if sys.platform.startswith("linux"):
        notes["linux_guidance"] = (
            "Native Linux Docker does not expose host.docker.internal unless Nexent "
            "containers are started with --add-host=host.docker.internal:host-gateway. "
            "Prefer --docker-host auto to register docker0 bridge IPs, or pass explicit "
            "--task*-server-url values."
        )
        notes["nexent_compose_extra_hosts_example"] = (
            "extra_hosts:\n  - \"host.docker.internal:host-gateway\""
        )
    return notes


def build_dkm_openapi_services(
    *,
    task1_server_url: str = "http://host.docker.internal:8000",
    task2_server_url: str = "http://host.docker.internal:8002",
    task3_server_url: str = "http://host.docker.internal:8003",
) -> list[dict[str, Any]]:
    """Build import payload inputs from the three local FastAPI applications."""

    from src.pipelines.task1_api_server import app as task1_app
    from src.pipelines.task2_api_server import app as task2_app
    from src.pipelines.task3_api_server import app as task3_app

    return [
        {
            "service_name": "nexent_dkm_task1",
            "server_url": _validate_base_url(task1_server_url),
            "openapi_json": task1_app.openapi(),
            "service_description": "DKM task 1 data processing API",
        },
        {
            "service_name": "nexent_dkm_task2",
            "server_url": _validate_base_url(task2_server_url),
            "openapi_json": task2_app.openapi(),
            "service_description": "DKM task 2 medical knowledge graph API",
        },
        {
            "service_name": "nexent_dkm_task3",
            "server_url": _validate_base_url(task3_server_url),
            "openapi_json": task3_app.openapi(),
            "service_description": "DKM task 3 graph analysis API",
        },
    ]


def select_dkm_tool_ids(
    tools: list[dict[str, Any]],
    services: list[dict[str, Any]],
) -> list[int]:
    """Select Nexent MCP tools generated from the three DKM OpenAPI specs."""

    operation_ids = _openapi_operation_ids(services)
    service_names = {
        str(service.get("service_name"))
        for service in services
        if service.get("service_name")
    }
    selected: list[int] = []
    for tool in tools:
        if tool.get("source") != "mcp" or tool.get("usage") != "outer-apis":
            continue
        mounted_name = str(tool.get("name") or "")
        if not any(service_name in mounted_name for service_name in service_names):
            continue
        origin_name = str(tool.get("origin_name") or "")
        if not any(
            operation_id in mounted_name or operation_id == origin_name
            for operation_id in operation_ids
        ):
            continue
        tool_id = tool.get("tool_id")
        if isinstance(tool_id, int) and tool_id not in selected:
            selected.append(tool_id)
    return selected


def build_dkm_agent_payload(
    *,
    tools: list[dict[str, Any]],
    services: list[dict[str, Any]],
    model_id: int | None = None,
    model_name: str | None = None,
    agent_name: str = "dkm_end_to_end_agent",
) -> dict[str, Any]:
    """Build the official Nexent ``/agent/update`` creation payload."""

    if not agent_name.isidentifier():
        raise ValueError("Nexent agent name must be a valid Python identifier.")
    tool_ids = select_dkm_tool_ids(tools, services)
    if not tool_ids:
        raise ValueError(
            "No DKM OpenAPI tools were discovered. Import services and refresh "
            "the Nexent tool catalog first."
        )
    if model_id is None and not model_name:
        raise ValueError("model_id or model_name is required to create a runnable agent.")
    return {
        "name": agent_name,
        "display_name": "DKM Data-Knowledge-Insight Agent",
        "description": (
            "Nexent agent for data processing, medical knowledge graph "
            "construction, and graph-driven analysis."
        ),
        "business_description": (
            "Orchestrates task 1 data processing, task 2 medical KG generation, "
            "and task 3 graph analysis through imported OpenAPI tools."
        ),
        "model_id": model_id,
        "model_name": model_name,
        "max_steps": 12,
        "provide_run_summary": True,
        "duty_prompt": (
            "Plan the data-knowledge-insight workflow. Submit task 1 first when "
            "raw data needs cleaning, use task 2 to build and query the medical "
            "knowledge graph, then use task 3 for NL2SQL, graph analytics, and "
            "visualization. Poll each status endpoint and read its report before "
            "passing artifacts to the next stage."
        ),
        "constraint_prompt": (
            "Keep DataMate in dry_run unless the user explicitly requests an "
            "online submission. Do not invent task IDs, graph paths, metrics, "
            "or external-service results."
        ),
        "few_shots_prompt": "",
        "enabled": True,
        "enabled_tool_ids": tool_ids,
        "related_agent_ids": [],
    }


def _validate_base_url(base_url: str) -> str:
    parsed = urlparse((base_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute http(s) URL.")
    if parsed.username or parsed.password:
        raise ValueError("base_url must not include credentials.")
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"


def _service_name(service: dict[str, Any]) -> str | None:
    for field in ("mcp_service_name", "service_name", "name"):
        value = service.get(field)
        if value:
            return str(value)
    return None


def _openapi_operation_ids(services: list[dict[str, Any]]) -> set[str]:
    operation_ids: set[str] = set()
    for service in services:
        paths = service.get("openapi_json", {}).get("paths", {})
        if not isinstance(paths, dict):
            continue
        for path_item in paths.values():
            if not isinstance(path_item, dict):
                continue
            for operation in path_item.values():
                if isinstance(operation, dict) and operation.get("operationId"):
                    operation_ids.add(str(operation["operationId"]))
    return operation_ids


def _openapi_service_matches(
    existing: dict[str, Any],
    requested: dict[str, Any],
) -> bool:
    existing_url = str(existing.get("server_url") or "").rstrip("/")
    requested_url = str(requested.get("server_url") or "").rstrip("/")
    existing_spec = existing.get("openapi_json")
    requested_spec = requested.get("openapi_json")
    if not isinstance(existing_spec, dict) or not isinstance(requested_spec, dict):
        return False
    return (
        existing_url == requested_url
        and _openapi_contract(existing_spec) == _openapi_contract(requested_spec)
    )


def _openapi_contract(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the API surface that determines imported tool compatibility."""

    components = spec.get("components")
    schemas = components.get("schemas", {}) if isinstance(components, dict) else {}
    return {
        "paths": spec.get("paths", {}),
        "schemas": schemas,
    }


def _request_failure(exc: Exception, base_url: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "unavailable",
        "base_url": base_url,
        "message": str(exc),
    }
    if isinstance(exc, HTTPError):
        result["http_status"] = exc.code
        try:
            result["body"] = exc.read().decode("utf-8", errors="replace")
        except Exception:
            result["body"] = ""
    elif isinstance(exc, (URLError, TimeoutError, OSError, ValueError)):
        result["error_type"] = type(exc).__name__
    return result
