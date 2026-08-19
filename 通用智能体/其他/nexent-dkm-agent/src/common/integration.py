"""Shared Nexent/DataMate integration probes and graph planning helpers."""

from __future__ import annotations

from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.operators.data_ops.datamate_client import DataMateClient, safe_datamate_call

_NON_NEXENT_MARKERS = ("jupyter", "tornado")


def probe_datamate(base_url: str, timeout: float = 3.0) -> dict[str, Any]:
    """Probe DataMate health and return a small operator sample."""

    if base_url.lower() == "none":
        return {"status": "skipped", "reason": "datamate-url=none"}
    try:
        client = DataMateClient(base_url, timeout=timeout)
    except ValueError as exc:
        return {"status": "invalid_config", "base_url": base_url, "message": str(exc)}

    health = safe_datamate_call(client.health)
    operators = safe_datamate_call(lambda: client.list_operators(size=5))
    templates = safe_datamate_call(
        lambda: client.list_cleaning_templates(page=0, size=3)
    )
    tasks = safe_datamate_call(lambda: client.list_cleaning_tasks(page=0, size=3))
    core_probes = (operators, templates, tasks)
    successful_core_probes = sum(
        _is_successful_datamate_response(result) for result in core_probes
    )
    health_is_healthy = health.get("status") == "healthy"
    if successful_core_probes == len(core_probes):
        status = "available"
        readiness_basis = (
            "health_and_core_api_probes"
            if health_is_healthy
            else "core_api_probes"
        )
    elif successful_core_probes:
        status = "partial"
        readiness_basis = "partial_core_api_probes"
    elif health_is_healthy:
        status = "partial"
        readiness_basis = "health_endpoint_only"
    else:
        status = "unavailable"
        readiness_basis = "no_successful_probe"
    return {
        "status": status,
        "readiness_basis": readiness_basis,
        "successful_core_probes": successful_core_probes,
        "core_probe_count": len(core_probes),
        "base_url": client.base_url,
        "health": health,
        "operator_sample": operators,
        "template_sample": templates,
        "task_sample": tasks,
        "write_mode": "probe_only_no_submit",
    }


def probe_nexent(base_url: str, timeout: float = 3.0) -> dict[str, Any]:
    """Probe a Nexent web/API endpoint without mutating remote state.

    Nexent v2.x ships a Next.js front-end whose HTML body does not contain
    the literal word 'nexent'. When the initial HTML probe is inconclusive,
    we issue a follow-up GET to ``/api/tool/openapi_services`` — a
    Nexent-specific JSON endpoint — to confirm the service identity.
    """

    if base_url.lower() == "none":
        return {"status": "skipped", "reason": "nexent-url=none"}
    try:
        request = Request(url=base_url, method="GET", headers={"Accept": "text/html,application/json"})
        with urlopen(request, timeout=timeout) as response:
            body_preview = response.read(1024).decode("utf-8", errors="replace")
            server_header = response.headers.get("Server", "")
            x_powered_by = response.headers.get("X-Powered-By", "")
            http_status = response.status
        non_nexent = _identify_non_nexent_service(server_header, body_preview)
        if non_nexent:
            return {
                "status": "not_nexent",
                "url": base_url,
                "http_status": http_status,
                "server_header": server_header,
                "detected_service": non_nexent,
                "body_preview": body_preview[:120],
            }
        detected_service = _identify_nexent_service(server_header, body_preview)
        if not detected_service:
            # Follow-up: hit a Nexent-specific JSON API endpoint to confirm.
            api_result = _probe_nexent_api_endpoint(base_url, timeout=timeout)
            if api_result.get("is_nexent"):
                return {
                    "status": "available",
                    "url": base_url,
                    "http_status": http_status,
                    "server_header": server_header,
                    "x_powered_by": x_powered_by,
                    "detected_service": "nexent",
                    "body_preview": body_preview[:120],
                    "api_confirmation": api_result,
                }
        return {
            "status": "available" if detected_service else "unknown_http_service",
            "url": base_url,
            "http_status": http_status,
            "server_header": server_header,
            "x_powered_by": x_powered_by,
            "detected_service": detected_service or "unknown_http_service",
            "body_preview": body_preview[:120],
        }
    except HTTPError as exc:
        body_preview = exc.read(1024).decode("utf-8", errors="replace")
        server_header = (exc.headers or {}).get("Server", "")
        non_nexent = _identify_non_nexent_service(server_header, body_preview)
        if non_nexent:
            return {
                "status": "not_nexent",
                "url": base_url,
                "http_status": exc.code,
                "server_header": server_header,
                "detected_service": non_nexent,
                "body_preview": body_preview[:120],
            }
        detected_service = _identify_nexent_service(server_header, body_preview)
        return {
            "status": (
                "available"
                if exc.code < 500 and detected_service
                else "unknown_http_service"
                if exc.code < 500
                else "unavailable"
            ),
            "url": base_url,
            "http_status": exc.code,
            "server_header": server_header,
            "detected_service": detected_service or "unknown_http_service",
            "message": str(exc),
        }
    except (URLError, TimeoutError, ValueError) as exc:
        return {"status": "unavailable", "url": base_url, "message": str(exc)}


def _probe_nexent_api_endpoint(base_url: str, timeout: float = 3.0) -> dict[str, Any]:
    """Confirm Nexent identity by hitting a Nexent-specific JSON API.

    The ``/api/tool/openapi_services`` endpoint returns
    ``{"message": "success", "data": [...]}`` on a real Nexent deployment.
    We treat a 200 JSON response with that envelope shape as confirmation.
    """

    api_path = "/api/tool/openapi_services"
    try:
        request = Request(
            url=f"{base_url.rstrip('/')}{api_path}",
            method="GET",
            headers={"Accept": "application/json"},
        )
        with urlopen(request, timeout=timeout) as response:
            payload = response.read(4096).decode("utf-8", errors="replace")
        is_nexent = (
            '"message"' in payload
            and '"success"' in payload
            and ('"data"' in payload or "[]" in payload)
        )
        return {
            "endpoint": api_path,
            "http_status": response.status,
            "is_nexent": is_nexent,
            "response_preview": payload[:200],
        }
    except Exception:
        return {"endpoint": api_path, "is_nexent": False}


def build_integration_report(
    datamate_url: str = "http://localhost:18000",
    nexent_url: str = "http://localhost:3000",
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Return a combined integration readiness report for demos and evidence."""

    datamate = probe_datamate(datamate_url, timeout=timeout)
    nexent = probe_nexent(nexent_url, timeout=timeout)
    statuses = {datamate.get("status"), nexent.get("status")}
    if statuses <= {"skipped"}:
        stack_status = "offline"
    elif statuses == {"available"}:
        stack_status = "ready"
    elif "available" in statuses or "partial" in statuses:
        stack_status = "partial"
    else:
        stack_status = "unavailable"
    return {
        "stack_status": stack_status,
        "datamate": datamate,
        "nexent": nexent,
    }


def summarize_graph_for_planning(graph: dict[str, Any]) -> dict[str, Any]:
    """Summarize a task-2 graph artifact for analysis planning."""

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    type_counts: dict[str, int] = {}
    for node in nodes:
        node_type = str(node.get("type", "Unknown"))
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    disease_count = type_counts.get("Disease", 0)
    symptom_count = type_counts.get("Symptom", 0)
    relation_counts: dict[str, int] = {}
    for edge in edges:
        relation = str(edge.get("predicate") or edge.get("type") or "RELATED")
        relation_counts[relation] = relation_counts.get(relation, 0) + 1

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "type_counts": type_counts,
        "disease_count": disease_count,
        "symptom_count": symptom_count,
        "relation_counts": relation_counts,
        "is_large_graph": len(nodes) >= 20 or len(edges) >= 25,
        "has_rich_disease_links": disease_count >= 3 and len(edges) >= 5,
    }


def _identify_non_nexent_service(server_header: str, body: str) -> str | None:
    fingerprint = f"{server_header} {body}".lower()
    if any(marker in fingerprint for marker in _NON_NEXENT_MARKERS):
        return "Jupyter/Tornado"
    return None


def _identify_nexent_service(server_header: str, body: str) -> str | None:
    fingerprint = f"{server_header} {body}".lower()
    return "nexent" if "nexent" in fingerprint else None


def _is_successful_datamate_response(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status", "")).lower()
    if status in {"unavailable", "invalid_config", "error", "failed"}:
        return False
    if "code" in payload:
        return str(payload.get("code")) in {"0", "200"}
    return status in {"available", "healthy", "ok", "success"}
