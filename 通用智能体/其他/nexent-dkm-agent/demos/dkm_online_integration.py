"""Probe or explicitly register the three DKM APIs in a live Nexent service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.integration import build_integration_report
from src.common.nexent_online import (
    NexentOnlineClient,
    build_dkm_agent_payload,
    build_dkm_openapi_services,
    build_task_server_urls,
    describe_docker_host_integration_notes,
    load_authorization,
    probe_json_health,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely probe or register DKM OpenAPI services in Nexent."
    )
    parser.add_argument(
        "--mode",
        choices=["probe", "prepare", "submit"],
        default="probe",
    )
    parser.add_argument("--nexent-url", default="http://localhost:3000")
    parser.add_argument("--datamate-url", default="http://localhost:18000")
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--allow-write", action="store_true")
    parser.add_argument("--force-update", action="store_true")
    parser.add_argument("--create-agent", action="store_true")
    parser.add_argument("--agent-name", default="dkm_end_to_end_agent")
    parser.add_argument("--model-id", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--docker-host",
        default=None,
        help=(
            "Host alias/IP Nexent containers use to reach task APIs. "
            "Use 'auto' on native Linux Docker to pick the docker0 bridge IP; "
            "omit to keep host.docker.internal (Docker Desktop / WSL)."
        ),
    )
    parser.add_argument(
        "--task1-server-url",
        default="http://host.docker.internal:8000",
    )
    parser.add_argument(
        "--task2-server-url",
        default="http://host.docker.internal:8002",
    )
    parser.add_argument(
        "--task3-server-url",
        default="http://host.docker.internal:8003",
    )
    parser.add_argument(
        "--task1-health-url",
        default="http://localhost:8000/health",
    )
    parser.add_argument(
        "--task2-health-url",
        default="http://localhost:8002/health",
    )
    parser.add_argument(
        "--task3-health-url",
        default="http://localhost:8003/health",
    )
    parser.add_argument(
        "--allow-unhealthy-task-apis",
        action="store_true",
        help="Allow submit even when host-side task API health probes fail.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


_DEFAULT_TASK_SERVER_URLS = (
    "http://host.docker.internal:8000",
    "http://host.docker.internal:8002",
    "http://host.docker.internal:8003",
)


def _apply_docker_host_args(args: argparse.Namespace) -> None:
    if not args.docker_host:
        return
    auto_urls = build_task_server_urls(docker_host=args.docker_host)
    for attr, default, auto_url in zip(
        ("task1_server_url", "task2_server_url", "task3_server_url"),
        _DEFAULT_TASK_SERVER_URLS,
        auto_urls,
    ):
        if getattr(args, attr) == default:
            setattr(args, attr, auto_url)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _apply_docker_host_args(args)
    nexent_skipped = args.nexent_url.strip().lower() == "none"
    client = None
    if not nexent_skipped:
        authorization = load_authorization(token_file=args.token_file)
        client = NexentOnlineClient(
            args.nexent_url,
            authorization=authorization,
            timeout=args.timeout,
        )
    services = build_dkm_openapi_services(
        task1_server_url=args.task1_server_url,
        task2_server_url=args.task2_server_url,
        task3_server_url=args.task3_server_url,
    )
    health_urls = {
        "task1": args.task1_health_url,
        "task2": args.task2_health_url,
        "task3": args.task3_health_url,
    }
    task_api_health = {
        task: probe_json_health(url, timeout=args.timeout)
        for task, url in health_urls.items()
    }
    report: dict[str, Any] = {
        "mode": args.mode,
        "write_allowed": bool(args.allow_write),
        "integration": build_integration_report(
            datamate_url=args.datamate_url,
            nexent_url=args.nexent_url,
            timeout=args.timeout,
        ),
        "task_api_health": task_api_health,
        "nexent_openapi_services": (
            client.list_openapi_services()
            if client is not None
            else {"status": "skipped", "reason": "nexent-url=none"}
        ),
        "service_specs": [_service_summary(service) for service in services],
        "docker_host_reachability": describe_docker_host_integration_notes(
            docker_host=args.docker_host,
            task1_server_url=args.task1_server_url,
            task2_server_url=args.task2_server_url,
            task3_server_url=args.task3_server_url,
        ),
    }
    _reconcile_nexent_status_from_openapi(report)
    if args.mode == "probe":
        report["status"] = _probe_status(report)
        return report

    if client is None:
        report["status"] = "blocked_nexent_skipped"
        report["message"] = (
            "Prepare and submit modes require a real --nexent-url. "
            "Use --mode probe when Nexent is intentionally skipped."
        )
        return report

    if args.mode == "prepare":
        report["imports"] = [
            _compact_import_result(
                client.import_openapi_service(
                    **service,
                    mode="dry_run",
                    allow_write=False,
                )
            )
            for service in services
        ]
        report["status"] = "prepared"
        return report

    unhealthy = [
        task
        for task, health in task_api_health.items()
        if health.get("status") != "available"
    ]
    if unhealthy and not args.allow_unhealthy_task_apis:
        report["status"] = "blocked_unhealthy_task_apis"
        report["blocked_tasks"] = unhealthy
        report["message"] = (
            "Start all three task APIs before submit, or explicitly pass "
            "--allow-unhealthy-task-apis."
        )
        return report

    import_results = [
        client.import_openapi_service(
            **service,
            force_update=args.force_update,
            mode="submit",
            allow_write=args.allow_write,
        )
        for service in services
    ]
    report["imports"] = [_compact_import_result(item) for item in import_results]
    if not all(item.get("status") == "verified" for item in import_results):
        report["status"] = "openapi_import_incomplete"
        return report

    report["tool_refresh"] = client.refresh_tool_catalog(
        allow_write=args.allow_write
    )
    tools = client.list_tools()
    report["tool_catalog"] = {
        "status": tools.get("status"),
        "tool_count": tools.get("tool_count", 0),
    }
    if args.create_agent:
        try:
            payload = build_dkm_agent_payload(
                tools=tools.get("tools", []),
                services=services,
                model_id=args.model_id,
                model_name=args.model_name,
                agent_name=args.agent_name,
            )
        except ValueError as exc:
            if "model_id or model_name" in str(exc):
                report["agent"] = _verify_or_report_missing_agent_model(
                    client,
                    args.agent_name,
                    str(exc),
                )
            else:
                report["agent"] = {
                    "status": "not_created",
                    "message": str(exc),
                }
        else:
            report["agent"] = client.create_agent(
                payload,
                mode="submit",
                allow_write=args.allow_write,
            )

    report["status"] = _submit_status(report, create_agent=args.create_agent)
    return report


def main() -> int:
    args = parse_args()
    report = run(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] in {"available", "partial", "prepared", "verified"} else 1


def _service_summary(service: dict[str, Any]) -> dict[str, Any]:
    paths = service["openapi_json"].get("paths", {})
    return {
        "service_name": service["service_name"],
        "server_url": service["server_url"],
        "path_count": len(paths),
        "operation_count": sum(
            1
            for path_item in paths.values()
            if isinstance(path_item, dict)
            for operation in path_item.values()
            if isinstance(operation, dict) and operation.get("operationId")
        ),
    }


def _compact_import_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = {key: value for key, value in result.items() if key != "payload"}
    payload = result.get("payload")
    if isinstance(payload, dict):
        compact["payload_summary"] = {
            "service_name": payload.get("service_name"),
            "server_url": payload.get("server_url"),
            "force_update": payload.get("force_update"),
        }
    return compact


def _verify_or_report_missing_agent_model(
    client: NexentOnlineClient,
    agent_name: str,
    message: str,
) -> dict[str, Any]:
    existing_agent = client.verify_existing_agent(agent_name)
    if existing_agent.get("status") == "verified":
        existing_agent["message"] = (
            "Existing Nexent agent verified without requiring model_id/model_name."
        )
        return existing_agent
    return {
        "status": "not_created",
        "message": message,
        "existing_agent": existing_agent,
    }


def _reconcile_nexent_status_from_openapi(report: dict[str, Any]) -> None:
    """Use authenticated OpenAPI metadata to confirm Nexent in full deployments."""

    services = report.get("nexent_openapi_services", {})
    if services.get("status") != "available":
        return

    integration = report.get("integration", {})
    nexent = integration.get("nexent", {})
    if nexent.get("status") not in {"unavailable", "unknown_http_service"}:
        return

    nexent.update(
        {
            "status": "available",
            "detected_service": "nexent",
            "api_confirmation": {
                "endpoint": "/api/tool/openapi_services",
                "is_nexent": True,
                "service_count": services.get("service_count", 0),
                "service_names": services.get("service_names", []),
            },
        }
    )
    if "message" in nexent:
        nexent["message"] = (
            f"{nexent['message']} API endpoint confirmed Nexent service."
        )

    statuses = {
        integration.get("datamate", {}).get("status"),
        nexent.get("status"),
    }
    if statuses == {"available"}:
        integration["stack_status"] = "ready"
    elif "available" in statuses or "partial" in statuses:
        integration["stack_status"] = "partial"


def _probe_status(report: dict[str, Any]) -> str:
    statuses = {
        report["integration"].get("stack_status"),
        report["nexent_openapi_services"].get("status"),
    }
    if statuses <= {"ready", "available"}:
        return "available"
    if "available" in statuses or "partial" in statuses:
        return "partial"
    return "unavailable"


def _submit_status(report: dict[str, Any], *, create_agent: bool) -> str:
    if report.get("tool_refresh", {}).get("status") != "refreshed":
        return "tool_refresh_incomplete"
    if report.get("tool_catalog", {}).get("status") != "available":
        return "tool_catalog_unavailable"
    if create_agent and report.get("agent", {}).get("status") != "verified":
        return "agent_creation_incomplete"
    return "verified"


if __name__ == "__main__":
    raise SystemExit(main())
