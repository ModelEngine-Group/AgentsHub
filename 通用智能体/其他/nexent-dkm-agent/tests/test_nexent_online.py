from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from demos import dkm_online_integration
from src.common.nexent_online import (
    NexentOnlineClient,
    build_dkm_agent_payload,
    build_dkm_openapi_services,
    build_task_server_urls,
    describe_docker_host_integration_notes,
    load_authorization,
    read_docker_bridge_host_ip,
    resolve_docker_host_alias,
    select_dkm_tool_ids,
)


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_nexent_online_client_rejects_unsafe_url():
    with pytest.raises(ValueError):
        NexentOnlineClient("file:///tmp/nexent")
    with pytest.raises(ValueError):
        NexentOnlineClient("http://user:pass@localhost:3000")


def test_dkm_probe_allows_nexent_to_be_explicitly_skipped(monkeypatch):
    monkeypatch.setattr(
        dkm_online_integration,
        "NexentOnlineClient",
        lambda *args, **kwargs: pytest.fail(
            "offline probe must not construct a Nexent client"
        ),
    )
    monkeypatch.setattr(
        dkm_online_integration,
        "build_integration_report",
        lambda **kwargs: {
            "stack_status": "partial",
            "datamate": {"status": "available"},
            "nexent": {"status": "skipped"},
        },
    )
    monkeypatch.setattr(
        dkm_online_integration,
        "probe_json_health",
        lambda *args, **kwargs: {"status": "unavailable"},
    )
    args = SimpleNamespace(
        mode="probe",
        nexent_url="none",
        datamate_url="http://localhost:8080",
        token_file=None,
        timeout=1.0,
        allow_write=False,
        force_update=False,
        create_agent=False,
        agent_name="dkm_end_to_end_agent",
        model_id=None,
        model_name=None,
        task1_server_url="http://host.docker.internal:8000",
        task2_server_url="http://host.docker.internal:8002",
        task3_server_url="http://host.docker.internal:8003",
        task1_health_url="http://localhost:8000/health",
        task2_health_url="http://localhost:8002/health",
        task3_health_url="http://localhost:8003/health",
        allow_unhealthy_task_apis=False,
        docker_host=None,
    )

    report = dkm_online_integration.run(args)

    assert report["status"] == "partial"
    assert report["nexent_openapi_services"]["status"] == "skipped"


def test_dkm_probe_confirms_full_auth_nexent_from_openapi(monkeypatch):
    class FakeNexentClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def list_openapi_services(self) -> dict:
            return {
                "status": "available",
                "service_count": 3,
                "service_names": [
                    "nexent_dkm_task1",
                    "nexent_dkm_task2",
                    "nexent_dkm_task3",
                ],
            }

    monkeypatch.setattr(
        dkm_online_integration,
        "NexentOnlineClient",
        FakeNexentClient,
    )
    monkeypatch.setattr(
        dkm_online_integration,
        "build_integration_report",
        lambda **kwargs: {
            "stack_status": "unavailable",
            "datamate": {"status": "skipped"},
            "nexent": {
                "status": "unknown_http_service",
                "message": "HTTP service responded, but Nexent was not detected.",
            },
        },
    )
    monkeypatch.setattr(
        dkm_online_integration,
        "probe_json_health",
        lambda *args, **kwargs: {"status": "available"},
    )
    args = SimpleNamespace(
        mode="probe",
        nexent_url="http://localhost:3000",
        datamate_url="none",
        token_file=None,
        timeout=1.0,
        allow_write=False,
        force_update=False,
        create_agent=False,
        agent_name="dkm_end_to_end_agent",
        model_id=None,
        model_name=None,
        task1_server_url="http://host.docker.internal:8000",
        task2_server_url="http://host.docker.internal:8002",
        task3_server_url="http://host.docker.internal:8003",
        task1_health_url="http://localhost:8000/health",
        task2_health_url="http://localhost:8002/health",
        task3_health_url="http://localhost:8003/health",
        allow_unhealthy_task_apis=False,
        docker_host=None,
    )

    report = dkm_online_integration.run(args)

    assert report["status"] == "partial"
    assert report["integration"]["stack_status"] == "partial"
    assert report["integration"]["nexent"]["status"] == "available"
    assert report["integration"]["nexent"]["api_confirmation"] == {
        "endpoint": "/api/tool/openapi_services",
        "is_nexent": True,
        "service_count": 3,
        "service_names": [
            "nexent_dkm_task1",
            "nexent_dkm_task2",
            "nexent_dkm_task3",
        ],
    }


def test_load_authorization_reads_token_file_without_exposing_path(tmp_path):
    token_file = tmp_path / "nexent.token"
    token_file.write_text("test-access-token\n", encoding="utf-8")

    authorization = load_authorization(token_file=token_file)

    assert authorization == "Bearer test-access-token"


def test_nexent_online_client_lists_openapi_services(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "message": "success",
                "data": [{"mcp_service_name": "dkm_task1"}],
            }
        )

    monkeypatch.setattr("src.common.nexent_online.urlopen", fake_urlopen)
    client = NexentOnlineClient(
        "http://localhost:3000",
        authorization="Bearer secret",
        timeout=4.5,
    )

    result = client.list_openapi_services()

    assert captured == {
        "url": "http://localhost:3000/api/tool/openapi_services",
        "authorization": "Bearer secret",
        "timeout": 4.5,
    }
    assert result["status"] == "available"
    assert result["service_names"] == ["dkm_task1"]


def test_nexent_openapi_import_is_dry_run_by_default(monkeypatch):
    monkeypatch.setattr(
        "src.common.nexent_online.urlopen",
        lambda *args, **kwargs: pytest.fail("dry-run must not perform HTTP requests"),
    )
    client = NexentOnlineClient("http://localhost:3000")

    result = client.import_openapi_service(
        service_name="dkm_task1",
        server_url="http://host.docker.internal:8000",
        openapi_json={"openapi": "3.1.0", "paths": {"/health": {"get": {}}}},
    )

    assert result["status"] == "prepared"
    assert result["submitted"] is False
    assert result["payload"]["service_name"] == "dkm_task1"


def test_nexent_openapi_import_requires_explicit_write_permission():
    client = NexentOnlineClient("http://localhost:3000")

    result = client.import_openapi_service(
        service_name="dkm_task1",
        server_url="http://host.docker.internal:8000",
        openapi_json={"openapi": "3.1.0", "paths": {}},
        mode="submit",
        allow_write=False,
    )

    assert result["status"] == "write_blocked"
    assert result["submitted"] is False


def test_nexent_openapi_import_verifies_registered_service(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request.method, request.full_url))
        if request.method == "POST":
            payload = json.loads(request.data.decode("utf-8"))
            assert payload["service_name"] == "dkm_task1"
            assert payload["server_url"] == "http://host.docker.internal:8000"
            return FakeResponse(
                {
                    "status": "success",
                    "data": {"mcp_refresh": {"success": True}},
                }
            )
        if len(requests) == 1:
            return FakeResponse({"message": "success", "data": []})
        return FakeResponse(
            {
                "message": "success",
                "data": [{"mcp_service_name": "dkm_task1"}],
            }
        )

    monkeypatch.setattr("src.common.nexent_online.urlopen", fake_urlopen)
    client = NexentOnlineClient("http://localhost:3000")

    result = client.import_openapi_service(
        service_name="dkm_task1",
        server_url="http://host.docker.internal:8000",
        openapi_json={"openapi": "3.1.0", "paths": {}},
        mode="submit",
        allow_write=True,
    )

    assert requests == [
        ("GET", "http://localhost:3000/api/tool/openapi_services"),
        ("POST", "http://localhost:3000/api/tool/openapi_service"),
        ("GET", "http://localhost:3000/api/tool/openapi_services"),
    ]
    assert result["status"] == "verified"
    assert result["submitted"] is True
    assert result["verified"] is True


def test_nexent_openapi_import_skips_matching_existing_service(monkeypatch):
    requests = []
    openapi_json = {
        "openapi": "3.1.0",
        "paths": {
            "/health": {
                "get": {"operationId": "health_check_health_get"},
            }
        },
    }

    def fake_urlopen(request, timeout):
        requests.append((request.method, request.full_url))
        return FakeResponse(
            {
                "message": "success",
                "data": [
                    {
                        "mcp_service_name": "dkm_task1",
                        "server_url": "http://host.docker.internal:8000",
                        "openapi_json": openapi_json,
                    }
                ],
            }
        )

    monkeypatch.setattr("src.common.nexent_online.urlopen", fake_urlopen)
    client = NexentOnlineClient("http://localhost:3000")

    result = client.import_openapi_service(
        service_name="dkm_task1",
        server_url="http://host.docker.internal:8000",
        openapi_json=openapi_json,
        mode="submit",
        allow_write=True,
    )

    assert requests == [
        ("GET", "http://localhost:3000/api/tool/openapi_services"),
    ]
    assert result["status"] == "verified"
    assert result["submitted"] is False
    assert result["preexisting"] is True


def test_nexent_openapi_import_detects_contract_change_with_same_operation_id(
    monkeypatch,
):
    requests = []
    existing_spec = {
        "openapi": "3.1.0",
        "paths": {
            "/api/task1/process": {
                "post": {
                    "operationId": "submit_task",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/OldRequest"}
                            }
                        }
                    },
                }
            }
        },
        "components": {"schemas": {"OldRequest": {"type": "object"}}},
    }
    requested_spec = {
        "openapi": "3.1.0",
        "paths": {
            "/api/task1/process": {
                "post": {
                    "operationId": "submit_task",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/NewRequest"}
                            }
                        }
                    },
                }
            }
        },
        "components": {"schemas": {"NewRequest": {"type": "object"}}},
    }

    def fake_urlopen(request, timeout):
        requests.append((request.method, request.full_url))
        return FakeResponse(
            {
                "message": "success",
                "data": [
                    {
                        "mcp_service_name": "dkm_task1",
                        "server_url": "http://host.docker.internal:8000",
                        "openapi_json": existing_spec,
                    }
                ],
            }
        )

    monkeypatch.setattr("src.common.nexent_online.urlopen", fake_urlopen)
    client = NexentOnlineClient("http://localhost:3000")

    result = client.import_openapi_service(
        service_name="dkm_task1",
        server_url="http://host.docker.internal:8000",
        openapi_json=requested_spec,
        mode="submit",
        allow_write=True,
    )

    assert requests == [
        ("GET", "http://localhost:3000/api/tool/openapi_services"),
    ]
    assert result["status"] == "update_blocked"
    assert result["requires_force_update"] is True


def test_nexent_openapi_import_blocks_conflicting_existing_service(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request.method, request.full_url))
        return FakeResponse(
            {
                "message": "success",
                "data": [
                    {
                        "mcp_service_name": "dkm_task1",
                        "server_url": "http://old-host:8000",
                        "openapi_json": {"openapi": "3.1.0", "paths": {}},
                    }
                ],
            }
        )

    monkeypatch.setattr("src.common.nexent_online.urlopen", fake_urlopen)
    client = NexentOnlineClient("http://localhost:3000")

    result = client.import_openapi_service(
        service_name="dkm_task1",
        server_url="http://host.docker.internal:8000",
        openapi_json={"openapi": "3.1.0", "paths": {"/health": {"get": {}}}},
        mode="submit",
        allow_write=True,
    )

    assert requests == [
        ("GET", "http://localhost:3000/api/tool/openapi_services"),
    ]
    assert result["status"] == "update_blocked"
    assert result["submitted"] is False
    assert result["requires_force_update"] is True


def test_build_dkm_openapi_services_exports_three_live_api_specs():
    services = build_dkm_openapi_services()

    assert [service["service_name"] for service in services] == [
        "nexent_dkm_task1",
        "nexent_dkm_task2",
        "nexent_dkm_task3",
    ]
    assert "/api/task1/process" in services[0]["openapi_json"]["paths"]
    assert "/api/task2/process" in services[1]["openapi_json"]["paths"]
    assert "/api/task3/process" in services[2]["openapi_json"]["paths"]


def test_nexent_refresh_tools_requires_explicit_write_permission(monkeypatch):
    monkeypatch.setattr(
        "src.common.nexent_online.urlopen",
        lambda *args, **kwargs: pytest.fail("write-blocked refresh must not call HTTP"),
    )
    client = NexentOnlineClient("http://localhost:3000")

    result = client.refresh_tool_catalog(allow_write=False)

    assert result["status"] == "write_blocked"
    assert result["submitted"] is False


def test_nexent_client_lists_tools(monkeypatch):
    monkeypatch.setattr(
        "src.common.nexent_online.urlopen",
        lambda *args, **kwargs: FakeResponse(
            [
                {
                    "tool_id": 41,
                    "name": "nexent_dkm_task1_submit_task_api_task1_process_post",
                    "source": "mcp",
                    "usage": "outer-apis",
                }
            ]
        ),
    )
    client = NexentOnlineClient("http://localhost:3000")

    result = client.list_tools()

    assert result["status"] == "available"
    assert result["tool_count"] == 1
    assert result["tools"][0]["tool_id"] == 41


def test_select_dkm_tool_ids_matches_mounted_operation_ids():
    services = build_dkm_openapi_services()
    tools = [
        {
            "tool_id": 41,
            "name": "nexent_dkm_task1_submit_task_api_task1_process_post",
            "origin_name": "submit_task_api_task1_process_post",
            "source": "mcp",
            "usage": "outer-apis",
        },
        {
            "tool_id": 42,
            "name": "other_service_submit_task_api_task1_process_post",
            "source": "mcp",
            "usage": "outer-apis",
        },
        {
            "tool_id": 43,
            "name": "submit_task_api_task1_process_post",
            "source": "local",
            "usage": None,
        },
    ]

    selected = select_dkm_tool_ids(tools, services)

    assert selected == [41]


def test_build_dkm_agent_payload_requires_discovered_tools():
    with pytest.raises(ValueError, match="No DKM OpenAPI tools"):
        build_dkm_agent_payload(
            tools=[],
            services=build_dkm_openapi_services(),
            model_id=7,
            model_name="main_model",
        )


def test_nexent_agent_create_verifies_by_name(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request.method, request.full_url))
        if request.full_url.endswith("/api/agent/list"):
            return FakeResponse([])
        if request.method == "POST":
            payload = json.loads(request.data.decode("utf-8"))
            assert payload["name"] == "dkm_end_to_end_agent"
            assert payload["enabled_tool_ids"] == [41, 42]
            return FakeResponse({"agent_id": 99})
        return FakeResponse(
            {
                "agent_id": 99,
                "latest_version_no": None,
            }
        )

    monkeypatch.setattr("src.common.nexent_online.urlopen", fake_urlopen)
    client = NexentOnlineClient("http://localhost:3000")

    result = client.create_agent(
        {
            "name": "dkm_end_to_end_agent",
            "display_name": "DKM Data-Knowledge-Insight Agent",
            "enabled_tool_ids": [41, 42],
        },
        mode="submit",
        allow_write=True,
    )

    assert requests == [
        ("GET", "http://localhost:3000/api/agent/list"),
        ("POST", "http://localhost:3000/api/agent/update"),
        ("GET", "http://localhost:3000/api/agent/by-name/dkm_end_to_end_agent"),
    ]
    assert result["status"] == "verified"
    assert result["agent_id"] == 99


def test_nexent_agent_create_verifies_matching_existing_agent(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request.method, request.full_url))
        if request.full_url.endswith("/api/agent/list"):
            return FakeResponse(
                [{"agent_id": 88, "name": "dkm_end_to_end_agent"}]
            )
        return FakeResponse(
            {
                "agent_id": 88,
                "name": "dkm_end_to_end_agent",
                "latest_version_no": 3,
            }
        )

    monkeypatch.setattr("src.common.nexent_online.urlopen", fake_urlopen)
    client = NexentOnlineClient("http://localhost:3000")

    result = client.create_agent(
        {
            "name": "dkm_end_to_end_agent",
            "enabled_tool_ids": [41],
        },
        mode="submit",
        allow_write=True,
    )

    assert requests == [
        ("GET", "http://localhost:3000/api/agent/list"),
        ("GET", "http://localhost:3000/api/agent/by-name/dkm_end_to_end_agent"),
    ]
    assert result["status"] == "verified"
    assert result["submitted"] is False
    assert result["verified"] is True
    assert result["preexisting"] is True
    assert result["agent_id"] == 88
    assert result["verification_scope"] == "identity_only"
    assert result["configuration_verified"] is False


def test_nexent_verify_existing_agent_is_read_only(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request.method, request.full_url))
        if request.full_url.endswith("/api/agent/list"):
            return FakeResponse(
                [{"agent_id": 88, "name": "dkm_end_to_end_agent"}]
            )
        return FakeResponse(
            {
                "agent_id": 88,
                "name": "dkm_end_to_end_agent",
                "latest_version_no": 3,
            }
        )

    monkeypatch.setattr("src.common.nexent_online.urlopen", fake_urlopen)
    client = NexentOnlineClient("http://localhost:3000")

    result = client.verify_existing_agent("dkm_end_to_end_agent")

    assert requests == [
        ("GET", "http://localhost:3000/api/agent/list"),
        ("GET", "http://localhost:3000/api/agent/by-name/dkm_end_to_end_agent"),
    ]
    assert result["status"] == "verified"
    assert result["submitted"] is False
    assert result["preexisting"] is True


def test_dkm_submit_verifies_existing_agent_without_model(monkeypatch):
    class FakeNexentClient:
        def __init__(self, *args, **kwargs) -> None:
            self.created = False

        def list_openapi_services(self) -> dict:
            return {
                "status": "available",
                "service_count": 3,
                "service_names": [
                    "nexent_dkm_task1",
                    "nexent_dkm_task2",
                    "nexent_dkm_task3",
                ],
            }

        def import_openapi_service(self, **kwargs) -> dict:
            return {"status": "verified", "service_name": kwargs["service_name"]}

        def refresh_tool_catalog(self, *, allow_write: bool) -> dict:
            return {"status": "refreshed", "submitted": allow_write}

        def list_tools(self) -> dict:
            return {
                "status": "available",
                "tool_count": 1,
                "tools": [
                    {
                        "tool_id": 41,
                        "name": (
                            "nexent_dkm_task1_"
                            "submit_task_api_task1_process_post"
                        ),
                        "origin_name": "submit_task_api_task1_process_post",
                        "source": "mcp",
                        "usage": "outer-apis",
                    }
                ],
            }

        def verify_existing_agent(self, name: str) -> dict:
            return {
                "status": "verified",
                "submitted": False,
                "verified": True,
                "preexisting": True,
                "agent_id": 88,
                "verification_scope": "identity_only",
            }

        def create_agent(self, *args, **kwargs) -> dict:
            pytest.fail("preexisting agent verification must not submit updates")

    monkeypatch.setattr(
        dkm_online_integration,
        "NexentOnlineClient",
        FakeNexentClient,
    )
    monkeypatch.setattr(
        dkm_online_integration,
        "build_integration_report",
        lambda **kwargs: {
            "stack_status": "partial",
            "datamate": {"status": "skipped"},
            "nexent": {"status": "available"},
        },
    )
    monkeypatch.setattr(
        dkm_online_integration,
        "probe_json_health",
        lambda *args, **kwargs: {"status": "available"},
    )
    args = SimpleNamespace(
        mode="submit",
        nexent_url="http://localhost:3000",
        datamate_url="none",
        token_file=None,
        timeout=1.0,
        allow_write=True,
        force_update=False,
        create_agent=True,
        agent_name="dkm_end_to_end_agent",
        model_id=None,
        model_name=None,
        task1_server_url="http://host.docker.internal:8000",
        task2_server_url="http://host.docker.internal:8002",
        task3_server_url="http://host.docker.internal:8003",
        task1_health_url="http://localhost:8000/health",
        task2_health_url="http://localhost:8002/health",
        task3_health_url="http://localhost:8003/health",
        allow_unhealthy_task_apis=False,
        docker_host=None,
    )

    report = dkm_online_integration.run(args)

    assert report["status"] == "verified"
    assert report["agent"]["status"] == "verified"
    assert report["agent"]["preexisting"] is True


def test_resolve_docker_host_alias_uses_explicit_value():
    assert resolve_docker_host_alias("10.0.0.8") == "10.0.0.8"


def test_resolve_docker_host_alias_auto_prefers_bridge_ip(monkeypatch):
    monkeypatch.setattr(
        "src.common.nexent_online.read_docker_bridge_host_ip",
        lambda: "172.17.0.1",
    )

    assert resolve_docker_host_alias("auto") == "172.17.0.1"


def test_build_task_server_urls_uses_resolved_host(monkeypatch):
    monkeypatch.setattr(
        "src.common.nexent_online.resolve_docker_host_alias",
        lambda explicit=None: "172.17.0.1",
    )

    assert build_task_server_urls(docker_host="auto") == (
        "http://172.17.0.1:8000",
        "http://172.17.0.1:8002",
        "http://172.17.0.1:8003",
    )


def test_read_docker_bridge_host_ip_parses_ip_addr_output(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = "3: docker0: <BROADCAST,MULTICAST,UP,LOWER_UP>\n    inet 172.17.0.1/16"

    monkeypatch.setattr("src.common.nexent_online.sys.platform", "linux")
    monkeypatch.setattr("src.common.nexent_online.shutil.which", lambda _: "/usr/sbin/ip")
    monkeypatch.setattr(
        "src.common.nexent_online.subprocess.run",
        lambda *args, **kwargs: FakeProc(),
    )

    assert read_docker_bridge_host_ip() == "172.17.0.1"


def test_apply_docker_host_args_rewrites_default_task_urls():
    args = SimpleNamespace(
        docker_host="auto",
        task1_server_url="http://host.docker.internal:8000",
        task2_server_url="http://host.docker.internal:8002",
        task3_server_url="http://host.docker.internal:8003",
    )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            dkm_online_integration,
            "build_task_server_urls",
            lambda docker_host=None: (
                "http://172.17.0.1:8000",
                "http://172.17.0.1:8002",
                "http://172.17.0.1:8003",
            ),
        )
        dkm_online_integration._apply_docker_host_args(args)

    assert args.task1_server_url == "http://172.17.0.1:8000"
    assert args.task3_server_url == "http://172.17.0.1:8003"


def test_describe_docker_host_integration_notes_includes_linux_guidance(monkeypatch):
    monkeypatch.setattr("src.common.nexent_online.sys.platform", "linux")
    monkeypatch.setattr(
        "src.common.nexent_online.read_docker_bridge_host_ip",
        lambda: "172.17.0.1",
    )

    notes = describe_docker_host_integration_notes(
        docker_host="auto",
        task1_server_url="http://172.17.0.1:8000",
        task2_server_url="http://172.17.0.1:8002",
        task3_server_url="http://172.17.0.1:8003",
    )

    assert notes["resolved_docker_host"] == "172.17.0.1"
    assert "linux_guidance" in notes
