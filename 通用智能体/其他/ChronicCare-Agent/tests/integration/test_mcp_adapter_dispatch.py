from __future__ import annotations

from fastapi.testclient import TestClient

from mcp_adapter import server


def _client(monkeypatch) -> tuple[TestClient, str]:
    known_tool = next(iter(server.get_tool_map()))
    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: {
            "tool_server_url": "http://tool-server:18088",
            "transport": "streamable-http",
            "sdk_available": True,
        },
    )
    monkeypatch.setattr(
        server,
        "execute_tool",
        lambda name, arguments: {
            "tool": name,
            "text": "调用成功",
            "data": {"echo": arguments},
        },
    )
    monkeypatch.setattr(server, "load_recent_traces", lambda limit: [])
    monkeypatch.setattr(
        server,
        "summarize_traces",
        lambda: {"status": "success", "total_calls": 0},
    )
    return TestClient(server.create_app()), known_tool


def _rpc(method: str, params: dict | None = None) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": f"{method}-1",
        "method": method,
        "params": params or {},
    }


def test_mcp_initialize_ping_and_tool_list(monkeypatch) -> None:
    client, _ = _client(monkeypatch)
    initialized = client.post("/mcp", json=_rpc("initialize"))
    ping = client.post("/mcp", json=_rpc("ping"))
    tools = client.post("/mcp", json=_rpc("tools/list"))
    assert initialized.status_code == 200
    assert initialized.json()["result"]["protocolVersion"] == "2024-11-05"
    assert ping.json()["result"] == {"pong": True}
    listed = tools.json()["result"]["tools"]
    assert listed
    assert all("conversation_id" in item["inputSchema"]["properties"] for item in listed)


def test_mcp_dispatches_tool_call_without_network(monkeypatch) -> None:
    client, known_tool = _client(monkeypatch)
    response = client.post(
        "/mcp",
        json=_rpc(
            "tools/call",
            {"name": known_tool, "arguments": {"query": "高血压"}},
        ),
    )
    result = response.json()["result"]
    assert result["isError"] is False
    assert result["content"][0]["text"] == "调用成功"
    assert result["structuredContent"] == {"echo": {"query": "高血压"}}


def test_mcp_returns_json_rpc_error_for_unknown_method_and_tool(monkeypatch) -> None:
    client, _ = _client(monkeypatch)
    unknown_method = client.post("/mcp", json=_rpc("unsupported"))
    unknown_tool = client.post(
        "/mcp",
        json=_rpc("tools/call", {"name": "missing", "arguments": {}}),
    )
    assert unknown_method.json()["error"]["code"] == -32601
    assert unknown_tool.json()["error"]["code"] == -32601


def test_mcp_http_invoke_rejects_unknown_tool(monkeypatch) -> None:
    client, _ = _client(monkeypatch)
    response = client.post("/invoke", json={"name": "missing", "arguments": {}})
    assert response.status_code == 404
