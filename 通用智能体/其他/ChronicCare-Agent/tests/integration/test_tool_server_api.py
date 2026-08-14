from __future__ import annotations

from fastapi.testclient import TestClient

from tool_server import app as tool_app


def _client(monkeypatch) -> TestClient:
    monkeypatch.setattr(tool_app, "load_server_config", lambda: {})
    monkeypatch.setattr(
        tool_app,
        "project_identity",
        lambda config: {
            "project": "ChronicCare-Agent",
            "base_url": "http://127.0.0.1:18088",
            "service_base_url": "http://chroniccare-runtime:18088",
        },
    )
    monkeypatch.setattr(tool_app, "safety_note", lambda config: "仅用于辅助分析")
    return TestClient(tool_app.app)


def test_tool_server_root_health_tools_and_openapi(monkeypatch) -> None:
    client = _client(monkeypatch)
    root = client.get("/")
    health = client.get("/health")
    tools = client.get("/tools")
    openapi = client.get("/openapi.json")
    assert root.status_code == 200
    assert root.json()["project"] == "ChronicCare-Agent"
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert tools.status_code == 200
    tool_rows = tools.json()["tools"]
    assert tools.json()["tool_count"] == len(tool_rows)
    assert len({row["name"] for row in tool_rows}) == len(tool_rows)
    assert any(row["path"] == "/analysis/open-query" for row in tool_rows)
    assert openapi.status_code == 200
    assert "/analysis/open-sql/query" in openapi.json()["paths"]


def test_tool_server_middleware_binds_and_resets_conversation(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        tool_app,
        "bind_conversation_context",
        lambda conversation_id: events.append(("bind", conversation_id)) or "token",
    )
    monkeypatch.setattr(
        tool_app,
        "reset_conversation_context",
        lambda token: events.append(("reset", token)),
    )
    client = _client(monkeypatch)
    response = client.get(
        "/health",
        headers={"X-ChronicCare-Conversation-ID": "conversation-001"},
    )
    assert response.status_code == 200
    assert events == [("bind", "conversation-001"), ("reset", "token")]


def test_tool_server_rejects_invalid_request_schema(monkeypatch) -> None:
    client = _client(monkeypatch)
    response = client.post("/analysis/open-sql/query", json={})
    assert response.status_code == 422
