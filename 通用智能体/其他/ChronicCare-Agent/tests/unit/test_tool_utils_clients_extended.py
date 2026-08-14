from __future__ import annotations

import io
import json
import socket
import urllib.error
from pathlib import Path

import pytest

from mcp_adapter.chroniccare_http import ChronicCareClient, ChronicCareHTTPError
from mcp_adapter.client import MCPAdapterClient
from orchestration import question_pipeline
from tool_server import utils


def test_load_server_config_yaml_and_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("server:\n  host: 127.0.0.1\n  port: 18088\nname: demo\n", encoding="utf-8")
    monkeypatch.setattr(utils, "resolve_path", lambda value: path)
    assert utils.load_server_config()["server"]["port"] == 18088
    monkeypatch.setattr(utils, "yaml", None)
    parsed = utils.load_server_config()
    assert parsed == {"server": {"host": "127.0.0.1", "port": 18088}, "name": "demo"}


def test_url_identity_safety_and_route_helpers() -> None:
    config = {
        "server": {
            "host": "0.0.0.0",
            "public_host": "public",
            "port": 19000,
            "browser_base_url": "http://browser/base/",
        },
        "safety": {"medical_safety_note": "安全"},
    }
    assert utils.build_service_base_url(config) == "http://public:19000"
    assert utils.build_base_url(config) == "http://browser/base"
    assert utils.public_artifact_url(config, "/x") == "http://browser/base/x"
    assert utils.service_artifact_url(config, "/x") == "http://public:19000/x"
    assert utils.project_identity(config)["project"] == "ChronicCare-Agent"
    assert utils.safety_note(config) == "安全"
    assert utils.artifact_route_path("") == "/"
    assert utils.artifact_route_path("x") == "/x"
    assert utils.artifact_route_path("/x") == "/x"


def test_artifact_status_optional_json_and_parent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    present = tmp_path / "value.json"
    present.write_text('{"value": 3}', encoding="utf-8")
    monkeypatch.setattr(
        utils, "resolve_path", lambda value: Path(value) if Path(value).is_absolute() else tmp_path / value
    )
    monkeypatch.setattr(utils, "relative_to_project", lambda path: path.as_posix())
    assert utils.artifact_status(present)["size_bytes"] > 0
    assert utils.artifact_status("missing")["exists"] is False
    assert utils.read_optional_json(present) == {"value": 3}
    assert utils.read_optional_json("missing") == {}
    assert utils.ensure_parent("a/b.json").parent.is_dir()
    assert utils.file_response_meta(present)["exists"] is True


def test_current_metrics_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "configs/current_metrics.json": {},
        "outputs/enhanced/current_metrics_snapshot.json": {"patients": 2000},
    }
    monkeypatch.setattr(utils, "read_optional_json", lambda path: values.get(path, {}))
    assert utils.load_current_metrics() == {"patients": 2000}
    monkeypatch.setattr(utils, "read_optional_json", lambda path: {})
    assert utils.load_current_metrics() == {}


def test_sqlite_fetch_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    connection = utils.sqlite3.connect(db)
    connection.execute("CREATE TABLE item(id INTEGER, name TEXT)")
    connection.executemany("INSERT INTO item VALUES (?, ?)", [(1, "a"), (2, "b")])
    connection.commit()
    connection.close()
    monkeypatch.setattr(utils, "load_server_config", lambda: {"paths": {"sqlite_db": str(db)}})
    monkeypatch.setattr(utils, "resolve_path", lambda value: Path(value))
    assert utils.fetch_rows("SELECT * FROM item ORDER BY id", [])[1]["name"] == "b"
    assert utils.fetch_one("SELECT * FROM item WHERE id = ?", [1])["name"] == "a"
    assert utils.fetch_one("SELECT * FROM item WHERE id = 99") == {}


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("disease_inventory_distribution.png", "disease_inventory_distribution.svg"),
        ("line_followup_trend_9d.png", "line_followup_trend_9d.svg"),
        ("followup_high_risk_45d.png", "line_followup_trend_high_risk_45d.svg"),
        ("hba1c_trend_3m.svg", "analysis_trend_hba1c_abnormal_3m.svg"),
        ("unchanged.jpg", "unchanged.jpg"),
    ],
)
def test_chart_alias_patterns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    filename: str,
    expected: str,
) -> None:
    monkeypatch.setattr(utils, "resolve_path", lambda value: tmp_path / str(value))
    assert utils._chart_alias_target(filename) == expected


@pytest.mark.parametrize(
    ("analysis_id", "expected"),
    [
        ("followup_high_risk_7_days", "analysis_future_followup_chart_bundle_high_risk_7d_chart"),
        ("analysis_followup_high_risk_9d_chart", "analysis_future_followup_chart_bundle_high_risk_9d_chart"),
        ("analysis_disease_distribution", "analysis_disease_inventory"),
        ("kg_subgraph_%E9%AB%98%E8%A1%80%E5%8E%8B", "kg_subgraph_高血压"),
        ("custom", "custom"),
    ],
)
def test_graph_driven_aliases(analysis_id: str, expected: str) -> None:
    assert utils._graph_driven_alias_target(analysis_id) == expected


def test_artifact_exists_for_static_dynamic_and_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(utils, "resolve_path", lambda value: tmp_path / str(value))
    assert utils.artifact_exists_for_route("") is False
    assert utils.artifact_exists_for_route("/artifacts/charts") is True
    assert utils.artifact_exists_for_route("/artifacts/report") is True
    assert utils.artifact_exists_for_route("/artifacts/kg_subgraph_高血压.svg") is True
    assert utils.artifact_exists_for_route("/artifacts/subgraphs/") is False
    assert utils.artifact_exists_for_route("/artifacts/graph-driven/kg_subgraph_高血压") is True
    assert utils.artifact_exists_for_route("/artifacts/open-nl2sql/missing.json") is False
    assert utils.artifact_exists_for_route("/other") is False

    chart = tmp_path / "outputs/runtime_generated/charts/custom.svg"
    chart.parent.mkdir(parents=True)
    chart.write_text("<svg/>", encoding="utf-8")
    assert utils.artifact_exists_for_route("/artifacts/charts/custom.png") is True

    subgraph = tmp_path / "outputs/runtime_generated/subgraphs/subgraph_demo.html"
    subgraph.parent.mkdir(parents=True)
    subgraph.write_text("html", encoding="utf-8")
    assert utils.artifact_exists_for_route("/artifacts/subgraphs/subgraph_demo") is True


def test_latest_subgraph_urls_and_metric_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(utils, "resolve_path", lambda value: tmp_path / str(value))
    config = {"server": {"browser_base_url": "http://browser", "host": "host", "port": 1}}
    assert utils.latest_subgraph_path() is None
    assert utils.latest_subgraph_public_url(config) == "http://browser/artifacts/graph.html"
    assert utils.latest_subgraph_service_url(config) == "http://host:1/artifacts/graph.html"
    first = tmp_path / "outputs/runtime_generated/subgraphs/a.html"
    first.parent.mkdir(parents=True)
    first.write_text("a", encoding="utf-8")
    assert utils.latest_subgraph_path() == first
    assert "/artifacts/subgraphs/a?v=" in utils.latest_subgraph_public_url(config)
    rows = [{"name": "a", "value": 1}, {"name": "b", "value": 3}]
    assert utils.summarize_metric_rows(rows, "name", "value", 1) == [{"name": "b", "value": 3}]


class _Response:
    def __init__(self, payload) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


def test_chroniccare_http_client_success_and_headers() -> None:
    client = ChronicCareClient("http://local/", timeout=4, conversation_id=" c1 ")
    captured = {}

    def open_request(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response({"status": "success"})

    client._opener.open = open_request
    assert client.get("/health")["status"] == "success"
    assert client.post("/tool", {"中文": "值"}, timeout=9)["status"] == "success"
    assert client.url_for("/x") == "http://local/x"
    assert captured["timeout"] == 9
    assert captured["request"].get_header("X-chroniccare-conversation-id") == "c1"
    assert captured["request"].data is not None


def test_chroniccare_http_client_error_branches() -> None:
    client = ChronicCareClient("http://local")
    assert ChronicCareClient._parse_json("u", '{"ok": true}') == {"ok": True}
    with pytest.raises(ChronicCareHTTPError, match="Expected JSON"):
        ChronicCareClient._parse_json("u", "bad")
    with pytest.raises(ChronicCareHTTPError, match="JSON object"):
        ChronicCareClient._parse_json("u", "[]")

    error = urllib.error.HTTPError("u", 500, "bad", {}, io.BytesIO(b"server error"))
    client._opener.open = lambda request, timeout: (_ for _ in ()).throw(error)
    with pytest.raises(ChronicCareHTTPError, match="HTTP 500"):
        client.get("/x")

    client._opener.open = lambda request, timeout: (_ for _ in ()).throw(urllib.error.URLError(socket.timeout()))
    with pytest.raises(ChronicCareHTTPError, match="timed out"):
        client.get("/x")
    client._opener.open = lambda request, timeout: (_ for _ in ()).throw(urllib.error.URLError("offline"))
    with pytest.raises(ChronicCareHTTPError, match="offline"):
        client.get("/x")
    client._opener.open = lambda request, timeout: (_ for _ in ()).throw(TimeoutError())
    with pytest.raises(ChronicCareHTTPError, match="timed out"):
        client.get("/x")


def test_mcp_adapter_client_success_and_rpc_error() -> None:
    client = MCPAdapterClient("http://mcp/")
    replies = [
        {"result": {"server": "ok"}},
        {"result": {"tools": [{"name": "a"}]}},
        {"result": {"content": "done"}},
    ]
    client._opener.open = lambda request, timeout: _Response(replies.pop(0))
    assert client.initialize()["server"] == "ok"
    assert client.list_tools() == [{"name": "a"}]
    assert client.call_tool("a")["content"] == "done"
    client._opener.open = lambda request, timeout: _Response({"error": {"message": "failed"}})
    with pytest.raises(RuntimeError, match="failed"):
        client.call_tool("a")


def test_question_pipeline_build_run_and_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        question_pipeline,
        "classify_question",
        lambda payload: {"intent": "data_summary", "normalized_entities": {}},
    )
    monkeypatch.setattr(
        question_pipeline,
        "build_query_plan",
        lambda classified, query: {"intent": classified["intent"], "query": query},
    )
    monkeypatch.setattr(question_pipeline, "execute_query_plan", lambda plan: {"answer": "raw"})
    monkeypatch.setattr(
        question_pipeline,
        "format_answer",
        lambda plan, payload: {**payload, "answer": "formatted"},
    )
    built = question_pipeline.build_question_pipeline("数据规模")
    assert built["plan"]["intent"] == "data_summary"
    result = question_pipeline.run_question_pipeline("数据规模")
    assert result["answer"] == "formatted" and result["rule_pipeline"]["classification"]
    monkeypatch.setattr(question_pipeline, "execute_query_plan", lambda plan: None)
    assert question_pipeline.run_question_pipeline("数据规模") is None
