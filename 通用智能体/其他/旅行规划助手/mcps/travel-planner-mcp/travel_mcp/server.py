# -*- coding: utf-8 -*-
"""零依赖 MCP 服务：同时支持 stdio 与 HTTP（Streamable HTTP + 旧版 SSE）。

用法：
    python -m travel_mcp --transport stdio
    python -m travel_mcp --transport http --host 0.0.0.0 --port 8765

HTTP 端点：
    POST /mcp                 Streamable HTTP（JSON-RPC，返回 application/json）
    GET  /sse                 旧版 HTTP+SSE（返回 endpoint 事件）
    POST /messages?session_id 旧版 SSE 的消息通道
    GET  /health              健康检查
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import tools as T

SERVER_NAME = "travel-planner-mcp"
SERVER_VERSION = "1.0.0"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

TOOL_SPECS = [
    {
        "name": "get_weather",
        "description": "查询城市未来 1-14 天天气预报（温度、降雨概率、风力）。用于判断穿衣、雨天备选与户外行程安排。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如 东京 / 曼谷 / 北京"},
                "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD，可留空表示从今天起"},
                "days": {"type": "integer", "description": "查询天数 1-14，默认 3", "minimum": 1, "maximum": 14},
            },
            "required": ["city"],
        },
        "annotations": {"readOnlyHint": True, "openWorldHint": True},
    },
    {
        "name": "search_poi",
        "description": "检索城市景点/街区/美食地点，返回片区、建议时长、门票、开放时间与贴士。用于挑选候选景点、核实开放信息。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"},
                "keyword": {"type": "string", "description": "关键词，如 夜景 / 博物馆 / 浅草"},
                "category": {"type": "string", "description": "类别过滤，如 寺庙 / 公园 / 购物"},
                "limit": {"type": "integer", "description": "返回条数 1-12，默认 6"},
            },
            "required": ["city"],
        },
        "annotations": {"readOnlyHint": True, "openWorldHint": True},
    },
    {
        "name": "plan_route",
        "description": "计算两点之间的距离与预计耗时，支持步行/驾车/骑行/公共交通估算。用于判断同一天行程是否顺路、排定出发时间。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "起点：景点名、城市名或“纬度,经度”"},
                "destination": {"type": "string", "description": "终点：景点名、城市名或“纬度,经度”"},
                "mode": {"type": "string", "enum": ["walk", "drive", "bike", "transit"], "description": "交通方式，默认 walk"},
                "city": {"type": "string", "description": "所在城市，用于解析景点名"},
            },
            "required": ["origin", "destination"],
        },
        "annotations": {"readOnlyHint": True, "openWorldHint": True},
    },
    {
        "name": "estimate_budget",
        "description": "按住宿/餐饮/市内交通/门票/其他分项估算旅行预算，可选估算往返大交通。用于回答“预算多少”“一万五够不够”。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "目的地城市"},
                "days": {"type": "integer", "description": "行程天数"},
                "travelers": {"type": "integer", "description": "出行人数，默认 1"},
                "style": {"type": "string", "enum": ["economy", "comfort", "premium"], "description": "消费档位，默认 comfort"},
                "origin_city": {"type": "string", "description": "出发城市，用于估算机票"},
                "include_flight": {"type": "boolean", "description": "是否包含往返大交通"},
            },
            "required": ["city", "days"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "build_itinerary",
        "description": "按片区聚类与游玩节奏生成逐日行程骨架（含时段、门票、午餐、片区提示）。生成完整行程时必须先调用此工具，再结合天气与开放信息调整。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "目的地城市"},
                "days": {"type": "integer", "description": "天数 1-10"},
                "pace": {"type": "string", "enum": ["relaxed", "moderate", "packed"], "description": "节奏，默认 moderate"},
                "interests": {"type": "array", "items": {"type": "string"}, "description": "兴趣标签，如 美食/博物馆/亲子/夜景"},
                "must_see": {"type": "array", "items": {"type": "string"}, "description": "必去景点名称列表"},
                "start_date": {"type": "string", "description": "出发日期 YYYY-MM-DD，用于避开周一闭馆"},
            },
            "required": ["city", "days"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "get_local_essentials",
        "description": "查询目的地生活信息：货币、插头电压、饮水、小费、紧急电话、交通卡、常用语。用于出行准备清单。",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
        "annotations": {"readOnlyHint": True},
    },
]

TOOL_FUNCS = {
    "get_weather": T.get_weather,
    "search_poi": T.search_poi,
    "plan_route": T.plan_route,
    "estimate_budget": T.estimate_budget,
    "build_itinerary": T.build_itinerary,
    "get_local_essentials": T.get_local_essentials,
}


class RpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _ok(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _call_tool(name: str, args: dict):
    if name not in TOOL_FUNCS:
        raise RpcError(-32602, f"未知工具：{name}")
    fn = TOOL_FUNCS[name]
    if not isinstance(args, dict):
        raise RpcError(-32602, "arguments 必须是对象")
    try:
        out = fn(**args)
    except TypeError as exc:
        raise RpcError(-32602, f"参数不匹配：{exc}") from exc
    except Exception as exc:  # 工具内部异常也返回结构化错误，避免拖垮会话
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "hint": "请换一种方式调用或直接告知用户工具暂不可用。"}
    text = json.dumps(out, ensure_ascii=False, indent=2)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": out,
        "isError": not bool(out.get("ok", False)),
    }


def dispatch(method: str, params: dict | None):
    params = params or {}
    if method == "initialize":
        requested = params.get("protocolVersion", "")
        version = requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": "旅行规划工具集：天气、景点、路线、预算、行程骨架、目的地生活信息。数据为演示用途，实时信息需复核。",
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": TOOL_SPECS}
    if method == "tools/call":
        name = params.get("name")
        if not name:
            raise RpcError(-32602, "缺少工具名 name")
        return _call_tool(name, params.get("arguments") or {})
    if method == "resources/list":
        return {"resources": []}
    if method == "prompts/list":
        return {"prompts": []}
    if method in ("notifications/initialized", "notifications/cancelled", "logging/setLevel"):
        return None
    raise RpcError(-32601, f"不支持的方法：{method}")


def handle_message(msg):
    """处理单条 JSON-RPC 消息，返回响应 dict 或 None（通知）。"""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return _err(None, -32600, "无效的 JSON-RPC 请求")
    method = msg.get("method")
    msg_id = msg.get("id")
    if not method:
        return _err(msg_id, -32600, "缺少 method")
    try:
        result = dispatch(method, msg.get("params"))
    except RpcError as exc:
        return _err(msg_id, exc.code, exc.message)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return _err(msg_id, -32603, f"服务内部错误：{exc}")
    if msg_id is None:
        return None
    return _ok(msg_id, result if result is not None else {})


# ---------------------------------------------------------------------------
# stdio
# ---------------------------------------------------------------------------

def serve_stdio():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps(_err(None, -32700, "JSON 解析失败")) + "\n")
            sys.stdout.flush()
            continue
        resp = handle_message(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

_SESSIONS: dict[str, "queue.Queue[str]"] = {}
_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "TravelMCP/1.0"

    def log_message(self, fmt, *args):  # 精简日志
        sys.stderr.write("[mcp] " + fmt % args + "\n")

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, Mcp-Session-Id, MCP-Protocol-Version")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, DELETE, OPTIONS")

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._json({"ok": True, "server": SERVER_NAME, "version": SERVER_VERSION, "tools": len(TOOL_SPECS)})
            return
        if path in ("/sse", "/sse/"):
            self._serve_sse()
            return
        if path in ("/mcp", "/mcp/"):
            self._serve_stream()
            return
        self._json({"error": "not_found"}, 404)

    def do_DELETE(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path not in ("/mcp", "/mcp/", "/messages", "/messages/"):
            self._json({"error": "not_found"}, 404)
            return
        try:
            payload = json.loads(self._read_body().decode("utf-8") or "null")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(_err(None, -32700, "JSON 解析失败"), 400)
            return

        if isinstance(payload, list):
            responses = [r for r in (handle_message(m) for m in payload) if r is not None]
            result = responses or None
        else:
            result = handle_message(payload)

        if path.startswith("/messages"):
            session_id = (parse_qs(parsed.query).get("session_id") or [""])[0]
            if result is not None and session_id:
                q = _SESSIONS.get(session_id)
                if q:
                    q.put(json.dumps(result, ensure_ascii=False))
            self.send_response(202)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if result is None:
            self.send_response(202)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._json(result)

    # -- 旧版 SSE ---------------------------------------------------------
    def _serve_stream(self):
        """Streamable HTTP 的可选 GET 流：仅用于保活/服务端通知。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                time.sleep(15)
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _serve_sse(self):
        session_id = uuid.uuid4().hex
        q: "queue.Queue[str]" = queue.Queue()
        with _LOCK:
            _SESSIONS[session_id] = q
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()

        def send(event, data):
            self.wfile.write(f"event: {event}\ndata: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

        send("endpoint", f"/messages?session_id={session_id}")
        try:
            while True:
                try:
                    data = q.get(timeout=15)
                    send("message", data)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _LOCK:
                _SESSIONS.pop(session_id, None)


def serve_http(host: str, port: int):
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    sys.stderr.write(f"[mcp] {SERVER_NAME} v{SERVER_VERSION} listening on http://{host}:{port}/mcp\n")
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Travel planner MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="http")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.transport == "stdio":
        serve_stdio()
    else:
        serve_http(args.host, args.port)


if __name__ == "__main__":
    main()
