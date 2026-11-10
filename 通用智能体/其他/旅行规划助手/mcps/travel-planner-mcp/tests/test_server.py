# -*- coding: utf-8 -*-
import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from travel_mcp.server import Handler


class TestMcpHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def rpc(self, method, params=None, msg_id=1):
        body = json.dumps({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/mcp", data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def test_initialize(self):
        out = self.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                      "clientInfo": {"name": "test", "version": "0"}})
        self.assertEqual(out["result"]["serverInfo"]["name"], "travel-planner-mcp")
        self.assertIn("tools", out["result"]["capabilities"])

    def test_tools_list(self):
        out = self.rpc("tools/list")
        names = {t["name"] for t in out["result"]["tools"]}
        self.assertEqual(len(names), 6)
        self.assertIn("build_itinerary", names)

    def test_tools_call(self):
        out = self.rpc("tools/call", {"name": "build_itinerary",
                                      "arguments": {"city": "成都", "days": 2, "pace": "relaxed"}})
        result = out["result"]
        self.assertFalse(result["isError"])
        payload = json.loads(result["content"][0]["text"])
        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["plan"]), 2)

    def test_unknown_tool(self):
        out = self.rpc("tools/call", {"name": "nope", "arguments": {}})
        self.assertEqual(out["error"]["code"], -32602)

    def test_health(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as resp:
            body = json.loads(resp.read().decode())
        self.assertTrue(body["ok"])


if __name__ == "__main__":
    unittest.main()
