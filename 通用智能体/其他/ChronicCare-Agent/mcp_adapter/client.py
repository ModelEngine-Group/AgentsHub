from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List


class MCPAdapterClient:
    def __init__(self, base_url: str, timeout: int = 10) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def initialize(self) -> Dict[str, Any]:
        return self._rpc("initialize", {"clientInfo": {"name": "chroniccare-local-client", "version": "0.1.0"}})

    def list_tools(self) -> List[Dict[str, Any]]:
        result = self._rpc("tools/list", {})
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self._rpc("tools/call", {"name": name, "arguments": arguments or {}})

    def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": f"{method}-1", "method": method, "params": params}
        request = urllib.request.Request(
            url=f"{self.base_url}/mcp",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"},
            method="POST",
        )
        with self._opener.open(request, timeout=self.timeout) as response:
            doc = json.loads(response.read().decode("utf-8"))
        if doc.get("error"):
            raise RuntimeError(doc["error"]["message"])
        return doc.get("result", {})
