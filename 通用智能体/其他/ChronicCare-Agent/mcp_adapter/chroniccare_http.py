from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


class ChronicCareHTTPError(RuntimeError):
    pass


class ChronicCareClient:
    def __init__(self, base_url: str, timeout: int = 30, conversation_id: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.conversation_id = str(conversation_id or "").strip() or None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(self, path: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        return self._request("GET", path, None, timeout=timeout)

    def post(self, path: str, payload: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        return self._request("POST", path, payload or {}, timeout=timeout)

    def url_for(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]],
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        url = self.url_for(path)
        data = None
        headers = {"Accept": "application/json"}
        if self.conversation_id:
            headers["X-ChronicCare-Conversation-ID"] = self.conversation_id
        request_timeout = timeout if timeout is not None else self.timeout
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = urllib.request.Request(url=url, data=data, headers=headers, method=method.upper())
        try:
            with self._opener.open(request, timeout=request_timeout) as response:
                raw = response.read().decode("utf-8")
                return self._parse_json(url, raw)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ChronicCareHTTPError(f"{method.upper()} {url} failed with HTTP {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, socket.timeout):
                raise ChronicCareHTTPError(f"{method.upper()} {url} timed out after {request_timeout}s") from exc
            raise ChronicCareHTTPError(f"{method.upper()} {url} failed: {reason}") from exc
        except TimeoutError as exc:
            raise ChronicCareHTTPError(f"{method.upper()} {url} timed out after {request_timeout}s") from exc

    @staticmethod
    def _parse_json(url: str, raw: str) -> Dict[str, Any]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ChronicCareHTTPError(f"Expected JSON response from {url}, but got: {raw[:200]}") from exc
        if not isinstance(data, dict):
            raise ChronicCareHTTPError(f"Expected JSON object response from {url}, but got {type(data).__name__}")
        return data
