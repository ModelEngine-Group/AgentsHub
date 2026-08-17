"""MinerU Agent 轻量解析 API 客户端。

该适配器负责本地文件签名上传、异步任务轮询和 Markdown 结果下载，
与 DataMate 清洗任务保持解耦。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_MINERU_BASE_URL = "https://mineru.net/api/v1/agent"
MAX_LIGHTWEIGHT_FILE_BYTES = 10 * 1024 * 1024
TERMINAL_STATES = {"done", "failed"}


class MineruRemoteError(RuntimeError):
    """远程 MinerU 请求或解析任务失败。"""


@dataclass(frozen=True)
class MineruParseResult:
    """一次远程文档解析的可审计结果。"""

    task_id: str
    source_name: str
    markdown_url: str
    markdown: str
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_name": self.source_name,
            "markdown_url": self.markdown_url,
            "output_chars": len(self.markdown),
            "elapsed_seconds": self.elapsed_seconds,
        }


class MineruAgentClient:
    """调用 MinerU 官方免 Token Agent API。"""

    def __init__(
        self,
        base_url: str = DEFAULT_MINERU_BASE_URL,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 2,
        request_timeout_seconds: float = 30,
        session: requests.Session | None = None,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = (
            None if timeout_seconds is None else max(10.0, float(timeout_seconds))
        )
        self.poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self.request_timeout_seconds = max(5.0, float(request_timeout_seconds))
        self.session = session or requests.Session()
        self._sleep = sleep
        self._monotonic = monotonic

    @classmethod
    def from_env(cls) -> "MineruAgentClient":
        """从统一环境变量创建客户端。"""

        timeout_value = os.environ.get("CCF_MINERU_TIMEOUT_SECONDS", "").strip()
        return cls(
            base_url=os.environ.get("CCF_MINERU_API", DEFAULT_MINERU_BASE_URL),
            timeout_seconds=float(timeout_value) if timeout_value else None,
            poll_interval_seconds=os.environ.get("CCF_MINERU_POLL_INTERVAL_SECONDS", "2"),
            request_timeout_seconds=os.environ.get("CCF_MINERU_REQUEST_TIMEOUT_SECONDS", "30"),
        )

    def probe(self) -> tuple[bool, str]:
        """通过无副作用的不存在任务查询验证远程接口可达。"""

        try:
            response = self.session.get(
                f"{self.base_url}/parse/mediflow-capability-probe",
                timeout=self.request_timeout_seconds,
            )
        except requests.RequestException as exc:
            return False, f"远程 MinerU 不可达：{type(exc).__name__}"
        if response.status_code >= 500:
            return False, f"远程 MinerU 响应 HTTP {response.status_code}"
        return True, f"远程 MinerU 响应 HTTP {response.status_code}"

    def parse_file(
        self,
        file_path: str | Path,
        *,
        language: str = "ch",
        enable_table: bool = True,
        is_ocr: bool = False,
        enable_formula: bool = True,
        page_range: str | None = None,
    ) -> MineruParseResult:
        """上传单个文档并等待 Markdown 解析结果。"""

        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size <= 0:
            raise MineruRemoteError(f"PDF 文件为空：{path.name}")
        if size > MAX_LIGHTWEIGHT_FILE_BYTES:
            raise MineruRemoteError(
                f"{path.name} 超过 MinerU 轻量接口 10 MB 限制；请拆分文件或配置精准解析服务"
            )

        payload: dict[str, Any] = {
            "file_name": path.name,
            "language": language,
            "enable_table": bool(enable_table),
            "is_ocr": bool(is_ocr),
            "enable_formula": bool(enable_formula),
        }
        if page_range:
            payload["page_range"] = page_range

        started = self._monotonic()
        create = self.session.post(
            f"{self.base_url}/parse/file",
            json=payload,
            timeout=self.request_timeout_seconds,
        )
        create.raise_for_status()
        create_data = self._api_data(create, "创建 MinerU 解析任务")
        task_id = str(create_data.get("task_id") or "").strip()
        upload_url = str(create_data.get("file_url") or "").strip()
        if not task_id or not upload_url:
            raise MineruRemoteError("MinerU 创建任务响应缺少 task_id 或 file_url")

        with path.open("rb") as stream:
            uploaded = self.session.put(
                upload_url,
                data=stream,
                timeout=max(self.request_timeout_seconds, 60),
            )
        uploaded.raise_for_status()

        status_data = self._wait_for_result(task_id, started)
        markdown_url = str(status_data.get("markdown_url") or "").strip()
        if not markdown_url:
            raise MineruRemoteError("MinerU 已完成任务但未返回 markdown_url")
        markdown_response = self.session.get(markdown_url, timeout=self.request_timeout_seconds)
        markdown_response.raise_for_status()
        markdown = markdown_response.text.strip()
        if not markdown:
            raise MineruRemoteError("MinerU 返回的 Markdown 内容为空")
        return MineruParseResult(
            task_id=task_id,
            source_name=path.name,
            markdown_url=markdown_url,
            markdown=markdown,
            elapsed_seconds=round(self._monotonic() - started, 3),
        )

    def _wait_for_result(self, task_id: str, started: float) -> dict[str, Any]:
        while self.timeout_seconds is None or self._monotonic() - started < self.timeout_seconds:
            try:
                response = self.session.get(
                    f"{self.base_url}/parse/{task_id}",
                    timeout=self.request_timeout_seconds,
                )
            except requests.RequestException as exc:
                print(f"[MinerU] status request failed; keep waiting: {type(exc).__name__}")
                self._sleep(self.poll_interval_seconds)
                continue
            response.raise_for_status()
            data = self._api_data(response, "查询 MinerU 解析任务")
            state = str(data.get("state") or "").strip().lower()
            if state in TERMINAL_STATES:
                if state == "failed":
                    detail = data.get("err_msg") or data.get("err_code") or "未知错误"
                    raise MineruRemoteError(f"MinerU 解析失败：{detail}")
                return data
            self._sleep(self.poll_interval_seconds)
        raise MineruRemoteError(f"MinerU 解析等待超时（{self.timeout_seconds:g} 秒）")

    @staticmethod
    def _api_data(response: requests.Response, action: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise MineruRemoteError(f"{action}返回非 JSON 响应") from exc
        if payload.get("code") != 0:
            raise MineruRemoteError(f"{action}失败：{payload.get('msg') or payload.get('code')}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise MineruRemoteError(f"{action}响应缺少 data")
        return data
