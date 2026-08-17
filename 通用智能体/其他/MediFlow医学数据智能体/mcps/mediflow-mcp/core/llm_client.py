# -*- coding: utf-8 -*-
"""LLM 统一调用出口，所有 LLM 请求经此模块发出。"""
import re
import json
import threading
import requests
from typing import Any, Optional


class LLMClient:
    def __init__(self,
                 base_url: str = "https://api.deepseek.com/v1/chat/completions",
                 model: str = "deepseek-chat",
                 timeout: int = 45,
                 api_key: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.temperature = max(0.0, min(2.0, float(temperature)))
        try:
            configured_max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            configured_max_tokens = 4096
        self.max_tokens = max(256, min(16384, configured_max_tokens))
        # Gap extraction can run in parallel.  Keep one HTTP session per
        # worker thread so repeated requests reuse TLS connections without
        # sharing a requests.Session instance across threads.
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=8,
                pool_maxsize=8,
                max_retries=0,
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._thread_local.session = session
        return session

    def chat(self, prompt: str, system: Optional[str] = None) -> str:
        """单轮对话，返回纯文本（已剥离 think 标签）"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = self._session().post(
            self.base_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return self._strip_think(content).strip()

    def chat_json(self, prompt: str, system: Optional[str] = None) -> Any:
        """要求模型输出 JSON，自动解包并解析。失败返回 None。"""
        raw = self.chat(prompt, system=system)
        return self._extract_json(raw)

    # ---------- 内部工具 ----------

    @staticmethod
    def _strip_think(text: str) -> str:
        """剥离 think 标签"""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    @staticmethod
    def _extract_json(text: str) -> Any:
        """从模型输出中提取 JSON（处理 markdown 包裹和前后噪声）"""
        text = LLMClient._strip_think(text).strip()

        # 去掉 ```json ... ``` 包裹
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

        # 直接尝试
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 截取第一个 [ 或 { 到对应的最后一个 ] 或 }
        for open_ch, close_ch in [("[", "]"), ("{", "}")]:
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
        return None
