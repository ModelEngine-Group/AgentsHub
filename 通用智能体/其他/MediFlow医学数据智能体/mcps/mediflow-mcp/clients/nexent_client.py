# -*- coding: utf-8 -*-
"""Nexent 平台适配客户端。

封装登录、智能体配置、版本发布、MCP 注册、工具发现和流式对话，
使业务服务只依赖稳定的方法接口。具体服务地址和凭据由运行环境提供。
"""
import json
import requests
from typing import Optional, Dict, Any, Iterator


class NexentClient:
    def __init__(self, config_base: str, runtime_base: str, email: str, password: str):
        """
        config_base: http://host:5010
        runtime_base: http://host:5014
        """
        self.config_base = config_base.rstrip("/")
        self.runtime_base = runtime_base.rstrip("/")
        self.email = email
        self.password = password
        self._token: Optional[str] = None

    # ---------- 认证 ----------
    def login(self) -> str:
        resp = requests.post(f"{self.config_base}/user/signin",
                             json={"email": self.email, "password": self.password})
        resp.raise_for_status()
        self._token = resp.json()["data"]["session"]["access_token"]
        return self._token

    @property
    def headers(self) -> Dict[str, str]:
        if not self._token:
            self.login()
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    # ---------- MCP 注册 ----------
    def add_mcp_server(self, mcp_url: str, service_name: str,
                       authorization_token: Optional[str] = None) -> Dict[str, Any]:
        """注册远程 MCP server；兼容新版 JSON 请求和旧版查询参数。"""
        payload = {
            "name": service_name,
            "server_url": mcp_url,
            "description": "MediFlow medical data MCP service",
            "source": "local",
            "tags": ["medical", "data"],
            "enabled": True,
        }
        if authorization_token:
            payload["authorization_token"] = authorization_token
        resp = requests.post(
            f"{self.config_base}/mcp/add", json=payload, headers=self.headers, timeout=30
        )
        if resp.status_code in (404, 405, 422):
            params = {"mcp_url": mcp_url, "service_name": service_name}
            if authorization_token:
                params["authorization_token"] = authorization_token
            resp = requests.post(
                f"{self.config_base}/mcp/add", params=params, headers=self.headers, timeout=30
            )
        resp.raise_for_status()
        return resp.json()

    def list_mcp_tools(self, mcp_url: str, service_name: str) -> Dict[str, Any]:
        """列出某 MCP server 暴露的工具；优先使用新版 mcp_id 接口。"""
        records_resp = requests.get(
            f"{self.config_base}/mcp/list", headers=self.headers, timeout=20
        )
        records_resp.raise_for_status()
        payload = records_resp.json()
        records = payload if isinstance(payload, list) else payload.get(
            "remote_mcp_server_list", payload.get("data", payload.get("mcp_list", []))
        )
        mcp_id = None
        for item in records if isinstance(records, list) else []:
            item_name = item.get("remote_mcp_server_name") or item.get("service_name") or item.get("name")
            item_url = item.get("remote_mcp_server") or item.get("mcp_url") or item.get("server_url")
            if item_name == service_name or str(item_url or "").rstrip("/") == mcp_url.rstrip("/"):
                mcp_id = item.get("mcp_id") or item.get("id")
                break

        if mcp_id is not None:
            resp = requests.get(
                f"{self.config_base}/mcp/tools",
                params={"mcp_id": mcp_id},
                headers=self.headers,
                timeout=30,
            )
        else:
            resp = requests.post(
                f"{self.config_base}/mcp/tools",
                params={"mcp_url": mcp_url, "service_name": service_name},
                headers=self.headers,
                timeout=30,
            )
        resp.raise_for_status()
        return resp.json()

    def scan_tools(self) -> Dict[str, Any]:
        """扫描所有 MCP server，同步工具到 tool 表（必须在绑定工具前调用）。"""
        resp = requests.get(f"{self.config_base}/tool/scan_tool", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def list_tools(self) -> list:
        """列出全部工具（含 MCP 工具，有 tool_id）。"""
        resp = requests.get(f"{self.config_base}/tool/list", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def list_agents(self) -> list:
        """列出当前租户下 Agent。"""
        resp = requests.get(f"{self.config_base}/agent/list", headers=self.headers)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "agents", "agent_list", "items"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        return []

    # ---------- Agent 管理 ----------
    def get_agent_info(self, agent_id: int, version_no: int = 0) -> Dict[str, Any]:
        """获取 Agent 详情（version_no=0 为 draft，正整数为发布版本）。"""
        resp = requests.post(f"{self.config_base}/agent/search_info",
                             headers=self.headers, json={"agent_id": agent_id, "version_no": version_no})
        resp.raise_for_status()
        return resp.json()

    def update_agent(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """更新 Agent draft (version_no=0)。config 中 agent_id 必填，其余字段可选更新。
        工具绑定用 enabled_tool_ids: [tool_id, ...]（需先 scan_tools 获取 tool_id）。
        更新后必须调用 publish_agent 才能在对话中生效。"""
        resp = requests.post(f"{self.config_base}/agent/update", headers=self.headers, json=config)
        resp.raise_for_status()
        return resp.json()

    def publish_agent(self, agent_id: int, version_name: str = "v-auto",
                      release_note: str = "") -> Dict[str, Any]:
        """发布 Agent 新版本（让 draft 的改动在对话中生效）。"""
        resp = requests.post(f"{self.config_base}/agent/{agent_id}/publish",
                             headers=self.headers,
                             json={"version_name": version_name, "release_note": release_note})
        resp.raise_for_status()
        return resp.json()

    # ---------- 对话（SSE 流式）----------
    def run_agent_stream(self, agent_id: int, query: str,
                         conversation_id: Optional[int] = None,
                         stream_timeout: Optional[float] = None) -> Iterator[Dict[str, Any]]:
        """运行 Agent，以 SSE 事件流形式返回（用于 main_pipeline 或测试）。
        事件类型：agent_new_run, step_count, model_output_thinking,
                  parse(工具调用), final_answer 等。"""
        payload = {"agent_id": agent_id, "query": query}
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        with requests.post(f"{self.runtime_base}/agent/run",
                           headers=self.headers, json=payload,
                           stream=True, timeout=stream_timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                ls = line.decode("utf-8", errors="replace")
                if ls.startswith("data:"):
                    raw = ls[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError:
                        pass

    def run_agent(self, agent_id: int, query: str) -> str:
        """运行 Agent，阻塞等待 final_answer，返回最终回复文本。"""
        final = ""
        for event in self.run_agent_stream(agent_id, query):
            if event.get("type") == "final_answer":
                final = event.get("content", "")
        return final

    # ---------- 知识库（/indices） ----------

    def list_knowledge_bases(self) -> list:
        """列出当前租户下所有知识库。"""
        resp = requests.get(f"{self.config_base}/indices", headers=self.headers)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data.get("data", data.get("indices", []))
        return data if isinstance(data, list) else []

    def create_knowledge_base(self, kb_name: str,
                               embedding_model_name: Optional[str] = None) -> Dict[str, Any]:
        """创建知识库（ES 索引）。kb_name 即知识库展示名。"""
        body = {}
        if embedding_model_name:
            body["embedding_model_name"] = embedding_model_name
        resp = requests.post(f"{self.config_base}/indices/{kb_name}",
                             headers=self.headers, json=body)
        resp.raise_for_status()
        return resp.json()

    def check_kb_exists(self, kb_name: str) -> bool:
        """检查知识库是否已存在（按用户显示名）。"""
        resp = requests.post(f"{self.config_base}/indices/check_exist",
                             headers=self.headers, json={"knowledge_name": kb_name})
        resp.raise_for_status()
        data = resp.json()
        # Nexent 返回 {"status": "exists_in_tenant"} 或 {"exists": true}
        status = data.get("status", "")
        return status == "exists_in_tenant" or bool(data.get("exists", data.get("exist", False)))

    def get_es_index_name(self, kb_name: str) -> Optional[str]:
        """根据用户知识库名查找实际 ES index name（用于文档导入 URL）。
        Nexent 创建 KB 时返回的 id 字段即为 ES index name（如 2-xxxxxxxx...）。
        若本地没缓存，通过创建返回值获取；如已存在则从 list 中匹配。"""
        resp = requests.get(f"{self.config_base}/indices", headers=self.headers)
        resp.raise_for_status()
        data = resp.json()
        # Nexent 返回示例：{"indices": ["1-xxx", "2-xxx"], "count": N}
        indices = data.get("indices", []) if isinstance(data, dict) else data
        # 尝试创建并捕获返回的 id（如果不存在）
        # 由于无直接映射 API，通过 check_exist 后重新 POST 创建来获取 id
        exists_resp = requests.post(f"{self.config_base}/indices/check_exist",
                                    headers=self.headers, json={"knowledge_name": kb_name})
        if exists_resp.json().get("status") == "exists_in_tenant":
            # 已存在：无法通过 API 直接查到 ES index name，需要调用方自行传入
            return None
        return None

    def import_documents(self, kb_es_index: str, docs: list) -> Dict[str, Any]:
        """向知识库导入文档列表。
        kb_es_index: 实际 ES index name（如 "2-f9c6bcf1202c4080a55a2df360f63e45"），
                     不是用户显示名。
        每条文档格式：
          {
            "content": "文本内容",
            "path_or_url": "唯一标识（如 medical_kg/disease_name）",
            "source_type": "local",
            "file_size": <int>,
            "filename": "xxx.txt",
            "metadata": {"title": ..., "languages": ["zh"], "author": ..., "date": ...}
          }
        """
        resp = requests.post(f"{self.config_base}/indices/{kb_es_index}/documents",
                             headers=self.headers, json=docs,
                             timeout=120)
        resp.raise_for_status()
        return resp.json()
