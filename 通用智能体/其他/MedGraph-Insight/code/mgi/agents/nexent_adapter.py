"""Nexent 集成适配层。

安装 nexent（pip install "med-graph-insight[nexent]"）并配置模型后，
可把本仓库的算子/工具注册为 Nexent Tool，由 Nexent CoreAgent 的
ReAct（think-act-observe）循环驱动整个「数据→知识→洞察」链路。

未安装 nexent 时本模块可安全导入，NEXENT_AVAILABLE=False，
框架自动使用 mgi.agents.base 中的内置规划-执行循环。
"""

from __future__ import annotations

import json
from typing import Any, Callable

try:
    from smolagents.tools import Tool as SmolTool
    from nexent.core.agents.core_agent import CoreAgent
    from nexent.core.models.openai_llm import OpenAIModel
    from nexent.core.utils.observer import MessageObserver

    NEXENT_AVAILABLE = True
except ImportError:
    NEXENT_AVAILABLE = False
    SmolTool = object  # type: ignore[assignment,misc]


def wrap_tool(name: str, description: str, fn: Callable[..., tuple[Any, str]]):
    """把 mgi 工具包装为 Nexent/smolagents Tool。"""
    if not NEXENT_AVAILABLE:
        raise RuntimeError("未安装 nexent，无法创建 Nexent Tool")

    class _WrappedTool(SmolTool):
        pass

    _WrappedTool.name = name
    _WrappedTool.description = description
    _WrappedTool.inputs = {
        "params_json": {"type": "string", "description": "工具参数（JSON 对象字符串）"},
    }
    _WrappedTool.output_type = "string"

    def forward(self, params_json: str = "{}") -> str:  # noqa: ANN001
        params = json.loads(params_json or "{}")
        _, summary = fn(None, **params)
        return summary

    _WrappedTool.forward = forward
    return _WrappedTool()


def build_core_agent(agent, model_id: str, api_key: str, api_base: str, max_steps: int = 8):
    """用 mgi 智能体的工具集构建 Nexent CoreAgent。

    agent: mgi.agents.base.BaseAgent 实例（提供 tools 与 tool_descriptions）。
    """
    if not NEXENT_AVAILABLE:
        raise RuntimeError("未安装 nexent。请 pip install nexent 后重试。")
    observer = MessageObserver()
    model = OpenAIModel(observer=observer, model_id=model_id, api_key=api_key, api_base=api_base)
    tools = [wrap_tool(n, agent.tool_descriptions[n], fn) for n, fn in agent.tools.items()]
    return CoreAgent(observer=observer, tools=tools, model=model,
                     name=agent.name, max_steps=max_steps)
