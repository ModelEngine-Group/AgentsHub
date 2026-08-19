"""智能体核心：任务理解 → 规划 → 执行 → 状态跟踪。

规划器三级后端（自动降级，保证可复现）：
1. Nexent CoreAgent（安装 nexent 且配置模型时）：本仓库工具注册为
   Nexent Tool，由 CoreAgent 的 ReAct 循环驱动；
2. OpenAI 兼容 LLM 规划（设置 MGI_LLM_API_KEY / MGI_LLM_BASE_URL / MGI_LLM_MODEL）；
3. 规则规划器（零依赖、确定性）：基于任务关键词与算子目录做规划，
   用于 CI 与无外部服务环境下的可复现 Demo。
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Step:
    """计划中的一步：调用某个工具/算子。"""

    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class StepRecord:
    step: Step
    status: str  # ok / failed / retried / skipped
    elapsed_ms: float = 0.0
    error: str = ""
    summary: str = ""
    summary_state: Any = None


@dataclass
class AgentRun:
    task: str
    plan: list[Step] = field(default_factory=list)
    records: list[StepRecord] = field(default_factory=list)
    result: Any = None
    planner: str = "rule"

    def report(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "planner": self.planner,
            "plan": [{"tool": s.tool, "params": s.params, "reason": s.reason} for s in self.plan],
            "steps": [
                {"tool": r.step.tool, "status": r.status,
                 "elapsed_ms": round(r.elapsed_ms, 2), "error": r.error, "summary": r.summary}
                for r in self.records
            ],
        }


class LLMPlanner:
    """OpenAI 兼容接口的 LLM 规划器。"""

    def __init__(self) -> None:
        self.api_key = os.getenv("MGI_LLM_API_KEY", "")
        self.base_url = os.getenv("MGI_LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("MGI_LLM_MODEL", "gpt-4o-mini")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def plan(self, task: str, tool_catalog: list[dict[str, Any]]) -> list[Step]:
        import requests

        prompt = (
            "你是数据处理编排智能体。根据任务与可用工具目录，输出 JSON 数组形式的执行计划，"
            '每个元素形如 {"tool": 工具名, "params": {…}, "reason": 一句话理由}。'
            "只输出 JSON。\n\n任务：" + task + "\n\n工具目录：" + json.dumps(tool_catalog, ensure_ascii=False)
        )
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0},
            timeout=60,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        return [Step(tool=s["tool"], params=s.get("params", {}), reason=s.get("reason", ""))
                for s in json.loads(content)]


class BaseAgent:
    """规划-执行智能体基类。

    子类提供：
    - tools: 工具名 -> callable(state, **params) -> (state, summary)
    - rule_plan(task): 规则规划器
    """

    name = "base_agent"
    max_retries = 1

    def __init__(self) -> None:
        self.tools: dict[str, Callable[..., tuple[Any, str]]] = {}
        self.tool_descriptions: dict[str, str] = {}
        self.llm = LLMPlanner()

    def register_tool(self, name: str, fn: Callable[..., tuple[Any, str]], description: str) -> None:
        self.tools[name] = fn
        self.tool_descriptions[name] = description

    def catalog(self) -> list[dict[str, Any]]:
        return [{"name": n, "description": d} for n, d in self.tool_descriptions.items()]

    # ---- 规划 -----------------------------------------------------------

    def rule_plan(self, task: str) -> list[Step]:  # pragma: no cover - overridden
        raise NotImplementedError

    def make_plan(self, task: str) -> tuple[list[Step], str]:
        if self.llm.available:
            try:
                plan = self.llm.plan(task, self.catalog())
                valid = [s for s in plan if s.tool in self.tools]
                if valid:
                    return valid, "llm"
            except Exception:
                pass  # 降级到规则规划器
        return self.rule_plan(task), "rule"

    # ---- 执行 -----------------------------------------------------------

    def run(self, task: str, state: Any = None) -> AgentRun:
        plan, planner = self.make_plan(task)
        run = AgentRun(task=task, plan=plan, planner=planner)
        for step in plan:
            record = self._execute_step(step, state)
            run.records.append(record)
            if record.status in ("ok", "retried"):
                state = record.summary_state
            elif record.status == "failed":
                break
        run.result = state
        return run

    def _execute_step(self, step: Step, state: Any) -> StepRecord:
        attempts = 0
        while True:
            attempts += 1
            start = time.perf_counter()
            try:
                new_state, summary = self.tools[step.tool](state, **step.params)
                return StepRecord(step=step, status="ok" if attempts == 1 else "retried",
                                  elapsed_ms=(time.perf_counter() - start) * 1000,
                                  summary=summary, summary_state=new_state)
            except Exception:
                if attempts > self.max_retries:
                    return StepRecord(step=step, status="failed",
                                      elapsed_ms=(time.perf_counter() - start) * 1000,
                                      error=traceback.format_exc(limit=3), summary_state=state)
