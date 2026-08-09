"""任务一：数据处理智能体（Data Processing Agent）。

能力：
1. 任务理解：从自然语言任务描述中识别需要的清洗/抽取/转换步骤；
2. 多算子组合与调度：将 DataMate 算子编排为流水线并调用 DataMateClient 执行；
3. 状态跟踪与异常处理：每步计时、记录输入输出规模、失败自动重试与降级。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseAgent, Step
from ..datamate.client import DataMateClient, OperatorInstance

# 任务关键词 -> DataMate 算子
KEYWORD_OPS: list[tuple[list[str], str, dict[str, Any]]] = [
    (["html", "标签", "网页"], "HtmlTagCleaner", {}),
    (["繁体", "简体", "繁简"], "TraditionalChineseCleaner", {}),
    (["空白", "空格", "格式", "归一", "规范"], "WhitespaceNormalizer", {}),
    (["去重", "重复"], "DedupFilter", {}),
    (["短文本", "过短", "长度"], "MinLengthFilter", {"min_len": 10}),
    (["脱敏", "隐私", "手机号", "匿名"], "PhoneAnonymizer", {}),
    (["身份证"], "IDCardAnonymizer", {}),
]

DEFAULT_ETL = ["HtmlTagCleaner", "TraditionalChineseCleaner", "WhitespaceNormalizer",
               "PhoneAnonymizer", "IDCardAnonymizer", "DedupFilter", "MinLengthFilter"]


class DataProcessingAgent(BaseAgent):
    """数据处理智能体：理解任务 → 编排 DataMate 算子 → 执行并跟踪状态。"""

    name = "data_processing_agent"

    def __init__(self, client: DataMateClient | None = None) -> None:
        super().__init__()
        self.client = client or DataMateClient()
        for op in self.client.list_operators():
            self.register_tool(op, self._make_tool(op), f"DataMate 文本算子 {op}")
        self.register_tool("load_jsonl", self._load_jsonl, "从 JSONL 文件加载文本记录（text 字段）")
        self.register_tool("save_jsonl", self._save_jsonl, "将处理结果保存为 JSONL")

    # ---- 工具 -----------------------------------------------------------

    def _make_tool(self, op_name: str):
        def tool(state: list[str], **params: Any) -> tuple[list[str], str]:
            res = self.client.run_cleaning(state or [], [OperatorInstance(op_name, params)])
            out = res["output"]
            return out, f"{op_name}[{res['backend']}]: {len(state or [])} -> {len(out)} 条"

        return tool

    def _load_jsonl(self, state: Any, path: str = "", field: str = "text") -> tuple[list[str], str]:
        lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
        records = [json.loads(ln)[field] for ln in lines if ln.strip()]
        return records, f"加载 {len(records)} 条记录 <- {path}"

    def _save_jsonl(self, state: list[str], path: str = "") -> tuple[list[str], str]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in state:
                f.write(json.dumps({"text": r}, ensure_ascii=False) + "\n")
        return state, f"保存 {len(state)} 条记录 -> {path}"

    # ---- 规则规划器 ------------------------------------------------------

    def rule_plan(self, task: str) -> list[Step]:
        task = task.lower()
        ops: list[Step] = []
        for keywords, op, params in KEYWORD_OPS:
            if any(k in task for k in keywords):
                ops.append(Step(tool=op, params=params, reason=f"任务提及 {keywords[0]}"))
        if not ops:  # 未识别到具体算子 -> 标准 ETL 流水线
            ops = [Step(tool=o, reason="标准文本 ETL 流水线") for o in DEFAULT_ETL]
        return ops

    # ---- 高层接口 --------------------------------------------------------

    def process_file(self, task: str, input_path: str, output_path: str):
        """端到端：加载 -> 规划的清洗流水线 -> 保存。"""
        plan, planner = self.make_plan(task)
        full_plan = ([Step(tool="load_jsonl", params={"path": input_path}, reason="加载输入")]
                     + plan
                     + [Step(tool="save_jsonl", params={"path": output_path}, reason="保存输出")])
        run = self._run_with_plan(task, full_plan, planner)
        return run

    def _run_with_plan(self, task: str, plan: list[Step], planner: str):
        from .base import AgentRun

        run = AgentRun(task=task, plan=plan, planner=planner)
        state: Any = None
        for step in plan:
            record = self._execute_step(step, state)
            run.records.append(record)
            if record.status in ("ok", "retried"):
                state = record.summary_state
            else:
                break
        run.result = state
        return run
