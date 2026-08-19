"""Task planners for the data processing agent."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol


CSV_STEPS = [
    {"step": "read_csv", "operator": "ReadCsvOperator", "goal": "读取结构化医疗记录"},
    {"step": "clean_missing", "operator": "CleanMissingOperator", "goal": "删除空值记录"},
    {"step": "normalize_text", "operator": "NormalizeTextOperator", "goal": "规范化文本字段"},
    {"step": "extract_fields", "operator": "ExtractFieldOperator", "goal": "抽取核心字段"},
    {"step": "save_csv", "operator": "SaveCsvOperator", "goal": "保存清洗后的结构化数据"},
]

TEXT_STEPS = [
    {"step": "read_text", "operator": "ReadTextOperator", "goal": "读取非结构化医疗文本"},
    {"step": "split_paragraph", "operator": "SplitParagraphOperator", "goal": "按疾病章节切分文本"},
    {"step": "save_text_json", "operator": "SaveJsonOperator", "goal": "保存清洗后的文本结构"},
]

KNOWN_STEPS = {step["step"] for step in [*CSV_STEPS, *TEXT_STEPS]}


class TaskPlanner(Protocol):
    """Protocol for natural-language task planners."""

    def plan(self, task_description: str) -> list[dict[str, str]]:
        """Create ordered processing steps from a task description."""
        ...


class RuleBasedTaskPlanner:
    """Small deterministic planner for the demo task descriptions."""

    def plan(self, task_description: str) -> list[dict[str, str]]:
        normalized = task_description.strip().lower()
        if not normalized:
            raise ValueError("任务描述不能为空")

        wants_csv = any(keyword in normalized for keyword in ["csv", "结构化", "表格", "记录"])
        wants_text = any(keyword in normalized for keyword in ["文本", "段落", "非结构化", "text"])

        if not wants_csv and not wants_text:
            wants_csv = True
            wants_text = True

        steps: list[dict[str, str]] = []
        if wants_csv:
            steps.extend(dict(step) for step in CSV_STEPS)
        if wants_text:
            steps.extend(dict(step) for step in TEXT_STEPS)
        return steps


class LLMTaskPlanner:
    """LLM-backed task planner with deterministic rule-based fallback."""

    def __init__(
        self,
        llm: Callable[[str], str],
        fallback_planner: TaskPlanner | None = None,
    ) -> None:
        self.llm = llm
        self.fallback_planner = fallback_planner or RuleBasedTaskPlanner()

    def plan(self, task_description: str) -> list[dict[str, str]]:
        normalized = task_description.strip()
        if not normalized:
            raise ValueError("任务描述不能为空")

        prompt = self._build_prompt(normalized)
        try:
            payload = self._parse_llm_payload(self.llm(prompt))
            steps = payload["steps"]
            return self._validate_steps(steps)
        except Exception:
            return self.fallback_planner.plan(task_description)

    def _parse_llm_payload(self, raw_response: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        stripped = raw_response.strip()
        for index, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "steps" in payload:
                self._validate_steps(payload["steps"])
                return payload
        return json.loads(self._extract_json_object(raw_response))

    def _extract_json_object(self, raw_response: str) -> str:
        stripped = raw_response.strip()
        if stripped.startswith("{"):
            return stripped

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM response does not contain a JSON object")
        return stripped[start : end + 1]

    def _build_prompt(self, task_description: str) -> str:
        allowed_steps = ", ".join(sorted(KNOWN_STEPS))
        return (
            "你是医疗数据处理任务规划器。"
            "请只返回 JSON，格式为 {\"steps\": [{\"step\": string, \"operator\": string, \"goal\": string}]}。"
            f"允许的 step 只能是：{allowed_steps}。"
            "如果用户只要求处理文本，或明确说不要处理 CSV，steps 必须且只能是 "
            "read_text, split_paragraph, save_text_json。"
            "如果用户只要求处理 CSV/结构化数据，steps 必须且只能是 "
            "read_csv, clean_missing, normalize_text, extract_fields, save_csv。"
            "如果用户同时要求 CSV 和文本，先输出 CSV steps，再输出文本 steps。"
            "不要输出用户明确排除的数据类型对应的 step。"
            f"用户任务：{task_description}"
        )

    def _validate_steps(self, steps: Any) -> list[dict[str, str]]:
        if not isinstance(steps, list) or not steps:
            raise ValueError("LLM plan must contain non-empty steps")

        validated: list[dict[str, str]] = []
        for item in steps:
            if not isinstance(item, dict):
                raise ValueError("LLM plan step must be an object")
            step = str(item.get("step", ""))
            operator = str(item.get("operator", ""))
            goal = str(item.get("goal", ""))
            if step not in KNOWN_STEPS:
                raise ValueError(f"Unknown LLM plan step: {step}")
            if not operator or not goal:
                raise ValueError("LLM plan step requires operator and goal")
            validated.append({"step": step, "operator": operator, "goal": goal})
        return validated
