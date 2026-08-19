"""Task-3 hybrid analysis planner.

Supports rule-based, LLM (via openai), and local model planning with
graceful fallback.  The LLM path uses the ``openai`` library directly,
consistent with the NL2SQL module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.common.llm_config import openai_extra_kwargs

logger = logging.getLogger(__name__)

# Reasoning-capable models (e.g. glm-5.1) spend part of the token budget on
# internal reasoning before emitting the answer; a small cap can be fully
# consumed by reasoning, yielding empty content (finish_reason="length").
# Default high and allow per-request override via llm_config["max_tokens"].
_DEFAULT_LLM_MAX_TOKENS = 4096

REGISTERED_ANALYSIS_OPERATORS = [
    "load_graph",
    "generate_statistical_summary",
    "generate_association_analysis",
    "generate_trend_analysis",
    "compute_centrality",
    "compute_shortest_paths",
    "detect_communities",
    "translate_question_to_sql",
    "execute_sql",
    "build_analysis_visualizations",
    "build_analysis_report",
]

# Intent -> trigger keywords used by the rule-based planner. The detected
# intents drive which optional operators the agent executes (see
# GraphAnalysisAgent.run), so the plan is not merely informational.
_ANALYSIS_INTENT_TRIGGERS = (
    ("statistics", ("统计", "分布", "count", "stat")),
    ("association", ("关联", "关系", "association", "relation")),
    ("trend", ("趋势", "trend")),
    ("graph_analytics", (
        "中心性", "关键节点", "重要节点", "核心节点", "枢纽", "社区", "社群",
        "聚类", "路径", "最短路径", "连通", "centrality", "community", "path", "hub",
    )),
    ("visualization", ("可视化", "图表", "chart", "visual")),
    ("nl2sql", ("sql", "查询", "哪些", "question")),
)

_AUTO_GRAPH_ANALYTICS_TRIGGERS = (
    "分析", "洞察", "可视化", "全面", "完整", "社区", "中心性", "枢纽",
    "graph", "community", "centrality", "hub",
)


def _should_auto_enable_graph_analytics(
    task_request: str | None,
    graph_context: dict[str, Any],
) -> bool:
    """Enable extended graph analytics only for explicit analysis requests."""

    if not task_request:
        return False
    if not (graph_context.get("is_large_graph") or graph_context.get("has_rich_disease_links")):
        return False
    normalized = task_request.lower()
    return any(
        trigger in normalized or trigger in task_request
        for trigger in _AUTO_GRAPH_ANALYTICS_TRIGGERS
    )


_PLANNING_SYSTEM_PROMPT = (
    "You are an analysis planning assistant. Given a task request and a list "
    "of available operators, decide which operators to run and in what order. "
    "Return a JSON object with keys: "
    '"task_type" (string), "operators" (list of operator names), '
    '"intent_keywords" (list of detected intent keywords), '
    '"confidence" (float 0-1). '
    "Only use operators from the available list. "
    "Output ONLY the JSON object, no markdown fences."
)


def plan_analysis_task(
    task_request: str | None = None,
    question: str | None = None,
    graph_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rule-based analysis planner.

    Detects requested analysis intents from the task request. When no intent is
    detected, defaults to the core analysis set (statistics/association/trend/
    visualization/nl2sql) but leaves extended graph analytics opt-in.
    Graph summaries from task-2 artifacts can auto-enable graph analytics.
    """
    normalized = (task_request or "").lower()
    intents = []
    for name, triggers in _ANALYSIS_INTENT_TRIGGERS:
        if any(trigger in normalized for trigger in triggers):
            intents.append(name)
    if question and "nl2sql" not in intents:
        intents.append("nl2sql")
    if not intents:
        intents = ["statistics", "association", "trend", "visualization", "nl2sql"]

    graph_context = graph_summary or {}
    if _should_auto_enable_graph_analytics(task_request, graph_context):
        if "graph_analytics" not in intents:
            intents.append("graph_analytics")

    confidence = 0.75 + min(len(intents), 4) * 0.05
    if graph_context:
        confidence = min(confidence + 0.05, 0.95)

    return {
        "task_type": "full_analysis",
        "planner_mode": "rule",
        "intent_keywords": intents,
        "operators": list(REGISTERED_ANALYSIS_OPERATORS),
        "question": question,
        "confidence": confidence,
        "graph_context": graph_context,
    }


class AnalysisHybridPlanner:
    """Plan analysis tasks using LLM (openai) > rule-based fallback."""

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        local_model_path: str | None = None,
    ) -> None:
        self.llm_config = llm_config
        if local_model_path and Path(local_model_path).is_dir():
            self.local_model_path = local_model_path
        else:
            if local_model_path:
                logger.info(
                    "local_model_path '%s' not found; local model planning disabled.",
                    local_model_path,
                )
            self.local_model_path = None

    def plan(
        self,
        request: str | None = None,
        question: str | None = None,
        graph_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Plan analysis task with local model > LLM > rule-based fallback."""
        if self.local_model_path:
            try:
                local_plan = self._local_model_plan(request, question)
                if local_plan is not None:
                    return local_plan
            except Exception:
                logger.warning("Local model planning failed, falling back.", exc_info=True)

        if self.llm_config:
            try:
                return self._llm_plan(request, question)
            except Exception:
                logger.warning("LLM planning failed, falling back to rules.", exc_info=True)

        return plan_analysis_task(request, question=question, graph_summary=graph_summary)

    def _local_model_plan(
        self,
        request: str | None,
        question: str | None,
    ) -> dict[str, Any] | None:
        """Plan using the task-3 fine-tuned planner, validated for analysis ops.

        Uses ``analysis_ops.local_model_planning.predict_plan``, whose prompt
        matches the analysis fine-tuning format, and keeps only operators that
        are registered for analysis. Returns None to signal a clean fallback.
        """

        from src.operators.analysis_ops.local_model_planning import predict_plan

        task_request = request or "分析医疗知识图谱数据"
        if question:
            task_request += f" 问题：{question}"

        result = predict_plan(model_path=self.local_model_path, request=task_request)
        if not result:
            return None

        operators = [
            op for op in result.get("operators", [])
            if op in REGISTERED_ANALYSIS_OPERATORS
        ]
        if len(operators) < 2:
            return None

        return {
            "task_type": result.get("task_type", "full_analysis"),
            "planner_mode": "local_model",
            "intent_keywords": result.get("intent_keywords", []),
            "operators": operators,
            "question": question,
            "confidence": min(result.get("confidence", 0.8), 0.99),
        }

    def _llm_plan(
        self,
        request: str | None,
        question: str | None,
    ) -> dict[str, Any]:
        """Use LLM (via openai) to generate an analysis plan."""
        import openai

        task_request = request or "分析医疗知识图谱数据"
        if question:
            task_request += f" 问题：{question}"

        user_content = json.dumps({
            "task_request": task_request,
            "available_operators": list(REGISTERED_ANALYSIS_OPERATORS),
        }, ensure_ascii=False)

        client = openai.OpenAI(
            base_url=self.llm_config["base_url"],
            api_key=self.llm_config["api_key"],
        )
        response = client.chat.completions.create(
            model=self.llm_config.get("model_name", "glm-5.1"),
            messages=[
                {"role": "system", "content": _PLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=self.llm_config.get("max_tokens", _DEFAULT_LLM_MAX_TOKENS),
            timeout=self.llm_config.get("timeout", 60.0),
            **openai_extra_kwargs(self.llm_config),
        )

        raw_text = (response.choices[0].message.content or "").strip()
        llm_result = _parse_plan_json(raw_text)

        operators = [
            op for op in llm_result.get("operators", [])
            if op in REGISTERED_ANALYSIS_OPERATORS
        ]
        if len(operators) < 2:
            raise ValueError("LLM returned too few valid operators; falling back.")

        return {
            "task_type": llm_result.get("task_type", "full_analysis"),
            "planner_mode": "llm",
            "intent_keywords": llm_result.get("intent_keywords", []),
            "operators": operators,
            "question": question,
            "confidence": min(llm_result.get("confidence", 0.8), 0.99),
        }


def _parse_plan_json(text: str) -> dict[str, Any]:
    """Parse LLM plan JSON, handling markdown fences."""
    import re

    cleaned = text.strip()
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {"operators": []}
