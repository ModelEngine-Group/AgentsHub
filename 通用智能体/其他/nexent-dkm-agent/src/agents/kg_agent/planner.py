"""Task 2 KG agent planner -- supports rule-based, LLM, and local model planning."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REGISTERED_KG_OPERATORS = frozenset({
    "extract_medical_entities",
    "extract_relations",
    "validate_triples",
    "build_medical_graph",
    "answer_graph_question",
    "build_kg_quality_report",
    "find_graph_entities",
    "query_graph_neighbors",
})


@dataclass(frozen=True)
class KGTaskUnderstanding:
    original_request: str
    task_type: str  # "kg_build", "kg_query", "kg_qa", "full_pipeline"
    data_type: str  # "text", "unknown"
    intent_keywords: list[str]
    constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KGTaskPlan:
    understanding: KGTaskUnderstanding
    operators: list[str]
    rationale: list[str]
    question: str | None = None
    confidence: float = 0.8
    planner_mode: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["understanding"] = self.understanding.to_dict()
        return d


_INTENT_RULES = [
    ("build", ("构建", "生成", "建立", "抽取", "build", "construct")),
    ("query", ("查询", "查找", "搜索", "query", "find", "search")),
    ("qa", ("问答", "回答", "哪些", "什么", "怎么", "question", "answer")),
    ("validate", ("验证", "校验", "检查", "validate", "check")),
    ("export", ("导出", "输出", "保存", "export", "save")),
]


class KGHybridPlanner:
    """Plan KG tasks using local_model > LLM > rule-based fallback."""

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
    ) -> KGTaskPlan:
        if self.local_model_path:
            try:
                result = self._local_model_plan(request, question)
                if result is not None:
                    return result
            except Exception:
                logger.warning("Local model planning failed, falling back.", exc_info=True)

        if self.llm_config:
            try:
                return self._llm_plan(request, question)
            except Exception:
                logger.warning("LLM planning failed, falling back to rules.", exc_info=True)

        return plan_kg_task(request, question=question)

    def _llm_plan(
        self,
        request: str | None,
        question: str | None,
    ) -> KGTaskPlan:
        from src.agents.data_processing_agent.llm_orchestrator import request_plan

        task_request = request or "构建医疗知识图谱并回答问题"
        if question:
            task_request += f" 问题：{question}"

        llm_result = request_plan(
            base_url=self.llm_config["base_url"],
            api_key=self.llm_config["api_key"],
            model_name=self.llm_config.get("model_name", "glm-5.1"),
            task_request=task_request,
            available_operators=list(REGISTERED_KG_OPERATORS),
            timeout=self.llm_config.get("timeout", 30.0),
            llm_config=self.llm_config,
        )

        operators = [op for op in llm_result.get("operators", []) if op in REGISTERED_KG_OPERATORS]
        if len(operators) < 2:
            raise ValueError("LLM returned too few valid operators; falling back.")

        original_request = (request or "").strip()
        understanding = KGTaskUnderstanding(
            original_request=original_request,
            task_type=llm_result.get("task_type", "full_pipeline"),
            data_type="text",
            intent_keywords=llm_result.get("intent_keywords", []),
            constraints=["no_secret_persistence"],
        )

        return KGTaskPlan(
            understanding=understanding,
            operators=operators,
            rationale=llm_result.get("rationale", ["LLM-generated plan."]),
            question=question,
            confidence=min(llm_result.get("confidence", 0.8), 0.99),
            planner_mode="llm",
        )

    def _local_model_plan(
        self,
        request: str | None,
        question: str | None,
    ) -> KGTaskPlan | None:
        try:
            from src.agents.data_processing_agent.local_model_planner import predict_plan
        except ImportError:
            return None

        result = predict_plan(
            model_path=self.local_model_path,
            task_request=request or "构建医疗知识图谱",
        )
        if result is None:
            return None

        operators = [op for op in result.get("operators", []) if op in REGISTERED_KG_OPERATORS]
        if len(operators) < 2:
            return None

        return KGTaskPlan(
            understanding=KGTaskUnderstanding(
                original_request=(request or "").strip(),
                task_type=result.get("task_type", "full_pipeline"),
                data_type="text",
                intent_keywords=result.get("intent_keywords", []),
            ),
            operators=operators,
            rationale=result.get("rationale", ["Local model plan."]),
            question=question,
            confidence=min(result.get("confidence", 0.7), 0.95),
            planner_mode="local_model",
        )


def plan_kg_task(
    request: str | None = None,
    question: str | None = None,
    text_length: int | None = None,
    input_text: str | None = None,
) -> KGTaskPlan:
    """Rule-based KG task planner."""

    original_request = (request or "").strip()
    normalized = original_request.lower()
    corpus_text = input_text or original_request
    small_corpus = text_length is not None and text_length < 500

    intent_keywords = [
        intent
        for intent, triggers in _INTENT_RULES
        if any(trigger in normalized for trigger in triggers)
    ]

    if question and any(token in (question or "") for token in ("哪些", "什么", "怎么", "?")):
        if "qa" not in intent_keywords:
            intent_keywords.append("qa")

    # If a question is provided, always include QA
    if question and "qa" not in intent_keywords:
        intent_keywords.append("qa")

    if any(token in corpus_text for token in ("并发", "并发症", "合并")):
        if "build" not in intent_keywords:
            intent_keywords.append("build")

    # Determine task type
    if question and small_corpus and "build" not in intent_keywords:
        task_type = "kg_qa"
    elif "query" in intent_keywords and "build" not in intent_keywords:
        task_type = "kg_query"
    elif "qa" in intent_keywords and "build" not in intent_keywords:
        task_type = "kg_qa"
    else:
        task_type = "full_pipeline"

    # Build operator sequence
    operators = []
    rationale = []

    if task_type == "full_pipeline":
        operators = [
            "extract_medical_entities",
            "extract_relations",
            "validate_triples",
            "build_medical_graph",
            "answer_graph_question",
        ]
        rationale = [
            "Extract medical entities from text.",
            "Generate relation triples between entities.",
            "Validate triples against KG schema.",
            "Build deduplicated knowledge graph.",
            "Answer user question using graph evidence.",
        ]
        if any(token in corpus_text for token in ("并发", "并发症", "合并")):
            rationale.append("Prioritize complication_of relation extraction for comorbidity cues.")
        if not small_corpus:
            operators.append("build_kg_quality_report")
            rationale.append("Generate quality report with readiness metrics.")
        else:
            rationale.append("Corpus too small for community discovery; skipped quality/community analysis.")
    elif task_type == "kg_qa":
        if small_corpus:
            operators = [
                "find_graph_entities",
                "answer_graph_question",
            ]
            rationale = [
                "Question-first plan for a small corpus: locate entities before rebuilding the graph.",
                "Answer question from graph evidence.",
            ]
        else:
            operators = [
                "extract_medical_entities",
                "extract_relations",
                "validate_triples",
                "build_medical_graph",
                "answer_graph_question",
            ]
            rationale = [
                "Extract entities needed for graph construction.",
                "Build relations and graph for QA.",
                "Answer question from graph evidence.",
            ]
    elif task_type == "kg_query":
        operators = [
            "find_graph_entities",
            "query_graph_neighbors",
        ]
        rationale = [
            "Search graph for matching entities.",
            "Retrieve neighbor relationships.",
        ]

    if "validate" in intent_keywords and "validate_triples" not in operators:
        operators.append("validate_triples")
        rationale.append("Validate triples against schema.")

    # A question must always be answerable, even for query-typed requests.
    # Otherwise a request like "查询X" + question="..." would silently skip QA.
    if question and "answer_graph_question" not in operators:
        operators.append("answer_graph_question")
        rationale.append("Answer user question using graph evidence.")

    understanding = KGTaskUnderstanding(
        original_request=original_request,
        task_type=task_type,
        data_type="text",
        intent_keywords=intent_keywords,
    )

    confidence = 0.5 + min(len(intent_keywords), 4) * 0.1
    if question:
        confidence += 0.1

    return KGTaskPlan(
        understanding=understanding,
        operators=operators,
        rationale=rationale,
        question=question,
        confidence=round(min(confidence, 0.95), 2),
        planner_mode="rule",
    )
