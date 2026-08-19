"""Task understanding and operator planning for task 1."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TASK = "清洗CSV数据，去重、填补缺失值、规范化字段类型并导出结果"

_INTENT_RULES = [
    ("deduplicate", ("去重", "重复", "dedup", "duplicate")),
    ("fill_missing", ("缺失", "空值", "空白", "missing", "null", "empty")),
    ("normalize_types", ("规范", "标准", "类型", "转换", "normalize", "type")),
    ("export", ("导出", "输出", "保存", "export", "save")),
    ("extract", ("抽取", "提取", "extract", "parse")),
    ("transform", ("变换", "转换列", "transform", "convert")),
    ("clean_text", ("清洗文本", "文本处理", "去除标签", "clean text")),
]

REGISTERED_OPERATORS = frozenset({
    "load_csv",
    "profile_schema",
    "drop_duplicate_rows",
    "drop_column",
    "fill_missing_values",
    "normalize_column_types",
    "export_clean_dataset",
    "validate_clean_dataset",
    "load_text",
    "clean_text",
    "extract_entities",
    "transform_columns",
})


@dataclass(frozen=True)
class TaskUnderstanding:
    """Structured understanding of a user data-processing request."""

    original_request: str
    task_type: str
    data_type: str
    intent_keywords: list[str]
    constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataTaskPlan:
    """Agent plan that can be executed by the task-1 scheduler."""

    understanding: TaskUnderstanding
    operators: list[str]
    rationale: list[str]
    quality_checks: list[str]
    confidence: float
    planner_mode: str = "rule"
    datamate_integration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "understanding": self.understanding.to_dict(),
            "operators": self.operators,
            "rationale": self.rationale,
            "quality_checks": self.quality_checks,
            "confidence": self.confidence,
            "planner_mode": self.planner_mode,
            "datamate_integration": self.datamate_integration,
        }


class HybridPlanner:
    """Plan data tasks using LLM, local model, or rule-based fallback.

    Priority: local_model > llm_config > rule-based.
    """

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        local_model_path: str | None = None,
        datamate_operators: list[str] | None = None,
    ) -> None:
        self.llm_config = llm_config
        self.local_model_path = local_model_path
        self.datamate_operators = datamate_operators or []

    def plan(
        self,
        request: str | None,
        data_profile: dict[str, Any] | None = None,
    ) -> DataTaskPlan:
        if self.local_model_path:
            try:
                result = self._local_model_plan(request, data_profile)
                if result is not None:
                    return result
            except Exception:
                logger.warning("Local model planning failed, falling back.", exc_info=True)

        if self.llm_config:
            try:
                return self._llm_plan(request, data_profile)
            except Exception:
                logger.warning("LLM planning failed, falling back to rules.", exc_info=True)

        plan = plan_data_task(request, data_profile)
        return self.enrich_plan(self.validate_plan_against_profile(plan, data_profile))

    def validate_plan_against_profile(
        self,
        plan: DataTaskPlan,
        data_profile: dict[str, Any] | None,
    ) -> DataTaskPlan:
        """Correct LLM/local plans that contradict the data profile."""

        if not data_profile:
            return plan

        operators = list(plan.operators)
        rationale = list(plan.rationale)
        changed = False

        if data_profile.get("duplicate_rows", 0) == 0 and "drop_duplicate_rows" in operators:
            operators.remove("drop_duplicate_rows")
            rationale.append("Skipped deduplication because the profile reports zero duplicate rows.")
            changed = True

        if _has_missing_values(data_profile) and "fill_missing_values" not in operators:
            if any(count > 0 for count in data_profile.get("missing_cells", {}).values()):
                pass
        elif not _has_missing_values(data_profile) and "fill_missing_values" in operators:
            operators.remove("fill_missing_values")
            rationale.append("Removed fill_missing_values because the profile has no missing cells.")
            changed = True

        high_missing = _high_missing_columns(data_profile)
        if high_missing and "drop_column" not in operators:
            operators.insert(max(operators.index("profile_schema"), 0) + 1, "drop_column")
            rationale.append(
                "Drop columns with >50% missing values: " + ", ".join(high_missing[:4])
                + ("." if len(high_missing) <= 4 else ", ...")
            )
            changed = True

        if not changed:
            return plan

        return DataTaskPlan(
            understanding=plan.understanding,
            operators=operators,
            rationale=rationale,
            quality_checks=plan.quality_checks,
            confidence=plan.confidence,
            planner_mode=plan.planner_mode,
            datamate_integration=plan.datamate_integration,
        )

    def enrich_plan(
        self,
        plan: DataTaskPlan,
        datamate_catalog: dict[str, Any] | None = None,
    ) -> DataTaskPlan:
        """Attach DataMate catalog mappings and rationale to an existing plan."""

        catalog = datamate_catalog or {}
        mappings = catalog.get("candidate_mappings", {})
        if not mappings and not self.datamate_operators:
            return plan

        selected_ids: list[str] = []
        operator_mappings: dict[str, Any] = {}
        for operator in plan.operators:
            mapping = mappings.get(operator, {})
            operator_ids = mapping.get("selected_operator_ids", [])
            if operator_ids:
                operator_mappings[operator] = mapping
                selected_ids.extend(operator_ids)

        if not selected_ids and self.datamate_operators:
            selected_ids = list(self.datamate_operators)

        if not selected_ids:
            return plan

        rationale = list(plan.rationale)
        rationale.append(
            "Mapped local operators to DataMate catalog entries: "
            + ", ".join(sorted(set(selected_ids))[:6])
            + ("." if len(set(selected_ids)) <= 6 else ", ...")
        )
        return DataTaskPlan(
            understanding=plan.understanding,
            operators=plan.operators,
            rationale=rationale,
            quality_checks=plan.quality_checks,
            confidence=min(plan.confidence + 0.05, 0.99),
            planner_mode=plan.planner_mode,
            datamate_integration={
                "status": "mapped",
                "operator_count": catalog.get("operator_count", len(self.datamate_operators)),
                "selected_operator_ids": sorted(set(selected_ids)),
                "operator_mappings": operator_mappings,
            },
        )

    def _llm_plan(
        self,
        request: str | None,
        data_profile: dict[str, Any] | None,
    ) -> DataTaskPlan:
        from src.agents.data_processing_agent.llm_orchestrator import (
            request_plan,
        )

        llm_result = request_plan(
            base_url=self.llm_config["base_url"],
            api_key=self.llm_config["api_key"],
            model_name=self.llm_config.get("model_name", "glm-5.1"),
            task_request=request or DEFAULT_TASK,
            data_profile=data_profile,
            available_operators=list(REGISTERED_OPERATORS),
            timeout=self.llm_config.get("timeout", 30.0),
            llm_config=self.llm_config,
        )

        operators = llm_result.get("operators", [])
        rationale = llm_result.get("rationale", [])
        operators = [op for op in operators if op in REGISTERED_OPERATORS]

        if len(operators) < 2:
            raise ValueError("LLM returned too few valid operators; falling back.")

        original_request = (request or DEFAULT_TASK).strip()
        understanding = TaskUnderstanding(
            original_request=original_request,
            task_type=llm_result.get("task_type", "cleaning"),
            data_type=llm_result.get("data_type", _infer_data_type(original_request, data_profile)),
            intent_keywords=llm_result.get("intent_keywords", []),
            constraints=["no_secret_persistence", "datamate_submit_requires_explicit_mode"],
        )

        confidence = min(llm_result.get("confidence", 0.8), 0.99)

        return self.validate_plan_against_profile(
            self.enrich_plan(
                DataTaskPlan(
                    understanding=understanding,
                    operators=operators,
                    rationale=rationale or ["LLM-generated plan."],
                    quality_checks=[
                        "row_count",
                        "column_count",
                        "missing_cells",
                        "duplicate_rows",
                        "output_rows",
                    ],
                    confidence=confidence,
                    planner_mode="llm",
                )
            ),
            data_profile,
        )

    def _local_model_plan(
        self,
        request: str | None,
        data_profile: dict[str, Any] | None,
    ) -> DataTaskPlan | None:
        try:
            from src.agents.data_processing_agent.local_model_planner import (
                predict_plan,
            )
        except ImportError:
            return None

        result = predict_plan(
            model_path=self.local_model_path,
            task_request=request or DEFAULT_TASK,
            data_profile=data_profile,
        )
        if result is None:
            return None

        operators = [op for op in result.get("operators", []) if op in REGISTERED_OPERATORS]
        if len(operators) < 2:
            return None

        original_request = (request or DEFAULT_TASK).strip()
        understanding = TaskUnderstanding(
            original_request=original_request,
            task_type=result.get("task_type", "cleaning"),
            data_type=result.get("data_type", _infer_data_type(original_request, data_profile)),
            intent_keywords=result.get("intent_keywords", []),
            constraints=["no_secret_persistence", "datamate_submit_requires_explicit_mode"],
        )

        return self.validate_plan_against_profile(
            self.enrich_plan(
                DataTaskPlan(
                    understanding=understanding,
                    operators=operators,
                    rationale=result.get("rationale", ["Local model generated plan."]),
                    quality_checks=["row_count", "column_count", "missing_cells", "duplicate_rows"],
                    confidence=min(result.get("confidence", 0.7), 0.95),
                    planner_mode="local_model",
                )
            ),
            data_profile,
        )


def plan_data_task(
    request: str | None,
    data_profile: dict[str, Any] | None = None,
) -> DataTaskPlan:
    """Convert a free-form task request and optional profile into an operator plan."""

    original_request = (request or DEFAULT_TASK).strip()
    normalized = original_request.lower()
    intent_keywords = [
        intent
        for intent, triggers in _INTENT_RULES
        if any(trigger in normalized for trigger in triggers)
    ]

    if data_profile:
        if data_profile.get("duplicate_rows", 0) > 0 and "deduplicate" not in intent_keywords:
            intent_keywords.append("deduplicate")
        if _has_missing_values(data_profile) and "fill_missing" not in intent_keywords:
            intent_keywords.append("fill_missing")
        if _has_non_text_columns(data_profile) and "normalize_types" not in intent_keywords:
            intent_keywords.append("normalize_types")
        if _all_text_columns(data_profile) and "extract" not in intent_keywords:
            intent_keywords.append("extract")

    if "export" not in intent_keywords:
        intent_keywords.append("export")

    task_type = "cleaning" if any(
        intent in intent_keywords
        for intent in ("deduplicate", "fill_missing", "normalize_types", "clean_text", "extract", "transform")
    ) else "profiling"
    data_type = _infer_data_type(original_request, data_profile)

    operators = []
    rationale = []

    if data_type == "text":
        operators = ["load_text", "clean_text"]
        rationale = ["Load and clean unstructured text input."]
    else:
        operators = ["load_csv", "profile_schema"]
        rationale = ["Understand the request and profile the input data."]

    if "deduplicate" in intent_keywords:
        if not data_profile or data_profile.get("duplicate_rows", 0) > 0:
            operators.append("drop_duplicate_rows")
            rationale.append("Remove duplicate records detected by the profile or request.")
    if "fill_missing" in intent_keywords:
        operators.append("fill_missing_values")
        rationale.append("Fill missing values with deterministic type-aware defaults.")
    high_missing = _high_missing_columns(data_profile) if data_profile else []
    if high_missing:
        operators.append("drop_column")
        rationale.append(
            "Drop columns with >50% missing values before imputation: "
            + ", ".join(high_missing[:4])
            + ("." if len(high_missing) <= 4 else ", ...")
        )
    if "normalize_types" in intent_keywords:
        operators.append("normalize_column_types")
        rationale.append("Normalize integer, float, boolean, and text representations.")
    if "clean_text" in intent_keywords and data_type != "text":
        if "load_csv" in operators:
            operators.remove("load_csv")
        if "profile_schema" in operators:
            operators.remove("profile_schema")
        operators = ["load_text", "clean_text"] + operators
        rationale.append("Clean unstructured text: remove HTML, normalize Unicode, redact PII.")
    if "extract" in intent_keywords:
        operators.append("extract_entities")
        rationale.append("Extract structured entities from text data.")
    if "transform" in intent_keywords:
        operators.append("transform_columns")
        rationale.append("Apply column-level transforms: select, rename, filter.")
    operators.extend(["export_clean_dataset", "validate_clean_dataset"])

    confidence = _estimate_confidence(intent_keywords, data_profile)
    understanding = TaskUnderstanding(
        original_request=original_request,
        task_type=task_type,
        data_type=data_type,
        intent_keywords=intent_keywords,
        constraints=["no_secret_persistence", "datamate_submit_requires_explicit_mode"],
    )
    return DataTaskPlan(
        understanding=understanding,
        operators=operators,
        rationale=rationale,
        quality_checks=[
            "row_count",
            "column_count",
            "missing_cells",
            "duplicate_rows",
            "output_rows",
            "post_clean_missing_cells",
            "post_clean_duplicate_rows",
        ],
        confidence=confidence,
        planner_mode="rule",
    )


def _infer_data_type(
    request: str,
    data_profile: dict[str, Any] | None,
) -> str:
    if data_profile:
        fname = str(data_profile.get("file_name", "")).lower()
        if fname.endswith(".csv"):
            return "structured_csv"
        if fname.endswith(".txt") or fname.endswith(".text"):
            return "text"
        if fname.endswith(".json"):
            return "json"
    lowered = request.lower()
    if "csv" in lowered or "表" in request or "结构化" in request:
        return "structured_csv"
    if "文本" in request or "text" in lowered:
        return "text"
    return "unknown"


def _has_missing_values(data_profile: dict[str, Any]) -> bool:
    return any(count > 0 for count in data_profile.get("missing_cells", {}).values())


def _has_non_text_columns(data_profile: dict[str, Any]) -> bool:
    return any(
        column.get("inferred_type") != "text"
        for column in data_profile.get("columns", [])
    )


def _all_text_columns(data_profile: dict[str, Any]) -> bool:
    columns = data_profile.get("columns", [])
    return bool(columns) and all(column.get("inferred_type") == "text" for column in columns)


def _high_missing_columns(data_profile: dict[str, Any]) -> list[str]:
    row_count = data_profile.get("row_count", 0) or 0
    if row_count <= 0:
        return []
    threshold = row_count * 0.5
    return sorted(
        column
        for column, missing in data_profile.get("missing_cells", {}).items()
        if missing > threshold
    )


def _estimate_confidence(
    intent_keywords: list[str],
    data_profile: dict[str, Any] | None,
) -> float:
    base = 0.45
    base += min(len(intent_keywords), 4) * 0.1
    if data_profile:
        base += 0.25
    return round(min(base, 0.95), 2)
