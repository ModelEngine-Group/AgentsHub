"""Data processing agent for task 1.

The agent is intentionally small and deterministic for the MVP:
- understand a task description by mapping it to a fixed processing plan
- execute existing operators in order
- track each step status
- write reproducible outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

from operators import (
    CleanMissingOperator,
    ExtractFieldOperator,
    NormalizeTextOperator,
    ReadCsvOperator,
    ReadTextOperator,
    SaveCsvOperator,
    SaveJsonOperator,
    SplitParagraphOperator,
)
from planners import RuleBasedTaskPlanner, TaskPlanner


CORE_FIELDS = [
    "patient_id",
    "age",
    "gender",
    "chief_complaint",
    "diagnosis",
    "medicine",
    "department",
]


@dataclass(frozen=True)
class StepExecutionPolicy:
    """Execution policy for one agent step."""

    retries: int = 0
    required: bool = True


class DataProcessingAgent:
    """Task 1 data processing agent.

    The class is the MVP implementation of an agent-like controller. It receives
    a user task description, converts it into an ordered plan, calls operators,
    tracks execution states, and returns output locations.
    """

    def __init__(
        self,
        data_dir: Path | str,
        outputs_dir: Path | str,
        planner: TaskPlanner | None = None,
        skip_optional_text_errors: bool = False,
        step_policies: dict[str, StepExecutionPolicy] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.outputs_dir = Path(outputs_dir)
        self.planner = planner or RuleBasedTaskPlanner()
        self.skip_optional_text_errors = skip_optional_text_errors
        self.step_policies = step_policies or {}
        self.medical_records_path = self.data_dir / "medical_records.csv"
        self.medical_text_path = self.data_dir / "medical_text.txt"
        self.cleaned_records_path = self.outputs_dir / "cleaned_records.csv"
        self.cleaned_text_path = self.outputs_dir / "cleaned_text.json"
        self.summary_path = self.outputs_dir / "task1_summary.json"
        self.steps: list[dict[str, Any]] = []
        self.errors: list[dict[str, str]] = []

    def plan(self, task_description: str) -> list[dict[str, str]]:
        """Create an ordered operator plan from the task description."""
        return self.planner.plan(task_description)

    def run(self, task_description: str) -> dict[str, Any]:
        """Execute the data processing plan."""
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.steps = []
        self.errors = []
        plan = self.plan(task_description)

        structured_result: dict[str, Any] | None = None
        text_result: dict[str, Any] | None = None
        try:
            structured_result = self._process_structured_records()
            text_result = self._process_text_records(required=not self.skip_optional_text_errors)
        except Exception as exc:
            failed_step = self.steps[-1]["step"] if self.steps else "unknown"
            if not self.errors:
                self.errors.append({"step": failed_step, "error": str(exc)})
            if not self.steps or self.steps[-1]["status"] != "failed":
                self._record(failed_step, "failed", str(exc))
            summary = {
                "task": "task1_data_processing_agent",
                "task_description": task_description,
                "status": "failed",
                "plan": plan,
                "structured_result": structured_result,
                "text_result": text_result,
                "execution_state": {"steps": self.steps, "errors": self.errors},
            }
            SaveJsonOperator().execute(summary, str(self.summary_path))
            return summary

        status = "partial_success" if self.errors else "success"

        summary = {
            "task": "task1_data_processing_agent",
            "task_description": task_description,
            "status": status,
            "plan": plan,
            "structured_result": structured_result,
            "text_result": text_result,
            "execution_state": {"steps": self.steps, "errors": self.errors},
        }
        SaveJsonOperator().execute(summary, str(self.summary_path))
        return summary

    def _record(self, step: str, status: str, detail: str) -> None:
        self.steps.append({"step": step, "status": status, "detail": detail})
        print(f"[agent] {step}: {status} - {detail}")

    def _execute_step(
        self,
        step: str,
        operation: Callable[[], Any],
        *,
        retries: int | None = None,
        required: bool | None = None,
        detail: str = "执行算子",
    ) -> Any:
        policy = self.step_policies.get(step, StepExecutionPolicy())
        effective_retries = policy.retries if retries is None else retries
        effective_required = policy.required if required is None else required
        self._record(step, "running", detail)
        attempt = 0
        while True:
            try:
                result = operation()
                self._record(step, "success", "执行成功")
                return result
            except Exception as exc:
                if attempt < effective_retries:
                    attempt += 1
                    self._record(step, "retrying", f"第 {attempt} 次重试: {exc}")
                    continue

                error = {"step": step, "error": str(exc)}
                self.errors.append(error)
                if effective_required:
                    self._record(step, "failed", str(exc))
                    raise
                self._record(step, "skipped", str(exc))
                return None

    def _process_structured_records(self) -> dict[str, Any]:
        df = self._execute_step(
            "read_csv",
            lambda: ReadCsvOperator().execute(str(self.medical_records_path)),
            detail=f"读取 CSV: {self.medical_records_path}",
        )

        df = self._execute_step(
            "clean_missing",
            lambda: CleanMissingOperator(strategy="drop").execute(df),
            detail="删除空值记录",
        )

        df = self._execute_step(
            "normalize_text",
            lambda: NormalizeTextOperator().execute(
                df,
                columns=["gender", "chief_complaint", "diagnosis", "medicine", "department"],
            ),
            detail="规范化文本字段",
        )

        df = self._execute_step(
            "extract_fields",
            lambda: ExtractFieldOperator().execute(df, CORE_FIELDS),
            detail="抽取核心字段",
        )

        saved_path = self._execute_step(
            "save_csv",
            lambda: SaveCsvOperator().execute(df, str(self.cleaned_records_path), encoding="utf-8-sig"),
            detail=f"保存 CSV: {self.cleaned_records_path}",
        )

        return {
            "input_path": str(self.medical_records_path),
            "output_path": saved_path,
            "row_count": len(df),
            "columns": list(df.columns),
            "diagnosis_count": df["diagnosis"].value_counts().to_dict(),
            "department_count": df["department"].value_counts().to_dict(),
        }

    def _process_text_records(self, *, required: bool = True) -> dict[str, Any] | None:
        text = self._execute_step(
            "read_text",
            lambda: ReadTextOperator().execute(str(self.medical_text_path)),
            required=required,
            detail=f"读取文本: {self.medical_text_path}",
        )
        if text is None:
            return None

        paragraphs = self._execute_step(
            "split_paragraph",
            lambda: SplitParagraphOperator().execute(text, min_length=20),
            detail="切分疾病段落",
        )

        items: list[dict[str, Any]] = []
        for index, paragraph in enumerate(paragraphs, start=1):
            title = "未知疾病"
            content = paragraph
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if lines and lines[0].startswith("## "):
                title = lines[0].replace("## ", "", 1).strip()
                content = "\n".join(lines[1:]).strip()
            items.append({"id": index, "title": title, "content": content, "char_count": len(content)})

        output = {
            "input_path": str(self.medical_text_path),
            "output_path": str(self.cleaned_text_path),
            "paragraph_count": len(items),
            "items": items,
        }
        self._execute_step(
            "save_text_json",
            lambda: SaveJsonOperator().execute(output, str(self.cleaned_text_path)),
            detail=f"保存文本 JSON: {self.cleaned_text_path}",
        )
        return {
            "input_path": output["input_path"],
            "output_path": output["output_path"],
            "paragraph_count": output["paragraph_count"],
        }
