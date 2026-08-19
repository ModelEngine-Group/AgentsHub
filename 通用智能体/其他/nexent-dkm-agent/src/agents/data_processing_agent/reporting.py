"""Structured run reporting for the task-1 data processing agent."""

from __future__ import annotations

from typing import Any


def build_quality_report(
    plan: dict[str, Any],
    profile: dict[str, Any],
    cleaning: dict[str, Any],
    validation: dict[str, Any],
    datamate: dict[str, Any],
) -> dict[str, Any]:
    """Summarize run evidence for demos, tests, and competition review."""

    operators = plan.get("operators", [])
    operator_catalog = datamate.get("operators", {})
    template = operator_catalog.get("cleaning_template", {})
    task = operator_catalog.get("cleaning_task", {})
    submission = template.get("submission", {})

    readiness = {
        "task_understanding": bool(plan.get("understanding")),
        "multi_operator_plan": len(operators) >= 4,
        "local_execution": cleaning.get("status") == "completed",
        "quality_validation": validation.get("status") == "passed",
        "datamate_catalog": operator_catalog.get("status") == "available",
        "datamate_template_ready": template.get("status") in {"ready", "skipped"},
        "datamate_task_ready_or_waiting": task.get("status")
        in {"ready", "waiting_for_dataset", "skipped", None},
        "datamate_submission_safe": (
            submission.get("mode") in {"dry_run", "skipped", None}
            and submission.get("submitted") is not True
        ),
    }
    datamate_status = datamate.get("status")
    datamate_skipped = datamate_status == "skipped"
    datamate_ready = all(
        readiness[name]
        for name in (
            "datamate_catalog",
            "datamate_template_ready",
            "datamate_task_ready_or_waiting",
            "datamate_submission_safe",
        )
    )
    local_ready = readiness["task_understanding"] and not _core_checks_fail(readiness)
    status = (
        "passed"
        if local_ready and (datamate_skipped or datamate_ready)
        else "warning"
    )
    if _core_checks_fail(readiness):
        status = "failed"
    if validation.get("status") == "failed" or cleaning.get("status") == "failed":
        status = "failed"

    return {
        "status": status,
        "metrics": {
            "input_rows": profile.get("row_count", 0),
            "output_rows": cleaning.get(
                "output_rows",
                cleaning.get("output_records", 0),
            ),
            "duplicate_rows_before": profile.get("duplicate_rows", 0),
            "duplicate_rows_removed": cleaning.get("duplicate_rows_removed", 0),
            "missing_values_before": _missing_total(profile),
            "missing_values_filled": cleaning.get("missing_values_filled", 0),
            "planned_operator_count": len(operators),
            "datamate_operator_count": operator_catalog.get("operator_count", 0),
            "plan_confidence": plan.get("confidence"),
        },
        "readiness": readiness,
        "datamate": {
            "status": datamate_status,
            "execution_mode": "offline" if datamate_skipped else "integrated",
            "template_status": template.get("status"),
            "task_status": task.get("status"),
            "submission_mode": submission.get("mode"),
            "submitted": submission.get("submitted", False),
        },
    }


def _core_checks_fail(readiness: dict[str, bool]) -> bool:
    return any(
        not readiness[name]
        for name in ("multi_operator_plan", "local_execution", "quality_validation")
    )


def _missing_total(profile: dict[str, Any]) -> int:
    return sum(profile.get("missing_cells", {}).values())
