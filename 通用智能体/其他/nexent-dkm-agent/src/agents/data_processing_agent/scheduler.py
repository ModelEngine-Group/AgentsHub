"""DAG-based scheduler with retry support for task 1."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.agents.data_processing_agent.state import TaskStateTracker


class StepSpec:
    """Specification for one DAG step."""

    def __init__(
        self,
        name: str,
        operation: Callable[[], Any],
        depends_on: list[str] | None = None,
        max_retries: int = 0,
        retry_delay: float = 0.5,
        message: str = "",
    ) -> None:
        self.name = name
        self.operation = operation
        self.depends_on = depends_on or []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.message = message


class OperatorScheduler:
    """Run named operations sequentially and record their state.

    Kept for backward compatibility. New code should use DAGScheduler.
    """

    def __init__(self, tracker: TaskStateTracker) -> None:
        self.tracker = tracker

    def run_step(
        self,
        name: str,
        operation: Callable[[], Any],
        message: str = "",
    ) -> Any:
        step = self.tracker.start_step(name)
        try:
            result = operation()
        except Exception as exc:
            self.tracker.fail_step(step, exc)
            raise

        self.tracker.complete_step(
            step,
            message=message,
            artifacts=_summarize_artifact(result),
        )
        return result


class DAGScheduler:
    """Execute steps respecting a dependency DAG with optional parallel execution."""

    def __init__(
        self,
        tracker: TaskStateTracker,
        max_workers: int = 2,
    ) -> None:
        self.tracker = tracker
        self.max_workers = max_workers

    def run_dag(
        self,
        steps: list[StepSpec],
    ) -> dict[str, Any]:
        """Execute all steps respecting dependencies. Returns step_name -> result map."""

        results: dict[str, Any] = {}
        step_map = {s.name: s for s in steps}
        completed_names: set[str] = set()

        # Detect cycles
        _validate_dag(steps)

        layers = _topological_layers(steps)

        for layer in layers:
            if len(layer) == 1:
                name = layer[0]
                step = step_map[name]
                results[name] = self._execute_with_retry(step)
                completed_names.add(name)
            else:
                # Parallel execution for independent steps in the same layer
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(layer))) as pool:
                    futures = {}
                    for name in layer:
                        step = step_map[name]
                        # Capture operation in closure correctly
                        futures[pool.submit(self._execute_with_retry, step)] = name

                    for future in as_completed(futures):
                        name = futures[future]
                        results[name] = future.result()
                        completed_names.add(name)

        return results

    def _execute_with_retry(self, step: StepSpec) -> Any:
        """Execute a step with retry support and state tracking."""

        last_exc: Exception | None = None
        for attempt in range(1 + step.max_retries):
            tracker_step = self.tracker.start_step(step.name)
            try:
                result = step.operation()
                self.tracker.complete_step(
                    tracker_step,
                    message=step.message + (f" (attempt {attempt + 1})" if attempt > 0 else ""),
                    artifacts=_summarize_artifact(result),
                )
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < step.max_retries:
                    self.tracker.fail_step(tracker_step, exc)
                    time.sleep(step.retry_delay)
                else:
                    self.tracker.fail_step(tracker_step, exc)
                    raise

        raise last_exc  # type: ignore[misc]


def _topological_layers(steps: list[StepSpec]) -> list[list[str]]:
    """Return steps grouped into layers that can execute in parallel."""
    step_map = {s.name: set(s.depends_on) for s in steps}
    remaining = dict(step_map)
    layers: list[list[str]] = []

    while remaining:
        ready = [name for name, deps in remaining.items() if not deps]
        if not ready:
            raise ValueError(f"Circular dependency detected among: {list(remaining.keys())}")
        layers.append(sorted(ready))
        for name in ready:
            del remaining[name]
        for deps in remaining.values():
            deps.difference_update(ready)

    return layers


def _validate_dag(steps: list[StepSpec]) -> None:
    """Check that all dependency references are valid."""
    names = {s.name for s in steps}
    for step in steps:
        for dep in step.depends_on:
            if dep not in names:
                raise ValueError(f"Step '{step.name}' depends on unknown step '{dep}'")


def _summarize_artifact(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in (
        "status",
        "row_count",
        "column_count",
        "duplicate_rows",
        "output_rows",
        "missing_values_filled",
    ):
        if key in result:
            summary[key] = result[key]
    return summary
