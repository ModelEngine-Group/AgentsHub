from __future__ import annotations

from typing import Any, Dict

from orchestration.supervisor import run_supervisor, supervisor_plan


def agent_plan(user_goal: str) -> Dict[str, Any]:
    return supervisor_plan(user_goal)


def agent_run(user_goal: str) -> Dict[str, Any]:
    return run_supervisor(user_goal)
