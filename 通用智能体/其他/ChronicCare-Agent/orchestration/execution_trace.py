from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from runtime_common.common import now_iso, write_json


def write_trace(
    path: Path,
    run_id: str,
    user_goal: str,
    plan: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    final_answer: str,
    safety_note: str,
    agents_used: List[str],
    artifacts_used: List[str],
) -> None:
    write_json(
        path,
        {
            "run_id": run_id,
            "created_at": now_iso(),
            "user_goal": user_goal,
            "plan": plan,
            "steps": steps,
            "tool_call_count": len(steps),
            "agents_used": agents_used,
            "artifacts_used": artifacts_used,
            "final_answer": final_answer,
            "safety_note": safety_note,
        },
    )
