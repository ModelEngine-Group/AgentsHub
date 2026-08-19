"""Task 1 pipeline entrypoint.

This module keeps the external task1 pipeline API stable, while delegating the
actual planning and execution to DataProcessingAgent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents import DataProcessingAgent

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_TASK_DESCRIPTION = "请清洗医疗CSV数据，抽取核心字段，并处理医疗文本"


def run(task_description: str = DEFAULT_TASK_DESCRIPTION) -> dict[str, Any]:
    """Run task 1 through DataProcessingAgent."""
    print("[task1] Data Processing Agent pipeline started")
    agent = DataProcessingAgent(data_dir=DATA_DIR, outputs_dir=OUTPUTS_DIR)
    result = agent.run(task_description)
    print(f"[task1] Summary saved: {OUTPUTS_DIR / 'task1_summary.json'}")
    print("[task1] Data Processing Agent pipeline finished successfully")
    return result


if __name__ == "__main__":
    run()
