"""Task 3 pipeline entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents import DataAnalysisAgent

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_TASK_DESCRIPTION = "请分析医疗知识图谱，生成统计结果、图表和报告"


def run(task_description: str = DEFAULT_TASK_DESCRIPTION) -> dict[str, Any]:
    """Run task 3 through DataAnalysisAgent."""
    print("[task3] Data Analysis Agent pipeline started")
    agent = DataAnalysisAgent(outputs_dir=OUTPUTS_DIR)
    result = agent.run(task_description)
    print(f"[task3] Summary saved: {OUTPUTS_DIR / 'analysis_summary.json'}")
    print(f"[task3] Report saved: {OUTPUTS_DIR / 'analysis_report.md'}")
    print("[task3] Data Analysis Agent pipeline finished successfully")
    return result


if __name__ == "__main__":
    run()
