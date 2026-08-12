"""Task 2 pipeline entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents import KGQAAgent, KnowledgeGraphAgent
from operators import SaveJsonOperator

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_TASK_DESCRIPTION = "请从医疗文本中抽取实体关系，并生成医疗知识图谱"
SAMPLE_QUESTIONS = [
    "肺炎有哪些症状？",
    "肺炎可以用什么药？",
    "哪些疾病关联胸痛？",
    "阿莫西林可以治疗哪些疾病？",
]


def run(task_description: str = DEFAULT_TASK_DESCRIPTION) -> dict[str, Any]:
    """Run task 2 through KnowledgeGraphAgent."""
    print("[task2] Knowledge Graph Agent pipeline started")
    agent = KnowledgeGraphAgent(data_dir=DATA_DIR, outputs_dir=OUTPUTS_DIR)
    result = agent.run(task_description)

    qa_agent = KGQAAgent(outputs_dir=OUTPUTS_DIR)
    qa_examples = [qa_agent.run(question) for question in SAMPLE_QUESTIONS]
    result["qa_examples"] = qa_examples
    SaveJsonOperator().execute(qa_examples, str(OUTPUTS_DIR / "medical_qa_examples.json"))

    print(f"[task2] Triples saved: {OUTPUTS_DIR / 'medical_triples.json'}")
    print(f"[task2] Graph saved: {OUTPUTS_DIR / 'medical_kg.json'}")
    print(f"[task2] QA examples saved: {OUTPUTS_DIR / 'medical_qa_examples.json'}")
    print("[task2] Knowledge Graph Agent pipeline finished successfully")
    return result


if __name__ == "__main__":
    run()
