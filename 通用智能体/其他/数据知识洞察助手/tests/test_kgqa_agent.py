from pathlib import Path

from agents import KGQAAgent, KnowledgeGraphAgent
from pipelines.task2_pipeline import run as run_task2_pipeline


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def build_test_graph() -> None:
    KnowledgeGraphAgent(data_dir=DATA_DIR, outputs_dir=OUTPUTS_DIR).run(
        "请从医疗文本中抽取实体关系，并生成医疗知识图谱"
    )


def test_kgqa_agent_answers_disease_symptoms_question():
    build_test_graph()
    agent = KGQAAgent(outputs_dir=OUTPUTS_DIR)

    result = agent.run("肺炎有哪些症状？")

    assert result["status"] == "success"
    assert result["question_type"] == "disease_symptoms"
    assert "发热" in result["answers"]
    assert {"subject": "肺炎", "predicate": "has_symptom", "object": "发热"} in result["evidence"]


def test_kgqa_agent_answers_disease_drugs_question():
    build_test_graph()
    agent = KGQAAgent(outputs_dir=OUTPUTS_DIR)

    result = agent.run("肺炎可以用什么药？")

    assert result["status"] == "success"
    assert result["question_type"] == "disease_drugs"
    assert "阿莫西林" in result["answers"]
    assert {"subject": "肺炎", "predicate": "treated_by", "object": "阿莫西林"} in result["evidence"]


def test_kgqa_agent_answers_symptom_to_diseases_question():
    build_test_graph()
    agent = KGQAAgent(outputs_dir=OUTPUTS_DIR)

    result = agent.run("哪些疾病关联胸痛？")

    assert result["status"] == "success"
    assert result["question_type"] == "symptom_diseases"
    assert "肺炎" in result["answers"]
    assert "冠心病" in result["answers"]
    assert {"subject": "肺炎", "predicate": "has_symptom", "object": "胸痛"} in result["evidence"]


def test_kgqa_agent_answers_drug_to_diseases_question():
    build_test_graph()
    agent = KGQAAgent(outputs_dir=OUTPUTS_DIR)

    result = agent.run("阿莫西林可以治疗哪些疾病？")

    assert result["status"] == "success"
    assert result["question_type"] == "drug_diseases"
    assert "肺炎" in result["answers"]
    assert "急性扁桃体炎" in result["answers"]
    assert {"subject": "肺炎", "predicate": "treated_by", "object": "阿莫西林"} in result["evidence"]


def test_task2_pipeline_outputs_qa_examples():
    result = run_task2_pipeline()

    qa_path = OUTPUTS_DIR / "medical_qa_examples.json"
    assert result["qa_examples"]
    assert qa_path.exists()
    assert any(item["question_type"] == "disease_symptoms" for item in result["qa_examples"])
