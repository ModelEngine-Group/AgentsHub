from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "data-knowledge-insight-demo"}


def test_task1_run_endpoint_runs_data_processing_agent():
    response = client.post("/task1/run", json={"task_description": "请清洗医疗数据"})

    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "success"
    assert data["structured_result"]["row_count"] == 15


def test_task2_build_kg_endpoint_builds_graph():
    response = client.post("/task2/build-kg", json={"task_description": "请生成医疗知识图谱"})
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "success"
    assert data["triple_count"] > 0
    assert data["graph"]["nodes"]


def test_task2_ask_endpoint_answers_question():
    client.post("/task2/build-kg", json={"task_description": "请生成医疗知识图谱"})

    response = client.post("/task2/ask", json={"question": "肺炎有哪些症状？"})
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "success"
    assert "发热" in data["answers"]


def test_task3_analyze_endpoint_runs_data_analysis_agent():
    client.post("/task1/run", json={"task_description": "请清洗医疗数据"})
    client.post("/task2/build-kg", json={"task_description": "请生成医疗知识图谱"})

    response = client.post("/task3/analyze", json={"task_description": "请分析医疗知识图谱"})
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "success"
    assert data["statistics"]["node_count"] > 0
    assert data["nl2sql_metrics"]["accuracy"] >= 0.85
