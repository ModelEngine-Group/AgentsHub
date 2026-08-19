import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.kg_agent.nexent_adapter import (
    MedicalKGAgentTool,
    build_nexent_agent_spec,
    build_nexent_tool_spec,
)
from src.operators.kg_ops import (
    answer_graph_question,
    build_medical_graph,
    build_kg_quality_report,
    evaluate_extraction_quality,
    extract_medical_entities,
    extract_relations,
    extract_relations_tensorized,
    find_graph_entities,
    query_graph_neighbors,
    validate_triples,
)
from src.pipelines.task2_kg_pipeline import run_task2_pipeline
from src.pipelines.task2_evaluation import run_task2_evaluation


SAMPLE_TEXT = """
记录 1:
患者张三，男，45岁。主诉：头晕、头痛 3 天。既往有高血压病史5年，
长期服用氨氯地平。查体：血压160/95mmHg。初步诊断：高血压。
建议：血常规、肝功能检查，继续服用阿司匹林。
---
记录 2:
患者李四，女，52岁。主诉：口渴、多尿 2 周。既往有糖尿病病史。
长期服用二甲双胍。实验室检查：空腹血糖12.5mmol/L。初步诊断：2型糖尿病。
建议行尿常规、血糖监测，加用辛伐他汀调节血脂。
"""


def test_extract_medical_entities_from_records():
    extraction = extract_medical_entities(SAMPLE_TEXT)

    assert extraction["record_count"] == 2
    first = extraction["records"][0]
    assert first["record_id"] == "record_1"
    assert first["entities"]["Disease"] == ["高血压"]
    assert set(first["entities"]["Symptom"]) >= {"头晕", "头痛"}
    assert set(first["entities"]["Drug"]) >= {"氨氯地平", "阿司匹林"}
    assert set(first["entities"]["Examination"]) >= {"血常规", "肝功能"}
    second = extraction["records"][1]
    assert second["entities"]["Disease"] == ["2型糖尿病"]
    assert "糖尿病" in second["normalization"]["aliases"]["2型糖尿病"]


def test_extract_relations_and_validate_triples():
    extraction = extract_medical_entities(SAMPLE_TEXT)
    triples = extract_relations(extraction["records"])
    validation = validate_triples(triples)

    assert validation["status"] == "passed"
    assert validation["invalid_count"] == 0
    assert any(
        triple["subject"] == "高血压"
        and triple["predicate"] == "has_symptom"
        and triple["object"] == "头晕"
        and triple["record_id"] == "record_1"
        and triple["confidence"] >= 0.70
        and "头晕" in triple["evidence"]
        for triple in triples
    )
    assert any(
        triple["subject"] == "2型糖尿病"
        and triple["predicate"] == "diagnosed_by"
        and triple["object"] == "血糖"
        for triple in triples
    )


def test_extract_relations_tensorized_cpu_matches_rule():
    """The CPU tensor-scoring path must produce the same triples as the rule path."""
    from src.operators.kg_ops import extract_relations_tensorized

    extraction = extract_medical_entities(SAMPLE_TEXT)
    rule_triples = extract_relations(extraction["records"])
    tensor_result = extract_relations_tensorized(extraction["records"], backend="cpu")

    assert tensor_result["status"] == "completed"
    assert tensor_result["scoring_backend"] == "cpu"
    assert tensor_result["candidate_count"] > 0

    def _norm(triples):
        return sorted((t["subject"], t["predicate"], t["object"]) for t in triples)

    assert _norm(tensor_result["triples"]) == _norm(rule_triples)
    # Tensorized predicates must respect the schema's object typing.
    for triple in tensor_result["triples"]:
        assert triple["subject_type"] == "Disease"


def test_kg_query_reuses_graph_and_skips_rebuild(tmp_path):
    """A kg_query plan must reuse the cached graph and skip the build chain."""
    out_dir = tmp_path / "task2"
    build = run_task2_pipeline(output_dir=out_dir, question="高血压有哪些症状和用药？")
    assert build.status == "completed"
    build_steps = [s["name"] for s in build.artifacts["run_state"]["steps"]]
    assert "build_graph" in build_steps
    assert build.artifacts["plan_execution"].get("graph_reused") is None

    query = run_task2_pipeline(
        output_dir=out_dir,
        task_request="查询高血压相关实体",
        question=None,
    )
    assert query.status == "completed"
    assert query.artifacts["plan"]["understanding"]["task_type"] == "kg_query"
    assert query.artifacts["plan_execution"]["graph_reused"] is True

    query_steps = [s["name"] for s in query.artifacts["run_state"]["steps"]]
    assert "load_graph_artifact" in query_steps
    # The expensive rebuild chain must be entirely skipped.
    for build_step in ("extract_entities", "extract_relations", "validate_triples",
                       "build_graph", "export_graph"):
        assert build_step not in query_steps
    assert query.artifacts["retrieval"]["status"] == "completed"


def test_kg_query_without_cached_graph_builds(tmp_path):
    """Without a cached graph, a query request must fall back to a full build."""
    out_dir = tmp_path / "task2_fresh"
    query = run_task2_pipeline(
        output_dir=out_dir,
        task_request="查询高血压相关实体",
        question=None,
    )
    assert query.status == "completed"
    steps = [s["name"] for s in query.artifacts["run_state"]["steps"]]
    # No cached graph existed, so the build chain runs.
    assert "build_graph" in steps
    assert query.artifacts["plan_execution"].get("graph_reused") is None


def test_relation_backend_tensor_uses_real_records():
    from src.operators.kg_ops.relation_features import build_features_from_records

    extraction = extract_medical_entities(SAMPLE_TEXT)
    scoring = build_features_from_records(extraction["records"])

    assert scoring["candidate_count"] > 0
    assert scoring["scheme"] == "relation_projection"
    import torch

    assert isinstance(scoring["features"], torch.Tensor)
    assert scoring["features"].shape[0] == scoring["candidate_count"]
    assert scoring["features"].shape[1] >= 16

    tensor_result = extract_relations_tensorized(extraction["records"], backend="cpu")
    assert tensor_result["status"] == "completed"
    assert tensor_result["candidate_count"] == scoring["candidate_count"]


def test_entity_extractor_loads_external_dictionary():
    from src.operators.kg_ops import entity_extractor

    entity_extractor.reload_entity_dictionary()
    assert "甲亢" in entity_extractor.ENTITY_DICTIONARY["Disease"]
    assert "系统性红斑狼疮" in entity_extractor.ENTITY_DICTIONARY["Disease"]

    text = "记录: 患者确诊甲亢，手抖、消瘦，甲巯咪唑治疗，复查甲状腺超声。"
    extraction = extract_medical_entities(text)
    entities = extraction["records"][0]["entities"]
    assert "甲亢" in entities["Disease"]
    assert "手抖" in entities["Symptom"]
    assert "甲巯咪唑" in entities["Drug"]
    assert "甲状腺超声" in entities["Examination"]


def test_planner_skips_communities_for_small_corpus():
    from src.agents.kg_agent.planner import plan_kg_task

    plan = plan_kg_task("构建医疗知识图谱", question="高血压有哪些症状？", text_length=120)
    assert "build_kg_quality_report" not in plan.operators
    assert any("community" in item.lower() or "small" in item.lower() for item in plan.rationale)


def test_planner_prioritizes_qa_for_question_input():
    from src.agents.kg_agent.planner import plan_kg_task

    plan = plan_kg_task("查询图谱", question="哪些疾病最多症状？", text_length=80)
    assert plan.operators[0] == "find_graph_entities"
    assert "answer_graph_question" in plan.operators


def test_extract_relations_tensorized_invalid_backend():
    from src.operators.kg_ops import extract_relations_tensorized

    extraction = extract_medical_entities(SAMPLE_TEXT)
    try:
        extract_relations_tensorized(extraction["records"], backend="gpu")
    except ValueError as exc:
        assert "backend" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for unsupported backend")


def test_extract_relations_tensorized_npu_reports_scoring_mode_or_fallback():
    """NPU path must either complete (with scoring_mode) or gracefully fall back.

    On hosts without an Ascend NPU the path falls back to CPU tensor scoring,
    which still yields valid triples. This test ensures the new cached-argmax
    default does not break the contract on non-NPU machines.
    """

    extraction = extract_medical_entities(SAMPLE_TEXT)
    result = extract_relations_tensorized(extraction["records"], backend="npu")
    assert result["status"] == "completed"
    assert "scoring_mode" in result or result.get("scoring_backend") in {"cpu", "rule"}
    assert isinstance(result["triples"], list)
    assert len(result["triples"]) > 0


def test_extract_relations_orients_complication_triples():
    text = "记录 1:\n患者赵六。既往有高血压病史，现诊断为高血压并发冠心病，建议心电图检查。"
    extraction = extract_medical_entities(text)
    triples = extract_relations(extraction["records"])
    validation = validate_triples(triples)

    assert validation["status"] == "passed"
    assert any(
        triple["subject"] == "冠心病"
        and triple["predicate"] == "complication_of"
        and triple["object"] == "高血压"
        and triple["confidence"] >= 0.68
        and "并发冠心病" in triple["evidence"]
        for triple in validation["triples"]
    )


def test_extract_relations_skips_unsignalled_complication_pairs():
    """Co-occurring diseases without a linguistic complication cue must not
    produce a spurious ``complication_of`` triple."""
    text = "记录 1:\n患者王五。既往有高血压，本次体检发现2型糖尿病，建议血糖监测。"
    extraction = extract_medical_entities(text)
    triples = extract_relations(extraction["records"])

    diseases = extraction["records"][0]["entities"]["Disease"]
    assert {"高血压", "2型糖尿病"}.issubset(set(diseases))
    assert not any(triple["predicate"] == "complication_of" for triple in triples)


def test_build_medical_graph_deduplicates_nodes_and_edges():
    extraction = extract_medical_entities(SAMPLE_TEXT)
    triples = extract_relations(extraction["records"])
    graph = build_medical_graph(triples, extraction["records"])

    node_ids = {node["id"] for node in graph["nodes"]}
    assert {"Disease:高血压", "Drug:阿司匹林", "Symptom:头痛"}.issubset(node_ids)
    assert graph["statistics"]["node_count"] == len(graph["nodes"])
    assert graph["statistics"]["edge_count"] == len(graph["edges"])
    assert graph["statistics"]["triple_count"] == len(triples)
    symptom_edge = next(edge for edge in graph["edges"] if edge["predicate"] == "has_symptom")
    assert symptom_edge["evidence"]


def test_graph_query_finds_entities_and_neighbors():
    extraction = extract_medical_entities(SAMPLE_TEXT)
    triples = extract_relations(extraction["records"])
    graph = build_medical_graph(triples, extraction["records"])

    matches = find_graph_entities("糖尿病", graph, entity_type="Disease")
    assert matches["status"] == "matched"
    assert matches["matches"][0]["name"] == "2型糖尿病"
    assert matches["matches"][0]["match_type"] in {"exact", "substring"}

    neighbors = query_graph_neighbors("糖尿病", graph, relation="diagnosed_by")
    target_names = {neighbor["target"]["name"] for neighbor in neighbors["neighbors"]}
    assert neighbors["status"] == "matched"
    assert {"血糖", "尿常规"}.issubset(target_names)
    assert all(neighbor["evidence"] for neighbor in neighbors["neighbors"])


def test_answer_graph_question_uses_graph_relations():
    extraction = extract_medical_entities(SAMPLE_TEXT)
    triples = extract_relations(extraction["records"])
    graph = build_medical_graph(triples, extraction["records"])

    answer = answer_graph_question("高血压有哪些症状和用药？", graph)

    assert answer["status"] == "answered"
    assert "头晕" in answer["answer"]
    assert "阿司匹林" in answer["answer"] or "氨氯地平" in answer["answer"]
    assert answer["evidence"]


def test_answer_graph_question_handles_complications():
    text = "记录 1:\n患者赵六。既往有高血压病史，现诊断为高血压并发冠心病，建议心电图检查。"
    extraction = extract_medical_entities(text)
    triples = extract_relations(extraction["records"])
    validation = validate_triples(triples)
    graph = build_medical_graph(validation["triples"], extraction["records"])

    answer = answer_graph_question("高血压有哪些并发症？", graph)

    assert answer["status"] == "answered"
    assert "冠心病" in answer["answer"]
    assert answer["evidence"][0]["predicate"] == "complication_of"


def test_task2_pipeline_exports_graph_and_answers_question(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    result = run_task2_pipeline(
        input_path=source,
        output_dir=tmp_path / "outputs",
        question="糖尿病需要做什么检查？",
    )

    assert result.task == "task2_kg_agent"
    assert result.status == "completed"
    assert result.artifacts["extraction"]["record_count"] == 2
    assert result.artifacts["validation"]["status"] == "passed"
    assert result.artifacts["qa"]["status"] == "answered"
    assert result.artifacts["quality_report"]["status"] == "passed"
    assert result.artifacts["quality_report"]["readiness"]["graph_exported"]

    graph_path = Path(result.artifacts["graph"]["output_path"])
    assert graph_path.exists()
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    assert payload["statistics"]["triple_count"] > 0


def test_quality_report_summarizes_graph_readiness():
    extraction = extract_medical_entities(SAMPLE_TEXT)
    triples = extract_relations(extraction["records"])
    validation = validate_triples(triples)
    graph = build_medical_graph(validation["triples"], extraction["records"])
    qa = answer_graph_question("糖尿病需要做什么检查？", graph)

    report = build_kg_quality_report(
        extraction=extraction,
        validation=validation,
        graph=graph,
        qa=qa,
        export={"status": "completed", "output_path": "outputs/task2/medical_kg.json"},
    )

    assert report["status"] == "passed"
    assert report["metrics"]["entity_total"] >= 8
    assert report["metrics"]["triple_count"] == graph["statistics"]["triple_count"]
    assert report["metrics"]["relation_type_count"] >= 3
    assert report["metrics"]["relation_coverage"] >= 0.6
    assert report["metrics"]["average_confidence"] > 0
    assert report["readiness"]["qa_answered"]


def test_task2_pipeline_reports_missing_input(tmp_path):
    result = run_task2_pipeline(input_path=tmp_path / "missing.txt")

    assert result.status == "failed"
    assert result.artifacts["error"]["type"] == "FileNotFoundError"
    assert result.artifacts["run_state"]["status"] == "failed"


def test_task2_evaluation_writes_compact_report(tmp_path):
    source = tmp_path / "notes.txt"
    report_path = tmp_path / "reports" / "task2_quality_report.json"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    payload = run_task2_evaluation(
        input_path=source,
        output_dir=tmp_path / "outputs",
        question="糖尿病需要做什么检查？",
        report_path=report_path,
    )

    assert payload["task"] == "task2_kg_agent"
    assert payload["status"] == "completed"
    assert payload["quality_report"]["status"] == "passed"
    assert payload["graph"]["triple_count"] > 0
    assert payload["qa"]["status"] == "answered"
    assert report_path.exists()


def test_nexent_adapter_wraps_task2_pipeline(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    tool = MedicalKGAgentTool(output_dir=str(tmp_path / "outputs"))
    payload = json.loads(
        tool.forward(
            input_path=str(source),
            question="高血压有哪些症状？",
        )
    )

    assert payload["task"] == "task2_kg_agent"
    assert payload["status"] == "completed"
    assert payload["artifacts"]["qa"]["status"] == "answered"

    tool_spec = build_nexent_tool_spec(output_dir=str(tmp_path / "outputs"))
    agent_spec = build_nexent_agent_spec(model_name="glm-5.1")
    assert tool_spec["name"] == "task2_medical_kg"
    assert agent_spec["name"] == "task2_medical_kg_agent"
    assert agent_spec["tools"][0]["name"] == "task2_medical_kg"


def test_task2_api_process_status_report_and_query(tmp_path):
    from fastapi.testclient import TestClient

    from src.pipelines.task2_api_server import app, _tasks

    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")
    _tasks.clear()

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["service"] == "task2_medical_kg_agent"

    operators = client.get("/api/task2/operators")
    assert operators.status_code == 200
    assert "extract_medical_entities" in operators.json()["operators"]
    assert "query_graph_neighbors" in operators.json()["operators"]

    submitted = client.post(
        "/api/task2/process",
        json={
            "input_path": str(source),
            "output_dir": str(tmp_path / "outputs"),
            "question": "糖尿病需要做什么检查？",
        },
    )
    assert submitted.status_code == 200
    task_id = submitted.json()["task_id"]
    assert submitted.json()["status"] == "completed"

    status = client.get(f"/api/task2/status/{task_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"

    report = client.get(f"/api/task2/report/{task_id}")
    assert report.status_code == 200
    assert report.json()["artifacts"]["quality_report"]["status"] == "passed"

    query = client.post(
        "/api/task2/query",
        json={"task_id": task_id, "entity": "糖尿病", "relation": "diagnosed_by"},
    )
    assert query.status_code == 200
    target_names = {neighbor["target"]["name"] for neighbor in query.json()["neighbors"]}
    assert {"血糖", "尿常规"}.issubset(target_names)


def test_plan_kg_task_full_pipeline():
    from src.agents.kg_agent.planner import plan_kg_task

    plan = plan_kg_task("构建医疗知识图谱", question="高血压有哪些症状？")
    assert plan.understanding.task_type == "full_pipeline"
    assert plan.operators[0] == "extract_medical_entities"
    assert "answer_graph_question" in plan.operators
    assert plan.planner_mode == "rule"
    assert plan.confidence >= 0.5


def test_plan_kg_task_query_only():
    from src.agents.kg_agent.planner import plan_kg_task

    plan = plan_kg_task("查询高血压相关的药物", question=None)
    assert plan.understanding.task_type == "kg_query"
    assert "find_graph_entities" in plan.operators
    assert plan.planner_mode == "rule"


def test_plan_kg_task_with_question():
    from src.agents.kg_agent.planner import plan_kg_task

    plan = plan_kg_task(question="高血压有哪些症状和用药？")
    assert plan.understanding.task_type == "kg_qa"
    assert "answer_graph_question" in plan.operators
    assert "build_medical_graph" in plan.operators


def test_kg_hybrid_planner_rule_fallback():
    from src.agents.kg_agent.planner import KGHybridPlanner

    planner = KGHybridPlanner(llm_config=None, local_model_path=None)
    plan = planner.plan("构建医疗知识图谱", question="高血压有哪些症状？")
    assert plan.planner_mode == "rule"
    assert plan.operators[0] == "extract_medical_entities"


def test_task2_plan_drives_qa_execution(tmp_path):
    """A query-only plan (no question) should skip the QA operator."""
    from src.agents.kg_agent.agent import MedicalKGAgent

    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    result = MedicalKGAgent().run(
        input_path=source,
        output_dir=tmp_path / "outputs",
        question=None,
        task_request="查询高血压相关的药物",
    )

    assert result.status == "completed"
    assert result.artifacts["plan_execution"]["qa_executed"] is False
    assert result.artifacts["qa"]["status"] == "skipped"


def test_plan_kg_task_query_with_question_includes_qa():
    """A query-typed request that also carries a question must still plan QA."""
    from src.agents.kg_agent.planner import plan_kg_task

    plan = plan_kg_task("查询高血压相关的药物", question="高血压有哪些症状？")
    assert plan.understanding.task_type == "kg_query"
    assert "answer_graph_question" in plan.operators


def test_task2_query_request_with_question_answers(tmp_path):
    """Regression: a query request + a question must answer it, not silently skip."""
    from src.agents.kg_agent.agent import MedicalKGAgent

    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    result = MedicalKGAgent().run(
        input_path=source,
        output_dir=tmp_path / "outputs",
        question="高血压有哪些症状？",
        task_request="查询高血压相关的药物",
    )

    assert result.status == "completed"
    assert result.artifacts["plan_execution"]["qa_executed"] is True
    assert result.artifacts["qa"]["status"] == "answered"


def test_task2_executes_query_operators_when_planned(tmp_path):
    """A kg_query plan must actually run the retrieval operators (not leave them dead)."""
    from src.agents.kg_agent.agent import MedicalKGAgent

    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    result = MedicalKGAgent().run(
        input_path=source,
        output_dir=tmp_path / "outputs",
        question=None,
        task_request="查询高血压相关的药物",
    )

    assert result.status == "completed"
    retrieval = result.artifacts["retrieval"]
    assert retrieval["status"] == "completed"
    assert retrieval["entities"]["status"] == "matched"
    assert "find_graph_entities" in result.artifacts["plan_execution"]["executed_operators"]


def test_task2_plan_executes_qa_when_question_present(tmp_path):
    from src.agents.kg_agent.agent import MedicalKGAgent

    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    result = MedicalKGAgent().run(
        input_path=source,
        output_dir=tmp_path / "outputs",
        question="高血压有哪些症状？",
    )

    assert result.status == "completed"
    assert result.artifacts["plan_execution"]["qa_executed"] is True
    assert result.artifacts["qa"]["status"] == "answered"


def test_task2_pipeline_with_planner_artifact(tmp_path):
    from src.pipelines.task2_kg_pipeline import run_task2_pipeline

    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    result = run_task2_pipeline(
        input_path=source,
        output_dir=tmp_path / "outputs",
        question="高血压有哪些症状？",
    )
    assert result.status == "completed"
    plan_artifact = result.artifacts.get("plan", {})
    assert "operators" in plan_artifact
    assert plan_artifact.get("planner_mode") == "rule"


def test_task2_demo_cli_llm_config_not_required(tmp_path):
    import subprocess

    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "demos/task2_demo.py",
            "--input", str(source),
            "--output-dir", str(tmp_path / "outputs"),
            "--question", "高血压有哪些症状？",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0
    assert "Status: completed" in proc.stdout
    assert "[Mode]" in proc.stdout


def test_task2_demo_cli_rejects_incomplete_llm_config(tmp_path):
    bad_config = tmp_path / "llm_config.env"
    bad_config.write_text("OPENAI_BASE_URL=https://example.test/v1\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "demos/task2_demo.py",
            "--llm-config", str(bad_config),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )

    assert proc.returncode == 2
    assert "LLM config is missing or incomplete" in proc.stdout


def test_task2_nexent_spec_cli_prints_agent_spec():
    proc = subprocess.run(
        [
            sys.executable,
            "demos/task2_nexent_spec.py",
            "--model-name", "glm-5.1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["name"] == "task2_medical_kg_agent"
    assert payload["model_name"] == "glm-5.1"
    assert payload["tools"][0]["name"] == "task2_medical_kg"


def test_task2_api_rejects_incomplete_llm_config(tmp_path):
    from fastapi.testclient import TestClient
    from src.pipelines.task2_api_server import app, _tasks

    bad_config = tmp_path / "llm_config.env"
    bad_config.write_text("OPENAI_BASE_URL=https://example.test/v1\n", encoding="utf-8")
    _tasks.clear()

    client = TestClient(app)
    response = client.post(
        "/api/task2/process",
        json={"llm_config_path": str(bad_config)},
    )

    assert response.status_code == 400
    assert "LLM config is missing or incomplete" in response.json()["detail"]


def test_llm_extractor_parse_json():
    from src.operators.kg_ops.llm_extractor import _parse_extraction_json

    raw_fenced = '```json\n{"entities": {"Disease": ["高血压"]}, "relations": []}\n```'
    result = _parse_extraction_json(raw_fenced)
    assert result["entities"]["Disease"] == ["高血压"]

    raw_plain = '{"entities": {"Symptom": ["头晕"]}, "relations": [{"subject": "高血压", "predicate": "has_symptom", "object": "头晕"}]}'
    result = _parse_extraction_json(raw_plain)
    assert result["entities"]["Symptom"] == ["头晕"]
    assert len(result["relations"]) == 1


def test_llm_extractor_split_chunks():
    from src.operators.kg_ops.llm_extractor import _split_chunks

    multi_record = "记录1：高血压\n---\n记录2：糖尿病"
    chunks = _split_chunks(multi_record)
    assert len(chunks) == 2

    single = "只有一段文字"
    chunks = _split_chunks(single)
    assert len(chunks) == 1


def test_llm_extractor_valid_relation():
    from src.operators.kg_ops.llm_extractor import _valid_relation

    valid = {"subject": "高血压", "predicate": "has_symptom", "object": "头晕"}
    assert _valid_relation(valid)

    invalid_pred = {"subject": "高血压", "predicate": "unknown_rel", "object": "头晕"}
    assert not _valid_relation(invalid_pred)

    missing_obj = {"subject": "高血压", "predicate": "has_symptom", "object": ""}
    assert not _valid_relation(missing_obj)


def test_llm_extractor_evidence_snippet():
    from src.operators.kg_ops.llm_extractor import _evidence_snippet

    text = "患者主诉头晕、头痛3天。既往有高血压病史5年，长期服用氨氯地平。"
    snippet = _evidence_snippet(text, "高血压", "氨氯地平")
    assert "高血压" in snippet or "氨氯地平" in snippet


def test_extract_entities_with_llm_fallback():
    """Test that LLM extraction gracefully falls back when LLM fails."""
    from unittest.mock import patch
    from src.operators.kg_ops.llm_extractor import extract_entities_with_llm

    bad_config = {"base_url": "http://invalid:9999", "api_key": "fake", "model_name": "test"}

    with patch("src.operators.kg_ops.llm_extractor._call_llm_extraction", side_effect=Exception("LLM unavailable")):
        result = extract_entities_with_llm(SAMPLE_TEXT, bad_config)
        # Should still return a valid structure even when LLM fails
        assert result["status"] == "completed"
        assert result["record_count"] >= 1


def test_extract_entities_with_llm_keeps_entities_per_record():
    """Each record must carry only the entities found in its own chunk, not the
    global union across all chunks."""
    from unittest.mock import patch
    from src.operators.kg_ops.llm_extractor import extract_entities_with_llm

    text = "记录 1:\n高血压患者出现头晕。\n---\n记录 2:\n糖尿病患者使用二甲双胍。"

    def fake_call(chunk_text, _config):
        if "高血压" in chunk_text:
            return {"entities": {"Disease": ["高血压"], "Symptom": ["头晕"]}, "relations": []}
        return {"entities": {"Disease": ["糖尿病"], "Drug": ["二甲双胍"]}, "relations": []}

    with patch("src.operators.kg_ops.llm_extractor._call_llm_extraction", side_effect=fake_call):
        result = extract_entities_with_llm(text, {"base_url": "x", "api_key": "y"})

    assert result["record_count"] == 2
    first, second = result["records"]
    assert first["entities"]["Disease"] == ["高血压"]
    assert "糖尿病" not in first["entities"]["Disease"]
    assert second["entities"]["Disease"] == ["糖尿病"]
    assert "高血压" not in second["entities"]["Disease"]
    # Aggregate counts still reflect the union across the document.
    assert result["entity_counts"]["Disease"] == 2


def test_extract_relations_with_llm_merges():
    """Test LLM relation extraction merges with rule-based triples."""
    from unittest.mock import patch
    from src.operators.kg_ops.llm_extractor import extract_relations_with_llm

    fake_llm_output = {
        "entities": {"Disease": ["高血压"]},
        "relations": [
            {"subject": "高血压", "predicate": "has_symptom", "object": "心悸", "confidence": 0.75}
        ],
    }
    rule_triples = [
        {
            "subject": "高血压", "predicate": "has_symptom", "object": "头晕",
            "record_id": "record_1", "confidence": 0.82,
            "subject_type": "Disease", "object_type": "Symptom",
            "evidence": "头晕、头痛",
        }
    ]

    with patch("src.operators.kg_ops.llm_extractor._call_llm_extraction", return_value=fake_llm_output):
        merged = extract_relations_with_llm("dummy text", {"base_url": "x", "api_key": "y"}, rule_triples)
        objects = {t["object"] for t in merged}
        assert "头晕" in objects  # from rule-based
        assert "心悸" in objects  # from LLM


def test_multi_hop_query_finds_paths():
    from src.operators.kg_ops.multi_hop_qa import multi_hop_query

    extraction = extract_medical_entities(SAMPLE_TEXT)
    triples = extract_relations(extraction["records"])
    validation = validate_triples(triples)
    graph = build_medical_graph(validation["triples"], extraction["records"])

    result = multi_hop_query(graph, "高血压", target_entity="阿司匹林", max_hops=3)
    assert result["status"] == "matched"
    assert result["path_count"] >= 1
    assert result["paths"][0]["hop_count"] >= 1


def test_multi_hop_query_unmatched():
    from src.operators.kg_ops.multi_hop_qa import multi_hop_query

    extraction = extract_medical_entities(SAMPLE_TEXT)
    triples = extract_relations(extraction["records"])
    validation = validate_triples(triples)
    graph = build_medical_graph(validation["triples"], extraction["records"])

    result = multi_hop_query(graph, "不存在的疾病")
    assert result["status"] == "unmatched"
    assert result.get("path_count", 0) == 0


def test_build_evidence_chain():
    from src.operators.kg_ops.multi_hop_qa import build_evidence_chain

    extraction = extract_medical_entities(SAMPLE_TEXT)
    triples = extract_relations(extraction["records"])
    validation = validate_triples(triples)
    graph = build_medical_graph(validation["triples"], extraction["records"])

    result = build_evidence_chain(graph, "高血压有哪些症状？", max_hops=2)
    assert result["status"] == "chains_found"
    assert result["chain_count"] >= 1
    assert result["entity_count"] >= 1


def test_answer_with_evidence_chain():
    from src.operators.kg_ops.multi_hop_qa import answer_with_evidence_chain

    extraction = extract_medical_entities(SAMPLE_TEXT)
    triples = extract_relations(extraction["records"])
    validation = validate_triples(triples)
    graph = build_medical_graph(validation["triples"], extraction["records"])

    result = answer_with_evidence_chain("高血压有哪些症状？", graph)
    assert result["reasoning_depth"] == "multi_hop"
    assert "推理链路" in result["answer"]
    assert result.get("chain_facts") is not None
    assert len(result["chain_facts"]) >= 1


def test_answer_with_evidence_chain_no_entities():
    from src.operators.kg_ops.multi_hop_qa import answer_with_evidence_chain

    extraction = extract_medical_entities(SAMPLE_TEXT)
    triples = extract_relations(extraction["records"])
    validation = validate_triples(triples)
    graph = build_medical_graph(validation["triples"], extraction["records"])

    result = answer_with_evidence_chain("完全无关的问题内容", graph)
    assert result["status"] in ("answered", "unanswered")


def test_task2_api_multi_hop_and_evidence_qa(tmp_path):
    from fastapi.testclient import TestClient
    from src.pipelines.task2_api_server import app, _tasks

    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")
    _tasks.clear()

    client = TestClient(app)

    # Check operators list includes new ones
    operators = client.get("/api/task2/operators")
    ops_list = operators.json()["operators"]
    assert "multi_hop_query" in ops_list
    assert "answer_with_evidence_chain" in ops_list

    # Submit task
    submitted = client.post(
        "/api/task2/process",
        json={
            "input_path": str(source),
            "output_dir": str(tmp_path / "outputs"),
            "question": "高血压有哪些症状？",
        },
    )
    assert submitted.status_code == 200
    task_id = submitted.json()["task_id"]

    # Multi-hop query
    mh_result = client.post(
        "/api/task2/multi-hop",
        json={
            "task_id": task_id,
            "start_entity": "高血压",
            "target_entity": "阿司匹林",
            "max_hops": 3,
            "max_paths": 3,
        },
    )
    assert mh_result.status_code == 200
    assert mh_result.json()["status"] == "matched"

    # Evidence QA
    eqa_result = client.post(
        "/api/task2/evidence-qa",
        json={"task_id": task_id, "question": "高血压有哪些症状？"},
    )
    assert eqa_result.status_code == 200
    assert eqa_result.json()["reasoning_depth"] == "multi_hop"

    # Evidence chain
    chain_result = client.post(
        "/api/task2/evidence-chain",
        json={"task_id": task_id, "question": "高血压有哪些症状？"},
    )
    assert chain_result.status_code == 200
    assert chain_result.json()["status"] in ("chains_found", "no_chains")


def test_task2_api_404_for_unknown_task():
    from fastapi.testclient import TestClient
    from src.pipelines.task2_api_server import app, _tasks

    _tasks.clear()
    client = TestClient(app)

    resp = client.get("/api/task2/status/nonexistent")
    assert resp.status_code == 404


def test_generate_kg_training_data():
    from data.training.generate_kg_training_data import generate_samples

    samples = generate_samples(n=10)
    assert len(samples) == 10
    for s in samples:
        assert "instruction" in s
        assert "input" in s
        assert "output" in s
        output = json.loads(s["output"])
        assert "entities" in output
        assert "relations" in output


def test_kg_training_instruction_matches_canonical_prompt():
    """Generated instructions must equal the shared prompt used at inference.

    A drift here would train an adapter on one prompt and query it with another,
    silently degrading the fine-tuned model to base behaviour.
    """
    from data.training.generate_kg_training_data import generate_samples
    from src.operators.kg_ops.kg_prompts import KG_INSTRUCTION

    sample = generate_samples(n=1)[0]
    assert sample["instruction"] == KG_INSTRUCTION


def test_local_model_ner_parse_entity_response():
    from src.operators.kg_ops.local_model_ner import _parse_entity_response

    raw = '{"entities": {"Disease": ["高血压"], "Symptom": ["头晕", "头痛"], "Drug": ["氨氯地平"]}, "relations": []}'
    result = _parse_entity_response(raw)
    assert result is not None
    assert "高血压" in result["Disease"]
    assert "头晕" in result["Symptom"]

    fenced = '```json\n{"entities": {"Disease": ["冠心病"]}, "relations": []}\n```'
    result = _parse_entity_response(fenced)
    assert result is not None
    assert "冠心病" in result["Disease"]

    assert _parse_entity_response("not json at all") is None


def test_local_model_ner_predict_without_model():
    from src.operators.kg_ops.local_model_ner import predict_kg_entities

    result = predict_kg_entities(None, "患者有高血压")
    assert result is None  # No model path


def test_task2_pipeline_with_local_model_path(tmp_path):
    """Pipeline should still work when local_model_path is set but model doesn't exist."""
    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    result = run_task2_pipeline(
        input_path=source,
        output_dir=tmp_path / "outputs",
        question="高血压有哪些症状？",
        local_model_path="/nonexistent/model/path",
    )
    # Should fall back to rule-based and succeed
    assert result.status == "completed"


# ---------------------------------------------------------------------------
# Neo4j Tests (mock-based, runs without actual Neo4j server)
# ---------------------------------------------------------------------------

def test_neo4j_predicate_mapping():
    from src.operators.kg_ops.neo4j_store import _PREDICATE_TO_REL_TYPE, _REL_TYPE_TO_PREDICATE

    assert _PREDICATE_TO_REL_TYPE["has_symptom"] == "HAS_SYMPTOM"
    assert _PREDICATE_TO_REL_TYPE["treated_by"] == "TREATED_BY"
    assert _REL_TYPE_TO_PREDICATE["HAS_SYMPTOM"] == "has_symptom"
    assert _REL_TYPE_TO_PREDICATE["TREATED_BY"] == "treated_by"


def test_neo4j_store_functions_import():
    """Import test to verify package structure."""
    import inspect

    from src.operators.kg_ops.neo4j_store import (
        graph_to_neo4j,
        neo4j_to_graph,
        clear_neo4j_graph,
        check_neo4j_connection,
        _get_driver,
    )
    for fn in (graph_to_neo4j, neo4j_to_graph, clear_neo4j_graph, check_neo4j_connection, _get_driver):
        assert callable(fn)
    for fn in (graph_to_neo4j, neo4j_to_graph, clear_neo4j_graph, check_neo4j_connection):
        assert inspect.signature(fn).parameters["password"].default is None
    assert graph_to_neo4j({})["status"] == "credentials_required"
    assert neo4j_to_graph()["status"] == "credentials_required"
    assert clear_neo4j_graph()["status"] == "credentials_required"
    assert check_neo4j_connection()["status"] == "credentials_required"


def test_neo4j_query_functions_import():
    import inspect

    from src.operators.kg_ops.neo4j_query import (
        neo4j_find_entities,
        neo4j_query_neighbors,
        neo4j_multi_hop,
        neo4j_answer_question,
        _get_driver,
        _score,
    )
    for fn in (neo4j_find_entities, neo4j_query_neighbors, neo4j_multi_hop, neo4j_answer_question, _get_driver):
        assert callable(fn)
    for fn in (
        neo4j_find_entities,
        neo4j_query_neighbors,
        neo4j_multi_hop,
        neo4j_answer_question,
    ):
        assert inspect.signature(fn).parameters["password"].default is None
    assert neo4j_find_entities("高血压")["status"] == "credentials_required"
    assert neo4j_query_neighbors("高血压")["status"] == "credentials_required"
    assert neo4j_multi_hop("高血压")["status"] == "credentials_required"
    assert neo4j_answer_question("高血压有哪些症状？")["status"] == "credentials_required"
    assert _score("高血压", "高血压") == 1.0
    assert _score("高血压", "高血压病") > 0.0


def test_neo4j_get_driver_returns_none_without_package():
    """When neo4j package is not installed, _get_driver returns None."""
    from unittest.mock import patch
    from src.operators.kg_ops.neo4j_store import _get_driver as store_driver
    from src.operators.kg_ops.neo4j_query import _get_driver as query_driver

    with patch.dict("sys.modules", {"neo4j": None, "neo4j.GraphDatabase": None}):
        # Force ImportError behavior
        result = store_driver("bolt://localhost:7687", "neo4j", "neo4j")
        assert result is None

        result = query_driver("bolt://localhost:7687", "neo4j", "neo4j")
        assert result is None


def test_graph_to_neo4j_without_driver():
    """graph_to_neo4j returns unavailable status when driver is not available."""
    from unittest.mock import patch
    from src.operators.kg_ops.neo4j_store import graph_to_neo4j

    graph = {
        "nodes": [{"id": "Disease:高血压", "name": "高血压", "type": "Disease", "record_ids": [], "mention_count": 1}],
        "edges": [],
    }

    with patch("src.operators.kg_ops.neo4j_store._get_driver", return_value=None):
        result = graph_to_neo4j(graph, "bolt://localhost:7687", "neo4j", "neo4j")
        assert result["status"] == "unavailable"


def test_neo4j_find_entities_without_driver():
    """neo4j_find_entities returns driver_unavailable when driver is not available."""
    from unittest.mock import patch
    from src.operators.kg_ops.neo4j_query import neo4j_find_entities

    with patch("src.operators.kg_ops.neo4j_query._get_driver", return_value=None):
        result = neo4j_find_entities("高血压", "bolt://localhost:7687", "neo4j", "neo4j")
        assert result["status"] == "driver_unavailable"
        assert result["matches"] == []


def test_task2_pipeline_with_neo4j_config_gracefully_fails(tmp_path):
    """Pipeline should succeed even when Neo4j is unavailable."""
    from unittest.mock import patch

    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    neo4j_config = {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "neo4j",
    }

    with patch("src.operators.kg_ops.neo4j_store._get_driver", return_value=None):
        result = run_task2_pipeline(
            input_path=source,
            output_dir=tmp_path / "outputs",
            question="高血压有哪些症状？",
            neo4j_config=neo4j_config,
        )
        # Pipeline should complete even if Neo4j persist fails
        assert result.status == "completed"
        neo4j_artifact = result.artifacts.get("neo4j")
        assert neo4j_artifact is not None
        assert neo4j_artifact.get("status") == "unavailable"


def test_check_neo4j_connection_without_driver():
    from unittest.mock import patch
    from src.operators.kg_ops.neo4j_store import check_neo4j_connection

    with patch("src.operators.kg_ops.neo4j_store._get_driver", return_value=None):
        result = check_neo4j_connection("bolt://localhost:7687", "neo4j", "neo4j")
        assert result["status"] == "driver_unavailable"


def test_neo4j_sanitize_label_rejects_unknown():
    from src.operators.kg_ops.neo4j_store import _sanitize_label

    assert _sanitize_label("Disease") == "Disease"
    assert _sanitize_label("Symptom") == "Symptom"
    assert _sanitize_label("UnknownType") == "Entity"
    assert _sanitize_label("") == "Entity"
    assert _sanitize_label("DROP DATABASE") == "Entity"


def test_neo4j_sanitize_cypher_label():
    from src.operators.kg_ops.neo4j_query import _sanitize_cypher_label

    assert _sanitize_cypher_label("Drug") == "Drug"
    assert _sanitize_cypher_label("Examination") == "Examination"
    assert _sanitize_cypher_label("HackerLabel") == "Entity"


def test_neo4j_sanitize_relationship_type():
    from src.operators.kg_ops.neo4j_query import _sanitize_relationship_type

    assert _sanitize_relationship_type("has_symptom") == "HAS_SYMPTOM"
    assert _sanitize_relationship_type("treated_by") == "TREATED_BY"
    assert _sanitize_relationship_type("HAS_SYMPTOM") == "HAS_SYMPTOM"
    assert _sanitize_relationship_type("HAS_SYMPTOM` MATCH (n) DETACH DELETE n") is None


def test_neo4j_valid_entity_labels_complete():
    from src.operators.kg_ops.neo4j_store import _VALID_ENTITY_LABELS

    expected = {"Disease", "Symptom", "Drug", "Examination", "Treatment"}
    assert _VALID_ENTITY_LABELS == expected


def test_neo4j_valid_rel_types_complete():
    from src.operators.kg_ops.neo4j_store import _VALID_REL_TYPES

    expected = {"HAS_SYMPTOM", "TREATED_BY", "DIAGNOSED_BY", "RECOMMENDED_TREATMENT", "COMPLICATION_OF"}
    assert _VALID_REL_TYPES == expected


def test_neo4j_clear_scope_is_task2_only():
    """Verify clear_neo4j_graph docstring mentions scope."""
    from src.operators.kg_ops.neo4j_store import clear_neo4j_graph

    doc = clear_neo4j_graph.__doc__ or ""
    assert "task-2" in doc.lower() or "task2" in doc.lower()
    assert "whitelist" in doc.lower() or "label" in doc.lower()


def test_pipeline_survives_llm_extraction_failure(tmp_path):
    """Regression: when LLM extraction fails, pipeline should still complete
    via rule-based fallback, not crash with KeyError('subject').
    """
    from unittest.mock import patch

    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    fake_llm_config = {"base_url": "http://fake", "api_key": "k", "model_name": "m"}

    with patch(
        "src.operators.kg_ops.llm_extractor._call_llm_extraction",
        side_effect=Exception("LLM unavailable"),
    ):
        result = run_task2_pipeline(
            input_path=source,
            output_dir=tmp_path / "outputs",
            question="高血压有哪些症状？",
            llm_config=fake_llm_config,
        )

    assert result.status == "completed"
    assert result.artifacts["graph"]["triple_count"] > 0


def test_merge_local_model_entities():
    from src.agents.kg_agent.agent import _merge_local_model_entities

    extraction = {
        "status": "completed",
        "record_count": 1,
        "records": [
            {
                "record_id": "record_1",
                "text": "患者有高血压",
                "entities": {
                    "Disease": ["高血压"],
                    "Symptom": ["头晕"],
                    "Drug": [],
                    "Examination": [],
                    "Treatment": [],
                },
                "normalization": {"aliases": {}},
                "mentions": {},
            }
        ],
        "entity_counts": {"Disease": 1, "Symptom": 1, "Drug": 0, "Examination": 0, "Treatment": 0},
        "normalization": {"aliases": {}},
    }

    local_entities = {
        "Disease": ["高血压"],       # already exists, should not duplicate
        "Symptom": ["头痛", "心悸"],  # 头痛 is new, 心悸 is new
        "Drug": ["氨氯地平"],         # entirely new
    }

    result = _merge_local_model_entities(extraction, local_entities)

    record_entities = result["records"][0]["entities"]
    assert "高血压" in record_entities["Disease"]
    assert record_entities["Disease"].count("高血压") == 1  # no duplicate
    assert "头晕" in record_entities["Symptom"]
    assert "头痛" in record_entities["Symptom"]
    assert "心悸" in record_entities["Symptom"]
    assert "氨氯地平" in record_entities["Drug"]
    assert result["entity_counts"]["Drug"] == 1


def test_pipeline_uses_local_model_ner_when_available(tmp_path):
    """When local_model_path is set and predict_kg_entities returns results,
    the pipeline should merge them into the extraction."""
    from unittest.mock import patch

    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    # Create a real directory so _effective_local_model_path passes validation
    model_dir = tmp_path / "fake_model"
    model_dir.mkdir()

    fake_ner_result = {
        "Disease": ["冠心病"],
        "Symptom": ["心悸"],
    }

    with patch(
        "src.agents.kg_agent.agent.predict_kg_entities",
        return_value=fake_ner_result,
    ):
        result = run_task2_pipeline(
            input_path=source,
            output_dir=tmp_path / "outputs",
            question="高血压有哪些症状？",
            local_model_path=str(model_dir),
        )

    assert result.status == "completed"
    # 冠心病 was added by local model, not in original rule-based extraction
    all_diseases = set()
    for record in result.artifacts["extraction"]["records"]:
        all_diseases.update(record["entities"].get("Disease", []))
    assert "冠心病" in all_diseases


def test_resolve_base_model_short_path_no_index_error():
    """Regression: _resolve_base_model must not crash on short paths."""
    from src.operators.kg_ops.local_model_ner import _resolve_base_model

    # Short path with fewer than 3 parent levels
    result = _resolve_base_model("/tmp/not-a-real-adapter")
    assert result == "Qwen/Qwen2.5-0.5B-Instruct"  # falls back gracefully

    # Single-level relative path
    result = _resolve_base_model("adapter")
    assert result == "Qwen/Qwen2.5-0.5B-Instruct"


def test_llm_extractor_fast_fail_on_first_chunk():
    """When LLM is unavailable, extraction should fast-fail after first chunk."""
    from unittest.mock import patch
    from src.operators.kg_ops.llm_extractor import extract_entities_with_llm

    config = {"base_url": "http://fake", "api_key": "k", "timeout": 1.0}
    text = "第一条记录。\n---\n第二条记录。\n---\n第三条记录。"

    with patch(
        "src.operators.kg_ops.llm_extractor._call_llm_extraction",
        side_effect=RuntimeError("LLM unavailable"),
    ):
        result = extract_entities_with_llm(text, config)

    assert result["status"] == "completed"
    assert result["llm_chunks_processed"] == 0  # first chunk failed, rest skipped


def test_pipeline_skips_llm_planning_after_extraction_failure(tmp_path):
    """When LLM extraction fails completely, planner should use rule-based."""
    from unittest.mock import patch

    source = tmp_path / "notes.txt"
    source.write_text(SAMPLE_TEXT, encoding="utf-8")

    config = {"base_url": "http://fake", "api_key": "k", "timeout": 1.0}

    # Mock LLM calls to fail
    with patch(
        "src.operators.kg_ops.llm_extractor._call_llm_extraction",
        side_effect=RuntimeError("LLM down"),
    ):
        result = run_task2_pipeline(
            input_path=source,
            output_dir=tmp_path / "outputs",
            question="高血压有哪些症状？",
            llm_config=config,
        )

    assert result.status == "completed"
    # Planner should have used rule-based mode (not LLM)
    assert result.artifacts["plan"]["planner_mode"] == "rule"


def test_neo4j_write_node_includes_type_property():
    """_write_node must SET n.type so round-trip read preserves the entity type."""
    from unittest.mock import MagicMock
    from src.operators.kg_ops.neo4j_store import _write_node

    tx = MagicMock()
    node = {"type": "Disease", "name": "高血压", "mention_count": 3, "record_ids": ["r1"]}
    _write_node(tx, node)

    # The SET clause should include n.type
    call_args = tx.run.call_args
    cypher = call_args[0][0]
    assert "n.type = $type" in cypher
    kwargs = call_args[1]
    assert kwargs["type"] == "Disease"


def test_neo4j_read_nodes_uses_label_fallback():
    """_read_nodes should use Neo4j labels when type property is missing."""
    from unittest.mock import MagicMock
    from src.operators.kg_ops.neo4j_store import _read_nodes

    mock_session = MagicMock()

    # Use a plain dict-like object for node properties (no 'type' key)
    class FakeNode(dict):
        pass

    node_props = FakeNode({"name": "高血压", "mention_count": 2})

    mock_record = {"n": node_props, "labels": ["Disease"]}
    mock_result = [mock_record]
    mock_session.run.return_value = mock_result

    nodes = _read_nodes(mock_session)
    assert len(nodes) == 1
    assert nodes[0]["id"] == "Disease:高血压"
    assert nodes[0]["type"] == "Disease"


# ---------------------------------------------------------------------------
# Multi-entity QA tests (Disease, Drug, Symptom, Examination, Treatment)
# ---------------------------------------------------------------------------


def test_qa_drug_centric_question():
    """QA should answer drug-centric questions."""
    extraction = extract_medical_entities(SAMPLE_TEXT)
    triples = extract_relations(extraction["records"])
    validation = validate_triples(triples)
    graph = build_medical_graph(validation["triples"], extraction["records"])

    result = answer_graph_question("氨氯地平治疗哪些疾病？", graph)
    assert result["status"] == "answered"
    assert "氨氯地平" in result["answer"]


def test_qa_symptom_centric_question():
    """QA should answer symptom-centric questions."""
    extraction = extract_medical_entities(SAMPLE_TEXT)
    triples = extract_relations(extraction["records"])
    validation = validate_triples(triples)
    graph = build_medical_graph(validation["triples"], extraction["records"])

    result = answer_graph_question("头晕和什么疾病相关？", graph)
    assert result["status"] == "answered"
    assert "头晕" in result["answer"]


def test_qa_disease_centric_still_works():
    """Original disease-centric QA should still work after refactoring."""
    extraction = extract_medical_entities(SAMPLE_TEXT)
    triples = extract_relations(extraction["records"])
    validation = validate_triples(triples)
    graph = build_medical_graph(validation["triples"], extraction["records"])

    result = answer_graph_question("高血压有哪些症状？", graph)
    assert result["status"] == "answered"
    assert "高血压" in result["answer"]


def test_extraction_quality_on_holdout_corpus():
    """The rule-based extractor should keep perfect precision and high recall on
    the held-out annotated corpus, with every miss being a genuine
    out-of-vocabulary entity (precision must stay 1.0 = no spurious entities)."""
    gold_path = ROOT / "benchmarks" / "data" / "kg_extraction_gold.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    report = evaluate_extraction_quality(gold["records"])
    overall = report["overall"]

    assert report["record_count"] == len(gold["records"])
    assert overall["precision"] == 1.0
    # Dictionary expansion lifted held-out recall to >= 0.95 (currently 1.0).
    assert overall["recall"] >= 0.95
    assert overall["f1"] >= 0.95
    # Every false negative must be a real entity the dictionary does not cover,
    # never a precision regression (no false positives anywhere).
    assert all(not diag["false_positives"] for diag in report["records"])


def test_oov_extraction_recall_meets_open_domain_target():
    """Suffix-pattern supplement should recover most OOV gold entities."""
    from src.operators.kg_ops import evaluate_extraction_vocabulary_split

    gold_path = ROOT / "benchmarks" / "data" / "kg_extraction_oov_gold.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    report = evaluate_extraction_vocabulary_split(gold["records"])
    oov = report["vocabulary_split"]["out_of_vocabulary"]

    assert oov["recall"] >= 0.95, report
    assert oov["precision"] >= 0.95, report
    assert report["oov_entity_count"] > 0


def test_evaluate_extraction_quality_penalizes_false_positives():
    """A stub extractor that hallucinates an entity must lose precision, proving
    the metric is not trivially satisfied."""

    def noisy_extractor(text: str):
        return {
            "records": [
                {"entities": {"Disease": ["高血压", "不存在的病"], "Symptom": [],
                              "Drug": [], "Examination": [], "Treatment": []}}
            ]
        }

    gold_records = [{
        "record_id": "g1",
        "text": "高血压",
        "entities": {"Disease": ["高血压"], "Symptom": [], "Drug": [],
                     "Examination": [], "Treatment": []},
    }]

    report = evaluate_extraction_quality(gold_records, extractor=noisy_extractor)
    assert report["overall"]["fp"] == 1
    assert report["overall"]["precision"] == 0.5


def test_relation_quality_on_holdout_corpus():
    """Relation extraction must reach perfect recall and high precision on the
    medically-annotated relation gold; complication-aware drug gating eliminates
    the historical over-pairing of complication-diseases with primary drugs."""
    from src.operators.kg_ops import evaluate_relation_quality

    gold_path = ROOT / "benchmarks" / "data" / "kg_relation_gold.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    report = evaluate_relation_quality(gold["records"], backend="rule")
    overall = report["overall"]

    assert report["record_count"] == len(gold["records"])
    assert overall["recall"] == 1.0
    assert overall["precision"] >= 0.95
    assert overall["f1"] >= 0.97
    # No missing relations.
    assert all(not diag["false_negatives"] for diag in report["records"])


def test_treated_by_gates_complication_disease_from_drugs():
    """When a record explicitly signals one disease as a complication of another
    (合并/并发/伴发), the complication (secondary) disease must not be paired
    with drugs that belong to the primary disease's treatment plan."""

    text = (
        "记录 1:\n老年男性，冠心病合并心力衰竭，活动后气短，"
        "服用阿司匹林及辛伐他汀，行心电图检查。"
    )
    extraction = extract_medical_entities(text)
    triples = extract_relations(extraction["records"])

    # The complication triple must still exist.
    assert any(
        t["subject"] == "心力衰竭"
        and t["predicate"] == "complication_of"
        and t["object"] == "冠心病"
        for t in triples
    )
    # Primary disease keeps the drug pairs.
    treated_drugs_for_primary = {
        t["object"]
        for t in triples
        if t["subject"] == "冠心病" and t["predicate"] == "treated_by"
    }
    assert {"阿司匹林", "辛伐他汀"}.issubset(treated_drugs_for_primary)
    # Complication (secondary) disease must NOT inherit the drugs.
    treated_drugs_for_complication = {
        t["object"]
        for t in triples
        if t["subject"] == "心力衰竭" and t["predicate"] == "treated_by"
    }
    assert not treated_drugs_for_complication


def test_treated_by_gates_symptomatic_drug_clue():
    """A drug followed by an ancillary purpose marker (退热/退烧/止痛/镇痛) is
    used for symptomatic relief, not for treating the disease itself, so no
    ``treated_by`` triple should be emitted for it."""

    text = "记录 1:\n患者哮喘控制不佳，夜间喘息、气短，使用布洛芬退热，复查胸部CT。"
    extraction = extract_medical_entities(text)
    triples = extract_relations(extraction["records"])

    assert any(
        t["subject"] == "支气管哮喘急性发作"
        and t["predicate"] == "has_symptom"
        and t["object"] == "喘息"
        for t in triples
    )
    # 布洛芬 is used for 退热, not asthma treatment.
    assert not any(
        t["subject"] == "支气管哮喘急性发作"
        and t["predicate"] == "treated_by"
        and t["object"] == "布洛芬"
        for t in triples
    )


def test_treated_by_keeps_explicit_drug_for_complication_disease():
    """A complication disease keeps drugs explicitly attributed to it."""

    records = [{
        "record_id": "explicit_complication_treatment",
        "text": "冠心病合并心力衰竭，阿司匹林治疗冠心病，呋塞米治疗心力衰竭。",
        "entities": {
            "Disease": ["冠心病", "心力衰竭"],
            "Symptom": [],
            "Drug": ["阿司匹林", "呋塞米"],
            "Examination": [],
            "Treatment": [],
        },
    }]

    triples = extract_relations(records)
    treated_pairs = {
        (triple["subject"], triple["object"])
        for triple in triples
        if triple["predicate"] == "treated_by"
    }

    assert treated_pairs == {
        ("冠心病", "阿司匹林"),
        ("心力衰竭", "呋塞米"),
    }


def test_treated_by_keeps_later_explicit_use_of_symptomatic_drug():
    """A prior symptomatic use must not hide a later explicit treatment use."""

    records = [{
        "record_id": "mixed_drug_purpose",
        "text": "布洛芬退热后，继续布洛芬治疗类风湿关节炎。",
        "entities": {
            "Disease": ["类风湿关节炎"],
            "Symptom": [],
            "Drug": ["布洛芬"],
            "Examination": [],
            "Treatment": [],
        },
    }]

    triples = extract_relations(records)

    assert any(
        triple["subject"] == "类风湿关节炎"
        and triple["predicate"] == "treated_by"
        and triple["object"] == "布洛芬"
        for triple in triples
    )


def test_treated_by_explicit_attribution_matches_tensor_backend():
    """Rule and tensor paths share explicit drug-disease attribution."""

    records = [{
        "record_id": "explicit_tensor_treatment",
        "text": "冠心病合并心力衰竭，阿司匹林治疗冠心病，呋塞米治疗心力衰竭。",
        "entities": {
            "Disease": ["冠心病", "心力衰竭"],
            "Symptom": [],
            "Drug": ["阿司匹林", "呋塞米"],
            "Examination": [],
            "Treatment": [],
        },
    }]

    rule_pairs = {
        (triple["subject"], triple["predicate"], triple["object"])
        for triple in extract_relations(records)
    }
    tensor_pairs = {
        (triple["subject"], triple["predicate"], triple["object"])
        for triple in extract_relations_tensorized(records, backend="cpu")["triples"]
    }

    assert tensor_pairs == rule_pairs


def test_relation_quality_cpu_backend_matches_rule():
    """The tensorized CPU relation backend must yield identical relation-level
    precision/recall/F1 to the rule backend (semantic equivalence)."""
    from src.operators.kg_ops import evaluate_relation_quality

    gold_path = ROOT / "benchmarks" / "data" / "kg_relation_gold.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    rule = evaluate_relation_quality(gold["records"], backend="rule")["overall"]
    cpu = evaluate_relation_quality(gold["records"], backend="cpu")["overall"]
    assert rule == cpu


def test_evaluate_relation_quality_penalizes_missing_relations():
    """A gold relation the extractor cannot find must lower recall, proving the
    metric is not trivially satisfied."""
    from src.operators.kg_ops import evaluate_relation_quality

    gold_records = [{
        "record_id": "g1",
        "text": "记录: 高血压患者头痛，服用氨氯地平。",
        "relations": [
            {"subject": "高血压", "predicate": "has_symptom", "object": "头痛"},
            # This relation cannot be extracted (entity absent from text).
            {"subject": "高血压", "predicate": "treated_by", "object": "不存在的药"},
        ],
    }]

    report = evaluate_relation_quality(gold_records, backend="rule")
    assert report["overall"]["fn"] >= 1
    assert report["overall"]["recall"] < 1.0
