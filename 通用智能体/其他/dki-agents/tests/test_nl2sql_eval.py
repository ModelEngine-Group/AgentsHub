"""NL2SQL 准确率验收测试（评测集 accuracy ≥ 0.85 为赛题验收线）。

依赖真实 demo 数据（data/raw/structured + outputs/kg）——若不存在则现场生成小语料。
"""

import sqlite3

import pytest

from dki.analysis.database import build_database
from dki.analysis.nl2sql_eval import (
    diagnose_candidate_set_with_gold,
    evaluate,
    evaluate_set,
)
from dki.kg.store import KnowledgeGraph
from dki.paths import OUTPUT_DIR, RAW_DATA_DIR


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory):
    visits = RAW_DATA_DIR / "structured" / "visits.csv"
    kg_path = OUTPUT_DIR / "kg" / "medical_kg.json"
    if not visits.exists():
        pytest.skip("请先运行 python -m dki gen-data 生成数据")
    kg = KnowledgeGraph.load(kg_path) if kg_path.exists() else None
    if kg is None:
        # 现场构建小图谱（评测集图谱题依赖）
        from dki.kg.build import build_kg_from_samples
        from dki.pipeline import LocalPipelineRunner, samples_from_dir
        from dki.pipeline.config import load_pipeline_config
        from dki.paths import PROJECT_ROOT

        process, _ = load_pipeline_config(PROJECT_ROOT / "configs" / "pipeline_kg_build.yaml")
        samples = samples_from_dir(RAW_DATA_DIR, pattern="*.txt")[:60]
        kg = build_kg_from_samples(LocalPipelineRunner().run(process, samples).samples)
    db = tmp_path_factory.mktemp("nl2sql") / "insight.db"
    build_database(db_path=db, visits_csv=visits,
                   patients_csv=RAW_DATA_DIR / "structured" / "patients.csv", kg=kg)
    return db


def test_rule_engine_accuracy_above_085(demo_db):
    res = evaluate(engine="rule", db_path=demo_db)
    print(f"\nNL2SQL rule accuracy: {res['accuracy']:.2%} ({res['n_correct']}/{res['n']})")
    for d in res["details"]:
        if not d.get("correct"):
            print(f"  [X] {d['question']} -> {d.get('pred_sql')}")
    assert res["n"] >= 40, "评测集规模不足"
    assert res["accuracy"] >= 0.85, f"NL2SQL 规则引擎准确率 {res['accuracy']:.2%} 低于验收线 85%"


def test_layered_eval_set_size_and_gold_executes(demo_db):
    """难题集扩充：合并集 ≥120 题、分难度、且每条 gold SQL 可在真实库上执行。"""
    from dki.analysis.database import run_sql
    from dki.analysis.nl2sql_eval import evaluate_layered, _load_items, HARD_EVAL_SET

    hard = _load_items(HARD_EVAL_SET)
    assert len(hard) >= 70, f"难题集规模不足：{len(hard)}"
    # 每条 gold 必须可执行（防脏 gold）
    for it in hard:
        run_sql(it["gold_sql"], demo_db)  # 不抛异常即通过
        assert it.get("difficulty") in {"easy", "medium", "hard"}

    layered = evaluate_layered(engine="rule", db_path=demo_db)
    assert layered["n"] >= 120, f"合并评测集规模不足：{layered['n']}"
    # 规则引擎在 easy/medium 层应保持高准确（hard 层留给 LLM 管线）。
    # 用 >=0.85 容差而非精确等于，兼容 fixture 小图谱下个别图谱题的行数差异。
    by = layered["by_difficulty"]
    assert by["easy"]["accuracy"] >= 0.85, f"easy 层规则准确率偏低：{by['easy']}"
    if "medium" in by:
        assert by["medium"]["accuracy"] >= 0.85, f"medium 层偏低：{by['medium']}"
    print(f"\nlayered rule EX={layered['accuracy']:.2%}; by_difficulty={by}")


class _AgentResult:
    def __init__(self, result):
        self.result = result

    def query(self, question):
        return dict(self.result)


def _small_eval_database(path):
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE patients(patient_id TEXT, region TEXT)")
    connection.executemany(
        "INSERT INTO patients VALUES (?, ?)",
        [("p1", "北京"), ("p2", "上海"), ("p3", "北京")],
    )
    connection.commit()
    connection.close()


def test_eval_carries_private_diagnostics_and_aggregates_capabilities(tmp_path):
    database = tmp_path / "eval.sqlite"
    _small_eval_database(database)
    sql = "SELECT region, COUNT(*) FROM patients GROUP BY region"
    agent = _AgentResult(
        {
            "sql": sql,
            "engine": "llm",
            "llm_meta": {"n_candidates": 2},
            "candidate_diagnostics": [{"sql": sql}, {"sql": "SELECT region FROM patients"}],
        }
    )

    result = evaluate_set(
        [
            {
                "id": "cap-1",
                "question": "返回每个地区的患者数",
                "gold_sql": sql,
                "difficulty": "hard",
                "category": "aggregation_grouping",
                "capabilities": ["grain", "output contract"],
            }
        ],
        "agent",
        database,
        nl=agent,
        strict=True,
    )

    detail = result["details"][0]
    assert detail["llm_meta"] == {"n_candidates": 2}
    assert len(detail["candidate_diagnostics"]) == 2
    assert detail["capabilities"] == ("grain", "output_contract")
    assert result["by_category"]["aggregation_grouping"]["accuracy"] == 1.0
    assert result["by_capability"]["grain"]["n"] == 1
    assert result["by_capability"]["output_contract"]["correct"] == 1
    assert "oracle_hit" not in detail
    assert "selector_regret" not in detail


def test_gold_aware_candidate_diagnostic_is_explicit_and_reports_regret(tmp_path):
    database = tmp_path / "candidates.sqlite"
    _small_eval_database(database)
    gold_sql = "SELECT patient_id FROM patients WHERE region='北京'"
    selected_sql = "SELECT patient_id FROM patients WHERE region='上海'"
    query_result = {
        "sql": selected_sql,
        "engine": "llm",
        "selected_candidate_index": 0,
        "candidate_diagnostics": [
            {"sql": selected_sql},
            {"sql": gold_sql},
            {"sql": gold_sql},
            {"sql": "SELECT missing FROM patients"},
        ],
    }

    diagnostic = diagnose_candidate_set_with_gold(
        query_result,
        gold_sql,
        database,
        question="返回北京患者编号",
        strict=True,
    )

    assert diagnostic["oracle_hit"] is True
    assert diagnostic["selected_hit"] is False
    assert diagnostic["selector_regret"] is True
    assert diagnostic["candidate_count"] == 4
    assert diagnostic["valid_candidate_count"] == 3
    assert diagnostic["unique_candidate_count"] == 2


def test_auto_engine_alias_uses_agent_query(tmp_path):
    database = tmp_path / "auto.sqlite"
    _small_eval_database(database)
    sql = "SELECT patient_id FROM patients"

    result = evaluate_set(
        [{"question": "返回患者编号", "gold_sql": sql}],
        "auto",
        database,
        nl=_AgentResult({"sql": sql, "engine": "llm"}),
        strict=True,
    )

    assert result["engine"] == "auto"
    assert result["accuracy"] == 1.0
    assert result["details"][0]["engine"] == "llm"
