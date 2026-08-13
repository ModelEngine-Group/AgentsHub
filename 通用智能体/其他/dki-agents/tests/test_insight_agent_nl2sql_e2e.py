"""Insight Agent -> Nl2SqlTool -> NL2SQL 的隔离端到端测试。"""

from __future__ import annotations

import json
import sqlite3

import pytest

from dki.agents.factory import build_insight_agent
from dki.agents.runtime import ScriptedModel, build_observer, run_agent_sync
from dki.agents.tools.insight_tools import Nl2SqlTool
from dki.analysis.nl2sql import NL2SQL


@pytest.fixture
def isolated_insight_db(tmp_path, monkeypatch):
    """建立最小只供本测试使用的数据库，并隔离工具的自动建库入口。"""
    db_path = tmp_path / "insight-agent-e2e.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT,
                gender TEXT,
                age INTEGER,
                region TEXT
            );
            CREATE TABLE visits (
                visit_id TEXT PRIMARY KEY,
                patient_id TEXT,
                visit_date TEXT,
                department TEXT,
                disease TEXT,
                drug_prescribed TEXT,
                exam TEXT,
                cost REAL,
                los_days INTEGER
            );
            CREATE TABLE kg_entities (
                name TEXT,
                type TEXT,
                mention_count INTEGER
            );
            CREATE TABLE kg_triples (
                head TEXT,
                head_type TEXT,
                relation TEXT,
                relation_zh TEXT,
                tail TEXT,
                tail_type TEXT,
                confidence REAL,
                count INTEGER
            );
            INSERT INTO patients VALUES
                ('P001', '张三', '男', 45, '北京'),
                ('P002', '李四', '女', 37, '上海');
            INSERT INTO visits VALUES
                ('V001', 'P001', '2025-01-01', '心内科', '高血压',
                 '氨氯地平', '血压', 120.0, 0),
                ('V002', 'P001', '2025-01-10', '心内科', '高血压',
                 '氨氯地平', '心电图', 180.0, 0),
                ('V003', 'P002', '2025-01-12', '神经内科', '偏头痛',
                 '布洛芬', '头颅CT', 240.0, 0);
            INSERT INTO kg_entities VALUES ('高血压', 'Disease', 2);
            INSERT INTO kg_triples VALUES
                ('高血压', 'Disease', 'treated_by', '治疗药物', '氨氯地平',
                 'Drug', 0.95, 2);
            """
        )

    import dki.agents.tools.insight_tools as insight_tools

    monkeypatch.setattr(insight_tools, "DB_PATH", db_path)
    monkeypatch.setattr(insight_tools, "_ensure_db", lambda: None)
    return db_path


def test_nl2sql_tool_returns_success_json_and_queries_once(
    isolated_insight_db, monkeypatch
):
    calls = []
    original_query = NL2SQL.query

    def counted_query(self, question):
        calls.append(question)
        return original_query(self, question)

    monkeypatch.setattr(NL2SQL, "query", counted_query)

    payload = json.loads(Nl2SqlTool().forward(question="按科室统计就诊量"))

    assert payload["ok"] is True
    assert payload["engine"] == "rule"
    assert payload["validated"] is True
    assert payload["columns"] == ["department", "visit_count"]
    assert payload["rows"] == [["心内科", 2], ["神经内科", 1]]
    assert calls == ["按科室统计就诊量"]


def test_nl2sql_tool_returns_json_for_rule_refusal(isolated_insight_db):
    payload = json.loads(
        Nl2SqlTool().forward(
            question="统计2025-01-01、2025-02-01和2025-03-01三天的就诊量"
        )
    )

    assert payload["ok"] is False
    assert payload["error"]
    assert payload["hint"]


def test_nl2sql_tool_wraps_setup_exception_as_error_json(
    isolated_insight_db, monkeypatch
):
    import dki.agents.tools.insight_tools as insight_tools

    def fail_to_prepare_database():
        raise RuntimeError("isolated database unavailable")

    monkeypatch.setattr(insight_tools, "_ensure_db", fail_to_prepare_database)

    payload = json.loads(Nl2SqlTool().forward(question="按科室统计就诊量"))

    assert payload["ok"] is False
    assert payload["error"] == "isolated database unavailable"
    assert payload["hint"]


def test_scripted_insight_agent_completes_nl2sql_loop_once(
    isolated_insight_db, monkeypatch
):
    calls = []
    original_query = NL2SQL.query

    def counted_query(self, question):
        calls.append(question)
        return original_query(self, question)

    monkeypatch.setattr(NL2SQL, "query", counted_query)
    responses = [
        (
            "思考：先调用 NL2SQL 工具取得可验证结果，再直接总结。\n"
            "<code>\n"
            "import json\n"
            'result = json.loads(nl2sql_query(question="按科室统计就诊量"))\n'
            'final_answer("查询完成：" + result["rows"][0][0] + "=" + '
            'str(result["rows"][0][1]))\n'
            "</code>"
        )
    ]
    agent = build_insight_agent(
        model=ScriptedModel(responses),
        observer=build_observer(),
        max_steps=3,
    )

    answer = run_agent_sync(agent, "请按科室统计就诊量，并告诉我数量最多的科室。")

    assert "查询完成：心内科=2" in str(answer)
    assert calls == ["按科室统计就诊量"]
