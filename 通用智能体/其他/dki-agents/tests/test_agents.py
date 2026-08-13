"""智能体与工具测试（不依赖网络/LLM：ScriptedModel 驱动）。"""

import inspect
import json

import pytest

from dki.agents.factory import build_data_processing_agent
from dki.agents.runtime import (
    ScriptedModel,
    _core_agent_accepts_context_runtime,
    _translate_code_tags_for_nexent_012,
    build_observer,
    run_agent_sync,
)
from dki.agents.tools import ListOperatorsTool, PreviewDataTool, RunPipelineTool, TaskStatusTool
from dki.llm import LlmNotConfigured


def test_nexent_012_public_agent_api_signatures():
    """Pin public Nexent imports without constructing a network client."""

    from nexent.core.agents import CoreAgent
    from nexent.core.models.openai_llm import OpenAIModel
    from nexent.core.utils.observer import MessageObserver

    observer_parameters = inspect.signature(
        MessageObserver
    ).parameters
    core_parameters = inspect.signature(CoreAgent).parameters
    model_parameters = inspect.signature(OpenAIModel).parameters
    run_parameters = inspect.signature(CoreAgent.run).parameters

    assert "lang" in observer_parameters
    assert {"observer", "prompt_templates"} <= set(core_parameters)
    assert {"observer", "temperature", "top_p"} <= set(model_parameters)
    assert {"task", "stream", "reset", "images", "additional_args", "max_steps"} <= set(
        run_parameters
    )
    assert any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in model_parameters.values()
    )
    assert any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in model_parameters.values()
    )


def test_context_runtime_requires_explicit_terminal_sdk_support():
    class _TerminalWithoutSupport:
        def __init__(self, *args, **kwargs):
            pass

    class _WrapperWithoutSupport(_TerminalWithoutSupport):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class _TerminalWithSupport(_TerminalWithoutSupport):
        def __init__(self, *args, context_runtime=None, **kwargs):
            super().__init__(*args, **kwargs)

    assert not _core_agent_accepts_context_runtime(_WrapperWithoutSupport)
    assert _core_agent_accepts_context_runtime(_TerminalWithSupport)


def test_code_protocol_adapter_preserves_contract_and_emits_runtime_fence():
    tagged = '思考。\n<code>\nfinal_answer("ok")\n</code>'
    translated = _translate_code_tags_for_nexent_012(tagged)
    assert translated == '思考。\n```py\nfinal_answer("ok")\n```'
    already_fenced = '说明。\n```py\nprint("not an action tag")\n```'
    assert _translate_code_tags_for_nexent_012(already_fenced) == already_fenced


def test_list_operators_tool():
    out = json.loads(ListOperatorsTool().forward(category="medical"))
    assert out["ok"]
    names = {m["op"] for m in out["medical_operators"]}
    assert names == {
        "MedicalTextCleaner", "MedicalSectionSplitter", "MedicalEntityExtractor",
        "MedicalRelationExtractor", "TripleBuilderValidator",
    }


def test_preview_tool_path_guard():
    out = json.loads(PreviewDataTool().forward(path="../nexent"))
    assert not out["ok"] and "越界" in out["error"]


def test_run_pipeline_and_status_tools(tmp_path):
    # 在 data/ 允许范围外造数据不行——用 data/work 临时目录
    from dki.paths import DATA_DIR

    work = DATA_DIR / "work" / "test_agents"
    work.mkdir(parents=True, exist_ok=True)
    (work / "a.txt").write_text("主诉：头痛。<p>x</p>", encoding="utf-8")
    pipeline = json.dumps([{"op": "MedicalTextCleaner", "params": {}}])
    out = json.loads(RunPipelineTool().forward(
        pipeline=pipeline, input_dir="data/work/test_agents", name="tool-test"))
    assert out["ok"] and out["status"] in {"PENDING", "RUNNING"}
    st = json.loads(TaskStatusTool().forward(task_id=out["task_id"], wait_seconds=5))
    assert st["ok"] and st["task"]["status"] == "COMPLETED"


def test_run_pipeline_bad_op_returns_hint():
    out = json.loads(RunPipelineTool().forward(
        pipeline=json.dumps([{"op": "NoSuchOp", "params": {}}]), input_dir="data/raw"))
    assert not out["ok"] and "list_operators" in out["hint"]


def test_scripted_agent_full_react_loop():
    """ScriptedModel 驱动 CoreAgent 完成 侦察→执行→查状态→final_answer 四步闭环。"""
    responses = [
        '思考：先查算子。\n<code>\nprint(list_operators(category="medical")[:200])\n</code>',
        '思考：提交流水线。\n<code>\nimport json\n'
        'p = [{"op": "MedicalTextCleaner", "params": {}}]\n'
        'r = json.loads(run_pipeline(pipeline=json.dumps(p), input_dir="data/raw", '
        'pattern="emr_00[01]*.txt", name="pytest-agent"))\nprint(r["status"])\ntid = r["task_id"]\n</code>',
        '思考：查状态并收尾。\n<code>\nimport json\n'
        'st = json.loads(get_task_status(task_id=tid, wait_seconds=5))\n'
        'final_answer("完成:" + st["task"]["status"])\n</code>',
    ]
    agent = build_data_processing_agent(
        model=ScriptedModel(responses), observer=build_observer(), max_steps=5)
    answer = run_agent_sync(agent, "清洗部分病历")
    assert "完成:COMPLETED" in str(answer)


def _patch_nl2sql(monkeypatch, payload):
    """让 Nl2SqlTool 走完整 forward 路径，但用给定结果替换真实引擎与建库。"""
    import dki.agents.tools.insight_tools as insight_tools
    import dki.analysis.nl2sql as nl2sql_module

    monkeypatch.setattr(insight_tools, "_ensure_db", lambda: None)

    class _Engine:
        def __init__(self, *_args, **_kwargs):
            pass

        def query(self, _question):
            return dict(payload)

    monkeypatch.setattr(nl2sql_module, "NL2SQL", _Engine)


def test_nl2sql_tool_preserves_terminal_abstention(monkeypatch):
    """终止性弃权必须原样透传，且不得给出"换个问法"的重试提示。"""
    from dki.agents.tools.insight_tools import Nl2SqlTool

    _patch_nl2sql(monkeypatch, {
        "question": "q", "sql": "", "engine": "none", "confidence": 0.0,
        "validated": False, "columns": [], "rows": [],
        "error": "该问题包含规则引擎无法完整表达的限定条件",
        "terminal": True, "disposition": "blocked",
    })

    out = json.loads(Nl2SqlTool().forward(question="q"))

    assert out["ok"] is False
    assert out["terminal"] is True
    assert out["disposition"] == "blocked"
    # 绝不能诱导智能体改写问法或放宽限定条件重试
    assert "换一种" not in out["hint"]
    assert "不要删减" in out["hint"]
    # 终止结论不得携带任何可被误用为结果的载荷
    assert "sql" not in out and "rows" not in out


def test_nl2sql_tool_marks_fenced_execution_failure_terminal(monkeypatch):
    """专属规则 SQL 执行失败同样是终止性的，不能回退成通用查询。"""
    from dki.agents.tools.insight_tools import Nl2SqlTool

    _patch_nl2sql(monkeypatch, {
        "question": "q", "sql": "", "engine": "none", "confidence": 0.0,
        "validated": False, "columns": [], "rows": [],
        "error": "该问题的专属规则 SQL 无法执行，不能交由通用回退处理",
        "terminal": True, "disposition": "fenced_execution_failure",
    })

    out = json.loads(Nl2SqlTool().forward(question="q"))

    assert out["ok"] is False and out["terminal"] is True
    assert out["disposition"] == "fenced_execution_failure"


def test_nl2sql_tool_keeps_retry_hint_for_recoverable_errors(monkeypatch):
    """普通未命中仍应保留可重试提示，终止语义不能误伤正常兜底。"""
    from dki.agents.tools.insight_tools import Nl2SqlTool

    _patch_nl2sql(monkeypatch, {
        "question": "q", "sql": "", "engine": "none", "confidence": 0.0,
        "validated": False, "columns": [], "rows": [],
        "error": "无法解析该问题（规则未命中且 LLM 未配置）",
    })

    out = json.loads(Nl2SqlTool().forward(question="q"))

    assert out["ok"] is False
    assert out.get("terminal") is None
    assert "换一种" in out["hint"]


def test_nl2sql_tool_passes_through_successful_query(monkeypatch):
    """成功路径不受终止性标记影响。"""
    from dki.agents.tools.insight_tools import Nl2SqlTool

    _patch_nl2sql(monkeypatch, {
        "question": "q", "sql": "SELECT 1", "engine": "rule", "confidence": 0.9,
        "validated": True, "columns": ["n"], "rows": [[1]],
    })

    out = json.loads(Nl2SqlTool().forward(question="q"))

    assert out["ok"] is True
    assert out["sql"] == "SELECT 1" and out["rows"] == [[1]]


def test_base_prompt_forbids_retrying_terminal_results():
    """系统提示必须区分可重试失败与终止性弃权。"""
    from dki.agents.runtime import build_prompt_templates

    system_prompt = build_prompt_templates(
        {"duty": "", "constraint": "", "few_shots": ""}
    )["system_prompt"]

    assert '"terminal": true' in system_prompt
    assert "禁止删减、放宽或忽略任何限定条件" in system_prompt


def test_llm_not_configured(monkeypatch):
    import dki.paths as paths

    monkeypatch.setattr(paths, "LLM_API_KEY", "")
    from dki import llm

    assert not llm.llm_available()
    with pytest.raises(LlmNotConfigured):
        llm.chat("hi")
