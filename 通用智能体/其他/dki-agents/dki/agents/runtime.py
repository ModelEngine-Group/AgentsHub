"""Nexent 智能体运行时：模型构建、提示词装配、Agent 工厂与同步运行。

与 Nexent SDK 的对接点：
- CoreAgent（nexent.core.agents 公共入口）：ReAct 循环 + <code> 动作协议 + observer 流式；
- OpenAIModel（nexent.core.models.openai_llm）：流式 LLM 封装（token 级 observer 推送）；
- MessageObserver / ProcessType（nexent.core.utils.observer）：过程消息总线。

另提供 ScriptedModel：预置响应序列的离线模型（smolagents Model 子类），
用于无 LLM 环境下确定性演示/测试完整智能体循环。
"""

import copy
import functools
import inspect
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from dki.llm import LlmNotConfigured, llm_available

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_CODE_TAG_PATTERN = re.compile(
    r"<code>[ \t]*\r?\n?(.*?)\r?\n?[ \t]*</code>",
    flags=re.DOTALL | re.IGNORECASE,
)
def _translate_code_tags_for_nexent_012(content: str) -> str:
    """Bridge DKI's ``<code>`` contract to a Markdown-fence Nexent executor.

    Public prompts, examples, and ScriptedModel intentionally use ``<code>``
    tags. Nexent builds that delegate parsing to smolagents expect Markdown
    Python fences instead. Translate only action tags at the model boundary;
    ordinary fenced text remains untouched and the project-level contract stays
    stable. Applied only when :func:`_executor_parses_code_tags` reports that
    the installed executor does *not* accept ``<code>`` itself.
    """

    def replacement(match: re.Match[str]) -> str:
        code = match.group(1).strip("\r\n")
        return f"```py\n{code}\n```"

    return _CODE_TAG_PATTERN.sub(replacement, content)


class _Nexent012CodeProtocolAdapter:
    """Callable proxy that preserves model attributes and adapts code syntax."""

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        message = self._model(*args, **kwargs)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            translated = _translate_code_tags_for_nexent_012(content)
            if translated != content:
                message = copy.copy(message)
                message.content = translated
        return message


@functools.lru_cache(maxsize=1)
def _executor_parses_code_tags() -> bool:
    """Whether the installed Nexent executor natively parses ``<code>`` actions.

    Nexent builds disagree on the action syntax. Some ship their own
    ``parse_code_blobs`` that accepts only ``<code>`` and deliberately refuses
    Markdown fences, so that Python examples inside retrieved documents are
    never executed; translating to ``py`` fences for those builds makes every
    action unparseable, which the executor reports as ``FinalAnswerError`` —
    the model's text is returned verbatim and no tool ever runs. Other builds
    delegate to smolagents and expect fences. Probe the exact function the ReAct
    step calls instead of assuming either contract.
    """

    probe = 'final_answer("probe")'
    try:
        from nexent.core.agents.core_agent import parse_code_blobs
    except ImportError:
        return False
    try:
        return parse_code_blobs(f"<code>\n{probe}\n</code>").strip() == probe
    except Exception:
        return False


# ---------------------------------------------------------------- observer / model
def build_observer(lang: str = "zh"):
    from nexent.core.utils.observer import MessageObserver

    return MessageObserver(lang=lang)


def build_model(observer=None):
    """从环境配置构建 Nexent OpenAIModel（观察器与智能体共享）。"""
    from dki.paths import LLM_API_KEY, LLM_API_URL, LLM_MODEL_NAME

    if not llm_available():
        raise LlmNotConfigured(
            "未配置 LLM。请在 .env 设置 LLM_API_KEY / LLM_MODEL_NAME / LLM_API_URL。"
        )
    from nexent.core.models.openai_llm import OpenAIModel

    observer = observer or build_observer()
    return OpenAIModel(
        observer=observer,
        model_id=LLM_MODEL_NAME,
        api_key=LLM_API_KEY,
        api_base=LLM_API_URL,
        temperature=0.2,
        top_p=0.95,
        ssl_verify=True,
    )


class ScriptedModel:
    """预置响应的离线模型（兼容 smolagents Model 接口）。

    responses 按调用顺序弹出；耗尽后返回一条兜底 final_answer，避免死循环。
    """

    def __new__(cls, responses: List[str]):
        # 动态继承，避免模块导入期强依赖 smolagents
        from smolagents.models import ChatMessage, MessageRole, Model

        class _Scripted(Model):
            def __init__(self, scripted_responses: List[str]):
                super().__init__(model_id="scripted-offline")
                self._responses = list(scripted_responses)
                self._idx = 0
                self.stop_event = threading.Event()  # 兼容 nexent 对 stop_event 的访问

            def generate(self, messages, stop_sequences=None, response_format=None,
                         tools_to_call_from=None, **kwargs) -> ChatMessage:
                if self._idx < len(self._responses):
                    content = self._responses[self._idx]
                    self._idx += 1
                else:
                    content = ('任务结束。\n<code>\nfinal_answer("（脚本响应已耗尽）")\n</code>')
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=content,
                )

            def __call__(
                self,
                messages,
                stop_sequences=None,
                grammar=None,
                tools_to_call_from=None,
                **kwargs,
            ) -> ChatMessage:
                # smolagents bundled by nexent==0.1.2 invokes Model directly;
                # its base __call__ is a no-op.  Keep generate() as the newer
                # compatibility surface and bridge the pinned SDK to it.
                return self.generate(
                    messages,
                    stop_sequences=stop_sequences,
                    tools_to_call_from=tools_to_call_from,
                    grammar=grammar,
                    **kwargs,
                )

        return _Scripted(responses)


# ---------------------------------------------------------------- 提示词装配
def load_agent_prompt(name: str) -> Dict[str, str]:
    """加载某个智能体的 duty/constraint/few_shots 三段提示词。"""
    path = PROMPTS_DIR / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        "duty": data.get("duty", ""),
        "constraint": data.get("constraint", ""),
        "few_shots": data.get("few_shots", ""),
    }


def build_prompt_templates(agent_prompt: Dict[str, str]) -> Dict[str, Any]:
    """把三段提示词注入基础模板，生成 smolagents PromptTemplates 结构。

    注意：CoreAgent 将 code_block_tags 置空并以 parse_code_blobs 解析 <code></code>，
    因此 <code> 标签在模板中硬编码（与 nexent backend 模板一致）；
    {{ tools }} 循环保留给 smolagents 的 populate_template 渲染。
    """
    from smolagents.agents import EMPTY_PROMPT_TEMPLATES

    base = yaml.safe_load(
        (PROMPTS_DIR / "base_system_prompt.yaml").read_text(encoding="utf-8")
    )["system_prompt"]
    system_prompt = (
        base.replace("__DUTY__", agent_prompt["duty"])
        .replace("__CONSTRAINT__", agent_prompt["constraint"])
        .replace("__FEW_SHOTS__", agent_prompt["few_shots"])
    )
    templates = copy.deepcopy(EMPTY_PROMPT_TEMPLATES)
    templates["system_prompt"] = system_prompt
    return templates


# ---------------------------------------------------------------- Agent 工厂
def _core_agent_accepts_context_runtime(core_agent: type[Any]) -> bool:
    """Return whether a CoreAgent inheritance chain explicitly supports it.

    Some Nexent distributions expose ``CoreAgent.__init__(*args, **kwargs)``
    while delegating to a smolagents base class that rejects unknown keyword
    arguments. A wrapper-level ``**kwargs`` is therefore not enough evidence
    that ``context_runtime`` is safe to pass; require either an explicit
    parameter on some implementation in the MRO, or an implementation whose
    ``**kwargs`` body demonstrably consumes ``context_runtime`` (newer builds
    ``kwargs.pop("context_runtime")`` without a named parameter and otherwise
    fail closed at run time with ``UnconfiguredContextRuntime``).
    """

    for implementation in getattr(core_agent, "__mro__", (core_agent,)):
        init = implementation.__dict__.get("__init__")
        if init is None:
            continue
        try:
            parameters = inspect.signature(init).parameters
        except (TypeError, ValueError):
            continue
        if "context_runtime" in parameters:
            return True
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            try:
                source = inspect.getsource(init)
            except (OSError, TypeError):
                continue
            if "context_runtime" in source:
                return True
    return False


def _context_runtime_kwargs(core_agent: type[Any]) -> dict[str, Any]:
    """Provide the optional legacy runtime only to compatible SDK builds."""

    if not _core_agent_accepts_context_runtime(core_agent):
        return {}
    try:
        from nexent.core.context_runtime.legacy.runtime import LegacyContextRuntime
    except ImportError:
        return {}
    return {"context_runtime": LegacyContextRuntime()}


def build_agent(
    name: str,
    description: str,
    tools: List[Any],
    prompt_name: str,
    model=None,
    observer=None,
    max_steps: int = 8,
):
    """构建一个 Nexent CoreAgent（工具 + 三段式提示词 + observer）。"""
    # PyPI nexent==0.1.2 exports CoreAgent from this public package.  Its
    # internal ``agents.core_agent`` and ``context_runtime`` paths do not
    # exist, so importing those repository-only paths breaks a clean install.
    from nexent.core.agents import CoreAgent

    observer = observer or build_observer()

    # nexent==0.1.2 的 CoreAgent 默认使用 UnconfiguredContextRuntime
    # （fail-closed，运行时报 "CoreAgent requires a context runtime"）。
    # 迁移前 build_agent 显式注入 LegacyContextRuntime；此处恢复该注入。
    # clean install 可能不暴露内部 context_runtime 路径，导入失败时保持
    # 不注入（由 SDK 默认路径兜底），与迁移后现状一致。
    observer = observer or build_observer()
    if model is None:
        model = build_model(observer)
    if not _executor_parses_code_tags():
        model = _Nexent012CodeProtocolAdapter(model)
    # 工具挂 observer（nexent 风格：工具过程消息也进 observer）
    for t in tools:
        if hasattr(t, "observer"):
            t.observer = observer

    templates = build_prompt_templates(load_agent_prompt(prompt_name))
    agent_kwargs = {
        "observer": observer,
        "prompt_templates": templates,
        "tools": tools,
        "model": model,
        "name": name,
        "description": description,
        "max_steps": max_steps,
        "additional_authorized_imports": ["json", "re", "statistics"],
    }
    agent_kwargs.update(_context_runtime_kwargs(CoreAgent))
    agent = CoreAgent(**agent_kwargs)
    return agent


# ---------------------------------------------------------------- 同步运行 + 过程渲染
_TYPE_LABEL = {
    "step_count": "",
    "parse": "",
    "execution_logs": "",
    "final_answer": "\n━━━ 最终回答 ━━━\n",
    "error": "\n[错误] ",
    "tool": "[工具] ",
    "agent_new_run": "",
    "token_count": None,           # 不展示
    "model_output_thinking": None,  # token 流不逐条展示（避免刷屏）
    "model_output_code": None,
    "model_output_deep_thinking": None,
}


def _render_messages(raw_messages: List[str]) -> Optional[str]:
    """把 observer 的 JSON 消息渲染到终端；返回捕获到的 final_answer。"""
    final = None
    for raw in raw_messages:
        try:
            msg = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        mtype, content = msg.get("type"), msg.get("content", "")
        label = _TYPE_LABEL.get(mtype, "")
        if label is None:
            continue
        if mtype == "final_answer":
            final = content
        text = f"{label}{content}" if isinstance(content, str) else f"{label}{content}"
        print(text)
    return final


def run_agent_sync(agent, task: str) -> str:
    """同步运行智能体：过程消息实时渲染，返回最终回答。"""
    result_box: Dict[str, Any] = {}

    def _run():
        try:
            result_box["answer"] = agent.run(task, stream=False)
        except Exception as e:  # 异常也走 observer 渲染 + 返回
            result_box["error"] = str(e)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    final_from_obs = None
    while worker.is_alive():
        got = _render_messages(agent.observer.get_cached_message())
        final_from_obs = got or final_from_obs
        time.sleep(0.3)
    worker.join()
    got = _render_messages(agent.observer.get_cached_message())
    final_from_obs = got or final_from_obs

    if "error" in result_box:
        return f"智能体运行出错: {result_box['error']}"
    answer = result_box.get("answer")
    return str(answer) if answer is not None else (final_from_obs or "")
