"""答疑卡片渲染。

为什么要单独渲染而不用模型自由发挥：课堂上最影响信任的是结构不稳定。
同一类问题每次输出位置不同的段落，学生就无法快速定位「提示」与「验算证据」。
本模块把卡片固定为五段，字段缺省时留占位，保证任何一次输出都可预期。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

__all__ = ["render_tutor_card", "SECTION_TITLES"]

SECTION_TITLES = ("当前卡点", "涉及知识点", "分层提示", "验算证据", "下一步")

_LEVEL_NAMES = {
    1: "第一步 · 方向",
    2: "第二步 · 关键步骤",
    3: "第三步 · 完整解答",
}


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _format_topic(topic: Any) -> str:
    if isinstance(topic, Mapping):
        code = str(topic.get("code") or "").strip()
        title = str(topic.get("title") or topic.get("name") or "").strip()
        if code and title:
            return f"【{code}】{title}"
        return title or code or "未标注"
    text = str(topic or "").strip()
    return text or "未标注"


def _format_hint(index: int, hint: Any) -> str:
    if isinstance(hint, Mapping):
        level = hint.get("level")
        text = str(hint.get("text") or hint.get("content") or "").strip()
    else:
        level, text = index, str(hint).strip()
    try:
        label = _LEVEL_NAMES.get(int(level), f"第 {level} 步")
    except (TypeError, ValueError):
        label = f"第 {index} 步"
    return f"- **{label}**：{text}"


def _format_verification(item: Any) -> str:
    if not isinstance(item, Mapping):
        return f"- {item}"
    tool = str(item.get("tool") or "手工推导").strip()
    claim = str(item.get("claim") or item.get("statement") or "").strip()
    result = item.get("result")
    if isinstance(result, Mapping):
        verdict = result.get("verdict") or ""
        value = result.get("value")
        summary = result.get("summary") or ""
        detail_parts = []
        if verdict:
            detail_parts.append(str(verdict))
        if value is not None and value != "":
            detail_parts.append(f"值 {value}")
        if summary:
            detail_parts.append(str(summary))
        detail = "；".join(detail_parts)
    else:
        detail = str(result or "").strip()
    claim_part = f"{claim} —— " if claim else ""
    tool_part = f"（`{tool}`）" if tool else ""
    return f"- {claim_part}{detail}{tool_part}".rstrip()


def render_tutor_card(payload: Mapping[str, Any]) -> dict:
    """把结构化答疑结果渲染成五段式 Markdown 卡片。

    必需字段：topic、blocker。其余缺省时输出占位说明，不抛异常——
    宁可少一段内容，也不要因为字段缺失让整次答疑失败。
    """
    if not isinstance(payload, Mapping):
        raise ValueError("payload 必须是对象")

    topic = payload.get("topic")
    blocker = str(payload.get("blocker") or "").strip()
    if not blocker:
        raise ValueError("缺少 blocker 字段：请先用一句话说明学生卡在哪里")

    knowledge_points = _as_list(payload.get("knowledge_points"))
    hints = _as_list(payload.get("hints"))
    verification = _as_list(payload.get("verification"))
    next_step = str(payload.get("next_step") or "").strip()
    integrity = str(payload.get("integrity_note") or "").strip()

    topic_text = _format_topic(topic)
    title = f"### {topic_text} · 答疑卡片"

    lines: list[str] = [title, ""]

    lines.append("**一、当前卡点**")
    lines.append(f"- 现象：{blocker}")
    diagnosis = str(payload.get("diagnosis") or "").strip()
    if diagnosis:
        lines.append(f"- 判断：{diagnosis}")
    lines.append("")

    lines.append("**二、涉及知识点**")
    if knowledge_points:
        for point in knowledge_points:
            if isinstance(point, Mapping):
                code = str(point.get("code") or "").strip()
                statement = str(point.get("statement") or point.get("text") or "").strip()
                prefix = f"【{code}】" if code else ""
                lines.append(f"- {prefix}{statement}")
            else:
                lines.append(f"- {point}")
    else:
        lines.append("- （本次未检索到对应条目，以下为通用推导，建议补充讲义或换关键词重试）")
    lines.append("")

    lines.append("**三、分层提示**")
    if hints:
        for index, hint in enumerate(hints, start=1):
            lines.append(_format_hint(index, hint))
    else:
        lines.append("- （按用户要求直接给出完整解答，未逐级提示）")
    lines.append("")

    lines.append("**四、验算证据**")
    if verification:
        for item in verification:
            lines.append(_format_verification(item))
    else:
        lines.append("- （本次未调用验算工具，结论来自推导；涉及数值结果时建议再验算一次）")
    lines.append("")

    lines.append("**五、下一步**")
    lines.append(f"- {next_step}" if next_step else "- 请先按第二步自行走一遍，再对照第三步检查")
    if integrity:
        lines.append(f"- 提示：{integrity}")
    lines.append("")
    lines.append("> 验算为数值证据，用于发现计算错误，不替代定义与定理证明。")

    markdown = "\n".join(lines).rstrip() + "\n"
    return {
        "markdown": markdown,
        "sections": list(SECTION_TITLES),
        "hint_levels": len(hints),
        "verified": bool(verification),
    }
