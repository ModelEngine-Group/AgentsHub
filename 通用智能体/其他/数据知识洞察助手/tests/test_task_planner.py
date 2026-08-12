import json

import pytest

from planners.task_planner import LLMTaskPlanner, RuleBasedTaskPlanner


def test_rule_based_planner_rejects_empty_task_description():
    planner = RuleBasedTaskPlanner()

    with pytest.raises(ValueError, match="任务描述不能为空"):
        planner.plan("  ")


def test_rule_based_planner_returns_csv_and_text_steps_for_combined_task():
    planner = RuleBasedTaskPlanner()

    plan = planner.plan("请清洗医疗CSV数据，并处理医疗文本")

    assert [item["step"] for item in plan] == [
        "read_csv",
        "clean_missing",
        "normalize_text",
        "extract_fields",
        "save_csv",
        "read_text",
        "split_paragraph",
        "save_text_json",
    ]


def test_rule_based_planner_returns_only_csv_steps_for_csv_task():
    planner = RuleBasedTaskPlanner()

    plan = planner.plan("只清洗医疗CSV结构化数据")

    assert [item["step"] for item in plan] == [
        "read_csv",
        "clean_missing",
        "normalize_text",
        "extract_fields",
        "save_csv",
    ]


def test_rule_based_planner_returns_only_text_steps_for_text_task():
    planner = RuleBasedTaskPlanner()

    plan = planner.plan("只处理医疗文本并切分段落")

    assert [item["step"] for item in plan] == [
        "read_text",
        "split_paragraph",
        "save_text_json",
    ]


def test_llm_task_planner_uses_llm_json_steps():
    prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps(
            {
                "steps": [
                    {"step": "read_text", "operator": "ReadTextOperator", "goal": "读取医疗文本"},
                    {"step": "split_paragraph", "operator": "SplitParagraphOperator", "goal": "切分医疗段落"},
                    {"step": "save_text_json", "operator": "SaveJsonOperator", "goal": "保存文本 JSON"},
                ]
            },
            ensure_ascii=False,
        )

    planner = LLMTaskPlanner(llm=fake_llm)

    plan = planner.plan("只处理医疗文本")

    assert prompts
    assert [item["step"] for item in plan] == ["read_text", "split_paragraph", "save_text_json"]


def test_llm_task_planner_falls_back_to_rules_when_llm_returns_invalid_json():
    planner = LLMTaskPlanner(llm=lambda prompt: "not json")

    plan = planner.plan("只清洗医疗CSV结构化数据")

    assert [item["step"] for item in plan] == [
        "read_csv",
        "clean_missing",
        "normalize_text",
        "extract_fields",
        "save_csv",
    ]


def test_llm_task_planner_accepts_json_fenced_response():
    def fake_llm(prompt: str) -> str:
        return """```json
{"steps": [{"step": "read_text", "operator": "ReadTextOperator", "goal": "读取医疗文本"}]}
```"""

    planner = LLMTaskPlanner(llm=fake_llm)

    plan = planner.plan("只读取医疗文本")

    assert plan == [{"step": "read_text", "operator": "ReadTextOperator", "goal": "读取医疗文本"}]


def test_llm_task_planner_uses_first_valid_steps_json_object():
    def fake_llm(prompt: str) -> str:
        return (
            '模板：{"steps": [{"step": string, "operator": string, "goal": string}]}\n'
            '实际：{"steps": [{"step": "read_text", "operator": "ReadTextOperator", "goal": "读取医疗文本"}]}'
        )

    planner = LLMTaskPlanner(llm=fake_llm)

    plan = planner.plan("只读取医疗文本")

    assert plan == [{"step": "read_text", "operator": "ReadTextOperator", "goal": "读取医疗文本"}]
