from __future__ import annotations

from typing import Any, Dict, List


class NexentAdapter:
    """Nexent-facing orchestration adapter.

    The deterministic planner returns transparent operator plans. A real Nexent
    runtime can later call the same tools with the same input/output schemas.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def plan(self, task: str, available_tools: List[str]) -> Dict[str, Any]:
        if self.enabled:
            raise NotImplementedError(
                "Nexent runtime binding is not configured yet. Use deterministic planner."
            )
        intent = self._infer_intent(task)
        return {
            "task": task,
            "runtime": "deterministic_rules",
            "tools": available_tools,
            "intent": intent,
            "steps": [
                {"order": index, "tool": tool, "reason": self._tool_reason(tool)}
                for index, tool in enumerate(available_tools, start=1)
            ],
            "rationale": "Use deterministic operators to guarantee a reproducible contest demo; keep the same tool contract for later Nexent runtime binding.",
        }

    def _infer_intent(self, task: str) -> Dict[str, bool]:
        return {
            "cleaning": any(keyword in task for keyword in ["清洗", "规范", "去噪"]),
            "extraction": any(keyword in task for keyword in ["抽取", "实体", "关系", "知识图谱"]),
            "transformation": any(keyword in task for keyword in ["转换", "结构化", "ETL", "切分", "处理", "构建"]),
            "analysis": any(keyword in task for keyword in ["分析", "统计", "指标", "可视化"]),
        }

    def _tool_reason(self, tool: str) -> str:
        reasons = {
            "raw_record_profiler": "识别输入数据规模、字段覆盖率和结构化程度。",
            "medical_record_normalizer": "把公开数据源记录规范为统一 ETL 输入契约。",
            "medical_text_loader": "将 JSONL 记录加载为内部 Document 对象。",
            "medical_text_cleaner": "清理空白、标点和文本噪声。",
            "medical_text_segmenter": "切分为句子级 Segment，供后续分析、抽取或检索使用。",
            "medical_table_profiler": "识别结构化表格字段、缺失率和字段类型。",
            "medical_table_normalizer": "把结构化医疗表格映射为统一字段契约。",
            "medical_table_document_builder": "将规范化表格行转换为可复用 Document。",
            "rule_entity_extractor": "基于来源字段词表抽取医学实体。",
            "rule_relation_extractor": "基于医学触发词抽取关系候选。",
            "triple_builder": "把实体关系转换为可验证三元组。",
            "triple_validator": "用图谱 schema 约束校验三元组。",
        }
        return reasons.get(tool, "执行智能体计划中的工具步骤。")
