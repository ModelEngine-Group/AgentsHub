"""任务二工具集：图谱构建 / 图谱查询与问答。"""

import json
from typing import Optional

from smolagents.tools import Tool

from dki.agents.tools.data_tools import _err, _notify, _ok, _safe_path
from dki.paths import OUTPUT_DIR

KG_PATH = OUTPUT_DIR / "kg" / "medical_kg.json"


def _load_kg():
    from dki.kg.store import KnowledgeGraph
    from dki.pipeline.artifacts import validate_task2_artifact

    validate_task2_artifact(KG_PATH)
    return KnowledgeGraph.load(KG_PATH)


class BuildKgTool(Tool):
    name = "build_knowledge_graph"
    description = (
        "构建医疗知识图谱：对输入目录执行标准知识抽取流水线"
        "（清洗→分段→实体→关系→三元组校验），跨文档聚合去重后保存图谱。"
        "返回 JSON（图谱路径与规模统计）。"
    )
    inputs = {
        "input_dir": {"type": "string", "description": "任务一 manifest 的清洗文本目录；省略时自动读取 manifest", "nullable": True},
        "pattern": {"type": "string", "description": "文件通配符，默认 *.txt", "nullable": True},
        "min_confidence": {"type": "number", "description": "三元组置信度阈值，默认 0.55", "nullable": True},
    }
    output_type = "string"

    def __init__(self, observer=None):
        super().__init__()
        self.observer = observer

    def forward(self, input_dir: Optional[str] = None, pattern: Optional[str] = None,
                min_confidence: Optional[float] = None) -> str:
        try:
            from dki.kg.build import build_kg_from_samples
            from dki.pipeline import LocalPipelineRunner, samples_from_dir
            from dki.pipeline.artifacts import (
                load_task1_manifest,
                task1_clean_dir,
                write_task2_receipt,
            )

            manifest = load_task1_manifest()
            in_dir = task1_clean_dir(manifest)
            if input_dir and _safe_path(input_dir).resolve() != in_dir.resolve():
                return _err("任务二只能读取任务一 manifest 产物", "先运行 task1 或 publish_task1_artifacts")
            samples = samples_from_dir(in_dir, pattern=pattern or "*.txt")
            if not samples:
                return _err(f"目录 {in_dir} 无匹配文件", "检查任务一 manifest 的 clean_text 产物")
            _notify(self.observer, f"构建知识图谱（{len(samples)} 份文档）")
            process = [
                {"MedicalTextCleaner": {"maskPii": True}},
                {"MedicalSectionSplitter": {}},
                {"MedicalEntityExtractor": {"backend": "dict"}},
                {"MedicalRelationExtractor": {}},
                {"TripleBuilderValidator": {"minConfidence": float(min_confidence or 0.55)}},
            ]
            runner = LocalPipelineRunner()
            result = runner.run(process, samples, name="agent-kg-build")
            if result.report.status.value not in {"COMPLETED", "PARTIAL_SUCCESS"}:
                return _err(
                    f"任务二流水线状态为 {result.report.status.value}，拒绝发布图谱",
                    "查看任务报告中的失败算子后重新构建",
                )
            kg = build_kg_from_samples(result.samples)
            kg.save(KG_PATH)
            receipt = write_task2_receipt(
                KG_PATH,
                task_id=result.report.task_id,
            )
            return _ok(
                task_id=result.report.task_id,
                pipeline_status=result.report.status.value,
                kg_path=str(KG_PATH),
                provenance_path=str(receipt),
                stats=kg.stats(),
            )
        except Exception as e:
            return _err(str(e), "检查输入目录是否含病历文本；或查看 get_task_status 定位失败算子")


class KgQueryTool(Tool):
    name = "kg_query"
    description = (
        "查询医疗知识图谱。action 取值："
        "stats(规模统计) / entity(实体详情,参数name) / neighbors(邻居,参数name+可选relation) / "
        "subgraph(子图,参数name+可选hops) / path(两实体路径,参数name+target) / "
        "top(核心实体,可选etype如Disease) / by_relation(某关系全部三元组,参数relation) / "
        "qa(自然语言问答,参数question)。返回 JSON。"
    )
    inputs = {
        "action": {"type": "string", "description": "stats/entity/neighbors/subgraph/path/top/by_relation/qa"},
        "name": {"type": "string", "description": "实体名（entity/neighbors/subgraph/path 用）", "nullable": True},
        "target": {"type": "string", "description": "目标实体名（path 用）", "nullable": True},
        "relation": {"type": "string", "description": "关系类型（neighbors/by_relation 用），如 HAS_SYMPTOM", "nullable": True},
        "hops": {"type": "integer", "description": "子图跳数，默认 1", "nullable": True},
        "etype": {"type": "string", "description": "实体类型过滤（top 用），如 Disease", "nullable": True},
        "question": {"type": "string", "description": "自然语言问题（qa 用）", "nullable": True},
    }
    output_type = "string"

    def __init__(self, observer=None):
        super().__init__()
        self.observer = observer

    def forward(self, action: str, name: Optional[str] = None, target: Optional[str] = None,
                relation: Optional[str] = None, hops: Optional[int] = None,
                etype: Optional[str] = None, question: Optional[str] = None) -> str:
        try:
            kg = _load_kg()
            _notify(self.observer, f"图谱查询 action={action}")
            if action == "stats":
                from dki.kg.schema import schema_description

                return _ok(stats=kg.stats(), schema=schema_description())
            if action == "entity":
                ent = kg.get_entity(name or "")
                return _ok(entity=ent) if ent else _err(f"实体不存在: {name}",
                                                        "用 top 查看核心实体名")
            if action == "neighbors":
                ns = kg.neighbors(name or "", relation=relation)
                return _ok(name=name, neighbors=ns[:20])
            if action == "subgraph":
                sub = kg.subgraph([name or ""], hops=hops or 1)
                return _ok(n_entities=len(sub["entities"]), n_triples=len(sub["triples"]),
                           triples=sub["triples"][:30])
            if action == "path":
                path = kg.find_path(name or "", target or "", max_hops=3)
                return _ok(found=bool(path), path=path)
            if action == "top":
                return _ok(top=kg.top_entities(etype=etype, k=10))
            if action == "by_relation":
                ts = kg.triples_by_relation(relation or "")
                return _ok(relation=relation, n=len(ts), triples=ts[:30])
            if action == "qa":
                from dki.kg.qa import answer_question

                res = answer_question(kg, question or "")
                return _ok(**res)
            return _err(f"未知 action: {action}",
                        "可用: stats/entity/neighbors/subgraph/path/top/by_relation/qa")
        except FileNotFoundError as e:
            return _err(str(e), "先调用 build_knowledge_graph")
        except Exception as e:
            return _err(str(e))
