"""任务二：医疗知识图谱生成与问答智能体。

图谱生成智能体自动编排三个核心算子：
  entity_extract（实体识别）→ relation_extract（关系抽取）→ triple_build（三元组生成与校验）
并将校验后的三元组写入 MedicalKG。

问答智能体（KGQAAgent）基于图谱回答症状/用药/检查/并发症/人群类问题。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseAgent, Step
from ..kg import MedicalKG, RELATION_TYPES, Triple
from ..operators.entity_extract import EntityExtractOperator
from ..operators.relation_extract import RelationExtractOperator
from ..operators.triple_builder import TripleBuildOperator


class KGBuildAgent(BaseAgent):
    """知识图谱生成智能体：自动编排 NER → RE → 三元组校验 → 入图。"""

    name = "kg_build_agent"

    def __init__(self, use_npu: bool = False) -> None:
        super().__init__()
        if use_npu:
            from ..operators.npu.ner_npu import EntityExtractNPUOperator

            self.ner: Any = EntityExtractNPUOperator()
        else:
            self.ner = EntityExtractOperator()
        self.re_op = RelationExtractOperator()
        self.tb_op = TripleBuildOperator()
        self.kg = MedicalKG()

        self.register_tool("load_corpus", self._load_corpus, "加载医疗语料（JSONL，含 id/text 字段）")
        self.register_tool("entity_extract", self._entity_extract,
                           "实体识别算子：抽取疾病/症状/药物/检查/治疗/部位/人群实体")
        self.register_tool("relation_extract", self._relation_extract,
                           "关系抽取算子：抽取疾病-症状/药物/检查等关系候选")
        self.register_tool("triple_build", self._triple_build,
                           "三元组生成与校验算子：Schema 校验、去重合并、置信度过滤")
        self.register_tool("write_graph", self._write_graph, "将校验后的三元组写入知识图谱")

    # ---- 工具 -----------------------------------------------------------

    def _load_corpus(self, state: Any, path: str = "") -> tuple[dict, str]:
        docs = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").strip().splitlines()]
        return {"docs": docs}, f"加载 {len(docs)} 篇医疗文档"

    def _entity_extract(self, state: dict, **params: Any) -> tuple[dict, str]:
        texts = [d["text"] for d in state["docs"]]
        res = self.ner.run(texts, **params)
        state["entities"] = res.output
        n = sum(len(e) for e in res.output)
        return state, f"实体识别: {n} 个实体, {res.elapsed_ms:.1f}ms, {res.throughput:.0f} 文档/秒"

    def _relation_extract(self, state: dict, **params: Any) -> tuple[dict, str]:
        pairs = [(d["text"], ents) for d, ents in zip(state["docs"], state["entities"])]
        res = self.re_op.run(pairs, **params)
        flat: list[Triple] = []
        for doc, triples in zip(state["docs"], res.output):
            for t in triples:
                flat.append(Triple(head=t.head, head_type=t.head_type, relation=t.relation,
                                   tail=t.tail, tail_type=t.tail_type, source=doc.get("id", ""),
                                   confidence=t.confidence, evidence=t.evidence))
        state["candidates"] = flat
        return state, f"关系抽取: {len(flat)} 个候选三元组, {res.elapsed_ms:.1f}ms"

    def _triple_build(self, state: dict, **params: Any) -> tuple[dict, str]:
        res = self.tb_op.run(state["candidates"], **params)
        state["triples"] = res.output
        r = self.tb_op.last_report
        return state, (f"三元组校验: {r.total} -> {r.valid} "
                       f"(schema剔除{r.dropped_schema}, 自环{r.dropped_self_loop}, 合并{r.dropped_duplicate})")

    def _write_graph(self, state: dict, **params: Any) -> tuple[dict, str]:
        self.kg.add_triples(state["triples"])
        s = self.kg.stats()
        return state, f"入图: {s['num_entities']} 实体 / {s['num_relations']} 关系"

    # ---- 规划 -----------------------------------------------------------

    def rule_plan(self, task: str) -> list[Step]:
        return [
            Step(tool="entity_extract", reason="先识别医疗实体"),
            Step(tool="relation_extract", reason="基于实体抽取关系候选"),
            Step(tool="triple_build", reason="Schema 校验与去重合并"),
            Step(tool="write_graph", reason="三元组入图"),
        ]

    def build_from_corpus(self, corpus_path: str, task: str = "从医疗语料构建知识图谱"):
        plan = [Step(tool="load_corpus", params={"path": corpus_path}, reason="加载语料")] + self.rule_plan(task)
        from .base import AgentRun

        run = AgentRun(task=task, plan=plan, planner="rule")
        state: Any = None
        for step in plan:
            record = self._execute_step(step, state)
            run.records.append(record)
            if record.status in ("ok", "retried"):
                state = record.summary_state
            else:
                break
        run.result = self.kg
        return run


# ---- 问答智能体 -----------------------------------------------------------

QA_INTENTS: list[tuple[list[str], str, str]] = [
    (["症状", "表现", "临床表现"], "HAS_SYMPTOM", "的常见症状"),
    (["吃什么药", "用什么药", "药物", "用药"], "TREATED_BY_DRUG", "的常用治疗药物"),
    (["怎么治", "治疗", "疗法"], "TREATED_BY", "的治疗方式"),
    (["检查", "确诊", "诊断"], "DIAGNOSED_BY", "的相关检查"),
    (["并发症", "并发"], "COMPLICATION_OF", "的并发症"),
    (["哪些人", "人群", "高发", "好发"], "HIGH_RISK_GROUP", "的高发人群"),
    (["部位", "累及", "影响"], "AFFECTS", "累及的部位"),
]


class KGQAAgent:
    """基于知识图谱的问答智能体：意图识别 + 实体链接 + 图查询 + 证据回溯。"""

    def __init__(self, kg: MedicalKG) -> None:
        self.kg = kg

    def answer(self, question: str) -> dict[str, Any]:
        entity = self._link_entity(question)
        if entity is None:
            return {"question": question, "answer": "未能在图谱中找到问题涉及的实体。", "matched": []}
        relation, suffix = self._detect_intent(question)
        if relation is None:
            hits = self.kg.neighbors(entity)
            answer = f"{entity} 的相关知识：" + "；".join(
                f"[{RELATION_TYPES[r][2]}] {t}" for r, t, _ in hits[:10]) if hits else f"图谱中暂无 {entity} 的信息。"
            return {"question": question, "entity": entity, "answer": answer,
                    "matched": [{"relation": r, "tail": t, "confidence": c} for r, t, c in hits]}
        hits = self.kg.neighbors(entity, relation)
        if relation == "COMPLICATION_OF":
            hits += [(r, h, c) for r, h, c in self.kg.reverse_neighbors(entity, relation)]
        if not hits:
            return {"question": question, "entity": entity, "relation": relation,
                    "answer": f"图谱中暂无 {entity}{suffix} 的记录。", "matched": []}
        tails = sorted({t for _, t, _ in hits})
        return {
            "question": question, "entity": entity, "relation": relation,
            "answer": f"{entity}{suffix}包括：{'、'.join(tails)}。",
            "matched": [{"relation": r, "tail": t, "confidence": c} for r, t, c in hits],
        }

    def _link_entity(self, question: str) -> str | None:
        cands = [e for e in self.kg.entities() if e in question]
        return max(cands, key=len) if cands else None

    def _detect_intent(self, question: str) -> tuple[str | None, str]:
        for keywords, rel, suffix in QA_INTENTS:
            if any(k in question for k in keywords):
                return rel, suffix
        return None, ""
