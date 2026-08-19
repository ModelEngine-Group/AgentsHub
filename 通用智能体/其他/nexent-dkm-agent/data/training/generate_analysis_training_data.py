"""Generate synthetic training data for the task-3 analysis agent.

Emits two JSONL datasets for QLoRA fine-tuning of a small language model:

1. Planning data (``analysis_planning_{train,val}.jsonl``): natural-language
   analysis requests -> a JSON operator plan, matching the schema produced by
   the rule/LLM planners (operators restricted to the registered analysis
   operators).
2. NL2SQL data (``analysis_nl2sql_{train,val}.jsonl``): analysis questions ->
   a single read-only SQL statement against the task-2 graph schema, reusing
   the canonical ``INTENT_SQL`` templates so the labels are executable.

The samples are derived from the same intents the runtime planner/translator
use, so a model trained on them produces drop-in-compatible outputs that fall
back gracefully when absent.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.analysis_ops.analysis_prompts import (
    NL2SQL_INSTRUCTION as _NL2SQL_INSTRUCTION,
)
from src.operators.analysis_ops.analysis_prompts import (
    PLANNING_INSTRUCTION as _PLANNING_INSTRUCTION,
)
from src.operators.analysis_ops.nl2sql import INTENT_SQL

random.seed(42)

_CORE_OPS = ["load_graph"]
_TAIL_OPS = ["build_analysis_visualizations", "build_analysis_report"]

# intent -> (request phrasings, operators contributed, intent keyword label)
_PLANNING_INTENTS: dict[str, tuple[list[str], list[str], str]] = {
    "statistics": (
        ["统计各类实体数量", "做基础统计分析", "汇总图谱规模指标", "compute graph statistics"],
        ["generate_statistical_summary"],
        "statistics",
    ),
    "association": (
        ["分析疾病与症状的关联", "做关联分析", "找出实体间的关系模式", "association analysis"],
        ["generate_association_analysis"],
        "association",
    ),
    "trend": (
        ["分析记录随时间的趋势", "做趋势分析", "看指标的变化趋势", "trend analysis over records"],
        ["generate_trend_analysis"],
        "trend",
    ),
    "graph_analytics": (
        ["分析核心枢纽节点", "找出关键节点和社区结构", "计算中心性与最短路径",
         "detect communities and key hubs"],
        ["compute_centrality", "compute_shortest_paths", "detect_communities"],
        "graph_analytics",
    ),
    "nl2sql": (
        ["用SQL查询图谱回答问题", "把问题转成SQL查询", "query the graph with SQL"],
        ["translate_question_to_sql", "execute_sql"],
        "nl2sql",
    ),
}

# intent -> question phrasings whose canonical answer is INTENT_SQL[intent].
_NL2SQL_QUESTIONS: dict[str, list[str]] = {
    "top_disease_symptoms": [
        "哪些疾病关联最多症状？", "症状最多的疾病有哪些", "每种疾病有多少症状",
        "各疾病的临床表现数量排名", "disease symptom counts",
    ],
    "top_disease_drugs": [
        "哪些疾病用药种类最多", "高血压用什么药", "疾病的治疗药物有多少种",
        "糖尿病患者需要服用哪些药", "each disease medication count",
    ],
    "top_disease_exams": [
        "哪些疾病需要的检查项目最多", "每种疾病做哪些化验", "疾病的检测手段统计",
        "which examinations diagnose each disease",
    ],
    "top_disease_treatments": [
        "各疾病推荐的治疗方案", "高血压怎么治疗", "疾病的疗法分别有哪些",
        "冠心病的处理办法有哪些", "recommended treatment per disease",
    ],
    "disease_complications": [
        "哪些疾病的并发症最多", "疾病之间的并发关系", "list complications of diseases",
    ],
    "relation_distribution": [
        "图谱中有哪些关系", "知识图谱里关系数量怎么分布", "what relation types exist",
    ],
    "list_diseases": [
        "列出所有疾病", "图谱里有哪些疾病", "which diseases are in the graph",
        "列举所有疾病名称",
    ],
    "top_mentioned_entities": [
        "提及次数最多的实体", "哪些实体出现最频繁", "most mentioned entities in the kg",
        "频次最高的节点是什么",
    ],
    "entity_distribution": [
        "各类实体的数量分布", "节点类型的统计", "entity type distribution",
        "知识图谱实体分布情况",
    ],
}


def generate_planning_samples(n: int = 600) -> list[dict[str, str]]:
    """Build planning samples: analysis request -> operator-plan JSON."""

    intents = list(_PLANNING_INTENTS)
    samples: list[dict[str, str]] = []
    for _ in range(n):
        chosen = random.sample(intents, random.randint(1, 3))
        operators = list(_CORE_OPS)
        keywords: list[str] = []
        request_parts: list[str] = []
        for intent in chosen:
            phrases, ops, label = _PLANNING_INTENTS[intent]
            request_parts.append(random.choice(phrases))
            for op in ops:
                if op not in operators:
                    operators.append(op)
            keywords.append(label)
        for op in _TAIL_OPS:
            if op not in operators:
                operators.append(op)

        request = "，".join(request_parts)
        output = json.dumps(
            {
                "task_type": "full_analysis",
                "operators": operators,
                "intent_keywords": keywords,
                "confidence": round(random.uniform(0.82, 0.97), 2),
            },
            ensure_ascii=False,
        )
        samples.append(
            {"instruction": _PLANNING_INSTRUCTION, "input": request, "output": output}
        )
    return samples


def generate_nl2sql_samples(n: int = 600) -> list[dict[str, str]]:
    """Build NL2SQL samples: analysis question -> canonical SELECT statement."""

    pairs = [
        (q, INTENT_SQL[intent])
        for intent, questions in _NL2SQL_QUESTIONS.items()
        for q in questions
    ]
    samples: list[dict[str, str]] = []
    for _ in range(n):
        question, sql = random.choice(pairs)
        samples.append(
            {"instruction": _NL2SQL_INSTRUCTION, "input": question, "output": sql}
        )
    return samples


def _write_split(samples: list[dict[str, str]], stem: str, output_dir: Path) -> tuple[Path, Path]:
    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train, val = samples[:split], samples[split:]
    train_path = output_dir / f"{stem}_train.jsonl"
    val_path = output_dir / f"{stem}_val.jsonl"
    for path, rows in ((train_path, train), (val_path, val)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return train_path, val_path


def main() -> int:
    output_dir = Path(__file__).resolve().parent
    planning = generate_planning_samples(600)
    nl2sql = generate_nl2sql_samples(600)

    p_train, p_val = _write_split(planning, "analysis_planning", output_dir)
    n_train, n_val = _write_split(nl2sql, "analysis_nl2sql", output_dir)

    print(f"Planning: {p_train.name} + {p_val.name} ({len(planning)} samples)")
    print(f"NL2SQL:   {n_train.name} + {n_val.name} ({len(nl2sql)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
