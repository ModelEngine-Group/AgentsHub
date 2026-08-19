# 任务二：医疗知识图谱生成与问答智能体

## 目标

任务二把 `data/medical_text.txt` 中的非结构化医疗文本转换为医疗知识图谱，并基于图谱回答简单医疗关系问题。

## Schema 设计

### 实体类型

| 类型 | 含义 | 示例 |
|---|---|---|
| Disease | 疾病 | 肺炎、冠心病、急性扁桃体炎 |
| Symptom | 症状 | 发热、咳嗽、胸痛 |
| Drug | 药物 | 阿莫西林、阿司匹林、二甲双胍 |
| Examination | 检查 | 胸部X线、CT检查、心电图 |
| Treatment | 治疗方式 | 呼吸机辅助通气、急诊PCI、扁桃体切除术 |

### 关系类型

| 关系 | 含义 | 示例 |
|---|---|---|
| has_symptom | 疾病具有某症状 | 肺炎 has_symptom 发热 |
| treated_by | 疾病可由某药物或治疗方式处理 | 肺炎 treated_by 阿莫西林 |
| requires_exam | 疾病诊断需要某检查 | 肺炎 requires_exam 胸部X线 |
| complication_of | 并发或相关疾病关系 | 当前 MVP 预留 |

## 智能体流程

```text
medical_text.txt
  ↓
ReadTextOperator
  ↓
SplitParagraphOperator
  ↓
EntityRecognitionOperator
  ↓
RelationExtractionOperator
  ↓
TripleGenerationOperator
  ↓
TripleValidationOperator
  ↓
GraphBuildOperator
  ↓
KnowledgeGraphAgent 输出图谱文件
  ↓
KGQAAgent 基于图谱问答
```

## 输出文件

| 文件 | 说明 |
|---|---|
| `outputs/medical_triples.json` | 医疗三元组列表 |
| `outputs/medical_kg.json` | nodes / edges 图谱 JSON |
| `outputs/medical_kg.graphml` | 可被图工具读取的 GraphML |
| `outputs/medical_kg.html` | 简单 HTML 图谱展示 |
| `outputs/medical_qa_examples.json` | 4 个示例问题的回答和证据 |

## 三元组样例

```json
{
  "subject": "肺炎",
  "predicate": "has_symptom",
  "object": "发热"
}
```

```json
{
  "subject": "肺炎",
  "predicate": "treated_by",
  "object": "阿莫西林"
}
```

```json
{
  "subject": "肺炎",
  "predicate": "requires_exam",
  "object": "胸部X线"
}
```

## 问答能力

当前 `KGQAAgent` 支持 4 类规则型问题：

| 问题类型 | 示例问题 | 查询逻辑 |
|---|---|---|
| 疾病查症状 | 肺炎有哪些症状？ | 查 `疾病 --has_symptom--> 症状` |
| 疾病查药物 | 肺炎可以用什么药？ | 查 `疾病 --treated_by--> 药物` |
| 症状反查疾病 | 哪些疾病关联胸痛？ | 查 `疾病 --has_symptom--> 胸痛` |
| 药物反查疾病 | 阿莫西林可以治疗哪些疾病？ | 查 `疾病 --treated_by--> 阿莫西林` |

## 运行命令

```bash
uv run python run_demo.py task2
```

## 验证命令

```bash
uv run pytest tests/test_kg_operators.py
uv run pytest tests/test_knowledge_graph_agent.py
uv run pytest tests/test_kgqa_agent.py
uv run pytest
```
