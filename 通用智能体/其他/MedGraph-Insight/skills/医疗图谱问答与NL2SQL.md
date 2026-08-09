# Skill：医疗图谱问答与 NL2SQL

本技能包描述 MedGraph-Insight 的问答与查询能力，对应 MCP 工具
`ask_medical_kg` 与 `query_kg_with_nl2sql`。

## 1. 何时用问答，何时用 NL2SQL

| 问题特征 | 选择 | 示例 |
| --- | --- | --- |
| 询问某实体的相关知识、需要自然语言解释与证据 | `ask_medical_kg` | 「2型糖尿病有哪些症状？」「痛风好发于哪些人群？」 |
| 需要跨关系推理 | `ask_medical_kg`（多跳） | 「心力衰竭的并发症用什么药？」 |
| 需要集合运算、计数、排序、阈值筛选 | `query_kg_with_nl2sql` | 「既有胸痛又有心悸的疾病」「症状超过4个的疾病」 |
| 需要精确 SQL 供人工核对 | `query_kg_with_nl2sql` | 「置信度大于0.85的用药关系」 |

## 2. GraphRAG 问答（ask_medical_kg）

- **实现**：`code/mgi/kg/graphrag.py`
- **实体链接**：`子串匹配 > 别名匹配 > 语义相似（TF-IDF 字符 n-gram + 余弦相似度）`
- **意图识别**：从问题中识别目标关系类型（症状/用药/检查/治疗/人群/并发症）
- **多跳推理**：如「并发症用什么药」沿 `COMPLICATION_OF` → `TREATED_BY_DRUG` 二跳汇聚
- **证据返回**：每条结论附带来源语料原文与置信度，可溯源

回答要求：

1. 必须基于返回的 `answer` 与 `evidence` 作答，不得凭记忆补充医学结论；
2. 证据为空时明确说明「图谱中暂无相关记录」；
3. 涉及具体诊疗决策时提示用户咨询专业医师。

## 3. NL2SQL（query_kg_with_nl2sql）

- **实现**：`code/mgi/analysis/nl2sql.py`
- **策略**：优先 LLM 生成（配置 `MGI_LLM_API_KEY` 时），回退到确定性语义模板解析器，
  保证离线可复现
- **数据视图**：图谱导出为 SQLite，含 `entities`（实体）与 `relations`（关系）两表

支持的查询能力：

| 能力 | 示例问题 | 生成 SQL 要点 |
| --- | --- | --- |
| 别名链接 | T2DM 有哪些症状 | 归一到 `2型糖尿病` |
| 多条件交集 | 既有胸痛又有心悸的疾病 | `INTERSECT` 子查询 |
| 聚合 + HAVING | 症状超过 4 个的疾病 | `GROUP BY head HAVING cnt > 4` |
| JOIN | 2型糖尿病的药物属于什么类型 | `relations JOIN entities ON tail=name` |
| 置信度过滤 | 置信度大于 0.85 的用药关系 | `WHERE confidence >= 0.85` |
| 反向查询 | 哪些疾病会并发心力衰竭 | `WHERE tail='心力衰竭'` |

**执行准确率**：在 40 条标注评测集（`knowledge/eval/nl2sql_eval.jsonl`）上
按结果集等价比对，准确率 **100% (40/40)**。

复现命令（在 `code/` 目录下）：

```bash
python -m mgi.cli eval-sql
```

## 4. 回答模板

```text
【结论】<基于图谱证据的直接回答>
【依据】<关系类型 + 证据原文/置信度，或生成的 SQL 与结果集>
【说明】<数据边界：图谱覆盖范围、是否需要人工复核>
```
