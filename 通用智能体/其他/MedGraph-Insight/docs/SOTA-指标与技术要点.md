# SOTA 指标与技术要点 · MedGraph-Insight

> 面向 2026 ModelEngine 应用赛道。本文汇总三大任务的**可复现指标**、评测方法与关键技术，所有数字均可用仓库内命令一键复现。

## 0. 一键复现

```bash
pip install -e ".[dev]"
pytest -q            # 51 项单元测试全绿
mgi eval-kg          # 任务二：NER/RE 金标准 P/R/F1
mgi eval-sql         # 任务三：NL2SQL 执行准确率
mgi dag              # 任务一：DAG 编排 + 数据质量 + 血缘
mgi pipeline         # 端到端全链路
```

## 1. 指标总览

| 维度 | 指标 | 结果 | 赛题基线/要求 | 评测方法 |
| --- | --- | --- | --- | --- |
| 任务二 NER | 实体级 P / R / F1 | **1.000 / 1.000 / 1.000** | — | 15 篇金标准，201 实体，精确边界+类型匹配 |
| 任务二 RE | 三元组级 P / R / F1 | **1.000 / 0.994 / 0.997** | — | 15 篇金标准，176 关系，(head,rel,tail) 精确匹配 |
| 任务三 NL2SQL | 执行准确率 | **100% (40/40)** | ≥ 85% | 结果集等价比对（`data/eval/nl2sql_eval.jsonl`） |
| 图谱规模 | 实体 / 关系 | **212 / 351** | — | 40 篇语料自动构建 |
| 工程 | 单元测试 | **51 passed** | — | `pytest -q` |

> 图谱实体类型分布：Disease 43 / Symptom 54 / Drug 42 / Examination 28 / Treatment 22 / BodyPart 13 / Population 10；关系类型：HAS_SYMPTOM 119 / TREATED_BY_DRUG 64 / DIAGNOSED_BY 60 / TREATED_BY 35 / HIGH_RISK_GROUP 29 / COMPLICATION_OF 25 / AFFECTS 17 / CONTRAINDICATED 2。

## 2. 任务一：数据处理智能体（DAG 编排 + 数据质量 + 血缘）

- **DAG 算子编排**（`mgi/datamate/dag.py`）：在线性流水线之上支持有向无环图，`OperatorDAG.add(id, op, inputs=[...])` 声明依赖；Kahn 拓扑排序 + **环检测**（存在环则报错）+ **分叉/汇合**（多上游取并集）。
- **数据质量指标（DQ）**：逐节点计算 `count / avg_len / empty_rate / dup_rate / pii_leak_rate / html_rate`，量化每个算子对数据质量的改善（如 `PhoneAnonymizer` 使 `pii_leak_rate 0.5 → 0`）。
- **血缘追踪（lineage）**：递归展开每个节点到数据源的上游链，保证处理过程可审计、可溯源。
- 双后端：配置 `DATAMATE_BASE_URL` 走真实 DataMate REST；否则本地内置算子回退，CI/离线可复现。

## 3. 任务二：医疗知识图谱与问答

- **混合 NER + 别名注入**：Aho-Corasick O(n) 多模式匹配 + 最长匹配消歧；从 `synonyms.json` 将别名/英文/缩写注入 AC 词典，`T2DM / type 2 diabetes / 糖尿病2型` 均可识别。
- **实体归一化链接**（`mgi/kg/normalize.py`）：`精确 > 别名 > 归一化` 三级链接到规范名；默认关闭模糊匹配以杜绝中文短词语义漂移（如「心悸→心梗」），仅长度门控 ≥4 时可选启用。
- **否定检测（ConText 风格）**：断言状态判定作用域限定到最近子句边界（按 `、，。；` 切分），命中「未见/否认/无」等触发词则关系判为 negated，置信度减半并在三元组构建阶段默认过滤（「未见胸痛」不生成 `HAS_SYMPTOM`）。
- **单话题疾病头消歧**：取篇章内频次最高疾病为唯一话题头实体，消除「篇内多疾病 × 多症状」的笛卡尔积误报（精度杀手），同时借跨句回退提升召回。
- **GraphRAG 语义检索问答**（`mgi/kg/graphrag.py`）：TF-IDF 字符 n-gram 索引 + 余弦相似度做语义实体链接；识别意图后检索子图证据并支持**多跳推理**（并发症链二跳汇聚），答案附证据来源。

## 4. 任务二加分：NPU 优化

将 Aho-Corasick 状态机 NER 重写为**稠密张量运算**（Unicode 码点矩阵 + 向量化相等比较 + 沿模式长度归约），消除数据依赖分支，天然适配 NPU/GPU SIMD。后端自动选择 `torch_npu`（昇腾 CANN）→ `torch` CPU → `numpy`，三者结果逐位一致（`test_npu_backend_matches_cpu`）。`mgi bench` 输出三后端吞吐/延迟/加速比对比。

**昇腾真机验证（Ascend 910B2C）**：设备端核函数 ≈ 7,402 万次字符比较/ms（较 CPU 侧高 ~50×），800~80,000 文档全部规模下三后端结果逐位一致；完成设备端稀疏化（仅回传命中索引）、批维分块防 OOM、解码向量化三项优化，40,000 文档端到端 6.0s→2.7s（2.2×）。真机报告：[docs/benchmark-ascend.md](docs/benchmark-ascend.md)。

## 5. 任务三：NL2SQL 与 BI 可视化

确定性语义解析器（LLM 可选，离线可复现），从基础问答扩展到生产级复杂查询：

| 能力 | 示例问题 | 生成 SQL 要点 |
| --- | --- | --- |
| 别名链接 | T2DM 有哪些症状 | 归一到 `2型糖尿病` + `HAS_SYMPTOM` |
| 多条件交集 | 既有胸痛又有心悸的疾病 | `INTERSECT` 子查询 |
| 聚合 + HAVING | 症状超过 4 个的疾病 | `GROUP BY head HAVING cnt > 4` |
| JOIN | 2型糖尿病的药物属于什么类型 | `relations JOIN entities ON tail=name` |
| 置信度过滤 | 置信度大于 0.85 的用药关系 | `WHERE confidence >= 0.85` |
| 反向查询 | 哪些疾病会并发心力衰竭 | `WHERE tail='心力衰竭'` |

BI 可视化：实体/关系分布柱状与饼图、**实体类型×关系类型覆盖热力图**、TopN 关联排行、**并发症多节点关系网络（度中心性加权）**、实体中心 networkx 子图，汇聚为自包含 HTML 图文报告；`mgi analyze --trace` 与 `mgi pipeline` 同步导出智能体规划/执行轨迹 JSON（可审计可回放）。

NPU 基准评测除吞吐/延迟外，还输出字符比较次数（工作量）、峰值张量内存、比较吞吐（并行度代理）等资源/能效指标；算法设计与 CPU↔NPU 对比分析见 [docs/npu-optimization.md](docs/npu-optimization.md)。

## 6. 评测集与语料

- 语料 `data/corpus/medical_notes.jsonl`：40 篇疾病案例/指南。
- 金标准 `data/eval/kg_gold.jsonl`：15 篇人工标注，201 实体 / 176 关系（NER/RE 评测）。
- `data/eval/nl2sql_eval.jsonl`：40 条（问题 + 金标准 SQL），覆盖多条件/聚合/JOIN/置信度/别名/反向查询。
