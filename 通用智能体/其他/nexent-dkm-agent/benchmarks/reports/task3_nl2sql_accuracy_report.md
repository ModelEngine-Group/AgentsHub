# 任务三 NL2SQL 准确率报告

## 目标

本报告衡量分析智能体所使用的模板 NL2SQL 翻译器的准确率。

## 方法

每个可识别问题映射到 `src/operators/analysis_ops/nl2sql.py` 中 `INTENT_SQL` 定义的标准分析意图之一。每个意图在任务二医疗 KG schema（表 `nodes`、`edges`）上对应唯一可执行 SQL 模板，因此在固定 schema 下，**意图分类准确率等价于 SQL 正确率**。

- 标注 benchmark：`benchmarks/data/nl2sql_benchmark.json`（76 题，含枢纽/症状→疾病/药物→疾病/疗法→疾病反查新意图）
- 翻译器：`classify_question_intent`（关键词打分，离线，无需 LLM）
- 指标：`correct_intent / total`，并附各意图分解与误分类问题列表，便于审阅

benchmark 有意混合标准表述、口语化中文同义词（如 不适 / 服用 / 处理办法）以及英文问题。

## 命令

```bash
python benchmarks/task3_nl2sql_benchmark.py --report benchmarks/reports/task3_nl2sql_report.json
```

## 结果

环境：Windows 本地 Python 运行时（离线，模板翻译器）。

### 意图分类基准（`nl2sql_benchmark.json`）

- 总题数：76
- 正确：76
- 准确率 100%（**76/76**）

### 执行级准确率（`nl2sql_execution_benchmark.json`）

- 总题数：18（含具体疾病过滤与症状/药物/疗法反查问句）
- 准确率 100%（**18/18**） — 实体感知翻译会针对具体疾病或实体过滤，而非返回全局聚合

### 扩展改写回归集（`nl2sql_holdout_benchmark.json`）

- 总题数：20（表述刻意区别于关键词原型）
- 准确率 100%（**20/20**）
- 文件名为兼容历史保留（含 holdout 字样），该集合是扩展改写回归测试，不是独立盲测集。

### 分路径执行（报告内 `independent_paths`）

- template：**18/18**（100%）
- llm / local_model：`not_configured`（需 `--llm-config` / `--local-model` 后评测）

机器可读输出：`benchmarks/reports/task3_nl2sql_report.json`。

## 说明

- 结果可完全离线复现，并由测试 `test_nl2sql_*_above_threshold` 守护，防止准确率回退。
- 配置 OpenAI 兼容 LLM 后，智能体使用 `translate_question_to_sql_with_llm`，优先尝试 LLM 生成 SQL，失败时回退到本模板翻译器，因此本报告测得的准确率是 LLM 启用路径的下界。
