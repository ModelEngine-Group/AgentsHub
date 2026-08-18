# chronic_nl2sql_analyze

作用：便携式 NL2SQL 分析算子入口。当前目录可独立迁移，运行时仅依赖本目录及共享 `_chronic_common` 实现。

输入文件：
- `data/sqlite/chroniccare.db`
- `configs/nl2sql_questions.json`（DataMate 内部标准评测输入）

输出文件：
- `outputs/reports/sql_candidates.json`
- `outputs/reports/nl2sql_eval_report.json`
- `outputs/reports/indicator_results.json`

单独运行示例：

```bash
python integrations/datamate/operators/chronic_nl2sql_analyze/process.py --project-root . --export-path outputs/operator_runs/chronic_nl2sql_analyze --params '{}'
```

当前定位：慢病链路第 10 步。

与后续分析链路的关系：该算子产出 SQL 候选、内部评测结果和指标分析证据；Nexent 对外能力清单由 `chroniccare_open_sql_examples` 动态提供，不直接暴露内部固定题集。
