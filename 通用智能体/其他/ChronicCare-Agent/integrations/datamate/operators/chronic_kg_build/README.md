# chronic_kg_build

作用：便携式图谱构建算子入口。当前目录可独立迁移，运行时仅依赖本目录及共享 `_chronic_common` 实现。

输入文件：
- `data/processed/entities/`
- `data/processed/triples/triples_clean.jsonl`

输出文件：
- `data/graph/graph.json`
- `data/graph/graph_summary.json`
- `data/processed/kg_build_report.json`

单独运行示例：

```bash
python integrations/datamate/operators/chronic_kg_build/process.py --project-root . --export-path outputs/operator_runs/chronic_kg_build --params '{}'
```

当前定位：慢病链路第 8 步。

与后续 Nexent/MCP 工具的关系：`/kg/summary`、图谱展示和综合智能体回答都直接消费这里的真实图谱指标。
