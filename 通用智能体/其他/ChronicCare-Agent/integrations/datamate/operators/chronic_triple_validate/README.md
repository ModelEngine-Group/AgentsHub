# chronic_triple_validate

作用：便携式三元组校验算子入口。当前目录可独立迁移，运行时仅依赖本目录及共享 `_chronic_common` 实现。

输入文件：
- `data/processed/relations/`

输出文件：
- `data/processed/triples/triples_clean.jsonl`
- `data/processed/triples/triples_rejected.jsonl`
- `data/processed/triple_validate_report.json`

单独运行示例：

```bash
python integrations/datamate/operators/chronic_triple_validate/process.py --project-root . --export-path outputs/operator_runs/chronic_triple_validate --params '{}'
```

当前定位：慢病链路第 7 步。

与后续 Nexent/MCP 工具的关系：clean triples 会直接用于图谱构建，进而影响图谱 API 返回的真实指标。
