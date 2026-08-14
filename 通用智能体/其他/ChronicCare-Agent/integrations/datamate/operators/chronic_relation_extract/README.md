# chronic_relation_extract

作用：便携式关系抽取算子入口。当前目录可独立迁移，运行时仅依赖本目录及共享 `_chronic_common` 实现。

输入文件：
- `data/processed/entities/`
- `data/processed/chunks/`

输出文件：
- `data/processed/relations/`
- `data/processed/relation_extract_report.json`

单独运行示例：

```bash
python integrations/datamate/operators/chronic_relation_extract/process.py --project-root . --export-path outputs/operator_runs/chronic_relation_extract --params '{}'
```

当前定位：慢病链路第 6 步。

与后续 Nexent/MCP 工具的关系：关系抽取结果会进入三元组校验和知识图谱构建，最终支撑 `/kg/summary` 等工具。
