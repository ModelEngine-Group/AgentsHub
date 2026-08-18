# chronic_table_clean

作用：便携式结构化清洗算子入口。当前目录可独立迁移，运行时仅依赖本目录及共享 `_chronic_common` 实现。

输入文件：
- `data/raw/structured/`

输出文件：
- `data/processed/clean_tables/`
- `data/processed/table_clean_report.json`

参数说明：
- `--project-root`
- `--export-path`
- `--params`

单独运行示例：

```bash
python integrations/datamate/operators/chronic_table_clean/process.py --project-root . --export-path outputs/operator_runs/chronic_table_clean --params '{}'
```

当前定位：慢病链路第 2 步。

与后续 Nexent/MCP 工具的关系：清洗后的结构化表会继续进入字段归一化、SQLite 装载与分析工具链。
