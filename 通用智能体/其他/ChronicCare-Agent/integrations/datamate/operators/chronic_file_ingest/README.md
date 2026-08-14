# chronic_file_ingest

作用：便携式原始数据接入算子入口。当前目录可独立迁移，运行时仅依赖本目录及共享 `_chronic_common` 实现。

输入文件：
- `data/raw/structured/`
- `data/raw/text/`

输出文件：
- `data/processed/manifest.json`
- `outputs/operator_runs/chronic_file_ingest/operator_run_summary.json`

参数说明：
- `--project-root`：项目根目录
- `--export-path`：算子运行输出目录
- `--params`：额外 JSON 参数

单独运行示例：

```bash
python integrations/datamate/operators/chronic_file_ingest/process.py --project-root . --export-path outputs/operator_runs/chronic_file_ingest --params '{}'
```

当前定位：慢病链路第 1 步。

与后续 Nexent/MCP 工具的关系：为后续清洗、图谱构建和分析产物提供原始文件登记证据。
