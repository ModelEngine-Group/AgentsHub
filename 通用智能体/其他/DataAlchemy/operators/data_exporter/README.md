# DataExporter - 数据导出算子

> 版本：1.0.0  
> 开发者：ccf-team

## 功能概述

支持 CSV/JSON/JSONL 格式导出的数据导出算子，完成 ETL 流水线的 Load 阶段。

## 处理顺序

```
validate_input → prepare_data → export_file → verify_output
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---:|:---|
| `outputFormat` | select | `csv` | 导出格式：csv / json / jsonl |
| `outputDir` | input | （必填） | 输出目录路径，支持 Docker 路径映射 |
| `outputFileName` | input | `export_output` | 输出文件名（不含扩展名） |
| `encoding` | select | `utf-8` | 输出编码：utf-8 / gbk / gb2312 |
| `includeHeader` | switch | `true` | CSV 是否包含表头 |
| `indexColumn` | switch | `false` | 是否包含行索引列 |
| `overwrite` | switch | `true` | 文件已存在时是否覆盖 |

## 输入输出

**输入**：来自 data_transformer / data_cleaner 的标准输出格式 `{data: [...], schema: {...}}`

**输出**：`{data: [...](透传), schema: {...}(透传), report: {export_summary}}`

### 支持的导出格式

| 格式 | 扩展名 | 说明 |
|:---|:---:|:---|
| CSV | `.csv` | 标准表格格式，支持表头/索引配置 |
| JSON | `.json` | JSON 数组格式 `[{...}, {...}]` |
| JSONL | `.jsonl` | 每行一个 JSON 对象 `{...}\n{...}\n` |

## 使用示例

```python
from data_exporter import DataExporter

exporter = DataExporter()

# 导出为 CSV
result = exporter.process(
    input_data={"data": [{"name": "张三", "age": 25}]},
    params={
        "outputFormat": "csv",
        "outputDir": "/mnt/data/export/",
        "outputFileName": "cleaned_data",
        "includeHeader": True
    }
)

print(result["report"]["summary"])
# 输出：数据导出完成：1 行，CSV 格式，输出到 /mnt/data/export/cleaned_data.csv，文件大小 0.1KB。
```

## 注意事项

1. 输出目录不存在时自动创建
2. Docker 路径 `/mnt/data/` 自动映射到 `D:/data/`
3. 文件已存在且 `overwrite=false` 时，自动追加时间戳（如 `_20260430_221500`）
4. JSON/JSONL 格式自动将 NaN 转换为 null
