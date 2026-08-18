# DataLoaderMapper - 数据加载算子

## 功能特性

- **多格式支持**：CSV、JSON、JSONL
- **批量处理**：多个文件一次加载
- **编码兼容**：自动检测 UTF-8、GBK、GB2312、UTF-8-SIG
- **大文件处理**：分块读取，避免内存溢出
- **合并策略**：
  - `concat`：直接拼接
  - `join`：关联合并
  - `union`：去重合并
- **容错机制**：
  - 部分成功：单文件失败不影响整体
  - 降级策略：连续错误自动降级
  - 快速失败：可选的错误处理模式

## 文件结构

```
data_loader/
├── __init__.py      # 注册入口
├── metadata.yml     # UI 配置
├── process.py       # 核心逻辑
├── requirements.txt # 依赖
└── README.md        # 本文档
```

## 使用示例

### Python 调用

```python
from data_loader import DataLoaderMapper

# 单文件加载
loader = DataLoaderMapper(
    file_paths=["D:/data/sample.csv"],
    file_formats=["csv"]
)

# 批量加载（自动合并）
loader = DataLoaderMapper(
    file_paths=["file1.csv", "file2.json", "file3.csv"],
    file_formats=["csv", "json", "csv"],
    merge_strategy="concat"
)

# 大文件分块处理
loader = DataLoaderMapper(
    file_paths=["D:/data/large.csv"],
    file_formats=["csv"],
    chunk_size=5000  # 每批 5000 行
)

# 执行
sample = {}
result = loader(sample)
print(result["data"])      # 加载的数据
print(result["count"])     # 数据条数
print(result["quality_score"])  # 质量评分
```

### DataMate 流水线配置

```yaml
operators:
  - name: DataLoader
    type: DataLoaderMapper
    params:
      file_paths: ["D:/data/input.csv"]
      file_formats: ["csv"]
      encoding: "utf-8"
      chunk_size: 10000
      merge_strategy: "concat"
```

## 输出字段

| 字段 | 类型 | 说明 |
|:---|:---:|:---|
| `data` | list | 加载的数据列表 |
| `count` | int | 数据条数 |
| `sources` | list | 来源文件路径 |
| `errors` | list | 错误信息列表 |
| `success_count` | int | 成功加载的文件数 |
| `error_count` | int | 失败的文件数 |
| `quality_score` | float | 质量评分（0-1） |
| `execution_time_ms` | float | 执行耗时（毫秒） |

## 版本历史

- **v2.2.0**：新增大文件分块处理
- **v2.1.0**：完善异常处理和降级策略
- **v2.0.0**：支持多文件批量加载和合并策略
- **v1.0.0**：初始版本

---

> CCF ModelEngine 赛道 · 数据处理智能体
