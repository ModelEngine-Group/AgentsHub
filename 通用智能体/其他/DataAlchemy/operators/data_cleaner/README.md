# DataCleaner 数据清洗算子

> 支持去重、空值处理、格式标准化、隐私脱敏的结构化数据清洗算子

## 功能说明

DataCleaner 是 CCF ModelEngine 赛道的数据清洗算子，提供以下功能：

| 功能 | 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---:|:---:|:---|
| **删除重复行** | `removeDuplicates` | switch | false | 删除数据中的完全重复行 |
| **空值处理** | `handleMissing` | select | "drop" | drop/fill/keep 三种策略 |
| **填充值** | `fillValue` | input | "" | 统一填充值 |
| **去除空格** | `trimWhitespace` | switch | true | 去除字符串首尾空格 |
| **格式标准化** | `standardizeFormat` | switch | false | 日期格式统一为 YYYY-MM-DD |
| **隐私脱敏** | `privacyCheck` | switch | false | 自动检测并脱敏敏感信息 |

## 处理顺序

```
trim_whitespace → remove_duplicates → handle_missing → privacy_check → standardize_format
```

1. **先去空格**：确保去重不会被首尾空格干扰
2. **再去重**：基于清理后数据去重
3. **空值处理**：按策略处理空值
4. **隐私脱敏**：最后处理，避免重复行浪费操作
5. **格式标准化**：脱敏后的数据格式可能已改变

## 隐私脱敏规则

| 类型 | 检测规则 | 脱敏示例 |
|:---|:---|:---|
| 手机号 | 11位，以1开头 | `13812345678` → `138****5678` |
| 身份证号 | 15位或18位 | `310101199001011234` → `310***********1234` |
| 姓名 | 2-4个中文字符 | `张三` → `患者1` |

## 输入输出格式

### 输入

```json
{
  "data": [
    {"name": "张三", "phone": "13812345678", "age": 25},
    {"name": "李四", "phone": "", "age": null}
  ],
  "schema": {
    "columns": ["name", "phone", "age"],
    "row_count": 2
  }
}
```

### 输出

```json
{
  "data": [...],
  "schema": {"columns": [...], "row_count": ...},
  "report": {
    "input_rows": 1000,
    "output_rows": 985,
    "duplicates_removed": 15,
    "missing_handled": 120,
    "privacy_masked": {"phone": 175, "id_card": 50, "name": 200},
    "quality_metrics": {
      "null_rate_before": 0.15,
      "null_rate_after": 0.02,
      "duplicate_rate_after": 0.0
    },
    "warnings": [
      {"type": "privacy_detection_failed", "field": "phone", "message": "..."}
    ],
    "summary": "数据清洗完成；原始数据 1000 行 → 清洗后 985 行；..."
  }
}
```

## 使用示例

```python
from data_cleaner import DataCleaner

# 初始化
cleaner = DataCleaner()

# 准备输入数据
input_data = {
    "data": [
        {"name": "张三", "phone": "13812345678", "age": 25},
        {"name": "张三 ", "phone": "13812345678", "age": 25},
        {"name": "李四", "phone": "", "age": None}
    ],
    "schema": {"columns": ["name", "phone", "age"], "row_count": 3}
}

# 配置参数
params = {
    "trimWhitespace": True,
    "removeDuplicates": True,
    "handleMissing": "drop",
    "privacyCheck": True
}

# 执行处理
result = cleaner.process(input_data, params)

# 获取处理摘要
summary = result["report"]["summary"]
print(summary)
# 输出: 数据清洗完成；删除重复行 1 行；处理空值 1 个；脱敏处理 2 处敏感信息。
```

## 架构设计亮点

1. **PIPELINE_ORDER 流水线设计**：明确的处理顺序，避免功能冲突
2. **PrivacyMasker 独立模块**：可测试、可替换、可扩展
3. **结构化 warnings**：支持下游智能体自动决策
4. **质量指标体系**：null_rate_before/after 对比，量化清洗效果
5. **容错机制**：可恢复错误记录警告继续执行，不可恢复错误快速失败

## 注意事项

- 空值填充使用统一填充值（字段级填充在后续版本扩展）
- 隐私脱敏支持指定字段（`privacyFields`）或全局扫描
- 日期标准化支持常见格式自动识别

## 版本历史

- **v1.0.0** (2026-04-28): 首次发布，支持5大核心功能

---

_算子版本：1.0.0_
_更新日期：2026-04-28_
