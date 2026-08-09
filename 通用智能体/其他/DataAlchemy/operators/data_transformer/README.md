# DataTransformer - 数据转换算子

> 版本：1.0.0  
> 开发者：ccf-team

## 功能概述

支持字段重命名、字段选择、类型转换、值替换、条件筛选、派生列的数据转换算子。

## 处理顺序

```
select_fields → drop_fields → rename_fields → type_conversion → value_mapping → filter_condition → derive_columns
```

## 参数说明

| 参数 | 类型 | 说明 |
|:---|:---|:---|
| `renameFields` | textarea(JSON) | 字段映射，如 `{"old": "new"}` |
| `selectFields` | textarea(逗号分隔) | 保留字段，如 `name,age` |
| `dropFields` | textarea(逗号分隔) | 删除字段，如 `id,tmp` |
| `typeConversions` | textarea(JSON) | 类型映射，如 `{"age": "int"}` |
| `valueMappings` | textarea(JSON) | 值映射，如 `{"gender": {"M": "男"}}` |
| `filterCondition` | input | 筛选表达式，如 `age > 18` |
| `deriveColumns` | textarea(JSON) | 派生列，如 `{"bmi": "weight/(height/100)**2"}` |

## 输入输出

**输入**：来自 data_cleaner 的 `{data: [...], schema: {...}}`  
**输出**：`{data: [...], schema: {...}, report: {...}}`

## 使用示例

```json
{
  "renameFields": "{\"name\": \"patient_name\"}",
  "selectFields": "patient_name,age",
  "typeConversions": "{\"age\": \"int\"}"
}
```

## 依赖

- pandas >= 1.5.0
- numpy >= 1.23.0

## 文件结构

```
data_transformer/
├── __init__.py          # 包入口
├── process.py           # 核心处理逻辑
├── metadata.yml         # UI配置与元数据
├── requirements.txt     # Python依赖
└── README.md            # 本文件
```
