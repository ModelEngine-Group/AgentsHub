# DataMate 集成说明

## 1. 本文目标

本文说明当前 Demo 如何使用 DataMate 的数据处理思想，并解释已经实现的 DataMate 风格算子适配样例。

当前阶段不直接修改 DataMate 主项目源码，而是在 Demo 内提供一个轻量 adapter，用来证明：

> Demo 中已有的数据处理算子，可以迁移为 DataMate `runtime/ops` 风格的 `sample -> sample` 算子。

## 2. DataMate 数据处理能力是什么？

根据 `DataMate/runtime/ops/README.md`，DataMate 的算子生态用于承载数据清洗、格式转换、文本切分、质量过滤等数据处理能力。

一个标准 DataMate 算子包通常包含：

```text
operator_package/
├── __init__.py        # 算子注册入口
├── metadata.yml       # 算子元数据、UI 参数定义、资源配置
├── process.py         # 算子核心逻辑
├── requirements.txt   # 可选依赖
└── README.md          # 可选说明文档
```

核心执行方式是：

```python
def execute(self, sample: dict) -> dict:
    ...
    return sample
```

其中 `sample` 是一条数据样本，常见字段包括：

- `text`：文本内容
- `data`：二进制数据
- `fileName`：文件名
- `fileType`：文件类型
- `filePath`：文件路径
- `ext_params`：额外参数

## 3. Demo 中已经实现的 DataMate 风格样例

当前 Demo 已新增：

```text
adapters/
├── __init__.py
├── datamate_adapter.py
├── datamate_task1_mappers.py
└── datamate_task1_package/
    ├── __init__.py
    ├── metadata.yml
    └── process.py
```

核心类：

```python
CleanTextDataMateMapper
NormalizeMedicalRecordMapper
ExtractMedicalFieldsMapper
```

它把 Demo 中已有的：

```python
CleanTextOperator
```

包装成 DataMate 风格：

```text
输入 sample 字典
  ↓
清洗 sample["text"]
  ↓
返回新的 sample 字典
```

`NormalizeMedicalRecordMapper` 和 `ExtractMedicalFieldsMapper` 进一步覆盖任务一结构化医疗记录的样本级处理：

```text
输入单条医疗记录 sample
  ↓
规范化 diagnosis / medicine / department 等字段
  ↓
抽取 patient_id / diagnosis / medicine 等核心字段
  ↓
返回新的 sample 字典
```

`adapters/datamate_task1_package/` 是一个可迁移到 DataMate `runtime/ops/user/` 的示例包。它包含 DataMate 规范要求的 `metadata.yml`、`process.py` 和 `__init__.py` 注册入口。

## 4. 使用示例

示例代码：

```python
from adapters.datamate_adapter import CleanTextDataMateMapper
from adapters.datamate_task1_mappers import ExtractMedicalFieldsMapper, NormalizeMedicalRecordMapper

sample = {
    "text": "  发热\n\n  咳嗽   3天。 ",
    "fileName": "record.txt",
    "fileType": "txt",
}

result = CleanTextDataMateMapper(
    remove_digits=True,
    remove_punctuation=True,
).execute(sample)

print(result)
```

输出效果：

```python
{
    "text": "发热 咳嗽 天",
    "fileName": "record.txt",
    "fileType": "txt",
}
```

可以看到：

- `text` 字段被清洗。
- `fileName`、`fileType` 等元数据被保留。
- 输入输出都符合 DataMate 的 `sample -> sample` 思路。

结构化医疗记录样例：

```python
sample = {
    "patient_id": "P001",
    "diagnosis": " 肺炎  ",
    "medicine": " 阿莫西林   头孢 ",
    "department": " 呼吸内科 ",
}

normalized = NormalizeMedicalRecordMapper(
    columns=["diagnosis", "medicine", "department"],
).execute(sample)

selected = ExtractMedicalFieldsMapper(
    fields=["patient_id", "diagnosis", "medicine", "department"],
).execute(normalized)
```

## 5. 测试方式

运行 DataMate adapter 专项测试：

```bash
uv run pytest tests/test_datamate_adapter.py
uv run pytest tests/test_datamate_task1_mappers.py
```

运行全量测试：

```bash
uv run pytest
```

当前验证结果：

- `tests/test_datamate_adapter.py`：2 个测试通过。
- `tests/test_datamate_task1_mappers.py`：3 个测试通过。

## 6. 为什么先选 CleanTextOperator？

优先选择 `CleanTextOperator` 的原因：

1. 输入简单：只需要一段文本。
2. 输出简单：返回清洗后的文本。
3. 非常接近 DataMate `Mapper` 算子的使用方式。
4. 不依赖数据库、文件系统、图谱文件或外部服务。
5. 适合作为第一个最小可迁移样例。

这比一开始迁移 `GraphBuildOperator` 或 `DataAnalysisAgent` 更稳，因为后者更像批处理或分析任务，不适合直接作为单条样本级算子。

## 7. 如果未来迁移到真实 DataMate，需要怎么做？

当前 Demo adapter 只是轻量验证样例。若要迁移到真实 DataMate `runtime/ops`，建议创建一个算子包，例如：

```text
DataMate/runtime/ops/user/clean_text_mapper/
├── __init__.py
├── metadata.yml
├── process.py
├── requirements.txt
└── README.md
```

本 Demo 已提供一个可参考的任务一 package 示例：

```text
adapters/datamate_task1_package/
├── __init__.py
├── metadata.yml
└── process.py
```

### 7.1 process.py 迁移思路

真实 DataMate 版本中，`process.py` 应继承 DataMate 的 `Mapper`：

```python
from datamate.core.base_op import Mapper


class CleanTextMapper(Mapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_digits = kwargs.get("removeDigits", False)
        self.remove_punctuation = kwargs.get("removePunctuation", False)

    def execute(self, sample: dict) -> dict:
        text = sample["text"]
        # 这里可以复用 Demo 中 CleanTextOperator 的清洗逻辑
        sample["text"] = clean_text(text, self.remove_digits, self.remove_punctuation)
        return sample
```

### 7.2 metadata.yml 迁移思路

`metadata.yml` 需要描述算子名称、输入输出、UI 参数和资源需求，例如：

```yaml
name: 医疗文本清洗算子
description: 清洗医疗文本中的多余空格、数字和标点
language: python
raw_id: CleanTextMapper
version: 1.0.0
vendor: datamate
modal: text
inputs: text
outputs: text
types:
  - cleaning
settings:
  removeDigits:
    name: 是否去除数字
    description: 开启后会删除文本中的数字
    type: switch
    defaultVal: "false"
    required: false
  removePunctuation:
    name: 是否去除标点
    description: 开启后会删除文本中的标点符号
    type: switch
    defaultVal: "false"
    required: false
```

### 7.3 __init__.py 注册思路

DataMate 算子需要注册到 `OPERATORS`：

```python
from datamate.core.base_op import OPERATORS


OPERATORS.register_module(
    module_name="CleanTextMapper",
    module_path="ops.user.clean_text_mapper.process",
)
```

## 8. 当前阶段不做什么？

当前阶段只做 Demo 内的适配验证，不做以下事情：

- 不直接修改 `DataMate/runtime/ops` 主项目源码。
- 不启动 DataMate 后端服务。
- 不连接 PostgreSQL、Milvus、Ray 等运行环境。
- 不上传真实算子包。
- 不处理 DataMate 前端 UI 配置。

这些内容会增加部署复杂度，不适合作为当前阶段的第一步。

## 9. 当前验收状态

已完成：

- [x] 明确 Demo 算子与 DataMate 算子接口差异。
- [x] 实现 `CleanTextDataMateMapper` 适配样例。
- [x] 实现任务一结构化记录 mapper：`NormalizeMedicalRecordMapper`、`ExtractMedicalFieldsMapper`。
- [x] 提供任务一 DataMate package 示例：`adapters/datamate_task1_package/`。
- [x] 编写 `tests/test_datamate_adapter.py`。
- [x] 编写 `tests/test_datamate_task1_mappers.py`。
- [x] 通过 adapter 专项测试。
- [x] 通过项目全量测试。
- [x] 说明未来迁移到真实 DataMate `runtime/ops` 的路径。

## 10. 下一步

DataMate 侧 7.1 完成后，建议进入 Nexent 侧 7.2 的实现：

```text
实现 1 个 Nexent 风格 Tool 适配样例
```

推荐优先封装：

```python
KGQAAgent.run(question)
```

为：

```python
ask_medical_kg(question: str) -> dict
```

然后在文档中说明如何迁移为 Nexent 的 MCP Tool。
