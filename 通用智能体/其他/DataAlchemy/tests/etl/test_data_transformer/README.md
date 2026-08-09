# test_data_transformer - DataTransformer 算子测试套件

> CCF ModelEngine 赛道 · 数据转换算子测试环境
> 最后更新：2026-04-30

---

## 📁 目录结构

```
test_data_transformer/
├── test_data/                      # 测试数据 fixtures
│   ├── __init__.py
│   └── test_data.py                # 样本数据、预期结果、辅助函数
├── test_data_transformer.py        # 主测试文件（59 个测试用例）
├── pytest.ini                      # pytest 配置
├── requirements.txt                # 测试依赖
├── __init__.py                     # 包入口
└── README.md                       # 本文档
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行全部测试

```bash
cd D:\PythonProject\OperatorTest\test_data_transformer
pytest test_data_transformer.py -v
```

### 3. 按类别运行

```bash
# 单元测试（30个）
pytest test_data_transformer.py -v -k "TestSelectFields or TestDropFields or TestRenameFields or TestTypeConversion or TestValueMapping or TestFilterCondition or TestDeriveColumns"

# 集成测试（8个）
pytest test_data_transformer.py -v -k "TestIntegration"

# 边界测试（6个）
pytest test_data_transformer.py -v -k "TestBoundary"

# 异常测试（5个）
pytest test_data_transformer.py -v -k "TestException"

# 端到端测试（3个）
pytest test_data_transformer.py -v -k "TestE2E"

# 性能测试（3个）
pytest test_data_transformer.py -v -k "TestPerformance"

# 补充测试（4个）
pytest test_data_transformer.py -v -k "TestSupplementary"
```

### 4. 生成覆盖率报告

```bash
pytest test_data_transformer.py --cov=data_transformer --cov-report=html
```

---

## 📋 测试覆盖

| 测试类别 | 用例数 | 覆盖内容 |
|:---|:---:|:---|
| **单元测试** | 30 | 7 大功能：select(4) + drop(3) + rename(4) + type_conversion(6) + value_mapping(3) + filter(5) + derive(5) |
| **集成测试** | 8 | Pipeline 顺序：select+drop, select+rename, rename+type, type+value, rename+derive, type+filter, 全流程, rename+filter |
| **边界测试** | 6 | 空数据、单行、全空参数、筛选为空、全字段选择、类型混合 |
| **异常测试** | 5 | JSON 错误、不支持类型、filter 语法错误、derive 除零、无效输入 |
| **端到端测试** | 3 | 医疗数据完整 ETL、大数据集性能、NaN 处理 |
| **性能测试** | 3 | 10000 行性能、步骤耗时、步骤级性能分析 |
| **补充测试** | 4 | 类型透传、输出格式校验、data_cleaner 对接、性能稳定性 |
| **总计** | **59** | — |

---

## 🧪 测试数据说明

测试数据定义在 `test_data/test_data.py` 中，可通过以下方式使用：

```python
from test_data.test_data import SAMPLE_DATA, EMPTY_PARAMS, create_input_data

# 使用预定义数据
data = create_input_data("basic")

# 使用空参数
params = {**EMPTY_PARAMS, "typeConversions": '{"age": "int"}'}
```

| 数据键名 | 说明 | 行数 |
|:---|:---|:---:|
| `basic` | 基础数据（name, age, dept） | 3 |
| `with_types` | 多类型数据（str, int, float, date, gender） | 2 |
| `with_spaces` | 列名含空格 | 2 |
| `with_zero` | 含零值（除零测试） | 2 |
| `type_mixed` | 类型混合列 | 3 |
| `with_null` | 含 NaN 值 | 3 |
| `medical` | 医疗数据（完整 ETL 场景） | 3 |
| `empty` | 空数据 | 0 |

---

## 📝 测试中发现并修复的 Bug

| Bug | 根因 | 修复方案 |
|:---|:---|:---|
| 日期混合格式失败 | pandas 2.0+ 不支持混合格式自动推断 | `pd.to_datetime(format="mixed")` |
| Int64 列存字符串失败 | Int64 dtype 不接受字符串赋值 | 赋值前 `astype(object)` |
| filter 表达式引号 | 测试写法 `== 张三` 应为 `== "张三"` | 修正测试表达式 |
| E2E 计数错误 | 3 行 gender M→男(2)+F→女(1)=3 | 修正期望值 |

---

## 🔧 添加新测试

```python
class TestMyNewFeature:
    """新功能测试"""

    def test_new_feature(self, transformer, basic_data, basic_params):
        """测试说明"""
        params = {**basic_params, "newParam": "value"}
        result = transformer.process(basic_data, params)

        assert result["report"]["output_rows"] == 3
```

然后在文件末尾的 `TestSupplementary` 类中添加即可。

---

> CCF ModelEngine 赛道 · DataTransformer 算子测试套件
