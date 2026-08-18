# OperatorTest - 算子测试环境

> 用于测试 CCF ModelEngine 比赛中开发的所有算子

## 📁 目录结构

```
D:\PythonProject\OperatorTest\
├── test_data/              # 测试数据文件
│   ├── sample_utf8.csv     # UTF-8 CSV 测试数据
│   ├── sample_gbk.csv      # GBK 编码测试数据
│   ├── sample_mixed.csv    # 中英文混合数据
│   ├── sample.json         # JSON 数组测试数据
│   ├── sample_single.json  # 单对象 JSON 测试数据
│   └── large_file.csv      # 大文件（10000行）测试数据
├── test_results/           # 测试报告输出目录
├── test_data_loader.py     # DataLoader 算子测试脚本
└── README.md               # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy
```

### 2. 运行测试

```bash
cd D:\PythonProject\OperatorTest
python test_data_loader.py
```

### 3. 查看报告

测试报告自动保存到 `test_results/` 目录，格式为 JSON。

## 📋 测试覆盖

| 测试类别 | 测试用例数 | 说明 |
|:---|:---:|:---|
| 功能正确性 | 4 | 单文件、多文件、多格式加载 |
| 编码兼容性 | 2 | UTF-8、GBK 编码支持 |
| 合并策略 | 2 | concat、union 合并逻辑 |
| 容错与降级 | 3 | 文件不存在、连续错误、快速失败 |
| 性能与分块 | 1 | 大文件分块处理 |
| 质量评分 | 1 | 评分逻辑验证 |
| 边界测试 | 2 | 空路径、格式自动检测 |
| **总计** | **15** | - |

## 📝 测试报告格式

```json
{
  "timestamp": "2026-04-28 12:00:00",
  "total": 15,
  "passed": 15,
  "failed": 0,
  "details": [
    {
      "status": "PASS",
      "msg": "✅ 单文件加载成功，共 5 条数据"
    }
  ]
}
```

## 🔧 添加新测试

在 `test_data_loader.py` 中添加新的测试函数：

```python
def test_my_new_feature():
    """测试XX：新功能测试"""
    log("=" * 50)
    log("测试XX：新功能测试")
    log("=" * 50)

    loader = DataLoaderMapper(...)
    result = loader({})

    assert_check(
        result["execute_result"] == True,
        "✅ 新功能正常",
        "❌ 新功能异常"
    )
```

然后在 `run_all_tests()` 函数的 `test_functions` 列表中添加即可。

## 📦 测试数据说明

| 文件名 | 用途 | 说明 |
|:---|:---|:---|
| `sample_utf8.csv` | 基础功能测试 | 5行标准中文数据 |
| `sample_gbk.csv` | 编码测试 | GBK 编码中文数据 |
| `sample_mixed.csv` | 混合数据测试 | 中英文混合 |
| `sample.json` | JSON 数组测试 | 3条 JSON 对象 |
| `sample_single.json` | 单对象测试 | 1条 JSON 对象 |
| `large_file.csv` | 大文件测试 | 10000 行数据 |

---

> CCF ModelEngine 赛道 · 算子测试环境
