# 任务一：数据处理智能体 (30分)

## 📋 任务目标

基于DataMate平台开发数据处理智能体，实现医疗数据的读取、清洗、验证和分析功能。

## 🎯 功能需求

| 功能 | 描述 | 优先级 |
|:---|:---|:---:|
| 数据读取 | 支持CSV/JSON/Parquet格式 | P0 |
| 数据清洗 | 去重、标准化、缺失值处理 | P0 |
| 数据验证 | 完整性检查、格式校验 | P1 |
| 统计计算 | 基础统计指标计算 | P1 |
| 数据导出 | 支持多种格式导出 | P2 |

## 📂 目录结构

```
task1-data-processing/
├── agents/                    # Nexent智能体配置
│   └── data_processing_agent.yaml
├── operators/                # DataMate算子
│   ├── csv_reader/
│   ├── json_reader/
│   ├── text_cleaner/
│   ├── data_validator/
│   └── stats_calculator/
├── pipeline/                 # 流水线配置
│   └── etl_pipeline.yaml
├── tests/                    # 测试
│   └── test_operators.py
└── data/                     # 测试数据
    └── samples/
```

## 🛠️ 算子清单

### 1. CSV读取算子 (csv_reader)

- **功能**: 读取CSV格式医疗数据
- **输入**: CSV文件路径
- **输出**: 标准化数据字典

### 2. JSON读取算子 (json_reader)

- **功能**: 读取JSON格式数据
- **输入**: JSON文件路径
- **输出**: 标准化数据字典

### 3. 文本清洗算子 (text_cleaner)

- **功能**: 文本标准化、去重、缺失值处理
- **输入**: 原始文本数据
- **输出**: 清洗后的文本数据

### 4. 数据验证算子 (data_validator)

- **功能**: 数据质量检查、完整性验证
- **输入**: 清洗后数据
- **输出**: 验证报告

### 5. 统计计算算子 (stats_calculator)

- **功能**: 计算基础统计指标
- **输入**: 验证通过数据
- **输出**: 统计结果

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行测试

```bash
pytest tests/ -v
```

### 3. 开发新算子

参考 `docs/DEVELOPMENT.md` 中的算子开发规范。

## 📊 验收标准

- [ ] 能正确读取CSV格式医疗数据
- [ ] 能识别并处理重复数据
- [ ] 能处理缺失值和异常值
- [ ] 能生成数据质量报告
- [ ] 能正确导出清洗后的数据

## 👤 负责人



---
*最后更新: 2026-04-26*
