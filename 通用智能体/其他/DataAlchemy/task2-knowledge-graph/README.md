# 任务二：知识图谱问答智能体 (40分)

## 📋 任务目标

基于Nexent和DataMate平台开发知识图谱问答智能体，实现医疗实体的识别、关系抽取、知识图谱构建和智能问答功能。

## 🎯 功能需求

| 功能 | 描述 | 优先级 |
|:---|:---|:---:|
| 实体识别 | 识别医疗文本中的疾病、症状、药物等实体 | P0 |
| 关系抽取 | 抽取实体之间的关系 | P0 |
| 知识图谱构建 | 将抽取结果存储为图谱结构 | P0 |
| 自然语言问答 | 基于知识图谱回答医疗问题 | P1 |
| 图谱可视化 | 支持图谱查询和展示 | P2 |

## 📂 目录结构

```
task2-knowledge-graph/
├── agents/                    # Nexent智能体配置
│   └── kg_qa_agent.yaml
├── operators/                # DataMate算子
│   ├── entity_extractor/
│   ├── relation_extractor/
│   └── triplet_builder/
├── schema/                    # 知识图谱Schema
│   └── medical_schema.json
├── graph/                     # 图谱存储
│   └── kg_storage.py
├── tests/                     # 测试
│   └── test_extraction.py
└── data/                      # 测试数据
    └── samples/
```

## 🛠️ 算子清单

### 1. 实体抽取算子 (entity_extractor)

- **功能**: 从医疗文本中识别实体
- **实体类型**: 疾病、症状、药物、治疗、检查
- **输入**: 医疗文本
- **输出**: 实体列表

### 2. 关系抽取算子 (relation_extractor)

- **功能**: 抽取实体之间的关系
- **关系类型**: 治疗、诊断、导致、并发
- **输入**: 实体列表+原始文本
- **输出**: 关系列表

### 3. 三元组构建算子 (triplet_builder)

- **功能**: 将实体和关系构建为三元组
- **输入**: 实体列表+关系列表
- **输出**: 三元组列表

## 🏥 医疗知识图谱Schema

```json
{
  "entities": [
    {"type": "疾病", "description": "医学疾病名称"},
    {"type": "症状", "description": "疾病表现症状"},
    {"type": "药物", "description": "药品名称"},
    {"type": "治疗", "description": "治疗方法"},
    {"type": "检查", "description": "医学检查项目"}
  ],
  "relations": [
    {"type": "导致", "source": "疾病", "target": "症状"},
    {"type": "治疗", "source": "药物", "target": "疾病"},
    {"type": "用于", "source": "药物", "target": "症状"},
    {"type": "诊断", "source": "检查", "target": "疾病"}
  ]
}
```

## 📊 验收标准

- [ ] 能识别医疗文本中的5种以上实体类型
- [ ] 能抽取实体间的4种以上关系
- [ ] 能构建可查询的知识图谱
- [ ] 能回答常见的医疗咨询问题
- [ ] 问答准确率达到90%以上

## 👤 负责人



---
*最后更新: 2026-04-26*
