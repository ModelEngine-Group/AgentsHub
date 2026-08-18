# KGRelationExtractor - 医疗关系抽取算子

## 功能
基于模式触发词 + 实体类型共现推理，从医疗文本中抽取4类关系：
- **导致**：疾病 → 症状（如糖尿病→多饮多尿）
- **治疗**：药物 → 疾病（如二甲双胍→糖尿病）
- **用于**：药物 → 症状（如布洛芬→退热）
- **禁忌**：药物 → 疾病（如阿司匹林→胃溃疡）

## 输入格式
```json
{
  "text": "患者糖尿病多年，服用二甲双胍治疗",
  "entities": [
    {"type": "疾病", "name": "糖尿病", "start": 2, "end": 5},
    {"type": "药物", "name": "二甲双胍", "start": 10, "end": 14}
  ]
}
```

## 输出格式
```json
{
  "relations": [
    {
      "head": "二甲双胍",
      "head_type": "药物",
      "relation": "治疗",
      "tail": "糖尿病",
      "tail_type": "疾病",
      "confidence": "high",
      "evidence": "服用二甲双胍治疗"
    }
  ]
}
```

## 处理流程
1. validate_input - 校验输入格式
2. prepare_entities - 准备文本和实体数据
3. extract_relations - 模式匹配 + 共现推理 → 合并去重
4. build_output - 构建输出结果

## 技术特点
- 20+中文触发词模式（导致/引起/治疗/禁忌等）
- 实体类型对有效性校验
- 高/中置信度分级
- 纯Python实现，零外部依赖
