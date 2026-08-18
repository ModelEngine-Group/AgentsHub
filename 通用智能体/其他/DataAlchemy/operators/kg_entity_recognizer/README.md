# KGEntityRecognizer - 医疗实体识别算子

## 功能
基于医疗词典 + 中文文本匹配，从医疗文本中识别预定义的4类实体：
- **疾病** (Disease)：如糖尿病、高血压、冠心病等
- **症状** (Symptom)：如发热、头痛、胸闷等
- **药物** (Drug)：如二甲双胍、阿司匹林等
- **检查** (Examination)：如血常规、CT、心电图等

## 输入格式
```json
{
  "text": "患者糖尿病多年，出现多饮多尿症状"
}
// 或批量模式
{
  "data": [{"text": "..."}, {"text": "..."}]
}
```

## 输出格式
```json
{
  "entities": [
    {"type": "疾病", "name": "糖尿病", "start": 2, "end": 5},
    {"type": "症状", "name": "多饮多尿", "start": 10, "end": 14}
  ],
  "entity_count": 2
}
```

## 处理流程
1. validate_input - 校验输入格式
2. extract_entities - 基于词典+正则抽取实体
3. build_output - 构建输出结果

## 技术特点
- 内置100+医疗实体词典，覆盖常见疾病、症状、药物、检查
- 首字索引加速匹配
- 贪心最长匹配，避免交叉覆盖
- 纯Python实现，零外部依赖
