# KGTripleGenerator - 知识图谱三元组生成算子

## 功能
将实体识别 + 关系抽取的结果组合为标准化的知识图谱三元组：
- 实体存在性校验与类型校验
- 关系-实体类型对合法性校验（如"治疗"关系要求药物→疾病）
- 去重（相同 head-relation-tail 只保留一条）
- 冲突检测（直接冲突 + 禁忌冲突）
- 上下文约束字段预留

## 输入格式
```json
{
  "entities": [
    {"type": "疾病", "name": "糖尿病"},
    {"type": "药物", "name": "二甲双胍"},
    {"type": "症状", "name": "多饮多尿"}
  ],
  "relations": [
    {"head": "二甲双胍", "relation": "治疗", "tail": "糖尿病", "confidence": "high"},
    {"head": "糖尿病", "relation": "导致", "tail": "多饮多尿", "confidence": "high"}
  ]
}
```

## 输出格式
```json
{
  "triples": [
    {
      "head": "二甲双胍", "head_type": "药物",
      "relation": "治疗",
      "tail": "糖尿病", "tail_type": "疾病",
      "confidence": "high",
      "context_constraints": []
    }
  ],
  "conflicts": [],
  "triple_count": 1
}
```

## 处理流程
1. validate_input - 校验输入格式
2. validate_schema - 实体/关系类型合法性校验
3. generate_triples - 组合实体+关系生成三元组
4. detect_conflicts - 冲突检测（直接冲突/禁忌冲突）
5. deduplicate - 去重
6. build_output - 构建输出结果

## 冲突检测类型
- **直接冲突**：same head+relation, different tail
- **禁忌冲突**：同一药物既治疗又禁忌同一疾病（critical级别）

## 技术特点
- 4类实体 + 4类关系的Schema约束校验
- 高/中/低三级置信度保留策略
- 纯Python实现，零外部依赖
