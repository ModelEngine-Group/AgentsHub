# 医疗知识图谱分析报告

## 总体统计

- 节点数量：87
- 边数量：93
- 实体类型分布：{'Disease': 15, 'Symptom': 34, 'Drug': 24, 'Examination': 5, 'Treatment': 9}
- 关系类型分布：{'has_symptom': 48, 'treated_by': 40, 'requires_exam': 5}

## 关键发现

- 关联症状最多的疾病是：尿路感染，症状数量为 5。
- 关联疾病最多的药物是：布洛芬，疾病数量为 3。
- 中心性最高的节点是：肺炎，中心性分数为 0.127907。
- NL2SQL 模板评估准确率：100%（6/6）。

## 图表输出

- `outputs/charts/disease_symptom_count.png`
- `outputs/charts/drug_disease_count.png`

## NL2SQL MVP

- SQLite 数据库：`outputs/medical_records.db`
- 评估样本数：6
- 正确映射数：6
- 准确率：100%
