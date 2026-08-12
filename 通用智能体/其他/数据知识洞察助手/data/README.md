# 数据源说明

本目录包含用于 Demo 的医疗示例数据。

## 文件列表

### medical_records.csv

结构化医疗记录数据，共 15 条。

**字段说明：**

| 字段名 | 含义 | 示例 |
|---|---|---|
| patient_id | 患者ID | 1 |
| age | 年龄 | 45 |
| gender | 性别 | 男/女 |
| chief_complaint | 主诉 | 咳嗽发热3天 |
| diagnosis | 诊断 | 肺炎 |
| medicine | 用药 | 阿莫西林 |
| admission_date | 入院日期 | 2024-01-15 |
| department | 科室 | 呼吸内科 |

### medical_text.txt

非结构化医疗文本，包含 15 种常见疾病的描述。

**涵盖疾病：**

肺炎、冠心病、偏头痛、急性胃肠炎、支气管炎、类风湿性关节炎、腰椎间盘突出、多囊卵巢综合征、2型糖尿病、缺铁性贫血、荨麻疹、急性胆囊炎、尿路感染、急性心肌梗死、急性扁桃体炎

**用途：**

- 知识图谱构建的原始文本输入
- 实体识别算子的训练/测试数据
- 关系抽取算子的输入数据
- 问答智能体的问答依据

## 数据使用

```python
# 读取 CSV
import pandas as pd
df = pd.read_csv("data/medical_records.csv")

# 读取文本
with open("data/medical_text.txt", "r", encoding="utf-8") as f:
    text = f.read()
```
