# API 使用说明

本 Demo 提供 FastAPI 接口，用于通过 HTTP 调用已经实现的数据处理、知识图谱和数据分析智能体。

## 启动服务

在 `data_knowledge_insight_demo/` 目录下运行：

```bash
uv run uvicorn api.main:app --reload
```

启动后打开自动接口文档：

```text
http://127.0.0.1:8000/docs
```

## 接口列表

### 健康检查

```http
GET /health
```

返回：

```json
{
  "status": "ok",
  "service": "data-knowledge-insight-demo"
}
```

### 运行任务一：数据处理

```http
POST /task1/run
```

请求体：

```json
{
  "task_description": "请清洗医疗数据"
}
```

会生成：

- `outputs/cleaned_records.csv`
- `outputs/cleaned_text.json`
- `outputs/task1_summary.json`

### 运行任务二：构建知识图谱

```http
POST /task2/build-kg
```

请求体：

```json
{
  "task_description": "请生成医疗知识图谱"
}
```

会生成：

- `outputs/medical_triples.json`
- `outputs/medical_kg.json`
- `outputs/medical_kg.graphml`
- `outputs/medical_kg.html`

### 运行任务二：图谱问答

```http
POST /task2/ask
```

请求体：

```json
{
  "question": "肺炎有哪些症状？"
}
```

返回包含：

- `answers`：答案列表
- `answer`：自然语言答案
- `evidence`：支撑答案的三元组证据

### 运行任务三：数据分析

```http
POST /task3/analyze
```

请求体：

```json
{
  "task_description": "请分析医疗知识图谱"
}
```

会生成：

- `outputs/analysis_summary.json`
- `outputs/analysis_report.md`
- `outputs/charts/disease_symptom_count.png`
- `outputs/charts/drug_disease_count.png`
- `outputs/medical_records.db`

## 测试命令

```bash
uv run pytest tests/test_api.py
uv run pytest
```
