# 课程数据分析助手

基于 Nexent 的课程数据分析智能体，面向课程成绩与学习行为数据，提供数据质量检查、统计摘要、相关性和条件查询能力。交付包同时覆盖 **MCP 工具、Skill 工作流、知识库口径**，可在本地复现后再导入 Nexent。

## 能力

- **数据质量检查**：样本量、字段完整性、缺失值、数值范围。
- **成绩统计**：均值、标准差、最小值、最大值、四分位数、中位数。
- **相关性分析**：出勤率、作业、期中、学习时长、实验成绩与期末成绩的相关系数。
- **风险统计**：按阈值统计期末成绩预警人数和占比。
- **条件查询**：按期末成绩、出勤率和性别筛选记录。
- **规范报告**：按数据质量、关键统计、相关性发现、下一步建议输出结论。

## 目录结构

```text
课程数据分析助手/
├── agent.json                         # Nexent 智能体导入配置
├── README.md
├── requirements.txt
├── data/course_records.csv           # 示例课程数据
├── knowledge_base/课程数据分析方法.md  # 知识库源文档
├── skills/数据分析工作流/SKILL.md      # Skill 源文档
├── skills/数据分析工作流.zip           # 可上传到 Nexent 的技能包
├── mcps/
│   ├── server.py                     # 标准库实现 + FastMCP 封装
│   └── course-data-analysis.json     # MCP 启动配置
└── tests/test_course_data_tools.py   # 本地校验测试
```

## 本地校验

```bash
cd 通用智能体/数据处理/课程数据分析助手
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

未安装 FastMCP 时，测试仍可直接验证工具函数逻辑；FastMCP 仅用于启动 MCP 服务。

## 导入 Nexent

1. **上传知识库**：在 Nexent 知识库中导入 `knowledge_base/课程数据分析方法.md`，索引名建议为 `课程数据分析方法`。
2. **上传 Skill**：进入智能体开发 → 选择技能 → 构建技能，上传 `skills/数据分析工作流.zip`。
3. **启动 MCP**：在智能体目录运行 `python mcps/server.py`；Nexent 侧新增 MCP 服务，名称为 `course-data-analysis`，URL 为 `http://host.docker.internal:8001/mcp`。
4. **导入智能体**：上传 `agent.json`，重新选择模型，将 `KnowledgeBaseSearchTool` 的 `index_names` 从 `<TO_CONFIG:课程数据分析方法>` 替换为实际知识库名。
5. **验证用例**：
   - `请分析这批课程数据并给出教学建议`
   - `期末成绩低于 70 分的学生有哪些？`
   - `出勤率和期末成绩有什么关系？`

## 设计说明

- MCP 工具只读取 `data/course_records.csv`，不联网、不写入、不删除文件。
- Skill 负责固定“检查 → 分析 → 查询 → 知识库校验 → 报告”的执行顺序。
- 知识库承载字段契约、指标口径、相关性解释、报告模板和治理边界。
- 所有统计数值由 MCP 计算；智能体不得手动估算或编造结果。

## 边界

- 数据仅用于课程教学分析，不用于个人画像、公开披露或自动化处置。
- 相关性不是因果性；样本量小于 5 时提示结论不稳定。
- 当前工具仅支持课程成绩与学习行为数据，不支持医疗、金融等其他领域数据。
