# DataMate / Nexent 集成增强规划

## 1. 当前状态

当前 `data_knowledge_insight_demo` 已经完成一个独立可运行的“数据 → 知识 → 洞察”闭环：

- 任务一：数据处理智能体，完成 CSV 清洗、文本处理和结果导出。
- 任务二：知识图谱智能体，完成实体识别、关系抽取、三元组生成、图谱构建和图谱问答。
- 任务三：数据分析智能体，完成图谱统计、可视化、SQLite 构建和 NL2SQL MVP。
- 调用方式：支持 `run_demo.py` 一键运行和 FastAPI 接口调用。
- 验证方式：已有 pytest 测试覆盖主要算子、智能体、API 和一键运行入口。

总计划下一阶段是第 7 阶段：DataMate / Nexent 集成增强。

## 2. 是否应该现在做集成增强？

结论：**应该做，但不应该直接大改 DataMate 或 Nexent 主项目源码。**

原因如下：

1. **应该做的原因**
   - 赛题目标不是只写一个孤立 Demo，而是要体现它与 DataMate、Nexent 的关系。
   - DataMate Runtime 明确包含 `runtime/ops` 算子生态，和我们 Demo 的 `operators/` 目录有天然对应关系。
   - Nexent 文档说明其工具系统基于 MCP，适合把我们的任务能力封装成可调用工具。
   - 做集成说明和适配样例，可以让答辩时说明“如何从 Demo 走向真实平台接入”。

2. **不应该直接大改主项目的原因**
   - DataMate 和 Nexent 都是较大的工程，直接修改主源码容易引入部署、依赖、数据库和服务启动复杂度。
   - 当前 Demo 已经稳定可测，直接耦合主项目可能破坏已有闭环。
   - 用户当前阶段更需要可理解、可展示、可验收的路线，而不是一次性深度集成。
   - 因此更合适的路线是：先做文档对照，再做轻量适配层，最后提供可迁移样例。

## 3. 推荐路线

推荐采用“三层递进”路线：

```text
第一层：结构对照文档
  说明 Demo operators/agents 与 DataMate/Nexent 架构如何对应。

第二层：轻量适配层
  在 Demo 内新增 adapters/，不直接改外部主项目。

第三层：最小可迁移样例
  提供 1 个 DataMate 风格算子适配样例 + 1 个 Nexent MCP 工具样例。
```

这条路线的优点是：

- 风险低：不破坏现有可运行 Demo。
- 可解释：每一步都能讲清楚为什么做。
- 可测试：适配层仍然可以用 pytest 验证。
- 可答辩：能展示从独立 Demo 到真实平台接入的演进路径。

## 4. 阶段划分与任务清单

### 4.1 阶段 A：结构对照与集成判断

目标：搞清楚 Demo、DataMate、Nexent 三者应该如何分工。

任务：

- [ ] 对比 Demo `operators/` 与 DataMate `runtime/ops` 的目录和职责。
- [ ] 对比 Demo `agents/` 与 Nexent `CoreAgent` / MCP 工具体系的职责。
- [ ] 输出可迁移能力清单：哪些算子适合迁移，哪些智能体能力适合封装为工具。
- [ ] 明确暂不做的内容：不启动完整 DataMate 服务、不修改 Nexent 数据库、不接真实 LLM 工具市场。

验收标准：

- 生成 `docs/datamate_nexent_mapping.md`。
- 文档中至少包含一张“Demo 模块 → DataMate/Nexent 对应关系表”。
- 文档中明确写出“做什么”和“不做什么”。

### 4.2 阶段 B：DataMate 风格算子适配

目标：证明 Demo 算子可以向 DataMate 算子生态迁移。

建议优先选择简单算子：`CleanMissingValueOperator` 或 `SplitParagraphOperator`。

任务：

- [ ] 新建 `adapters/datamate_adapter.py`。
- [ ] 定义一个轻量 `DataMateOperatorAdapter`。
- [ ] 包装 1 个现有 Demo 算子为 DataMate 风格调用接口。
- [ ] 编写测试 `tests/test_datamate_adapter.py`。
- [ ] 编写说明文档，解释如何迁移到 DataMate `runtime/ops`。

验收标准：

- `uv run pytest tests/test_datamate_adapter.py` 通过。
- 适配样例不破坏现有 `uv run pytest` 全量测试。
- 文档能说明输入、输出、执行方式和迁移路径。

### 4.3 阶段 C：Nexent MCP 工具适配

目标：证明 Demo 智能体能力可以作为 Nexent 工具被调用。

当前推荐方案是独立 FastMCP Server + Nexent 远程 MCP 接入。工具粒度保持业务级，不把底层算子逐个暴露给 Nexent。

任务：

- [x] 新建 `adapters/nexent_mcp_tools.py`。
- [x] 设计 4 个业务级 MCP 工具函数：
  - `process_medical_data_tool(task_description: str, ...) -> dict`
  - `build_medical_knowledge_graph_tool(task_description: str, ...) -> dict`
  - `ask_medical_kg_tool(question: str, ...) -> dict`
  - `analyze_medical_data_tool(task_description: str, ...) -> dict`
- [x] 保持普通 Python 函数先可测试，再可选注册到 FastMCP。
- [x] 为 `ask_medical_kg_tool(...)` 和 `analyze_medical_data_tool(...)` 增加前置产物检查，缺失时返回 `missing_prerequisite`。
- [x] 在 README 和任务文档中给出 Nexent MCP 注册地址：`http://127.0.0.1:8001/sse`，Docker 下使用 `http://host.docker.internal:8001/sse`。
- [x] 编写测试 `tests/test_nexent_mcp_tools.py`。

验收标准：

- `uv run pytest tests/test_nexent_mcp_tools.py -v` 通过。
- 4 个工具函数分别复用现有 `DataProcessingAgent`、`KnowledgeGraphAgent`、`KGQAAgent` 和 `DataAnalysisAgent`。
- 文档中说明如何把独立 FastMCP Server 注册为 Nexent 远程 MCP 服务。

### 4.4 阶段 D：统一集成文档与答辩材料

目标：把集成增强变成可展示、可答辩的成果。

任务：

- [ ] 编写 `docs/datamate_integration.md`。
- [ ] 编写 `docs/nexent_integration.md`。
- [ ] 更新项目 README，增加“与 DataMate/Nexent 的关系”。
- [ ] 在总计划中勾选第 7 阶段相关项目。

验收标准：

- 文档能够回答：Demo 为什么能代表 DataMate/Nexent 方向？
- 文档能够回答：如果后续要接入真实平台，应该改哪里？
- `uv run pytest` 全量通过。

## 5. 不建议现在做的事情

以下内容先不做，避免任务失控：

- 不直接修改 DataMate `runtime/ops` 主源码。
- 不启动 DataMate PostgreSQL、Milvus、Ray 等完整服务。
- 不直接修改 Nexent 后端数据库或前端配置。
- 不要求在单元测试中启动真实 Nexent 服务；Demo 侧提供可选 FastMCP Server 入口。
- 不做复杂权限、租户、工具市场发布流程。

这些内容可以放到后续“深度集成”阶段。

## 6. 推荐执行顺序

建议后续按下面顺序推进：

1. 写 `docs/datamate_nexent_mapping.md`。
2. 实现 DataMate 风格算子适配样例。
3. 实现 Nexent MCP 工具适配样例。
4. 补充两个集成说明文档。
5. 更新 README 和总计划勾选项。
6. 运行 API 测试和全量测试。

## 7. 最终验收标准

第 7 阶段完成时，应该满足：

- [ ] 有清晰的 DataMate / Nexent 对照说明。
- [ ] 有至少 1 个 DataMate 风格算子适配样例。
- [x] 有 4 个 Nexent MCP 工具适配样例。
- [x] 有对应测试证明适配样例可运行。
- [ ] 有文档说明如何迁移到真实 DataMate / Nexent 项目。
- [ ] `uv run pytest` 全量通过。

## 8. 当前结论

本阶段应该做，但要控制范围：

> 先做“文档 + 轻量适配层 + 最小样例”，不要现在直接深度侵入 DataMate / Nexent 主工程。

这样既能体现赛题要求中的平台关联，又能保持当前 Demo 稳定、可运行、可测试。
