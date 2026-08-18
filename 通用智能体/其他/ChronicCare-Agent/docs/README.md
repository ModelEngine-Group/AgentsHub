# ChronicCare-Agent 文档中心

本目录保存 ChronicCare-Agent 决赛交付版的设计、部署、测试、合规和前端证据文档。项目面向慢病随访场景，以 Nexent 单智能体作为交互与任务规划入口，将 DataMate 数据处理、医疗知识图谱、受控 Open SQL、BI 可视化和 Ascend NPU 增强串联为“数据—知识—洞察”闭环。

## 文档结构

1. [技术报告](技术报告.md)（[PDF版](技术报告.pdf)）：了解项目目标、三项任务实现、前端问答和创新价值。
2. [系统架构](architecture.md)：了解模块边界、数据流、MCP 调用和 CPU/NPU 分工。
3. [任务要求映射](task_requirements_mapping.md)：按比赛要求和评分点核对实现证据。
4. [部署与复现指南](deployment_guide.md)：启动服务、接入 Nexent 并执行验收。
5. [测试与性能报告](testing_and_benchmarks.md)：复核 NL2SQL、SQL Guard、图谱、DAG 和 NPU 指标。
6. [数据与模型来源](data_and_model_provenance.md)：了解数据性质、数据规模、模型来源和使用边界。
7. [开源合规说明](open_source_compliance.md)：了解许可证、第三方依赖和数据合规边界。
8. [MCP 与 API 参考](mcp_and_api_reference.md)：查询工具职责、参数、安全边界和长任务行为。
9. [前端证据索引](frontend_evidence_index.md)：将 Nexent 前端问题映射到 42 张截图。

## 当前版本事实

| 项目 | 当前值 |
| --- | ---: |
| 数据版本 | `synthetic_chroniccare` |
| 患者 | 2,000 人 |
| 随访记录 | 8,231 条 |
| 检验记录 | 131,323 条 |
| 用药记录 | 18,248 条 |
| 慢病类型 | 20 种 |
| 知识图谱节点 | 197,404 个 |
| 知识图谱边 | 396,928 条 |
| 实体类型 | 14 类 |
| 关系类型 | 15 类 |
| NL2SQL本地回归盲测 | 239/240（99.58%） |
| DataMate 主线算子 | 11 个 |
| NPU 增强算子 | 2 个 |

上述指标来自`configs/current_metrics.json`和当前主线数据库/图谱产物，文档不对外展示内部图谱评分。

## 文档说明

- 当前数据为参考真实慢病随访业务的字段结构、业务关系和合理取值范围，采用程序化规则与固定随机种子合成的数据，不包含可识别的真实患者身份信息。
- 前端代表性问题和答案集中写在 `技术报告.md`；本目录的证据索引只负责问题、工具和截图的对应关系。
- NPU 数值必须按同一运行批次解释。当前正式报告采用 CPU 64 线程、NPU batch size 1024，CPU/NPU 各预热一次，CPU 2,048 条、NPU 2,048 条和 NPU 全量分别真实计时。
- 项目仅用于慢病随访数据处理、知识组织和辅助分析，不提供临床诊断，不替代医生决策。
