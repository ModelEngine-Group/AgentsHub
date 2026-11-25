# CHANGELOG

## v1.0.0

首个版本，对应 AgentsHub 首次提交。

**新增**

- 智能体定义 `agent.json`：含角色、约束、5 组示例对话，声明 7 个 MCP 工具与知识库检索工具；
- Skill `calculus-tutoring-playbook`：五步答疑流程、三级提示规范、6 类错误归因、验算时机与表述规范、学术诚信边界；
- 知识库 9 篇：按同济第七版大纲组织的 7 章讲义（37 个知识点、127 条易错点）+ 常见错误速查 + 使用说明，统一 `HM-xx-xx` 编号；
- MCP 服务 `math_qa_mcp.py`：`locate_topic`、`eval_expr`、`check_limit`、`check_derivative`、`check_integral`、`compare_expressions`、`render_tutor_card`，支持 stdio / SSE / Streamable HTTP；
- 零依赖验算内核 `math_qa_engine`：递归下降表达式解析、极限收敛性判定、Richardson 外推求导、自适应 Gauss–Legendre 积分、表达式等价比对、五段式卡片渲染；
- 单元测试 60 个，覆盖解析、极限、导数、积分、等价比对、知识点索引与卡片结构；
- 评测集 `evaluation/cases.json`（10 条）与 3 份真实答疑记录 `examples/`。

**已知取舍**

- `model_ids` 留空、`verification_config` 置为 `null`：前者与租户绑定，后者在不同平台版本上行为不一致，交由导入者按环境配置；
- 数值验算只做核对，不做符号推导，避免把「可能算错」包装成「已证明」。
