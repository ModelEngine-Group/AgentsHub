# 验证记录

归档包在 2026-08-12 完成以下验证：

| 验证项 | 环境 | 结果 |
|---|---|---|
| 目录、JSON、工具声明与敏感信息扫描 | Python 3.12.13 | 通过 |
| 离线数据处理冒烟测试 | Python 3.12.13 | 通过 |
| Nexent MCP 初始化握手 | Nexent v2.4.0 | 通过 |
| Nexent MCP 服务注册与 6 工具扫描 | Nexent v2.4.0 | 通过 |
| Nexent `agent.json` 导入与官方导出 | Nexent v2.4.0 | 通过 |
| DataMate 健康、模板、网关与算子目录检查 | DataMate `965fb70` | 通过 |
| `require_datamate=true` 强制连接调用 | Nexent `/tool/validate` | 通过 |
| 数据处理 -> 知识图谱 -> NL2SQL 端到端调用 | Nexent `/tool/validate` | 通过 |
| MCP SSE 服务启动与初始化握手 | Python 3.12.13 / MCP 1.29.0 | 通过 |
| MCP 六工具发现 | Python 3.12.13 / MCP 1.29.0 | 通过 |
| MCP `process_medical_data` 实际调用 | Python 3.12.13 / MCP 1.29.0 | 通过 |

真实端到端验证结果：

- Nexent 从 `http://host.docker.internal:18100/sse` 发现 6 个 MCP 工具。
- DataMate `connected=true`，`fallback.used=false`；健康接口、5 组清洗模板、网关和 210 个算子目录均可访问。
- 任务一处理 3 条集成测试记录，成功率 `1.0`。
- 任务二生成 5 个节点、4 条边，三元组有效率与证据覆盖率均为 `1.0`。
- 任务三生成并校验只读 SQL，返回 4 类实体分布，状态为 `ok`。
- `agent.json` 在无预置模型的新 Nexent 实例中完成导入和官方导出；聊天路由需在目标实例中选择可用的工具调用模型。

验证输入只用于集成测试，运行产物已从开源归档中移除。

复核命令：

```bash
python3 scripts/verify_package.py
PYTHONPATH="mcps/src:mcps" python3 -m unittest discover -s tests -v
```
