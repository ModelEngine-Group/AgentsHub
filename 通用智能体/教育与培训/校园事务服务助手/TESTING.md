# 平台测试记录

时间：2026-11-19；本机 Nexent；助手 ID 4。

1. 费用查询：实际调用 read_skill_md 和 knowledge_base_search，回答演示资料未配置校园卡补办费用，并引用资料。通过。
2. 目录及工单查询：实际调用 read_skill_md、list_services、get_my_request_status；列出三项演示服务，明确工单未认证，未虚构进度。通过。
3. MCP 函数检查：目录数量、未认证工单、未接入开放时间、未知事项四项断言通过。
4. 平台导出：成功取得包含 agent.json 与 Skill 的 ZIP。

限制：完整验收矩阵、跨实例导入、真实校园认证与业务连接未验证。临时 MCP 进程不保证随容器重启恢复。
