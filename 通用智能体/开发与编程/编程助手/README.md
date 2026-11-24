# 编程助手

基于 Nexent v2.8.5 的中文 Python 编程助手，结合 DeepSeek Flash、编程 Skill、MCP 静态检查和本地向量知识库。支持代码解释、排错建议和测试设计，不执行用户代码。

## 文件
- agent.json：用户最终平台导出配置。
- exports/programming-assistant.zip：完整平台导出，含 Skill ZIP，推荐通过此包导入。
- mcps/：AST分析服务；skills/：Skill源文件。
- knowledge/：原创教学知识库；embedding/：中文向量服务。
- TEST_REPORT.md：实测记录；evidence/：用户提供的调试截图。

## 部署与导入
1. 准备可用 Nexent v2.8.5 实例与租户管理员，配置聊天模型。示例使用 DeepSeek Flash，自行提供密钥。
2. 在Python 3.11独立环境安装 mcps/requirements.txt，运行 mcps/server.py，默认监听127.0.0.1:8765。容器平台需使用受控内网可达地址；容器内可运行 python -c 'from server import mcp; mcp.run(transport="streamable-http",host="0.0.0.0",port=8765)'。服务名 programming-assistant。
3. 按 DEPLOYMENT.md 部署本地向量模型，确认 /health 就绪。
4. 在Nexent添加向量模型：OpenAI兼容服务商，模型名 BAAI/bge-small-zh-v1.5，类型embedding，512维，Base URL为服务地址加/v1，API Key填占位值local。
5. 新建“Python编程规范”，选择向量模型，按 knowledge/python-guide.md 五个二级标题创建5个文本分块，填写标题和来源python-guide.md。不要同时重复导入全文。
6. 在智能体列表导入 exports/programming-assistant.zip，重新选择目标租户模型和MCP。确认Skill已绑定；缺失时上传 skills/programming-assistant/SKILL.md 后添加。
7. 移除源实例旧知识库引用，选择新建的库。knowledge_base_search设置top_k=3、hybrid、rerank=false。
8. 检查提示词、模型、Skill、MCP和知识库，按DEMO.md调试，通过后发布。

## 移植注意
导出中agent_id、tenant_id、model_ids、知识库index_names都是源实例标识，不能直接用于新实例。容器MCP地址只在原网络可达。知识库内容需单独导入，导入向导不等于完成知识迁移。密钥不得提交版本库。

## 状态
本机完整流程已跑通，用户最新截图显示已发布。未验证干净环境重建、CI或PDF/Word上传解析；重型文档解析服务仍暂停。本库通过平台原生手工分块服务完成向量入库。

## 提交
目录：通用智能体/开发与编程/编程助手。
PR标题：[new agent] 编程助手。
目前准备提交材料，未推送、未创建PR。许可证沿用仓库MIT。
