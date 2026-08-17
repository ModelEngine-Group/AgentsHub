# MCP 服务

`mediflow-mcp/` 是三个 Nexent 智能体共用的服务运行包，负责把数据处理、医学知识抽取和数据分析能力以 MCP 工具形式提供给 Nexent。

包含内容：

- FastMCP 服务入口和工具注册；
- 医学实体、关系和术语处理逻辑；
- DataMate 自定义算子；
- 知识图谱、分析数据库和平台客户端适配；
- 可选的离线数据库构建代码。

运行数据库不重复放在源码目录中，而是压缩保存在上级 [`knowledge_base/`](../knowledge_base/)。首次启动前，在归档根目录执行 `python prepare_runtime_assets.py`，再按照 [`mediflow-mcp/README.md`](mediflow-mcp/README.md) 启动服务。
