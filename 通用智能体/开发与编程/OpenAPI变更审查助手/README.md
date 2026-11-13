# OpenAPI 变更审查助手

给发布前的 API 审查提供可追溯的变更清单。比较旧版和新版 OpenAPI 3.0 或 3.1 JSON。报告包含契约位置、变更前后值和影响等级。工具检查接口移除、请求参数和请求体必填化、请求枚举收紧、响应字段移除和响应媒体类型变化，也提示鉴权、序列化、响应枚举扩展等需要人工确认的变化。

## 试用

运行 `python mcps/server.py examples/before.json examples/after.json`。示例应标出新增必填参数、请求枚举收紧、响应字段移除和响应枚举扩展。用同一份规格作为两份输入，可确认示例没有比较噪声。

在 Nexent 中使用时，安装 `fastmcp==2.14.7`，运行 `python mcps/server.py --host 0.0.0.0 --port 8080`。把服务部署在 Nexent 可达的私有网络，按实际主机名添加 Streamable HTTP 地址 `http://主机名:8080/mcp` 并扫描工具。注册名称使用 `openapi-contract-review`。导入 `agent-export.zip`，选择支持工具调用的模型，确认 `review_openapi` 已绑定，然后发布。导入包依据 Nexent 2.8.2 的格式组装。在聊天里分别粘贴旧、新 JSON，明确哪个版本已被客户端使用。

工具解析文档内部 JSON Pointer，外部引用和超出比较范围的复杂组合模式提示人工复核。比较针对契约文本，实际服务行为和 SDK 反序列化行为仍需契约测试。参考 [OpenAPI 3.0.4](https://spec.openapis.org/oas/v3.0.4.html) 和 [Google API 兼容性指南](https://google.aip.dev/180)。
