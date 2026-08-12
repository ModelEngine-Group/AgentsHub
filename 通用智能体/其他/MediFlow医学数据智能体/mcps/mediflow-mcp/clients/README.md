# 外部平台适配

`clients/` 封装 Nexent 和 DataMate 的 HTTP 访问逻辑，让业务模块不直接拼接平台接口。

| 文件 | 作用 |
| --- | --- |
| `nexent_client.py` | Nexent 登录、智能体管理、MCP 工具发现和流式对话适配 |
| `__init__.py` | Python 包入口 |

平台地址、账号和密钥从运行环境读取，不写入代码。这里负责通信和响应解析；数据处理、医学抽取和数据分析编排分别由 `mcp_server/` 下的服务模块完成。
