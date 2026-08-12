# 配置模板

本目录提供不含密钥的配置样例：

| 文件 | 用途 |
| --- | --- |
| `.env.example` | MCP 服务、模型、Nexent、DataMate 和数据库路径的环境变量模板 |
| `config.example.yaml` | 便于阅读的结构化配置参考 |
| `nexent-mcp.example.json` | 在 Nexent 中手动登记 MCP 服务时可参考的请求内容 |

使用时复制模板，在目标环境填写实际值：

```bash
cd ../mcps/mediflow-mcp
cp ../../config/.env.example .env.runtime
cp ../../config/config.example.yaml config.yaml
```

至少需要配置模型服务、MCP 地址、Nexent 地址，以及使用数据处理智能体时的 DataMate 地址。数据库路径保持为模板中的相对路径即可；运行前先在归档根目录执行 `python prepare_runtime_assets.py`。

如果使用 Nexent 原生知识库，知识库源文档和绑定关系在 Nexent 中配置，不写入公开配置模板。不要把密码、Token、Cookie 或真实内网地址保存到仓库。

MCP 服务名默认是 `medical-ai`。在 Nexent 中注册并扫描该服务后，再导入三个智能体配置。
