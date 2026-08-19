# MCP / OpenAPI 服务

将下列 OpenAPI 契约导入 Nexent（MCP 配置 / OpenAPI 服务），并确保对应任务 API 已在宿主机监听：

| 文件 | 默认服务地址 |
| --- | --- |
| `nexent_dkm_task1.openapi.json` | `http://host.docker.internal:8000` |
| `nexent_dkm_task2.openapi.json` | `http://host.docker.internal:8002` |
| `nexent_dkm_task3.openapi.json` | `http://host.docker.internal:8003` |

Linux 原生 Docker 请将 `host.docker.internal` 换成宿主机可达 IP（如 `172.17.0.1`），详见 `../docs/online_integration.md`。
