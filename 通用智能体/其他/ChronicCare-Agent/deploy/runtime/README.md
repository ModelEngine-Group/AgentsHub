# 统一运行容器

本目录提供ChronicCare-Agent正式交付版的单容器运行配置。

## 容器内进程

- FastAPI Tool Server：端口`18088`
- MCP Adapter：端口`18188`
- 可选Streamlit辅助Dashboard：端口`18501`；正式交互与测试以Nexent前端为准
- 运行日志：`logs/runtime/`

## 启动命令

```bash
docker compose up -d --build chroniccare-runtime
```

## 运行方式

`chroniccare-runtime`是本地和正式交付环境的统一运行入口。根目录`docker-compose.yml`是唯一推荐配置；`docker-compose.server.yml`仅用于受控服务器环境。
