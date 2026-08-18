# Tool Server拆分容器

该镜像用于独立运行ChronicCare-Agent的FastAPI Tool Server，属于开发调试和拆分部署入口。正式交付采用`chroniccare-runtime`单容器。

- 基础镜像：`python:3.12.13-slim`
- 容器端口：`18088`
- 默认启动命令：`uvicorn tool_server.app:app --host 0.0.0.0 --port 18088`

数据卷挂载目录：

- `/app/data`
- `/app/outputs`
- `/app/configs`
- `/app/models`
