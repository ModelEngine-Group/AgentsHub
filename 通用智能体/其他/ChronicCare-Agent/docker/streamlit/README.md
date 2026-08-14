# Streamlit拆分容器

该镜像用于独立运行ChronicCare-Agent Dashboard，属于开发调试和拆分部署入口。正式交付采用`chroniccare-runtime`单容器。

- 基础镜像：`python:3.12.13-slim`
- 容器端口：`8501`
- 默认依赖的 Tool Server 地址：
  - `CHRONICCARE_TOOL_SERVER_URL=http://chroniccare-tool-server:18088`

数据卷挂载目录：

- `/app/data`
- `/app/outputs`
- `/app/configs`
