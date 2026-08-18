# Docker 部署配置

本项目依赖 **Nexent** 和 **DataMate** 两个基础平台。以下配置文件提供了 Docker Compose 环境部署方案。

---

## 目录说明

| 文件/目录 | 说明 |
|:---|:---|
| `nexent-docker-compose.yml` | Nexent 平台编排（11 个服务） |
| `nexent-deploy.sh` | Nexent 一键部署脚本 |
| `nexent-env.example` | Nexent 环境变量模板（**需修改为实际值后保存为 `.env`**） |
| `nexent-dockerfiles/` | Nexent 各服务的 Dockerfile |
| `nexent-sql/` | Nexent 数据库初始化 SQL |
| `nexent-scripts/` | Nexent 辅助脚本 |
| `datamate-docker-compose.yml` | DataMate 平台编排（18 个服务） |

---

## 使用步骤

### 1. 环境准备

- 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- 确保 WSL 2 已启用（Windows）

### 2. 部署 Nexent

```bash
# 进入 docker 目录
cd docker

# 复制环境变量模板并填写
cp nexent-env.example .env
# 编辑 .env 填入你的 API Key 和密码

# 启动 Nexent
docker compose -f nexent-docker-compose.yml up -d
```

### 3. 部署 DataMate

```bash
docker compose -f datamate-docker-compose.yml up -d
```

### 4. 启动 MCP 服务器

```bash
# 在项目根目录
cd mcp_server
pip install -r ../requirements.txt
python server.py
```

### 5. 访问 Nexent 控制台

- Nexent Web: http://localhost:3000
- MCP 绑定地址: `http://host.docker.internal:8089`

---

## 注意事项

- `.env` 中包含 LLM API Key、数据库密码等敏感信息，**不要提交到 Git**
- Nexent 需要至少 8GB 可用内存
- 首次启动时需拉取 Docker 镜像（约 2-3GB 下载量）
- MCP 服务器需在 Docker 环境外运行（宿主机）
