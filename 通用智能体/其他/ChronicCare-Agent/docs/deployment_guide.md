# ChronicCare-Agent 部署与复现指南

## 1. 部署范围

本交付包采用单容器 `chroniccare-runtime`。容器统一提供：

- Tool Server：`18088`
- MCP Adapter：`18188`
- 可选Streamlit辅助Dashboard：`18501`

Nexent 和 DataMate 是外部平台/运行时，不包含在本项目镜像中。Nexent 通过 MCP Endpoint 调用 ChronicCare；需要完整 DataMate/NPU 重跑时，宿主机还需具备对应运行时和硬件环境。

## 2. 环境要求

### 2.1 CPU基础环境

- Linux
- Docker Engine和Docker Compose插件
- Python 3.12（仅在宿主机直接运行脚本时需要）
- 可用端口18088、18188、18501

Python依赖统一记录在根目录 `requirements.txt`。

### 2.2 NPU可选环境

- Ascend 910B3或兼容设备
- 与设备匹配的驱动和CANN
- 与CANN匹配的PyTorch、`torch_npu`
- 本地 `bge-small-zh-v1.5` 模型权重
- 可用的外部 `datamate-runtime`

通用CPU镜像不会强制安装硬件相关包，也不包含模型权重。

### 2.3 正式NPU基准环境快照

正式性能报告使用的外部NPU环境固定如下，机器可读副本见`outputs/evaluation/npu_environment_report.json`：

| 项目 | 固定值 |
| --- | --- |
| NPU | 单张Ascend 910B3 |
| Ascend驱动 | 25.5.0（ascendhal 7.35.23） |
| CANN Toolkit | 8.5.0（内部版本`V100R001C25SPC001B232`） |
| DataMate基线 | commit `6136834ce00075f0a844e26dcc7fe1cc9e0d8dd9` |
| DataMate运行镜像 | `ghcr.io/modelengine-group/datamate-runtime@sha256:0fcdd4293270d8c6f19e0ffed9584e9f61989116735c4f39a8445eab421eeb9d` |
| Python | 3.11.14 |
| PyTorch / torch_npu | 2.8.0+cpu / 2.8.0 |
| Transformers | 4.57.6 |
| BGE模型 | `bge-small-zh-v1.5`，目录内容SHA-256 `688f9664eb65edea0f73f78464c767a759a230b60ae74001af9492be6a67e94c` |
| 性能参数 | CPU 64线程；CPU batch 64；NPU batch 1024；max_length 64；每轮各预热1次 |

交付包可独立复现项目代码、数据快照、图谱、SQLite、MCP服务和正式证据。Nexent交互界面、DataMate完整运行时及NPU基准还需按上表准备外部组件；这些大型上游平台、驱动与模型权重不随本项目包重复分发。

## 3. 配置

复制环境变量模板：

```bash
cp .env.example .env
```

默认离线/模板Open SQL无需API Key。若启用LLM候选SQL，可配置：

```dotenv
OPEN_SQL_LLM_ENABLED=true
OPEN_SQL_LLM_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=<本地API Key>
```

真实`.env`、API Key、代理凭据、SSH信息和服务器私有路径均不进入仓库。

## 4. 一行启动

在 `ChronicCare-Agent` 根目录执行：

```bash
docker compose up -d --build chroniccare-runtime
```

查看状态：

```bash
docker compose ps
docker logs --tail 200 chroniccare-runtime
```

停止服务：

```bash
docker compose down
```

### 4.1 Docker安全边界

正式`docker-compose.yml`是唯一推荐入口，不挂载宿主机`/var/run/docker.sock`或`/usr/bin/docker`，因此容器不能控制宿主机Docker。常规统计、图谱、Open SQL、MCP和可视化功能均使用该配置。

完整DataMate/NPU重跑需要调用外部`datamate-runtime`时，应由具备权限的运维人员在宿主机显式执行第7节脚本。`docker-compose.server.yml`保留管理型容器编排能力并挂载Docker socket，该能力等同较高宿主机权限，只能用于隔离、受控、单租户服务器，不能暴露给不可信用户或公共网络。

## 5. 健康检查

```bash
curl --noproxy "*" -sS http://127.0.0.1:18088/health
curl --noproxy "*" -sS http://127.0.0.1:18188
curl --noproxy "*" -sS http://127.0.0.1:18088/analysis/open-sql/schema
```

浏览器入口：

- 可选辅助Dashboard：`http://127.0.0.1:18501`（功能交互与测试以Nexent前端为准）
- Tool Server：`http://127.0.0.1:18088`
- MCP Endpoint：`http://127.0.0.1:18188/mcp`

## 6. 接入Nexent

1. 启动 `chroniccare-runtime`。
2. 在Nexent中注册Streamable HTTP MCP服务器。
3. MCP地址使用 `http://宿主机可达地址:18188/mcp`。
4. 导入或参考 `integrations/nexent/chroniccare_agent_prompt.md`。
5. 后端MCP服务公开38个工具；按默认清单绑定33个工具，其中保留2个调试工具，5个历史兼容工具不默认暴露。
6. 先测试健康检查、数据规模和Open SQL Schema，再执行DataMate/NPU长任务。

示例配置见：

```text
integrations/nexent/chroniccare_mcp_config.example.json
```

当Nexent运行在容器内时，`127.0.0.1` 指向Nexent容器本身，必须改成它能够访问的宿主机地址或同一Docker网络中的服务名。

## 7. DataMate全流程

需要外部DataMate运行时时，在宿主机项目根目录执行：

```bash
python3 scripts/run_datamate_full_pipeline.py
python3 scripts/sync_datamate_outputs_to_mainline.py
python3 scripts/check_datamate_full_pipeline.py
```

完整链路会更新：

- `data/processed/`
- `data/graph/`
- `data/sqlite/`
- `outputs/runtime_generated/`中的运行时报告、图表和子图
- `configs/current_metrics.json`

Nexent前端调用 `chroniccare_datamate_pipeline_run` 时，也必须等待本轮真实执行完成。

运行时报告、图表和子图不作为随包固定证据；已复核的前端实测证据统一保存在`docs/assets/`。

## 8. NPU运行

NPU增强应先检查：

- NPU设备可见；
- `npu-smi`可用；
- CANN环境可用；
- `torch_npu`与PyTorch版本匹配；
- BGE模型目录已挂载；
- DataMate NPU算子能够加载模型。

正式对比口径为CPU 64线程、NPU batch size 1024；同一批2,048条样本独立执行五轮，每轮CPU和NPU各预热一次，正式结论采用五轮耗时的算术平均值。若报告返回 `fallback_used=true`，只能说明功能回退成功，不能宣称NPU加速。

## 9. 外部访问与SSH转发

远程服务器通过本地端口转发提供访问，例如：

```bash
ssh -N \
  -L 13000:127.0.0.1:3000 \
  -L 18088:127.0.0.1:18088 \
  -L 18188:127.0.0.1:18188 \
  -L 18501:127.0.0.1:18501 \
  用户@服务器
```

端口13000仅为Nexent前端示例，实际以Nexent部署端口为准。

## 10. 常见问题

### MCP显示不可用

检查18188端口、`/mcp`路径、Nexent到宿主机的网络连通性，以及工具是否仍绑定在当前Agent。

### 图表或图谱链接打不开

检查返回URL中的主机和端口是否对浏览器可达。容器内部地址或服务器 `localhost` 不能直接被笔记本浏览器访问时，需要修改public URL或建立SSH转发。

### Open SQL返回连接错误

模板模式不需要外部模型。启用LLM候选时检查API Key、Base URL和代理；LLM失败不能绕过SQL Guard。

### NPU任务超时

首次模型加载、编译和全量运行耗时较长。运行前需确认NPU未被其他进程占用，并为Nexent长任务配置足够的超时时间；性能结果必须来自本轮真实执行。


