# 初步准备与部署记录

## 目标

本阶段先完成比赛开发前的基础准备：

- 拉取官方 Nexent 仓库，用于 Docker 部署、Agent 框架参考和接口集成。
- 拉取官方 DataMate 仓库，用于数据处理能力参考、部署和接口集成。
- 保持比赛主开发工程独立在 `nexent-dkm-agent/` 目录，避免直接修改官方仓库源码。
- 记录当前可复现的版本、部署入口、验证结果和剩余阻塞。

## 当前目录

```text
nexent-dkm-agent/              当前 Git 工作目录
├─ nexent/                     官方 Nexent 参考仓库，本地部署与接口参考
├─ DataMate/                   官方 DataMate 参考仓库，本地部署与接口参考
└─ nexent-dkm-agent/           比赛主开发工程，代码、配置、测试和文档放这里
```

`nexent/` 和 `DataMate/` 只作为本地参考与部署目录，最终提交不包含官方仓库源码。

## Windows + WSL 开发环境（重要）

本仓库在 **Windows 10/11** 上的日常开发、演示程序、pytest、答辩证据采集，均在 **Windows 原生 Python 3.12** 中执行；但 **Nexent / DataMate / Neo4j 的 Docker 部署与运维命令必须在 WSL Ubuntu 内执行**，不能假设 Windows PowerShell 可直接调用 `docker`。

| 场景 | 推荐环境 | 说明 |
| --- | --- | --- |
| 任务一/二/三演示、基准测试、pytest、证据打包 | Windows PowerShell + 项目根目录 Python | 见各任务文档与根目录 `README.md` |
| Nexent Docker 部署 | **WSL Ubuntu** + `nexent/docker/deploy.local.sh` | Windows 侧通常无 `docker` 命令 |
| DataMate Docker 启动/恢复 | **WSL**（脚本：`scripts/start_datamate_wsl.ps1` 会保活 WSL 并执行就绪检查） | 宿主机端口 `18000` / `30000` |
| Neo4j 冒烟测试 | **Windows Docker Desktop** 或 WSL Docker（二选一，以本机已有容器为准） | Bolt `7687`，Browser `7474` |
| Ascend NPU 复验 | **Linux 服务器**（openEuler），**不在 WSL/Windows 上跑** | 见 `npu_optimization.md` |

WSL 路径映射示例：Windows 仓库 `C:\Users\...\nexent-dkm-agent` 在 WSL 中为 `/mnt/c/Users/.../nexent-dkm-agent`。官方 Nexent 仓库若 checkout 在 Windows 挂载盘，`deploy.sh` 可能因 CRLF 失败，请使用同目录下的 **`deploy.local.sh`（LF 版本）**。

DataMate 在 WSL 中恢复：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_datamate_wsl.ps1 -Mode start -WaitSeconds 90
```

等价 Linux 路径（在 DataMate 官方 checkout 旁）：

```bash
bash scripts/start_datamate_linux.sh --mode start --wait-seconds 90
```

## Linux 原生部署（重要）

在**原生 Linux**（非 WSL 环境）上，Python 主工程、Docker、Nexent/DataMate/Neo4j
可在同一台机器运行，通常比 Windows+WSL 分工更简单。在线集成时 Nexent 容器访问
宿主机任务 API 需额外注意网络，详见 [在线集成 - Linux 原生 Docker 注意点](online_integration.md#linux-原生-docker-注意点)。

| 场景 | 推荐环境 | 说明 |
| --- | --- | --- |
| 任务一/二/三演示、pytest、证据打包 | Linux Python 3.12 + venv | 与 Windows 命令相同 |
| Nexent Docker 部署 | `../nexent/docker/deploy.sh` | 原生 `docker`/`docker compose` |
| DataMate Docker 启动/恢复 | `scripts/start_datamate_linux.sh` | 默认 `http://localhost:18000` |
| Neo4j 冒烟测试 | `docker compose -f docker-compose.neo4j.yml up -d` | Bolt `7687` |
| Nexent 在线集成（容器→宿主机 API） | `dkm_online_integration.py --docker-host auto` | 或配置 `host.docker.internal:host-gateway` |
| Ascend NPU 复验 | openEuler / aarch64 + CANN | 见 `server_environment.md` |

## 官方仓库版本

| 项目 | 本地路径 | 远程地址 | 当前 commit |
| --- | --- | --- | --- |
| Nexent | `../nexent` | `https://github.com/ModelEngine-Group/nexent.git` | `1614aab67bfc85ca390f4c23d7df4474b2664802` |
| DataMate | `../DataMate` | `https://gitlink.org.cn/modelengine-group/DataMate.git` | `8f47ceb3ccc4a786b1dade5d78c3c8f48620d3cb` |

## 官方部署入口

### Nexent

官方 README 推荐 Docker 部署（在官方 Nexent checkout `../nexent` 内执行）：

```bash
cd ../nexent/docker
cp .env.example .env
bash deploy.sh
```

官方要求：

- Docker 24+
- Docker Compose v2+
- 最低 4 CPU / 8 GiB 内存 / 40 GiB 磁盘
- 推荐 8 CPU / 16 GiB 内存 / 100 GiB 磁盘

服务启动后，默认访问入口为：

```text
http://localhost:3000
```

### DataMate

官方 README 推荐 Docker 或 Make 部署。

快速 Docker 部署：

```bash
wget -qO docker-compose.yml https://raw.githubusercontent.com/ModelEngine-Group/DataMate/refs/heads/main/deployment/docker/datamate/docker-compose.yml \
  && REGISTRY=ghcr.io/modelengine-group/ docker compose up -d
```

本地源码部署：

```bash
cd DataMate
make install
```

如果没有 `make`，可在官方 DataMate checkout 内直接使用 Docker Compose：

```bash
cd DataMate
REGISTRY=ghcr.io/modelengine-group/ docker compose -f deployment/docker/datamate/docker-compose.yml --profile milvus up -d
```

服务启动后，默认访问入口为：

```text
http://localhost:30000    # Web UI（浏览器）
http://localhost:18000    # REST API（集成脚本、任务一 DataMate 客户端请用此端口）
```

## 当前验证结果

已完成：

- Nexent 官方仓库已拉取到本地。
- DataMate 官方仓库已拉取到本地。
- 已记录两个官方仓库的远程地址和 commit。
- Nexent 已通过 WSL Docker 部署。
- DataMate 核心服务已通过 WSL Docker Compose 部署。
- 比赛工程基础演示可运行。
- 比赛工程基础测试可运行。

### 部署环境

Windows PowerShell 当前没有 `docker` 命令，但 WSL Ubuntu 内 Docker 可用：

```text
Docker version 29.4.1
Docker Compose version v5.1.3
```

实际部署命令均通过 WSL 执行。

### Nexent 部署结果

部署模式：

- version: `speed`
- mode: `production`
- image source: general
- root dir: `$HOME/nexent-data`
- built-in skills: disabled
- terminal tool container: disabled

关键命令：

```bash
cd /mnt/c/path/to/nexent-dkm-agent/nexent/docker
cp -n .env.example .env
bash deploy.local.sh --version speed --mode production --enable-terminal N --is-mainland N --enable-skills N --root-dir "$HOME/nexent-data"
```

说明：由于官方仓库位于 Windows 挂载路径下，`deploy.sh` 存在 CRLF 换行导致 WSL bash 解析失败，本地生成了 `deploy.local.sh` 作为 LF 版本部署脚本。`nexent/` 是被忽略的官方参考仓库，该本地脚本不提交到比赛仓库。

当前容器：

- `nexent-web`
- `nexent-config`
- `nexent-runtime`
- `nexent-mcp`
- `nexent-northbound`
- `nexent-data-process`
- `nexent-postgresql`
- `nexent-redis`
- `nexent-minio`
- `nexent-elasticsearch`

访问验证：

- WSL 内 `http://127.0.0.1:3000` 返回 `307 /zh`，随后可返回 Nexent 页面。
- Windows 侧当前 WSL IP 入口 `http://172.21.112.21:3000` 返回 `200`（示例 IP，以 `wsl hostname -I` 为准）。WSL 重启后 IP 可能变化，可用 `wsl hostname -I` 重新查看。
- Windows 侧 `http://localhost:3000` 暂未转发成功，后续如需固定 localhost 入口，可再配置 Windows portproxy。

### DataMate 部署结果

部署模式：

- core services only
- registry: `ghcr.io/modelengine-group/`
- Milvus、Label Studio、Mineru、DeerFlow 等可选 Docker 服务配置（profile）暂未启用。

由于本机已有 `honcho-database-1` 占用宿主机 `127.0.0.1:5432`，DataMate 使用本地 compose 文件将数据库宿主端口改为：

```text
15432:5432
```

容器内部仍使用 `datamate-database:5432`，不影响 DataMate 服务间通信。

关键命令（Linux 服务器优先；`docker-compose.local.yml` 是本机部署时在官方 DataMate checkout 下生成的本地覆盖文件，不属于比赛仓库提交内容）：

```bash
cd /data/nexent-dkm-agent/nexent-dkm-agent
bash scripts/start_datamate_linux.sh --mode up --wait-seconds 90
```

如果官方 DataMate checkout 不在比赛仓库同级目录，可显式传入 compose 目录：

```bash
bash scripts/start_datamate_linux.sh \
  --compose-dir /path/to/DataMate/deployment/docker/datamate \
  --mode up \
  --wait-seconds 90
```

如果容器已经存在但处于 stopped 状态，可只重新启动已有容器：

```bash
bash scripts/start_datamate_linux.sh --mode start --wait-seconds 90
```

手工等价命令如下：

```bash
cd /path/to/DataMate/deployment/docker/datamate
REGISTRY=ghcr.io/modelengine-group/ docker compose -f docker-compose.local.yml up -d
```

Windows + WSL 仅作为本地演示辅助路径。WSL 环境建议用比赛仓库内 PowerShell 脚本启动，脚本会先保活一个临时 WSL 进程，再启动 DataMate，并要求算子、清洗模板和清洗任务三项数据库相关核心 API 全部成功后才判定就绪：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_datamate_wsl.ps1 -Mode up -WaitSeconds 90
```

如果 WSL 中容器已经存在但因 WSL 重启 / SIGTERM 正常退出，可只重新启动已有容器：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_datamate_wsl.ps1 -Mode start -WaitSeconds 90
```

默认 WSL 发行版为 `Ubuntu`；如名称不同，显式传入 `-Distro <名称>`。Linux
启动脚本固定使用 LF 行尾，仓库通过 `.gitattributes` 防止 Windows checkout
将 `*.sh` 转回 CRLF。

### DataMate SIGTERM 排查结论

2026-06-05 在 Windows + WSL Docker 环境中复现过 DataMate 容器约 1-3 分钟后优雅退出：

- `datamate-runtime` 日志显示收到 `signal 15`。
- `datamate-backend-python` 日志显示正常 shutdown。
- `docker inspect` 显示 `datamate-database`、`datamate-runtime`、`datamate-frontend`、`datamate-backend-python` 均为 `exit=0`，且 restart policy 为 `on-failure`。
- 同一时间 Nexent 容器也出现重启，但 Nexent 使用 `restart=always`，因此能自动恢复。

结论：本地 SIGTERM 来源不是 DataMate 算子或应用崩溃，而是 WSL / Docker daemon 生命周期导致容器收到优雅停止信号。DataMate 核心服务正常退出时不会触发 `on-failure` 自恢复，所以看起来像只有 DataMate 不稳定。临时保活命令 `wsl -e sleep 1800` 存活期间，DataMate health 连续返回 `200`，算子目录可读取 `totalElements=210`。

2026-06-15 使用 `scripts/start_datamate_wsl.ps1 -Mode start` 再次恢复并验证了已有
PostgreSQL 持久卷。脚本已避免把 Docker Compose 写入 stderr 的正常启动进度误判为
PowerShell 异常，并通过 `scripts/datamate_readiness.py` 同时验证健康检查与 3/3
核心业务接口；只有 health 成功但业务接口失败时不会误报 ready。

相关证据已保存到 ignored 路径：

```text
outputs/competition_evidence/datamate-sigterm-debug/
```

当前容器：

- `datamate-frontend`
- `datamate-backend`
- `datamate-backend-python`
- `datamate-gateway`
- `datamate-database`
- `datamate-runtime`

访问验证：

- Windows 侧 `http://localhost:30000` 返回 `200`。
- `datamate-backend-python` 日志显示 DataMate Python Backend 已启动，MCP HTTP server listening at `/api/mcp`。
- Windows 侧 `http://localhost:18000/api/cleaning/templates?page=0&size=3` 返回 `200`；2026-06-15 只读探测共回查到 7 个模板。
- Windows 侧 `http://localhost:18000/api/cleaning/tasks?page=0&size=3` 返回 `200`；2026-06-15 共回查到 3 条历史任务，其中 2 条为 `COMPLETED`、1 条为 `FAILED`。
- Windows 侧 `http://localhost:18000/api/v1/jobs?page=0&size=3` 返回 `404`；当前部署应优先使用 `/api/cleaning/templates` 和 `/api/cleaning/tasks`，不是 execution-engine OpenAPI 中的 `/api/v1/jobs`。

### 镜像拉取处理

部署过程中遇到过镜像源 EOF / 不可用（unavailable）。处理方式：

- Nexent 改用 general image source，并先逐个拉取基础镜像和 Nexent 核心镜像。
- DataMate 先逐个补拉 `datamate-backend`、`datamate-frontend`、`datamate-gateway`、`datamate-database`、`datamate-backend-python`、`datamate-runtime`，再执行 `compose up`。

已验证命令：

```bash
cd nexent-dkm-agent
python demos/task1_demo.py
python demos/task2_demo.py
python demos/task2_evaluate.py
python benchmarks/task2_kg_benchmark.py --iterations 5
python demos/task3_demo.py
python demos/task3_evaluate.py --question "哪些疾病关联最多症状？"
python demos/task3_nexent_spec.py --model-name main_model
python demos/task3_smoke.py --iterations 3
python benchmarks/task3_analysis_benchmark.py --iterations 5
python -m pytest
python -m compileall src demos tests benchmarks
```

当前结果：

- 任务一演示可完成自然语言任务理解、CSV 画像、自动规划、状态追踪、清洗导出、清洗后质量校验、运行质量报告、Nexent 适配、DataMate 算子目录映射，并生成 DataMate 清洗模板/任务提交数据；文本输入支持 HTML 清洗、Unicode 规范化、PII 脱敏和医疗实体抽取；支持 LLM 自主规划（`--llm --llm-config`）、本地微调模型规划（`--local-model`）和 REST API 服务（`--serve`），均由 `task1_demo.py` 统一承载。
- 任务二演示可完成医疗实体抽取（Disease/Symptom/Drug/Examination/Treatment）、关系抽取（has_symptom/treated_by/diagnosed_by/recommended_treatment/complication_of）、三元组校验与去重、图谱构建与导出、单跳/多跳问答、证据链推理；支持 LLM 增强抽取（`--llm-config`）、本地小模型 NER 抽取（`--local-model`）、混合规划、REST API v2.0（含多跳查询和证据链 QA 端点）、QLoRA 微调训练数据生成与微调脚本、Neo4j 图数据库持久化与 Cypher 查询（`--neo4j-uri`），均由 `task2_demo.py` 统一承载。
- 任务三演示可复用任务二图谱并在缺失时自动补全上游，完成图谱统计、疾病关联分析、记录趋势分析、中心性/路径/社区分析、LLM 增强 NL2SQL 与模板回退、静态仪表盘、ECharts 交互仪表盘、REST API、Nexent 适配、评测/冒烟测试和 CPU/NPU 基准测试；本地模型路径通过 `task3_demo.py --local-model` 进入主链路。
- `python -m pytest -q` 全量通过：**437/437**（2026-07-03 答辩证据包与当前主分支，Windows/Python 3.12）；Ascend 910B3 无卡预配置 **737/737**、插卡后 **770/770**（NPU 专项 **43/43**，2026-06-24 NPU 快照，`final-version`；2026-06-16 历史为 380/380；2026-06-14 历史为 289/289）。用例数随功能演进增长，以各环境实际运行输出为准。
- `compileall` 通过。
- 本地小模型微调（任务一/二/三）链路已在 CUDA 12.8 / RTX 5070 上实测通过，详见 [local_model_finetune.md](local_model_finetune.md)。
- LLM 配置由 `src/common/llm_config.py` 统一读取，REST API 本地路径由 `src/common/path_security.py` 校验；API 服务默认监听 `127.0.0.1`。

## 环境限制与复现边界

- Windows PowerShell PATH 中仍没有 `docker` 命令；当前依赖 WSL 内 Docker 执行部署和运维命令。
- Nexent 的 Windows `localhost:3000` 入口暂未转发成功，可先使用当前 WSL IP（示例 `http://172.21.112.21:3000`，以 `wsl hostname -I` 为准）。WSL 重启后 IP 可能变化。
- DataMate 的可选 Milvus / Label Studio / Mineru / DeerFlow profiles 尚未启用；当前只部署核心服务，足够作为任务一 DataMate 能力参考和接口摸底起点。

## 复现与安全说明

- 复现任务一、任务二、任务三时，建议分别运行对应演示、评测/冒烟测试、基准测试和 `python -m pytest -q`。
- LLM 配置、模型权重、运行输出和本地缓存仅作为本地运行产物保留，不提交真实 key 或大文件。
- 如需对外开放 API，请显式传 `--host 0.0.0.0` 并确认网络可信。
