# Nexent / DataMate 真实在线集成

本文说明如何在不修改 Nexent、DataMate 官方源码的前提下，把本项目三套 REST API
注册为 Nexent OpenAPI 工具，并对 DataMate 清洗资源执行可回查的在线验证。

> **Windows 开发机**：Nexent 与 DataMate 容器在 **WSL Ubuntu + Docker** 中运行；任务一/二/三 API 在 **Windows PowerShell** 中启动（`python demos/task*_demo.py --serve`）。Neo4j 可用 Windows Docker Desktop 或 WSL Docker。详见 [初步准备与部署记录](preparation.md#windows--wsl-开发环境重要)。

> **Linux 原生 Docker**：任务 API 与集成脚本可在同一台 Linux 主机直接运行；但 Nexent 容器默认**不能**解析 `host.docker.internal`，见下文 [Linux 原生 Docker 注意点](#linux-原生-docker-注意点)。

## 集成边界

- `probe`（探测）：只读探测 Nexent、DataMate 和三套任务 API，不修改远端状态。
- `prepare`（准备）：生成三套 OpenAPI 导入摘要，不向 Nexent 写入。
- `submit`（提交）：导入 OpenAPI 服务并刷新 Nexent 工具目录，必须显式传入 `--allow-write`。
- `submit --create-agent`：进一步通过 Nexent 官方 `/api/agent/update` 创建 DKM Agent，并通过 `/api/agent/by-name/{name}` 按名称回查。
- DataMate 的 `datamate_mode=submit` 会创建模板和清洗任务；任务创建后会立即执行。本项目会提取服务端资源 ID，再调用详情接口确认资源真实存在。
- 任务一 REST API 默认仅接受服务启动时配置的 DataMate 地址，并拒绝所有正式提交请求。只有可信环境下以 `--allow-api-datamate-write` 启动任务一 API 后，Nexent 工具调用才可能进入 DataMate 写入路径。

NPU 基准测试与在线集成相互独立。本文流程不修改已有 NPU 数据或结论。

## 验证层级（2026-07-02 在线结论）

**说明**：本节验证外部服务连通与写入回查，与 NL2SQL/抽取 benchmark 无关。

集成验证按依赖层级自下而上：**L1 任务 API** → **L2 DataMate** → **L3 Nexent** → **L4 Neo4j**。2026-07-02 在线探测 `stack_status=ready`；`collect_competition_evidence` 离线采集默认 `stack_status=offline`（外部服务 skipped）。

| 层级 | 组件 | 验证什么 | 结论 | 证据（答辩包内） |
| --- | --- | --- | --- | --- |
| L1 | Task1 API `:8000/health` | HTTP 200 + `status=healthy` | available | `probe-20260702-final.json` → `task_api_health.task1` |
| L1 | Task2 API `:8002/health` | HTTP 200 + 服务标识 | available | 同上 → `task_api_health.task2` |
| L1 | Task3 API `:8003/health` | HTTP 200 + 服务标识 | available | 同上 → `task_api_health.task3` |
| L2 | DataMate `:18000` | health + 核心 API **3/3**；submit 回查 | 已验证 | `probe-20260702-final.json`；`datamate-submit-20260702-final.json` |
| L3 | Nexent `:3000` | JWT + OpenAPI 三服务 + Agent 回查 | 已验证（48 tools） | `probe-20260702-final.json` 等 |
| L4 | Neo4j Bolt | 连接/写入/读回/Cypher/KG QA | `passed=true`；26/29 | `task2-neo4j-live-smoke-20260702-final.json` |

完整表格与离线采集说明见 [技术答辩材料 §6.3](competition_defense_document.md) 与根目录 [README 在线集成](../README.md#在线集成2026-07-02)。

## 2026-07-03 复验补充（本地复跑）

答辩包 `evidence/online_integration/` 另含下列 JSON，供复现对照：

- `probe-20260703-fullstack.json` — JWT 刷新后全栈 probe，`stack_status=ready`，DataMate 3/3，Nexent OpenAPI 三服务，三套任务 API 均 available。
- `datamate-submit-20260703-rerun.json` — `catalog_summary` 在线 submit 复验（template/task 均为 `verified`）。
- `../benchmarks/task1_datamate_submit.json` — 任务一 pipeline submit benchmark（修复 dest 名冲突后 `passed=true`）。

复跑前可先执行 `python scripts/refresh_nexent_token.py` 刷新 JWT（见 §4 鉴权）。

## 1. 启动三套任务 API

Nexent 运行在容器中时，需要访问宿主机 API。仅在可信本机网络中使用
`0.0.0.0`：

```bash
python demos/task1_demo.py --serve --host 0.0.0.0 --port 8000
python demos/task2_demo.py --serve --host 0.0.0.0 --port 8002
python demos/task3_demo.py --serve --host 0.0.0.0 --port 8003
```

上面的默认启动命令保持 DataMate API 写入关闭。需要验证写入时，应先人工确认源数据集，再显式启动：

```bash
python demos/task1_demo.py --serve --host 0.0.0.0 --port 8000 \
  --datamate-url http://localhost:18000 \
  --allow-api-datamate-write
```

请求中的 `datamate_url` 必须与服务启动时的 `--datamate-url` 一致；`none` 始终允许用于完全禁用 DataMate。

宿主机健康检查地址为 `http://localhost:8000/health`、`:8002/health` 和
`:8003/health`。默认注册给 Nexent 容器的地址使用
`host.docker.internal:8000/8002/8003`（适用于 Docker Desktop / WSL）。

如果 Nexent 运行在**原生 Linux Docker** 中，见下一节；也可通过
`--task*-server-url` 显式改为 Nexent 容器可达的宿主机 IP。

## Linux 原生 Docker 注意点

Nexent 容器需要回调宿主机上的三套任务 API（8000/8002/8003）。在 Docker
Desktop（Windows/macOS/WSL）中，`host.docker.internal` 通常可直接使用；在**原生
Linux Docker** 中默认不存在该主机名，OpenAPI 导入虽可能显示 `verified`，但
Nexent 实际调用工具时会失败。

推荐三种做法（任选其一）：

| 方案 | 做法 | 适用场景 |
| --- | --- | --- |
| A. 启用 `host.docker.internal` | 启动 Nexent 栈时增加 `extra_hosts: ["host.docker.internal:host-gateway"]`，或 `docker run --add-host=host.docker.internal:host-gateway` | 希望与 Windows 证据 URL 保持一致 |
| B. 使用 docker0 网桥 IP | 集成脚本加 `--docker-host auto`，自动把 OpenAPI `server_url` 写成 `http://<docker0-ip>:8000` 等（常见为 `172.17.0.1`） | 原生 Linux 快速复现 |
| C. 显式宿主机 IP | 传入 `--task1-server-url http://172.17.0.1:8000` 等（以 `ip -4 addr show docker0` 为准） | 自定义网络或 bridge 非默认 |

Linux 完整在线集成示例：

```bash
# 1. 启动任务 API（宿主机）
python demos/task1_demo.py --serve --host 0.0.0.0 --port 8000 &
python demos/task2_demo.py --serve --host 0.0.0.0 --port 8002 &
python demos/task3_demo.py --serve --host 0.0.0.0 --port 8003 &

# 2. 启动 Nexent / DataMate（同级官方仓库，见 preparation.md）
cd ../nexent/docker && bash deploy.sh
cd /path/to/nexent-dkm-agent/nexent-dkm-agent
bash scripts/start_datamate_linux.sh --mode start --wait-seconds 90

# 3. 探测 + 导入（--docker-host auto 为 Linux 推荐）
python demos/dkm_online_integration.py \
  --mode probe \
  --nexent-url http://localhost:3000 \
  --datamate-url http://localhost:18000 \
  --docker-host auto \
  --token-file .local/nexent.token \
  --output outputs/competition_evidence/online-integration/probe-linux.json

python demos/dkm_online_integration.py \
  --mode submit \
  --allow-write \
  --force-update \
  --docker-host auto \
  --token-file .local/nexent.token \
  --output outputs/competition_evidence/online-integration/openapi-submit-linux.json
```

`probe` / `prepare` / `submit` 的 JSON 报告会附带
`docker_host_reachability` 字段，记录解析后的宿主机别名、三套 `server_url` 与
Linux 部署说明。DataMate 启动脚本见 `scripts/start_datamate_linux.sh`。

## 2. 只读探测

```bash
python demos/dkm_online_integration.py \
  --mode probe \
  --nexent-url http://localhost:3000 \
  --datamate-url http://localhost:18000
```

探测内容包括 Nexent Web 指纹、Nexent OpenAPI 服务列表、DataMate health /
算子目录 / 清洗模板与任务样本，以及三套任务 API 的 `/health`。

需要明确跳过 Nexent、只验证 DataMate 与任务 API 时，可以使用：

```bash
python demos/dkm_online_integration.py \
  --mode probe \
  --nexent-url none \
  --datamate-url http://localhost:18000
```

`nexent-url=none` 只允许用于 `probe`。`prepare` 和 `submit` 仍要求真实 Nexent
地址，避免离线参数误入写入流程。

## 3. 准备导入

```bash
python demos/dkm_online_integration.py \
  --mode prepare \
  --output outputs/competition_evidence/online-integration/prepare.json
```

该命令只输出服务名、容器可达 URL、路径数量和操作数量，不写入 Nexent。

## 4. 鉴权

普通 Nexent 部署需要 Bearer 令牌。为避免令牌出现在进程列表或 Git 历史中，将令牌写入 ignored 路径 `.local/nexent.token`。文件内容可以是裸令牌，也可以是完整的 `Bearer ...`。不要提交该文件。Nexent 极速模式若不要求鉴权，可以省略 `--token-file`。

本仓当前 full 鉴权复验在项目外层保存令牌：`../.local/nexent.token`。从 `nexent-dkm-agent/` 项目目录直接复跑时，可优先使用该路径；如果需要整理到项目内，则先复制到 `.local/nexent.token`，该目录已由外层 `.gitignore` 忽略。

```powershell
Test-Path ..\.local\nexent.token
python scripts/refresh_nexent_token.py
python demos/dkm_online_integration.py `
  --mode probe `
  --nexent-url http://localhost:3000 `
  --datamate-url none `
  --token-file ..\.local\nexent.token `
  --output outputs/competition_evidence/online-integration/probe-full-auth-token.json
```

## 5. 导入 OpenAPI 服务

确认三套任务 API 健康后执行：

```bash
python demos/dkm_online_integration.py \
  --mode submit \
  --allow-write \
  --force-update \
  --token-file ../.local/nexent.token \
  --output outputs/competition_evidence/online-integration/openapi-submit.json
```

成功条件：

1. 三个 OpenAPI 导入结果均为已验证（`verified`）；
2. Nexent 工具目录刷新成功（`refreshed`）；
3. 刷新后的 `/api/tool/list` 可正常读取。

脚本会在写入前读取已有 OpenAPI 服务。同名服务的 URL、路径和 `components.schemas` 契约一致时，结果为已验证（`verified`）且标记为已存在（`preexisting=true`），不会重复写入；若配置不一致，默认返回更新被阻断（`update_blocked`）。只有人工核对后显式增加 `--force-update`，才会覆盖同名服务配置。

默认情况下，只要任一任务 API 健康检查失败，`submit` 就会停止。仅用于诊断网络
差异时才使用 `--allow-unhealthy-task-apis`，该结果不能作为完整在线证据。

## 6. 创建 DKM Agent

先在 Nexent 中确认可用模型 ID 或模型名，然后执行：

```bash
python demos/dkm_online_integration.py \
  --mode submit \
  --allow-write \
  --force-update \
  --create-agent \
  --model-id 1 \
  --model-name main_model \
  --token-file ../.local/nexent.token \
  --output outputs/competition_evidence/online-integration/agent-submit.json
```

程序会从刷新后的工具目录中，仅选择由三套 DKM OpenAPI `operationId` 生成的 `mcp / outer-apis` 工具；匹配时同时校验服务命名空间，避免其他服务的同名操作被误选。程序创建 `dkm_end_to_end_agent` 后按名称回查 Agent ID。脚本不会自动删除或覆盖同名 Agent；发现同名记录时会再次调用按名称查询接口，ID 一致则按已验证且已存在（`verified + preexisting=true`）处理，因此同一命令可安全重复执行。

当前 Nexent 按名称查询响应只返回 ID 和版本号，因此该状态的 `verification_scope` 为 `identity_only`（仅身份回查）、`configuration_verified=false`。完整工具和模型配置以创建请求体为准；该状态不覆盖逐字段配置回读。

## 7. DataMate 在线提交

先只读检查：

```bash
python demos/task1_demo.py \
  --datamate-url http://localhost:18000 \
  --datamate-mode dry_run
```

只有在已有 DataMate 源数据集 ID 且明确允许创建任务时执行。为保证复现结果能证明远端写入和回查，推荐直接调用任务一使用的 DataMateClient，提交文本清洗算子对应的模板/任务：

```powershell
$srcDatasetId="<existing-datamate-source-dataset-id>"
$destName="task1_submit_smoke_cleaned_$(Get-Date -Format yyyyMMddHHmmss)"
@"
import json
from pathlib import Path
from src.operators.data_ops.datamate_client import DataMateClient

client = DataMateClient("http://localhost:18000", timeout=8)
summary = client.catalog_summary(
    ["drop_duplicate_rows", "normalize_column_types"],
    src_dataset_id="$srcDatasetId",
    src_dataset_name="task1_submit_smoke",
    dest_dataset_name="$destName",
    mode="submit",
)
result = {
    "template_status": summary["cleaning_template"]["submission"]["status"],
    "template_resource_id": summary["cleaning_template"]["submission"]["resource_id"],
    "task_status": summary["cleaning_task"]["submission"]["status"],
    "task_resource_id": summary["cleaning_task"]["submission"]["resource_id"],
    "dest_dataset_name": "$destName",
}
Path("outputs/competition_evidence/online-integration").mkdir(parents=True, exist_ok=True)
Path("outputs/competition_evidence/online-integration/datamate-submit.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, ensure_ascii=False, indent=2))
if result["template_status"] != "verified" or result["task_status"] != "verified":
    raise SystemExit(1)
"@ | python -
```

提交结果中的 `resource_id`、`verification` 和 `status=verified` 分别证明服务端返回
资源 ID、详情接口可回查和资源创建已确认。若只有 `submitted_unverified`，不能将其
描述为完整在线成功。

## 8. 证据文件

在线集成证据 JSON 位于 `outputs/competition_evidence/online-integration/`（答辩包内为 `competition_submission/defense-package-final/evidence/online_integration/`）：

- `datamate-submit-20260702-final.json` — DataMate 正式提交回查
- `datamate-readiness-20260702-final.json` — DataMate 就绪检查（算子/模板/任务）
- `task2-neo4j-live-smoke-20260702-final.json` — Neo4j 冒烟测试
- `probe-20260702-final.json`、`prepare-20260702-final.json` — Nexent 探测与准备
- `openapi-submit-no-token-rerun-20260702.json`、`agent-submit-no-token-rerun-20260702.json` — OpenAPI 导入与 Agent 回查

2026-07-03 复验补充：

- `probe-20260703-fullstack.json` — 全栈 probe（JWT 刷新后）
- `datamate-submit-20260703-rerun.json` — DataMate submit 复验
- `../benchmarks/task1_datamate_submit.json` — 任务一 submit benchmark

DataMate 在 WSL 中恢复：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_datamate_wsl.ps1 `
  -Mode start -WaitSeconds 90
```

就绪探测（算子、模板、任务三项均成功时返回码为 0）：

```bash
python scripts/datamate_readiness.py \
  --url http://localhost:18000 \
  --timeout 8
```

2026-06-18 历史对照 JSON 保留于 `outputs/competition_evidence/online-integration/`；2026-06-16 及更早对照 JSON 见答辩包 `../competition_submission/defense-package-final/evidence/online_integration/`（见该目录 README）。NPU 复验见 [NPU 优化说明](npu_optimization.md)。
