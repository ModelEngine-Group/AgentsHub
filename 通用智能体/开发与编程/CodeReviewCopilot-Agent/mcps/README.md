# MCP 服务：code-review-fs

代码审查助手的「取数与确定性扫描」层。五个工具全部**只读**：前四个取到真实代码、diff 和物理行号，第五个按规则实现产出可复现的候选列表。没有这一层，报告里的 `文件:行号` 只能靠模型编，候选集也会随模型波动。

## 启动

```bash
# 建议在工作区内自建虚拟环境，不要装进系统解释器
python -m venv .venv
. .venv/Scripts/activate        # Linux/macOS 用 . .venv/bin/activate
pip install -r requirements.txt

# 本机使用：只监听回环地址
python code_review_mcp_server.py --transport sse --host 127.0.0.1 --port 8766
# Nexent 在 Docker 里时使用 0.0.0.0，并限定可读仓库根目录（见下方安全边界）
python code_review_mcp_server.py --transport sse --host 0.0.0.0 --port 8766 --allowed-root /path/to/repos
# 控制台打印：MCP URL: http://<服务器地址>:8766/sse
```

端口用 `--port` 指定。8766 是本机实测空闲的端口（8765 被另一个本地服务占用）；若你的环境里 8766 也被占，换任意空闲端口即可，但**要同步把下面平台里填的 URL 改掉**。

### 安全边界

服务没有鉴权，只应监听可信回环或容器桥接网络，不要暴露到公网。`--allowed-root` 可重复指定，用于把 `repo_path` 限制在特定仓库根目录内；指定后所有工具都拒绝范围外的路径。未指定时保持向后兼容（不限制），本地调试可用，生产部署建议始终指定。

在 Nexent「智能体开发 → 工具 → MCP 配置」里新建一条：

| 字段 | 值 |
| --- | --- |
| `mcp_server_name` | `code-review-fs` |
| `mcp_url` | `http://<服务器地址>:8766/sse` |

如果 Nexent 跑在 Docker 里而服务跑在宿主机，`<服务器地址>` 用 `host.docker.internal`，不要用 `127.0.0.1`（容器内的 127.0.0.1 指向容器自己）。仓库中已有作品用的是同一个写法。

### 依赖装不上时（国内网络）

直连 PyPI 慢/超时的话，用本机 `uv` 走清华镜像，并清掉终端会话里的代理变量：

```powershell
$env:HTTP_PROXY=$null; $env:HTTPS_PROXY=$null; $env:http_proxy=$null; $env:https_proxy=$null
uv pip install --python .\.venv\Scripts\python.exe --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

本机实测：直连 pypi.org 的 pip/uv 都在拉 metadata 阶段超时，换清华源后一次装成 `mcp 1.30.0`。

### Docker 容器访问与 Host 白名单

`FastMCP` 构造时若 host 是 `localhost`/`127.0.0.1`，会自动启用 DNS rebinding 保护，Host 白名单只含 localhost。平台从容器带 `Host: host.docker.internal:8766` 过来会被 uvicorn 以 **`421 Invalid Host header`** 拒绝 —— 这不是端口或防火墙问题。

本服务构造时显式用 `host="0.0.0.0"`，并在支持 `transport_security` 的 mcp 版本上把 `host.docker.internal:*` 加入白名单（保护保留）；旧版 mcp 无该模块时自动降级，仍可访问。

验证（在 `nexent-runtime` 容器内）：

```bash
docker exec nexent-runtime curl -s --noproxy '*' -o /dev/null -w "%{http_code}\n" http://host.docker.internal:8766/sse
# 应返回 200
```

## 工具清单

| 工具 | 入参 | 返回 |
| --- | --- | --- |
| `get_repo_meta` | `repo_path` | 分支、短 HEAD、未提交文件数、是否为 git 仓库 |
| `list_changed_files` | `repo_path`, `base_ref` | `base_ref..HEAD` 的变更文件与状态（A/M/D） |
| `read_diff` | `repo_path`, `base_ref`, `file_path`, `max_lines` | 统一 diff，`+` 行对应新文件；最多 400 行 |
| `read_file` | `repo_path`, `file_path`, `start_line`, `end_line` | 带物理行号的源码行，最多 200 行 |
| `scan_candidates` | `repo_path`, `file_paths`, `diff_text` | 带 `fingerprint` 的候选发现项；传 `diff_text` 时只保留 `+` 行 |

所有工具返回 JSON 字符串。可预期错误返回 `{"error": "..."}`，智能体据此停止审查而不是编内容。

### `scan_candidates` 的规则来源

它**不自带规则**，而是运行时 import 同一 agent 目录下 `skills/code-review-rules/scripts/scan.py`——三件套里 Skill 提供规则，MCP 提供执行通道，规则只有一份实现。参数行为：

- 传 `diff_text`（`read_diff` 的原文）：解析 unified diff，只保留 `+` 行上的候选，`scope=added_lines`
- 传 `file_paths`（逗号分隔相对路径）：按文件全文扫描，`scope=whole_files`
- 两者都不传：扫描仓库内全部 `.py` 文件

每一项候选附 `fingerprint = sha1(rule_id|file|line)[:12]`，两次运行结果一致。生成代码与第三方目录按扫描范围规则排除后列在 `excluded_files`；读不到或违反只读白名单的文件进 `skipped_files`，两者都不静默忽略。候选**不等于上报**：是否豁免由智能体查知识库《豁免清单》决定。

> 部署约束：`mcps/` 与 `skills/` 必须同属一个 agent 目录。找不到 `scan.py` 时工具返回明确 error，智能体按约束停止，不会退化成凭感觉判定。

## 只读边界

- 不写文件、不执行被审查代码、不联网；`git` 调用固定参数列表，`shell=False`，15 秒超时
- 拒绝绝对路径、`..`、`.git/`、依赖目录、`.env*` 与常见密钥文件名
- 单文件上限 200 KB，单次 diff 上限 400 行，单行读取上限 200 行
- 错误信息只返回类型名和一句可判定说明，不回传堆栈

这几条同时是仓库审核标准里「不含恶意代码或数据窃取逻辑」的直接证据。

## 逻辑层单独验证

`fs_tools.py` 只用标准库，可以不装 `mcp` 单独测：

```bash
python -c "import fs_tools; print(fs_tools.dumps(fs_tools.read_file('..', 'eval/samples/services/payment.py', 1, 12)))"
python -c "import fs_tools; print(fs_tools.dumps(fs_tools.scan_candidates('..', file_paths=['eval/samples/services/payment.py'])))"
```

第二条会同时验证「规则实现能被找到」和「候选带 fingerprint」。期望输出 `candidate_count=2`（CR-001、CR-002）。

`eval/samples/` 不是 git 仓库，`read_diff` 会返回一条明确的 error；`read_file` 仍可用。这也是为什么平台验收必须先用 `eval/init_fixture_repo.py` 生成一个带两次提交的 fixture 仓库——直接拿散文件当输入，A/B/C 三组消融只会得到「无法获取变更」。C 组消融（关 MCP）的正确做法是停掉 8766 端口的 SSE 服务，而不是指向非 git 目录。

## 打包说明

`skills/` 里的 zip 是给平台上传用的；本目录与 `skills/code-review-rules/` 保留明文源码，方便审核者直接看到规则和服务实现，不必解压二进制。
