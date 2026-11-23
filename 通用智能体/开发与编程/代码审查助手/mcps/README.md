# GitHub MCP 接入说明

本目录提供两种接法。**推荐方式 A**：不需要跑容器，也不需要写 MCP Server，直接用 Nexent 的「API 转 MCP」。

## 方式 A：API 转 MCP（推荐）

### 1. 创建 Fine-grained PAT

GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens：

| 权限 | 级别 | 用途 |
|---|---|---|
| Metadata | Read | 基础仓库信息 |
| Contents | Read | 读取文件（可选工具） |
| Pull requests | Read and write | 读取 PR、提交 review |

只勾选需要接入的仓库，不要给全账号权限。

### 2. 在 Nexent 中添加

1. 智能体开发 → 工具 → **MCP 配置** → 集成类型选「API 转换为 MCP」
2. 服务名称：`github-pr`
3. Base URL：`https://api.github.com`
4. OpenAPI JSON：粘贴 `github-pr.openapi.json` 全文
5. 认证：Bearer，填上一步的 PAT
6. 保存后做连通性校验

生成 4 个工具：

| 工具 | 用途 |
|---|---|
| `github_get_pr` | 读 PR 基本信息、head sha |
| `github_list_pr_files` | 读变更文件与 patch（拼成 diff） |
| `github_get_file_content` | 读完整文件补上下文（可选） |
| `github_create_review` | 提交 review 与逐行评论 |

### 3. 在智能体里启用

编辑「代码审查助手」→ 工具 → MCP → 勾选上面 4 个（至少前两个 + `github_create_review`）→ 保存。

### 4. 验证

```bash
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/repos/OWNER/REPO/pulls/1
```

能看到 PR JSON 即通。

## 方式 B：官方 GitHub MCP Server

适合需要 issue、搜索、代码扫描等更多工具的团队。Nexent 支持远程或容器方式，具体启动参数以 [github/github-mcp-server](https://github.com/github/github-mcp-server) 的 README 为准。

注意：

- 容器方式需要 Docker，且 Nexent 容器要能访问 Docker socket 或同网络；
- 远程方式需要 Nexent 容器能出网到 GitHub；
- 工具名与方式 A 不同，需在智能体里重新勾选。

## 网络与权限

- Nexent 若跑在 Docker 中，MCP 配置里的 `127.0.0.1` 指向容器自己；GitHub 是公网地址，通常只需容器能出网。
- 公司有代理时，在 Nexent 容器内配置 `HTTPS_PROXY`，否则连通性校验会失败。
- GitHub 认证用户限流 5000 次/小时；`files` 接口单页最多 100 个文件，超过要翻页。

## 回帖安全策略

1. 先生成评论草稿，人工确认后再调用 `github_create_review`。
2. 默认 `event=COMMENT`，不要用 `REQUEST_CHANGES` 阻塞合并。
3. 评论里的行号必须先通过 `review_diff.py --mode verify`。
4. 只对本次 diff 的新增行评论，不对未变更代码留言。
5. Token 只放 Nexent 的 MCP 配置里，不要写进 agent.json 或仓库。

## 成本与失败处理

- `github_list_pr_files` 返回的 `patch` 对大文件可能为空；此时让用户粘贴该文件 diff，或跳过并在报告里说明。
- PR 超过 3000 个文件时 GitHub 不返回完整列表，应改用本地 `git diff`。
- 401：Token 过期或权限不足；403：限流或仓库未授权；404：仓库/PR 不存在或无权限（私有仓库常见）。
