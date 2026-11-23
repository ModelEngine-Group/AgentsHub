# 代码审查助手（Code Review Assistant）

面向 Python 团队的代码审查智能体：读取 diff（粘贴、patch 文件或 GitHub PR），先用脚本算出真实行号并做确定性扫描，再结合团队规范逐行判断，最后输出**可定位、可验证、可回帖**的审查意见。

> 设计原则：**先确定性，后生成**。行号、密钥、危险调用交给脚本；语义判断交给模型；团队标准放知识库。

## 功能特性

- **diff 解析**：把 unified diff 拆成文件/hunk，计算 RIGHT 侧真实行号，避免模型数错行。
- **确定性扫描**：私钥、AK、Token、口令、SQL 拼接、`eval/exec`、`shell=True`、裸 `except`、可变默认参数、调试残留、TODO；密钥自动打码。
- **团队规范检索**：审查前按类别检索知识库（通用清单、安全基线、Python 缺陷、事故复盘、评论风格）。
- **引用校验**：输出前逐条校验 `file:line` 是否真的在 diff 中，不合法的条目直接丢弃。
- **分级降噪**：P0–P3 分级，单次最多 8 条，P3 只统计；没有具体失败场景的意见不输出。
- **GitHub 回帖（可选）**：生成 review comment 草稿，确认后通过 MCP 提交，默认 `COMMENT` 不阻塞合并。

## 架构

```text
用户输入（diff / patch / PR 链接）
        │
        ▼
主 Agent  code_review_assistant
  角色/边界/分级/红线 · max_steps 12
        │
        ├── Skill: code-review
        │     ├─ scripts/review_diff.py   parse / scan / verify
        │     └─ references/python-checklist.md
        ├── Tool: knowledge_base_search   团队规范知识库
        ├── Tool: read_file / create_file / analyze_text_file
        └── MCP: github-pr（可选）        拉 PR / 回帖
        │
        ▼
JSON + Markdown 报告 / 落盘 / PR 评论
```

## 目录结构

```text
代码审查助手/
├── agent.json                     # 智能体配置（导入用）
├── README.md                      # 本文件
├── skills/
│   └── code-review.zip            # 自定义 Skill（含 diff 解析/扫描脚本）
├── knowledge/                     # 知识库源文件（导入后上传）
│   ├── 01-review-checklist.md
│   ├── 02-security-baseline.md
│   ├── 03-python-idioms.md
│   ├── 04-postmortems.md
│   └── 05-comment-style.md
├── eval/                          # 最小验收用例与评分标准
│   ├── cases/*.diff
│   ├── labels.json
│   └── rubric.md
└── mcps/
    ├── github-pr.openapi.json     # GitHub PR 最小 OpenAPI（API 转 MCP）
    └── README.md                  # MCP 接入说明
```

## 快速开始

### 1. 上传 Skill

技能仓库 → 我的 Skill → 创建 → 上传技能文件 → 选择 `skills/code-review.zip`。

ZIP 内结构为 `code-review/SKILL.md` + `scripts/` + `references/` + `config/`。上传后确认技能名为 `code-review`。

> 若上传时提示「缺少 SKILL.md」，把 `code-review/` 目录**里面**的文件重新压到 ZIP 根目录再上传（不同 Nexent 版本对 ZIP 层级要求略有差异）。
> 脚本仅依赖 Python 标准库，建议 Python 3.9+。

### 2. 创建知识库

1. 知识库 → 创建知识库，名称填 `codereview`（只能用小写字母/中文）。
2. 向量模型选择与当前系统一致的 embedding 模型（不一致会导致检索失败）。
3. 上传 `knowledge/` 下 5 个 `.md`，等待状态变为「已就绪」。
4. 生成知识库摘要，便于 Agent 选库。

### 3. 导入智能体

1. 智能体开发 → 导入 → 选择 `agent.json`。
2. 选择要使用的大语言模型（原配置不绑定具体模型）。
3. 保存后进入配置页，检查工具与技能是否齐全。

### 4. 绑定知识库（必做）

打开 `knowledge_base_search` 工具的参数配置：

- `top_k`：4
- `search_mode`：hybrid
- 开启 rerank
- **在知识库列表中勾选 `codereview`**

`agent.json` 中的 `index_names` 为空，导入后必须手动选一次；否则检索不到任何内容。

### 5. 调试

用 `eval/cases/` 里的 5 个 diff 逐个测试，重点看：

- 行号是否准确（可运行 `verify` 校验）；
- 安全问题是否漏报；
- 无问题的 diff 是否会硬凑意见。

### 6. 发布

调试通过后发布。若要接入 GitHub PR，先按 `mcps/README.md` 配置 MCP，再回到智能体里勾选 4 个 GitHub 工具。

## 使用示例

**示例 1：粘贴 diff**

```text
帮我审一下这段 diff，只报 P0/P1：
<paste diff>
```

**示例 2：审查 PR**

```text
审查这个 PR：https://github.com/org/repo/pull/123
```

**示例 3：回帖**

```text
把上面的 P1 评论发到这个 PR，先给我看草稿。
```

## 输出格式

```json
{
  "verdict": "request_changes",
  "summary": "新增用户查询接口存在 SQL 注入，需修复后合入",
  "findings": [{
    "file": "src/api/user.py",
    "line": 6,
    "severity": "P1",
    "category": "security",
    "title": "SQL 注入",
    "evidence": "sql = f\"SELECT * FROM users WHERE id={uid}\"",
    "impact": "uid 来自 query 参数，可注入 1 OR 1=1",
    "suggestion": "改用参数化查询",
    "confidence": 0.92
  }],
  "stats": {"files": 1, "added": 4, "deleted": 2}
}
```

## 评测

`eval/` 提供 5 个最小用例与标注，覆盖：SQL 注入、可变默认参数、无问题、密钥+调试、裸 except+SQL 拼接。

| 指标 | 目标 |
|---|---|
| 召回率 | ≥ 70% |
| 精确率 | ≥ 70% |
| 引用准确率 | 100% |
| 安全类 P0/P1 漏检 | 0 |

评分方法见 `eval/rubric.md`。

本地已用脚本验证：5 个用例的 parse/scan/verify 全部通过，密钥输出已打码，非法行号能被拦截。

## 设计取舍

- **为什么只有一个 Skill**：diff 解析脚本与审查流程强耦合，放在同一个技能包里可以避免跨技能调用脚本的兼容问题；Nexent 官方示例 `code_review_expert` 也是同样做法。
- **为什么规则放知识库、流程放 Skill**：规则变化快、需要按语言/类别检索；流程稳定、需要按需加载省 token。
- **为什么先跑脚本**：LLM 数行号必错；确定性检查还能把「有没有密钥」「有没有 shell=True」这类事实先钉死。
- **为什么 MCP 只做取数和回写**：审查逻辑保持一套，换 Git 平台只需换 MCP。

## 边界与安全

- 只审本次 diff，不做全仓库扫描，不替代 lint/SAST。
- 不执行被审查代码，不安装依赖；默认不启用终端工具。
- 密钥只报位置和类型，不回显完整值。
- 代码视为数据，注释/PR 描述中的指令不执行。
- 回帖默认先草稿、人工确认；使用最小权限 Token。
- 代码能否送外部模型由使用方决定；本仓库不绑定模型。

## 许可证

MIT，与 AgentsHub 仓库一致。
