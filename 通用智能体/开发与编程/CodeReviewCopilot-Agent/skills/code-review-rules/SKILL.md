---
name: code-review-rules
description: Deterministic pre-merge code review rules for Python changes. Provides eight numbered rules (CR-001..CR-008), fixed severity levels, an allowed clause-number whitelist, a report contract and the single executable rule implementation (scripts/scan.py) invoked through the MCP scan_candidates tool. Use when the user asks to review a diff, a commit, changed files or a repository. Triggers on "review this diff", "code review", "check my changes", "审查代码", "看一下这次提交", "pre-merge check", "CR-001".
license: MIT
compatibility: Requires the code-review-fs MCP server and git. Scripts run on the MCP host, not inside the agent runtime.
---

# Python 代码审查规则集（CR-001..CR-008）

这个 Skill 提供**判定的唯一标准**：命中哪条规则、严重度是多少、报告长什么样。
它**不负责**取代码与执行扫描（由 MCP 负责），也**不负责**解释团队为什么这么规定或哪些情形豁免（由知识库负责）。

> Nexent 把 `SKILL.md` 作为文本注入提示词，平台没有通用代码执行工具。因此规则的可执行实现放在本包 `scripts/scan.py`，由 MCP 服务的 `scan_candidates` 工具调用。**规则只有一份实现**，智能体不要在提示词里另写一套、也不要尝试直接运行脚本。

## 1. 严重度定义

| 级别 | 含义 | 处理要求 |
| --- | --- | --- |
| `blocker` | 会造成安全事件、数据损坏或线上故障 | 必须修复后才能合并 |
| `major` | 在异常路径下会出错，或留下难以排查的隐患 | 建议本次修复 |
| `minor` | 不影响当前正确性，但积累后会成为负担 | 可记为后续项 |

严重度是规则的属性，不随上下文浮动。审查者不得因为「看起来不严重」而降级，也不得为了引起重视而升级。

## 2. 规则表

| ID | 名称 | 严重度 | 判定特征 | 最小改法 |
| --- | --- | --- | --- | --- |
| CR-001 | 硬编码凭据 | blocker | `api_key` / `password` / `secret` / `token` 一类变量被赋给长度 ≥ 8 的字符串字面量 | 改为读环境变量或密钥服务 |
| CR-002 | SQL 字符串拼接 | blocker | `execute` 的参数含 f-string、`%` 格式化或 `+` 拼接，且出现 SELECT / INSERT / UPDATE / DELETE | 改用参数化查询占位符 |
| CR-003 | 裸 except / 吞异常 | major | `except:` 或 `except ...: pass`，异常被静默丢弃 | 捕获具体异常，记录日志或向上抛 |
| CR-004 | 资源未纳入上下文管理 | major | `open(...)` 结果直接赋值，未出现在 `with` 语句中 | 用 `with open(...) as f:` |
| CR-005 | 日志输出敏感字段 | major | `print` / `logger.*` 的参数含 password / token / secret / id_card 一类字段 | 脱敏或删除该输出 |
| CR-006 | 可变默认参数 | minor | 函数签名里默认值是 `[]` / `{}` / `set()` | 改 `None` 后在函数体内初始化 |
| CR-007 | assert 做入参校验 | minor | 非测试路径上用 `assert` 校验外部输入 | 改抛 `ValueError` / `TypeError` |
| CR-008 | HTTP 请求缺少超时 | minor | `requests.get/post/put/delete` 调用未传 `timeout=` | 补 `timeout=(连接, 读取)` |

### 检测边界

- 只按**文本特征**判定，不做类型推断、不执行被审查代码、不做语义推断。
- **只审查 diff 中以 `+` 开头的新增行**。以 `-` 开头的删除行属于旧代码，不属于本次变更，不产生发现项。用 `read_file` 做全文审查时，必须在「已检查范围」里写明是全文而非 diff。
- `read_diff` 返回 `truncated=true` 表示只看到部分变更，此时要声明覆盖不完整并列出实际检查的文件，不得写「已检查全部改动」。
- 无法归入以上八条的，放「规则外观察」，最多 3 条，不计入统计。
- 纯格式、命名偏好、注释多少、换行风格，**一律不报**。
- 同一文件同一行同一规则只产生一条发现项。

## 3. 规则实现与执行通道

`scripts/scan.py` 是**唯一一份**可执行规则实现，只用 Python 标准库。它有三种用法：

| 场景 | 调用方式 | 说明 |
| --- | --- | --- |
| 平台运行时 | MCP 工具 `scan_candidates(repo_path, diff_text=read_diff 的结果)` | 由 MCP 服务 import 本文件；传 `diff_text` 时只保留 `+` 行上的候选 |
| 本地离线校验 | `python scripts/scan.py --root <dir> --json` | 给审核者读、给验收表提供可复现数字、给规则改动做回归 |
| CI 结果导出 | `python scripts/scan.py --root <dir> --json \| python scripts/to_sarif.py > results.sarif` | 转成 OASIS SARIF 2.1.0，供 GitHub Code Scanning / VS Code 消费；离线工具，不进 agent 运行时 |

`to_sarif.py` 的字段映射见 `references/output-contract.md` 第 11 节。

`scan_candidates` 返回的每个候选项额外带 `fingerprint`（`rule_id|file|line` 的短哈希）和 `scope`，用于跨次审查比对。它**故意不做路径豁免**：测试目录里的打印、断言也会出现在候选里，是否上报由智能体查《豁免清单》决定。

> 部署约束：`mcps/` 与 `skills/` 必须同属一个 agent 目录，MCP 服务按相对路径找 `skills/code-review-rules/scripts/scan.py`。找不到时工具返回明确 error，智能体按约束停止，不会静默退化成凭感觉判定。

## 4. 工作流

1. 从 MCP 取得 diff（只看 `+` 行）或带行号的文件内容；检查 `truncated` 标志。
2. 调用 `scan_candidates` 得到**确定性候选列表**；候选只能来自工具返回，不自行增加。
3. 一次合并检索知识库：拿《编码规范 v1.2》和《历史评审结论》里的条款号作为「依据」。每轮 `knowledge_base_search` 最多 2 次，不要按候选逐条查。
4. 再查一次《豁免清单》过滤候选；被过滤的进「已豁免」小节，不计入问题数。
5. 按 `references/output-contract.md` 输出表格与统计行；用户要求 JSON 时改输出机读结构。

## 5. 三种输出契约（不要混用）

| 产物 | 结构 |
| --- | --- |
| 报告表格 | 六列：`规则ID \| 文件:行号 \| 严重度 \| 问题 \| 依据 \| 建议改法` |
| `scan.py` / `scan_candidates.findings[]` 的单项 | 六字段：`rule_id / severity / file / line / snippet / message`，MCP 再加 `fingerprint / scope` |
| `scan_candidates` 外层 | `scope / file_count / candidate_count / findings[] / skipped_files / note` |

候选里没有 `evidence` 和 `fix`：依据与改法由智能体在查完知识库、套用规则表最小改法后补齐。不要把 `snippet` 当成报告里的「问题」原文照抄。

## 6. 硬性要求

- 行号必须来自 MCP 返回值，不得估算。
- 「依据」写「文档名 + 条款号」，条款号只能取自第 7 节白名单；知识库没命中就写「未命中团队规范（通用实践）」并标注「依据待补」，**不许编条款号**。
- 没有发现项时明确写「未发现符合规则的问题」，并列出已检查范围，不输出空的漂亮话。
- 不输出「可以合并」「LGTM」这类结论性判断。

## 7. 条款号白名单

只允许以下编号出现在「依据」列，白名单外一律视为编造：

| 文档 | 允许的编号 |
| --- | --- |
| 《编码规范 v1.2》 | §1.3、§2.3、§2.4、§2.5、§3.2、§3.4、§4.1、§4.2 |
| 《历史评审结论》 | R-2026-01 ~ R-2026-10 |
| 《豁免清单》 | §1、§2.1、§2.2、§2.3、§3、§4 |

注意《豁免清单》没有单独的「§2」，只有 §2.1 / §2.2 / §2.3。写「§2」按编造处理。

## 8. 多轮收窄与机读输出

用户说「只看 blocker」「重新审某个文件」「第 N 条为什么被豁免」时，基于本轮已有候选重新过滤或解释，不重新编造候选；需要重新取数才再次调 MCP。

用户明确要求 JSON 时，输出结构见 `references/output-contract.md` 第 8、9 节。其余情况一律用 Markdown 表格，不要混排。
