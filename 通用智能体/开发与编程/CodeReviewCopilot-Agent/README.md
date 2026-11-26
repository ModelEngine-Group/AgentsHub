# 【开发与编程】代码审查助手（code_review_copilot）

## 简介

一个面向 Python 项目的提交前代码审查智能体，基于 Nexent v2.7.0。它把审查拆成三件互相不能替代的事：**MCP 取到真实代码、行号，并按规则实现做确定性候选扫描**、**Skill 提供判定的唯一标准（规则表、严重度、条款号白名单、输出契约）**、**知识库给出团队条款依据并过滤误报**。输出是一张每行都带「依据」列的表格，能回溯到具体条款；也能按请求输出机读 JSON，或离线导出 SARIF 2.1.0 供 CI 使用。

> **与仓库内已有条目的区别**：`通用智能体/开发与编程/` 下已有的 `代码解析助手`、`项目架构研究助手` 做的是「读懂代码、解释结构」。本作品做的是「按固定规则判定并给出可回溯依据」，且明确要求三件套同时在 `agent.json` 里留下痕迹（`mcp_info`、`skill_names`、`knowledge_base_search`），便于审核者直接核验。两者定位不重叠。

> **关于 `agent.json` 的来源**：本文件由 **Nexent v2.7.0 平台导出**（2026-11-15，草稿版 v0，导出包 `code_review_copilot.zip`），不是手工构造的模板。已脱敏：`tenant_id` 置空、`author` 为 `Community Contributor`，全文件无 API Key、无邮箱、无本机绝对路径。
>
> `model_ids`（`[3]`）和 `index_names`（`1-07fbb0fc69ae422fb98a275712ed0933`）是本机数据库 ID，`mcp_url` 指向本机自建服务——**导入到别的环境必须重新选择模型、重新勾选知识库、改成自己的 MCP 地址**。导出包里的 `skills/code-review-rules.zip` 与本目录 `skills/` 下的明文包逐字节一致（4 个文件、28410 字节），已核对。
>
> 早期模板用过 `<TO_CONFIG:...>` 占位符，但导入向导 `AgentImportWizard.tsx` 会把这种字符串识别成「待配置」，据此判定 MCP 未安装并要求现场安装，而同名 MCP 会被后端以 409 拒绝。现在填真实值，向导按 name+URL 直接匹配到平台记录。

> **脱敏与授权**：不含租户 UUID、邮箱、API Key、本机绝对路径。三份知识库文档与六个评审样例均为本作品原创编写，不含真实凭据、真实业务代码或个人信息；样例里的 `sk-live-...` 是随机编造的演示字符串。本作品以 **MIT** 授权发布，见 `LICENSE`。

## 功能

- **只读取数**：通过自建 MCP 服务读变更列表、统一 diff 和带物理行号的源码。工具拒绝绝对路径、`..`、`.git/`、`.env*` 与常见密钥文件，不写文件、不执行被审查代码、不联网。
- **确定性候选扫描**：第 5 个 MCP 工具 `scan_candidates` 调用 Skill 包内**唯一一份**规则实现 `scripts/scan.py`，产出带 `fingerprint` 的候选列表。传 `read_diff` 的结果时只保留 `+` 行候选，与「只审新增行」对齐。候选集可复现，不随模型波动。
- **扫描范围排除**：主动跳过 `vendor/`、`third_party/`、`generated/`、`migrations/`、`snapshots/`、`*_pb2.py` 等生成或第三方代码，并写进 `excluded_files` 明示，不静默忽略。对齐开源审查工具（PR-Agent 等）的 scope paths 实践。
- **SARIF 2.1.0 离线导出**：`scripts/to_sarif.py` 把候选映射成 OASIS 标准的 `ruleId / level / locations / partialFingerprints`，可直接喂给 GitHub Code Scanning、VS Code SARIF Viewer。这是 CI 用导出工具，**不进入智能体运行时**。
- **规则化判定**：Skill `code-review-rules` 提供 8 条编号规则（CR-001~CR-008）、固定严重度、条款号白名单和输出契约。Skill 在平台里是注入提示词的**规则文本**，不直接执行；可执行实现由 MCP 通道调用。
- **依据可回溯**：报告的「依据」列必须写「文档名 + 条款号」，编号只能取自白名单。检索为空时如实标注「依据待补」，不编条款号。
- **检索有预算**：`knowledge_base_search` 每轮最多 2 次（一次合并查条款、一次查豁免），不按候选逐条检索。
- **误报过滤**：《豁免清单》明确哪些情形不上报。被豁免的候选仍要列在「已豁免」小节，不能悄悄丢掉。
- **范围完整性**：`read_diff` 返回 `truncated=true` 时必须写出实际覆盖的文件；`excluded_files` / `skipped_files` 非空时也要说明覆盖范围，不能声称已检查全部改动。
- **多轮收窄与机读输出**：支持「只看 blocker」「重新审某个文件」「第 N 条为什么被豁免」，以及按请求输出固定结构的 JSON；`fingerprint` 用来稳定指认具体条目。
- **读不到就停**：MCP 不可用或 `scan_candidates` 报错时直接说明原因并停止，不编造文件名、行号或代码片段，也不改用自己逐行判断兜底。
- **不越权下结论**：不判断「可以合并」，不重写整个文件，只给最小改法。

## 运行环境与依赖

| 项目 | 配置 |
| --- | --- |
| 平台版本 | Nexent v2.7.0 |
| 智能体变量名 | `code_review_copilot` |
| 显示名 | `代码审查助手` |
| 大语言模型 | 任意支持 function calling 的对话模型（开发参考配置 `deepseek-v4-flash-0731`） |
| 内置工具 | `knowledge_base_search`（`hybrid`，`top_k=3`，`inputs` 含 `query` 与可空的 `index_names`） |
| 知识库 | `code-review-standards`，3 份原创文档，平台自动分块 |
| Skill | `code-review-rules`（`skills/code-review-rules.zip`，规则文本 + 唯一规则实现 + SARIF 导出） |
| MCP | `code-review-fs`，自建 SSE 服务，**5 个只读工具** |
| MCP 依赖 | Python 3.10+、`mcp>=1.2.0,<2`；规则实现与逻辑层仅标准库 |
| 向量模型 | 选平台已配置的向量模型即可（本机实测建库用 `BAAI/bge-m3`，连通有效；`text-embedding-v4`（1024 维）为另一可选配置） |
| `max_steps` | 20（每轮固定 MCP 4~5 次 + KB 2 次） |

MCP 五个工具：`get_repo_meta`、`list_changed_files`、`read_diff`、`read_file`（取数）+ `scan_candidates`（确定性扫描）。

## 目录结构

```text
CodeReviewCopilot-Agent/
├── README.md                         # 本文档
├── agent.json                        # 配置模板（建议平台导出后覆盖）
├── CHANGELOG.md
├── LICENSE                           # MIT
├── skills/
│   ├── code-review-rules.zip         # 平台导入用（内含 SKILL.md + scripts + references）
│   └── code-review-rules/            # 同内容明文，便于审核者直接阅读
│       ├── SKILL.md
│       ├── scripts/scan.py           # 唯一规则实现：MCP 调用 + 本地离线校验
│       ├── scripts/to_sarif.py      # 离线导出 SARIF 2.1.0
│       └── references/output-contract.md
├── mcps/
│   ├── code_review_mcp_server.py     # 5 个只读 MCP 工具
│   ├── fs_tools.py                  # 纯标准库逻辑层，含 scan_candidates 与范围排除
│   ├── requirements.txt
│   └── README.md
├── knowledge/                        # 知识库源文件
│   ├── README.md                     # 导入步骤、分块建议、召回自检
│   ├── 编码规范_v1.2.md
│   ├── 历史评审结论_节选.md
│   └── 豁免清单.md
└── eval/
    ├── README.md
    ├── init_fixture_repo.py          # 生成可测 diff 的 git fixture 仓库
    └── samples/                      # 5 个缺陷样例 + 1 个干净负样本
```

> **部署约束**：`mcps/` 与 `skills/` 必须同属一个 agent 目录。`fs_tools.py` 按相对路径加载 `skills/code-review-rules/scripts/scan.py`；找不到时工具返回明确 error 并停止，不会静默退化。

## 导入与配置

1. **导入 Skill**：「技能」→ 上传 `skills/code-review-rules.zip`，确认技能名为 `code-review-rules`。frontmatter 已用平台真实 `SkillLoader` 解析验证：`name`/`description` 正常，`license`/`compatibility` 会被平台安全忽略。
2. **建知识库**：左侧「智能体开发」→「知识库配置」→ 新建 `code-review-standards`，向量模型选平台已配置的模型（本机实测用 `BAAI/bge-m3`），上传三份 `.md`（不要传 `knowledge/README.md`），等向量化完成。平台自动分块，**没有「按 `##` 分块」的设置项**；`hybrid / top_k=3 / rerank=false` 是工具参数，在第 4 步的「工具配置」弹窗里设，不在建库时设。
3. **启动并启用 MCP**：按 `mcps/README.md` 在虚拟环境里装依赖并启动 SSE 服务，拿到形如 `http://<host>:8766/sse` 的 URL。Nexent 跑在 Docker 里时用 `host.docker.internal`。平台「工具 → MCP 配置」里除了新增 `code-review-fs`，**还必须把「启用状态」开关打开**并等工具列表刷新完 —— 只创建记录不算启用，`enabled=false` 时平台不会把 5 个工具注册进工具表，后续导入 `agent.json` 会报 `Cannot find tool ... in mcp.`。
4. **导入智能体**：导入 `agent.json`。向导第 2 步「MCP 服务器」应显示 `code-review-fs` **已安装**（因为 `mcp_url` 已是真实值、平台里也有同名同 URL 的记录），不要点「安装」。导入后在「模型与提示词」里**重新选择模型**（`model_ids` 是本机示例 ID，输出预留 Token 调到 ≥ 8192），并在「工具配置」弹窗里核对 `knowledge_base_search` 已绑定 `code-review-standards`、参数为 `hybrid / top_k=3 / rerank=false`。
5. **核对接线**：确认 `skill_names` 非空、工具列表里 `source: mcp` 的有 **5 个**、`knowledge_base_search` 的 `index_names` 与本机知识库一致。此时再跑 `knowledge/README.md` 第 3 节的 4 条召回自检（平台没有独立的知识库检索调试窗口，只能在智能体调试窗口或对话里跑）。在调试窗口调一次 `scan_candidates`，确认它能找到 Skill 包里的规则实现并返回候选。
6. **准备验收仓库**：跑 `python eval/init_fixture_repo.py`，按输出拿到一个带两次提交的 git 仓库路径。验收表 A/B/C 三组都用这个路径，不要直接用 `eval/samples/`。
7. **重新导出**：平台里跑通第 6 节至少一条用例后重新导出 `agent.json` 覆盖模板，并按第 7 节做脱敏检查。

**安全提醒**：不要把 API Key、租户密码、登录令牌、本机绝对路径提交到仓库。fixture 仓库是本地验收产物，也不要提交。

## 使用示例与预期输出

### 示例 1：完整审查

输入：

```
帮我审查 <fixture 仓库路径> 最近一次提交的改动
```

预期：依次调用 `get_repo_meta` → `list_changed_files` → `read_diff` → `scan_candidates`（传入 diff 原文）→ `knowledge_base_search`（条款）→ `knowledge_base_search`（豁免），输出六列表格，每行带条款号，末尾有统计行与「已豁免」小节。报告里**不应出现 `legacy.py`**（删除文件）或 `utils/clean.py`（干净负样本）。

### 示例 2：测试文件（验证知识库确实在起作用）

输入：

```
tests/test_login.py 这段也一起看看
```

预期：`read_file` 读到原文、`scan_candidates` 按文件扫描报出第 7、9 行两条候选；查《豁免清单》§2.1、§2.2 后**不上报**，输出「未发现符合规则的问题」并把两条列为已豁免。

### 示例 3：只粘贴代码、不给仓库路径

预期：直接追问仓库路径与 `base_ref`，说明需要读取真实文件才能做确定性扫描；不基于粘贴内容给结论，也不标注「相对行号」。

### 示例 4：MCP 不可用

预期：输出「无法获取变更」并停止，附启动命令；不出现任何编造的文件名或行号。

## 验收测试记录

### A. 本地可复现部分（已实测）

`eval/samples/` 有 5 个带已知缺陷的文件和 1 个干净负样本。三套命令都能跑：

```bash
# 离线校验：直接跑规则实现
python skills/code-review-rules/scripts/scan.py --root eval/samples

# 平台同款：走 MCP 逻辑层（含 fingerprint / diff 只审 + 行）
python -c "import sys; sys.path.insert(0,'mcps'); import fs_tools as f; \
import subprocess; d=subprocess.run(['git','-C','<fixture 仓库路径>','diff','--no-color','HEAD~1','HEAD'], \
capture_output=True,text=True).stdout; \
print(f.dumps(f.scan_candidates('<fixture 仓库路径>', diff_text=d)))"

# 导出 SARIF 2.1.0（CI 用）
python skills/code-review-rules/scripts/scan.py --root eval/samples --json \
  | python skills/code-review-rules/scripts/to_sarif.py > results.sarif
```

实测（Python 3.13.14，2026-11-15）：

| 样例 | 候选 | 应上报 | 应豁免 |
| --- | --- | --- | --- |
| `services/payment.py` | CR-001、CR-002 | 2 | 0 |
| `services/sync.py` | CR-003、CR-004 | 2 | 0 |
| `utils/http.py` | CR-006、CR-008 | 2 | 0 |
| `utils/validate.py` | CR-005、CR-007 | 2 | 0 |
| `tests/test_login.py` | CR-005、CR-007 | 0 | 2 |
| `utils/clean.py` | 无 | 0 | 0 |
| **合计** | **10** | **8** | **2** |

`scan_candidates` 对 fixture 的 `HEAD~1..HEAD` diff 扫描：`scope=added_lines`、`file_count=6`、`candidate_count=10`、10 个 `fingerprint` 互不相同且两次运行完全一致、`legacy.py` 出现 0 次、`clean.py` 出现 0 次、`excluded_files` 与 `skipped_files` 均为空。全文模式同样是 10 条。

补充验证：

- **范围排除**：传混合列表 `services/payment.py + vendor/lib.py + generated/schema_pb2.py + utils/clean.py`，得到 2 条候选、`excluded_files` 恰好列出 2 个生成/第三方路径并带原因、`skipped_files` 为空。10 条基准检出集不受影响。
- **SARIF 导出**：`version=2.1.0`、8 条 `rules`（只声明实际命中的规则）、10 条 `results`、level 覆盖 `error/warning/note`、每条都有 `physicalLocation` 与 `partialFingerprints`。
- **Skill 可导入**：用 Nexent 运行时里的真实 `SkillLoader` 解析 `SKILL.md` 成功，`name=code-review-rules`、`description` 493 字符；附加的 `license`/`compatibility` 被平台按白名单过滤，不影响导入。

三个探针：`tests/test_login.py` 测豁免清单是否真被检索、`utils/clean.py` 测不凑数、`legacy.py`（仅 fixture 基线且第二次提交删除）测只审 `+` 行。

### B. 三组消融对照（平台已实测）

准备：`python eval/init_fixture_repo.py`，得到一个真实 git 仓库路径。

同一输入 `帮我审查 <fixture 仓库路径> 最近一次提交的改动`，**每次只改一项配置**，跑三遍：

| 组 | 配置 | 预期差异 | 平台实测 |
| --- | --- | --- | --- |
| A | 完整：MCP + Skill + 知识库 | 上报 8 条，依据列全部有条款号，已豁免 2 条，不含 `legacy.py` / `clean.py` | ✅ 与预期一致：8 条 / 依据 8/8 有条款号 / 豁免 2 条（§2.1、§2.2），无 `legacy.py` / `clean.py`（2026-11-15，agent_id=3） |
| B | 关知识库：保留工具，把 `index_names` 清空使其返回空结果 | `tests/test_login.py` 的 2 条被误报，上报变 10 条；依据列全部变「未命中团队规范」 | ✅ 与预期一致：上报 10 条（`tests/test_login.py:7`、`:9` 被报出），依据 10 条全部「未命中团队规范（通用实践）（依据待补）」，无编造条款号；已豁免 0 条（2026-11-15） |
| C | 关 MCP：停掉 8766 端口服务 | 不输出任何具体问题，不编造行号（实测为平台报错，不产出报告） | ✅ `Run Agent Error: unhandled errors in a TaskGroup (1 sub-exception)`，0 条结论 |

> **实测说明**：三组已于 2026-11-15 在 Nexent v2.7.0（agent_id=3，模型 `deepseek-v4-flash-0731`）实测完成，结果均与预期一致。B 组通过把 `index_names` 改为不存在的 ID 实现关知识库（跑完已还原）；C 组停掉 8766 服务实测，平台在 MCP 连接层直接抛异常而非降级为文字说明，见「已知限制」。

> B 组务必**保留 `knowledge_base_search` 工具**、只清空索引。删掉整个工具会触发「工具不存在」错误，测不到「检索返回空」这条路径。

> 候选集由 `scan_candidates` 确定性产生，A/B 两组的条数差不再受模型波动影响。

### C. 功能用例自检（#9 / #10 已实测，#1-#8 待平台复测）

| # | 输入 | 期望 | 平台实测 |
| --- | --- | --- | --- |
| 1 | 审查 fixture 仓库的改动 | 8 条，依据列有条款号，不报 `legacy.py` / `clean.py`；日志里 `scan_candidates` 被调用 | 待填 |
| 2 | 单独看 `tests/test_login.py` | 0 条上报，2 条已豁免且注明 §2.1 / §2.2；已检查范围写 `:1-9` | 待填 |
| 3 | 断开 MCP 后重试 | 停止并说明原因，不编代码、不改用自己判断兜底 | 待填 |
| 4 | 只粘贴一段代码、不给路径 | 追问仓库路径，不给审查结论 | 待填 |
| 5 | 问「成都欢乐谷门票多少钱」 | 不拿编码规范内容硬凑答案 | 待填 |
| 6 | 「给我 JSON 结果」 | 按 output-contract 第 9 节输出四段结构，`summary.reported` 等于 `findings` 长度 | 待填 |
| 7 | 「只看 blocker」 | 基于本轮候选过滤，不产生新候选 | 待填 |
| 8 | 同一仓库连审两次 | `fingerprint` 集合完全一致 | 待填 |
| 9 | 本地导出 SARIF | `results.sarif` 含 10 条结果、8 条规则声明、每条有 `partialFingerprints` | ✅ 已实测 |
| 10 | 扫描含 `vendor/` 的仓库 | `excluded_files` 列出被跳过的路径并带原因，报告不声称「已检查全部文件」 | ✅ 已实测（工具层） |

## 设计说明

**为什么三件套必须同时存在，而不是配上去充数。** 三者分别解决一类无法互相替代的失败：没有 MCP，行号只能靠模型编、候选也无从确定性产生；没有 Skill，同一段 f-string 拼 SQL 这次报 blocker、下次报 major；没有知识库，「依据」列空掉且测试文件产生误报。消融对照表就是把这个判断变成可观测的数字。

**为什么把规则执行放进 MCP 通道，而不是留在提示词里。** 查证过 Nexent 的实现：Skill 是把 `SKILL.md` 作为 `skill_content` 文本注入，平台没有通用代码执行工具。所以「让智能体运行 Skill 里的脚本」做不到。现在由 MCP 的 `scan_candidates` 调用 Skill 包内**同一份** `scan.py`：Skill 定标准、MCP 做执行，规则只有一份实现，候选集可复现。SKILL.md 的 frontmatter、目录结构（`scripts/` + `references/`）、渐进披露也都对齐了 Agent Skills 开放规范。

**为什么豁免规则放在知识库，不写进扫描器。** 扫描器刻意只做文本匹配，测试目录里的 `print` 也会进候选。把豁免写死在代码里，知识库就变成装饰；放到知识库里，智能体必须真的检索一次才能正确定稿。

**为什么范围排除放在工具层。** 开源审查工具（PR-Agent 等）的配置指南都强调 scope paths：排除 lockfiles、generated、vendored、snapshots。这跟「豁免」是不同层次的问题 —— 生成代码根本不该进入扫描范围，否则会淹没真正的问题。排除结果写进 `excluded_files` 而不是静默丢掉，是为了让报告能诚实说明覆盖范围。

**为什么 SARIF 只做离线导出、不改成运行时主格式。** SARIF 2.1.0 是 OASIS 标准，我们的字段（`ruleId`/`level`/`locations`/`partialFingerprints`）本来就是它的子集，加一个转换脚本成本很低。但 SARIF 是给机器和 CI 看的，人读不了；课程演示和审核看的是带「依据」列的表格。所以保留表格为主输出，SARIF 只在需要接 GitHub Code Scanning 或 VS Code 时导出。同理，没有引入置信度评分或 PR 打分 —— 那类主观数值会和「可回溯依据」的定位冲突。

**为什么条款号要有白名单。** 让模型自由引用「团队规范第 3 章」，它很容易顺手编一个不存在的编号。白名单把允许出现的编号收窄到 8 个条款、10 条历史结论和 6 条豁免条款，越界一律按编造处理。

**为什么只审 `+` 行。** diff 里 `-` 行是被删掉的旧代码，把它报成「本次改动的问题」是典型误报。`scan_candidates` 传 `diff_text` 时直接按 `+` 行过滤候选，从工具层面就把这条路堵死；fixture 里被删除且含密钥的 `legacy.py` 用来验证它确实生效。

**为什么给每条候选加 fingerprint。** `sha1(rule_id|file|line)` 让「第 N 条」这种随过滤顺序变化的指认方式可以被稳定标识替代，也正好对应 SARIF 的 `partialFingerprints`（跨运行去重与生命周期跟踪）。这一步只输出指纹、不做存储，MCP 保持纯只读。

**为什么 MCP 服务只读。** 代码审查的输入必须是本机仓库的真实 diff，通用搜索类 MCP 拿不到。只读边界（拒绝 `.env`、`.git`、绝对路径，15 秒超时，不执行被审查代码）同时也是仓库审核标准第 5 条的直接证据。

## 已知限制

- 规则实现是逐行文本匹配，不做 AST 语义分析；跨行的 HTTP 调用、`with` 跨行写法可能漏检或误检。刻意保留逐行匹配，是为了让检出集可预测、可复现。
- SARIF 是离线导出格式，不是智能体运行时输出；导出的是**候选**，不含知识库依据与豁免判定。
- `mcps/` 与 `skills/` 必须在同一个 agent 目录下；单独部署 MCP 服务而不带 Skill 明文包会返回明确错误。
- 默认 `base_ref = HEAD~1`，只覆盖最近一次提交；多提交范围需要在请求里显式给 `base_ref`。
- `eval/samples/` 本身不是 git 仓库；要测 diff 类工具必须先用 `eval/init_fixture_repo.py` 生成 fixture 仓库。
- 只粘贴代码而不给仓库路径时不提供审查（见示例 3），这是为了保证候选能由确定性工具产出。
- MCP 服务需自行启动。**连接层故障时平台会直接抛 `Run Agent Error: unhandled errors in a TaskGroup`，整个 run 终止，不会降级成「无法获取变更」的文字说明**；提示词里的停止规则只对工具返回 `{"error": "..."}` 的情况生效（如路径不存在、越权被拒）。C 组消融实测记录了这个差异。不管哪种情况，都不会产出编造的行号或发现项。
- 支持 Python 语法；其他语言未覆盖。
