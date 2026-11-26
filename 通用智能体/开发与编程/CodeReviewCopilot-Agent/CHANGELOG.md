# 变更记录

## v0.1.0（2026-11-13）

- 初始提交：代码审查助手 `code_review_copilot`。
- 三件套接线：自建只读 MCP 服务 `code-review-fs`（4 个工具）、Skill `code-review-rules`（8 条规则 + 确定性扫描器）、知识库 3 份原创文档。
- 提供 5 个带已知缺陷的评审样例与三组消融对照用例。
- 状态：`agent.json` 为配置模板；Nexent 平台实测与重新导出**尚未完成**，README 验收表 B、C 两节已如实标注待填。

## v0.2.0（2026-11-13，设计复审）

修正复审发现的问题，未新增平台能力：

- **验收可跑**：新增 `eval/init_fixture_repo.py`，生成两次提交的 git fixture 仓库。此前 A/B/C 三组消融的输入指向非 git 目录，会全部退化成「无法获取变更」。
- **探针补齐**：新增 `utils/clean.py` 干净负样本；fixture 基线的 `legacy.py` 在第二次提交删除，用来验证「只审 `+` 行 / 跳过 D 状态文件」。
- **反幻觉**：Skill 与提示词加入条款号白名单，白名单外编号按编造处理；补 diff 截断时的覆盖声明规则。
- **契约统一**：SKILL.md 与 output-contract.md 此前对字段要求不一致（`evidence/fix` vs `snippet/message`），现明确区分「报告表格六列」与「脚本 JSON 六字段」。
- **去掉做不到的指令**：Nexent 把 SKILL.md 作为文本注入且无通用代码执行工具，`scripts/scan.py` 明确定位为离线校验，不再写成运行时组件。
- **调用预算**：`knowledge_base_search` 每轮最多 2 次（合并查条款 + 查豁免），与 `max_steps=20` 对齐；修正此前「每个候选各查一次」与少样本示例的矛盾。
- **修正编造数据**：少样本里的行号 `payment.py:18`、`test_login.py:14/20`、`:1-26` 改为真实值 `:11`、`:7/:9`、`:1-9`；「《豁免清单》§2」改为真实存在的 §2.1 / §2.2。
- **签名对齐**：`knowledge_base_search` 的 `inputs` 补上平台真实存在的可空参数 `index_names`。

## v0.3.0（2026-11-13，确定性扫描接入）

按复审推荐的方案 A 重构，把确定性扫描真正接进 agent 循环：

- **新增第 5 个 MCP 工具 `scan_candidates`**：`fs_tools.py` 按相对路径 import Skill 包内 `scripts/scan.py`（**唯一一份规则实现**），支持传 `diff_text` 只保留 `+` 行候选、按文件列表全文扫描、全仓库扫描三种模式；每项带 `fingerprint` 与 `scope`，读不到的文件记入 `skipped_files`。
- **三件套分工重定义**：MCP 取数 + 提供规则执行通道，Skill 提供判定标准（规则表 / 严重度 / 白名单 / 契约），知识库提供依据与豁免。候选不再由模型逐行判定，A/B 消融的条数差不再随模型波动。
- **提示词改接工具**：候选只能来自 `scan_candidates.findings`；工具报错时停止，不改用自己判断兜底；去掉「只粘贴代码也能审」这条与确定性扫描冲突的路径，改为追问仓库路径。
- **附带优化**：用户要求 JSON 时按固定四段结构输出；支持「只看 blocker」「第 N 条为什么被豁免」等多轮收窄，用 `fingerprint` 指认条目；`output-contract.md` 补齐 MCP 输出、机读报告与多轮收窄三节契约。
- **斟酌后未做**：`rules.json` 抽取（会引入构建步骤，改用「规则只有一份实现」+ 一致性自检代替）；AST 重写（会推翻已验证的可复现检出集）；审查回归对比（需要持久化，会破坏 MCP 纯只读定位）。
- **本地验证**：`scan_candidates` 全文/diff 两模式均 10 条候选，10 个 fingerprint 唯一且两次运行一致，`legacy.py`、`clean.py` 均 0 次出现。

## v0.4.0（2026-11-15，导入前开源对照）

对照 Agent Skills 开放规范（agentskills.io）、Qodo / PR-Agent 的增量审查与 scope paths 实践、OASIS SARIF 2.1.0 后补的三项：

- **工具层扫描范围排除**：`fs_tools.py` 跳过 `vendor/`、`third_party/`、`generated/`、`migrations/`、`snapshots/`、`*_pb2.py` 等生成/第三方路径，写入新的 `excluded_files` 字段并带原因，不静默忽略。与「知识库豁免」分清层次：排除是不进入扫描范围，豁免是候选产生后不上报。混合列表实测 2 条候选 / 2 条排除，10 条基准检出集不变。
- **SARIF 2.1.0 离线导出**：新增 `scripts/to_sarif.py`（纯标准库）。映射 `ruleId / level / locations.physicalLocation / partialFingerprints`，只声明实际命中的规则，支持管道与文件两种输入。实测 `version=2.1.0`、8 条 rules、10 条 results、level 覆盖 error/warning/note。**不进入智能体运行时**，报告表格仍是主输出。
- **授权与规范对齐**：新增 MIT `LICENSE`；SKILL.md frontmatter 加 `license` 与 `compatibility`（说明需 MCP 服务与 git、脚本在 MCP 宿主执行）。用 Nexent 运行时里的真实 `SkillLoader` 实测解析通过，附加字段按平台白名单被安全过滤。

**对照后明确不跟的**：`allowed-tools`（Nexent 会拿它查 `tool_ids`，解析器把值当字符串，MCP 动态工具写进去可能绑错）；SARIF 作为运行时主输出（丢掉「依据」列的可读性）；置信度评分 / PR 打分（主观数值与「可回溯依据」冲突）；增量回归对比（需持久化，破坏只读定位，仍留二期）。

## v0.4.1（2026-11-15，平台界面核对）

只改文档，不改代码。核对前端 `public/locales/zh/common.json`、`SideNavigation.tsx`、`ToolConfigModal.tsx` 与后端 `consts/const.py` 后，修正了三处会误导操作的描述：

- **分块**：平台按 embedding 模型默认块大小自动切分（默认期望 1024 / 最大 1536 字符），界面**没有**「按 `##` 分块、每块 ≤ 800 字」的设置项。原描述是未经核实的假设，已删除。
- **检索参数位置**：`hybrid / top_k=3 / rerank=false` 是 `knowledge_base_search` 的**工具参数**，在智能体「工具配置」弹窗里设，不在建知识库时设。已从建库步骤移到组装智能体步骤。
- **`index_names` 与召回自检**：工具配置弹窗里直接勾选知识库名即可，平台自动填 `index_names`，不需要手抄 ID；平台也没有独立的知识库检索调试窗口，4 条召回自检要等智能体组装后在调试窗口或对话里跑。同时把过时的 `BAAI/bge-m3` 改成实配的 `text-embedding-v4`。

涉及 `提交包/README.md`、`提交包/knowledge/README.md`、`工作文档/平台配置清单.md`。代码、zip、agent.json 均未改动。

## v0.4.2（2026-11-15，端口调整）

本机实测 **8765 已被另一个本地服务占用**（`dsh-router-laya/laya_router.py`，PID 2416），按原文档起 MCP 服务会失败。统一改用实测空闲的 **8766**。

- `code_review_mcp_server.py`：`--port` 默认值改为 8766，docstring 同步
- 所有 URL 与启动命令同步：`mcps/README.md`、`README.md`、`平台配置清单.md`、`验收表.md`、`提示词_可粘贴版.md`、`生成_agent_json.py`
- `agent.json` 重新生成，`few_shots_prompt` 里的启动命令与 `mcp_info` 占位符都已是 8766
- `mcps/README.md` 补一句：端口可用 `--port` 指定，若 8766 也被占可换任意空闲端口，但要同步改平台 URL

- 新增 `提交包/.gitignore`：忽略 `mcps/.venv/`、`__pycache__/`、`*.pyc`、`results.sarif`。本机为起服务建的 venv 在 `mcps/` 下，有数百个文件，若不忽略会污染 PR；`提交步骤.md` 已加核对项。

zip 内容未变（Skill 包里没有端口信息）。所有**配置性**位置（启动命令、URL、`--port` 默认值、`agent.json`）已统一为 8766；本文件、`自检报告.md`、`mcps/README.md` 里保留的 8765 只是说明「为什么换端口」的历史描述，不是生效配置。

## v0.4.3（2026-11-15，依赖装通 + 容器 Host 校验修复）

- **依赖装通**：直连 PyPI 的 pip/uv 都被会话代理拖到超时；改用 **PowerShell + 清空会话代理 + `uv` 走清华 PyPI 镜像** 一次装成 `mcp 1.30.0`。`mcps/README.md` 补了这条可复现命令。
- **修复 `421 Invalid Host header`**：`FastMCP` 构造时 host 为 localhost 会自动启用 DNS rebinding 保护，Host 白名单只含 localhost；平台从 Docker 容器带 `Host: host.docker.internal:8766` 访问会被 uvicorn 拒绝。`code_review_mcp_server.py` 改为构造时用 `host="0.0.0.0"`，并在支持 `transport_security` 的 mcp 版本上把 `host.docker.internal:*` 加入白名单（**保留保护**）；旧版自动降级。
- **端到端实测**：服务起在 `0.0.0.0:8766`，宿主机与 `nexent-runtime` 容器访问 `/sse` 均返回 200；用真实 MCP 客户端握手 `initialize` + `list_tools` 得到 5 个工具（含 `scan_candidates`），实调 `scan_candidates` 返回 2 条候选与 `fingerprint`。

zip 与 `agent.json` 未改动（Skill 包和提示词里没有 Host 白名单信息）。

## v0.4.4（2026-11-15，导入向导占位符修复）

导入 `agent.json` 时向导第 2 步显示 `code-review-fs`「未安装」、URL 输入框为空。查证 `AgentImportWizard.tsx:53-58,701-715`：`needsConfig()` 把 `<TO_CONFIG:` 开头的字符串识别为「待配置」，于是 `isInstalled=false`、`editedUrl=""`，并把占位符里的文字当提示显示 —— 截图现象是设计行为，不是配置丢失。

但**不能在弹窗里手填 URL 点「安装」**：`handleInstallMcp` 调 `/mcp/add`，而 `remote_mcp_app.py:204` 对同名 MCP 返回 409；平台 `mcp_record_t` 里已有一条 `code-review-fs` / `http://host.docker.internal:8766/sse`。

修复：`agent.json` 的 `mcp_info.mcp_url` 与 `knowledge_base_search.params.index_names` 改为本机实测真实值，向导按 name+URL 即可匹配到已有记录、直接显示「已安装」。同步改 `生成_agent_json.py`（单一来源）、`README.md`、`平台配置清单.md`。提示词、Skill zip、MCP 服务代码未变。

真实值符合仓库惯例（现有条目导出文件也带本机 `index_names`），README 已注明导入到别的环境必须重新勾选知识库、重填 MCP URL。

## v0.4.5（2026-11-15，导入报错 `Agent import error.` 修复）

导入点「安装」后报 500，前端只显示兜底文案；后端日志给出真实异常：`Agent import error: Cannot find tool MCPTool in mcp.`。

根因链（两处）：

1. **配置包写错工具键**：`生成_agent_json.py` 给 5 个 MCP 工具写了 `class_name: "MCPTool"`、`usage: null`。导入时 `import_agent_by_agent_id` 按 `f"{class_name}&{source}"` 查 `ag_tool_info_t`，`validate_import_tool_params` 查不到就抛 `AgentToolImportError`。平台实际把 MCP 工具注册成 `class_name=工具名`、`usage=MCP 服务名`（已用 `get_tool_from_remote_mcp_server` 只读探测确认），仓库里现有 MCP agent 也是这个写法。已改。
2. **平台侧 MCP 未启用**：`mcp_record_t` 里 `code-review-fs` 的 `enabled=false`、`status` 空；`get_all_mcp_tools` 只扫 `enabled and status` 都为真的记录，所以 5 个工具从未写进 `ag_tool_info_t`。需要在 MCP 配置页打开「启用状态」并等工具列表刷新。

同步改 `生成_agent_json.py`（class_name / usage）、重新生成 `agent.json`、`README.md` 与 `平台配置清单.md` 补「启用 MCP + 刷新工具」前置步骤。提示词、Skill zip、MCP 服务代码、端口均未改动。

## v0.4.6（2026-11-15，同会话二次运行提前结束）

平台实测时，同一次对话里连跑第二个审查任务，流程在 step 3 就结束了。runtime 日志显示：

```
MODEL OUTPUT
继续执行：读取全部 diff 并加载审查规则技能。
Final answer: 继续执行：读取全部 diff 并加载审查规则技能。
```

模型把一句过程旁白当成了最终答案，既没输出 `<code>` 调工具、也没出报告，平台按 `Final answer` 收尾。触发条件是同一对话累积了 14 条历史消息（上一次完整报告还在上下文里），模型误以为在续写。

修复：`constraint_prompt` 加第 18 条硬约束——「每一轮必须二选一：输出 `<code>` 块调用工具，或输出完整审查报告；不得只写『继续执行…』这类过程说明就结束，平台会当作最终答案」。重新生成 `agent.json`（constraint 从 1518 字增至 1644 字）。

注意：平台里已导入的智能体仍是旧提示词，这条要重新导入或手动粘贴才生效；临时规避办法是**在新对话里跑**。Skill zip、MCP 服务代码、端口、工具键均未改动。

## v0.4.7（2026-11-15，C 组实测：平台连接层不降级）

C 组消融（停掉 8766）实测结果与设计预期不同：不是模型说「无法获取变更」并停止，而是平台直接抛 `Run Agent Error: unhandled errors in a TaskGroup (1 sub-exception)`，整个 agent run 终止，0 条审查结论。

说明：平台的 MCP 客户端在连接失败时于 TaskGroup 内抛异常，跑不到模型那一层。提示词里「工具不可用或返回错误时说明无法获取变更」这条，只对**工具返回 `{"error": "..."}`** 的情况生效（例如仓库路径不存在、越权路径被拒），对**连接层故障**无效。

结论不变：没有 MCP 就不会产生任何具体结论，也不会编行号；但表现形式是平台报错而非文字说明。已写入 `README.md`「已知限制」与 `验收表.md` C 组。

## v0.5.0（2026-11-15，平台导出覆盖 + 脱敏）

A/B/C 三组消融跑完后，从平台导出 `code_review_copilot.zip`（含 `agent.json` + `skills/code-review-rules.zip`），覆盖 `提交包/agent.json`：

- **来源变更**：由手工构造的配置模板改为**真实平台导出件**（草稿版 v0）。README「关于 agent.json 的来源」同步改写。
- **脱敏**：`tenant_id` 置空；`author` 已是 `Community Contributor`；复扫确认无 API Key、无邮箱、无本机绝对路径。
- **字段核对**：6 个工具（5 个 `source: mcp` + 1 个 `KnowledgeBaseSearchTool&local`）的 `class_name`/`usage` 与平台注册表一致；`skill_names=["code-review-rules"]`；`model_ids=[3]` + `model_names=["deepseek-v4-flash-0731"]`；`mcp_url` 与实跑一致；三段提示词 1421 / 1643 / 2753 字（constraint 含第 18 条）。
- **Skill 包一致性**：导出包内 `skills/code-review-rules.zip` 与本地 `skills/code-review-rules/` 明文包逐字节一致（4 个文件、28410 字节），确认平台原样保存、未改写。
- `生成_agent_json.py` 加警告：今后以平台导出为准，脚本仅用于无平台环境时重建模板，运行会覆盖导出件。

## v0.5.1（2026-11-17，实测结果回写）

只改文档，不改代码、zip、`agent.json`。消融 A/B/C 实测完成后批量回写过时表述：

- **`提交包/README.md`**：消融表 A/B 组「平台实测」列填入实测数字（此前仅 C 组有）；B 节标题与「如实说明」段改为实测完成状态；功能用例标题改为「#9 / #10 已实测，#1-#8 待复测」；向量模型描述修正——建库实际用 `BAAI/bge-m3` 且连通有效（此前误写为「旧稿，不用改」），`text-embedding-v4` 为另一可选配置；删除已知限制里「agent.json 为配置模板、平台实测与重新导出尚未完成」的过时条目（与 v0.5.0 矛盾）。
- **`工作文档/PR说明.md`**：提交前置条件警告改为「平台实测已完成」；三处 `[待填]` 换成实测数字；MCP 行「服务不可用时按提示词直接停止」改为与 C 组实测一致（连接层直接报错，见 README 已知限制）；「平台功能用例 8 条，实测结果见」改为如实表述（#1-#8 未逐条复测）；自检第 3 项勾选并注明覆盖范围。
- **`工作文档/验收表.md`**：引言与运行环境回填（模型 `deepseek-v4-flash-0731`、日期 2026-11-15）；第三节标题改「已实测」；删除排查表里与 v0.4.1 矛盾的「按 `##` 分块」残留。
- **`工作文档/自检报告.md`**：第三节加状态更新标注——该表为 2026-11-13 快照，「未执行」项已被第六~十三节推翻。
- **`工作文档/项目说明_v0.5.0.md`**：修正 Skill 行「缺了会怎样」——原写成「B 组实测」，实为设计推断且未实测（B 组关的是知识库；无「关 Skill」消融组），已如实标注。
- **`工作文档/平台配置清单.md`**：修正第 0 节与第 2 节残留的向量模型旧表述（同 README）；头部进度更新为「第 0~6 节已完成，仅剩第 7 节提交」；第 5 节 4 项核对勾选回填。
- **`工作文档/提交步骤.md`**：开头的「平台实测还没跑完，不要执行」警告改为前置条件已满足；阶段 0 的 5 项核对勾选回填。

功能用例 #1-#8 与 PR 提交仍待完成。

## v0.5.2（2026-11-26，PR #222 自动审视意见修复）

AtomGit AI 审视给出 7 条建议（阻塞 0 / 建议 7），逐条核对后全部处理：

- **`eval/init_fixture_repo.py`**：`shutil.rmtree(..., onexc=...)` 改为 `onerror=...`。`onexc` 是 Python 3.12 才有的参数，且回调签名按旧三元组解包，在 3.12+ 重建 fixture 时会报错。
- **`mcps/fs_tools.py`**：新增 `configure_allowed_roots()` 与 `ALLOWED_ROOTS`，`_root()` 拒绝白名单外的 `repo_path`；`_added_lines()` 修正 `+++` / `---` 前缀误判——diff 文件头是带空格的 `+++ b/...` / `--- a/...`，已在前面处理，源码行以 `+++` 开头时不能跳过。
- **`mcps/code_review_mcp_server.py`**：新增 `--allowed-root`（可重复）参数并在启动时调用配置函数；启动示例区分本机回环与 Docker 场景。
- **`mcps/README.md`**：启动命令默认改为 `127.0.0.1`，新增「安全边界」小节说明无鉴权服务不应暴露公网、`--allowed-root` 的用法与默认行为。Docker 接入仍用 `0.0.0.0`——这是 `host.docker.internal` 访问的实测必需配置，未照建议直接改掉。
- **`agent.json`**：few_shots 启动示例同步改为本机回环地址，并说明 Docker 场景需加 `--allowed-root`。
- **`knowledge/豁免清单.md`**：§3 的 `# noqa` 豁免限定为非生产路径，并写明与文末「生产路径一律不豁免」的优先级关系。
- **`skills/code-review-rules/SKILL.md`**：CR-002 判定特征补上 `.format()`，与《编码规范 v1.2》§3.2 及 `scan.py` 已有实现对齐（实现本就覆盖，仅规则表文字漏写）。zip 已重建，4 文件与明文一致。

未改：平台已导入的 agent（提示词仍在平台侧，不受交付包文字修订影响）；三组消融证据与功能用例状态。
