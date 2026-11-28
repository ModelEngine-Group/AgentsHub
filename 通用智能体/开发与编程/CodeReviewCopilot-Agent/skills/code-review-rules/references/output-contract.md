# 发现项输出契约

## 1. JSON 契约（`scripts/scan.py` 的输出）

每个发现项是一个对象，六个字段全部必填，不缺项、不增项：

```json
{
  "rule_id": "CR-002",
  "severity": "blocker",
  "file": "services/payment.py",
  "line": 18,
  "snippet": "cursor.execute(f\"SELECT * FROM orders WHERE id = {user_id}\")",
  "message": "f-string 拼接用户输入进 SQL，存在注入风险"
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `rule_id` | string | `CR-001` ~ `CR-008` 之一 |
| `severity` | string | `blocker` / `major` / `minor`，由规则固定 |
| `file` | string | 相对仓库根的路径，用 `/` 分隔，不含盘符和绝对路径 |
| `line` | int | 1 起始的物理行号 |
| `snippet` | string | 触发行去掉首尾空白后的原文，超过 160 字符截断 |
| `message` | string | 一句话说明问题，不含修复建议 |

扫描器按顺序遍历文件、按行号升序输出，同一输入的结果可复现。

## 2. 报告表格契约（智能体输出）

固定六列，顺序不变：

```markdown
| 规则ID | 文件:行号 | 严重度 | 问题 | 依据 | 建议改法 |
| --- | --- | --- | --- | --- | --- |
| CR-001 | services/payment.py:7 | blocker | 生产密钥硬编码在源码中 | 《编码规范 v1.2》§4.1 | 改为读环境变量 |
```

| 列 | 来源 | 缺失时怎么办 |
| --- | --- | --- |
| 规则ID | Skill 规则表 | 不允许缺失 |
| 文件:行号 | MCP 读取结果 | 不允许缺失；读不到就整体停止 |
| 严重度 | Skill 规则表 | 不允许缺失 |
| 问题 | 规则表 `message` + 实际代码 | 不允许缺失 |
| 依据 | 知识库检索结果 | 写「未命中团队规范（通用实践）」+ 标注「依据待补」 |
| 建议改法 | Skill 规则表最小改法 | 不允许缺失 |

表格后必须跟一行统计：

```
统计：blocker 2 / major 3 / minor 3；已豁免 2 条。
```

## 3. 已豁免小节

命中豁免的候选**不进表格**，单独列在后面：

```markdown
已豁免（不计入问题数）：
- CR-005 候选 tests/test_login.py:14 调试打印 → 《豁免清单》§2.1
```

每条必须写清：规则ID、文件:行号、候选内容、豁免条款。只写「已忽略 2 条」不算合格。

## 4. 无发现项时的输出

```markdown
未发现符合规则的问题。

已检查范围：services/payment.py（HEAD~1..HEAD）。
已豁免（不计入问题数）：...
```

不许用「代码质量良好」「建议继续保持」这类无信息量的话代替。

## 5. 两个契约不要混用

- **脚本 JSON**（第 1 节）是候选发现项：`rule_id / severity / file / line / snippet / message`。它不含 `evidence`、`fix`。
- **报告表格**（第 2 节）才含「依据」和「建议改法」，由智能体在查完知识库、套用规则表最小改法后补齐。

不要把脚本的 `snippet` 当成报告里的「问题」原文照抄，也不要在脚本输出里要求 `evidence` 字段。

## 6. 条款号白名单

「依据」列的编号只允许：

| 文档 | 允许的编号 |
| --- | --- |
| 《编码规范 v1.2》 | §1.3、§2.3、§2.4、§2.5、§3.2、§3.4、§4.1、§4.2 |
| 《历史评审结论》 | R-2026-01 ~ R-2026-10 |
| 《豁免清单》 | §1、§2.1、§2.2、§2.3、§3、§4 |

《豁免清单》没有单独的「§2」。《编码规范 v1.2》没有 §2.1、§3.1 这类编号。白名单外的编号一律按编造处理，改写成「未命中团队规范（通用实践）」并标注「依据待补」。

## 7. 范围标注

- 用 `read_diff` 时只审 `+` 行，「已检查范围」写 `<base_ref>..HEAD 的 + 行`。
- 用 `read_file` 时写 `文件:起止行（全文审查，非 diff）`。
- `truncated=true` 时必须在报告开头声明覆盖不完整并列出实际检查的文件，不能写「已检查全部改动」。

## 8. `scan_candidates` MCP 工具输出契约

外层固定七字段：

```json
{
  "scope": "added_lines",
  "file_count": 6,
  "candidate_count": 10,
  "findings": [],
  "excluded_files": [],
  "skipped_files": [],
  "note": "findings 只是候选，是否上报由知识库《豁免清单》决定"
}
```

`findings[]` 每一项 = 第 1 节的六字段 + 两个附加字段：

| 字段 | 说明 |
| --- | --- |
| `fingerprint` | `sha1(rule_id\|file\|line)[:12]`，同一位置同一规则恒定，供跨次审查比对 |
| `scope` | `added_lines`（传了 diff_text）/ `whole_files`（按文件或全仓库扫描） |

规则：

- `scope=added_lines` 时，`findings` 里的 `(file, line)` 必须全部出现在 diff 的 `+` 行中；删除行与状态为 D 的文件不产生候选。
- 读不到或违反只读白名单的文件进 `skipped_files`，**不静默忽略**；报告里要提一句。
- `candidate_count` 是候选数，不是最终上报数。豁免过滤后才得到上报数。
- 工具返回 `{"error": "..."}` 时智能体停止，不改用自己逐行判断兜底。

## 9. 机读报告（用户要求 JSON 时）

固定四段，不缺项、不增项：

```json
{
  "scope": "HEAD~1..HEAD 的 + 行",
  "summary": {"blocker": 2, "major": 3, "minor": 3, "exempted": 2, "reported": 8},
  "findings": [
    {
      "rule_id": "CR-001",
      "file": "services/payment.py",
      "line": 7,
      "severity": "blocker",
      "fingerprint": "57eb442beb09",
      "problem": "生产密钥硬编码在源码中",
      "evidence": ["《编码规范 v1.2》§4.1", "《历史评审结论》R-2026-03"],
      "fix": "API_KEY = os.environ[\"PAYMENT_API_KEY\"]"
    }
  ],
  "exempted": [
    {"rule_id": "CR-005", "file": "tests/test_login.py", "line": 7,
     "clause": "《豁免清单》§2.1"}
  ]
}
```

`evidence` 数组里的编号仍受第 6 节白名单约束；空依据用 `["未命中团队规范（通用实践）"]` 并在同项加 `"evidence_pending": true`。`summary.reported` 必须等于 `findings` 长度。

## 10. 多轮收窄

「只看 blocker」「重新审某个文件」「第 N 条为什么被豁免」这类追问，基于本轮已有的 `findings` / `exempted` 重新过滤或解释，不产生工具里没有的新候选。用户给出新的 `base_ref` 或新路径时才重新调 MCP。指认具体条目用 `fingerprint`，不要用「第三条」这种随过滤顺序变化的序号。

## 11. SARIF 2.1.0 离线导出

`scripts/to_sarif.py` 把候选发现项转成 OASIS SARIF 2.1.0，让结果能被 GitHub Code Scanning、VS Code SARIF Viewer、Azure DevOps 等通用工具消费。它**不进入智能体运行时**，报告表格仍是给人看的主输出。

| 本作品字段 | SARIF 字段 | 说明 |
| --- | --- | --- |
| `rule_id` | `results[].ruleId` + `tool.driver.rules[].id` | 只声明实际产生结果的规则 |
| `severity` | `results[].level` | `blocker→error`、`major→warning`、`minor→note`；同时保留在 `properties.severity` |
| `message` | `results[].message.text` | 一句话问题说明 |
| `file` / `line` | `locations[].physicalLocation.artifactLocation.uri` / `region.startLine` | 仓库相对路径 + 物理行号，供 PR 内联标注 |
| `fingerprint` | `results[].partialFingerprints.codeReviewRuleLine` | 跨运行稳定标识，用于去重与生命周期跟踪 |
| `snippet` / `scope` | `results[].properties.*` | 自定义扩展，不影响标准字段 |

命令：

```bash
python scripts/scan.py --root <dir> --json | python scripts/to_sarif.py > results.sarif
python scripts/to_sarif.py --input findings.json --output results.sarif   # 也可读 scan_candidates 的返回对象
```

输入可以是发现项数组，也可以是含 `findings` 数组的对象。输出固定 `version: "2.1.0"`，`runs[].properties.candidateCount` 等于 `results` 长度。

## 12. 扫描范围排除

`scan_candidates` 会主动跳过生成代码与第三方目录（`vendor/`、`third_party/`、`generated/`、`migrations/`、`snapshots/`、`*_pb2.py`、`*_generated.py` 等），并把它们列在 `excluded_files` 里，**不静默忽略**。这与「豁免」是两回事：排除的是根本不该进入扫描范围的文件，豁免是候选产生后由知识库《豁免清单》决定不上报。

`excluded_files` 和 `skipped_files` 的区别：前者是主动排除（带原因），后者是读不到或被只读白名单拒绝。报告里若 `excluded_files` 非空，应提一句覆盖范围，不要声称「已检查全部文件」。
