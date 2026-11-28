# 评审样例与复现方式

`eval/samples/` 有 5 个**故意含缺陷**的 Python 文件和 1 个**干净负样本**，用来把「能不能发现问题、会不会凑数」变成可以数的数字。全部为本作品原创编写，不是从任何真实项目复制的代码。

## 一、离线复现（不需要 Nexent）

在本作品目录下执行：

```bash
python skills/code-review-rules/scripts/scan.py --root eval/samples --json
python skills/code-review-rules/scripts/scan.py --root eval/samples/utils/clean.py
```

不加 `--json` 输出人读表格。扫描器只用标准库，不需要装 `mcp`。**平台运行时用的是同一份脚本**：MCP 的 `scan_candidates` 工具会 import 它并做确定性扫描；这里直接跑是离线校验，两者规则完全一致，不存在第二套实现。

同目录还有 `to_sarif.py`，把结果转成 OASIS SARIF 2.1.0 供 CI 消费（离线导出，不进运行时）：

```bash
python skills/code-review-rules/scripts/scan.py --root eval/samples --json \
  | python skills/code-review-rules/scripts/to_sarif.py > results.sarif
```

## 二、样例清单

| 文件 | 植入的缺陷 | 期望规则 | 上报? |
| --- | --- | --- | --- |
| `services/payment.py` | 硬编码 `sk-live-...` 密钥；f-string 拼 SQL | CR-001 / CR-002 | 是 |
| `services/sync.py` | `open` 未用 `with`；`except Exception: pass` | CR-004 / CR-003 | 是 |
| `utils/http.py` | `requests.get` 无 `timeout`；`items=[]` 默认参数 | CR-008 / CR-006 | 是 |
| `utils/validate.py` | `assert` 校验入参；日志打印 password | CR-007 / CR-005 | 是 |
| `tests/test_login.py` | 调试打印 token；测试断言 | CR-005 / CR-007 | **否，应豁免** |
| `utils/clean.py` | 无 | 无 | **否，干净负样本** |

合计 10 条候选 → 8 条上报 + 2 条豁免。`utils/clean.py` 单独跑得到 0 条。

## 三、三个探针

| 探针 | 验证什么 | 失败表现 |
| --- | --- | --- |
| `tests/test_login.py` 的 2 条 | 知识库《豁免清单》是否真的被检索 | 关掉知识库后上报数 8 → 10 |
| `utils/clean.py` | 「没有发现项时不凑数」约束 | 凭空报出格式、命名之类的主观意见 |
| `legacy.py`（仅 fixture 基线） | 「只审 `+` 行 / 跳过 D 状态文件」约束 | 把已删除的旧代码里那把密钥报成问题 |

`tests/test_login.py` 那两条之所以一定出现在离线原始输出里，是因为扫描器按设计**不做路径豁免**；只有智能体查了《豁免清单》§2.1、§2.2 才会把它们移到「已豁免」。

## 四、平台验收必须先建 fixture 仓库

`eval/samples/` 是散文件，不是 git 仓库，`read_diff` 在这里会返回 error。若直接拿它当验收输入，A/B/C 三组消融只会得到「无法获取变更」，核心证据根本跑不出来。

```bash
python eval/init_fixture_repo.py                    # 首次生成
python eval/init_fixture_repo.py --recreate        # 改过样例后重建（仅限脚本自己生成的目录）
```

脚本做两次提交：

- **基线**：6 个文件只有一句 docstring，另有 `legacy.py` 含硬编码密钥
- **缺陷提交**：写入 5 个带缺陷样例 + `clean.py`，删除 `legacy.py`

于是 `HEAD~1..HEAD` 的 `+` 行候选恰好 10 条，与散文件扫描一致；同时 `legacy.py` 状态为 D、diff 全是 `-` 行，专门测「删除行不报」。脚本会打印验收表该用的仓库路径。

> 默认生成在 `工作文档/_fixture_repo`，是本地验收产物，**不要提交进 PR**。目录带 `.code_review_fixture` 标记；不带标记的目录即使加 `--recreate` 也不会被删。

## 五、本地实测输出（2026-11-13，Python 3.13.14）

```
CR-001 | services/payment.py:7  | blocker | 凭据被硬编码为字符串字面量，随代码分发即泄漏
CR-002 | services/payment.py:11 | blocker | SQL 语句由字符串拼接或格式化生成，存在注入风险
CR-004 | services/sync.py:7     | major   | 文件句柄未纳入 with 上下文管理，异常路径下可能泄漏
CR-003 | services/sync.py:15    | major   | 裸 except 或异常被静默吞掉，故障无法定位
CR-005 | tests/test_login.py:7  | major   | 日志或打印语句输出了敏感字段
CR-007 | tests/test_login.py:9  | minor   | 用 assert 校验外部输入，python -O 下校验会被移除
CR-008 | utils/http.py:7        | minor   | HTTP 请求未设置超时，可能长期挂起
CR-006 | utils/http.py:10       | minor   | 函数默认值是可变对象，跨调用会共享状态
CR-007 | utils/validate.py:9    | minor   | 用 assert 校验外部输入，python -O 下校验会被移除
CR-005 | utils/validate.py:10   | major   | 日志或打印语句输出了敏感字段
候选发现项合计：10
```

fixture 仓库生成后同样跑出 10 条；`utils/clean.py` 单独跑是「候选发现项合计：0」。

## 六、边界说明

- 样例里的 `sk-live-8f3a92c1d7e64b05` 与 `legacy.py` 里的 `sk-live-legacy0000000` 都是随机编造的字符串，不是可用密钥。
- `init_fixture_repo.py` 需要本机有 `git`；`git config` 只写这个仓库的 `.git/config`，不改全局配置。
- Windows 上 git object 文件带只读属性，脚本的 `--recreate` 已处理，不会因权限中断。
- `scan_candidates` 按相对路径加载 `skills/code-review-rules/scripts/scan.py`，所以 `mcps/` 与 `skills/` 必须同属一个 agent 目录；找不到时工具返回明确 error，不会静默退化。
