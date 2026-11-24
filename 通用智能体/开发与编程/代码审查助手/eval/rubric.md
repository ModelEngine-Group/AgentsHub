# 验收 rubric

## 怎么跑

1. 在 Nexent 里打开「代码审查助手」，把 `eval/cases/*.diff` 逐个粘贴（或上传）。
2. 每次要求：「输出 JSON + Markdown，不要回帖」。
3. 保存 Agent 输出的 JSON 为 `eval/out/<case-id>.json`。
4. 对每个输出运行：

```bash
python skills/code-review/scripts/review_diff.py \
  --mode verify --diff eval/cases/<case-id>.diff \
  --findings eval/out/<case-id>.json
```

`invalid` 必须为 0。

## 指标

| 指标 | 定义 | v0.1 目标 |
|---|---|---|
| 召回率 | 命中的 expected 条数 / expected 总条数 | ≥ 70% |
| 精确率 | 人工判定有效的条数 / 全部 findings | ≥ 70% |
| 引用准确率 | verify 通过的行号 / 全部行号 | 100% |
| 噪声 | P3 逐条展开数 + 空泛评论数 | ≤ 2 / 次 |
| 安全底线 | 安全类用例的 P0/P1 是否漏检 | 0 漏检 |

命中判定：同一文件、行号 ±1、级别相差 ≤1 级、类别一致，且 `must_mention` 至少命中 2 个关键词。

## 单条判定

一条意见算「有效」，必须同时满足：

- 指向 diff 中的真实变更行；
- 能说出具体失败场景（输入/并发/环境 → 后果）；
- 修复建议可执行；
- 不是纯风格说教。

否则记为误报。**漏报不扣精确率，但会拉低召回率。**

## 记录表

| 用例 | expected | 命中 | 误报 | invalid | 结论 |
|---|---:|---:|---:|---:|---|
| 01_sql_injection | 1 |  |  |  |  |
| 02_mutable_default | 2 |  |  |  |  |
| 03_clean | 0 |  |  |  |  |
| 04_secret_and_debug | 2 |  |  |  |  |
| 05_bare_except_sql | 2 |  |  |  |  |

## 常见失败模式

| 现象 | 原因 | 修法 |
|---|---|---|
| 行号偏 1–2 行 | 模型自己数行 | 强制用 parse 输出，输出前 verify |
| 把 `print` 标成 P1，把注入标成 P2 | 分级标准不清 | 在约束提示词里固定级别表 |
| 每个文件都提「加测试」 | 没有具体对象 | 反噪声规则：必须指出缺哪个用例 |
| 对未变更代码提意见 | 读了整文件 | 只允许引用 diff 行；verify 兜底 |
| 密钥被完整复述 | 模型忽略打码 | 约束 + 脚本 evidence 已打码；提示词再强调 |
| 无问题时硬凑 | 缺少「无问题」示例 | few-shots 里放一个 approve 示例 |
