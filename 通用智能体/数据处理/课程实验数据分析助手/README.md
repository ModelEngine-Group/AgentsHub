# 课程实验数据分析助手

## 简介

面向**上实验课的大学生**（大多没有统计学背景，用中文提问）的数据分析助手。

用户上传一份 **UTF-8 编码的 CSV**，它把文件变成可追问的分析结论：数据概况 → 数据质量检查 → 筛选与分组统计 → 趋势比较 → 图表（HTML）→ 分析简报。

输出遵循固定顺序：**先给结论，再给表格或图表，最后说明筛选条件、有效样本数与计算口径**，并支持对同一张表连续追问。

## 设计取向

学生实验数据最常见的坑不是「算不出来」，而是**算错了还理直气壮**。所以这个智能体在两个地方做了额外约束：

1. **所有数值必须基于文件实际计算**，不允许凭印象或估算（提示词里写死）。
2. **每个结论都要交代口径** —— 筛选条件、有效样本数、缺失去向，让结论可复核。

`metric-calculation-audit` 与 `metric-anomaly-diagnosis` 两个技能专门用来复核指标计算与异常归因。

## 功能特性

### 14 个技能，按用途分四组

**① 输入与清洗**

| 技能 | 作用 |
|---|---|
| `table-health-check-and-cleaning` | 缺失值、重复记录、字段类型、异常值检查与清洗建议 |
| `analyze-image` | 识别截图或图片中的表格/图表内容 |

**② 查询与统计**

| 技能 | 作用 |
|---|---|
| `nl-to-analysis-sql` | 把中文问题转成对当前 CSV 的筛选/分组统计 |

**③ 可视化**

| 技能 | 作用 |
|---|---|
| `data-visualization` | 通用图表生成（HTML） |
| `trend-anomaly-visualization` | 时序趋势与突变点标注 |
| `grouped-comparison-viz` | 分组对比（实验组 vs 对照组） |
| `chart-review-redraw` | 图表审查与重绘 |
| `chart-data-report` | 图表配套的数据说明 |

**④ 指标可信度**

| 技能 | 作用 |
|---|---|
| `metric-calculation-audit` | 复核指标计算口径 |
| `metric-anomaly-diagnosis` | 异常值归因 |

**⑤ 产出文档**

| 技能 | 作用 |
|---|---|
| `docx` / `pdf` / `xlsx` | 生成 Word / PDF / Excel 产物 |
| `skill-creator` | 按需创建新技能 |

> `docx`、`pdf`、`xlsx`、`skill-creator`、`analyze-image` 属于平台常用技能，本仓库仍一并提供 zip，方便目标环境直接上传。

### 工具

| 工具 | 说明 |
|---|---|
| `knowledge_base_search` | 检索数据字典、指标口径、缺失值规则、实验说明 |
| `analyze_text_file` | 读取用户上传的文本文件 |
| `create_file` / `create_directory` / `delete_file` / `delete_directory` | 平台本地文件工具，用于生成图表与报告产物 |

无自定义 MCP、无子智能体。

## 技术架构

- **模型**：`deepseek-flash`
- **最大执行步数**：15
- **变量名**：`course_lab_data_analyst`
- **知识库**：必需（提供数据字典与指标口径），见下方导入顺序

## 使用示例

用户输入：

```
帮我检查这份实验数据有没有问题，然后按实验组算一下平均值和标准差。
```

智能体流程：

1. `analyze_text_file` 读取 CSV
2. `table-health-check-and-cleaning` 统计缺失、重复、类型异常
3. `nl-to-analysis-sql` 按实验组分组统计
4. `metric-calculation-audit` 复核口径（用 n 还是 n-1、缺失是否剔除）
5. 输出：结论 → 分组统计表 → 有效样本数与计算口径说明

## 文件结构

```
课程实验数据分析助手/
├── agent.json                    # Nexent 平台导出的智能体配置
├── README.md                     # 本文件
└── skills/                       # 14 个技能包（可直接上传平台）
    ├── table-health-check-and-cleaning.zip
    ├── analyze-image.zip
    ├── nl-to-analysis-sql.zip
    ├── data-visualization.zip
    ├── trend-anomaly-visualization.zip
    ├── grouped-comparison-viz.zip
    ├── chart-review-redraw.zip
    ├── chart-data-report.zip
    ├── metric-calculation-audit.zip
    ├── metric-anomaly-diagnosis.zip
    ├── docx.zip
    ├── pdf.zip
    ├── xlsx.zip
    └── skill-creator.zip
```

## 配置参数

| 参数 | 值 |
|---|---|
| 显示名 | 数据分析助手 |
| 变量名 | `course_lab_data_analyst` |
| 最大执行步数 | 15 |
| 绑定技能数 | 14 |
| 工具数 | 6 |
| 作者 | Akutagawa2 |

## 导入方式

**顺序不能换**（技能要先装，否则导入时找不到）：

1. **先装技能**：智能体开发 → 选择技能 → **构建技能**，依次上传 `skills/` 下 14 个 `.zip`
2. **再建知识库**：准备数据字典 / 指标口径 / 缺失值规则文档并建库，或先导入后重新绑定
3. **导入智能体**：智能体开发 → 导入 → 选 `agent.json`，按提示绑定模型与知识库
4. **验证**：上传一份 UTF-8 CSV，问「这份数据有多少行、哪些字段」

## 本提交相对平台导出原件的改动

为了让仓库可公开使用，对 `agent.json` 做了 4 处必要处理（其余结构、提示词、技能绑定与原导出一致）：

| 改动 | 原因 |
|---|---|
| `author` 由个人邮箱改为 `Akutagawa2` | 避免个人邮箱在公开仓库长期暴露 |
| 移除 `mcp_info` 中的 `outer-apis` 条目 | 该 MCP 的 `mcp_url` 为空字符串，是本地调试残留，导入时会失败 |
| 移除工具 `local_test_tool_name` | 调试期的占位工具，无实际实现 |
| 从 `skill_names` 移除 `destination-card-retrieval` | 该技能属于另一智能体，误绑定，已一并移除其 zip |

## 已知限制

- **仅支持单个 UTF-8 编码 CSV**。GBK/Excel 另存的 CSV 需先转码，否则中文列名会乱码。
- **依赖知识库**。未绑定数据字典时，字段含义与指标口径只能靠字段名推测，结论可靠性下降。
- **`agent.json` 的描述文字仍提到「由 MCP 工具计算」**。原环境该 MCP 为空配置（见上表），本提交已移除；实际计算依赖平台本地文件工具与技能脚本，描述文字未改写以保持导出原件一致。
- 面向教学场景，**不替代正式统计分析结论**；显著性检验、样本量设计等请用专业统计工具。
- 生成结果会写入平台本地文件系统，清理时机由平台策略决定。
