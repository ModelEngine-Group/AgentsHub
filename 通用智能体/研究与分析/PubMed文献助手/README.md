# PubMed文献助手

## 简介

PubMed 文献助手是一个专业的学术文献检索助手，能够通过 PubMed 工具搜索学术文献，并通过深入分析总结解答用户的学术问题。所有返回的 URL 都会以 `[题目](URL)` 的形式呈现，方便用户一键跳转。

## 功能特性

### 1. 文献检索
- 支持 `pubmed_search` 进行综合检索
  - `days_back`：时间范围（默认 90 天，0 表示不限）
  - `max_results`：返回数量（1-100，默认 20）
  - `sort_by`：`relevance` / `date` / `pubdate`
  - `response_format`：`compact` / `standard` / `detailed`
- 支持 `pubmed_quick_search` 进行快速检索（默认 10 篇，最多 20 篇）

### 2. 详情获取
- 通过 `pubmed_get_details` 获取指定 PMID 的完整信息
- 支持 `include_full_text=True` 拿全文链接

### 3. 关键信息抽取
- 通过 `pubmed_extract_key_info` 提取摘要关键信息
- 摘要长度可控：500-6000 字符，默认 5000

### 4. 回答结构
- 标题以 `[题目](URL)` 链接形式呈现
- 包含作者、期刊/年份、核心摘要、PMID 等字段
- 给出"查看全文"链接

### 5. 智能工具选择
- 用户未明确指定条件时，优先使用 `pubmed_quick_search` 提升响应速度
- 需要深入分析时切换到 `pubmed_search` + `pubmed_get_details` + `pubmed_extract_key_info`

## 技术架构

### 模型信息
- **主模型**: Qwen/Qwen3-32B
- **业务逻辑模型**: Qwen/Qwen3-32B

### 工具集
需要先部署安装 pubmed MCP 服务

| 工具名称 | 功能说明 |
|---------|---------|
| `pubmed_quick_search` | 快速检索 PubMed 文献（默认 10 篇，最多 20 篇） |
| `pubmed_search` | 综合检索（1-100 篇，默认 20，支持时间范围、排序、格式） |
| `pubmed_get_details` | 获取指定 PMID 的完整文献信息（支持全文链接） |
| `pubmed_extract_key_info` | 抽取关键信息（摘要长度 500-6000 字符） |

### 外部集成
| 服务 | 用途 |
|------|------|
| **pubmed MCP** | PubMed 文献检索与解析引擎 |

## 工作流程

```
用户提出学术问题
  ↓
判断问题复杂度
  ├─ 简单检索 → pubmed_quick_search
  └─ 深入分析 → pubmed_search → pubmed_get_details → pubmed_extract_key_info
        ↓
整理文献条目
  ├─ 标题 → [题目](URL)
  ├─ 作者、期刊/年份
  ├─ 核心摘要
  └─ 查看全文 / PMID
        ↓
输出综合回答
```

## 使用示例

**示例 1：检索最新研究**

> 查找最近 3 个月关于"阿尔茨海默病治疗"的最新研究论文

Agent 调用 `pubmed_search(days_back=90, sort_by="date", max_results=15)` 检索，输出包含链接、作者、期刊、摘要的列表。

**示例 2：获取指定文献详情**

> 获取 PMID 为 12345678 的论文详细信息

Agent 调用 `pubmed_get_details(pmids="12345678", include_full_text=True)`，输出标题、作者、通讯作者、期刊信息、完整摘要、关键词、PubMed 页面/全文 PDF/DOI 等多链接。

**示例 3：快速粗筛**

> 快速查找 5 篇关于"糖尿病预防"的论文

Agent 调用 `pubmed_quick_search(query="糖尿病 预防", max_results=5)`，输出简要信息列表。

## 文件结构

```
General/research_and_analysis/pubmed_paper_assistant/
├── agent.json                    # Agent 配置文件
└── README.md                     # 本文件
```

## 配置参数

| 参数 | 值 |
|------|-----|
| Agent ID | 1985 |
| 最大执行步骤 | 7 |
| 启用状态 | 是 |
| 作者 | Nexent |

## 核心原则

- **链接必留**：所有结果中的 URL 完整保留并以 `[题目](URL)` 形式呈现
- **工具分级**：根据需要选择 `quick_search` 或深度检索组合
- **参数严格**：严格遵守各工具的参数约束（数量、长度、排序、格式）
- **响应规范**：返回的 PMID、作者、期刊、摘要等字段要准确
- **优先速度**：未明确指定条件时优先使用快速检索
- **可溯源**：每篇文献都给出 PubMed 页面 / DOI 等可追溯入口