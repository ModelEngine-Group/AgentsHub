# 【医疗】医学文献检索助手

## 简介

医学文献检索助手是一个基于 PubMed API 的智能检索代理，专门帮助临床科研人员快速检索、筛选和提取医学文献。具备解析临床问题、构建专业检索式及获取文献详细信息的能力，能够提供精准的文献列表、核心结论摘要及相关文献扩展建议。

## 功能特性

### 1. PICO 临床问题解析
从用户输入的临床问题中提取结构化检索要素：
- **P**（疾病/人群）：目标患者群体
- **I**（干预措施）：诊断/治疗/暴露因素
- **C**（对照）：对照方案（可选）
- **O**（结局指标）：临床终点

### 2. 专业检索式构建
- 支持布尔逻辑运算（AND, OR, NOT）
- 支持 PubMed 字段标签（`[pt]`, `[Title]`, `[Author]` 等）
- 支持 MeSH 主题词检索
- 支持研究类型过滤（Meta-Analysis, RCT, Systematic Review 等）

### 3. 多维文献操作
| 工具 | 功能说明 |
|------|---------|
| `pubmed_search` | 检索 PubMed 数据库，支持时间限制、排序、格式选择 |
| `pubmed_get_details` | 根据 PMID 获取完整文献信息（摘要、作者、期刊、DOI、MeSH 词） |
| `pubmed_extract_info` | 定向提取特定字段（作者、摘要、关键词、DOI），更省 Token |
| `pubmed_find_related` | 查找相似文献或综述文章，扩展检索范围 |

### 4. 智能筛选排序
- 研究类型优先级：**Meta 分析 > RCT > 队列研究**
- 时间优先：近 3 年文献优先
- 最多展示 5 篇最相关文献

### 5. 结构化输出格式

```
### 检索策略
- 检索式：【布尔逻辑检索式】
- 时间范围：【】
- 返回数量：【】

### 文献列表（最多5篇）
1.【标题】
   - 期刊/年份：【】/【】
   - 研究类型：【】
   - 核心结论：【1句话】
   - PMID：【】| DOI：【】
   - 链接：https://pubmed.ncbi.nlm.nih.gov/【PMID】/

### 建议扩展
- 如需更全面结果，建议放宽：【】
```

## 技术架构

### 模型信息
- **主模型**: deepseek-ai/DeepSeek-V4-Flash
- **业务逻辑模型**: Qwen/Qwen3.5-397B-A17B

### 工具集
| 工具名称 | 功能说明 |
|---------|---------|
| `pubmed_search` | 检索 PubMed 数据库，支持 query、days_back、sort_by、format 等参数 |
| `pubmed_get_details` | 获取指定 PMID 的完整文献信息（单次最多 20 个 PMID） |
| `pubmed_extract_info` | 提取特定字段：basic_info、authors、abstract_summary、keywords、doi_link |
| `pubmed_find_related` | 查找相似文献（similar）或综述文章（reviews） |

### 外部集成
- **PubMed API**：通过 MCP 协议接入，实现实时文献检索
- **MCP 端点来源**：`https://www.modelscope.cn/mcp/servers/liueic/mcp-pubmed-llm-server`

## 使用示例

**示例 1：PICO 检索**

用户输入：
```
查找关于 SGLT2 抑制剂治疗射血分数保留的心力衰竭的最新研究
```

Agent 构建检索式 `(SGLT2 inhibitor OR empagliflozin) AND HFpEF`，限定近 1 年，返回 10 篇结果。筛选后展示 5 篇，包括 Circulation 的 RCT 研究（SGLT2 抑制剂降低 HFpEF 住院风险）和 NEJM 的临床试验（恩格列净改善生活质量）。

**示例 2：字段定向提取**

用户指定 PMID 35123456，Agent 使用 `pubmed_extract_info` 定向提取作者、摘要和 DOI 信息，无需拉取全文记录。

**示例 3：基于核心文献扩展**

用户基于 PMID 34567890 查找相关综述，Agent 调用 `pubmed_find_related` 找到 Nature Reviews 和 Lancet Oncology 上的综述文章，再拉取详情展示。

**示例 4：Meta 分析专项检索**

检索过去一年关于二甲双胍治疗 2 型糖尿病的 Meta 分析，构建检索式 `Metformin AND Type 2 Diabetes AND Meta-Analysis[pt]`，返回 15 篇并筛选展示。

## 工作流程

```
用户问题 → PICO 解析 → 构建检索式 → pubmed_search
    → pubmed_get_details（获取前10-15篇详情）
    → 筛选排序（Meta > RCT > 队列研究，近3年优先）
    → pubmed_find_related（扩展检索）
    → 结构化输出
```

## 文件结构

```
Industry/medical/pubmed_search_assistant/
├── agent.json                    # Agent 配置文件
└── README.md                     # 本文件
```

## 配置参数

| 参数 | 值 |
|------|-----|
| Agent ID | 2910 |
| 最大执行步骤 | 5 |
| 启用状态 | 是 |
| 作者 | admin@wmc.com |

## 核心原则

- **检索先行**：必须先执行 `pubmed_search`，禁止跳过初始检索直接获取详情
- **时间限定**：默认检索近 365 天内的文献
- **定向提取**：仅需特定字段时使用 `pubmed_extract_info` 替代全记录拉取
- **扩展有序**：`pubmed_find_related` 仅在识别核心文献后使用
- **筛选严谨**：按研究类型优先级筛选，最终展示不超过 5 篇
- **检索式规范**：必须使用布尔运算符及 PubMed 字段标签
