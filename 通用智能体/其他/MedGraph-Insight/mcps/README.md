# MCP 服务说明 · MedGraph-Insight

本目录提供 MedGraph-Insight 的 **MCP 服务**，把「数据处理 → 医疗知识图谱 → 分析洞察」
全链路能力暴露为 6 个 MCP 工具，供 Nexent 平台或任意 MCP 客户端调用。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

无需 `pip install` 本作品本体：服务启动时会自动把同级的 `../code` 加入 `sys.path`。

## 2. 启动服务

### 接入 Nexent 平台（SSE）

```bash
python mgi_mcp_server.py --transport sse --host 0.0.0.0 --port 8765
```

启动后 MCP URL 为：

```text
http://<服务器地址>:8765/sse
```

在 Nexent「智能体开发 → 选择工具 → MCP 配置」中填入该地址，服务名使用 `medgraph-insight`。

### 本地 MCP 客户端（stdio）

```bash
python mgi_mcp_server.py
```

可用 MCP Inspector 连接调试：

```bash
npx @modelcontextprotocol/inspector python mgi_mcp_server.py
```

## 3. 工具清单

| 工具 | 对应赛题任务 | 说明 |
| --- | --- | --- |
| `clean_medical_texts` | 任务一 | DAG 编排 7 个清洗算子，返回拓扑序、逐节点数据质量指标（重复率/PII 泄漏率/HTML 残留率）与血缘 |
| `build_medical_kg` | 任务二 | 实体识别 → 关系抽取 → 三元组校验，构建并持久化医疗知识图谱 |
| `ask_medical_kg` | 任务二 | GraphRAG 检索问答，支持别名/英文/缩写链接与多跳推理，返回图谱证据 |
| `kg_overview` | 任务二 | 图谱概览：实体/关系总数与类型分布 |
| `query_kg_with_nl2sql` | 任务三 | 自然语言转 SQL 并执行，返回 SQL 与结果集 |
| `generate_bi_report` | 任务三 | 生成自包含 HTML 图文 BI 报告 |

## 4. 产物位置

首次调用会在 `../code/output/` 下生成：

- `medical_kg.json`：知识图谱
- `medical_kg.sqlite`：供 NL2SQL 查询的关系视图
- `analysis_report.html`：图文 BI 报告

图谱构建后会被缓存复用；语料变更时用 `build_medical_kg(rebuild=true)` 重建。

## 5. 注意事项

- **中文字体**：`generate_bi_report` 生成的图表若显示方框，请安装中文字体
  （Ubuntu：`sudo apt-get install -y fonts-noto-cjk`）。
- **端口暴露**：`--host 0.0.0.0` 会把服务暴露到所有网卡，且本服务**不带鉴权**。
  在公网环境请改绑内网地址、置于反向代理之后，或加装网关鉴权。
- **离线可用**：全部工具默认不依赖外部大模型或联网服务，便于复现。
