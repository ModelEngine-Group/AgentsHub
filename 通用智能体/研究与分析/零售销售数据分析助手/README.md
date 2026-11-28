# 零售销售数据分析助手

在 Nexent 中查询模拟零售订单的净销售额及品类排名。首版支持一个自然月和可选的一个品类。只给月份时按 2026 年解释。样例覆盖 2026 年 10 月至 12 月。

## 启动 MCP

在本目录操作。Nexent 通过 Docker 部署时，将 MCP 容器接入 Nexent 使用的 Compose 网络。选择配置、运行和数据处理服务共用的网络，替换命令中的 `<nexent-network>`。如果部署使用了其他容器名，以 `docker ps` 所示名称为准。

```powershell
docker inspect nexent-config nexent-mcp nexent-runtime nexent-data-process --format '{{.Name}} {{range $name, $value := .NetworkSettings.Networks}}{{$name}} {{end}}'
docker build -f mcps/Dockerfile -t retail-sales-mcp:1.0 .
docker run -d --restart unless-stopped --name retail-sales-mcp --network <nexent-network> retail-sales-mcp:1.0
```

Nexent 的 MCP 配置使用名称 `retail-sales`、传输 `Streamable HTTP`、地址 `http://retail-sales-mcp:8765/mcp`。同步工具后应出现 `summarize_retail_sales`。容器无需发布宿主机端口。

如果 Nexent 的后端和 MCP 都直接运行在同一主机，可以使用 Python 3.10+ 启动服务，地址改为 `http://127.0.0.1:8765/mcp`。

```powershell
python -m pip install -r mcps/requirements.txt
python mcps/server.py
```

Windows 上也可以让 Docker Desktop 中的 Nexent 调用宿主机进程。在启动服务前执行 `$env:MCP_HOST = '0.0.0.0'`，把 Nexent 中的 MCP 地址设为 `http://host.docker.internal:8765/mcp`。保持该进程运行。这个服务仅用于本地模拟数据演示。

## 配置 Nexent

1. 登录 Nexent，在模型管理中测试一个对话模型，确认文本 Embedding 模型可用。没有现成 Embedding 时，可添加 OpenAI API 兼容的 `retail-demo-text-384`。服务地址取 MCP 地址的同一主机和端口，路径改为 `/v1/embeddings`，API Key 字段填 `local-demo`。这个演示端点使用 384 维字符特征哈希做词面检索。
2. 创建知识库「零售销售指标口径」，上传 `knowledge/retail-metrics.md`，等待解析和索引成功，并试搜「净销售额」。如文件处理服务不可用，可在该知识库把文档全文保存为一条手工知识分块，再测试检索。`data/orders.csv` 随 MCP 服务部署。
3. 在技能管理中上传 `skills/retail-sales-analysis/SKILL.md`。注册 MCP 并同步工具。
4. 在「智能体开发」导入 `agent.json`。选择目标环境的对话模型，确认 `summarize_retail_sales` 和 `retail-sales-analysis` 已启用，把 `knowledge_base_search` 指向刚创建的知识库，保存并调试。如果导入提示缺少工具，先在 MCP 配置中同步工具列表，再重试。

## 验收

在调试页输入「11 月各品类的净销售额是多少？哪个品类最高？」并检查运行记录。应能看到知识库检索和 MCP 调用。2026 年 11 月共 9 条有效记录，已完成 ¥2,441.00，已退款 ¥340.25，净销售额 **¥2,100.75**。

| 排名 | 品类 | 已完成 | 已退款 | 净销售额 |
| --- | --- | ---: | ---: | ---: |
| 1 | 数码 | 1,420.00 | 200.00 | 1,220.00 |
| 2 | 家居 | 570.50 | 90.00 | 480.50 |
| 3 | 食品 | 450.50 | 50.25 | 400.25 |

查询 2026 年 11 月食品应只返回 ¥400.25。查询 2026 年 9 月或 11 月服饰时应提示无可统计数据。服饰只有取消订单。

## 导出与 PR

本目录的 `agent.json` 来自 Nexent 导出。同租户使用已有依赖已完成重新导入、资源绑定和示例复测。在另一套环境复现时，按上述步骤部署依赖，再将配置绑定到该环境的模型、MCP 和知识库。

在 AgentsHub 的 `通用智能体/研究与分析/零售销售数据分析助手/` 提交 `agent.json`、本说明、样例数据、指标文档和自定义依赖。PR 标题为 `[new agent] 零售销售数据分析助手`。
