# 运行与接入说明

本文件说明如何把归档包接入一个新的 Nexent 环境。它不要求使用额外发布脚本。

## 运行 MCP

1. 准备 Python 3.10 或更高版本；
2. 在归档根目录执行 `python prepare_runtime_assets.py`；
3. 复制 `mcps/mediflow-mcp/.env.example` 为 `.env.runtime`，填写模型、MCP、Nexent 和 DataMate 地址；
4. 在 `mcps/mediflow-mcp/` 安装 `requirements.txt` 中的依赖；
5. 执行 `python mcp_server/server.py`，确认 `http://<host>:8900/mcp` 可被 Nexent 访问。

## 接入 DataMate

数据处理智能体使用 `operators/` 下的自定义算子。请在目标 DataMate 的算子管理界面上传对应目录，并完成算子注册；算子目录中的 `process.py` 和 `metadata.yml` 是运行所需文件。

## 接入 Nexent

1. 在 Nexent 中注册 MCP 地址 `http://<host>:8900/mcp`，服务名使用 `medical-ai`；
2. 扫描并确认工具已经出现；
3. 导入 `agents/数据处理智能体/agent.json`、`agents/知识图谱智能体/agent.json` 和 `agents/数据分析智能体/agent.json`；
4. 为每个智能体选择目标环境模型并确认工具绑定；
5. 如果需要基于原文的知识库问答，在 Nexent 中上传源文档、等待索引完成，再绑定知识库；
6. 保存并发布。

## 验证方式

可以在 Nexent 的开始问答页面分别发送以下问题：

- “请整理这批医疗文本并返回质量信息。”
- “请抽取这段资料中的疾病、症状和治疗关系。”
- “同时包含症状咳嗽和检查 X 线的疾病有哪些？”

如果需要直接检查服务，可确认 MCP 地址返回工具列表，并检查相应智能体是否产生了工具调用记录。
