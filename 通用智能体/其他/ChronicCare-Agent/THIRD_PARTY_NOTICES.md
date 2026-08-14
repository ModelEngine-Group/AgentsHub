# 第三方组件声明

本文件记录当前版本的直接运行依赖及外部集成边界，用于组件归属和许可证核对，不替代各上游项目的完整许可证文本。

| 组件 | 固定版本或边界 | 许可证 | 上游来源 | 用途 |
| --- | --- | --- | --- | --- |
| Nexent | `v2.2.0` / `a57538fb68e23d4e6b5439c3f13add014edbfd7e` | MIT | [GitHub](https://github.com/ModelEngine-Group/nexent) | Agent前端与编排平台 |
| DataMate | `6136834ce00075f0a844e26dcc7fe1cc9e0d8dd9` | MIT | [GitHub](https://github.com/ModelEngine-Group/DataMate) | Mapper算子运行时 |
| pandas | 2.3.3 | BSD-3-Clause | [PyPI](https://pypi.org/project/pandas/2.3.3/) | 表格数据处理 |
| PyYAML | 6.0.3 | MIT | [PyPI](https://pypi.org/project/PyYAML/6.0.3/) | 配置解析 |
| NetworkX | 3.6.1 | BSD-3-Clause | [PyPI](https://pypi.org/project/networkx/3.6.1/) | 图数据处理 |
| PyVis | 0.3.2 | BSD-3-Clause | [PyPI](https://pypi.org/project/pyvis/0.3.2/) | 图谱可视化 |
| Plotly | 5.24.1 | MIT | [PyPI](https://pypi.org/project/plotly/5.24.1/) | 图表生成 |
| Streamlit | 1.58.0 | Apache-2.0 | [PyPI](https://pypi.org/project/streamlit/1.58.0/) | 可选辅助Dashboard |
| FastAPI | 0.124.4 | MIT | [PyPI](https://pypi.org/project/fastapi/0.124.4/) | HTTP API |
| Uvicorn | 0.38.0 | BSD-3-Clause | [PyPI](https://pypi.org/project/uvicorn/0.38.0/) | ASGI服务 |
| Pydantic | 2.12.5 | MIT | [PyPI](https://pypi.org/project/pydantic/2.12.5/) | 数据结构与校验 |
| Pillow | 12.1.1 | HPND | [PyPI](https://pypi.org/project/pillow/12.1.1/) | PNG图表及图谱预览渲染 |
| Requests | 2.32.5 | Apache-2.0 | [PyPI](https://pypi.org/project/requests/2.32.5/) | HTTP客户端 |
| HTTPX | 0.28.1 | BSD-3-Clause | [PyPI](https://pypi.org/project/httpx/0.28.1/) | HTTP客户端 |
| SQLGlot | 30.12.0 | MIT | [PyPI](https://pypi.org/project/sqlglot/30.12.0/) | SQL解析与安全校验 |
| MCP Python SDK | 1.26.0 | MIT | [PyPI](https://pypi.org/project/mcp/1.26.0/) | MCP适配器 |
| pytest | 8.4.2 | MIT | [PyPI](https://pypi.org/project/pytest/8.4.2/) | 自动化测试 |
| pytest-cov | 7.0.0 | MIT | [PyPI](https://pypi.org/project/pytest-cov/7.0.0/) | 测试覆盖率采集 |
| Ruff | 0.14.14 | MIT | [PyPI](https://pypi.org/project/ruff/0.14.14/) | Python静态检查 |

## 未随交付包分发的组件

- [`bge-small-zh-v1.5`](https://huggingface.co/BAAI/bge-small-zh-v1.5)模型权重：MIT License；不包含在交付包中，部署和再分发仍须保留上游许可与来源信息。
- Ascend驱动、CANN、PyTorch、`torch_npu`、Transformers及相关NPU运行时：由目标DataMate环境提供，版本须与目标Ascend软件栈匹配。
- 可选[`deepseek-chat`](https://platform.deepseek.com/)接口：仅在显式启用时接收自然语言问题和白名单Schema约束，遵循服务条款与隐私政策。
- ChronicCare数据：参考真实慢病随访业务的字段结构、业务关系和合理取值范围，采用程序化规则与固定随机种子合成，不包含可识别的真实患者身份信息。

