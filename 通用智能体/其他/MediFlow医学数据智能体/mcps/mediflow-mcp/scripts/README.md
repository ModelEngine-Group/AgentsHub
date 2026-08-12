# 运行辅助模块

本目录只保留 MCP 运行时需要的环境变量读取模块：

| 文件 | 作用 |
| --- | --- |
| [`runtime_env.py`](runtime_env.py) | 读取 `.env.runtime` 和进程环境变量，供数据处理流程获取路径与凭据 |
| [`__init__.py`](__init__.py) | Python 包标记文件 |

MCP 注册、智能体创建、模型选择和知识库绑定均在 Nexent 界面完成，不需要运行发布脚本。DataMate 算子也通过 DataMate 的算子管理功能接入。
