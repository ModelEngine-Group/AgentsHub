# 依赖与环境

本文记录项目复现所需的 Python 依赖分层、外部仓库版本、LLM 配置方式和 API 路径安全边界。

## Python 依赖分层

依赖文件按运行场景拆成三层，避免普通 CPU/CUDA 环境被 NPU 专用栈绑定：

| 文件 | 用途 | 关键依赖 |
| --- | --- | --- |
| `requirements.txt` | 通用依赖：运行任务一/二/三主链路，包含 REST API、Neo4j/LLM 可选增强、本地小模型训练与推理。适用于 CPU 或 CUDA GPU 环境。 | `pandas`, `fastapi`, `uvicorn`, `httpx`, `neo4j`, `openai`, `torch`, `transformers`, `peft`, `datasets`, `accelerate`, `safetensors` |
| `requirements-dev.txt` | 本地开发依赖：在通用依赖之上叠加测试和代码检查。 | `-r requirements.txt` + `pytest`, `ruff` |
| `requirements-npu.txt` | NPU 环境依赖：记录 Ascend 910B3 实测 torch/torch_npu 前置条件，并安装非 torch Python 依赖。torch 栈由 Ascend/CANN 环境或专用安装流程提供，不作为普通 pip requirements 强制安装项。 | 前置：CANN 8.5.0、`torch 2.9.0+cpu`、`torch_npu 2.9.0`（历史 910B2C 口径：CANN 8.5.1、`torch_npu 2.9.0rc1`）；pip：`transformers`, `peft`, `datasets`, `accelerate`, `safetensors` 和非 torch 通用依赖 |

安装命令：

```powershell
# 通用依赖（CPU / CUDA GPU）
python -m pip install -r requirements.txt

# 开发依赖（测试 + lint）
python -m pip install -r requirements-dev.txt

# NPU 依赖（Ascend；需先激活匹配的 CANN + torch_npu 运行时）
python -m pip install -r requirements-npu.txt
```

验证环境：Windows + NVIDIA RTX 5070 Laptop，CUDA 12.8，`torch 2.11.0+cu128`。基座模型 `Qwen/Qwen2.5-0.5B-Instruct` 经 ModelScope 下载；HuggingFace 不可达时可设置 `HF_ENDPOINT=https://hf-mirror.com` 或改用 `modelscope`。本地小模型微调与验证见 [本地小模型微调与验证](local_model_finetune.md)。

可选包按需安装：

- `bitsandbytes`：任务一 4-bit QLoRA，仅 CUDA 环境建议安装。
- `trl`：若安装则使用 `SFTTrainer`；缺失时自动回退 `transformers.Trainer`。

## 模型与 API 配置

- 仓库不保存模型 API key、token 或带鉴权信息的 base URL。
- 私有 LLM 测试应通过本地 shell 环境变量、Nexent 运行时 UI 或本地配置文件传入。
- 三个任务的演示程序和 API 服务统一使用 `src/common/llm_config.py` 加载配置。
- 本地小模型和 LLM API 属于同一可选增强层的两种实现：已有本地 adapter 时可用 `--local-model` 做离线规划、抽取或 NL2SQL；未准备 adapter 或希望在线增强时，可通过 `--llm-config` 读取 LLM API key 完成同类能力。两者都不是主链路必需项，缺失时回退规则或模板。
- 共享加载器支持两种本地配置格式：
  - `.env`：`OPENAI_API_KEY=...`、`OPENAI_BASE_URL=...`、`OPENAI_MODEL=...`、`OPENAI_TIMEOUT=...`、`OPENAI_MAX_TOKENS=...`
  - `.json`：`{"api_key": "...", "base_url": "...", "model_name": "...", "timeout": 120, "max_tokens": 2048}`，也支持 OpenAI 兼容的大写字段别名。
- DeepSeek V4 复现建议使用 `OPENAI_BASE_URL=https://api.deepseek.com`、`OPENAI_MODEL=deepseek-v4-flash`、`OPENAI_THINKING=disabled`；该字段会透传为 OpenAI-compatible `extra_body.thinking`，避免 reasoning token 消耗后返回空正文。
- `base_url` 必须是绝对 `http(s)` 地址，不能嵌入用户名或密码。
- 命令行和 API 服务统一通过 `--llm-config` 传入配置；任务一保留 `--env-file` 作为兼容别名。
- API server 中类似路径的字段会经过 `src/common/path_security.py` 校验，只允许访问项目工作区或系统临时目录内的路径。

## Nexent

- 来源：`ModelEngine-Group/nexent`
- 本地路径：`../nexent`
- 远端：`https://github.com/ModelEngine-Group/nexent.git`
- 版本：`1614aab67bfc85ca390f4c23d7df4474b2664802`
- 用途：Docker 部署、智能体框架参考、接口集成验证。
- 默认端口：Web **3000**（部署边界见 [preparation.md](preparation.md)）。
- 本项目触点：三个任务通过各自 `nexent_adapter.py` 导出 Nexent AgentConfig /
  ToolConfig 兼容结构；`src/common/nexent_online.py` 和
  `demos/dkm_online_integration.py` 对接官方 OpenAPI 服务、工具目录与 Agent 管理接口，
  不修改官方 Nexent 源码。

## DataMate

- 来源：`ModelEngine-Group/DataMate`
- 本地路径：`../DataMate`
- 远端：`https://gitlink.org.cn/modelengine-group/DataMate.git`
- 版本：`8f47ceb3ccc4a786b1dade5d78c3c8f48620d3cb`
- 用途：数据处理能力参考和任务一集成验证。
- 默认端口：API **18000**、Web UI **30000**（部署边界见 [preparation.md](preparation.md)）。
- 本项目触点：任务一通过 DataMate Python backend 的 `/api/health`、
  `/api/operators/list`、`/api/cleaning/templates` 和 `/api/cleaning/tasks` 接口进行
  探测、试运行和可选正式提交；写入后通过资源 ID 详情接口回查。
