# 高中数学 MCP 服务

服务通过 Streamable HTTP 提供两项只读工具。 `quadratic_check` 接收整数系数并核对二次式的根、顶点和判别式。 `fraction_check` 核对分数四则运算。工具只接受结构化参数。

## Docker 启动示例

下例在已经部署 Nexent 的主机上运行，复用可访问的 Nexent v2.7.0 数据处理镜像。进入本目录，将 `nexent_network` 替换成 Nexent 配置与运行服务所在的实际 Docker 网络。

```powershell
docker run -d --name nexent-math-course --network nexent_network --restart unless-stopped --memory 1g --cpus 2 --mount "type=bind,source=$((Get-Location).Path),target=/course,readonly" --workdir /course --entrypoint python ccr.ccs.tencentyun.com/nexent-hub/nexent-data-process:v2.7.0 -B -m uvicorn server:app --host 0.0.0.0 --port 8765
```

在 Nexent 中新增远程 MCP `high_school_math`，服务器地址设置为 `http://nexent-math-course:8765/mcp`。用连通性检查确认两个工具名称，启用服务后再将工具绑定到智能体。若修改容器名称或端口，同步修改 `server.py` 的允许主机、测试地址和 Nexent 配置。示例容器端口只在 Docker 网络内可见。

## 可选本地嵌入

若已配置文本向量模型，可直接使用现有模型。需要复现本地嵌入时，从 [模型发布页](https://huggingface.co/BAAI/bge-small-zh-v1.5) 获取 Safetensors 权重、配置和 tokenizer 文件，把 `config.json`、`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`、`vocab.txt` 与 `model.safetensors` 放入 `model/`。本次使用的权重 SHA-256 为 `354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026`。

在 Nexent 模型管理添加 OpenAI 兼容文本 Embedding，模型名 `bge-small-zh-v1.5`，基础地址 `http://nexent-math-course:8765/v1`。测试成功后可创建 512 维文本索引。数学 MCP 本身可以独立于嵌入权重运行。

`verify-mcp.py` 从可访问此 Docker 网络的环境运行，要求 Python 安装 FastMCP 客户端。预期二次式根为 2、3，顶点纵坐标为 -1/4，分数算例结果为 3/10。
