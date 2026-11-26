# 高等数学答疑 MCP 服务

只读服务，7 个工具，全部为纯函数：输入表达式字符串，输出结构化结果。不访问网络、不读写文件、不执行用户提交的代码。

## 依赖

验算内核（`math_qa_engine/`）只用标准库，可脱离 MCP 单独运行与测试。服务外壳需要：

```
mcp>=1.2.0
```

## 启动

```bash
pip install -r requirements.txt

python math_qa_mcp.py --selftest                        # 自检，不需要 mcp 包
python math_qa_mcp.py --transport stdio                 # stdio 接入
python math_qa_mcp.py --transport http --port 8770      # Streamable HTTP（Nexent 用这个）
python math_qa_mcp.py --transport sse  --port 8770      # 旧版 SSE
```

在 Nexent 中新增远程 MCP，名称 `higher_math_qa`，地址：

```
http://127.0.0.1:8770/mcp
```

`allowed_hosts` 按 `--port` 自动生成，改端口不需要改代码。若换成别的容器名部署，需要同步 `build_server()` 中主机名那一项，否则会被 DNS 重绑定保护拦下并返回 421。

## 工具一览

| 工具 | 输入要点 | 返回要点 |
| --- | --- | --- |
| `locate_topic` | `query`、`limit` | 匹配的知识点编号、所属章节、命中原因 |
| `eval_expr` | `expr`、`var`、`at` | 代入值；无定义或趋于无穷时返回 `null` |
| `check_limit` | `expr`、`var`、`point`、`direction` | `verdict`（exists/diverges/oscillates/one_sided_*/not_exists）、估计值、双侧采样 |
| `check_derivative` | `expr`、`var`、`point`、`order` | 导数值、误差估计、`one_sided`（可识别不可导点） |
| `check_integral` | `expr`、`var`、`a`、`b` | 积分值、误差、`verdict`（converged/unstable/diverges）、所用方法 |
| `compare_expressions` | `expr_a`、`expr_b`、`allow_constant` | `verdict`（equivalent/constant_offset/opposite_sign/different/incomparable） |
| `render_tutor_card` | `payload` | 五段式 Markdown 卡片 |

## 表达式语法

- 四则：`+ - * /`，乘方用 `^`（右结合），支持隐式乘法 `2x`、`(x+1)(x-1)`；
- 函数：`sin cos tan cot sec csc arcsin arccos arctan sinh cosh tanh exp ln log lg sqrt abs floor ceil sign max min`；
- 常数：`pi`、`e`、`inf`；
- 输入容错：全角括号、`×`、`÷`、`π`、`²` 等常见写法会被自动归一化。

不支持求和号、积分号、分段函数、矩阵。传入非法表达式会返回 `{"error": ..., "error_type": "syntax"}`，不会静默猜测。

## 数值方法与可信度

| 工具 | 方法 | 精度 |
| --- | --- | --- |
| `check_limit` | 单侧递减步长采样 + 收敛性判别，差分突增处截断（双精度噪声） | 通常 1e-6 |
| `check_derivative` | 中心差分 + Richardson 外推 | 约 1e-9 |
| `check_integral` | 自适应 16 点 Gauss–Legendre；端点无界时平方代换；无穷区间换元 | 通常 1e-10 |

所有返回都带 `evidence_type: 数值…（非证明）`。`check_integral` 在相对误差偏大时返回 `verdict: unstable`，用于提示「该反常积分可能发散，数值不可采信」。

## 性能

单次调用通常在毫秒级。`check_integral` 对强振荡被积函数可能触发较多细分，实测常规题目不超过 50 ms。
