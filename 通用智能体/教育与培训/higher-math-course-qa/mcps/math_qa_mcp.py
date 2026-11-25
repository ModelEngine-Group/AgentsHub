#!/usr/bin/env python3
"""高等数学课堂答疑助手 · MCP 服务。

只读服务：不写文件、不执行用户代码、不访问网络。所有工具都是纯函数，
输入表达式字符串，输出可直接嵌进答疑卡片的结构化结果。

启动方式（三种任选）：

    python math_qa_mcp.py --transport stdio            # 本地 stdio
    python math_qa_mcp.py --transport sse --port 8770  # SSE
    python math_qa_mcp.py --transport http --port 8770 # Streamable HTTP

不装任何依赖也能自检：

    python math_qa_mcp.py --selftest
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

# 允许直接以脚本方式运行（仓库里的用法就是 python math_qa_mcp.py）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_qa_engine import (  # noqa: E402
    MathSyntaxError,
    CHAPTERS,
    check_derivative,
    check_integral,
    check_limit,
    compare_expressions,
    evaluate,
    locate_topic,
    render_tutor_card,
)

DEFAULT_PORT = 8770
SERVER_NAME = "higher_math_qa"


def _guard(func):
    """把可预期的输入错误转成结构化错误，避免整次调用失败。"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MathSyntaxError as exc:
            return {"error": str(exc), "error_type": "syntax"}
        except ValueError as exc:
            return {"error": str(exc), "error_type": "input"}

    return wrapper


# --------------------------------------------------------------------------
# 工具实现（同时被 selftest 与单测直接调用）
# --------------------------------------------------------------------------

@_guard
def locate_topic_tool(query: str, limit: int = 3) -> dict:
    """按关键词定位讲义章节与知识点编号。"""
    result = locate_topic(query, limit=max(1, min(int(limit or 3), 7)))
    if not result["matches"]:
        result["chapters"] = [
            {"code": c["code"], "title": c["title"]} for c in CHAPTERS
        ]
    return result


@_guard
def eval_expr_tool(expr: str, var: str = "x", at: str = "0") -> dict:
    """在给定点对表达式求值，用于核对代入结果。"""
    point = evaluate(at, variables=[])
    value = evaluate(expr, variables={var: point})
    finite = isinstance(value, float) and math.isfinite(value)
    return {
        "expr": expr,
        "var": var,
        "at": at,
        "value": float(f"{value:.10g}") if finite else None,
        "note": (
            "代入求值，不涉及极限过程"
            if finite
            else "该点处表达式无定义或趋于无穷（如分母为零、负数开偶次方）"
        ),
    }


@_guard
def check_limit_tool(expr: str, var: str = "x", point: str = "0", direction: str = "both") -> dict:
    """数值判定极限是否存在：区分收敛、发散、振荡与单侧存在。"""
    return check_limit(expr, var=var, point=point, direction=direction)


@_guard
def check_derivative_tool(expr: str, var: str = "x", point: str = "0", order: int = 1) -> dict:
    """验算某点的一阶或二阶导数值。"""
    return check_derivative(expr, var=var, point=point, order=int(order or 1))


@_guard
def check_integral_tool(expr: str, var: str = "x", a: str = "0", b: str = "1") -> dict:
    """验算定积分，端点无界时按反常积分处理。"""
    return check_integral(expr, var=var, a=a, b=b)


@_guard
def compare_expressions_tool(
    expr_a: str,
    expr_b: str,
    var: str = "x",
    allow_constant: bool = False,
) -> dict:
    """比对两个表达式是否等价：支持相差常数、互为相反数两种提示。"""
    return compare_expressions(expr_a, expr_b, var=var, allow_constant=bool(allow_constant))


@_guard
def render_tutor_card_tool(payload: Mapping[str, Any]) -> dict:
    """把答疑结论渲染成固定五段式卡片。"""
    return render_tutor_card(payload)


# --------------------------------------------------------------------------
# MCP 服务
# --------------------------------------------------------------------------

def build_server(port: int = DEFAULT_PORT):  # pragma: no cover - 依赖外部包，无法在纯标准库环境测试
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.server.transport_security import TransportSecuritySettings
    except ModuleNotFoundError as exc:  # mcp 2.x 把 FastMCP 改名为 MCPServer，API 不兼容
        raise SystemExit(
            "当前安装的 mcp 版本不含 FastMCP。mcp 2.x 已将其更名为 MCPServer 且参数有变，"
            "本服务按 mcp 1.x 的 API 编写。请执行：pip install 'mcp>=1.2.0,<2'，"
            "或改用 requirements.txt 安装。原始错误：" + str(exc)
        ) from exc

    mcp = FastMCP(
        SERVER_NAME,
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            # 按实际端口生成：换 --port 时无需改代码，否则会被 421 拦下
            allowed_hosts=[
                f"127.0.0.1:{port}",
                f"localhost:{port}",
                f"higher-math-course-qa:{port}",
            ],
            allowed_origins=[],
        ),
    )

    mcp.tool(name="locate_topic", description="按关键词定位讲义章节与知识点编号。")(locate_topic_tool)
    mcp.tool(name="eval_expr", description="在给定点对表达式求值，用于核对代入结果。")(eval_expr_tool)
    mcp.tool(name="check_limit", description="数值判定极限是否存在：区分收敛、发散、振荡与单侧存在。")(check_limit_tool)
    mcp.tool(name="check_derivative", description="验算某点的一阶或二阶导数值。")(check_derivative_tool)
    mcp.tool(name="check_integral", description="验算定积分，端点无界时按反常积分处理。")(check_integral_tool)
    mcp.tool(name="compare_expressions", description="比对两个表达式是否等价：支持相差常数、互为相反数两种提示。")(compare_expressions_tool)
    mcp.tool(name="render_tutor_card", description="把答疑结论渲染成固定五段式卡片。")(render_tutor_card_tool)
    return mcp


def run_selftest() -> int:
    """不依赖 mcp 包的自检：覆盖七个工具的正常与异常路径。"""
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if not condition:
            failures.append(f"{name}: {detail}")

    limit = check_limit_tool("sin(x)/x", point="0")
    check("sin(x)/x 极限", limit.get("verdict") == "exists" and abs(limit["value"] - 1) < 1e-6,
          json.dumps(limit, ensure_ascii=False)[:200])

    one_side = check_limit_tool("sqrt(x)", point="0")
    check("sqrt(x) 单侧极限", one_side.get("verdict") == "one_sided_right",
          one_side.get("verdict", ""))

    deriv = check_derivative_tool("x^2", point="3")
    check("x^2 在 3 处导数", abs(deriv["raw_value"] - 6) < 1e-5, str(deriv.get("raw_value")))

    integral = check_integral_tool("x^2", a="0", b="1")
    check("∫x² 0→1", abs(integral["raw_value"] - 1 / 3) < 1e-8, str(integral.get("raw_value")))

    improper = check_integral_tool("1/sqrt(x)", a="0", b="1")
    check("∫1/√x 0→1", abs(improper["raw_value"] - 2) < 1e-6, str(improper.get("raw_value")))

    compare = compare_expressions_tool("x^2", "x*x")
    check("x² 与 x·x 等价", compare.get("equivalent") is True, compare.get("verdict", ""))

    offset = compare_expressions_tool("x^2+1", "x^2", allow_constant=True)
    check("不定积分相差常数", offset.get("verdict") == "constant_offset", offset.get("verdict", ""))

    located = locate_topic_tool("等价无穷小替换")
    check("知识点定位", bool(located["matches"]) and located["matches"][0]["code"] == "HM-01-04",
          json.dumps(located.get("matches", [])[:1], ensure_ascii=False)[:120])

    card = render_tutor_card_tool(
        {
            "topic": {"code": "HM-01-04", "title": "无穷小与等价无穷小替换"},
            "blocker": "在加减法里直接替换 sin x ~ x",
            "hints": [{"level": 1, "text": "先看是否满足乘除结构"}],
            "verification": [{"tool": "check_limit", "claim": "sin x - x 与 x³ 同阶", "result": limit}],
        }
    )
    check("卡片渲染", card.get("markdown", "").count("**") >= 8 and card.get("sections"),
          str(card)[:120])

    bad = check_limit_tool("sin(x", point="0")
    check("语法错误被捕获", "error" in bad, str(bad)[:120])

    if failures:
        print("[selftest] FAILED")
        for item in failures:
            print("  -", item)
        return 1
    print("[selftest] OK · 7 个工具、9 项断言全部通过")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="高等数学课堂答疑助手 MCP 服务")
    parser.add_argument("--selftest", action="store_true", help="运行内置自检后退出")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "http"],
                        help="传输方式，默认 stdio")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()

    mcp = build_server(args.port)
    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        if args.transport == "sse":
            mcp.run(transport="sse")
        else:
            mcp.run(transport="streamable-http")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
