"""数学表达式解析与求值（纯标准库，无第三方依赖）。

设计目标只有两条：
1. 安全：不调用 eval，不执行任意代码，仅支持白名单内的运算与函数；
2. 可诊断：语法错误必须指出位置与可能的写法问题，而不是抛出通用的 TypeError。

使用方式：

    f = compile_expr("sin(x)/x", variables=["x"])
    f({"x": 0.5})        # -> 0.958851...
"""

from __future__ import annotations

import math
import re
from typing import Callable, Dict, Iterable, Mapping, Sequence

__all__ = [
    "MathSyntaxError",
    "compile_expr",
    "evaluate",
    "normalize",
    "FUNCTIONS",
]


class MathSyntaxError(ValueError):
    """表达式无法解析。消息直接面向用户，避免暴露实现细节。"""


# --------------------------------------------------------------------------
# 预处理：把常见的中文输入习惯转成标准写法
# --------------------------------------------------------------------------

_NORMALIZE_MAP = {
    "×": "*",
    "·": "*",
    "÷": "/",
    "−": "-",
    "–": "-",
    "—": "-",
    "（": "(",
    "）": ")",
    "，": ",",
    "π": "pi",
    "∞": "inf",
    "²": "^2",
    "³": "^3",
    "¹": "^1",
    "⁰": "^0",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
    "⁺": "+",
    "⁻": "-",
}


def normalize(expr: str) -> str:
    """统一全角符号、乘号与上标写法。"""
    if not isinstance(expr, str):
        raise MathSyntaxError("表达式必须是字符串")
    text = expr.strip()
    for src, dst in _NORMALIZE_MAP.items():
        text = text.replace(src, dst)
    return text


# --------------------------------------------------------------------------
# 词法分析
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<op>[+\-*/^(),|])
    """,
    re.VERBOSE,
)


def _tokenize(text: str) -> list[tuple[str, str, int]]:
    tokens: list[tuple[str, str, int]] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            raise MathSyntaxError(
                f"第 {pos + 1} 个字符 “{text[pos]}” 无法识别；"
                "支持数字、字母变量、+ - * / ^ ( ) , 与常用函数名"
            )
        kind = match.lastgroup or ""
        value = match.group()
        if kind != "space":
            tokens.append((kind, value, pos))
        pos = match.end()
    tokens.append(("end", "", len(text)))
    return tokens


# --------------------------------------------------------------------------
# 函数与常数白名单
# --------------------------------------------------------------------------

def _wrap(fn: Callable[..., float]) -> Callable[..., float]:
    """把 math 中可能抛出的定义域错误统一转成 nan，便于数值采样继续执行。"""

    def inner(*args: float) -> float:
        try:
            return float(fn(*args))
        except (ValueError, OverflowError):
            return math.nan
        except ZeroDivisionError:
            return math.nan

    return inner


FUNCTIONS: Dict[str, tuple[int, Callable[..., float]]] = {
    "sin": (1, _wrap(math.sin)),
    "cos": (1, _wrap(math.cos)),
    "tan": (1, _wrap(math.tan)),
    "cot": (1, _wrap(lambda x: 1.0 / math.tan(x))),
    "sec": (1, _wrap(lambda x: 1.0 / math.cos(x))),
    "csc": (1, _wrap(lambda x: 1.0 / math.sin(x))),
    "asin": (1, _wrap(math.asin)),
    "acos": (1, _wrap(math.acos)),
    "atan": (1, _wrap(math.atan)),
    # 学生更习惯写 arctan / arcsin / arccos，一并接受
    "arctan": (1, _wrap(math.atan)),
    "arcsin": (1, _wrap(math.asin)),
    "arccos": (1, _wrap(math.acos)),
    "sinh": (1, _wrap(math.sinh)),
    "cosh": (1, _wrap(math.cosh)),
    "tanh": (1, _wrap(math.tanh)),
    "exp": (1, _wrap(math.exp)),
    "ln": (1, _wrap(math.log)),
    "log": (1, _wrap(math.log)),
    "lg": (1, _wrap(math.log10)),
    "sqrt": (1, _wrap(math.sqrt)),
    "abs": (1, _wrap(abs)),
    "floor": (1, _wrap(math.floor)),
    "ceil": (1, _wrap(math.ceil)),
    "sign": (1, _wrap(lambda x: float((x > 0) - (x < 0)))),
    "max": (2, _wrap(max)),
    "min": (2, _wrap(min)),
}

CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
}


def _div(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        if a == 0.0:
            return math.nan
        return math.copysign(math.inf, a)


def _pow(a: float, b: float) -> float:
    try:
        return float(a ** b)
    except ZeroDivisionError:
        return math.inf
    except OverflowError:
        return math.copysign(math.inf, a)
    except (ValueError, TypeError):
        return math.nan


# --------------------------------------------------------------------------
# 语法分析：递归下降，边解析边生成闭包（避免构建中间 AST）
# --------------------------------------------------------------------------

# 一个「可求值节点」就是 env -> float 的函数
Node = Callable[[Mapping[str, float]], float]


class _Parser:
    def __init__(self, text: str, variables: Sequence[str]) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.index = 0
        self.variables = set(variables)

    # -- 词法游标 ---------------------------------------------------------
    def peek(self) -> tuple[str, str, int]:
        return self.tokens[self.index]

    def take(self) -> tuple[str, str, int]:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def accept(self, kind: str, value: str | None = None) -> bool:
        token = self.peek()
        if token[0] == kind and (value is None or token[1] == value):
            self.index += 1
            return True
        return False

    def expect(self, kind: str, value: str | None = None) -> tuple[str, str, int]:
        if not self.accept(kind, value):
            want = value or kind
            got = self.peek()
            raise MathSyntaxError(
                f"第 {got[2] + 1} 个字符处应出现 “{want}”，实际读到 “{got[1] or '结尾'}”"
            )
        return self.tokens[self.index - 1]

    # -- 语法规则 ---------------------------------------------------------
    def parse(self) -> Node:
        node = self.expression()
        if self.peek()[0] != "end":
            token = self.peek()
            raise MathSyntaxError(
                f"第 {token[2] + 1} 个字符处出现多余的 “{token[1]}”"
            )
        return node

    def expression(self) -> Node:
        node = self.term()
        while True:
            if self.accept("op", "+"):
                right = self.term()
                node = _binary(node, right, lambda a, b: a + b)
            elif self.accept("op", "-"):
                right = self.term()
                node = _binary(node, right, lambda a, b: a - b)
            else:
                return node

    def term(self) -> Node:
        node = self.unary()
        while True:
            if self.accept("op", "*"):
                node = _binary(node, self.unary(), lambda a, b: a * b)
            elif self.accept("op", "/"):
                node = _binary(node, self.unary(), _div)
            elif self._starts_implicit_factor():
                # 隐式乘法：2x、3(x+1)、2sin(x)、(x+1)(x-1)
                node = _binary(node, self.power(), lambda a, b: a * b)
            else:
                return node

    def _starts_implicit_factor(self) -> bool:
        kind = self.peek()[0]
        return kind in {"number", "name"} or (kind == "op" and self.peek()[1] == "(")

    def unary(self) -> Node:
        if self.accept("op", "-"):
            inner = self.unary()
            return lambda env: -inner(env)
        if self.accept("op", "+"):
            return self.unary()
        return self.power()

    def power(self) -> Node:
        base = self.atom()
        if self.accept("op", "^"):
            exponent = self.unary()  # 右结合，且允许 2^-1
            return _binary(base, exponent, _pow)
        return base

    def atom(self) -> Node:
        kind, value, pos = self.peek()

        if kind == "number":
            self.take()
            number = float(value)
            return lambda env: number

        if kind == "op" and value == "(":
            self.take()
            node = self.expression()
            self.expect("op", ")")
            return node

        if kind == "name":
            self.take()
            lowered = value.lower()
            if lowered in FUNCTIONS:
                arity, fn = FUNCTIONS[lowered]
                if not self.accept("op", "("):
                    raise MathSyntaxError(
                        f"函数 “{value}” 需要带括号，例如 {value}(x)"
                    )
                args = [self.expression()]
                while self.accept("op", ","):
                    args.append(self.expression())
                self.expect("op", ")")
                if len(args) != arity:
                    raise MathSyntaxError(
                        f"函数 “{value}” 需要 {arity} 个参数，实际给了 {len(args)} 个"
                    )
                if arity == 1:
                    first = args[0]
                    return lambda env: fn(first(env))
                if arity == 2:
                    first, second = args
                    return lambda env: fn(first(env), second(env))
                return lambda env: fn(*[arg(env) for arg in args])

            if lowered in CONSTANTS:
                constant = CONSTANTS[lowered]
                return lambda env: constant

            if value in self.variables:
                return lambda env, name=value: env.get(name, math.nan)

            allowed = "、".join(sorted(self.variables)) or "（无）"
            raise MathSyntaxError(
                f"未知符号 “{value}”；当前允许的变量为 {allowed}。"
                "如需常数请用 pi / e，函数请用 sin、cos、ln、sqrt 等"
            )

        raise MathSyntaxError(
            f"第 {pos + 1} 个字符处缺少操作数，实际读到 “{value or '结尾'}”"
        )


def _binary(left: Node, right: Node, op: Callable[[float, float], float]) -> Node:
    def inner(env: Mapping[str, float]) -> float:
        return op(left(env), right(env))

    return inner


def compile_expr(expr: str, variables: Iterable[str] = ("x",)) -> Node:
    """把表达式编译成可调用对象；语法问题抛 MathSyntaxError。"""
    text = normalize(expr)
    if not text:
        raise MathSyntaxError("表达式为空")
    parser = _Parser(text, [str(v) for v in variables])
    return parser.parse()


def evaluate(expr: str, variables: Mapping[str, float] | None = None) -> float:
    """一次性求值，主要用于常数表达式与测试。"""
    env = dict(variables or {})
    return compile_expr(expr, variables=list(env.keys()) or ["x"])(env)
