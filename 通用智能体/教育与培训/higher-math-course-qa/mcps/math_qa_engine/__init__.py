"""高等数学课堂答疑助手的本地验算内核。

三层职责：
- parser  : 把字符串表达式安全地编译成可求值函数；
- numeric : 极限 / 导数 / 定积分 / 等价性的数值验算；
- topics  : 章节与知识点索引，提供稳定的引用编号；
- card    : 五段式答疑卡片渲染。
"""

from .card import render_tutor_card
from .numeric import check_derivative, check_integral, check_limit, compare_expressions
from .parser import MathSyntaxError, compile_expr, evaluate
from .topics import CHAPTERS, TOPICS, locate_topic

__all__ = [
    "MathSyntaxError",
    "compile_expr",
    "evaluate",
    "check_limit",
    "check_derivative",
    "check_integral",
    "compare_expressions",
    "render_tutor_card",
    "locate_topic",
    "CHAPTERS",
    "TOPICS",
]

__version__ = "1.0.0"
