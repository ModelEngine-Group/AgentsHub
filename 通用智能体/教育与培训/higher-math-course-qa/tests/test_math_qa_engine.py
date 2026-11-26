"""高等数学答疑内核的单元测试。

运行方式（两种均可）：

    cd tests && python -m unittest test_math_qa_engine -v
    python -m unittest discover -s tests -v

覆盖重点不是「工具能跑」，而是几个容易回归的判定：
- (1-cos x)/x² 在 0 处的双精度抵消（曾误判为振荡）；
- ln x 在 0 处应判发散而不是振荡；
- |x| 在 0 处应判不可导（中心差分会给出假的 0）；
- 发散的反常积分必须标注 unstable，而不是返回一个看起来正常的数字。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT / "mcps"))

import math  # noqa: E402

from math_qa_engine import (  # noqa: E402
    MathSyntaxError,
    check_derivative,
    check_integral,
    check_limit,
    compare_expressions,
    evaluate,
    locate_topic,
    render_tutor_card,
)
from math_qa_engine.parser import compile_expr, normalize  # noqa: E402


class ParserTests(unittest.TestCase):
    def test_precedence(self) -> None:
        self.assertEqual(evaluate("1+2*3", {"x": 0}), 7.0)
        self.assertEqual(evaluate("(1+2)*3", {"x": 0}), 9.0)

    def test_power_is_right_associative(self) -> None:
        self.assertEqual(evaluate("2^3^2", {"x": 0}), 512.0)

    def test_unary_minus_binds_looser_than_power(self) -> None:
        # -x^2 必须理解为 -(x^2)
        self.assertEqual(evaluate("-x^2", {"x": 3}), -9.0)

    def test_implicit_multiplication(self) -> None:
        self.assertEqual(evaluate("2x", {"x": 4}), 8.0)
        self.assertEqual(evaluate("(x+1)(x-1)", {"x": 3}), 8.0)
        self.assertAlmostEqual(evaluate("3sin(x)", {"x": math.pi / 2}), 3.0)

    def test_constants(self) -> None:
        self.assertAlmostEqual(evaluate("pi", {"x": 0}), math.pi)
        self.assertAlmostEqual(evaluate("e^1", {"x": 0}), math.e)

    def test_normalize_fullwidth_and_superscript(self) -> None:
        self.assertEqual(normalize("π／2"), "pi／2")
        self.assertEqual(evaluate("x²", {"x": 5}), 25.0)
        self.assertEqual(evaluate("2×3", {"x": 0}), 6.0)

    def test_unknown_symbol_message(self) -> None:
        with self.assertRaises(MathSyntaxError) as ctx:
            evaluate("y+1", {"x": 1})
        self.assertIn("未知符号", str(ctx.exception))
        self.assertIn("x", str(ctx.exception))

    def test_unbalanced_parentheses(self) -> None:
        with self.assertRaises(MathSyntaxError) as ctx:
            evaluate("sin(x", {"x": 1})
        self.assertIn("应出现", str(ctx.exception))

    def test_function_requires_parentheses(self) -> None:
        with self.assertRaises(MathSyntaxError) as ctx:
            evaluate("sin x + 1", {"x": 1})
        self.assertIn("括号", str(ctx.exception))

    def test_function_arity_checked(self) -> None:
        with self.assertRaises(MathSyntaxError):
            evaluate("max(1)", {"x": 1})

    def test_empty_expression(self) -> None:
        with self.assertRaises(MathSyntaxError):
            evaluate("   ", {"x": 1})

    def test_division_by_zero_returns_infinity(self) -> None:
        self.assertTrue(math.isinf(evaluate("1/x", {"x": 0})))

    def test_out_of_domain_returns_nan(self) -> None:
        self.assertTrue(math.isnan(evaluate("sqrt(x)", {"x": -1})))

    def test_compile_reuse(self) -> None:
        node = compile_expr("x^2+1", variables=["x"])
        self.assertEqual(node({"x": 2}), 5.0)
        self.assertEqual(node({"x": -3}), 10.0)


class LimitTests(unittest.TestCase):
    def test_classic_sine_over_x(self) -> None:
        result = check_limit("sin(x)/x", point="0")
        self.assertEqual(result["verdict"], "exists")
        self.assertAlmostEqual(result["value"], 1.0, delta=1e-6)

    def test_cancellation_case_one_minus_cos(self) -> None:
        """回归：小步长时 1-cos(x) 会抵消为 0，不能因此判成振荡。"""
        result = check_limit("(1-cos(x))/x^2", point="0")
        self.assertEqual(result["verdict"], "exists")
        self.assertAlmostEqual(result["value"], 0.5, delta=1e-3)

    def test_log_limit_diverges(self) -> None:
        """回归：ln x 在 0+ 应判发散，而不是振荡。"""
        result = check_limit("ln(x)", point="0")
        self.assertEqual(result["verdict"], "diverges")

    def test_inverse_diverges(self) -> None:
        result = check_limit("1/x", point="0")
        self.assertEqual(result["verdict"], "diverges")

    def test_oscillation_detected(self) -> None:
        result = check_limit("sin(1/x)", point="0")
        self.assertEqual(result["verdict"], "oscillates")

    def test_jump_discontinuity(self) -> None:
        result = check_limit("abs(x)/x", point="0")
        self.assertEqual(result["verdict"], "not_exists")

    def test_one_sided_only(self) -> None:
        result = check_limit("sqrt(x)", point="0")
        self.assertEqual(result["verdict"], "one_sided_right")
        self.assertLess(abs(result["value"]), 1e-3)

    def test_limit_at_infinity_is_single_sided(self) -> None:
        result = check_limit("e^x", point="-inf")
        self.assertEqual(result["direction"], "tendency")
        self.assertEqual(result["verdict"], "exists")
        self.assertLess(abs(result["value"]), 1e-9)

    def test_second_important_limit(self) -> None:
        result = check_limit("(1+1/x)^x", point="inf")
        self.assertEqual(result["verdict"], "exists")
        self.assertAlmostEqual(result["value"], math.e, delta=1e-5)

    def test_convergent_but_monotone_not_misjudged(self) -> None:
        """回归：单调递增但收敛的序列不能判成发散。"""
        result = check_limit("2-1/x", point="inf")
        self.assertEqual(result["verdict"], "exists")
        self.assertAlmostEqual(result["value"], 2.0, delta=1e-3)

    def test_right_limit_of_x_ln_x(self) -> None:
        result = check_limit("x*ln(x)", point="0")
        self.assertEqual(result["verdict"], "one_sided_right")
        self.assertLess(abs(result["value"]), 1e-3)

    def test_invalid_direction(self) -> None:
        with self.assertRaises(MathSyntaxError):
            check_limit("x", point="0", direction="up")


class DerivativeTests(unittest.TestCase):
    def test_first_order(self) -> None:
        self.assertAlmostEqual(check_derivative("sin(x)", point="0")["raw_value"], 1.0, delta=1e-7)

    def test_first_order_polynomial(self) -> None:
        self.assertAlmostEqual(check_derivative("x^3", point="2")["raw_value"], 12.0, delta=1e-4)

    def test_second_order(self) -> None:
        self.assertAlmostEqual(check_derivative("x^2", point="3", order=2)["raw_value"], 2.0, delta=1e-4)

    def test_cusp_is_not_differentiable(self) -> None:
        """回归：中心差分会把 |x| 在 0 处算成 0，必须靠左右导数识别。"""
        result = check_derivative("abs(x)", point="0")
        self.assertEqual(result["value"], "不可导")
        self.assertFalse(result["one_sided"]["differentiable"])
        self.assertAlmostEqual(result["one_sided"]["left"], -1.0, delta=1e-6)
        self.assertAlmostEqual(result["one_sided"]["right"], 1.0, delta=1e-6)

    def test_smooth_point_is_differentiable(self) -> None:
        self.assertTrue(check_derivative("sin(x)", point="0")["one_sided"]["differentiable"])

    def test_undefined_point(self) -> None:
        result = check_derivative("ln(x)", point="0")
        self.assertEqual(result["value"], "未定义")

    def test_infinite_point_rejected(self) -> None:
        with self.assertRaises(MathSyntaxError):
            check_derivative("x", point="inf")


class IntegralTests(unittest.TestCase):
    def test_polynomial(self) -> None:
        self.assertAlmostEqual(check_integral("x^2", a="0", b="1")["raw_value"], 1 / 3, delta=1e-9)

    def test_trigonometric(self) -> None:
        self.assertAlmostEqual(check_integral("sin(x)", a="0", b="pi")["raw_value"], 2.0, delta=1e-8)

    def test_sqrt_singularity_at_endpoint(self) -> None:
        """回归：端点 1/√x 奇性需靠平方代换，否则误差停在 1e-5。"""
        result = check_integral("1/sqrt(x)", a="0", b="1")
        self.assertAlmostEqual(result["raw_value"], 2.0, delta=1e-7)

    def test_arcsine_integral(self) -> None:
        result = check_integral("1/sqrt(1-x^2)", a="0", b="1")
        self.assertAlmostEqual(result["raw_value"], math.pi / 2, delta=1e-7)

    def test_gaussian_over_whole_line(self) -> None:
        result = check_integral("exp(-x^2)", a="-inf", b="inf")
        self.assertAlmostEqual(result["raw_value"], math.sqrt(math.pi), delta=1e-6)

    def test_divergent_improper_flagged(self) -> None:
        """回归：∫₁^∞ dx/x 发散，不能返回 24 这类看似正常的数字。"""
        result = check_integral("1/x", a="1", b="inf")
        self.assertEqual(result["verdict"], "unstable")
        self.assertEqual(result["confidence"], "low")

    def test_convergent_improper(self) -> None:
        result = check_integral("1/x^2", a="1", b="inf")
        self.assertEqual(result["verdict"], "converged")
        self.assertAlmostEqual(result["raw_value"], 1.0, delta=1e-6)

    def test_reversed_interval(self) -> None:
        self.assertAlmostEqual(check_integral("x^2", a="1", b="0")["raw_value"], -1 / 3, delta=1e-9)

    def test_zero_length_interval(self) -> None:
        with self.assertRaises(MathSyntaxError):
            check_integral("x", a="1", b="1")


class CompareTests(unittest.TestCase):
    def test_equivalent_identity(self) -> None:
        result = compare_expressions("sin(x)^2+cos(x)^2", "1")
        self.assertTrue(result["equivalent"])
        self.assertEqual(result["verdict"], "equivalent")

    def test_different(self) -> None:
        self.assertFalse(compare_expressions("x^2", "x^3")["equivalent"])

    def test_constant_offset_allowed(self) -> None:
        result = compare_expressions("x^2+1", "x^2", allow_constant=True)
        self.assertEqual(result["verdict"], "constant_offset")
        self.assertTrue(result["equivalent"])

    def test_constant_offset_rejected_by_default(self) -> None:
        result = compare_expressions("x^2+1", "x^2")
        self.assertFalse(result["equivalent"])

    def test_opposite_sign_hint(self) -> None:
        self.assertEqual(compare_expressions("-x^2", "x^2")["verdict"], "opposite_sign")

    def test_incomparable_when_domain_too_narrow(self) -> None:
        result = compare_expressions("sqrt(-1-x^2)", "x")
        self.assertEqual(result["verdict"], "incomparable")


class TopicTests(unittest.TestCase):
    def test_locate_by_code(self) -> None:
        result = locate_topic("HM-01-04")
        self.assertEqual(result["matches"][0]["code"], "HM-01-04")

    def test_locate_by_keyword(self) -> None:
        result = locate_topic("分部积分怎么用")
        self.assertEqual(result["matches"][0]["code"], "HM-04-03")

    def test_locate_equivalence_infinitesimal(self) -> None:
        self.assertEqual(locate_topic("等价无穷小替换")["matches"][0]["code"], "HM-01-04")

    def test_empty_query(self) -> None:
        result = locate_topic("   ")
        self.assertEqual(result["matches"], [])
        self.assertIn("查询为空", result["note"])

    def test_every_topic_has_valid_chapter(self) -> None:
        from math_qa_engine.topics import TOPICS, chapter_of

        for topic in TOPICS:
            self.assertIsNotNone(chapter_of(topic["code"]), topic["code"])
            self.assertTrue(topic["keywords"], topic["code"])


class CardTests(unittest.TestCase):
    def test_five_sections_rendered(self) -> None:
        result = render_tutor_card(
            {
                "topic": {"code": "HM-01-04", "title": "无穷小与等价无穷小替换"},
                "blocker": "在加减法里直接替换 sin x ~ x",
                "diagnosis": "把乘除可替换的条件套用到了加减法",
                "knowledge_points": [{"code": "HM-01-04", "statement": "等价替换只在乘除因子中成立"}],
                "hints": [{"level": 1, "text": "先看结构是乘除还是加减"}],
                "verification": [
                    {
                        "tool": "check_limit",
                        "claim": "(sin x - x)/x³ → -1/6",
                        "result": {"verdict": "exists", "value": "-0.1666"},
                    }
                ],
                "next_step": "用泰勒展开到三阶再算一次",
            }
        )
        markdown = result["markdown"]
        for title in ("一、当前卡点", "二、涉及知识点", "三、分层提示", "四、验算证据", "五、下一步"):
            self.assertIn(title, markdown)
        self.assertIn("HM-01-04", markdown)
        self.assertIn("check_limit", markdown)

    def test_missing_blocker_rejected(self) -> None:
        with self.assertRaises(ValueError):
            render_tutor_card({"topic": {"code": "HM-01-01"}})

    def test_placeholders_when_empty(self) -> None:
        result = render_tutor_card({"topic": "极限", "blocker": "不会求"})
        self.assertIn("未检索到对应条目", result["markdown"])
        self.assertIn("未调用验算工具", result["markdown"])


class ToolLayerTests(unittest.TestCase):
    def test_syntax_error_is_structured(self) -> None:
        from math_qa_mcp import check_limit_tool

        result = check_limit_tool("sin(x", point="0")
        self.assertIn("error", result)
        self.assertEqual(result["error_type"], "syntax")

    def test_eval_tool(self) -> None:
        from math_qa_mcp import eval_expr_tool

        self.assertEqual(eval_expr_tool("2x^2", at="3")["value"], 18.0)

    def test_eval_tool_undefined_point(self) -> None:
        from math_qa_mcp import eval_expr_tool

        result = eval_expr_tool("1/x", at="0")
        self.assertIsNone(result["value"])

    def test_locate_tool_returns_chapter_hint_when_missed(self) -> None:
        from math_qa_mcp import locate_topic_tool

        result = locate_topic_tool("zzz 不存在的关键词")
        self.assertEqual(result["matches"], [])
        self.assertTrue(result["chapters"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
