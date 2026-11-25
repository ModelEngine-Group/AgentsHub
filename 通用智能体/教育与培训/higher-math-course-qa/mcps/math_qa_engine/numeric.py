"""数值验算：极限、导数、定积分与表达式等价性。

定位说明（重要）：
本模块给出的是**数值证据**，不是严格证明。所有返回结果都带 `evidence_type`
与 `confidence` 字段，提示词层被要求如实转述这一点。对课堂答疑而言，
数值证据足以纠正 90% 以上的计算错误，但不能替代"用定义或定理完成证明"。

数值方法选型：
- 极限：单侧递减步长采样 + 收敛性判别（区分「收敛」「发散」「振荡」「单侧存在」）；
- 导数：中心差分 + Richardson 外推（精度约 1e-9，足够发现符号与系数错误）；
- 积分：自适应 Gauss–Legendre（16 点），开区间取点，天然适合端点无界的反常积分；
- 等价性：多点采样比对，支持「相差一个常数」与「互为相反数」两种常见情形。
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping, Sequence

from .parser import MathSyntaxError, compile_expr, normalize

__all__ = [
    "check_limit",
    "check_derivative",
    "check_integral",
    "compare_expressions",
    "DEFAULT_TOL",
]

DEFAULT_TOL = 1e-6
_FINITE_LIMIT = 1e12


# --------------------------------------------------------------------------
# 公共小工具
# --------------------------------------------------------------------------

def _finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _round(value: float) -> float | str:
    """返回值统一保留 10 位有效数字；非有限值转成可读字符串。"""
    if value is None:
        return "未定义"
    if isinstance(value, str):
        return value
    if math.isnan(value):
        return "未定义（NaN）"
    if math.isinf(value):
        return "+∞" if value > 0 else "-∞"
    if value == 0:
        return 0.0
    if abs(value) >= 1e6 or abs(value) < 1e-6:
        return float(f"{value:.6e}")
    return float(f"{value:.10g}")


def _parse_point(point: str | float) -> tuple[float, bool]:
    """解析极限点。支持 0、pi/2、inf、+inf、-inf。"""
    if isinstance(point, (int, float)):
        value = float(point)
        return value, math.isinf(value)
    text = normalize(str(point)).strip().lower()
    if text in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf, True
    if text in {"-inf", "-infinity"}:
        return -math.inf, True
    try:
        value = float(text)
        return value, math.isinf(value)
    except ValueError:
        pass
    try:
        value = compile_expr(text, variables=[])({})
    except MathSyntaxError as exc:
        raise MathSyntaxError(f"极限点 “{point}” 无法解析：{exc}") from exc
    if not _finite(value):
        return value, math.isinf(value)
    return value, False


def _make_function(expr: str, var: str) -> Callable[[float], float]:
    node = compile_expr(expr, variables=[var])

    def f(x: float) -> float:
        try:
            value = node({var: float(x)})
        except (ZeroDivisionError, ValueError, OverflowError):
            return math.nan
        return float(value)

    return f


# --------------------------------------------------------------------------
# 极限
# --------------------------------------------------------------------------

def _sample_side(
    f: Callable[[float], float],
    point: float,
    *,
    side: int,
    infinite: bool,
    steps: int = 9,
) -> dict:
    """从某一侧逼近极限点，返回样本序列与收敛性判定。

    判定顺序是有意安排的：先判发散，再做精度噪声截断，最后判收敛。
    顺序颠倒会把 ln x 这类「量级无穷增长」误判成振荡，
    也会把 (1-cos x)/x² 这类「先收敛后落入双精度噪声」误判成振荡。
    """
    positions: list[float] = []
    values: list[float] = []
    for k in range(1, steps + 1):
        x = side * 10.0**k if infinite else point + side * (10.0**-k)
        positions.append(x)
        values.append(f(x))

    samples = [{"x": _round(p), "f(x)": _round(v)} for p, v in zip(positions, values)]
    if infinite:
        label = "趋向 +∞" if side > 0 else "趋向 -∞"
    else:
        label = "右" if side > 0 else "左"

    finite_pairs = [(p, v) for p, v in zip(positions, values) if _finite(v)]
    if not finite_pairs:
        return {
            "side": label,
            "verdict": "undefined",
            "value": None,
            "samples": samples,
            "note": f"{label}侧全部采样点无定义",
        }

    vals = [v for _, v in finite_pairs]
    mags = [abs(v) for v in vals]
    diffs = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]

    # 一、发散：量级单调不减，且差值没有收缩迹象（ln x、1/x 属此类）
    non_decreasing = all(mags[i + 1] >= mags[i] * 0.999 for i in range(len(mags) - 1))
    no_shrink = bool(diffs) and diffs[-1] >= 0.5 * diffs[0]
    if len(vals) >= 3 and mags[-1] > 3 and non_decreasing and no_shrink:
        sign = 1 if vals[-1] > 0 else -1
        return {
            "side": label,
            "verdict": "diverges",
            "value": math.inf * sign,
            "samples": samples,
            "note": (
                f"{label}侧量级持续增大（末值 {_round(vals[-1])}），"
                f"判为趋向{'正' if sign > 0 else '负'}无穷"
            ),
        }

    # 二、精度噪声截断：差分突然放大多半意味着已进入双精度噪声区
    cutoff = len(vals)
    for i in range(1, len(diffs)):
        if diffs[i] > 4 * diffs[i - 1] and diffs[i] > 1e-9 * max(1.0, mags[i + 1]):
            cutoff = i + 1
            break
    truncated = cutoff < len(vals)
    vals = vals[:cutoff]
    if truncated:
        # 只把参与判定的样本交给调用方：噪声段的 0.0 会误导模型把极限读成 0
        samples = [{"x": _round(p), "f(x)": _round(v)} for p, v in finite_pairs[:cutoff]]
        samples.append({"x": "更小步长", "f(x)": "已忽略（双精度噪声）"})

    if len(vals) < 3:
        return {
            "side": label,
            "verdict": "oscillates",
            "value": None,
            "samples": samples,
            "note": f"{label}侧相邻样本差值持续放大，无收敛趋势，判为振荡或未收敛",
        }

    last = vals[-4:] if len(vals) >= 4 else vals
    spread = max(last) - min(last)
    scale = max(1.0, max(abs(v) for v in last))

    tail_diffs = [abs(vals[i + 1] - vals[i]) for i in range(max(0, len(vals) - 5), len(vals) - 1)]
    ratios = [tail_diffs[i + 1] / tail_diffs[i] for i in range(len(tail_diffs) - 1) if tail_diffs[i] > 0]
    shrinking = (sorted(ratios)[len(ratios) // 2] < 0.6) if ratios else True

    noise_note = "（已忽略更小步长的采样，避免双精度噪声干扰）" if truncated else ""

    if spread <= 1e-6 * scale:
        return {
            "side": label,
            "verdict": "converges",
            "value": sum(last) / len(last),
            "samples": samples,
            "note": f"{label}侧收敛，末段极差 {spread:.2e}{noise_note}",
        }

    if shrinking and spread <= 1e-3 * scale:
        return {
            "side": label,
            "verdict": "converges",
            "value": sum(last) / len(last),
            "samples": samples,
            "note": f"{label}侧收敛（末段极差 {spread:.2e}，精度一般）{noise_note}",
        }

    if shrinking:
        return {
            "side": label,
            "verdict": "converges_slowly",
            "value": vals[-1],
            "samples": samples,
            "note": (
                f"{label}侧仍在变化（末段极差 {spread:.2e}），"
                f"估计值取最后一个样本，置信度偏低{noise_note}"
            ),
        }

    return {
        "side": label,
        "verdict": "oscillates",
        "value": None,
        "samples": samples,
        "note": f"{label}侧样本无收敛趋势（末段极差 {spread:.2e}），判为振荡或未收敛",
    }


def check_limit(
    expr: str,
    var: str = "x",
    point: str | float = "0",
    direction: str = "both",
) -> dict:
    """数值判定函数在给定点的极限。"""
    direction = (direction or "both").lower()
    if direction not in {"both", "left", "right"}:
        raise MathSyntaxError("direction 只能取 both、left、right")

    f = _make_function(expr, var)
    value, infinite = _parse_point(point)

    if infinite:
        # 无穷远点只有一个趋近方向，left/right 之分没有意义
        side = 1 if value > 0 else -1
        key = "right" if side > 0 else "left"
        effective = key
        sides: dict[str, dict] = {key: _sample_side(f, value, side=side, infinite=True)}
    else:
        effective = direction
        sides = {}
        if direction in {"both", "right"}:
            sides["right"] = _sample_side(f, value, side=1, infinite=False)
        if direction in {"both", "left"}:
            sides["left"] = _sample_side(f, value, side=-1, infinite=False)

    result: dict = {
        "expr": normalize(expr),
        "var": var,
        "point": str(point),
        "direction": "tendency" if infinite else direction,
        "evidence_type": "数值采样（非证明）",
        "sides": sides,
    }

    right = sides.get("right")
    left = sides.get("left")

    def usable(side: dict | None) -> bool:
        return side is not None and side["verdict"] in {"converges", "converges_slowly"} and side["value"] is not None

    if effective == "both":
        if usable(left) and usable(right):
            diff = abs(left["value"] - right["value"])
            scale = max(1.0, abs(left["value"]), abs(right["value"]))
            if diff / scale <= 1e-4:
                result["verdict"] = "exists"
                result["value"] = (left["value"] + right["value"]) / 2
                result["confidence"] = "high" if (
                    left["verdict"] == right["verdict"] == "converges"
                ) else "medium"
                result["summary"] = f"左右极限一致，极限约为 {_round(result['value'])}"
            else:
                result["verdict"] = "not_exists"
                result["value"] = None
                result["confidence"] = "high"
                result["summary"] = (
                    f"左右极限不相等（左 {_round(left['value'])}，右 {_round(right['value'])}），极限不存在"
                )
        elif usable(right):
            result["verdict"] = "one_sided_right"
            result["value"] = right["value"]
            result["confidence"] = "medium"
            result["summary"] = f"仅右极限存在，约为 {_round(right['value'])}"
        elif usable(left):
            result["verdict"] = "one_sided_left"
            result["value"] = left["value"]
            result["confidence"] = "medium"
            result["summary"] = f"仅左极限存在，约为 {_round(left['value'])}"
        elif any(s["verdict"] == "diverges" for s in sides.values()):
            result["verdict"] = "diverges"
            result["value"] = None
            result["confidence"] = "high"
            result["summary"] = "至少一侧趋于无穷，极限不存在（发散）"
        else:
            result["verdict"] = "oscillates"
            result["value"] = None
            result["confidence"] = "medium"
            result["summary"] = "至少一侧无收敛趋势，极限不存在（振荡或未收敛）"
    else:
        side = right or left
        assert side is not None
        mapping = {
            "converges": "exists",
            "converges_slowly": "exists",
            "diverges": "diverges",
            "oscillates": "oscillates",
            "undefined": "undefined",
        }
        result["verdict"] = mapping[side["verdict"]]
        result["value"] = side["value"] if mapping[side["verdict"]] == "exists" else None
        result["confidence"] = "high" if side["verdict"] == "converges" else "medium"
        result["summary"] = side["note"]

    return result


# --------------------------------------------------------------------------
# 导数
# --------------------------------------------------------------------------

def check_derivative(
    expr: str,
    var: str = "x",
    point: str | float = "0",
    order: int = 1,
) -> dict:
    """中心差分 + Richardson 外推，验算某点的一阶或二阶导数值。"""
    if order not in (1, 2):
        raise MathSyntaxError("order 目前仅支持 1 或 2")

    f = _make_function(expr, var)
    x, infinite = _parse_point(point)
    if infinite:
        raise MathSyntaxError("导数验算需要有限点，无穷远点请先换元")

    h0 = 1e-3 * max(1.0, abs(x))

    def central(step: float) -> float:
        if order == 1:
            return (f(x + step) - f(x - step)) / (2 * step)
        return (f(x + step) - 2 * f(x) + f(x - step)) / (step * step)

    coarse = central(h0)
    fine = central(h0 / 2)
    extrapolated = (4 * fine - coarse) / 3

    samples = []
    for step in (h0, h0 / 2, h0 / 4):
        samples.append({"h": _round(step), "估计值": _round(central(step))})

    # 中心差分对 |x| 这类尖点会给出「恰好等于 0」的假结果，
    # 因此一阶导数额外比较左右差商：两者显著不等即判为不可导。
    one_sided = None
    if order == 1:
        left_q = (f(x) - f(x - h0)) / h0
        right_q = (f(x + h0) - f(x)) / h0
        if _finite(left_q) and _finite(right_q):
            scale_q = max(1.0, abs(left_q), abs(right_q))
            gap = abs(left_q - right_q) / scale_q
            one_sided = {
                "differentiable": gap <= 2e-2,
                "left": _round(left_q),
                "right": _round(right_q),
                "relative_gap": _round(gap),
            }

    stable = _finite(coarse) and _finite(extrapolated)
    error = abs(fine - coarse) / 3 if stable else float("nan")

    result: dict = {
        "expr": normalize(expr),
        "var": var,
        "point": x,
        "order": order,
        "samples": samples,
        "evidence_type": "中心差分 + Richardson 外推（非证明）",
    }

    if one_sided is not None:
        result["one_sided"] = one_sided
        if not one_sided["differentiable"]:
            result.update(
                value="不可导",
                raw_value=None,
                estimated_error="—",
                confidence="high",
                summary=(
                    f"左导数 ≈ {one_sided['left']}，右导数 ≈ {one_sided['right']}；"
                    "左右导数不相等，该点不可导（绝对值、分段函数分界点常见此情形）"
                ),
            )
            return result

    result.update(
        value=_round(extrapolated) if stable else "未定义",
        raw_value=extrapolated if stable else None,
        estimated_error=_round(error) if stable else "未定义",
        confidence="high" if stable and error < 1e-6 else ("medium" if stable else "low"),
        summary=(
            f"f^({order})({_round(x)}) ≈ {_round(extrapolated)}，估计误差 {_round(error)}"
            if stable
            else "该点附近函数值异常（可能无定义或不可导），无法给出导数估计"
        ),
    )
    return result


# --------------------------------------------------------------------------
# 定积分
# --------------------------------------------------------------------------

def _gauss_legendre(n: int) -> tuple[list[float], list[float]]:
    """用 Newton 迭代求 n 点 Gauss–Legendre 节点与权重（无需硬编码表）。"""
    nodes: list[float] = []
    weights: list[float] = []
    for i in range(1, n + 1):
        # 初值：Chebyshev 近似
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = -p1 / dp
            x += dx
            if abs(dx) < 1e-15:
                break
        p0, p1 = 1.0, x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1)
        nodes.append(x)
        weights.append(2.0 / ((1 - x * x) * dp * dp))
    return nodes, weights


_GL_NODES, _GL_WEIGHTS = _gauss_legendre(16)


def _gl_integrate(f: Callable[[float], float], a: float, b: float) -> float:
    half = (b - a) / 2
    mid = (a + b) / 2
    total = 0.0
    for node, weight in zip(_GL_NODES, _GL_WEIGHTS):
        value = f(mid + half * node)
        if not math.isfinite(value):
            return math.nan
        total += weight * value
    return total * half


def _adaptive(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float,
    depth: int = 0,
    max_depth: int = 24,
) -> tuple[float, float]:
    """自适应细分，返回（积分值，误差估计）。"""
    whole = _gl_integrate(f, a, b)
    mid = (a + b) / 2
    left = _gl_integrate(f, a, mid)
    right = _gl_integrate(f, mid, b)
    if not all(math.isfinite(v) for v in (whole, left, right)):
        return math.nan, math.nan
    refined = left + right
    if depth >= max_depth or abs(refined - whole) <= tol * (1 + abs(refined)):
        return refined, abs(refined - whole)
    l_val, l_err = _adaptive(f, a, mid, tol, depth + 1, max_depth)
    r_val, r_err = _adaptive(f, mid, b, tol, depth + 1, max_depth)
    return l_val + r_val, l_err + r_err


def check_integral(
    expr: str,
    var: str = "x",
    a: str | float = "0",
    b: str | float = "1",
    tol: float = 1e-10,
) -> dict:
    """数值验算定积分（含端点无界的反常积分）。"""
    f = _make_function(expr, var)
    lower, lower_inf = _parse_point(a)
    upper, upper_inf = _parse_point(b)

    if lower_inf or upper_inf:
        if lower_inf and upper_inf and lower * upper > 0:
            raise MathSyntaxError("积分区间端点不能同时为同号无穷，请拆分处理")
        try:
            if lower_inf and upper_inf:
                # (-∞, +∞)：x = tan(π(t-1/2))，dx = π / cos²(π(t-1/2)) dt
                def transformed(t: float) -> float:
                    angle = math.pi * (t - 0.5)
                    cos = math.cos(angle)
                    if cos == 0:
                        return math.nan
                    return f(math.tan(angle)) * math.pi / (cos * cos)

                method = "双无穷区间正切换元 + 自适应 Gauss–Legendre"
            elif upper_inf:
                # [a, +∞)：x = a + t/(1-t)
                def transformed(t: float) -> float:
                    if t >= 1:
                        return math.nan
                    return f(lower + t / (1 - t)) / ((1 - t) ** 2)

                method = "无穷区间换元 x=a+t/(1-t) + 自适应 Gauss–Legendre"
            else:
                # (-∞, b]：x = b - t/(1-t)
                def transformed(t: float) -> float:
                    if t >= 1:
                        return math.nan
                    return f(upper - t / (1 - t)) / ((1 - t) ** 2)

                method = "无穷区间换元 x=b-t/(1-t) + 自适应 Gauss–Legendre"

            value, err = _adaptive(transformed, 0.0, 1.0, tol)
        except RecursionError:
            value, err = math.nan, math.nan
    else:
        if lower > upper:
            lower, upper = upper, lower
            swapped = True
        else:
            swapped = False

        # 端点无界时，x = 端点 ± w·t²（t∈[0,1]）可消去 1/√ 型奇性：
        # 自适应细分对 t^(-1/2) 收敛极慢，代换后被积函数变为光滑函数。
        low_singular = not math.isfinite(f(lower))
        high_singular = not math.isfinite(f(upper))
        width = upper - lower
        if width == 0:
            raise MathSyntaxError("积分区间长度为 0")
        if low_singular or high_singular:
            if low_singular:
                def transformed(t: float) -> float:
                    return f(lower + width * t * t) * (2 * width * t)
            else:
                def transformed(t: float) -> float:
                    return f(upper - width * t * t) * (2 * width * t)

            value, err = _adaptive(transformed, 0.0, 1.0, tol)
            method = "端点平方代换 + 自适应 Gauss–Legendre（16 点）"
            if low_singular and high_singular:
                method += "；两端均无界，结果仅供参考"
        else:
            value, err = _adaptive(f, lower, upper, tol)
            method = "自适应 Gauss–Legendre（16 点）"
        if swapped:
            value = -value

    converged = math.isfinite(value)
    if converged:
        rel_err = abs(err) / max(1.0, abs(value))
        if rel_err <= 1e-6:
            verdict, confidence = "converged", "high"
        elif rel_err <= 1e-4:
            verdict, confidence = "converged", "medium"
        else:
            verdict, confidence = "unstable", "low"
    else:
        rel_err = float("inf")
        verdict, confidence = "diverges", "low"

    if verdict == "converged":
        summary = f"∫ 在 [{a}, {b}] 上约为 {_round(value)}，误差估计 {_round(err)}"
    elif verdict == "unstable":
        summary = (
            f"数值结果为 {_round(value)}，但相对误差 {rel_err:.1e} 偏大："
            "常见原因是区间内有奇点，或该反常积分本身发散（如 p≤1 的 ∫₁^∞ x^-p）。"
            "此结果不可直接采信，请先判断敛散性"
        )
    else:
        summary = "数值结果不收敛，可能为发散的反常积分或表达式在该区间无定义"

    return {
        "expr": normalize(expr),
        "var": var,
        "interval": [str(a), str(b)],
        "verdict": verdict,
        "value": _round(value) if converged else "发散或无法计算",
        "raw_value": value if converged else None,
        "error_estimate": _round(err) if math.isfinite(err) else "未定义",
        "relative_error": _round(rel_err) if math.isfinite(rel_err) else "未定义",
        "method": method,
        "evidence_type": "自适应数值积分（非证明）",
        "confidence": confidence,
        "summary": summary,
    }


# --------------------------------------------------------------------------
# 表达式等价性
# --------------------------------------------------------------------------

_DEFAULT_POINTS = (-2.3, -1.7, -0.9, -0.37, 0.23, 0.61, 1.11, 1.83, 2.47, 3.19)


def compare_expressions(
    expr_a: str,
    expr_b: str,
    var: str = "x",
    allow_constant: bool = False,
    tol: float = 1e-6,
) -> dict:
    """多点采样比对两个表达式。用于核对「我的答案与标准答案是否一致」。"""
    f_a = _make_function(expr_a, var)
    f_b = _make_function(expr_b, var)

    checked = []
    for x in _DEFAULT_POINTS:
        va, vb = f_a(x), f_b(x)
        if math.isfinite(va) and math.isfinite(vb):
            checked.append((x, va, vb))

    if len(checked) < 3:
        return {
            "verdict": "incomparable",
            "equivalent": False,
            "reason": "两个表达式在公共定义域内的有效采样点不足 3 个，"
            "可能存在定义域差异或表达式无法求值",
            "checked": [{"x": _round(x), "a": _round(va), "b": _round(vb)} for x, va, vb in checked],
            "evidence_type": "数值采样（非证明）",
            "confidence": "low",
        }

    diffs = [va - vb for _, va, vb in checked]
    max_abs = max(
        abs(va - vb) / max(1.0, abs(va), abs(vb)) for _, va, vb in checked
    )

    result: dict = {
        "expr_a": normalize(expr_a),
        "expr_b": normalize(expr_b),
        "var": var,
        "checked": [
            {"x": _round(x), "a": _round(va), "b": _round(vb)} for x, va, vb in checked
        ],
        "max_relative_difference": _round(max_abs),
        "evidence_type": "数值采样（非证明）",
        "confidence": "high" if max_abs < tol else "medium",
    }

    if max_abs < tol:
        result.update(
            verdict="equivalent",
            equivalent=True,
            reason="所有采样点上两者一致",
        )
        return result

    opposite = max(abs(va + vb) / max(1.0, abs(va), abs(vb)) for _, va, vb in checked)
    if opposite < tol:
        result.update(
            verdict="opposite_sign",
            equivalent=False,
            reason="两者互为相反数，检查符号或移项方向",
        )
        return result

    spread = max(diffs) - min(diffs)
    offset = sum(diffs) / len(diffs)
    scale = max(1.0, max(abs(vb) for _, _, vb in checked))
    if allow_constant or spread / scale < tol:
        result.update(
            verdict="constant_offset",
            equivalent=allow_constant,
            offset=_round(offset),
            reason=(
                f"两者相差常数 {_round(offset)}；不定积分结果允许相差常数 C，"
                "若题目要求定积分或指定原函数则不能接受"
            ),
        )
        return result

    first = next(
        (
            {"x": _round(x), "a": _round(va), "b": _round(vb)}
            for x, va, vb in checked
            if abs(va - vb) / max(1.0, abs(va), abs(vb)) > tol
        ),
        None,
    )
    result.update(
        verdict="different",
        equivalent=False,
        first_mismatch=first,
        reason="存在采样点差异超出容差，两者不相等",
    )
    return result
