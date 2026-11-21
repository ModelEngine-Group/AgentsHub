#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate and calculate a travel budget without network or filesystem access."""

import argparse
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from html import escape
import json
import sys

CENT = Decimal("0.01")


def number(value, field):
    """Parse a bounded, finite, nonnegative decimal."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{field} must be a number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not result.is_finite() or result < 0 or result > Decimal("1000000000"):
        raise ValueError(f"{field} must be finite and between 0 and 1000000000")
    return result


def amount(value):
    return format(value.quantize(CENT, rounding=ROUND_HALF_UP), ".2f")


def cell(value, limit=160):
    return escape(str(value)[:limit], quote=False).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def calculate(payload):
    """Return calculated rows and an authoritative Markdown budget table."""
    if not isinstance(payload, dict):
        raise ValueError("input must be an object")
    budget = number(payload.get("budget"), "budget").quantize(CENT, rounding=ROUND_HALF_UP)
    currency = payload.get("currency", "CNY")
    if not isinstance(currency, str) or not currency.strip() or len(currency) > 12:
        raise ValueError("currency must be a nonempty code of at most 12 characters")
    items = payload.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= 40:
        raise ValueError("items must contain 1 to 40 entries")
    rows = []
    total = Decimal("0")
    lines = [f"币种：{cell(currency)}；全体预算：{amount(budget)}", "", "| 项目 | 计算式 | 全体金额 | 数量口径与依据 |", "|---|---|---:|---|"]
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
            raise ValueError(f"items[{index}] needs a nonempty name")
        factors = item.get("factors", [1])
        if not isinstance(factors, list) or not 1 <= len(factors) <= 8:
            raise ValueError(f"items[{index}].factors must contain 1 to 8 numbers")
        numbers = [number(value, f"items[{index}].factors") for value in factors]
        price = number(item.get("unit_price"), f"items[{index}].unit_price")
        subtotal = price
        for factor in numbers:
            subtotal *= factor
            if subtotal > Decimal("1000000000000"):
                raise ValueError("item amount is too large")
        subtotal = subtotal.quantize(CENT, rounding=ROUND_HALF_UP)
        total += subtotal
        expression = " × ".join([format(value, "f") for value in numbers] + [format(price, "f")])
        basis = "；".join(filter(None, [cell(item.get("quantity_note", "")), cell(item.get("basis", "待确认"))]))
        rows.append({"name": item["name"], "expression": expression, "amount": amount(subtotal)})
        lines.append(f"| {cell(item['name'], 80)} | {expression} | {amount(subtotal)} | {basis} |")
    balance = budget - total
    lines.extend([f"| **合计** | 各项小计之和 | **{amount(total)}** | 全体费用 |", f"| **预算余额** | {amount(budget)} − {amount(total)} | **{amount(balance)}** | {'超预算' if balance < 0 else '预算内'} |"])
    return {"status": "success", "currency": currency, "budget": amount(budget), "items": rows, "total": amount(total), "balance": amount(balance), "over_budget": balance < 0, "markdown": "\n".join(lines)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON object containing budget and items")
    args = parser.parse_args()
    try:
        if len(args.input) > 20000:
            raise ValueError("input exceeds 20000 characters")
        result = calculate(json.loads(args.input))
    except (ValueError, TypeError, InvalidOperation) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
