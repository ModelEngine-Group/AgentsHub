# SPDX-License-Identifier: MIT
"""Budget success, decimal rounding, and invalid-input regressions."""

import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calculate_budget.py"
SPEC = importlib.util.spec_from_file_location("travel_budget", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BudgetTests(unittest.TestCase):
    def test_room_nights_and_round_trip(self):
        result = MODULE.calculate({"budget": 900, "items": [{"name": "住宿", "factors": [1, 2], "unit_price": 300}, {"name": "交通", "factors": [2], "unit_price": 200}]})
        self.assertEqual([row["amount"] for row in result["items"]], ["600.00", "400.00"])
        self.assertEqual(result["total"], "1000.00")
        self.assertEqual(result["balance"], "-100.00")
        self.assertTrue(result["over_budget"])

    def test_family_regression(self):
        result = MODULE.calculate({"budget": 3000, "items": [{"name": "成人高铁", "factors": [2, 2], "unit_price": 73}, {"name": "儿童高铁", "factors": [1, 2], "unit_price": 37}, {"name": "住宿", "factors": [1, 2], "unit_price": 480}, {"name": "餐饮", "factors": [3, 3], "unit_price": 90}, {"name": "当地交通", "factors": [3], "unit_price": 60}, {"name": "门票", "factors": [1], "unit_price": 150}]})
        self.assertEqual(result["items"][1]["amount"], "74.00")
        self.assertEqual(result["total"], "2466.00")
        self.assertEqual(result["balance"], "534.00")

    def test_decimal_rounding(self):
        result = MODULE.calculate({"budget": "1.00", "items": [{"name": "测试", "factors": [3], "unit_price": "0.335"}]})
        self.assertEqual(result["total"], "1.01")
        self.assertEqual(result["balance"], "-0.01")

    def test_free_item(self):
        result = MODULE.calculate({"budget": 0, "items": [{"name": "免费", "factors": [3], "unit_price": 0}]})
        self.assertEqual(result["balance"], "0.00")

    def test_invalid_amounts(self):
        for invalid in [-1, "NaN", "Infinity", True, None, "not-money"]:
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                MODULE.calculate({"budget": 100, "items": [{"name": "费用", "unit_price": invalid}]})

    def test_empty_and_oversized_rows(self):
        for items in [[], [{"name": "费用", "unit_price": 1}] * 41]:
            with self.subTest(count=len(items)), self.assertRaises(ValueError):
                MODULE.calculate({"budget": 100, "items": items})


if __name__ == "__main__":
    unittest.main()
