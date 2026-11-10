# -*- coding: utf-8 -*-
import unittest

from travel_mcp import tools as T


class TestCityAndPoi(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(T.normalize_city("Tokyo"), "东京")
        self.assertEqual(T.normalize_city("清迈"), "清迈")
        self.assertIsNone(T.normalize_city("火星"))

    def test_search_poi_local(self):
        out = T.search_poi("东京", keyword="夜景", limit=5)
        self.assertTrue(out["ok"])
        self.assertGreater(out["count"], 0)
        self.assertLessEqual(out["count"], 5)
        self.assertEqual(out["results"][0]["source"], "本地精选库")

    def test_search_poi_unknown_city(self):
        out = T.search_poi("火星", keyword="山")
        self.assertFalse(out["ok"])
        self.assertIn("hint", out)

    def test_essentials(self):
        out = T.get_local_essentials("曼谷")
        self.assertTrue(out["ok"])
        self.assertIn("currency", out)
        self.assertEqual(out["police"], "191")


class TestBudget(unittest.TestCase):
    def test_budget_with_flight(self):
        out = T.estimate_budget("东京", 5, 2, "comfort", origin_city="上海", include_flight=True)
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["items"]), 5)
        self.assertTrue(out["flight"]["ok"])
        low, high = out["total_range"]
        self.assertLess(low, high)
        self.assertGreater(high, 10000)

    def test_budget_invalid_days(self):
        out = T.estimate_budget("东京", "abc")
        self.assertFalse(out["ok"])


class TestItinerary(unittest.TestCase):
    def test_three_days_tokyo(self):
        out = T.build_itinerary("东京", 3, "moderate", interests=["美食", "文化"])
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["plan"]), 3)
        names = []
        for day in out["plan"]:
            spots = [b for b in day["blocks"] if b["type"] == "景点"]
            self.assertGreaterEqual(len(spots), 1)
            self.assertLessEqual(len(spots), 3)
            names.extend(s["name"] for s in spots)
        self.assertEqual(len(names), len(set(names)), "同一景点不应重复安排")

    def test_monday_closure_avoided(self):
        # 2026-11-02 是周一
        out = T.build_itinerary("东京", 2, "packed", start_date="2026-11-02")
        self.assertTrue(out["ok"])
        first_day = out["plan"][0]
        for b in first_day["blocks"]:
            if b["type"] == "景点":
                self.assertNotIn("周一闭馆", b["open"])
                self.assertNotIn("周一闭园", b["open"])

    def test_must_see_missing_warns(self):
        out = T.build_itinerary("东京", 1, must_see=["不存在的美术馆"])
        self.assertTrue(out["ok"])
        self.assertTrue(any("不存在的美术馆" in w for w in out["warnings"]))


class TestRoute(unittest.TestCase):
    def test_known_pois(self):
        out = T.plan_route("浅草寺", "东京晴空塔", mode="walk", city="东京")
        self.assertTrue(out["ok"])
        self.assertGreater(out["distance_km"], 0)
        self.assertGreater(out["duration_min"], 0)


class TestWeather(unittest.TestCase):
    def test_live_weather(self):
        out = T.get_weather("东京", days=3)
        if not out.get("ok"):
            self.skipTest(f"外网不可用：{out.get('error')}")
        self.assertEqual(len(out["daily"]), 3)
        self.assertIn("weather", out["daily"][0])


if __name__ == "__main__":
    unittest.main()
