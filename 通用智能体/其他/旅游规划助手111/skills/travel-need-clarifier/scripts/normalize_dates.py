#!/usr/bin/env python3
"""日期归一化与风险标签辅助脚本。

用法:
    python normalize_dates.py --base 2026-11-22 --phrase "下周末"
    python normalize_dates.py --base 2026-11-22 --phrase "五一" --tz Asia/Shanghai
输出 JSON：绝对日期列表 + 每个日期的风险主/副标签。
"""
import argparse
import datetime as dt
import json

HOLIDAYS = {
    "元旦": ((1, 1), (1, 3)),
    "劳动节": ((5, 1), (5, 5)),
    "国庆": ((10, 1), (10, 7)),
}

SEASON_WINDOWS = [
    ("梅雨", (6, 15), (7, 15)),
    ("台风", (7, 1), (9, 30)),
    ("红叶", (10, 15), (12, 15)),
]


def parse_phrase(base: dt.date, phrase: str):
    phrase = phrase.strip()
    if phrase == "下周末":
        days_to_sat = (5 - base.weekday()) % 7 or 7
        sat = base + dt.timedelta(days=days_to_sat)
        return [sat, sat + dt.timedelta(days=1)]
    if phrase == "月底":
        last = (base.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
        return [last - dt.timedelta(days=i) for i in range(4, -1, -1)]
    for name, (s, e) in HOLIDAYS.items():
        if name in phrase:
            year = base.year if base.month <= s[0] else base.year + 1
            start = dt.date(year, *s)
            end = dt.date(year, *e)
            return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    # 绝对日期透传
    try:
        return [dt.date.fromisoformat(phrase)]
    except ValueError:
        raise SystemExit(f"无法解析日期短语: {phrase}")


def tag(d: dt.date):
    main, sub = "无特殊标记", []
    for name, (s, e) in HOLIDAYS.items():
        if (d.month, d.day) >= s and (d.month, d.day) <= e:
            main = "节假日客流"
    for name, start, end in SEASON_WINDOWS:
        if start <= (d.month, d.day) <= end:
            sub.append(f"季节窗口:{name}")
    if d.weekday() >= 5 and main == "无特殊标记":
        main = "补班日" if d.weekday() == 6 else main
    return {"date": d.isoformat(), "weekday": d.strftime("%A"),
            "main_tag": main, "sub_tags": sub}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="会话当前日期 YYYY-MM-DD")
    ap.add_argument("--phrase", required=True)
    ap.add_argument("--tz", default="Asia/Shanghai")
    a = ap.parse_args()
    base = dt.date.fromisoformat(a.base)
    dates = parse_phrase(base, a.phrase)
    print(json.dumps({"tz": a.tz, "input": a.phrase,
                      "result": [tag(d) for d in dates]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
