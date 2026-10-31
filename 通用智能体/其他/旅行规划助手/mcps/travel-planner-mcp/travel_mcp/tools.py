# -*- coding: utf-8 -*-
"""旅行规划工具的实现（纯 Python 标准库，可直接单元测试）。

对外暴露 6 个工具：
    get_weather / search_poi / plan_route / estimate_budget /
    build_itinerary / get_local_essentials

所有工具都返回可 JSON 序列化的 dict，并遵循同一约定：
    ok=True 时包含业务字段；ok=False 时包含 error 与 hint（给模型的下一步建议）。
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import urllib.parse
import urllib.request

from . import data

UA = "NexentTravelPlanner/1.0 (demo; contact: example@example.com)"
DEFAULT_TIMEOUT = 8

WMO = {
    0: "晴", 1: "晴间多云", 2: "多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "强毛毛雨",
    56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "阵雨", 81: "强阵雨", 82: "暴雨",
    85: "阵雪", 86: "强阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷暴",
}

PACE = {"relaxed": 2, "moderate": 3, "packed": 4}
PACE_LABEL = {"relaxed": "轻松", "moderate": "适中", "packed": "紧凑"}


# ---------------------------------------------------------------------------
# 基础工具函数
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: int = DEFAULT_TIMEOUT):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _fail(error: str, hint: str = ""):
    return {"ok": False, "error": error, "hint": hint}


def normalize_city(name: str) -> str | None:
    if not name:
        return None
    key = str(name).strip().lower()
    if key in data.CITY_ALIASES:
        return data.CITY_ALIASES[key]
    for city in data.CITIES:
        if city in key:
            return city
    return None


def _city_or_fail(city: str):
    c = normalize_city(city)
    if not c:
        return None, _fail(
            f"暂不支持城市「{city}」的离线数据",
            "可改用通用知识回答，或调用 search_poi 通过在线地图检索；不要编造该城市数据。",
        )
    return c, None


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * r * math.asin(math.sqrt(a)), 1)


def _geocode(name: str):
    """在线地理编码（Nominatim），返回 (lat, lon, display_name) 或 None。"""
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": name, "format": "jsonv2", "limit": 1, "accept-language": "zh-CN,en"}
    )
    try:
        rows = _get_json(url)
    except Exception:
        return None
    if not rows:
        return None
    row = rows[0]
    try:
        return float(row["lat"]), float(row["lon"]), row.get("display_name", name)
    except (KeyError, TypeError, ValueError):
        return None


def _find_poi(city: str, name: str):
    if not name:
        return None
    best = None
    for p in data.POIS:
        if p["city"] != city:
            continue
        if p["name"] == name:
            return p
        if name in p["name"] or p["name"] in name:
            best = best or p
    return best


def _resolve_point(city: str, text: str):
    """把“景点名 / lat,lon / 任意地名”解析成坐标。"""
    text = (text or "").strip()
    if not text:
        return None
    if "," in text:
        parts = [x.strip() for x in text.split(",")]
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1]), text
            except ValueError:
                pass
    p = _find_poi(city, text)
    if p:
        return p["lat"], p["lon"], p["name"]
    c = normalize_city(text)
    if c:
        meta = data.CITIES[c]
        return meta["lat"], meta["lon"], c
    return _geocode(text)


# ---------------------------------------------------------------------------
# 1. 天气
# ---------------------------------------------------------------------------

def get_weather(city: str, start_date: str = "", days: int = 3) -> dict:
    """查询城市未来天气（Open-Meteo，免费、无需 Key）。"""
    c, err = _city_or_fail(city)
    if err:
        return err
    meta = data.CITIES[c]
    try:
        days = max(1, min(int(days), 14))
    except (TypeError, ValueError):
        days = 3

    params = {
        "latitude": meta["lat"], "longitude": meta["lon"],
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max",
        "timezone": "auto", "forecast_days": days,
    }
    if start_date:
        try:
            start = _dt.date.fromisoformat(start_date)
            today = _dt.date.today()
            if start < today or (start - today).days > 15:
                return _fail(
                    f"日期 {start_date} 超出天气预报范围（今天起 16 天内）",
                    "请改为查询近期天气，或说明“出行日期较远，天气需临近再看”。",
                )
            end = start + _dt.timedelta(days=days - 1)
            params.pop("forecast_days", None)
            params["start_date"] = start.isoformat()
            params["end_date"] = end.isoformat()
        except ValueError:
            return _fail(f"日期格式不正确：{start_date}", "请使用 YYYY-MM-DD 格式。")

    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    try:
        raw = _get_json(url)
    except Exception as exc:  # 网络失败时给出可执行的降级建议
        return _fail(
            f"天气服务暂时不可用（{type(exc).__name__}）",
            "不要编造天气；告知用户“实时天气暂不可查”，建议出行前 1-3 天再确认，并给出按季节的穿衣建议。",
        )

    daily = raw.get("daily") or {}
    dates = daily.get("time") or []
    rows = []
    for i, d in enumerate(dates):
        code = (daily.get("weather_code") or [None] * len(dates))[i]
        rows.append({
            "date": d,
            "weather": WMO.get(code, "未知"),
            "temp_min_c": (daily.get("temperature_2m_min") or [None] * len(dates))[i],
            "temp_max_c": (daily.get("temperature_2m_max") or [None] * len(dates))[i],
            "rain_prob_%": (daily.get("precipitation_probability_max") or [None] * len(dates))[i],
            "wind_kmh": (daily.get("wind_speed_10m_max") or [None] * len(dates))[i],
        })
    return {
        "ok": True, "city": c, "days": len(rows), "daily": rows,
        "source": "Open-Meteo", "queried_at": _dt.datetime.now().isoformat(timespec="minutes"),
        "note": "天气预报存在不确定性，出行前请再次确认；户外行程请预留雨天备选。",
    }


# ---------------------------------------------------------------------------
# 2. 景点检索
# ---------------------------------------------------------------------------

def search_poi(city: str, keyword: str = "", category: str = "", limit: int = 6) -> dict:
    """在本地精选景点库中检索，必要时回退到 OpenStreetMap。"""
    c, err = _city_or_fail(city)
    if err:
        return err
    try:
        limit = max(1, min(int(limit), 12))
    except (TypeError, ValueError):
        limit = 6

    kw = (keyword or "").strip().lower()
    cat = (category or "").strip()

    def match(p):
        hay = " ".join([p["name"], p["district"], p["category"], " ".join(p["tags"]), p["tip"]]).lower()
        if kw and kw not in hay:
            return False
        if cat and cat not in p["category"] and cat not in " ".join(p["tags"]):
            return False
        return True

    local = [p for p in data.POIS if p["city"] == c and match(p)]
    local = local[:limit]
    results = [{
        "name": p["name"], "district": p["district"], "category": p["category"],
        "suggested_hours": p["duration_h"], "ticket_cny": p["ticket_cny"],
        "open": p["open"], "tags": p["tags"], "tip": p["tip"],
        "lat": p["lat"], "lon": p["lon"], "source": "本地精选库",
    } for p in local]

    if results:
        return {
            "ok": True, "city": c, "count": len(results), "results": results,
            "note": "门票与开放时间为示例数据，预订前请以官方渠道为准。",
        }

    # 本地无结果时回退在线检索（适合长尾景点）
    query = f"{keyword or category} {c}"
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": query, "format": "jsonv2", "limit": limit, "accept-language": "zh-CN,en", "addressdetails": 0}
    )
    try:
        rows = _get_json(url)
    except Exception:
        return _fail(
            f"本地库没有「{keyword or category}」，在线检索也暂时不可用",
            "请基于常识给出候选，并明确提示“未核实开放时间与门票”。",
        )
    if not rows:
        return _fail(
            f"没有找到「{keyword or category}」相关地点",
            "请让用户补充更具体的名称，或换一个关键词（如景点类型）。",
        )
    out = [{
        "name": r.get("display_name", "").split(",")[0],
        "address": r.get("display_name", ""),
        "category": r.get("type", "poi"), "lat": float(r["lat"]), "lon": float(r["lon"]),
        "source": "OpenStreetMap",
    } for r in rows]
    return {
        "ok": True, "city": c, "count": len(out), "results": out,
        "note": "结果来自 OpenStreetMap，未经人工校验；门票、开放时间需另行确认。",
    }


# ---------------------------------------------------------------------------
# 3. 路线与距离
# ---------------------------------------------------------------------------

_OSRM_PROFILE = {"walk": "foot", "walking": "foot", "drive": "driving", "driving": "driving",
                 "bike": "cycling", "cycling": "cycling"}
_SPEED_KMH = {"walk": 4.5, "drive": 28.0, "bike": 14.0, "transit": 20.0}


def plan_route(origin: str, destination: str, mode: str = "walk", city: str = "") -> dict:
    """计算两点间距离与预计耗时（OSRM；transit 使用估算）。"""
    mode = (mode or "walk").lower()
    if mode not in _SPEED_KMH:
        mode = "walk"
    c = normalize_city(city) or normalize_city(origin) or normalize_city(destination) or ""
    a = _resolve_point(c, origin)
    b = _resolve_point(c, destination)
    if not a:
        return _fail(f"无法定位起点「{origin}」", "请改用“景点名”或“纬度,经度”，或先调用 search_poi 获取坐标。")
    if not b:
        return _fail(f"无法定位终点「{destination}」", "请改用“景点名”或“纬度,经度”，或先调用 search_poi 获取坐标。")

    straight = _haversine_km(a[0], a[1], b[0], b[1])
    result = {
        "ok": True, "origin": a[2], "destination": b[2], "mode": mode,
        "straight_km": straight,
    }

    # OSRM 公共实例只有驾车路网；步行/骑行用其路网距离，再按速度折算耗时
    route_km = None
    try:
        url = (f"https://router.project-osrm.org/route/v1/driving/"
               f"{a[1]},{a[0]};{b[1]},{b[0]}?overview=false")
        raw = _get_json(url)
        route = (raw.get("routes") or [None])[0]
        if route:
            route_km = round(route["distance"] / 1000, 1)
            if mode == "drive":
                result.update({
                    "distance_km": route_km,
                    "duration_min": max(1, round(route["duration"] / 60)),
                    "source": "OSRM", "approx": False,
                    "note": "按实际驾车路网估算，未含堵车与停车时间。",
                })
                return result
    except Exception:
        route_km = None

    distance = route_km if route_km is not None else round(straight * 1.3, 1)
    extra = 6 if mode == "transit" else 0
    minutes = round(distance / _SPEED_KMH[mode] * 60 + extra)
    result.update({
        "distance_km": distance, "duration_min": max(1, minutes),
        "source": "OSRM 路网 + 速度折算" if route_km is not None else "估算",
        "approx": True,
        "note": ("按路网距离与常见速度折算；" if route_km is not None else "按直线距离 × 路网系数估算；")
                + "transit 未含换乘与候车，实际请以地图 App 为准。",
    })
    return result


# ---------------------------------------------------------------------------
# 4. 预算
# ---------------------------------------------------------------------------

def _flight_range(origin_city: str, city: str):
    o = normalize_city(origin_city)
    if not o or o == city:
        return None
    a, b = data.CITIES[o], data.CITIES[city]
    km = _haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
    for limit, low, high, label in data.FLIGHT_BANDS:
        if km <= limit:
            return {"distance_km": km, "low": low, "high": high, "band": label}
    return None


def estimate_budget(city: str, days: int, travelers: int = 1, style: str = "comfort",
                    origin_city: str = "", include_flight: bool = False) -> dict:
    """按“住宿/餐饮/市内交通/门票/其他”分项估算预算。"""
    c, err = _city_or_fail(city)
    if err:
        return err
    try:
        days = max(1, min(int(days), 30))
        travelers = max(1, min(int(travelers), 10))
    except (TypeError, ValueError):
        return _fail("天数或人数不是有效整数", "请使用整数，例如 days=5, travelers=2。")
    style = (style or "comfort").lower()
    if style not in data.BUDGET[c]:
        style = "comfort"

    unit = data.BUDGET[c][style]
    per_day = sum(unit.values())
    rows = []
    for name, value in unit.items():
        rows.append({"item": name, "per_person_per_day": value,
                     "subtotal": value * days * travelers})
    local_total = per_day * days * travelers

    flight = None
    if include_flight:
        if not origin_city:
            flight = {"ok": False, "reason": "未提供出发城市，无法估算大交通",
                      "hint": "请询问用户出发城市，或让用户自行填写机票预算。"}
        else:
            fr = _flight_range(origin_city, c)
            if fr:
                flight = {
                    "ok": True, "from": normalize_city(origin_city), "to": c,
                    "round_trip_low": fr["low"] * travelers,
                    "round_trip_high": fr["high"] * travelers,
                    "per_person_low": fr["low"], "per_person_high": fr["high"],
                    "distance_km": fr["distance_km"], "band": fr["band"],
                    "note": "按距离分档的粗略区间，不含行李与旺季溢价。",
                }
            else:
                flight = {"ok": False, "reason": "出发城市暂不支持", "hint": "请让用户提供机票预算。"}

    total_low = int(local_total * 0.9)
    total_high = int(local_total * 1.15)
    if flight and flight.get("ok"):
        total_low += flight["round_trip_low"]
        total_high += flight["round_trip_high"]

    return {
        "ok": True, "city": c, "days": days, "travelers": travelers,
        "style": style, "style_label": data.STYLE_LABEL[style],
        "currency": "CNY",
        "items": rows,
        "local_total": local_total,
        "local_total_range": [total_low if not flight else int(local_total * 0.9),
                              total_high if not flight else int(local_total * 1.15)],
        "flight": flight,
        "total_range": [total_low, total_high],
        "per_person_total_range": [total_low // travelers, total_high // travelers],
        "assumptions": [
            "不含购物与自费娱乐",
            "住宿按双人拼房的人均价格估算",
            f"价格为 {data.STYLE_LABEL[style]}档示例区间，旺季可能上浮 30%-60%",
        ],
        "note": "预算为规划用估算，不是报价；下单前请以实际查询为准。",
    }


# ---------------------------------------------------------------------------
# 5. 行程编排
# ---------------------------------------------------------------------------

def _score(p, interests):
    s = 0.0
    tags = set(p["tags"]) | {p["category"]}
    for it in interests or []:
        it = str(it).strip()
        if not it:
            continue
        if it in tags or any(it in t for t in tags):
            s += 3
        elif it in p["name"] or it in p["tip"]:
            s += 1
    return s


def build_itinerary(city: str, days: int, pace: str = "moderate",
                    interests=None, must_see=None, start_date: str = "") -> dict:
    """基于片区聚类生成逐日行程骨架（确定性算法，便于复现与评估）。"""
    c, err = _city_or_fail(city)
    if err:
        return err
    try:
        days = max(1, min(int(days), 10))
    except (TypeError, ValueError):
        return _fail("天数不是有效整数", "请传入 1-10 的整数。")
    pace = (pace or "moderate").lower()
    if pace not in PACE:
        pace = "moderate"
    per_day = PACE[pace]
    interests = [str(x) for x in (interests or []) if str(x).strip()]
    must_see = [str(x) for x in (must_see or []) if str(x).strip()]

    pool = [dict(p) for p in data.POIS if p["city"] == c]
    by_name = {p["name"]: p for p in pool}
    warnings = []
    forced = []
    for name in must_see:
        hit = next((p for p in pool if name in p["name"] or p["name"] in name), None)
        if hit:
            forced.append(hit)
        else:
            warnings.append(f"必去点「{name}」不在本地精选库中，未纳入自动编排；请手动补充并核实开放信息。")

    # 整日项目（主题乐园、远郊一日游）单独占一天，不参与逐日拼装
    full_day = [p for p in pool if p["duration_h"] >= 6]
    dedicated = [p for p in full_day if any(p["name"] in n or n in p["name"] for n in must_see)]
    for p in full_day:
        if p not in dedicated:
            warnings.append(f"「{p['name']}」为整日项目（约 {p['duration_h']:.0f} 小时），已从自动编排中排除；如需安排请加入 must_see 单独占一天。")
    pool = [p for p in pool if p["duration_h"] < 6]
    forced = [p for p in forced if p in pool]

    for p in pool:
        p["_score"] = _score(p, interests) + (5 if any(p is f for f in forced) else 0)

    # 按片区聚合，片区内按得分排序
    districts = {}
    for p in pool:
        districts.setdefault(p["district"], []).append(p)
    for d in districts:
        districts[d].sort(key=lambda x: (-x["_score"], -x["duration_h"]))

    district_order = sorted(districts, key=lambda d: (-sum(1 for p in districts[d] if p["_score"] > 0),
                                                      -len(districts[d])))

    used = set()
    plan = []
    cursor = 0
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    today = _dt.date.today()
    start = None
    if start_date:
        try:
            start = _dt.date.fromisoformat(start_date)
        except ValueError:
            warnings.append(f"start_date「{start_date}」格式无效，已忽略闭馆校验。")

    normal_days = max(0, days - len(dedicated))
    for day in range(1, normal_days + 1):
        weekday = None
        if start:
            weekday = (start + _dt.timedelta(days=day - 1)).weekday()

        chosen = [p for p in forced if p["name"] not in used][:per_day]
        for p in chosen:
            used.add(p["name"])

        # 以片区为块轮转填满当天：先保证同一天少折返，再考虑兴趣得分
        guard = 0
        while len(chosen) < per_day and guard < len(district_order) * 2:
            d = district_order[cursor % len(district_order)]
            cursor += 1
            guard += 1
            for p in districts[d]:
                if len(chosen) >= per_day:
                    break
                if p["name"] in used:
                    continue
                if weekday == 0 and ("周一闭馆" in p["open"] or "周一闭园" in p["open"]):
                    continue
                chosen.append(p)
                used.add(p["name"])
        if len(chosen) < per_day:
            for d in district_order:
                for p in districts[d]:
                    if len(chosen) >= per_day:
                        break
                    if p["name"] in used:
                        continue
                    chosen.append(p)
                    used.add(p["name"])
                if len(chosen) >= per_day:
                    break
        if not chosen:
            # 允许重复使用高分景点，避免空行程
            chosen = sorted(pool, key=lambda x: -x["_score"])[:per_day]

        blocks = []
        clock = 9 * 60
        lunch_done = False

        def add_lunch(at_min, tip):
            blocks.append({
                "_start": at_min, "type": "午餐", "name": "就近用餐（约 1 小时）",
                "time": "", "tip": tip,
            })

        def add_spot(p):
            nonlocal clock
            duration = int(p["duration_h"] * 60)
            start_min, end_min = clock, clock + duration
            blocks.append({
                "_start": start_min, "type": "景点", "name": p["name"],
                "district": p["district"], "duration_h": p["duration_h"],
                "ticket_cny": p["ticket_cny"], "open": p["open"], "tip": p["tip"],
                "time": f"{start_min // 60:02d}:{start_min % 60:02d}-{end_min // 60:02d}:{end_min % 60:02d}",
            })
            clock = end_min + 30

        for p in chosen:
            if not lunch_done and clock >= 11 * 60 + 30:
                add_lunch(clock, "避开景区门口的高价餐厅，步行 10 分钟范围更划算。")
                clock += 60
                lunch_done = True
            add_spot(p)
        if not lunch_done:
            add_lunch(max(clock, 12 * 60), "按当地作息安排午餐，错峰排队更短。")

        # 按开始时间排序，并补全午餐时段文案
        blocks.sort(key=lambda b: b["_start"])
        for b in blocks:
            if b["type"] == "午餐":
                start_min = b["_start"]
                b["time"] = f"{start_min // 60:02d}:{start_min % 60:02d}-{start_min // 60 + 1:02d}:{start_min % 60:02d}"
            b.pop("_start", None)

        plan.append({
            "day": day,
            "weekday": weekdays[weekday] if weekday is not None else None,
            "area": "、".join(dict.fromkeys(p["district"] for p in chosen)),
            "blocks": blocks,
            "ticket_total_cny": sum(int(p["ticket_cny"]) for p in chosen),
            "walk_note": "同日景点已按片区聚合，片区间建议地铁/打车，单程约 15-30 分钟。",
        })

    for i, p in enumerate(dedicated):
        day_index = normal_days + i + 1
        weekday = (start + _dt.timedelta(days=day_index - 1)).weekday() if start else None
        plan.append({
            "day": day_index,
            "weekday": weekdays[weekday] if weekday is not None else None,
            "area": p["district"],
            "blocks": [{
                "time": "09:00-17:00", "type": "全天项目", "name": p["name"],
                "district": p["district"], "duration_h": p["duration_h"],
                "ticket_cny": p["ticket_cny"], "open": p["open"], "tip": p["tip"],
            }],
            "ticket_total_cny": int(p["ticket_cny"]),
            "walk_note": "整日项目：建议开园前 30 分钟到达，园内用餐；当天不再安排其他景点。",
        })

    return {
        "ok": True, "city": c, "days": days, "pace": pace, "pace_label": PACE_LABEL[pace],
        "interests": interests, "must_see": must_see,
        "plan": plan, "warnings": warnings,
        "totals": {"ticket_cny": sum(d["ticket_total_cny"] for d in plan)},
        "note": "这是行程骨架：请结合天气、预约时段、营业时间与用户体力二次调整；不要把它当作已核实的事实。",
    }


# ---------------------------------------------------------------------------
# 6. 目的地生活信息
# ---------------------------------------------------------------------------

def get_local_essentials(city: str) -> dict:
    """返回货币、插头、饮水、紧急电话、交通卡、常用语等。"""
    c, err = _city_or_fail(city)
    if err:
        return err
    info = data.ESSENTIALS[c]
    return {
        "ok": True, "city": c, "country": data.CITIES[c]["country"],
        **info,
        "verify_note": "紧急电话与入境政策可能调整，出行前请核对当地官方渠道。",
    }
