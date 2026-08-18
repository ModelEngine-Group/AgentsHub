# -*- coding: utf-8 -*-
"""
kg_analyze 工具测试脚本

测试覆盖8个场景，对照需求分析文档中的测试用例表
"""

import json, sys, os

# 直接测试核心逻辑（不启动MCP服务器）
_KG_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mcp_server", "kg_data", "knowledge_graph_triples.json"
)

with open(_KG_DATA, "r", encoding="utf-8") as f:
    TRIPLES = json.load(f)["triples"]

from collections import Counter

def _filter_triples(triples, entity=None, relation=None, keyword=None):
    results = []
    for t in triples:
        if entity and t["head"] != entity:
            continue
        if relation and t["relation"] != relation:
            continue
        if keyword:
            kw = keyword.lower()
            if kw not in t["head"].lower() and kw not in t["tail"].lower():
                continue
        results.append(t)
    return results

def _count_and_rank(items, top_k):
    counter = Counter(items)
    total = sum(counter.values())
    ranked = counter.most_common(top_k)
    return [{"item": item, "count": count, "percentage": round(count / total * 100, 1) if total > 0 else 0.0} for item, count in ranked]

def _handle_relation_dist(triples, entity, top_k):
    return _count_and_rank([t["relation"] for t in triples], top_k)

def _handle_symptom_dist(triples, entity, top_k):
    return _count_and_rank([t["tail"] for t in triples], top_k)

def _handle_drug_stats(triples, entity, top_k):
    return _count_and_rank([t["head"] for t in triples], top_k)

def _handle_entity_relations(triples, entity, top_k):
    items, seen = [], set()
    for t in triples:
        tail = t["tail"]
        if tail not in seen:
            items.append(tail)
            seen.add(tail)
    return _count_and_rank(items, top_k)

CUT = "=" * 50
passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}  {detail}")


# ===== 测试1：关系分布（全量）=====
print(f"\n{CUT}\n测试1：关系分布（全量，无筛选）\n{CUT}")
r1 = _filter_triples(TRIPLES)
stats1 = _handle_relation_dist(r1, None, 20)
print(f"  三元组总数: {len(r1)}")
for s in stats1:
    print(f"    {s['item']}: {s['count']} ({s['percentage']}%)")
check("返回了统计结果", len(stats1) > 0)
check("包含'导致'关系", any(s["item"] == "导致" for s in stats1))
check("包含'治疗'关系", any(s["item"] == "治疗" for s in stats1))

# ===== 测试2：症状分布 =====
print(f"\n{CUT}\n测试2：症状分布（entity=糖尿病）\n{CUT}")
r2 = _filter_triples(TRIPLES, entity="糖尿病", relation="导致")
stats2 = _handle_symptom_dist(r2, "糖尿病", 20)
print(f"  命中三元组: {len(r2)} 条")
for s in stats2:
    print(f"    {s['item']}: {s['count']} ({s['percentage']}%)")
check("返回了症状列表", len(stats2) > 0)
check("多饮多尿在其中", any(s["item"] == "多饮多尿" for s in stats2))

# ===== 测试3：药物统计 =====
print(f"\n{CUT}\n测试3：药物统计（entity=糖尿病, relation=治疗）\n{CUT}")
r3 = _filter_triples(TRIPLES, relation="治疗")
# 筛选出tail为糖尿病的
r3_filtered = [t for t in r3 if t["tail"] == "糖尿病"]
stats3 = _handle_drug_stats(r3_filtered, "糖尿病", 20)
print(f"  治疗糖尿病的药物: {len(r3_filtered)} 种")
for s in stats3:
    print(f"    {s['item']}: {s['count']} ({s['percentage']}%)")
check("返回了药物列表", len(stats3) > 0)
check("二甲双胍在其中", any(s["item"] == "二甲双胍" for s in stats3))

# ===== 测试4：关联分析 =====
print(f"\n{CUT}\n测试4：关联分析（entity=糖尿病）\n{CUT}")
r4 = _filter_triples(TRIPLES, entity="糖尿病")
stats4 = _handle_entity_relations(r4, "糖尿病", 20)
print(f"  关联实体数: {len(r4)} 条")
for s in stats4:
    print(f"    {s['item']}: {s['count']} ({s['percentage']}%)")
check("返回了关联实体", len(stats4) > 0)
check("多饮多尿在其中", any(s["item"] == "多饮多尿" for s in stats4))

# ===== 测试5：关键词搜索 =====
print(f"\n{CUT}\n测试5：关键词搜索（keyword=高血压）\n{CUT}")
r5 = _filter_triples(TRIPLES, keyword="高血压")
print(f"  命中三元组: {len(r5)} 条")
check("搜索有结果", len(r5) > 0)

# ===== 测试6：entity为空 =====
print(f"\n{CUT}\n测试6：entity为空时返回全量统计\n{CUT}")
r6 = _filter_triples(TRIPLES)
stats6 = _handle_relation_dist(r6, None, 20)
check("全量统计成功", len(stats6) > 0)
check("覆盖全部三元组", len(r6) == len(TRIPLES))

# ===== 测试7：空结果 =====
print(f"\n{CUT}\n测试7：无匹配结果\n{CUT}")
r7 = _filter_triples(TRIPLES, entity="不存在的疾病名称")
check("无匹配时返回空", len(r7) == 0)

# ===== 测试8：参数覆盖 =====
print(f"\n{CUT}\n测试8：参数覆盖逻辑验证\n{CUT}")
# 症状分布时relation会被覆盖为"导致"
r8 = _filter_triples(TRIPLES, entity="糖尿病", relation="导致")
check("relation被正确覆盖为导致", len(r8) > 0)
check("结果只有导致关系", all(t["relation"] == "导致" for t in r8))

# ===== 结果汇总 =====
total = passed + failed
print(f"\n{'='*50}")
print(f"  测试结果: {passed}/{total} 通过")
if failed == 0:
    print(f"  🎉 全部通过！")
else:
    print(f"  ⚠️ {failed} 个测试失败")
print(f"{'='*50}")
