---
name: travel-budget-checker
description: 精确核算旅行预算的数量乘单价、分类小计、总计与余额，输出可原样使用的中文 Markdown 表格。涉及多人、房间数、住宿晚数、往返交通或预算冲突时使用。
version: 1.0.0
author: "20241311"
---

# 旅行预算核算

在给出任何含金额的预算表前，必须运行本技能。它只做本地十进制算术，不联网，不读取密钥，不查询价格；单价及计费口径由你根据用户输入或明确标注的估算提供。

## 调用

先整理预算条目。`factors` 中每个数都是数量，`unit_price` 是最后相乘的单价；`quantity_note` 说明数量单位。每行必须说明单价来自用户、官方来源还是估算。儿童票价未知时，使用明确的估算金额，不把年龄优惠换算当作已核实票价。

```python
import json

payload = {
    "currency": "CNY",
    "budget": "900",
    "items": [
        {"name": "住宿", "factors": [1, 2], "unit_price": "300", "quantity_note": "1间×2晚", "basis": "用户给定"},
        {"name": "往返交通", "factors": [2], "unit_price": "200", "quantity_note": "2人，每人往返", "basis": "用户给定"}
    ]
}
arguments = "--input '" + json.dumps(payload, ensure_ascii=False) + "'"
result = run_skill_script(
    skill_name="travel-budget-checker",
    script_path="scripts/calculate_budget.py",
    params=arguments,
    source="skill"
)
checked = json.loads(result)
print(checked["markdown"])
```

例子的住宿为 600 元、交通为 400 元、总计 1000 元、余额为 -100 元。其他餐饮等项目尚未提供，不应把这一小计称作完整旅行总预算。

## 使用结果

- **原样使用返回的 `markdown` 作为预算表，不手工重新输入算式、小计、总计或余额。**
- 修改人数、单价、住宿晚数、门票选择或机动金后，重新运行技能，以最后一次结果为准。
- 给完整行程时，条目应覆盖城际交通、住宿、餐饮、当地交通、主方案门票和机动金；两个二选一项目只计入主方案选中的那个。
- `factors` 的人数必须与 `quantity_note` 一致，不能把全体每天餐费再次乘人数。
- 表格数值是算术结果，不代表实时可预订的价格。仍需在正文说明价格依据和尚待确认的政策。
- 若输入不合法，修正数据后重跑，不把错误消息写成有效预算。

## 输入与输出

脚本接受 `--input JSON`，输出 JSON。金额采用 Decimal，四舍五入到分；`balance` 小于零表示超预算。无需额外 Python 依赖，Python 3.10+ 即可运行。
