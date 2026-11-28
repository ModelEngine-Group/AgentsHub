import csv
import os
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from retail_embeddings import embeddings


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "orders.csv"
CENT = Decimal("0.01")


@dataclass
class Sales:
    completed: Decimal = Decimal("0.00")
    refunded: Decimal = Decimal("0.00")
    order_count: int = 0

    @property
    def net(self) -> Decimal:
        return self.completed - self.refunded

    def as_result(self) -> dict:
        return {
            "completed_cny": f"{self.completed:.2f}",
            "refunded_cny": f"{self.refunded:.2f}",
            "net_cny": f"{self.net:.2f}",
            "order_count": self.order_count,
        }


mcp = FastMCP(
    "retail-sales",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8765")),
    stateless_http=True,
    json_response=True,
)
mcp.custom_route("/v1/embeddings", methods=["POST"])(embeddings)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
def summarize_retail_sales(month: str, category: str | None = None) -> dict[str, Any]:
    """按 YYYY-MM 汇总模拟订单的净销售额。category 可选，按品类全名筛选。"""
    if not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", month) or month.startswith("0000"):
        raise ValueError("month 必须是有效的 YYYY-MM，例如 2026-11")
    if category is not None:
        category = category.strip()
        if not category:
            raise ValueError("category 须为品类全名，或省略该参数")

    groups: dict[str, Sales] = {}
    with DATA_FILE.open(encoding="utf-8-sig", newline="") as source:
        for order in csv.DictReader(source):
            order_month = date.fromisoformat(order["order_date"]).isoformat()[:7]
            order_category = order["category"]
            if order_month != month or (category is not None and order_category != category):
                continue
            status = order["status"]
            if status == "cancelled":
                continue
            if status not in {"completed", "refunded"}:
                raise ValueError(f"无法识别订单状态：{status}")
            amount = Decimal(order["amount_cny"])
            if not amount.is_finite() or amount < 0 or amount != amount.quantize(CENT):
                raise ValueError(f"订单 {order['order_id']} 的金额无效")
            sales = groups.setdefault(order_category, Sales())
            if status == "completed":
                sales.completed += amount
            else:
                sales.refunded += amount
            sales.order_count += 1

    totals = Sales()
    rows = []
    rank = 0
    previous_net = None
    ranked = sorted(groups.items(), key=lambda item: (-item[1].net, item[0]))
    for position, (name, sales) in enumerate(ranked, start=1):
        if sales.net != previous_net:
            rank = position
        previous_net = sales.net
        rows.append({"rank": rank, "category": name, **sales.as_result()})
        totals.completed += sales.completed
        totals.refunded += sales.refunded
        totals.order_count += sales.order_count

    return {
        "status": "ok" if rows else "no_data",
        "month": month,
        "category": category,
        "currency": "CNY",
        "totals": totals.as_result() if rows else None,
        "rows": rows,
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
