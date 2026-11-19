"""Read-only campus demo MCP. All records are fictional, never school policy."""
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("campus-demo", host="0.0.0.0", port=8010)
SERVICES = {
    "certificate": {"name": "在读证明", "office": "示例教务服务中心", "materials": ["校内身份认证", "用途说明"]},
    "repair": {"name": "宿舍报修", "office": "示例后勤服务中心", "materials": ["楼栋与房间", "故障描述"]},
    "card": {"name": "校园卡补办", "office": "示例校园卡中心", "materials": ["校内身份认证"]},
}

def result(data=None, error=None):
    return {"demo": True, "notice": "虚构演示数据，不代表任何学校政策", "data": data,
            "error": error, "retrieved_at": datetime.now(timezone.utc).isoformat()}

@mcp.tool()
def list_services(keyword: str = "") -> dict:
    """Search fictional campus services. Returns demo data only."""
    return result([{"service_id": key, **value} for key, value in SERVICES.items()
                   if keyword in value["name"] or keyword in key])

@mcp.tool()
def get_service_availability(service_id: str) -> dict:
    """Return unavailable for live opening hours; no school system is connected."""
    if service_id not in SERVICES:
        return result(error={"code": "NOT_ACCESSIBLE", "message": "事项不可访问"})
    return result(error={"code": "UNAVAILABLE", "message": "尚未连接学校实时开放时间服务"})

@mcp.tool()
def get_my_request_status(request_id: str) -> dict:
    """Fail closed: no authenticated school identity adapter is configured."""
    return result(error={"code": "UNAUTHENTICATED", "message": "请通过学校认证入口查询；演示服务不提供个人工单"})

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
