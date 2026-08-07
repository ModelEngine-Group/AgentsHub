from __future__ import annotations

import asyncio

from fastmcp import Client


EXPECTED = {
    "run_pipeline",
    "query_graph",
    "answer_medical_question",
    "run_analysis",
    "get_benchmark",
    "get_quality_report",
}


async def verify() -> None:
    async with Client("http://127.0.0.1:8100/sse") as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools}
        assert names == EXPECTED, (names, EXPECTED)
    print("sse_tools=6")


if __name__ == "__main__":
    asyncio.run(verify())
