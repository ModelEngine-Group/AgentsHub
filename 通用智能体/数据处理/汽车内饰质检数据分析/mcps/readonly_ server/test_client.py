# -*- coding: utf-8 -*-
"""Smoke test: connect to the local read-only MCP and call get_stats."""
from __future__ import annotations

import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main() -> None:
    async with streamablehttp_client("http://127.0.0.1:5071/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [tool.name for tool in tools.tools]
            print(f"tools={len(names)}")
            print(", ".join(names))
            result = await session.call_tool("get_stats", {})
            print(result.content[0].text)
            assert len(names) == 18, f"expected 18 tools, got {len(names)}"
            payload = json.loads(result.content[0].text)
            assert "total" in payload and "by_task_type" in payload


if __name__ == "__main__":
    asyncio.run(main())
