from fastmcp import FastMCP
from analysis_core import analyze_python

mcp = FastMCP("programming-assistant")
mcp.tool()(analyze_python)
if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8765)
