"""Benchmark-only variant of the web-research MCP server.

Same FastMCP app, with the Tavily call swapped for a canned result, so
bench_mcp_overhead.py measures protocol cost rather than network latency.
Rebinding the module global keeps the dispatch path identical to production.
"""
import mcp_servers.web_research as server

CANNED_RESULT = [
    {
        "content": "Chain-of-thought prompting elicits intermediate reasoning steps.",
        "source": "https://example.com/cot",
        "title": "Chain-of-Thought Prompting",
    }
]


async def _stub_tavily_search(query: str, max_results: int = 5) -> list[dict]:
    return CANNED_RESULT[:max_results]


server.tavily_search_tool = _stub_tavily_search


if __name__ == "__main__":
    server.mcp.run()
