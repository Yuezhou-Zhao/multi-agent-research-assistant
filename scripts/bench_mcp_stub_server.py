"""Benchmark-only variant of the web-research MCP server.

Serves the same FastMCP app as mcp_servers.web_research, with the Tavily
call swapped for a canned result. That isolates what the benchmark is
actually trying to measure — stdio transport + JSON-RPC + FastMCP
dispatch — from Tavily's network latency, which is two orders of
magnitude larger and would swamp the signal entirely.

The tool body is replaced by rebinding the module global that the
`tavily_search` wrapper resolves at call time, so the protocol surface,
schema, and dispatch path are byte-for-byte what production serves.

Not imported by the app. Only scripts/bench_mcp_overhead.py spawns it.
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
