"""web-research MCP server — the Web sub-agent's tool set over stdio.

Implementations stay in rag/tools.py; this module is a protocol surface
over them. The function docstrings below are published as the
protocol-level tool descriptions, so they are what a client reads when it
enumerates this server.

    python -m mcp_servers.web_research
"""
from mcp.server.fastmcp import FastMCP

from rag.tools import tavily_search_tool, url_scraper_tool

mcp = FastMCP("web-research")


@mcp.tool()
async def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for pages relevant to `query`.

    Returns up to `max_results` results, each with the extracted page
    content, its source URL, and its title. Searches a general web index,
    not arXiv.
    """
    return await tavily_search_tool(query, max_results=max_results)


@mcp.tool()
async def url_scraper(url: str, timeout: float = 15.0) -> str:
    """Fetch a single URL and return its visible text, with markup stripped.

    Use this to read the full text of a page that tavily_search returned
    only a snippet of. Gives up after `timeout` seconds.
    """
    return await url_scraper_tool(url, timeout=timeout)


if __name__ == "__main__":
    mcp.run()
