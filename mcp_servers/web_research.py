"""web-research MCP server — the Web sub-agent's tool set, served over MCP.

This is the WebResearchAgent's complete tool set (tavily_search,
url_scraper) exposed as a standalone process speaking the Model Context
Protocol, rather than as functions the agent node imports directly.

What actually moved: nothing. The implementations still live in
rag/tools.py and are imported here unchanged — this module is a protocol
surface over them, not a reimplementation. That keeps rag/tools.py
directly unit-testable without an agent runtime or a transport in the
loop (the property its own docstring claims), and keeps this refactor
honest: the diff is a boundary, not a rewrite.

Why the Web tool set and not the arXiv one: web_agent_node's Tavily call
is the only tool invocation on the live query path. arxiv_agent_node runs
the FAISS TwoStageRetriever, and rag.tools.arxiv_search_tool is used only
by rag/indexer.py when building the index offline. Wrapping that one
would have put an MCP server on a batch script and called it agent
tooling. FAISS retrieval stays in-process deliberately: it is the
latency-critical path, and a process hop there would be paid on every
query for nothing.

Note that the function docstrings below are no longer only for humans.
FastMCP publishes them as the protocol-level `description` on each tool,
so they are what a client sees when it enumerates this server — the text
an LLM would read to decide whether to call the tool.

Transport is stdio: the client spawns this module as a subprocess. That
needs no ports, no compose service, and no Dockerfile change, since the
repo is already on PYTHONPATH inside the image.

Run standalone (e.g. to inspect it with an MCP client):
    python -m mcp_servers.web_research
"""
from mcp.server.fastmcp import FastMCP

from rag.tools import tavily_search_tool, url_scraper_tool

mcp = FastMCP("web-research")


@mcp.tool()
async def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for pages relevant to `query`.

    Returns up to `max_results` results, each with the extracted page
    content, its source URL, and its title. Use this to find current or
    non-academic material; it searches a general web index, not arXiv.
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
