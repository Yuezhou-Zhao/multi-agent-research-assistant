"""WebResearchAgent — sub-agent whose tool set is served over MCP.

Searches over state["query"] (NOT search_payload — HyDE is tuned to
produce a hypothetical academic paper, which is a bad web-search prompt).
Writes to state["web_chunks"] only.

Tools come from the web-research MCP server over stdio, discovered at
connect time rather than imported. Still disjoint from ArXivResearchAgent
in both tools and state slice; results merge later in merge_results_node.

Search degrades in layers: MCP -> in-process rag.tools -> empty
web_chunks (arxiv-only, a valid degraded state).
"""
import logging

from backend.state import AcademicResearchState
from rag.mcp_client import web_research_mcp
from rag.tools import tavily_search_tool

log = logging.getLogger(__name__)

# Served by the MCP server, not imported here. rag.mcp_client checks these
# against the server's list_tools response at connect time.
TOOLS = ["tavily_search", "url_scraper"]


async def _search(query: str, max_results: int = 5) -> list[dict]:
    """Search via MCP, falling back to the in-process implementation."""
    try:
        return await web_research_mcp.call(
            "tavily_search", {"query": query, "max_results": max_results}
        )
    except Exception as exc:
        log.warning(
            "web-research MCP call failed, falling back to in-process tool: %s", exc
        )
        return await tavily_search_tool(query, max_results=max_results)


async def web_agent_node(state: AcademicResearchState) -> dict:
    # NOTE: no `status` write in this node — see arxiv_agent.py's note.
    # Both sub-agents fire concurrently via Send; only one can write to a
    # LastValue channel per superstep. status is owned by sequential nodes.
    try:
        results = await _search(state["query"], max_results=5)
    except Exception as exc:
        log.warning("Web sub-agent search failed, degrading to empty results: %s", exc)
        return {"web_chunks": []}

    chunks = [
        {
            "content": r["content"],
            "source": r["source"],
            "title": r.get("title", ""),
        }
        for r in results
    ]
    return {"web_chunks": chunks}
