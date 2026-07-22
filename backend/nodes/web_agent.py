"""WebResearchAgent — sub-agent whose tool set is served over MCP.

Runs a web search over state["query"] (NOT search_payload — HyDE is
tuned to produce a hypothetical academic paper, which is a bad web-search
prompt; the raw query performs better against a general web index).
Writes to state["web_chunks"] only.

The search no longer happens through an imported function. This node
calls the `tavily_search` tool on the web-research MCP server
(mcp_servers/web_research.py), which runs as a separate process and
advertises its tool set over the protocol — so tool execution is
decoupled from graph orchestration, and the tools this agent can reach
are discovered at connect time rather than fixed by an import statement.
The arXiv sub-agent's FAISS retrieval deliberately stayed in-process: it
is latency-critical, and a process hop there would be paid on every query
for no decoupling benefit worth having.

Independent from ArXivResearchAgent: disjoint tool set (no FAISS, no
arxiv API, no pdf parser), disjoint state slice (state["web_chunks"],
never state["arxiv_chunks"]), and never reads what the other sub-agent
wrote — the merge happens later in merge_results_node after the Send
fan-out fans back in.

Failure handling is layered, so adding the MCP boundary cannot make this
node less reliable than it was before:
  1. MCP path fails (server won't spawn, protocol error, tool raises) ->
     fall back to calling rag.tools in-process. Same implementation, one
     less process.
  2. That fails too (Tavily down, no API key) -> empty web_chunks and
     continue arxiv-only, which is the degraded-but-valid state this node
     has always produced.

Budget: zero LLM calls in the base path (web search is a search API, not
an LLM). The budget allows a possible tool-picking LLM call per
invocation; when that loop is not exercised, total_llm_calls is left
unchanged here.
"""
import logging

from backend.state import AcademicResearchState
from rag.mcp_client import web_research_mcp
from rag.tools import tavily_search_tool

log = logging.getLogger(__name__)

# Served by the web-research MCP server, not imported into this module.
# Still disjoint from arxiv_agent.py's TOOLS by construction — and now the
# boundary is a process boundary, not just a naming convention.
# rag.mcp_client checks these against the server's list_tools response at
# connect time, so a rename on the server side fails loudly.
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
        # Both paths failed -> warn + continue with arxiv-only. An empty
        # web_chunks is a valid degraded state, not a failure.
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
