"""WebResearchAgent — sub-agent with independent tool set (Section 4.2).

Runs Tavily search over state["query"] (NOT search_payload — HyDE is
tuned to produce a hypothetical academic paper, which is a bad web-search
prompt; the raw query performs better against a general web index).
Writes to state["web_chunks"] only.

Independent from ArXivResearchAgent per Section 1.2: disjoint tool set
(no FAISS, no arxiv API, no pdf parser), disjoint state slice
(state["web_chunks"], never state["arxiv_chunks"]), and never reads what
the other sub-agent wrote — the merge happens later in merge_results_node
after the Send fan-out fans back in.

Budget: zero LLM calls in the base path (Tavily is a search API, not an
LLM). Section 2.2 accounts for a possible tool-picking LLM call per
invocation; when that loop is not exercised, total_llm_calls is left
unchanged here. Tavily failures degrade gracefully to an empty
web_chunks list rather than crashing the whole job — Section 6 Tier 2
"asyncio error handling" specifies exactly this behavior for the Tavily
timeout case.
"""
import logging

from backend.state import AcademicResearchState
from rag.tools import tavily_search_tool, url_scraper_tool

log = logging.getLogger(__name__)

# Independent tool set (Section 1.2, Section 4.2) — disjoint from
# arxiv_agent.py's TOOLS by construction, not by convention.
TOOLS = [tavily_search_tool, url_scraper_tool]


async def web_agent_node(state: AcademicResearchState) -> dict:
    # NOTE: no `status` write in this node — see arxiv_agent.py's note.
    # Both sub-agents fire concurrently via Send; only one can write to a
    # LastValue channel per superstep. status is owned by sequential nodes.
    try:
        results = await tavily_search_tool(state["query"], max_results=5)
    except Exception as exc:
        # Section 6 Tier 2: Tavily failure -> warn + continue with arxiv-
        # only. Empty web_chunks is a valid degraded state, not a failure.
        log.warning("Web sub-agent Tavily call failed, degrading to empty results: %s", exc)
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
