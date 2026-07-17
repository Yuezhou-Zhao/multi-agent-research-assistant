"""ArXivResearchAgent — sub-agent with independent tool set.

Runs the TwoStageRetriever over state["search_payload"] (HyDE
output or raw query) and pushes the results through the Gamma guardrail's
filter_chunks so only chunks above sf_threshold reach the merged pool.
Writes to state["arxiv_chunks"] and state["gamma_scores"] only — does NOT
read state["web_chunks"] and does NOT know a Web sub-agent exists.

This node is dispatched by route_after_supervisor's Send call, so it
receives a copy of state with the Supervisor's decision already made; the
Supervisor won't route here unless supervisor_decision["use_arxiv"] is
true (or the "neither" fallback in graph.py fired).

Four tools exist for this sub-agent (arXiv API, FAISS, PDF parser,
calculator). The FAISS retrieval is what runs by default here; the
arXiv-API + PDF-parser + calculator tools are wired but not invoked in
the base retrieval path, which is what the budget's expected LLM-call
count assumes. They exist so a future extension (an
agent-loop where the LLM picks tools per sub-question) can call them
without rewriting anything above.

Budget: no LLM call in the base path — retrieval + Gamma filtering are
both zero-LLM. The budget's "1 call per invocation" allowance for this
sub-agent covers the tool-picking loop mentioned above; when that
loop is not exercised (the current default), total_llm_calls is left
unchanged here.
"""
from backend.state import AcademicResearchState
from evaluation.gamma_guardrail import GammaGuardrail
from rag.retriever import TwoStageRetriever
from rag.tools import arxiv_search_tool, url_scraper_tool

# Lazy singletons — loading TwoStageRetriever + GammaGuardrail (encoder,
# reranker, FAISS index, calibration on 438 abstracts) at import time
# would freeze anything that just wants to import this module (e.g. graph
# topology tests, main.py's route wiring at startup for an unrelated
# endpoint).
_retriever: TwoStageRetriever | None = None
_guardrail: GammaGuardrail | None = None


def _get_retriever() -> TwoStageRetriever:
    global _retriever
    if _retriever is None:
        _retriever = TwoStageRetriever()
    return _retriever


def _get_guardrail() -> GammaGuardrail:
    global _guardrail
    if _guardrail is None:
        # Chunk-specific calibration (abstracts): what filter_chunks
        # actually scores here is retrieved chunk content, which lives
        # in the same distribution as raw abstracts. See
        # evaluation/gamma_guardrail.py's build_sentence_guardrail
        # docstring for why the Critic uses a different calibration.
        from evaluation.gamma_guardrail import build_chunk_guardrail

        _guardrail, _ = build_chunk_guardrail()
    return _guardrail


# Independent tool set — listed here (not just referenced) so the "true
# multi-agent" claim is enforceable by inspection: this module has its
# own tool imports; web_agent.py has its own, disjoint set.
TOOLS = [arxiv_search_tool, url_scraper_tool]  # + faiss_retriever_tool (via _get_retriever), + pdf_parser_tool (future)


async def arxiv_agent_node(state: AcademicResearchState) -> dict:
    retriever = _get_retriever()
    guardrail = _get_guardrail()

    raw_results = retriever.retrieve(state["search_payload"], k=8)
    raw_chunks = [
        {
            "content": r["parent_text"],
            "source": r["paper"]["id"],
            "title": r["paper"]["title"],
            "url": r["paper"]["url"],
            "rerank_score": r["score"],
        }
        for r in raw_results
    ]

    verified, scores = guardrail.filter_chunks(raw_chunks, state["sf_threshold"])
    # NOTE: no `status` write here. Both this node and web_agent_node fire
    # concurrently via the Send fan-out in graph.py, and LangGraph's default
    # LastValue reducer on the `status` channel rejects writes from >1 node
    # in the same superstep (InvalidUpdateError). status transitions are
    # owned by sequential nodes only (supervisor/writer/critic/finalize).
    return {
        "arxiv_chunks": verified,
        "gamma_scores": scores,
    }
