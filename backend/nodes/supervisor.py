"""SupervisorAgent — query classification + Send API dispatch.

Routes retrieval to the ArXiv sub-agent, the Web sub-agent, or both in
parallel. This classification step plus route_after_supervisor's Send
fan-out (backend/graph.py) is what makes the Researcher tier genuinely
multi-agent: the Supervisor decides independently,
then two sub-agents with disjoint tool sets run concurrently and write to
non-overlapping state fields (arxiv_chunks vs web_chunks).

Also holds merge_results_node — the fan-in after that Send dispatch. Zero
LLM calls: it's a reduce over two lists the independent sub-agents wrote,
not a decision.
"""
import json

from langchain_openai import ChatOpenAI

from backend.state import AcademicResearchState, llm_call_update

LLM_MODEL = "gpt-4o-mini"


class SupervisorAgent:
    CLASSIFICATION_PROMPT = """Classify this research query. Respond in JSON only:
{{"use_arxiv": bool, "use_web": bool, "reason": "one sentence"}}

Query: {query}

use_arxiv: true if query needs academic papers, methods, benchmarks
use_web: true if query needs recent news, events, or non-academic sources"""

    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    FALLBACK_DECISION = {
        "use_arxiv": True,
        "use_web": False,
        "reason": "classifier reply was malformed; defaulting to arxiv-only",
    }

    async def classify(self, query: str) -> dict:
        response = await self.llm.ainvoke(self.CLASSIFICATION_PROMPT.format(query=query))
        # A malformed reply must not kill the job: fall back to the same
        # arxiv-only default route_after_supervisor uses when a decision
        # names neither source. response_format=json_object makes this
        # rare, not impossible.
        try:
            decision = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            return dict(self.FALLBACK_DECISION)
        if not isinstance(decision, dict):
            return dict(self.FALLBACK_DECISION)
        return decision


_default_agent: SupervisorAgent | None = None


def _get_default_agent() -> SupervisorAgent:
    # Lazy singleton: constructing ChatOpenAI eagerly at import time would
    # require OPENAI_API_KEY to already be in the environment just to
    # import this module (e.g. for graph-topology tests) — defer it to
    # first real use instead.
    global _default_agent
    if _default_agent is None:
        _default_agent = SupervisorAgent()
    return _default_agent


async def supervisor_node(state: AcademicResearchState) -> dict:
    decision = await _get_default_agent().classify(state["query"])
    return {"supervisor_decision": decision, **llm_call_update(state)}


def merge_results_node(state: AcademicResearchState) -> dict:
    """Fan-in after the ArXiv/Web Send dispatch: concatenate whatever each
    independent sub-agent wrote to its own state slice. Zero LLM calls.

    This is also where the researching -> context_eval status transition
    is written, since the sub-agents themselves can't safely touch
    `status` (they run concurrently via Send; LastValue channels only
    accept one write per superstep). merge_results runs after fan-in,
    sequentially, so it can.
    """
    merged = state.get("arxiv_chunks", []) + state.get("web_chunks", [])
    return {"merged_chunks": merged, "status": "context_eval"}
