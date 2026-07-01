"""finalize_node / force_finalize_node — the two graph exits (Section 4.6).

finalize is the happy path: Critic approved the draft, so we commit it
as final_answer and mark the job done. force_finalize is the graceful
degradation path: either the outer circuit breaker exhausted
(critic_loop_count >= max_critic_loops) or the global LLM budget hit its
cap — in both cases we still return whatever the last draft was, since
"no answer" is worse than "an answer with a degradation flag." Section
2.2's whole point is that the budget is a hard ceiling with explicit
degradation signaling.

Both nodes are zero-LLM.
"""
import re

from backend.state import AcademicResearchState

_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def _extract_citations(draft: str) -> list[str]:
    """Deduped, order-preserving citation ids extracted from the draft's
    [id] markers — useful for the UI's citation panel (Section 5.1 /
    Chainlit, Week 6)."""
    seen = set()
    ordered = []
    for match in _CITATION_RE.findall(draft):
        cid = match.strip()
        if cid and cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def finalize_node(state: AcademicResearchState) -> dict:
    draft = state["draft"]
    return {
        "final_answer": draft,
        "citations": _extract_citations(draft),
        "status": "approved",
    }


def force_finalize_node(state: AcademicResearchState) -> dict:
    draft = state.get("draft", "")
    if state["llm_budget_exceeded"]:
        reason = "budget_exceeded"
        status = "budget_exceeded"
    else:
        reason = f"circuit breaker fired (critic_loop_count={state['critic_loop_count']})"
        status = "force_finalized"

    return {
        "final_answer": draft,
        "citations": _extract_citations(draft),
        "status": status,
        "failure_reason": reason,
    }
