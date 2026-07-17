"""finalize_node / force_finalize_node — the two graph exits.

finalize is the happy path: Critic approved the draft, so we commit it
as final_answer and mark the job done. force_finalize is the graceful
degradation path: either the outer circuit breaker exhausted
(critic_loop_count >= max_critic_loops) or the global LLM budget hit its
cap — in both cases we still return whatever the last draft was, since
"no answer" is worse than "an answer with a degradation flag." The
budget is a hard ceiling with explicit degradation signaling.

Both nodes are zero-LLM.

The draft coming in uses 1-based `[N]` markers indexing
into merged_chunks. resolve_citations() rewrites them to real
`[source_id]` markers for the user-visible final_answer, and returns
the ordered id list for the citations panel. Doing the resolution here
(rather than in Critic) means the internal draft flowing between graph
nodes stays index-based, so all the guardrail/L2 checks compare against
one consistent representation.
"""
from backend.state import AcademicResearchState
from evaluation.citation_check import resolve_citations


def finalize_node(state: AcademicResearchState) -> dict:
    draft = state["draft"]
    merged_chunks = state.get("merged_chunks") or state.get("verified_chunks", [])
    resolved, ordered_ids = resolve_citations(draft, merged_chunks)
    return {
        "final_answer": resolved,
        "citations": ordered_ids,
        "status": "approved",
    }


def force_finalize_node(state: AcademicResearchState) -> dict:
    draft = state.get("draft", "")
    merged_chunks = state.get("merged_chunks") or state.get("verified_chunks", [])
    resolved, ordered_ids = resolve_citations(draft, merged_chunks)

    if state["llm_budget_exceeded"]:
        reason = "budget_exceeded"
        status = "budget_exceeded"
    else:
        reason = f"circuit breaker fired (critic_loop_count={state['critic_loop_count']})"
        status = "force_finalized"

    return {
        "final_answer": resolved,
        "citations": ordered_ids,
        "status": status,
        "failure_reason": reason,
    }
