"""refine_node — inner active-retrieval loop (Section 2.1 / 4.6).

Only reachable when route_after_context_eval decides coverage_score < 0.5
AND refinement_count < 1 AND budget not exceeded. Its job: retrieve more
evidence targeted at whichever sub-question was worst-covered, merge into
verified_chunks, bump refinement_count, then loop back to context_eval.

Zero-LLM by design: coverage was decided by embeddings, and the refined
retrieval reuses the ArXiv sub-agent's TwoStageRetriever with a targeted
query. Section 2.2's "Refined Search (inner) 1 call per invocation"
accounts for an optional tool-picking LLM call — the base path here
doesn't invoke one, so total_llm_calls is left unchanged when
refinement is triggered.
"""
import numpy as np

from backend.nodes.arxiv_agent import _get_guardrail, _get_retriever
from backend.state import AcademicResearchState

# Reuse the same encoder + guardrail the ArXiv sub-agent uses — the
# refined query needs to live in the same embedding space that
# context_eval used to declare a gap in the first place.


def _worst_covered_sub_question(state: AcademicResearchState) -> str:
    """Return the sub-question with the *lowest* max-cosine-sim to any
    verified chunk — the same worst-covered pick that made
    coverage_score = min across sub-questions in context_eval_node."""
    sub_questions = state["planner_queries"]
    chunks = state["verified_chunks"]
    if not sub_questions:
        return state["query"]
    if not chunks:
        return sub_questions[0]

    encoder = _get_guardrail().encoder
    q_embs = encoder.encode(sub_questions)
    c_embs = encoder.encode([c["content"] for c in chunks])
    q_embs = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-10)
    c_embs = c_embs / (np.linalg.norm(c_embs, axis=1, keepdims=True) + 1e-10)
    sim_matrix = q_embs @ c_embs.T
    max_sims = sim_matrix.max(axis=1)
    worst_idx = int(max_sims.argmin())
    return sub_questions[worst_idx]


def refine_node(state: AcademicResearchState) -> dict:
    target = _worst_covered_sub_question(state)
    retriever = _get_retriever()
    guardrail = _get_guardrail()

    raw_results = retriever.retrieve(target, k=8)
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
    new_verified, new_scores = guardrail.filter_chunks(raw_chunks, state["sf_threshold"])

    # Dedup on source id — refined retrieval on a similar sub-question can
    # surface papers we already have; keep the union with the incumbent
    # verified_chunks so context_eval sees the enlarged evidence pool.
    existing = state.get("verified_chunks", [])
    existing_ids = {c["source"] for c in existing}
    additions = [(c, s) for c, s in zip(new_verified, new_scores) if c["source"] not in existing_ids]

    merged_chunks = existing + [c for c, _ in additions]
    merged_scores = list(state.get("gamma_scores", [])) + [s for _, s in additions]

    return {
        "verified_chunks": merged_chunks,
        "gamma_scores": merged_scores,
        "refinement_count": state["refinement_count"] + 1,
        "status": "refining",
    }
