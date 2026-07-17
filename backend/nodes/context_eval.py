"""context_eval_node — coverage check, zero LLM calls.

Checks whether the retrieved chunks cover every Planner sub-question.
Uses embedding cosine similarity only — no LLM call, so this can run every
outer/inner loop iteration for free (an LLM-based coverage check would add
up to 3 extra calls, one per outer loop — exactly the budget growth
this design avoids).

Coverage logic: for each sub-question, take the max cosine similarity to
any verified chunk; coverage_score is the min of those per-question maxima
(the worst-covered sub-question determines whether we refine).
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.state import AcademicResearchState

# Must match rag/indexer.py's EMBEDDING_MODEL: coverage_score is a cosine
# similarity between sub-question and chunk embeddings, so both need to
# live in the same embedding space as the chunks were indexed in.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

encoder = SentenceTransformer(EMBEDDING_MODEL)


def context_eval_node(state: AcademicResearchState) -> dict:
    sub_questions = state["planner_queries"]
    chunks = state["verified_chunks"]

    if not chunks or not sub_questions:
        return {"coverage_score": 0.0}

    q_embs = encoder.encode(sub_questions)
    c_embs = encoder.encode([c["content"] for c in chunks])

    q_embs = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-10)
    c_embs = c_embs / (np.linalg.norm(c_embs, axis=1, keepdims=True) + 1e-10)

    sim_matrix = q_embs @ c_embs.T   # (n_q, n_c)
    max_sims = sim_matrix.max(axis=1)  # (n_q,) best chunk per sub-question
    coverage_score = float(max_sims.min())  # worst-covered sub-question wins

    return {"coverage_score": coverage_score}
