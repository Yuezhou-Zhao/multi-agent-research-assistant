"""HyDEOperator — idempotent pre-flight, NOT a LangGraph node (Section 4.3).

Executes once before the LangGraph state machine starts (Section 2.1's
"Pre-flight Layer (OUTSIDE state machine)"). Its result is frozen into
state["search_payload"] for the job's entire lifetime. This module lives
under backend/nodes/ per Section 7.1's file structure, but `execute()` is
called directly by the FastAPI handler (backend/main.py, Week 5) before
the graph is invoked — it is deliberately not registered as a graph node.

Why decoupled from the graph: a standard demo would put HyDE inside the
Researcher node, so a Critic rollback -> Researcher rerun -> HyDE
reinvokes every time. With max 3 rollbacks x 2 sub-agents, that's up to 6
wasted HyDE calls. Here: 1 call, cached in state, reused for every
downstream retrieval regardless of how many rollbacks follow.

Budget accounting (Section 2.2): the LLM call happens before either
quality gate runs, so it counts against total_llm_calls whenever
hyde_enabled=True — even if the result is then discarded by a gate and
the fallback (raw query) is used instead. The caller (main.py, Week 5)
should seed new_job_state()'s total_llm_calls with 1 whenever
hyde_enabled=True, since this function's own call happens outside the
graph and can't update AcademicResearchState itself.
"""
import numpy as np
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

# Must match rag/indexer.py, backend/nodes/context_eval.py, and
# evaluation/gamma_guardrail.py — the HyDE quality gate's cosine
# similarity needs to live in the same embedding space as everything
# else that compares embeddings in this project.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Section 2.2's $0.001/call cost estimate is GPT-4o-mini.
LLM_MODEL = "gpt-4o-mini"

MIN_HYP_DOC_WORDS = 50
# Section 4.3's literal threshold. Measured empirically with
# BAAI/bge-small-en-v1.5: real 50+-word "unrelated" text against a typical
# query (recipe text, sports commentary, gibberish, random word salad,
# even French prose or digit strings) scored 0.35-0.49 cosine similarity —
# all *above* this gate. That's this embedding model's anisotropic
# baseline similarity floor, not a bug — it means gate 2 rarely fires in
# practice with this model. Kept at the spec'd value rather than lowered
# to "make it trigger more," since inventing a stricter threshold to hit
# a target trigger rate isn't a decision this project gets to make
# unilaterally. See tests/test_preflight.py for how gate 2's *logic* is
# tested deterministically (mocked encoder) despite this.
MIN_QUALITY_COSINE_SIM = 0.3


class HyDEOperator:
    PROMPT = (
        "Write a single paragraph from a peer-reviewed CS paper that "
        "directly answers: {query}. Use technical terminology and "
        "declarative statements. No title, no references."
    )

    def __init__(self, llm=None, encoder: SentenceTransformer | None = None):
        self.llm = llm or ChatOpenAI(model=LLM_MODEL, temperature=0.3)
        self.encoder = encoder or SentenceTransformer(EMBEDDING_MODEL)

    async def execute(self, query: str, hyde_enabled: bool) -> tuple[str, bool]:
        """Returns (search_payload, hyde_used).

        Fallback to the raw query (hyde_used=False) if HyDE is disabled,
        or if either quality gate fails:
          gate 1 (length):  hypothetical doc < 50 words
          gate 2 (semantic): cosine_sim(query, hyp_doc) < 0.3 — HyDE
            produced something too unrelated to the actual query to be a
            useful retrieval anchor.
        """
        if not hyde_enabled:
            return query, False

        hyp_doc = (await self.llm.ainvoke(self.PROMPT.format(query=query))).content.strip()

        if len(hyp_doc.split()) < MIN_HYP_DOC_WORDS:
            return query, False

        q_emb = self.encoder.encode([query])[0]
        d_emb = self.encoder.encode([hyp_doc])[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        d_emb = d_emb / (np.linalg.norm(d_emb) + 1e-10)
        if float(q_emb @ d_emb) < MIN_QUALITY_COSINE_SIM:
            return query, False

        return hyp_doc, True
