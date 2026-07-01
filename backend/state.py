"""AcademicResearchState — the shared state schema for the LangGraph
orchestrator (Planner/Supervisor/Writer/Critic tier).

See Section 3.1 of agent_system_design_v2.md for the authoritative spec.
"""
from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph import add_messages


class AcademicResearchState(TypedDict):

    # ── Core content ─────────────────────────────────────────────────
    query: str
    search_payload: str          # HyDE output or raw query; IMMUTABLE after preflight
    planner_queries: list[str]   # 3 sub-questions from Planner
    supervisor_decision: dict    # {use_arxiv: bool, use_web: bool, reason: str}

    # Retrieved and verified evidence
    arxiv_chunks: list[dict]     # raw from ArXiv sub-agent
    web_chunks: list[dict]       # raw from Web sub-agent
    merged_chunks: list[dict]    # merged by Supervisor after parallel retrieval
    verified_chunks: list[dict]  # post-Gamma-filter

    # gamma_scores: per-chunk survival-function score from the calibrated
    # embedding-distance guardrail (see evaluation/gamma_guardrail.py, Week 4).
    # This is a cheap prefilter score, not a statistical guarantee of
    # correctness — do not describe it as such in docs or demo copy.
    gamma_scores: list[float]

    # Generation
    draft: str
    final_answer: str
    citations: list[str]
    critic_feedback: Optional[str]  # error context injected on rollback

    # ── Loop control (circuit breaker + budget) ───────────────────────
    critic_loop_count: int       # outer: Critic → Researcher (max 3)
    max_critic_loops: int        # snapshotted at job creation, default 3
    refinement_count: int        # inner: Context Eval → refined search (max 1)
    total_llm_calls: int         # global counter, incremented after each LLM call
    max_llm_calls: int           # hard cap, snapshotted at job creation, default 15
    llm_budget_exceeded: bool    # True when total_llm_calls >= max_llm_calls

    status: Literal[
        "preflight", "planning", "supervising",
        "researching_arxiv", "researching_web",
        "context_eval", "refining",
        "writing", "reviewing",
        "approved", "failed",
        "force_finalized", "budget_exceeded",
    ]
    failure_reason: Optional[str]

    # ── Per-job config (snapshotted at POST /research, immutable) ─────
    hyde_enabled: bool           # from Chainlit toggle at job creation
    sf_threshold: float          # from Chainlit slider at job creation
    job_id: str

    # ── Evaluation metadata ────────────────────────────────────────────
    coverage_score: float        # min cosine_sim across sub-questions
    hallucination_flags: list[bool]
    cascade_decisions: list[str]  # per-sentence: "reject" | "approve" | "escalate"
    llm_calls_avoided: int       # Gamma-only decisions (for metrics dashboard)
    node_latencies: dict[str, float]

    # ── LangGraph messages ─────────────────────────────────────────────
    messages: Annotated[list, add_messages]


def new_job_state(
    job_id: str,
    query: str,
    hyde_enabled: bool = True,
    sf_threshold: float = 0.15,
    max_critic_loops: int = 3,
    max_llm_calls: int = 15,
) -> AcademicResearchState:
    """Build the initial state for a job.

    hyde_enabled, sf_threshold, max_critic_loops, and max_llm_calls are
    snapshotted here (Section 1.3 #6 / Section 4.6): later changes to a
    Chainlit slider or toggle must not affect an in-flight job.
    """
    return AcademicResearchState(
        query=query,
        search_payload="",
        planner_queries=[],
        supervisor_decision={},
        arxiv_chunks=[],
        web_chunks=[],
        merged_chunks=[],
        verified_chunks=[],
        gamma_scores=[],
        draft="",
        final_answer="",
        citations=[],
        critic_feedback=None,
        critic_loop_count=0,
        max_critic_loops=max_critic_loops,
        refinement_count=0,
        total_llm_calls=0,
        max_llm_calls=max_llm_calls,
        llm_budget_exceeded=False,
        status="preflight",
        failure_reason=None,
        hyde_enabled=hyde_enabled,
        sf_threshold=sf_threshold,
        job_id=job_id,
        coverage_score=0.0,
        hallucination_flags=[],
        cascade_decisions=[],
        llm_calls_avoided=0,
        node_latencies={},
        messages=[],
    )
