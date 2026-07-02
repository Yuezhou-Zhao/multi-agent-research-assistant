"""Critic — three-layer cascade (Section 4.7 diagram + Section 4.9 update).

L1: Gamma sentence scorer over the draft (zero LLM, ~ms per sentence).
    Rejects on majority-reject. Otherwise produces a per-sentence
    approve/escalate/reject cascade.
L2a: Rule-based structural citation check (zero LLM). Section 4.8:
     strips dangling `[N]` markers and enforces MAX_UNCITED_SENTENCES.
L2b: Per-sentence embedding grounding check (zero LLM, Section 4.9).
     Sentences whose cosine similarity to their cited chunk's text
     falls below GROUNDING_THRESHOLD are downgraded in the L1 cascade
     (approve -> escalate, escalate -> reject). Then majority-reject
     re-tested on the updated cascade.
L3: LLM judge (single gpt-4o-mini call). Only fires if L1, L2a, and
    L2b all left the cascade non-majority-reject.

Section 2.2 accounts for max 3 Critic LLM calls (once per outer loop);
the cascade means the actual number is usually much lower.
llm_calls_avoided (Section 3.1) increments by 1 whenever any L1/L2a/L2b
layer short-circuits L3 — the primary metric for the cascade value
story (Section 5.1 target: >=60% of Critic decisions resolved by
Gamma+rules alone).
"""
import json

from langchain_openai import ChatOpenAI

from backend.state import AcademicResearchState, llm_call_update
from evaluation.citation_check import check_citations
from evaluation.citation_grounding import (
    GROUNDING_THRESHOLD,
    check_citation_grounding,
)
from evaluation.gamma_guardrail import GammaGuardrail

LLM_MODEL = "gpt-4o-mini"


class CriticAgent:
    JUDGE_PROMPT = """You are an academic Critic. Determine whether this draft answers the query using ONLY the cited sources. Respond in JSON only:
{{"approved": bool, "reason": "one sentence"}}

Query: {query}

Draft:
{draft}

Cited source excerpts (numbered as they appear in the draft):
{citations}

Reject if: the draft makes claims not supported by the cited excerpts,
misrepresents a cited source, or omits a critical aspect of the query."""

    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    async def judge(
        self, query: str, draft: str, indexed_cited_chunks: list[tuple[int, dict]]
    ) -> dict:
        """indexed_cited_chunks is (1-based-index, chunk) so the judge sees
        the same [N] format the draft uses — matches Section 4.8's
        index-based citation architecture."""
        citations = "\n\n".join(
            f"[{i}] {c['content']}" for i, c in indexed_cited_chunks
        )
        response = await self.llm.ainvoke(
            self.JUDGE_PROMPT.format(query=query, draft=draft, citations=citations)
        )
        return json.loads(response.content)


_default_agent: CriticAgent | None = None
_guardrail: GammaGuardrail | None = None


def _get_default_agent() -> CriticAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = CriticAgent()
    return _default_agent


def _get_guardrail() -> GammaGuardrail:
    global _guardrail
    if _guardrail is None:
        # Sentence-specific calibration (Writer-style exemplars): what
        # score_and_route scores here is Writer's synthesized-and-cited
        # prose, which lives in a different distribution than raw
        # abstracts. Using the abstract-calibrated guardrail here caused
        # 100% force_finalized on a 5-query diagnostic batch; see
        # evaluation/gamma_guardrail.py's build_sentence_guardrail
        # docstring for the full trail.
        from evaluation.gamma_guardrail import build_sentence_guardrail

        _guardrail, _ = build_sentence_guardrail()
    return _guardrail


async def critic_node(state: AcademicResearchState) -> dict:
    draft = state["draft"]
    merged_chunks = state.get("merged_chunks") or state.get("verified_chunks", [])
    new_loop_count = state["critic_loop_count"] + 1
    base_update = {
        "critic_loop_count": new_loop_count,
    }

    # ── Layer 1: Gamma sentence scorer ────────────────────────────────
    guardrail = _get_guardrail()
    cascade_decisions, mean_sf = guardrail.score_and_route(draft, state["sf_threshold"])
    reject_count = cascade_decisions.count("reject")
    l1_rejected = reject_count > len(cascade_decisions) / 2  # majority-reject
    # Policy note: Section 4.7's code lists per-sentence decisions but
    # doesn't specify how to combine them into a draft-level verdict.
    # Original implementation used "any reject -> rollback"; measured on a
    # 5-query batch that rejected 100% of drafts (30% reject rate per
    # sentence * 5-7 sentences per draft -> always >= 1 reject). Switched
    # to majority-reject so L1 only auto-rollbacks when Gamma is
    # confidently negative overall; minority rejects still escalate the
    # whole draft to L3 (LLM judge), which was the point of L3 existing.
    # llm_calls_avoided still increments when L1 short-circuits — that's
    # the Section 5.1 metric.
    if l1_rejected:
        return {
            **base_update,
            "cascade_decisions": cascade_decisions,
            "status": "reviewing",
            "critic_feedback": (
                f"Gamma guardrail flagged {reject_count}/{len(cascade_decisions)} sentences "
                f"as likely-hallucinated (SF < {GammaGuardrail.CERTAIN_WRONG})."
            ),
            "llm_calls_avoided": state["llm_calls_avoided"] + 1,
        }

    # ── Layer 2a: structural citation check ────────────────────────────
    # check_citations strips dangling [N] markers as a repair pass (see
    # evaluation/citation_check.py). On pass, downstream sees the
    # sanitized draft; on fail, we still roll back to Writer with feedback
    # listing what was hallucinated so the retry has a chance to converge.
    citation_report = check_citations(draft, merged_chunks)
    if not citation_report.passed:
        return {
            **base_update,
            "cascade_decisions": cascade_decisions,
            "status": "reviewing",
            "critic_feedback": f"Citation check failed: {citation_report.summary()}",
            "llm_calls_avoided": state["llm_calls_avoided"] + 1,
        }

    sanitized_draft = citation_report.sanitized_draft or draft

    # ── Layer 2b: per-sentence embedding grounding (Section 4.9) ──────
    # Catches citation misattribution: structurally valid [N] pointing
    # at a chunk whose content doesn't support the sentence's claim.
    # Downgrades approve/escalate for ungrounded sentences; if the
    # updated cascade becomes majority-reject, roll back with feedback
    # naming the specific ungrounded sentences.
    grounding_report = check_citation_grounding(
        sanitized_draft, cascade_decisions, merged_chunks, guardrail.encoder
    )
    updated_cascade = grounding_report.updated_cascade
    updated_reject_count = updated_cascade.count("reject")
    if updated_reject_count > len(updated_cascade) / 2:
        return {
            **base_update,
            "cascade_decisions": updated_cascade,
            "status": "reviewing",
            "critic_feedback": (
                f"Citation grounding failed: {grounding_report.summary()}"
            ),
            "llm_calls_avoided": state["llm_calls_avoided"] + 1,
        }

    # ── Layer 3: LLM judge (only if L1, L2a, L2b all passed) ──────────
    # Judge sees chunks with the same 1-based [N] labels the Writer used
    # in the draft (Section 4.8) — otherwise it can't map "[3]" in the
    # draft to any specific evidence.
    indexed_cited_chunks = [
        (i, merged_chunks[i - 1])
        for i in sorted(citation_report.cited_indices)
        if 1 <= i <= len(merged_chunks)
    ]
    verdict = await _get_default_agent().judge(state["query"], sanitized_draft, indexed_cited_chunks)
    approved = bool(verdict.get("approved"))
    return {
        **base_update,
        "draft": sanitized_draft,
        "cascade_decisions": updated_cascade,
        "status": "approved" if approved else "reviewing",
        "critic_feedback": None if approved else f"LLM judge rejected: {verdict.get('reason', '')}",
        **llm_call_update(state),
    }
