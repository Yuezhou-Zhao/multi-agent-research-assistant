"""Critic — three-layer cascade (Section 4.7 diagram).

L1: Gamma sentence scorer over the draft (zero LLM, ~ms per sentence).
    Rejects on any "reject" (SF < CERTAIN_WRONG). If all sentences either
    "approve" or "escalate", proceeds to L2.
L2: Citation grounding, rule-based (zero LLM). Rejects if any bracket
    references an id that isn't in merged_chunks, or any sentence lacks
    a citation. If clean, proceeds to L3.
L3: LLM judge (single gpt-4o-mini call). Reads the draft + critic-view
    citations and returns an approve/reject decision with a reason.
    ONLY fires if L1 and L2 both passed.

llm_calls_avoided (Section 3.1) increments by 1 whenever L1 or L2
short-circuits the L3 judge — this is the primary metric for the cascade
value story (Section 5.1 target: >60% of Critic decisions resolved by
Gamma alone). critic_loop_count increments regardless of outcome —
Section 4.6's outer circuit breaker (max_critic_loops=3) needs to see
every Critic pass, approved or not.

Section 2.2 accounts for max 3 Critic LLM calls (once per outer loop);
the cascade means the actual number is usually much lower.
"""
import json

from langchain_openai import ChatOpenAI

from backend.state import AcademicResearchState, llm_call_update
from evaluation.citation_check import check_citations
from evaluation.gamma_guardrail import GammaGuardrail

LLM_MODEL = "gpt-4o-mini"


class CriticAgent:
    JUDGE_PROMPT = """You are an academic Critic. Determine whether this draft answers the query using ONLY the cited sources. Respond in JSON only:
{{"approved": bool, "reason": "one sentence"}}

Query: {query}

Draft:
{draft}

Cited source excerpts (id -> content):
{citations}

Reject if: the draft makes claims not supported by the cited excerpts,
misrepresents a cited source, or omits a critical aspect of the query."""

    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    async def judge(self, query: str, draft: str, cited_chunks: list[dict]) -> dict:
        citations = "\n\n".join(f"[{c['source']}] {c['content']}" for c in cited_chunks)
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
        from evaluation.gamma_guardrail import build_default_guardrail

        _guardrail, _ = build_default_guardrail()
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
    l1_rejected = "reject" in cascade_decisions
    if l1_rejected:
        # Any single "reject" is enough to send us back — cheap, no LLM.
        return {
            **base_update,
            "cascade_decisions": cascade_decisions,
            "status": "reviewing",
            "critic_feedback": (
                f"Gamma guardrail flagged {cascade_decisions.count('reject')} sentence(s) "
                f"as likely-hallucinated (SF < {GammaGuardrail.CERTAIN_WRONG})."
            ),
            "llm_calls_avoided": state["llm_calls_avoided"] + 1,
        }

    # ── Layer 2: citation grounding ────────────────────────────────────
    citation_report = check_citations(draft, merged_chunks)
    if not citation_report.passed:
        return {
            **base_update,
            "cascade_decisions": cascade_decisions,
            "status": "reviewing",
            "critic_feedback": f"Citation check failed: {citation_report.summary()}",
            "llm_calls_avoided": state["llm_calls_avoided"] + 1,
        }

    # ── Layer 3: LLM judge (only if L1+L2 both passed) ─────────────────
    cited_chunks = [c for c in merged_chunks if c["source"] in citation_report.cited_source_ids]
    verdict = await _get_default_agent().judge(state["query"], draft, cited_chunks)
    approved = bool(verdict.get("approved"))
    return {
        **base_update,
        "cascade_decisions": cascade_decisions,
        "status": "approved" if approved else "reviewing",
        "critic_feedback": None if approved else f"LLM judge rejected: {verdict.get('reason', '')}",
        **llm_call_update(state),
    }
