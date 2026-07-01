"""Writer — draft the answer with inline citations (Section 2.1 diagram).

Reads verified_chunks (the Gamma-filtered evidence pool that came out of
merge_results -> context_eval -> maybe refine) and produces an answer
that cites each supporting chunk by its `source` id in [id] brackets.
The Critic's rule-based citation grounding (evaluation/citation_check.py)
enforces this bracket format, so we tell the LLM about it explicitly.

On rollback (Critic sent us back with critic_feedback set), the previous
draft is included and the feedback is prepended — this is the "error
context injected on rollback" line in Section 2.1's Critic node box.
Without it, a rerun would produce the same draft and rollback forever.
"""
from langchain_openai import ChatOpenAI

from backend.state import AcademicResearchState, llm_call_update

LLM_MODEL = "gpt-4o-mini"


class WriterAgent:
    # Prompt hardened after measuring that gpt-4o-mini in the Writer role
    # will happily generate plausible-looking arxiv IDs (2305.17306v1,
    # 2402.00559v4, etc.) that don't exist in the source pool despite
    # a generic "do not invent citations" instruction. Fix: enumerate the
    # legal IDs explicitly at prompt end and repeat the constraint at
    # both the top and bottom of the message. Empirically this brings
    # the dangling-citation rate way down; the Critic's L2 grounding
    # check still catches whatever slips through.
    BASE_PROMPT = """You are an academic research writer. Answer the user's query using ONLY the provided source excerpts. Every sentence must cite at least one source using the exact `[source_id]` format shown below.

Query: {query}

Sources (cite by the id in [square brackets]):
{sources}

STRICT CITATION RULES:
- The ONLY valid citation ids are: {valid_ids}
- You MUST copy these ids exactly, character-for-character. Do NOT invent, guess, or modify any id.
- If you cannot support a claim from the sources above, do not make the claim.
- Every sentence must end with at least one [source_id] citation.
- Write 4-8 complete sentences.
- Do NOT include a header, title, or trailing bibliography — just the answer paragraph.
"""

    FEEDBACK_ADDENDUM = """

IMPORTANT: Your previous draft was rejected. Feedback:
{feedback}

Your previous draft (which was WRONG — do NOT reproduce it):
{previous_draft}

You MUST produce a substantively different draft this time. Do not lightly reword the previous draft — restructure it. Cite different combinations of sources where possible."""

    # Base temperature is low so first-attempt drafts stay stable; on
    # rollback the temperature is raised so the Writer actually explores
    # a different phrasing/structure. Without this bump, gpt-4o-mini
    # produced byte-for-byte identical drafts across all 3 outer loops
    # even with critic_feedback in the prompt.
    BASE_TEMPERATURE = 0.2
    ROLLBACK_TEMPERATURE = 0.8

    def __init__(self, llm=None, llm_rollback=None):
        self.llm = llm or ChatOpenAI(model=LLM_MODEL, temperature=self.BASE_TEMPERATURE)
        self.llm_rollback = llm_rollback or ChatOpenAI(
            model=LLM_MODEL, temperature=self.ROLLBACK_TEMPERATURE
        )

    def _format_sources(self, chunks: list[dict]) -> str:
        lines = []
        for c in chunks:
            title = c.get("title") or ""
            title_prefix = f"{title} — " if title else ""
            lines.append(f"[{c['source']}] {title_prefix}{c['content']}")
        return "\n\n".join(lines)

    async def write(
        self,
        query: str,
        chunks: list[dict],
        previous_draft: str = "",
        critic_feedback: str | None = None,
    ) -> str:
        valid_ids = ", ".join(f"[{c['source']}]" for c in chunks)
        prompt = self.BASE_PROMPT.format(
            query=query,
            sources=self._format_sources(chunks),
            valid_ids=valid_ids,
        )
        llm = self.llm
        if critic_feedback:
            prompt += self.FEEDBACK_ADDENDUM.format(
                feedback=critic_feedback, previous_draft=previous_draft
            )
            llm = self.llm_rollback
        response = await llm.ainvoke(prompt)
        return response.content.strip()


_default_agent: WriterAgent | None = None


def _get_default_agent() -> WriterAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = WriterAgent()
    return _default_agent


async def writer_node(state: AcademicResearchState) -> dict:
    chunks = state.get("verified_chunks") or state.get("merged_chunks", [])
    draft = await _get_default_agent().write(
        query=state["query"],
        chunks=chunks,
        previous_draft=state.get("draft", ""),
        critic_feedback=state.get("critic_feedback"),
    )
    return {
        "draft": draft,
        "status": "writing",
        # Consume critic_feedback so the next Writer call (if we roll back
        # again) sees only fresh feedback, not the stale prior round's.
        "critic_feedback": None,
        **llm_call_update(state),
    }
