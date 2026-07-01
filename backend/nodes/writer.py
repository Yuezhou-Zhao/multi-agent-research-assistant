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
    BASE_PROMPT = """You are an academic research writer. Answer the user's query using ONLY the provided source excerpts. Every sentence must cite at least one source in [source_id] brackets.

Query: {query}

Sources (cite by the id in [square brackets]):
{sources}

Requirements:
- Write 4-8 complete sentences.
- Every sentence must end with at least one [source_id] citation.
- Only cite ids from the sources above; do not invent citations.
- Do NOT include a header, title, or trailing bibliography — just the answer paragraph.
"""

    FEEDBACK_ADDENDUM = """

Your previous draft was rejected by the Critic with this feedback:
{feedback}

Previous draft (do not repeat its errors):
{previous_draft}
"""

    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model=LLM_MODEL, temperature=0.2)

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
        prompt = self.BASE_PROMPT.format(query=query, sources=self._format_sources(chunks))
        if critic_feedback:
            prompt += self.FEEDBACK_ADDENDUM.format(
                feedback=critic_feedback, previous_draft=previous_draft
            )
        response = await self.llm.ainvoke(prompt)
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
