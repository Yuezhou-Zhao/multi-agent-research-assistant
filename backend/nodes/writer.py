"""Writer — draft the answer with index-based citations.

Design decision: Writer cites by
1-based INDEX into the presented chunk list — [1], [2], [3] — never by
raw arXiv id. A deterministic post-processing step in finalizer_node
resolves those indices back to real arXiv ids/URLs for the user-visible
final_answer. Rationale, in one sentence: confabulating specific
identifiers under pressure to "know" one is a well-documented LLM
failure mode, so we remove the failure mode instead of tuning around it.

The LLM's job becomes "which of these N sources supports this sentence?"
— structurally impossible to cite something that doesn't exist because
the whitelist is 1..N and check_citations can just do a range check.

Later fix: index-based citation stopped fabricated IDs but did NOT stop
*citation misattribution* — a structurally-valid [N] pointing at a paper whose
content doesn't support the specific claim being made. n=10 labeling
found this in 36/65 sentences. Two-part fix: this file's BASE_PROMPT
adds an explicit grounding rule ("do NOT describe methods from your
pretrained knowledge and then cite whichever chunk has a plausible
title"), and critic.py adds a zero-LLM embedding-similarity check
between each cited sentence and its cited chunk's text.

On rollback (Critic set critic_feedback), the previous draft is included
and a rollback-temperature LLM is used so the retry actually explores a
different phrasing.
"""
from langchain_openai import ChatOpenAI

from backend.state import AcademicResearchState, llm_call_update

LLM_MODEL = "gpt-4o-mini"


class WriterAgent:
    BASE_PROMPT = """You are an academic research writer. Answer the user's query using ONLY the numbered source excerpts below. Every sentence must cite at least one source using the exact `[N]` format, where N is the source's number (1..{n_sources}).

Query: {query}

Sources:
{sources}

STRICT CITATION RULES:
- Cite ONLY by source number, in square brackets: [1], [2], ..., [{n_sources}].
- Valid source numbers are: {valid_numbers}. Any other number is invalid.
- To cite multiple sources for one sentence, use separate brackets: "... [1] [3]." NOT "[1, 3]" or "[1,3]".
- Never write an arXiv id, URL, author name, or year inside the brackets.
- Every sentence must end with at least one [N] citation.
- Write 4-8 complete sentences.
- Do NOT include a header, title, or trailing bibliography — just the answer paragraph.

CRITICAL GROUNDING RULE (this is more important than the citation rules above):
- Every specific technical claim you attach a citation to — method names, algorithm descriptions, benchmark scores, dataset names, numeric results — MUST appear in the cited chunk's text as shown above.
- Do NOT describe methods from your own pretrained knowledge. The base model likely knows many of these papers (e.g. FG-PRM, CoNLI, FLARE, HyDE), but that memory is unreliable for citation-attaching: describing what you *think* a paper says and then citing whichever nearby source has a plausible-looking title is worse than a citation-free sentence.
- If a chunk describes method X, cite it for claims about X. Do NOT use that chunk as a citation for method Y from your training data just because Y is in the same subfield.
- If no chunk supports the specific detail you want to include, either (a) omit the specific detail and stay general to what the chunks actually say, or (b) drop the sentence entirely. Never invent a link between a claim and a chunk.
- When in doubt, quote or paraphrase directly from the cited chunk rather than synthesizing from memory.
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
        """Number sources 1..N so the LLM cites by index, not by raw id."""
        lines = []
        for i, c in enumerate(chunks, start=1):
            title = c.get("title") or ""
            title_prefix = f"{title} — " if title else ""
            lines.append(f"[{i}] {title_prefix}{c['content']}")
        return "\n\n".join(lines)

    async def write(
        self,
        query: str,
        chunks: list[dict],
        previous_draft: str = "",
        critic_feedback: str | None = None,
    ) -> str:
        n = len(chunks)
        valid_numbers = ", ".join(str(i) for i in range(1, n + 1)) if n else "(none)"
        prompt = self.BASE_PROMPT.format(
            query=query,
            sources=self._format_sources(chunks),
            n_sources=n,
            valid_numbers=valid_numbers,
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
