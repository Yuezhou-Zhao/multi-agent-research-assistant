"""Planner — decompose the query into 3 sub-questions (Section 2.1 diagram).

Sub-questions get fanned out to the Supervisor next, and each one is
individually scored by context_eval_node (Section 4.5) — coverage_score
is the *min* across all 3, so the Planner's job is to produce
non-degenerate, non-overlapping sub-questions. Three is a fixed, spec'd
count, not a tuning knob: the worst-case LLM budget analysis in Section
2.2 assumes a single Planner call producing three sub-questions.
"""
import json

from langchain_openai import ChatOpenAI

from backend.state import AcademicResearchState, llm_call_update

LLM_MODEL = "gpt-4o-mini"  # Section 2.2's cost model


class PlannerAgent:
    PROMPT = """Decompose this research query into exactly 3 focused sub-questions
that together cover the query completely. Respond in JSON only:
{{"sub_questions": ["question 1", "question 2", "question 3"]}}

Query: {query}

Each sub-question should be self-contained and answerable from academic
literature or web sources. Do not repeat the original query verbatim."""

    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    async def plan(self, query: str) -> list[str]:
        response = await self.llm.ainvoke(self.PROMPT.format(query=query))
        parsed = json.loads(response.content)
        sub_questions = parsed["sub_questions"]
        if not isinstance(sub_questions, list) or len(sub_questions) != 3:
            raise ValueError(f"Planner did not return exactly 3 sub-questions: {sub_questions!r}")
        return sub_questions


_default_agent: PlannerAgent | None = None


def _get_default_agent() -> PlannerAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = PlannerAgent()
    return _default_agent


async def planner_node(state: AcademicResearchState) -> dict:
    sub_questions = await _get_default_agent().plan(state["query"])
    return {
        "planner_queries": sub_questions,
        "status": "planning",
        **llm_call_update(state),
    }
