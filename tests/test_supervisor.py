"""Tests for SupervisorAgent classification and its node wrappers
(Section 4.1).

route_after_supervisor's Send-fan-out routing already has dedicated tests
in tests/test_state_machine.py — it's Week 1 code (lives in graph.py).
This file covers what's new in Week 4: SupervisorAgent.classify()'s real
LLM classification (live gpt-4o-mini calls, representative query types),
and the supervisor_node / merge_results_node wrappers around it.
"""
import os

import pytest

from backend.nodes.supervisor import SupervisorAgent, merge_results_node, supervisor_node
from backend.state import new_job_state


def base_state(**overrides):
    state = new_job_state(job_id="test-job", query="placeholder")
    state.update(overrides)
    return state


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="live gpt-4o-mini calls — skipped when no API key (e.g. CI)",
)
class TestSupervisorAgentClassification:
    """Real gpt-4o-mini calls — classification correctness on
    representative query types (Section 4.1's decision logic)."""

    @staticmethod
    @pytest.fixture(scope="class")
    def agent():
        return SupervisorAgent()

    @pytest.mark.asyncio
    async def test_academic_query_routes_to_arxiv(self, agent):
        decision = await agent.classify(
            "What retrieval methods do recent RAG papers benchmark against dense passage retrieval?"
        )
        assert decision["use_arxiv"] is True
        assert isinstance(decision["reason"], str) and decision["reason"]

    @pytest.mark.asyncio
    async def test_current_events_query_routes_to_web(self, agent):
        decision = await agent.classify("What AI product launches were announced this week?")
        assert decision["use_web"] is True

    @pytest.mark.asyncio
    async def test_hybrid_query_routes_to_both(self, agent):
        decision = await agent.classify(
            "Compare the HyDE paper's approach to dense retrieval benchmarks, "
            "and also summarize any AI conference announcements from this month."
        )
        assert decision["use_arxiv"] is True
        assert decision["use_web"] is True

    @pytest.mark.asyncio
    async def test_response_is_valid_decision_shape(self, agent):
        decision = await agent.classify("How does attention work in transformers?")
        assert set(decision.keys()) >= {"use_arxiv", "use_web", "reason"}
        assert isinstance(decision["use_arxiv"], bool)
        assert isinstance(decision["use_web"], bool)


class TestSupervisorNode:
    """supervisor_node's budget accounting, isolated from the real LLM
    call via a monkeypatched agent — this shouldn't cost a live call just
    to check total_llm_calls bookkeeping."""

    @pytest.mark.asyncio
    async def test_increments_llm_call_count(self, monkeypatch):
        import backend.nodes.supervisor as supervisor_module

        class FakeAgent:
            async def classify(self, query):
                return {"use_arxiv": True, "use_web": False, "reason": "fake"}

        monkeypatch.setattr(supervisor_module, "_get_default_agent", lambda: FakeAgent())

        state = base_state(query="anything", total_llm_calls=2, max_llm_calls=15)
        result = await supervisor_node(state)

        assert result["supervisor_decision"] == {"use_arxiv": True, "use_web": False, "reason": "fake"}
        assert result["total_llm_calls"] == 3
        assert result["llm_budget_exceeded"] is False

    @pytest.mark.asyncio
    async def test_sets_budget_exceeded_when_call_hits_cap(self, monkeypatch):
        import backend.nodes.supervisor as supervisor_module

        class FakeAgent:
            async def classify(self, query):
                return {"use_arxiv": True, "use_web": False, "reason": "fake"}

        monkeypatch.setattr(supervisor_module, "_get_default_agent", lambda: FakeAgent())

        state = base_state(query="anything", total_llm_calls=14, max_llm_calls=15)
        result = await supervisor_node(state)
        assert result["llm_budget_exceeded"] is True


class TestMergeResultsNode:
    def test_merges_arxiv_and_web_chunks(self):
        state = base_state(
            arxiv_chunks=[{"content": "a1"}, {"content": "a2"}],
            web_chunks=[{"content": "w1"}],
        )
        result = merge_results_node(state)
        assert result["merged_chunks"] == [
            {"content": "a1"},
            {"content": "a2"},
            {"content": "w1"},
        ]

    def test_handles_empty_arxiv_chunks(self):
        state = base_state(arxiv_chunks=[], web_chunks=[{"content": "w1"}])
        result = merge_results_node(state)
        assert result["merged_chunks"] == [{"content": "w1"}]

    def test_handles_both_empty(self):
        state = base_state(arxiv_chunks=[], web_chunks=[])
        result = merge_results_node(state)
        assert result["merged_chunks"] == []

    def test_returns_merged_chunks_and_status(self):
        """merge_results is the fan-in point after the ArXiv/Web Send
        dispatch, so it owns the researching -> context_eval status
        transition (the sub-agents can't safely write status; they run
        concurrently and would clash on the LastValue channel)."""
        state = base_state(arxiv_chunks=[{"content": "a"}], web_chunks=[])
        result = merge_results_node(state)
        assert set(result.keys()) == {"merged_chunks", "status"}
        assert result["status"] == "context_eval"
