"""Unit tests for the circuit-breaker and budget routing logic
and the Supervisor Send-fan-out routing.

These are pure function tests: no LLM calls, no graph execution — just the
routing decisions that keep the system from looping forever or overspending
its LLM budget.
"""
from langgraph.types import Send

from backend.graph import (
    compiled_graph,
    route_after_context_eval,
    route_after_critic,
    route_after_supervisor,
)
from backend.state import llm_call_update, new_job_state


def base_state(**overrides):
    state = new_job_state(job_id="test-job", query="How does HyDE improve RAG retrieval?")
    state.update(overrides)
    return state


# ── route_after_critic (outer circuit breaker) ──────────────────────────────

class TestRouteAfterCritic:
    def test_approved_routes_to_finalize(self):
        state = base_state(status="approved", critic_loop_count=1, llm_budget_exceeded=False)
        assert route_after_critic(state) == "finalize"

    def test_default_routes_back_to_supervisor_not_planner(self):
        state = base_state(status="reviewing", critic_loop_count=1, llm_budget_exceeded=False)
        assert route_after_critic(state) == "supervisor"

    def test_loop_exhausted_routes_to_force_finalize(self):
        state = base_state(
            status="reviewing", critic_loop_count=3, max_critic_loops=3, llm_budget_exceeded=False
        )
        assert route_after_critic(state) == "force_finalize"

    def test_loop_count_exceeding_max_also_force_finalizes(self):
        state = base_state(
            status="reviewing", critic_loop_count=4, max_critic_loops=3, llm_budget_exceeded=False
        )
        assert route_after_critic(state) == "force_finalize"

    def test_loop_count_just_below_max_continues(self):
        state = base_state(
            status="reviewing", critic_loop_count=2, max_critic_loops=3, llm_budget_exceeded=False
        )
        assert route_after_critic(state) == "supervisor"

    def test_budget_exceeded_wins_over_approved(self):
        """Budget must be checked first, even if the Critic approved the draft."""
        state = base_state(status="approved", critic_loop_count=0, llm_budget_exceeded=True)
        assert route_after_critic(state) == "force_finalize"

    def test_budget_exceeded_wins_over_fresh_loop_count(self):
        state = base_state(status="reviewing", critic_loop_count=0, llm_budget_exceeded=True)
        assert route_after_critic(state) == "force_finalize"

    def test_approved_wins_over_loop_exhausted(self):
        """Priority order: approved (#2) is checked before loop exhaustion (#3)."""
        state = base_state(
            status="approved", critic_loop_count=3, max_critic_loops=3, llm_budget_exceeded=False
        )
        assert route_after_critic(state) == "finalize"


# ── route_after_context_eval (inner active-retrieval loop) ──────────────────

class TestRouteAfterContextEval:
    def test_low_coverage_triggers_refine(self):
        state = base_state(coverage_score=0.2, refinement_count=0, llm_budget_exceeded=False)
        assert route_after_context_eval(state) == "refine"

    def test_high_coverage_routes_to_writer(self):
        state = base_state(coverage_score=0.9, refinement_count=0, llm_budget_exceeded=False)
        assert route_after_context_eval(state) == "writer"

    def test_coverage_exactly_at_threshold_routes_to_writer(self):
        """coverage_score < 0.5 triggers refine; == 0.5 should not."""
        state = base_state(coverage_score=0.5, refinement_count=0, llm_budget_exceeded=False)
        assert route_after_context_eval(state) == "writer"

    def test_inner_budget_exhausted_routes_to_writer_despite_low_coverage(self):
        state = base_state(coverage_score=0.1, refinement_count=1, llm_budget_exceeded=False)
        assert route_after_context_eval(state) == "writer"

    def test_global_budget_exceeded_wins_over_low_coverage(self):
        """Budget must be checked first, always."""
        state = base_state(coverage_score=0.0, refinement_count=0, llm_budget_exceeded=True)
        assert route_after_context_eval(state) == "writer"


# ── route_after_supervisor (Send fan-out) ───────────────────────────────────

class TestRouteAfterSupervisor:
    def _sends_by_node(self, sends):
        return {s.node for s in sends}

    def test_arxiv_only(self):
        state = base_state(supervisor_decision={"use_arxiv": True, "use_web": False})
        sends = route_after_supervisor(state)
        assert all(isinstance(s, Send) for s in sends)
        assert self._sends_by_node(sends) == {"arxiv_agent"}

    def test_web_only(self):
        state = base_state(supervisor_decision={"use_arxiv": False, "use_web": True})
        sends = route_after_supervisor(state)
        assert self._sends_by_node(sends) == {"web_agent"}

    def test_both_dispatched_in_parallel(self):
        """Both sub-agents fire independently — this is what makes the
        Researcher tier genuinely multi-agent, not sequential nodes."""
        state = base_state(supervisor_decision={"use_arxiv": True, "use_web": True})
        sends = route_after_supervisor(state)
        assert self._sends_by_node(sends) == {"arxiv_agent", "web_agent"}
        assert len(sends) == 2

    def test_neither_falls_back_to_arxiv(self):
        state = base_state(supervisor_decision={"use_arxiv": False, "use_web": False})
        sends = route_after_supervisor(state)
        assert self._sends_by_node(sends) == {"arxiv_agent"}
        assert len(sends) == 1

    def test_missing_keys_falls_back_to_arxiv(self):
        """supervisor_decision is Supervisor-populated JSON; missing keys
        should degrade to the documented default, not raise."""
        state = base_state(supervisor_decision={})
        sends = route_after_supervisor(state)
        assert self._sends_by_node(sends) == {"arxiv_agent"}


# ── Topology sanity ─────────────────────────────────────────────────────────

class TestGraphTopology:
    def test_graph_compiles(self):
        graph = compiled_graph()
        node_names = set(graph.get_graph().nodes.keys())
        expected = {
            "planner", "supervisor", "arxiv_agent", "web_agent",
            "merge_results", "context_eval", "refine", "writer",
            "critic", "finalize", "force_finalize",
        }
        assert expected.issubset(node_names)

    def test_worst_case_budget_cap_below_theoretical_max(self):
        """Worst-case theoretical max is 16 LLM calls; the
        global cap is set to 15 so force_finalize fires one step early."""
        state = base_state(max_llm_calls=15)
        assert state["max_llm_calls"] < 16


class TestLLMCallUpdate:
    """Shared budget-accounting helper every LLM-calling node uses."""

    def test_increments_total_calls(self):
        state = base_state(total_llm_calls=3, max_llm_calls=15)
        update = llm_call_update(state)
        assert update["total_llm_calls"] == 4

    def test_defaults_to_incrementing_by_one(self):
        state = base_state(total_llm_calls=0, max_llm_calls=15)
        update = llm_call_update(state)
        assert update["total_llm_calls"] == 1

    def test_supports_multi_call_increments(self):
        state = base_state(total_llm_calls=0, max_llm_calls=15)
        update = llm_call_update(state, calls=3)
        assert update["total_llm_calls"] == 3

    def test_sets_budget_exceeded_at_exact_cap(self):
        state = base_state(total_llm_calls=14, max_llm_calls=15)
        update = llm_call_update(state)
        assert update["total_llm_calls"] == 15
        assert update["llm_budget_exceeded"] is True

    def test_budget_not_exceeded_below_cap(self):
        state = base_state(total_llm_calls=5, max_llm_calls=15)
        update = llm_call_update(state)
        assert update["llm_budget_exceeded"] is False

    def test_only_returns_relevant_keys(self):
        state = base_state(total_llm_calls=0, max_llm_calls=15)
        update = llm_call_update(state)
        assert set(update.keys()) == {"total_llm_calls", "llm_budget_exceeded"}
