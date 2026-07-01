"""LangGraph orchestrator — topology + routing/circuit-breaker logic.

All node bodies are real as of Week 5 (Section 7.2) — this module is now
purely the wiring layer between them plus the three routing functions
(Section 4.6 / 4.1). Node logic lives in backend/nodes/*.py; the HyDE
pre-flight operator (Section 4.3) is intentionally NOT a node here — per
Section 2.1 it runs once, outside the state machine, before the graph is
invoked, with its output frozen into state["search_payload"] by
backend/main.py.
"""
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from backend.nodes.arxiv_agent import arxiv_agent_node
from backend.nodes.context_eval import context_eval_node
from backend.nodes.critic import critic_node
from backend.nodes.finalizer import finalize_node, force_finalize_node
from backend.nodes.planner import planner_node
from backend.nodes.refine import refine_node
from backend.nodes.supervisor import merge_results_node, supervisor_node
from backend.nodes.web_agent import web_agent_node
from backend.nodes.writer import writer_node
from backend.state import AcademicResearchState


# ── Routing functions (Section 4.6 / 4.1) — final logic, fully tested ──────

def route_after_supervisor(state: AcademicResearchState) -> list[Send]:
    """Fan-out to the ArXiv and/or Web sub-agents via LangGraph's Send API.

    This is what makes the Researcher tier true Multi-Agent (Section 1.2):
    concurrent, independent sub-agent dispatch rather than sequential nodes
    sharing state. Falls back to arxiv_agent if the Supervisor's decision
    names neither source.
    """
    sends = []
    decision = state["supervisor_decision"]

    if decision.get("use_arxiv"):
        sends.append(Send("arxiv_agent", state))
    if decision.get("use_web"):
        sends.append(Send("web_agent", state))

    if not sends:  # fallback: always use arxiv
        sends.append(Send("arxiv_agent", state))

    return sends


def route_after_context_eval(state: AcademicResearchState) -> str:
    """Inner loop routing (refinement_count, max 1). Budget check first, always.

    Priority order:
      1. Global budget exceeded -> writer (degrade gracefully, use what we have)
      2. Inner budget (refinement_count >= 1) exhausted -> writer
      3. coverage_score < 0.5 -> refine (trigger one refined search)
      4. Default -> writer
    """
    if state["llm_budget_exceeded"]:
        return "writer"
    if state["refinement_count"] >= 1:
        return "writer"
    if state["coverage_score"] < 0.5:
        return "refine"
    return "writer"


def route_after_critic(state: AcademicResearchState) -> str:
    """Outer circuit breaker (critic_loop_count, max 3) + global LLM budget.

    Priority order (budget must be checked FIRST):
      1. Global budget exceeded -> force_finalize (cost control)
      2. Approved -> finalize
      3. Outer loop exhausted -> force_finalize (graceful degradation)
      4. Default -> supervisor (restart from Supervisor, not Planner)
    """
    if state["llm_budget_exceeded"]:
        return "force_finalize"
    if state["status"] == "approved":
        return "finalize"
    if state["critic_loop_count"] >= state["max_critic_loops"]:
        return "force_finalize"
    return "supervisor"


# ── Graph topology ──────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Wire up the node topology described in Section 2.1's diagram.

    planner -> supervisor -(Send fan-out)-> {arxiv_agent, web_agent}
            -> merge_results -> context_eval -(route)-> {refine -> context_eval, writer}
            -> writer -> critic -(route)-> {finalize, force_finalize, supervisor}
    """
    graph = StateGraph(AcademicResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("arxiv_agent", arxiv_agent_node)
    graph.add_node("web_agent", web_agent_node)
    graph.add_node("merge_results", merge_results_node)
    graph.add_node("context_eval", context_eval_node)
    graph.add_node("refine", refine_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("force_finalize", force_finalize_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        ["arxiv_agent", "web_agent"],
    )
    graph.add_edge("arxiv_agent", "merge_results")
    graph.add_edge("web_agent", "merge_results")
    graph.add_edge("merge_results", "context_eval")

    graph.add_conditional_edges(
        "context_eval",
        route_after_context_eval,
        {"refine": "refine", "writer": "writer"},
    )
    graph.add_edge("refine", "context_eval")

    graph.add_edge("writer", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "finalize": "finalize",
            "force_finalize": "force_finalize",
            "supervisor": "supervisor",
        },
    )

    graph.add_edge("finalize", END)
    graph.add_edge("force_finalize", END)

    return graph


def compiled_graph():
    return build_graph().compile()
