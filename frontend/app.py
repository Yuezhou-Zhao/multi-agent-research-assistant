"""Chainlit frontend — Week 6, Section 7.2.

Four pieces the design doc calls out:
  1. Nested Steps showing the real execution trace (Planner ->
     Supervisor -> Send fan-out -> sub-agents -> Context Eval -> Writer
     -> Critic), including the rollback path when the outer loop fires
     more than once.
  2. Sidebar controls for sf_threshold + hyde_enabled, snapshotted at
     job submission per Section 3.1's immutability rule — sliders
     moved mid-job cannot affect a running job. This is enforced
     structurally: on_message reads sidebar values ONCE into
     new_job_state(), and the graph then reads from state, never from
     cl.user_session. tests/test_immutability.py locks that contract.
  3. Red-highlight rendering of Gamma-L1-flagged sentences in the
     final draft view, tied to state["cascade_decisions"].
  4. Live metrics panel with total_llm_calls/max, llm_calls_avoided,
     coverage_score, critic_loop_count, and cascade approve/escalate/
     reject counts, refreshed after each node completes.

Launch: `./venv/bin/chainlit run frontend/app.py -w` from repo root.
"""
import uuid

import chainlit as cl
from chainlit.input_widget import Slider, Switch
from dotenv import load_dotenv

load_dotenv()

from backend.graph import compiled_graph
from backend.nodes.preflight import HyDEOperator
from backend.state import new_job_state
from evaluation.gamma_guardrail import GammaGuardrail

# Lazy singletons — HyDEOperator + compiled_graph both trigger model
# loading (encoders, guardrail calibration) at construction, so defer
# them until the first message rather than paying the cost on every
# chat-open ping.
_graph = None
_hyde: HyDEOperator | None = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = compiled_graph()
    return _graph


def _get_hyde() -> HyDEOperator:
    global _hyde
    if _hyde is None:
        _hyde = HyDEOperator()
    return _hyde


# ── 2. Sidebar controls (Section 3.1 snapshot rule) ─────────────────────

DEFAULT_SETTINGS = {"hyde_enabled": True, "sf_threshold": 0.15}


@cl.on_chat_start
async def start():
    await cl.ChatSettings(
        [
            Switch(
                id="hyde_enabled",
                label="HyDE pre-flight",
                initial=DEFAULT_SETTINGS["hyde_enabled"],
            ),
            Slider(
                id="sf_threshold",
                label="Gamma SF threshold (chunk filter)",
                initial=DEFAULT_SETTINGS["sf_threshold"],
                min=0.0,
                max=1.0,
                step=0.01,
            ),
        ]
    ).send()
    cl.user_session.set("settings", dict(DEFAULT_SETTINGS))
    await cl.Message(
        content=(
            "**Academic Research Agent — ready.**\n\n"
            "Ask a research question. HyDE toggle + Gamma SF threshold are in the "
            "sidebar (⚙️).\n\n"
            "Sidebar values are **snapshotted at submit time** — sliders moved while "
            "a job is running have zero effect on that job (Section 3.1 immutability)."
        )
    ).send()


@cl.on_settings_update
async def settings_updated(settings: dict):
    cl.user_session.set("settings", settings)


# ── Helpers for 3 (red-highlight) and 4 (metrics) ──────────────────────

def _cascade_counts(decisions: list[str]) -> tuple[int, int, int]:
    return (
        decisions.count("approve"),
        decisions.count("escalate"),
        decisions.count("reject"),
    )


def _highlight_cascade(draft: str, decisions: list[str]) -> str:
    """Wrap L1-rejected sentences in a red-tinted HTML span. Chainlit
    renders Markdown + inline HTML, so span backgrounds work."""
    sentences = GammaGuardrail._split_sentences(draft)
    if not sentences or len(decisions) != len(sentences):
        # Sanitization can strip citation markers; if the sentence count
        # diverges from the cascade recorded pre-sanitize, don't try to
        # misalign highlights — just show the draft plain.
        return draft
    parts = []
    for sent, decision in zip(sentences, decisions):
        if decision == "reject":
            parts.append(
                f'<span style="background-color: #ff6b6b40; border-radius: 3px; '
                f'padding: 0 3px;">{sent}</span>'
            )
        else:
            parts.append(sent)
    return " ".join(parts)


def _metrics_markdown(state: dict) -> str:
    a, e, r = _cascade_counts(state.get("cascade_decisions") or [])
    total_sents = a + e + r
    return (
        "**Live Metrics**\n\n"
        f"| Metric | Value |\n"
        f"| --- | --- |\n"
        f"| LLM budget | **{state.get('total_llm_calls', 0)} / "
        f"{state.get('max_llm_calls', 15)}** |\n"
        f"| llm_calls_avoided (Gamma / L2 prefilter) | **{state.get('llm_calls_avoided', 0)}** |\n"
        f"| coverage_score | **{state.get('coverage_score', 0.0):.3f}** |\n"
        f"| Critic loops | **{state.get('critic_loop_count', 0)} / "
        f"{state.get('max_critic_loops', 3)}** |\n"
        f"| Cascade (a/e/r) | **{a}** / **{e}** / **{r}** ({total_sents} sentences) |\n"
        f"| Status | `{state.get('status', 'preflight')}` |"
    )


# ── Per-node step body renderers ───────────────────────────────────────

def _describe_supervisor(decision: dict) -> str:
    parts = []
    if decision.get("use_arxiv"):
        parts.append("arxiv")
    if decision.get("use_web"):
        parts.append("web")
    routes = ", ".join(parts) if parts else "arxiv (fallback)"
    reason = decision.get("reason", "")
    return f"routes to: **{routes}**\n\n{reason}"


def _step_body(node: str, update: dict, state: dict) -> str:
    if node == "planner":
        qs = update.get("planner_queries", [])
        return "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs)) or "(empty)"
    if node == "supervisor":
        return _describe_supervisor(update.get("supervisor_decision", {}))
    if node == "arxiv_agent":
        chunks = update.get("arxiv_chunks", [])
        scores = update.get("gamma_scores", [])
        return (
            f"**{len(chunks)}** chunks verified after Gamma chunk-filter\n\n"
            f"SF scores: `{[round(s, 3) for s in scores]}`"
        )
    if node == "web_agent":
        return f"**{len(update.get('web_chunks', []))}** web chunks retrieved"
    if node == "merge_results":
        return f"**{len(update.get('merged_chunks', []))}** chunks merged"
    if node == "context_eval":
        cov = update.get("coverage_score", 0.0)
        # Routing decision is computed by route_after_context_eval in graph.py;
        # we mirror its logic here purely for readable trace display.
        if state.get("llm_budget_exceeded"):
            route = "writer (budget exceeded)"
        elif state.get("refinement_count", 0) >= 1:
            route = "writer (inner budget exhausted)"
        elif cov < 0.5:
            route = "refine"
        else:
            route = "writer"
        return f"coverage_score = **{cov:.3f}** → next: **{route}**"
    if node == "refine":
        return f"refinement_count = {update.get('refinement_count', '?')}"
    if node == "writer":
        d = update.get("draft", "")
        return f"draft ({len(d)} chars):\n\n{d}"
    if node == "critic":
        cascade = update.get("cascade_decisions", [])
        a, e, r = _cascade_counts(cascade)
        status = update.get("status", "?")
        feedback = update.get("critic_feedback")
        lines = [
            f"cascade: **{a}** approve · **{e}** escalate · **{r}** reject",
            f"decision: **{status}**",
        ]
        if feedback:
            lines.append(f"feedback: {feedback}")
        return "\n\n".join(lines)
    if node in ("finalize", "force_finalize"):
        cits = update.get("citations", [])
        return f"citations: {cits}"
    # Fallback — trim so the step doesn't blow up on unexpected shapes
    return str(update)[:800]


# ── 1. Trace + streaming (Section 2.1 topology) ────────────────────────

NODE_STEP_TITLES = {
    "planner": "Planner — decompose query into 3 sub-questions",
    "supervisor": "Supervisor — classify + Send fan-out",
    "arxiv_agent": "ArXiv Sub-Agent — retrieve + Gamma filter",
    "web_agent": "Web Sub-Agent — Tavily search",
    "merge_results": "Merge Results (fan-in, zero LLM)",
    "context_eval": "Context Eval (embedding coverage, zero LLM)",
    "refine": "Refine — targeted retrieval on worst-covered sub-question",
    "writer": "Writer — draft with [N] index citations",
    "critic": "Critic — L1 Gamma → L2 grounding → L3 judge",
    "finalize": "Finalize (approved)",
    "force_finalize": "Force finalize (degraded — circuit breaker or budget cap)",
}


@cl.on_message
async def main(message: cl.Message):
    # ── 2 (again): snapshot sidebar values into state at submit time.
    # ONLY place sidebar values enter the graph. Everything downstream
    # reads state, never cl.user_session.
    settings = cl.user_session.get("settings") or DEFAULT_SETTINGS
    sf_threshold = float(settings.get("sf_threshold", DEFAULT_SETTINGS["sf_threshold"]))
    hyde_enabled = bool(settings.get("hyde_enabled", DEFAULT_SETTINGS["hyde_enabled"]))

    job_id = str(uuid.uuid4())
    state = new_job_state(
        job_id=job_id,
        query=message.content,
        hyde_enabled=hyde_enabled,
        sf_threshold=sf_threshold,
    )

    # 4. Live metrics panel — one message that gets .update()-d after
    # each node completes. Sent first so it stays pinned above the trace.
    metrics_msg = cl.Message(content=_metrics_markdown(state))
    await metrics_msg.send()

    async def refresh_metrics():
        metrics_msg.content = _metrics_markdown(state)
        await metrics_msg.update()

    # HyDE pre-flight runs OUTSIDE the graph (Section 2.1). Trace it as
    # its own top-level Step so the demo shows the pre-flight/state-
    # machine split explicitly.
    async with cl.Step(name="HyDE Pre-flight (outside state machine)", type="tool") as step:
        step.input = f"hyde_enabled={hyde_enabled}"
        search_payload, hyde_used = await _get_hyde().execute(
            state["query"], hyde_enabled
        )
        state["search_payload"] = search_payload
        if hyde_used:
            new_total = state["total_llm_calls"] + 1
            state["total_llm_calls"] = new_total
            state["llm_budget_exceeded"] = new_total >= state["max_llm_calls"]
        step.output = (
            f"hyde_used={hyde_used}\n\n"
            + (
                f"payload preview: {search_payload[:400]}..."
                if hyde_used
                else "using raw query as search_payload"
            )
        )
    await refresh_metrics()

    # ── Stream through the compiled graph, mounting each node update
    #    as a Chainlit Step. astream(stream_mode="updates") yields
    #    {node_name: partial_state} after each node completes — that's
    #    exactly what we want to render as trace + refresh metrics from.
    graph = _get_graph()
    critic_pass = 0

    async for event in graph.astream(state, stream_mode="updates"):
        for node, update in event.items():
            state.update(update)

            if node == "critic":
                critic_pass += 1
                title = (
                    f"Critic pass {critic_pass}/{state['max_critic_loops']} "
                    f"— L1 → L2 → L3"
                )
            else:
                title = NODE_STEP_TITLES.get(node, node)

            async with cl.Step(name=title, type="tool") as step:
                step.output = _step_body(node, update, state)

            await refresh_metrics()

    # ── 3. Final answer with red-highlighted L1 rejects ────────────────
    final_answer = state.get("final_answer", "")
    cascade = state.get("cascade_decisions") or []
    highlighted = _highlight_cascade(final_answer, cascade)
    citations = state.get("citations") or []

    status = state.get("status", "?")
    header = {
        "approved": "✅ Approved",
        "force_finalized": "⚠️ Force-finalized (circuit breaker hit max_critic_loops)",
        "budget_exceeded": "⚠️ Force-finalized (global LLM budget cap)",
        "failed": "❌ Failed",
    }.get(status, f"Status: {status}")

    reason = state.get("failure_reason")
    reason_line = f"\n\n_reason: {reason}_" if reason else ""

    await cl.Message(
        content=(
            f"### {header}{reason_line}\n\n"
            f"{highlighted or '(no draft produced)'}\n\n"
            f"---\n"
            f"**Citations:** {', '.join(citations) or '—'}"
        )
    ).send()
