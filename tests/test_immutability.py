"""Config immutability rule — mid-job sidebar changes must not
affect a running job's snapshotted state.

The rule is enforced structurally: frontend/app.py's on_message reads
cl.user_session ONCE into new_job_state() and everything downstream
reads from state, never from cl.user_session again. These tests lock
that contract as a testable invariant even without a running Chainlit
server:

  1. new_job_state() returns an independent state per call — mutating a
     later call's state (or the settings dict feeding it) cannot leak
     back into an earlier call's state.
  2. The graph code path never imports chainlit. That's the enforcement
     mechanism: even if someone edited a node to read cl.user_session,
     the import would fail in a unit test. This structural check catches
     that regression at CI time, not at demo time.
"""
from backend.state import new_job_state


class TestSnapshotIndependence:
    def test_two_jobs_have_independent_config(self):
        """Different sidebar values at different submits produce
        distinct state dicts — a job created at sf_threshold=0.15 keeps
        that value even after a later job is created at 0.5."""
        job_a = new_job_state(job_id="a", query="q1", sf_threshold=0.15, hyde_enabled=True)
        job_b = new_job_state(job_id="b", query="q2", sf_threshold=0.50, hyde_enabled=False)

        assert job_a["sf_threshold"] == 0.15
        assert job_a["hyde_enabled"] is True
        assert job_b["sf_threshold"] == 0.50
        assert job_b["hyde_enabled"] is False

    def test_mutating_state_after_creation_does_not_affect_a_sibling(self):
        """Two states created back-to-back should not share dict
        references; overwriting one's sf_threshold has no effect on the
        other. This is the structural version of "sliders moved after
        submit cannot affect a running job."""
        job_a = new_job_state(job_id="a", query="q1", sf_threshold=0.15)
        job_b = new_job_state(job_id="b", query="q2", sf_threshold=0.15)

        # Simulate the user dragging the slider mid-run — but only the
        # NEW job's snapshot changes. The old one is a plain TypedDict
        # dict; nothing about creating job_b touches job_a's memory.
        job_b["sf_threshold"] = 0.9

        assert job_a["sf_threshold"] == 0.15

    def test_list_and_dict_slices_are_independent(self):
        """The mutable fields (planner_queries, merged_chunks, etc.) must
        be fresh containers per job, not shared class-level defaults."""
        job_a = new_job_state(job_id="a", query="q1")
        job_b = new_job_state(job_id="b", query="q2")

        job_a["planner_queries"].append("cross-talk")
        job_a["merged_chunks"].append({"source": "x", "content": "y"})
        job_a["node_latencies"]["writer"] = 1.23

        assert job_b["planner_queries"] == []
        assert job_b["merged_chunks"] == []
        assert job_b["node_latencies"] == {}

    def test_budget_caps_snapshotted_per_job(self):
        job_a = new_job_state(job_id="a", query="q1", max_llm_calls=15, max_critic_loops=3)
        job_b = new_job_state(job_id="b", query="q2", max_llm_calls=5, max_critic_loops=1)
        assert job_a["max_llm_calls"] == 15
        assert job_a["max_critic_loops"] == 3
        assert job_b["max_llm_calls"] == 5
        assert job_b["max_critic_loops"] == 1


class TestBackendIsUiIndependent:
    """The graph layer must not read from chainlit's user_session —
    that's the structural enforcement of the snapshot rule. If a future edit
    slips a `cl.user_session.get(...)` into a node, this test catches it
    at CI time rather than at demo time."""

    def test_no_chainlit_import_in_backend_or_rag_or_evaluation(self):
        import ast
        from pathlib import Path

        roots = [
            Path(__file__).resolve().parent.parent / "backend",
            Path(__file__).resolve().parent.parent / "rag",
            Path(__file__).resolve().parent.parent / "evaluation",
        ]
        offenders = []
        for root in roots:
            for py in root.rglob("*.py"):
                try:
                    tree = ast.parse(py.read_text())
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "chainlit" or alias.name.startswith("chainlit."):
                                offenders.append(str(py))
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and (
                            node.module == "chainlit" or node.module.startswith("chainlit.")
                        ):
                            offenders.append(str(py))

        assert not offenders, (
            f"chainlit imports leaked into non-UI code, which would let a "
            f"running-job read sidebar state and break the per-job snapshot "
            f"rule: {offenders}"
        )
