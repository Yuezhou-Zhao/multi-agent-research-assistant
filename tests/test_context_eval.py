"""Tests for context_eval_node (Section 4.5) — coverage scoring via
embedding cosine similarity, zero LLM calls.
"""
from backend.nodes.context_eval import context_eval_node
from backend.state import new_job_state


def base_state(**overrides):
    state = new_job_state(job_id="test-job", query="How does HyDE improve RAG retrieval?")
    state.update(overrides)
    return state


class TestContextEvalNode:
    def test_no_chunks_returns_zero_coverage(self):
        state = base_state(planner_queries=["what is HyDE?"], verified_chunks=[])
        result = context_eval_node(state)
        assert result["coverage_score"] == 0.0

    def test_no_sub_questions_returns_zero_coverage(self):
        state = base_state(
            planner_queries=[], verified_chunks=[{"content": "HyDE generates a hypothetical document."}]
        )
        result = context_eval_node(state)
        assert result["coverage_score"] == 0.0

    def test_relevant_chunk_gives_high_coverage(self):
        state = base_state(
            planner_queries=["How does HyDE improve retrieval accuracy?"],
            verified_chunks=[
                {
                    "content": (
                        "HyDE (Hypothetical Document Embeddings) generates a hypothetical "
                        "answer document and embeds it, improving retrieval by matching "
                        "query intent to declarative document style."
                    )
                }
            ],
        )
        result = context_eval_node(state)
        assert result["coverage_score"] > 0.5

    def test_irrelevant_chunk_gives_low_coverage(self):
        state = base_state(
            planner_queries=["How does HyDE improve retrieval accuracy?"],
            verified_chunks=[{"content": "The recipe calls for two cups of flour and a pinch of salt."}],
        )
        result = context_eval_node(state)
        assert result["coverage_score"] < 0.5

    def test_coverage_is_worst_covered_sub_question(self):
        """coverage_score = min across sub-questions of their best-matching
        chunk — one well-covered question should not mask a gap elsewhere."""
        state = base_state(
            planner_queries=[
                "How does HyDE improve retrieval accuracy?",
                "What is the capital of France?",
            ],
            verified_chunks=[
                {
                    "content": (
                        "HyDE generates a hypothetical answer document to improve dense "
                        "retrieval by aligning query and document embedding styles."
                    )
                }
            ],
        )
        result = context_eval_node(state)
        # The France question has no matching chunk at all, so the min
        # (worst-covered) should pull the overall score down.
        assert result["coverage_score"] < 0.5

    def test_only_returns_coverage_score_key(self):
        """Node should return a partial update, not the full state."""
        state = base_state(
            planner_queries=["q"], verified_chunks=[{"content": "some content"}]
        )
        result = context_eval_node(state)
        assert set(result.keys()) == {"coverage_score"}
