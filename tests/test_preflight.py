"""Tests for HyDEOperator: the length and cosine-similarity
quality gates, plus one real (live LLM) integration check.

Gate tests use a fake LLM client so the boundary conditions (word count,
cosine similarity) are deterministic rather than depending on whatever a
live model happens to generate.
"""
import os

import numpy as np
import pytest

from backend.nodes.preflight import HyDEOperator


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, content: str):
        self._content = content

    async def ainvoke(self, prompt: str):
        return FakeResponse(self._content)


class TestHyDEDisabled:
    @pytest.mark.asyncio
    async def test_disabled_returns_raw_query_without_llm_call(self):
        op = HyDEOperator(llm=FakeLLM("should never be used"))
        payload, used = await op.execute("what is HyDE?", hyde_enabled=False)
        assert payload == "what is HyDE?"
        assert used is False


class TestLengthQualityGate:
    @pytest.mark.asyncio
    async def test_short_hypothetical_doc_falls_back(self):
        short_doc = "HyDE improves retrieval."  # well under 50 words
        op = HyDEOperator(llm=FakeLLM(short_doc))
        payload, used = await op.execute("what is HyDE?", hyde_enabled=True)
        assert payload == "what is HyDE?"
        assert used is False

    @pytest.mark.asyncio
    async def test_long_relevant_doc_passes_length_gate(self):
        long_relevant_doc = (
            "Hypothetical Document Embeddings (HyDE) is a retrieval technique that generates a "
            "hypothetical answer document for a given query using a language model, then embeds "
            "that hypothetical document and uses it to search a dense vector index. This approach "
            "closes the semantic gap between short interrogative queries and long declarative "
            "documents, since queries and passages naturally occupy different regions of "
            "embedding space. By retrieving based on a generated document's embedding rather than "
            "the raw query embedding, HyDE improves recall on tasks where the query phrasing "
            "diverges substantially from how the answer is typically written in source documents."
        )
        op = HyDEOperator(llm=FakeLLM(long_relevant_doc))
        payload, used = await op.execute(
            "How does HyDE improve retrieval-augmented generation?", hyde_enabled=True
        )
        assert used is True
        assert payload == long_relevant_doc


class FakeOrthogonalEncoder:
    """Returns orthogonal vectors for the query vs. everything else, so
    cosine_sim is deterministically 0 — isolates gate 2's logic from
    whatever a real embedding model happens to produce for any given pair
    of texts (see note below on why real text doesn't reliably do this)."""

    def __init__(self, query_text: str):
        self.query_text = query_text

    def encode(self, texts):
        return np.array([[1.0, 0.0]]) if texts[0] == self.query_text else np.array([[0.0, 1.0]])


class TestSemanticQualityGate:
    @pytest.mark.asyncio
    async def test_unrelated_embedding_falls_back(self):
        # Empirical note: with BAAI/bge-small-en-v1.5, real "unrelated"
        # 50+-word English text against this query measured 0.35-0.49
        # cosine similarity in manual testing (recipe, sports commentary,
        # gibberish, random word salad, even French and digit strings) —
        # all *above* MIN_QUALITY_COSINE_SIM=0.3. That's a real property
        # of this embedding model's anisotropic baseline similarity, not
        # a bug: it means gate 2 rarely fires in practice with this
        # model/query combination. To test the gate's logic deterministically
        # regardless of that baseline, this test mocks the encoder instead
        # of hunting for real text that happens to score low enough.
        query = "How does HyDE improve retrieval-augmented generation?"
        long_doc = " ".join(["placeholder"] * 60)  # 60 words, clears the length gate
        op = HyDEOperator(llm=FakeLLM(long_doc), encoder=FakeOrthogonalEncoder(query))
        payload, used = await op.execute(query, hyde_enabled=True)
        assert payload == query
        assert used is False


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="live OpenAI call — skipped when no API key (e.g. CI)",
)
class TestRealIntegration:
    """Live OpenAI call — confirms the operator works end-to-end, not just
    its gate boundary logic."""

    @pytest.mark.asyncio
    async def test_real_hyde_call_produces_relevant_payload(self):
        op = HyDEOperator()
        payload, used = await op.execute(
            "How does HyDE improve retrieval-augmented generation?", hyde_enabled=True
        )
        assert used is True
        assert len(payload.split()) >= 50
        assert payload.lower() != "how does hyde improve retrieval-augmented generation?"
