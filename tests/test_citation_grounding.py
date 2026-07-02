"""Tests for evaluation/citation_grounding.py (Section 4.9).

The core failure mode being caught here: a sentence with a structurally
valid [N] citation, but whose content doesn't match the cited chunk's
text. Uses a fake encoder for boundary/downgrade tests so the mechanics
are deterministic, and one real-encoder integration test so we know the
whole path actually detects a misattribution end-to-end.
"""
import numpy as np
import pytest

from evaluation.citation_grounding import (
    GROUNDING_THRESHOLD,
    _downgrade,
    _extract_cited_indices,
    check_citation_grounding,
)


class FakeEncoder:
    """Encoder that maps each unique input string to a fixed 2-D vector.

    Grounded pairs (sentence and cited-chunk content share a marker
    substring) map to the same vector -> cosine sim = 1.0. Ungrounded
    pairs (different marker) map to orthogonal vectors -> sim = 0.0.
    """

    def __init__(self, groups: dict[str, np.ndarray]):
        # groups: marker -> vector (length 2)
        self._groups = groups

    def _vec(self, text: str) -> np.ndarray:
        for marker, vec in self._groups.items():
            if marker in text:
                return vec
        return np.array([0.0, 0.0])

    def encode(self, texts: list[str], normalize_embeddings: bool = False):
        arr = np.stack([self._vec(t) for t in texts])
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
            arr = arr / norms
        return arr


class TestUtilities:
    def test_downgrade_approve_to_escalate(self):
        assert _downgrade("approve") == "escalate"

    def test_downgrade_escalate_to_reject(self):
        assert _downgrade("escalate") == "reject"

    def test_downgrade_reject_stays_reject(self):
        assert _downgrade("reject") == "reject"

    def test_extract_cited_indices_basic(self):
        assert _extract_cited_indices("body [1] and [3].", n_chunks=5) == [1, 3]

    def test_extract_cited_indices_dedupes(self):
        assert _extract_cited_indices("[1] and again [1].", n_chunks=5) == [1]

    def test_extract_cited_indices_filters_out_of_range(self):
        assert _extract_cited_indices("[1] and [99].", n_chunks=5) == [1]

    def test_extract_cited_indices_ignores_non_integer(self):
        assert _extract_cited_indices("[1] and [source_id].", n_chunks=5) == [1]


class TestCascadeAlignment:
    def test_returns_original_when_sentence_count_diverges(self):
        """If sentence-count and cascade-length don't line up, we can't
        safely per-sentence-downgrade — pass through unchanged rather
        than mis-align."""
        draft = "First. Second. Third."
        cascade = ["approve", "approve"]  # 2 vs 3 — length mismatch
        report = check_citation_grounding(
            draft, cascade, merged_chunks=[{"content": "x"}], encoder=FakeEncoder({})
        )
        assert report.updated_cascade == cascade
        assert report.n_ungrounded == 0

    def test_no_cited_sentences_passes_through(self):
        """Uncited sentences are L2a's problem, not L2b's."""
        draft = "First. Second."
        cascade = ["approve", "escalate"]
        report = check_citation_grounding(
            draft, cascade, merged_chunks=[{"content": "x"}], encoder=FakeEncoder({})
        )
        assert report.updated_cascade == ["approve", "escalate"]
        assert report.n_ungrounded == 0


class TestGroundedVsUngrounded:
    """Fake encoder: sentence body says "topic-A" and chunk 1's content
    says "topic-A" -> cosine sim 1.0 (grounded). Sentence says "topic-A"
    but chunk 2 says "topic-B" -> cosine sim 0.0 (ungrounded)."""

    def _run(self, draft, cascade, merged_chunks):
        encoder = FakeEncoder(
            {"topic-A": np.array([1.0, 0.0]), "topic-B": np.array([0.0, 1.0])}
        )
        return check_citation_grounding(draft, cascade, merged_chunks, encoder)

    def test_grounded_sentence_not_downgraded(self):
        report = self._run(
            draft="Body about topic-A [1].",
            cascade=["approve"],
            merged_chunks=[{"content": "topic-A chunk"}],
        )
        assert report.updated_cascade == ["approve"]
        assert report.n_ungrounded == 0
        assert report.per_sentence[0]["sim"] == pytest.approx(1.0)

    def test_ungrounded_approve_downgrades_to_escalate(self):
        report = self._run(
            draft="Body about topic-A [1].",
            cascade=["approve"],
            merged_chunks=[{"content": "topic-B chunk"}],
        )
        assert report.updated_cascade == ["escalate"]
        assert report.n_ungrounded == 1
        assert report.per_sentence[0]["sim"] == pytest.approx(0.0)

    def test_ungrounded_escalate_downgrades_to_reject(self):
        report = self._run(
            draft="Body about topic-A [1].",
            cascade=["escalate"],
            merged_chunks=[{"content": "topic-B chunk"}],
        )
        assert report.updated_cascade == ["reject"]
        assert report.n_ungrounded == 1

    def test_ungrounded_reject_stays_reject(self):
        report = self._run(
            draft="Body about topic-A [1].",
            cascade=["reject"],
            merged_chunks=[{"content": "topic-B chunk"}],
        )
        assert report.updated_cascade == ["reject"]
        # Not counted as newly-downgraded since it was already reject.
        assert report.n_ungrounded == 0

    def test_multiple_cited_chunks_use_centroid(self):
        """Sentence cites two chunks; if EITHER supports it, centroid
        stays close enough. Here both grounded."""
        report = self._run(
            draft="Body about topic-A [1] [2].",
            cascade=["approve"],
            merged_chunks=[
                {"content": "topic-A chunk"},
                {"content": "topic-A other"},
            ],
        )
        assert report.updated_cascade == ["approve"]

    def test_summary_names_bad_sentences(self):
        report = self._run(
            draft="Body about topic-A [1]. Second about topic-A [1].",
            cascade=["approve", "approve"],
            merged_chunks=[{"content": "topic-B chunk"}],
        )
        assert "sent#1" in report.summary()
        assert "sent#2" in report.summary()
        assert "cited[1]" in report.summary()


class TestThresholdBoundary:
    """Sentences with sim >= threshold stay; below downgrades."""

    def _encoder_at_sim(self, sim: float) -> FakeEncoder:
        # Two unit vectors with cosine sim = `sim`: [1, 0] and [sim, sqrt(1-sim^2)]
        return FakeEncoder(
            {
                "sent-marker": np.array([1.0, 0.0]),
                "chunk-marker": np.array([sim, (1 - sim * sim) ** 0.5]),
            }
        )

    def test_clearly_above_threshold_is_grounded(self):
        report = check_citation_grounding(
            draft="sent-marker text [1].",
            cascade_decisions=["approve"],
            merged_chunks=[{"content": "chunk-marker text"}],
            encoder=self._encoder_at_sim(GROUNDING_THRESHOLD + 0.05),
        )
        assert report.updated_cascade == ["approve"]

    def test_just_below_threshold_downgrades(self):
        report = check_citation_grounding(
            draft="sent-marker text [1].",
            cascade_decisions=["approve"],
            merged_chunks=[{"content": "chunk-marker text"}],
            encoder=self._encoder_at_sim(GROUNDING_THRESHOLD - 0.05),
        )
        assert report.updated_cascade == ["escalate"]


class TestRealIntegration:
    """Real bge-small-en-v1.5 encoder. On-topic sentence + on-topic
    chunk should score above threshold; on-topic sentence + off-topic
    chunk (the "misattribution" failure mode) should score below."""

    @staticmethod
    @pytest.fixture(scope="class")
    def encoder():
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("BAAI/bge-small-en-v1.5")

    def test_grounded_pair_stays(self, encoder):
        report = check_citation_grounding(
            draft=(
                "Retrieval-augmented generation grounds language model outputs "
                "in retrieved documents [1]."
            ),
            cascade_decisions=["approve"],
            merged_chunks=[
                {
                    "content": (
                        "We introduce RAG, an approach that combines a "
                        "retriever with a language model to ground outputs "
                        "in retrieved documents."
                    )
                }
            ],
            encoder=encoder,
        )
        # Real integration: exact sim varies, but this pair is on-topic
        # enough it should stay approved.
        assert report.updated_cascade == ["approve"]

    def test_misattribution_pair_downgrades(self, encoder):
        """Simulates the exact failure mode from the n=10 labeling
        exercise: sentence describes 'FG-PRM' but cited chunk is about
        REFIND. Should downgrade approve -> escalate."""
        report = check_citation_grounding(
            draft=(
                "FG-PRM is a fine-grained process reward model that categorizes "
                "hallucinations into six types [1]."
            ),
            cascade_decisions=["approve"],
            merged_chunks=[
                {
                    "content": (
                        "REFIND is a retrieval-augmented method for evaluating "
                        "factuality of language model outputs by checking whether "
                        "each generated claim is supported by retrieved passages."
                    )
                }
            ],
            encoder=encoder,
        )
        assert report.updated_cascade == ["escalate"]
        assert report.n_ungrounded == 1
