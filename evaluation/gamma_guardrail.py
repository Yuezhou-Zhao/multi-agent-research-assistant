"""GammaGuardrail — calibrated embedding-distance prefilter (Section 4.7).

FRAMING — read before writing anything about this elsewhere in the repo
or out loud in an interview: this is a calibrated embedding-distance
prefilter, a cheap (zero-LLM, low-latency) way to decide "does this text
look like the trustworthy academic text we calibrated on, in embedding
space." It is NOT a statistical guarantee of factual correctness, and it
is not tied to any specific arXiv preprint's claim. Section 1.3 #4 is
explicit that it "performs on par with simpler baselines (raw cosine
distance / empirical percentile)" — the Gamma modeling is an engineering
choice for a smooth, thresholdable score, not a novel statistical result.
Known limitation (Section 8): the "Semantic Illusion" problem — a
hallucination that is semantically close to a correct answer will not
read as distant in embedding space, so this prefilter will not catch it.
It exists to cut LLM Critic calls on the easy cases, not to catch every
hallucination.

requirements.txt originally scoped this as "local import from
prototype-uncertainty project" — that project isn't part of this repo, so
this module reimplements the same core technique directly with
scipy.stats.gamma (already a Section 7.3 dependency for exactly this).

How it works:
  1. calibrate(reference_texts) embeds a reference corpus of trustworthy
     academic text, takes its centroid, and fits a Gamma distribution to
     the reference population's own cosine-distance-to-centroid values.
     ("What does trustworthy academic text look like in embedding
     space" — Section 4.7.) Section 4.7 calibrates on arXiv high-citation
     abstracts; arXiv's API doesn't expose citation counts, so this
     project calibrates on real arXiv abstracts from the indexed corpus
     as a proxy — documented here rather than silently substituted.
  2. survival_score(texts) embeds new text, computes its distance to that
     same centroid, and evaluates the fitted Gamma's survival function
     (SF = 1 - CDF) at that distance. A distance typical of the reference
     population's own distances scores a high SF (looks like real
     academic text); a distance far outside what the reference population
     produces scores a low SF (atypical -> flagged).
  3. Two call sites, two different uses of that score:
       filter_chunks(chunks, sf_threshold) — single-cutoff keep/reject
         gate over retrieved evidence (Section 4.2, ArXiv/Web sub-agents).
         This is where the per-job sf_threshold slider (Section 3.1)
         actually changes behavior.
       score_and_route(draft, sf_threshold) — three-way cascade
         (reject/approve/escalate) over Writer's draft sentences
         (Section 4.7's Critic). The cascade boundaries are the fixed
         CERTAIN_WRONG/CERTAIN_RIGHT constants below, exactly as the
         Section 4.7 code lists them; sf_threshold is accepted here for
         signature parity with filter_chunks (both are snapshotted
         together in Section 3.1) but does not move these boundaries —
         matching the literal Section 4.7 code, which never references
         its own sf_threshold parameter in the cascade body.

Measured latency (this M5 Pro, calibrated on 438 real arXiv abstracts):
Section 4.7 claims "<2ms p99 CPU inference." The Gamma survival-function
math itself measures ~0.006ms/sentence given an already-computed
embedding — comfortably under that claim. Full survival_score() latency
(embed + score) measures ~2.7ms/sentence, dominated by the BGE-small
forward pass. Both numbers are reported (see this module's __main__
block) rather than only citing whichever one clears the bar.
"""
import re

import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer

# Must match rag/indexer.py and backend/nodes/context_eval.py — every
# cosine-similarity comparison in this project shares one embedding space.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class GammaGuardrail:
    CERTAIN_WRONG = 0.05
    CERTAIN_RIGHT = 0.25

    def __init__(self, encoder: SentenceTransformer | None = None):
        self.encoder = encoder or SentenceTransformer(EMBEDDING_MODEL)
        self._centroid: np.ndarray | None = None
        self._gamma_params: tuple[float, float, float] | None = None  # (shape, loc, scale)

    @property
    def is_calibrated(self) -> bool:
        return self._gamma_params is not None

    def calibrate(self, reference_texts: list[str]) -> dict:
        """Fit the Gamma survival model on a reference corpus of
        trustworthy academic text. Returns calibration stats for logging.
        """
        if len(reference_texts) < 2:
            raise ValueError("Need at least 2 reference texts to calibrate a distribution.")

        embeddings = self.encoder.encode(reference_texts, normalize_embeddings=True)
        centroid = embeddings.mean(axis=0)
        self._centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

        distances = self._distance_to_centroid(embeddings)
        # floc=0: cosine distance is non-negative by construction, so we
        # don't let the fit shift the distribution's support left of zero.
        shape, loc, scale = stats.gamma.fit(distances, floc=0)
        self._gamma_params = (shape, loc, scale)

        return {
            "n_reference": len(reference_texts),
            "gamma_shape": float(shape),
            "gamma_scale": float(scale),
            "distance_mean": float(distances.mean()),
            "distance_std": float(distances.std()),
        }

    def _distance_to_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        cosine_sim = normed @ self._centroid
        return 1.0 - cosine_sim

    def survival_score(self, texts: list[str]) -> np.ndarray:
        if not self.is_calibrated:
            raise RuntimeError("GammaGuardrail.calibrate() must run before scoring.")
        embeddings = self.encoder.encode(texts, normalize_embeddings=True)
        distances = self._distance_to_centroid(embeddings)
        shape, loc, scale = self._gamma_params
        return stats.gamma.sf(distances, shape, loc=loc, scale=scale)

    def filter_chunks(self, chunks: list[dict], sf_threshold: float) -> tuple[list[dict], list[float]]:
        """Section 4.2: keep only chunks whose SF clears sf_threshold."""
        if not chunks:
            return [], []
        scores = self.survival_score([c["content"] for c in chunks])
        verified = [c for c, s in zip(chunks, scores) if s >= sf_threshold]
        verified_scores = [float(s) for s in scores if s >= sf_threshold]
        return verified, verified_scores

    def score_and_route(self, draft: str, sf_threshold: float) -> tuple[list[str], float]:
        """Section 4.7: three-way cascade over Writer's draft sentences.

        Returns (per-sentence routing decisions, mean_sf_score).
        routing: "reject" | "approve" | "escalate"
        """
        sentences = self._split_sentences(draft)
        if not sentences:
            return [], 0.0
        sf_scores = self.survival_score(sentences)

        decisions = []
        for sf in sf_scores:
            if sf < self.CERTAIN_WRONG:
                decisions.append("reject")
            elif sf > self.CERTAIN_RIGHT:
                decisions.append("approve")
            else:
                decisions.append("escalate")

        return decisions, float(sf_scores.mean())

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]


def build_default_guardrail() -> GammaGuardrail:
    """Calibrate a GammaGuardrail on the real indexed arXiv corpus
    (rag/indexer.py's abstracts) rather than a synthetic reference set."""
    from rag.indexer import load_index

    _, metadata = load_index()
    reference_texts = [p["summary"] for p in metadata["paper_by_id"].values()]

    guardrail = GammaGuardrail()
    stats_ = guardrail.calibrate(reference_texts)
    return guardrail, stats_


if __name__ == "__main__":
    import time

    from scipy import stats as scipy_stats

    guardrail, calib_stats = build_default_guardrail()
    print(f"Calibrated: {calib_stats}")

    test_sentences = [
        "Retrieval-augmented generation combines a retriever with a language model to ground outputs in external documents.",
        "The banana wore a small hat and danced across the moon while singing opera.",
    ]

    guardrail.survival_score(test_sentences)  # warm up
    t0 = time.time()
    for _ in range(50):
        scores = guardrail.survival_score(test_sentences)
    full_ms = (time.time() - t0) / 50 * 1000 / len(test_sentences)
    print(f"scores: {scores}  (on-topic vs. nonsense — separation is the point)")

    # Section 4.7 claims "<2ms p99 CPU inference" — measured full latency
    # (embed + score) is a few ms, dominated by the embedding forward pass;
    # the Gamma math itself, given an already-computed embedding, is the
    # part that's actually sub-2ms. Report both rather than assert one.
    embeddings = guardrail.encoder.encode(test_sentences, normalize_embeddings=True)
    distances = guardrail._distance_to_centroid(embeddings)
    shape, loc, scale = guardrail._gamma_params
    scipy_stats.gamma.sf(distances, shape, loc=loc, scale=scale)  # warm up
    t0 = time.time()
    for _ in range(200):
        scipy_stats.gamma.sf(distances, shape, loc=loc, scale=scale)
    gamma_only_ms = (time.time() - t0) / 200 * 1000 / len(test_sentences)

    print(f"mean per-sentence latency, embed+score (cold path): {full_ms:.2f}ms")
    print(f"mean per-sentence latency, Gamma math only (embedding already computed): {gamma_only_ms:.4f}ms")
