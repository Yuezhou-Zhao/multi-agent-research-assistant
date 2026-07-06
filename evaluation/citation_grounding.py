"""Per-sentence citation grounding via embedding similarity (Section 4.9).

Zero-LLM check that catches *citation misattribution*: sentences whose
`[N]` citation is structurally valid (in-range, resolvable to a real
paper) but semantically wrong (the cited chunk's actual text doesn't
support the sentence's specific claim).

Motivation. Section 4.8's index-based citation fix eliminated fabricated
ARXIV IDs — no more `[2305.17306v1]` that doesn't exist. But it did NOT
catch the case where the LLM describes a well-known method from
pretrained knowledge and then attaches whichever nearby chunk index has
a plausible-looking title. Empirically (n=10 labeling exercise): 36/65
sentences carried a valid-range citation pointing at the wrong paper
(e.g. FG-PRM described from memory, cited to a REFIND chunk).

Signal. Same idea as backend/nodes/context_eval.py's coverage check
(Section 4.5), just applied at the sentence-citation pair rather than
query-subquestion pair:
  sim(sentence_embedding, mean(cited_chunks_embeddings))
Sentences whose similarity falls below GROUNDING_THRESHOLD are
downgraded in the L1 cascade decisions (approve -> escalate, escalate
-> reject, reject stays reject) — a sentence Gamma was ready to approve
gets one more chance to be rescued at L3; a sentence Gamma already
escalated gets rejected outright.

Threshold. Set to 0.82 by sweeping F1 against the human-reviewed
cascade_labels.csv (see GROUNDING_THRESHOLD below for the measured
precision/recall table and the recall-over-precision rationale). Earlier
hand-picked values (0.5, then 0.65) fired too rarely — 0.65 caught only
12 of 37 real misattributions.
"""
import re
from dataclasses import dataclass, field

import numpy as np

from evaluation.gamma_guardrail import GammaGuardrail

_CITATION_RE = re.compile(r"\[(\d+)\]")

# Empirical measurements against BAAI/bge-small-en-v1.5 on the actual
# failure mode from the n=10 labeling exercise:
#
#   correct grounded (RAG sentence → RAG chunk):          sim = 0.858
#   correctly matched (FG-PRM sentence → FG-PRM chunk):   sim = 0.968
#   misattribution (FG-PRM sentence → REFIND chunk):      sim = 0.566
#   same-subfield adjacent (CoNLI sentence → Semantic
#   Illusion chunk):                                       sim = 0.837
#   wildly unrelated (HyDE sentence → recipe chunk):      sim = 0.442
#
# Final threshold set by sweeping against the human-reviewed labels in
# experiments/results/cascade_labels.csv (58 cited sentences: 37
# hallucinated / 13 correct / 8 uncertain). Positive class = hallucinated;
# L2b downgrades a sentence whose sim < threshold. Measured precision/recall
# on those labels (experiments/threshold_validation.py --source final):
#
#   threshold 0.65:  precision 1.000  recall 0.324  F1 0.490  (catches 12/37)
#   threshold 0.82:  precision 0.818  recall 0.973  F1 0.889  (catches 36/37)
#
# 0.65 (the earlier hand-picked value) barely fires — it misses 25 of 37
# real misattributions, defeating the purpose of the check. 0.82 is the
# F1-optimal on the labeled set and is robustly confirmed by the AI-drafted
# labels too (same 0.82 optimum, see threshold_validation_aidraft.md).
#
# The cost of 0.82 is precision: ~8 of 13 correct same-subfield citations
# fall below it and get downgraded. But L2b downgrades approve -> escalate
# (i.e. hands the sentence to the L3 LLM judge, which re-checks it) rather
# than to a final verdict, so a false downgrade costs an extra L3 call, not
# a wrong answer. Given that the misattribution failure mode is exactly what
# L2b exists to catch, we prioritize recall here and let L3 clean up the
# false positives. The residual hallucinated/correct sim overlap in
# [0.65, 0.85] is the documented Semantic-Illusion limitation (Section 8) —
# an embedding-only signal can't separate them cleanly; a stronger signal
# (NLI / the L3 judge) is what closes it.
GROUNDING_THRESHOLD = 0.82


@dataclass
class GroundingReport:
    updated_cascade: list[str] = field(default_factory=list)
    # Per-sentence detail so the critic_feedback message and any debug
    # UI can point at which specific sentences got downgraded and why.
    per_sentence: list[dict] = field(default_factory=list)
    n_ungrounded: int = 0

    def summary(self) -> str:
        if self.n_ungrounded == 0:
            return "grounding OK"
        by_source = [
            f"sent#{i+1} → cited{s['cited']} sim={s['sim']:.2f}"
            for i, s in enumerate(self.per_sentence)
            if not s["grounded"] and s["cited"]
        ]
        return (
            f"{self.n_ungrounded} sentence(s) below grounding threshold "
            f"{GROUNDING_THRESHOLD}: {'; '.join(by_source)}"
        )


def _extract_cited_indices(sentence: str, n_chunks: int) -> list[int]:
    """1-based indices this sentence carries as citations, filtered to
    those that are in-range against merged_chunks."""
    out = []
    seen = set()
    for match in _CITATION_RE.findall(sentence):
        try:
            idx = int(match)
        except ValueError:
            continue
        if 1 <= idx <= n_chunks and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _downgrade(decision: str) -> str:
    """Downgrade one step in the cascade severity: approve -> escalate
    -> reject; already-reject stays reject."""
    return {"approve": "escalate", "escalate": "reject"}.get(decision, decision)


def check_citation_grounding(
    draft: str,
    cascade_decisions: list[str],
    merged_chunks: list[dict],
    encoder,
    threshold: float = GROUNDING_THRESHOLD,
) -> GroundingReport:
    """Score each sentence against the centroid of its cited chunks.
    Sentences with no citations are passed through (uncited handling is
    L2a's job, not this check's). Returns an updated cascade whose
    approve/escalate slots are downgraded for ungrounded sentences.
    """
    sentences = GammaGuardrail._split_sentences(draft)
    if not sentences or len(cascade_decisions) != len(sentences):
        # Sentence-cascade misalignment — safest is to pass through
        # rather than mis-downgrade decisions off-by-one.
        return GroundingReport(updated_cascade=list(cascade_decisions))

    n_chunks = len(merged_chunks)
    updated = list(cascade_decisions)
    per_sentence: list[dict] = []
    n_ungrounded = 0

    # Batch-embed sentences and the (unique) chunks they cite for speed.
    sent_embeddings = encoder.encode(sentences, normalize_embeddings=True)
    chunk_embedding_cache: dict[int, np.ndarray] = {}

    def chunk_emb(idx: int) -> np.ndarray:
        if idx not in chunk_embedding_cache:
            text = merged_chunks[idx - 1]["content"]
            chunk_embedding_cache[idx] = encoder.encode(
                [text], normalize_embeddings=True
            )[0]
        return chunk_embedding_cache[idx]

    for i, (sent, decision) in enumerate(zip(sentences, updated)):
        cited = _extract_cited_indices(sent, n_chunks)
        if not cited:
            per_sentence.append(
                {"sentence": sent, "cited": [], "sim": None, "grounded": None,
                 "original": decision, "updated": decision}
            )
            continue

        # Cited-chunk centroid, then cosine sim (embeddings normalized -> dot product).
        chunk_centroid = np.mean([chunk_emb(c) for c in cited], axis=0)
        chunk_centroid = chunk_centroid / (np.linalg.norm(chunk_centroid) + 1e-10)
        sim = float(sent_embeddings[i] @ chunk_centroid)

        grounded = sim >= threshold
        new_decision = decision if grounded else _downgrade(decision)
        if not grounded and new_decision != decision:
            n_ungrounded += 1
        updated[i] = new_decision

        per_sentence.append(
            {
                "sentence": sent,
                "cited": cited,
                "sim": sim,
                "grounded": grounded,
                "original": decision,
                "updated": new_decision,
            }
        )

    return GroundingReport(
        updated_cascade=updated,
        per_sentence=per_sentence,
        n_ungrounded=n_ungrounded,
    )
