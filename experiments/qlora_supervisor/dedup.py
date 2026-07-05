"""Train/eval disjointness enforcement (Section: independence).

Two tiers:
  1. Exact: normalized-string match (lowercase, collapsed whitespace,
     stripped trailing punctuation).
  2. Near-duplicate: cosine similarity above DEDUP_COSINE_THRESHOLD under
     bge-small-en-v1.5 (the same encoder the rest of the project uses).
     Catches paraphrases the exact match misses.

Used to (a) dedup within a freshly generated pool, and (b) remove any eval
query that collides with the training pool, so the held-out eval set is
genuinely disjoint.
"""
import numpy as np

from . import config
from .common import normalize_query

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _encoder


def dedup_within(rows: list[dict]) -> tuple[list[dict], int]:
    """Drop exact + near-duplicate queries within one pool. Returns
    (kept_rows, n_dropped). Order-stable: keeps the first occurrence."""
    kept: list[dict] = []
    seen_norm: set[str] = set()
    for r in rows:
        norm = normalize_query(r["query"])
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        kept.append(r)

    if len(kept) < 2:
        return kept, len(rows) - len(kept)

    # Near-dup pass via embeddings.
    enc = _get_encoder()
    embs = enc.encode([r["query"] for r in kept], normalize_embeddings=True)
    keep_mask = [True] * len(kept)
    for i in range(len(kept)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(kept)):
            if keep_mask[j] and float(embs[i] @ embs[j]) >= config.DEDUP_COSINE_THRESHOLD:
                keep_mask[j] = False
    deduped = [r for r, keep in zip(kept, keep_mask) if keep]
    return deduped, len(rows) - len(deduped)


def remove_overlap(eval_rows: list[dict], train_rows: list[dict]) -> tuple[list[dict], int]:
    """Drop any eval query that exactly- or near-matches a training query.
    Returns (disjoint_eval_rows, n_removed)."""
    train_norm = {normalize_query(r["query"]) for r in train_rows}
    stage1 = [r for r in eval_rows if normalize_query(r["query"]) not in train_norm]

    if not stage1 or not train_rows:
        return stage1, len(eval_rows) - len(stage1)

    enc = _get_encoder()
    train_embs = enc.encode([r["query"] for r in train_rows], normalize_embeddings=True)
    eval_embs = enc.encode([r["query"] for r in stage1], normalize_embeddings=True)
    # For each eval query, max cosine against any training query.
    sims = eval_embs @ train_embs.T  # (n_eval, n_train)
    max_sim = sims.max(axis=1)
    disjoint = [r for r, s in zip(stage1, max_sim) if s < config.DEDUP_COSINE_THRESHOLD]
    return disjoint, len(eval_rows) - len(disjoint)
