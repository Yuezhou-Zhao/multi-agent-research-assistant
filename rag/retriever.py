"""rag/retriever.py — TwoStageRetriever: BM25 + FAISS + RRF fusion + BGE rerank.

Section 6 names this architecture ("RAG vs basic vector search": two-stage,
BM25+FAISS+RRF fusion -> BGE reranker) but the design doc gives no code
listing for it (unlike chunker/HyDE/gamma, which have full snippets in
Section 4) — this module is my own implementation of that described
pipeline, built on top of the Parent-Child index from Week 2
(rag/indexer.py) and reusing its embedding model for the dense leg so both
legs score the exact same child chunks.

Stage 1 (recall):   BM25 (sparse, exact term match) and FAISS (dense,
                     semantic) rank the full child-chunk corpus
                     independently, then get fused via Reciprocal Rank
                     Fusion (RRF) — this avoids picking a weighting between
                     BM25 and cosine scores, which live on incomparable
                     scales.
Stage 2 (precision): a cross-encoder re-scores (query, child_text) pairs
                     from the RRF-fused candidate pool only, and re-sorts
                     by that score. Cross-encoders are too slow to run
                     over the whole corpus — stage 1 exists to narrow the
                     field first.

Reranker model: Section 6 specifies BGE-reranker-v2-m3 by default. Measured
on this M5 Pro against this corpus's real ~128-token child chunks (not
short synthetic strings), reranking 20 candidates took ~500-640ms/query
end-to-end — at/over Section 8's stopping rule L1 (>500ms/query on M5 Pro
-> switch to cross-encoder/ms-marco-MiniLM-L-6-v2). MPS (Apple Silicon GPU)
helped (a 20-pair batch alone dropped from ~1.1s to ~77ms) but didn't close
the gap once real passage lengths and the surrounding retrieve() overhead
were included, so L1 is genuinely triggered here, not worked around.
Default is therefore the L1 fallback model; BGE-reranker-v2-m3 remains
available via the `reranker_model` / `reranker_backend` constructor args
for anyone running this on hardware where it clears the budget.
"""
from pathlib import Path

import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.indexer import INDEX_DIR, load_index

BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # Section 6 default
FALLBACK_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Section 8 L1
RERANKER_MODEL = FALLBACK_RERANKER_MODEL
RRF_K = 60  # standard RRF constant (Cormack et al., 2009)

_RERANKER_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Fuse multiple ranked ID lists into one score per ID.

    score(id) = sum over rankings containing id of 1 / (k + rank + 1)
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


class TwoStageRetriever:
    def __init__(
        self,
        index_dir: Path = INDEX_DIR,
        reranker_model: str = RERANKER_MODEL,
        reranker_backend: str = "cross_encoder",  # or "flagembedding" for BGE
        bm25_top_k: int = 30,
        faiss_top_k: int = 30,
        fusion_top_k: int = 20,
    ):
        self.index_dir = index_dir
        self.bm25_top_k = bm25_top_k
        self.faiss_top_k = faiss_top_k
        self.fusion_top_k = fusion_top_k
        self.reranker_backend = reranker_backend

        self.faiss_index, self.metadata = load_index(index_dir)
        self.encoder = SentenceTransformer(self.metadata["embedding_model"])

        if reranker_backend == "flagembedding":
            from FlagEmbedding import FlagReranker

            self.reranker = FlagReranker(reranker_model, use_fp16=True, devices=[_RERANKER_DEVICE])
        elif reranker_backend == "cross_encoder":
            self.reranker = CrossEncoder(reranker_model, device=_RERANKER_DEVICE)
        else:
            raise ValueError(f"Unknown reranker_backend: {reranker_backend!r}")

        self.child_ids: list[str] = self.metadata["child_order"]
        self.child_texts: list[str] = self.metadata["child_texts"]
        self._bm25 = BM25Okapi([text.lower().split() for text in self.child_texts])

    def _rerank_scores(self, pairs: list[list[str]]) -> list[float]:
        if self.reranker_backend == "flagembedding":
            scores = self.reranker.compute_score(pairs, normalize=True)
            return [scores] if isinstance(scores, float) else list(scores)
        return [float(s) for s in self.reranker.predict(pairs)]

    def _bm25_rank(self, query: str, top_k: int) -> list[str]:
        scores = self._bm25.get_scores(query.lower().split())
        top_idx = scores.argsort()[::-1][:top_k]
        return [self.child_ids[i] for i in top_idx]

    def _faiss_rank(self, query: str, top_k: int) -> list[str]:
        q_emb = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        _, indices = self.faiss_index.search(q_emb, top_k)
        return [self.child_ids[i] for i in indices[0] if i != -1]

    def _resolve(self, child_id: str) -> dict:
        parent_id = self.metadata["child_to_parent"][child_id]
        paper_id = self.metadata["child_to_paper"][child_id]
        return {
            "child_id": child_id,
            "parent_text": self.metadata["parent_store"][parent_id],
            "paper": self.metadata["paper_by_id"][paper_id],
        }

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Two-stage retrieve: BM25 + FAISS -> RRF fuse -> rerank -> top k."""
        bm25_ranking = self._bm25_rank(query, self.bm25_top_k)
        faiss_ranking = self._faiss_rank(query, self.faiss_top_k)
        fused = reciprocal_rank_fusion([bm25_ranking, faiss_ranking])

        candidates = sorted(fused, key=fused.get, reverse=True)[: self.fusion_top_k]
        child_text_by_id = dict(zip(self.child_ids, self.child_texts))
        pairs = [[query, child_text_by_id[cid]] for cid in candidates]
        rerank_scores = self._rerank_scores(pairs)

        ranked = sorted(zip(candidates, rerank_scores), key=lambda cs: cs[1], reverse=True)
        results = []
        for child_id, score in ranked[:k]:
            result = self._resolve(child_id)
            result["score"] = float(score)
            results.append(result)
        return results

    def retrieve_dense_only(self, query: str, k: int = 5) -> list[dict]:
        """Single-stage baseline (FAISS only, no BM25/RRF/rerank) — used by
        the recall comparison in this module's __main__ block."""
        child_ids = self._faiss_rank(query, k)
        return [self._resolve(cid) for cid in child_ids]


if __name__ == "__main__":
    import random
    import time

    retriever = TwoStageRetriever()
    random.seed(0)
    sample_papers = random.sample(list(retriever.metadata["paper_by_id"].values()), 100)

    def recall_at_5(retrieve_fn) -> float:
        hits = 0
        for paper in sample_papers:
            results = retrieve_fn(paper["title"], k=5)
            if any(r["paper"]["id"] == paper["id"] for r in results):
                hits += 1
        return hits / len(sample_papers)

    single_stage_recall = recall_at_5(retriever.retrieve_dense_only)

    t0 = time.time()
    two_stage_recall = recall_at_5(retriever.retrieve)
    per_query_ms = (time.time() - t0) / len(sample_papers) * 1000

    print(f"Single-stage (FAISS only)                         Recall@5: {single_stage_recall:.3f}")
    print(f"Two-stage (BM25+FAISS+RRF+{retriever.reranker_backend} rerank) Recall@5: {two_stage_recall:.3f}")
    print(f"Two-stage mean latency/query: {per_query_ms:.1f}ms")
