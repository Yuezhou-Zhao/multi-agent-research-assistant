"""rag/indexer.py — arXiv paper download + FAISS index build.

Builds the local corpus the ArXiv sub-agent retrieves from
(faiss_retriever_tool). Indexes title+abstract metadata, not full PDF
text: full text is fetched on demand via pdf_parser_tool when an agent
actually needs it, which keeps index construction fast and network-light.

Embedding model: BAAI/bge-small-en-v1.5 — the same encoder reused
wherever else this project computes cosine similarity (HyDE quality
gate, context_eval coverage scoring), so all of those comparisons live
in the same embedding space.
"""
import asyncio
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.chunker import ParentChildChunker
from rag.tools import arxiv_search_tool

INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Spans the CS sub-areas the demo queries touch: RAG, agents,
# hallucination detection. Broad enough to hit ~500 unique papers.
DEFAULT_QUERIES = [
    "retrieval augmented generation",
    "large language model agents",
    "hallucination detection language models",
    "chain of thought reasoning",
    "vector database embedding retrieval",
]


async def fetch_corpus(queries: list[str] = DEFAULT_QUERIES, per_query: int = 100) -> list[dict]:
    """Download paper metadata across several queries, deduped by arXiv id."""
    seen: dict[str, dict] = {}
    for query in queries:
        papers = await arxiv_search_tool(query, max_results=per_query)
        for paper in papers:
            seen[paper["id"]] = paper
        await asyncio.sleep(1)  # be polite to the arXiv API between queries
    return list(seen.values())


def build_index(papers: list[dict], index_dir: Path = INDEX_DIR) -> dict:
    """Chunk, embed, and index a paper corpus. Returns build stats."""
    chunker = ParentChildChunker(parent_tokens=512, child_tokens=128, overlap=20)
    child_records = []  # (child_id, child_text, paper_id)
    for paper in papers:
        doc_text = f"{paper['title']}\n\n{paper['summary']}"
        for cid, text in chunker.chunk_document(paper["id"], doc_text):
            child_records.append((cid, text, paper["id"]))

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [text for _, text, _ in child_records]
    embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # normalized embeddings -> inner product == cosine
    index.add(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))

    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "child_order": [cid for cid, _, _ in child_records],
        # child_texts is aligned with child_order — needed by the BM25 leg
        # of TwoStageRetriever (rag/retriever.py), which must run over the
        # exact same chunks the FAISS leg was built from.
        "child_texts": texts,
        "parent_store": chunker.parent_store,
        "child_to_parent": chunker.child_to_parent,
        "paper_by_id": {p["id"]: p for p in papers},
        "child_to_paper": {cid: pid for cid, _, pid in child_records},
    }
    with open(index_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return {
        "num_papers": len(papers),
        "num_parents": len(chunker.parent_store),
        "num_children": len(child_records),
        "embedding_dim": dim,
    }


def load_index(index_dir: Path = INDEX_DIR):
    index = faiss.read_index(str(index_dir / "faiss.index"))
    with open(index_dir / "metadata.json") as f:
        metadata = json.load(f)
    return index, metadata


def search(query: str, k: int = 5, index_dir: Path = INDEX_DIR) -> list[dict]:
    """Retrieve top-k child chunks for `query`, resolved to parent text.

    This is the retrieval half of the Parent-Child design:
    match on the precise child embedding, return the full-context parent.
    """
    index, metadata = load_index(index_dir)
    encoder = SentenceTransformer(metadata["embedding_model"])
    q_emb = encoder.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        child_id = metadata["child_order"][idx]
        parent_id = metadata["child_to_parent"][child_id]
        paper_id = metadata["child_to_paper"][child_id]
        results.append(
            {
                "child_id": child_id,
                "score": float(score),
                "parent_text": metadata["parent_store"][parent_id],
                "paper": metadata["paper_by_id"][paper_id],
            }
        )
    return results


if __name__ == "__main__":
    start = time.time()
    corpus = asyncio.run(fetch_corpus())
    print(f"Fetched {len(corpus)} unique papers in {time.time() - start:.1f}s")

    build_start = time.time()
    stats = build_index(corpus)
    print(f"Built index in {time.time() - build_start:.1f}s: {stats}")
