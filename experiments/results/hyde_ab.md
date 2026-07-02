## HyDE A/B — Section 5.2

| # | Query | Rollbacks (off) | Rollbacks (on) | SF (off) | SF (on) | Rerank (off) | Rerank (on) |
|---|---|---|---|---|---|---|---|
| 1 | How does HyDE improve retrieval-augmented generation for ... | 1 | 1 | 0.643 | 0.589 | 2.556 | 2.556 |
| 2 | What is the attention mechanism in transformer models? | 3 | 1 | 0.585 | 0.571 | -2.938 | -2.938 |
| 3 | How does chain of thought reasoning improve language mode... | 1 | 1 | 0.814 | 0.751 | 5.850 | 5.411 |
| 4 | What methods are used to detect hallucinations in languag... | 3 | 3 | 0.684 | 0.787 | 3.754 | 3.754 |
| 5 | How does dense passage retrieval compare to BM25 for open... | 1 | 1 | 0.642 | 0.663 | 1.460 | 1.460 |
| 6 | What is retrieval-augmented generation and how does it work? | 1 | 1 | 0.620 | 0.658 | 3.319 | 3.319 |
| 7 | How does reranking improve retrieval quality in RAG systems? | 1 | 1 | 0.777 | 0.717 | 5.151 | 5.151 |
| 8 | What role does query expansion play in information retrie... | 1 | 1 | 0.531 | 0.549 | 1.906 | 1.906 |
| 9 | How do embedding models represent semantic similarity bet... | 1 | 1 | 0.375 | 0.413 | 2.736 | 3.654 |
| 10 | What is self-consistency decoding for language models? | 1 | 1 | 0.759 | 0.731 | -0.665 | -0.665 |
**Aggregate (n=10):**

|                     | HyDE off | HyDE on |
|---------------------|----------|---------|
| mean rollbacks       | 1.40 | 1.20 |
| mean chunk SF        | 0.643 | 0.643 |
| mean rerank score    | 2.313 | 2.361 |
| approved / n         | 9/10 | 9/10 |
