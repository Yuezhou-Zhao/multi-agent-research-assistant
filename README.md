# Multi-Agent Research Assistant

![tests](https://github.com/Yuezhou-Zhao/multi-agent-research-assistant/actions/workflows/ci.yml/badge.svg)

A production-minded research assistant that answers technical questions by
retrieving across arXiv and the web, drafting a cited synthesis, and
**self-correcting through a three-layer verification cascade** that keeps
LLM cost bounded by construction.

Built with `gpt-4o-mini` as the only API-side language model — no managed
vector database, no opaque orchestration layer.

> **Author:** Yuezhou Zhao

## Why this project

Most RAG systems either invoke an LLM to verify every generated claim, at
significant cost, or perform no verification at all. This project
investigates whether **a cheap-first verification cascade can close that
gap** — resolving most sentence-level verdicts with zero-LLM checks and
escalating only genuinely uncertain cases to an LLM judge — **under a hard
upper bound on cost** (≤ 15 calls per query, ~$0.016 worst case).

## What happens to one query

```
"How does chain-of-thought prompting work?"
       │
       ▼   Planner    — split into focused sub-questions
       ▼   Retrieval  — arXiv + Web searched in parallel
       ▼   Writer     — draft an answer with [1][2] citations
       ▼   Critic     — verify each sentence; may reject and retry
       ▼
  Cited answer  (low-confidence sentences flagged 🚩)
```

## Demo

![Approved answer: Gamma red-flags on suspect sentences, real resolved citations](docs/img/demo_answer.png)

*A real query end-to-end: the Critic approves after one visible rollback; 🚩
marks the sentences Gamma's L1 flagged as likely-hallucinated (2 of 6 in this
run), and the Writer's `[N]` index citations arrive resolved to real arXiv
IDs by the Finalizer.*

The Chainlit UI visualizes the run as it happens:

- a **nested execution trace** — every node expandable, including each
  rollback iteration when the Critic sends the draft back
- a **live metrics panel** — budget used vs. cap, LLM calls saved by the
  cascade, coverage score, and the per-band cascade counts
- **🚩 inline highlighting** of the sentences the guardrail flagged as
  low-confidence
- resolved **`[N]` → arXiv ID** citations in the final answer

<details>
<summary>Live metrics panel + execution trace, and the whole run animated</summary>

![Live metrics: LLM budget 5/15, coverage 0.854, cascade bands, approved status](docs/img/demo_metrics.png)

![The run animated: status planning → researching → writing → reviewing → approved](docs/img/demo_run.gif)

</details>

## Headline results

| | |
|---|---|
| **LLM budget** | ≤ 15 calls/query, enforced — worst-case ~$0.016/query; degraded answers are flagged, never silently truncated |
| **Zero-LLM resolution** | 58.3% of sentences verified with zero LLM calls — the cheap-first cascade settles most verdicts before the judge is ever invoked |
| **Hallucination detection** | F1 0.50 → 0.562 vs. the guardrail alone (oracle-judge ceiling: 0.600) |
| **Router distillation** | 0.446 → 0.838 route accuracy — distilled from gpt-4o-mini into a 1.5B on-device model, within 1.6 pt of the teacher at ~5× lower p95 latency |

Full tables, methodology, and the negative results: **[docs/results.md](docs/results.md)**.

## Architecture

![System overview — pipeline, three-layer cascade, and measured results](docs/system_overview.svg)

- **Parallel retrieval (multi-agent).** The arXiv and web sub-agents own
  disjoint tool sets, run concurrently via LangGraph's `Send` API, and write
  to non-overlapping state slices; results merge only at a fan-in node.
  (Concurrency is async within one process — the independence is about
  tools and state, not distributed infrastructure.)
- **Shared-state synthesis.** Planner, Writer, and Critic share one state
  on purpose: their data dependencies are tight, and separate agents would
  add serialization overhead for no benefit.
- **Hybrid retrieval pipeline.** The retriever combines lexical (BM25) and
  semantic (FAISS) search, fuses the two rankings with reciprocal-rank
  fusion, then reranks with a cross-encoder over parent-child chunks
  (128-token children for recall, 512-token parents for context —
  hand-written, ~50 lines). BGE-reranker-v2-m3 was the original reranker;
  it measured over the 500 ms/query latency budget on real chunks, so the
  shipped default is `ms-marco-MiniLM-L-6-v2` (BGE stays available via a
  constructor arg).

## The verification cascade

Every draft sentence is scored cheapest-first. Only genuinely uncertain
sentences ever reach an LLM:

```
        each draft sentence
                │
                ▼
   L1  embedding confidence      ~2 ms, 0 LLM   → confident approve / reject
                │  (uncertain)
                ▼
   L2a citation format check                    → reject if [N] out of range
                │
                ▼
   L2b citation grounding        0 LLM           → reject if sentence not
                │                                   supported by cited chunk
                ▼  (only the still-uncertain band)
   L3  LLM judge                 1 API call       → final verdict
```

1. **L1 — Gamma guardrail** ([`evaluation/gamma_guardrail.py`](evaluation/gamma_guardrail.py)):
   a calibrated embedding-distance score, ~2 ms, zero LLM. Resolves the
   confident approve/reject bands outright.
2. **L2a — citation structural check** ([`evaluation/citation_check.py`](evaluation/citation_check.py)):
   the Writer cites by *index* `[1..N]` and the Finalizer resolves indices
   to real arXiv IDs afterward — a fabricated identifier is impossible by
   construction.
3. **L2b — citation grounding** ([`evaluation/citation_grounding.py`](evaluation/citation_grounding.py)):
   `cosine(sentence, cited chunks)` catches citations that point at a real
   but *wrong* paper — a failure mode found by hand-labeling 65 sentences
   (36 were misattributed). Precision 0.89 / recall 0.62 against those
   labels, zero LLM.
4. **L3 — LLM judge**: runs only on the unresolved band, only while the
   budget allows.

Loop control is bounded on three axes: outer circuit breaker (max 3
rollbacks) + inner refinement (max 1) + global LLM budget (max 15 calls).
One design decision worth reading: the grounding threshold that maximized
F1 (0.82) made the correction loop non-convergent in live runs, so the
shipped value is 0.70 — the sweep and the reasoning are in
[docs/results.md](docs/results.md#choosing-the-grounding-threshold).

## Local-inference extension: distilling the router

To eliminate one API call on every query, the routing decision
(arXiv / web / both) was distilled from gpt-4o-mini into a local
**Qwen2.5-1.5B + LoRA**, evaluated on a 130-row human-corrected held-out
set:

| Metric | gpt-4o-mini (API) | Qwen2.5-1.5B zero-shot | + LoRA (local) |
|---|---:|---:|---:|
| route accuracy | 0.854 | 0.446 | **0.838** |
| latency p95 | 5.90 s | 0.58 s | **1.15 s** |

The zero-shot control shows the fine-tuning did the work (+39.2 points).
The wins are latency and on-device independence — the dollar saving at
this call volume is negligible, and the student inherits the teacher's
weakness on the `both` class. Full write-up:
[`experiments/lora_supervisor/RESULTS.md`](experiments/lora_supervisor/RESULTS.md).

**Note on the serving path:** the shipped app keeps the gpt-4o-mini
routing call. The student lives in [`experiments/lora_supervisor/`](experiments/lora_supervisor/)
and was measured on Apple-Silicon MPS (fp16 via `transformers` + PEFT);
it is deliberately not wired into the Docker serving path, where a 1.5B
model on shared CPU cores would be slower than the API it replaces. The
distillation targets on-device deployment, not the cloud demo.

## Setup & run

Prerequisites: Python 3.11, API keys in `.env` (copy
[`.env.example`](.env.example)): `OPENAI_API_KEY`, `TAVILY_API_KEY`.

### Local

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -c requirements.lock

python -m rag.indexer            # build the FAISS index (one-time, ~90s)
python -m scripts.smoke_test     # end-to-end sanity check → [smoke] OK

chainlit run frontend/app.py     # UI on :8000
uvicorn backend.main:app --port 8001   # optional: headless JSON API
```

The UI streams the live execution trace (including rollback iterations),
sidebar toggles for HyDE / `sf_threshold` (snapshotted per job), red-flag
highlighting of guardrail-rejected sentences, and a live metrics panel.

The FastAPI service exposes the same pipeline headlessly:

```bash
curl -X POST localhost:8001/research -H 'content-type: application/json' \
     -d '{"query": "How does chain-of-thought prompting work?"}'
# → {"job_id": "..."}   then poll /status/{id} and /result/{id}
```

### Docker

```bash
docker compose build
docker compose up        # UI on :8000, API on :8001; index auto-builds on first run

# Verify the container end-to-end (not just that it built):
docker compose run --rm --entrypoint python research-agent -m scripts.smoke_test
```

The image builds `linux/arm64` natively. For slow networks there are
BuildKit cache mounts and a PyPI mirror override
(`--build-arg PIP_INDEX_URL=...`).

For a public deployment, [`docker/docker-compose.prod.yml`](docker/docker-compose.prod.yml)
adds a Caddy reverse proxy with automatic TLS and basic-auth in front of
both ports.

## Testing

**104 automated tests** — 99 run without any API key; 5 live-LLM
integration tests auto-skip when `OPENAI_API_KEY` is absent, so CI stays
green.

```bash
pytest -q
```

CI runs the keyless suite on every push
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Coverage focuses
on what's easy to get subtly wrong: circuit-breaker/budget routing,
parent-child chunk round-tripping, cascade routing, citation index
mechanics, the L2b grounding check (including a real-encoder test
reproducing an actual misattribution), and an AST-based lock asserting the
backend never imports the UI framework.

## Project structure

```
backend/
  state.py          shared state schema + per-job config snapshot
  graph.py          LangGraph state machine + Send fan-out routing
  main.py           FastAPI job API (POST /research, GET /status, /result)
  nodes/            preflight, planner, supervisor, arxiv/web agents,
                    context_eval, writer, critic, finalizer
rag/                parent-child chunker, two-stage retriever, indexer, tools
evaluation/         guardrail (L1), citation check (L2a), grounding (L2b)
frontend/app.py     Chainlit UI
experiments/        HyDE A/B, cascade effectiveness, threshold validation
  lora_supervisor/  router distillation (data prep, train, eval)
scripts/            smoke test, demo dry-run harness
docker/             Dockerfile + entrypoint.sh (base image), plus a Caddy +
                    prod-compose overlay (docker-compose.prod.yml) for a
                    public deploy with TLS/basic-auth
tests/              104 unit + integration tests
docs/results.md     full evaluation write-up
```

## Known limitations

- **Semantic Illusion.** Embedding similarity can't reliably detect
  semantically plausible hallucinations — e.g. citing the wrong paper from
  the same subfield, which sits in the same embedding neighborhood as the
  right one. Addressing this needs NLI or an LLM judge.
- **HyDE showed no measurable retrieval win** at n=10 (the aggregate
  improvement traces to a single query). It stays on because it's cheap
  and harmless; rerunning at n=50 is the planned check.

## Next steps

Each open limitation has a concrete follow-up experiment:

- **HyDE at scale.** The n=10 improvement (1.40 → 1.20 rollbacks) traces
  to a single query, so it may be noise. Rerun
  [`experiments/hyde_ab.py`](experiments/hyde_ab.py) at n=50 to see if the
  effect survives a larger sample.
- **Larger L3 ground truth.** The measured cascade F1 (0.562) rests on
  only 12 gradable escalate-band sentences. Extend
  [`experiments/measure_l3.py`](experiments/measure_l3.py) to a larger
  human-annotated batch so the judge's reliability becomes a stable
  estimate, not a directional one.
- **NLI-based grounding.** L2b's embedding similarity can't separate a
  correctly-cited from a wrongly-cited same-subfield paper (the embedding
  ceiling). Replace the cosine check with an NLI model — testing whether
  the cited chunk *entails* the sentence — to attack the same-subfield
  misattribution the embedding signal misses.
- **Targeted rollback feedback.** Rollbacks currently hand the Writer the
  whole draft for a global rewrite. Appending the specific L2b failures
  instead ("sentence K cited [J] but isn't supported by it — rewrite only
  that sentence, or drop the citation") would turn retries into local
  edits — and may make the F1-optimal 0.82 threshold convergent enough to
  ship, recovering the recall that 0.70 trades away.

Full evaluation write-up: [docs/results.md](docs/results.md).
