"""HyDE A/B experiment — Section 5.2, 10 test queries × {off, on}.

For each (query, hyde_enabled) run, records:
  - Critic rollback count (state["critic_loop_count"])
  - Mean Gamma SF of verified chunks (mean of state["gamma_scores"])
  - Mean BGE-family reranker score of retrieved chunks
    (rerank_score is stashed on each chunk by arxiv_agent.py)

Emits a Markdown table matching Section 5.2's schema plus a JSON sidecar
with the raw numbers, saved under experiments/results/. The README
Week 7 update pulls the Markdown block from here verbatim.

Runs 20 total jobs (~10-15 min end-to-end on M5 Pro depending on
rollback rate). Same-seed reproducibility: sf_threshold and query set
are constants below, and everything else (retrieval, Gamma) is
deterministic given the built index.
"""
import asyncio
import json
import sys
import time
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from backend.graph import compiled_graph
from backend.nodes.preflight import HyDEOperator
from backend.state import new_job_state

# 10 queries picked to sit inside the indexed corpus's topic set (see
# rag/indexer.py's DEFAULT_QUERIES) so retrieval always has real chunks
# to work with — otherwise "HyDE effect" gets confounded with "empty
# retrieval effect".
QUERIES = [
    "How does HyDE improve retrieval-augmented generation for academic papers?",
    "What is the attention mechanism in transformer models?",
    "How does chain of thought reasoning improve language model performance?",
    "What methods are used to detect hallucinations in language models?",
    "How does dense passage retrieval compare to BM25 for open-domain QA?",
    "What is retrieval-augmented generation and how does it work?",
    "How does reranking improve retrieval quality in RAG systems?",
    "What role does query expansion play in information retrieval?",
    "How do embedding models represent semantic similarity between text passages?",
    "What is self-consistency decoding for language models?",
]

SF_THRESHOLD = 0.15
RESULTS_DIR = _REPO_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def run_one(query: str, hyde_enabled: bool, hyde: HyDEOperator, graph) -> dict:
    state = new_job_state(
        job_id=f"hyde-{hyde_enabled}-{hash(query) & 0xffff:x}",
        query=query,
        hyde_enabled=hyde_enabled,
        sf_threshold=SF_THRESHOLD,
    )
    t0 = time.time()
    search_payload, hyde_used = await hyde.execute(query, hyde_enabled)
    state["search_payload"] = search_payload
    if hyde_used:
        new_total = state["total_llm_calls"] + 1
        state["total_llm_calls"] = new_total
        state["llm_budget_exceeded"] = new_total >= state["max_llm_calls"]

    final = await graph.ainvoke(state)
    elapsed = time.time() - t0

    verified_chunks = final.get("verified_chunks") or final.get("merged_chunks") or []
    rerank_scores = [
        c.get("rerank_score") for c in verified_chunks if c.get("rerank_score") is not None
    ]
    return {
        "query": query,
        "hyde_enabled": hyde_enabled,
        "hyde_used": hyde_used,
        "rollbacks": final["critic_loop_count"],
        "status": final["status"],
        "mean_gamma_sf": mean(final["gamma_scores"]) if final["gamma_scores"] else None,
        "mean_rerank_score": mean(rerank_scores) if rerank_scores else None,
        "coverage_score": final["coverage_score"],
        "total_llm_calls": final["total_llm_calls"],
        "llm_calls_avoided": final["llm_calls_avoided"],
        "elapsed_s": round(elapsed, 1),
    }


def _fmt(x, digits=3) -> str:
    return f"{x:.{digits}f}" if x is not None else "—"


def to_markdown(runs: list[dict]) -> str:
    by_query: dict[str, dict[bool, dict]] = {}
    for r in runs:
        by_query.setdefault(r["query"], {})[r["hyde_enabled"]] = r

    header = (
        "| # | Query | Rollbacks (off) | Rollbacks (on) | "
        "SF (off) | SF (on) | Rerank (off) | Rerank (on) |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for i, (query, pair) in enumerate(by_query.items(), start=1):
        off = pair.get(False, {})
        on = pair.get(True, {})
        short = query if len(query) <= 60 else query[:57] + "..."
        rows.append(
            f"| {i} | {short} | "
            f"{off.get('rollbacks', '—')} | {on.get('rollbacks', '—')} | "
            f"{_fmt(off.get('mean_gamma_sf'))} | {_fmt(on.get('mean_gamma_sf'))} | "
            f"{_fmt(off.get('mean_rerank_score'))} | {_fmt(on.get('mean_rerank_score'))} |"
        )
    return header + "\n".join(rows)


def summary(runs: list[dict]) -> str:
    off_runs = [r for r in runs if not r["hyde_enabled"]]
    on_runs = [r for r in runs if r["hyde_enabled"]]

    def _mean_across(runs_, key):
        vals = [r[key] for r in runs_ if r[key] is not None]
        return mean(vals) if vals else None

    off_rollbacks = mean(r["rollbacks"] for r in off_runs)
    on_rollbacks = mean(r["rollbacks"] for r in on_runs)
    off_sf = _mean_across(off_runs, "mean_gamma_sf")
    on_sf = _mean_across(on_runs, "mean_gamma_sf")
    off_rr = _mean_across(off_runs, "mean_rerank_score")
    on_rr = _mean_across(on_runs, "mean_rerank_score")
    n = len(off_runs)
    off_approved = sum(1 for r in off_runs if r["status"] == "approved")
    on_approved = sum(1 for r in on_runs if r["status"] == "approved")

    return (
        f"\n**Aggregate (n={n}):**\n\n"
        f"|                     | HyDE off | HyDE on |\n"
        f"|---------------------|----------|---------|\n"
        f"| mean rollbacks       | {off_rollbacks:.2f} | {on_rollbacks:.2f} |\n"
        f"| mean chunk SF        | {_fmt(off_sf)} | {_fmt(on_sf)} |\n"
        f"| mean rerank score    | {_fmt(off_rr)} | {_fmt(on_rr)} |\n"
        f"| approved / n         | {off_approved}/{n} | {on_approved}/{n} |\n"
    )


async def main():
    hyde = HyDEOperator()
    graph = compiled_graph()
    runs = []

    # Interleave off/on per query so any transient API weather affects
    # both halves roughly equally rather than clustering in one half.
    for i, query in enumerate(QUERIES, start=1):
        for enabled in (False, True):
            print(f"[{i}/{len(QUERIES)}] hyde={enabled} : {query[:60]}...", flush=True)
            r = await run_one(query, enabled, hyde, graph)
            print(
                f"  -> status={r['status']} rollbacks={r['rollbacks']} "
                f"SF={_fmt(r['mean_gamma_sf'])} rerank={_fmt(r['mean_rerank_score'])} "
                f"({r['elapsed_s']}s)",
                flush=True,
            )
            runs.append(r)

    json_path = RESULTS_DIR / "hyde_ab.json"
    md_path = RESULTS_DIR / "hyde_ab.md"
    json_path.write_text(json.dumps(runs, indent=2))

    md = "## HyDE A/B — Section 5.2\n\n" + to_markdown(runs) + summary(runs)
    md_path.write_text(md)

    print("\n" + "=" * 60)
    print(md)
    print("\nsaved:", md_path)
    print("saved:", json_path)


if __name__ == "__main__":
    asyncio.run(main())
