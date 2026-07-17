"""Cascade effectiveness experiment.

For each Critic invocation in the pipeline, this script logs:
  - Gamma-only resolve rate: fraction of sentences where SF < 0.05 (reject)
    OR SF > 0.25 (approve) — decided by Gamma alone, no LLM.
  - LLM Critic invocation rate: fraction of sentences in the uncertain
    "escalate" band (would have fired the L3 judge in isolation).
  - Per-sentence cascade decision + SF score + which chunks the L3
    judge would have seen.

WHAT THIS SCRIPT DOES NOT COMPUTE: the companion metric, "Final F1
vs Gamma-only F1." That needs per-sentence ground-truth
correct/hallucinated labels. This script instead dumps a labeling
sheet (experiments/results/cascade_labels.csv) with every scored
sentence for the user to hand-label; a companion `compute_f1.py`
consumes the labeled CSV to produce the F1 numbers. Per user
instruction: n=10 first so the labeling cost is scoped before
committing to n=20.

Runs deterministically at sf_threshold=0.15 (default) and hyde on so
the trace matches what the Chainlit demo shows.
"""
import asyncio
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from backend.graph import compiled_graph
from backend.nodes.preflight import HyDEOperator
from backend.state import new_job_state
from evaluation.gamma_guardrail import GammaGuardrail, build_sentence_guardrail

# Same query set as HyDE A/B for direct comparability. Extend to 20
# when the user opts in (add 10 more here).
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


async def run_one(query: str, hyde: HyDEOperator, graph, guardrail: GammaGuardrail) -> dict:
    """Run one query end-to-end, then re-score the final draft's sentences
    so we can dump each with its SF + decision for the labeling sheet.
    (Re-scoring is fine — Gamma is deterministic and cheap.)"""
    state = new_job_state(
        job_id=f"cascade-{hash(query) & 0xffff:x}",
        query=query,
        hyde_enabled=True,
        sf_threshold=SF_THRESHOLD,
    )
    t0 = time.time()
    search_payload, hyde_used = await hyde.execute(query, True)
    state["search_payload"] = search_payload
    if hyde_used:
        new_total = state["total_llm_calls"] + 1
        state["total_llm_calls"] = new_total
        state["llm_budget_exceeded"] = new_total >= state["max_llm_calls"]

    final = await graph.ainvoke(state)
    elapsed = time.time() - t0

    # Rescore the final draft's sentences for the labeling sheet.
    # cascade_decisions on state is from the LAST Critic pass, but the
    # final_answer may be sanitized by L2a — rescore against the
    # sanitized text to keep sentence alignment consistent.
    sentences = GammaGuardrail._split_sentences(final["final_answer"] or "")
    sf_scores = guardrail.survival_score(sentences) if sentences else []
    per_sentence = []
    for sent, sf in zip(sentences, sf_scores):
        if sf < GammaGuardrail.CERTAIN_WRONG:
            gamma_decision = "reject"
        elif sf > GammaGuardrail.CERTAIN_RIGHT:
            gamma_decision = "approve"
        else:
            gamma_decision = "escalate"
        per_sentence.append(
            {
                "sentence": sent,
                "sf": float(sf),
                "gamma_decision": gamma_decision,
            }
        )

    return {
        "query": query,
        "status": final["status"],
        "critic_loop_count": final["critic_loop_count"],
        "cascade_decisions": final["cascade_decisions"],
        "final_answer": final["final_answer"],
        "citations": final["citations"],
        "coverage_score": final["coverage_score"],
        "total_llm_calls": final["total_llm_calls"],
        "llm_calls_avoided": final["llm_calls_avoided"],
        "per_sentence": per_sentence,
        "elapsed_s": round(elapsed, 1),
    }


def aggregate(runs: list[dict]) -> dict:
    """Headline numbers, computed from the per-sentence
    Gamma decisions on the final drafts."""
    counter = Counter()
    for r in runs:
        for s in r["per_sentence"]:
            counter[s["gamma_decision"]] += 1
    total = sum(counter.values()) or 1
    gamma_resolved = counter["approve"] + counter["reject"]
    return {
        "n_queries": len(runs),
        "n_sentences": total,
        "counts": dict(counter),
        "gamma_only_resolve_rate": gamma_resolved / total,
        "llm_invocation_rate": counter["escalate"] / total,
    }


def write_labeling_sheet(runs: list[dict], path: Path) -> None:
    """Dump one row per (query, sentence) with SF + Gamma decision so
    the user can add a `label` column in a spreadsheet. compute_f1.py
    reads this back once labels are filled in."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "query_idx",
                "query",
                "sentence_idx",
                "sentence",
                "sf",
                "gamma_decision",
                "label",  # user fills in: correct | hallucinated | (blank to skip)
            ]
        )
        for qi, r in enumerate(runs, start=1):
            for si, s in enumerate(r["per_sentence"], start=1):
                w.writerow(
                    [
                        qi,
                        r["query"],
                        si,
                        s["sentence"],
                        f"{s['sf']:.4f}",
                        s["gamma_decision"],
                        "",  # blank for user
                    ]
                )


def to_markdown(runs: list[dict], agg: dict) -> str:
    lines = [
        "## Cascade Effectiveness (n=" f"{agg['n_queries']}" ")",
        "",
        f"Total sentences scored: **{agg['n_sentences']}**",
        "",
        "| Band | Count | Fraction |",
        "|------|------:|---------:|",
        f"| approve (SF > {GammaGuardrail.CERTAIN_RIGHT}) — Gamma alone | "
        f"{agg['counts'].get('approve', 0)} | "
        f"{agg['counts'].get('approve', 0) / agg['n_sentences']:.1%} |",
        f"| escalate (uncertain, needs LLM judge) | "
        f"{agg['counts'].get('escalate', 0)} | "
        f"{agg['counts'].get('escalate', 0) / agg['n_sentences']:.1%} |",
        f"| reject (SF < {GammaGuardrail.CERTAIN_WRONG}) — Gamma alone | "
        f"{agg['counts'].get('reject', 0)} | "
        f"{agg['counts'].get('reject', 0) / agg['n_sentences']:.1%} |",
        "",
        f"**Gamma-only resolve rate:** {agg['gamma_only_resolve_rate']:.1%} "
        f"(design target: ≥ 60%)",
        "",
        f"**LLM Critic invocation rate (escalate band):** "
        f"{agg['llm_invocation_rate']:.1%}",
        "",
        "**Final F1 vs Gamma-only F1:** _pending human labels — see "
        "`experiments/results/cascade_labels.csv`. Run "
        "`python -m experiments.compute_f1` after labeling to fill in._",
    ]
    return "\n".join(lines)


async def main():
    hyde = HyDEOperator()
    graph = compiled_graph()
    guardrail, _ = build_sentence_guardrail()

    runs = []
    for i, query in enumerate(QUERIES, start=1):
        print(f"[{i}/{len(QUERIES)}] {query[:60]}...", flush=True)
        r = await run_one(query, hyde, graph, guardrail)
        counts = Counter(s["gamma_decision"] for s in r["per_sentence"])
        print(
            f"  -> status={r['status']} loops={r['critic_loop_count']} "
            f"gamma counts a/e/r = {counts['approve']}/{counts['escalate']}/{counts['reject']} "
            f"({r['elapsed_s']}s)",
            flush=True,
        )
        runs.append(r)

    agg = aggregate(runs)

    (RESULTS_DIR / "cascade.json").write_text(json.dumps(runs, indent=2))
    write_labeling_sheet(runs, RESULTS_DIR / "cascade_labels.csv")
    (RESULTS_DIR / "cascade.md").write_text(to_markdown(runs, agg))

    print("\n" + "=" * 60)
    print(to_markdown(runs, agg))
    print()
    print(f"labeling sheet: {RESULTS_DIR / 'cascade_labels.csv'}")


if __name__ == "__main__":
    asyncio.run(main())
