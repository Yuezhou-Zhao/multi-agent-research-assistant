"""Demo dry-run — verify the two scripted demo scenarios headlessly.

Runs the demo's two deliberately-chosen queries through the compiled graph
(the exact code the Chainlit UI drives) and checks each against its intended
beat: a query in the corpus's sweet spot that should converge (approved), and
a genuinely hard query that should degrade gracefully (force_finalize via the
circuit breaker). The system is non-deterministic (Writer temperature 0.8 on
rollback), so these are the beats chosen to be *reliable* — run this before a
live demo to confirm they still hold.

Makes real LLM calls (two full pipeline runs). Run:
  python -m scripts.demo_dryrun
"""
import asyncio
import os
import sys
import time
from pathlib import Path

# Models are cached after first use; skip HuggingFace HEAD update-checks so a
# throttled network can't add 10s stalls per model load and corrupt timings.
# Must precede the backend imports.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from backend.graph import compiled_graph
from backend.nodes.preflight import HyDEOperator
from backend.state import new_job_state

# (label, query, hyde, sf, expected terminal status)
SCENARIOS = [
    ("happy path — should converge",
     "How does chain of thought reasoning improve language model performance?",
     True, 0.15, "approved"),
    ("graceful degradation — should force_finalize",
     "How does HyDE improve RAG retrieval?",
     True, 0.15, "force_finalized"),
]


async def _run(query: str, hyde_enabled: bool, sf_threshold: float) -> dict:
    state = new_job_state(
        job_id="demo", query=query, hyde_enabled=hyde_enabled, sf_threshold=sf_threshold
    )
    payload, hyde_used = await HyDEOperator().execute(query, hyde_enabled)
    state["search_payload"] = payload
    if hyde_used:
        n = state["total_llm_calls"] + 1
        state["total_llm_calls"] = n
        state["llm_budget_exceeded"] = n >= state["max_llm_calls"]

    t = time.perf_counter()
    final = await compiled_graph().ainvoke(state)
    elapsed = time.perf_counter() - t
    cascade = final.get("cascade_decisions", []) or []
    return {
        "status": final["status"],
        "rollbacks": final["critic_loop_count"],
        "flagged": cascade.count("reject") + cascade.count("escalate"),
        "n_sent": len(cascade),
        "llm_calls": final["total_llm_calls"],
        "elapsed": elapsed,
    }


async def main() -> int:
    print("Demo dry-run — scripted demo beats\n")
    surprises = []
    for label, query, hyde, sf, expected in SCENARIOS:
        print(f"running [{label}]\n  {query!r} ...", flush=True)
        r = await _run(query, hyde, sf)
        ok = r["status"] == expected
        if not ok:
            surprises.append(label)
        print(
            f"  status={r['status']} (expected {expected})  rollbacks={r['rollbacks']}  "
            f"flagged={r['flagged']}/{r['n_sent']}  llm={r['llm_calls']}  "
            f"{r['elapsed']:.0f}s  -> {'OK' if ok else 'DIVERGED'}\n"
        )

    print("=" * 60)
    if surprises:
        print(f"DRY-RUN: {len(surprises)} beat(s) diverged — the system is "
              f"non-deterministic, so re-run once; if it persists, review "
              f"before the live demo:")
        for s in surprises:
            print(f"  - {s}")
        return 1
    print("DRY-RUN OK — both demo beats behaved as scripted.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
