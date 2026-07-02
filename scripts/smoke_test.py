"""Container smoke test — Section 7.2 Week 7 verify step.

Runs one query end-to-end through the compiled graph and asserts the
job hit a terminal status (`approved`, `force_finalized`, or
`budget_exceeded` — all three mean the pipeline completed without
crashing). Used inside the Docker container to prove the image actually
runs, not just that it builds.

Exits 0 on any terminal status + non-empty final_answer, non-zero
otherwise. Prints a compact one-line summary + first 400 chars of the
final answer.

Run from repo root, or inside the container:
  docker compose run --rm --entrypoint python research-agent -m scripts.smoke_test
"""
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from backend.graph import compiled_graph
from backend.nodes.preflight import HyDEOperator
from backend.state import new_job_state

TERMINAL_STATUSES = {"approved", "force_finalized", "budget_exceeded"}

QUERY = "How does chain of thought reasoning improve language model performance?"


async def main() -> int:
    print(f"[smoke] query: {QUERY}")

    state = new_job_state(
        job_id="smoke", query=QUERY, hyde_enabled=True, sf_threshold=0.15
    )

    hyde = HyDEOperator()
    search_payload, hyde_used = await hyde.execute(QUERY, True)
    state["search_payload"] = search_payload
    if hyde_used:
        new_total = state["total_llm_calls"] + 1
        state["total_llm_calls"] = new_total
        state["llm_budget_exceeded"] = new_total >= state["max_llm_calls"]

    graph = compiled_graph()
    final = await graph.ainvoke(state)

    status = final["status"]
    answer = final.get("final_answer") or ""
    print(
        f"[smoke] status={status}  loops={final['critic_loop_count']}/3  "
        f"llm_calls={final['total_llm_calls']}/{final['max_llm_calls']}  "
        f"avoided={final['llm_calls_avoided']}  "
        f"citations={final.get('citations', [])}"
    )
    print(f"[smoke] final_answer preview:\n{answer[:400]}...")

    if status not in TERMINAL_STATUSES:
        print(f"[smoke] FAIL — non-terminal status: {status!r}")
        return 1
    if not answer.strip():
        print("[smoke] FAIL — empty final_answer")
        return 2
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
