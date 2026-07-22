"""Container smoke test — end-to-end verify step.

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


async def _check_mcp_server() -> bool:
    """Verify the web-research MCP server is reachable and advertises its tools.

    Worth its own check rather than trusting the end-to-end run: web_agent_node
    fails open to an in-process Tavily call, so a missing or broken server
    produces a perfectly green pipeline with a component silently bypassed.
    This is the check that catches, for instance, mcp_servers/ not being COPYed
    into the image.
    """
    from rag.mcp_client import WebResearchMCPClient

    client = WebResearchMCPClient()
    try:
        names = await client.tool_names()
    except Exception as exc:
        print(f"[smoke] FAIL — web-research MCP server unreachable: {exc}")
        return False
    finally:
        await client.aclose()

    missing = {"tavily_search", "url_scraper"} - names
    if missing:
        print(f"[smoke] FAIL — MCP server missing tools: {sorted(missing)}")
        return False

    print(f"[smoke] mcp: web-research OK, tools={sorted(names)}")
    return True


async def main() -> int:
    print(f"[smoke] query: {QUERY}")

    if not await _check_mcp_server():
        return 1

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
