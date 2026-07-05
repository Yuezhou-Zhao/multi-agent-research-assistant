"""ROLE 2 — training labels via the PRODUCTION Supervisor (gpt-4o-mini).

Runs every training query through the exact production classifier by
importing SupervisorAgent from backend/nodes/supervisor.py — NOT a copied
prompt. This guarantees the distillation targets are byte-identical to what
the deployed Supervisor would emit, so the fine-tuned student is imitating
production, not a reconstruction of it.

Output train_set.jsonl rows:
  {id, query, intended_category, use_arxiv, use_web, reason, route,
   label_model, label_source}

Resumable: re-running skips ids already in the output file, so an
interrupted run (network blip) continues instead of re-billing finished rows.

Run:
  python -m experiments.qlora_supervisor.label_training
"""
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from backend.nodes.supervisor import SupervisorAgent  # noqa: E402

from . import config  # noqa: E402
from .common import (  # noqa: E402
    append_jsonl,
    done_ids,
    read_jsonl,
    route_of,
    with_retries,
)


async def _label_one(agent: SupervisorAgent, row: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        decision = await with_retries(lambda: agent.classify(row["query"]))
    use_arxiv = bool(decision.get("use_arxiv", False))
    use_web = bool(decision.get("use_web", False))
    return {
        "id": row["id"],
        "query": row["query"],
        "intended_category": row.get("intended_category"),
        "use_arxiv": use_arxiv,
        "use_web": use_web,
        "reason": decision.get("reason", ""),
        "route": route_of(use_arxiv, use_web),
        "label_model": config.TRAIN_LABEL_MODEL,
        "label_source": "production_supervisor_prompt",
    }


async def main() -> int:
    queries = read_jsonl(config.TRAIN_QUERIES_PATH)
    if not queries:
        raise SystemExit(
            f"No training queries at {config.TRAIN_QUERIES_PATH}. "
            f"Run generate_queries.py first."
        )

    already = done_ids(config.TRAIN_SET_PATH)
    todo = [q for q in queries if q["id"] not in already]
    print(f"ROLE 2 — labeling {len(todo)} training queries with "
          f"{config.TRAIN_LABEL_MODEL} (production Supervisor); "
          f"{len(already)} already done")

    agent = SupervisorAgent()
    sem = asyncio.Semaphore(config.MAX_CONCURRENCY)

    tasks = [asyncio.create_task(_label_one(agent, q, sem)) for q in todo]
    done = 0
    for coro in asyncio.as_completed(tasks):
        try:
            labeled = await coro
        except Exception as e:  # noqa: BLE001
            print(f"  ! failed a query after retries: {type(e).__name__}: {e}")
            continue
        append_jsonl(config.TRAIN_SET_PATH, labeled)
        done += 1
        if done % 50 == 0:
            print(f"  labeled {done}/{len(todo)}")

    rows = read_jsonl(config.TRAIN_SET_PATH)
    dist: dict[str, int] = {}
    for r in rows:
        dist[r["route"]] = dist.get(r["route"], 0) + 1
    print(f"  wrote {config.TRAIN_SET_PATH} — {len(rows)} rows")
    print(f"  route distribution: {dist}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
