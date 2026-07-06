"""ROLE 3 — held-out eval labels via qwen3.5-397b-a17b (independent oracle).

Labels the DISJOINT eval pool with a DIFFERENT model than ROLE 2, using the
IDENTICAL routing rubric (the production Supervisor's CLASSIFICATION_PROMPT,
imported so the rubric can't drift). Because the eval judge (qwen) differs
from the model the student distills from (gpt-4o-mini), eval accuracy later
measures generalization to an independent oracle — not memorization of one
model's quirks.

Also flags SPOTCHECK_N eval labels for the human to verify before the eval
set is trusted, prioritizing the most decision-relevant rows: cases where
qwen's route disagrees with the query's intended category (likely-hard or
possibly-mislabeled), then a stratified fill across routes.

Outputs:
  eval_set.jsonl        full independent labels
  eval_spotcheck.csv    SPOTCHECK_N rows with blank human-review columns

Run:
  python -m experiments.lora_supervisor.label_eval
"""
import asyncio
import csv
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
    extract_json,
    qwen_chat,
    read_jsonl,
    route_of,
)

# Same rubric as production (ROLE 2), applied by a different model.
_RUBRIC = SupervisorAgent.CLASSIFICATION_PROMPT

# intended_category → the route we'd expect if the generator was on-target.
# 'ambiguous' has no single expected route, so it never counts as a
# disagreement.
_EXPECTED_ROUTE = {"arxiv_only": "arxiv", "web_only": "web", "both": "both"}


async def _label_one(row: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        reply = await qwen_chat(_RUBRIC.format(query=row["query"]), temperature=0.0)
    try:
        decision = extract_json(reply)
    except ValueError:
        decision = {}
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
        "label_model": config.QWEN_MODEL,
        "label_source": "production_rubric_independent_judge",
    }


def _select_spotcheck(rows: list[dict], n: int) -> list[dict]:
    """Pick n rows to hand-verify: disagreements first (qwen route ≠ the
    intended category's expected route), then a deterministic stratified
    fill across routes. Deterministic — sorted by id, no RNG."""
    rows = sorted(rows, key=lambda r: r["id"])

    def is_disagreement(r: dict) -> bool:
        exp = _EXPECTED_ROUTE.get(r.get("intended_category"))
        return exp is not None and r["route"] != exp

    disagreements = [r for r in rows if is_disagreement(r)]
    picked = list(disagreements[:n])

    if len(picked) < n:
        picked_ids = {r["id"] for r in picked}
        rest = [r for r in rows if r["id"] not in picked_ids]
        # Stride-sample the rest for spread rather than taking a contiguous run.
        if rest:
            stride = max(1, len(rest) // (n - len(picked)))
            for r in rest[::stride]:
                if len(picked) >= n:
                    break
                picked.append(r)
    return picked[:n]


def _write_spotcheck(picked: list[dict]) -> None:
    cols = [
        "id", "query", "intended_category",
        "qwen_use_arxiv", "qwen_use_web", "qwen_route", "qwen_reason",
        "flag_reason",
        "human_agree_Y_N", "human_route_if_diff", "human_note",
    ]
    with open(config.SPOTCHECK_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in picked:
            exp = _EXPECTED_ROUTE.get(r.get("intended_category"))
            flag = (
                f"route '{r['route']}' != expected '{exp}' for category "
                f"'{r['intended_category']}'"
                if exp and r["route"] != exp
                else "stratified sample"
            )
            w.writerow(
                {
                    "id": r["id"],
                    "query": r["query"],
                    "intended_category": r["intended_category"],
                    "qwen_use_arxiv": r["use_arxiv"],
                    "qwen_use_web": r["use_web"],
                    "qwen_route": r["route"],
                    "qwen_reason": r["reason"],
                    "flag_reason": flag,
                    "human_agree_Y_N": "",
                    "human_route_if_diff": "",
                    "human_note": "",
                }
            )


async def main() -> int:
    queries = read_jsonl(config.EVAL_QUERIES_PATH)
    if not queries:
        raise SystemExit(
            f"No eval queries at {config.EVAL_QUERIES_PATH}. "
            f"Run generate_queries.py --pool eval first."
        )
    config.require_qwen()

    already = done_ids(config.EVAL_SET_PATH)
    todo = [q for q in queries if q["id"] not in already]
    print(f"ROLE 3 — labeling {len(todo)} held-out eval queries with "
          f"{config.QWEN_MODEL} (independent judge); {len(already)} already done")

    sem = asyncio.Semaphore(config.MAX_CONCURRENCY)
    tasks = [asyncio.create_task(_label_one(q, sem)) for q in todo]
    done = 0
    for coro in asyncio.as_completed(tasks):
        try:
            labeled = await coro
        except Exception as e:  # noqa: BLE001
            print(f"  ! failed a query after retries: {type(e).__name__}: {e}")
            continue
        append_jsonl(config.EVAL_SET_PATH, labeled)
        done += 1
        if done % 25 == 0:
            print(f"  labeled {done}/{len(todo)}")

    rows = read_jsonl(config.EVAL_SET_PATH)
    dist: dict[str, int] = {}
    for r in rows:
        dist[r["route"]] = dist.get(r["route"], 0) + 1
    print(f"  wrote {config.EVAL_SET_PATH} — {len(rows)} rows")
    print(f"  route distribution: {dist}")

    picked = _select_spotcheck(rows, config.SPOTCHECK_N)
    _write_spotcheck(picked)
    n_disagree = sum(
        1 for r in picked
        if _EXPECTED_ROUTE.get(r.get("intended_category"))
        and r["route"] != _EXPECTED_ROUTE[r["intended_category"]]
    )
    print(f"  wrote {config.SPOTCHECK_PATH} — {len(picked)} rows for human review "
          f"({n_disagree} disagreements + {len(picked) - n_disagree} stratified)")
    print("  → fill human_agree_Y_N / human_route_if_diff / human_note, "
          "then hand back for trust sign-off.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
