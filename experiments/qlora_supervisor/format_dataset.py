"""Format train_set.jsonl into chat examples + handle class imbalance.

Each training example is the EXACT production Supervisor prompt as the user
turn and the gpt-4o-mini JSON label as the assistant turn — faithful
distillation of the production classifier onto a small local student.

Class imbalance: the production routing is arxiv-heavy — train route
distribution is roughly arxiv 436 / web 149 / both 57 / neither 0, so `both`
is ~9% of the set. Left alone, a small student tends to collapse `both`
into `arxiv`.

We handle this by **oversampling** the minority routes rather than a
class-weighted loss. Reason: the target is a structured JSON *generation*,
so the training signal is a token-level LM loss over the whole completion —
attaching a single per-example class weight to that token loss is awkward and
couples the weight to unrelated tokens (the prompt echo, the `reason` text).
Oversampling rebalances what the model actually *sees* per route directly and
composes cleanly with the standard causal-LM objective. The multiplier is
capped (`--balance-ratio`, default 1.0 = match the majority) so the tradeoff
against overfitting the few real `both` examples is explicit and tunable.

Run:
  python -m experiments.qlora_supervisor.format_dataset            # ratio 1.0
  python -m experiments.qlora_supervisor.format_dataset --balance-ratio 0.5
"""
import argparse
import collections
import json
import sys

from backend.nodes.supervisor import SupervisorAgent

from . import config
from .common import read_jsonl

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_CHAT_PATH = config.DATA_DIR / "train_chat.jsonl"


def _example(row: dict) -> dict:
    """One chat example: production prompt in, gpt-4o-mini JSON label out."""
    user = SupervisorAgent.CLASSIFICATION_PROMPT.format(query=row["query"])
    # Faithful target = exactly the fields the production prompt asks for.
    target = json.dumps(
        {
            "use_arxiv": bool(row["use_arxiv"]),
            "use_web": bool(row["use_web"]),
            "reason": row.get("reason", ""),
        }
    )
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "route": row["route"],
    }


def _oversample(rows: list[dict], ratio: float) -> tuple[list[dict], dict]:
    """Duplicate minority-route rows until each route reaches
    round(majority * ratio). ratio=1.0 → full balance to the majority count.
    Deterministic (cycles through each route's rows in order)."""
    by_route: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_route[r["route"]].append(r)
    majority = max(len(v) for v in by_route.values())
    target = max(1, round(majority * ratio))

    out: list[dict] = []
    plan = {}
    for route, items in by_route.items():
        if len(items) >= target:
            out.extend(items)
            plan[route] = len(items)
        else:
            dup = [items[i % len(items)] for i in range(target)]
            out.extend(dup)
            plan[route] = target
    return out, plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance-ratio", type=float, default=1.0,
                        help="minority routes oversampled to majority*ratio "
                             "(1.0=full balance, 0.0=no oversampling)")
    args = parser.parse_args()

    rows = read_jsonl(config.TRAIN_SET_PATH)
    if not rows:
        raise SystemExit(f"No train set at {config.TRAIN_SET_PATH}")

    orig = collections.Counter(r["route"] for r in rows)
    print(f"original route distribution: {dict(orig)}  (n={len(rows)})")

    if args.balance_ratio > 0:
        rows, plan = _oversample(rows, args.balance_ratio)
        print(f"after oversample (ratio={args.balance_ratio}): {plan}  "
              f"(n={len(rows)})")
    else:
        print("no oversampling")

    examples = [_example(r) for r in rows]
    with open(TRAIN_CHAT_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {TRAIN_CHAT_PATH} — {len(examples)} chat examples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
