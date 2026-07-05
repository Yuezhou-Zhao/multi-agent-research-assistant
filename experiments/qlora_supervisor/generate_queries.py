"""ROLE 1 — synthetic query generation (qwen3.5-397b-a17b).

Generates two DISJOINT pools of research queries:
  - train_queries.jsonl  (~TRAIN_PER_CATEGORY × 4 categories)
  - eval_queries.jsonl   (~EVAL_PER_CATEGORY  × 4 categories, disjoint)

Diversity is forced by generating across four intended routing categories
(arxiv_only / web_only / both / ambiguous). Style is calibrated against the
real queries already used in the HyDE A/B and cascade experiments, so the
synthetic pool matches the phrasing the production Supervisor actually sees.

The intended category is metadata only — the real training/eval LABEL comes
from ROLE 2 / ROLE 3. Here we just want a varied, realistic query pool.

Run:
  python -m experiments.qlora_supervisor.generate_queries --pool both
  python -m experiments.qlora_supervisor.generate_queries --pool both --smoke
"""
import argparse
import asyncio
import json
import sys

from . import config
from .common import qwen_chat, extract_json
from .dedup import dedup_within, remove_overlap

# Real queries from the existing experiments — the style anchor.
_STYLE_EXEMPLARS = [
    "How does chain of thought reasoning improve language model performance?",
    "What methods are used to detect hallucinations in language models?",
    "How does dense passage retrieval compare to BM25 for open-domain QA?",
    "What role does query expansion play in information retrieval?",
    "How do embedding models represent semantic similarity between texts?",
]

_CATEGORY_GUIDANCE = {
    "arxiv_only": (
        "answerable from ACADEMIC PAPERS — methods, architectures, "
        "benchmarks, theory, empirical results. Timeless and technical. "
        "Must NOT depend on recent news, product releases, or current events."
    ),
    "web_only": (
        "need RECENT NEWS, events, product announcements, pricing, tutorials, "
        "or other non-academic/practical information. Must NOT be answerable "
        "from a static academic paper corpus alone."
    ),
    "both": (
        "genuinely need BOTH academic grounding AND recent developments — "
        "e.g. comparing a classic method's original papers to how 2025 "
        "industry systems apply it, or a technique's theory plus its latest "
        "real-world adoption."
    ),
    "ambiguous": (
        "genuinely UNDERSPECIFIED or borderline — short/vague, or mixing "
        "academic and practical signals, so a router could reasonably pick "
        "either source. These probe the classifier's decision boundary."
    ),
}


def _gen_prompt(category: str, n: int) -> str:
    exemplars = "\n".join(f"- {q}" for q in _STYLE_EXEMPLARS)
    return f"""You are generating a diverse benchmark of research queries for a
retrieval router that decides whether to search academic papers (arXiv), the
web, or both.

Generate {n} DISTINCT research queries in this category:
  {category}: queries that are {_CATEGORY_GUIDANCE[category]}

Style — match the phrasing of these real examples (concise, one sentence,
natural research questions):
{exemplars}

Requirements:
- {n} queries, all clearly different from each other (vary topic, subfield,
  phrasing, and length).
- Cover many areas: NLP, vision, RL, systems, theory, retrieval, agents, etc.
- Do NOT number them or add commentary.

Respond with ONLY a JSON array of {n} strings."""


async def _generate_pool(pool_name: str, per_category: int, batch_size: int) -> list[dict]:
    prefix = pool_name  # "train" or "eval"
    rows: list[dict] = []
    counter = 0
    for category in config.CATEGORIES:
        collected: list[str] = []
        # Over-generate to absorb dedup shrinkage; cap batches for safety.
        max_batches = max(3, (per_category // batch_size + 1) * 3)
        for _ in range(max_batches):
            if len(collected) >= per_category:
                break
            want = min(batch_size, per_category - len(collected) + batch_size // 2)
            reply = await qwen_chat(_gen_prompt(category, want), temperature=1.0)
            try:
                arr = extract_json(reply)
            except ValueError:
                continue
            if isinstance(arr, dict):  # some models wrap in {"queries": [...]}
                arr = next((v for v in arr.values() if isinstance(v, list)), [])
            for q in arr:
                if isinstance(q, str) and q.strip():
                    collected.append(q.strip())
        collected = collected[:per_category]
        for q in collected:
            counter += 1
            rows.append(
                {
                    "id": f"{prefix}-{counter:06d}",
                    "query": q,
                    "intended_category": category,
                    "generator": config.QWEN_MODEL,
                    "pool": pool_name,
                }
            )
        print(f"  [{pool_name}] {category}: collected {len(collected)}")
    return rows


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=["train", "eval", "both"], default="both")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run (3/category, 1 batch) to validate the endpoint + pipeline",
    )
    args = parser.parse_args()

    config.require_qwen()
    per_train = 3 if args.smoke else config.TRAIN_PER_CATEGORY
    per_eval = 3 if args.smoke else config.EVAL_PER_CATEGORY
    batch = 5 if args.smoke else config.GEN_BATCH_SIZE

    train_rows = None
    if args.pool in ("train", "both"):
        print("ROLE 1 — generating TRAIN pool (qwen)")
        train_rows = await _generate_pool("train", per_train, batch)
        train_rows, dropped = dedup_within(train_rows)
        print(f"  train after dedup: {len(train_rows)} (dropped {dropped})")
        from .common import write_jsonl

        write_jsonl(config.TRAIN_QUERIES_PATH, train_rows)
        print(f"  wrote {config.TRAIN_QUERIES_PATH}")

    if args.pool in ("eval", "both"):
        print("ROLE 1 — generating EVAL pool (qwen, held-out)")
        eval_rows = await _generate_pool("eval", per_eval, batch)
        eval_rows, dropped = dedup_within(eval_rows)
        print(f"  eval after within-dedup: {len(eval_rows)} (dropped {dropped})")

        # Enforce disjointness vs. the training pool.
        from .common import read_jsonl, write_jsonl

        train_for_overlap = train_rows if train_rows is not None else read_jsonl(
            config.TRAIN_QUERIES_PATH
        )
        if train_for_overlap:
            eval_rows, removed = remove_overlap(eval_rows, train_for_overlap)
            print(f"  eval after train-overlap removal: {len(eval_rows)} "
                  f"(removed {removed} that collided with train)")
        else:
            print("  WARNING: no train pool found — cannot verify disjointness")
        write_jsonl(config.EVAL_QUERIES_PATH, eval_rows)
        print(f"  wrote {config.EVAL_QUERIES_PATH}")

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
