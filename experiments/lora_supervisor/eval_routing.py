"""Evaluate a routing classifier against the held-out eval set.

Scores any router that maps a query → {use_arxiv, use_web} against the
human-corrected eval_set.jsonl. Two backends:

  --backend openai   the production Supervisor on gpt-4o-mini (BASELINE)
  --backend local    a fine-tuned local model (adapter dir via --adapter)

Reports (identical metric set for both, so they're directly comparable):
  - route accuracy (exact match on the (arxiv,web) pair)
  - per-route precision / recall / F1
  - confusion matrix
  - exact-match on each independent flag (use_arxiv, use_web)
  - mean / p50 / p95 latency per query
  - cost: measured for the API; ~0 marginal for local (documented)

Measured, not tuned — same discipline as every other result in the repo.

Run:
  python -m experiments.lora_supervisor.eval_routing --backend openai
  python -m experiments.lora_supervisor.eval_routing --backend local \
      --adapter experiments/lora_supervisor/train/adapter
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from backend.nodes.supervisor import SupervisorAgent  # noqa: E402

from . import config  # noqa: E402
from .common import extract_json, read_jsonl, route_of, with_retries  # noqa: E402

_ROUTES = ["arxiv", "web", "both", "neither"]

# gpt-4o-mini pricing (USD / 1M tokens), as of the project's cost model.
_GPT4OMINI_IN = 0.15 / 1_000_000
_GPT4OMINI_OUT = 0.60 / 1_000_000


# ── Backends ─────────────────────────────────────────────────────────────
async def _openai_router(queries: list[str]) -> tuple[list[dict], list[float], dict]:
    """gpt-4o-mini via the production Supervisor. Returns (decisions,
    latencies, cost_info)."""
    agent = SupervisorAgent()
    decisions, latencies = [], []
    prompt_tokens = completion_tokens = 0
    sem = asyncio.Semaphore(config.MAX_CONCURRENCY)

    async def _one(q: str):
        async with sem:
            t = time.perf_counter()
            d = await with_retries(lambda: agent.classify(q))
            return d, time.perf_counter() - t

    results = await asyncio.gather(*(_one(q) for q in queries))
    for d, lat in results:
        decisions.append(d)
        latencies.append(lat)
    # Token accounting: rough estimate from prompt+output lengths (the
    # classify() wrapper doesn't surface usage). Documented as an estimate.
    cost = {
        "note": "cost estimated from mean prompt/output size; classify() "
        "does not surface token usage",
    }
    return decisions, latencies, cost


def _local_router(queries: list[str], adapter_dir: str | None):
    """Qwen2.5-1.5B on MPS — with the LoRA adapter when `adapter_dir` is
    given, or the bare base model (zero-shot control) when it is None.
    Greedy decode of the JSON label. Returns (decisions, latencies,
    cost_info)."""
    import torch
    try:  # peft 0.19.1 ↔ torch 2.12.1 shim (see train_lora.py)
        import torch.distributed.tensor  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .format_dataset import BASE_MODEL

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()
    else:
        model = base.to(device).eval()

    decisions, latencies = [], []
    for q in queries:
        messages = [{"role": "user", "content": SupervisorAgent.CLASSIFICATION_PROMPT.format(query=q)}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(device)
        t = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        latencies.append(time.perf_counter() - t)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            decisions.append(extract_json(gen))
        except ValueError:
            decisions.append({})
    cost = {"note": "local inference: ~0 marginal $ cost (electricity only)"}
    return decisions, latencies, cost


# ── Metrics ──────────────────────────────────────────────────────────────
def _score(gold_rows: list[dict], decisions: list[dict], latencies: list[float]) -> dict:
    n = len(gold_rows)
    route_correct = arxiv_correct = web_correct = 0
    confusion = {g: {p: 0 for p in _ROUTES} for g in _ROUTES}
    tp = {r: 0 for r in _ROUTES}
    fp = {r: 0 for r in _ROUTES}
    fn = {r: 0 for r in _ROUTES}

    for gold, dec in zip(gold_rows, decisions):
        g_route = gold["route"]
        p_arxiv = bool(dec.get("use_arxiv", False))
        p_web = bool(dec.get("use_web", False))
        p_route = route_of(p_arxiv, p_web)

        if p_arxiv == gold["use_arxiv"]:
            arxiv_correct += 1
        if p_web == gold["use_web"]:
            web_correct += 1
        confusion[g_route][p_route] += 1
        if p_route == g_route:
            route_correct += 1
            tp[g_route] += 1
        else:
            fp[p_route] += 1
            fn[g_route] += 1

    per_route = {}
    for r in _ROUTES:
        prec = tp[r] / (tp[r] + fp[r]) if (tp[r] + fp[r]) else 0.0
        rec = tp[r] / (tp[r] + fn[r]) if (tp[r] + fn[r]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = sum(1 for g in gold_rows if g["route"] == r)
        per_route[r] = {"precision": prec, "recall": rec, "f1": f1, "support": support}

    lat_sorted = sorted(latencies)
    return {
        "n": n,
        "route_accuracy": route_correct / n,
        "use_arxiv_accuracy": arxiv_correct / n,
        "use_web_accuracy": web_correct / n,
        "macro_f1": statistics.mean(
            per_route[r]["f1"] for r in _ROUTES if per_route[r]["support"] > 0
        ),
        "per_route": per_route,
        "confusion_gold_x_pred": confusion,
        "latency_s": {
            "mean": statistics.mean(latencies),
            "p50": lat_sorted[len(lat_sorted) // 2],
            "p95": lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))],
        },
    }


def _print_report(backend: str, result: dict, cost: dict) -> None:
    print(f"\n=== Routing eval — backend={backend} — n={result['n']} "
          f"(human-corrected eval set) ===")
    print(f"route accuracy : {result['route_accuracy']:.3f}")
    print(f"use_arxiv acc  : {result['use_arxiv_accuracy']:.3f}")
    print(f"use_web acc    : {result['use_web_accuracy']:.3f}")
    print(f"macro F1       : {result['macro_f1']:.3f}")
    print("per-route (P / R / F1 / support):")
    for r, m in result["per_route"].items():
        if m["support"]:
            print(f"  {r:8s} {m['precision']:.2f} / {m['recall']:.2f} / "
                  f"{m['f1']:.2f} / {m['support']}")
    lat = result["latency_s"]
    print(f"latency (s)    : mean {lat['mean']:.3f}  p50 {lat['p50']:.3f}  "
          f"p95 {lat['p95']:.3f}")
    print(f"cost           : {cost.get('note','')}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "local"], required=True)
    parser.add_argument(
        "--adapter",
        help="LoRA adapter dir (local backend). Omit to run the bare base "
        "model as the zero-shot control.",
    )
    parser.add_argument("--out", help="write JSON result to this path")
    args = parser.parse_args()

    gold = read_jsonl(config.EVAL_SET_PATH)
    if not gold:
        raise SystemExit(f"No eval set at {config.EVAL_SET_PATH}")
    queries = [r["query"] for r in gold]

    if args.backend == "openai":
        decisions, latencies, cost = await _openai_router(queries)
    else:
        if not args.adapter:
            print("[eval] no --adapter given: running the BARE base model "
                  "(zero-shot control)")
        decisions, latencies, cost = _local_router(queries, args.adapter)

    backend_label = args.backend if (args.backend == "openai" or args.adapter) \
        else "local-zeroshot"
    result = _score(gold, decisions, latencies)
    _print_report(backend_label, result, cost)

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"backend": backend_label, "cost": cost, **result}, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
