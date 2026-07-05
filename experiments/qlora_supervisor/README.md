# Supervisor QLoRA Distillation — Data Preparation

Prepares the training + eval data to fine-tune a **small local model** (e.g.
Qwen2.5-1.5B via QLoRA) that replaces the Supervisor's single `gpt-4o-mini`
routing call (`backend/nodes/supervisor.py`) with local inference — then lets
us report accuracy vs. the API version plus the cost/latency delta.

**This directory only prepares the data.** The QLoRA fine-tune itself is run
separately, later.

## Three independent data roles (no model grades its own homework)

| Role | Model | Produces |
|---|---|---|
| **1 · Query generation** | `qwen3.5-397b-a17b` | diverse synthetic query pool (train + disjoint eval) |
| **2 · Training labels** | `gpt-4o-mini`, **exact production Supervisor prompt** | `train_set.jsonl` — the distillation targets |
| **3 · Held-out eval labels** | `qwen3.5-397b-a17b` (independent judge) | `eval_set.jsonl` + `eval_spotcheck.csv` |

**Why this design:**

- The **training labeler (gpt-4o-mini)** is the model the student imitates. It
  uses the *actual* production classifier — `SupervisorAgent` is imported, not
  re-copied, so the targets can't drift from what production emits.
- The **eval labeler (qwen)** is a *different* model applying the *identical*
  routing rubric. So eval accuracy later measures whether the fine-tuned
  student generalizes to an **independent oracle**, not whether it memorized
  gpt-4o-mini's quirks. This is the "not grading its own homework" property.
- The **generator (qwen)** produces both pools. Generation ≠ labeling, so this
  overlap is acceptable; the independence that matters is enforced elsewhere:
  - train queries ∩ eval queries = ∅ (exact + embedding dedup, `dedup.py`)
  - train labeler ≠ eval labeler (the key axis above)
- **20 eval labels are flagged for human spot-check** before the eval set is
  trusted — disagreements (qwen route ≠ the query's intended category) first,
  then a stratified sample.

## Layout

```
config.py            endpoints (env), category/count targets, paths
common.py            qwen client, robust JSON parse, retries, JSONL I/O
dedup.py             exact + embedding-based train/eval disjointness
generate_queries.py  ROLE 1
label_training.py    ROLE 2
label_eval.py        ROLE 3
data/                generated artifacts (gitignored)
```

## Setup

Set the qwen endpoint in `.env` (see `.env.example`):

```
QWEN_BASE_URL=http://<host>:<port>/<model-path>/v1
QWEN_MODEL=qwen3.5-397b-a17b
QWEN_API_KEY=...
```

The endpoint is OpenAI-compatible (vLLM). `OPENAI_API_KEY` (already set for the
main project) drives ROLE 2.

## Run

```bash
# 0. Validate endpoint + full pipeline on tiny counts first (3/category, 1 batch)
python -m experiments.qlora_supervisor.generate_queries --pool both --smoke

# 1. ROLE 1 — generate the real pools (~800 train / ~120 eval, disjoint)
python -m experiments.qlora_supervisor.generate_queries --pool both

# 2. ROLE 2 — label the training pool with gpt-4o-mini (production Supervisor)
python -m experiments.qlora_supervisor.label_training

# 3. ROLE 3 — label the held-out eval pool with qwen + emit the spot-check sheet
python -m experiments.qlora_supervisor.label_eval

# 4. Hand-verify data/eval_spotcheck.csv (fill the human_* columns), then the
#    eval set is trusted and ready to score a fine-tuned student against.
```

All labeling steps are **resumable** — re-running skips ids already in the
output file, so a network interruption continues instead of re-billing.

## Outputs (in `data/`, gitignored)

| File | Role | Contents |
|---|---|---|
| `train_queries.jsonl` | 1 | generated training queries + intended category |
| `eval_queries.jsonl` | 1 | generated eval queries (disjoint from train) |
| `train_set.jsonl` | 2 | `{query, use_arxiv, use_web, reason, route, label_model}` |
| `eval_set.jsonl` | 3 | same schema, labeled by the independent qwen judge |
| `eval_spotcheck.csv` | 3 | 20 rows with blank `human_*` columns to verify |

## Config knobs (`config.py`)

`TRAIN_PER_CATEGORY` (200 → 800 total), `EVAL_PER_CATEGORY` (30 → 120),
`DEDUP_COSINE_THRESHOLD` (0.92), `MAX_CONCURRENCY` (8), `SPOTCHECK_N` (20).
