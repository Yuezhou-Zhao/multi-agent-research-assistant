"""Config for the Supervisor QLoRA distillation data-prep pipeline.

The whole point of this experiment is to replace the Supervisor's single
gpt-4o-mini routing call (backend/nodes/supervisor.py) with a small local
model fine-tuned via QLoRA — then report accuracy vs. the API version plus
the cost/latency delta. This module only prepares the *data*; the fine-tune
itself is run separately.

THREE INDEPENDENT DATA ROLES (so no model grades its own homework):

  ROLE 1 — Query generation        (qwen3.5-397b-a17b)
    Generates a diverse synthetic query pool spanning four intended
    routing categories. This is a *generation* role, not a labeling role.

  ROLE 2 — Training labels          (gpt-4o-mini, EXACT production prompt)
    Runs the whole training pool through the real Supervisor classifier
    (SupervisorAgent, imported — not re-copied — so it cannot drift from
    production). The student model distills FROM these labels.

  ROLE 3 — Held-out eval labels     (qwen3.5-397b-a17b)
    Labels a disjoint eval pool with a DIFFERENT model than ROLE 2, using
    the identical routing rubric. Because the eval judge is a different
    model from the one the student imitates, eval accuracy measures
    generalization to an independent oracle — not memorization of
    gpt-4o-mini's quirks. ~20 eval labels are flagged for human
    spot-check before the eval set is trusted.

Independence summary:
  - train queries  ∩  eval queries   = ∅  (enforced by dedup, see dedup.py)
  - train labeler (gpt-4o-mini)      ≠  eval labeler (qwen)   ← key axis
  - generator (qwen) overlaps both pools: acceptable, since generation is
    not labeling. Documented as a deliberate choice in README.md.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env before reading any endpoint config, so every script in this
# package (generate/label_training/label_eval) sees the qwen + OpenAI vars
# regardless of which one is the entry point.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ── Endpoints (env-configured; never hardcode secrets in committed code) ──
# ROLE 2 training labeler is gpt-4o-mini via the production SupervisorAgent,
# which reads OPENAI_API_KEY itself. ROLES 1 & 3 use this qwen endpoint.
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.5-397b-a17b")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")

TRAIN_LABEL_MODEL = "gpt-4o-mini"  # must match SupervisorAgent.LLM_MODEL

# ── Intended generation categories (ROLE 1) ──────────────────────────────
# These drive DIVERSITY of generation only. The actual training/eval label
# is whatever the labeler decides (use_arxiv, use_web) — the intended
# category is metadata, not ground truth.
CATEGORIES = ("arxiv_only", "web_only", "both", "ambiguous")

# Targets (within the user's requested ranges: ~600-1,100 train / 100-150 eval)
TRAIN_PER_CATEGORY = 200          # → 800 training queries
EVAL_PER_CATEGORY = 30            # → 120 held-out eval queries
GEN_BATCH_SIZE = 25               # queries requested per qwen generation call
SPOTCHECK_N = 20                  # eval labels flagged for human review

# ── Concurrency / robustness ─────────────────────────────────────────────
MAX_CONCURRENCY = 8              # simultaneous label calls
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0          # seconds; exponential backoff

# ── Near-duplicate threshold for train/eval disjointness ─────────────────
# Cosine (bge-small-en-v1.5) above this = treat as the same query and drop
# from eval. Exact normalized-string match is also always dropped.
DEDUP_COSINE_THRESHOLD = 0.92

# ── Paths ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

TRAIN_QUERIES_PATH = DATA_DIR / "train_queries.jsonl"      # ROLE 1 out
EVAL_QUERIES_PATH = DATA_DIR / "eval_queries.jsonl"        # ROLE 1 out (disjoint)
TRAIN_SET_PATH = DATA_DIR / "train_set.jsonl"             # ROLE 2 out
EVAL_SET_PATH = DATA_DIR / "eval_set.jsonl"               # ROLE 3 out
SPOTCHECK_PATH = DATA_DIR / "eval_spotcheck.csv"          # ROLE 3 out (human review)
STATS_PATH = DATA_DIR / "dataset_stats.json"


def require_qwen() -> None:
    """Fail fast with an actionable message if the qwen endpoint isn't set."""
    missing = [
        name
        for name, val in (
            ("QWEN_BASE_URL", QWEN_BASE_URL),
            ("QWEN_API_KEY", QWEN_API_KEY),
        )
        if not val
    ]
    if missing:
        raise SystemExit(
            f"Missing env var(s): {', '.join(missing)}. "
            f"Set them in .env (see .env.example). This experiment needs the "
            f"qwen3.5-397b-a17b endpoint for query generation and eval labeling."
        )
