"""Measure the REAL L3 judge on the escalate band.

compute_f1.py's "Final F1" gives the escalate band an ORACLE L3 — it assumes
the LLM judge lands on the true label. This script replaces that assumption
with the actual production Critic L3 (gpt-4o-mini, CriticAgent imported from
backend/nodes/critic.py — not re-implemented) run per escalate sentence, and
recomputes the cascade F1 with the measured verdicts.

Per-sentence protocol: treat each escalate sentence as a one-sentence draft,
re-index its [arxiv_id] citations to the [N] form the judge expects, supply
the matching cited chunks, and read the judge's approved/rejected verdict
(approved -> predicts "correct", rejected -> predicts "hallucinated").

Reports, positive class = hallucinated:
  - L3 standalone on the gradable escalate sentences (its real reliability)
  - Full cascade with MEASURED L3, next to the oracle upper bound (0.600)
Writes experiments/results/cascade_l3_measured.md.

  python -m experiments.measure_l3
"""
import asyncio
import csv
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")

from backend.nodes.critic import CriticAgent  # noqa: E402
from rag.indexer import load_index  # noqa: E402

RESULTS_DIR = _REPO_ROOT / "experiments" / "results"
LABELS_PATH = RESULTS_DIR / "cascade_labels.csv"
OUT_MD = RESULTS_DIR / "cascade_l3_measured.md"

_ARXIV_ID_RE = re.compile(r"\[(\d{4}\.\d{4,5}(?:v\d+)?)\]")
VALID = {"correct", "hallucinated"}


def _f1(tp, fp, fn):
    if tp == 0:
        return 0.0, 0.0, 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _reindex(sentence: str, paper_by_id: dict):
    """Return (reindexed_draft, indexed_cited_chunks) — [arxiv_id] markers
    rewritten to [1],[2],... with the matching chunk contents, the exact
    shape CriticAgent.judge expects. Skips ids missing from the index."""
    ids = []
    for cid in _ARXIV_ID_RE.findall(sentence):
        if cid not in ids and cid in paper_by_id:
            ids.append(cid)
    draft = sentence
    indexed = []
    for k, cid in enumerate(ids, start=1):
        draft = draft.replace(f"[{cid}]", f"[{k}]")
        paper = paper_by_id[cid]
        content = f"{paper.get('title','')} — {paper.get('summary','')}"
        indexed.append((k, {"content": content}))
    # Any un-resolved [id] markers left in the draft are stripped so the judge
    # isn't confused by dangling refs.
    draft = _ARXIV_ID_RE.sub("", draft)
    return draft, indexed


async def main() -> int:
    _, metadata = load_index()
    paper_by_id = metadata["paper_by_id"]

    with open(LABELS_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    escalate = [r for r in rows if r["gamma_decision"].strip() == "escalate"]
    agent = CriticAgent()

    # Run the real judge on every escalate sentence.
    measured = []  # (row, human_label, predicted_label, approved)
    for r in escalate:
        draft, indexed = _reindex(r["sentence"], paper_by_id)
        verdict = await agent.judge(r["query"], draft, indexed)
        approved = bool(verdict.get("approved", False))
        pred = "correct" if approved else "hallucinated"
        measured.append((r, r["label"].strip().lower(), pred, approved))

    # ── L3 standalone on the gradable escalate sentences ──────────────
    gradable = [(h, p) for (_r, h, p, _a) in measured if h in VALID]
    tp = sum(1 for h, p in gradable if p == "hallucinated" and h == "hallucinated")
    fp = sum(1 for h, p in gradable if p == "hallucinated" and h == "correct")
    fn = sum(1 for h, p in gradable if p == "correct" and h == "hallucinated")
    tn = sum(1 for h, p in gradable if p == "correct" and h == "correct")
    correct_calls = sum(1 for h, p in gradable if h == p)
    l3_p, l3_r, l3_f = _f1(tp, fp, fn)

    # ── Full cascade with MEASURED L3 ─────────────────────────────────
    # resolved bands: approve->correct, reject->hallucinated (Gamma decides)
    pairs = []
    for r in rows:
        lab = r["label"].strip().lower()
        if lab not in VALID:
            continue
        band = r["gamma_decision"].strip()
        if band == "approve":
            pairs.append(("correct", lab))
        elif band == "reject":
            pairs.append(("hallucinated", lab))
    # escalate band: use the MEASURED L3 verdict
    for _r, h, p, _a in measured:
        if h in VALID:
            pairs.append((p, h))

    ctp = sum(1 for pred, lab in pairs if pred == "hallucinated" and lab == "hallucinated")
    cfp = sum(1 for pred, lab in pairs if pred == "hallucinated" and lab == "correct")
    cfn = sum(1 for pred, lab in pairs if pred == "correct" and lab == "hallucinated")
    cp, cr, cf = _f1(ctp, cfp, cfn)

    # ── Report ────────────────────────────────────────────────────────
    lines = [
        "## Cascade F1 — measured L3 judge",
        "",
        f"Escalate band: **{len(escalate)}** sentences "
        f"({len(gradable)} gradable correct/hallucinated, "
        f"{len(escalate)-len(gradable)} uncertain dropped).",
        "",
        "### L3 judge (gpt-4o-mini) standalone on the escalate band",
        f"- accuracy on gradable escalate sentences: **{correct_calls}/{len(gradable)}**",
        f"- precision {l3_p:.3f} / recall {l3_r:.3f} / F1 **{l3_f:.3f}** "
        f"(positive = hallucinated; TP {tp} FP {fp} FN {fn} TN {tn})",
        "",
        "### Full cascade F1 (positive = hallucinated)",
        "| variant | F1 |",
        "|---|---|",
        "| Gamma-only (L1) | 0.500 |",
        "| Full cascade, **oracle** L3 (upper bound) | 0.600 |",
        f"| Full cascade, **measured** L3 | **{cf:.3f}** |",
        "",
        f"Measured cascade: precision {cp:.3f} / recall {cr:.3f} / F1 **{cf:.3f}** "
        f"(n={len(pairs)}).",
        "",
        "**What this does and doesn't show.** The measured number is the honest "
        "end-to-end cascade F1 — no oracle. The gap from the 0.600 upper bound "
        "is exactly how often the real gpt-4o-mini judge disagrees with a human "
        "on the hardest (escalate-band) sentences, the ones L1/L2 could not "
        "resolve. It is a small-n figure (escalate band is only "
        f"{len(gradable)} gradable sentences), so treat it as directional, not "
        "a precise operating point.",
    ]
    md = "\n".join(lines)
    print(md)
    OUT_MD.write_text(md + "\n")
    print(f"\nsaved: {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
