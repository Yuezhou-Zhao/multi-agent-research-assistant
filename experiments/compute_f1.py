"""compute_f1.py — Section 5.3's "Final F1 vs Gamma-only F1" sub-metric.

Reads experiments/results/cascade_labels.csv, which cascade_effectiveness.py
produced with a blank `label` column. Once you fill each row's label as
`correct` or `hallucinated` (blank rows are skipped as unlabeled),
running this script computes:

  Gamma-only F1: treat SF < CERTAIN_WRONG as "predicted hallucinated"
                 and SF > CERTAIN_RIGHT as "predicted correct";
                 escalates are dropped as "no prediction" (Gamma alone
                 doesn't decide those).
  Final F1:      the pipeline's actual decision — same as Gamma for
                 approve/reject sentences, and for escalates we assume
                 L3's binary judge would land on the majority ground
                 truth (this is a portfolio-scale approximation, called
                 out in the README).

Emits results/cascade_f1.md and appends the F1 row into cascade.md.
"""
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.gamma_guardrail import GammaGuardrail

RESULTS_DIR = _REPO_ROOT / "experiments" / "results"
LABELS_PATH = RESULTS_DIR / "cascade_labels.csv"

VALID_LABELS = {"correct", "hallucinated"}


def _binary_f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute():
    rows = []
    with open(LABELS_PATH) as f:
        for row in csv.DictReader(f):
            label = row["label"].strip().lower()
            if label not in VALID_LABELS:
                continue
            rows.append(
                {
                    "sf": float(row["sf"]),
                    "gamma_decision": row["gamma_decision"],
                    "label": label,
                }
            )

    if not rows:
        print(
            f"No labeled rows found in {LABELS_PATH}. "
            f"Fill the `label` column with 'correct' or 'hallucinated' first."
        )
        return

    # Gamma-only: treat approve as predicted-correct, reject as
    # predicted-hallucinated; escalates have no prediction.
    resolved = [r for r in rows if r["gamma_decision"] != "escalate"]
    escalates = [r for r in rows if r["gamma_decision"] == "escalate"]

    # Class of interest for F1 = "hallucinated" (the failure mode we want
    # to catch), so positive class = hallucinated.
    def _score(preds_vs_labels: list[tuple[str, str]]) -> float:
        tp = sum(1 for p, y in preds_vs_labels if p == "hallucinated" and y == "hallucinated")
        fp = sum(1 for p, y in preds_vs_labels if p == "hallucinated" and y == "correct")
        fn = sum(1 for p, y in preds_vs_labels if p == "correct" and y == "hallucinated")
        return _binary_f1(tp, fp, fn)

    gamma_pairs = [
        ("hallucinated" if r["gamma_decision"] == "reject" else "correct", r["label"])
        for r in resolved
    ]
    gamma_f1 = _score(gamma_pairs)

    # "Final F1" = Gamma pairs + (for escalates) assume L3 lands on the
    # true label. Documented in the README as an upper-bound proxy.
    final_pairs = gamma_pairs + [(r["label"], r["label"]) for r in escalates]
    final_f1 = _score(final_pairs)

    print("F1 (positive class = hallucinated):")
    print(f"  Gamma-only F1: {gamma_f1:.3f}  (n_resolved = {len(resolved)})")
    print(f"  Final F1:      {final_f1:.3f}  (n_total = {len(rows)}, n_escalate = {len(escalates)})")

    out = RESULTS_DIR / "cascade_f1.md"
    out.write_text(
        "## Cascade F1 — Section 5.3 (positive class = hallucinated)\n\n"
        f"- Labeled rows: **{len(rows)}** "
        f"({sum(1 for r in rows if r['label'] == 'hallucinated')} hallucinated, "
        f"{sum(1 for r in rows if r['label'] == 'correct')} correct)\n"
        f"- Gamma-only F1: **{gamma_f1:.3f}** "
        f"(resolved-band only, n = {len(resolved)})\n"
        f"- Final F1: **{final_f1:.3f}** "
        f"(escalates given oracle-L3 upper bound; n = {len(rows)})\n"
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    compute()
