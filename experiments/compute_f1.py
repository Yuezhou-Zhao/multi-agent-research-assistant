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
    all_rows = []
    with open(LABELS_PATH) as f:
        for row in csv.DictReader(f):
            all_rows.append(row["gamma_decision"].strip())
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

    # Labeled-batch band distribution (all 65 rows incl. uncertain) — the
    # README's "75.4% on the labeled batch" figure sources from here.
    n_all = len(all_rows)
    band = {b: all_rows.count(b) for b in ("approve", "reject", "escalate")}
    n_resolved_all = band["approve"] + band["reject"]

    # If the measured-L3 run exists, surface its number instead of a bare
    # pointer (measure_l3.py writes cascade_l3_measured.md).
    measured_line = (
        "- Final F1 (measured L3): the actual gpt-4o-mini judge run on the "
        "escalate band — run `python -m experiments.measure_l3` to produce "
        "`cascade_l3_measured.md`.\n"
    )
    l3_md = RESULTS_DIR / "cascade_l3_measured.md"
    if l3_md.exists():
        import re
        m = re.search(r"\*\*measured\*\* L3 \| \*\*([0-9.]+)\*\*", l3_md.read_text())
        if m:
            measured_line = (
                f"- Final F1 (measured L3): **{m.group(1)}** — the actual "
                f"gpt-4o-mini judge on the escalate band; see "
                f"`cascade_l3_measured.md`.\n"
            )

    out = RESULTS_DIR / "cascade_f1.md"
    out.write_text(
        "## Cascade F1 — Section 5.3 (positive class = hallucinated)\n\n"
        f"- Labeled batch: **{n_all}** sentences; band distribution "
        f"approve **{band['approve']}** / reject **{band['reject']}** / "
        f"escalate **{band['escalate']}** → Gamma-only resolve rate "
        f"**{n_resolved_all}/{n_all} = {n_resolved_all / n_all:.1%}** "
        f"(the README's labeled-batch figure; the post-fix self-consistency "
        f"batch resolves 58.3% — see `cascade.md`)\n"
        f"- Binary-F1 rows: **{len(rows)}** "
        f"({sum(1 for r in rows if r['label'] == 'hallucinated')} hallucinated, "
        f"{sum(1 for r in rows if r['label'] == 'correct')} correct; "
        f"`uncertain` dropped)\n"
        f"- Gamma-only F1: **{gamma_f1:.3f}** "
        f"(resolved-band only, n = {len(resolved)})\n"
        f"- Final F1 (oracle L3, upper bound): **{final_f1:.3f}** "
        f"(escalates assumed judged correctly; n = {len(rows)})\n"
        + measured_line
    )
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    compute()
