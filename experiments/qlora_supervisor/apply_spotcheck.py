"""Apply human spot-check corrections to the eval set (ROLE 3 follow-up).

Reads the human-reviewed eval_spotcheck.csv and, for every row the human
marked disagree (human_agree_Y_N == N), overwrites that eval_set.jsonl row's
route with human_route_if_diff. Rows marked agree are left as the qwen judge
labeled them but tagged human_verified. Untouched eval rows (not in the
spot-check subset) keep their qwen labels.

Also reports the human/qwen agreement rate — scoped to the spot-check subset
(the hardest / most-ambiguous rows, selected disagreement-first), NOT the
full eval set. That scoping matters: the 20 rows were deliberately chosen to
be the hard cases, so agreement here is a lower bound, not the eval set's
overall label quality.

Idempotent: re-running re-derives corrections from the CSV; already-applied
rows just get re-confirmed.

Run:
  python -m experiments.qlora_supervisor.apply_spotcheck
"""
import csv
import sys

from . import config
from .common import read_jsonl, write_jsonl

# route label → (use_arxiv, use_web)
_ROUTE_FLAGS = {
    "arxiv": (True, False),
    "web": (False, True),
    "both": (True, True),
    "neither": (False, False),
}


def main() -> int:
    if not config.SPOTCHECK_PATH.exists():
        raise SystemExit(f"No spot-check file at {config.SPOTCHECK_PATH}")
    if not config.EVAL_SET_PATH.exists():
        raise SystemExit(f"No eval set at {config.EVAL_SET_PATH}")

    with open(config.SPOTCHECK_PATH) as f:
        spot = list(csv.DictReader(f))

    agree = disagree = blank = 0
    corrections: dict[str, str] = {}   # id -> human route
    verified: set[str] = set()         # ids the human confirmed
    for r in spot:
        v = r["human_agree_Y_N"].strip().upper()
        if v == "Y":
            agree += 1
            verified.add(r["id"])
        elif v == "N":
            disagree += 1
            human_route = r["human_route_if_diff"].strip().lower()
            if human_route not in _ROUTE_FLAGS:
                raise SystemExit(
                    f"Row {r['id']}: human_route_if_diff={human_route!r} not one "
                    f"of {list(_ROUTE_FLAGS)}"
                )
            corrections[r["id"]] = human_route
        else:
            blank += 1

    reviewed = agree + disagree
    if blank:
        print(f"WARNING: {blank} spot-check rows have a blank verdict — skipped")
    if not reviewed:
        raise SystemExit("No Y/N verdicts found in the spot-check file.")

    # Apply to the eval set.
    rows = read_jsonl(config.EVAL_SET_PATH)
    n_applied = 0
    for row in rows:
        rid = row["id"]
        if rid in corrections:
            new_route = corrections[rid]
            ua, uw = _ROUTE_FLAGS[new_route]
            row.setdefault("original_qwen_route", row["route"])
            row["route"] = new_route
            row["use_arxiv"] = ua
            row["use_web"] = uw
            row["human_corrected"] = True
            row["label_source"] = "human_reviewed_override"
            n_applied += 1
        elif rid in verified:
            row["human_verified"] = True
    write_jsonl(config.EVAL_SET_PATH, rows)

    pct = 100.0 * agree / reviewed
    print(f"Spot-check applied to {config.EVAL_SET_PATH.name}")
    print(f"  reviewed subset: {reviewed} rows (the hard/most-ambiguous cases)")
    print(f"  human agreed with qwen: {agree}/{reviewed} ({pct:.0f}%)")
    print(f"  human overrides applied: {n_applied}")
    print(f"  NOTE: {pct:.0f}% is scoped to the disagreement-first hard subset, "
          f"not the full {len(rows)}-row eval set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
