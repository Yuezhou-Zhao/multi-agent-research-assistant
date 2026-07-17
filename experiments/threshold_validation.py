"""Threshold validation for the L2b citation-grounding check.

Sweeps the GROUNDING_THRESHOLD in evaluation/citation_grounding.py against
labeled cascade sentences: does the embedding-grounding similarity actually
separate hallucinated (misattributed) citations from correct ones, and what
threshold maximizes F1?

Labels: the human-reviewed `label` column in
experiments/results/cascade_labels.csv — these are the numbers that go
into the README.

Method:
  1. Parse [XXXX.XXXXXvY] arxiv ids out of each labeled sentence.
  2. Look up each cited paper's title+summary from the FAISS index metadata.
  3. sim = cosine(encode(sentence), centroid(encode(cited_chunks))).
  4. Bucket sims by the chosen label column; report per-bucket distribution,
     confusion at the current GROUNDING_THRESHOLD, and the F1-optimal sweep.
  5. Save results/threshold_validation.md + .json.
"""
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rag.indexer import load_index

RESULTS_DIR = _REPO_ROOT / "experiments" / "results"

# Matches arxiv ids inside brackets: "[2403.15450v1]", "[2504.10529v1]".
_ARXIV_ID_RE = re.compile(r"\[(\d{4}\.\d{4,5}(?:v\d+)?)\]")

# Set in main() from CLI args (default: the human-reviewed final labels).
LABEL_COL = "label"
LABELS_PATH = RESULTS_DIR / "cascade_labels.csv"
OUT_MD = RESULTS_DIR / "threshold_validation.md"
OUT_JSON = RESULTS_DIR / "threshold_validation.json"
BANNER = "FINAL — human-reviewed labels"


def load_labeled_rows():
    # utf-8-sig tolerates a BOM; the file is normalized UTF-8.
    with open(LABELS_PATH, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def lookup_chunks(paper_by_id: dict, cited_ids: list[str]) -> list[dict]:
    """Return [{id, content}, ...] for each cited id that resolves to a
    paper in the indexed corpus. IDs that miss the index (rare — the
    corpus was built July 2 and the draft labels reference IDs from
    that corpus) are skipped and reported."""
    out = []
    for cid in cited_ids:
        if cid in paper_by_id:
            paper = paper_by_id[cid]
            content = f"{paper.get('title', '')} — {paper.get('summary', '')}"
            out.append({"id": cid, "content": content})
    return out


def cosine_sim(encoder: SentenceTransformer, sentence: str, chunks: list[dict]) -> float:
    """Sentence embedding vs mean-of-cited-chunks embedding (same as
    evaluation/citation_grounding.check_citation_grounding)."""
    s_emb = encoder.encode([sentence], normalize_embeddings=True)[0]
    c_embs = encoder.encode(
        [c["content"] for c in chunks], normalize_embeddings=True
    )
    centroid = c_embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
    return float(s_emb @ centroid)


def describe(sims: list[float]) -> dict:
    if not sims:
        return {"n": 0}
    return {
        "n": len(sims),
        "mean": statistics.mean(sims),
        "median": statistics.median(sims),
        "min": min(sims),
        "max": max(sims),
        "stdev": statistics.stdev(sims) if len(sims) > 1 else 0.0,
    }


def _print_sim_hist(sims: list[float], label: str, buckets: list[float]) -> str:
    """Text histogram of sims within specific ranges. Compact enough
    to sit inside the Markdown output."""
    if not sims:
        return f"  {label}: (none)"
    lines = [f"  {label} (n={len(sims)}):"]
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        n = sum(1 for s in sims if lo <= s < hi)
        bar = "█" * n
        lines.append(f"    [{lo:.2f}, {hi:.2f}): {bar} {n}")
    # Above the last bucket
    n_over = sum(1 for s in sims if s >= buckets[-1])
    lines.append(f"    [{buckets[-1]:.2f}, 1.00]: {'█' * n_over} {n_over}")
    return "\n".join(lines)


def _confusion_at_threshold(rows_with_sims: list[dict], threshold: float) -> dict:
    """For threshold T, an L2b system that downgrades sim < T (i.e.
    predicts 'hallucinated' for sim < T, 'correct' for sim >= T).
    Compute TP/FP/FN/TN vs the human-reviewed label."""
    tp = fp = fn = tn = 0
    for r in rows_with_sims:
        if r["sim"] is None:
            continue
        label = r["label"]
        predicted_hallucinated = r["sim"] < threshold
        if label == "hallucinated":
            if predicted_hallucinated:
                tp += 1
            else:
                fn += 1
        elif label == "correct":
            if predicted_hallucinated:
                fp += 1
            else:
                tn += 1
        # 'uncertain' rows are dropped from the confusion matrix; they
        # cannot be graded either way.
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _suggest_threshold(rows_with_sims: list[dict]) -> tuple[float, dict]:
    """Scan thresholds in 0.02 steps between 0.4 and 0.9, pick the one
    with highest F1 for the hallucinated class."""
    best = None
    for t in np.arange(0.40, 0.9001, 0.02):
        cm = _confusion_at_threshold(rows_with_sims, float(t))
        if best is None or cm["f1"] > best["f1"]:
            best = cm
    return float(best["threshold"]), best


def main() -> int:
    global LABEL_COL, LABELS_PATH, OUT_MD, OUT_JSON, BANNER
    LABEL_COL = "label"
    LABELS_PATH = RESULTS_DIR / "cascade_labels.csv"
    OUT_MD = RESULTS_DIR / "threshold_validation.md"
    OUT_JSON = RESULTS_DIR / "threshold_validation.json"
    BANNER = "FINAL — human-reviewed labels"

    encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    _, metadata = load_index()
    paper_by_id: dict = metadata["paper_by_id"]

    labeled = load_labeled_rows()
    print(f"\n{BANNER}\n")
    print(f"Loaded {len(labeled)} labeled rows from {LABELS_PATH.name}")

    rows_with_sims = []
    skipped_no_citation = 0
    skipped_unresolved = 0
    for row in labeled:
        sentence = row["sentence"]
        cited_ids = _ARXIV_ID_RE.findall(sentence)
        if not cited_ids:
            skipped_no_citation += 1
            rows_with_sims.append(
                {
                    "query_idx": int(row["query_idx"]),
                    "sentence_idx": int(row["sentence_idx"]),
                    "label": row[LABEL_COL],
                    "cited_ids": [],
                    "sim": None,
                    "skipped_reason": "no citation in sentence",
                }
            )
            continue
        chunks = lookup_chunks(paper_by_id, cited_ids)
        if not chunks:
            skipped_unresolved += 1
            rows_with_sims.append(
                {
                    "query_idx": int(row["query_idx"]),
                    "sentence_idx": int(row["sentence_idx"]),
                    "label": row[LABEL_COL],
                    "cited_ids": cited_ids,
                    "sim": None,
                    "skipped_reason": "no cited id resolves to indexed paper",
                }
            )
            continue
        sim = cosine_sim(encoder, sentence, chunks)
        rows_with_sims.append(
            {
                "query_idx": int(row["query_idx"]),
                "sentence_idx": int(row["sentence_idx"]),
                "sentence": sentence,
                "label": row[LABEL_COL],
                "cited_ids": cited_ids,
                "resolved_ids": [c["id"] for c in chunks],
                "sim": sim,
            }
        )

    print(
        f"  scored={sum(1 for r in rows_with_sims if r['sim'] is not None)}  "
        f"skipped(no citation)={skipped_no_citation}  "
        f"skipped(unresolved)={skipped_unresolved}"
    )

    # Bucket sims by AI-drafted label
    by_label = defaultdict(list)
    for r in rows_with_sims:
        if r["sim"] is None:
            continue
        by_label[r["label"]].append(r["sim"])

    labels_of_interest = ["hallucinated", "correct", "uncertain"]
    stats_per_label = {lbl: describe(by_label[lbl]) for lbl in labels_of_interest}

    # Confusion matrix at the current threshold
    from evaluation.citation_grounding import GROUNDING_THRESHOLD

    cm_current = _confusion_at_threshold(rows_with_sims, GROUNDING_THRESHOLD)
    best_t, cm_best = _suggest_threshold(rows_with_sims)

    # ── Console + markdown output ──────────────────────────────────
    intro = (
        "Analysis uses the human-reviewed `label` column from "
        "`experiments/results/cascade_labels.csv`. **These are the "
        "finalized labels** — the numbers here are the ones that go into "
        "the README."
    )
    md_lines = [
        f"# {BANNER}",
        "",
        "## Threshold validation — FINAL",
        "",
        intro,
        "",
        "### Per-label sim distribution",
        "",
        "| label | n | mean | median | min | max | stdev |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lbl in labels_of_interest:
        s = stats_per_label[lbl]
        if s["n"] == 0:
            md_lines.append(f"| {lbl} | 0 | — | — | — | — | — |")
        else:
            md_lines.append(
                f"| {lbl} | {s['n']} | {s['mean']:.3f} | {s['median']:.3f} | "
                f"{s['min']:.3f} | {s['max']:.3f} | {s['stdev']:.3f} |"
            )
    md_lines.append("")

    # Histograms
    md_lines.append("### sim histograms per label\n")
    md_lines.append("```")
    for lbl in labels_of_interest:
        md_lines.append(
            _print_sim_hist(
                by_label[lbl],
                lbl,
                buckets=[0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
            )
        )
    md_lines.append("```\n")

    # Current threshold report
    md_lines.append(
        f"### Confusion at current threshold (GROUNDING_THRESHOLD = "
        f"{GROUNDING_THRESHOLD})\n"
    )
    md_lines.append(
        "Positive class = **hallucinated**. Uncertain rows dropped from "
        "the confusion matrix (they can't be graded either way).\n"
    )
    md_lines.append(
        "| | predicted hallucinated | predicted correct |\n"
        "|---|---:|---:|\n"
        f"| **actual hallucinated** | {cm_current['tp']} (TP) | "
        f"{cm_current['fn']} (FN) |\n"
        f"| **actual correct** | {cm_current['fp']} (FP) | "
        f"{cm_current['tn']} (TN) |\n"
    )
    md_lines.append(
        f"- precision: **{cm_current['precision']:.3f}**\n"
        f"- recall: **{cm_current['recall']:.3f}**\n"
        f"- F1: **{cm_current['f1']:.3f}**\n"
    )

    # Best-threshold suggestion
    md_lines.append(
        f"### Suggested threshold (max F1 on hallucinated class, sweep 0.40 → 0.90)\n"
    )
    md_lines.append(
        f"- best threshold: **{best_t:.2f}**  "
        f"(F1 = {cm_best['f1']:.3f}, precision = {cm_best['precision']:.3f}, "
        f"recall = {cm_best['recall']:.3f})\n"
    )

    # Failure modes at current threshold
    md_lines.append(
        f"### Sentences the current threshold ({GROUNDING_THRESHOLD}) mis-classifies\n"
    )
    fps = [
        r for r in rows_with_sims
        if r["sim"] is not None and r["label"] == "correct"
        and r["sim"] < GROUNDING_THRESHOLD
    ]
    fns = [
        r for r in rows_with_sims
        if r["sim"] is not None and r["label"] == "hallucinated"
        and r["sim"] >= GROUNDING_THRESHOLD
    ]
    md_lines.append(f"- FP (AI-labeled 'correct' but sim < threshold): **{len(fps)}**")
    md_lines.append(f"- FN (AI-labeled 'hallucinated' but sim ≥ threshold): **{len(fns)}**")
    md_lines.append("")
    if fns:
        md_lines.append("Sample FNs (misattributions L2b would let through):")
        for r in fns[:8]:
            md_lines.append(
                f"  - q{r['query_idx']}s{r['sentence_idx']}  sim={r['sim']:.3f}  "
                f"cited={r['cited_ids']}"
            )

    md = "\n".join(md_lines)
    print(md)

    OUT_MD.write_text(md + "\n")
    OUT_JSON.write_text(
        json.dumps(
            {
                "banner": BANNER,
                "current_threshold": GROUNDING_THRESHOLD,
                "current": cm_current,
                "best": {"threshold": best_t, **cm_best},
                "per_label_stats": stats_per_label,
                "rows": rows_with_sims,
            },
            indent=2,
        )
    )
    print(f"\nsaved: {OUT_MD}")
    print(f"saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
