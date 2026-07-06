# ⚠️  AI-DRAFTED LABELS, PENDING HUMAN REVIEW

## Threshold validation (Section 4.9) — preliminary

Analysis uses `claude_suggested_label` from `experiments/results/cascade_labels_draft.csv`. **These are AI-drafted labels, not human-reviewed.** The number that goes into Section 4.9's final writeup and the README must come from the finalized human-reviewed `cascade_labels.csv`.

### Per-label sim distribution

| label | n | mean | median | min | max | stdev |
|---|---:|---:|---:|---:|---:|---:|
| hallucinated | 36 | 0.673 | 0.675 | 0.508 | 0.848 | 0.084 |
| correct | 12 | 0.780 | 0.794 | 0.650 | 0.889 | 0.082 |
| uncertain | 10 | 0.732 | 0.742 | 0.592 | 0.862 | 0.080 |

### sim histograms per label

```
  hallucinated (n=36):
    [0.40, 0.50):  0
    [0.50, 0.55): ███ 3
    [0.55, 0.60): █████ 5
    [0.60, 0.65): █████ 5
    [0.65, 0.70): █████████ 9
    [0.70, 0.75): ███████ 7
    [0.75, 0.80): █████ 5
    [0.80, 0.85): ██ 2
    [0.85, 0.90):  0
    [0.90, 1.00]:  0
  correct (n=12):
    [0.40, 0.50):  0
    [0.50, 0.55):  0
    [0.55, 0.60):  0
    [0.60, 0.65):  0
    [0.65, 0.70): ███ 3
    [0.70, 0.75): █ 1
    [0.75, 0.80): ██ 2
    [0.80, 0.85): ███ 3
    [0.85, 0.90): ███ 3
    [0.90, 1.00]:  0
  uncertain (n=10):
    [0.40, 0.50):  0
    [0.50, 0.55):  0
    [0.55, 0.60): █ 1
    [0.60, 0.65): █ 1
    [0.65, 0.70): ██ 2
    [0.70, 0.75): ██ 2
    [0.75, 0.80): ██ 2
    [0.80, 0.85): █ 1
    [0.85, 0.90): █ 1
    [0.90, 1.00]:  0
```

### Confusion at current threshold (GROUNDING_THRESHOLD = 0.82)

Positive class = **hallucinated**. Uncertain rows dropped from the confusion matrix (they can't be graded either way).

| | predicted hallucinated | predicted correct |
|---|---:|---:|
| **actual hallucinated** | 35 (TP) | 1 (FN) |
| **actual correct** | 7 (FP) | 5 (TN) |

- precision: **0.833**
- recall: **0.972**
- F1: **0.897**

### Suggested threshold (max F1 on hallucinated class, sweep 0.40 → 0.90)

- best threshold: **0.82**  (F1 = 0.897, precision = 0.833, recall = 0.972)

### Sentences the current threshold (0.82) mis-classifies

- FP (AI-labeled 'correct' but sim < threshold): **7**
- FN (AI-labeled 'hallucinated' but sim ≥ threshold): **1**

Sample FNs (misattributions L2b would let through):
  - q4s3  sim=0.848  cited=['2508.01862v1']
