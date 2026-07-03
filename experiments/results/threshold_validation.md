# ⚠️  AI-DRAFTED LABELS, PENDING HUMAN REVIEW

## Preliminary threshold validation (Section 4.9)

Analysis uses `claude_suggested_label` from `experiments/results/cascade_labels_draft.csv`. **These are AI-drafted labels, not human-reviewed.** The number that goes into Section 4.9's final writeup and the README must come from the finalized human-reviewed `cascade_labels.csv`.

### Per-label sim distribution

| AI label | n | mean | median | min | max | stdev |
|---|---:|---:|---:|---:|---:|---:|
| hallucinated | 36 | 0.673 | 0.675 | 0.508 | 0.848 | 0.084 |
| correct | 12 | 0.780 | 0.794 | 0.650 | 0.889 | 0.082 |
| uncertain | 10 | 0.732 | 0.742 | 0.592 | 0.862 | 0.080 |

### sim histograms per AI label

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

### Confusion at current threshold (GROUNDING_THRESHOLD = 0.65)

Positive class = **hallucinated**. Uncertain rows dropped from the confusion matrix (they can't be graded either way).

| | predicted hallucinated | predicted correct |
|---|---:|---:|
| **actual hallucinated** | 13 (TP) | 23 (FN) |
| **actual correct** | 0 (FP) | 12 (TN) |

- precision: **1.000**
- recall: **0.361**
- F1: **0.531**

### Suggested threshold (max F1 on hallucinated class, sweep 0.40 → 0.90)

- best threshold: **0.82**  (F1 = 0.897, precision = 0.833, recall = 0.972)

### Sentences the current threshold (0.65) mis-classifies

- FP (AI-labeled 'correct' but sim < threshold): **0**
- FN (AI-labeled 'hallucinated' but sim ≥ threshold): **23**

Sample FNs (misattributions L2b would let through):
  - q1s2  sim=0.762  cited=['2403.15450v1', '2602.07739v2']
  - q1s4  sim=0.723  cited=['2207.03030v1', '2606.28358v1']
  - q2s3  sim=0.725  cited=['2604.10697v1', '2311.04589v3']
  - q2s4  sim=0.700  cited=['2501.09997v3']
  - q2s5  sim=0.736  cited=['2312.12141v4']
  - q3s3  sim=0.736  cited=['2406.09136v2']
  - q3s4  sim=0.657  cited=['2502.12134v2']
  - q3s6  sim=0.709  cited=['2506.03673v1']
