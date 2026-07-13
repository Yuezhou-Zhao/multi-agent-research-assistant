# FINAL — human-reviewed labels

## Threshold validation (Section 4.9) — FINAL

Analysis uses the human-reviewed `label` column from `experiments/results/cascade_labels.csv`. **These are the finalized labels** — the numbers here are the ones that go into Section 4.9's final writeup and the README.

### Per-label sim distribution

*Rows here total **58**, not 65: these are the citation-carrying subset of
the 65 labeled sentences (only cited sentences can be grounding-scored).
All 37 hallucinated sentences carry citations; 6 `correct` and 1 `uncertain`
do not — hence 13 correct / 8 uncertain below vs. the full batch's 19 / 9.*

| label | n | mean | median | min | max | stdev |
|---|---:|---:|---:|---:|---:|---:|
| hallucinated | 37 | 0.674 | 0.676 | 0.508 | 0.848 | 0.083 |
| correct | 13 | 0.779 | 0.782 | 0.650 | 0.889 | 0.079 |
| uncertain | 8 | 0.733 | 0.742 | 0.592 | 0.862 | 0.089 |

### sim histograms per label

```
  hallucinated (n=37):
    [0.40, 0.50):  0
    [0.50, 0.55): ███ 3
    [0.55, 0.60): █████ 5
    [0.60, 0.65): ████ 4
    [0.65, 0.70): ███████████ 11
    [0.70, 0.75): ███████ 7
    [0.75, 0.80): █████ 5
    [0.80, 0.85): ██ 2
    [0.85, 0.90):  0
    [0.90, 1.00]:  0
  correct (n=13):
    [0.40, 0.50):  0
    [0.50, 0.55):  0
    [0.55, 0.60):  0
    [0.60, 0.65):  0
    [0.65, 0.70): ███ 3
    [0.70, 0.75): █ 1
    [0.75, 0.80): ███ 3
    [0.80, 0.85): ███ 3
    [0.85, 0.90): ███ 3
    [0.90, 1.00]:  0
  uncertain (n=8):
    [0.40, 0.50):  0
    [0.50, 0.55):  0
    [0.55, 0.60): █ 1
    [0.60, 0.65): █ 1
    [0.65, 0.70): █ 1
    [0.70, 0.75): ██ 2
    [0.75, 0.80): █ 1
    [0.80, 0.85): █ 1
    [0.85, 0.90): █ 1
    [0.90, 1.00]:  0
```

### Confusion at current threshold (GROUNDING_THRESHOLD = 0.7)

Positive class = **hallucinated**. Uncertain rows dropped from the confusion matrix (they can't be graded either way).

| | predicted hallucinated | predicted correct |
|---|---:|---:|
| **actual hallucinated** | 23 (TP) | 14 (FN) |
| **actual correct** | 3 (FP) | 10 (TN) |

- precision: **0.885**
- recall: **0.622**
- F1: **0.730**

### Suggested threshold (max F1 on hallucinated class, sweep 0.40 → 0.90)

- best threshold: **0.82**  (F1 = 0.889, precision = 0.818, recall = 0.973)

### Sentences the current threshold (0.7) mis-classifies

- FP (labeled 'correct' but sim < threshold): **3**
- FN (labeled 'hallucinated' but sim ≥ threshold): **14**

Sample FNs (misattributions L2b would let through):
  - q1s2  sim=0.762  cited=['2403.15450v1', '2602.07739v2']
  - q1s4  sim=0.723  cited=['2207.03030v1', '2606.28358v1']
  - q2s3  sim=0.725  cited=['2604.10697v1', '2311.04589v3']
  - q2s5  sim=0.736  cited=['2312.12141v4']
  - q3s3  sim=0.736  cited=['2406.09136v2']
  - q3s6  sim=0.709  cited=['2506.03673v1']
  - q4s3  sim=0.848  cited=['2508.01862v1']
  - q5s6  sim=0.788  cited=['2108.06279v2', '2601.06196v3']
