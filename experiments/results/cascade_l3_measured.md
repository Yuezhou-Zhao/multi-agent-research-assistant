## Cascade F1 — measured L3 judge

Escalate band: **16** sentences (12 gradable correct/hallucinated, 4 uncertain dropped).

### L3 judge (gpt-4o-mini) standalone on the escalate band
- accuracy on gradable escalate sentences: **8/12**
- precision 0.600 / recall 1.000 / F1 **0.750** (positive = hallucinated; TP 6 FP 4 FN 0 TN 2)

### Full cascade F1 (positive = hallucinated)
| variant | F1 |
|---|---|
| Gamma-only (L1) | 0.500 |
| Full cascade, **oracle** L3 (upper bound) | 0.600 |
| Full cascade, **measured** L3 | **0.562** |

Measured cascade: precision 0.667 / recall 0.486 / F1 **0.562** (n=56).

**What this does and doesn't show.** The measured number is the honest end-to-end cascade F1 — no oracle. The gap from the 0.600 upper bound is exactly how often the real gpt-4o-mini judge disagrees with a human on the hardest (escalate-band) sentences, the ones L1/L2 could not resolve. It is a small-n figure (escalate band is only 12 gradable sentences), so treat it as directional, not a precise operating point.
