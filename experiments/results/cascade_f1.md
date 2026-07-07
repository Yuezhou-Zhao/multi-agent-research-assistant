## Cascade F1 — Section 5.3 (positive class = hallucinated)

- Labeled rows: **56** (37 hallucinated, 19 correct)
- Gamma-only F1: **0.500** (resolved-band only, n = 44)
- Final F1 (oracle L3, upper bound): **0.600** (escalates assumed judged correctly; n = 56)
- Final F1 (measured L3): the actual gpt-4o-mini judge run on the escalate band — see `cascade_l3_measured.md` (`python -m experiments.measure_l3`).
