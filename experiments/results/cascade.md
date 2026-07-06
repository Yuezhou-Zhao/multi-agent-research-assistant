## Cascade Effectiveness — Section 5.3 (n=10)

Total sentences scored: **60**

| Band | Count | Fraction |
|------|------:|---------:|
| approve (SF > 0.25) — Gamma alone | 18 | 30.0% |
| escalate (uncertain, needs LLM judge) | 25 | 41.7% |
| reject (SF < 0.05) — Gamma alone | 17 | 28.3% |

**Gamma-only resolve rate:** 58.3% (Section 5.3 target: ≥ 60%)

**LLM Critic invocation rate (escalate band):** 41.7%

**Final F1 vs Gamma-only F1:** Gamma-only **0.500**, full cascade **0.600** (positive class = hallucinated, human-labeled n=56; see `cascade_f1.md`). The L3 judge on the escalate band lifts F1 by 10 points over Gamma alone.