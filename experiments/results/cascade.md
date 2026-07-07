## Cascade Effectiveness — Section 5.3 (n=10)

Total sentences scored: **60**

| Band | Count | Fraction |
|------|------:|---------:|
| approve (SF > 0.25) — Gamma alone | 18 | 30.0% |
| escalate (uncertain, needs LLM judge) | 25 | 41.7% |
| reject (SF < 0.05) — Gamma alone | 17 | 28.3% |

**Gamma-only resolve rate:** 58.3% (Section 5.3 target: ≥ 60%)

**LLM Critic invocation rate (escalate band):** 41.7%

**Final F1 vs Gamma-only F1** (positive class = hallucinated, human-labeled n=56): Gamma-only **0.500** → full cascade **0.600** with an *oracle* L3 (upper bound, assumes every escalate judged perfectly) → full cascade **0.562** with the *measured* real gpt-4o-mini judge on the escalate band (`cascade_l3_measured.md`). The measured judge catches all 6 hallucinated escalates (recall 1.0) but over-rejects 4 correct ones, landing just below the oracle.

*Note: this n=60 self-consistency batch was collected on the shipped post-4.9 system (after the writer-prompt hardening); the human-labeled F1 batch (n=65) predates that change — see the README's sentence-count note.*