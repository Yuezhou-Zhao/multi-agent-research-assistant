## Cascade F1 — Section 5.3 (positive class = hallucinated)

- Labeled batch: **65** sentences; band distribution approve **28** / reject **21** / escalate **16** → Gamma-only resolve rate **49/65 = 75.4%** (the README's labeled-batch figure; the post-fix self-consistency batch resolves 58.3% — see `cascade.md`)
- Binary-F1 rows: **56** (37 hallucinated, 19 correct; `uncertain` dropped)
- Gamma-only F1: **0.500** (resolved-band only, n = 44)
- Final F1 (oracle L3, upper bound): **0.600** (escalates assumed judged correctly; n = 56)
- Final F1 (measured L3): **0.562** — the actual gpt-4o-mini judge on the escalate band; see `cascade_l3_measured.md`.
