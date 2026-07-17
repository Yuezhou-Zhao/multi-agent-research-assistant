# Evaluation results

Full tables and methodology behind the README's headline numbers. Raw
per-query/per-sentence outputs live in
[`experiments/results/`](../experiments/results/); every number here is
reproducible from the scripts in [`experiments/`](../experiments/).

## Cascade effectiveness

Band distribution over a 10-query, 60-sentence self-consistency run of the
shipped system:

| Band | Count | Fraction |
|---|---:|---:|
| approve (guardrail alone) | 18 | 30.0% |
| escalate (needs LLM judge) | 25 | 41.7% |
| reject (guardrail alone) | 17 | 28.3% |

**Zero-LLM resolve rate: 58.3%** — the free layers settle nearly 6 of every
10 sentences, so the LLM judge is invoked on ~42%. The design target was
≥60%; 58.3% sits just under it. Full breakdown:
[`cascade.md`](../experiments/results/cascade.md).

## Hallucination-detection F1

All 65 sentences of a labeled batch were hand-labeled
`correct` / `hallucinated` / `uncertain`; positive class = hallucinated:

| | F1 | Note |
|---|---:|---|
| Guardrail alone (L1) | 0.500 | 19 hallucinated sentences look distributionally normal and slip into `approve` — the Semantic Illusion |
| Full cascade, oracle L3 (upper bound) | 0.600 | assumes a perfect judge on the escalate band |
| Full cascade, **measured L3** | **0.562** | the real gpt-4o-mini judge: catches all 6 hallucinated escalate-band sentences (recall 1.0) but over-rejects 4 correct ones (precision 0.60) |

The measured judge is conservative on the hardest band — it catches every
hallucination but rejects some correct citations, which is why it lands
just below the oracle ceiling. Small-n caveat: only 12 gradable
escalate-band sentences, so the L3 row is directional. Breakdowns:
[`cascade_f1.md`](../experiments/results/cascade_f1.md),
[`cascade_l3_measured.md`](../experiments/results/cascade_l3_measured.md).

### Why the sentence counts differ across tables

**65** = the human-labeled batch (all correctness metrics). **58** = the 65
that carry a citation (the grounding check needs a cited chunk). **56** =
the 65 minus 9 `uncertain` labels (binary F1 needs a decisive label). The
**60** in the band table is a separate, unlabeled re-run taken *after* the
Writer-prompt hardening, which made the Writer more cautious (more
`escalate`) — that's why its resolve rate (58.3%) sits below the labeled
batch's (75.4%, 49/65). Band distribution = the shipped post-fix system;
correctness metrics = the labeled pre-fix batch.

## Citation misattribution and the L2b grounding check

A labeling exercise surfaced a failure mode the structural citation check
(L2a) cannot catch: in **36/65** labeled sentences the `[N]` marker was
structurally valid and resolved to a real paper, but that paper's content
didn't support the claim. The Writer was describing well-known methods from
pretrained memory and attaching whichever nearby chunk had a plausible
title (e.g. describing FG-PRM but citing the REFIND paper).

The fix is two-part and zero-LLM: a hardened Writer grounding prompt, plus
the **L2b embedding grounding check** —
`cosine(encode(sentence), centroid(encode(cited chunks)))`, downgrading the
cascade decision one step when a sentence isn't supported by its cited
chunk. Against the human labels it flags misattributed citations at
**precision 0.89 / recall 0.62 / F1 0.73** (threshold 0.70).

Known ceiling: embedding similarity cannot separate a correctly-cited
same-subfield paper from a wrongly-cited one — both sit in the same
embedding neighborhood. That residual class needs NLI or an LLM judge.

### Choosing the grounding threshold

Swept against the labeled set, **0.82 was F1-optimal** (recall 0.97,
F1 0.89). It was also operationally unviable: 0.82 sits *above* the
correct-citation similarity mean (0.779), so it flagged ~88% of all
sentences — the self-correction loop never converged and most queries
force-finalized, non-deterministically (the same query approved on one run
and force-finalized the next). The dry-run harness
([`scripts/demo_dryrun.py`](../scripts/demo_dryrun.py)) caught this; the
labeled metric alone did not.

The sweep was then redone with an explicit operability column (`%flag` —
how much of a real draft gets downgraded). **0.70 shipped**: it sits
between the class means, flags ~50% of sentences, and keeps the correction
loop convergent — trading recall (0.62 vs 0.97) for a system that
actually converges. Full sweep:
[`threshold_validation.md`](../experiments/results/threshold_validation.md).

## HyDE A/B — a negative result (n = 10 queries)

Each query run twice — HyDE off vs. on — everything else fixed:

|                    | HyDE off | HyDE on |
|--------------------|:--------:|:-------:|
| mean rollbacks     |   1.40   |   1.20  |
| mean chunk SF      |   0.643  |  0.643  |
| mean rerank score  |   2.313  |  2.361  |
| approved / n       |   9/10   |  9/10   |

HyDE did not measurably help on this corpus. Retrieval-quality metrics are
flat across 8 of 10 queries, and the entire aggregate rollback improvement
traces to a single query (Q2: 3 → 1 rollbacks). It stays enabled by default
because it does no measurable harm and costs one cached LLM call (never
re-invoked on rollback). The A/B harness
([`hyde_ab.py`](../experiments/hyde_ab.py)) exists to re-test at larger *n*
before making any stronger claim. Per-query table:
[`hyde_ab.md`](../experiments/results/hyde_ab.md).

## Router distillation (LoRA)

The Supervisor's routing call (gpt-4o-mini, decides arXiv / web / both)
runs before every retrieval. It was distilled into a local Qwen2.5-1.5B +
LoRA and measured against the same 130-row held-out eval set as the API
model (labels judged by an independent model, then human-corrected on the
hard cases — methodology in
[`lora_supervisor/METHODOLOGY.md`](../experiments/lora_supervisor/METHODOLOGY.md)):

| Metric | gpt-4o-mini (API) | Qwen2.5-1.5B zero-shot (control) | + LoRA (local) |
|---|---:|---:|---:|
| route accuracy | 0.854 | 0.446 | **0.838** (−1.6 pt) |
| macro F1 | 0.782 | 0.327 | 0.752 |
| `both`-class F1 | 0.486 | 0.000 | 0.424 |
| latency mean | 2.95 s | 0.54 s | **0.97 s** (~3×) |
| latency p95 | 5.90 s | 0.58 s | **1.15 s** (~5×) |
| marginal cost / call | ~$0.000034 | ~$0 | ~$0 |

Reading the whole row:

- The zero-shot control is what proves the fine-tuning mattered: the bare
  base model routes at 0.446 and never predicts `both` at all; LoRA lifts
  it +39.2 points, to within 1.6 points of the teacher.
- The student inherits the teacher's blind spot on the `both` class
  (F1 0.42 vs the teacher's own weak 0.49) — a distilled model can't exceed
  its teacher's signal on the hardest class; oversampling lifted `both`
  precision but not recall.
- The dollar saving is negligible at this call volume (~$0.03 / 1,000
  calls); the real wins are latency and independence — no network round
  trip, no query text leaving the machine.
- The latency comparison is API-with-network vs. local MPS inference —
  which is exactly the deployment question being asked, but worth stating.
- It's LoRA (adapters on an fp16 base), not 4-bit QLoRA: bitsandbytes 4-bit
  is CUDA-only, and a 1.5B model in fp16 (~3 GB) fits easily in unified
  memory.

Full write-up and class-imbalance handling:
[`lora_supervisor/RESULTS.md`](../experiments/lora_supervisor/RESULTS.md).

## Guardrail framing and measured latency

The Gamma guardrail is a calibrated embedding-distance prefilter — a cheap
way to score "does this text look like the trustworthy academic text we
calibrated on." It is not a statistical guarantee of correctness, and it
performs on par with simpler baselines (raw cosine distance / empirical
percentile); the Gamma modeling buys a smooth, thresholdable score.

Two separate calibrations are used — raw arXiv abstracts for chunk
filtering, Writer-style sentence exemplars for draft scoring — because the
two inputs occupy different embedding distributions (a pooled calibration
rejected 68% of Writer sentences and approved 0/5 queries in a diagnostic
batch).

Measured on 438 real arXiv abstracts: the survival-function math itself is
~0.006 ms/sentence given an embedding; full embed+score is
~2.7 ms/sentence, dominated by the BGE-small forward pass.

## Open follow-ups

- **HyDE at scale** — rerun the A/B at n=50 to determine whether the
  single-query effect survives a larger sample.
- **Larger L3 ground truth** — extend
  [`measure_l3.py`](../experiments/measure_l3.py) beyond 12 gradable
  escalate-band sentences so the judge's reliability becomes a stable
  estimate.
- **NLI-based grounding** — replace L2b's cosine check with an entailment
  model to attack the same-subfield misattribution class that embedding
  similarity cannot separate.
