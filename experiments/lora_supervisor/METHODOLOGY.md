# Labeling Methodology & Routing Decision Rule

Records how the Supervisor-distillation datasets were labeled, the routing
boundary used to resolve ambiguous cases, and the human spot-check result.
See `README.md` for the three-role pipeline mechanics.

## Routing decision rule

One rule, applied in both directions — route on **what is actually needed to
answer the specific question**, not on whether domain-sounding vocabulary
appears in it:

> **Route `both`** when a query asks whether a theoretical / foundational
> method *holds up in current, practical, or deployed real-world use* — arXiv
> describes what was *proposed*, not whether it is *in production*, so
> verifying present-day status needs the web channel too.
> **Otherwise route to whichever single channel's content actually answers
> the specific question asked**, regardless of whether academic vocabulary
> appears in the query.

Two diagnostic questions that operationalize it:

1. *"Does this query need to verify current / practical / deployed status,
   separate from what the literature proposed?"* → if yes, **include web**.
2. *"Does this channel's content actually contribute to the answer?"* → if
   no, **drop it**, even when the topic sounds academic.

This is deliberately a content rule, not a keyword rule: a query full of
academic terms ("sparse MoE architectures", "metric-learning losses") still
routes `both` if it's really asking how the proposed idea fares in practice,
and a domain-heavy query routes to a single channel when only that channel's
content answers it.

## Human spot-check result

Of the 130-row held-out eval set, **20 rows** were flagged for human review —
selected disagreement-first (qwen's route ≠ the query's intended generation
category), i.e. deliberately the **hardest / most-ambiguous** cases.

- **Human agreement with the qwen judge: 14/20 (70%)** on this hard subset.
- This 70% is **scoped to the disagreement-first hard subset**, and is a
  lower bound on eval-set label quality — NOT the agreement rate over the
  full 130-row eval set. The 110 unreviewed rows were the ones qwen labeled
  in line with generation intent, where agreement is expected to be higher.
- All 6 disagreements were resolved by the routing rule above; every one was
  a boundary case between `arxiv` and `both`, plus a single `both`→`web`.

### Corrections applied (`apply_spotcheck.py`)

| id | qwen | human | why (rule) |
|---|---|---|---|
| eval-000105 | both | **web** | commercial-use / licensing status of video-gen models — practical, arXiv doesn't answer it; the academic channel adds nothing |
| eval-000117 | arxiv | **both** | whether recent RAG systems overcome a proposed limitation in practice |
| eval-000121 | arxiv | **both** | how sparse-MoE shifted from theory to deployed use |
| eval-000132 | arxiv | **both** | relationship of original metric-learning losses to their current use |
| eval-000134 | arxiv | **both** | extent modern semantic parsers actually use formal grammars |
| eval-000140 | arxiv | **both** | whether econometric causal-inference methods translate to real deployment |

The eval set is therefore **human-corrected on the flagged disagreement
cases**, not purely qwen-labeled. Each corrected row carries
`human_corrected: true` and `original_qwen_route` for provenance; agreed rows
carry `human_verified: true`.

## Why the eval labeler differs from the training labeler

Training labels come from **gpt-4o-mini** (the production Supervisor prompt,
imported not copied). Eval labels come from **qwen3.5-397b-a17b** applying the
identical rubric, then human-corrected on the hard cases. Because the eval
judge is a different model from the one the student distills, eval accuracy
measures generalization to an independent (and, on the hard subset, human-
verified) oracle — not memorization of gpt-4o-mini's quirks.
