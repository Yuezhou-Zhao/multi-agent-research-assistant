# Supervisor Distillation — Results

Distilled the production Supervisor's routing call (`gpt-4o-mini`,
`backend/nodes/supervisor.py`) into a **local Qwen2.5-1.5B + LoRA** and
measured it against the same held-out, human-corrected eval set the API model
was scored on. Numbers are measured, not tuned toward a target.

## Head-to-head (n=130 human-corrected eval set)

| Metric | gpt-4o-mini (API baseline) | Qwen2.5-1.5B **zero-shot** (control) | Qwen2.5-1.5B + LoRA (local student) |
|---|---:|---:|---:|
| **route accuracy** | **0.854** | 0.446 | **0.838** |
| use_arxiv accuracy | 0.985 | 0.477 | 0.969 |
| use_web accuracy | 0.869 | 0.600 | 0.869 |
| macro F1 | 0.782 | 0.327 | 0.752 |
| `both`-class F1 | 0.486 | 0.000 | 0.424 |
| **latency mean** | 2.95 s | 0.54 s | **0.97 s** |
| **latency p95** | 5.90 s | 0.58 s | **1.15 s** |
| marginal cost / call | ~$0.000034 (API) | ~$0 (local) | ~$0 (local) |

## Takeaways

- **The fine-tuning did the work — measured, not assumed.** The same base
  model zero-shot scores **0.446** route accuracy with `both`-class F1
  **0.000** (it never predicts the combined route and over-predicts `web`,
  recall 0.97 at precision 0.35). LoRA lifts route accuracy **+39.2 points**
  to 0.838 — from roughly coin-flip to within 1.6 points of the teacher.
  (Zero-shot latency is lower simply because its outputs are shorter and
  often malformed; it is not a real speed advantage.)

- **Near-parity routing at a fraction of the latency.** The 1.5B local student
  lands within **1.6 points** of gpt-4o-mini on route accuracy (0.838 vs
  0.854) while running **~3× faster on mean latency and ~5× faster on p95**
  (0.97 s / 1.15 s local vs 2.95 s / 5.90 s API). For a per-query gating call
  that runs before every retrieval, cutting the tail from ~6 s to ~1 s is the
  meaningful win.
- **The real payoff is latency + independence, not dollars.** At this call size
  gpt-4o-mini is only ~$0.03 / 1,000 calls, so the cost saving is negligible in
  absolute terms — stated plainly rather than dressed up. The engineering wins
  that matter: the routing decision now runs **on-device** (no network round
  trip, no external dependency or rate limit, no query text leaving the
  machine), which is also why the latency and its tail collapse.
- **The student inherits the teacher's blind spot on `both`.** Both models are
  weakest on the `both` route (student F1 0.42, teacher 0.49). That's expected:
  the student distills gpt-4o-mini's labels, and gpt-4o-mini itself
  under-predicts `both` (recall 0.39) — including the exact human-corrected
  cases the eval set reinforces. A distilled student can't exceed its teacher's
  signal on the hardest class. Oversampling lifted `both` **precision**
  (0.64 → 0.70) but not recall; closing the recall gap would need a stronger
  teacher signal on `both` (or human-labeled `both` training data), not more
  duplication of the same 57 examples.

## Method

- **LoRA on fp16 base via MPS — not literal 4-bit QLoRA.** bitsandbytes 4-bit
  is CUDA-only and doesn't run on Apple-Silicon MPS; more fundamentally, a 1.5B
  model in fp16 is ~3 GB and fits trivially in the M5 Pro's 48 GB unified
  memory, so 4-bit quantization (which exists to fit large models onto small
  VRAM) buys nothing here. We use LoRA adapters on the fp16 base — same
  parameter-efficient-fine-tuning idea, right-sized to the hardware. Trained in
  fp32 for MPS stability (no loss scaler on MPS), inference in fp16 for
  realistic local latency.
- **LoRA config:** r=16, α=32, dropout=0.05, targets q/k/v/o + gate/up/down
  proj; 3 epochs, lr 2e-4 cosine, effective batch 16. Loss computed on the
  assistant JSON completion only (prompt tokens masked). Final train loss ~0.04.
- **Class imbalance → oversampling** (see `format_dataset.py` for the full
  rationale): the training routes are arxiv 436 / web 149 / both 57, so `both`
  is ~9%. We oversample minority routes to parity (→ 436 each, 642 → 1308
  examples) rather than a class-weighted loss, because the target is structured
  JSON *generation* and a per-example weight on the token-level LM loss is
  awkward. The multiplier is tunable via `--balance-ratio`.

### Caveats

- The latency comparison is **API (includes network round-trip) vs local
  (on-device MPS)**. That is precisely the deployment question being asked —
  "keep the API call or run locally?" — so it's the right comparison, but the
  API number is not pure model compute.
- Eval labels are qwen-judged + human-corrected on the hard subset (see
  `METHODOLOGY.md`); the training labels are gpt-4o-mini. Student and teacher
  are therefore both scored against an independent oracle, not against the
  teacher's own labels.

## Reproduce

```bash
# data already prepared (train_set.jsonl / eval_set.jsonl committed)
python -m experiments.lora_supervisor.format_dataset          # → train_chat.jsonl (1308)
HF_HUB_OFFLINE=1 python -m experiments.lora_supervisor.train_lora   # → train/adapter (~18 min, MPS)
python -m experiments.lora_supervisor.eval_routing --backend openai --out data/baseline_gpt4omini.json
HF_HUB_OFFLINE=1 python -m experiments.lora_supervisor.eval_routing \
    --backend local --adapter train/adapter --out data/student_qwen1.5b_lora.json
# zero-shot control (bare base model, no adapter):
HF_HUB_OFFLINE=1 python -m experiments.lora_supervisor.eval_routing \
    --backend local --out data/student_qwen1.5b_zeroshot.json
```

## Future work

- **Model-size ablation** (Qwen2.5-0.5B / 3B): 1.5B was chosen as the
  smallest tier expected to reliably emit structured JSON, not ablated.
- **Closing the `both`-class recall gap** needs a stronger teacher signal on
  that class (human-labeled `both` training data), not more oversampling.
