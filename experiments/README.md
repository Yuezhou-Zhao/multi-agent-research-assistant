# Experiments

Scripts and outputs behind the numbers in the top-level [README](../README.md)
and [docs/results.md](../docs/results.md). Each script is runnable standalone
and regenerates its corresponding file in [`results/`](results/).

| Script | Measures |
|---|---|
| [`cascade_effectiveness.py`](cascade_effectiveness.py) | Per-band resolve rate of the verification cascade (zero-LLM vs. escalated) |
| [`compute_f1.py`](compute_f1.py) | Hallucination-detection F1 from the human-labeled sentence batch |
| [`measure_l3.py`](measure_l3.py) | The real LLM-judge (L3) accuracy on the escalate band |
| [`threshold_validation.py`](threshold_validation.py) | Sweeps the L2b grounding threshold; the source of the 0.70-vs-0.82 decision |
| [`hyde_ab.py`](hyde_ab.py) | HyDE on/off A/B comparison |

[`results/`](results/) holds each script's output (Markdown + JSON, and the
underlying labeled CSV). [`lora_supervisor/`](lora_supervisor/) is a
self-contained sub-project: distilling the Supervisor's routing call into a
local Qwen2.5-1.5B + LoRA model — see its own
[README](lora_supervisor/README.md).
