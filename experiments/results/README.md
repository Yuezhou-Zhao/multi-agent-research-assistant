# Results

Generated output from the scripts in [`experiments/`](../). Each experiment
has a `.md` (human-readable) and, where applicable, a `.json` (raw numbers)
version.

| File | From | Content |
|---|---|---|
| `cascade.md` / `cascade.json` | `cascade_effectiveness.py` | Per-band cascade resolve rates |
| `cascade_f1.md` | `compute_f1.py` | Hallucination-detection F1 (Gamma alone vs. full cascade) |
| `cascade_l3_measured.md` | `measure_l3.py` | Cascade F1 with the real (not oracle) LLM judge |
| `cascade_labels.csv` | — | The human-reviewed sentence labels every F1 number above is computed from |
| `threshold_validation.md` / `.json` | `threshold_validation.py` | The L2b grounding-threshold sweep (0.70 vs. 0.82) |
| `hyde_ab.md` / `.json` | `hyde_ab.py` | HyDE on/off A/B comparison |

Full narrative write-up, with the reasoning behind each number:
[docs/results.md](../../docs/results.md).
