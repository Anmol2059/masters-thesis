# experiments/

One script per experiment. All write JSON results to `results/`.

| Script | Experiment | What it tests |
|--------|-----------|---------------|
| `run_asr.py` | 1 — ASR adaptation | Whisper vanilla vs domain-prompted |
| `run_translation.py` | 2 — MT adaptation | NLLB / Qwen vanilla / Qwen + glossary |
| `run_pipeline.py` | 3 — Full pipeline | End-to-end audio → EN across all conditions |
| `compare_glossaries.py` | 4 — Glossary methods | Manual vs TF-IDF vs LLM-generated glossary |

See [../docs/architecture.md](../docs/architecture.md) for the full experiment design.
