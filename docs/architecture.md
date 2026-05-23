# System Architecture

Two systems evaluated head-to-head on 130 Spanish EU Parliament speeches from EPIC v2.0, scored against professional English interpreter output.

---

## Pipeline Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  EPIC v2.0 Corpus                                                    │
│  130 Spanish EU Parliament speeches  (~6.3 h total, ~2.9 min avg)   │
└────────────────────────┬─────────────────────────────────────────────┘
                         │  audio (WAV, 16kHz mono)
          ┌──────────────┴──────────────────────────┐
          ▼  System A: Cascaded                     ▼  System B: End-to-end
┌──────────────────┐                    ┌────────────────────────┐
│ Whisper large-v3 │                    │  SeamlessM4T v2        │
│ (faster-whisper  │                    │  facebook/seamless-    │
│  int8, CUDA)     │                    │  m4t-v2-large          │
│  ES audio →      │                    │                        │
│  ES text         │                    │  ES audio → EN text    │
└────────┬─────────┘                    │  (joint model, no      │
         │                             │   intermediate step)   │
         ▼                             └───────────┬────────────┘
┌──────────────────┐                               │
│  NLLB-200-3.3B   │                               │
│  ES text → EN    │                               │
└────────┬─────────┘                               │
         │  English text                           │  English text
         └──────────────────┬──────────────────────┘
                            ▼
             ┌──────────────────────────────┐
             │  Multi-faceted Evaluation    │
             │  vs gold EN interpreter      │
             │  transcript                  │
             │                              │
             │  COMET-DA    (alignment)     │
             │  COMET-Kiwi  (faithfulness)  │
             │  TermAcc     (terminology)   │
             │  BLEU / chrF (surface)       │
             │  Composite   (weighted)      │
             └──────────────────────────────┘
```

---

## Models

| Role | Model | VRAM | Notes |
|------|-------|------|-------|
| ASR (System A) | `Systran/faster-whisper-large-v3` (int8) | ~3 GB | Standard open-source ASR baseline; most cited in cascaded ST literature |
| MT (System A) | `facebook/nllb-200-3.3B` (fp16) | ~7 GB | Dedicated translation model; most cited in 2025-26 cascaded ST papers |
| End-to-end (System B) | `facebook/seamless-m4t-v2-large` (fp16) | ~8 GB | Joint model; trained on Europarl — gives inherent domain advantage |
| Eval (COMET-DA) | `Unbabel/wmt22-comet-da` | ~2 GB | Interpreter alignment |
| Eval (COMET-Kiwi) | `Unbabel/wmt22-cometkiwi-da` | ~2 GB | Source faithfulness (reference-free) |
| **Total peak** | | **~22 GB** | Fits on one RTX 6000 Ada (48 GB) |

---

## Why These Models

**Whisper large-v3**: The de facto standard for open-source ASR in 2026. Used as-is without fine-tuning, representing a general-purpose ASR system applied to a specialised domain.

**NLLB-200-3.3B**: The most commonly cited open-source MT model in cascaded speech translation research (2025–26). A December 2025 multi-benchmark study found cascaded systems with NLLB-3.3B marginally outperform SeamlessM4T v2 on eng→X directions, providing empirical grounding for this choice.

**SeamlessM4T v2**: Meta's state-of-the-art end-to-end speech translation model. Important confound: trained on SeamlessAlign corpus including Europarl (EU Parliament data), giving it an inherent domain advantage over System A. Both systems evaluated zero-shot — no fine-tuning.

---

## Evaluation Protocol

See [docs/evaluation.md](evaluation.md) for the full protocol specification and weight justification.

Summary of metrics:

| Metric | What it measures | Weight in composite |
|--------|-----------------|---------------------|
| COMET-DA | Semantic closeness to interpreter reference | 0.35 |
| COMET-Kiwi | Faithfulness to Spanish source (no reference) | 0.30 |
| TermAcc | EU terminology accuracy (IATE-derived) | 0.20 |
| BLEU/100 | Surface overlap (included for comparability) | 0.15 |

---

## Experiments

| # | Script | Input | What it measures |
|---|--------|-------|-----------------|
| 1 | `run_asr.py` | Audio | Whisper ASR quality vs gold ES transcript (WER, CER) |
| 2 | `run_pipeline.py --backend cascaded` | Audio | System A: Whisper + NLLB-3.3B end-to-end |
| 3 | `run_pipeline.py --backend seamless` | Audio | System B: SeamlessM4T v2 end-to-end |

Run experiments 2 and 3 in parallel across both GPUs with `run_parallel.sh`.

---

## Code Layout

```
src/
├── asr.py          Transcriber (Whisper) + SeamlessTranscriber
├── translator.py   NLLB-3.3B wrapper
├── metrics.py      WER, BLEU, chrF, COMET-DA, COMET-Kiwi, TermAcc, composite
├── glossary.py     IATE glossary loading
├── epic_parser.py  EPIC transcript markup cleaner
└── utils.py        Logging, output dir, per-file cache helpers

experiments/
├── run_asr.py        Experiment 1: ASR quality
└── run_pipeline.py   Experiments 2 & 3: cascaded vs seamless

scripts/
├── prepare_epic.py       Clean transcripts, pair with audio
├── download_models.py    Pull all models from HuggingFace
└── extract_glossary_*.py IATE glossary construction

data/
└── download_epic.py      Resumable EPIC v2.0 downloader
```
