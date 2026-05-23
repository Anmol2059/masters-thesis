# System Architecture

End-to-end offline pipeline: Spanish speech → ASR → Machine Translation → Evaluation against professional interpreter output.

---

## Pipeline Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  EPIC v2.0 Corpus                                                    │
│  130 Spanish EU Parliament speeches  (~6.3 h total, ~2.9 min avg)   │
└────────────────────────┬─────────────────────────────────────────────┘
                         │  audio (WAV, 16kHz mono)
                         ▼
               ┌──────────────────┐
               │  Whisper large-v3│  ← vanilla  OR  + domain initial_prompt
               │  (faster-whisper │
               │   int8, CUDA)    │
               └────────┬─────────┘
                        │  Spanish text (hypothesis)
         ┌──────────────┼──────────────────────┐
         ▼                                     ▼
  ┌─────────────┐                     ┌──────────────────┐
  │  ASR eval   │                     │   Translation    │
  │  vs gold ES │                     │   ES → EN        │
  │  transcript │                     │                  │
  │             │                     │  A) NLLB-600M    │
  │  WER  CER   │                     │  B) Qwen2.5-7B   │
  │  (overall + │                     │     vanilla      │
  │  domain     │                     │  C) Qwen2.5-7B   │
  │  terms)     │                     │     + glossary   │
  └─────────────┘                     └───────┬──────────┘
                                              │  English text
                                              ▼
                                   ┌──────────────────────┐
                                   │  Evaluation          │
                                   │  vs gold EN interp   │
                                   │  transcript          │
                                   │                      │
                                   │  BLEU  chrF  COMET   │
                                   │  Term Accuracy       │
                                   └──────────────────────┘
```

---

## Models

| Role | Model | Size | VRAM | Quantisation |
|------|-------|------|------|-------------|
| ASR | `faster-whisper large-v3` | 1.55B | ~3 GB | int8 |
| Translation (main) | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` | 7B | ~5 GB | GPTQ 4-bit |
| Translation (baseline) | `facebook/nllb-200-distilled-600M` | 600M | ~2 GB | fp16 |
| Evaluation | `Unbabel/wmt22-comet-da` | — | ~2 GB | fp16 |
| **Total peak** | | | **~12 GB** | fits on 24 GB GPU |

---

## Domain Adaptation Methods

### ASR — Whisper `initial_prompt`

Whisper's `initial_prompt` parameter biases the beam search decoder toward specific vocabulary. We feed a comma-separated list of EU Parliament terms:

```python
# vanilla
model.transcribe(audio, language="es")

# domain-adapted
model.transcribe(audio, language="es",
    initial_prompt="Parlamento Europeo, Comisión Europea, BCE, PIB, "
                   "ponente, enmienda, directiva, codecisión ...")
```

### Translation — Glossary in System Prompt

Qwen receives a glossary block in its system message constraining how specific terms must be rendered:

```python
system = """Translate Spanish to English. Use this glossary strictly:
- ponente → rapporteur (NOT speaker)
- Comisión → Commission (NOT committee)
- directiva → directive (NOT guideline)
- PIB → GDP
- BCE → ECB
..."""
```

NLLB is a seq2seq model with no prompt interface — used as a glossary-free baseline.

---

## Glossary Construction (Experiment 4)

Three methods compared:

| Method | How | Output |
|--------|-----|--------|
| **Manual** | Expert selects ~25–50 high-stakes EU terms | `glossaries/eu_parliament_es_en.json` |
| **TF-IDF** | Top-k n-grams by TF-IDF score over EPIC corpus | `glossaries/eu_parliament_es_en_tfidf.json` |
| **LLM-generated** | Qwen prompted to extract + translate domain terms | `glossaries/eu_parliament_es_en_llm.json` |

---

## Experiments

| # | Name | Input | What varies | Key metric |
|---|------|-------|-------------|------------|
| 1 | ASR adaptation | Audio | vanilla vs domain prompt | WER, CER |
| 2 | MT adaptation | Gold ES text | model × glossary method | BLEU, COMET, TermAcc |
| 3 | Full pipeline | Audio | ASR × MT condition | BLEU, COMET |
| 4 | Glossary methods | Gold ES text | manual vs TF-IDF vs LLM | BLEU, COMET, TermAcc |

---

## Code Layout

```
src/
├── asr.py          Whisper wrapper — loads model, applies domain prompt
├── translator.py   Qwen + NLLB wrappers — glossary injection
├── glossary.py     Load JSON glossary, format for prompt / ASR prompt
├── metrics.py      WER, CER, BLEU, chrF, COMET, term accuracy
└── epic_parser.py  Strip EPIC markup from raw transcripts

experiments/
├── run_asr.py          Experiment 1
├── run_translation.py  Experiment 2
├── run_pipeline.py     Experiment 3
└── compare_glossaries.py Experiment 4

scripts/
├── prepare_epic.py           Clean transcripts, pair with audio
├── download_models.py        Pull models from HuggingFace Hub
├── extract_glossary_tfidf.py Auto-glossary via TF-IDF
└── extract_glossary_llm.py   Auto-glossary via Qwen

data/
└── download_epic.py   Resumable downloader for EPIC v2.0 from Zenodo
```
