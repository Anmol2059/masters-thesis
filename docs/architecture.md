# System Architecture

End-to-end offline pipeline: Spanish speech → ASR / Speech Translation → Evaluation against professional interpreter output.

---

## Pipeline Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  EPIC v2.0 Corpus                                                    │
│  130 Spanish EU Parliament speeches  (~6.3 h total, ~2.9 min avg)   │
└────────────────────────┬─────────────────────────────────────────────┘
                         │  audio (WAV, 16kHz mono)
          ┌──────────────┴──────────────────────────┐
          ▼                                         ▼
┌──────────────────┐                    ┌────────────────────────┐
│ Whisper large-v3 │                    │  SeamlessM4T v2        │
│ (faster-whisper  │                    │  (end-to-end ST)       │
│  int8, CUDA)     │                    │                        │
│                  │  vanilla OR        │  ES audio → EN text    │
│  ES audio →      │  + domain prompt   │  (no adaptation)       │
│  ES text         │                    └───────────┬────────────┘
└────────┬─────────┘                               │
         │  Spanish text (hypothesis)              │  English text
   ┌─────┴────────────┐                            │
   ▼                  ▼                            │
┌──────────┐  ┌──────────────────────────────┐    │
│ ASR eval │  │   Translation   ES → EN      │    │
│ vs gold  │  │                              │    │
│ ES text  │  │  A) NLLB-3.3B  (baseline)   │    │
│          │  │  B) Qwen3-8B   vanilla       │    │
│ WER  CER │  │  C) Qwen3-8B   + glossary   │    │
└──────────┘  └──────────────┬───────────────┘    │
                             │  English text       │
                             └──────────┬──────────┘
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

| Role | Model | VRAM | Notes |
|------|-------|------|-------|
| ASR | `Systran/faster-whisper-large-v3` (int8) | ~3 GB | Standard open-source ASR baseline |
| End-to-end ST | `facebook/seamless-m4t-v2-large` (fp16) | ~8 GB | Direct ES audio → EN text, no cascade |
| MT baseline | `facebook/nllb-200-3.3B` (fp16) | ~7 GB | Dedicated translation model |
| MT main | `Qwen/Qwen3-8B` (BitsAndBytes 4-bit) | ~6 GB | Instruction-tuned LLM + glossary injection |
| Evaluation | `Unbabel/wmt22-comet-da` | ~2 GB | Neural MT quality metric |
| **Total peak** | | **~26 GB** | Fits on a single RTX 6000 Ada (48 GB) |

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

SeamlessM4T v2 has no comparable adaptation interface — used as a domain-agnostic end-to-end baseline.

### Translation — Glossary in System Prompt

Qwen3 receives a glossary block in its system message constraining how specific terms must be rendered. Chain-of-thought is disabled (`enable_thinking=False`) for inference speed:

```python
system = """Translate Spanish to English. Use this glossary strictly:
- ponente → rapporteur (NOT speaker)
- Comisión → Commission (NOT committee)
- directiva → directive (NOT guideline)
- PIB → GDP
- BCE → ECB
..."""
```

NLLB-3.3B is a seq2seq model with no prompt interface — used as a glossary-free baseline.

---

## Glossary Construction (Experiment 4)

Three methods compared:

| Method | How | Output |
|--------|-----|--------|
| **Manual** | Expert selects ~25–50 high-stakes EU terms | `glossaries/eu_parliament_es_en.json` |
| **TF-IDF** | Top-k n-grams by TF-IDF score over EPIC corpus | `glossaries/eu_parliament_es_en_tfidf.json` |
| **LLM-generated** | Qwen3 prompted to extract + translate domain terms | `glossaries/eu_parliament_es_en_llm.json` |

---

## Experiments

| # | Name | Input | What varies | Key metric |
|---|------|-------|-------------|------------|
| 1 | ASR adaptation | Audio | Whisper vanilla vs domain prompt vs SeamlessM4T | WER, CER |
| 2 | MT adaptation | Gold ES text | NLLB-3.3B / Qwen3 vanilla / Qwen3 + glossary | BLEU, COMET, TermAcc |
| 3 | Full pipeline | Audio | Cascaded (Whisper+Qwen3) vs end-to-end (SeamlessM4T) + domain adaptation | BLEU, COMET |
| 4 | Glossary methods | Gold ES text | manual vs TF-IDF vs LLM-generated | BLEU, COMET, TermAcc |

The central comparison in Experiment 3 — cascaded domain-adapted vs end-to-end — is the paper's primary research contribution.

---

## Code Layout

```
src/
├── asr.py          Transcriber (Whisper) + SeamlessTranscriber (SeamlessM4T v2)
├── translator.py   Qwen3 + NLLB wrappers with glossary injection
├── glossary.py     Load JSON glossary, format for prompt / ASR prompt
├── metrics.py      WER, CER, BLEU, chrF, COMET, term accuracy
└── epic_parser.py  Strip EPIC markup from raw transcripts

experiments/
├── run_asr.py              Experiment 1 (--backend whisper|seamless)
├── run_translation.py      Experiment 2
├── run_pipeline.py         Experiment 3 (--backend cascaded|seamless)
└── compare_glossaries.py   Experiment 4

scripts/
├── prepare_epic.py           Clean transcripts, pair with audio
├── download_models.py        Pull models from HuggingFace Hub
├── extract_glossary_tfidf.py Auto-glossary via TF-IDF
└── extract_glossary_llm.py   Auto-glossary via Qwen3

data/
└── download_epic.py   Resumable downloader for EPIC v2.0 from Zenodo
```
