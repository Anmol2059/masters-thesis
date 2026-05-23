# Domain-Adapted Speech Translation Evaluated Against Professional Interpreters

**Masters Thesis — Anmol Guragain, 2026**

> Spanish audio → ASR → Translate to English → Compare with what a professional interpreter actually said.
> How much does a domain glossary improve ASR and translation? We find out using real European Parliament interpreting data.

---

## What This Is

We take **Spanish speech** from the European Parliament (EPIC v2.0 corpus), run it through ASR and machine translation, and compare our English output against **what a real professional interpreter said** for the same speech.

We then show that injecting a **domain-specific glossary** into both the ASR and translation steps significantly improves output quality — especially on institutional and technical terminology.

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   EPIC v2.0: Spanish source audio                                   │
│   (EU Parliament speeches, extracted from video)                    │
│                                                                     │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   ASR (Whisper)  │  ← Run with & without domain prompt
              │   Spanish audio  │
              │   → Spanish text │
              └────────┬────────┘
                       │
          ┌────────────┼────────────────┐
          ▼                             ▼
   ┌─────────────┐              ┌──────────────┐
   │ Compare ASR │              │  Translate   │  ← Run with & without glossary
   │ output with │              │  ES → EN     │
   │ EPIC gold   │              │              │
   │ Spanish     │              │  Models:     │
   │ transcript  │              │  • Qwen2.5-7B│
   │             │              │  • NLLB-600M │
   │ → WER, CER  │              └──────┬───────┘
   └─────────────┘                     │
                                       ▼
                              ┌──────────────────┐
                              │ Compare our EN   │
                              │ translation with │
                              │ EPIC gold EN     │
                              │ interpreter      │
                              │ transcript       │
                              │                  │
                              │ → BLEU, COMET,   │
                              │   chrF, TermAcc  │
                              └──────────────────┘
```

**This repo is the offline experimental pipeline.** For the real-time interpreter assistance system, see [InterpretBench](https://github.com/Anmol2059/interpret-bench).

---

## Data: EPIC v2.0

We use the [European Parliament Interpreting Corpus v2.0](https://zenodo.org/records/13856205) (Lobascio, Liu & Russo, 2024).

**What it contains (what we use):**

| Component | What it is | How we use it |
|-----------|-----------|---------------|
| Spanish source **audio** | EU Parliament speeches by Spanish MEPs (extracted from video via ffmpeg) | Input to our ASR pipeline |
| Spanish source **transcript** | Gold human transcription of the Spanish speech | Ground truth to measure ASR accuracy (WER) |
| English interpreter **transcript** | Gold human transcription of what the professional interpreter said | Ground truth to measure our translation quality (BLEU, COMET) |
| Metadata | Speaker info, speech rate, topic, delivery mode | Analysis of where our system struggles |

**What we do NOT use:** video files, Italian data, POS tags, alignments (for now).

**Size for ES→EN direction:** ~9-10 hours of paired audio, hundreds of speech segments.

### Download & Prepare

```bash
# 1. Download from Zenodo (manual — browser required)
#    https://zenodo.org/records/13856205
#    Get: 05_transcripts_v2.0.zip (3.5 MB)
#         06_recordings_v2.0.zip  (7.5 GB)

# 2. Place in data/epic_raw/ and extract
mkdir -p data/epic_raw
mv 05_transcripts_v2.0.zip data/epic_raw/
mv 06_recordings_v2.0.zip data/epic_raw/
cd data/epic_raw && unzip '*.zip' && cd ../..

# 3. Prepare: extract audio from video, clean transcripts, pair them
python scripts/prepare_epic.py \
    --epic-dir data/epic_raw \
    --output-dir data/epic_processed \
    --direction es-en
```

`prepare_epic.py` does:
1. Finds Spanish source speech videos → extracts audio (`ffmpeg -vn -ac 1 -ar 16000`)
2. Parses Spanish transcripts → removes markup (`(.)`, `-ehm-`, `//`, `~word </correct/>`)
3. Parses English interpreter transcripts → same cleanup
4. Pairs them by speech ID
5. Outputs:

```
data/epic_processed/
├── audio/
│   ├── es_001.wav          # Spanish source audio (16kHz mono)
│   ├── es_002.wav
│   └── ...
├── transcripts_es/
│   ├── es_001.txt          # Gold Spanish transcript (cleaned)
│   └── ...
├── transcripts_en_interp/
│   ├── es_001.txt          # Gold English interpreter output (cleaned)
│   └── ...
└── manifest.json           # Maps everything together + metadata
```

---

## Models

| Component | Model | Params | VRAM | Purpose |
|-----------|-------|--------|------|---------|
| ASR | `faster-whisper large-v3` (int8) | 1.55B | ~3 GB | Spanish speech → Spanish text |
| Translation (main) | `Qwen2.5-7B-Instruct` (GPTQ 4-bit) | 7B | ~5 GB | Spanish text → English text |
| Translation (baseline) | `NLLB-200-distilled-600M` | 600M | ~2 GB | Spanish text → English text (no glossary) |
| Evaluation | `unbabel-comet` (wmt22-comet-da) | — | ~2 GB | COMET scoring |
| **Total** | | | **~12 GB** | Fits on 24GB GPU |

```bash
# Download all models
python scripts/download_models.py
```

---

## Experiments

### Experiment 1: ASR — Does Domain Prompting Help?

Whisper's `initial_prompt` biases recognition toward specific vocabulary. We test whether feeding EU Parliament terminology reduces errors.

```bash
# Vanilla ASR (no domain prompt)
python experiments/run_asr.py \
    --audio-dir data/epic_processed/audio/ \
    --gold-dir data/epic_processed/transcripts_es/ \
    --model large-v3 \
    --condition vanilla \
    --output results/asr_vanilla.json

# Domain-adapted ASR (glossary terms as initial prompt)
python experiments/run_asr.py \
    --audio-dir data/epic_processed/audio/ \
    --gold-dir data/epic_processed/transcripts_es/ \
    --model large-v3 \
    --condition domain \
    --glossary glossaries/eu_parliament_es.json \
    --output results/asr_domain.json
```

**What changes:**
```python
# Vanilla
model.transcribe(audio, language="es")

# Domain-adapted — inject terms Whisper often gets wrong
model.transcribe(audio, language="es",
    initial_prompt="Parlamento Europeo, Comisión Europea, BCE, PIB, "
                   "subvenciones, directiva, ponente, enmienda, codecisión")
```

**Metrics:** WER (overall), WER (domain terms only), CER

### Experiment 2: Translation — Does a Glossary Help?

We translate **gold Spanish transcripts** (not ASR output) to isolate translation quality.

```bash
# NLLB baseline (no glossary possible)
python experiments/run_translation.py \
    --source-dir data/epic_processed/transcripts_es/ \
    --reference-dir data/epic_processed/transcripts_en_interp/ \
    --model nllb \
    --output results/mt_nllb.json

# Qwen vanilla (no glossary)
python experiments/run_translation.py \
    --source-dir data/epic_processed/transcripts_es/ \
    --reference-dir data/epic_processed/transcripts_en_interp/ \
    --model qwen \
    --condition vanilla \
    --output results/mt_qwen_vanilla.json

# Qwen with glossary
python experiments/run_translation.py \
    --source-dir data/epic_processed/transcripts_es/ \
    --reference-dir data/epic_processed/transcripts_en_interp/ \
    --model qwen \
    --condition domain \
    --glossary glossaries/eu_parliament_es_en.json \
    --output results/mt_qwen_domain.json
```

**What changes:**
```python
# Vanilla
system = "Translate the following Spanish text to English."

# Domain-adapted — glossary in system prompt
system = """Translate Spanish to English. Use this glossary strictly:
- ponente → rapporteur (NOT speaker)
- Comisión → Commission (NOT committee, when referring to EU)
- directiva → directive (NOT guideline)
- enmienda → amendment
- codecisión → co-decision
- PIB → GDP
- BCE → ECB (European Central Bank)
- subvenciones agrícolas → agricultural subsidies
"""
```

**Metrics:** BLEU, COMET, chrF, Term Accuracy (% of glossary terms correctly translated)

### Experiment 3: Full Pipeline (ASR + Translation)

End-to-end from audio. Shows how ASR errors compound with translation errors.

```bash
# Vanilla ASR → Vanilla translation
python experiments/run_pipeline.py \
    --audio-dir data/epic_processed/audio/ \
    --reference-dir data/epic_processed/transcripts_en_interp/ \
    --condition vanilla \
    --output results/pipeline_vanilla.json

# Domain ASR → Domain translation
python experiments/run_pipeline.py \
    --audio-dir data/epic_processed/audio/ \
    --reference-dir data/epic_processed/transcripts_en_interp/ \
    --condition domain \
    --glossary glossaries/eu_parliament_es_en.json \
    --output results/pipeline_domain.json

# Ablations: domain ASR + vanilla MT, vanilla ASR + domain MT
python experiments/run_pipeline.py --condition domain-asr-only ...
python experiments/run_pipeline.py --condition domain-mt-only ...
```

### Experiment 4: Glossary Construction Methods

How to build the glossary? Compare three approaches:

```bash
# Method 1: Manual — expert picks ~50 terms (already in glossaries/)
# Method 2: TF-IDF — auto-extract from EPIC Spanish transcripts
python scripts/extract_glossary_tfidf.py \
    --corpus data/epic_processed/transcripts_es/ \
    --top-k 200 \
    --output glossaries/eu_parliament_es_en_tfidf.json

# Method 3: LLM-generated — ask Qwen to identify domain terms
python scripts/extract_glossary_llm.py \
    --corpus data/epic_processed/transcripts_es/ \
    --output glossaries/eu_parliament_es_en_llm.json

# Compare all glossaries on translation quality
python experiments/compare_glossaries.py \
    --source-dir data/epic_processed/transcripts_es/ \
    --reference-dir data/epic_processed/transcripts_en_interp/ \
    --glossaries manual tfidf llm \
    --output results/glossary_comparison.json
```

---

## Expected Results

### Table 1: ASR Domain Adaptation

| Condition | WER ↓ | WER (domain terms) ↓ | CER ↓ |
|-----------|-------|----------------------|-------|
| Whisper large-v3 vanilla | | | |
| Whisper large-v3 + domain prompt | | | |

### Table 2: Translation Quality (from gold transcripts)

| Model | Glossary | BLEU ↑ | COMET ↑ | chrF ↑ | TermAcc ↑ |
|-------|----------|--------|---------|--------|-----------|
| NLLB-600M | — | | | | |
| Qwen2.5-7B | none | | | | |
| Qwen2.5-7B | manual | | | | |
| Qwen2.5-7B | TF-IDF | | | | |
| Qwen2.5-7B | LLM-gen | | | | |

### Table 3: Full Pipeline (audio → English)

| ASR | MT | BLEU ↑ | COMET ↑ | chrF ↑ |
|-----|-----|--------|---------|--------|
| vanilla | NLLB | | | |
| vanilla | Qwen vanilla | | | |
| vanilla | Qwen + glossary | | | |
| domain | Qwen vanilla | | | |
| domain | Qwen + glossary | | | |

### Table 4: Glossary Construction Methods

| Method | # Terms | TermPrec ↑ | BLEU ↑ | COMET ↑ |
|--------|---------|-----------|--------|---------|
| Manual (expert) | ~50 | | | |
| TF-IDF auto | ~200 | | | |
| LLM-generated | ~100 | | | |
| Manual + LLM | ~120 | | | |

---

## Project Structure

```
masters-thesis/
├── README.md
├── requirements.txt
├── Dockerfile
├── run.sh
│
├── scripts/
│   ├── download_models.py            # Download Whisper, Qwen, NLLB, COMET
│   ├── prepare_epic.py               # Extract audio + parse EPIC transcripts
│   ├── extract_glossary_tfidf.py     # Build glossary via TF-IDF
│   └── extract_glossary_llm.py       # Build glossary via LLM
│
├── experiments/
│   ├── run_asr.py                    # Exp 1: ASR vanilla vs domain
│   ├── run_translation.py            # Exp 2: Translation vanilla vs glossary
│   ├── run_pipeline.py               # Exp 3: Full audio → English
│   └── compare_glossaries.py         # Exp 4: Glossary methods comparison
│
├── src/
│   ├── asr.py                        # Whisper wrapper + domain prompting
│   ├── translator.py                 # Qwen + NLLB translation
│   ├── glossary.py                   # Glossary loading, formatting, injection
│   ├── metrics.py                    # WER, BLEU, COMET, chrF, term accuracy
│   └── epic_parser.py               # Parse EPIC transcript format
│
├── glossaries/
│   ├── eu_parliament_es.json         # Spanish ASR domain terms
│   ├── eu_parliament_es_en.json      # ES→EN translation glossary (manual)
│   └── README.md                     # Glossary format documentation
│
├── configs/
│   └── default.yaml                  # Model paths, hyperparams
│
├── data/                             # gitignored
│   ├── epic_raw/                     # Downloaded EPIC v2.0 zips
│   └── epic_processed/              # Extracted audio + clean transcripts
│
├── results/                          # gitignored
│
└── paper/                            # Thesis LaTeX
    ├── main.tex
    ├── figures/
    └── tables/
```

---

## Differences from InterpretBench

| | This repo (masters-thesis) | InterpretBench |
|---|---|---|
| **Mode** | Offline experiments on EPIC data | Real-time streaming |
| **Input** | Pre-recorded Spanish audio files | Live dual-mic audio |
| **ASR** | One ASR (Spanish source only) | Two ASRs (source + interpreter) |
| **Evaluation** | BLEU/COMET against interpreter transcript | LLM-as-judge real-time scoring |
| **Focus** | Domain adaptation experiments | Interpreter assistance tool |
| **Glossary** | Main contribution — how to build & use | Used but not studied |

---

## Setup

```bash
git clone https://github.com/Anmol2059/masters-thesis.git
cd masters-thesis

conda create -n thesis python=3.11
conda activate thesis

pip install -r requirements.txt

# Download models (~10 GB)
python scripts/download_models.py

# Prepare EPIC data (after manual download from Zenodo)
python scripts/prepare_epic.py --epic-dir data/epic_raw --output-dir data/epic_processed --direction es-en

# Run all experiments
python experiments/run_asr.py --condition vanilla --output results/asr_vanilla.json
python experiments/run_asr.py --condition domain --glossary glossaries/eu_parliament_es.json --output results/asr_domain.json
python experiments/run_translation.py --model nllb --output results/mt_nllb.json
python experiments/run_translation.py --model qwen --condition vanilla --output results/mt_qwen_vanilla.json
python experiments/run_translation.py --model qwen --condition domain --glossary glossaries/eu_parliament_es_en.json --output results/mt_qwen_domain.json
python experiments/run_pipeline.py --condition vanilla --output results/pipeline_vanilla.json
python experiments/run_pipeline.py --condition domain --glossary glossaries/eu_parliament_es_en.json --output results/pipeline_domain.json
python experiments/compare_glossaries.py --glossaries manual tfidf llm --output results/glossary_comparison.json
```

---

## Requirements

```
faster-whisper>=1.0.0
transformers>=4.40.0
auto-gptq>=0.7.0
torch>=2.1.0
sacrebleu>=2.3.0
unbabel-comet>=2.2.0
jiwer
scikit-learn
ffmpeg-python
tqdm
pyyaml
```

**Hardware:** NVIDIA GPU ≥16 GB VRAM (tested on RTX 3090/4090 24 GB)

---

## Citation

```bibtex
@mastersthesis{guragain2026domain,
  title  = {Domain-Adapted Speech Translation Evaluated Against
            Professional Interpreter Output},
  author = {Anmol Guragain},
  year   = {2026}
}
```

MIT License
