# Domain-Adapted Speech Translation Evaluated Against Professional Interpreters

**Masters Thesis — Anmol Guragain, 2026**

Spanish EU Parliament speeches → ASR → Machine Translation → compared against what a professional interpreter actually said in English.
Core question: **how much does a domain glossary improve ASR and translation quality?**

---

## Pipeline

```
EPIC v2.0: 130 Spanish EU Parliament speeches (~6.3 h, avg 2.9 min each)
                         │
                         ▼
              ┌──────────────────┐
              │ Whisper large-v3 │  vanilla  vs.  + EU Parliament domain prompt
              │  Spanish audio   │
              │  → Spanish text  │
              └────────┬─────────┘
                       │
         ┌─────────────┼──────────────────┐
         ▼                                ▼
  ┌─────────────┐               ┌──────────────────┐
  │ ASR eval vs │               │  Translate ES→EN │  NLLB-600M baseline
  │ gold ES     │               │                  │  Qwen2.5-7B vanilla
  │ transcript  │               │                  │  Qwen2.5-7B + glossary
  │ WER / CER   │               └────────┬─────────┘
  └─────────────┘                        │
                                         ▼
                              ┌──────────────────────┐
                              │ Eval vs gold English │
                              │ interpreter output   │
                              │ BLEU · chrF · COMET  │
                              │ Term Accuracy        │
                              └──────────────────────┘
```

---

## Dataset

[EPIC v2.0](https://zenodo.org/records/13856205) — European Parliament Interpreting Corpus (Lobascio, Liu & Russo, 2024)

| | |
|--|--|
| Speeches (ES→EN) | 130 |
| Total audio | ~6.3 hours |
| Average speech | 2.9 min (range: 0.15 – 25.2 min) |
| Source | Spanish MEP speeches (MP4 → 16 kHz mono WAV) |
| Gold transcripts | Human-transcribed Spanish source + English interpreter output |

Full corpus details: [docs/dataset.md](docs/dataset.md)

---

## Models

| Role | Model | VRAM |
|------|-------|------|
| ASR | `faster-whisper large-v3` (int8) | ~3 GB |
| Translation | `Qwen2.5-7B-Instruct` (GPTQ 4-bit) | ~5 GB |
| Baseline MT | `NLLB-200-distilled-600M` | ~2 GB |
| Evaluation | `Unbabel/wmt22-comet-da` | ~2 GB |
| **Total** | | **~12 GB** (fits on 24 GB GPU) |

---

## Experiments

| # | Script | What it measures |
|---|--------|-----------------|
| 1 | `experiments/run_asr.py` | ASR — vanilla vs domain-prompted Whisper (WER, CER) |
| 2 | `experiments/run_translation.py` | MT — NLLB / Qwen vanilla / Qwen + glossary (BLEU, COMET, TermAcc) |
| 3 | `experiments/run_pipeline.py` | Full pipeline audio→EN across all ASR × MT conditions |
| 4 | `experiments/compare_glossaries.py` | Glossary construction: manual vs TF-IDF vs LLM-generated |

Architecture details: [docs/architecture.md](docs/architecture.md)

---

## Setup

```bash
git clone https://github.com/Anmol2059/masters-thesis.git
cd masters-thesis

conda create -n thesis python=3.11 && conda activate thesis

# Install PyTorch first (adjust CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Download EPIC v2.0 corpus (~10 GB, resumable)
python data/download_epic.py --data-dir data/epic_raw

# Clean transcripts + pair with audio
python scripts/prepare_epic.py --epic-dir data/epic_raw --output-dir data/epic_processed

# Download models (~10 GB)
python scripts/download_models.py
```

---

## Run All Experiments

```bash
bash run.sh
```

Or individually — see [experiments/README.md](experiments/README.md).

---

## Project Structure

```
masters-thesis/
├── data/
│   ├── download_epic.py        Resumable EPIC v2.0 downloader
│   └── README.md
├── scripts/
│   ├── prepare_epic.py         Clean transcripts, pair with audio
│   ├── download_models.py      Pull models from HuggingFace
│   ├── extract_glossary_tfidf.py
│   └── extract_glossary_llm.py
├── experiments/
│   ├── run_asr.py              Experiment 1
│   ├── run_translation.py      Experiment 2
│   ├── run_pipeline.py         Experiment 3
│   └── compare_glossaries.py   Experiment 4
├── src/
│   ├── asr.py                  Whisper wrapper + domain prompting
│   ├── translator.py           Qwen + NLLB wrappers
│   ├── glossary.py             Glossary loading + formatting
│   ├── metrics.py              WER, BLEU, COMET, chrF, TermAcc
│   └── epic_parser.py          EPIC transcript markup cleaner
├── glossaries/
│   ├── eu_parliament_es.json       ASR domain terms (ES)
│   └── eu_parliament_es_en.json    Translation glossary (ES→EN, manual)
├── docs/
│   ├── dataset.md              Corpus documentation
│   └── architecture.md         Pipeline + experiment design
├── configs/default.yaml
├── requirements.txt
├── run.sh                      Full pipeline in one command
└── paper/                      Thesis LaTeX
```

---

## Related

This is the **offline experiment repo**. The real-time interpreter assistance system is at [InterpretBench](https://github.com/Anmol2059/interpret-bench).

---

## Citation

```bibtex
@mastersthesis{guragain2026domain,
  title  = {Domain-Adapted Speech Translation Evaluated Against Professional Interpreter Output},
  author = {Anmol Guragain},
  year   = {2026}
}
```

MIT License
