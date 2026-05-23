# Domain-Adapted Speech Translation Evaluated Against Professional Interpreters

**Masters Thesis — Anmol Guragain, 2026**

Spanish EU Parliament speeches → ASR → Machine Translation → compared against what a professional interpreter actually said in English.
Core question: **how much does a domain glossary improve ASR and translation quality?**

---

## Pipeline

```
EPIC v2.0: 130 Spanish EU Parliament speeches (~6.3 h, avg 2.9 min each)
                         │  audio (WAV, 16kHz mono)
          ┌──────────────┴──────────────────────────┐
          ▼  cascaded                               ▼  end-to-end
┌──────────────────┐                    ┌────────────────────────┐
│ Whisper large-v3 │ vanilla /          │ SeamlessM4T v2         │
│  ES audio →      │ + domain prompt    │ ES audio → EN text     │
│  ES text         │                    └───────────┬────────────┘
└────────┬─────────┘                               │
         │                                         │
   ┌─────┴──────┐  ┌─────────────────────────┐    │
   ▼            ▼  │  Translate ES→EN         │    │
┌──────────┐    │  │  NLLB-3.3B  baseline    │    │
│ ASR eval │    └─►│  Qwen3-8B   vanilla      │    │
│ WER / CER│       │  Qwen3-8B   + glossary  │    │
└──────────┘       └────────────┬────────────┘    │
                                │  English text   │
                                └────────┬────────┘
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

## Systems

| System | Components | VRAM |
|--------|-----------|------|
| **A — Cascaded** | `faster-whisper large-v3` + `facebook/nllb-200-3.3B` | ~10 GB |
| **B — End-to-end** | `facebook/seamless-m4t-v2-large` | ~8 GB |
| Eval | `Unbabel/wmt22-comet-da` + `wmt22-cometkiwi-da` | ~4 GB |
| **Peak** | | **~22 GB** (one RTX 6000 Ada, 48 GB) |

---

## Experiments

| # | Script | What it measures |
|---|--------|-----------------|
| 1 | `run_asr.py` | Whisper ASR quality vs gold ES transcript (WER, CER) |
| 2 | `run_pipeline.py --backend cascaded` | System A: Whisper + NLLB-3.3B (COMET-DA, COMET-Kiwi, TermAcc, BLEU) |
| 3 | `run_pipeline.py --backend seamless` | System B: SeamlessM4T v2 end-to-end (same metrics) |

Run experiments 2 & 3 in parallel across both GPUs: `bash run_parallel.sh`

Evaluation protocol: [docs/evaluation.md](docs/evaluation.md)

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
