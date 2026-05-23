# 🎙️ InterpretBench

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-≥16GB_VRAM-f97316?logo=nvidia&logoColor=white)]()
[![Whisper](https://img.shields.io/badge/ASR-Whisper_large--v3-8b5cf6)]()
[![Qwen](https://img.shields.io/badge/LLM-Qwen2.5--7B-ec4899)]()
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)]()

**Real-time AI-powered interpreter quality assessment for Spanish ↔ English**

*Listens to both speaker and interpreter simultaneously — evaluates accuracy, fluency, pragmatics & terminology on the fly.*

</div>

---

## 🏗️ Architecture

```
                  ┌─ Stream A: Source (ES) ──► Whisper large-v3 ──► Qwen2.5-7B (domain MT) ─┐
  Audio Input ───┤                                                                             ├──► LLM Judge ──► 📊 Dashboard
                  └─ Stream B: Interp (EN) ──► Whisper large-v3-turbo                         │
                                                NLLB-200-600M (baseline) ──────────────────────┘
```

### 📦 Models & VRAM

| Module | Model | VRAM |
|--------|-------|------|
| 🎤 Source ASR | `faster-whisper large-v3` (int8) | ~3 GB |
| 🎧 Interpreter ASR | `faster-whisper large-v3-turbo` (int8) | ~2 GB |
| 🌐 MT Reference | `Qwen2.5-7B-Instruct` GPTQ 4-bit + glossary | ~5 GB |
| ⚖️ Quality Judge | Qwen2.5-7B (shared, different prompt) | — |
| 📊 Baseline MT | `NLLB-200-distilled-600M` | ~2 GB |
| | **Total** | **~12 GB** |

### 🔍 Evaluation Dimensions

| Dimension | What it measures |
|-----------|-----------------|
| ✅ Accuracy | Semantic faithfulness — COMET + LLM judge |
| ❌ Omissions / Additions | Missed or hallucinated content |
| 💬 Fluency | Grammaticality and naturalness |
| 🎭 Pragmatic quality | Register, tone, cultural adaptation *(novel)* |

---

## ⚡ Quick Start

```bash
git clone https://github.com/Anmol2059/interpret-bench.git
cd interpret-bench
chmod +x run.sh && ./run.sh
```

> `run.sh` auto-creates `.venv`, installs deps, downloads models (~15 GB), and launches the dashboard.

**CUDA override:**
```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./run.sh
```

### 🐳 Docker

```bash
docker build -t interpretbench .
docker run --gpus all -p 7860:7860 -v $(pwd)/models:/app/models interpretbench
```

---

## 🚀 Usage

**🎙️ Live streaming + dashboard:**
```bash
python stream.py \
  --source-device 0 --interp-device 1 \
  --domain economics \
  --glossary glossaries/economics_es_en.json \
  --ui
```

**📁 Offline evaluation:**
```bash
python evaluate.py \
  --source-audio audio/source_es.wav \
  --interpreter-audio audio/interp_en.wav \
  --domain economics \
  --glossary glossaries/economics_es_en.json \
  --output results/session_001.json
```

**📊 Benchmarks:**
```bash
python benchmarks/asr.py       --dataset data/fisher_callhome_es/
python benchmarks/mt.py        --source-transcripts data/src.json --references data/refs.json
python benchmarks/evaluator.py --sessions data/test_sessions/
```

**⬇️ Models only:**
```bash
python scripts/download_models.py
```

---

## 📁 Project Structure

```
interpret-bench/
├── 📄 stream.py              ← live streaming entry point
├── 📄 evaluate.py            ← offline evaluation entry point
├── 🐳 Dockerfile
├── 🚀 run.sh
├── 📋 requirements.txt
│
├── src/                      ← core library
│   ├── asr/                  · Transcriber, DomainAdapter, VAD
│   ├── translation/          · Translator (Qwen), NLLBBaseline, GlossaryLoader
│   ├── evaluator/            · LLMJudge, COMETScorer, metrics
│   ├── streaming/            · AudioCapture, Buffer, Pipeline
│   └── ui/                   · Gradio dashboard
│
├── benchmarks/               ← research experiments
│   ├── asr.py                · ASR model comparison (WER, latency)
│   ├── mt.py                 · MT model comparison (BLEU, COMET)
│   └── evaluator.py          · LLM judge ablation
│
├── configs/                  ← YAML configs
│   ├── default.yaml
│   ├── economics.yaml
│   └── legal.yaml
│
├── glossaries/               ← domain terminology (ES→EN)
│   ├── economics_es_en.json
│   ├── legal_es_en.json
│   └── medical_es_en.json
│
├── scripts/                  ← utilities
│   ├── download_models.py    · check + download all HF models
│   ├── create_glossary.py    · interactive glossary builder
│   └── prepare_fisher.py     · prepare Fisher/Callhome dataset
│
├── data/                     ← datasets (gitignored)
└── results/                  ← outputs (gitignored)
```

---

## 🌍 Domain Adaptation

Glossaries are injected at two levels:

1. **🎤 ASR** — terms in Whisper `initial_prompt` to bias recognition of acronyms  
2. **🌐 MT + Judge** — full glossary in system prompt to enforce correct translations

**Add your own domain:**
```bash
python scripts/create_glossary.py
```
Then point `--glossary` at your new file and create a matching config in `configs/`.

---

## 📚 Citation

```bibtex
@mastersthesis{interpretbench2026,
  title  = {InterpretBench: Domain-Adapted Real-Time Interpretation Quality
             Assessment with LLM-as-Judge Evaluation},
  author = {Anmol Guragain},
  year   = {2026}
}
```

---

<div align="center">
MIT License · Built with faster-whisper · Qwen2.5 · NLLB-200 · COMET · Silero VAD · Gradio
</div>
