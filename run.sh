#!/usr/bin/env bash
# run.sh — Full pipeline: setup → data → models → ASR eval → parallel pipeline experiments
# For the parallel pipeline run only, you can also call: bash run_parallel.sh
set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"
DATA_RAW="data/epic_raw"
DATA_PROCESSED="data/epic_processed"
GLOSSARY_ES="glossaries/eu_parliament_es.json"
GLOSSARY_ES_EN="glossaries/eu_parliament_es_en.json"
TS=$(date +%Y%m%d_%H%M%S)

# ── 1. Virtual environment ───────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[run.sh] Creating virtual environment ..."
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
export PYTHONPATH="${PYTHONPATH:-.}"
# Use local disk for HF cache — NFS (/autofs/thau00a) is too slow for large models
export HF_HOME="${HF_HOME:-/mnt/thau08a/aguragain/hf_cache}"
echo "[run.sh] Python: $(which python)  TS=${TS}"

# ── 2. PyTorch (auto-detect CUDA version) ───────────────────────────────────
python - <<'EOF'
import importlib.util, subprocess, sys, os, json, pathlib

if importlib.util.find_spec("torch") is None:
    url = os.environ.get("TORCH_INDEX_URL", "")
    if not url:
        try:
            v = json.loads(pathlib.Path("/usr/local/cuda/version.json").read_text())
            major, minor = v["cuda"]["version"].split(".")[:2]
            # PyTorch wheel index publishes up to cu126 for CUDA 12.x
            tag = f"cu12{min(int(minor), 6)}" if int(major) == 12 else "cu118"
        except Exception:
            tag = "cu126"
        url = f"https://download.pytorch.org/whl/{tag}"
    print(f"[run.sh] Installing PyTorch ({url}) ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "torch", "torchvision", "torchaudio", "--index-url", url])
else:
    import torch
    print(f"[run.sh] torch {torch.__version__} already installed.")
EOF

# ── 3. Python dependencies ───────────────────────────────────────────────────
pip install --quiet -r requirements.txt

# ── 4. Download EPIC v2.0 (resumable — safe to re-run) ──────────────────────
echo "[run.sh] Downloading EPIC v2.0 ..."
python data/download_epic.py --data-dir "$DATA_RAW"

# ── 5. Prepare: clean transcripts + symlink audio ───────────────────────────
if [ ! -f "$DATA_PROCESSED/manifest.json" ]; then
    echo "[run.sh] Preparing data ..."
    python scripts/prepare_epic.py \
        --epic-dir "$DATA_RAW" \
        --output-dir "$DATA_PROCESSED"
else
    echo "[run.sh] Data already prepared (manifest exists) — skipping."
fi

# ── 6. Download models ───────────────────────────────────────────────────────
echo "[run.sh] Downloading models ..."
python scripts/download_models.py

# ── 7. Experiment 1: ASR quality (Whisper vs gold ES transcripts) ────────────
echo "[run.sh] Experiment 1: ASR ..."
python experiments/run_asr.py \
    --audio-dir "$DATA_PROCESSED/audio" \
    --gold-dir  "$DATA_PROCESSED/transcripts_es" \
    --backend whisper \
    --output-dir "results/asr_whisper_${TS}" \
    --log-dir logs

# ── 8. Experiments 2 & 3: Pipeline — cascaded + seamless in parallel ─────────
echo "[run.sh] Experiments 2 & 3: Pipeline (parallel across 2 GPUs) ..."
bash run_parallel.sh

echo ""
echo "[run.sh] All done. Results in results/"
ls -d results/*_${TS}/ 2>/dev/null || true
