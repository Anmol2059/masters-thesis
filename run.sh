#!/usr/bin/env bash
# run.sh — full thesis pipeline: setup → download → prepare → experiments
set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"
DATA_RAW="data/epic_raw"
DATA_PROCESSED="data/epic_processed"
DIRECTION="es-en"

# ── 1. Virtual environment ───────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[run.sh] Creating virtual environment ..."
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "[run.sh] Python: $(which python)"

# ── 2. Dependencies ──────────────────────────────────────────────────────────
pip install --quiet --upgrade pip

python - <<'EOF'
import importlib.util, subprocess, sys, os
if importlib.util.find_spec("torch") is None:
    url = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu121")
    print(f"[run.sh] Installing PyTorch from {url} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "torch", "torchvision", "torchaudio", "--index-url", url])
EOF

pip install --quiet -r requirements.txt

# ── 3. Download EPIC v2.0 (resumable — safe to re-run) ──────────────────────
echo "[run.sh] Downloading EPIC v2.0 ..."
python data/download_epic.py \
    --data-dir "$DATA_RAW" \
    --direction "$DIRECTION"

# ── 4. Prepare: clean transcripts + pair with audio ─────────────────────────
echo "[run.sh] Preparing data ..."
python scripts/prepare_epic.py \
    --epic-dir "$DATA_RAW" \
    --output-dir "$DATA_PROCESSED" \
    --direction "$DIRECTION"

# ── 5. Download models ───────────────────────────────────────────────────────
echo "[run.sh] Downloading models ..."
python scripts/download_models.py

# ── 6. Experiments ───────────────────────────────────────────────────────────
echo "[run.sh] Experiment 1: ASR ..."
python experiments/run_asr.py \
    --audio-dir "$DATA_PROCESSED/audio" \
    --gold-dir  "$DATA_PROCESSED/transcripts_es" \
    --model large-v3 \
    --condition vanilla \
    --output results/asr_vanilla.json

python experiments/run_asr.py \
    --audio-dir "$DATA_PROCESSED/audio" \
    --gold-dir  "$DATA_PROCESSED/transcripts_es" \
    --model large-v3 \
    --condition domain \
    --glossary glossaries/eu_parliament_es.json \
    --output results/asr_domain.json

echo "[run.sh] Experiment 2: Translation ..."
python experiments/run_translation.py \
    --source-dir    "$DATA_PROCESSED/transcripts_es" \
    --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
    --model nllb \
    --output results/mt_nllb.json

python experiments/run_translation.py \
    --source-dir    "$DATA_PROCESSED/transcripts_es" \
    --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
    --model qwen \
    --condition vanilla \
    --output results/mt_qwen_vanilla.json

python experiments/run_translation.py \
    --source-dir    "$DATA_PROCESSED/transcripts_es" \
    --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
    --model qwen \
    --condition domain \
    --glossary glossaries/eu_parliament_es_en.json \
    --output results/mt_qwen_domain.json

echo "[run.sh] Experiment 3: Full pipeline ..."
python experiments/run_pipeline.py \
    --audio-dir     "$DATA_PROCESSED/audio" \
    --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
    --condition vanilla \
    --output results/pipeline_vanilla.json

python experiments/run_pipeline.py \
    --audio-dir     "$DATA_PROCESSED/audio" \
    --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
    --condition domain \
    --glossary glossaries/eu_parliament_es_en.json \
    --output results/pipeline_domain.json

echo "[run.sh] Experiment 4: Glossary comparison ..."
python experiments/compare_glossaries.py \
    --source-dir    "$DATA_PROCESSED/transcripts_es" \
    --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
    --glossaries manual tfidf llm \
    --output results/glossary_comparison.json

echo "[run.sh] All experiments done. Results in results/"
