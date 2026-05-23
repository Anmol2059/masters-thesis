#!/usr/bin/env bash
# run.sh — full thesis pipeline: setup → download → prepare → all experiments (single GPU)
# For parallel 2-GPU Exp 3, use run_parallel.sh instead.
set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"
DATA_RAW="data/epic_raw"
DATA_PROCESSED="data/epic_processed"
DIRECTION="es-en"
GLOSSARY_ES="glossaries/eu_parliament_es.json"
GLOSSARY_ES_EN="glossaries/eu_parliament_es_en.json"
TS=$(date +%Y%m%d_%H%M%S)

# ── 1. Virtual environment ───────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[run.sh] Creating virtual environment ..."
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "[run.sh] Python: $(which python)  TS: $TS"

# ── 2. Dependencies ──────────────────────────────────────────────────────────
pip install --quiet --upgrade pip

python - <<'EOF'
import importlib.util, subprocess, sys, os, json, pathlib

if importlib.util.find_spec("torch") is None:
    # Detect CUDA version from /usr/local/cuda/version.json, fall back to cu126
    url = os.environ.get("TORCH_INDEX_URL", "")
    if not url:
        try:
            v = json.loads(pathlib.Path("/usr/local/cuda/version.json").read_text())
            major, minor = v["cuda"]["version"].split(".")[:2]
            # PyTorch wheel index only publishes up to cu126 for CUDA 12.x
            if int(major) == 12:
                tag = f"cu12{min(int(minor), 6)}"
            elif int(major) == 11:
                tag = "cu118"
            else:
                tag = "cu126"
        except Exception:
            tag = "cu126"
        url = f"https://download.pytorch.org/whl/{tag}"
    print(f"[run.sh] Installing PyTorch ({url}) ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "torch", "torchvision", "torchaudio", "--index-url", url])
EOF

pip install --quiet -r requirements.txt

# ── 3. Download EPIC v2.0 (resumable) ───────────────────────────────────────
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

# ── 6. Experiment 1: ASR ─────────────────────────────────────────────────────
echo "[run.sh] Experiment 1: ASR ..."

python experiments/run_asr.py \
    --audio-dir "$DATA_PROCESSED/audio" \
    --gold-dir  "$DATA_PROCESSED/transcripts_es" \
    --backend whisper --condition vanilla \
    --output-dir "results/asr_whisper_vanilla_${TS}" --log-dir logs

python experiments/run_asr.py \
    --audio-dir "$DATA_PROCESSED/audio" \
    --gold-dir  "$DATA_PROCESSED/transcripts_es" \
    --backend whisper --condition domain \
    --glossary "$GLOSSARY_ES" \
    --output-dir "results/asr_whisper_domain_${TS}" --log-dir logs

python experiments/run_asr.py \
    --audio-dir "$DATA_PROCESSED/audio" \
    --gold-dir  "$DATA_PROCESSED/transcripts_es" \
    --backend seamless \
    --output-dir "results/asr_seamless_${TS}" --log-dir logs

# ── 7. Experiment 2: Translation ─────────────────────────────────────────────
echo "[run.sh] Experiment 2: Translation ..."

python experiments/run_translation.py \
    --source-dir    "$DATA_PROCESSED/transcripts_es" \
    --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
    --model nllb \
    --output-dir "results/mt_nllb_${TS}" --log-dir logs

python experiments/run_translation.py \
    --source-dir    "$DATA_PROCESSED/transcripts_es" \
    --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
    --model qwen --condition vanilla \
    --output-dir "results/mt_qwen_vanilla_${TS}" --log-dir logs

python experiments/run_translation.py \
    --source-dir    "$DATA_PROCESSED/transcripts_es" \
    --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
    --model qwen --condition domain \
    --glossary "$GLOSSARY_ES_EN" \
    --output-dir "results/mt_qwen_domain_${TS}" --log-dir logs

# ── 8. Experiment 3: Full pipeline ───────────────────────────────────────────
echo "[run.sh] Experiment 3: Full pipeline ..."

python experiments/run_pipeline.py \
    --audio-dir     "$DATA_PROCESSED/audio" \
    --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
    --source-dir    "$DATA_PROCESSED/transcripts_es" \
    --backend cascaded --condition vanilla \
    --output-dir "results/pipeline_cascaded_vanilla_${TS}" --log-dir logs

python experiments/run_pipeline.py \
    --audio-dir     "$DATA_PROCESSED/audio" \
    --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
    --source-dir    "$DATA_PROCESSED/transcripts_es" \
    --backend cascaded --condition domain \
    --glossary "$GLOSSARY_ES_EN" \
    --output-dir "results/pipeline_cascaded_domain_${TS}" --log-dir logs

python experiments/run_pipeline.py \
    --audio-dir     "$DATA_PROCESSED/audio" \
    --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
    --source-dir    "$DATA_PROCESSED/transcripts_es" \
    --backend seamless \
    --output-dir "results/pipeline_seamless_${TS}" --log-dir logs

# ── 9. Experiment 4: Glossary comparison ─────────────────────────────────────
echo "[run.sh] Experiment 4: Glossary comparison ..."
python experiments/compare_glossaries.py \
    --source-dir    "$DATA_PROCESSED/transcripts_es" \
    --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
    --glossaries manual tfidf llm \
    --output-dir "results/glossary_comparison_${TS}" --log-dir logs

echo "[run.sh] All experiments done. Results in results/"
