#!/usr/bin/env bash
# run_parallel.sh — Run both pipeline experiments simultaneously across 2 GPUs.
#
# GPU 0: System A — Whisper large-v3 + NLLB-3.3B (cascaded)
# GPU 1: System B — SeamlessM4T v2 (end-to-end)
#
# Prerequisites: run steps 1-5 of run.sh first (venv, deps, data, models).
set -euo pipefail

VENV_DIR=".venv"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="${PYTHONPATH:-.}"

DATA_PROCESSED="data/epic_processed"
GLOSSARY="glossaries/eu_parliament_es_en.json"
TS=$(date +%Y%m%d_%H%M%S)

# Sanity check data exists
if [ ! -d "$DATA_PROCESSED/audio" ]; then
    echo "[ERROR] $DATA_PROCESSED/audio not found — run prepare_epic.py first."
    exit 1
fi

echo "[parallel] TS=${TS}"
echo "[parallel] GPU 0 → cascaded (Whisper + NLLB-3.3B)"
echo "[parallel] GPU 1 → seamless (SeamlessM4T v2)"
echo ""

# ── GPU 0: cascaded pipeline ─────────────────────────────────────────────────
(
    CUDA_VISIBLE_DEVICES=0 python experiments/run_pipeline.py \
        --audio-dir     "$DATA_PROCESSED/audio" \
        --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
        --source-dir    "$DATA_PROCESSED/transcripts_es" \
        --glossary      "$GLOSSARY" \
        --backend cascaded \
        --output-dir "results/pipeline_cascaded_${TS}" \
        --log-dir logs
) &
PID_A=$!

# ── GPU 1: SeamlessM4T v2 end-to-end ─────────────────────────────────────────
(
    CUDA_VISIBLE_DEVICES=1 python experiments/run_pipeline.py \
        --audio-dir     "$DATA_PROCESSED/audio" \
        --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
        --source-dir    "$DATA_PROCESSED/transcripts_es" \
        --glossary      "$GLOSSARY" \
        --backend seamless \
        --output-dir "results/pipeline_seamless_${TS}" \
        --log-dir logs
) &
PID_B=$!

echo "[parallel] Running... PIDs: GPU0=$PID_A  GPU1=$PID_B"
echo "[parallel] Follow progress (run in separate terminals):"
echo "           tail -f \$(ls -t logs/pipeline_cascaded_*.log | head -1)"
echo "           tail -f \$(ls -t logs/pipeline_seamless_*.log | head -1)"
echo ""

# Wait for both, report outcome independently
GPU0_OK=true
GPU1_OK=true
wait $PID_A || GPU0_OK=false
wait $PID_B || GPU1_OK=false

echo ""
$GPU0_OK && echo "[GPU 0] cascaded  ✓  results/pipeline_cascaded_${TS}/" \
          || echo "[GPU 0] cascaded  FAILED — check logs/pipeline_cascaded_*.log"
$GPU1_OK && echo "[GPU 1] seamless  ✓  results/pipeline_seamless_${TS}/" \
          || echo "[GPU 1] seamless  FAILED — check logs/pipeline_seamless_*.log"
