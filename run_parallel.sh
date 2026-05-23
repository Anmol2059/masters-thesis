#!/usr/bin/env bash
# run_parallel.sh — Experiment 3 across both GPUs simultaneously.
#
# GPU 0: cascaded pipeline (Whisper → Qwen3), vanilla + domain conditions
# GPU 1: SeamlessM4T v2 end-to-end
#
# Run AFTER: data is prepared and models are downloaded (run.sh steps 1-5).
set -euo pipefail

VENV_DIR=".venv"
source "$VENV_DIR/bin/activate"

DATA_PROCESSED="data/epic_processed"
GLOSSARY_ES_EN="glossaries/eu_parliament_es_en.json"
TS=$(date +%Y%m%d_%H%M%S)

echo "[parallel] Starting Exp 3 on 2 GPUs  (TS=${TS})"

# ── GPU 0: cascaded conditions (sequential within GPU) ───────────────────────
(
    echo "[GPU 0] cascaded vanilla ..."
    CUDA_VISIBLE_DEVICES=0 python experiments/run_pipeline.py \
        --audio-dir     "$DATA_PROCESSED/audio" \
        --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
        --source-dir    "$DATA_PROCESSED/transcripts_es" \
        --backend cascaded --condition vanilla \
        --output-dir "results/pipeline_cascaded_vanilla_${TS}" \
        --log-dir logs

    echo "[GPU 0] cascaded domain ..."
    CUDA_VISIBLE_DEVICES=0 python experiments/run_pipeline.py \
        --audio-dir     "$DATA_PROCESSED/audio" \
        --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
        --source-dir    "$DATA_PROCESSED/transcripts_es" \
        --backend cascaded --condition domain \
        --glossary "$GLOSSARY_ES_EN" \
        --output-dir "results/pipeline_cascaded_domain_${TS}" \
        --log-dir logs
) &
PID_A=$!

# ── GPU 1: SeamlessM4T end-to-end ────────────────────────────────────────────
(
    echo "[GPU 1] seamless e2e ..."
    CUDA_VISIBLE_DEVICES=1 python experiments/run_pipeline.py \
        --audio-dir     "$DATA_PROCESSED/audio" \
        --reference-dir "$DATA_PROCESSED/transcripts_en_interp" \
        --source-dir    "$DATA_PROCESSED/transcripts_es" \
        --backend seamless \
        --output-dir "results/pipeline_seamless_${TS}" \
        --log-dir logs
) &
PID_B=$!

echo "[parallel] GPU 0 (cascaded) PID=$PID_A  |  GPU 1 (seamless) PID=$PID_B"
echo "[parallel] Tail logs in real time with:"
echo "           tail -f logs/pipeline_cascaded_*.log"
echo "           tail -f logs/pipeline_seamless_e2e_*.log"

wait $PID_A && echo "[GPU 0] done." || echo "[GPU 0] FAILED — check logs."
wait $PID_B && echo "[GPU 1] done." || echo "[GPU 1] FAILED — check logs."

echo "[parallel] Exp 3 complete. Results:"
ls -d results/pipeline_*_${TS}/
