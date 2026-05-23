#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

# ── 1. Virtual environment ───────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[run.sh] Creating virtual environment at $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "[run.sh] Virtual environment active: $(which python)"

# ── 2. Install / update dependencies ────────────────────────────────────────
echo "[run.sh] Installing requirements ..."
pip install --quiet --upgrade pip
# Install PyTorch first if not present (auto-detect CUDA)
python - <<'EOF'
import importlib.util, subprocess, sys
if importlib.util.find_spec("torch") is None:
    print("[run.sh] torch not found — attempting CPU install. Override TORCH_INDEX_URL for CUDA.")
    url = __import__("os").environ.get(
        "TORCH_INDEX_URL", "https://download.pytorch.org/whl/cpu"
    )
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "torch", "torchvision", "torchaudio", "--index-url", url
    ])
EOF
pip install --quiet -r requirements.txt

# ── 3. Download models if needed ─────────────────────────────────────────────
echo "[run.sh] Checking models ..."
python scripts/download_models.py

# ── 4. Launch ────────────────────────────────────────────────────────────────
echo "[run.sh] Starting InterpretBench ..."
exec python stream.py --ui "$@"
