# ─── Base image ───────────────────────────────────────────────────────────────
# CPU variant (default). For CUDA support swap to:
#   nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM python:3.11-slim

# ─── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        portaudio19-dev \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ─── Copy source ─────────────────────────────────────────────────────────────
COPY . .
RUN chmod +x run.sh

# ─── Download models at build time (set HF_TOKEN env var for gated models) ───
# Comment out to download at runtime instead
RUN python scripts/download_models.py || echo "[Docker] Models will be downloaded at runtime"

# ─── Expose Gradio default port ───────────────────────────────────────────────
EXPOSE 7860

# ─── Persist model cache across runs ─────────────────────────────────────────
VOLUME ["/app/models"]

ENV HF_HOME=/app/models/hf_cache
ENV PYTHONUNBUFFERED=1

CMD ["./run.sh"]
